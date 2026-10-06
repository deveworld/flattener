import { Models, readManifest } from './models';
import { scan } from './scanner';
import {
  cleanDocument,
  emptyMask,
  estimatePaper,
  maskCount,
  pageConfidence,
  suggestFingers,
} from './cleanup';
import { createSteps, encode } from './previews';
import { sharpenPixels } from './sharpen';
import type { Backend, Pixels, ScanTrace, WorkerReply, WorkerRequest } from './types';

let models: Models | undefined;
let preparation: Promise<Models> | undefined;
let busy = false;
const report = (event: WorkerReply) => self.postMessage(event);

function prepare(base: string, backend?: Backend): Promise<Models> {
  if (models) return Promise.resolve(models);
  if (preparation) return preparation;
  report({ type: 'preparation', state: 'loading' });
  preparation = (async () => {
    const candidate = new Models(await readManifest(base), base, (event) => {
      if (event.type !== 'progress' || busy) report(event);
    });
    try {
      await candidate.load(backend);
      try {
        await candidate.warmup();
      } catch (error) {
        if (candidate.backend !== 'webgpu') throw error;
        await candidate.fallback();
        await candidate.warmup();
      }
      models = candidate;
      return candidate;
    } catch (error) {
      await candidate.dispose();
      throw error;
    }
  })()
    .then((candidate) => {
      report({ type: 'preparation', state: 'ready' });
      return candidate;
    })
    .catch((error) => {
      report({ type: 'preparation', state: 'error' });
      throw error;
    })
    .finally(() => {
      preparation = undefined;
    });
  return preparation;
}

async function decode(file: File): Promise<Pixels> {
  if (file.size > 40 * 1024 * 1024)
    throw new Error('This photo is larger than 40 MB. Choose a smaller JPEG, PNG, or WebP.');
  if (file.type && !['image/jpeg', 'image/png', 'image/webp'].includes(file.type))
    throw new Error(
      'Choose a JPEG, PNG, or WebP photo. Convert HEIC files before opening them here.',
    );
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  } catch {
    throw new Error('This image could not be opened. Try saving it as JPEG or PNG.');
  }
  try {
    if (bitmap.width * bitmap.height > 20_000_000)
      throw new Error(
        'This photo exceeds 20 megapixels. Resize it before scanning to keep browser memory manageable.',
      );
    if (Math.min(bitmap.width, bitmap.height) < 64)
      throw new Error('This photo is too small. Choose an image at least 64 pixels on each side.');
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height),
      context = canvas.getContext('2d');
    if (!context)
      throw new Error(
        'This browser cannot read images. Try a current version of Chrome, Edge, Firefox, or Safari.',
      );
    context.fillStyle = '#fff';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(bitmap, 0, 0);
    return {
      width: canvas.width,
      height: canvas.height,
      data: context.getImageData(0, 0, canvas.width, canvas.height).data,
    };
  } finally {
    bitmap.close();
  }
}

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type === 'prepare') {
    void prepare(event.data.base, event.data.backend).catch(() => {});
    return;
  }
  if (event.data.type !== 'scan' || busy) return;
  busy = true;
  try {
    const { file, options, base, backend } = event.data;
    let trace: ScanTrace | undefined;
    report({
      type: 'progress',
      stage: 'Opening your photo',
      detail: 'Your photo stays in this browser.',
    });
    const source = await decode(file);
    if (!models)
      report({
        type: 'progress',
        stage: 'Preparing scanner',
        detail: 'Your photo will be processed automatically.',
      });
    models = await prepare(base, backend);
    const start = performance.now();
    let output;
    try {
      trace = event.data.trace ? { placements: [], orientation: [], passes: [] } : undefined;
      output = await scan(source, models, options, report, { trace });
    } catch (error) {
      // Retry actual GPU/runtime failures, never turn a failed geometry check into success.
      if (
        models.backend !== 'webgpu' ||
        !/GPU|device lost|session|kernel|WebGPU|ORT|buffer|allocation/i.test(String(error))
      )
        throw error;
      report({ type: 'reset-steps' });
      await models.fallback();
      trace = event.data.trace ? { placements: [], orientation: [], passes: [] } : undefined;
      output = await scan(source, models, options, report, { trace });
    }
    const { result, warnings, passes, rotation } = output;
    const steps = createSteps(report),
      size = emptyMask(result.width, result.height);
    report({
      type: 'progress',
      stage: 'Checking the page edges',
      detail: 'Looking for possible fingers to review.',
    });
    const confidence = pageConfidence(
      output.grid,
      output.probability,
      output.gw,
      output.gh,
      size.width,
      size.height,
    );
    const suggestion = suggestFingers(confidence, size.width, size.height, result);
    if (maskCount(suggestion)) await steps.mask(result, suggestion);
    report({
      type: 'progress',
      stage: 'Correcting the paper color',
      detail: 'Estimating lighting and balancing the paper color.',
    });
    const field = estimatePaper(result, suggestion);
    await steps.paper(result, field);
    const corrected = cleanDocument(result, size, true, field, { excluded: suggestion });
    await steps.image(
      corrected,
      'color',
      'Color corrected',
      'Paper lighting and color balanced. You can switch back to Original color.',
    );
    // The scan is returned in the chosen color; edits recalculate from ``raw``.
    const raw = await encode(result),
      blob = await encode(options.cleanColor ? corrected : sharpenPixels(result, output.sharpen));
    report({
      type: 'result',
      blob,
      raw,
      suggestion,
      metadata: {
        backend: models.backend,
        seconds: (performance.now() - start) / 1000,
        width: result.width,
        height: result.height,
        warnings,
        passes,
        rotation,
        source: { width: source.width, height: source.height },
        model: models.manifest.version,
        sharpen: output.sharpen,
      },
      trace,
    });
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    const message =
      /backend|dynamically imported|out of memory|allocation|memory access|session|kernel/i.test(
        detail,
      )
        ? 'Your browser could not run the scanner. Try again with a smaller image, or use a current desktop browser.'
        : /Failed to fetch|NetworkError/i.test(detail)
          ? 'The scanner could not be downloaded. Check your connection and try again.'
          : detail;
    report({ type: 'error', message });
  } finally {
    busy = false;
  }
};
