// Run the web scanner's pipeline outside the browser and record every decision.
//
//   node --import tsx scripts/trace.ts DECODED_DIR OUT_DIR [fast|refined|both]
//
// DECODED_DIR comes from tools/web/decode_photos.py. For each photo and mode this writes
// <stem>.<mode>.json (the scan trace, warnings and output size) and <stem>.<mode>.rgba,
// the flattened page before color or finger edits. ONNX Runtime Web runs on its WASM
// backend with FLATTENER_THREADS threads (default 2).
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import * as ort from 'onnxruntime-web';
import { scan, type ScanDebug, type ScanModels } from '../src/scanner';
import type { Manifest, Options, ScanTrace } from '../src/types';

type Photo = { stem: string; width: number; height: number };

const [decoded, out, which = 'both'] = process.argv.slice(2);
if (!decoded || !out) {
  console.error('usage: trace.ts DECODED_DIR OUT_DIR [fast|refined|both]');
  process.exit(2);
}
ort.env.wasm.numThreads = Number(process.env.FLATTENER_THREADS ?? 2);
ort.env.logLevel = 'error';

const manifest: Manifest = JSON.parse(readFileSync('public/models/manifest.json', 'utf8'));
const sessions = Object.fromEntries(
  await Promise.all(
    (['gate', 'orient', 'dewarp'] as const).map(async (role) => [
      role,
      await ort.InferenceSession.create(
        readFileSync(`public/models/${manifest.models[role].file}`),
        {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        },
      ),
    ]),
  ),
) as Record<keyof Manifest['models'], ort.InferenceSession>;

const models: ScanModels = {
  manifest,
  async run(role, inputs) {
    const feed: Record<string, ort.Tensor> = {};
    for (const [name, data] of Object.entries(inputs))
      feed[name] = new ort.Tensor('float32', data, manifest.models[role].inputs[name]);
    const outputs = await sessions[role].run(feed);
    return Object.fromEntries(
      Object.entries(outputs).map(([key, value]) => [
        key,
        new Float32Array(value.data as Float32Array),
      ]),
    );
  },
};

// The preview images are not needed here.
const steps = {
  page: async () => {},
  mesh: async () => {},
  image: async () => {},
  mask: async () => {},
  paper: async () => {},
} as unknown as ScanDebug['steps'];

mkdirSync(out, { recursive: true });
const photos: Photo[] = JSON.parse(readFileSync(join(decoded, 'photos.json'), 'utf8'));
const modes = which === 'both' ? (['fast', 'refined'] as const) : [which as 'fast' | 'refined'];
for (const photo of photos) {
  const pixels = new Uint8ClampedArray(readFileSync(join(decoded, `${photo.stem}.rgba`)));
  for (const quality of modes) {
    // A scan may paint missing page corners into its photo; each mode gets a fresh copy.
    const image = { width: photo.width, height: photo.height, data: pixels.slice() };
    const options: Options = {
      quality,
      orient: true,
      maxSide: 2560,
      sharpen: true,
      cleanColor: true,
    };
    const trace: ScanTrace = { placements: [], orientation: [], passes: [] };
    const started = performance.now();
    let record: Record<string, unknown>;
    try {
      const result = await scan(image, models, options, () => {}, { trace, steps });
      writeFileSync(join(out, `${photo.stem}.${quality}.rgba`), result.result.data);
      record = {
        photo,
        quality,
        seconds: (performance.now() - started) / 1000,
        warnings: result.warnings,
        trace,
      };
    } catch (error) {
      record = { photo, quality, error: String(error), trace };
    }
    writeFileSync(join(out, `${photo.stem}.${quality}.json`), JSON.stringify(record));
    const t = record.trace as ScanTrace;
    console.log(
      photo.stem,
      quality,
      record.error ?? `${t.output?.width}x${t.output?.height}`,
      'rotation',
      t.rotation,
      t.orientation.map((o) => `${o.stage}:k${o.k}${o.accepted ? '' : '?'}`).join(' '),
    );
  }
}
