import { cleanDocument, estimatePaper, rotatePixels } from './cleanup';
import { cropPixels, FULL_CROP } from './crop';
import { encode } from './previews';
import { sharpenPixels } from './sharpen';
import type { EditReply, EditRequest, Pixels } from './types';

// Rotating or cropping leaves the corrected page as it was: the last one is kept and reused
// while its color mode, sharpening, removal mask and source stay the same. The main thread
// starts a new worker for every scan, so a cached page never outlives its scan.
let cached: { size: number; key: string; mask: Uint8Array; page: Pixels } | undefined;

self.onmessage = async ({ data }: MessageEvent<EditRequest>) => {
  let bitmap: ImageBitmap | undefined;
  const reply = (message: EditReply) => self.postMessage(message);
  try {
    const key = `${data.settings.cleanColor}/${data.sharpen}`;
    const mask = data.settings.mask.data;
    if (
      cached &&
      cached.size === data.raw.size &&
      cached.key === key &&
      cached.mask.length === mask.length &&
      cached.mask.every((v, i) => v === mask[i])
    ) {
      const result = rotatePixels(
        cropPixels(cached.page, data.settings.crop ?? FULL_CROP),
        data.settings.rotation,
      );
      reply({
        type: 'edited',
        id: data.id,
        blob: await encode(result),
        width: result.width,
        height: result.height,
      });
      return;
    }
    bitmap = await createImageBitmap(data.raw);
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height),
      ctx = canvas.getContext('2d')!;
    ctx.drawImage(bitmap, 0, 0);
    const source = {
      width: bitmap.width,
      height: bitmap.height,
      data: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data,
    };
    const excluded = {
      ...data.settings.mask,
      data: data.settings.mask.data.map((v, i) => (v || data.suggestion.data[i] ? 1 : 0)),
    };
    // Color and removal work on the whole page; the crop follows, then the quarter turns.
    let page = cleanDocument(
      source,
      data.settings.mask,
      data.settings.cleanColor,
      estimatePaper(source, excluded),
      { excluded },
    );
    // Sharpening only in Original color (see ``scan`` in scanner.ts).
    if (!data.settings.cleanColor) page = sharpenPixels(page, data.sharpen);
    cached = { size: data.raw.size, key, mask: mask.slice(), page };
    const result = rotatePixels(
      cropPixels(page, data.settings.crop ?? FULL_CROP),
      data.settings.rotation,
    );
    reply({
      type: 'edited',
      id: data.id,
      blob: await encode(result),
      width: result.width,
      height: result.height,
    });
  } catch (error) {
    reply({
      type: 'error',
      id: data.id,
      message: error instanceof Error ? error.message : 'The edit failed. Try a smaller selection.',
    });
  } finally {
    bitmap?.close();
  }
};
