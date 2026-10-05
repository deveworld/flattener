import { cleanDocument, estimatePaper, rotatePixels } from './cleanup';
import { encode } from './previews';
import type { EditReply, EditRequest } from './types';

self.onmessage = async ({ data }: MessageEvent<EditRequest>) => {
  let bitmap: ImageBitmap | undefined;
  const reply = (message: EditReply) => self.postMessage(message);
  try {
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
    const result = rotatePixels(
      cleanDocument(
        source,
        data.settings.mask,
        data.settings.cleanColor,
        estimatePaper(source, excluded),
      ),
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
