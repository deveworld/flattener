import { bilinear } from './geometry';
import { paperAt } from './cleanup';
import type { Grid, Mask, PaperField, Pixels, WorkerReply } from './types';

export async function encode(image: Pixels): Promise<Blob> {
  const canvas = new OffscreenCanvas(image.width, image.height),
    ctx = canvas.getContext('2d')!;
  ctx.putImageData(
    new ImageData(new Uint8ClampedArray(image.data), image.width, image.height),
    0,
    0,
  );
  return canvas.convertToBlob({ type: 'image/png' });
}

async function canvasFor(image: Pixels, maxSide = 800) {
  const scale = Math.min(1, maxSide / Math.max(image.width, image.height));
  const canvas = new OffscreenCanvas(
    Math.max(1, Math.round(image.width * scale)),
    Math.max(1, Math.round(image.height * scale)),
  );
  const ctx = canvas.getContext('2d')!;
  const pixels = new ImageData(new Uint8ClampedArray(image.data), image.width, image.height);
  if (scale === 1) ctx.putImageData(pixels, 0, 0);
  else {
    const bitmap = await createImageBitmap(pixels, {
      resizeWidth: canvas.width,
      resizeHeight: canvas.height,
      resizeQuality: 'high',
    });
    try {
      ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    } finally {
      bitmap.close();
    }
  }
  return { canvas, ctx };
}

export function createSteps(report: (reply: WorkerReply) => void) {
  async function send(id: string, title: string, detail: string, canvas: OffscreenCanvas) {
    report({
      type: 'step',
      id,
      title,
      detail,
      blob: await canvas.convertToBlob({ type: 'image/png' }),
    });
  }
  return {
    async page(image: Pixels, probability: Float32Array, width: number, height: number) {
      const { canvas, ctx } = await canvasFor(image);
      const overlay = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let y = 0; y < canvas.height; y++)
        for (let x = 0; x < canvas.width; x++) {
          const p = bilinear(
            probability,
            width,
            height,
            ((x + 0.5) * width) / canvas.width - 0.5,
            ((y + 0.5) * height) / canvas.height - 0.5,
          );
          const i = (y * canvas.width + x) * 4;
          const a = p > 0.5 ? 0.28 : 0.62;
          const color = p > 0.5 ? [32, 142, 89] : [34, 40, 36];
          for (let c = 0; c < 3; c++)
            overlay.data[i + c] = overlay.data[i + c] * (1 - a) + color[c] * a;
        }
      ctx.putImageData(overlay, 0, 0);
      await send(
        'page',
        'Page detected',
        'Green shows the paper area found by the document model.',
        canvas,
      );
    },
    async mesh(image: Pixels, grid: Grid, pass: number) {
      const { canvas, ctx } = await canvasFor(image);
      const point = (r: number, c: number) => {
        const i = (r * grid.cols + c) * 2;
        return [
          ((grid.data[i] + 1) * canvas.width) / 2,
          ((grid.data[i + 1] + 1) * canvas.height) / 2,
        ] as const;
      };
      const path = () => {
        ctx.beginPath();
        for (let r = 0; r < grid.rows; r += 4) {
          ctx.moveTo(...point(r, 0));
          for (let c = 1; c < grid.cols; c++) ctx.lineTo(...point(r, c));
        }
        for (let c = 0; c < grid.cols; c += 3) {
          ctx.moveTo(...point(0, c));
          for (let r = 1; r < grid.rows; r++) ctx.lineTo(...point(r, c));
        }
      };
      path();
      ctx.lineWidth = 3;
      ctx.strokeStyle = '#163928bb';
      ctx.stroke();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = '#a2ffbd';
      ctx.stroke();
      await send(
        `mesh-${pass}`,
        pass === 1 ? 'Page shape estimated' : `Edges refined · ${pass - 1}`,
        'These lines follow the calculated page surface. They become straight in the scan.',
        canvas,
      );
    },
    async image(image: Pixels, id: string, title: string, detail: string) {
      await send(id, title, detail, (await canvasFor(image)).canvas);
    },
    async mask(image: Pixels, mask: Mask) {
      const { canvas, ctx } = await canvasFor(image),
        overlay = ctx.getImageData(0, 0, canvas.width, canvas.height);
      for (let y = 0; y < canvas.height; y++)
        for (let x = 0; x < canvas.width; x++) {
          const mx = Math.min(mask.width - 1, Math.floor((x * mask.width) / canvas.width));
          const my = Math.min(mask.height - 1, Math.floor((y * mask.height) / canvas.height));
          if (!mask.data[my * mask.width + mx]) continue;
          const i = (y * canvas.width + x) * 4;
          for (let c = 0; c < 3; c++)
            overlay.data[i + c] = overlay.data[i + c] * 0.45 + [211, 65, 48][c] * 0.55;
        }
      ctx.putImageData(overlay, 0, 0);
      await send(
        'fingers',
        'Possible fingers',
        'Red marks a proposed removal area. Review it with Remove fingers before applying.',
        canvas,
      );
    },
    async paper(image: Pixels, field: PaperField) {
      const scale = Math.min(1, 640 / Math.max(image.width, image.height));
      const width = Math.round(image.width * scale),
        height = Math.round(image.height * scale);
      const data = new Uint8ClampedArray(width * height * 4);
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4;
          for (let c = 0; c < 3; c++) data[i + c] = paperAt(field, x, y, width, height, c);
          data[i + 3] = 255;
        }
      await send(
        'light',
        'Paper lighting estimated',
        'The estimated paper color and lighting, used to reduce shadows and color cast.',
        (await canvasFor({ width, height, data })).canvas,
      );
    },
  };
}
