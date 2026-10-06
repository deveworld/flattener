import type { Pixels } from './types';

/** A crop as fractions of the unrotated scan: 0 is the left or top edge, 1 the right or bottom. */
export type Crop = { left: number; top: number; right: number; bottom: number };

export const FULL_CROP: Crop = { left: 0, top: 0, right: 1, bottom: 1 };
export const MIN_CROP = 0.05;

export function isFullCrop(crop: Crop | undefined): boolean {
  return (
    !crop ||
    (crop.left <= 0.001 && crop.top <= 0.001 && crop.right >= 0.999 && crop.bottom >= 0.999)
  );
}

/** ``crop`` after ``turns`` clockwise quarter turns of the image (``rotatePixels``). */
export function turnCrop(crop: Crop, turns: number): Crop {
  let { left, top, right, bottom } = crop;
  for (let k = 0; k < ((turns % 4) + 4) % 4; k++)
    [left, top, right, bottom] = [1 - bottom, left, 1 - top, right];
  return { left, top, right, bottom };
}

/** Pixel bounds of ``crop`` on an image, at least one pixel each way. */
export function cropBounds(crop: Crop, width: number, height: number) {
  const x0 = Math.min(width - 1, Math.max(0, Math.round(crop.left * width)));
  const y0 = Math.min(height - 1, Math.max(0, Math.round(crop.top * height)));
  const x1 = Math.max(x0 + 1, Math.min(width, Math.round(crop.right * width)));
  const y1 = Math.max(y0 + 1, Math.min(height, Math.round(crop.bottom * height)));
  return { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
}

export function cropPixels(image: Pixels, crop: Crop): Pixels {
  if (isFullCrop(crop)) return image;
  const box = cropBounds(crop, image.width, image.height);
  const data = new Uint8ClampedArray(box.width * box.height * 4);
  for (let y = 0; y < box.height; y++) {
    const start = ((box.y + y) * image.width + box.x) * 4;
    data.set(image.data.subarray(start, start + box.width * 4), y * box.width * 4);
  }
  return { width: box.width, height: box.height, data };
}
