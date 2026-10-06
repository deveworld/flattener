import type { Pixels } from './types';

// Unsharp masking of the rendered page, as the Python scanner's ``scan.sharpen``: the
// luma detail (Y minus its Gaussian blur) is added to R, G and B alike, so colors keep
// their hue; detail within THRESHOLD levels (paper grain, JPEG noise) is left alone, and
// values are clipped to 0..255.

const SIGMA = 1.5;
const THRESHOLD = 2;

/** Strength for a render that samples ``density`` source pixels per output pixel (by
 * area): off when the page is enlarged, stronger the more it is reduced. */
export function sharpenAmount(density: number): number {
  if (!Number.isFinite(density) || density < 1) return 0;
  return Math.min(1, 0.3 + 0.5 * (density - 1));
}

function kernel(sigma: number): Float32Array {
  const radius = Math.ceil(sigma * 3),
    weights = new Float32Array(radius * 2 + 1);
  let total = 0;
  for (let i = -radius; i <= radius; i++) {
    weights[i + radius] = Math.exp(-(i * i) / (2 * sigma * sigma));
    total += weights[i + radius];
  }
  return weights.map((w) => w / total);
}

export function sharpenPixels(image: Pixels, amount: number): Pixels {
  if (amount <= 0) return image;
  const { width, height, data } = image;
  const luma = new Float32Array(width * height);
  for (let i = 0; i < luma.length; i++)
    luma[i] = (data[i * 4] * 299 + data[i * 4 + 1] * 587 + data[i * 4 + 2] * 114) / 1000;
  const weights = kernel(SIGMA),
    radius = (weights.length - 1) / 2;
  const across = new Float32Array(luma.length),
    blurred = new Float32Array(luma.length);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++)
        sum += luma[y * width + Math.min(width - 1, Math.max(0, x + k))] * weights[k + radius];
      across[y * width + x] = sum;
    }
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let sum = 0;
      for (let k = -radius; k <= radius; k++)
        sum += across[Math.min(height - 1, Math.max(0, y + k)) * width + x] * weights[k + radius];
      blurred[y * width + x] = sum;
    }
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < luma.length; i++) {
    const detail = luma[i] - blurred[i];
    const step = Math.abs(detail) <= THRESHOLD ? 0 : Math.round(detail * amount);
    out[i * 4] = data[i * 4] + step;
    out[i * 4 + 1] = data[i * 4 + 1] + step;
    out[i * 4 + 2] = data[i * 4 + 2] + step;
    out[i * 4 + 3] = data[i * 4 + 3];
  }
  return { width, height, data: out };
}
