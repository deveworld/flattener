import { bilinear, placedPoint } from './geometry';
import type { OrientConfig, Pixels, Placement } from './types';

/** Port of scan.orient.page_views, including energy-selected text patches. */
export function orientationViews(image: Pixels, placement: Placement, config: OrientConfig) {
  const gray = new Float32Array(image.width * image.height);
  for (let i = 0; i < gray.length; i++) {
    // PIL's fixed-point ITU-R 601 conversion, including its rounding.
    gray[i] =
      (image.data[i * 4] * 19595 +
        image.data[i * 4 + 1] * 38470 +
        image.data[i * 4 + 2] * 7471 +
        32768) >>
      16;
  }
  const levels = new Map<number, { data: Float32Array; width: number; height: number }>();
  levels.set(1, { data: gray, width: image.width, height: image.height });
  function level(density: number) {
    const factor = Math.max(
      1,
      Math.min(
        Math.floor(density / 1.5),
        Math.floor(image.width / 2),
        Math.floor(image.height / 2),
      ),
    );
    if (!levels.has(factor)) {
      const width = Math.floor(image.width / factor),
        height = Math.floor(image.height / factor);
      const data = new Float32Array(width * height);
      for (let y = 0; y < height; y++)
        for (let x = 0; x < width; x++) {
          let sum = 0;
          for (let j = 0; j < factor; j++)
            for (let i = 0; i < factor; i++)
              sum += gray[(y * factor + j) * image.width + x * factor + i];
          data[y * width + x] = sum / (factor * factor);
        }
      levels.set(factor, { data, width, height });
    }
    return { ...levels.get(factor)!, factor };
  }
  const w = placement.width,
    h = placement.height;
  const inside = (x: number, y: number) =>
    x >= -1e-9 && x <= w + 1e-9 && y >= -1e-9 && y <= h + 1e-9;
  function sampler(density: number) {
    const source = level(density);
    return (x: number, y: number) => {
      if (!inside(x, y)) return config.pad_value;
      const q = placedPoint(placement, (2 * x) / w - 1, (2 * y) / h - 1);
      // Quantize the normalized grid to float32 just as grid_sample does.
      const nx = Math.fround(((q[0] + 1) * image.width) / (source.width * source.factor) - 1);
      const ny = Math.fround(((q[1] + 1) * image.height) / (source.height * source.factor) - 1);
      return Math.floor(
        bilinear(
          source.data,
          source.width,
          source.height,
          ((nx + 1) * source.width) / 2 - 0.5,
          ((ny + 1) * source.height) / 2 - 0.5,
        ) + 0.5,
      );
    };
  }
  const side = config.global_size,
    scale = side / Math.max(w, h);
  const left = (side - w * scale) / 2,
    top = (side - h * scale) / 2;
  const global = new Float32Array(side * side),
    mask = new Uint8Array(side * side);
  const read = sampler(1 / scale);
  for (let y = 0; y < side; y++)
    for (let x = 0; x < side; x++) {
      const px = (x + 0.5 - left) / scale,
        py = (y + 0.5 - top) / scale,
        i = y * side + x;
      mask[i] = +inside(px, py);
      global[i] = read(px, py);
    }
  const size = config.energy_grid,
    count = new Uint32Array(size * size);
  const total = new Float64Array(count.length),
    squares = new Float64Array(count.length);
  for (let y = 1; y < side - 1; y++)
    for (let x = 1; x < side - 1; x++) {
      const i = y * side + x;
      if (!mask[i] || !mask[i - 1] || !mask[i + 1] || !mask[i - side] || !mask[i + side]) continue;
      const magnitude = Math.hypot(
        global[i + 1] - global[i - 1],
        global[i + side] - global[i - side],
      );
      const c = Math.max(0, Math.min(size - 1, Math.floor(((x + 0.5 - left) / scale / w) * size)));
      const r = Math.max(0, Math.min(size - 1, Math.floor(((y + 0.5 - top) / scale / h) * size)));
      const cell = r * size + c;
      count[cell]++;
      total[cell] += magnitude;
      squares[cell] += magnitude * magnitude;
    }
  const energy = Array.from(count, (n, i) =>
    n < 4 ? 0 : Math.fround(Math.sqrt(Math.max(0, squares[i] / n - (total[i] / n) ** 2))),
  );
  const candidates: number[] = [];
  for (let y = 1; y < size - 1; y++)
    for (let x = 1; x < size - 1; x++) candidates.push(y * size + x);
  const cells = candidates.sort((a, b) => energy[b] - energy[a]).slice(0, config.patches);
  const patch = config.patch_size,
    step = Math.max(w, h) / config.page_long_side,
    sample = sampler(step);
  const patches = new Float32Array(config.patches * patch * patch);
  for (let k = 0; k < cells.length; k++) {
    const cx = (((cells[k] % size) + 0.5) * w) / size,
      cy = ((Math.floor(cells[k] / size) + 0.5) * h) / size;
    for (let y = 0; y < patch; y++)
      for (let x = 0; x < patch; x++) {
        patches[k * patch * patch + y * patch + x] = sample(
          cx + (x + 0.5 - patch / 2) * step,
          cy + (y + 0.5 - patch / 2) * step,
        );
      }
  }
  return { global, patches, cells, energy };
}
