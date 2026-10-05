import { bilinear, gridPoint } from './geometry';
import type { Grid, Mask, PaperField, Pixels } from './types';

const clamp = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, n));

export function emptyMask(width: number, height: number, maxSide = 512): Mask {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  width = Math.max(1, Math.round(width * scale));
  height = Math.max(1, Math.round(height * scale));
  return { width, height, data: new Uint8Array(width * height) };
}

export function maskCount(mask: Mask): number {
  return mask.data.reduce((n, v) => n + (v ? 1 : 0), 0);
}

function validateMask(mask: Mask, maxCoverage = 0.2) {
  if (
    !Number.isInteger(mask.width) ||
    !Number.isInteger(mask.height) ||
    mask.width < 1 ||
    mask.height < 1 ||
    mask.width > 2048 ||
    mask.height > 2048 ||
    mask.data.length !== mask.width * mask.height
  )
    throw new Error('The removal selection is invalid. Clear it and mark the area again.');
  if (maskCount(mask) > mask.data.length * maxCoverage)
    throw new Error('Select a smaller area to remove (less than 20% of the page).');
}

function selected(mask: Mask | undefined, x: number, y: number, w: number, h: number) {
  if (!mask) return false;
  const mx = clamp(Math.floor(((x + 0.5) * mask.width) / w), 0, mask.width - 1);
  const my = clamp(Math.floor(((y + 0.5) * mask.height) / h), 0, mask.height - 1);
  return !!mask.data[my * mask.width + mx];
}

/** The actual gate probability, resampled through the final dewarp map. */
export function pageConfidence(
  grid: Grid,
  probability: Float32Array,
  gw: number,
  gh: number,
  width: number,
  height: number,
): Float32Array {
  const result = new Float32Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const [sx, sy] = gridPoint(grid, (x + 0.5) / width, (y + 0.5) / height);
      result[y * width + x] = bilinear(
        probability,
        gw,
        gh,
        ((sx + 1) * gw) / 2 - 0.5,
        ((sy + 1) * gh) / 2 - 0.5,
      );
    }
  return result;
}

/** Proposals only: the page gate is not a dedicated hand segmenter.
 * Ignore the uncertain outer rim and interior holes; keep localized edge intrusions.
 * Appearance must also differ from nearby paper, preventing the gate's uncertain
 * blank margins and neutral shadows from becoming removal proposals. Colors are
 * relative to this document's paper, not a fixed skin-tone range.
 * Users review the proposal before any pixels are removed.
 */
export function suggestFingers(
  confidence: Float32Array,
  width: number,
  height: number,
  image?: Pixels,
): Mask {
  if (confidence.length !== width * height) throw new Error('Invalid page confidence map.');
  const mask = emptyMask(width, height, Math.max(width, height));
  const visited = new Uint8Array(width * height),
    queue = new Int32Array(width * height);
  const short = Math.min(width, height),
    rim = Math.max(1, Math.floor(short * 0.018));
  const edgeDistance = (x: number, y: number) => Math.min(x, y, width - 1 - x, height - 1 - y);
  const appearance = new Uint8Array(width * height).fill(1);
  if (image) {
    const excluded = { width, height, data: Uint8Array.from(confidence, (p) => (p < 0.5 ? 1 : 0)) };
    const field = estimatePaper(image, excluded);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const ratios = [];
        for (let c = 0; c < 3; c++) {
          const value = bilinear(
            image.data,
            image.width,
            image.height,
            ((x + 0.5) * image.width) / width - 0.5,
            ((y + 0.5) * image.height) / height - 0.5,
            4,
            c,
          );
          ratios.push(value / Math.max(60, paperAt(field, x, y, width, height, c)));
        }
        // Relative chroma is stable as a finger gets darker toward its boundary.
        // An absolute channel difference would only retain its brighter center.
        const mean = (ratios[0] + ratios[1] + ratios[2]) / 3;
        appearance[y * width + x] =
          (Math.max(...ratios) - Math.min(...ratios)) / Math.max(0.08, mean) > 0.18 &&
          Math.hypot(ratios[0] - 1, ratios[1] - 1, ratios[2] - 1) > 0.2
            ? 1
            : 0;
      }
  }
  for (let i = 0; i < confidence.length; i++) {
    const x = i % width,
      y = Math.floor(i / width);
    if (
      visited[i] ||
      !appearance[i] ||
      confidence[i] >= 0.45 ||
      !Number.isFinite(confidence[i]) ||
      edgeDistance(x, y) < rim
    )
      continue;
    let head = 0,
      tail = 1,
      near = Infinity,
      deep = 0;
    let left = width,
      right = 0,
      top = height,
      bottom = 0;
    queue[0] = i;
    visited[i] = 1;
    while (head < tail) {
      const p = queue[head++],
        px = p % width,
        py = Math.floor(p / width);
      const d = edgeDistance(px, py);
      near = Math.min(near, d);
      deep = Math.max(deep, d);
      left = Math.min(left, px);
      right = Math.max(right, px);
      top = Math.min(top, py);
      bottom = Math.max(bottom, py);
      for (const n of [
        px ? p - 1 : -1,
        px + 1 < width ? p + 1 : -1,
        py ? p - width : -1,
        py + 1 < height ? p + width : -1,
      ]) {
        if (
          n < 0 ||
          visited[n] ||
          !appearance[n] ||
          !(confidence[n] < 0.45) ||
          edgeDistance(n % width, Math.floor(n / width)) < rim
        )
          continue;
        visited[n] = 1;
        queue[tail++] = n;
      }
    }
    const bw = right - left + 1,
      bh = bottom - top + 1;
    if (
      near > short * 0.04 ||
      deep < short * 0.055 ||
      deep > short * 0.3 ||
      tail < width * height * 0.0008 ||
      tail > width * height * 0.12 ||
      bw > width * 0.6 ||
      bh > height * 0.6 ||
      tail / (bw * bh) < 0.22
    )
      continue;
    for (let j = 0; j < tail; j++) mask.data[queue[j]] = 1;
    // Extend accepted intrusions through the discarded rim to the page edge.
    if (left <= rim + 1)
      for (let yy = top; yy <= bottom; yy++)
        if (mask.data[yy * width + left])
          for (let xx = 0; xx < left; xx++) mask.data[yy * width + xx] = 1;
    if (right >= width - rim - 2)
      for (let yy = top; yy <= bottom; yy++)
        if (mask.data[yy * width + right])
          for (let xx = right; xx < width; xx++) mask.data[yy * width + xx] = 1;
    if (top <= rim + 1)
      for (let xx = left; xx <= right; xx++)
        if (mask.data[top * width + xx])
          for (let yy = 0; yy < top; yy++) mask.data[yy * width + xx] = 1;
    if (bottom >= height - rim - 2)
      for (let xx = left; xx <= right; xx++)
        if (mask.data[bottom * width + xx])
          for (let yy = bottom; yy < height; yy++) mask.data[yy * width + xx] = 1;
  }
  // Extend confirmed components over their connected colored boundary, including
  // softer model confidence at the finger tip. Do not grow into neutral paper/text.
  let head = 0,
    tail = 0;
  for (let i = 0; i < mask.data.length; i++) if (mask.data[i]) queue[tail++] = i;
  while (image && head < tail) {
    const p = queue[head++],
      x = p % width,
      y = Math.floor(p / width);
    for (const n of [
      x ? p - 1 : -1,
      x + 1 < width ? p + 1 : -1,
      y ? p - width : -1,
      y + 1 < height ? p + width : -1,
    ]) {
      if (
        n < 0 ||
        mask.data[n] ||
        !appearance[n] ||
        edgeDistance(n % width, Math.floor(n / width)) > short * 0.3
      )
        continue;
      mask.data[n] = 1;
      queue[tail++] = n;
    }
  }
  const grown = mask.data.slice(),
    radius = Math.max(1, Math.round(short * 0.008));
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      if (!mask.data[y * width + x]) continue;
      for (let dy = -radius; dy <= radius; dy++)
        for (let dx = -radius; dx <= radius; dx++)
          if (
            dx * dx + dy * dy <= radius * radius &&
            x + dx >= 0 &&
            x + dx < width &&
            y + dy >= 0 &&
            y + dy < height
          )
            grown[(y + dy) * width + x + dx] = 1;
    }
  mask.data = grown;
  if (maskCount(mask) > mask.data.length * 0.18) mask.data.fill(0);
  return mask;
}

/** Smooth local paper color, estimated from the bright, low-chroma pixels in tiles.
 * This is a lightweight background-estimation method, not a reproduction of a neural
 * shadow remover. Excluded pixels never contribute to the paper estimate.
 */
export function estimatePaper(image: Pixels, excluded?: Mask): PaperField {
  if (excluded) validateMask(excluded, 1);
  const scale = Math.min(1, 512 / Math.max(image.width, image.height));
  const sw = Math.max(1, Math.round(image.width * scale)),
    sh = Math.max(1, Math.round(image.height * scale));
  const cols = Math.max(2, Math.round(sw / 32)),
    rows = Math.max(2, Math.round(sh / 32));
  const data = new Float32Array(cols * rows * 3),
    valid = new Uint8Array(cols * rows);
  const samples: number[][] = Array.from({ length: cols * rows }, () => []);
  for (let y = 0; y < sh; y++)
    for (let x = 0; x < sw; x++) {
      if (selected(excluded, x, y, sw, sh)) continue;
      const sx = ((x + 0.5) * image.width) / sw - 0.5,
        sy = ((y + 0.5) * image.height) / sh - 0.5;
      const r = bilinear(image.data, image.width, image.height, sx, sy, 4, 0);
      const g = bilinear(image.data, image.width, image.height, sx, sy, 4, 1);
      const b = bilinear(image.data, image.width, image.height, sx, sy, 4, 2);
      const hi = Math.max(r, g, b),
        lo = Math.min(r, g, b);
      if (hi < 55 || hi - lo > hi * 0.45) continue;
      const cell =
        Math.min(rows - 1, Math.floor((y * rows) / sh)) * cols +
        Math.min(cols - 1, Math.floor((x * cols) / sw));
      samples[cell].push(r, g, b, 0.2126 * r + 0.7152 * g + 0.0722 * b);
    }
  for (let cell = 0; cell < samples.length; cell++) {
    const values = samples[cell],
      lumas = [];
    for (let j = 3; j < values.length; j += 4) lumas.push(values[j]);
    if (lumas.length < 8) continue;
    lumas.sort((a, b) => a - b);
    const threshold = lumas[Math.floor(lumas.length * 0.65)];
    let count = 0;
    for (let j = 0; j < values.length; j += 4)
      if (values[j + 3] >= threshold) {
        for (let c = 0; c < 3; c++) data[cell * 3 + c] += values[j + c];
        count++;
      }
    for (let c = 0; c < 3; c++) data[cell * 3 + c] /= count;
    valid[cell] = 1;
  }
  const known = Array.from(valid.keys()).filter((i) => valid[i]);
  if (!known.length) {
    // Low confidence: retain dark/graphic pages rather than amplify noise.
    data.fill(248);
    return { width: cols, height: rows, data };
  }
  for (let i = 0; i < valid.length; i++)
    if (!valid[i]) {
      let total = 0;
      for (const j of known) {
        const d =
          ((i % cols) - (j % cols)) ** 2 + (Math.floor(i / cols) - Math.floor(j / cols)) ** 2;
        const w = 1 / (d * d + 0.01);
        total += w;
        for (let c = 0; c < 3; c++) data[i * 3 + c] += data[j * 3 + c] * w;
      }
      for (let c = 0; c < 3; c++) data[i * 3 + c] /= total;
    }
  // A small spatial filter suppresses tile transitions without blurring the document.
  const smooth = data.slice();
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < cols; x++)
      for (let c = 0; c < 3; c++) {
        let sum = 0,
          weight = 0;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++) {
            const w = (dx === 0 ? 2 : 1) * (dy === 0 ? 2 : 1);
            sum +=
              data[(clamp(y + dy, 0, rows - 1) * cols + clamp(x + dx, 0, cols - 1)) * 3 + c] * w;
            weight += w;
          }
        smooth[(y * cols + x) * 3 + c] = sum / weight;
      }
  return { width: cols, height: rows, data: smooth };
}

export function paperAt(
  field: PaperField,
  x: number,
  y: number,
  width: number,
  height: number,
  c: number,
) {
  return bilinear(
    field.data,
    field.width,
    field.height,
    ((x + 0.5) * field.width) / width - 0.5,
    ((y + 0.5) * field.height) / height - 0.5,
    3,
    c,
  );
}

/** Non-destructive edits: every invocation starts with the same uncorrected scan. */
export function cleanDocument(
  image: Pixels,
  mask: Mask,
  cleanColor: boolean,
  field?: PaperField,
): Pixels {
  validateMask(mask);
  const data = image.data.slice(),
    { width, height } = image;
  const hasMask = maskCount(mask) > 0;
  if (!cleanColor && !hasMask) return { width, height, data };
  field ??= estimatePaper(image, mask);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4,
        remove = hasMask && selected(mask, x, y, width, height);
      if (!cleanColor && !remove) continue;
      for (let c = 0; c < 3; c++) {
        const paper = paperAt(field, x, y, width, height, c);
        // Mask-only fill leaves every unselected source pixel exactly unchanged.
        const value = remove ? paper : image.data[i + c];
        data[i + c] = cleanColor ? value * clamp(248 / Math.max(paper, 70), 0.85, 2.5) : value;
      }
    }
  return { width, height, data };
}

export function rotateMask(mask: Mask, turns: number): Mask {
  let result = { ...mask, data: mask.data.slice() };
  for (let k = 0; k < ((turns % 4) + 4) % 4; k++) {
    const { width, height, data } = result,
      next = new Uint8Array(data.length);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) next[x * height + height - 1 - y] = data[y * width + x];
    result = { width: height, height: width, data: next };
  }
  return result;
}

export function rotatePixels(image: Pixels, turns: number): Pixels {
  let result = image;
  for (let k = 0; k < ((turns % 4) + 4) % 4; k++) {
    const { width, height, data } = result,
      next = new Uint8ClampedArray(data.length);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++)
        next.set(
          data.subarray((y * width + x) * 4, (y * width + x) * 4 + 4),
          (x * height + height - 1 - y) * 4,
        );
    result = { width: height, height: width, data: next };
  }
  return result;
}
