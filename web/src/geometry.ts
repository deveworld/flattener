import type { Grid, Pixels, Placement } from './types';

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
export const median = (a: number[]) => a.sort((x, y) => x - y)[Math.floor((a.length - 1) / 2)];

export function whole(image: { width: number; height: number }): Placement {
  return { matrix: [1, 0, 0, 1], offset: [0, 0], width: image.width, height: image.height };
}

export function boxPlacement(box: number[], image: Pixels): Placement {
  const [l, t, r, b] = box;
  return {
    matrix: [(r - l) / image.width, 0, 0, (b - t) / image.height],
    offset: [(l + r) / image.width - 1, (t + b) / image.height - 1],
    width: r - l,
    height: b - t,
  };
}

export function turnPlacement(p: Placement, k: number): Placement {
  let result = p;
  for (let i = 0; i < ((k % 4) + 4) % 4; i++) {
    const [a, b, c, d] = result.matrix;
    result = { ...result, matrix: [b, -a, d, -c], width: result.height, height: result.width };
  }
  return result;
}

export function placedPoint(p: Placement, x: number, y: number): [number, number] {
  const [a, b, c, d] = p.matrix;
  return [a * x + b * y + p.offset[0], c * x + d * y + p.offset[1]];
}

export function bilinear(
  data: ArrayLike<number>,
  width: number,
  height: number,
  x: number,
  y: number,
  channels = 1,
  channel = 0,
): number {
  x = clamp(x, 0, width - 1);
  y = clamp(y, 0, height - 1);
  const x0 = Math.floor(x),
    y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, width - 1),
    y1 = Math.min(y0 + 1, height - 1);
  const dx = x - x0,
    dy = y - y0;
  const top =
    data[(y0 * width + x0) * channels + channel] * (1 - dx) +
    data[(y0 * width + x1) * channels + channel] * dx;
  const bottom =
    data[(y1 * width + x0) * channels + channel] * (1 - dx) +
    data[(y1 * width + x1) * channels + channel] * dx;
  return top * (1 - dy) + bottom * dy;
}

/** PyTorch grid_sample: pixel-edge normalized coordinates, border padding. */
export function modelInput(
  image: Pixels,
  p: Placement,
  height: number,
  width: number,
): Float32Array {
  const data = new Float32Array(3 * width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const q = placedPoint(p, (2 * (x + 0.5)) / width - 1, (2 * (y + 0.5)) / height - 1);
      const sx = ((Math.fround(q[0]) + 1) * image.width) / 2 - 0.5;
      const sy = ((Math.fround(q[1]) + 1) * image.height) / 2 - 0.5;
      for (let c = 0; c < 3; c++) {
        data[c * width * height + y * width + x] =
          bilinear(image.data, image.width, image.height, sx, sy, 4, c) / 255;
      }
    }
  return data;
}

/** Separable triangle filter used by interpolate(bilinear, antialias=True). */
export function gateInput(image: Pixels, height: number, width: number): Float32Array {
  function taps(input: number, output: number) {
    const scale = input / output,
      support = Math.max(scale, 1);
    return Array.from({ length: output }, (_, i) => {
      const center = (i + 0.5) * scale,
        list: [number, number][] = [];
      let sum = 0;
      for (
        let j = Math.max(0, Math.floor(center - support));
        j < Math.min(input, Math.ceil(center + support));
        j++
      ) {
        const weight = Math.max(0, 1 - Math.abs((j + 0.5 - center) / support));
        if (weight) {
          list.push([j, weight]);
          sum += weight;
        }
      }
      return list.map(([j, w]) => [j, w / sum] as const);
    });
  }
  const xt = taps(image.width, width),
    yt = taps(image.height, height);
  const temp = new Float32Array(image.height * width * 3),
    out = new Float32Array(width * height * 3);
  for (let y = 0; y < image.height; y++)
    for (let x = 0; x < width; x++) {
      let red = 0,
        green = 0,
        blue = 0;
      for (const [j, w] of xt[x]) {
        const i = (y * image.width + j) * 4;
        red += (image.data[i] / 255) * w;
        green += (image.data[i + 1] / 255) * w;
        blue += (image.data[i + 2] / 255) * w;
      }
      const i = (y * width + x) * 3;
      temp[i] = red;
      temp[i + 1] = green;
      temp[i + 2] = blue;
    }
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let red = 0,
        green = 0,
        blue = 0;
      for (const [j, w] of yt[y]) {
        const i = (j * width + x) * 3;
        red += temp[i] * w;
        green += temp[i + 1] * w;
        blue += temp[i + 2] * w;
      }
      const i = y * width + x;
      out[i] = red;
      out[width * height + i] = green;
      out[2 * width * height + i] = blue;
    }
  return out;
}

export function mirroredInput(data: Float32Array, height: number, width: number): Float32Array {
  const out = new Float32Array(data.length);
  for (let c = 0; c < 3; c++)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        out[c * width * height + y * width + x] =
          data[c * width * height + y * width + width - 1 - x];
      }
  return out;
}

export function fromCHW(data: Float32Array, rows: number, cols: number): Grid {
  const result = new Float64Array(rows * cols * 2),
    n = rows * cols;
  for (let i = 0; i < n; i++) {
    result[2 * i] = data[i];
    result[2 * i + 1] = data[n + i];
  }
  return { rows, cols, data: result };
}

/** The mean of a grid and its mirrored-input prediction, or the grid alone when the two
 * disagree (half the median node distance above 0.03). ``info`` receives that distance and
 * the half distance at every node, a map of where the two predictions disagree. */
export function blendFlip(
  grid: Grid,
  flipped: Grid,
  info: { gap?: number; gaps?: number[] } = {},
): Grid {
  const mirror = new Float64Array(grid.data.length),
    gaps = [];
  for (let y = 0; y < grid.rows; y++)
    for (let x = 0; x < grid.cols; x++) {
      const a = (y * grid.cols + x) * 2,
        b = (y * grid.cols + grid.cols - 1 - x) * 2;
      mirror[a] = -flipped.data[b];
      mirror[a + 1] = flipped.data[b + 1];
      gaps.push(Math.hypot(grid.data[a] - mirror[a], grid.data[a + 1] - mirror[a + 1]) / 2);
    }
  info.gap = median(gaps);
  info.gaps = gaps;
  if (info.gap > 0.03) return grid;
  return { ...grid, data: grid.data.map((v, i) => Math.fround((v + mirror[i]) / 2)) };
}

export function compose(grid: Grid, placement: Placement): Grid {
  const data = new Float64Array(grid.data.length);
  for (let i = 0; i < data.length; i += 2) {
    const q = placedPoint(placement, grid.data[i], grid.data[i + 1]);
    data[i] = q[0];
    data[i + 1] = q[1];
  }
  return { ...grid, data };
}

export function outline(grid: Grid, width: number, height: number): [number, number][] {
  const points: [number, number][] = [];
  for (let r = 0; r < grid.rows; r++)
    for (let c = 0; c < grid.cols; c++) {
      if (r > 0 && r < grid.rows - 1 && c > 0 && c < grid.cols - 1) continue;
      const i = (r * grid.cols + c) * 2;
      points.push([((grid.data[i] + 1) * width) / 2, ((grid.data[i + 1] + 1) * height) / 2]);
    }
  return points;
}

export function pageAngle(grid: Grid, width: number, height: number): number | null {
  const angles: number[] = [],
    xs: number[] = [],
    ys: number[] = [];
  for (const r of [0, grid.rows - 1])
    for (let c = 0; c < grid.cols - 1; c++) {
      const i = (r * grid.cols + c) * 2;
      const dx = ((grid.data[i + 2] - grid.data[i]) * width) / 2;
      const dy = ((grid.data[i + 3] - grid.data[i + 1]) * height) / 2;
      const length = Math.hypot(dx, dy);
      if (length > 0) {
        xs.push(dx / length);
        ys.push(dy / length);
        angles.push(Math.atan2(dy, dx));
      }
    }
  if (!angles.length || Math.hypot(mean(xs), mean(ys)) < 0.5) return null;
  const center = Math.atan2(mean(ys), mean(xs));
  const offsets = angles
    .map((a) => Math.atan2(Math.sin(a - center), Math.cos(a - center)))
    .sort((a, b) => a - b);
  const drop = Math.floor(offsets.length / 4);
  return ((center + mean(offsets.slice(drop, offsets.length - drop))) * 180) / Math.PI;
}

export function tightPlacement(grid: Grid, image: Pixels, angle: number, margin = 0.04): Placement {
  const theta = (angle * Math.PI) / 180,
    cos = Math.cos(theta),
    sin = Math.sin(theta);
  const frame = outline(grid, image.width, image.height).map(([x, y]) => [
    cos * x + sin * y,
    -sin * x + cos * y,
  ]);
  const lo = [Math.min(...frame.map((v) => v[0])), Math.min(...frame.map((v) => v[1]))];
  const hi = [Math.max(...frame.map((v) => v[0])), Math.max(...frame.map((v) => v[1]))];
  const width = (hi[0] - lo[0]) * (1 + 2 * margin),
    height = (hi[1] - lo[1]) * (1 + 2 * margin);
  const cx = (lo[0] + hi[0]) / 2,
    cy = (lo[1] + hi[1]) / 2;
  return {
    matrix: [
      (cos * width) / image.width,
      (-sin * height) / image.width,
      (sin * width) / image.height,
      (cos * height) / image.height,
    ],
    offset: [
      (2 * (cos * cx - sin * cy)) / image.width - 1,
      (2 * (sin * cx + cos * cy)) / image.height - 1,
    ],
    width,
    height,
  };
}

/** 4-connected components of gate cells at or above 0.5, largest first (cell indices). */
export function pageComponents(probability: Float32Array, height: number, width: number) {
  const visited = new Uint8Array(width * height),
    queue = new Int32Array(width * height);
  const found: number[][] = [];
  for (let start = 0; start < probability.length; start++) {
    if (visited[start] || probability[start] < 0.5) continue;
    let read = 0,
      write = 1;
    queue[0] = start;
    visited[start] = 1;
    while (read < write) {
      const i = queue[read++],
        x = i % width,
        y = Math.floor(i / width);
      for (const n of [
        x > 0 ? i - 1 : -1,
        x < width - 1 ? i + 1 : -1,
        y > 0 ? i - width : -1,
        y < height - 1 ? i + width : -1,
      ]) {
        if (n >= 0 && !visited[n] && probability[n] >= 0.5) {
          visited[n] = 1;
          queue[write++] = n;
        }
      }
    }
    found.push(Array.from(queue.subarray(0, write)));
  }
  return found.sort((a, b) => b.length - a.length);
}

/** Pixel box of gate cells, grown by 4% of its size per side. */
function cellsBox(cells: number[], height: number, width: number, image: Pixels): number[] {
  let l = width,
    r = 0,
    t = height,
    b = 0;
  for (const i of cells) {
    const x = i % width,
      y = Math.floor(i / width);
    l = Math.min(l, x);
    r = Math.max(r, x + 1);
    t = Math.min(t, y);
    b = Math.max(b, y + 1);
  }
  const mx = (r - l) * 0.04,
    my = (b - t) * 0.04;
  return [
    Math.floor((Math.max(0, l - mx) / width) * image.width),
    Math.floor((Math.max(0, t - my) / height) * image.height),
    Math.ceil((Math.min(width, r + mx) / width) * image.width),
    Math.ceil((Math.min(height, b + my) / height) * image.height),
  ];
}

/** Largest connected page component, used only as a crop hint. */
export function pageBox(
  probability: Float32Array,
  height: number,
  width: number,
  image: Pixels,
): number[] | null {
  const [best] = pageComponents(probability, height, width);
  if (!best || best.length < width * height * 0.03) return null;
  return cellsBox(best, height, width, image);
}

/** The grid's boundary nodes in pixels, in order around the page. */
export function outlinePolygon(grid: Grid, width: number, height: number): [number, number][] {
  const point = (r: number, c: number): [number, number] => {
    const i = (r * grid.cols + c) * 2;
    return [((grid.data[i] + 1) * width) / 2, ((grid.data[i + 1] + 1) * height) / 2];
  };
  const points: [number, number][] = [];
  for (let c = 0; c < grid.cols; c++) points.push(point(0, c));
  for (let r = 1; r < grid.rows; r++) points.push(point(r, grid.cols - 1));
  for (let c = grid.cols - 2; c >= 0; c--) points.push(point(grid.rows - 1, c));
  for (let r = grid.rows - 2; r > 0; r--) points.push(point(r, 0));
  return points;
}

function insidePolygon(polygon: [number, number][], x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i],
      [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export const SPLIT_RULES = {
  dominance: 0.85, // a largest piece below this share of the page cells: the gate split the page
  minArea: 0.01, // pieces smaller than this share of the grid are ignored
  minInside: 0.5, // a piece belongs to the page when this share of it lies in the outline
  minPrecision: 0.5, // the outline is used when this share of it is page cells (any piece)
};

/** Whether the gate split one page into several pieces (``SPLIT_RULES.dominance``). */
export function gateSplit(components: number[][]): boolean {
  const total = components.reduce((sum, cells) => sum + cells.length, 0);
  return components.length > 1 && components[0].length / total < SPLIT_RULES.dominance;
}

/**
 * A crop box from a whole-photo dewarp grid when the gate split the page: the outline must
 * hold some gate pieces and be mostly page cells (the Python scanner's rule). Null otherwise.
 */
export function splitPageBox(
  grid: Grid,
  components: number[][],
  height: number,
  width: number,
  image: Pixels,
): number[] | null {
  const polygon = outlinePolygon(grid, image.width, image.height);
  if (polygon.some(([x, y]) => !Number.isFinite(x) || !Number.isFinite(y))) return null;
  const inside = new Uint8Array(width * height);
  let cells = 0;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      if (
        insidePolygon(
          polygon,
          ((x + 0.5) / width) * image.width,
          ((y + 0.5) / height) * image.height,
        )
      ) {
        inside[y * width + x] = 1;
        cells++;
      }
  if (!cells) return null;
  // A piece that also covers clutter outside the outline (tissues touching a cover's
  // title) is still paper inside it, so every piece's cells count for the precision.
  let held = false,
    paper = 0;
  for (const piece of components) {
    const within = piece.reduce((sum, i) => sum + inside[i], 0);
    paper += within;
    if (piece.length >= width * height * SPLIT_RULES.minArea)
      held ||= within / piece.length >= SPLIT_RULES.minInside;
  }
  if (!held || paper / cells < SPLIT_RULES.minPrecision) return null;
  const xs = polygon.map(([x]) => x),
    ys = polygon.map(([, y]) => y);
  const [l, r, t, b] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const mx = (r - l) * 0.04,
    my = (b - t) * 0.04;
  const box = [
    Math.floor(Math.max(0, l - mx)),
    Math.floor(Math.max(0, t - my)),
    Math.ceil(Math.min(image.width, r + mx)),
    Math.ceil(Math.min(image.height, b + my)),
  ];
  return box[2] - box[0] >= 2 && box[3] - box[1] >= 2 ? box : null;
}

export function quarterTurn(grid: Grid, k: number): Grid {
  let result = grid;
  for (let n = 0; n < ((k % 4) + 4) % 4; n++) {
    const { rows, cols, data } = result,
      next = new Float64Array(data.length);
    for (let r = 0; r < cols; r++)
      for (let c = 0; c < rows; c++) {
        const to = (r * rows + c) * 2,
          from = (c * cols + cols - 1 - r) * 2;
        next[to] = data[from];
        next[to + 1] = data[from + 1];
      }
    result = { rows: cols, cols: rows, data: next };
  }
  return result;
}

export function gridPoint(grid: Grid, u: number, v: number): [number, number] {
  const x = clamp(u, 0, 1) * (grid.cols - 1),
    y = clamp(v, 0, 1) * (grid.rows - 1);
  return [
    bilinear(grid.data, grid.cols, grid.rows, x, y, 2, 0),
    bilinear(grid.data, grid.cols, grid.rows, x, y, 2, 1),
  ];
}

export function checkGrid(grid: Grid): string | null {
  if (!grid.data.every(Number.isFinite))
    return 'The page shape could not be estimated. Try a sharper photo.';
  const g = grid.data;
  for (let y = 0; y < grid.rows - 1; y++)
    for (let x = 0; x < grid.cols - 1; x++) {
      const a = (y * grid.cols + x) * 2,
        b = a + 2,
        c = a + grid.cols * 2,
        d = c + 2;
      for (const [u, v, p, q] of [
        [a, b, a, c],
        [a, b, b, d],
        [c, d, a, c],
        [c, d, b, d],
      ]) {
        const det = (g[v] - g[u]) * (g[q + 1] - g[p + 1]) - (g[v + 1] - g[u + 1]) * (g[q] - g[p]);
        if (det <= 1e-10)
          return 'The predicted page folds over itself. Use a photo with all four edges visible.';
      }
    }
  let outside = 0;
  for (let y = 0; y < 64; y++)
    for (let x = 0; x < 64; x++) {
      const [a, b] = gridPoint(grid, (x + 0.5) / 64, (y + 0.5) / 64);
      if (Math.abs(a) > 1 || Math.abs(b) > 1) outside++;
    }
  return outside / 4096 > 0.05
    ? 'Too much of the page falls outside the photo. Include the full page and try again.'
    : null;
}

export function outputSize(
  grid: Grid,
  source: { width: number; height: number },
  maxSide: number,
): [number, number] {
  const rows: number[] = [],
    cols: number[] = [];
  const distance = (a: number, b: number) =>
    Math.hypot(
      ((grid.data[a] - grid.data[b]) * source.width) / 2,
      ((grid.data[a + 1] - grid.data[b + 1]) * source.height) / 2,
    );
  for (let r = 0; r < grid.rows; r++) {
    let length = 0;
    for (let c = 1; c < grid.cols; c++)
      length += distance((r * grid.cols + c) * 2, (r * grid.cols + c - 1) * 2);
    rows.push(length);
  }
  for (let c = 0; c < grid.cols; c++) {
    let length = 0;
    for (let r = 1; r < grid.rows; r++)
      length += distance((r * grid.cols + c) * 2, ((r - 1) * grid.cols + c) * 2);
    cols.push(length);
  }
  let aspect = mean(rows) / mean(cols);
  if (
    !Number.isFinite(aspect) ||
    aspect < 0.15 ||
    aspect > 6 ||
    Math.max(...rows) < 16 ||
    mean(cols) < 16
  ) {
    throw new Error(
      'The page is too small or narrow to scan. Move closer and include all four edges.',
    );
  }
  const standards = [1 / Math.sqrt(2), 152 / 225, 128 / 188, 8.5 / 11];
  const candidates = [...standards, ...standards.map((x) => 1 / x)].sort(
    (a, b) => Math.abs(aspect - a) / a - Math.abs(aspect - b) / b,
  );
  if (Math.abs(aspect - candidates[0]) / candidates[0] <= 0.015) aspect = candidates[0];
  const continuousWidth = Math.max(...rows),
    continuousHeight = continuousWidth / aspect;
  const scale = Math.min(
    1,
    maxSide / Math.max(continuousWidth, continuousHeight),
    Math.sqrt(4_000_000 / (continuousWidth * continuousHeight)),
  );
  let width = Math.max(
    1,
    scale === 1 ? Math.round(continuousWidth) : Math.floor(continuousWidth * scale),
  );
  let height = Math.max(1, Math.round(width / aspect));
  while (width > 1 && (Math.max(width, height) > maxSide || width * height > 4_000_000)) {
    width--;
    height = Math.max(1, Math.round(width / aspect));
  }
  return [height, width];
}

/** Edge-aligned grid nodes, pixel-center evaluation; sample the original once. */
export function render(image: Pixels, grid: Grid, height: number, width: number): Pixels {
  const data = new Uint8ClampedArray(height * width * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const [qx, qy] = gridPoint(grid, (x + 0.5) / width, (y + 0.5) / height);
      const sx = ((Math.fround(qx) + 1) * image.width) / 2 - 0.5;
      const sy = ((Math.fround(qy) + 1) * image.height) / 2 - 0.5,
        i = (y * width + x) * 4;
      for (let c = 0; c < 3; c++)
        data[i + c] = Math.floor(
          bilinear(image.data, image.width, image.height, sx, sy, 4, c) + 0.5,
        );
      data[i + 3] = 255;
    }
  return { width, height, data };
}
