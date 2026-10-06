import type { PageOutline, Point } from './types';

export function pageOutline(
  probability: Float32Array,
  width: number,
  height: number,
): PageOutline | null {
  if (probability.length !== width * height || width < 2 || height < 2)
    throw new Error('Invalid page mask dimensions.');
  if (!probability.every(Number.isFinite)) return null;
  const visited = new Uint8Array(probability.length);
  const queue = new Int32Array(probability.length);
  const pieces: number[][] = [];
  for (let start = 0; start < probability.length; start++) {
    if (visited[start] || probability[start] < 0.55) continue;
    let head = 0,
      tail = 1;
    queue[0] = start;
    visited[start] = 1;
    while (head < tail) {
      const i = queue[head++],
        x = i % width,
        y = Math.floor(i / width);
      for (const next of [
        x > 0 ? i - 1 : -1,
        x + 1 < width ? i + 1 : -1,
        y > 0 ? i - width : -1,
        y + 1 < height ? i + width : -1,
      ]) {
        if (next < 0 || visited[next] || probability[next] < 0.55) continue;
        visited[next] = 1;
        queue[tail++] = next;
      }
    }
    pieces.push(Array.from(queue.subarray(0, tail)));
  }
  pieces.sort((a, b) => b.length - a.length);
  const largest = joinPieces(pieces, width, height);
  if (!largest || largest.length < probability.length * 0.025) return null;
  const confidence = largest.reduce((sum, i) => sum + probability[i], 0) / largest.length;
  if (confidence < 0.7) return null;

  let mask: Uint8Array = new Uint8Array(probability.length);
  let clipped = false;
  for (const i of largest) {
    mask[i] = 1;
    const x = i % width,
      y = Math.floor(i / width);
    if (x === 0 || y === 0 || x === width - 1 || y === height - 1) clipped = true;
  }
  // Small gaps (text near the edge, a fingertip) are closed; thin strips joining the page to
  // something beside it (a patterned cloth the gate half-takes for paper) are cut. The guide
  // then follows the page's own outline, concave edges included.
  mask = morph(morph(mask, width, height, 2, 'grow'), width, height, 2, 'shrink');
  mask = morph(morph(mask, width, height, 2, 'shrink'), width, height, 2, 'grow');
  // Trace cell corners: a corner is on the page when any of its four cells is, so the
  // outline runs along the mask's outer cell edges.
  const corners = new Uint8Array((width + 1) * (height + 1));
  for (let y = 0; y <= height; y++)
    for (let x = 0; x <= width; x++)
      for (const [cx, cy] of [
        [x - 1, y - 1],
        [x, y - 1],
        [x - 1, y],
        [x, y],
      ])
        if (cx >= 0 && cy >= 0 && cx < width && cy < height && mask[cy * width + cx])
          corners[y * (width + 1) + x] = 1;
  const traced = outerBoundary(corners, width + 1, height + 1);
  if (traced.length < 3) return null;
  const outer = simplify(traced, 1.25).map(([x, y]): Point => [x / width, y / height]);
  if (outer.length < 3) return null;
  return { points: resample(outer, 64), area: largest.length / probability.length, clipped };
}

/** ``mask`` grown or shrunk by ``radius`` cells (a square neighbourhood), keeping only the
 * largest connected part after a shrink. */
function morph(
  mask: Uint8Array,
  width: number,
  height: number,
  radius: number,
  mode: 'grow' | 'shrink',
): Uint8Array {
  const keep = mode === 'grow' ? 1 : 0;
  const pass = (source: Uint8Array, horizontal: boolean) => {
    const out = new Uint8Array(source.length);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        let hit = 1 - keep;
        for (let d = -radius; d <= radius && hit !== keep; d++) {
          const xx = horizontal ? x + d : x,
            yy = horizontal ? y : y + d;
          // Outside the frame counts as page when shrinking, so edge pages keep their edge.
          const v =
            xx < 0 || yy < 0 || xx >= width || yy >= height ? 1 - keep : source[yy * width + xx];
          if (v === keep) hit = keep;
        }
        out[y * width + x] = hit;
      }
    return out;
  };
  const result = pass(pass(mask, true), false);
  if (mode === 'grow') return result;
  // The largest connected part: a strip cut off by the shrink does not come back.
  const label = new Int32Array(result.length).fill(-1),
    queue = new Int32Array(result.length);
  let best = -1,
    bestSize = 0;
  for (let start = 0, n = 0; start < result.length; start++) {
    if (!result[start] || label[start] >= 0) continue;
    let head = 0,
      tail = 1;
    queue[0] = start;
    label[start] = n;
    while (head < tail) {
      const i = queue[head++],
        x = i % width;
      for (const next of [x > 0 ? i - 1 : -1, x + 1 < width ? i + 1 : -1, i - width, i + width])
        if (next >= 0 && next < result.length && result[next] && label[next] < 0) {
          label[next] = n;
          queue[tail++] = next;
        }
    }
    if (tail > bestSize) [best, bestSize] = [n, tail];
    n++;
  }
  return Uint8Array.from(label, (v) => (v === best && best >= 0 ? 1 : 0));
}

/** The outer boundary cells of a mask, in order (Moore neighbour tracing from the first cell
 * in reading order). */
function outerBoundary(mask: Uint8Array, width: number, height: number): Point[] {
  const at = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x] === 1;
  let start = -1;
  for (let i = 0; i < mask.length && start < 0; i++) if (mask[i]) start = i;
  if (start < 0) return [];
  const sx = start % width,
    sy = Math.floor(start / width);
  // Clockwise neighbours, starting west.
  const dirs = [
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
    [0, 1],
    [-1, 1],
  ];
  const points: Point[] = [[sx, sy]];
  let [x, y] = [sx, sy],
    from = 0; // the start was entered from the west (outside, as it is first in its row)
  for (let steps = 0; steps < 4 * mask.length; steps++) {
    let found = false;
    for (let k = 1; k <= 8; k++) {
      const d = (from + k) % 8,
        nx = x + dirs[d][0],
        ny = y + dirs[d][1];
      if (at(nx, ny)) {
        // Continue from the neighbour, searching from the direction we came from.
        from = (d + 4) % 8;
        [x, y] = [nx, ny];
        found = true;
        break;
      }
    }
    if (!found || (x === sx && y === sy)) break;
    points.push([x, y]);
  }
  return points;
}

/** Douglas–Peucker on a closed polygon with ``tolerance`` in the points' units: the ring is
 * cut at its first point and the point farthest from it, and each half is simplified. */
function simplify(points: Point[], tolerance: number): Point[] {
  if (points.length < 4) return points;
  const line = (chain: Point[]): Point[] => {
    const keep = new Uint8Array(chain.length);
    keep[0] = keep[chain.length - 1] = 1;
    const split = (a: number, b: number) => {
      const [ax, ay] = chain[a],
        [bx, by] = chain[b],
        span = Math.hypot(bx - ax, by - ay) || 1;
      let worst = -1,
        at = -1;
      for (let i = a + 1; i < b; i++) {
        const d = Math.abs((chain[i][0] - ax) * (by - ay) - (chain[i][1] - ay) * (bx - ax)) / span;
        if (d > worst) [worst, at] = [d, i];
      }
      if (worst > tolerance) {
        keep[at] = 1;
        split(a, at);
        split(at, b);
      }
    };
    split(0, chain.length - 1);
    return chain.filter((_, i) => keep[i]);
  };
  let far = 0;
  for (let i = 1; i < points.length; i++)
    if (
      Math.hypot(points[i][0] - points[0][0], points[i][1] - points[0][1]) >
      Math.hypot(points[far][0] - points[0][0], points[far][1] - points[0][1])
    )
      far = i;
  const first = line(points.slice(0, far + 1)),
    second = line([...points.slice(far), points[0]]);
  return [...first.slice(0, -1), ...second.slice(0, -1)];
}

/** The largest piece joined with pieces of the same page that a dark band (a printed
 * title bar, a ruled header) split from it: stacked above or below with a small gap, about
 * as wide and overlapping it across. The live guide cannot run the dewarp model, so it
 * joins only such clear cases. */
function joinPieces(pieces: number[][], width: number, height: number): number[] | null {
  if (!pieces.length) return null;
  const bounds = (cells: number[]) => {
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
    return { l, r, t, b };
  };
  const joined = pieces[0].slice();
  let box = bounds(joined);
  for (const piece of pieces.slice(1)) {
    if (piece.length < width * height * 0.02) break;
    const other = bounds(piece);
    const overlap = Math.min(box.r, other.r) - Math.max(box.l, other.l);
    const widths = [box.r - box.l, other.r - other.l];
    const gap = Math.max(other.t - box.b, box.t - other.b);
    if (
      overlap >= 0.6 * Math.min(...widths) &&
      Math.min(...widths) >= 0.7 * Math.max(...widths) &&
      gap <= 0.1 * height
    ) {
      joined.push(...piece);
      box = bounds(joined);
    }
  }
  return joined;
}

function resample(points: Point[], count: number): Point[] {
  const lengths = points.map((p, i) => {
    const q = points[(i + 1) % points.length];
    return Math.hypot(q[0] - p[0], q[1] - p[1]);
  });
  const perimeter = lengths.reduce((a, b) => a + b, 0);
  let edge = 0,
    start = 0;
  return Array.from({ length: count }, (_, i) => {
    const distance = (perimeter * i) / count;
    while (edge < lengths.length - 1 && start + lengths[edge] < distance) start += lengths[edge++];
    const t = (distance - start) / lengths[edge];
    const a = points[edge],
      b = points[(edge + 1) % points.length];
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  });
}

export function smoothOutline(previous: PageOutline | null, next: PageOutline): PageOutline {
  if (!previous || previous.points.length !== next.points.length) return next;
  const count = next.points.length;
  let offset = 0,
    best = Infinity;
  for (let shift = 0; shift < count; shift++) {
    const distance = previous.points.reduce((sum, p, i) => {
      const q = next.points[(i + shift) % count];
      return sum + (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
    }, 0);
    if (distance < best) {
      best = distance;
      offset = shift;
    }
  }
  if (Math.sqrt(best / count) > 0.06) return next;
  return {
    ...next,
    points: previous.points.map((p, i) => {
      const q = next.points[(i + offset) % count];
      return [p[0] + (q[0] - p[0]) * 0.65, p[1] + (q[1] - p[1]) * 0.65];
    }),
  };
}

export function containVideo(
  width: number,
  height: number,
  frameWidth: number,
  frameHeight: number,
) {
  const scale = Math.min(width / frameWidth, height / frameHeight);
  const w = frameWidth * scale,
    h = frameHeight * scale;
  return { width: w, height: h, left: (width - w) / 2, top: (height - h) / 2 };
}

export type Guidance = 'none' | 'shape' | 'clipped' | 'close' | 'small' | 'tilted' | 'ok';

/** Starting values, to be tuned on phones. */
export const GUIDANCE = {
  maxAspect: 4, // a page outline longer than this many times its width is not one page
  maxArea: 0.85, // share of the frame above which the page is too close
  minArea: 0.18, // share of the frame below which the page is too small
  maxSideRatio: 1.3, // opposite sides differing more than this: the camera is tilted
};

/** One capture hint from the live outline, in priority order: no page or an odd shape,
 * an edge cut by the frame, too close, too small, tilted, then ok. ``aspect`` is the
 * frame's width over its height. */
export function guidance(outline: PageOutline | null, aspect: number): Guidance {
  if (!outline || outline.points.length < 4) return 'none';
  const points = outline.points.map(([x, y]) => [x * aspect, y] as Point);
  const pick = (score: (p: Point) => number) =>
    points.reduce((best, p) => (score(p) > score(best) ? p : best));
  const tl = pick(([x, y]) => -x - y),
    br = pick(([x, y]) => x + y),
    tr = pick(([x, y]) => x - y),
    bl = pick(([x, y]) => y - x);
  const length = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const top = length(tl, tr),
    bottom = length(bl, br),
    left = length(tl, bl),
    right = length(tr, br);
  const width = (top + bottom) / 2,
    height = (left + right) / 2;
  if (!width || !height || Math.max(width / height, height / width) > GUIDANCE.maxAspect)
    return 'shape';
  if (outline.clipped) return 'clipped';
  if (outline.area > GUIDANCE.maxArea) return 'close';
  if (outline.area < GUIDANCE.minArea) return 'small';
  const ratio = (a: number, b: number) => Math.max(a, b) / Math.max(Math.min(a, b), 1e-6);
  if (Math.max(ratio(top, bottom), ratio(left, right)) > GUIDANCE.maxSideRatio) return 'tilted';
  return 'ok';
}

/** A small grayscale copy of an image. */
export type Gray = { width: number; height: number; data: Float32Array };

/**
 * Where the live preview lies in a camera photo. ``scale`` is the share of the photo's width
 * that the preview spans (1: the same view; 0.6: the photo shows a wider view, as from
 * another lens); ``score`` is the normalized cross-correlation of the best match (-1..1).
 * A coarse search over scale and offset is refined once around the best match.
 */
export function previewInPhoto(preview: Gray, photo: Gray): { scale: number; score: number } {
  const pw = preview.width,
    ph = preview.height;
  let mean = 0;
  for (const v of preview.data) mean += v;
  mean /= preview.data.length;
  const match = (scale: number, dx: number, dy: number) => {
    // Preview pixel (u, v) falls on photo pixel (cx + (u + 0.5 - pw / 2) s, cy + ...).
    const s = (scale * photo.width) / pw,
      cx = photo.width / 2 + dx * photo.width,
      cy = photo.height / 2 + dy * photo.width;
    let n = 0,
      sq = 0,
      sqq = 0,
      spq = 0,
      spp = 0,
      sp = 0;
    for (let v = 0; v < ph; v++) {
      const y = cy + (v + 0.5 - ph / 2) * s - 0.5;
      if (y < 0 || y > photo.height - 1) continue;
      const y0 = Math.floor(y),
        y1 = Math.min(y0 + 1, photo.height - 1),
        fy = y - y0;
      for (let u = 0; u < pw; u++) {
        const x = cx + (u + 0.5 - pw / 2) * s - 0.5;
        if (x < 0 || x > photo.width - 1) continue;
        const x0 = Math.floor(x),
          x1 = Math.min(x0 + 1, photo.width - 1),
          fx = x - x0;
        const row0 =
            photo.data[y0 * photo.width + x0] * (1 - fx) + photo.data[y0 * photo.width + x1] * fx,
          row1 =
            photo.data[y1 * photo.width + x0] * (1 - fx) + photo.data[y1 * photo.width + x1] * fx;
        const q = row0 * (1 - fy) + row1 * fy,
          p = preview.data[v * pw + u] - mean;
        n++;
        sp += p;
        spp += p * p;
        sq += q;
        sqq += q * q;
        spq += p * q;
      }
    }
    // Most of the preview must fall inside the photo.
    if (n < pw * ph * 0.7) return -1;
    const covariance = spq - (sp * sq) / n,
      pv = spp - (sp * sp) / n,
      qv = sqq - (sq * sq) / n;
    return pv > 0 && qv > 0 ? covariance / Math.sqrt(pv * qv) : -1;
  };
  let best = { scale: 1, dx: 0, dy: 0, score: -1 };
  const search = (scales: number[], offsets: number[], around = best) => {
    const start = around;
    for (const scale of scales)
      for (const ox of offsets)
        for (const oy of offsets) {
          const score = match(start.scale + scale, start.dx + ox, start.dy + oy);
          if (score > best.score)
            best = { scale: start.scale + scale, dx: start.dx + ox, dy: start.dy + oy, score };
        }
  };
  const range = (from: number, to: number, step: number) =>
    Array.from({ length: Math.round((to - from) / step) + 1 }, (_, i) => from + i * step);
  // Scales 0.35..1.15 and offsets up to 12% of the photo width (lenses sit apart).
  search(range(-0.65, 0.15, 0.05), range(-0.12, 0.12, 0.02), { scale: 1, dx: 0, dy: 0, score: -1 });
  search(range(-0.04, 0.04, 0.01), range(-0.015, 0.015, 0.005));
  return { scale: best.scale, score: best.score };
}
