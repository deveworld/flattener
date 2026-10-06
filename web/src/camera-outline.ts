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

  const mask = new Uint8Array(probability.length);
  for (const i of largest) mask[i] = 1;
  const stride = width + 1;
  const boundary = new Set<number>();
  let clipped = false;
  for (const i of largest) {
    const x = i % width,
      y = Math.floor(i / width),
      a = y * stride + x;
    if (x === 0 || y === 0 || x === width - 1 || y === height - 1) clipped = true;
    if (
      y === 0 ||
      !mask[i - width] ||
      x === width - 1 ||
      !mask[i + 1] ||
      y === height - 1 ||
      !mask[i + width] ||
      x === 0 ||
      !mask[i - 1]
    ) {
      for (const corner of [a, a + 1, a + stride, a + stride + 1]) boundary.add(corner);
    }
  }
  // Bridge mask gaps caused by text or fingers in the live framing guide.
  const outer = convexHull(
    Array.from(boundary, (i) => [(i % stride) / width, Math.floor(i / stride) / height]),
  );
  if (outer.length < 3) return null;
  return { points: resample(outer, 64), area: largest.length / probability.length, clipped };
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

function convexHull(points: Point[]): Point[] {
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (a: Point, b: Point, c: Point) =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const half = (ordered: Point[]) => {
    const chain: Point[] = [];
    for (const point of ordered) {
      while (
        chain.length >= 2 &&
        cross(chain[chain.length - 2], chain[chain.length - 1], point) <= 0
      )
        chain.pop();
      chain.push(point);
    }
    return chain.slice(0, -1);
  };
  return [...half(points), ...half(points.slice().reverse())];
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

/** Root-mean-square move of the outline points between two frames, after matching the
 * starting point, as a share of the page size (the square root of its frame area). */
export function outlineMotion(previous: PageOutline, next: PageOutline): number {
  if (previous.points.length !== next.points.length) return Infinity;
  const count = next.points.length;
  let best = Infinity;
  for (let shift = 0; shift < count; shift++) {
    let sum = 0;
    for (let i = 0; i < count; i++) {
      const p = previous.points[i],
        q = next.points[(i + shift) % count];
      sum += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
    }
    best = Math.min(best, sum);
  }
  return Math.sqrt(best / count) / Math.sqrt(Math.max(next.area, 1e-6));
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

/** Auto capture: a steady, well-framed page starts a countdown; after a photo the next one
 * waits until the page changes, so one page is not taken twice. */
export const AUTO = {
  steady: 0.02, // outline motion per frame below this share of the page size is steady
  change: 0.12, // motion above this is a page change (a turned page, a hand)
  lostMs: 500, // no page for this long is a page change too; shorter gaps are detector misses
  steadyMs: 700, // steady time before the countdown
  countdownMs: 1300,
};

/**
 * The auto-capture decision, fed only with fresh detector replies. Replies that arrive
 * while a photo is taken are not fed (the preview can freeze or go dark then), so the
 * first outline after the photo is compared with the last one before it. A shown outline
 * that expires because replies are late is ``stale``, which is not a page change.
 */
export class AutoCapture {
  private armed = true;
  private steadySince: number | null = null;
  private lostSince: number | null = null;
  /** The last outline the detector found, kept through gaps and through a photo. */
  private seen: PageOutline | null = null;

  /** A photo was taken and the page has not changed since. */
  get waiting() {
    return !this.armed;
  }

  reset() {
    this.armed = true;
    this.steadySince = this.lostSince = null;
    this.seen = null;
  }

  /** A detector reply (``outline`` null: no page) at ``now`` ms; ``ok``: the framing hint is
   * clear. Returns the countdown progress, 0..1; 1 takes the photo. */
  frame(outline: PageOutline | null, ok: boolean, now: number): number {
    if (!outline) {
      this.steadySince = null;
      this.lostSince ??= now;
      if (now - this.lostSince >= AUTO.lostMs) this.armed = true;
      return 0;
    }
    this.lostSince = null;
    const motion = this.seen ? outlineMotion(this.seen, outline) : Infinity;
    this.seen = outline;
    if (!this.armed) {
      this.armed = motion > AUTO.change;
      return 0;
    }
    if (!ok || motion > AUTO.steady) {
      this.steadySince = null;
      return 0;
    }
    this.steadySince ??= now;
    return Math.min(1, Math.max(0, (now - this.steadySince - AUTO.steadyMs) / AUTO.countdownMs));
  }

  /** The shown outline is out of date: no countdown until fresh replies come. */
  stale() {
    this.steadySince = null;
  }

  /** A photo was taken; the next waits for a page change. */
  taken() {
    this.armed = false;
    this.steadySince = this.lostSince = null;
  }
}
