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
  let largest: number[] = [];
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
    if (tail > largest.length) largest = Array.from(queue.subarray(0, tail));
  }
  if (largest.length < probability.length * 0.025) return null;
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
