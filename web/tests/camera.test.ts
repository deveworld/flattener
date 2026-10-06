import test from 'node:test';
import assert from 'node:assert/strict';
import { containVideo, pageOutline, smoothOutline } from '../src/camera-outline';
import type { PageOutline, Point } from '../src/types';

function mask(width: number, height: number, inside: (x: number, y: number) => boolean) {
  return Float32Array.from({ length: width * height }, (_, i) =>
    inside((i % width) + 0.5, Math.floor(i / width) + 0.5) ? 0.95 : 0.05,
  );
}

test('camera outline follows a slanted page and leaves separate objects out', () => {
  const width = 80,
    height = 60;
  const p = mask(
    width,
    height,
    (x, y) => (y >= 10 && y < 50 && x >= 15 + y / 5 && x < 60 + y / 5) || (x < 3 && y < 3),
  );
  const outline = pageOutline(p, width, height)!;
  assert.ok(outline);
  assert.equal(outline.points.length, 64);
  assert.equal(outline.clipped, false);
  assert.ok(Math.abs(outline.area - (45 * 40) / (width * height)) < 0.001);
  for (const [nx, ny] of outline.points) {
    const x = nx * width,
      y = ny * height;
    const distance = Math.min(
      Math.abs(y - 10),
      Math.abs(y - 50),
      Math.abs(x - (15 + y / 5)),
      Math.abs(x - (60 + y / 5)),
    );
    assert.ok(distance < 1.5, `Point left the page boundary: ${x}, ${y}`);
  }
});

test('interior holes do not replace the outer page boundary', () => {
  const p = mask(
    40,
    40,
    (x, y) => x >= 5 && x < 35 && y >= 5 && y < 35 && !(x >= 12 && x < 28 && y >= 12 && y < 28),
  );
  const outline = pageOutline(p, 40, 40)!;
  for (const [x, y] of outline.points) assert.ok(Math.min(x, y, 1 - x, 1 - y) <= 0.125 + 1e-8);
});

test('small gaps in the mask do not indent the live framing guide', () => {
  const p = mask(
    40,
    40,
    (x, y) => x >= 5 && x < 35 && y >= 5 && y < 35 && !(x >= 15 && x < 25 && y < 12),
  );
  const outline = pageOutline(p, 40, 40)!;
  assert.equal(outline.clipped, false);
  for (const [x, y] of outline.points) assert.ok(Math.min(x, y, 1 - x, 1 - y) <= 0.125 + 1e-8);
});

test('camera detection rejects empty, tiny, uncertain and invalid masks', () => {
  assert.equal(pageOutline(new Float32Array(400), 20, 20), null);
  assert.equal(
    pageOutline(
      mask(20, 20, (x, y) => x < 2 && y < 2),
      20,
      20,
    ),
    null,
  );
  assert.equal(pageOutline(new Float32Array(400).fill(0.6), 20, 20), null);
  assert.equal(pageOutline(new Float32Array(400).fill(NaN), 20, 20), null);
  assert.throws(() => pageOutline(new Float32Array(2), 20, 20));
});

test('a page touching the frame is marked as clipped', () => {
  assert.equal(
    pageOutline(
      mask(30, 40, (x, y) => x < 20 && y > 5 && y < 35),
      30,
      40,
    )?.clipped,
    true,
  );
});

test('outline smoothing aligns changing contour starts and resets after a large move', () => {
  const points: Point[] = [
    [0.2, 0.2],
    [0.7, 0.2],
    [0.7, 0.8],
    [0.2, 0.8],
  ];
  const previous: PageOutline = { points, area: 0.3, clipped: false };
  const shifted = { ...previous, points: [...points.slice(2), ...points.slice(0, 2)] };
  assert.deepEqual(smoothOutline(previous, shifted).points, points);
  const moved = { ...previous, points: points.map(([x, y]): Point => [x + 0.2, y]) };
  assert.deepEqual(smoothOutline(previous, moved), moved);
});

test('overlay matches contained video in portrait and landscape letterboxing', () => {
  assert.deepEqual(containVideo(400, 600, 1600, 900), {
    width: 400,
    height: 225,
    left: 0,
    top: 187.5,
  });
  assert.deepEqual(containVideo(600, 400, 900, 1600), {
    width: 225,
    height: 400,
    left: 187.5,
    top: 0,
  });
});

test('capture guidance names one problem at a time, in priority order', async () => {
  const { guidance } = await import('../src/camera-outline');
  const quad = (pts: [number, number][], area: number, clipped = false) => ({
    points: pts,
    area,
    clipped,
  });
  const square = (x0: number, y0: number, x1: number, y1: number): [number, number][] => [
    [x0, y0],
    [x1, y0],
    [x1, y1],
    [x0, y1],
  ];
  assert.equal(guidance(null, 0.75), 'none');
  assert.equal(guidance(quad(square(0.45, 0.1, 0.5, 0.9), 0.04), 0.75), 'shape');
  assert.equal(guidance(quad(square(0, 0.1, 0.8, 0.9), 0.64, true), 0.75), 'clipped');
  assert.equal(guidance(quad(square(0.02, 0.02, 0.98, 0.98), 0.92), 0.75), 'close');
  assert.equal(guidance(quad(square(0.4, 0.4, 0.6, 0.6), 0.04), 0.75), 'small');
  const keystone: [number, number][] = [
    [0.35, 0.15],
    [0.65, 0.15],
    [0.85, 0.85],
    [0.15, 0.85],
  ];
  assert.equal(guidance(quad(keystone, 0.35), 0.75), 'tilted');
  assert.equal(guidance(quad(square(0.15, 0.1, 0.85, 0.9), 0.56), 0.75), 'ok');
});

test('auto capture waits for a steady page and re-arms only after a big change', async () => {
  const { AUTO, autoStep } = await import('../src/camera-outline');
  let state = { armed: true, steadySince: null as number | null };
  let step = autoStep(state, 0.005, true, 0);
  state = step.state;
  step = autoStep(state, 0.005, true, AUTO.steadyMs + AUTO.countdownMs / 2);
  assert.ok(step.progress > 0.4 && step.progress < 0.6);
  step = autoStep(step.state, 0.05, true, AUTO.steadyMs + AUTO.countdownMs); // a shake
  assert.equal(step.progress, 0);
  step = autoStep(step.state, 0.005, false, 5000); // badly framed
  assert.equal(step.progress, 0);
  // After a capture nothing happens until the page changes a lot.
  state = { armed: false, steadySince: null };
  for (const motion of [0.005, 0.03, 0.08]) {
    step = autoStep(state, motion, true, 10_000);
    assert.equal(step.state.armed, false);
  }
  assert.equal(autoStep(state, 0.2, true, 10_000).state.armed, true);
  assert.equal(autoStep(state, null, true, 10_000).state.armed, true);
});

test('the live outline joins a page that a dark band split, but not a separate object', () => {
  const [w, h] = [40, 50];
  const probability = new Float32Array(w * h);
  const fill = (x0: number, y0: number, x1: number, y1: number) => {
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) probability[y * w + x] = 0.95;
  };
  fill(8, 6, 32, 20); // title area
  fill(8, 23, 32, 45); // illustration below a dark band
  const joined = pageOutline(probability, w, h)!;
  const ys = joined.points.map(([, y]) => y);
  assert.ok(Math.min(...ys) <= 6 / h + 0.01 && Math.max(...ys) >= 45 / h - 0.01);
  probability.fill(0);
  fill(8, 23, 32, 45);
  fill(2, 0, 10, 8); // a narrow object off to the side
  const alone = pageOutline(probability, w, h)!;
  assert.ok(Math.min(...alone.points.map(([, y]) => y)) >= 23 / h - 0.01);
});
