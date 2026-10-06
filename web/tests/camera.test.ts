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
