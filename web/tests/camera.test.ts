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
    (x, y) => x >= 5 && x < 35 && y >= 5 && y < 35 && !(x >= 18 && x < 21 && y < 7),
  );
  const outline = pageOutline(p, 40, 40)!;
  assert.equal(outline.clipped, false);
  for (const [x, y] of outline.points) assert.ok(Math.min(x, y, 1 - x, 1 - y) <= 0.125 + 1e-8);
});

test('the guide follows a concave page edge and leaves a cloth strip joined to the page out', () => {
  const width = 60,
    height = 80;
  // A page whose top edge sags 12 cells in the middle (a curled sheet), and a thin strip of
  // cloth the gate took for paper, joined to its right edge.
  const sag = (x: number) => 10 + 12 * Math.sin((Math.PI * (x - 8)) / 44);
  const p = mask(
    width,
    height,
    (x, y) =>
      (x >= 8 && x < 52 && y >= sag(x) && y < 70) || (x >= 52 && x < 58 && y >= 40 && y < 42),
  );
  const outline = pageOutline(p, width, height)!;
  const top = outline.points.filter(
    ([x, y]) => y * height < 40 && x * width > 20 && x * width < 40,
  );
  assert.ok(top.length > 0 && top.every(([, y]) => y * height > 18), 'the top edge sags');
  assert.ok(
    outline.points.every(([x]) => x * width <= 52.5),
    'the strip is left out',
  );
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

/** A smooth random scene ``width`` x ``height`` (fixed seed), and a camera view of it. */
function scene(width: number, height: number) {
  let seed = 7;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const blobs = Array.from({ length: 60 }, () => [
    random(),
    random() * 0.75,
    0.02 + random() * 0.06,
    random(),
  ]);
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let v = 0;
      for (const [bx, by, r, a] of blobs) {
        const d = ((x / width - bx) ** 2 + (y / width - by) ** 2) / (r * r);
        v += a * Math.exp(-d);
      }
      data[y * width + x] = v;
    }
  return { width, height, data };
}
function view(
  photo: { width: number; height: number; data: Float32Array },
  width: number,
  height: number,
  scale: number,
  dx: number,
  dy: number,
) {
  const data = new Float32Array(width * height),
    s = (scale * photo.width) / width;
  for (let v = 0; v < height; v++)
    for (let u = 0; u < width; u++) {
      const x = Math.round(photo.width / 2 + dx * photo.width + (u + 0.5 - width / 2) * s - 0.5),
        y = Math.round(photo.height / 2 + dy * photo.width + (v + 0.5 - height / 2) * s - 0.5);
      data[v * width + u] =
        photo.data[
          Math.min(photo.height - 1, Math.max(0, y)) * photo.width +
            Math.min(photo.width - 1, Math.max(0, x))
        ];
    }
  return { width, height, data };
}

test('a photo from a wider lens than the preview is recognized', async () => {
  const { previewInPhoto } = await import('../src/camera-outline');
  const photo = scene(256, 192);
  const same = previewInPhoto(view(photo, 64, 48, 1, 0, 0), photo);
  assert.ok(Math.abs(same.scale - 1) < 0.03 && same.score > 0.9, JSON.stringify(same));
  // An ultra-wide photo: the preview spans 60% of it, a little off center.
  const wide = previewInPhoto(view(photo, 64, 48, 0.6, 0.04, -0.02), photo);
  assert.ok(Math.abs(wide.scale - 0.6) < 0.03 && wide.score > 0.9, JSON.stringify(wide));
  // A 16:9 preview cut from a 4:3 photo of the same width is the same view.
  const cut = previewInPhoto(view(photo, 64, 36, 1, 0, 0), photo);
  assert.ok(Math.abs(cut.scale - 1) < 0.03, JSON.stringify(cut));
});
