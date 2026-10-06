import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  blendFlip,
  checkGrid,
  compose,
  gateInput,
  gridPoint,
  mirroredInput,
  modelInput,
  outputSize,
  pageAngle,
  pageBox,
  quarterTurn,
  render,
  tightPlacement,
  turnPlacement,
  whole,
} from '../src/geometry';
import { orientationViews } from '../src/orientation';
import type { Grid, Pixels } from '../src/types';

const ref = JSON.parse(readFileSync(new URL('./fixtures/python.json', import.meta.url), 'utf8'));
const image: Pixels = {
  ...ref.source,
  data: new Uint8ClampedArray(ref.source.width * ref.source.height * 4),
};
for (let y = 0; y < image.height; y++)
  for (let x = 0; x < image.width; x++) {
    const i = (y * image.width + x) * 4;
    image.data.set(
      [
        (x * 17 + y * 3) % 256,
        (x * 5 + y * 13) % 256,
        ((Math.floor(x / 7) + Math.floor(y / 5)) % 2) * 230,
        255,
      ],
      i,
    );
  }
const grid: Grid = { ...ref.grid, data: new Float64Array(ref.grid.data) };
function close(actual: ArrayLike<number>, expected: ArrayLike<number>, tolerance = 1e-6) {
  assert.equal(actual.length, expected.length);
  let max = 0;
  for (let i = 0; i < actual.length; i++) max = Math.max(max, Math.abs(actual[i] - expected[i]));
  assert.ok(max <= tolerance, `maximum error ${max} > ${tolerance}`);
}

test('dewarp pixel-center preprocessing matches PyTorch', () => {
  close(modelInput(image, whole(image), 13, 11), ref.model_input, 4e-5);
});
test('antialiased page gate preprocessing matches PyTorch', () => {
  close(gateInput(image, 19, 17), ref.gate_input, 1e-6);
});
test('page angle, tight crop, and affine composition match Python', () => {
  const angle = pageAngle(grid, image.width, image.height)!;
  close([angle], [ref.angle], 1e-10);
  const placement = tightPlacement(grid, image, angle);
  close(placement.matrix, ref.placement.matrix, 1e-12);
  close(placement.offset, ref.placement.offset, 1e-12);
  close([placement.height, placement.width], [ref.placement.height, ref.placement.width], 1e-10);
  close(compose(grid, placement).data, ref.composed, 1e-12);
});
test('dense map and one-sample rendering match Python', () => {
  const dense = [];
  for (let y = 0; y < 17; y++)
    for (let x = 0; x < 13; x++) dense.push(...gridPoint(grid, (x + 0.5) / 13, (y + 0.5) / 17));
  close(dense, ref.dense, 1e-7);
  const pixels = render(image, grid, 17, 13).data;
  const rgb = Array.from(pixels).filter((_, i) => i % 4 !== 3);
  close(rgb, ref.render, 1); // Final float32 rounding can move an 8-bit value by one.
  assert.deepEqual(outputSize(grid, image, 2560), ref.output_size);
});
test('counterclockwise quarter turns preserve Python orientation and source coordinates', () => {
  close(quarterTurn(grid, 1).data, ref.turned, 0);
  close(quarterTurn(grid, 4).data, grid.data, 0);
  const p = whole(image);
  assert.deepEqual(turnPlacement(p, 4), p);
  assert.equal(turnPlacement(p, 1).width, image.height);
});
test('orientation letterboxing, energy ranking, and text patches match Python', () => {
  const views = orientationViews(image, whole(image), ref.orientation.config);
  close(views.global, ref.orientation.global, 0);
  close(views.patches, ref.orientation.patches, 0);
  const cells = views.cells.flatMap((i) => [Math.floor(i / 8), i % 8]);
  assert.deepEqual(cells, ref.orientation.cells);
  close(views.energy, ref.orientation.energy, 2e-5);
});
test('flip TTA reindexes the mirrored grid, and rejects disagreeing predictions', () => {
  const mirrored: Grid = { ...grid, data: new Float64Array(grid.data.length) };
  for (let y = 0; y < grid.rows; y++)
    for (let x = 0; x < grid.cols; x++) {
      const a = (y * grid.cols + x) * 2,
        b = (y * grid.cols + grid.cols - 1 - x) * 2;
      mirrored.data[a] = -grid.data[b];
      mirrored.data[a + 1] = grid.data[b + 1];
    }
  close(blendFlip(grid, mirrored).data, grid.data, 1e-7);
  mirrored.data.fill(5);
  assert.equal(blendFlip(grid, mirrored), grid);
  const input = modelInput(image, whole(image), 13, 11);
  assert.deepEqual(mirroredInput(mirroredInput(input, 13, 11), 13, 11), input);
});
test('geometry guard rejects non-finite, folded, and out-of-frame maps', () => {
  assert.equal(checkGrid(grid), null);
  assert.ok(checkGrid({ ...grid, data: grid.data.map(() => NaN) }));
  assert.ok(checkGrid({ ...grid, data: grid.data.map((v, i) => (i % 2 ? v : -v)) }));
  assert.ok(checkGrid({ ...grid, data: grid.data.map((v) => v + 3) }));
});
test('page detection rejects empty input and uses the largest connected component', () => {
  const p = new Float32Array(100);
  assert.equal(pageBox(p, 10, 10, image), null);
  p[0] = 1;
  for (let y = 3; y < 8; y++) for (let x = 2; x < 8; x++) p[y * 10 + x] = 0.9;
  const box = pageBox(p, 10, 10, image)!;
  assert.ok(box[0] > 0 && box[1] > 0);
  assert.ok(box[2] < image.width && box[3] < image.height);
});
test('output size limits preserve a usable document aspect', () => {
  const [h, w] = outputSize(grid, image, 100);
  assert.ok(h <= 100 && w <= 100);
  assert.throws(
    () => outputSize({ ...grid, data: grid.data.map(() => 0) }, image, 100),
    /too small/,
  );
});

test('a page the gate split into pieces is taken from the whole-photo outline', async () => {
  const { gateSplit, pageComponents, splitPageBox } = await import('../src/geometry');
  const [h, w] = [40, 30];
  const probability = new Float32Array(h * w);
  // The page spans x 8..22, y 6..34; a dark band at rows 16..18 splits it in two.
  for (let y = 6; y < 34; y++)
    for (let x = 8; x < 22; x++) if (y < 16 || y > 18) probability[y * w + x] = 0.9;
  const image = { width: 300, height: 400, data: new Uint8ClampedArray(300 * 400 * 4) };
  const components = pageComponents(probability, h, w);
  assert.equal(components.length, 2);
  assert.ok(gateSplit(components));
  const grid = (x0: number, x1: number, y0: number, y1: number) => {
    const rows = 5,
      cols = 4,
      data = new Float64Array(rows * cols * 2);
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        data[(r * cols + c) * 2] = x0 + ((x1 - x0) * c) / (cols - 1);
        data[(r * cols + c) * 2 + 1] = y0 + ((y1 - y0) * r) / (rows - 1);
      }
    return { rows, cols, data };
  };
  // The outline of the page in normalized values: x 80..220 px, y 60..340 px.
  const box = splitPageBox(grid(-0.467, 0.467, -0.7, 0.7), components, h, w, image)!;
  assert.ok(box[0] <= 80 && box[2] >= 220 && box[1] <= 60 && box[3] >= 340, String(box));
  // An outline over the background holds no piece.
  assert.equal(splitPageBox(grid(0.6, 0.95, 0.6, 0.95), components, h, w, image), null);
});

/** A 45 × 31 page grid in photo-normalized coordinates: ``page(u, v)`` gives page units,
 * mapped through a perspective homography into a 3000 × 4000 photo. */
function perspectiveGrid(page: (u: number, v: number) => [number, number]) {
  const rows = 45,
    cols = 31,
    data = new Float64Array(rows * cols * 2);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      const [u, v] = page(c / (cols - 1), r / (rows - 1));
      const z = 1 + 0.25 * u - 0.15 * v;
      const x = (600 + 1700 * u + 300 * v) / z,
        y = (700 + 200 * u + 2300 * v) / z;
      data[2 * (r * cols + c)] = (2 * x) / 3000 - 1;
      data[2 * (r * cols + c) + 1] = (2 * y) / 4000 - 1;
    }
  return { rows, cols, data };
}

test('grid distortion ignores perspective and smooth curl and finds a local bend', async () => {
  const { gridDistortion } = await import('../src/geometry');
  const flat = gridDistortion(
    perspectiveGrid((u, v) => [u, v]),
    3000,
    4000,
  )!;
  assert.ok(flat.bend < 1e-6 && flat.strain < 1e-5, JSON.stringify(flat.bend));
  // A page bowed along its width (curl) is smooth: the cubic takes it out.
  const curl = gridDistortion(
    perspectiveGrid((u, v) => [u + 0.02 * Math.sin(Math.PI * v), v + 0.03 * u * (1 - u)]),
    3000,
    4000,
  )!;
  assert.ok(curl.bend < 0.002, `curl ${curl.bend}`);
  // An edge pulled inward over a short stretch (an occluder at the left edge, rows 26–34).
  const pulled = gridDistortion(
    perspectiveGrid((u, v) => [u + 0.03 * Math.exp(-((v - 0.68) ** 2) / 0.003 - u / 0.08), v]),
    3000,
    4000,
  )!;
  assert.ok(pulled.bend > 0.015, `bend ${pulled.bend}`);
  assert.equal(pulled.bendAt[1], 0);
  assert.ok(Math.abs(pulled.bendAt[0] - 30) <= 2, `at ${pulled.bendAt}`);
  assert.ok(pulled.notch > 0.015 && pulled.notchAt[1] === 0, `notch ${pulled.notch}`);
  assert.ok(pulled.strain > 20 * curl.strain);
});

test('a dog-eared page corner is found from the gate and painted with paper only off the page', async () => {
  const { cornerGaps, fillCorners } = await import('../src/geometry');
  const gw = 144,
    gh = 192;
  // The page covers cells 20..124 × 30..170; a dog-ear cuts its top-left corner diagonally.
  const gate = (cut: boolean) =>
    Float32Array.from({ length: gw * gh }, (_, i) => {
      const x = i % gw,
        y = Math.floor(i / gw);
      const inPage = x >= 20 && x < 124 && y >= 30 && y < 170;
      return inPage && !(cut && x - 20 + (y - 30) < 22) ? 0.95 : 0.05;
    });
  assert.equal(cornerGaps(gate(false), gw, gh, 1440, 1920).length, 0);
  const gaps = cornerGaps(gate(true), gw, gh, 1440, 1920);
  assert.equal(gaps.length, 1);
  const cx = gaps[0].reduce((s, p) => s + p[0], 0) / gaps[0].length,
    cy = gaps[0].reduce((s, p) => s + p[1], 0) / gaps[0].length;
  assert.ok(cx < 340 && cy < 440, `${cx} ${cy}`); // near the page's top-left corner (200, 300)
  // Paper (230) on the page, a dark background (40) elsewhere.
  const probability = gate(true);
  const image = { width: 1440, height: 1920, data: new Uint8ClampedArray(1440 * 1920 * 4) };
  for (let y = 0; y < 1920; y++)
    for (let x = 0; x < 1440; x++) {
      const p = probability[Math.floor(y / 10) * gw + Math.floor(x / 10)] >= 0.5;
      image.data.set(p ? [230, 230, 230, 255] : [40, 40, 40, 255], (y * 1440 + x) * 4);
    }
  const before = image.data.slice();
  const restore = fillCorners(image, gaps, probability, gw, gh);
  const at = (x: number, y: number) => image.data[(y * 1440 + x) * 4];
  assert.ok(at(215, 315) > 200, 'the missing corner is paper colored');
  assert.equal(at(100, 100), 40, 'background outside the gap stays');
  assert.equal(at(700, 900), 230, 'the page stays');
  restore();
  assert.deepEqual(image.data, before);
});
