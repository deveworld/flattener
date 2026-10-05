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
