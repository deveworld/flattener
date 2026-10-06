import test from 'node:test';
import assert from 'node:assert/strict';
import { sharpenAmount, sharpenPixels } from '../src/sharpen';

test('sharpening strength follows the reduction and is off when enlarging', () => {
  assert.equal(sharpenAmount(0.8), 0);
  assert.equal(sharpenAmount(1), 0.3);
  assert.ok(sharpenAmount(2) > sharpenAmount(1.5));
  assert.equal(sharpenAmount(4), 1);
});

test('an edge gains contrast, flat paper and small grain stay, values stay in range', () => {
  const width = 20,
    height = 4,
    data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const v = x < 10 ? 40 : 245;
      data.set([v, v, v, 255], (y * width + x) * 4);
    }
  data.set([246, 245, 245, 255], (0 * width + 18) * 4); // one level of grain
  const out = sharpenPixels({ width, height, data }, 1).data;
  const at = (x: number, y = 1) => out[(y * width + x) * 4];
  assert.ok(at(9) < 40 && at(10) > 245 - 1); // darker ink and brighter paper at the edge
  assert.equal(at(10), 255);
  assert.equal(at(2), 40); // flat ink far from the edge is unchanged
  assert.equal(out[(0 * width + 18) * 4], 246); // grain below the threshold is kept
  assert.equal(sharpenPixels({ width, height, data }, 0).data, data);
});
