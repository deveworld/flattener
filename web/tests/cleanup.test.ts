import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cleanDocument,
  emptyMask,
  estimatePaper,
  maskCount,
  pageConfidence,
  rotateMask,
  rotatePixels,
  suggestFingers,
} from '../src/cleanup';
import { quarterTurn } from '../src/geometry';
import type { Pixels } from '../src/types';

function paper(width = 256, height = 320): Pixels {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const shade = 0.68 + (0.3 * x) / width;
      const ink = x > 40 && x < width - 40 && y % 24 > 14 && y % 24 < 18;
      const value = ink ? [30, 32, 29] : [246, 227, 202];
      const i = (y * width + x) * 4;
      data.set([...value.map((v) => Math.round(v * shade)), 255], i);
    }
  return { width, height, data };
}

test('color correction removes a smooth shadow and warm cast without washing out text', () => {
  const source = paper(),
    before = source.data.slice();
  const result = cleanDocument(source, emptyMask(source.width, source.height), true);
  for (const x of [32, 100, 200, 230]) {
    const i = (10 * source.width + x) * 4;
    const rgb = Array.from(result.data.slice(i, i + 3));
    assert.ok(Math.min(...rgb) > 237, `${x}: ${rgb}`);
    assert.ok(Math.max(...rgb) - Math.min(...rgb) <= 3);
  }
  assert.ok(result.data[(16 * source.width + 100) * 4] < 50, 'Text stays dark');
  assert.deepEqual(source.data, before, 'Source never mutates');
});

test('lighting interpolation preserves pixel centers and border colors', () => {
  const source: Pixels = {
    width: 3,
    height: 3,
    data: new Uint8ClampedArray(Array.from({ length: 9 }, () => [100, 100, 100, 255]).flat()),
  };
  const field = {
    width: 2,
    height: 2,
    data: new Float32Array([128, 192, 160, 224].flatMap((v) => [v, v, v])),
  };
  const corrected = cleanDocument(source, emptyMask(3, 3), true, field);
  const expected = [128, 160, 192, 144, 176, 208, 160, 192, 224];
  assert.deepEqual(
    Array.from(corrected.data),
    expected.flatMap((paper) => {
      const value = Math.round((100 * 248) / paper);
      return [value, value, value, 255];
    }),
  );
});

test('colored ink keeps its hue and original color is an exact no-op', () => {
  const source = paper();
  for (let y = 70; y < 76; y++)
    for (let x = 80; x < 120; x++) source.data.set([150, 30, 20, 255], (y * source.width + x) * 4);
  const mask = emptyMask(source.width, source.height);
  assert.deepEqual(cleanDocument(source, mask, false).data, source.data);
  const output = cleanDocument(source, mask, true);
  const i = (72 * source.width + 100) * 4;
  assert.ok(output.data[i] > output.data[i + 1] * 3 && output.data[i + 1] > output.data[i + 2]);
});

test('the ink stretch darkens faint text and whitens paper; contrast off keeps the division', () => {
  const source = paper();
  for (let i = 0; i < source.data.length; i += 4)
    if (source.data[i] < 120)
      source.data.set(
        [105, 102, 98].map((v) => v + (i % 7)),
        i,
      );
  const mask = emptyMask(source.width, source.height);
  const divided = cleanDocument(source, mask, true, undefined, { contrast: false }),
    stretched = cleanDocument(source, mask, true);
  const text = (16 * source.width + 128) * 4,
    blank = (10 * source.width + 128) * 4;
  assert.ok(stretched.data[text] < divided.data[text] - 60, `${stretched.data[text]}`);
  assert.ok(stretched.data[text] < 70);
  assert.ok(stretched.data[blank] >= 253 && divided.data[blank] < 252);
});

test('light tints and photo midtones keep their color under the ink stretch', () => {
  const source = paper();
  // A pale pink print (paper times 1, 0.92, 0.92) in the same shadow as the paper (smaller than a paper cell; a large
  // one is taken as paper) and a photo-like gradient block on the same page.
  for (let y = 104; y < 120; y++)
    for (let x = 56; x < 72; x++) {
      const shade = 0.68 + (0.3 * x) / source.width;
      source.data.set(
        [...[246, 209, 186].map((v) => Math.round(v * shade)), 255],
        (y * source.width + x) * 4,
      );
    }
  for (let y = 100; y < 220; y++)
    for (let x = 140; x < 236; x++) {
      const v = 70 + ((x - 140) * 130) / 96;
      source.data.set([v, v * 0.9, v * 0.75, 255], (y * source.width + x) * 4);
    }
  const mask = emptyMask(source.width, source.height);
  const divided = cleanDocument(source, mask, true, undefined, { contrast: false }),
    stretched = cleanDocument(source, mask, true);
  const pink = (112 * source.width + 64) * 4;
  for (let c = 0; c < 3; c++)
    assert.ok(Math.abs(stretched.data[pink + c] - divided.data[pink + c]) <= 3, 'the tint stays');
  let change = 0,
    count = 0;
  for (let y = 120; y < 200; y++)
    for (let x = 160; x < 216; x++, count++)
      for (let c = 0; c < 3; c++) {
        const i = (y * source.width + x) * 4 + c;
        change += Math.abs(stretched.data[i] - divided.data[i]);
      }
  assert.ok(change / (count * 3) < 6, `photo change ${change / (count * 3)}`);
});

test('removal fills selected pixels from paper and leaves all other pixels unchanged', () => {
  const source = paper(),
    mask = emptyMask(source.width, source.height);
  for (let y = 110; y < 155; y++)
    for (let x = 0; x < 32; x++) {
      mask.data[y * mask.width + x] = 1;
      source.data.set([148, 96, 70, 255], (y * source.width + x) * 4);
    }
  const sourceCopy = source.data.slice(),
    result = cleanDocument(source, mask, false);
  for (let i = 0; i < mask.data.length; i++) {
    if (!mask.data[i])
      assert.deepEqual(
        result.data.subarray(i * 4, i * 4 + 4),
        source.data.subarray(i * 4, i * 4 + 4),
      );
    else assert.ok(result.data[i * 4 + 1] > 145, 'Finger replaced by local paper');
  }
  assert.deepEqual(source.data, sourceCopy);
  assert.deepEqual(
    cleanDocument(source, emptyMask(source.width, source.height), false).data,
    sourceCopy,
    'Reset recovers every original pixel',
  );
});

test('large or malformed removal masks are rejected; dark pages remain stable', () => {
  const source = paper(64, 80),
    mask = emptyMask(64, 80);
  mask.data.fill(1);
  assert.throws(() => cleanDocument(source, mask, true), /smaller area/);
  assert.throws(
    () => cleanDocument(source, { width: 64, height: 80, data: new Uint8Array(10) }, false),
    /invalid/,
  );
  for (let i = 0; i < source.data.length; i += 4) source.data.set([12, 15, 20, 255], i);
  const output = cleanDocument(source, emptyMask(64, 80), true);
  assert.deepEqual(output.data, source.data);
  assert.ok(estimatePaper(source).data.every(Number.isFinite));
});

test('automatic proposals retain a localized edge intrusion, ignore inner graphics and uncertain rims', () => {
  const width = 256,
    height = 320,
    confidence = new Float32Array(width * height).fill(0.98);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      if (
        Math.min(x, y, width - x - 1, height - y - 1) < 3 ||
        (x < 34 && y >= 100 && y < 155) ||
        (x >= 120 && x < 160 && y >= 200 && y < 230)
      )
        confidence[y * width + x] = 0.05;
    }
  const mask = suggestFingers(confidence, width, height);
  assert.ok(maskCount(mask) > 34 * 55);
  assert.equal(mask.data[125 * width + 5], 1);
  assert.equal(mask.data[210 * width + 130], 0);
  assert.equal(mask.data[20 * width + 1], 0);
  confidence.fill(0.1);
  assert.equal(
    maskCount(suggestFingers(confidence, width, height)),
    0,
    'Ambiguous large region not proposed',
  );
  confidence.fill(0.98);
  assert.equal(maskCount(suggestFingers(confidence, width, height)), 0);
});

test('confidence follows final grid coordinates including rotation', () => {
  const data = new Float64Array([-1, -1, 1, -1, -1, 1, 1, 1]);
  const probability = new Float32Array([0.1, 0.2, 0.3, 0.4]);
  const result = pageConfidence({ rows: 2, cols: 2, data }, probability, 2, 2, 2, 2);
  assert.deepEqual(result, probability);
  const turned = pageConfidence(
    quarterTurn({ rows: 2, cols: 2, data }, 1),
    probability,
    2,
    2,
    2,
    2,
  );
  assert.deepEqual(turned, new Float32Array([0.2, 0.4, 0.1, 0.3]));
});

test('appearance rejects blank-paper gate errors and neutral shadows across skin colors', () => {
  const width = 256,
    height = 320,
    source = paper(width, height);
  const confidence = new Float32Array(width * height).fill(0.98);
  for (let y = 100; y < 155; y++) for (let x = 0; x < 34; x++) confidence[y * width + x] = 0.1;
  assert.equal(
    maskCount(suggestFingers(confidence, width, height, source)),
    0,
    'Blank paper never proposed',
  );
  for (const rgb of [
    [214, 158, 121],
    [125, 74, 42],
    [84, 49, 28],
  ]) {
    const data = source.data.slice();
    for (let y = 100; y < 155; y++)
      for (let x = 0; x < 34; x++) data.set([...rgb, 255], (y * width + x) * 4);
    const result = suggestFingers(confidence, width, height, { ...source, data });
    assert.ok(result.data[125 * width + 15], `Missed tone ${rgb}`);
  }
  const shadow = source.data.slice();
  for (let y = 100; y < 155; y++)
    for (let x = 0; x < 34; x++) for (let c = 0; c < 3; c++) shadow[(y * width + x) * 4 + c] *= 0.6;
  assert.equal(
    maskCount(suggestFingers(confidence, width, height, { ...source, data: shadow })),
    0,
  );
});

test('a detected finger grows over its connected colored boundary even where gate confidence is wrong', () => {
  const width = 256,
    height = 320,
    source = paper(width, height);
  const confidence = new Float32Array(width * height).fill(0.99);
  for (let y = 100; y < 155; y++)
    for (let x = 0; x < 34; x++) {
      source.data.set([137, 81, 48, 255], (y * width + x) * 4);
      if (x < 25 && y > 104 && y < 150) confidence[y * width + x] = 0.1;
    }
  const mask = suggestFingers(confidence, width, height, source);
  assert.equal(mask.data[102 * width + 31], 1, 'Finger edge included');
  assert.equal(mask.data[90 * width + 31], 0, 'Surrounding paper excluded');
});

test('dark finger boundaries retain the same relative color evidence as their center', () => {
  const width = 256,
    height = 320,
    source = paper(width, height);
  for (let i = 0; i < width * height; i++) source.data.set([251, 250, 246, 255], i * 4);
  const confidence = new Float32Array(width * height).fill(0.99);
  for (let y = 100; y < 155; y++)
    for (let x = 0; x < 34; x++) {
      const rgb = y < 110 || y > 145 ? [101, 81, 86, 255] : [120, 96, 102, 255];
      source.data.set(rgb, (y * width + x) * 4);
      confidence[y * width + x] = 0.1;
    }
  const mask = suggestFingers(confidence, width, height, source);
  assert.equal(mask.data[103 * width + 15], 1);
  assert.equal(mask.data[150 * width + 15], 1);
});

test('mask and image quarter turns stay aligned and round-trip without resampling', () => {
  const source = paper(48, 64),
    mask = emptyMask(48, 64);
  mask.data[10 * 48 + 7] = 1;
  const turned = rotateMask(mask, 1);
  assert.equal(turned.width, 64);
  assert.equal(turned.data[7 * 64 + 63 - 10], 1);
  assert.deepEqual(rotateMask(turned, -1), mask);
  const rotated = rotatePixels(source, 1);
  assert.deepEqual(
    rotated.data.slice((7 * 64 + 53) * 4, (7 * 64 + 53) * 4 + 4),
    source.data.slice((10 * 48 + 7) * 4, (10 * 48 + 7) * 4 + 4),
  );
  assert.deepEqual(rotatePixels(rotated, 3), source);
});
