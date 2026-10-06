import test from 'node:test';
import assert from 'node:assert/strict';
import { cropBounds, cropPixels, isFullCrop, turnCrop } from '../src/crop';
import { rotatePixels } from '../src/cleanup';
import { pdfFromJpegs } from '../src/pdf';
import type { Pixels } from '../src/types';

function numbered(width: number, height: number): Pixels {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) data.set([i % 256, Math.floor(i / 256), 0, 255], i * 4);
  return { width, height, data };
}

test('a crop turns with the image it belongs to', () => {
  const image = numbered(40, 30);
  const crop = { left: 0.1, top: 0.2, right: 0.6, bottom: 0.9 };
  for (const turns of [1, 2, 3]) {
    const cropThenTurn = rotatePixels(cropPixels(image, crop), turns);
    const turnThenCrop = cropPixels(rotatePixels(image, turns), turnCrop(crop, turns));
    assert.deepEqual(
      [turnThenCrop.width, turnThenCrop.height],
      [cropThenTurn.width, cropThenTurn.height],
    );
    assert.deepEqual(turnThenCrop.data, cropThenTurn.data);
  }
  assert.deepEqual(turnCrop(crop, 4), crop);
});

test('crops keep at least one pixel and skip full-frame copies', () => {
  const image = numbered(10, 10);
  assert.equal(cropPixels(image, { left: 0, top: 0, right: 1, bottom: 1 }), image);
  assert.ok(isFullCrop(undefined));
  assert.deepEqual(cropBounds({ left: 0.5, top: 0.5, right: 0.5, bottom: 0.5 }, 10, 10), {
    x: 5,
    y: 5,
    width: 1,
    height: 1,
  });
  const piece = cropPixels(image, { left: 0.2, top: 0.3, right: 0.5, bottom: 0.6 });
  assert.deepEqual([piece.width, piece.height], [3, 3]);
  assert.deepEqual([...piece.data.subarray(0, 4)], [32, 0, 0, 255]); // pixel (2, 3)
});

test('a PDF lists every page and its cross-reference offsets point at the objects', async () => {
  const jpeg = new Uint8Array([0xff, 0xd8, 0xff, 0xd9]);
  const pdf = new Uint8Array(
    await pdfFromJpegs([
      { jpeg, width: 1200, height: 1600 },
      { jpeg, width: 1600, height: 1200 },
    ]).arrayBuffer(),
  );
  const text = new TextDecoder('latin1').decode(pdf);
  assert.ok(text.startsWith('%PDF-1.4'));
  assert.match(text, /\/Count 2/);
  assert.match(text, /\/MediaBox \[0 0 631\.42 841\.89\]/);
  assert.match(text, /\/MediaBox \[0 0 841\.89 631\.42\]/);
  const xref = Number(text.match(/startxref\n(\d+)/)![1]);
  assert.ok(text.slice(xref).startsWith('xref'));
  const entries = [...text.slice(xref).matchAll(/(\d{10}) 00000 n/g)].map((m) => Number(m[1]));
  assert.equal(entries.length, 8);
  entries.forEach((offset, index) =>
    assert.ok(text.slice(offset).startsWith(`${index + 1} 0 obj`), `object ${index + 1}`),
  );
});
