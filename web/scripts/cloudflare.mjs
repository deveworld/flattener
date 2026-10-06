// Prepare a built site for Cloudflare Pages: `node scripts/cloudflare.mjs dist cloudflare`.
// Pages limits each file to 25 MiB, so larger models are split into parts that the app
// joins again (`parts` in the manifest). `_headers` makes the site cross-origin isolated,
// so the WASM runtime can use threads without the service worker.
import {
  cpSync,
  openSync,
  readFileSync,
  readSync,
  rmSync,
  writeFileSync,
  closeSync,
} from 'node:fs';
import { join } from 'node:path';

const PART_BYTES = 20 * 1024 * 1024;
const [source, target] = process.argv.slice(2);
if (!source || !target) {
  console.error('usage: node scripts/cloudflare.mjs <built site> <output directory>');
  process.exit(2);
}
rmSync(target, { recursive: true, force: true });
cpSync(source, target, { recursive: true });

const models = join(target, 'models');
const manifestPath = join(models, 'manifest.json');
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
for (const model of Object.values(manifest.models)) {
  if (model.bytes <= PART_BYTES) continue;
  const path = join(models, model.file);
  const handle = openSync(path, 'r');
  const parts = [];
  for (let offset = 0; offset < model.bytes; offset += PART_BYTES) {
    const buffer = Buffer.alloc(Math.min(PART_BYTES, model.bytes - offset));
    readSync(handle, buffer, 0, buffer.length, offset);
    const name = `${model.file}.part${parts.length + 1}`;
    writeFileSync(join(models, name), buffer);
    parts.push(name);
  }
  closeSync(handle);
  rmSync(path);
  model.parts = parts;
  console.log(`${model.file}: ${parts.length} parts`);
}
writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
writeFileSync(
  join(target, '_headers'),
  '/*\n  Cross-Origin-Opener-Policy: same-origin\n  Cross-Origin-Embedder-Policy: require-corp\n',
);
