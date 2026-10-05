import { defineConfig } from 'vite';
import { copyFileSync, createReadStream, mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

// Keep runtime binaries on the same origin; no CDN or processing endpoint is needed.
const runtime = resolve('node_modules/onnxruntime-web/dist');
const files = ['ort-wasm-simd-threaded.asyncify.wasm', 'ort-wasm-simd-threaded.asyncify.mjs'];
const license = resolve('../LICENSE');
const runtimePackage = JSON.parse(readFileSync(resolve(runtime, '../package.json'), 'utf8'));
if (runtimePackage.version !== '1.23.2') {
  throw new Error('Review LICENSE for the new ONNX Runtime version before building.');
}

export default defineConfig({
  base: './',
  resolve: {
    conditions: ['onnxruntime-web-use-extern-wasm', 'module', 'browser', 'development|production'],
  },
  worker: { format: 'es' },
  plugins: [
    {
      name: 'local-onnx-runtime',
      configureServer(server) {
        server.middlewares.use('/LICENSE', (_req, res) => {
          res.setHeader('Content-Type', 'text/plain; charset=utf-8');
          createReadStream(license).pipe(res);
        });
        server.middlewares.use('/runtime', (req, res, next) => {
          const name = req.url?.split('?')[0]?.slice(1);
          if (!name || !files.includes(name)) return next();
          res.setHeader(
            'Content-Type',
            name.endsWith('.wasm') ? 'application/wasm' : 'text/javascript',
          );
          createReadStream(resolve(runtime, name)).pipe(res);
        });
      },
      closeBundle() {
        copyFileSync(license, resolve('dist/LICENSE'));
        mkdirSync('dist/runtime', { recursive: true });
        for (const name of files)
          copyFileSync(resolve(runtime, name), resolve('dist/runtime', name));
      },
    },
  ],
});
