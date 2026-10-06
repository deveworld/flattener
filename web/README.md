# Flattener in the browser

A TypeScript + Vite app for scanning document photos locally.
ONNX Runtime Web runs the page detector, orientation model and dewarper in a Web Worker, using WebGPU or a CPU/WASM fallback.
Photos stay in browser memory.

## Run locally

Use Node.js 22.12 or newer and Python 3.12.
The exporter downloads the default [scanner models](https://huggingface.co/DevWorld/flattener) on first use.

From the repository root:

```sh
uv sync
uv run python tools/models/export_onnx.py
cd web
npm ci
npm run dev
```

The exporter writes `manifest.json` and three FP32 ONNX models to `public/models/`.
Model files are generated locally and excluded from Git.
Pass `--bundle /path/to/bundle.json` to export a custom bundle.
The same files for the default models are on Hugging Face; download them instead of exporting with
`uvx hf download DevWorld/flattener manifest.json dewarp.onnx gate.onnx orient.onnx --revision 1.1.0 --local-dir web/public/models`.

## Build and host

The app is also deployed to https://flattener.worldsw.dev on Cloudflare, as an assets-only Worker (`wrangler.jsonc`).
Deploy it from `web/` with `npm run deploy:cloudflare` after `npx wrangler login`.
`scripts/cloudflare.mjs` splits models above Cloudflare's 25 MiB file limit into parts, which the app joins again, and adds a `_headers` file with the cross-origin isolation headers.

The app is deployed to https://deveworld.github.io/flattener/ by `.github/workflows/pages.yml` on each push to `main` that changes `web/`.
The workflow downloads the ONNX models from Hugging Face, runs the tests and publishes `web/dist/`.
GitHub Pages cannot set the cross-origin isolation headers below.
`public/coi-serviceworker.js` adds them through a service worker, so the first visit reloads once and the WASM fallback can use threads.

```sh
npm run build
npm run preview
```

Deploy `web/dist/` to a static host over HTTPS.
The build includes the models, fonts, sample image, workers and WASM runtime.
Relative asset paths support hosting beneath a subdirectory.
Serve `runtime/*.mjs` as JavaScript and `runtime/*.wasm` as `application/wasm`.
Enable WASM multithreading by serving the app and worker scripts with these response headers:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The development and preview servers set these headers automatically.
Use localhost during development.

The first scan downloads about 36 MB of models and 25 MB of runtime before compression.
Models are cached in browser storage when available.
Reopening the app loads its application assets and manifest from the host.
The WASM fallback uses up to four threads according to the device's logical core count.
Browsers without cross-origin isolation or WASM thread support use one thread.

## Scanning and editing

- Open **Camera** for a live page outline. Taking a photo starts flattening automatically.
  The shutter takes a still photo at the camera's full resolution where the browser supports it (`ImageCapture.takePhoto`), and a frame of the preview video otherwise.
  Some phones take the still photo with another lens than the preview (an ultra-wide one at 0.6x while the preview is at 1x).
  Each photo's view is matched against the preview; a photo with a different view is not used, the next try asks for the browser's default photo size, and if that also differs, that camera uses preview frames while the page is open.
  Photos above 20 megapixels are reduced when taken. The scan details show the method, the photo and preview sizes, a rejected photo's view, and the capture time.
  One hint at a time guides the framing, judged from the page outline only: no page, page cut off at the edge, too close, too small, tilted.
  The scanner loads and warms up while you frame the page, then stays ready for the next photo.
  You can take a photo while it prepares; processing starts as soon as it is ready.
  Camera access requires HTTPS or localhost and browser permission.
  If the live camera cannot open, **Use the camera app** takes the photo with the device's camera app, which needs no site permission.
  After the camera is allowed in the site settings, the live camera reopens by itself.
  The camera closes after capture, on leaving the page, or when the tab is hidden.
- Open a JPEG, PNG or WebP by file picker, drop or paste. Flattening starts as soon as the photo opens.
- Scans start with Quick processing and automatic orientation.
  Choose **Refine scan** on the result to run flip averaging and two crop refinement passes.
  The current scan stays visible and can be saved while refinement runs.
  Canceling or failing refinement keeps that scan and its edits; completing refinement replaces them.
- Follow page detection, geometry and color correction under **See processing steps**.
- Use **Clean color** to balance paper color and smooth lighting variation, darken text and whiten the paper.
  New scans start in the color mode you chose last.
- Review suggested fingers near page edges, then adjust the selection with the brush.
- Crop by dragging the frame's corners, edges or inside; rotate, undo edits, compare with the source and inspect at 100%.
- Save the scan as PNG, JPG or PDF. **Add to PDF** collects scans as pages, kept while you go back (**<**) to scan the next one, and **Save PDF** writes them into one file.
- Use **Scan settings** to change options and scan again.
  **Sharpen text in Original color** (on by default) sharpens the page in Original color, more when the photo was reduced more, and not at all when it was enlarged.
  Clean color does not sharpen: it already darkens text, and sharpening on top of it lowered OCR accuracy on test photos.

The browser flattens one photo at a time.
Inputs are limited to 40 MB and 20 megapixels, with at least 64 pixels on each side.
Output is capped at four megapixels and the selected long edge, subject to the source's sampling density.
The browser handles EXIF orientation, color profiles and decoding.
Processing time starts after model download and setup.

For book splitting, batches and the full rendering options, use the [Python scanner](../README.md#scan).

## Languages

The interface is in Korean when the browser's first preferred language is Korean, and in English otherwise.
`src/i18n.ts` translates the page and every text the app writes, using the English text as the key; Korean text lives in `src/i18n/ko.ts`.
Add an entry there for any new English interface text.

## Color and finger cleanup

Edits are recalculated from the original flattened image.
The editor supports mouse, touch and keyboard input: arrows move the brush, Space paints and Shift gives smaller movements.
Masks use up to 512 pixels on the longest side.
Brush strokes and applied edits each have 20 undo states.
Removal covers up to 20% of the page.
Canceling or failing an edit keeps the previous result.

Finger suggestions use the page detector's mask and the appearance of regions near the page edge.
Review the selection before applying it; gloves, fingers within the page and colored edge artwork may need brush adjustments.
Filling replaces selected pixels with surrounding paper color, including any writing or illustrations inside the selection.

Color correction estimates paper brightness and color from local patches, then adjusts the image toward that estimate.
In the same patches it estimates the ink level (a low percentile of brightness relative to the paper) and stretches the range between ink and paper so that text turns dark; levels near the paper ease to white.
Patches with many midtones (photos, colored print, dark pages) are stretched and whitened less, a page with few text patches is stretched less overall, and light tints keep their color.
Suggested fingers are left out of both estimates and are not stretched, so text beside a finger is corrected like the rest of the page.
For strong shadow boundaries, textured or colored paper and large illustrations, compare with **Original color**.
See [Bako et al., Removing Shadows from Images of Documents](https://civc.ucsb.edu/graphics/Papers/ACCV2016_DocShadow/) for background-estimation methods.

Processing previews show the initial scan before manual edits.
Canceling a scan stops its worker; cached models remain available for the next scan.

## Development

```sh
npm test
npm run build
```

Tests cover geometry against Python fixtures, color correction, sharpening, removal masks, rotations, capture guidance and the photo view check.
Regenerate the fixtures with `uv run python tools/web/generate_fixtures.py`.

Browser checks used Chromium on Linux with WASM.
Testing on hardware WebGPU devices is pending.

## Assets and licenses

Generate `public/sample.png` with `uv run python tools/web/generate_sample.py`.
The sample is a synthetic document.
Manrope comes from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/manrope) under [SIL OFL 1.1](public/fonts/OFL.txt).

The build copies [LICENSE](../LICENSE) to `dist/LICENSE`.
Update its component notices when upgrading ONNX Runtime.
