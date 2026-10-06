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
  The scanner loads and warms up while you frame the page, then stays ready for the next photo.
  You can take a photo while it prepares; processing starts as soon as it is ready.
  Camera access requires HTTPS or localhost and browser permission.
  The camera closes after capture, on leaving the page, or when the tab is hidden.
- Open a JPEG, PNG or WebP by file picker, drop or paste.
- Scans start with Quick processing and automatic orientation.
  Choose **Refine scan** on the result to run flip averaging and two crop refinement passes.
  The current scan stays visible and can be saved while refinement runs.
  Canceling or failing refinement keeps that scan and its edits; completing refinement replaces them.
- Follow page detection, geometry and color correction under **See processing steps**.
- Use **Clean color** to balance paper color and smooth lighting variation.
- Review suggested fingers near page edges, then adjust the selection with the brush.
- Undo edits, rotate the result, compare it with the source, inspect at 100% and save a PNG.
- Use **Scan settings** to change options and scan again.

The browser processes one page at a time.
Inputs are limited to 40 MB and 20 megapixels, with at least 64 pixels on each side.
Output is capped at four megapixels and the selected long edge, subject to the source's sampling density.
The browser handles EXIF orientation, color profiles and decoding.
Processing time starts after model download and setup.

For book splitting, batches and the full rendering options, use the [Python scanner](../README.md#scan).

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
For strong shadow boundaries, textured or colored paper and large illustrations, compare with **Original color**.
See [Bako et al., Removing Shadows from Images of Documents](https://civc.ucsb.edu/graphics/Papers/ACCV2016_DocShadow/) for background-estimation methods.

Processing previews show the initial scan before manual edits.
Canceling a scan stops its worker; cached models remain available for the next scan.

## Development

```sh
npm test
npm run build
```

Tests cover geometry against Python fixtures, color correction, removal masks and rotations.
Regenerate the fixtures with `uv run python tools/web/generate_fixtures.py`.

Browser checks used Chromium on Linux with WASM.
Testing on hardware WebGPU devices is pending.

## Assets and licenses

Generate `public/sample.png` with `uv run python tools/web/generate_sample.py`.
The sample is a synthetic document.
Manrope comes from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/manrope) under [SIL OFL 1.1](public/fonts/OFL.txt).

The build copies [LICENSE](../LICENSE) to `dist/LICENSE` and links it from the footer.
Update its component notices when upgrading ONNX Runtime.
