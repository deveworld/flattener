# Flattener in the browser

A static TypeScript + Vite application that runs Flattener's page gate, orientation
model, and dewarper in a dedicated Web Worker using ONNX Runtime Web. It tries
WebGPU first and falls back to WASM on the CPU. Photos stay in browser memory;
there is no processing server, upload endpoint, telemetry, or third-party CDN.

## Run locally

Use Node.js 22.12 or newer and the repository's Python 3.12 environment.
From the repository root:

```sh
uv sync --locked --group web-export
uv run --locked --group web-export python tools/export_web.py --bundle /path/to/bundle-v9.json
cd web
npm ci
npm run dev
```

In this development workspace the verified bundle is at
`~/.cache/flattener/scan-weights/bundle-v9.json`. Checkpoints and generated ONNX
files are not committed. A fresh clone needs a compatible release bundle before
it can process photos. Public weight hosting is not configured by this change.

The exporter verifies the source bundle and release checks, exports FP32 models,
runs the ONNX checker, and compares each model with PyTorch on varying input.
It writes `manifest.json` and three self-contained `.onnx` files to `public/models/`.
The release used here exports to 35,637,392 bytes total (about 35.6 MB).
The unused training 3D head is omitted from the dewarper export.

## Build and host

```sh
npm run build
npm run preview
```

Deploy the complete `web/dist/` directory to a static host over HTTPS. Relative
asset paths support hosting beneath a subdirectory. The build includes the model
files present in `public/models/`, fonts, the sample, worker, and WASM runtime.
Keep `runtime/*.mjs` served as JavaScript and `runtime/*.wasm` as
`application/wasm`. Localhost is also a secure context. Plain HTTP on a remote
host cannot provide the secure APIs used by the scanner.

No cross-origin isolation headers are needed: the WASM fallback uses one thread.
The first scan downloads about 36 MB of models plus roughly 25 MB of runtime
(before HTTP compression). Subsequent model loads use a hash-verified Cache API
entry where browser storage is available. This is not an installable offline PWA;
reopening the page still requires its application assets and manifest.

## Current capabilities

- Open one JPEG, PNG, or WebP by file picker, drop, or paste.
- Detect the dominant page and correct perspective and paper curvature.
- Choose Quick (one prediction) or Refined (flip TTA and two tight crop passes).
- Estimate text direction before and after correction; rotate the result manually.
- Watch real page masks, predicted surface meshes, crop refinements, the flattened
  page, estimated paper lighting, and color correction as they are computed.
  Revisit these images under **See processing steps** after scanning.
- Balance paper color and reduce smooth lighting variation with **Clean color**;
  switch to **Original color** to preserve the uncorrected scan's colors.
- Propose possible fingers near page edges; review and edit the red selection
  with an add/erase brush before filling it with surrounding paper color.
- Undo brush strokes, undo applied edits, or reset color, removal, and rotation.
- Compare the source and scan, inspect at 100%, and download a PNG.
- Cancel a scan by terminating its worker; cached models are reusable afterward.
- Reject non-finite, folded, or substantially out-of-frame predicted geometry.

The workspace shows one primary action at a time: choose a photo, flatten it,
then save the result. Refined correction and automatic orientation are selected
by default. Open **Scan settings** to change options, scan again, or inspect image
dimensions and processing details. Interface text stays at least 16px on mobile
and desktop.

## Color and finger cleanup

All cleanup runs locally. A second worker recalculates edits from the retained,
unedited flattened image, so repeated toggles and rotations do not accumulate
resampling or color loss. The editor supports pointer, touch, and keyboard input.
Arrow keys move the brush; Space paints; Shift gives smaller movements. The mask
uses at most 512 pixels on its longest side, with up to 20 brush undo states and
20 applied edit states. Removal is limited to 20% of the page. Canceling an edit
or failing to start its worker keeps the previous downloadable result.

Finger suggestions reuse the existing page model, whose training masks exclude
synthetic thumbs (`tools/gate/scenes.py`). Its probabilities are mapped through
the final dewarp grid, then filtered for localized edge intrusions and appearance
that differs from the surrounding paper. This is **not a dedicated hand model**:
it can miss fingertips, gloves, low-chroma hands, and fingers in the middle of a
page, or confuse colored edge artwork with a finger. Suggestions are never applied
without review. The brush can add missed areas or protect content. Fill restores
paper background only; **it cannot recover covered writing or illustrations**.

Color correction estimates a smooth RGB paper field from bright, low-chroma
samples in local tiles and applies bounded per-channel gains to the original
pixels. This follows the general background-estimation principle described by
[Bako et al., Removing Shadows from Images of Documents](https://civc.ucsb.edu/graphics/Papers/ACCV2016_DocShadow/),
with an independent lightweight implementation rather than a reproduction of the
paper or a neural shadow-removal model. Strong shadow boundaries, textured or
colored paper, and large illustrations may need **Original color**. Cleanup has
synthetic regression coverage, not a real-world quality benchmark.

Processing previews come directly from the gate tensor, accepted dewarp grids,
actual resampled image, and paper estimate. There are no simulated intermediate
results or artificial delays. Downloads have byte-based progress; inference uses
stage names and intermediate images because the runtime does not expose a reliable
completion percentage. The history describes the initial scan, before manual edits.

Input limits are 40 MB and 20 megapixels, with at least 64 pixels on each side.
Output is capped at the selected long edge and four megapixels without upscaling
past the estimated page sampling density. The displayed processing time excludes
the initial model download and session setup. EXIF orientation follows the browser's
image decoder; ICC handling and resizing can differ between browsers.

This is a browser implementation for **single pages**. It does not reproduce the
full Python scanner's candidate search, perspective fallback, content checks,
adaptive settle pass, focal-length aspect correction, sharpening, or book split.
Its page gate is a largest-component crop hint. Browser decoding can also differ
from Pillow. The Python release's SOTA benchmark scores are not browser benchmark
results. HEIC, PDF, OCR, batches, and automatic book splitting are not implemented.

## Verification

```sh
npm test
npm run build
```

The 20 tests compare preprocessing, antialiasing, crop placement, text patch selection,
grid composition, output sizing, quarter turns, and rendering against fixtures
generated by the existing Python implementation. They also exercise geometry
rejection, component selection, flip TTA disagreement, paper color and shadow
correction, colored-ink preservation, exact no-op/reset behavior, mask-only pixel
changes, selection limits, finger proposal false positives, and rotated masks.
Regenerate the geometry fixture
from the repository root with `uv run python tools/web_reference.py`.

The browser checks used headless Chromium 154 on Linux. On 2026-10-05, before adding
cleanup and processing previews, the synthetic sample completed Refined processing
on WASM in approximately 6.1 seconds and Quick processing in 1.4 seconds after model
setup. Those timings exclude the new preview and cleanup work. Comparison, rotation, 100% view, PNG
download, cancellation, retry with cached models, invalid-image recovery, and
worker-start failure recovery were checked. Layouts were checked from 320px to
1440px wide. A plain static server also completed a scan from a subdirectory URL.
Only same-origin asset GET requests were observed; photos were never sent.
The cleanup extension was also checked with the original sample and a procedural
thumb generated by the existing gate augmentation. Actual intermediate events,
color changes, pixel-exact undo/reset, automatic suggestions, brush and keyboard
editing, erase/undo, PNG saving, and 320–1440px layouts were verified. Worker-start
failure, edit cancellation/retry, scan cancellation/retry, and persistence of a
cleared removal selection were also checked. The original
sample produced no finger proposal after the appearance filter was added. Browser
checks are smoke tests; synthetic thumb detection does not establish accuracy on
real hands.
With previews and cleanup enabled, a Refined pass of the original sample took
approximately 6.8 seconds on this same WASM setup after model initialization.
The environment did not expose a usable GPU adapter, so hardware WebGPU speed
and parity still need testing on an actual supported device. This measurement is
a smoke check, not a benchmark or a promised latency.

`public/sample.png` is original synthetic material, visibly labeled in the app;
regenerate it with `uv run python tools/make_web_sample.py`. Manrope is self-hosted
under [SIL OFL 1.1](public/fonts/OFL.txt), sourced from
[Google Fonts](https://github.com/google/fonts/tree/main/ofl/manrope).
Font SHA-256: `3ae11c49db0455a3cc33e37d380f20fdb8c7f8b41dc07625c177e3d87a9d6ae6`.
ONNX Runtime Web is MIT licensed and incorporates components under MPL-2.0,
BSD and other licenses. The build copies the repository's [LICENSE](../LICENSE),
including component notices and MPL source links, to `dist/LICENSE`; the footer
links to it. The build pins the reviewed runtime version so an upgrade requires
updating those notices. Font licenses cover the font files, not the raster sample.
