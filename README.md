# Flattener

**Open-source SOTA document scanner.**

Turn phone photos of documents and open books into flat, upright scans. Flattener finds
the page, corrects perspective and curved or crumpled paper, and splits book spreads
into two pages. Everything runs locally.

Inference, training, data preparation and evaluation code are included. All three
scanner models are trained from scratch on redistributable data.

## SOTA results

Flattener outperforms the best published results on **DocUNet and DIR300 across all
three metrics**, and achieves the **best LD and AD on UVDoc**.

Scores are **MS-SSIM ↑ / LD ↓ / AD ↓**. Higher MS-SSIM and lower LD and AD are better.

| Benchmark | Flattener | Best published |
|---|---|---|
| DocUNet | **0.589 / 4.91 / 0.208** | 0.571 / 5.51 / 0.218 |
| DIR300 | **0.718 / 3.78 / 0.126** | 0.702 / 4.26 / 0.131 |
| UVDoc | 0.767 / **4.53 / 0.088** | 0.800 / 5.83 / 0.098 |

These results use two tight crop refinement passes and flip test-time augmentation
(TTA). The best published scores were re-checked with the same evaluator. See the
[model comparison and reproduction commands](examples/dewarp/README.md#scores).

On **68 real photos of Korean documents**, Flattener produces 75 page results, including
seven spreads split into two pages. It flattens 70 of them; the remaining five are
close-ups with the page cut off by the frame. Median processing time is **1.86 seconds
per photo on a laptop CPU with two threads**.

## Features

- **Automatic scanning:** page detection, upright orientation, perspective and curve
  correction, output sizing and sharpening in one pipeline.
- **Open books:** find the fold, split the spread and flatten each page, with left-to-right
  or right-to-left page order.
- **One final resampling:** compose the geometry, then sample the source photo directly
  for each output page.
- **Useful failure handling:** record cut-off pages, uncertain orientation and other
  conditions with guidance. Return the original when a page cannot be flattened.
- **Manual control:** override the page region, corners, orientation, aspect ratio or
  output size. Save a correction plan and render it again from the same source.

## Install

The pinned environment targets Linux x86_64, Python 3.12 and CPU PyTorch.

```bash
uv sync --locked --python 3.12
```

## Browser app

The [TypeScript + Vite web app](web/README.md) processes a document photo entirely
in your browser, using WebGPU when available and a local CPU/WASM fallback.
Choose a photo, flatten it, compare the original and result, and save a PNG.
Photos are never uploaded to a processing server.

With Node.js 22.12 or newer and a compatible scanner bundle:

```bash
uv sync --locked --group web-export
uv run --locked --group web-export python tools/export_web.py --bundle /path/to/bundle-v9.json
cd web
npm ci
npm run dev
```

`npm run build` produces a static site in `web/dist/`, including the locally
exported models. The browser edition currently handles single pages; see its
[capabilities, hosting instructions, and verification notes](web/README.md).

## Scan

Scanning requires model checkpoints and a bundle JSON file. Public weight downloads
will be linked here once published; the [training guides](#training) describe the models
and data preparation.

```bash
uv run --locked python -m flattener.scan photo.jpg --out scans --bundle bundle-v9.json
```

Pass multiple photos to scan a batch. Each photo produces `<stem>.jpg` and a
`<stem>.scan.json` record; a split spread produces `<stem>-p1` and `<stem>-p2`.
The record includes the outcome, reasons, guidance, model identity and stage timings.
Outcomes are `rectified`, `rectified_with_warnings`, `perspective`, `original_returned`
and `failed`.

From Python:

```python
from flattener.scan.io import save_scan
from flattener.scan.pipeline import Scanner

scanner = Scanner.load("bundle-v9.json")
result = scanner.scan("photo.jpg")
save_scan(result, "scans", "photo")  # scans/photo.jpg, scans/photo.scan.json
```

Use `scanner.scan_pages("book.jpg")` to get one or two page results with automatic spread
splitting, then save each result with `save_scan`.

| Option | Use |
|---|---|
| `--pages auto\|1\|2` | Detect spreads automatically, keep one image or split into two |
| `--page-order ltr\|rtl` | Choose the reading order of split pages |
| `--region L,T,R,B` / `--quad x1,y1,...,x4,y4` | Set the page box or its four corners |
| `--long-side N` / `--output-size H,W` | Set the output dimensions |
| `--format png` | Save lossless PNG instead of JPEG |
| `--sharpen N` | Set sharpening strength; 0 turns it off |
| `--timing run.jsonl` | Record processing times; add `--no-cache` for repeated measurements |

Run `uv run --locked python -m flattener.scan --help` for all options. CUDA inference is
available with a CUDA-enabled PyTorch installation and `--device cuda`.

### Model bundle

A bundle names the scanner's three models. Paths are relative to the bundle file.
The current model combination is:

```json
{
  "schema": "flattener-scan-bundle-1",
  "dewarp": {"path": "soup-c70-ck/step_00040000.pt", "tta": "flip"},
  "gate": {"path": "page-gate-v1.2/best.pt"},
  "orient": {"path": "orient-v3/step_00004000.pt"}
}
```

`"tta": "flip"` predicts the image and its mirror, then averages the grids after turning
the mirrored prediction back. When the two disagree, it keeps the original prediction.
This adds about 0.4 seconds per photo on a CPU.

## How it works

The scanner reads the photo with EXIF orientation applied, finds the page and turns it
upright. The dewarp model predicts a coordinate grid, crops around the page and predicts
again. The scanner checks the geometry and page content, splits spreads when present,
then renders and sharpens each output page from the source photo.

Large JPEGs can be decoded at reduced resolution. Model outputs are cached so manual
edits reuse earlier work.

| Model | Size | Trained on |
|---|---|---|
| Dewarp (UVDoc architecture) | 8M | SyntheticDoc pages, UVDoc raw captures with our pages laid on them, crease and fold renders |
| Page gate | 0.56M | synthetic desk scenes and SmartDoc frames |
| Orientation (OrientNet) | 0.44M | synthetic pages, printed web pages, spreads |

## Training

The training guides cover data preparation, model configurations and checkpoints:

- [Dewarp](examples/dewarp/README.md): SyntheticDoc, re-textured captures, crease and fold
  renders, fine-tuning and the final weight blend.
- [Page gate](examples/gate/README.md): synthetic scenes and SmartDoc capture frames.
- [Orientation](examples/orient/README.md): page views, text patches, training and export.

Checkpoints record their training lineage, data sources and hashes. The scanner checks
that the full lineage starts from scratch on release data before accepting default
weights. To inspect a checkpoint:

```bash
uv run --locked python -m flattener.scan.shippable checkpoint.pt
```

## Evaluate

Reproduce the DocUNet configuration used in the results table:

```bash
uv run --locked python -m flattener.dewarp.predict --weights checkpoint.pt docunet \
    --tta flip --input-dir CROPS --output-dir PREDICTIONS \
    --force-render --quantize truncate --rerun 2 --rect
uv run --locked python -m flattener.benchmark docunet PREDICTIONS \
    --convention 128_excl64 --ref-resize uvdoc_view2_double --output docunet.json
```

`CROPS` is the DocUNet cropped-photo directory. Reference scans use the evaluator's
default location, or a directory supplied with `--references-dir`.

The benchmark commands cover DocUNet, DIR300, UVDoc, OCR and evaluator calibration.
LD and AD require the SIFT flow sources (`FLATTENER_SIFTFLOW_SOURCE`). Reports record
the model, input hashes, processing settings and prediction failures. See
`uv run --locked python -m flattener.benchmark --help`.

Run the default test suite:

```bash
uv run --locked pytest -q
```

## License

Project code is Apache-2.0. UVDoc-derived code and ONNX Runtime Web are MIT;
the bundled fonts are OFL-1.1. The web runtime also includes components under
MPL-2.0, BSD and other licenses. Copyright notices, license texts and the MPL
source links are in [LICENSE](LICENSE).

Noto Sans KR is used by the synthetic training-page generators; scanning does
not load it. Manrope is used by the web interface and sample generator. Font
license copies accompany the font files. OFL applies to the font software,
not the generated page images.

The benchmark's external SIFT-flow implementation has no verified package-wide
redistribution license; see the provenance note in LICENSE. External datasets
and pretrained weights have their own terms; the scanner's release checks
reject external-weight lineage.
