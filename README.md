# flattener

Turns a phone photo of a document into a flat scan: it finds the page, turns it upright,
flattens curled or crumpled paper and resamples the photo once. Open books are split into
two pages. When a page is cut off or missing, the scan says why and how to retake it.

All models are trained from scratch on redistributable data. Code is Apache-2.0; weight
downloads will be listed here once published.

## Results

MS-SSIM ↑ / LD ↓ / AD ↓, tight crop with flip test-time augmentation. The best published
scores were re-checked with the same evaluator.

| Benchmark | flattener | Best published |
|---|---|---|
| DocUNet | **0.589 / 4.91 / 0.208** | 0.571 / 5.51 / 0.218 |
| DIR300 | **0.718 / 3.78 / 0.126** | 0.702 / 4.26 / 0.131 |
| UVDoc | 0.767 / **4.53 / 0.088** | 0.800 / 5.83 / 0.098 |

On 68 real photos of Korean documents, 70 of 75 results are flattened (seven spreads split
into two pages); the other five are close-ups cut off by the frame. Median time on a
laptop CPU (2 threads): 1.86 s per photo.

## Install

Linux x86_64, Python 3.12, CPU PyTorch; versions are pinned in `uv.lock`.

```bash
uv sync --locked --python 3.12
uv run --locked pytest -q
```

## Use

```bash
python -m flattener.scan IMG... --out DIR --bundle bundle-v9.json
```

```python
from flattener.scan.io import save_scan
from flattener.scan.pipeline import Scanner

result = Scanner.load("bundle-v9.json").scan("photo.jpg")
save_scan(result, "out", "photo")  # out/photo.jpg, out/photo.scan.json
```

Each photo gives `<stem>.jpg` and a `<stem>.scan.json` record with the outcome
(`rectified`, `rectified_with_warnings`, `perspective`, `original_returned`, `failed`),
reason codes and guidance. Spreads give `<stem>-p1` and `<stem>-p2`. Useful options:
`--pages auto|1|2`, `--sharpen N` (0 turns it off), `--long-side N`, `--region`, `--quad`,
`--device cuda`. See `python -m flattener.scan --help`.

A bundle names the three models; paths are relative to the bundle file:

```json
{
  "schema": "flattener-scan-bundle-1",
  "dewarp": {"path": "soup-c70-ck/step_00040000.pt", "tta": "flip"},
  "gate": {"path": "page-gate-v1.2/best.pt"},
  "orient": {"path": "orient-v3/step_00004000.pt"}
}
```

`"tta": "flip"` also predicts the mirrored image and averages the two unless they disagree
(about 0.4 s more per photo on a CPU).

## How it works

Read (EXIF, large JPEGs decoded smaller) → find the page (page gate) → orient (quarter-turn
classifier) → dewarp (predict a grid, crop to it, predict again) → check (paper, cut-off,
neighbouring page) → render once and sharpen → split spreads. Details are in
[docs/INTERFACE_SPEC.md](docs/INTERFACE_SPEC.md).

| Model | Size | Trained on |
|---|---|---|
| Dewarp (UVDoc architecture) | 8M | SyntheticDoc pages, UVDoc raw captures with our pages laid on them, crease and fold renders |
| Page gate | 0.56M | synthetic desk scenes and SmartDoc frames |
| Orientation | 0.44M | synthetic pages, printed web pages, spreads |

Training guides: [dewarp](examples/dewarp/README.md), [page gate](examples/gate/README.md),
[orientation](examples/orient/README.md). `python -m flattener.scan.shippable CKPT` checks
that a checkpoint's lineage starts from scratch on releasable data.

## Evaluate

```bash
python -m flattener.dewarp.predict --weights W docunet --tta flip --input-dir CROPS \
    --output-dir OUT --force-render --quantize truncate --rerun 2 --rect
python -m flattener.benchmark docunet OUT --ref-resize uvdoc_view2_double --output docunet.json
```

`python -m flattener.benchmark --help` lists DocUNet, DIR300, UVDoc, OCR and calibration
commands. LD and AD need the SIFT flow sources (`FLATTENER_SIFTFLOW_SOURCE`).

## License

Apache-2.0, with the UVDoc-derived model code under MIT and the bundled font under OFL-1.1.
Data and code sources are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
