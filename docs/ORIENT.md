# OrientNet-1 data, training and export

OrientNet-1 (`flattener.scan.orient`) predicts a page's rotation in clockwise quarter turns.
Apply `numpy.rot90(view, k)` to turn a prediction of `k` upright.
The scanner runs it before dewarping and again on the flattened preview.

The model has 0.44M parameters.
It combines a 256x256 grayscale page view with four 128x128 patches selected from text-rich areas at a page long side of 1024 pixels.

The model includes a 4,000-step fine-tune on forms, printed web pages and spreads.

## Data

The fine-tune uses these sources:

| Source | Kind | Train / val pages |
|---|---|---|
| Flattened SyntheticDoc renders | `syntheticdoc_flat` | 40,000 / 1,000 |
| Upright SyntheticDoc warped crops | `syntheticdoc_crop` | 40,000 / 1,000 |
| Generated pages with text, forms, tables and figures | `documents` | 40,000 / 2,000 |
| Generated two-page spreads | `spread` | 12,000 / 600 |
| Blank, figure-only and ruled paper | `text_free` | 3,000 / 300 |

The model was trained on generated pages with Korean, English and mixed text.
The generators produce English text.

SyntheticDoc sources support full-resolution downloads and compact copies.
Flat pages and crops use separate renders (`part`).
Release builds require samples with track `release` and a release license class.
A `pages` source can add upright page images listed in a `manifest.jsonl` (SmartDoc templates, scanner residual pages) with per-row track and license class.

The trainer draws `k` uniformly per item and augments inside the single sampling pass: margins -3% to +12% per side, skew (uniform +-5 degrees on flat pages, N(0, 3) degrees clipped to +-10 on crops), a smooth residual warp up to 3% on flat pages, illumination, shadow, contrast, gamma, blur, noise and JPEG.

## Training and export

Use [tools/data/orientation/build_data.py](../tools/data/orientation/build_data.py) to build training data.
For trainer options, run `python -m flattener.scan.orient_train --help`.

The data builder writes page batches and a sample manifest.
The trainer resumes from `checkpoints/latest.json`.
`best.json` names the checkpoint with the best mean accuracy across classes.
Use that checkpoint in the export command.
Install `onnx` and `onnxscript` for export; `onnxruntime` adds a comparison with PyTorch:

```sh
python -m flattener.scan.orient export \
    --checkpoint path/to/checkpoint.pt \
    --output orientnet-1.onnx
```

## Checkpoints

Checkpoints contain the model, training settings and calibration.
Load them with `Orienter.load(path)`.

Validation fits a temperature and confidence threshold on synthetic pages, limiting accepted wrong predictions to 0.5% of all labeled validation pages.
Choose the operating threshold on the real-photo development split and set it in the bundle's `orient.threshold` or with `--orient-threshold`.
Pass the same value to `export --threshold`.
The scanner defaults to 0.9.
