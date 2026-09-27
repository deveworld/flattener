# OrientNet-1 data, training and export

OrientNet-1 (`flattener.scan.orient`) tells how many clockwise quarter turns `k` the
text of a page view is turned; `numpy.rot90(view, k)` makes it upright. One model serves
the scanner twice: on the deskewed page crop before dewarp and on the flat preview after
it. It reads a 256x256 letterboxed grayscale view and four 128x128 patches cut at page
long side 1024 from the cells with the most text energy, shares one depthwise-separable
trunk across the five views and pools them by a text-saliency softmax (0.44M parameters,
0.34 GFLOPs per page, about 10 ms on two laptop CPU threads).

| File | Use |
|---|---|
| `data-orient-release.json` | build spec for `tools/orient/build_data.py` |
| `train-orient-release.json` | train config for `python -m flattener.scan.orient_train` |

## Data

| Source | Kind | Train / val pages |
|---|---|---|
| SyntheticDoc clean renders flattened through their ground-truth maps (long side 768-1024) | `syntheticdoc_flat` | 120,000 / 1,000 |
| SyntheticDoc warped crops, turned so the page's up vector points up (15% stored margin) | `syntheticdoc_crop` | 120,000 / 1,000 |
| Own Korean/English/mixed pages: `synthetic/documents.py` pages and composed pages of procedural text, formulas, tables and figures in Noto Sans KR | `documents` | 40,000 / 2,000 |
| Blank, figure-only and ruled paper (uniform target) | `text_free` | 3,000 / 300 |

The SyntheticDoc sources read `../dewarp/data-release-raw.json` (the full-resolution
download named by `FLATTENER_DATA`); `../dewarp/data-release.json` (the compact copy) also
works. Flat pages and crops use disjoint renders (`part`). A release build refuses any
sample without track `release` and a release license class. A `pages` source can add
upright page images listed in a `manifest.jsonl` (SmartDoc templates, scanner residual
pages) with per-row track and license class.

The trainer draws `k` uniformly per item and augments inside the single sampling pass:
margins -3% to +12% per side, skew (uniform +-5 degrees on flat pages, N(0, 3) degrees
clipped to +-10 on crops), a smooth residual warp up to 3% on flat pages, illumination,
shadow, contrast, gamma, blur, noise and JPEG.

## Home PC (i5-11400, RTX 3060 Ti)

Windows `cmd.exe`, from the repository root:

```bat
set FLATTENER_DATA=D:\flattener-data
set FLATTENER_CACHE=E:\cache
set FLATTENER_RUNS=E:\runs
python tools\orient\build_data.py --spec examples\orient\data-orient-release.json ^
    --output %FLATTENER_CACHE%\orient-release-v1 --workers 11
python -m flattener.scan.orient_train --config examples\orient\train-orient-release.json
```

The build writes one part file per 500 pages and resumes from them after an interruption;
the trainer resumes from `checkpoints/latest.json`. `best.json` names the checkpoint with
the best validation macro accuracy. Export it (needs `onnx` and `onnxscript`; with
`onnxruntime` installed the export records its logit difference to torch):

```bat
python -m flattener.scan.orient export ^
    --checkpoint %FLATTENER_RUNS%\orient-v1\checkpoints\step_000NNNNN.pt ^
    --output %FLATTENER_RUNS%\orient-v1\orientnet-1.onnx
```

Expected cost, extrapolated from laptop measurements (Core Ultra 5 228V, 2 workers):

| Stage | Basis | Home PC |
|---|---|---|
| Build, 283k pages, CPU | per page: flat 0.10 s, crop 0.06 s, own 0.13 s, text-free 0.01 s (with decode); 11 workers | about 50 min, plus 1-3 min to index SyntheticDoc |
| Train, 16,000 steps x 128 = 2.05M samples | 11.5 ms per item per loader worker; 10 workers give about 600-700 items/s, below what the 3060 Ti needs (about 1 GFLOP per sample for forward and backward) | about 55 min |
| Validation, 16 x 16,300 items | same loader rate | about 7 min |
| Total | | about 2 h, $0 |

Disk: about 30-35 GB of grayscale JPEG at quality 85 (measured at quality 90: flat
150 kB, crop 160 kB, own 120 kB, text-free 60 kB per page).

## Records and the ship gate

Checkpoints and their `.pt.json` sidecars use the dewarp record layout, so
`flattener.scan.shippable` gates them like dewarp weights:

- `track` and `lineage`: `{"kind": "scratch"}`, then the training step with its track,
  `data_spec_sha256`, `orient_data_sha256` (the built pages) and code sha256;
- `data_spec`: a `flattener-dewarp-data-1` spec naming every source of the build (the
  SyntheticDoc dewarp sources, and own pages as `converted` sources), with
  `data_spec_sha256` its sha256;
- `data_sources`: per split, each source's track, license and license-class counts;
- `orient_data`: the build summary; `calibration`: temperature and threshold.

`Orienter.load` and `export` refuse weights outside the release lineage unless
`research=True` (`orient.release_problems`, which ends with the scanner's ship gate
`scan.shippable`). A fine-tuned model also needs the data spec of every parent training
step; the lineage entry of its parent carries it.

Validation fits a temperature and the lowest threshold with at most 0.5% silent errors on
the synthetic val split (`calibration.source = "synthetic_val"`). The design sets the
operating threshold on the real-photo dev split; pass it to `export --threshold` once it
is measured. Until then the scanner starts from 0.9.
