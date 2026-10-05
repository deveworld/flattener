# Page gate data and training

The page gate (`flattener.regions.context`, `region-context-unet-2.0`) marks the page on
a 192x144 view of the whole photo. The scanner uses its mask to crop the first pass, to
turn away photos without a page and to judge the dewarp grid. An open-book spread counts
as one page region; the scanner flattens it whole and splits it afterwards. Other
documents under the page, desk objects and walls are not page. v1.2 has channels
16/32/64/128 (0.56M parameters).

| File | Use |
|---|---|
| `data-gate-release-v1.2.json` | build spec for `tools/gate/build_data.py` (the default gate) |
| `train-gate-release-v1.2.json` | train config for `python -m flattener.regions.train` |
| `*-v1.json`, `*-v1.1.json` | earlier builds: without the SmartDoc frames, documents under the page or white surfaces |

## Data

| Source | Kind | Train / val |
|---|---|---|
| SyntheticDoc renders (compact copy) placed in whole-photo scenes | `syntheticdoc_scene` | 60,000 / 1,000 |
| Own Korean/English pages, forms and sparse pages in scenes | `documents_scene` | 24,000 / 500 |
| Scenes without a page | `negative_scene` | 5,000 / 250 |
| SmartDoc 2015 challenge 1 frames, every sixth; documents 1-3 train, 4 val, 5 held out | `smartdoc_capture` | 2,529 / 300 |

The released gate was trained while the own pages drew Korean and English text; since
2026-10-05 the page generator draws English text only.

Scenes (`tools/gate/scenes.py`) go through the dewarp composite placement: tilt, quarter
turns, perspective and area.
- A close-up zooms the page past the frame.
- A facing page, drawn along the page's left or right edge, is part of the mask.
- A book edge (page stack) is drawn under the page and is left out of the mask.
- Documents under the page are left out of the mask.
- Laptops, keyboards, screens, book covers, white panels, mugs and cables are drawn around the page, as are plain light walls with a floor. None of them is in the mask.
- Thumbs over the page edge are cut out of the mask.
- Darker pages, cast shadows, blur, noise and JPEG round the scene off.

The SmartDoc frames manifest comes from `tools/gate/smartdoc_frames.py` over the extracted
`frames.tar.gz` of the v2.0.0 release (`FLATTENER_SMARTDOC_FULL`).

The builder records a dewarp-schema `data_spec` and a per-split license manifest. The
trainer copies both into every checkpoint for the ship gate. Photos taken for KR-Real
are for testing only and never enter a build.

## Training

```bash
python tools/gate/build_data.py --spec examples/gate/data-gate-release-v1.2.json \
    --output "$FLATTENER_CACHE/gate-release-v1.2" --workers 10
python -m flattener.regions.train --config examples/gate/train-gate-release-v1.2.json
```

- Loss: per-image BCE plus soft Dice. Items take a random quarter turn, mirror and colour jitter.
- Optimiser: AdamW with warmup and cosine decay, AMP on CUDA.
- Validation: scores IoU at 0.5 per row, or one minus the claimed share for a scene without a page. The mean over sources picks `best.pt`.
- Schedule: 30,000 steps of 64. On an RTX 3060 Ti the run is CPU-bound, at about 330 images/s (95 minutes).
- Build speed: about 60 scenes/s with 10 processes when the renders sit on an SSD.

## Results

| | v0 | v1.2 |
|---|---|---|
| SmartDoc documents 5 (72 frames), IoU | 0.939 | 0.977 |
| frames below IoU 0.9 | 7 | 0 |
| KR-Real vs v0, same scanner | — | 4 better, 2 worse |

Measure the SmartDoc IoU with `python tools/gate/eval_smartdoc.py GATE.pt [GATE2.pt ...]`.

On KR-Real, v1.2 improves four photos:
- a book in front of a laptop, now found;
- a spread, now split into two pages;
- a page cut at the right edge, now flattened cleanly;
- a page next to a keyboard, which is no longer included.

It worsens two:
- a steep close-up, now flattened upside down;
- a page whose output picks up a strip of the book under it.
