# Page gate data and training

The page detector (`flattener.regions.context`) predicts a mask from a 192x144 view of the photo.
The scanner uses it to find the page, crop the first pass and check the dewarp grid.
An open book forms one region, which is split after flattening.
The model has 16/32/64/128 channels and 0.56M parameters.

## Data

| Source | Kind | Train / val |
|---|---|---|
| SyntheticDoc renders (compact copy) placed in whole-photo scenes | `syntheticdoc_scene` | 60,000 / 1,000 |
| Generated Korean/English pages, forms and sparse pages in scenes | `documents_scene` | 24,000 / 500 |
| Background scenes | `negative_scene` | 5,000 / 250 |
| SmartDoc 2015 challenge 1 frames, every sixth; documents 1-3 train, 4 val, 5 held out | `smartdoc_capture` | 2,529 / 300 |

The model was trained on generated pages with Korean and English text.
The page generator produces English text.

Scenes (`tools/data/pages/scenes.py`) vary page tilt, rotation, perspective and size.

- Close-ups extend the page beyond the frame.
- Facing pages belong to the page mask.
- Page stacks, underlying documents and thumbs are labeled as background.
- Backgrounds include laptops, keyboards, screens, book covers, white panels, mugs, cables, walls and floors.
- Augmentations add darker paper, shadows, blur, noise and JPEG compression.

The SmartDoc frames manifest comes from `tools/data/pages/smartdoc_frames.py` over the extracted `frames.tar.gz` of the v2.0.0 release (`FLATTENER_SMARTDOC_FULL`).

Checkpoints include training settings and source licenses.

## Training

Use [tools/data/pages/build_data.py](../tools/data/pages/build_data.py) to build training data.
For trainer options, run `python -m flattener.regions.train --help`.

- Loss: binary cross-entropy plus soft Dice per image, with random quarter turns, mirroring and color jitter.
- Optimizer: AdamW with warmup, cosine decay and mixed precision on CUDA.
- Validation: IoU at a 0.5 threshold for page images; one minus the predicted page fraction for background scenes.
  The mean across sources selects `best.pt`.
- Schedule: 30,000 steps of 64.

## Results

The model achieves 0.977 mean IoU on 72 held-out SmartDoc document 5 frames.
All 72 frames have IoU of at least 0.9.
Measure IoU with `python tools/data/pages/eval_smartdoc.py checkpoint.pt`.
