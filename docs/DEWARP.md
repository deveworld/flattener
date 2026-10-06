# Dewarp model

The dewarp model blends two checkpoints equally: one trained on clean pages and one trained with additional creases and folds.
Both use SyntheticDoc renders and pages composited on UVDoc captures of deformed paper.
Training uses captures outside the UVDoc benchmark split.

## Data preparation

- [Compact SyntheticDoc renders](../tools/data/syntheticdoc/compact.py) (CC BY 4.0).
- [Fetch CC0 backgrounds](../tools/data/backgrounds/fetch_polyhaven.py).
- [Re-texture UVDoc captures](../tools/data/uvdoc/retexture.py) (MIT).
- [Render creases and folds](../tools/data/creases/build.py).

The re-texturing tool uses the paper shape and shading from the captures and supplies page content and backgrounds.

## Training and inference

Each fine-tune runs 40,000 steps at batch 32, with learning rate 5e-5, 1,000 warmup steps, cosine decay and seed 1.
For trainer options, run `python -m flattener.dewarp.train --help`.

Blend the clean-page and crease/fold checkpoints with:

```sh
python tools/models/blend.py blended.pt clean-pages.pt creases-and-folds.pt --weight 0.5
```

The bundle uses flip test-time augmentation (`"tta": "flip"`).
Each call predicts the source and its mirror, then averages the aligned grids.
If they differ by more than 3% of the image side, it uses the source prediction.

See [benchmark results and evaluation commands](BENCHMARKS.md).
