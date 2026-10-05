# Dewarp training

These configs rebuild the scanner's dewarp model, the one in `bundle-v9.json`.
The checkpoints of every step are in the private Hugging Face backup `DevWorld/flattener`:
`ancestors/` holds steps 1-3, `parents/` holds step 4, and `soup-c70-ck/` holds step 5.

The configs name machine-specific locations with environment variables. The trainer
expands `${VAR}` in `data_spec`, `output_dir`, `init_path` and in data-spec source roots.
An unset variable stops the run with its name. The data-spec hash keeps the unexpanded
roots, so a run resumes on another machine with the same spec.

| Variable | Points to |
|---|---|
| `FLATTENER_CACHE` | directory holding `syntheticdoc-release/` (written by `tools/syntheticdoc/compact.py`), `backgrounds/`, the re-textured UVDoc builds and the crease/fold build |
| `FLATTENER_RUNS` | parent directory of run outputs |
| `FLATTENER_INIT_CHECKPOINT` | the checkpoint a fine-tune starts from, with its `.pt.json` sidecar next to it |
| `FLATTENER_DATA` | data root with `release/syntheticdoc` (the full-resolution SyntheticDoc download read by `data-release-raw.json`, which the OrientNet data specs use) and the research-track data |

## Data

```sh
# SyntheticDoc renders (CC BY 4.0), compact copy at 0.75 scale
python tools/syntheticdoc/compact.py --source <download-root>/release/syntheticdoc \
    --output "$FLATTENER_CACHE/syntheticdoc-release" --workers 8
# CC0 background textures for the training composites
python tools/backgrounds/fetch_polyhaven.py --output "$FLATTENER_CACHE/backgrounds/polyhaven-768" \
    --count 300 --categories floor wall wood fabric plaster plaster-concrete concrete tiles \
    leather metal indoor clean cotton denim wool fleece "raw wood"
# Clean pages laid on UVDoc's raw captures of deformed paper (MIT, unpacked to
# $FLATTENER_CACHE/uvdoc_raw), 60,000 items each; repeat for v2, v3 and v4
python tools/uvdoc/retexture.py --spec examples/dewarp/uvdoc-retexture-release-v1.json \
    --output "$FLATTENER_CACHE/uvdoc-retexture-release-v1"
# Crease and fold renders on cleaned pages
python tools/crumple/build.py --spec examples/dewarp/crumple-release-v7.json \
    --output "$FLATTENER_CACHE/crumple-release-v7"
```

`retexture.py` lays pages on UVDoc's raw photos of blank paper through their texture
coordinate maps. The paper's shape and shading come from real captures, while the pages,
backgrounds and stored maps are ours. The captures of the UVDoc benchmark are left out.
v1 and v2 use SyntheticDoc renders flattened through their ground truth. v3 and v4
divide the render's light out of those pages (`clean_texture`) and put 30% of them on
patterned backgrounds; their `split_name` reuses v1's capture split.

## Steps

| Step | Config | Starts from | Data |
|---|---|---|---|
| 1. `dewarp-v1` | trained before this folder; its sidecar records the run | scratch | SyntheticDoc (`release-clean-v3-extra`), 140,625 steps at batch 64 |
| 2. `dewarp-v2` | `train-release-perspective-fold-ft.json` | step 1 | `data-release.json`, with perspective and facing-page composites |
| 3. `dewarp-uvdoch` | `train-release-uvdoc-half-ft.json` | step 2 | `data-release-uvdoc-half.json`: SyntheticDoc .5, re-textured v1 .25, v2 .25 |
| 4a. `dewarp-uvdocc70` | `train-release-uvdoc-clean70-ft.json` | step 3 | `data-release-uvdoc-clean70.json`: SyntheticDoc .3, v3 .35, v4 .35 |
| 4b. `dewarp-uvdocck` | `train-release-uvdoc-clean-crumple-ft.json` | step 3 | `data-release-uvdoc-clean-crumple.json`: SyntheticDoc .35, v3 .25, v4 .25, crease/fold v7 .15 |
| 5. `soup-c70-ck` | `python tools/soup.py OUT 4a 4b --weight 0.5` | 4a, 4b | none (a 50/50 weight blend) |

Each fine-tune runs 40,000 steps at batch 32, with learning rate 5e-5 (1,000 warmup
steps, cosine decay) and seed 1. That takes about 3 h on an RTX 3060 Ti at about 120
samples/s, with NCHW and AMP; channels_last ran about 4x slower on that card.

```sh
FLATTENER_INIT_CHECKPOINT=<step 2>/step_00040000.pt \
    python -m flattener.dewarp.train --config examples/dewarp/train-release-uvdoc-half-ft.json
```

Step 4b on its own pulls a curled facing page into one real book photo. The blend keeps
4b's DIR300 gains without that. `bundle-v9.json` runs the blend with flip test-time
augmentation (`"tta": "flip"` on the dewarp entry). Each model call also predicts the
mirror image and averages the turned-back grid in, unless the two disagree by more than
3% of the image side.

## Scores

Scores are MS-SSIM / LD / AD from the calibrated evaluator (`python -m flattener.benchmark`).
"One pass" is a single prediction. "Tight crop" adds two turned tight-crop passes
(`--rerun 2 --rect`).

| Model | DocUNet | UVDoc bench | DIR300 |
|---|---|---|---|
| 2. `dewarp-v2`, one pass | 0.573 / 5.42 / 0.240 | 0.652 / 7.11 / 0.211 | 0.643 / 6.70 / 0.255 |
| 3. `dewarp-uvdoch`, one pass | 0.575 / 5.29 / 0.231 | 0.725 / 5.08 / 0.114 | 0.684 / 4.70 / 0.176 |
| 4a. `dewarp-uvdocc70`, one pass | 0.577 / 5.22 / 0.234 | 0.738 / 4.59 / 0.098 | 0.693 / 4.42 / 0.166 |
| 4b. `dewarp-uvdocck`, one pass | 0.582 / 5.18 / 0.227 | 0.741 / 4.75 / 0.106 | – |
| 5. blend, one pass | 0.581 / 5.16 / 0.228 | 0.739 / 4.69 / 0.103 | 0.699 / 4.31 / 0.159 |
| 5. blend, tight crop | 0.581 / 4.98 / 0.214 | 0.753 / 4.80 / 0.109 | 0.714 / 3.95 / 0.133 |
| 5. blend, tight crop, flip TTA (`bundle-v9`) | 0.589 / 4.91 / 0.208 | 0.767 / 4.53 / 0.088 | 0.718 / 3.78 / 0.126 |
| best published | 0.571 / 5.51 / 0.218 | 0.800 / 5.83 / 0.098 | 0.702 / 4.26 / 0.131 |

The UVDoc benchmark is made by UVDoc's own `create_final.py` from held-out captures, so
its MS-SSIM rewards copying that generator's look. Real photos (KR-Real, DocUNet,
DIR300) decide.

```sh
python -m flattener.dewarp.predict --weights W --device cuda docunet --tta flip \
    --input-dir CROPS --output-dir OUT --force-render --quantize truncate --rerun 2 --rect
python -m flattener.benchmark docunet OUT --ref-resize uvdoc_view2_double --references-dir REFS --output R.json
python -m flattener.benchmark dir300-predict --weights W --device cuda --tta flip --inputs-dir DIST \
    --output-dir OUT3 --force-render --quantize truncate --rerun 2 --rect
python -m flattener.benchmark dir300 OUT3 --references-dir GT --output R3.json
python -m flattener.benchmark uvdoc-predict --weights W --device cuda --tta flip --output-dir OUTU \
    --root UVDOC --rerun 2 --rect
python -m flattener.benchmark uvdoc OUTU --root UVDOC --output RU.json
```

## Tried and dropped

These were trained and scored. Their configs were removed, but the code options remain.

| Change | Result |
|---|---|
| 1440x1024 input with a 90x64 grid | worse on every benchmark: the backbone's grid is input/16, so a bigger input shrinks its view of the page |
| an 89x61 output grid at 712x488 | no gain: the 45x31 target resized to 89x61 is 0.04 px from the 89x61 one |
| procedural crumple renders alone (`tools/crumple`, v1-v6) | DIR300 and UVDoc bench rose, but real book photos got worse (facing page taken in, cut pages cropped, spreads unsplit) |
| 3-D target from the crumple surfaces (`loss_grid3d`) | DocUNet worse with it |
| 30% and 70% shares of the raw re-textured items | 50% balanced the benchmarks best; 70% raised DIR300 and lowered DocUNet |
| Doc3D page geometry with our light (`"geometry": "doc3d_grid"`) | below the UVDoc-capture model on all three benchmarks |
| UVDoc's compositing recipe (`"recipe": "uvdoc"`) | UVDoc-bench MS-SSIM 0.776 but LD 7.22 and DIR300 down: it teaches a border inside the page |
| axis-alignment and page-space losses | no change in DIR300 AD |
| a second seed, and seed blends | within run noise |
| sheets of paper under the page (`stack_prob`, with and without a contact shadow) | fixed a stacked-paper photo, but DIR300 single-pass LD rose from 4.42 to 5.00-5.35: the model cut pages short at column rules and header bands |

## Notes

The start record of a fine-tune carries the parent's data spec, so
`python -m flattener.scan.shippable` checks every training step of the chain from the
child's sidecar alone.

`train-research-uvdoc-recipe.json` with `data-research-doc3d-uvdoc.json` maps the
upstream UVDoc training recipe (research track, never shipped).

On Windows `cmd.exe`, set the variables with `set FLATTENER_CACHE=E:\cache` and so on.
Command-line options (`--output-dir`, `--data-spec`, `--init-path`, `--num-workers`, ...)
override the file. Keep configs with machine paths outside the repository.
