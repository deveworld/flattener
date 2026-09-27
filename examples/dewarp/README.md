# Dewarp training examples

The configs name machine-specific locations with environment variables. The trainer
expands `${VAR}` in `data_spec`, `output_dir`, `init_path` and in data-spec source roots;
an unset variable stops the run with its name. The data-spec hash keeps the unexpanded
roots, so a run resumes on another machine with the same spec.

| Variable | Used by | Points to |
|---|---|---|
| `FLATTENER_CACHE` | `data-release.json`, `train-release-perspective.json` | directory holding `syntheticdoc-release/`, the copy written by `tools/syntheticdoc/compact.py`, and `backgrounds/` |
| `FLATTENER_DATA` | `data-research-doc3d-uvdoc.json`, `data-release-raw.json` | data root with `research/doc3d` and `research/uvdoc_final`, or with `release/syntheticdoc` (the full-resolution SyntheticDoc download of `tools/syntheticdoc/sdtool.py`) |
| `FLATTENER_RUNS` | `train-*.json` | parent directory of run outputs |
| `FLATTENER_UVDOC_WEIGHTS` | `train-release-uvdoc-init.json` | UVDoc `model/best_model.pkl` (MIT; sha256 `7e90861b…da23`) |
| `FLATTENER_INIT_CHECKPOINT` | `train-release-hr.json`, `train-release-grid89.json` | release-track dewarp checkpoint to fine-tune (its `.pt.json` sidecar next to it) |

Build the compact release copy from a finished (or partial) SyntheticDoc download, then train:

```sh
python tools/syntheticdoc/compact.py --source <download-root>/release/syntheticdoc \
    --output "$FLATTENER_CACHE/syntheticdoc-release" --workers 8
python -m flattener.dewarp.train --config examples/dewarp/train-release-uvdoc-init.json
```

`train-release-hr.json` fine-tunes a release checkpoint at 1440x1024 input with the
backbone's own 90x64 grid. It reads the full-resolution download through
`data-release-raw.json`, since the compact copy (0.75 scale, 1080x768) is smaller than
that input. The raw val split selects the same 2,000 samples as the compact one. Weights
trained at 712x488 do not carry over to another input size without training: on 400
val samples a 712x488 checkpoint scored 2.4 px docunet EPE at its own size, 73 px at
1024x704 and 146 px at 1440x1024. channels_last trained ~1.55x faster than NCHW on an
RTX 5090 (199 against 129 samples/s at batch 32, 12 GiB).

`train-release-grid89.json` fine-tunes a release checkpoint at 712x488 input with an
89x61 output grid, about twice the backbone's 45x31 nodes per side (UVDoc's own target
grid size). The model resizes the bridge features to the grid before the heads
(bilinear, align_corners), so every other node sits on a backbone feature; targets,
losses, validation and `predict` all follow `model.grid_size`. It reads the compact
copy through `data-release.json`. The page extent that places training and validation
crops comes from the grid nodes, so about 5% of the crops differ from the 45x31 ones and
the validation signature differs from a 45x31 run. On the first 400 val samples a
712x488 checkpoint scored 2.40 px docunet EPE at its own grid, 2.41 px with its 45x31
output resized to 89x61 (the like-for-like figure on 89x61 nodes) and 12.7 px as the
89x61 model before training. The 45x31 target resized to 89x61 has a mean EPE of 0.04 px
against the 89x61 target, so on SyntheticDoc maps the denser grid adds little the 45x31
grid cannot represent.

`train-release-perspective.json` repeats the recipe of the best release model so far
(scratch, 9.0M samples at batch 64) with composites on 30% of the samples: the page is placed on a new background with one
homography (in-plane tilt, corner shifts up to 12% of the page, page area 5-100% of the
frame) that maps the image and the target grid alike. Models trained on SyntheticDoc
alone undo only about two thirds of the foreshortening of a tilted camera, which is most
of the gap to UVDoc on its benchmark. The backgrounds are CC0 textures:

```sh
python tools/backgrounds/fetch_polyhaven.py --output "$FLATTENER_CACHE/backgrounds/polyhaven-768" \
    --count 300 --categories floor wall wood fabric plaster plaster-concrete concrete tiles \
    leather metal indoor clean cotton denim wool fleece "raw wood"
```

The `tilt15` validation view shows the same pages through a tilted camera (frame
corners moved by up to 15%). `python -m flattener.dewarp.probe` compares a checkpoint's
page and tilt views on the val split: grid error in percent of the page diagonal and
the under-correction factor (0: the foreshortening is undone, 1: it is left in). On 200
local SyntheticDoc val renders that best release model scored 0.27% / 0.01 on the page
view and 2.02% / 0.40 on the tilted view.

On Windows `cmd.exe`, set the variables with `set FLATTENER_CACHE=E:\cache` and so on.
Command-line options (`--output-dir`, `--data-spec`, `--init-path`, `--num-workers`, ...)
override the file. Keep configs with machine paths outside the repository.
