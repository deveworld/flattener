# Dewarp training examples

The configs name machine-specific locations with environment variables. The trainer
expands `${VAR}` in `data_spec`, `output_dir`, `init_path` and in data-spec source roots;
an unset variable stops the run with its name. The data-spec hash keeps the unexpanded
roots, so a run resumes on another machine with the same spec.

| Variable | Used by | Points to |
|---|---|---|
| `FLATTENER_CACHE` | `data-release.json`, `train-release-perspective*.json` | directory holding `syntheticdoc-release/`, the copy written by `tools/syntheticdoc/compact.py`, and `backgrounds/` |
| `FLATTENER_DATA` | `data-research-doc3d-uvdoc.json`, `data-release-raw.json` | data root with `research/doc3d` and `research/uvdoc_final`, or with `release/syntheticdoc` (the full-resolution SyntheticDoc download of `tools/syntheticdoc/sdtool.py`) |
| `FLATTENER_RUNS` | `train-*.json` | parent directory of run outputs |
| `FLATTENER_UVDOC_WEIGHTS` | `train-release-uvdoc-init.json` | UVDoc `model/best_model.pkl` (MIT; sha256 `7e90861b…da23`) |
| `FLATTENER_INIT_CHECKPOINT` | `train-release-hr.json`, `train-release-grid89.json`, `train-release-perspective-ft.json` | release-track dewarp checkpoint to fine-tune (its `.pt.json` sidecar next to it) |

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

`train-release-perspective-ft.json` is the short version for an RTX 3060 Ti (8 GB): it
fine-tunes a release checkpoint with the same composites (30%, corner shifts up to 12%,
no quarter turns) on the compact copy through `data-release.json`. 40,000 steps at
batch 32 are 1.28M samples, about 3 h at the 120 samples/s the 3060 Ti trained a
712x488 model at (NCHW, AMP, 2.9 GiB peak at batch 32), plus about 15 min of validation.
The learning rate is 5e-5 with 1,000 warmup steps and cosine decay, the rate of the
1440x1024 fine-tune and, per sample, of the 89x61 one. channels_last stays off: it ran
about 4x slower on that card. `max_seconds` (4.5 h) stops a slower run near the end of
the schedule, and a rerun resumes it. Seed 1 keeps the augmentation draws apart from
those of the seed-0 parent; validation does not depend on the seed, so its EPE compares
with the parent's.

`crumple-release-v*.json` are build specs for `tools/crumple/build.py`: each item
flattens a SyntheticDoc render (or draws an own page), bends it with a procedural
surface (`tools/crumple/render.py`), photographs it through a pinhole camera onto a
background and stores the exact map, in the converted layout. `data-release-crumple*.json`
add a build to the compact copy, and `train-release-crumple*-ft.json` fine-tune a release
checkpoint on it like `train-release-perspective-ft.json`.

| Build | Surfaces | Camera | Paper | 40k-step fine-tune from dewarp-v2 |
|---|---|---|---|---|
| v1 | crumple | tilt up to 32° | stretched on slopes | DIR300 +0.022 MS-SSIM; two real book photos worse |
| v2 | crumple .55, curve .45 | curve: steep and close | stretched | DIR300 0.670, UVDoc bench 0.666; two book photos worse |
| v5 | crease .25, fold .2, curve .15, flat .15, spread .25 | 40% of every mode close | keeps its length, 3-D shape stored | best UVDoc-bench AD and DIR300 LD; DocUNet -0.015; book photos worse |
| v6 | as v5, spread .2, leaf .15 | as v5 | as v5 | no better on book photos; pages cut off by the photo cropped again |

The v1-v2 renderer put each texture point straight above its place on the ground plane,
so folds and curls stretched the paper where real paper keeps its length and
foreshortens text. From v5 the surface is laid flat first (`isometric`), and each sample
also stores the surface's shape in the page's own frame (`grid3d`), which becomes the
model's 3-D target (`loss_grid3d`; `train-release-crumple6-no3d-ft.json` turns it off:
DocUNet 0.563 against 0.558 with it). `spread` maps both pages of an open book (the
scanner splits them at the gutter); `leaf` maps one page whose facing page mostly leaves
the photo.

Every fine-tune on these renders raised the single-sheet benchmarks (DIR300, UVDoc bench)
and made real book photos worse: the facing page taken into the page, pages cut off by
the photo cropped to their visible text, spreads left unsplit. The renders hold single
sheets with all four edges in view, as SyntheticDoc does. `tools/soup.py` blends a
fine-tune with its parent; the 50/50 blend of the v2 fine-tune with dewarp-v2 kept the
real photos close to dewarp-v2 and scored DocUNet 0.576 / 5.33 / 0.234, while the blend
with the v5 fine-tune (whose 3-D head had trained) fell below both parents.

`uvdoc-retexture-release-v*.json` are build specs for `tools/uvdoc/retexture.py`, which
lays clean pages (SyntheticDoc renders flattened through their ground truth, or own
generated pages) on UVDoc's raw photos of deformed blank paper through their texture
coordinate maps (MIT, (c) 2023 Tanguy Magne). The paper's shape and shading come from
real captures; pages, backgrounds and the stored map are ours. UVDoc's own composites
stay off the release track (DTD backgrounds, partly non-commercial pages). The captures
of the UVDoc benchmark are left out. Each spec builds 60,000 training items from 962
captures; `data-release-uvdoc*.json` add the builds to the compact copy. The spec's name
seeds the draws and also the capture split, so v2 trains on 44 of the 46 captures v1
holds out: the UVDoc validation error of the runs that read v2 is not held out. A
further build sets `split_name` to the first build's name.

| Fine-tune | UVDoc share | Start | DocUNet | UVDoc bench | DIR300 |
|---|---|---|---|---|---|
| `train-release-uvdoc-ft.json` | 30% (v1) | dewarp-v2, 40k steps | 0.575 / 5.20 / 0.230 | 0.710 / 5.28 / 0.136 | 0.679 / 4.85 / 0.180 |
| `train-release-uvdoc-half-ft.json` | 50% (v1, v2) | dewarp-v2, 40k steps | 0.575 / 5.29 / 0.231 | 0.725 / 5.08 / 0.114 | 0.684 / 4.70 / 0.176 |
| `train-release-uvdoc-half-ft2.json` | 50% | the row above, 40k more | 0.575 / 5.20 / 0.227 | 0.725 / 5.07 / 0.111 | 0.684 / 4.70 / 0.175 |
| `train-release-uvdoc-70-ft.json` | 70% | the 50% 40k row, 40k more | 0.573 / 5.33 / 0.230 | 0.726 / 4.97 / 0.108 | 0.686 / 4.63 / 0.172 |
| `train-release-uvdoc-clean-ft.json` | 50% (v3, v4: cleaned pages, patterns) | the 50% 40k row, 40k more | 0.582 / 5.16 / 0.227 | 0.734 / 5.07 / 0.109 | 0.691 / 4.48 / 0.166 |
| `train-release-uvdoc-recipe-ft.json` | 50% (v5, v6: UVDoc's recipe) | the 50% 40k row, 40k more | 0.582 / 5.26 / 0.234 | 0.756 / 6.64 / 0.128 | 0.688 / 4.63 / 0.174 |

Scores are MS-SSIM / LD / AD from one prediction. With AADD's tight turned crop and two
more predictions (`--rerun 2 --rect`) the last row scores DocUNet 0.576 / 4.97 / 0.212,
UVDoc bench 0.748 / 4.68 / 0.110 and DIR300 0.700 / 4.23 / 0.143. Unlike the procedural
renders, these fine-tunes kept real book photos close to dewarp-v2; two of 68 got worse
(one spread left unsplit, one close-up whose page runs past the photo returned as the
original). Twice the steps at the same share changed each score by at most 0.004. The
80k-step 50% model was the scanner's dewarp model in `bundle-v5.json`. At 70% the
benchmarks pull apart: with the tight crop DIR300 rises to 0.703 / 4.17 / 0.140 and the
UVDoc bench to 0.752 MS-SSIM, while DocUNet falls to 0.571 / 5.09 / 0.218, UVDoc-bench
LD and AD worsen (4.98, 0.121) and DIR300 OCR CER rises from 0.156 to 0.175. The 50/50
blend of the two (`tools/soup.py`) lands between them (tight crop: DocUNet 0.574 / 4.92 /
0.213, UVDoc bench 0.753 / 4.72 / 0.110, DIR300 0.702 / 4.22 / 0.142; CER 0.180).

A research-only check (never shipped) fine-tuned the same start on UVDoc's own
composites, which lay pages on the same captures with identical labels: one prediction
scored DocUNet 0.584, UVDoc bench 0.793 and DIR300 0.700. The difference was how our
pages looked. They were SyntheticDoc renders flattened through their ground truth, so
they kept the render's creases, shadows and grey cast, which then sat on another
capture's shape as false shading. `clean_texture` divides that light out and
`pattern_background` adds busy patterned surfaces (v3, v4); the fine-tune on them is the
scanner's dewarp model (`bundle-v6.json`; tight crop: DocUNet 0.579 / 4.97 / 0.215,
UVDoc bench 0.754 / 4.85 / 0.116, DIR300 0.706 / 4.12 / 0.139). The UVDoc benchmark
itself is made by UVDoc's `create_final.py` from held-out captures. Copying its recipe
(`"recipe": "uvdoc"`: the paper shown about two pixels past its labelled edge, the
background toned to the paper, 712x488; v5, v6) raised UVDoc-bench MS-SSIM to 0.776
with the tight crop but worsened its LD from 4.85 to 7.22 and lowered DIR300: the model
learns to set the border inside the visible edge, which suits only that benchmark.

`doc3d-retexture-release-v1.json` lays the same clean pages on Doc3D's page geometry
(`"geometry": "doc3d_grid"`; the 45x31 grids of UVDoc's `Doc3D_grid.zip`, MIT, about
100k renders; Doc3D's renders, albedo and lighting are not used): the grid is refined
to 177x121 nodes, the page drawn through it and lit from its 3-D shape. 59,053 of
60,000 items were kept (the rest turn over in view). `train-release-uvdoc-doc3d-ft.json`
gives them half of the UVDoc share (SyntheticDoc .5, UVDoc .25, Doc3D .25, from the
50% 40k model): with the tight crop DocUNet 0.568 / 4.97 / 0.215, UVDoc bench 0.738 /
4.69 / 0.122 and DIR300 0.695 / 4.33 / 0.142, below the 50% row on all three
benchmarks, while DocUNet OCR CER improved (0.142 against 0.169). Doc3D's shapes with
our synthetic light do not replace real captures.

The start record of a fine-tune carries the parent's data spec, so
`python -m flattener.scan.shippable` checks every training step of the chain from the
child's sidecar alone. Fine-tunes trained before that record only the spec's sha256: the
gate then reads the parent's sidecar at the recorded `init_path`, or takes the parent's
sidecar or spec file through `--spec`, and otherwise answers `unverified_data_spec`.

On Windows `cmd.exe`, set the variables with `set FLATTENER_CACHE=E:\cache` and so on.
Command-line options (`--output-dir`, `--data-spec`, `--init-path`, `--num-workers`, ...)
override the file. Keep configs with machine paths outside the repository.
