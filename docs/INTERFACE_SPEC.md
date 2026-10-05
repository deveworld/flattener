# Interface specification

Renderer spec `0.1`. The public API is in `flattener`, page handling in `flattener.regions`, dewarping in `flattener.dewarp`, the scanner in `flattener.scan`, and evaluation in `flattener.benchmark`.

## Input and output

```python
render(image, backward_map, *, target_mask=None, config=None) -> RenderResult
resize_backward_map(coarse_map, output_size, *, config=None) -> dense_map
```

| Item | Contract |
| --- | --- |
| Image | Orientation-corrected NumPy RGB `uint8 (H_src, W_src, 3)` |
| Coordinates | NumPy `float32 (H_out, W_out, 2)`, last axis `(x, y)`, absolute normalized coordinates of the source |
| Target mask | NumPy `bool (H_out, W_out)`; all `True` when omitted |
| Size | `(height, width)`, positive integers, `bool` excluded |
| Success result | RGB `uint8` image and three `bool` masks of the output size |
| Input ownership | The caller's arrays are left unchanged |

File input is single-frame 8-bit RGB or grayscale JPEG/PNG. `load_image` applies the EXIF orientation and copies grayscale into RGB. It rejects files with alpha, a palette, CMYK or multiple frames, and reads pixel values without ICC conversion.

## Coordinates

Output `[i, j]` reads the source position that `backward_map[i, j]` points to. Pixel `(0, 0)` is the centre of the top-left pixel; x grows to the right and y grows downward.

```text
g_x = 2 * (x + 0.5) / W_src - 1
g_y = 2 * (y + 0.5) / H_src - 1
x = (g_x + 1) * W_src / 2 - 0.5
y = (g_y + 1) * H_src / 2 - 0.5
I[i, j] = (2 * (j + 0.5) / W_out - 1, 2 * (i + 0.5) / H_out - 1)
```

`identity_map(output_size)` builds the reference grid I. When the input and output sizes match, it is the identity transform; otherwise it is a half-pixel resize. The normalized bounds `-1, 1` are the outer edges of the source pixels. The conversion functions are `pixel_to_normalized` and `normalized_to_pixel`.

Upsampling a low-resolution map is `I_out + bilinear_resize(coarse_map - I_coarse)`. It uses `align_corners=False` and `antialias=False` and extends the edge values of the residual. It needs a finite `float32` map, a grid of ≥2 on each axis, and an output of ≥ the grid size on each axis. The identity and constant shifts are preserved exactly; general deformations get interpolation error.

The `(Gh, Gw, 2)` grids of `flattener.dewarp` use the same value convention. Nodes are edge-aligned, so the corner nodes are the page corners, and `grid_to_map` turns a grid into the coordinate map above. Coordinate conversions for UVDoc (`align_corners=True`) and SyntheticDoc are in `flattener.dewarp.coords`.

## Sampling and masks

RGB is converted to [0,1] in CPU `float32` and sampled with `grid_sample(mode="bilinear", padding_mode="border", align_corners=False)`. The result is quantized as `floor(clip(value, 0, 1) * 255 + 0.5)`.

- `source_mask`: the coordinate before substitution is finite and lies in [-1,1] on both axes.
- `target_mask`: positions the caller marks as the correction target, where correspondences are defined.
- `valid_mask`: the intersection of the two masks. Other pixels are filled with `fill_rgb`.

NaN or infinity inside the target region is a failure. Non-finite values outside the target are allowed; they are replaced with safe values only in the copy used for sampling. How far source coordinates fall out of range is used for checks and statistics.

## Validation and failure

Checks run in this order: input shape/dtype/size → target finiteness and empty region → out-of-range ratio → local folds and degeneracy → valid pixel count. The local check uses the bilinear patch of each cell whose four corners are all valid in the target. It computes `det(dP/ds, dP/dt) / (4/(W_out*H_out))` at the four corners in `float64` and fails when a value is at or below the lower bound. The check works per cell; the caller checks the result for global overlap and occlusion.

`RenderConfig` defaults are `16_000_000` pixels each for input and output, `8192` per side, a target out-of-range limit of `0.25`, a Jacobian ratio lower bound of `1e-6`, fill colour `(255,255,255)`, and failure policy `"raise"`. When there is no cell to check, `fold_check="not_applicable"`; when processing stops before the check, it is `null`.

`RenderResult` has `status`, `reason_code`, `message`, `image`, the three masks, `used_fallback` and `metadata`. The statuses are `success`, `input_error`, `invalid_transform`, `model_error`, `render_error` and `io_error`.

- `raise`: raises a `FlattenerError` that carries the status, reason and metadata.
- `return`: the image and masks of a failed result are `None`.
- `original`: a copy of the validated, orientation-corrected source with `used_fallback=True`. The failure status is kept and the masks are `None`. If input validation fails, the `return` policy applies.

An invalid failure policy and errors in the helpers that return arrays raise exceptions. The fallback size is the source size. The metadata records the configuration, sizes, model hash, version, check statistics, failing stage and per-stage times.

## Files

`render_file(input_path, backward_map, *, target_mask=None, config=None, output_dir=None, overwrite=False)` chains decoding, rendering and saving. `save_result(result, output_dir, *, overwrite=False)` writes the image and mask PNGs and a JSON file, and returns a dict of paths. Overwriting must be requested explicitly, and each file is replaced atomically. Only one writer may write to a directory.

In region inference metadata, `processing_status` is the final processing status and `quality_status` is `"not_evaluated"`. `success` means the geometry checks and rendering finished.

## Page region API

`PageRegion(left, top, right, bottom)` is a rectangle on source pixel edges. The right and bottom ends are exclusive, and the whole image is `(0,0,width,height)`. A manual region must have a positive size and lie within the source.

`crop_page(image, output_size, *, region=None, segmenter=None, region_config=None, config=None)` makes a rectangular crop, and `unwarp_page(image, predictor, output_size, *, region=None, segmenter=None, region_config=None, config=None)` dewarps a curved page within the region.

Give exactly one of `region` and `segmenter`. The segmenter is a page gate loaded with `flattener.regions.context.load_context_segmenter(checkpoint)`, and it detects printed pages. `estimate_page_region(image, segmenter, config=...)` returns the region estimate. Automatic segmentation is experimental, so check the estimated region in the result.

Both functions sample the source only once. When the estimated mask does not cover the region well enough, `crop_page` widens the region and checks again. The caller sets the output aspect ratio, and the returned masks follow the renderer convention above.

The `predictor` of `unwarp_page` is a `Dewarper` from `flattener.dewarp`. Other values are rejected with `invalid_predictor`. It predicts once on an integer-pixel crop of the region with 4% of the size added on each side, places the grid in source coordinates, and densifies it to `output_size`. For a segmenter mask, the estimated coverage is only checked (`page_coverage_insufficient`), and the region is not widened. For a manual rectangle, the coverage is only recorded (`coverage_enforced: false`).

## Page gate

`load_context_segmenter(checkpoint, *, device="cpu")` in `flattener.regions.context` reads a `region-context-unet-2.0` checkpoint. A `<file>.sha256` file with one line of hex SHA-256 must sit next to it. The schemas are `independent-locator-pilot-1.0` (the `best.pt` of the locator research) and `region-context-1` (weights, config, preprocessing version, provenance record). Preprocessing is `regions.segmentation.preprocess_image`. A loaded model has `predict_probability`, `model_id` and `model_sha256`, so it can be passed as `segmenter=`.

`gate_page_region(probability, source_size, *, config=None)` and `gate_page(image, segmenter, *, config=None)` run `detect_region` with an area limit of 1.0 to set the page box. When the mask covers more than 90% of the frame or touches three or more sides of the border, `fills_frame=True`, `reasons=("page_fills_frame",)`, and the box is the whole image. This case is a success. `page_not_found`, `page_area_rejected`, `ambiguous_page_region` and `uncertain_page_region` are raised as a `FlattenerError` with status `model_error`. `confident` is true when dominance is ≥0.95 and the mean probability is ≥0.85.

The default weights are v1.2 (`page-gate-v1.2/best.pt` in `bundle-v9.json`, `region-context-1`): channels 16/32/64/128, 561,025 parameters, input 192×144. They were trained from scratch on synthetic scenes and frames of SmartDoc capture videos. The two pages of an open book form one region. Documents lying underneath, objects on the desk and walls are not taken as pages. Data, training and results are in [`examples/gate/README.md`](../examples/gate/README.md).

The earlier v0 weights are the `best.pt` of the locator research (140,545 parameters, input 192×144). They were trained from scratch for 8 epochs on 3,840 synthetic scenes of our own, 1,584 warped synthetic images made from SmartDoc source documents, and 528 SmartDoc capture photos.

- `tools/locator_v0.py install` copies `best.pt` from `outputs/models/locator.zip` to `locator-v0-pilot.pt` and writes `page-gate-v0.pt`, which holds the weights and the provenance record (`tools/locator_v0_provenance.json`). Both files get a `.sha256` file. `matches_recorded` in the result tells whether the export matches the recorded sha256.
- The provenance record was extracted from the research data manifest and frozen. Its paths (`outputs/dataset`, `outputs/coordinate-fold/locator.zip`) are the locations at research time. These strings are part of the export's sha256, so they stay unchanged. The same holds for the paths in `locator_v0_smartdoc.json`.
- `reproduce` measures 48 SmartDoc capture validation samples (`gate_v0` in `outputs/benchmarks/smartdoc/manifest.json`, files in `gate-v0/`) the same way as the research (sigmoid, bilinear upsampling to 384×288, threshold 0.5). On CPU the IoU equals the recorded 0.9508340768516064, the per-image loss difference is at most 2×10⁻⁷, and the probability difference from the archived model code is 0.
- `src/flattener/regions/locator_v0_smartdoc.json` lists the 22 SmartDoc documents v0 used for training (`exclude_from_evaluation`), the 4 used for checkpoint selection (`selection_exposed`), and the 4 not used (`unexposed`). When evaluating v0 on SmartDoc, drop all of the first 22 regardless of background and report the 4 selection documents separately. Read the file with `context.smartdoc_exposure()`.
- v0 has never been evaluated on photos without a page.

## Scanner

`ScanResult`, the result of `flattener.scan.pipeline.Scanner`, uses only existing `Status` values. Scanner decisions are `reason_code` strings. For each code, `scan.records.REASONS` holds the status, the severity (`info`, `warning`, `failure`) and the English guidance. `outcome` is `rectified`, `rectified_with_warnings`, `perspective`, `original_returned` or `failed`. Fallback results (perspective correction, returning the original) keep the failure status and reason and have `used_fallback=True`. Renderer check reasons (`fold_or_degenerate` and others) are recorded as they are.

The grid of a scan plan `ScanPlan` holds flattener values of the source. Nodes are edge-aligned and ordered with the page upright. The homography of a perspective plan maps an output position `(u, v, 1)` (u, v ∈ [-1, 1]) to homogeneous coordinates of source flattener values. `--inset` e and `--border` b apply the same way to both plans, so output u shows page position (1 − e + b)·u. In both cases the dense map samples the source once. `plan_sha256` is the sha256 of the plan's canonical JSON; a preview and a saved result with the same value use the same geometry.

The stages are read (EXIF once, draft decode for JPEGs over 16MP) → detect (page gate, plus an agreement rule with a single whole-photo inference, run only when needed) → orient (OrientNet) → infer (first inference, one crop re-inference and deskew; one more when the crop re-inference moves the page by more than 4%) → verify (orientation check on the flat preview, comparison with the gate region, paper and text-line checks on a small flat preview, node movement between passes ≤ 4% of the page diagonal) → plan (output size auto) → render (one `flattener.render` call, sharpening) → save.

- The page region is the outline the dewarping model finds; the page gate serves as a reference. The gate sets the crop of the first inference on large photos, filters out photos without a page, and helps choose among candidates. When the gate region is larger than the flattened page (a neighbouring page, the side of the book, paper underneath), the result is returned as is with a `page_region_larger` warning. In that case the flattened result must look like paper. On a preview with a 384-pixel long side, the median saturation must be ≤0.28 and dark pixels (brightness below 0.35) must be ≤35%. Otherwise the photo is treated as having no page (a laptop keyboard, a wooden desk), and the original is returned with `page_not_found`. Also, when blank paper of the page colour runs along more than half of one side of the first crop box (a side that is not a photo edge), the crop may have cut the page (the gate missed a white margin). Then, if the outline from a single whole-photo inference contains ≥0.95 of the gate region, the area it adds to the flattened page is at most the page area, and at least half of that added area is page colour, this candidate fails (`unstable_geometry`, `page_continues`) and the whole-photo outline candidate goes next. The page colour is the median of text-free pixels inside the outline, and colours within 0.08 on every channel count as the same. When the gate region extends past all four sides of the page by more than 5% of the page area, the result is treated as a flattened part of the page and fails. Against a manual rectangle, a coverage below 0.95 fails. When the gate splits the region into several pieces, the outline from a single whole-photo inference is used as the page, with an `ambiguous_page_region` or `two_page_candidate` warning. If the outline contains none of the pieces, or the pieces it contains cover less than half of the outline area, the original is returned.
- When the flattened result is one piece of the page blown up, the photo is treated as a close-up whose page is cut off by the photo edges, and the original is returned (`page_cut_off`, "The page runs out of the photo. Step back and take it again with the whole page in view"). On a preview with a 384-pixel long side, the result counts as a piece when text lines cross the page and number at least 8 and fewer than 20 (a spacing longer than 1/8 of the page is read as a layout such as columns or tables), the line spacing is regular (autocorrelation ≥0.4), and text runs into more than half of the edge band on two or more sides. The same reason applies when the gate saw the page filling the frame (`page_fills_frame`) and the flattened grid contains less than 45% of the gate region, and when more than 5% of the output of the plan about to be accepted lies outside the photo. However, when at most 25% lies outside the photo and the orientation check agrees twice (confident before flattening, and confident that the flattened preview is upright), the visible part is flattened and returned with a `page_cut_off` warning (evidence `outside`, the cut sides) and reshoot guidance. The check runs twice because on cut-off close-ups a single orientation call has been wrong even at 0.99 confidence. When the original is returned, candidate attempts stop, since the other candidates see the same photo, and no perspective correction is done even with `--fallback perspective`. Manual regions and manual rectangles skip this check, so a user can set the region by hand to get a flattened result.
- When the gate region runs along more than 15% of a photo side, the page-finding stage marks that side as a `page_cut_off` warning candidate. The gate region also includes the facing page cut off by the photo edge, so the warning is kept only when the accepted plan's grid comes within 1% of the side length of that side (evidence `mask_and_page`, distance per side `gap`). When the page itself is whole, there is no warning and no reshoot guidance. A photo split into two pages is judged again per page with each page's own grid. On KR-Real, records with this warning fell from 44 to 8 of 74, and the remaining 8 are photos where the page really runs out of the photo (dev: distance ≤ −0.05 on cut sides, ≥0.04 on whole pages).
- Candidates are tried in this order: the default path, no deskew, the region found by the other locator, and a single whole-photo pass (`rectify`). The first one that passes is used. Records name them `default`, `no_deskew`, `other_box` and `whole_photo` in the same order. The last candidate also re-indexes the grid nodes by orientation k to set the page upright. When the default path fails and a later candidate passes, a `candidate_fallback` warning is recorded. When only the crop re-inference fails, the first inference result is used (`rerun_discarded`). Perspective-corrected results from `--fallback perspective` carry `curvature_not_corrected`.
- Two-page split (`ScanOptions.pages`, default `auto`): the flat preview of the accepted plan is read as two open pages (`look.fold`) when it is wide (width/height ≥1.0), the middle band (35–65% of the width) has a vertical line with no text, and both sides have at least 2.5 times more text than that line. When a picture runs into the fold and there is no text-free line, the grid outline of the wide plan is used. If the top edge bends down or the bottom edge bends up by 8 degrees or more at a column in the middle band, measured between the two columns on either side, that column is taken as the fold (`_outline_fold`; `fold_found.evidence` in the record is `outline`). On KR-Real dev, open books measured 15.5 degrees and single sheets 3.4 degrees or less. The grid is split in two at the fold, and each page is cropped on its own and run through the model once more. If the new grid stays within 4% of the page diagonal of the split grid (median), the new grid is used; if it moves further (it caught the neighbouring page as well), the split grid is used. Each page is then sized, checked and recorded on its own (reason `two_pages`, `metadata.page` in the record). If only one page fails the checks, only that page gets a fallback result. `pages=2` splits at the middle when no fold is found, and `pages=1` never splits. Perspective plans and `uvdoc` node grids are not split.
- Manual corrections (`--region`, `--quad`, `--quarter`, `--no-dewarp`, `--long-side`, `--output-size`, `--aspect`) go through the same stages as the automatic path and are recorded as `manual`. `--no-dewarp` also asks OrientNet when no orientation is given. The quadrilateral comes from `--quad`, then `--region`, then the page gate's region; without a gate, the four corners of the outline from a single whole-photo inference are used. `--quad` points are given clockwise from the top left. Counter-clockwise order would mean a flipped page, so it is an input error. A perspective-corrected result requested with `--no-dewarp` counts as scanned. Model outputs are cached per input sha256, so only the stages after a change are recomputed.
- The decision thresholds (`scan.locate.THRESHOLDS`, `scan.pipeline.VERIFY`) are design starting values and are written to the scan record. They still need to be set on the SmartDoc train split and synthetic val. The values of the paper, text-line and crop checks, and the 0.5 for gate pieces, were set on KR-Real dev, SmartDoc train, and 76 photos without a page (background crops and texture photos).
- `--timing F.jsonl` writes one line per photo with per-stage times (including decode, `infer_pass_*` per pass, `render_map`/`render_check`/`render_sample`/`render_sharpen`, `preview`), `end_to_end` up to saving, and cache hits. The first line holds the model load time. Use `--no-cache` when timing the same photo repeatedly.
- By default the renderer's map-check verdict is computed from the grid, and `flattener.render` runs only on a small proxy (`ScanOptions.fast_check`, default `True`). The saved image is the same. `--full-check` (`fast_check=False`) checks at full resolution. On 20 KR-Real 12MP photos, the median end-to-end time fell from 1.52 s to 1.26 s (p95 from 2.51 s to 1.75 s), and all saved images were identical (laptop CPU, 2 threads). The method is in `scan/fast_check.py`, and the verdict comparison is in `tools/fast_check_agreement.py`.
- Flattened images get one sharpening pass by default (`ScanOptions.sharpen`, `--sharpen`, default 1.0, 0 turns it off, maximum 4). The difference between luma (Y) and its Gaussian blur with a 1.5-pixel standard deviation is multiplied by the strength and added equally to R, G and B, so colours stay the same. Where the difference is 2 levels or less (paper grain, JPEG noise), nothing is added (`scan/sharpen.py`). This restores stroke edges that soften slightly when the flattening resampling maps about one source pixel to one output pixel. It is not applied to fill pixels outside the photo, to the original returned on failure, or to previews. The strength is part of the plan (`ScanPlan.sharpen`), so a saved plan renders the same image under other options. The JSON of a plan with strength 0 omits this key, so plans saved before sharpening existed keep their `plan_sha256`.
- `rectify_batch(scanner.dewarper, inputs, out, **scanner.batch_kwargs(options))` builds a benchmark prediction folder with the scanner. `predictions.json` records the pipeline and the scanner's quantization, inset and pass settings. A folder holding predictions from another pipeline, and arguments that differ from the scanner settings, are rejected.

## Dewarp prediction

Predictions from `python -m flattener.dewarp.predict --weights CKPT {image,docunet}` go through the `flattener.render` checks (coordinate range, coverage, folds). With `--tta flip`, each model call predicts the input and its left-right mirror in one batch, flips the mirrored grid back (reverses the column order and negates x), and averages the two grids. When the median node distance between the two grids exceeds 3% of the image side (`TTA_MAX_GAP`), only the grid of the original input is used. The prediction record's evidence includes `tta`. The scanner turns it on with the `"tta"` key of the bundle's `dewarp` entry. Photos that fail the checks are not saved. Their status (`invalid_transform` and the reason) goes to the output and to `predictions.json`, and the command exits with code 1. `--force-render` saves failed photos too. Use it to score benchmarks the same way as published methods. The failure records and exit code 1 remain, and `prediction_failures` in the report counts the failed photos. `--weights` also accepts the public UVDoc weights `best_model.pkl`.

- `--quantize truncate` saves sampled values as `floor(x*255)` (default `round`). The UVDoc and SyntheticDoc paper numbers are results saved this way and scored the way the UVDoc script does (`--ref-resize uvdoc_view2_double`). Switching between rounding and truncation alone changes LD by about 0.3.
- `--inset` moves the predicted page outline inward before rendering. Output coordinate u (−1…1) shows position (1−e)·u of the predicted grid, so each side moves in by e/2 of the page size. The value is a preset name, `E`, or `EX,EY`. The preset `syntheticdoc_half_cell` (x 1/177, y 1/121) is the half-cell shift that SyntheticDoc inference introduces when it upsamples a 121×177 map with `align_corners=True`.
- `--rerun R` predicts once on the whole photo, then crops and predicts R more times. Each time it takes the axis-aligned box of the current map's page outline, adds 4% on each side, crops at integer pixels, and composes the new grid with the crop position to bring it back to source coordinates. `--deskew` estimates the page angle from the direction of the top and bottom edges and, when it is 45° or less, rotates the page upright before predicting. Skip reasons (`degenerate_outline`, `inconsistent_edges`, `angle_above_limit`), boxes and angles are recorded in `crops`.
- Quantization, inset and rerun settings are recorded in `predictions.json` and `evidence`, and predictions made with other settings are not reused. `uvdoc-predict`, `dir300-predict`, `docunet-stress` and `tools/score_docunet_local.py` take the same options.
- Training validation records EPE separately for each view. The views are a crop fitted tightly to the page (`page`), a crop with a margin as in DocUNet (`docunet`), the whole photo (`full_frame`), crops rotated by ±90° (`rot90`, `rot270`), and a small page that fills only 10–25% of a patterned canvas (`small_page`). The best checkpoint is chosen on the one view set by `val_select` (default `page`). Samples without valid ground-truth points count as `not_evaluable`.

## Evaluation

`flattener.benchmark` computes DocUNet MS-SSIM, LD and AD the same way as the public MATLAB scripts, and OCR ED and CER the same way as the pytesseract calls.

- DocUNet inputs are the cropped photos `outputs/benchmarks/docunet/crop/`, the scans `reference/docunet-reference-{k}.png`, and the original photos `photo/` (`full_frame` in the stress evaluation). Per-file hashes and sources are in `manifest.json` in the same folder.
- `--convention` chooses among `130_raw` (PaperEdge, DocTr family), `128_excl64` (UVDoc, 128 images without document 64) and `130_rot64`; the default is all three. `--ref-resize uvdoc_view2_double` reproduces how the UVDoc script resizes the scan twice for view 2. To compare with papers, use 8-bit truncation (`--quantize truncate`), `128_excl64` and `uvdoc_view2_double`.
- AD is reported as `lsqr_4iter`, which follows the MATLAB `lsqr` defaults. The exact least-squares value `exact_lstsq` is stored as well.
- LD and AD need the SIFT flow C++ source (Ce Liu). Point `--siftflow-source` or `FLATTENER_SIFTFLOW_SOURCE` at the `SIFTflow` folder from DocUNet `eval.zip` or UVDoc `eval_code`, and the first run builds it with a C++ compiler and ninja (`build-siftflow`).
- There are four evaluator calibration checks. They test whether PaperEdge result images reproduce the MATLAB values PaperEdge published (`paperedge_matlab`), whether the same images reproduce the PaperEdge row of the UVDoc paper table (`uvdoc_table_paperedge`), whether the public UVDoc results reproduce the UVDoc row of the same table (`uvdoc_table_uvdoc`), and whether Tesseract 5.0.1 reproduces the DocScanner OCR table (`ocr_table`). The results are in `outputs/benchmarks/docunet/calibration.json`.
- `comparable` in the OCR report is `true` only when the text was read with a Tesseract build that passed the OCR score reproduction check and the scan hashes match. The build list is `flattener.benchmark.ocr.CALIBRATED_TESSERACT_BUILDS`. It currently holds one build, the UB-Mannheim Windows build `v5.0.1.20220118` (Leptonica 1.78.0, tessdata_fast `eng`). On Linux, unpack the installer with `7z x` and run it under Wine (`--tesseract TREE/tesseract.exe --launcher wine`, with `WINEPREFIX` and `WINEDEBUG=-all` as environment variables). conda-forge `tesseract=5.0.1` (Leptonica 1.82.0) does not pass the reproduction check.

The stress evaluation (`docunet-stress --weights CKPT --output-dir OUT --force-render`) alters the DocUNet photos and scores them with the same evaluator. The variants are `crop` (baseline), `full_frame` (the original photo with its background, EXIF orientation applied), `rot90` and `rot270` (the crop rotated by ±90°), and `small_page` (10–25% of the area of a patterned canvas). The output is rendered at the size of the upright crop. Scans are scored upright, so a model that returns a rotated page as it is scores low on `rot90` and `rot270`. Results go to `OUT/stress.json` and to `OUT/<variant>/docunet.json` for each variant. The baseline for the public UVDoc weights is in `outputs/benchmarks/docunet/stress/uvdoc-mit/`.

The UVDoc benchmark (50 real photos) follows the conventions of the public UVDoc code (`uvdocBenchmark_pred.py`, `evalScriptUVDoc.m`, MIT).

- `uvdoc-prepare UVDoc_benchmark.zip` checks the SHA-256 of the archive (https://igl.ethz.ch/projects/uvdoc/UVDoc_benchmark.zip), unpacks it into `outputs/benchmarks/uvdoc-bench/`, and writes `manifest.json`.
- Training stops with a `benchmark_data` error when a dewarp training data folder is inside `outputs/benchmarks/`, or when that folder or a parent folder holds a benchmark `manifest.json`.
- `uvdoc-predict` predicts a grid from the photo `img/{id}.png`, applies the grid to the warped texture `warped_textures/{id}.png`, and writes the result to `OUT/uwp_texture/`. Bilinear interpolation, 0 outside the source, and saving with `floor(x*255)` all match the public code.
- `uvdoc` compares `uwp_texture` with the flat texture `texture_sample/{id}.png` using the same MATLAB emulation as DocUNet, and averages over the 50 images. The reference is the 50-image values in the UVDoc README erratum (MS-SSIM 0.785, AD 0.119).
- Evaluator check: the public UVDoc result images give MS-SSIM 0.7854, AD 0.1186 and LD 6.812, which match the erratum values and the LD 6.810 in AADD table S1.

DIR300 (300 real photos, 90 for OCR) is our own implementation of DocGeoNet's public protocol (`ssim_ld_eval_DIR300.m`, `OCR_eval_DIR300.py`).

- The data is `dist300/` (input photos) and `gt300/` (reference images) in `outputs/benchmarks/dir300/`, for non-commercial evaluation only. Both folders must match the official files by hash.
- `dir300-predict` corrects each photo at its original size. Because of the largest photo (3472×4624), the renderer limit is set to 20 million pixels.
- `dir300` resizes the reference image to 598,400 pixels and the corrected result to the same size, measures MS-SSIM and LD (DocUNet `evalUnwarp.m`) and AD (PaperEdge `evalAlignedUnwarp.m`), and takes the plain mean over the 300 pairs. `dir300-ocr` reads the 90 images in the DocGeoNet list with pytesseract defaults.
- DIR300 photos include background, so models that flatten the whole photo in one pass score much lower than methods that find the page first. Paper tables mix both kinds.
- Evaluator check: predicting with the UVDoc MIT weights and `--quantize truncate` gives 0.6215 / 7.742 / 0.2183, which matches 0.622 / 7.73 / 0.218 in the TADoc table.

`tools/score_docunet_local.py CKPT RUN` and `tools/score_dir300_local.py CKPT RUN` predict and score in one step and write reports to `$FLATTENER_CACHE_DIR/scores/RUN/` (default `~/.cache/flattener/scores/RUN/`), with one file name per setting (for example `docunet-uvdoc_view2_double-truncate.json`). Per-model scores are collected in `outputs/models/scores.json`.

## Result evidence

`predictions.json`, training logs, `status.json`, `best.json` and `flattener.benchmark` reports contain an `evidence` entry. `evidence_kind` tells what kind of result it is.

| `evidence_kind` | Meaning |
| --- | --- |
| `new_model_real_photo` | Results of correcting and scoring DocUNet, UVDoc benchmark and DIR300 real photos with a checkpoint we trained (10,000 steps or more). This is the only kind that counts as real-photo performance of a new model |
| `gt_crop_validation`, `gt_full_frame_validation` | Validation during training. Samples with ground truth are cropped to the ground-truth page region or fed as the full frame |
| `external_pretrained` | Results of models trained elsewhere, such as the public UVDoc weights or results the authors published. Used as the comparison baseline |
| `smoke_training` | Results of checkpoints trained for fewer than 10,000 steps |
| `docunet_standard`, `docunet_stress`, `uvdoc_benchmark`, `dir300_standard` | Results where the prediction folder has no `predictions.json`, so the model source is unknown, or results of the first research model (`own_previous_model`) |
| `evaluator_calibration`, `predictions`, `training_loss` | Evaluator calibration check reports, prediction records without scores, training loss |

`measurement` (what was measured), `model_provenance` (`own_checkpoint`, `own_previous_model`, `external_pretrained`), `checkpoint_step` and `initialization` (`scratch`, or `external_pretrained` for UVDoc weights) are recorded as well. Fine-tuning that starts from UVDoc weights also counts as `new_model_real_photo`; read its gain against the UVDoc baseline of the same variant.
