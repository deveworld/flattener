# Benchmarks

Python evaluation results for DocUNet, DIR300 and UVDoc.

## SOTA results

Scores are **MS-SSIM ↑ / LD ↓ / AD ↓**.
"SOTA" means the best scores among models with released code and weights.
The one exception is UVDoc MS-SSIM, where the UVDoc model scores higher.

| Benchmark | Flattener | Flattener without TTA | Best released model |
|---|---|---|---|
| DocUNet | **0.589 / 4.91 / 0.208** | 0.581 / 4.98 / 0.214 | 0.571 / 5.51 / 0.218 (SyntheticDoc HR) |
| DIR300 | **0.718 / 3.78 / 0.126** | 0.714 / 3.95 / 0.133 | 0.65 / 5.73 / 0.186 (D2Dewarp) |
| UVDoc | 0.767 / **4.53 / 0.088** | 0.753 / 4.80 / 0.109 | **0.785** / 6.81 / 0.119 (UVDoc) |

Flattener runs two tight crop refinement passes and flip test-time augmentation (TTA).
No published method uses flip TTA, so the table also lists scores without it.
DocUNet uses 128 images, without the two views of document 64, as in the UVDoc and SyntheticDoc papers.

Some papers report better scores but have not released code or weights:

| Method | DocUNet | DIR300 | UVDoc |
|---|---|---|---|
| ForCenNet-DOC3D | 0.579 / 4.91 / 0.19 | 0.709 / 4.73 / 0.136 | – |
| AADD | 0.543 / 6.25 / 0.278 | 0.702 / 4.26 / 0.131 | 0.800 / 5.83 / 0.098 |
| TADoc | 0.530 / 7.11 / 0.334 | 0.692 / 4.33 / 0.170 | – |

Flattener with TTA beats all of them on DIR300.
On DocUNet, ForCenNet-DOC3D has a better AD and the same LD.
On UVDoc, AADD has a better MS-SSIM.
AADD also re-crops and predicts again at test time, with one extra pass on DocUNet and UVDoc and two on DIR300.
The main ForCenNet model is left out because it trains on the flat scans of DocUNet and DIR300.

## Reproduce

Run these commands from the repository root with the final checkpoint.
Download `dewarp.pt` from [Hugging Face](https://huggingface.co/DevWorld/flattener/tree/1.1.0) and use its path for `checkpoint.pt`.
Replace the input paths with your dataset locations.

### DocUNet

```bash
uv run python -m flattener.dewarp.predict --weights checkpoint.pt docunet \
    --tta flip --input-dir docunet-crops --output-dir docunet-predictions \
    --force-render --quantize truncate --rerun 2 --rect
uv run python -m tools.benchmark docunet docunet-predictions \
    --convention 128_excl64 --ref-resize uvdoc_view2_double --output docunet.json
```

Replace `docunet-crops` with the DocUNet cropped-photo directory.
Reference scans use the evaluator's default location, or a directory supplied with `--references-dir`.

### DIR300

```bash
uv run python -m tools.benchmark dir300-predict --weights checkpoint.pt --tta flip \
    --inputs-dir dir300-photos --output-dir dir300-predictions \
    --force-render --quantize truncate --rerun 2 --rect
uv run python -m tools.benchmark dir300 dir300-predictions \
    --references-dir dir300-scans --output dir300-results.json
```

### UVDoc

```bash
uv run python -m tools.benchmark uvdoc-predict --weights checkpoint.pt --tta flip \
    --output-dir uvdoc-predictions --root uvdoc-benchmark --rerun 2 --rect
uv run python -m tools.benchmark uvdoc uvdoc-predictions \
    --root uvdoc-benchmark --output uvdoc-results.json
```

The benchmark commands cover DocUNet, DIR300, UVDoc, OCR and evaluator calibration.
LD and AD require the SIFT flow sources (`FLATTENER_SIFTFLOW_SOURCE`).
Reports include model settings, per-image scores and failures.
See `uv run python -m tools.benchmark --help`.

## Renderer performance

Measure CPU rendering time and memory use on a generated image:

```bash
uv run python -m tools.benchmark.renderer --height 1024 --width 1024
```

This tool requires Linux and prints its results as JSON.

## Sources

| Method | Source | Scores taken from |
|---|---|---|
| SyntheticDoc HR | [Woortmann et al., ECCV 2026](https://arxiv.org/abs/2609.15503), [code and weights](https://github.com/tanguymagne/SyntheticDoc) | Our evaluator on the released high-resolution checkpoint; the paper reports 0.570 / 5.51 / 0.219 (Table 5) |
| D2Dewarp | [CVPR 2026](https://arxiv.org/abs/2507.08492), [code and weights](https://github.com/xiaomore/D2Dewarp) | DIR300 table in the paper |
| UVDoc | [Verhoeven et al., SIGGRAPH Asia 2023](https://arxiv.org/abs/2302.02887), [code and weights](https://github.com/tanguymagne/UVDoc) | MS-SSIM and AD from the corrected UVDoc benchmark results in the repository README; LD from AADD Table S1 |
| ForCenNet | [ICCV 2025](https://arxiv.org/abs/2507.19804), [repository](https://github.com/caipeng328/ForCenNet) | Tables 1 and 2, ForCenNet-DOC3D rows |
| AADD | [AAAI 2026](https://arxiv.org/abs/2507.15000), [repository](https://github.com/chaoyunwang/AADD) | DocUNet and DIR300 from Table 1, UVDoc from Table S1 |
| TADoc | [arXiv 2025](https://arxiv.org/abs/2508.06988) | Table 1, average-method rows |

DvD (SIGGRAPH Asia 2025) has released weights and scores 0.549 / 6.61 / 0.279 on DocUNet; its DIR300 scores are in a supplement we did not check.
