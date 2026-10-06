# Benchmarks

Python evaluation results for DocUNet, DIR300 and UVDoc.

## SOTA results

Scores are **MS-SSIM ↑ / LD ↓ / AD ↓**.

| Benchmark | Flattener | Best published |
|---|---|---|
| DocUNet | **0.589 / 4.91 / 0.208** | 0.571 / 5.51 / 0.218 |
| DIR300 | **0.718 / 3.78 / 0.126** | 0.702 / 4.26 / 0.131 |
| UVDoc | 0.767 / **4.53 / 0.088** | 0.800 / 5.83 / 0.098 |

Evaluation uses two tight crop refinement passes and flip test-time augmentation (TTA).

## Reproduce

Run these commands from the repository root with the final checkpoint.
Download `dewarp.pt` from [Hugging Face](https://huggingface.co/DevWorld/flattener/tree/1.0.0) and use its path for `checkpoint.pt`.
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
