# Tools

Run these commands from the repository root.

| Task | Command |
|---|---|
| Export browser models | `uv run --group web-export python tools/models/export_onnx.py` |
| Blend trained checkpoints | `uv run python tools/models/blend.py dewarp.pt first.pt second.pt --weight 0.5` |
| Evaluate model predictions | `uv run python -m tools.benchmark --help` |
| Measure renderer performance | `uv run python -m tools.benchmark.renderer` |
| Prepare dewarp data | `tools/data/syntheticdoc/compact.py`, `tools/data/uvdoc/retexture.py`, `tools/data/creases/build.py` |
| Prepare page detection data | `tools/data/pages/build_data.py` |
| Prepare orientation data | `tools/data/orientation/build_data.py` |
| Download backgrounds | `tools/data/backgrounds/fetch_polyhaven.py` |
| Generate the browser sample | `uv run python tools/web/generate_sample.py` |
| Update browser test fixtures | `uv run python tools/web/generate_fixtures.py` |

See [training documentation](../README.md#training) for the model recipes and [benchmarks](../docs/BENCHMARKS.md) for evaluation.
