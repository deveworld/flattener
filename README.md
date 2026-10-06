# Flattener

**Open-source SOTA document scanner.**

Turn document photos and open books into flat, upright scans.
Flattener detects pages, corrects orientation and curvature, and splits book spreads.
Everything runs locally.
Inference, training, data preparation and evaluation code are included.

See [benchmark results and reproduction instructions](https://github.com/deveworld/flattener/blob/main/docs/BENCHMARKS.md).

## Scan

Flattener needs Python 3.12 and is tested on Linux x86_64 with CPU PyTorch.

```bash
pip install flattener-scan --extra-index-url https://download.pytorch.org/whl/cpu
flattener photo.jpg --out scans
```

The extra index installs the CPU build of PyTorch.
Without it, pip on Linux installs the CUDA build, which is several gigabytes larger.

To work from a clone of this repository, use `uv sync --python 3.12` and run `uv run flattener`.

The first scan downloads the Apache-2.0 models from [Hugging Face](https://huggingface.co/DevWorld/flattener), version `1.0.0`.
Later scans reuse the local Hugging Face cache and work offline.
Set `HF_HOME` to choose a different cache directory.

Pass multiple photos for a batch.
Results include an image and a `.scan.json` record with outcomes and timings.
Book spreads produce separate page files.
Use `--region`, `--quad`, `--quarter` or `--output-size` for manual control; run `flattener --help` for all options.

From Python:

```python
from flattener.scan.io import save_scan
from flattener.scan.pipeline import Scanner

scanner = Scanner.load()
result = scanner.scan("photo.jpg")
save_scan(result, "scans", "photo")
```

Use `scanner.scan_pages("book.jpg")` for automatic spread splitting.

## Browser app

Scan single pages locally with WebGPU or CPU/WASM, edit the result and save a PNG.
See the [web app guide](https://github.com/deveworld/flattener/blob/main/web/README.md) for setup, capabilities and static hosting.

## Training

All three scanner models are trained from scratch on redistributable data.

- [Dewarp](https://github.com/deveworld/flattener/blob/main/docs/DEWARP.md)
- [Page detection](https://github.com/deveworld/flattener/blob/main/docs/PAGE_GATE.md)
- [Orientation](https://github.com/deveworld/flattener/blob/main/docs/ORIENT.md)

Run the Python tests with `uv run pytest -q`.

## License

Project code is [Apache-2.0](https://github.com/deveworld/flattener/blob/main/LICENSE).
