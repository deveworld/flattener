# Flattener

**Open-source SOTA document scanner.**

**Demo:** https://flattener.worldsw.dev/ (runs in your browser; photos are not uploaded)

Turn document photos and open books into flat, upright scans.
Flattener detects pages, corrects orientation and curvature, and splits book spreads.
Everything runs locally.
Inference, training, data preparation and evaluation code are included.

See [benchmark results and reproduction instructions](https://github.com/deveworld/flattener/blob/main/docs/BENCHMARKS.md).

## Scan

```bash
uvx flattener-scan -i photo.jpg scan.jpg
```

Or install it with `pip install flattener-scan` and run `flattener-scan -i photo.jpg scan.jpg`.
Flattener needs Python 3.12 or newer and is tested on Linux x86_64.

The models run on the CPU with ONNX Runtime, so the install stays small.
To use an NVIDIA GPU on Linux, add PyTorch: `uvx --with torch flattener-scan -i photo.jpg scan.jpg`, or `pip install "flattener-scan[torch]"`.
With PyTorch installed, a CUDA GPU is used automatically; `--device cpu` or `--backend onnx` keeps the CPU.

The first scan downloads the Apache-2.0 models from [Hugging Face](https://huggingface.co/DevWorld/flattener), version `1.1.0`.
Later scans reuse the local Hugging Face cache and work offline.
Set `HF_HOME` to choose a different cache directory.

Without an output name, the scan is saved as `photo-scan.jpg` in the current directory.
Repeat `-i` for a batch and give an output directory: `flattener-scan -i a.jpg -i b.jpg scans/`.
Book spreads produce separate page files, such as `scan-p1.jpg` and `scan-p2.jpg`.
Existing files are kept unless you pass `-y`.
Use `--record` to save a JSON record next to each scan, or `--json` to print a one-line summary per page.
Use `--region`, `--quad`, `--quarter` or `--output-size` for manual control; run `flattener-scan --help` for all options.

From Python:

```python
from flattener.scan.io import save_scan_files
from flattener.scan.pipeline import Scanner

scanner = Scanner.load()
result = scanner.scan("photo.jpg")
save_scan_files(result, "scan.jpg")
```

Use `scanner.scan_pages("book.jpg")` for automatic spread splitting.

To work from a clone of this repository, run `uv sync` and `uv run flattener-scan`.
The development environment includes CPU PyTorch for training and evaluation.

## Browser app

Scan single pages locally with WebGPU or CPU/WASM, edit the result and save a PNG.
Try it at https://flattener.worldsw.dev/ or https://deveworld.github.io/flattener/.
See the [web app guide](https://github.com/deveworld/flattener/blob/main/web/README.md) for setup, capabilities and static hosting.

## Training

All three scanner models are trained from scratch on redistributable data.

- [Dewarp](https://github.com/deveworld/flattener/blob/main/docs/DEWARP.md)
- [Page detection](https://github.com/deveworld/flattener/blob/main/docs/PAGE_GATE.md)
- [Orientation](https://github.com/deveworld/flattener/blob/main/docs/ORIENT.md)

Run the Python tests with `uv run pytest -q`.

## License

Project code is [Apache-2.0](https://github.com/deveworld/flattener/blob/main/LICENSE).
