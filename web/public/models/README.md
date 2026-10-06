# Browser models

Generate the ONNX bundle from the repository root:

```sh
uv run python tools/models/export_onnx.py
```

The exporter downloads the default models from [Hugging Face](https://huggingface.co/DevWorld/flattener) on first use.
Use `--bundle /path/to/bundle.json` to export local models.
The exported default models are also on Hugging Face at revision `1.1.0`; download `manifest.json`, `dewarp.onnx`, `gate.onnx` and `orient.onnx` into this directory to skip the export.
The build includes `manifest.json`, `dewarp.onnx`, `gate.onnx` and `orient.onnx` in the deployed site.
These files are generated locally and excluded from Git.
The browser loads them from the site's host and processes photos locally.
