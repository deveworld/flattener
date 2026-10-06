# Browser models

Generate the ONNX bundle from the repository root:

```sh
uv run --group web-export python tools/models/export_onnx.py
```

The exporter downloads the default models from [Hugging Face](https://huggingface.co/DevWorld/flattener) on first use.
Use `--bundle /path/to/bundle.json` to export local models.
The build includes `manifest.json`, `dewarp.onnx`, `gate.onnx` and `orient.onnx` in the deployed site.
These files are generated locally and excluded from Git.
The browser loads them from the site's host and processes photos locally.
