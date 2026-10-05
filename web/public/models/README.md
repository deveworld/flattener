Place the generated ONNX bundle here with:

```sh
uv run --group web-export python tools/export_web.py --bundle /path/to/bundle.json
```

`manifest.json`, `dewarp.onnx`, `gate.onnx`, and `orient.onnx` are generated locally
and excluded from Git. They must accompany a deployed static build. The exporter
checks release lineage, checkpoint hashes, ONNX validity, and PyTorch parity.

The browser downloads models from this same origin. Photos never go to a server.
