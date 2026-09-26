# Dewarp training examples

The configs name machine-specific locations with environment variables. The trainer
expands `${VAR}` in `data_spec`, `output_dir`, `init_path` and in data-spec source roots;
an unset variable stops the run with its name. The data-spec hash keeps the unexpanded
roots, so a run resumes on another machine with the same spec.

| Variable | Used by | Points to |
|---|---|---|
| `FLATTENER_CACHE` | `data-release.json` | directory holding `syntheticdoc-release/`, the copy written by `tools/syntheticdoc/compact.py` |
| `FLATTENER_DATA` | `data-research-doc3d-uvdoc.json` | data root with `research/doc3d` and `research/uvdoc_final` |
| `FLATTENER_RUNS` | `train-*.json` | parent directory of run outputs |
| `FLATTENER_UVDOC_WEIGHTS` | `train-release-uvdoc-init.json` | UVDoc `model/best_model.pkl` (MIT; sha256 `7e90861b…da23`) |

Build the compact release copy from a finished (or partial) SyntheticDoc download, then train:

```sh
python tools/syntheticdoc/compact.py --source <download-root>/release/syntheticdoc \
    --output "$FLATTENER_CACHE/syntheticdoc-release" --workers 8
python -m flattener.dewarp.train --config examples/dewarp/train-release-uvdoc-init.json
```

On Windows `cmd.exe`, set the variables with `set FLATTENER_CACHE=E:\cache` and so on.
Command-line options (`--output-dir`, `--data-spec`, `--init-path`, `--num-workers`, ...)
override the file. Keep configs with machine paths outside the repository.
