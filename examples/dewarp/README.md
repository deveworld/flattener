# Dewarp training examples

The configs name machine-specific locations with environment variables. The trainer
expands `${VAR}` in `data_spec`, `output_dir`, `init_path` and in data-spec source roots;
an unset variable stops the run with its name. The data-spec hash keeps the unexpanded
roots, so a run resumes on another machine with the same spec.

| Variable | Used by | Points to |
|---|---|---|
| `FLATTENER_CACHE` | `data-release.json` | directory holding `syntheticdoc-release/`, the copy written by `tools/syntheticdoc/compact.py` |
| `FLATTENER_DATA` | `data-research-doc3d-uvdoc.json`, `data-release-raw.json` | data root with `research/doc3d` and `research/uvdoc_final`, or with `release/syntheticdoc` (the full-resolution SyntheticDoc download of `tools/syntheticdoc/sdtool.py`) |
| `FLATTENER_RUNS` | `train-*.json` | parent directory of run outputs |
| `FLATTENER_UVDOC_WEIGHTS` | `train-release-uvdoc-init.json` | UVDoc `model/best_model.pkl` (MIT; sha256 `7e90861b…da23`) |
| `FLATTENER_INIT_CHECKPOINT` | `train-release-hr.json` | release-track dewarp checkpoint to fine-tune (its `.pt.json` sidecar next to it) |

Build the compact release copy from a finished (or partial) SyntheticDoc download, then train:

```sh
python tools/syntheticdoc/compact.py --source <download-root>/release/syntheticdoc \
    --output "$FLATTENER_CACHE/syntheticdoc-release" --workers 8
python -m flattener.dewarp.train --config examples/dewarp/train-release-uvdoc-init.json
```

`train-release-hr.json` fine-tunes a release checkpoint at 1440x1024 input with the
backbone's own 90x64 grid. It reads the full-resolution download through
`data-release-raw.json`, since the compact copy (0.75 scale, 1080x768) is smaller than
that input. The raw val split selects the same 2,000 samples as the compact one. Weights
trained at 712x488 do not carry over to another input size without training: on 400
val samples a 712x488 checkpoint scored 2.4 px docunet EPE at its own size, 73 px at
1024x704 and 146 px at 1440x1024. channels_last trained ~1.55x faster than NCHW on an
RTX 5090 (199 against 129 samples/s at batch 32, 12 GiB).

On Windows `cmd.exe`, set the variables with `set FLATTENER_CACHE=E:\cache` and so on.
Command-line options (`--output-dir`, `--data-spec`, `--init-path`, `--num-workers`, ...)
override the file. Keep configs with machine paths outside the repository.
