"""Download the default scanner models and reuse the Hugging Face cache."""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from ..contracts import FlattenerError, Status

MODEL_REPO = "DevWorld/flattener"
MODEL_REVISION = "1.0.0"
MODEL_FILES = ("bundle.json", "dewarp.pt", "gate.pt", "orient.pt")


def download_bundle() -> Path:
    """Return a complete local bundle, downloading it on the first use."""
    settings = {
        "repo_id": MODEL_REPO,
        "revision": MODEL_REVISION,
        "allow_patterns": list(MODEL_FILES),
        "token": False,
    }
    try:
        cached = Path(snapshot_download(**settings, local_files_only=True))
    except LocalEntryNotFoundError:
        cached = None
    if cached is not None and all((cached / name).is_file() for name in MODEL_FILES):
        return cached / "bundle.json"

    try:
        directory = Path(snapshot_download(**settings))
    except Exception as error:
        raise FlattenerError(
            Status.IO_ERROR,
            "model_download_failed",
            f"Could not download {MODEL_REPO} {MODEL_REVISION} from Hugging Face. "
            "Check your connection and retry, or use --bundle PATH with local model files.",
        ) from error
    missing = [name for name in MODEL_FILES if not (directory / name).is_file()]
    if missing:
        raise FlattenerError(
            Status.IO_ERROR,
            "model_download_failed",
            f"The model download is incomplete: {', '.join(missing)}. "
            "Retry online, or use --bundle PATH with local model files.",
        )
    return directory / "bundle.json"
