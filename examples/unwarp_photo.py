"""Run a coordinate-study checkpoint on a photo with explicit output dimensions."""

import argparse
from pathlib import Path

from flattener import RenderConfig, Status
from flattener.image_io import load_image, save_result
from flattener.learning.coordinate_study import load_coordinate_predictor
from flattener.learning.inference import unwarp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trusted coordinate-study best.pt with its .sha256 sidecar",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--on-error", choices=("raise", "return", "original"), default="raise")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = RenderConfig(on_error=args.on_error)
    predictor = load_coordinate_predictor(args.checkpoint)
    image = load_image(args.input, config=config)
    result = unwarp(image, predictor, (args.height, args.width), config=config)
    result.metadata["input_path"] = str(args.input)
    result.metadata["model_scope"] = (
        "Experimental synthetic-data model; real-photo quality unverified"
    )
    paths = save_result(result, args.output_dir, overwrite=args.overwrite)
    print(f"{result.status}: {paths}")
    return 0 if result.status == Status.SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
