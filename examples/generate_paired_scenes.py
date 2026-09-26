"""Generate a small paired printed/blank scene dataset in a new directory."""

import argparse
import json
from pathlib import Path

from flattener.synthetic.paired_scenes import PairedSceneConfig, generate_paired_scene_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--train-documents", type=int, default=96)
    parser.add_argument("--validation-documents", type=int, default=24)
    parser.add_argument("--test-documents", type=int, default=24)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=144)
    parser.add_argument("--training-panels", choices=("matched", "independent"), default="matched")
    args = parser.parse_args()
    config = PairedSceneConfig(
        seed=args.seed,
        train_documents=args.train_documents,
        validation_documents=args.validation_documents,
        test_documents=args.test_documents,
        image_size=(args.height, args.width),
        training_panels=args.training_panels,
    )
    manifest = generate_paired_scene_dataset(args.output, config)
    print(
        json.dumps(
            {
                "directory": str(args.output),
                "counts": manifest["counts"],
                "preprocessing_qa": manifest["preprocessing_qa"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
