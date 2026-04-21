#!/usr/bin/env python3
"""Helper script to print random COCO caption annotations from the validation split."""

import argparse
import json
import random
from pathlib import Path


def load_coco_captions(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_random_captions(data: dict, count: int = 5) -> list[dict]:
    annotations = data.get("annotations", [])
    if not annotations:
        raise ValueError("No annotations found in COCO file.")
    return random.sample(annotations, min(count, len(annotations)))


def print_captions(annotations: list[dict]) -> None:
    for i, ann in enumerate(annotations, start=1):
        image_id = ann.get("image_id")
        caption = ann.get("caption")
        ann_id = ann.get("id")
        print(f"{i}. image_id={image_id} ann_id={ann_id}\n   caption={caption}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print random captions from COCO captions_val2017.json."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "annotations" / "captions_val2017.json",
        help="Path to the COCO captions_val2017.json file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of random captions to print.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    data = load_coco_captions(args.path)
    captions = sample_random_captions(data, args.count)
    print_captions(captions)


if __name__ == "__main__":
    main()
