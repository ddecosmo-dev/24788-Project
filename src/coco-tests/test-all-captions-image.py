#!/usr/bin/env python3
"""Print all COCO captions for a single random validation image."""

import argparse
import json
import random
from pathlib import Path


def load_coco_captions(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_random_image_id(data: dict) -> int:
    image_ids = [image["id"] for image in data.get("images", [])]
    if not image_ids:
        raise ValueError("No images found in COCO file.")
    return random.choice(image_ids)


def get_captions_for_image(data: dict, image_id: int) -> list[dict]:
    return [ann for ann in data.get("annotations", []) if ann.get("image_id") == image_id]


def print_image_captions(image_id: int, captions: list[dict]) -> None:
    print(f"Random image_id: {image_id}")
    print(f"Found {len(captions)} caption(s):\n")
    for i, ann in enumerate(captions, start=1):
        print(f"{i}. ann_id={ann.get('id')}\n   caption={ann.get('caption')}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print all COCO captions for a single random image from captions_val2017.json."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "annotations" / "captions_val2017.json",
        help="Path to the COCO captions_val2017.json file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible image selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    data = load_coco_captions(args.path)
    image_id = choose_random_image_id(data)
    captions = get_captions_for_image(data, image_id)
    print_image_captions(image_id, captions)


if __name__ == "__main__":
    main()
