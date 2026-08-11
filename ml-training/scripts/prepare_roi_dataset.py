#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from lipseg.config import LipROIConfig, load_config
from lipseg.dataset_validation import iter_samples, resolve_inside, validate_source
from lipseg.license_manifest import DatasetSource, load_sources
from lipseg.splits import subject_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fixed-size lip ROI samples.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=TRAINING_ROOT / "configs" / "lip_roi.json")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def expanded_box(mask: np.ndarray, config: LipROIConfig) -> tuple[int, int, int, int]:
    foreground_y, foreground_x = np.where(mask > 0)
    if foreground_x.size == 0 or foreground_y.size == 0:
        raise ValueError("mask has no foreground")

    x0 = float(foreground_x.min())
    x1 = float(foreground_x.max() + 1)
    y0 = float(foreground_y.min())
    y1 = float(foreground_y.max() + 1)
    width = x1 - x0
    height = y1 - y0
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    width *= 1.0 + 2.0 * config.margin_x
    height *= 1.0 + 2.0 * config.margin_y

    target_aspect = config.width / config.height
    if width / height < target_aspect:
        width = height * target_aspect
    else:
        height = width / target_aspect
    return (
        int(np.floor(center_x - width * 0.5)),
        int(np.floor(center_y - height * 0.5)),
        int(np.ceil(center_x + width * 0.5)),
        int(np.ceil(center_y + height * 0.5)),
    )


def crop_with_padding(image: Image.Image, box: tuple[int, int, int, int], fill: int | tuple[int, ...]) -> Image.Image:
    x0, y0, x1, y1 = box
    output = Image.new(image.mode, (x1 - x0, y1 - y0), fill)
    source_box = (
        max(0, x0),
        max(0, y0),
        min(image.width, x1),
        min(image.height, y1),
    )
    if source_box[0] >= source_box[2] or source_box[1] >= source_box[3]:
        return output
    source = image.crop(source_box)
    output.paste(source, (source_box[0] - x0, source_box[1] - y0))
    return output


def prepared_name(source: DatasetSource, line_number: int, sample: dict) -> str:
    identity = f"{source.name}:{line_number}:{sample['image']}:{sample['mask']}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{source.name}-{digest}"


def prepare_sample(
    source: DatasetSource,
    line_number: int,
    sample: dict,
    config: LipROIConfig,
    output: Path,
) -> dict:
    image_path = resolve_inside(source.root, sample["image"])
    mask_path = resolve_inside(source.root, sample["mask"])
    with Image.open(image_path) as opened_image, Image.open(mask_path) as opened_mask:
        image = opened_image.convert("RGB")
        mask = opened_mask.convert("L")
        box = expanded_box(np.asarray(mask, dtype=np.uint8), config)
        roi_image = crop_with_padding(image, box, (0, 0, 0)).resize(
            (config.width, config.height),
            Image.Resampling.BILINEAR,
        )
        roi_mask = crop_with_padding(mask, box, 0).resize(
            (config.width, config.height),
            Image.Resampling.NEAREST,
        )

    split = subject_split(str(sample["subject_id"]), config.split_seed, config.splits)
    name = prepared_name(source, line_number, sample)
    image_relative = Path("images") / f"{name}.jpg"
    mask_relative = Path("masks") / f"{name}.png"
    (output / split / image_relative).parent.mkdir(parents=True, exist_ok=True)
    (output / split / mask_relative).parent.mkdir(parents=True, exist_ok=True)
    roi_image.save(output / split / image_relative, quality=95)
    roi_mask.save(output / split / mask_relative)
    return {
        "image": image_relative.as_posix(),
        "mask": mask_relative.as_posix(),
        "subject_id": str(sample["subject_id"]),
        "consent_record_id": str(sample["consent_record_id"]),
        "source": source.name,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sources = load_sources(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    indexes = {}
    prepared_count = 0
    try:
        for source in sources:
            validate_source(source)
            for line_number, sample in iter_samples(source):
                record = prepare_sample(source, line_number, sample, config, args.output)
                split = subject_split(record["subject_id"], config.split_seed, config.splits)
                if split not in indexes:
                    index_path = args.output / split / "index.jsonl"
                    index_path.parent.mkdir(parents=True, exist_ok=True)
                    indexes[split] = index_path.open("w", encoding="utf-8")
                indexes[split].write(json.dumps(record, ensure_ascii=True) + "\n")
                prepared_count += 1
    finally:
        for index_file in indexes.values():
            index_file.close()
    print(f"prepared roi samples={prepared_count} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
