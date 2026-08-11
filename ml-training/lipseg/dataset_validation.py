from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from lipseg.license_manifest import DatasetSource

ALLOWED_MASK_IDS = {0, 1, 2, 3}
REQUIRED_FIELDS = {"image", "mask", "subject_id", "consent_record_id"}


def resolve_inside(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = (root / relative_path).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"path escapes dataset root: {relative_path}") from error
    return resolved_path


def iter_samples(source: DatasetSource):
    with source.samples_index.open("r", encoding="utf-8") as samples_file:
        for line_number, line in enumerate(samples_file, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source.name}:{line_number}: invalid JSON: {error}") from error


def validate_sample(source: DatasetSource, line_number: int, sample: dict) -> tuple[str, bool]:
    missing = REQUIRED_FIELDS - sample.keys()
    if missing:
        raise ValueError(f"{source.name}:{line_number}: missing fields: {sorted(missing)}")

    image_path = resolve_inside(source.root, sample["image"])
    mask_path = resolve_inside(source.root, sample["mask"])
    if not image_path.is_file():
        raise ValueError(f"{source.name}:{line_number}: missing image: {image_path}")
    if not mask_path.is_file():
        raise ValueError(f"{source.name}:{line_number}: missing mask: {mask_path}")

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"{source.name}:{line_number}: image/mask dimensions differ: {image.size} != {mask.size}"
            )
        mask_values = set(np.unique(np.asarray(mask.convert("L"), dtype=np.uint8)).tolist())

    invalid_ids = mask_values - ALLOWED_MASK_IDS
    if invalid_ids:
        raise ValueError(f"{source.name}:{line_number}: unsupported mask ids: {sorted(invalid_ids)}")
    if 1 not in mask_values or 2 not in mask_values:
        raise ValueError(f"{source.name}:{line_number}: upper and lower lip labels are required")

    subject_id = str(sample["subject_id"]).strip()
    consent_record_id = str(sample["consent_record_id"]).strip()
    if not subject_id:
        raise ValueError(f"{source.name}:{line_number}: empty subject_id")
    if not consent_record_id:
        raise ValueError(f"{source.name}:{line_number}: empty consent_record_id")
    return subject_id, 3 in mask_values


def validate_source(source: DatasetSource) -> tuple[int, int, int]:
    errors = source.validate_commercial_use()
    if errors:
        raise ValueError(f"{source.name}: commercial-use validation failed: {'; '.join(errors)}")

    sample_count = 0
    inner_mouth_count = 0
    subjects: Counter[str] = Counter()
    for line_number, sample in iter_samples(source):
        subject_id, has_inner_mouth = validate_sample(source, line_number, sample)
        sample_count += 1
        subjects[subject_id] += 1
        inner_mouth_count += int(has_inner_mouth)
    if sample_count == 0:
        raise ValueError(f"{source.name}: no samples found")
    return sample_count, len(subjects), inner_mouth_count
