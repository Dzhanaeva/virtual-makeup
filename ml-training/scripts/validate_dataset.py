#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from lipseg.dataset_validation import validate_source
from lipseg.license_manifest import load_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate lip segmentation dataset rights and masks.")
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sources = load_sources(args.manifest)
        total_samples = 0
        total_subjects = 0
        for source in sources:
            samples, subjects, inner_mouth_samples = validate_source(source)
            total_samples += samples
            total_subjects += subjects
            print(
                f"validated source={source.name} samples={samples} subjects={subjects} "
                f"inner_mouth_samples={inner_mouth_samples}"
            )
        print(f"dataset valid sources={len(sources)} samples={total_samples} subjects={total_subjects}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"dataset validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
