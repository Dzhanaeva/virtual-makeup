#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from lipseg.config import load_config
from lipseg.license_manifest import load_sources


def verify_config() -> None:
    config = load_config(TRAINING_ROOT / "configs" / "lip_roi.json")
    assert config.width == 192
    assert config.height == 96
    assert config.classes == ["background", "upper_lip", "lower_lip", "inner_mouth"]


def verify_license_guard() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        samples_index = root / "samples.jsonl"
        samples_index.write_text("", encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "datasets": [
                        {
                            "name": "blocked-example",
                            "root": ".",
                            "samples_index": "samples.jsonl",
                            "owner": "unknown",
                            "license_id": "unknown",
                            "commercial_use_approved": False,
                            "annotation_use_approved": False,
                            "consent_basis": "",
                            "approved_by": "",
                            "approved_at": "",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        [source] = load_sources(manifest)
        errors = source.validate_commercial_use()
        assert errors


def verify_model_if_available() -> None:
    try:
        import torch

        from lipseg.model import LipSegmentationNet
    except ImportError:
        print("smoke model_forward=skipped reason=torch_not_installed")
        return
    model = LipSegmentationNet(class_count=4)
    result = model(torch.zeros(1, 3, 96, 192))
    assert tuple(result.shape) == (1, 4, 96, 192)
    print("smoke model_forward=passed")


def main() -> int:
    verify_config()
    verify_license_guard()
    verify_model_if_available()
    print("smoke config=passed license_guard=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
