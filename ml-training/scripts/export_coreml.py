#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import coremltools as ct
except ImportError as error:
    raise SystemExit(
        "coremltools is required for export. Activate ml-training/.venv and run "
        "`python -m pip install -r ml-training/requirements.txt`."
    ) from error

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from lipseg.model import LipSegmentationNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained lip ROI model as FP16 CoreML package.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    width = int(checkpoint["width"])
    height = int(checkpoint["height"])
    model = LipSegmentationNet(class_count=len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    example = torch.zeros(1, 3, height, width)
    traced = torch.jit.trace(model, example)
    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.ImageType(
                name="lip_roi",
                shape=example.shape,
                scale=1.0 / 255.0,
                color_layout=ct.colorlayout.RGB,
            )
        ],
        outputs=[ct.TensorType(name="class_logits")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
    )
    coreml_model.author = "Virtual Makeup"
    coreml_model.short_description = "Lip-only ROI segmentation for commercial virtual makeup runtime."
    coreml_model.version = "1"
    coreml_model.user_defined_metadata["class_names"] = ",".join(classes)
    coreml_model.user_defined_metadata["roi_size"] = f"{width}x{height}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    coreml_model.save(args.output)
    print(f"exported coreml package={args.output} classes={classes} roi={width}x{height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
