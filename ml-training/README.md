# Lip ROI Segmentation Training

This directory contains the commercial-safe training pipeline for the lip-only
segmentation model used by Virtual Makeup.

The pipeline intentionally does not download public face datasets or pretrained
face-parsing weights. Public face datasets often restrict commercial use or have
unclear provenance. Add only internal data or externally licensed data that has
passed legal review.

## Model Contract

The exported CoreML model is named `LipSegmentation.mlpackage`.

- Input: RGB lip ROI image, `192 x 96`.
- Output: logits, shape `1 x 4 x 96 x 192`.
- Classes:
  - `0`: background
  - `1`: upper lip
  - `2`: lower lip
  - `3`: inner mouth

The iOS runtime must crop the ROI from MediaPipe landmarks. MediaPipe remains the
hard outer bound. The neural mask refines pixels inside that bound and must never
expand beyond it.

## Data Layout

Create a dataset manifest from `manifests/datasets.example.json`. Each dataset
entry points to a JSONL sample index:

```json
{"image":"images/subject-001-frame-001.jpg","mask":"masks/subject-001-frame-001.png","subject_id":"subject-001","consent_record_id":"consent-001"}
```

Masks are single-channel PNG files using the four class IDs defined above.
Subjects, not frames, are assigned to train, validation, and test splits.

## Setup

```sh
cd ml-training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Workflow

```sh
python scripts/validate_dataset.py \
  --manifest manifests/datasets.json

python scripts/prepare_roi_dataset.py \
  --manifest manifests/datasets.json \
  --output data/lip-roi

python scripts/train.py \
  --data data/lip-roi \
  --output artifacts/lipseg

python scripts/export_coreml.py \
  --checkpoint artifacts/lipseg/lipseg-best.pt \
  --output artifacts/LipSegmentation.mlpackage
```

Do not add source images, annotations, checkpoints, or exported models to git
until their distribution terms have been reviewed.

Before adding an exported model to Xcode, complete
`MODEL_CARD.template.md` and follow `IOS_INTEGRATION.md`.
