# LipSegmentation Model Card

## Artifact

- Model package:
- SHA-256:
- Training commit:
- Export date:
- Reviewer:

## Intended Use

Lip-only ROI segmentation for real-time virtual makeup rendering in the Virtual
Makeup iOS application.

## Training Data

- Reviewed manifest path:
- Dataset versions:
- Subject count:
- Frame count:
- Consent or legal basis:
- Commercial-use approval reference:

## Model

- Architecture: `LipSegmentationNet`
- Initialization: scratch-only, no third-party pretrained weights
- Input: RGB `192 x 96`
- Output classes: background, upper lip, lower lip, inner mouth

## Validation

- Foreground IoU:
- Upper-lip IoU:
- Lower-lip IoU:
- Inner-mouth IoU:
- Boundary F-score:
- Failure review notes:

## Device Validation

- Devices:
- p50 CoreML latency:
- p95 CoreML latency:
- p50 displayed-mask age:
- p95 displayed-mask age:
- Thermal test duration:
- Thermal result:

## Known Limitations

- 
