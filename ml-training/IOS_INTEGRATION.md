# iOS Integration Contract

Do not replace the existing `faceParsing.mlmodel` runtime until a reviewed
`LipSegmentation.mlpackage` has passed the device acceptance checks below.

## Runtime Input

- Build a lip ROI from the current MediaPipe lip landmarks.
- Expand the landmark bounds using the same margins as `configs/lip_roi.json`.
- Crop and resize the camera image to RGB `192 x 96`.
- Run `LipSegmentation` asynchronously outside ARKit renderer callbacks.

## Runtime Output

The output tensor is `class_logits` with shape `1 x 4 x 96 x 192`.

- `0`: background
- `1`: upper lip
- `2`: lower lip
- `3`: inner mouth

Map the upper and lower lip probabilities back to the current MediaPipe UV
region. MediaPipe remains the hard geometric bound: CoreML may remove pixels
inside that region but must not add pixels outside it. Exclude class `3` from
pigment and gloss.

## Temporal Rules

- Keep the last accepted mask and its MediaPipe pose.
- Warp that mask with the updated MediaPipe mesh between neural inferences.
- Reject stale results when ROI pose, scale, or mouth-open state changed beyond
  the tuned threshold while inference was running.
- Apply a short exponential moving average to mask probabilities in UV space,
  not in screen space.
- Fall back to the MediaPipe mask when model confidence is low or the neural
  result is stale.

## Device Acceptance

Validate on real supported iPhones using recorded scenarios:

- neutral face under bright, indoor, and dim light;
- slow and fast head rotation;
- speaking;
- wide smile;
- mouth opening and closing;
- partial profile;
- short tracking loss and reacquisition.

Record:

- p50 and p95 CoreML inference latency;
- p50 and p95 end-to-end mask age at display time;
- foreground IoU for upper lip, lower lip, and inner mouth;
- boundary F-score;
- visible failures per recorded minute.

The model is not production-ready until the model card records the reviewed
dataset manifest, training commit, exported model checksum, metrics, and device
test results.
