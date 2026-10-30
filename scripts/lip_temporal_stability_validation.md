# Contour stability validation

Measured on 2026-10-16 with `scripts/check_lip_temporal_stability.swift`.
These are deterministic filter measurements, not a replay of a device recording.

## Method

The synthetic mouth has 80 points in the same four-ring order as
`CanonicalLipGeometry.attentionLipIndices`, with a neutral width of 80 pixels.
Upper and lower sections are evaluated independently, including their shared corners.
The frozen legacy comparator implements the previous maximum-of-median/p95/aperture
activity policy, its confidence/dead-zone response, and the raw-contour shape bypass.
Both policies receive the same samples at each frequency.

- Idle: independent uniform noise in [-1.2, 1.2] pixels on each coordinate,
  fixed LCG seed `0x12345678`, three seconds total, first second excluded from RMS.
- Isolated spike: one lower outer point moves by 3, 4.5, or 6 pixels for one observation.
  The quiet-contour peak includes the return observation. A separate noisy control/spike
  pair measures the additional output displacement caused by the spike alone.
- Opening: both lip sections move in opposite directions, reaching an additional
  8 pixels at their centres over 200 ms.
- Smile: horizontal expansion reaches 12% over 200 ms (4.8 pixels at a corner).
- Expression delay: interpolated crossing of 90% of the target, relative to the
  input's 90% crossing. Tested with current ARKit expression evidence and with only
  confirmation from the next MediaPipe observation.
- Recovery: inconsistent but AR-bounded candidates, alternating 6-pixel candidates,
  common-pose comparison after rigid transport, and a four-second temporal gap.

The helper checks also require that a first coherent observation stays at nominal
alpha <= 0.70, that a following coherent observation can reach 0.96, that corresponding
ARKit evidence allows 0.96 immediately, and that ARKit cannot validate an isolated
unsupported point. A settled second observation can confirm a persistent new shape.

## Results

| Observation rate | Idle RMS, old -> new (px) | Quiet spike peak, old -> new (px) | Noisy 3px spike extra peak, old -> new (px) | Noisy 6px spike extra peak, old -> new (px) |
| --- | --- | --- | --- | --- |
| 15 Hz | 0.592 -> 0.418 | 3-6 -> 0.365 | 3.041 -> 0.403 | 6.034 -> 0.418 |
| 30 Hz | 0.783 -> 0.377 | 3-6 -> 0.196 | 3.404 -> 0.145 | 6.398 -> 0.144 |
| 60 Hz | 0.801 -> 0.332 | 3-6 -> 0.102 | 2.897 -> 0.210 | 5.895 -> 0.211 |

| Observation rate | Opening 90% lag, old -> new (ms) | Smile 90% lag, old -> new (ms) | Fresh recovery confirmation |
| --- | --- | --- | --- |
| 15 Hz | 34.5 -> 15.7 | 413.6 -> 29.6 (29.9 without AR evidence) | 133.3 ms |
| 30 Hz | 17.0 -> 2.2 | 452.7 -> 11.5 | 66.7 ms |
| 60 Hz | 14.0 -> 8.3 | 488.6 -> 22.8 | 50.0 ms |

In the other expression cases, the two evidence paths have the same rounded 90% lag.
The large legacy smile delay includes its slow final settling and dead zone; it is
not an estimate of total camera-to-display latency. An isolated spike leaves contour
activity at zero. Twelve inconsistent candidates never confirm recovery, including
the alternating 6-pixel case. A pause restarts the consensus run.

Existing checks for elapsed-time response, 15/30/60 Hz activity, duplicate render
callbacks, rigid lead reversal/stop, and source-depth reprojection continue to pass.

## Landmark submission pacing

A ten-second 30 FPS capture sequence with callback delivery jitter of 1-7 ms and a
12 ms detector service time exercises the fair-state low-motion interval. The old
0.034 s callback-time gate submitted 15.0 FPS. With the interval reduced to
0.030 s but still measured from callback arrival, it also submitted 15.0 FPS;
the 0.030 s capture-time gate submitted 30.0 FPS. The check also verifies that an older callback cannot replace
the newest pending frame and that non-monotonic or non-finite capture times are
rejected. This measures scheduling capacity when the detector keeps up, not actual
MediaPipe throughput on a device.

## Parameter rationale

- Spatial support uses three of five neighbours (two along the same lip section,
  three corresponding points on the other rings), direction cosine >= 0.55 and
  magnitude ratio 0.5-2. Current ARKit opening/jaw/smile evidence can reduce the
  neighbour requirement to two, but must match the local movement direction.
  Opposing upper/lower opening movements never vote against each other.
- The 1.4-pixel outlier correction limit scales with mouth width. A nominal outlier
  alpha of 0.14 bounds a 30 Hz correction to 0.196 pixels, matching the measured
  3-6 pixel spike peaks. Correction is bounded relative to the filtered point,
  so a spike's return cannot pull the mask toward the previous bad raw sample.
- Unconfirmed coherent motion uses alpha up to 0.60, or 0.70 for a geometrically
  valid large shape change. Confirmed motion uses 0.82-0.96. All alphas and stability
  gains use elapsed-time response. An 80 ms trust hold lets a confirmed expression
  finish settling after raw velocity returns to zero; the smile check exposed a
  long settling tail without this hold.
- Full contour activity needs spatially coherent coverage of at least 40% of the
  points. Local responses remain independent, so one moving lip section stays fast
  and a sparse spike cannot disable smoothing over the whole mouth.
- Recovery needs at least three consistent observations spanning at least 50 ms.
  Gaps over 220 ms or runs over 550 ms restart consensus. In a common ARKit pose,
  median point disagreement must be <= 3.5% of mouth width, p95 <= 7.5%, centre
  disagreement <= 5%, and width ratio within 0.90-1.10. The tighter median rejects
  the alternating 6-pixel recovery scenario at an 80-pixel mouth width.
- A 650 ms temporal gap starts a new landmark epoch and texture generation, just
  after the existing freshness fade ends at 620 ms. Recovery clears the old mesh
  and texture before publishing the new position. The renderer's existing 70 ms
  visibility ramp restores the mask after fresh consensus and texture completion.

## Device verification and pause diagnostics

The standalone test covers numerical filtering and consensus. ARKit/MediaPipe
callback scheduling, real camera alignment, texture completion races, thermal load,
and the visible recovery ramp require a supported iPhone.

The cause of the reported four-second pause is not established. Camera and render
ages and processed-result age are emitted with the existing five-second performance
summary, together with background/active events. ARSession interruptions/errors
invalidate tracking. Existing throttled submit/callback diagnostics now include
`submitCallMs`, `callbackQueueDelayMs`, and `watchdogQueueLag`; the last field measures
lateness from the actual scheduled watchdog deadline on the landmark queue. Stale
results are checked again before publication, and long processing gaps invalidate
the sequence. Compare these fields with existing preprocessing/surface times and
camera/render rates to distinguish a stopped camera, detector/queue delay, and a
render pause without inferring a cause from a late timeout alone.
