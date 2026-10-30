# AGENTS.md

## Source of truth

This file describes the current project conventions for coding agents and contributors.
Keep it up to date when architecture, build steps, dependencies, or project layout change.
If this file conflicts with the actual source code, Xcode project settings, or checked-in
configuration, the code and configuration win. Update this file after reconciling the mismatch.

## Project overview

Virtual Makeup is a SwiftUI iOS app for real-time virtual lipstick and lip liner preview. The app opens an
AR camera view, detects a face with MediaPipe Tasks Vision, tracks lip landmarks, builds a
SceneKit lip mesh, and renders generated lipstick and liner textures over the user's mouth.

Current core technologies:

- SwiftUI for the app shell and user controls.
- `UIViewRepresentable` bridge for `ARSCNView`.
- ARKit face tracking and SceneKit rendering for camera, face anchor, occlusion, and lip mesh.
- MediaPipe Tasks Vision `FaceLandmarker` for live lip landmark detection.
- Metal for camera-aware lipstick colour, linear-RGB light and
  shadow processing with sRGB input/output, texture preservation, a temporally stabilized
  camera-derived shine mask, and the feathered lip alpha mask; the legacy CPU compositor is
  fallback-only and follows the same colour-space and material-signal contract.
- CocoaPods for `MediaPipeTasksVision`.
- Bundled model/assets: `face_landmarker.task`, `face_model_with_iris.obj`, and
  `Assets.xcassets`.
- `ml-training/` for an optional future commercial-safe, lip-only CoreML model training and
  export workflow. No CoreML segmentation model is currently bundled or executed at runtime.

## Important files

- `Virtual Makeup/Virtual_MakeupApp.swift`: SwiftUI app entry point.
- `Virtual Makeup/ContentView.swift`: top-level UI, face detection status, lipstick presets, and
  the temporary lip liner product/shade catalogue.
- `Virtual Makeup/FaceTrackingView.swift`: thin SwiftUI/`ARSCNView` bridge.
- `Virtual Makeup/FaceTrackingCoordinator.swift`: ARKit and MediaPipe callback orchestration,
  tracking state, smoothing, prediction, and render request scheduling. Gloss, oil, and satin
  can refresh camera reflections from newer AR frames between accepted landmark contours;
  `textureQueue` prepares the frame and coalesces pending reflection requests.
- `Virtual Makeup/LipstickTypes.swift`: product metadata, lipstick material profiles, and the
  independent wax-pencil profile and immutable sRGB liner style snapshot.
- `Virtual Makeup/LipModels.swift` and `CanonicalLipGeometry.swift`: lip/face geometry data,
  coordinate transforms, canonical topology, and mesh lookup data.
- `Virtual Makeup/LipTemporalStability.swift`: frame-rate-independent lip filter response,
  spatial and temporal motion confidence for each lip section, bounded point corrections,
  reacquisition consensus, rigid display lead, camera-plane inverse projection math, and
  capture-timestamp pacing for landmark submission. It also composes the oriented
  source-image-to-viewport mapping used by camera-driven lip reflections.
- `Virtual Makeup/LipDiagnostics.swift`: throttled debug output and Release-available
  performance logging for power/thermal state, frame rates, and contour health.
  It also samples the camera-derived lip reflection at most 10 times per second
  and prints a one-second summary of bright-region movement in ordinary builds.
- `Virtual Makeup/LipInnerBoundary.swift`: shared inner lip coverage and recessed-colour
  profile, including a full closed-mouth fill, the open-mouth cutout, and
  hysteretic ARKit mouth-closure classification.
- `Virtual Makeup/LipMeshRenderer.swift`, `LipTextureRenderer.swift`, and
  `MetalLipColorCompositor.swift`: SceneKit geometry, camera-aware lipstick and liner colour,
  and the independent soft lip liner mask. Both colour layers sample the same captured camera
  frame and canonical UV map; liner pigment stays separate from lipstick finish and opacity.
  The screen-space mesh keeps the tracked outer contour as an exact UV/screen vertex ring
  and adds a separate outer carrier ring for feather coverage. The liner uses one closed
  reference based on the tracked outer UV landmarks. Its lower samples shift
  slightly outward and taper back to the original corners; the lower mask is
  narrower on the lip and extends a little past its edge. The upper stroke
  retains its original profile. Material opacity controls pigment density
  separately from the mask shape. Metal and CPU liner colour set pigment value
  from absolute camera pixel brightness before applying local light and shadow detail.
  All finishes combine pigment coverage and reflection coverage as separate
  layers in linear RGB before the shared contour and mouth mask; the CPU camera
  fallback uses the same alpha composition. Pigment density scales only the
  pigment layer. Gloss retains its lower-lip reflection gate, with white
  reflection coverage independent of pigment density. SceneKit intensity scales
  the result.
  The shimmer finish uses the wet camera-reflection response with continuous
  pearlescent micro-grain and fixed, tiny two-strength reflections in canonical
  UV space. Most particles are soft; a small fraction have a sharper bright
  core. Their colour and intensity respond to captured light, reflection
  geometry, and source-frame viewing angle on both Metal and CPU paths. Pearl
  and particle coverage are optical layers over the pigment; their strength
  does not scale with pigment density. The common lip and mouth mask still gates
  the result.
  Tune its visible strength through `LipMaterialProfile.shimmerIntensity`; both
  compositors consume that same product value.
  The Metal compositor applies one analytic outer alpha
  curve, then a bounded outer-only blur that cannot raise its coverage or refill the mouth.
  Shimmer particles use a separate inner-mouth gate: when the mouth opens, their
  reflection fades out several texture pixels before the inner lip boundary.
  The lipstick's shared soft inner alpha remains unchanged.
  Gloss extracts the current camera reflection from four source-image samples
  in linear RGB, then uses a lower-lip annulus gate. Oil and satin use four guarded
  samples. All camera-derived highlights use stabilized shine history: small
  fluctuations in either direction receive a slower response while lip motion
  is low; larger changes and moving contours remain responsive. Gloss keeps
  a fast rise for newly captured reflections. A one-pixel cross filter and
  symmetric slow response stabilize only low-strength highlight edges. Weak
  fringe pixels also carry temporal confidence: while lip motion is low, new
  pixels need repeated evidence even when bright, and existing ones fade after
  repeated signal loss. Metal shine history stores the stabilized ungated mask
  and confidence in two half-float channels so the contour gate is applied once per frame;
  the CPU fallback keeps the same confidence separate from its shine history.
  Their finish thresholds, concentration, gain, and cap come from
  `LipMaterialProfile`; satin also retains bright colour detail from the captured lip texture
  while gloss and oil require a more neutral residual. No normal lobe or fixed UV highlight is
  added. The source-image mapping
  includes frame orientation, crop, and mirror. Satin keeps a soft value shoulder on pale
  pigment and converts the same camera light to more visible reflection coverage as the shade
  approaches white. The strongest camera-derived satin reflections add a narrow white core
  over the softer product-coloured sheen on both Metal and CPU paths. Other finishes and liner retain their
  established sample mapping and finish response.
- `Virtual Makeup/BlushRenderer.swift`: SceneKit blush rendering.
- `Virtual Makeup/VirtualMakeup-Bridging-Header.h`: Objective-C bridging header configured in
  the Xcode project.
- `Virtual-Makeup-Info.plist`: app Info.plist, including camera permission text.
- `Podfile` and `Podfile.lock`: CocoaPods dependency definition and lockfile.
- `Virtual Makeup.xcworkspace`: preferred Xcode entry point when CocoaPods are installed.
- `Virtual Makeup.xcodeproj`: app project settings and target configuration.
- `ml-training/README.md`: lip-only model contract and training workflow.
- `ml-training/COMMERCIAL_DATA_POLICY.md`: required provenance rules for training data and weights.

Do not manually edit generated CocoaPods files under `Pods/`. Change `Podfile`, then run
`pod install` and keep generated project files consistent with the repository's chosen
dependency policy.

## Build and verification

Prefer opening/building the workspace, not the project, after Pods are installed:

```sh
open "Virtual Makeup.xcworkspace"
```

For command-line verification, use the workspace and the `Virtual Makeup` scheme when available:

```sh
xcodebuild -workspace "Virtual Makeup.xcworkspace" -scheme "Virtual Makeup" -destination 'generic/platform=iOS' build
```

This app depends on camera and AR face tracking behavior. Simulator builds can catch compile
errors, but meaningful validation requires a real device that supports `ARFaceTrackingConfiguration`.

There is currently no dedicated test target. When extracting pure logic from the tracking,
geometry, or rendering units, add focused tests for deterministic geometry, color, smoothing,
and texture math.

For the inner lip coverage and colour profile, run the standalone check from the repository root:

```sh
xcrun swiftc -module-cache-path /tmp/virtual-makeup-swift-cache "Virtual Makeup/LipInnerBoundary.swift" scripts/check_lip_inner_boundary.swift -o /tmp/virtual-makeup-inner-boundary-check
/tmp/virtual-makeup-inner-boundary-check
```

It checks the CPU profile across mouth openings and lip thicknesses, and compares it with
the production Metal functions across texture scales when a Metal GPU is available.

For temporal stability and source-frame projection math:

```sh
xcrun swiftc -module-cache-path /tmp/virtual-makeup-swift-cache "Virtual Makeup/LipTemporalStability.swift" scripts/check_lip_temporal_stability.swift -o /tmp/virtual-makeup-temporal-check
/tmp/virtual-makeup-temporal-check
```

This deterministic check compares the old and current rigid lead, verifies matching
activity and filter response at 15/30/60 Hz, and measures synthetic yaw/pitch parallax.
It also compares the previous global contour filter with the current point filter on seeded
noise, 3-6 pixel spikes, mouth opening, smile, and recovery after a pause. The scenarios,
measurements, and parameter rationale are in `scripts/lip_temporal_stability_validation.md`.
It checks fair-state landmark pacing against a 30 FPS camera with irregular callback delivery,
plus source-image mapping through rotation, mirroring, crop, and neighbor offsets.
Real camera alignment and display-image latency still require on-device capture.

For the lip-only CoreML training utilities:

```sh
python3 -m compileall ml-training
python3 ml-training/scripts/smoke_test.py
```

## Architecture guidelines

- Keep SwiftUI state and user controls in SwiftUI views. Keep camera/session/rendering code behind
  the `UIViewRepresentable` bridge.
- Treat `Coordinator` as the boundary for ARKit, MediaPipe delegate callbacks, and rendering state.
  Avoid pushing UI-only concerns into it.
- In the screen-space lip path, retain the MediaPipe contour with the ARKit pose
  from its source frame. Move it with the current rendered face pose and the
  same transport for reliability checks, smoothing, and display. At yaw/pitch,
  use per-point source-frame ARKit face depth with exact source reprojection;
  fall back to the fixed face-local screen basis or bounded similarity transform
  if depth is unavailable. The MediaPipe facial matrix belongs to its captured
  frame and is only required for the face-surface path.
- Derive adaptive smoothing activity from successive accepted raw MediaPipe
  contours after rigid transport. Keep the per-observation noise floor before
  converting residual displacement to a rate; apply elapsed-time response to
  confidence and alpha as well.
- Evaluate neighbouring motion separately on the upper and lower lip sections. A fast point
  response requires spatial support plus a following observation or corresponding ARKit
  expression evidence; ARKit does not supply the MediaPipe lip shape. Commit motion history
  only after publication checks pass. Shape changes request a validated fast response rather
  than bypassing smoothing. Reacquisition requires a run of mutually consistent candidates
  transported into one ARKit pose. After a temporal gap, invalidate the landmark epoch and
  texture generation before reacquiring; never interpolate from the old mask to a new location.
- Keep real-time paths small and predictable. Do not block ARKit renderer callbacks, AR session
  callbacks, or MediaPipe delegate queues with heavy synchronous work.
- Preserve the existing queue/lock model unless replacing it with a clearly safer concurrency
  design. `textureQueue` owns one texture request until its GPU completion and readback finish; pending
  texture work is coalesced. Never reuse compositor targets before completion. Shared mutable render, texture, detection, and viewport state must remain synchronized.
- UI state updates from background callbacks must happen on the main thread or main actor.
- Prefer extracting cohesive private types from `FaceTrackingView.swift` over adding more unrelated
  responsibility to `Coordinator`.
- Keep model and asset names stable unless all bundle lookups and Xcode target membership are
  updated together.

## Swift coding style

- Follow idiomatic Swift with early `guard` exits, small focused helpers, value types for geometry
  data, and explicit names for coordinate spaces.
- Avoid force unwraps and force casts in production paths.
- Prefer `let` over `var` when values do not change.
- Keep comments sparse and useful. Comment non-obvious math, coordinate transforms, thread-safety
  decisions, or MediaPipe/ARKit assumptions.
- Use `#if DEBUG` for diagnostic logging. Throttle logs in frame-by-frame paths.
- Keep files ASCII unless the existing file already contains localized user-facing text.
- Keep Russian user-facing strings consistent with the current UI copy unless localization is being
  introduced deliberately.

## AR, MediaPipe, and rendering rules

- Check `ARFaceTrackingConfiguration.isSupported` before starting face tracking.
- Keep MediaPipe live stream timestamps monotonic and avoid overlapping landmark detection work.
- Pace new landmark submissions by ARFrame capture timestamps; when a detector is busy or a
  frame is rate-limited, retain only the newest pending frame.
- Prune pending frame state so delayed MediaPipe callbacks cannot grow memory usage unbounded.
- Validate landmark counts, transforms, viewport size, lip bounds, and finite numeric values before
  rendering.
- Keep coordinate transforms explicit: captured image space, viewport space, normalized landmark
  space, UV space, AR face local space, and SceneKit space should not be mixed implicitly.
- For camera-driven gloss, oil, and satin, transport the accepted contour into the sampled AR
  frame, keep its source-image-to-viewport transform with the texture request, and accept only
  newer capture times in the same tracking epoch. Do not derive reflection movement from UV time.
- Do not regenerate high-cost textures more often than necessary. Preserve low-latency throttling
  and superseded-request checks when changing texture rendering.
- Keep the Metal lip target compact, retain source `CVMetalTexture` objects until command-buffer
  completion, and preserve the failure-counted CPU fallback when changing the compositor.
  The compositor completes asynchronously on `textureQueue`; it must not call
  `waitUntilCompleted`. Temporal resets invalidate outstanding shine-history completions.
- Keep liner RGB in its own opaque sRGB texture and the stroke coverage in the alpha-only
  material property. Apply liner pigment density and user intensity once at the SceneKit
  material, and reject texture completions from an older liner style revision. The camera-lit
  liner pass runs in the lip command buffer only while the liner is enabled; CPU fallback uses
  the same source-frame carrier mapping and linear-RGB light/detail bounds.
- Keep the tracked outer UV landmarks as the liner reference base. Offset only
  the sampled lower segment outward, tapering to zero at both mouth corners.
  Expand SceneKit coverage with a separate carrier ring so the exterior stroke
  remains on the mesh. The DEBUG-only `LipMeshRenderer.debugLinerAlignmentEnabled`
  flag overlays the tracked upper contour in cyan and the sampled liner reference in orange.
- Keep UIKit screen/window reads in SwiftUI or the view's main-thread layout callbacks; cache
  the resulting viewport/display values for SceneKit's rendering queue.
- Update the screen-space lip mesh in SceneKit's per-frame
  `didApplyAnimationsAtTime` phase; face-anchor callbacks cache the latest tracked
  anchor and node. Compute rigid prediction only for new, timestamped ARFrame
  poses whose camera transform matches the renderer camera, and hold its lead
  across duplicate render callbacks. Hide the mesh if camera frame updates stop.
  Treat ARFrame capture timestamps separately
  from the unknown presentation timestamp of ARSCNView's camera background.
- Clear SceneKit nodes and tracking state when anchors are removed, sessions stop, tracking is lost,
  or detection becomes stale.

## Dependency and project hygiene

- Update `Podfile.lock` whenever dependency versions change.
- Keep Xcode project target membership in sync with added Swift files, models, tasks, and assets.
- Avoid committing personal Xcode state such as `xcuserdata` changes unless the repository already
  intentionally tracks that exact file and the change is required.
- Do not change signing, bundle identifier, deployment targets, or supported platforms as a side
  effect of unrelated work.
- Current project settings show an iOS deployment target of `18.0`; the `Podfile` declares
  `platform :ios, '15.0'`. If deployment support is changed, reconcile both intentionally.
- Treat `ml-training/data/` and `ml-training/artifacts/` as local-only by default. Do not commit
  face images, masks, checkpoints, or exported models until dataset rights and distribution terms
  are reviewed for the commercial product.
- Do not use CelebAMask-HQ, weights trained on CelebAMask-HQ, unknown-license datasets, or scraped
  images in production training. Every training source must pass the checked-in commercial manifest
  validation.

## Change workflow

- Read the relevant code before editing. Existing implementation details are the first source of
  truth.
- Keep changes narrowly scoped to the requested behavior.
- Preserve user changes in the working tree. Do not revert unrelated edits.
- After meaningful changes, run the most relevant build or verification command available in the
  local environment and report any limitation clearly.
- If architecture, dependencies, build commands, assets, or file ownership change, update this
  `AGENTS.md` in the same change.
