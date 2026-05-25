# AGENTS.md

## Source of truth

This file describes the current project conventions for coding agents and contributors.
Keep it up to date when architecture, build steps, dependencies, or project layout change.
If this file conflicts with the actual source code, Xcode project settings, or checked-in
configuration, the code and configuration win. Update this file after reconciling the mismatch.

## Project overview

Virtual Makeup is a SwiftUI iOS app for real-time virtual lipstick preview. The app opens an
AR camera view, detects a face with MediaPipe Tasks Vision, tracks lip landmarks, builds a
SceneKit lip mesh, and renders a generated lipstick texture over the user's mouth.

Current core technologies:

- SwiftUI for the app shell and user controls.
- `UIViewRepresentable` bridge for `ARSCNView`.
- ARKit face tracking and SceneKit rendering for camera, face anchor, occlusion, and lip mesh.
- MediaPipe Tasks Vision `FaceLandmarker` for live lip landmark detection.
- CocoaPods for `MediaPipeTasksVision`.
- Bundled model/assets: `face_landmarker.task`, `faceParsing.mlmodel`, `face_model_with_iris.obj`,
  and `Assets.xcassets`.

## Important files

- `Virtual Makeup/Virtual_MakeupApp.swift`: SwiftUI app entry point.
- `Virtual Makeup/ContentView.swift`: top-level UI, face detection status, and lipstick presets.
- `Virtual Makeup/FaceTrackingView.swift`: AR view bridge, MediaPipe live stream handling,
  lip contour processing, smoothing, texture generation, and SceneKit mesh rendering.
- `Virtual Makeup/VirtualMakeup-Bridging-Header.h`: Objective-C bridging header configured in
  the Xcode project.
- `Virtual-Makeup-Info.plist`: app Info.plist, including camera permission text.
- `Podfile` and `Podfile.lock`: CocoaPods dependency definition and lockfile.
- `Virtual Makeup.xcworkspace`: preferred Xcode entry point when CocoaPods are installed.
- `Virtual Makeup.xcodeproj`: app project settings and target configuration.

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

There is currently no dedicated test target. When extracting pure logic from
`FaceTrackingView.swift` into smaller units, add focused tests for deterministic geometry,
color, smoothing, and texture math.

## Architecture guidelines

- Keep SwiftUI state and user controls in SwiftUI views. Keep camera/session/rendering code behind
  the `UIViewRepresentable` bridge.
- Treat `Coordinator` as the boundary for ARKit, MediaPipe delegate callbacks, and rendering state.
  Avoid pushing UI-only concerns into it.
- Keep real-time paths small and predictable. Do not block ARKit renderer callbacks, AR session
  callbacks, or MediaPipe delegate queues with heavy synchronous work.
- Preserve the existing queue/lock model unless replacing it with a clearly safer concurrency
  design. Shared mutable render, texture, detection, and viewport state must remain synchronized.
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
- Prune pending frame state so delayed MediaPipe callbacks cannot grow memory usage unbounded.
- Validate landmark counts, transforms, viewport size, lip bounds, and finite numeric values before
  rendering.
- Keep coordinate transforms explicit: captured image space, viewport space, normalized landmark
  space, UV space, AR face local space, and SceneKit space should not be mixed implicitly.
- Do not regenerate high-cost textures more often than necessary. Preserve low-latency throttling
  and superseded-request checks when changing texture rendering.
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

## Change workflow

- Read the relevant code before editing. Existing implementation details are the first source of
  truth.
- Keep changes narrowly scoped to the requested behavior.
- Preserve user changes in the working tree. Do not revert unrelated edits.
- After meaningful changes, run the most relevant build or verification command available in the
  local environment and report any limitation clearly.
- If architecture, dependencies, build commands, assets, or file ownership change, update this
  `AGENTS.md` in the same change.
