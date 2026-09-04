import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

final class Coordinator: NSObject, ARSCNViewDelegate, ARSessionDelegate, FaceLandmarkerLiveStreamDelegate {

    private struct ContourSmoothingParameters {
        let alpha: CGFloat
        let deadZone: CGFloat
        let isStationary: Bool
        let stability: CGFloat
        let activity: CGFloat
        let medianResidual: CGFloat
        let p95Residual: CGFloat
    }
    private struct PendingBlushStyle {
        let color: UIColor
        let opacity: Double
    }

    @Binding private var isFaceDetected: Bool
    weak var sceneView: ARSCNView?

    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let landmarkQueue = DispatchQueue(label: "VirtualMakeup.MediaPipeLandmarks", qos: .userInteractive)
    private let textureQueue = DispatchQueue(label: "VirtualMakeup.LipTexture", qos: .userInteractive)
    private let lipTextureRenderer = LipTextureRenderer()
    private let lipMeshRenderer = LipMeshRenderer()
    private let blushRenderer = BlushRenderer()
    private let arRendererFPS = FPSMeter("ar_renderer")
    private let mediaPipeFPS = FPSMeter("mediapipe_result")
    private let textureFPS = FPSMeter("lip_texture")
    private let viewportLock = NSLock()
    private let detectionLock = NSLock()
    private let trackingLock = NSLock()
    private let motionLock = NSLock()
    private let mouthStateLock = NSLock()
    private let textureStateLock = NSLock()
    private let meshStateLock = NSLock()
    private let rendererStateLock = NSLock()

    private var faceLandmarker: FaceLandmarker?
    private var cachedViewportSize: CGSize = .zero
    private var cachedRenderScale: CGFloat = 2
    private var cachedInterfaceOrientation: UIInterfaceOrientation = .unknown
    private var cachedViewportRevision = 0
    private var isDetectingLandmarks = false
    private var activeLandmarkTimestampInMilliseconds: Int?
    private var latestPendingFrameContext: FrameContext?
    private var landmarkTrackingEpoch = 0
    private var landmarkReadyEpoch: Int? = 0
    private var lastLandmarkTimestampInMilliseconds = -1
    private var lastAcceptedLandmarkTimestampInMilliseconds = -1
    private var lastLandmarkSubmitTime: CFTimeInterval = 0
    private var lastLandmarkSubmitPose: LipMotionPose?
    private var activeLiveFrame: PendingLiveFrame?
    private var faceSurfaceTopologyCache: FaceSurfaceTopologyCache?
    private var shouldExpandFaceSurfaceTopology = false
    private var faceSurfaceTopologyAnchorIdentifier: UUID?
    private var missedDetectionCount = 0
    private var smoothedLipContour: LipContour?
    private var smoothedLipPose: LipPose?
    private var contourStabilityConfidence: CGFloat = 0
    private var contourIsStationary = false
    private var isSessionActive = false
    private var cameraTrackingIsNormal = false
    private var hasTrackedFaceAnchor = false
    private var trackedFaceAnchorIdentifier: UUID?
    private var latestLipMotionSample: LipMotionSample?
    private var latestRenderMotionSample: LipMotionSample?
    private var latestRenderCenterDelta: CGVector?
    private var latestRenderPredictionLead = CGVector.zero
    // Captured once per AR anchor and intentionally preserved across detector
    // resets. These face-local points then move only with the head/camera pose.
    private var rigidLipReference: RigidLipReference?
    private var textureGeneration = 0
    private var nextTextureRequestID = 0
    private var latestTextureRequestID = 0
    private var latestInstalledTextureRequestID = 0
    private var isRenderingTexture = false
    private var pendingTextureRequest: LipTextureRequest?
    private var isLipTextureRefreshPending = false
    private var lastTextureSubmitTime: CFTimeInterval?
    private var lastAcceptedLipShapeTime: CFTimeInterval?
    private var lastDisplayedLipTextureTime: CFTimeInterval?
    private var lastAcceptedMotionPose: LipMotionPose?
    private var latestMeshContour: LipContour?
    private var latestMeshContourCaptureTime: CFTimeInterval?
    private var latestMeshContourAcceptedAt: CFTimeInterval?
    private var latestMeshMotionPose: LipMotionPose?
    private var previousRealContourSample: AcceptedRealContourSample?
    private var latestRealContourSample: AcceptedRealContourSample?
    private var latestMeshAnchorIdentifier: UUID?
    private var latestMeshViewportRevision: Int?
    private var latestMeshTrackingEpoch: Int?
    private var latestLipTexture: LipTexture?
    private var latestLipTextureGeneration: Int?
    private var latestLipTextureTrackingEpoch: Int?
    private var latestLipTextureAnchorIdentifier: UUID?
    private var latestLipTextureViewportRevision: Int?
    private var lastTextureMouthOpen: Bool?
    private var pendingTextureMouthOpen: Bool?
    private var pendingTextureMouthOpenSince: CFTimeInterval = 0
    private var lastLoggedMediaPipeLandmarkCount: Int?
    private var neutralMouthWidth: Float?
    private let lightingLock = NSLock()
    private var smoothedLipLightingFactor: CGFloat = 1
    private var lastLightingUpdateTime: CFTimeInterval = 0
    private var pendingLipOpacity: Double?
    private var pendingBlushStyle: PendingBlushStyle?
    private var rendererClearIsPending = false

    private static let outerLipIndices = CanonicalLipGeometry.outerLipIndices
    private static let innerLipIndices = CanonicalLipGeometry.innerLipIndices
    private static let attentionLipIndices = CanonicalLipGeometry.attentionLipIndices
    private static let arKitMouthLeftIndex = 249
    private static let arKitMouthRightIndex = 684
    private static let arKitMouthTopIndex = 24
    private static let arKitMouthBottomIndex = 25
    private static let maxMotionCompensationAge: CFTimeInterval = 0.62
    // Source age remains a final safety bound. MediaPipe runs near 15 FPS and
    // can reject several uncertain closed-mouth samples in a row. Keep the
    // last real contour under current AR motion during that short gap, then
    // fade it before the final source-age cutoff instead of blinking on one or
    // two rejected callbacks. Accepted input itself may already be 0.25s old.
    private static let maxRealContourDisplayAge: CFTimeInterval = 0.82
    private static let contourFreshnessFadeStart: CFTimeInterval = 0.40
    private static let contourFreshnessFadeEnd: CFTimeInterval = 0.62
    private static let maxSurfaceCarrierHoldAge: CFTimeInterval = 0.50
    // A depth-only carrier is a temporary fallback, not a stable attachment.
    // Reacquire it frequently so stale surface depth cannot create visible
    // parallax lag while the head or camera moves.
    private static let maxDepthOnlySurfaceCarrierHoldAge: CFTimeInterval = 0.20
    // Stable barycentric triangle keys remain topologically valid for the same
    // hold window as their carrier. Cutting the indexed path at 0.18s forced a
    // fragile full reacquisition precisely while the carrier was recovering.
    private static let maxIndexedSurfaceSnapshotAge: CFTimeInterval = 0.50
    private static let maxAcceptedLandmarkResultAge: CFTimeInterval = 0.25
    private static let landmarkCallbackTimeout: CFTimeInterval = 0.30
    // Keep the AR pose for exactly the same hold window as the real contour so
    // compensation cannot switch off abruptly before the contour itself does.
    private static let maxCurrentMotionSampleAge: CFTimeInterval = 0.45
    private static let maxTextureRequestAgeWithTexture: CFTimeInterval = 0.30
    private static let maxInitialTextureRequestAge: CFTimeInterval = 0.85
    private static let missedDetectionsBeforeReset = 14
    private static let minTextureRenderInterval: CFTimeInterval = 0.033
    private static let minLowLatencyTextureRenderInterval: CFTimeInterval = 0.020
    // 512 keeps enough facial detail for the 478-point landmarker while
    // reducing A14 inference latency and GPU contention with SceneKit/Metal.
    // Lip texture resolution is independent and remains unchanged.
    private static let mediaPipeInputLongEdge: CGFloat = 512
    private static let fastLandmarkMotionDelta: CGFloat = 0.055
    private static let maxExpressionPredictionHorizon: CFTimeInterval = 0.045
    private static let maxExpressionPredictionFactor: CFTimeInterval = 1.0
    private static let maxExpressionPredictionPixels: CGFloat = 2.0
    // SceneKit presents custom geometry one display step after ARKit has
    // estimated the face pose on some devices. Predict only rigid screen
    // translation; expression, scale and rotation remain measured values.
    private static let maxRigidDisplayPredictionHorizon: CFTimeInterval = 0.020
    private static let maxRigidDisplayPredictionPixels: CGFloat = 3.5
    private static let neutralAmbientIntensity: CGFloat = 950
    private static let minLipLightingFactor: CGFloat = 0.45
    private static let maxLipLightingFactor: CGFloat = 1
    private static let lightingAdaptationTime: CFTimeInterval = 0.40
    private static let lightingHysteresis: CGFloat = 0.020
    private static let lipMeshVerticalAlignmentOffset: Float = 0.026
    private static let maxLipMeshVerticalAlignment: Float = 0.00135

    init(isFaceDetected: Binding<Bool>) {
        _isFaceDetected = isFaceDetected
        super.init()
        faceLandmarker = Self.makeFaceLandmarker(liveStreamDelegate: self)
    }

    func startSession() {
        guard ARFaceTrackingConfiguration.isSupported else {
            DispatchQueue.main.async { self.isFaceDetected = false }
            return
        }

        setSessionActive(true)
        resetLipTrackingAsync()
        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        // preferredFramesPerSecond controls SceneKit, but AR face anchors are
        // still paced by the selected camera format. Prefer the fastest
        // supported TrueDepth format so rigid lip attachment is updated at the
        // camera rate instead of being limited to the 30 FPS default.
        let supportedFormats = ARFaceTrackingConfiguration.supportedVideoFormats
        if let fastestFrameRate = supportedFormats.map(\.framesPerSecond).max(),
           let fastestFormat = supportedFormats
            .filter({ $0.framesPerSecond == fastestFrameRate })
            .max(by: {
                $0.imageResolution.width * $0.imageResolution.height <
                    $1.imageResolution.width * $1.imageResolution.height
            }) {
            configuration.videoFormat = fastestFormat
            LipDebugLog.throttled(
                "lip_camera_format",
                "lip_camera fps=\(fastestFormat.framesPerSecond) resolution=\(Int(fastestFormat.imageResolution.width))x\(Int(fastestFormat.imageResolution.height))"
            )
        }
        sceneView?.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopSession() {
        setSessionActive(false)
        resetLipTrackingAsync()
        requestRendererClear()
        setTrackedFaceAnchor(nil, isTracked: false)
        DispatchQueue.main.async { self.isFaceDetected = false }
    }

    func updateLipstick(color: LipstickColor,
                        opacity: Double,
                        finish: LipFinish,
                        density: LipstickDensity,
                        texture: LipstickTexture) {
        rendererStateLock.lock()
        pendingLipOpacity = opacity
        rendererStateLock.unlock()
        guard lipTextureRenderer.updateStyle(
            color: color,
            finish: finish,
            density: density,
            texture: texture
        ) else {
            return
        }
        invalidatePendingTextures()
    }

    func updateBlush(color: UIColor, opacity: Double) {
        rendererStateLock.lock()
        pendingBlushStyle = PendingBlushStyle(color: color, opacity: opacity)
        rendererStateLock.unlock()
    }

    func updateViewport(size: CGSize,
                        scale: CGFloat,
                        orientation: UIInterfaceOrientation) {
        guard size.width > 1, size.height > 1 else {
            return
        }

        viewportLock.lock()
        let sizeChanged = cachedViewportSize != size
        let resolvedOrientation: UIInterfaceOrientation
        if orientation == .unknown, sizeChanged {
            resolvedOrientation = .unknown
        } else if orientation == .unknown {
            resolvedOrientation = cachedInterfaceOrientation
        } else {
            resolvedOrientation = orientation
        }
        let geometryChanged = sizeChanged ||
            cachedInterfaceOrientation != resolvedOrientation
        let renderScaleChanged = cachedRenderScale != scale
        if geometryChanged {
            cachedViewportRevision &+= 1
        }
        cachedViewportSize = size
        cachedRenderScale = scale
        cachedInterfaceOrientation = resolvedOrientation
        viewportLock.unlock()

        if geometryChanged {
            resetLipTrackingAsync()
        } else if renderScaleChanged {
            invalidatePendingTextures()
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard let faceAnchor = anchor as? ARFaceAnchor else {
            return nil
        }

        if setTrackedFaceAnchor(faceAnchor.identifier, isTracked: true) {
            resetLipTrackingAsync()
        }
        let faceIsTracked = currentTrackedFaceAnchorIdentifier() == faceAnchor.identifier
        DispatchQueue.main.async { self.isFaceDetected = faceIsTracked }
        let faceNode = SCNNode()
        applyPendingRendererState()
        lipMeshRenderer.attach(to: faceNode)
        blushRenderer.attach(to: faceNode)
        return faceNode
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        let frameStart = CACurrentMediaTime()
        defer {
            arRendererFPS.tick(workMilliseconds: (CACurrentMediaTime() - frameStart) * 1000)
        }

        guard let faceAnchor = anchor as? ARFaceAnchor else {
            return
        }
        applyPendingRendererState()

        let trackingStateChanged = setTrackedFaceAnchor(
            faceAnchor.identifier,
            isTracked: faceAnchor.isTracked
        )
        if !faceAnchor.isTracked {
            if trackingStateChanged {
                resetAndRequestRendererClear()
                applyPendingRendererState()
            }
        } else {
            if trackingStateChanged {
                resetLipTrackingAsync()
            }
            // Metal applies the smoothed lighting factor before compositing.
            // Keep SceneKit neutral to avoid correcting the same texture twice;
            // the CPU fallback still needs the material multiplier.
            let sceneKitLightingFactor = lipTextureRenderer.supportsRealtimeCompositing ?
                1 : currentLipLightingFactor()
            lipMeshRenderer.updateLightingFactor(sceneKitLightingFactor)
            lipMeshRenderer.updateOccluder(faceAnchor: faceAnchor, renderer: renderer, faceNode: node)
            blushRenderer.attach(to: node)

            blushRenderer.render(faceAnchor: faceAnchor)

            let mouthFrame = localMouthFrame(from: faceAnchor)
            let renderMotionSample = makeRenderLipMotionSample(
                for: faceAnchor,
                renderer: renderer,
                faceNode: node
            )
            let meshState = currentLipMeshState(
                for: faceAnchor.identifier,
                renderMotionSample: renderMotionSample,
                faceGeometry: faceAnchor.geometry,
                renderer: renderer,
                faceNode: node
            )
            if let mouthFrame,
               let meshState {
                lipMeshRenderer.render(
                    contour: meshState.contour,
                    texture: meshState.texture,
                    faceGeometry: faceAnchor.geometry,
                    mouthFrame: mouthFrame,
                    renderer: renderer,
                    faceNode: node,
                    contourAge: meshState.contourAge,
                    motionDelta: meshState.motionDelta,
                    freshnessVisibility: meshState.freshnessVisibility
                )
            } else {
                let availability = currentLipMeshAvailability()
                LipDebugLog.throttled(
                    "lip_scene_skip",
                    interval: 0.6,
                    "lip_scene skip tracked=\(faceAnchor.isTracked) mouthFrame=\(mouthFrame != nil) meshState=\(meshState != nil) hasContour=\(availability.contour) hasTexture=\(availability.texture)"
                )
                lipMeshRenderer.renderUnavailable()
            }
        }

        let faceIsTracked = faceAnchor.isTracked &&
            currentTrackedFaceAnchorIdentifier() == faceAnchor.identifier
        DispatchQueue.main.async {
            self.isFaceDetected = faceIsTracked
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        guard let faceAnchor = anchor as? ARFaceAnchor,
              currentTrackedFaceAnchorIdentifier() == faceAnchor.identifier else {
            return
        }
        setTrackedFaceAnchor(faceAnchor.identifier, isTracked: false)
        resetAndRequestRendererClear()
        applyPendingRendererState()
        DispatchQueue.main.async {
            self.isFaceDetected = false
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        if case .normal = camera.trackingState {
            setCameraTrackingIsNormal(true)
            return
        }

        setCameraTrackingIsNormal(false)
        setTrackedFaceAnchor(nil, isTracked: false)
        resetAndRequestRendererClear()
        DispatchQueue.main.async {
            self.isFaceDetected = false
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let receivedAt = CACurrentMediaTime()
        updateLipLightingEstimate(from: frame)

        guard currentSessionIsActive(),
              case .normal = frame.camera.trackingState else {
            return
        }
        let sourceAgeAtReceipt = receivedAt - frame.timestamp
        guard sourceAgeAtReceipt.isFinite,
              sourceAgeAtReceipt >= 0,
              sourceAgeAtReceipt <= Self.maxAcceptedLandmarkResultAge else {
            LipDebugLog.throttled(
                "lip_sync_stale_source",
                interval: 0.6,
                "lip_sync reject stage=source sourceAgeMs=\(String(format: "%.1f", sourceAgeAtReceipt * 1000))"
            )
            return
        }
        setCameraTrackingIsNormal(true)

        let viewport = currentViewport()
        let viewportSize = viewport.size
        let activeAnchorIdentifier = currentTrackedFaceAnchorIdentifier()
        let trackedFaceAnchors = frame.anchors
            .compactMap { $0 as? ARFaceAnchor }
            .filter(\.isTracked)
        let faceAnchor = activeAnchorIdentifier.flatMap { activeIdentifier in
            trackedFaceAnchors.first(where: { $0.identifier == activeIdentifier })
        } ?? trackedFaceAnchors.first
        guard viewportSize.width > 1,
              viewportSize.height > 1,
              viewport.orientation != .unknown,
              let faceAnchor,
              let motionReference = projectedLipMotionPose(
                from: faceAnchor,
                camera: frame.camera,
                orientation: viewport.orientation,
                viewportSize: viewportSize
              ) else {
            return
        }

        if setTrackedFaceAnchor(faceAnchor.identifier, isTracked: true) {
            resetLipTrackingAsync()
        }

        let timestampInMilliseconds = Int(frame.timestamp * 1000)
        let displayTransform = frame.displayTransform(
            for: viewport.orientation,
            viewportSize: viewportSize
        )
        let faceSurfaceProjectionInput = FaceSurfaceProjectionInput(
            geometry: faceAnchor.geometry,
            anchorTransform: faceAnchor.transform,
            camera: frame.camera,
            orientation: viewport.orientation,
            viewportSize: viewportSize,
            anchorIdentifier: faceAnchor.identifier
        )
        let trackingEpoch = currentLandmarkTrackingEpoch()
        let context = FrameContext(
            pixelBuffer: frame.capturedImage,
            frameTimestamp: frame.timestamp,
            timestampInMilliseconds: timestampInMilliseconds,
            receivedAt: receivedAt,
            viewportSize: viewportSize,
            renderScale: viewport.scale,
            interfaceOrientation: viewport.orientation,
            capturedImageToViewportTransform: displayTransform,
            faceSurfaceProjectionInput: faceSurfaceProjectionInput,
            viewportRevision: viewport.revision,
            trackingEpoch: trackingEpoch,
            anchorIdentifier: faceAnchor.identifier,
            motionReference: motionReference
        )

        guard currentSessionIsActive(),
              currentTrackedFaceAnchorIdentifier() == faceAnchor.identifier,
              currentViewport().revision == viewport.revision,
              currentLandmarkTrackingEpoch() == trackingEpoch else {
            return
        }
        setLatestLipMotionSample(
            LipMotionSample(
                pose: motionReference,
                sampledAt: frame.timestamp,
                frameTimestamp: frame.timestamp,
                trackingEpoch: trackingEpoch,
                anchorIdentifier: faceAnchor.identifier,
                viewportRevision: viewport.revision
            )
        )
        offerLandmarkFrame(context)
    }

    private func updateLipLightingEstimate(from frame: ARFrame) {
        let ambientIntensity = frame.lightEstimate?.ambientIntensity
        let targetFactor = targetLipLightingFactor(from: ambientIntensity)
        let now = CACurrentMediaTime()

        lightingLock.lock()
        let elapsed = lastLightingUpdateTime > 0 ? max(0, now - lastLightingUpdateTime) : 1.0 / 60.0
        let alpha = 1 - exp(-elapsed / Self.lightingAdaptationTime)
        let targetDelta = targetFactor - smoothedLipLightingFactor
        let stabilizedTarget = abs(targetDelta) < Self.lightingHysteresis ? smoothedLipLightingFactor : targetFactor
        smoothedLipLightingFactor += (stabilizedTarget - smoothedLipLightingFactor) * CGFloat(alpha)
        smoothedLipLightingFactor = max(Self.minLipLightingFactor, min(smoothedLipLightingFactor, Self.maxLipLightingFactor))
        let smoothedFactor = smoothedLipLightingFactor
        lastLightingUpdateTime = now
        lightingLock.unlock()

        LipDebugLog.throttled(
            "lip_lighting",
            interval: 1.0,
            "lip_lighting ambient=\(String(format: "%.1f", ambientIntensity ?? -1)) target=\(String(format: "%.3f", targetFactor)) smoothed=\(String(format: "%.3f", smoothedFactor))"
        )
    }

    private func targetLipLightingFactor(from ambientIntensity: CGFloat?) -> CGFloat {
        guard let ambientIntensity,
              ambientIntensity.isFinite,
              ambientIntensity > 0 else {
            return 1
        }

        let normalized = max(0, min(ambientIntensity / Self.neutralAmbientIntensity, 1))
        guard normalized < 0.90 else {
            return 1
        }

        let lowLightCurve = pow(normalized / 0.90, 0.72)
        let factor = Self.minLipLightingFactor + lowLightCurve * (1 - Self.minLipLightingFactor)
        return max(Self.minLipLightingFactor, min(factor, Self.maxLipLightingFactor))
    }

    private func currentLipLightingFactor() -> CGFloat {
        lightingLock.lock()
        let factor = smoothedLipLightingFactor
        lightingLock.unlock()
        return factor
    }

    private static func makeFaceLandmarker(liveStreamDelegate: FaceLandmarkerLiveStreamDelegate) -> FaceLandmarker? {
        guard let modelPath = Bundle.main.path(forResource: "face_landmarker", ofType: "task") else {
            return nil
        }

        func makeLandmarker(delegate: Delegate) -> FaceLandmarker? {
            let options = FaceLandmarkerOptions()
            let baseOptions = BaseOptions()
            baseOptions.modelAssetPath = modelPath
            baseOptions.delegate = delegate
            options.baseOptions = baseOptions
            options.runningMode = .liveStream
            options.faceLandmarkerLiveStreamDelegate = liveStreamDelegate
            options.numFaces = 1
            options.minFaceDetectionConfidence = 0.48
            options.minFacePresenceConfidence = 0.48
            options.minTrackingConfidence = 0.48
            options.outputFaceBlendshapes = false
            options.outputFacialTransformationMatrixes = true
            return try? FaceLandmarker(options: options)
        }

        return makeLandmarker(delegate: .GPU) ?? makeLandmarker(delegate: .CPU)
    }

    private func initializeFaceLandmarkerIfNeeded() {
        guard faceLandmarker == nil else {
            return
        }
        faceLandmarker = Self.makeFaceLandmarker(liveStreamDelegate: self)
    }

    private struct LipDetection {
        let contour: LipContour
        let pixelBuffer: CVPixelBuffer
        let imageSize: CGSize
    }

    private struct LipSurfaceBindingCandidate {
        let binding: LipSurfaceBinding
        let cameraDepth: Float
        let mouthSideScore: Float
    }

    private struct FrameContext {
        let pixelBuffer: CVPixelBuffer
        let frameTimestamp: CFTimeInterval
        let timestampInMilliseconds: Int
        let receivedAt: CFTimeInterval
        let viewportSize: CGSize
        let renderScale: CGFloat
        let interfaceOrientation: UIInterfaceOrientation
        let capturedImageToViewportTransform: CGAffineTransform
        let faceSurfaceProjectionInput: FaceSurfaceProjectionInput
        let viewportRevision: Int
        let trackingEpoch: Int
        let anchorIdentifier: UUID
        let motionReference: LipMotionPose
    }

    private struct LipMotionSample {
        let pose: LipMotionPose
        let sampledAt: CFTimeInterval
        let frameTimestamp: CFTimeInterval
        let trackingEpoch: Int
        let anchorIdentifier: UUID
        let viewportRevision: Int
    }

    private struct RigidLipReference {
        let anchorIdentifier: UUID
        let localCenter: SIMD3<Float>
        let localLeft: SIMD3<Float>
        let localRight: SIMD3<Float>
    }

    private struct PreparedMediaPipeInput {
        let pixelBuffer: CVPixelBuffer
        let inputToCapturedImageTransform: CGAffineTransform
    }

    private struct PendingLiveFrame {
        let pixelBuffer: CVPixelBuffer
        let imageSize: CGSize
        let viewportSize: CGSize
        let renderScale: CGFloat
        let generation: Int
        let frameTimestamp: CFTimeInterval
        let timestampInMilliseconds: Int
        let receivedAt: CFTimeInterval
        let submittedAt: CFTimeInterval
        let inputToCapturedImageTransform: CGAffineTransform
        let capturedImageToViewportTransform: CGAffineTransform
        let faceSurfaceSnapshot: FaceSurfaceSnapshot
        let viewportRevision: Int
        let trackingEpoch: Int
        let anchorIdentifier: UUID
        let motionReference: LipMotionPose
        let engineIdentifier: ObjectIdentifier
    }

    private struct LipTextureRequest {
        let contour: LipContour
        let pixelBuffer: CVPixelBuffer
        let imageSize: CGSize
        let viewportSize: CGSize
        let renderScale: CGFloat
        let lightingFactor: CGFloat
        let excludesInnerMouth: Bool
        let lowLatency: Bool
        let generation: Int
        let requestID: Int
        let motionReference: LipMotionPose
        let motionDelta: CGFloat
        let sourceCaptureTime: CFTimeInterval
        let trackingEpoch: Int
        let anchorIdentifier: UUID
        let viewportRevision: Int
        let createdAt: CFTimeInterval
    }

    private struct LipMeshState {
        let contour: LipContour
        let texture: LipTexture
        let contourAge: CFTimeInterval
        let motionDelta: CGFloat
        let freshnessVisibility: CGFloat
    }

    private struct AcceptedRealContourSample {
        let contour: LipContour
        let capturedAt: CFTimeInterval
        let motionReference: LipMotionPose
        let trackingEpoch: Int
        let anchorIdentifier: UUID
        let viewportRevision: Int
        let predictionEligible: Bool
    }

    private func offerLandmarkFrame(_ context: FrameContext) {
        guard currentTrackedFaceAnchorIdentifier() == context.anchorIdentifier else {
            return
        }

        var contextToSubmit: FrameContext?
        detectionLock.lock()
        guard context.trackingEpoch == landmarkTrackingEpoch,
              context.timestampInMilliseconds > lastLandmarkTimestampInMilliseconds else {
            detectionLock.unlock()
            return
        }

        if landmarkReadyEpoch != context.trackingEpoch {
            if latestPendingFrameContext.map({
                context.timestampInMilliseconds > $0.timestampInMilliseconds
            }) ?? true {
                latestPendingFrameContext = context
            }
            detectionLock.unlock()
            return
        }

        if isDetectingLandmarks {
            if latestPendingFrameContext.map({
                context.timestampInMilliseconds > $0.timestampInMilliseconds
            }) ?? true {
                latestPendingFrameContext = context
            }
            detectionLock.unlock()
            return
        }

        let now = CACurrentMediaTime()
        let motionDelta = motionDelta(
            from: lastLandmarkSubmitPose,
            to: context.motionReference
        )
        let minimumInterval = minimumLandmarkSubmitInterval(motionDelta: motionDelta)
        if lastLandmarkSubmitTime > 0,
           now - lastLandmarkSubmitTime < minimumInterval {
            latestPendingFrameContext = context
            detectionLock.unlock()
            return
        }

        isDetectingLandmarks = true
        activeLandmarkTimestampInMilliseconds = context.timestampInMilliseconds
        latestPendingFrameContext = nil
        lastLandmarkTimestampInMilliseconds = context.timestampInMilliseconds
        lastLandmarkSubmitTime = now
        lastLandmarkSubmitPose = context.motionReference
        contextToSubmit = context
        detectionLock.unlock()

        guard let contextToSubmit else {
            return
        }
        landmarkQueue.async { [weak self] in
            self?.submitLiveStreamFrame(contextToSubmit)
        }
    }

    private func submitLiveStreamFrame(_ context: FrameContext) {
        guard isActiveLandmarkContext(context),
              currentTrackedFaceAnchorIdentifier() == context.anchorIdentifier,
              currentViewport().revision == context.viewportRevision else {
            completeLandmarkDetection(
                timestampInMilliseconds: context.timestampInMilliseconds,
                trackingEpoch: context.trackingEpoch
            )
            return
        }

        let preprocessingStartedAt = CACurrentMediaTime()
        let sourceAgeBeforePreprocessing = preprocessingStartedAt - context.frameTimestamp
        guard sourceAgeBeforePreprocessing.isFinite,
              sourceAgeBeforePreprocessing >= 0,
              sourceAgeBeforePreprocessing <= Self.maxAcceptedLandmarkResultAge,
              let preparedInput = makeMediaPipeInputBuffer(
                from: context.pixelBuffer,
                orientation: context.interfaceOrientation
              ) else {
            LipDebugLog.throttled(
                "lip_sync_preprocess_reject",
                interval: 0.6,
                "lip_sync reject stage=preprocess timestamp=\(context.timestampInMilliseconds) sourceAge=\(String(format: "%.3f", sourceAgeBeforePreprocessing))"
            )
            completeLandmarkDetection(
                timestampInMilliseconds: context.timestampInMilliseconds,
                trackingEpoch: context.trackingEpoch
            )
            return
        }

        let sourceAgeAfterPreprocessing = CACurrentMediaTime() - context.frameTimestamp
        guard isActiveLandmarkContext(context),
              currentTrackedFaceAnchorIdentifier() == context.anchorIdentifier,
              currentViewport().revision == context.viewportRevision,
              sourceAgeAfterPreprocessing.isFinite,
              sourceAgeAfterPreprocessing >= 0,
              sourceAgeAfterPreprocessing <= Self.maxAcceptedLandmarkResultAge else {
            LipDebugLog.throttled(
                "lip_sync_postprocess_reject",
                interval: 0.6,
                "lip_sync reject stage=postprocess timestamp=\(context.timestampInMilliseconds) sourceAge=\(String(format: "%.3f", sourceAgeAfterPreprocessing))"
            )
            completeLandmarkDetection(
                timestampInMilliseconds: context.timestampInMilliseconds,
                trackingEpoch: context.trackingEpoch
            )
            return
        }

        if faceLandmarker == nil {
            initializeFaceLandmarkerIfNeeded()
        }
        guard let faceLandmarker else {
            handleMissedLipDetection()
            completeLandmarkDetection(
                timestampInMilliseconds: context.timestampInMilliseconds,
                trackingEpoch: context.trackingEpoch
            )
            return
        }

        let surfaceProjectionStartedAt = CACurrentMediaTime()
        let preferredSurfaceTriangles = preferredFaceSurfaceTriangleKeys(
            for: context.faceSurfaceProjectionInput
        )
        guard let faceSurfaceSnapshot = makeFaceSurfaceSnapshot(
            from: context.faceSurfaceProjectionInput,
            preferredTriangleKeys: preferredSurfaceTriangles
        ) else {
            faceSurfaceTopologyCache = nil
            shouldExpandFaceSurfaceTopology = true
            LipDebugLog.throttled(
                "lip_surface_snapshot_reject",
                interval: 0.6,
                "lip_surface reject stage=snapshot timestamp=\(context.timestampInMilliseconds)"
            )
            handleMissedLipDetection()
            completeLandmarkDetection(
                timestampInMilliseconds: context.timestampInMilliseconds,
                trackingEpoch: context.trackingEpoch
            )
            return
        }

        let surfaceProjectionFinishedAt = CACurrentMediaTime()
        let sourceAgeAfterSurfaceProjection =
            surfaceProjectionFinishedAt - context.frameTimestamp
        let preprocessingMilliseconds =
            (sourceAgeAfterPreprocessing - sourceAgeBeforePreprocessing) * 1000
        let surfaceProjectionMilliseconds =
            (surfaceProjectionFinishedAt - surfaceProjectionStartedAt) * 1000
        guard isActiveLandmarkContext(context),
              currentTrackedFaceAnchorIdentifier() == context.anchorIdentifier,
              currentViewport().revision == context.viewportRevision,
              sourceAgeAfterSurfaceProjection.isFinite,
              sourceAgeAfterSurfaceProjection >= 0,
              sourceAgeAfterSurfaceProjection <= Self.maxAcceptedLandmarkResultAge else {
            LipDebugLog.throttled(
                "lip_surface_snapshot_stale",
                interval: 0.6,
                "lip_surface reject stage=snapshot_stale timestamp=\(context.timestampInMilliseconds) sourceAge=\(String(format: "%.3f", sourceAgeAfterSurfaceProjection))"
            )
            completeLandmarkDetection(
                timestampInMilliseconds: context.timestampInMilliseconds,
                trackingEpoch: context.trackingEpoch
            )
            return
        }

        let imageSize = CGSize(
            width: CVPixelBufferGetWidth(preparedInput.pixelBuffer),
            height: CVPixelBufferGetHeight(preparedInput.pixelBuffer)
        )
        let engineIdentifier = ObjectIdentifier(faceLandmarker)
        let submittedAt = CACurrentMediaTime()
        activeLiveFrame = PendingLiveFrame(
            pixelBuffer: preparedInput.pixelBuffer,
            imageSize: imageSize,
            viewportSize: context.viewportSize,
            renderScale: context.renderScale,
            generation: currentTextureGeneration(),
            frameTimestamp: context.frameTimestamp,
            timestampInMilliseconds: context.timestampInMilliseconds,
            receivedAt: context.receivedAt,
            submittedAt: submittedAt,
            inputToCapturedImageTransform: preparedInput.inputToCapturedImageTransform,
            capturedImageToViewportTransform: context.capturedImageToViewportTransform,
            faceSurfaceSnapshot: faceSurfaceSnapshot,
            viewportRevision: context.viewportRevision,
            trackingEpoch: context.trackingEpoch,
            anchorIdentifier: context.anchorIdentifier,
            motionReference: context.motionReference,
            engineIdentifier: engineIdentifier
        )

        do {
            let image = try MPImage(pixelBuffer: preparedInput.pixelBuffer, orientation: .up)
            // MediaPipe's Objective-C BOOL + NSError API imports into Swift as
            // throwing Void, so a rejected submission is handled by catch.
            try faceLandmarker.detectAsync(
                image: image,
                timestampInMilliseconds: context.timestampInMilliseconds
            )
        } catch {
            activeLiveFrame = nil
            LipDebugLog.throttled(
                "lip_sync_submit_error",
                interval: 0.6,
                "lip_sync reject stage=submit timestamp=\(context.timestampInMilliseconds) error=\(error.localizedDescription)"
            )
            handleMissedLipDetection()
            completeLandmarkDetection(
                timestampInMilliseconds: context.timestampInMilliseconds,
                trackingEpoch: context.trackingEpoch
            )
            return
        }

        LipDebugLog.throttled(
            "lip_sync_submit",
            interval: 0.6,
            "lip_sync submit timestamp=\(context.timestampInMilliseconds) captureToReceiptMs=\(String(format: "%.1f", (context.receivedAt - context.frameTimestamp) * 1000)) captureToSubmitMs=\(String(format: "%.1f", (submittedAt - context.frameTimestamp) * 1000)) preprocessMs=\(String(format: "%.1f", preprocessingMilliseconds)) surfaceMs=\(String(format: "%.1f", surfaceProjectionMilliseconds)) input=\(Int(imageSize.width))x\(Int(imageSize.height)) surfaceTriangles=\(faceSurfaceSnapshot.triangles.count) indexed=\(preferredSurfaceTriangles != nil) epoch=\(context.trackingEpoch) viewport=\(context.viewportRevision)"
        )
        scheduleLandmarkCallbackWatchdog(
            timestampInMilliseconds: context.timestampInMilliseconds,
            trackingEpoch: context.trackingEpoch,
            engineIdentifier: engineIdentifier
        )
    }

    private func scheduleLandmarkCallbackWatchdog(timestampInMilliseconds: Int,
                                                  trackingEpoch: Int,
                                                  engineIdentifier: ObjectIdentifier) {
        landmarkQueue.asyncAfter(deadline: .now() + Self.landmarkCallbackTimeout) { [weak self] in
            guard let self,
                  let frame = self.activeLiveFrame,
                  frame.timestampInMilliseconds == timestampInMilliseconds,
                  frame.trackingEpoch == trackingEpoch,
                  frame.engineIdentifier == engineIdentifier else {
                return
            }

            self.activeLiveFrame = nil
            LipDebugLog.throttled(
                "lip_sync_timeout",
                interval: 0.6,
                "lip_sync reject stage=callback timestamp=\(timestampInMilliseconds) reason=timeout sourceAge=\(String(format: "%.3f", CACurrentMediaTime() - frame.frameTimestamp))"
            )
            self.completeLandmarkDetection(
                timestampInMilliseconds: timestampInMilliseconds,
                trackingEpoch: trackingEpoch
            )
        }
    }

    func faceLandmarker(_ faceLandmarker: FaceLandmarker,
                        didFinishDetection result: FaceLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        let engineIdentifier = ObjectIdentifier(faceLandmarker)
        landmarkQueue.async { [weak self] in
            self?.handleLiveStreamResult(
                result,
                timestampInMilliseconds: timestampInMilliseconds,
                engineIdentifier: engineIdentifier,
                error: error
            )
        }
    }

    private func handleLiveStreamResult(_ result: FaceLandmarkerResult?,
                                        timestampInMilliseconds: Int,
                                        engineIdentifier: ObjectIdentifier,
                                        error: Error?) {
        let resultStart = CACurrentMediaTime()
        defer {
            mediaPipeFPS.tick(workMilliseconds: (CACurrentMediaTime() - resultStart) * 1000)
        }

        guard let frame = activeLiveFrame,
              frame.timestampInMilliseconds == timestampInMilliseconds,
              frame.engineIdentifier == engineIdentifier else {
            LipDebugLog.throttled(
                "lip_sync_orphan_callback",
                interval: 0.6,
                "lip_sync ignore stage=callback timestamp=\(timestampInMilliseconds) reason=orphan"
            )
            return
        }
        activeLiveFrame = nil
        defer {
            completeLandmarkDetection(
                timestampInMilliseconds: timestampInMilliseconds,
                trackingEpoch: frame.trackingEpoch
            )
        }

        let callbackAt = CACurrentMediaTime()
        let captureToCallback = callbackAt - frame.frameTimestamp
        guard frame.trackingEpoch == currentLandmarkTrackingEpoch(),
              currentTrackedFaceAnchorIdentifier() == frame.anchorIdentifier else {
            LipDebugLog.throttled(
                "lip_sync_old_epoch",
                interval: 0.6,
                "lip_sync reject stage=callback timestamp=\(timestampInMilliseconds) reason=old_epoch"
            )
            return
        }
        guard currentViewport().revision == frame.viewportRevision else {
            LipDebugLog.throttled(
                "lip_sync_old_viewport",
                interval: 0.6,
                "lip_sync reject stage=callback timestamp=\(timestampInMilliseconds) reason=old_viewport"
            )
            return
        }
        guard captureToCallback.isFinite,
              captureToCallback >= 0,
              captureToCallback <= Self.maxAcceptedLandmarkResultAge else {
            LipDebugLog.throttled(
                "lip_sync_stale_callback",
                interval: 0.6,
                "lip_sync reject stage=callback timestamp=\(timestampInMilliseconds) reason=stale captureToCallbackMs=\(String(format: "%.1f", captureToCallback * 1000))"
            )
            return
        }
        guard timestampInMilliseconds > lastAcceptedLandmarkTimestampInMilliseconds else {
            LipDebugLog.throttled(
                "lip_sync_out_of_order",
                interval: 0.6,
                "lip_sync reject stage=callback timestamp=\(timestampInMilliseconds) reason=out_of_order last=\(lastAcceptedLandmarkTimestampInMilliseconds)"
            )
            return
        }

        logMediaPipeLandmarkCount(result)

        if let error {
            LipDebugLog.throttled(
                "lip_detection_error",
                "lip_detection reject reason=mediapipe_error error=\(error.localizedDescription)"
            )
            handleMissedLipDetection()
            return
        }
        guard currentTrackedFaceAnchor() else {
            LipDebugLog.throttled(
                "lip_detection_no_ar_tracking",
                "lip_detection reject reason=no_ar_tracking timestamp=\(timestampInMilliseconds)"
            )
            handleMissedLipDetection()
            return
        }
        guard let result else {
            LipDebugLog.throttled(
                "lip_detection_no_result",
                "lip_detection reject reason=no_result timestamp=\(timestampInMilliseconds)"
            )
            handleMissedLipDetection()
            return
        }
        guard let landmarks = result.faceLandmarks.first else {
            LipDebugLog.throttled(
                "lip_detection_no_landmarks",
                "lip_detection reject reason=no_landmarks faces=\(result.faceLandmarks.count) timestamp=\(timestampInMilliseconds)"
            )
            handleMissedLipDetection()
            return
        }
        guard isUsableFaceTransform(result.facialTransformationMatrixes.first) else {
            LipDebugLog.throttled(
                "lip_detection_bad_transform",
                "lip_detection reject reason=bad_face_geometry_transform matrices=\(result.facialTransformationMatrixes.count)"
            )
            handleMissedLipDetection()
            return
        }

        let faceGeometryPose = FaceGeometryPose(transform: result.facialTransformationMatrixes.first)
        guard let detection = lipDetection(
            from: landmarks,
            frame: frame,
            faceGeometryPose: faceGeometryPose
        ) else {
            LipDebugLog.throttled(
                "lip_detection_nil",
                "lip_detection reject reason=lip_detection_nil landmarks=\(landmarks.count) viewport=\(Int(frame.viewportSize.width))x\(Int(frame.viewportSize.height)) image=\(Int(frame.imageSize.width))x\(Int(frame.imageSize.height))"
            )
            handleMissedLipDetection()
            return
        }
        guard detection.contour.isUsable(in: frame.viewportSize) else {
            LipDebugLog.throttled(
                "lip_detection_unusable",
                "lip_detection reject reason=unusable_contour \(lipContourDebugSummary(detection.contour)) viewport=\(Int(frame.viewportSize.width))x\(Int(frame.viewportSize.height))"
            )
            handleMissedLipDetection()
            return
        }
        guard isReliableLipUpdate(
            detection.contour,
            viewportSize: frame.viewportSize,
            motionReference: frame.motionReference
        ) else {
            LipDebugLog.throttled(
                "lip_detection_unreliable",
                "lip_detection reject reason=unreliable_update \(lipContourDebugSummary(detection.contour))"
            )
            handleMissedLipDetection()
            return
        }

        let lowLatencyShapeChange = shouldUseRawContour(
            previous: smoothedLipContour,
            current: detection.contour,
            previousMotionPose: lastAcceptedMotionPose,
            currentMotionPose: frame.motionReference
        )
        let smoothedContour: LipContour
        if lowLatencyShapeChange {
            contourStabilityConfidence = 0
            contourIsStationary = false
            smoothedContour = detection.contour
        } else {
            smoothedContour = adaptivelySmoothed(
                detection.contour,
                motionReference: frame.motionReference
            )
        }
        let heldSurfaceCarrier: (
            contour: LipContour,
            createdAt: CFTimeInterval,
            age: CFTimeInterval
        )? = {
            guard let previous = smoothedLipContour,
                  hasCompleteSurfaceCarrier(previous),
                  previous.surfaceVertexCount == frame.faceSurfaceSnapshot.vertexCount,
                  previous.surfaceTriangleIndexCount ==
                    frame.faceSurfaceSnapshot.triangleIndexCount,
                  previous.surfaceTopologySignature ==
                    frame.faceSurfaceSnapshot.topologySignature,
                  let createdAt = previous.surfaceCarrierCreatedAt else {
                return nil
            }
            let age = CACurrentMediaTime() - createdAt
            let maximumHoldAge = previous.surfaceCarrierSupportsDeformation ?
                Self.maxSurfaceCarrierHoldAge :
                Self.maxDepthOnlySurfaceCarrierHoldAge
            guard age.isFinite,
                  age >= 0,
                  age <= maximumHoldAge else {
                return nil
            }
            // Barycentric bindings point into the current AR face topology and
            // are safe for a short retry window. Expiry deliberately falls
            // through to the exhaustive current-surface reacquisition below.
            return (previous, createdAt, age)
        }()

        var stableContour: LipContour
        if let heldSurfaceCarrier,
           let refreshedBindings = refreshedDepthBindings(
               from: heldSurfaceCarrier.contour,
               for: detection.contour,
               snapshot: frame.faceSurfaceSnapshot
           ) {
            // Keep the same stable AR triangles whenever their current
            // reprojection remains valid. The indexed lookup makes this O(80)
            // instead of searching hundreds of triangles for every landmark.
            var fastBoundContour = smoothedContour
            fastBoundContour.surfaceBindingsByIndex = refreshedBindings
            fastBoundContour.surfaceVertexCount =
                heldSurfaceCarrier.contour.surfaceVertexCount
            fastBoundContour.surfaceTriangleIndexCount =
                heldSurfaceCarrier.contour.surfaceTriangleIndexCount
            fastBoundContour.surfaceTopologySignature =
                heldSurfaceCarrier.contour.surfaceTopologySignature
            fastBoundContour.surfaceCarrierCreatedAt = frame.frameTimestamp
            fastBoundContour.surfaceCarrierSupportsDeformation = true
            stableContour = fastBoundContour
            LipDebugLog.throttled(
                "lip_surface_reuse_depth_carrier",
                interval: 0.4,
                "lip_surface reuse carrier=indexed_delta_warp shape=mediapipe timestamp=\(timestampInMilliseconds)"
            )
        } else if let heldSurfaceCarrier {
            // Publishing fresh MediaPipe X/Y must not wait for an exhaustive
            // carrier rebuild. The previous barycentric topology is used only
            // for depth, while visible X/Y remains the current detector shape.
            // Full reacquisition runs after this bounded carrier window expires.
            var freshContourWithHeldDepth = smoothedContour
            freshContourWithHeldDepth.surfaceBindingsByIndex =
                heldSurfaceCarrier.contour.surfaceBindingsByIndex
            freshContourWithHeldDepth.surfaceVertexCount =
                heldSurfaceCarrier.contour.surfaceVertexCount
            freshContourWithHeldDepth.surfaceTriangleIndexCount =
                heldSurfaceCarrier.contour.surfaceTriangleIndexCount
            freshContourWithHeldDepth.surfaceTopologySignature =
                heldSurfaceCarrier.contour.surfaceTopologySignature
            freshContourWithHeldDepth.surfaceCarrierCreatedAt =
                heldSurfaceCarrier.createdAt
            freshContourWithHeldDepth.surfaceCarrierSupportsDeformation = false
            stableContour = freshContourWithHeldDepth
            LipDebugLog.throttled(
                "lip_surface_fast_publish",
                interval: 0.4,
                "lip_surface publish=fresh_mediapipe carrier=held_depth rebind=deferred carrierAgeMs=\(String(format: "%.1f", heldSurfaceCarrier.age * 1000)) timestamp=\(timestampInMilliseconds)"
            )
        } else if let currentDepthContour = contourBoundToFaceSurface(
            detection.contour,
            publishing: smoothedContour,
            snapshot: frame.faceSurfaceSnapshot,
            carrierCreatedAt: frame.frameTimestamp
        ) {
            // Initial acquisition and bounded carrier expiry are the only
            // synchronous exhaustive searches on the landmark queue.
            stableContour = currentDepthContour
        } else {
            missedDetectionCount = 0
            faceSurfaceTopologyCache = nil
            shouldExpandFaceSurfaceTopology = true
            LipDebugLog.throttled(
                "lip_surface_hold_real_contour",
                interval: 0.4,
                "lip_surface hold_previous shape=mediapipe reason=no_current_depth timestamp=\(timestampInMilliseconds)"
            )
            // Keep the last complete MediaPipe shape/depth pair untouched. Its
            // capture-time age gate hides it after a short hold; never publish a
            // generated replacement.
            return
        }

        let detectedInnerOpening = stableContour.innerOpeningRatio
        stableContour.renderedInnerOpeningRatio =
            frame.motionReference.isConfidentlyClosedMouth ?
                0 : max(
                    detectedInnerOpening,
                    frame.motionReference.opening * 0.85
                )
        LipDebugLog.throttled(
            "lip_aperture_fusion",
            interval: 0.5,
            "lip_aperture raw=\(debugFloat(stableContour.innerOpeningRatio)) rendered=\(debugFloat(stableContour.effectiveInnerOpeningRatio)) arOpening=\(debugFloat(frame.motionReference.opening)) jawOpen=\(debugFloat(frame.motionReference.jawOpen)) smile=\(debugFloat(frame.motionReference.smile)) closed=\(frame.motionReference.isConfidentlyClosedMouth)"
        )

        missedDetectionCount = 0
        let motionReference = frame.motionReference
        let textureMotionDelta = lastAcceptedMotionPose.map {
            motionDelta(from: $0, to: motionReference)
        } ?? 0

        lastAcceptedLandmarkTimestampInMilliseconds = timestampInMilliseconds
        smoothedLipContour = stableContour
        smoothedLipPose = stableContour.pose
        lastAcceptedMotionPose = motionReference
        setLatestMeshContour(
            stableContour,
            motionReference: motionReference,
            capturedAt: frame.frameTimestamp,
            anchorIdentifier: frame.anchorIdentifier,
            viewportRevision: frame.viewportRevision,
            trackingEpoch: frame.trackingEpoch,
            predictionEligible: true
        )
        markLipShapeAccepted(capturedAt: frame.frameTimestamp)
        let detectionAcceptedAt = CACurrentMediaTime()
        LipDebugLog.throttled(
            "lip_detection_accept",
            interval: 0.6,
            "lip_detection accept lowLatency=\(lowLatencyShapeChange) captureToAcceptMs=\(String(format: "%.1f", (detectionAcceptedAt - frame.frameTimestamp) * 1000)) submitToAcceptMs=\(String(format: "%.1f", (detectionAcceptedAt - frame.submittedAt) * 1000)) callbackToAcceptMs=\(String(format: "%.1f", (detectionAcceptedAt - callbackAt) * 1000)) \(lipContourDebugSummary(stableContour))"
        )

        let textureNow = detectionAcceptedAt
        if lipTextureRenderer.supportsRealtimeCompositing {
            // The GPU compositor is bounded to a tiny 128x64 target, so the
            // sampled lip relief can follow every accepted MediaPipe keyframe.
            guard shouldSubmitTextureRender(
                lowLatency: true,
                now: textureNow
            ) else {
                return
            }
        } else {
            // Compatibility fallback: avoid running the legacy per-pixel CPU
            // compositor continuously because it competes with tracking.
            let textureMouthIsOpen =
                stableContour.effectiveInnerOpeningRatio >= 0.065
            guard shouldSubmitStableTextureRender(
                mouthOpen: textureMouthIsOpen,
                hasTexture: currentLipMeshAvailability().texture,
                now: textureNow
            ) else {
                return
            }
        }

        let request = LipTextureRequest(
            contour: stableContour,
            pixelBuffer: detection.pixelBuffer,
            imageSize: detection.imageSize,
            viewportSize: frame.viewportSize,
            renderScale: frame.renderScale,
            lightingFactor: currentLipLightingFactor(),
            excludesInnerMouth: true,
            lowLatency: true,
            generation: currentTextureGeneration(),
            requestID: reserveTextureRequestID(),
            motionReference: motionReference,
            motionDelta: textureMotionDelta,
            sourceCaptureTime: frame.frameTimestamp,
            trackingEpoch: frame.trackingEpoch,
            anchorIdentifier: frame.anchorIdentifier,
            viewportRevision: frame.viewportRevision,
            createdAt: CACurrentMediaTime()
        )
        enqueueTextureRender(request)
    }

    private func logMediaPipeLandmarkCount(_ result: FaceLandmarkerResult?) {
        let faceCount = result?.faceLandmarks.count ?? 0
        let landmarkCount = result?.faceLandmarks.first?.count ?? 0
        guard lastLoggedMediaPipeLandmarkCount != landmarkCount else {
            return
        }

        lastLoggedMediaPipeLandmarkCount = landmarkCount
        print(
            "mediapipe_landmarks faces=\(faceCount) points=\(landmarkCount) attention_or_v2=\(landmarkCount >= 478)"
        )
    }

    private func lipContourDebugSummary(_ contour: LipContour) -> String {
        let boundsText: String
        if let bounds = contour.bounds {
            boundsText = "bounds=(x:\(debugFloat(bounds.minX)) y:\(debugFloat(bounds.minY)) w:\(debugFloat(bounds.width)) h:\(debugFloat(bounds.height)))"
        } else {
            boundsText = "bounds=nil"
        }

        let poseText: String
        if let pose = contour.pose {
            poseText = "pose=(c:\(debugFloat(pose.center.x)),\(debugFloat(pose.center.y)) w:\(debugFloat(pose.width)) a:\(debugFloat(pose.angle)))"
        } else {
            poseText = "pose=nil"
        }

        let carrierMode: String
        if contour.surfaceCarrierSupportsDeformation {
            carrierMode = "deforming"
        } else if !contour.surfaceBindingsByIndex.isEmpty {
            carrierMode = "depth_only"
        } else {
            carrierMode = "none"
        }
        return "\(boundsText) \(poseText) opening=\(debugFloat(mouthOpeningRatio(contour))) outer=\(contour.outer.count) inner=\(contour.inner.count) mesh=\(contour.meshPointsByIndex.count) carrier=\(carrierMode)"
    }

    private func debugFloat(_ value: CGFloat) -> String {
        String(format: "%.2f", Double(value))
    }

    private func lipDetection(from landmarks: [NormalizedLandmark],
                              frame: PendingLiveFrame,
                              faceGeometryPose: FaceGeometryPose?) -> LipDetection? {
        let outer = mappedLipLandmarks(
            for: Self.outerLipIndices,
            landmarks: landmarks,
            viewportSize: frame.viewportSize,
            inputToCapturedImageTransform: frame.inputToCapturedImageTransform,
            capturedImageToViewportTransform: frame.capturedImageToViewportTransform
        )
        let inner = mappedLipLandmarks(
            for: Self.innerLipIndices,
            landmarks: landmarks,
            viewportSize: frame.viewportSize,
            inputToCapturedImageTransform: frame.inputToCapturedImageTransform,
            capturedImageToViewportTransform: frame.capturedImageToViewportTransform
        )
        let meshPoints = mappedLipMeshPoints(
            for: Self.attentionLipIndices,
            landmarks: landmarks,
            viewportSize: frame.viewportSize,
            inputToCapturedImageTransform: frame.inputToCapturedImageTransform,
            capturedImageToViewportTransform: frame.capturedImageToViewportTransform
        )

        guard outer.points.count == Self.outerLipIndices.count,
              inner.points.count == Self.innerLipIndices.count,
              outer.points3D.count == Self.outerLipIndices.count,
              inner.points3D.count == Self.innerLipIndices.count,
              outer.uv.count == Self.outerLipIndices.count,
              inner.uv.count == Self.innerLipIndices.count,
              meshPoints.count == Self.attentionLipIndices.count else {
            return nil
        }

        return LipDetection(
            contour: LipContour(
                outer: outer.points,
                inner: inner.points,
                outer3D: outer.points3D,
                inner3D: inner.points3D,
                outerUV: outer.uv,
                innerUV: inner.uv,
                meshPointsByIndex: meshPoints,
                faceGeometryPose: faceGeometryPose
            ),
            pixelBuffer: frame.pixelBuffer,
            imageSize: frame.imageSize
        )
    }

    private func contourBoundToFaceSurface(_ surfaceContour: LipContour,
                                           publishing publishedContour: LipContour,
                                           snapshot: FaceSurfaceSnapshot,
                                           carrierCreatedAt: CFTimeInterval) -> LipContour? {
        guard let pose = surfaceContour.pose,
              carrierCreatedAt.isFinite,
              snapshot.vertexCount > 0,
              !snapshot.triangles.isEmpty else {
            return nil
        }

        // MediaPipe remains the source of the captured contour. The binding
        // supplies depth and a stable ARKit motion carrier between MediaPipe
        // keyframes. ARFaceGeometry has a coarse mouth opening, so its nearest
        // lip surface can legitimately be several screen points from an inner
        // landmark without replacing the visible contour with the AR mesh.
        let maximumDepthProjectionError = max(30.0, pose.width * 0.34)
        var bindings: [Int: LipSurfaceBinding] = [:]
        var errors: [CGFloat] = []
        var edgeFallbackCount = 0
        bindings.reserveCapacity(Self.attentionLipIndices.count)
        errors.reserveCapacity(Self.attentionLipIndices.count)
        let previousBindings: [Int: LipSurfaceBinding]
        if let previous = smoothedLipContour,
           let previousCarrierCreatedAt = previous.surfaceCarrierCreatedAt {
            let previousCarrierAge = CACurrentMediaTime() - previousCarrierCreatedAt
            let maximumPreviousCarrierAge = previous.surfaceCarrierSupportsDeformation ?
                Self.maxSurfaceCarrierHoldAge :
                Self.maxDepthOnlySurfaceCarrierHoldAge
            if previousCarrierAge.isFinite,
               previousCarrierAge >= 0,
               previousCarrierAge <= maximumPreviousCarrierAge,
               previous.surfaceVertexCount == snapshot.vertexCount,
               previous.surfaceTriangleIndexCount == snapshot.triangleIndexCount,
               previous.surfaceTopologySignature == snapshot.topologySignature {
                previousBindings = previous.surfaceBindingsByIndex
            } else {
                previousBindings = [:]
            }
        } else {
            previousBindings = [:]
        }
        let triangleSwitchHysteresis = max(0.55, pose.width * 0.006)

        for landmarkIndex in Self.attentionLipIndices {
            guard let point = surfaceContour.meshPointsByIndex[landmarkIndex]?.screen else {
                return nil
            }
            guard let candidate = closestFaceSurfaceBinding(
                    to: point,
                    landmarkIndex: landmarkIndex,
                    triangles: snapshot.triangles,
                    previousBinding: previousBindings[landmarkIndex],
                    triangleSwitchHysteresis: triangleSwitchHysteresis
                  ) else {
                LipDebugLog.throttled(
                    "lip_surface_incomplete",
                    interval: 0.6,
                    "lip_surface reject reason=no_candidate index=\(landmarkIndex) coverage=\(bindings.count)/\(Self.attentionLipIndices.count)"
                )
                return nil
            }
            guard candidate.binding.projectionError <= maximumDepthProjectionError else {
                LipDebugLog.throttled(
                    "lip_surface_incomplete",
                    interval: 0.6,
                    "lip_surface reject reason=depth_projection_error index=\(landmarkIndex) coverage=\(bindings.count)/\(Self.attentionLipIndices.count) error=\(debugFloat(candidate.binding.projectionError)) maxAllowed=\(debugFloat(maximumDepthProjectionError)) point=(\(debugFloat(point.x)),\(debugFloat(point.y)))"
                )
                return nil
            }
            bindings[landmarkIndex] = candidate.binding
            errors.append(candidate.binding.projectionError)
            if candidate.binding.projectionError > 0.25 {
                edgeFallbackCount += 1
            }
        }

        let sortedErrors = errors.sorted()
        guard sortedErrors.count == Self.attentionLipIndices.count else {
            return nil
        }
        let medianError = sortedErrors[sortedErrors.count / 2]
        let p95Index = min(
            sortedErrors.count - 1,
            Int((Double(sortedErrors.count - 1) * 0.95).rounded(.up))
        )
        let p95Error = sortedErrors[p95Index]
        let maximumError = sortedErrors.last ?? .greatestFiniteMagnitude
        // The inner seam often falls inside ARKit's coarse mouth opening and is
        // therefore clamped to the nearest lip edge. Judge the carrier by the
        // robust distribution instead of rejecting an otherwise accurate frame
        // for a handful of 7-10 point edge fallbacks.
        let maximumMedianError = max(3.5, pose.width * 0.045)
        let maximumP95Error = max(10.0, pose.width * 0.11)
        let surfaceProjectionQualityIsReliable = medianError <= maximumMedianError &&
            p95Error <= maximumP95Error
        if !surfaceProjectionQualityIsReliable {
            LipDebugLog.throttled(
                "lip_surface_quality_reject",
                interval: 0.6,
                "lip_surface deforming reject reason=quality median=\(debugFloat(medianError)) p95=\(debugFloat(p95Error)) max=\(debugFloat(maximumError)) fallbacks=\(edgeFallbackCount)"
            )
        }
        let surfaceBindingIsCoherent = surfaceProjectionQualityIsReliable &&
            hasCoherentSurfaceBinding(
                bindings,
                surfaceContour: surfaceContour,
                snapshot: snapshot
            )
        // Delta-warp consumes only how the bound AR points moved since this
        // keyframe. It does not render ARKit's coarse mesh as the lip shape, so
        // reliable projections are sufficient even when that mesh cannot
        // reproduce every MediaPipe annulus triangle verbatim.
        let supportsSurfaceDeformation = surfaceProjectionQualityIsReliable

        var boundContour = publishedContour
        boundContour.surfaceBindingsByIndex = bindings
        boundContour.surfaceVertexCount = snapshot.vertexCount
        boundContour.surfaceTriangleIndexCount = snapshot.triangleIndexCount
        boundContour.surfaceTopologySignature = snapshot.topologySignature
        // Age the carrier from the AR frame that produced the bindings, not
        // from the delayed MediaPipe callback that happened to accept them.
        boundContour.surfaceCarrierCreatedAt = carrierCreatedAt
        boundContour.surfaceCarrierSupportsDeformation = supportsSurfaceDeformation
        if surfaceBindingIsCoherent {
            shouldExpandFaceSurfaceTopology = false
        } else {
            faceSurfaceTopologyCache = nil
            shouldExpandFaceSurfaceTopology = true
        }
        if supportsSurfaceDeformation {
            LipDebugLog.throttled(
                "lip_surface_accept",
                interval: 0.6,
                "lip_surface accept mode=delta_warp coherent=\(surfaceBindingIsCoherent) coverage=\(bindings.count)/\(Self.attentionLipIndices.count) median=\(debugFloat(medianError)) p95=\(debugFloat(p95Error)) max=\(debugFloat(maximumError)) fallbacks=\(edgeFallbackCount)"
            )
        } else {
            // Keep independently validated bindings as per-vertex depth
            // samples, but do not let a weak projection drive visible motion.
            LipDebugLog.throttled(
                "lip_surface_depth_only",
                interval: 0.6,
                "lip_surface accept mode=depth_only reason=\(surfaceProjectionQualityIsReliable ? "local_continuity" : "projection_quality") coverage=\(bindings.count)/\(Self.attentionLipIndices.count) median=\(debugFloat(medianError)) p95=\(debugFloat(p95Error)) max=\(debugFloat(maximumError)) fallbacks=\(edgeFallbackCount)"
            )
        }
        return boundContour
    }

    private func refreshedDepthBindings(
        from previousContour: LipContour,
        for currentContour: LipContour,
        snapshot: FaceSurfaceSnapshot
    ) -> [Int: LipSurfaceBinding]? {
        guard let pose = currentContour.pose else {
            return nil
        }
        let maximumProjectionError = max(30.0, pose.width * 0.34)
        var refreshed: [Int: LipSurfaceBinding] = [:]
        var errors: [CGFloat] = []
        refreshed.reserveCapacity(Self.attentionLipIndices.count)
        errors.reserveCapacity(Self.attentionLipIndices.count)

        for landmarkIndex in Self.attentionLipIndices {
            guard let point = currentContour.meshPointsByIndex[landmarkIndex]?.screen,
                  let previousBinding = previousContour.surfaceBindingsByIndex[landmarkIndex],
                  let triangle = snapshot.trianglesByKey[
                      FaceSurfaceTriangleKey(previousBinding.vertexIndices)
                  ],
                  let candidate = projectedCandidate(
                      from: previousBinding,
                      point: point,
                      triangle: triangle,
                      expectedSide: CanonicalLipGeometry.expectedSurfaceSide(
                          for: landmarkIndex
                      )
                  ),
                  candidate.binding.projectionError <= maximumProjectionError else {
                return nil
            }
            refreshed[landmarkIndex] = candidate.binding
            errors.append(candidate.binding.projectionError)
        }
        guard refreshed.count == Self.attentionLipIndices.count,
              errors.count == Self.attentionLipIndices.count else {
            return nil
        }
        let sortedErrors = errors.sorted()
        let medianError = sortedErrors[sortedErrors.count / 2]
        let p95Index = min(
            sortedErrors.count - 1,
            Int((Double(sortedErrors.count - 1) * 0.95).rounded(.up))
        )
        let p95Error = sortedErrors[p95Index]
        let maximumMedianError = max(3.5, pose.width * 0.045)
        let maximumP95Error = max(10.0, pose.width * 0.11)
        guard medianError <= maximumMedianError,
              p95Error <= maximumP95Error else {
            LipDebugLog.throttled(
                "lip_surface_reuse_quality_reject",
                interval: 0.6,
                "lip_surface reuse reject reason=quality median=\(debugFloat(medianError)) p95=\(debugFloat(p95Error))"
            )
            return nil
        }
        return refreshed
    }

    private func hasCoherentSurfaceBinding(
        _ bindings: [Int: LipSurfaceBinding],
        surfaceContour: LipContour,
        snapshot: FaceSurfaceSnapshot
    ) -> Bool {
        let mouthFrame = snapshot.mouthFrame
        let mouthWidth = mouthFrame.width
        guard mouthWidth.isFinite, mouthWidth > 0.008 else {
            return false
        }

        var positions: [Int: SIMD3<Float>] = [:]
        positions.reserveCapacity(bindings.count)
        for (landmarkIndex, binding) in bindings {
            let firstIndex = Int(binding.vertexIndices.x)
            let secondIndex = Int(binding.vertexIndices.y)
            let thirdIndex = Int(binding.vertexIndices.z)
            guard snapshot.sourceVertices.indices.contains(firstIndex),
                  snapshot.sourceVertices.indices.contains(secondIndex),
                  snapshot.sourceVertices.indices.contains(thirdIndex) else {
                return false
            }
            let weights = binding.barycentricWeights
            let point = snapshot.sourceVertices[firstIndex] * weights.x +
                snapshot.sourceVertices[secondIndex] * weights.y +
                snapshot.sourceVertices[thirdIndex] * weights.z
            let relative = point - mouthFrame.center
            let normalizedPoint = SIMD3<Float>(
                simd_dot(relative, mouthFrame.xAxis) / mouthWidth,
                simd_dot(relative, mouthFrame.downAxis) / mouthWidth,
                simd_dot(relative, mouthFrame.normalAxis) / mouthWidth
            )
            let expectedSide = CanonicalLipGeometry.expectedSurfaceSide(for: landmarkIndex)
            guard point.x.isFinite,
                  point.y.isFinite,
                  point.z.isFinite else {
                return false
            }
            guard abs(normalizedPoint.x) <= 0.88,
                  abs(normalizedPoint.y) <= 0.62,
                  abs(normalizedPoint.z) <= 0.56,
                  expectedSide == 0 || normalizedPoint.y * expectedSide >= -0.12 else {
                LipDebugLog.throttled(
                    "lip_surface_continuity_point",
                    interval: 0.6,
                    "lip_surface continuity reject reason=mouth_region index=\(landmarkIndex) local=(\(String(format: "%.3f", normalizedPoint.x)),\(String(format: "%.3f", normalizedPoint.y)),\(String(format: "%.3f", normalizedPoint.z))) side=\(String(format: "%.0f", expectedSide))"
                )
                return false
            }
            positions[landmarkIndex] = point
        }

        let maximumEdgeLength = mouthWidth * 0.56
        let maximumNormalStep = mouthWidth * 0.30
        let maximumTwiceArea = mouthWidth * mouthWidth * 0.28
        let screenToSurfaceScale = mouthWidth / Float(max(surfaceContour.pose?.width ?? 1, 1))
        var orientationVote = 0
        for triangle in CanonicalLipGeometry.lipMeshTriangles {
            guard let first = positions[triangle.0],
                  let second = positions[triangle.1],
                  let third = positions[triangle.2],
                  let firstScreen = surfaceContour.meshPointsByIndex[triangle.0]?.screen,
                  let secondScreen = surfaceContour.meshPointsByIndex[triangle.1]?.screen,
                  let thirdScreen = surfaceContour.meshPointsByIndex[triangle.2]?.screen else {
                return false
            }
            let firstLocal = SIMD2<Float>(
                simd_dot(first - mouthFrame.center, mouthFrame.xAxis),
                simd_dot(first - mouthFrame.center, mouthFrame.downAxis)
            )
            let secondLocal = SIMD2<Float>(
                simd_dot(second - mouthFrame.center, mouthFrame.xAxis),
                simd_dot(second - mouthFrame.center, mouthFrame.downAxis)
            )
            let thirdLocal = SIMD2<Float>(
                simd_dot(third - mouthFrame.center, mouthFrame.xAxis),
                simd_dot(third - mouthFrame.center, mouthFrame.downAxis)
            )
            let localSignedArea = Self.signedArea(
                firstLocal,
                secondLocal,
                thirdLocal
            )
            let screenSignedArea = Self.signedArea(
                SIMD2<Float>(Float(firstScreen.x), Float(firstScreen.y)),
                SIMD2<Float>(Float(secondScreen.x), Float(secondScreen.y)),
                SIMD2<Float>(Float(thirdScreen.x), Float(thirdScreen.y))
            )
            let expectedLocalArea = abs(screenSignedArea) * screenToSurfaceScale * screenToSurfaceScale
            if expectedLocalArea > mouthWidth * mouthWidth * 0.000_01,
               abs(localSignedArea) >= expectedLocalArea * 0.03 {
                orientationVote += localSignedArea * screenSignedArea >= 0 ? 1 : -1
            }
        }
        let expectedOrientationRelation: Float = orientationVote >= 0 ? 1 : -1
        let arOpeningRatio: Float
        if snapshot.sourceVertices.indices.contains(Self.arKitMouthTopIndex),
           snapshot.sourceVertices.indices.contains(Self.arKitMouthBottomIndex) {
            arOpeningRatio = simd_length(
                snapshot.sourceVertices[Self.arKitMouthBottomIndex] -
                    snapshot.sourceVertices[Self.arKitMouthTopIndex]
            ) / mouthWidth
        } else {
            arOpeningRatio = 0
        }
        let mediaPipeMouthIsOpen = mouthOpeningRatio(surfaceContour) >= 0.025
        let arMouthIsOpen = arOpeningRatio >= 0.08
        if arMouthIsOpen && !mediaPipeMouthIsOpen {
            LipDebugLog.throttled(
                "lip_surface_opening_signal_mismatch",
                interval: 0.6,
                "lip_surface continuity reject reason=opening_signal_mismatch ar=\(String(format: "%.3f", arOpeningRatio)) mediaPipe=\(debugFloat(mouthOpeningRatio(surfaceContour)))"
            )
            return false
        }
        let mouthIsOpen = mediaPipeMouthIsOpen || arMouthIsOpen
        if mouthIsOpen {
            guard let upperCenter = positions[13],
                  let lowerCenter = positions[14],
                  let upperScreen = surfaceContour.meshPointsByIndex[13]?.screen,
                  let lowerScreen = surfaceContour.meshPointsByIndex[14]?.screen,
                  let pose = surfaceContour.pose else {
                return false
            }
            let localOpening = simd_dot(
                lowerCenter - upperCenter,
                mouthFrame.downAxis
            )
            let screenDown = SIMD2<Float>(
                -Float(sin(pose.angle)),
                Float(cos(pose.angle))
            )
            let screenOpening = simd_dot(
                SIMD2<Float>(
                    Float(lowerScreen.x - upperScreen.x),
                    Float(lowerScreen.y - upperScreen.y)
                ),
                screenDown
            )
            let expectedLocalOpening = max(screenOpening, 0) * screenToSurfaceScale
            guard localOpening > mouthWidth * 0.004,
                  localOpening >= expectedLocalOpening * 0.15 else {
                LipDebugLog.throttled(
                    "lip_surface_inner_aperture",
                    interval: 0.6,
                    "lip_surface continuity reject reason=inner_aperture local=\(String(format: "%.3f", localOpening / mouthWidth)) expected=\(String(format: "%.3f", expectedLocalOpening / mouthWidth)) arOpen=\(String(format: "%.3f", arOpeningRatio))"
                )
                return false
            }
        }
        var resolvableTriangleCount = 0
        var collapsedTriangleCount = 0
        var expandedTriangleCount = 0
        var flippedTriangleCount = 0
        var invalidInnerBandTriangleCount = 0

        for triangle in CanonicalLipGeometry.lipMeshTriangles {
            guard let first = positions[triangle.0],
                  let second = positions[triangle.1],
                  let third = positions[triangle.2],
                  let firstScreen = surfaceContour.meshPointsByIndex[triangle.0]?.screen,
                  let secondScreen = surfaceContour.meshPointsByIndex[triangle.1]?.screen,
                  let thirdScreen = surfaceContour.meshPointsByIndex[triangle.2]?.screen else {
                return false
            }
            let firstEdge = second - first
            let secondEdge = third - second
            let thirdEdge = first - third
            let firstLocal = SIMD2<Float>(
                simd_dot(first - mouthFrame.center, mouthFrame.xAxis),
                simd_dot(first - mouthFrame.center, mouthFrame.downAxis)
            )
            let secondLocal = SIMD2<Float>(
                simd_dot(second - mouthFrame.center, mouthFrame.xAxis),
                simd_dot(second - mouthFrame.center, mouthFrame.downAxis)
            )
            let thirdLocal = SIMD2<Float>(
                simd_dot(third - mouthFrame.center, mouthFrame.xAxis),
                simd_dot(third - mouthFrame.center, mouthFrame.downAxis)
            )
            let localSignedArea = Self.signedArea(firstLocal, secondLocal, thirdLocal)
            let screenSignedArea = Self.signedArea(
                SIMD2<Float>(Float(firstScreen.x), Float(firstScreen.y)),
                SIMD2<Float>(Float(secondScreen.x), Float(secondScreen.y)),
                SIMD2<Float>(Float(thirdScreen.x), Float(thirdScreen.y))
            )
            let expectedLocalArea = abs(screenSignedArea) * screenToSurfaceScale * screenToSurfaceScale
            let areaIsResolvable = expectedLocalArea > mouthWidth * mouthWidth * 0.000_01
            let touchesInnerBoundary = CanonicalLipGeometry.isInnerLipIndex(triangle.0) ||
                CanonicalLipGeometry.isInnerLipIndex(triangle.1) ||
                CanonicalLipGeometry.isInnerLipIndex(triangle.2)
            let maximumObservedEdge = max(
                simd_length(firstEdge),
                max(simd_length(secondEdge), simd_length(thirdEdge))
            )
            let maximumObservedNormalStep = max(
                abs(simd_dot(firstEdge, mouthFrame.normalAxis)),
                max(
                    abs(simd_dot(secondEdge, mouthFrame.normalAxis)),
                    abs(simd_dot(thirdEdge, mouthFrame.normalAxis))
                )
            )
            let observedTwiceArea = simd_length(simd_cross(firstEdge, third - first))
            guard maximumObservedEdge <= maximumEdgeLength,
                  maximumObservedNormalStep <= maximumNormalStep,
                  observedTwiceArea <= maximumTwiceArea else {
                LipDebugLog.throttled(
                    "lip_surface_continuity_geometry",
                    interval: 0.6,
                    "lip_surface continuity reject reason=geometry triangle=\(triangle.0),\(triangle.1),\(triangle.2) edge=\(String(format: "%.3f", maximumObservedEdge / mouthWidth)) normal=\(String(format: "%.3f", maximumObservedNormalStep / mouthWidth)) area=\(String(format: "%.3f", observedTwiceArea / (mouthWidth * mouthWidth)))"
                )
                return false
            }
            guard areaIsResolvable else {
                collapsedTriangleCount += 1
                if touchesInnerBoundary {
                    invalidInnerBandTriangleCount += 1
                }
                continue
            }
            resolvableTriangleCount += 1
            let minimumAreaRatio: Float = mouthIsOpen && touchesInnerBoundary ? 0.40 : 0.30
            let maximumAreaRatio: Float = 4.0
            let observedArea = abs(localSignedArea)
            if observedArea < expectedLocalArea * minimumAreaRatio {
                collapsedTriangleCount += 1
                if touchesInnerBoundary {
                    invalidInnerBandTriangleCount += 1
                }
            } else if observedArea > expectedLocalArea * maximumAreaRatio {
                expandedTriangleCount += 1
                if touchesInnerBoundary {
                    invalidInnerBandTriangleCount += 1
                }
            } else if localSignedArea * screenSignedArea * expectedOrientationRelation < 0 {
                flippedTriangleCount += 1
                if touchesInnerBoundary {
                    invalidInnerBandTriangleCount += 1
                }
            }
        }
        // The material renders every triangle, so accepting even one collapsed
        // or flipped face can create a visible patch. Reject the whole carrier
        // and use the deterministic projected ring until all faces are coherent.
        let triangleCount = CanonicalLipGeometry.lipMeshTriangles.count
        let maximumCollapsedTriangleCount = 0
        let maximumFlippedTriangleCount = 0
        let maximumInvalidTriangleCount = 0
        let invalidTriangleCount = collapsedTriangleCount + expandedTriangleCount + flippedTriangleCount
        guard resolvableTriangleCount == triangleCount,
              !mouthIsOpen || invalidInnerBandTriangleCount == 0,
              invalidTriangleCount <= maximumInvalidTriangleCount,
              collapsedTriangleCount <= maximumCollapsedTriangleCount,
              flippedTriangleCount <= maximumFlippedTriangleCount else {
            LipDebugLog.throttled(
                "lip_surface_continuity_metrics",
                interval: 0.6,
                "lip_surface continuity reject collapsed=\(collapsedTriangleCount)/\(maximumCollapsedTriangleCount) expanded=\(expandedTriangleCount) flipped=\(flippedTriangleCount)/\(maximumFlippedTriangleCount) innerInvalid=\(invalidInnerBandTriangleCount) mouthOpen=\(mouthIsOpen) invalid=\(invalidTriangleCount)/\(maximumInvalidTriangleCount) resolvable=\(resolvableTriangleCount) triangles=\(triangleCount)"
            )
            return false
        }
        LipDebugLog.throttled(
            "lip_surface_continuity_accept",
            interval: 0.6,
            "lip_surface continuity accept collapsed=\(collapsedTriangleCount) expanded=\(expandedTriangleCount) flipped=\(flippedTriangleCount) innerInvalid=\(invalidInnerBandTriangleCount) mouthOpen=\(mouthIsOpen) invalid=\(invalidTriangleCount) resolvable=\(resolvableTriangleCount) triangles=\(triangleCount)"
        )
        return true
    }

    private static func signedArea(_ first: SIMD2<Float>,
                                   _ second: SIMD2<Float>,
                                   _ third: SIMD2<Float>) -> Float {
        (second.x - first.x) * (third.y - first.y) -
            (second.y - first.y) * (third.x - first.x)
    }

    private func closestFaceSurfaceBinding(
        to point: CGPoint,
        landmarkIndex: Int,
        triangles: [ProjectedFaceSurfaceTriangle],
        previousBinding: LipSurfaceBinding?,
        triangleSwitchHysteresis: CGFloat
    ) -> LipSurfaceBindingCandidate? {
        var best: LipSurfaceBindingCandidate?
        let expectedSide = CanonicalLipGeometry.expectedSurfaceSide(for: landmarkIndex)
        for triangle in triangles {
            let signedMouthSide = triangle.normalizedMouthY * expectedSide
            let targetMouthY = expectedSide * 0.10
            let mouthSideScore = -abs(triangle.normalizedMouthY - targetMouthY)
            guard expectedSide == 0 || signedMouthSide >= -0.04 else {
                continue
            }
            guard let screenWeights = closestScreenBarycentricWeights(
                point: point,
                first: triangle.first,
                second: triangle.second,
                third: triangle.third
            ),
            triangle.cameraDepths.x.isFinite,
            triangle.cameraDepths.y.isFinite,
            triangle.cameraDepths.z.isFinite,
            triangle.cameraDepths.x > 0.01,
            triangle.cameraDepths.y > 0.01,
            triangle.cameraDepths.z > 0.01 else {
                continue
            }

            let inverseDepthWeights = SIMD3<Float>(
                screenWeights.x / triangle.cameraDepths.x,
                screenWeights.y / triangle.cameraDepths.y,
                screenWeights.z / triangle.cameraDepths.z
            )
            let weightSum = inverseDepthWeights.x + inverseDepthWeights.y + inverseDepthWeights.z
            guard weightSum.isFinite, weightSum > 0.000_001 else {
                continue
            }
            let surfaceWeights = inverseDepthWeights / weightSum
            guard surfaceWeights.x.isFinite,
                  surfaceWeights.y.isFinite,
                  surfaceWeights.z.isFinite,
                  surfaceWeights.x >= -0.001,
                  surfaceWeights.y >= -0.001,
                  surfaceWeights.z >= -0.001,
                  surfaceWeights.x <= 1.001,
                  surfaceWeights.y <= 1.001,
                  surfaceWeights.z <= 1.001 else {
                continue
            }

            let projectedPoint = CGPoint(
                x: triangle.first.x * CGFloat(screenWeights.x) +
                    triangle.second.x * CGFloat(screenWeights.y) +
                    triangle.third.x * CGFloat(screenWeights.z),
                y: triangle.first.y * CGFloat(screenWeights.x) +
                    triangle.second.y * CGFloat(screenWeights.y) +
                    triangle.third.y * CGFloat(screenWeights.z)
            )
            let projectionError = hypot(projectedPoint.x - point.x, projectedPoint.y - point.y)
            let cameraDepth = simd_dot(surfaceWeights, triangle.cameraDepths)
            guard projectionError.isFinite,
                  cameraDepth.isFinite,
                  cameraDepth > 0.01 else {
                continue
            }

            let candidate = LipSurfaceBindingCandidate(
                binding: LipSurfaceBinding(
                    vertexIndices: triangle.vertexIndices,
                    barycentricWeights: surfaceWeights,
                    projectionError: projectionError,
                    referenceScreenPoint: projectedPoint
                ),
                cameraDepth: cameraDepth,
                mouthSideScore: mouthSideScore
            )
            if let currentBest = best {
                let errorImprovement = projectionError < currentBest.binding.projectionError - 0.05
                let equivalentError = abs(projectionError - currentBest.binding.projectionError) <= 0.05
                let sideImprovement = mouthSideScore > currentBest.mouthSideScore + 0.015
                let equivalentSide = abs(mouthSideScore - currentBest.mouthSideScore) <= 0.015
                if errorImprovement ||
                    (equivalentError && sideImprovement) ||
                    (equivalentError && equivalentSide && cameraDepth < currentBest.cameraDepth) {
                    best = candidate
                }
            } else {
                best = candidate
            }
        }
        guard let best else {
            return nil
        }
        if let previousBinding,
           previousBinding.vertexIndices == best.binding.vertexIndices,
           let triangle = triangles.first(where: {
            $0.vertexIndices == previousBinding.vertexIndices
           }),
           let previousCandidate = projectedCandidate(
            from: previousBinding,
            point: point,
            triangle: triangle,
            expectedSide: expectedSide
           ) {
            let movementRange = max(triangleSwitchHysteresis * 2, 0.001)
            let blendAmount = Float(min(
                max(
                    0.68 + previousCandidate.binding.projectionError / movementRange * 0.32,
                    0.68
                ),
                1
            ))
            var blendedWeights = previousBinding.barycentricWeights +
                (best.binding.barycentricWeights - previousBinding.barycentricWeights) * blendAmount
            let blendedWeightSum = blendedWeights.x + blendedWeights.y + blendedWeights.z
            if blendedWeightSum.isFinite, blendedWeightSum > 0.000_001 {
                blendedWeights /= blendedWeightSum
                let blendedBinding = LipSurfaceBinding(
                    vertexIndices: best.binding.vertexIndices,
                    barycentricWeights: blendedWeights,
                    projectionError: best.binding.projectionError,
                    referenceScreenPoint: best.binding.referenceScreenPoint
                )
                if let blendedCandidate = projectedCandidate(
                    from: blendedBinding,
                    point: point,
                    triangle: triangle,
                    expectedSide: expectedSide
                ) {
                    return blendedCandidate
                }
            }
        }
        guard let previousBinding,
              previousBinding.vertexIndices != best.binding.vertexIndices,
              let previousTriangle = triangles.first(where: {
                $0.vertexIndices == previousBinding.vertexIndices
              }),
              let previousCandidate = projectedCandidate(
                from: previousBinding,
                point: point,
                triangle: previousTriangle,
                expectedSide: expectedSide
              ),
              previousCandidate.binding.projectionError <=
                best.binding.projectionError + triangleSwitchHysteresis,
              previousCandidate.mouthSideScore >= best.mouthSideScore - 0.015,
              abs(previousCandidate.cameraDepth - best.cameraDepth) <=
                max(0.004, best.cameraDepth * 0.012) else {
            return best
        }
        return previousCandidate
    }

    private func projectedCandidate(from binding: LipSurfaceBinding,
                                    point: CGPoint,
                                    triangle: ProjectedFaceSurfaceTriangle,
                                    expectedSide: Float) -> LipSurfaceBindingCandidate? {
        let weights = binding.barycentricWeights
        let weightSum = weights.x + weights.y + weights.z
        let signedMouthSide = triangle.normalizedMouthY * expectedSide
        let targetMouthY = expectedSide * 0.10
        let mouthSideScore = -abs(triangle.normalizedMouthY - targetMouthY)
        guard weights.x.isFinite,
              weights.y.isFinite,
              weights.z.isFinite,
              abs(weightSum - 1) < 0.002,
              expectedSide == 0 || signedMouthSide >= -0.04 else {
            return nil
        }

        let weightedDepths = weights * triangle.cameraDepths
        let cameraDepth = weightedDepths.x + weightedDepths.y + weightedDepths.z
        guard cameraDepth.isFinite, cameraDepth > 0.01 else {
            return nil
        }
        let screenWeights = weightedDepths / cameraDepth
        let projectedPoint = CGPoint(
            x: triangle.first.x * CGFloat(screenWeights.x) +
                triangle.second.x * CGFloat(screenWeights.y) +
                triangle.third.x * CGFloat(screenWeights.z),
            y: triangle.first.y * CGFloat(screenWeights.x) +
                triangle.second.y * CGFloat(screenWeights.y) +
                triangle.third.y * CGFloat(screenWeights.z)
        )
        let projectionError = hypot(projectedPoint.x - point.x, projectedPoint.y - point.y)
        guard projectionError.isFinite else {
            return nil
        }
        return LipSurfaceBindingCandidate(
            binding: LipSurfaceBinding(
                vertexIndices: binding.vertexIndices,
                barycentricWeights: weights,
                projectionError: projectionError,
                referenceScreenPoint: projectedPoint
            ),
            cameraDepth: cameraDepth,
            mouthSideScore: mouthSideScore
        )
    }

    private func closestScreenBarycentricWeights(point: CGPoint,
                                                 first: CGPoint,
                                                 second: CGPoint,
                                                 third: CGPoint) -> SIMD3<Float>? {
        let denominator =
            (second.y - third.y) * (first.x - third.x) +
            (third.x - second.x) * (first.y - third.y)
        guard denominator.isFinite, abs(denominator) > 0.000_001 else {
            return nil
        }

        let firstWeight = (
            (second.y - third.y) * (point.x - third.x) +
            (third.x - second.x) * (point.y - third.y)
        ) / denominator
        let secondWeight = (
            (third.y - first.y) * (point.x - third.x) +
            (first.x - third.x) * (point.y - third.y)
        ) / denominator
        let thirdWeight = 1 - firstWeight - secondWeight
        if firstWeight >= -0.000_1,
           secondWeight >= -0.000_1,
           thirdWeight >= -0.000_1 {
            let clamped = SIMD3<Float>(
                Float(max(0, firstWeight)),
                Float(max(0, secondWeight)),
                Float(max(0, thirdWeight))
            )
            let sum = clamped.x + clamped.y + clamped.z
            guard sum > 0.000_001 else {
                return nil
            }
            return clamped / sum
        }

        func edgeWeights(from start: CGPoint,
                         to end: CGPoint,
                         startWeights: SIMD3<Float>,
                         endWeights: SIMD3<Float>) -> (weights: SIMD3<Float>, errorSquared: CGFloat)? {
            let dx = end.x - start.x
            let dy = end.y - start.y
            let lengthSquared = dx * dx + dy * dy
            guard lengthSquared.isFinite, lengthSquared > 0.000_001 else {
                return nil
            }
            let unclamped = ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared
            let amount = max(0, min(unclamped, 1))
            let closest = CGPoint(x: start.x + dx * amount, y: start.y + dy * amount)
            let errorX = closest.x - point.x
            let errorY = closest.y - point.y
            return (
                startWeights + (endWeights - startWeights) * Float(amount),
                errorX * errorX + errorY * errorY
            )
        }

        let firstBasis = SIMD3<Float>(1, 0, 0)
        let secondBasis = SIMD3<Float>(0, 1, 0)
        let thirdBasis = SIMD3<Float>(0, 0, 1)
        let candidates = [
            edgeWeights(from: first, to: second, startWeights: firstBasis, endWeights: secondBasis),
            edgeWeights(from: second, to: third, startWeights: secondBasis, endWeights: thirdBasis),
            edgeWeights(from: third, to: first, startWeights: thirdBasis, endWeights: firstBasis)
        ].compactMap { $0 }
        return candidates.min(by: { $0.errorSquared < $1.errorSquared })?.weights
    }

    private func isReliableLipUpdate(_ contour: LipContour,
                                     viewportSize: CGSize,
                                     motionReference: LipMotionPose?) -> Bool {
        guard let currentPose = contour.pose,
              hasReliableLipTopology(contour) else {
            return false
        }

        guard let previousPose = smoothedLipPose else {
            return true
        }

        let expectedPose: LipPose
        if let lastAcceptedMotionPose,
           let motionReference {
            expectedPose = motionCompensatedPose(
                previousPose,
                from: lastAcceptedMotionPose,
                to: motionReference
            )
        } else {
            expectedPose = previousPose
        }

        let expectedWidth = max(expectedPose.width, 1)
        let centerDistance = hypot(
            currentPose.center.x - expectedPose.center.x,
            currentPose.center.y - expectedPose.center.y
        ) / expectedWidth
        let widthRatio = currentPose.width / max(expectedPose.width, 1)
        let angleDelta = abs(Self.normalizedAngle(currentPose.angle - expectedPose.angle))
        let viewportDrift = hypot(
            currentPose.center.x - expectedPose.center.x,
            currentPose.center.y - expectedPose.center.y
        ) / max(min(viewportSize.width, viewportSize.height), 1)
        let previousOpening = smoothedLipContour.map { mouthOpeningRatio($0) } ?? 0
        let currentOpening = mouthOpeningRatio(contour)
        let openingDelta = abs(currentOpening - previousOpening)

        if widthRatio < 0.70 || widthRatio > 1.42 {
            LipDebugLog.throttled(
                "lip_detection_pose_innovation",
                interval: 0.5,
                "lip_detection hold reason=pose_innovation center=\(debugFloat(centerDistance)) widthRatio=\(debugFloat(widthRatio)) angle=\(debugFloat(angleDelta)) viewport=\(debugFloat(viewportDrift))"
            )
            return false
        }
        if widthRatio > 1.30 && openingDelta > 0.16 {
            LipDebugLog.throttled(
                "lip_detection_pose_innovation",
                interval: 0.5,
                "lip_detection hold reason=expression_spike widthRatio=\(debugFloat(widthRatio)) openingDelta=\(debugFloat(openingDelta))"
            )
            return false
        }
        // Projection disagreement grows temporarily during fast camera and
        // mouth motion. Widen only this bounded gate while ARKit independently
        // confirms motion; the topology checks above remain authoritative.
        let referenceMotionDelta = motionDelta(
            from: lastAcceptedMotionPose,
            to: motionReference
        )
        let motionActivity = referenceMotionDelta.isFinite ?
            min(max(referenceMotionDelta / 0.45, 0), 1) : 0
        let maximumCenterDistance = 0.22 + motionActivity * 0.08
        let maximumViewportDrift = 0.06 + motionActivity * 0.025
        let maximumAngleDelta = 0.25 + motionActivity * 0.08
        if centerDistance > maximumCenterDistance ||
            viewportDrift > maximumViewportDrift {
            LipDebugLog.throttled(
                "lip_detection_pose_innovation",
                interval: 0.5,
                "lip_detection hold reason=pose_innovation center=\(debugFloat(centerDistance)) widthRatio=\(debugFloat(widthRatio)) angle=\(debugFloat(angleDelta)) viewport=\(debugFloat(viewportDrift))"
            )
            return false
        }
        if angleDelta > maximumAngleDelta {
            LipDebugLog.throttled(
                "lip_detection_pose_innovation",
                interval: 0.5,
                "lip_detection hold reason=pose_innovation center=\(debugFloat(centerDistance)) widthRatio=\(debugFloat(widthRatio)) angle=\(debugFloat(angleDelta)) viewport=\(debugFloat(viewportDrift))"
            )
            return false
        }
        if let previousContour = smoothedLipContour,
           let previousMotionPose = lastAcceptedMotionPose,
           let motionReference,
           !hasReliableARLocalShapeInnovation(
               previous: previousContour,
               previousMotionPose: previousMotionPose,
               current: contour,
               currentMotionPose: motionReference
           ) {
            return false
        }
        return true
    }

    private func hasReliableARLocalShapeInnovation(
        previous: LipContour,
        previousMotionPose: LipMotionPose,
        current: LipContour,
        currentMotionPose: LipMotionPose
    ) -> Bool {
        var magnitudes: [CGFloat] = []
        magnitudes.reserveCapacity(Self.attentionLipIndices.count)
        var total = CGVector.zero
        var largePointCount = 0

        for landmarkIndex in Self.attentionLipIndices {
            guard let previousPoint = previous.meshPointsByIndex[landmarkIndex]?.screen,
                  let currentPoint = current.meshPointsByIndex[landmarkIndex]?.screen else {
                return false
            }
            let previousLocal = normalizedLipPoint(
                previousPoint,
                pose: previousMotionPose
            )
            let currentLocal = normalizedLipPoint(
                currentPoint,
                pose: currentMotionPose
            )
            let innovation = CGVector(
                dx: currentLocal.x - previousLocal.x,
                dy: currentLocal.y - previousLocal.y
            )
            let magnitude = hypot(innovation.dx, innovation.dy)
            guard innovation.dx.isFinite,
                  innovation.dy.isFinite,
                  magnitude.isFinite else {
                return false
            }
            total = CGVector(
                dx: total.dx + innovation.dx,
                dy: total.dy + innovation.dy
            )
            magnitudes.append(magnitude)
            if magnitude > 0.20 {
                largePointCount += 1
            }
        }

        guard magnitudes.count == Self.attentionLipIndices.count else {
            return false
        }
        let sorted = magnitudes.sorted()
        let median = sorted[sorted.count / 2]
        let p95Index = min(
            sorted.count - 1,
            Int((Double(sorted.count - 1) * 0.95).rounded(.up))
        )
        let p95 = sorted[p95Index]
        let inverseCount = 1 / CGFloat(sorted.count)
        let meanShift = hypot(
            total.dx * inverseCount,
            total.dy * inverseCount
        )
        // Pursing compresses the inner seam and moves most lip landmarks at
        // once. That is a coherent expression, not the sparse outlier this
        // gate is meant to reject. Give closed/near-closed lips a little more
        // local-shape headroom while retaining the stricter open-mouth gate.
        let isNearClosed = min(
            mouthOpeningRatio(previous),
            mouthOpeningRatio(current)
        ) <= 0.12
        let referenceMotionDelta = motionDelta(
            from: previousMotionPose,
            to: currentMotionPose
        )
        let motionActivity = referenceMotionDelta.isFinite ?
            min(max(referenceMotionDelta / 0.45, 0), 1) : 0
        let maximumMeanShift: CGFloat =
            (isNearClosed ? 0.14 : 0.12) + motionActivity * 0.07
        let maximumMedian: CGFloat =
            (isNearClosed ? 0.15 : 0.10) + motionActivity * 0.08
        let maximumP95: CGFloat =
            (isNearClosed ? 0.27 : 0.22) + motionActivity * 0.20
        let maximumLargePointCount =
            (isNearClosed ? 10 : 8) + Int((motionActivity * 24).rounded())
        let isReliable = meanShift <= maximumMeanShift &&
            median <= maximumMedian &&
            p95 <= maximumP95 &&
            largePointCount <= maximumLargePointCount
        if !isReliable {
            LipDebugLog.throttled(
                "lip_detection_shape_innovation",
                interval: 0.5,
                "lip_detection hold reason=shape_innovation mean=\(debugFloat(meanShift)) median=\(debugFloat(median)) p95=\(debugFloat(p95)) large=\(largePointCount) motionActivity=\(debugFloat(motionActivity))"
            )
        }
        return isReliable
    }

    private func hasReliableLipTopology(_ contour: LipContour) -> Bool {
        guard let outerBounds = bounds(for: contour.outer),
              let innerBounds = bounds(for: contour.inner),
              outerBounds.width > 1,
              outerBounds.height > 1,
              innerBounds.width > 1,
              innerBounds.height > 1 else {
            return false
        }

        let expandedOuter = outerBounds.insetBy(
            dx: -outerBounds.width * 0.18,
            dy: -outerBounds.height * 0.24
        )
        let innerCorners = [
            CGPoint(x: innerBounds.minX, y: innerBounds.minY),
            CGPoint(x: innerBounds.maxX, y: innerBounds.minY),
            CGPoint(x: innerBounds.minX, y: innerBounds.maxY),
            CGPoint(x: innerBounds.maxX, y: innerBounds.maxY)
        ]
        guard innerCorners.allSatisfy({ expandedOuter.contains($0) }) else {
            return false
        }

        guard innerBounds.width <= outerBounds.width * 1.08,
              innerBounds.height <= outerBounds.height * 1.05 else {
            return false
        }

        return hasConsistentScreenTriangleTopology(contour)
    }

    private func hasConsistentScreenTriangleTopology(_ contour: LipContour) -> Bool {
        guard let pose = contour.pose,
              pose.width.isFinite,
              pose.width > 1 else {
            return false
        }

        let minimumResolvableArea = max(
            Float(0.02),
            Float(pose.width * pose.width) * 0.000_002
        )
        var positiveCount = 0
        var negativeCount = 0
        var positiveArea: Float = 0
        var negativeArea: Float = 0

        for triangle in CanonicalLipGeometry.lipMeshTriangles {
            guard let first = contour.meshPointsByIndex[triangle.0]?.screen,
                  let second = contour.meshPointsByIndex[triangle.1]?.screen,
                  let third = contour.meshPointsByIndex[triangle.2]?.screen else {
                return false
            }
            let area = Self.signedArea(
                SIMD2<Float>(Float(first.x), Float(first.y)),
                SIMD2<Float>(Float(second.x), Float(second.y)),
                SIMD2<Float>(Float(third.x), Float(third.y))
            )
            guard area.isFinite else {
                return false
            }
            guard abs(area) > minimumResolvableArea else {
                continue
            }
            if area > 0 {
                positiveCount += 1
                positiveArea += area
            } else {
                negativeCount += 1
                negativeArea += -area
            }
        }

        let resolvedCount = positiveCount + negativeCount
        let totalArea = positiveArea + negativeArea
        guard resolvedCount >= 48,
              totalArea > 0 else {
            LipDebugLog.throttled(
                "lip_detection_triangle_topology",
                interval: 0.5,
                "lip_detection hold reason=triangle_collapse resolved=\(resolvedCount) minArea=\(debugFloat(CGFloat(minimumResolvableArea)))"
            )
            return false
        }

        let minorityCount = min(positiveCount, negativeCount)
        let minorityArea = min(positiveArea, negativeArea)
        let minorityAreaRatio = minorityArea / totalArea
        // When lips close or purse, the inner seam becomes a real fold in the
        // 2D projection. A small minority of triangles may therefore change
        // sign even though the outer and inner contours remain valid. The
        // stricter open-mouth limits still catch broad detector corruption.
        let isNearClosed = mouthOpeningRatio(contour) <= 0.12
        let maximumMinorityCount = isNearClosed ? 20 : 10
        let maximumMinorityAreaRatio: Float = isNearClosed ? 0.05 : 0.012
        let isConsistent = minorityCount <= maximumMinorityCount &&
            minorityAreaRatio <= maximumMinorityAreaRatio
        if !isConsistent {
            LipDebugLog.throttled(
                "lip_detection_triangle_topology",
                interval: 0.5,
                "lip_detection hold reason=triangle_fold positive=\(positiveCount) negative=\(negativeCount) minorityRatio=\(debugFloat(CGFloat(minorityAreaRatio)))"
            )
        }
        return isConsistent
    }

    private func bounds(for points: [CGPoint]) -> CGRect? {
        guard let first = points.first else {
            return nil
        }

        var minX = first.x
        var maxX = first.x
        var minY = first.y
        var maxY = first.y

        for point in points.dropFirst() {
            minX = min(minX, point.x)
            maxX = max(maxX, point.x)
            minY = min(minY, point.y)
            maxY = max(maxY, point.y)
        }

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private func motionCompensatedPose(_ pose: LipPose,
                                       from source: LipMotionPose,
                                       to target: LipMotionPose) -> LipPose {
        let rotation = target.angle - source.angle
        // LipContour.transformed applies the same bounds. The wider range is
        // intentional: ARKit follows camera-depth motion at render rate, while
        // the MediaPipe keyframe can be more than one tenth of a second old.
        let scale = min(max(target.width / max(source.width, 1), 0.55), 1.80)
        let offset = CGPoint(
            x: pose.center.x - source.center.x,
            y: pose.center.y - source.center.y
        )
        let cosine = cos(rotation)
        let sine = sin(rotation)
        let center = CGPoint(
            x: target.center.x + (offset.x * cosine - offset.y * sine) * scale,
            y: target.center.y + (offset.x * sine + offset.y * cosine) * scale
        )

        return LipPose(
            center: center,
            width: pose.width * scale,
            angle: pose.angle + rotation
        )
    }

    private func motionDelta(from source: LipMotionPose?, to target: LipMotionPose?) -> CGFloat {
        guard let source, let target else {
            return .greatestFiniteMagnitude
        }

        let poseDelta = poseMotionDelta(from: source, to: target)
        guard poseDelta.isFinite else {
            return .greatestFiniteMagnitude
        }

        let openingDelta = abs(target.opening - source.opening)
        let expressionDelta = max(
            abs(target.smile - source.smile),
            abs(target.pucker - source.pucker),
            abs(target.upperRaise - source.upperRaise),
            abs(target.lowerDrop - source.lowerDrop)
        )
        return poseDelta + openingDelta * 1.45 + expressionDelta * 0.30
    }

    private func expressionMotionDelta(from source: LipMotionPose?,
                                       to target: LipMotionPose?) -> CGFloat {
        guard let source, let target else {
            return 0
        }
        return max(
            abs(target.opening - source.opening),
            abs(target.smile - source.smile),
            abs(target.pucker - source.pucker),
            abs(target.upperRaise - source.upperRaise),
            abs(target.lowerDrop - source.lowerDrop)
        )
    }

    private func poseMotionDelta(from source: LipMotionPose?, to target: LipMotionPose?) -> CGFloat {
        guard let source, let target else {
            return .greatestFiniteMagnitude
        }

        let centerDelta = hypot(
            target.center.x - source.center.x,
            target.center.y - source.center.y
        ) / max(source.width, 1)
        let scaleDelta = abs(target.width / max(source.width, 1) - 1)
        let angleDelta = abs(Self.normalizedAngle(target.angle - source.angle))
        return centerDelta +
            scaleDelta * 0.65 +
            angleDelta * 0.35
    }

    private static func normalizedAngle(_ angle: CGFloat) -> CGFloat {
        var value = angle
        while value > .pi {
            value -= .pi * 2
        }
        while value < -.pi {
            value += .pi * 2
        }
        return value
    }

    private func isUsableFaceTransform(_ transform: TransformMatrix?) -> Bool {
        guard let transform else {
            return true
        }
        guard transform.rows == 4, transform.columns == 4 else {
            return false
        }

        for row in 0..<transform.rows {
            for column in 0..<transform.columns {
                let index = Int(row * transform.columns + column)
                if !transform.data[index].isFinite {
                    return false
                }
            }
        }
        return true
    }

    private func markLipShapeAccepted(capturedAt: CFTimeInterval) {
        motionLock.lock()
        lastAcceptedLipShapeTime = capturedAt
        motionLock.unlock()
    }

    private func markLipTextureDisplayed(sourceCapturedAt: CFTimeInterval) {
        motionLock.lock()
        lastDisplayedLipTextureTime = sourceCapturedAt
        motionLock.unlock()
    }

    private func clearLipShapeFreshness() {
        motionLock.lock()
        lastAcceptedLipShapeTime = nil
        lastDisplayedLipTextureTime = nil
        motionLock.unlock()
    }

    private func hasFreshLipShape() -> Bool {
        motionLock.lock()
        let timestamp = lastAcceptedLipShapeTime
        motionLock.unlock()

        guard let timestamp else {
            return false
        }
        let age = CACurrentMediaTime() - timestamp
        return age.isFinite &&
            age >= 0 &&
            age <= Self.maxMotionCompensationAge
    }

    private func hasFreshDisplayedLipTexture() -> Bool {
        motionLock.lock()
        let timestamp = lastDisplayedLipTextureTime
        motionLock.unlock()

        guard let timestamp else {
            return false
        }
        let age = CACurrentMediaTime() - timestamp
        return age.isFinite &&
            age >= 0 &&
            age <= Self.maxRealContourDisplayAge
    }

    private func handleMissedLipDetection() {
        missedDetectionCount += 1
        meshStateLock.lock()
        let retainedContour = latestMeshContour
        meshStateLock.unlock()
        let retainedCarrierCreatedAt = retainedContour?.surfaceCarrierCreatedAt
        let retainedAge = retainedCarrierCreatedAt.map {
            CACurrentMediaTime() - $0
        } ?? .greatestFiniteMagnitude
        let maximumRetainedCarrierAge = retainedContour?.surfaceCarrierSupportsDeformation == true ?
            Self.maxSurfaceCarrierHoldAge :
            Self.maxDepthOnlySurfaceCarrierHoldAge
        if let retainedContour,
           hasCompleteSurfaceCarrier(retainedContour),
           retainedAge.isFinite,
           retainedAge >= 0,
           retainedAge <= maximumRetainedCarrierAge,
           currentLipMeshAvailability().contour {
            missedDetectionCount = min(
                missedDetectionCount,
                Self.missedDetectionsBeforeReset + 1
            )
            LipDebugLog.throttled(
                "lip_surface_hold",
                interval: 0.8,
                "lip_surface hold reason=mediapipe_miss misses=\(missedDetectionCount)"
            )
            return
        }
        guard missedDetectionCount > Self.missedDetectionsBeforeReset else {
            return
        }

        resetLipTracking()
        requestRendererClear()
    }

    private func isActiveLandmarkContext(_ context: FrameContext) -> Bool {
        detectionLock.lock()
        let isActive = isDetectingLandmarks &&
            landmarkTrackingEpoch == context.trackingEpoch &&
            activeLandmarkTimestampInMilliseconds == context.timestampInMilliseconds
        detectionLock.unlock()
        return isActive
    }

    private func completeLandmarkDetection(timestampInMilliseconds: Int,
                                           trackingEpoch: Int) {
        var nextContext: FrameContext?
        detectionLock.lock()
        if activeLandmarkTimestampInMilliseconds == timestampInMilliseconds {
            isDetectingLandmarks = false
            activeLandmarkTimestampInMilliseconds = nil
            if let pending = latestPendingFrameContext,
               pending.trackingEpoch == landmarkTrackingEpoch,
               landmarkReadyEpoch == landmarkTrackingEpoch {
                nextContext = pending
                latestPendingFrameContext = nil
            } else if latestPendingFrameContext?.trackingEpoch != landmarkTrackingEpoch {
                latestPendingFrameContext = nil
            }
        }
        detectionLock.unlock()

        if let nextContext {
            offerLandmarkFrame(nextContext)
        }
    }

    private func currentLandmarkTrackingEpoch() -> Int {
        detectionLock.lock()
        let epoch = landmarkTrackingEpoch
        detectionLock.unlock()
        return epoch
    }

    @discardableResult
    private func invalidateLandmarkPipeline() -> Int {
        detectionLock.lock()
        landmarkTrackingEpoch &+= 1
        let epoch = landmarkTrackingEpoch
        landmarkReadyEpoch = nil
        latestPendingFrameContext = nil
        lastLandmarkSubmitTime = 0
        lastLandmarkSubmitPose = nil
        detectionLock.unlock()
        return epoch
    }

    private func markLandmarkPipelineReady(trackingEpoch: Int) {
        var pendingContext: FrameContext?
        detectionLock.lock()
        if landmarkTrackingEpoch == trackingEpoch {
            landmarkReadyEpoch = trackingEpoch
            if !isDetectingLandmarks,
               let pending = latestPendingFrameContext,
               pending.trackingEpoch == trackingEpoch {
                pendingContext = pending
                latestPendingFrameContext = nil
            }
        }
        detectionLock.unlock()

        if let pendingContext {
            offerLandmarkFrame(pendingContext)
        }
    }

    private func enqueueTextureRender(_ request: LipTextureRequest) {
        textureStateLock.lock()
        if isRenderingTexture {
            pendingTextureRequest = request
            textureStateLock.unlock()
            return
        }

        isRenderingTexture = true
        textureStateLock.unlock()

        textureQueue.async { [weak self] in
            self?.renderTextureRequest(request)
        }
    }

    private func shouldSubmitTextureRender(lowLatency: Bool, now: CFTimeInterval) -> Bool {
        textureStateLock.lock()
        defer {
            textureStateLock.unlock()
        }

        let minimumInterval = lowLatency ?
            Self.minLowLatencyTextureRenderInterval :
            Self.minTextureRenderInterval

        guard let lastTextureSubmitTime else {
            lastTextureSubmitTime = now
            return true
        }

        guard now - lastTextureSubmitTime >= minimumInterval else {
            return false
        }

        self.lastTextureSubmitTime = now
        return true
    }

    private func shouldSubmitStableTextureRender(mouthOpen: Bool,
                                                 hasTexture: Bool,
                                                 now: CFTimeInterval) -> Bool {
        textureStateLock.lock()
        defer {
            textureStateLock.unlock()
        }

        if isRenderingTexture {
            return false
        }

        if isLipTextureRefreshPending {
            lastTextureMouthOpen = mouthOpen
            pendingTextureMouthOpen = nil
            pendingTextureMouthOpenSince = 0
            lastTextureSubmitTime = now
            return true
        }

        guard hasTexture else {
            lastTextureMouthOpen = mouthOpen
            pendingTextureMouthOpen = nil
            pendingTextureMouthOpenSince = 0
            lastTextureSubmitTime = now
            return true
        }

        guard lastTextureMouthOpen != mouthOpen else {
            pendingTextureMouthOpen = nil
            pendingTextureMouthOpenSince = 0
            return false
        }

        if pendingTextureMouthOpen != mouthOpen {
            pendingTextureMouthOpen = mouthOpen
            pendingTextureMouthOpenSince = now
            return false
        }

        let requiredStableDuration: CFTimeInterval = mouthOpen ? 0.08 : 0.09
        guard now - pendingTextureMouthOpenSince >= requiredStableDuration else {
            return false
        }

        lastTextureMouthOpen = mouthOpen
        pendingTextureMouthOpen = nil
        pendingTextureMouthOpenSince = 0
        lastTextureSubmitTime = now
        return true
    }

    private func isTextureRenderInFlight() -> Bool {
        textureStateLock.lock()
        let isInFlight = isRenderingTexture
        textureStateLock.unlock()
        return isInFlight
    }

    private func renderTextureRequest(_ request: LipTextureRequest) {
        let textureStart = CACurrentMediaTime()
        let texture = lipTextureRenderer.makeTexture(
            contour: request.contour,
            pixelBuffer: request.pixelBuffer,
            imageSize: request.imageSize,
            viewportSize: request.viewportSize,
            renderScale: request.renderScale,
            excludesInnerMouth: request.excludesInnerMouth,
            lightingFactor: request.lightingFactor,
            lowLatency: request.lowLatency,
            motionDelta: request.motionDelta
        )
        textureFPS.tick(workMilliseconds: (CACurrentMediaTime() - textureStart) * 1000)

        let isTracked = currentTrackedFaceAnchor()
        let hasFreshShape = hasFreshLipShape()
        let currentGeneration = currentTextureGeneration()
        let latestRequestID = currentLatestTextureRequestID()
        let generationOK = currentGeneration == request.generation
        let isSuperseded = latestRequestID != request.requestID
        let now = CACurrentMediaTime()
        let renderRequestAge = now - request.createdAt
        let sourceAge = now - request.sourceCaptureTime
        let availability = currentLipMeshAvailability()
        let maxRequestAge = availability.texture ?
            Self.maxTextureRequestAgeWithTexture :
            Self.maxInitialTextureRequestAge
        let requestStillFresh = sourceAge.isFinite &&
            sourceAge >= 0 &&
            sourceAge <= maxRequestAge
        let temporalContextIsCurrent =
            request.trackingEpoch == currentLandmarkTrackingEpoch() &&
            request.anchorIdentifier == currentTrackedFaceAnchorIdentifier() &&
            request.viewportRevision == currentViewport().revision
        let requestMotionDelta = motionDelta(from: request.motionReference, to: currentLipMotionPose())
        // The canonical texture remains valid while the SceneKit mesh follows
        // the current AR pose. Accept a fresh open-mouth mask through stronger
        // motion so an older closed mask cannot stretch across the teeth.
        let requestStillAligned = requestMotionDelta < 0.80
        if let texture,
           isTracked,
           hasFreshShape,
           availability.contour,
           generationOK,
           !isSuperseded,
           temporalContextIsCurrent,
           requestStillFresh,
           requestStillAligned,
           installLatestLipTextureIfCurrent(texture, request: request) {
            markLipTextureDisplayed(sourceCapturedAt: request.sourceCaptureTime)
            LipDebugLog.throttled(
                "lip_texture_accept",
                interval: 0.6,
                "lip_texture accept stableUV=true requestID=\(request.requestID) latestRequestID=\(latestRequestID) sourceAge=\(String(format: "%.2f", sourceAge)) renderAge=\(String(format: "%.2f", renderRequestAge)) motion=\(String(format: "%.2f", requestMotionDelta)) generation=\(request.generation) lowLatency=\(request.lowLatency)"
            )
            DispatchQueue.main.async { [weak self] in
                self?.isFaceDetected = true
            }
        } else {
            LipDebugLog.throttled(
                "lip_texture_reject",
                interval: 0.6,
                "lip_texture reject stableUV=true hasTexture=\(texture != nil) tracked=\(isTracked) freshShape=\(hasFreshShape) hasContour=\(availability.contour) generationOK=\(generationOK) temporalContextOK=\(temporalContextIsCurrent) superseded=\(isSuperseded) freshRequestDiagnostic=\(requestStillFresh) alignedDiagnostic=\(requestStillAligned) sourceAge=\(String(format: "%.2f", sourceAge)) renderAge=\(String(format: "%.2f", renderRequestAge)) motion=\(String(format: "%.2f", requestMotionDelta)) requestID=\(request.requestID) latestRequestID=\(latestRequestID) requestGeneration=\(request.generation) currentGeneration=\(currentGeneration)"
            )
        }

        textureStateLock.lock()
        if let nextRequest = pendingTextureRequest {
            pendingTextureRequest = nil
            textureStateLock.unlock()
            textureQueue.async { [weak self] in
                self?.renderTextureRequest(nextRequest)
            }
        } else {
            isRenderingTexture = false
            textureStateLock.unlock()
        }
    }

    private func makeMediaPipeInputBuffer(from pixelBuffer: CVPixelBuffer,
                                          orientation: UIInterfaceOrientation) -> PreparedMediaPipeInput? {
        let imageOrientation = mediaPipeInputOrientation(for: orientation)
        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
            .oriented(forExifOrientation: Int32(imageOrientation.rawValue))
        let normalizedImage = sourceImage.transformed(
            by: CGAffineTransform(
                translationX: -sourceImage.extent.origin.x,
                y: -sourceImage.extent.origin.y
            )
        )
        let sourceWidth = max(normalizedImage.extent.width, 1)
        let sourceHeight = max(normalizedImage.extent.height, 1)
        let inputScale = min(1, Self.mediaPipeInputLongEdge / max(sourceWidth, sourceHeight))
        let inputImage = inputScale < 0.999 ?
            normalizedImage.transformed(by: CGAffineTransform(scaleX: inputScale, y: inputScale)) :
            normalizedImage
        let width = max(Int((sourceWidth * inputScale).rounded(.up)), 1)
        let height = max(Int((sourceHeight * inputScale).rounded(.up)), 1)

        var outputBuffer: CVPixelBuffer?
        let attributes: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
            kCVPixelBufferIOSurfacePropertiesKey: [:]
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attributes as CFDictionary,
            &outputBuffer
        )
        guard status == kCVReturnSuccess, let outputBuffer else {
            return nil
        }

        ciContext.render(
            inputImage,
            to: outputBuffer,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        return PreparedMediaPipeInput(
            pixelBuffer: outputBuffer,
            inputToCapturedImageTransform: inputToCapturedImageTransform(
                for: imageOrientation
            )
        )
    }

    private func mediaPipeInputOrientation(for interfaceOrientation: UIInterfaceOrientation) -> CGImagePropertyOrientation {
        switch interfaceOrientation {
        case .portrait:
            return .right
        case .portraitUpsideDown:
            return .left
        case .landscapeLeft:
            return .down
        case .landscapeRight:
            return .up
        case .unknown:
            return .right
        @unknown default:
            return .right
        }
    }

    private func inputToCapturedImageTransform(for orientation: CGImagePropertyOrientation) -> CGAffineTransform {
        switch orientation {
        case .up:
            return .identity
        case .right:
            return CGAffineTransform(a: 0, b: -1, c: 1, d: 0, tx: 0, ty: 1)
        case .down:
            return CGAffineTransform(a: -1, b: 0, c: 0, d: -1, tx: 1, ty: 1)
        case .left:
            return CGAffineTransform(a: 0, b: 1, c: -1, d: 0, tx: 1, ty: 0)
        default:
            return .identity
        }
    }

    private func mappedLipLandmarks(for indices: [Int],
                                    landmarks: [NormalizedLandmark],
                                    viewportSize: CGSize,
                                    inputToCapturedImageTransform: CGAffineTransform,
                                    capturedImageToViewportTransform: CGAffineTransform) -> (points: [CGPoint], points3D: [SIMD3<Float>], uv: [CGPoint]) {
        var points: [CGPoint] = []
        var points3D: [SIMD3<Float>] = []
        var uv: [CGPoint] = []
        points.reserveCapacity(indices.count)
        points3D.reserveCapacity(indices.count)
        uv.reserveCapacity(indices.count)

        for index in indices {
            guard landmarks.indices.contains(index),
                  let canonicalUV = CanonicalLipGeometry.normalizedUV(for: index) else {
                continue
            }

            let landmark = landmarks[index]
            guard landmark.x.isFinite,
                  landmark.y.isFinite,
                  landmark.z.isFinite else {
                continue
            }
            let viewportNormalizedPoint = CGPoint(
                x: CGFloat(landmark.x),
                y: CGFloat(landmark.y)
            )
            .applying(inputToCapturedImageTransform)
            .applying(capturedImageToViewportTransform)
            let point = CGPoint(
                x: viewportNormalizedPoint.x * viewportSize.width,
                y: viewportNormalizedPoint.y * viewportSize.height
            )
            guard point.x.isFinite, point.y.isFinite else {
                continue
            }

            points.append(point)
            points3D.append(SIMD3<Float>(landmark.x, landmark.y, landmark.z))
            uv.append(canonicalUV)
        }

        return (points, points3D, uv)
    }

    private func mappedLipMeshPoints(for indices: [Int],
                                     landmarks: [NormalizedLandmark],
                                     viewportSize: CGSize,
                                     inputToCapturedImageTransform: CGAffineTransform,
                                     capturedImageToViewportTransform: CGAffineTransform) -> [Int: LipMeshPoint] {
        var pointsByIndex: [Int: LipMeshPoint] = [:]
        pointsByIndex.reserveCapacity(indices.count)

        for index in indices {
            guard landmarks.indices.contains(index),
                  let canonicalUV = CanonicalLipGeometry.normalizedUV(for: index) else {
                continue
            }

            let landmark = landmarks[index]
            guard landmark.x.isFinite,
                  landmark.y.isFinite,
                  landmark.z.isFinite else {
                continue
            }
            let viewportNormalizedPoint = CGPoint(
                x: CGFloat(landmark.x),
                y: CGFloat(landmark.y)
            )
            .applying(inputToCapturedImageTransform)
            .applying(capturedImageToViewportTransform)
            let point = CGPoint(
                x: viewportNormalizedPoint.x * viewportSize.width,
                y: viewportNormalizedPoint.y * viewportSize.height
            )
            guard point.x.isFinite, point.y.isFinite else {
                continue
            }

            pointsByIndex[index] = LipMeshPoint(
                screen: point,
                normalized: SIMD3<Float>(landmark.x, landmark.y, landmark.z),
                uv: canonicalUV
            )
        }

        return pointsByIndex
    }


    private func adaptivelySmoothed(
        _ contour: LipContour,
        motionReference: LipMotionPose?
    ) -> LipContour {
        guard var previous = smoothedLipContour,
              previous.outer.count == contour.outer.count,
              previous.inner.count == contour.inner.count else {
            return contour
        }

        if shouldUseRawContour(
            previous: previous,
            current: contour,
            previousMotionPose: lastAcceptedMotionPose,
            currentMotionPose: motionReference
        ) {
            contourStabilityConfidence = 0
            contourIsStationary = false
            return contour
        }

        if let sourceMotionPose = lastAcceptedMotionPose,
           let targetMotionPose = motionReference {
            // Follow rigid head/camera motion with ARKit. Aligning to the raw
            // MediaPipe lip pose would copy its center/scale jitter into every
            // point before the low-pass filter can remove it.
            let sourcePose = LipPose(
                center: sourceMotionPose.center,
                width: sourceMotionPose.width,
                angle: sourceMotionPose.angle
            )
            let targetPose = LipPose(
                center: targetMotionPose.center,
                width: targetMotionPose.width,
                angle: targetMotionPose.angle
            )
            previous = previous.transformed(from: sourcePose, to: targetPose)
        } else if let sourcePose = smoothedLipPose,
                  let targetPose = contour.pose {
            previous = previous.transformed(from: sourcePose, to: targetPose)
        }

        let parameters = smoothingParameters(
            for: contour,
            alignedPrevious: previous
        )
        let alpha = parameters.alpha
        let outer = zip(previous.outer, contour.outer).map {
            smooth(
                previous: $0.0,
                current: $0.1,
                alpha: alpha,
                deadZone: parameters.deadZone
            )
        }
        let inner = zip(previous.inner, contour.inner).map {
            smooth(
                previous: $0.0,
                current: $0.1,
                alpha: alpha,
                deadZone: parameters.deadZone
            )
        }
        let outer3D = smoothDepth(
            previous: previous.outer3D,
            current: contour.outer3D,
            alpha: min(max(Float(alpha), 0), 1)
        )
        let inner3D = smoothDepth(
            previous: previous.inner3D,
            current: contour.inner3D,
            alpha: min(max(Float(alpha), 0), 1)
        )
        let meshAlpha = min(max(Float(alpha), 0), 1)
        var meshPoints = contour.meshPointsByIndex
        if previous.meshPointsByIndex.count == contour.meshPointsByIndex.count {
            for (index, currentPoint) in contour.meshPointsByIndex {
                guard let previousPoint = previous.meshPointsByIndex[index] else {
                    continue
                }

                meshPoints[index] = LipMeshPoint(
                    screen: smooth(
                        previous: previousPoint.screen,
                        current: currentPoint.screen,
                        alpha: alpha,
                        deadZone: parameters.deadZone
                    ),
                    normalized: previousPoint.normalized + (currentPoint.normalized - previousPoint.normalized) * meshAlpha,
                    uv: currentPoint.uv
                )
            }
        }
        LipDebugLog.throttled(
            "lip_contour_stability",
            interval: 0.75,
            "lip_stability state=\(parameters.isStationary ? "idle" : "adaptive") stability=\(debugFloat(parameters.stability)) activity=\(debugFloat(parameters.activity)) alpha=\(debugFloat(parameters.alpha)) deadZonePx=\(debugFloat(parameters.deadZone)) medianPx=\(debugFloat(parameters.medianResidual)) p95Px=\(debugFloat(parameters.p95Residual))"
        )
        return LipContour(
            outer: outer,
            inner: inner,
            outer3D: outer3D,
            inner3D: inner3D,
            outerUV: contour.outerUV,
            innerUV: contour.innerUV,
            meshPointsByIndex: meshPoints,
            faceGeometryPose: contour.faceGeometryPose ?? previous.faceGeometryPose,
            isStationary: parameters.isStationary
        )
    }

    private func shouldUseRawContour(
        previous: LipContour?,
        current: LipContour,
        previousMotionPose: LipMotionPose?,
        currentMotionPose: LipMotionPose?
    ) -> Bool {
        guard var comparablePrevious = previous else {
            return true
        }
        if let previousMotionPose,
           let currentMotionPose {
            comparablePrevious = comparablePrevious.transformed(
                from: LipPose(
                    center: previousMotionPose.center,
                    width: previousMotionPose.width,
                    angle: previousMotionPose.angle
                ),
                to: LipPose(
                    center: currentMotionPose.center,
                    width: currentMotionPose.width,
                    angle: currentMotionPose.angle
                )
            )
        }
        guard let previousBounds = comparablePrevious.bounds,
              let currentBounds = current.bounds,
              previousBounds.width > 1,
              previousBounds.height > 1 else {
            return true
        }

        let widthChange = abs(currentBounds.width / previousBounds.width - 1)
        let heightChange = abs(currentBounds.height / previousBounds.height - 1)
        let openingChange = abs(
            mouthOpeningRatio(current) - mouthOpeningRatio(comparablePrevious)
        )
        let widthRatio = currentBounds.width / max(previousBounds.width, 1)
        let heightRatio = currentBounds.height / max(previousBounds.height, 1)

        if widthRatio > 1.20 || heightRatio > 1.26 {
            return true
        }

        // Moderate expressions stay on the adaptive path; returning raw points
        // for a 2% aperture change made detector noise bypass all smoothing.
        return widthChange > 0.120 ||
            heightChange > 0.180 ||
            openingChange > 0.055
    }

    private func mouthOpeningRatio(_ contour: LipContour) -> CGFloat {
        guard let outerBounds = bounds(for: contour.outer),
              let innerBounds = bounds(for: contour.inner),
              outerBounds.width > 1 else {
            return 0
        }

        return innerBounds.height / outerBounds.width
    }

    private func smoothingParameters(
        for contour: LipContour,
        alignedPrevious: LipContour
    ) -> ContourSmoothingParameters {
        guard let currentPose = contour.pose else {
            contourIsStationary = false
            contourStabilityConfidence = 0
            return ContourSmoothingParameters(
                alpha: 0.78,
                deadZone: 0,
                isStationary: false,
                stability: 0,
                activity: 1,
                medianResidual: 0,
                p95Residual: 0
            )
        }

        var residuals: [CGFloat] = []
        residuals.reserveCapacity(Self.attentionLipIndices.count)
        for landmarkIndex in Self.attentionLipIndices {
            guard let previousPoint = alignedPrevious.meshPointsByIndex[landmarkIndex]?.screen,
                  let currentPoint = contour.meshPointsByIndex[landmarkIndex]?.screen else {
                continue
            }
            let residual = hypot(
                currentPoint.x - previousPoint.x,
                currentPoint.y - previousPoint.y
            )
            if residual.isFinite {
                residuals.append(residual)
            }
        }
        residuals.sort()
        let medianResidual = residuals.isEmpty ? 0 : residuals[residuals.count / 2]
        let p95Index = min(
            max(Int(ceil(CGFloat(residuals.count) * 0.95)) - 1, 0),
            max(residuals.count - 1, 0)
        )
        let p95Residual = residuals.isEmpty ? 0 : residuals[p95Index]

        let openingMotion = abs(
            mouthOpeningRatio(contour) - mouthOpeningRatio(alignedPrevious)
        )

        func smoothActivity(_ value: CGFloat, quiet: CGFloat, active: CGFloat) -> CGFloat {
            let t = min(max((value - quiet) / max(active - quiet, 0.000_1), 0), 1)
            return t * t * (3 - 2 * t)
        }

        // Residuals are measured after ARKit has removed rigid head/camera
        // motion. They are a better stability signal than a hard pose/opening
        // threshold, which previously toggled moving/idle even at sub-pixel
        // residuals. Aperture change remains an early expression signal.
        // MediaPipe's normalized jitter becomes a larger pixel displacement
        // as the face approaches the camera. Scale the activity thresholds by
        // the detected mouth width so the same physical noise receives the
        // same smoothing at every distance.
        let residualScale = min(max(currentPose.width / 80, 0.75), 2.50)
        let medianActivity = smoothActivity(
            medianResidual,
            quiet: 0.25 * residualScale,
            active: 1.65 * residualScale
        )
        let p95Activity = smoothActivity(
            p95Residual,
            quiet: 0.65 * residualScale,
            active: 3.60 * residualScale
        )
        let openingActivity = smoothActivity(
            openingMotion,
            quiet: 0.006,
            active: 0.055
        )
        let activity = max(medianActivity, p95Activity, openingActivity)
        let targetStability = 1 - activity
        // Leave a stable state immediately when real motion appears, but build
        // confidence gradually so one quiet detector sample cannot freeze an
        // actively changing expression.
        let confidenceGain: CGFloat = targetStability < contourStabilityConfidence ?
            0.82 : 0.30
        contourStabilityConfidence +=
            (targetStability - contourStabilityConfidence) * confidenceGain
        contourStabilityConfidence = min(max(contourStabilityConfidence, 0), 1)

        if contourIsStationary {
            if contourStabilityConfidence < 0.32 {
                contourIsStationary = false
            }
        } else if contourStabilityConfidence > 0.68 {
            contourIsStationary = true
        }

        // Alpha now changes continuously. Even before the idle hysteresis has
        // engaged, a quiet residual never falls back to the old near-raw 0.99.
        let alpha = min(
            max(
                0.18 + activity * 0.43 +
                    (1 - contourStabilityConfidence) * 0.07,
                0.18
            ),
            0.68
        )
        let maximumDeadZone = min(
            max(currentPose.width * 0.0036, 0.26),
            0.85
        )
        let deadZone = maximumDeadZone * contourStabilityConfidence
        return ContourSmoothingParameters(
            alpha: alpha,
            deadZone: deadZone,
            isStationary: contourIsStationary,
            stability: contourStabilityConfidence,
            activity: activity,
            medianResidual: medianResidual,
            p95Residual: p95Residual
        )
    }

    private func smooth(
        previous: CGPoint,
        current: CGPoint,
        alpha: CGFloat,
        deadZone: CGFloat = 0
    ) -> CGPoint {
        let dx = current.x - previous.x
        let dy = current.y - previous.y
        let distance = hypot(dx, dy)
        guard distance.isFinite,
              distance > deadZone else {
            return previous
        }
        let retainedDistance = distance - deadZone
        let scale = retainedDistance / distance * alpha
        return CGPoint(
            x: previous.x + dx * scale,
            y: previous.y + dy * scale
        )
    }

    private func smoothDepth(previous: [SIMD3<Float>],
                             current: [SIMD3<Float>],
                             alpha: Float) -> [SIMD3<Float>] {
        guard previous.count == current.count else {
            return current
        }

        return zip(previous, current).map {
            $0.0 + ($0.1 - $0.0) * alpha
        }
    }

    private func aspectFillTransform(for imageSize: CGSize, in viewportSize: CGSize) -> CGAffineTransform {
        guard imageSize.width > 0,
              imageSize.height > 0,
              viewportSize.width > 0,
              viewportSize.height > 0 else {
            return .identity
        }

        let scale = max(viewportSize.width / imageSize.width, viewportSize.height / imageSize.height)
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        let tx = (viewportSize.width - scaledWidth) * 0.5
        let ty = (viewportSize.height - scaledHeight) * 0.5
        return CGAffineTransform(translationX: tx, y: ty).scaledBy(x: scale, y: scale)
    }

    private func localMouthFrame(from faceAnchor: ARFaceAnchor) -> FaceLocalMouthFrame? {
        let vertices = faceAnchor.geometry.vertices
        let requiredIndex = max(
            Self.arKitMouthLeftIndex,
            Self.arKitMouthRightIndex,
            Self.arKitMouthTopIndex,
            Self.arKitMouthBottomIndex
        )
        guard vertices.count > requiredIndex else {
            return nil
        }

        let left = vertices[Self.arKitMouthLeftIndex]
        let right = vertices[Self.arKitMouthRightIndex]
        let top = vertices[Self.arKitMouthTopIndex]
        let bottom = vertices[Self.arKitMouthBottomIndex]
        let widthVector = right - left
        let width = simd_length(widthVector)
        guard width.isFinite, width > 0.008 else {
            return nil
        }
        let openingRatio = simd_length(bottom - top) / max(width, 0.0001)
        mouthStateLock.lock()
        if openingRatio < 0.34 {
            if let previousNeutral = neutralMouthWidth {
                neutralMouthWidth = min(previousNeutral * 0.96 + width * 0.04, width)
            } else {
                neutralMouthWidth = width
            }
        }
        let referenceWidth = neutralMouthWidth.map { min(max($0, width * 0.72), width) } ?? width
        mouthStateLock.unlock()
        let smileExpansion = max(0, min(width / max(referenceWidth, 0.0001) - 1, 1.2))

        let xAxis = simd_normalize(widthVector)
        var downVector = SIMD3<Float>(0, -1, 0)
        downVector -= xAxis * simd_dot(downVector, xAxis)
        guard simd_length(downVector) > 0.002 else {
            return nil
        }

        let downAxis = simd_normalize(downVector)
        var normalAxis = simd_cross(xAxis, downAxis)
        guard simd_length(normalAxis) > 0.0001 else {
            return nil
        }
        normalAxis = simd_normalize(normalAxis)
        if normalAxis.z < 0 {
            normalAxis = -normalAxis
        }

        let verticalAlignment: Float = 0
        let center = (left + right + top + bottom) * 0.25
        guard center.x.isFinite,
              center.y.isFinite,
              center.z.isFinite else {
            return nil
        }

        let blendShapes = faceAnchor.blendShapes
        let jawOpen = max(
            Float(openingRatio),
            FaceBlendShapeValue.value(.jawOpen, in: blendShapes)
        )
        let smileLeft = max(
            FaceBlendShapeValue.value(.mouthSmileLeft, in: blendShapes),
            FaceBlendShapeValue.value(.mouthDimpleLeft, in: blendShapes) * 0.55
        )
        let smileRight = max(
            FaceBlendShapeValue.value(.mouthSmileRight, in: blendShapes),
            FaceBlendShapeValue.value(.mouthDimpleRight, in: blendShapes) * 0.55
        )
        let upperLipRaise = max(
            FaceBlendShapeValue.value(.mouthUpperUpLeft, in: blendShapes),
            FaceBlendShapeValue.value(.mouthUpperUpRight, in: blendShapes)
        )
        let lowerLipDrop = max(
            FaceBlendShapeValue.value(.mouthLowerDownLeft, in: blendShapes),
            FaceBlendShapeValue.value(.mouthLowerDownRight, in: blendShapes)
        )
        let mouthPucker = FaceBlendShapeValue.value(.mouthPucker, in: blendShapes)
        let mouthFunnel = FaceBlendShapeValue.value(.mouthFunnel, in: blendShapes)

        return FaceLocalMouthFrame(
            center: center,
            xAxis: xAxis,
            downAxis: downAxis,
            normalAxis: normalAxis,
            width: width,
            smileExpansion: smileExpansion,
            openingRatio: openingRatio,
            referenceWidth: referenceWidth,
            verticalAlignment: verticalAlignment,
            jawOpen: jawOpen,
            smileLeft: smileLeft,
            smileRight: smileRight,
            upperLipRaise: upperLipRaise,
            lowerLipDrop: lowerLipDrop,
            mouthPucker: mouthPucker,
            mouthFunnel: mouthFunnel
        )
    }

    private func preferredFaceSurfaceTriangleKeys(
        for input: FaceSurfaceProjectionInput
    ) -> Set<FaceSurfaceTriangleKey>? {
        guard let contour = smoothedLipContour,
              hasCompleteSurfaceCarrier(contour),
              let carrierCreatedAt = contour.surfaceCarrierCreatedAt else {
            return nil
        }
        let carrierAge = CACurrentMediaTime() - carrierCreatedAt
        let maximumIndexedAge = contour.surfaceCarrierSupportsDeformation ?
            Self.maxIndexedSurfaceSnapshotAge :
            Self.maxDepthOnlySurfaceCarrierHoldAge
        let currentTriangleIndices = input.geometry.triangleIndices
        guard carrierAge.isFinite,
              carrierAge >= 0,
              carrierAge <= maximumIndexedAge,
              contour.surfaceVertexCount == input.geometry.vertices.count,
              contour.surfaceTriangleIndexCount == currentTriangleIndices.count,
              contour.surfaceTopologySignature == faceSurfaceTopologySignature(
                  vertexCount: input.geometry.vertices.count,
                  triangleIndices: currentTriangleIndices
              ) else {
            return nil
        }
        let keys = Set(contour.surfaceBindingsByIndex.values.map {
            FaceSurfaceTriangleKey($0.vertexIndices)
        })
        return keys.count >= 8 ? keys : nil
    }

    private func makeFaceSurfaceSnapshot(
        from input: FaceSurfaceProjectionInput,
        preferredTriangleKeys: Set<FaceSurfaceTriangleKey>?
    ) -> FaceSurfaceSnapshot? {
        guard input.viewportSize.width > 1,
              input.viewportSize.height > 1,
              input.orientation != .unknown else {
            return nil
        }

        let vertices = input.geometry.vertices
        let triangleIndices = input.geometry.triangleIndices
        guard triangleIndices.count >= 3,
              triangleIndices.count.isMultiple(of: 3),
              let mouthFrame = faceSurfaceMouthFrame(vertices: vertices) else {
            return nil
        }
        let topologySignature = faceSurfaceTopologySignature(
            vertexCount: vertices.count,
            triangleIndices: triangleIndices
        )
        guard let topology = faceSurfaceTopology(
            vertices: vertices,
            triangleIndices: triangleIndices,
            mouthFrame: mouthFrame,
            topologySignature: topologySignature,
            anchorIdentifier: input.anchorIdentifier
        ) else {
            return nil
        }
        let trianglesToProject: [SIMD3<Int32>]
        let usesPreferredTriangleSubset: Bool
        if let preferredTriangleKeys {
            let directlyBoundTriangles = topology.triangles.filter {
                preferredTriangleKeys.contains(FaceSurfaceTriangleKey($0))
            }
            var boundVertexIndices = Set<Int32>()
            for triangle in directlyBoundTriangles {
                boundVertexIndices.insert(triangle.x)
                boundVertexIndices.insert(triangle.y)
                boundVertexIndices.insert(triangle.z)
            }
            // Exact binding triangles are enough only while the expression is
            // unchanged. During a pucker a landmark legitimately crosses onto
            // a neighbouring ARFaceGeometry triangle. Include that one-ring
            // neighbourhood so the exhaustive fallback can rebind locally
            // instead of watching the indexed carrier decay to 8-12 faces.
            let localTriangleNeighborhood = topology.triangles.filter {
                boundVertexIndices.contains($0.x) ||
                    boundVertexIndices.contains($0.y) ||
                    boundVertexIndices.contains($0.z)
            }
            if directlyBoundTriangles.count >= 8,
               localTriangleNeighborhood.count >= 24 {
                trianglesToProject = localTriangleNeighborhood
                usesPreferredTriangleSubset = true
            } else {
                trianglesToProject = topology.triangles
                usesPreferredTriangleSubset = false
            }
        } else {
            trianglesToProject = topology.triangles
            usesPreferredTriangleSubset = false
        }

        let viewMatrix = input.camera.viewMatrix(for: input.orientation)
        let extendedViewport = CGRect(
            x: -input.viewportSize.width * 0.20,
            y: -input.viewportSize.height * 0.20,
            width: input.viewportSize.width * 1.40,
            height: input.viewportSize.height * 1.40
        )
        let cameraColumn = input.camera.transform.columns.3
        let cameraPosition = SIMD3<Float>(cameraColumn.x, cameraColumn.y, cameraColumn.z)
        var projectedVertices: [Int: (point: CGPoint, depth: Float, world: SIMD3<Float>)] = [:]
        projectedVertices.reserveCapacity(trianglesToProject.count * 2)

        func projectedVertex(at index: Int) -> (point: CGPoint, depth: Float, world: SIMD3<Float>)? {
            if let cached = projectedVertices[index] {
                return cached
            }
            guard vertices.indices.contains(index) else {
                return nil
            }

            let local = vertices[index]
            let world = input.anchorTransform * SIMD4<Float>(local.x, local.y, local.z, 1)
            guard world.x.isFinite,
                  world.y.isFinite,
                  world.z.isFinite,
                  world.w.isFinite,
                  abs(world.w) > 0.000_001 else {
                return nil
            }

            let worldPoint = SIMD3<Float>(
                world.x / world.w,
                world.y / world.w,
                world.z / world.w
            )
            let cameraPoint = viewMatrix * SIMD4<Float>(worldPoint.x, worldPoint.y, worldPoint.z, 1)
            guard cameraPoint.z.isFinite,
                  cameraPoint.w.isFinite,
                  abs(cameraPoint.w) > 0.000_001 else {
                return nil
            }
            let depth = -cameraPoint.z / cameraPoint.w
            guard depth.isFinite, depth > 0.01 else {
                return nil
            }

            let screenPoint = input.camera.projectPoint(
                worldPoint,
                orientation: input.orientation,
                viewportSize: input.viewportSize
            )
            guard screenPoint.x.isFinite,
                  screenPoint.y.isFinite,
                  extendedViewport.contains(screenPoint) else {
                return nil
            }

            let projected = (point: screenPoint, depth: depth, world: worldPoint)
            projectedVertices[index] = projected
            return projected
        }

        var projectedTriangles: [ProjectedFaceSurfaceTriangle] = []
        projectedTriangles.reserveCapacity(trianglesToProject.count)
        for triangle in trianglesToProject {
            let firstIndex = Int(triangle.x)
            let secondIndex = Int(triangle.y)
            let thirdIndex = Int(triangle.z)
            guard let first = projectedVertex(at: firstIndex),
                  let second = projectedVertex(at: secondIndex),
                  let third = projectedVertex(at: thirdIndex) else {
                continue
            }

            if !usesPreferredTriangleSubset {
                var localNormal = simd_cross(
                    vertices[secondIndex] - vertices[firstIndex],
                    vertices[thirdIndex] - vertices[firstIndex]
                )
                guard simd_length_squared(localNormal) > 0.000_000_000_1 else {
                    continue
                }
                localNormal = simd_normalize(localNormal)
                let worldNormal4 = input.anchorTransform * SIMD4<Float>(
                    localNormal.x,
                    localNormal.y,
                    localNormal.z,
                    0
                )
                var worldNormal = SIMD3<Float>(worldNormal4.x, worldNormal4.y, worldNormal4.z)
                let worldCentroid = (first.world + second.world + third.world) / 3
                var directionToCamera = cameraPosition - worldCentroid
                guard simd_length_squared(worldNormal) > 0.000_001,
                      simd_length_squared(directionToCamera) > 0.000_001 else {
                    continue
                }
                worldNormal = simd_normalize(worldNormal)
                directionToCamera = simd_normalize(directionToCamera)
                // ARFaceGeometry contains mixed winding around the folded lip
                // seam. The old global winding sign occasionally classified
                // nearly every valid mouth triangle as back-facing, reducing a
                // full snapshot from hundreds of triangles to 2-8. Projection
                // and binding validity do not depend on winding direction.
                guard abs(simd_dot(worldNormal, directionToCamera)) > 0.005 else {
                    continue
                }
            }

            let twiceArea =
                (second.point.x - first.point.x) * (third.point.y - first.point.y) -
                (second.point.y - first.point.y) * (third.point.x - first.point.x)
            guard twiceArea.isFinite,
                  usesPreferredTriangleSubset || abs(twiceArea) > 0.01 else {
                continue
            }
            let localCentroid = (
                vertices[firstIndex] + vertices[secondIndex] + vertices[thirdIndex]
            ) / 3
            let normalizedMouthY = simd_dot(
                localCentroid - mouthFrame.center,
                mouthFrame.downAxis
            ) / mouthFrame.width
            guard normalizedMouthY.isFinite else {
                continue
            }

            projectedTriangles.append(
                ProjectedFaceSurfaceTriangle(
                    vertexIndices: triangle,
                    first: first.point,
                    second: second.point,
                    third: third.point,
                    cameraDepths: SIMD3<Float>(first.depth, second.depth, third.depth),
                    normalizedMouthY: normalizedMouthY
                )
            )
        }

        guard !projectedTriangles.isEmpty else {
            return nil
        }
        let projectedTrianglesByKey = Dictionary(
            projectedTriangles.map {
                (FaceSurfaceTriangleKey($0.vertexIndices), $0)
            },
            uniquingKeysWith: { first, _ in first }
        )
        return FaceSurfaceSnapshot(
            vertexCount: vertices.count,
            triangleIndexCount: triangleIndices.count,
            topologySignature: topologySignature,
            sourceVertices: vertices,
            mouthFrame: mouthFrame,
            triangles: projectedTriangles,
            trianglesByKey: projectedTrianglesByKey
        )
    }

    private func faceSurfaceTopology(vertices: [SIMD3<Float>],
                                     triangleIndices: [Int16],
                                     mouthFrame: FaceSurfaceMouthFrame,
                                     topologySignature: UInt64,
                                     anchorIdentifier: UUID) -> FaceSurfaceTopologyCache? {
        if faceSurfaceTopologyAnchorIdentifier != anchorIdentifier {
            faceSurfaceTopologyAnchorIdentifier = anchorIdentifier
            shouldExpandFaceSurfaceTopology = false
            faceSurfaceTopologyCache = nil
        }
        if let cached = faceSurfaceTopologyCache,
           cached.vertexCount == vertices.count,
           cached.triangleIndexCount == triangleIndices.count,
           cached.topologySignature == topologySignature,
           cached.anchorIdentifier == anchorIdentifier {
            return cached
        }

        func normalizedMouthPoint(_ point: SIMD3<Float>) -> SIMD3<Float>? {
            let relative = point - mouthFrame.center
            let normalized = SIMD3<Float>(
                simd_dot(relative, mouthFrame.xAxis) / mouthFrame.width,
                simd_dot(relative, mouthFrame.downAxis) / mouthFrame.width,
                simd_dot(relative, mouthFrame.normalAxis) / mouthFrame.width
            )
            guard normalized.x.isFinite,
                  normalized.y.isFinite,
                  normalized.z.isFinite else {
                return nil
            }
            return normalized
        }

        func overlapsMouthRegion(_ a: SIMD3<Float>,
                                 _ b: SIMD3<Float>,
                                 _ c: SIMD3<Float>) -> Bool {
            guard let first = normalizedMouthPoint(a),
                  let second = normalizedMouthPoint(b),
                  let third = normalizedMouthPoint(c) else {
                return false
            }
            let minX = min(first.x, second.x, third.x)
            let maxX = max(first.x, second.x, third.x)
            let minY = min(first.y, second.y, third.y)
            let maxY = max(first.y, second.y, third.y)
            let minZ = min(first.z, second.z, third.z)
            let maxZ = max(first.z, second.z, third.z)
            let horizontalLimit: Float = shouldExpandFaceSurfaceTopology ? 0.80 : 0.72
            let verticalLimit: Float = shouldExpandFaceSurfaceTopology ? 0.55 : 0.42
            let depthLimit: Float = shouldExpandFaceSurfaceTopology ? 0.50 : 0.38
            return minX <= horizontalLimit && maxX >= -horizontalLimit &&
                minY <= verticalLimit && maxY >= -verticalLimit &&
                minZ <= depthLimit && maxZ >= -depthLimit
        }

        var candidateTriangles: [SIMD3<Int32>] = []
        candidateTriangles.reserveCapacity(triangleIndices.count / 3)
        for offset in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let firstIndex = Int(triangleIndices[offset])
            let secondIndex = Int(triangleIndices[offset + 1])
            let thirdIndex = Int(triangleIndices[offset + 2])
            guard vertices.indices.contains(firstIndex),
                  vertices.indices.contains(secondIndex),
                  vertices.indices.contains(thirdIndex) else {
                continue
            }
            candidateTriangles.append(
                SIMD3<Int32>(Int32(firstIndex), Int32(secondIndex), Int32(thirdIndex))
            )
        }

        var reachedVertices = Set([
            Self.arKitMouthLeftIndex,
            Self.arKitMouthRightIndex,
            Self.arKitMouthTopIndex,
            Self.arKitMouthBottomIndex
        ])
        var selectedTriangleIndices = Set<Int>()
        for _ in 0..<8 {
            var expandedVertices = reachedVertices
            let previousTriangleCount = selectedTriangleIndices.count
            for (triangleOffset, triangle) in candidateTriangles.enumerated()
                where !selectedTriangleIndices.contains(triangleOffset) {
                let firstIndex = Int(triangle.x)
                let secondIndex = Int(triangle.y)
                let thirdIndex = Int(triangle.z)
                guard reachedVertices.contains(firstIndex) ||
                        reachedVertices.contains(secondIndex) ||
                        reachedVertices.contains(thirdIndex),
                      overlapsMouthRegion(
                    vertices[firstIndex],
                    vertices[secondIndex],
                    vertices[thirdIndex]
                  ) else {
                    continue
                }
                selectedTriangleIndices.insert(triangleOffset)
                expandedVertices.insert(firstIndex)
                expandedVertices.insert(secondIndex)
                expandedVertices.insert(thirdIndex)
            }
            reachedVertices = expandedVertices
            if selectedTriangleIndices.count == previousTriangleCount {
                break
            }
        }

        var mouthTriangles = shouldExpandFaceSurfaceTopology ?
            candidateTriangles.filter { triangle in
                overlapsMouthRegion(
                    vertices[Int(triangle.x)],
                    vertices[Int(triangle.y)],
                    vertices[Int(triangle.z)]
                )
            } :
            selectedTriangleIndices.sorted().map { candidateTriangles[$0] }
        if mouthTriangles.count < 12 {
            mouthTriangles = candidateTriangles.filter { triangle in
                overlapsMouthRegion(
                    vertices[Int(triangle.x)],
                    vertices[Int(triangle.y)],
                    vertices[Int(triangle.z)]
                )
            }
        }
        guard mouthTriangles.count >= 12 else {
            return nil
        }
        let cache = FaceSurfaceTopologyCache(
            vertexCount: vertices.count,
            triangleIndexCount: triangleIndices.count,
            topologySignature: topologySignature,
            anchorIdentifier: anchorIdentifier,
            triangles: mouthTriangles
        )
        faceSurfaceTopologyCache = cache
        LipDebugLog.throttled(
            "lip_surface_topology",
            interval: 2,
            "lip_surface topology vertices=\(vertices.count) mouthTriangles=\(mouthTriangles.count) expanded=\(shouldExpandFaceSurfaceTopology)"
        )
        return cache
    }

    private func faceSurfaceMouthFrame(vertices: [SIMD3<Float>]) -> FaceSurfaceMouthFrame? {
        let requiredIndex = max(
            Self.arKitMouthLeftIndex,
            Self.arKitMouthRightIndex,
            Self.arKitMouthTopIndex,
            Self.arKitMouthBottomIndex
        )
        guard vertices.count > requiredIndex else {
            return nil
        }

        let left = vertices[Self.arKitMouthLeftIndex]
        let right = vertices[Self.arKitMouthRightIndex]
        let top = vertices[Self.arKitMouthTopIndex]
        let bottom = vertices[Self.arKitMouthBottomIndex]
        let widthVector = right - left
        let mouthWidth = simd_length(widthVector)
        guard mouthWidth.isFinite, mouthWidth > 0.008 else {
            return nil
        }

        let xAxis = simd_normalize(widthVector)
        var downVector = bottom - top
        downVector -= xAxis * simd_dot(downVector, xAxis)
        if simd_length(downVector) < 0.002 {
            downVector = SIMD3<Float>(0, -1, 0)
            downVector -= xAxis * simd_dot(downVector, xAxis)
        }
        guard simd_length(downVector) > 0.000_1 else {
            return nil
        }
        let downAxis = simd_normalize(downVector)
        var normalAxis = simd_cross(xAxis, downAxis)
        guard simd_length(normalAxis) > 0.000_1 else {
            return nil
        }
        normalAxis = simd_normalize(normalAxis)
        if normalAxis.z < 0 {
            normalAxis = -normalAxis
        }

        let center = (left + right + top + bottom) * 0.25
        guard center.x.isFinite, center.y.isFinite, center.z.isFinite else {
            return nil
        }
        return FaceSurfaceMouthFrame(
            center: center,
            xAxis: xAxis,
            downAxis: downAxis,
            normalAxis: normalAxis,
            width: mouthWidth
        )
    }

    private func makeRenderLipMotionSample(
        for faceAnchor: ARFaceAnchor,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode
    ) -> LipMotionSample? {
        guard faceAnchor.isTracked else {
            return nil
        }

        let viewport = currentViewport()
        let renderedAt = CACurrentMediaTime()
        let trackingEpoch = currentLandmarkTrackingEpoch()
        guard viewport.size.width > 1,
              viewport.size.height > 1,
              currentTrackedFaceAnchorIdentifier() == faceAnchor.identifier,
              let pose = projectedRenderLipMotionPose(
                  from: faceAnchor,
                  renderer: renderer,
                  faceNode: faceNode,
                  viewportSize: viewport.size
              ) else {
            return nil
        }

        let sample = LipMotionSample(
            pose: pose,
            sampledAt: renderedAt,
            frameTimestamp: renderedAt,
            trackingEpoch: trackingEpoch,
            anchorIdentifier: faceAnchor.identifier,
            viewportRevision: viewport.revision
        )
        return displayCompensatedRenderMotionSample(sample)
    }

    private func displayCompensatedRenderMotionSample(
        _ sample: LipMotionSample
    ) -> LipMotionSample {
        motionLock.lock()
        let previousSample = latestRenderMotionSample
        let previousDelta = latestRenderCenterDelta
        let previousLead = latestRenderPredictionLead
        latestRenderMotionSample = sample

        guard let previousSample,
              previousSample.trackingEpoch == sample.trackingEpoch,
              previousSample.anchorIdentifier == sample.anchorIdentifier,
              previousSample.viewportRevision == sample.viewportRevision else {
            latestRenderCenterDelta = nil
            latestRenderPredictionLead = .zero
            motionLock.unlock()
            return sample
        }

        let sampleInterval = sample.sampledAt - previousSample.sampledAt
        let currentDelta = CGVector(
            dx: sample.pose.center.x - previousSample.pose.center.x,
            dy: sample.pose.center.y - previousSample.pose.center.y
        )
        let timingIsContinuous = sampleInterval.isFinite &&
            sampleInterval >= 0.008 &&
            sampleInterval <= 0.060 &&
            currentDelta.dx.isFinite &&
            currentDelta.dy.isFinite
        guard timingIsContinuous else {
            latestRenderCenterDelta = nil
            latestRenderPredictionLead = .zero
            motionLock.unlock()
            return sample
        }

        let currentDistance = hypot(currentDelta.dx, currentDelta.dy)
        let motionNoiseFloor = min(
            max(sample.pose.width * 0.0015, 0.12),
            0.35
        )

        var regularizedDelta = currentDelta
        var directionConfidence: CGFloat = previousDelta == nil ? 0.55 : 1
        if let previousDelta {
            let previousDistance = hypot(previousDelta.dx, previousDelta.dy)
            if previousDistance > motionNoiseFloor {
                let directionCosine = min(
                    max(
                        (currentDelta.dx * previousDelta.dx +
                            currentDelta.dy * previousDelta.dy) /
                            max(currentDistance * previousDistance, 0.000_1),
                        -1
                    ),
                    1
                )
                let directionProgress = min(
                    max((directionCosine - 0.15) / 0.65, 0),
                    1
                )
                directionConfidence = directionProgress * directionProgress *
                    (3 - 2 * directionProgress)
                regularizedDelta = CGVector(
                    dx: currentDelta.dx * 0.65 + previousDelta.dx * 0.35,
                    dy: currentDelta.dy * 0.65 + previousDelta.dy * 0.35
                )
            } else {
                directionConfidence = 0.55
            }
        }
        latestRenderCenterDelta = regularizedDelta

        var requestedLead = CGVector.zero
        if currentDistance > motionNoiseFloor,
           directionConfidence > 0.01 {
            let predictionHorizon = min(
                sampleInterval * 0.58,
                Self.maxRigidDisplayPredictionHorizon
            )
            let predictionFactor = CGFloat(predictionHorizon / sampleInterval) *
                directionConfidence
            requestedLead = CGVector(
                dx: regularizedDelta.dx * predictionFactor,
                dy: regularizedDelta.dy * predictionFactor
            )
            let requestedDistance = hypot(
                requestedLead.dx,
                requestedLead.dy
            )
            let maximumLeadDistance = min(
                max(sample.pose.width * 0.030, 1.25),
                Self.maxRigidDisplayPredictionPixels
            ) * max(directionConfidence, 0.45)
            if requestedDistance > maximumLeadDistance {
                let scale = maximumLeadDistance / requestedDistance
                requestedLead = CGVector(
                    dx: requestedLead.dx * scale,
                    dy: requestedLead.dy * scale
                )
            }
        }

        guard requestedLead.dx.isFinite,
              requestedLead.dy.isFinite else {
            latestRenderPredictionLead = .zero
            motionLock.unlock()
            return sample
        }
        let lead = stabilizedRigidPredictionLead(
            requested: requestedLead,
            previous: previousLead,
            mouthWidth: sample.pose.width
        )
        latestRenderPredictionLead = lead
        motionLock.unlock()

        let predictedPose = LipMotionPose(
            center: CGPoint(
                x: sample.pose.center.x + lead.dx,
                y: sample.pose.center.y + lead.dy
            ),
            width: sample.pose.width,
            angle: sample.pose.angle,
            opening: sample.pose.opening,
            jawOpen: sample.pose.jawOpen,
            smile: sample.pose.smile,
            pucker: sample.pose.pucker,
            upperRaise: sample.pose.upperRaise,
            lowerDrop: sample.pose.lowerDrop
        )
        LipDebugLog.throttled(
            "lip_rigid_display_prediction",
            interval: 0.75,
            "lip_motion rigidLeadPx=\(debugFloat(hypot(lead.dx, lead.dy))) requestedPx=\(debugFloat(hypot(requestedLead.dx, requestedLead.dy))) dtMs=\(String(format: "%.1f", sampleInterval * 1000)) direction=\(debugFloat(directionConfidence))"
        )
        return LipMotionSample(
            pose: predictedPose,
            sampledAt: sample.sampledAt,
            frameTimestamp: sample.frameTimestamp,
            trackingEpoch: sample.trackingEpoch,
            anchorIdentifier: sample.anchorIdentifier,
            viewportRevision: sample.viewportRevision
        )
    }

    private func stabilizedRigidPredictionLead(
        requested: CGVector,
        previous: CGVector,
        mouthWidth: CGFloat
    ) -> CGVector {
        let requestedDistance = hypot(requested.dx, requested.dy)
        let previousDistance = hypot(previous.dx, previous.dy)
        let directionDot = requested.dx * previous.dx +
            requested.dy * previous.dy
        let alpha: CGFloat
        if requestedDistance < 0.01 {
            // Decay over a few render frames instead of snapping a 1-2 px
            // prediction correction back to zero in a single frame.
            alpha = 0.62
        } else if previousDistance < 0.01 {
            alpha = 0.58
        } else if directionDot < 0 {
            // A genuine direction reversal should react quickly, but still
            // pass through a bounded transition to avoid a visible twitch.
            alpha = 0.74
        } else {
            alpha = 0.48
        }

        var result = CGVector(
            dx: previous.dx + (requested.dx - previous.dx) * alpha,
            dy: previous.dy + (requested.dy - previous.dy) * alpha
        )
        let change = CGVector(
            dx: result.dx - previous.dx,
            dy: result.dy - previous.dy
        )
        let changeDistance = hypot(change.dx, change.dy)
        let maximumStep = min(
            max(mouthWidth * 0.012, 0.65),
            1.10
        )
        if changeDistance > maximumStep {
            let scale = maximumStep / changeDistance
            result = CGVector(
                dx: previous.dx + change.dx * scale,
                dy: previous.dy + change.dy * scale
            )
        }
        if hypot(result.dx, result.dy) < 0.04 {
            return .zero
        }
        return result
    }

    private func projectedLipMotionPose(from faceAnchor: ARFaceAnchor,
                                        camera: ARCamera,
                                        orientation: UIInterfaceOrientation,
                                        viewportSize: CGSize) -> LipMotionPose? {
        let vertices = faceAnchor.geometry.vertices
        guard let rigidReference = rigidLipReference(for: faceAnchor) else {
            return nil
        }

        let extendedViewport = CGRect(
            x: -viewportSize.width * 0.25,
            y: -viewportSize.height * 0.25,
            width: viewportSize.width * 1.5,
            height: viewportSize.height * 1.5
        )

        func projectedPoint(_ local: SIMD3<Float>) -> CGPoint? {
            let world = faceAnchor.transform * SIMD4<Float>(local.x, local.y, local.z, 1)
            guard world.w.isFinite, abs(world.w) > 0.000_001 else {
                return nil
            }
            let point = camera.projectPoint(
                SIMD3<Float>(
                    world.x / world.w,
                    world.y / world.w,
                    world.z / world.w
                ),
                orientation: orientation,
                viewportSize: viewportSize
            )
            guard point.x.isFinite,
                  point.y.isFinite,
                  extendedViewport.contains(point) else {
                return nil
            }
            return point
        }

        guard let rigidCenter = projectedPoint(rigidReference.localCenter),
              let rigidLeft = projectedPoint(rigidReference.localLeft),
              let rigidRight = projectedPoint(rigidReference.localRight),
              let currentTop = projectedPoint(vertices[Self.arKitMouthTopIndex]),
              let currentBottom = projectedPoint(vertices[Self.arKitMouthBottomIndex]) else {
            return nil
        }
        let expression = lipMotionExpression(from: faceAnchor)

        return LipMotionPose(
            rigidCenter: rigidCenter,
            rigidLeft: rigidLeft,
            rigidRight: rigidRight,
            openingDistance: hypot(
                currentBottom.x - currentTop.x,
                currentBottom.y - currentTop.y
            ),
            jawOpen: expression.jawOpen,
            smile: expression.smile,
            pucker: expression.pucker,
            upperRaise: expression.upperRaise,
            lowerDrop: expression.lowerDrop
        )
    }

    private func projectedRenderLipMotionPose(
        from faceAnchor: ARFaceAnchor,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode,
        viewportSize: CGSize
    ) -> LipMotionPose? {
        let vertices = faceAnchor.geometry.vertices
        guard let rigidReference = rigidLipReference(for: faceAnchor) else {
            return nil
        }
        let extendedViewport = CGRect(
            x: -viewportSize.width * 0.25,
            y: -viewportSize.height * 0.25,
            width: viewportSize.width * 1.5,
            height: viewportSize.height * 1.5
        )

        func projectedPoint(_ local: SIMD3<Float>) -> CGPoint? {
            let localPoint = SCNVector3(local.x, local.y, local.z)
            let worldPoint = faceNode.convertPosition(localPoint, to: nil)
            let projected = renderer.projectPoint(worldPoint)
            let point = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
            guard projected.z.isFinite,
                  projected.z >= 0,
                  projected.z <= 1,
                  point.x.isFinite,
                  point.y.isFinite,
                  extendedViewport.contains(point) else {
                return nil
            }
            return point
        }

        guard let rigidCenter = projectedPoint(rigidReference.localCenter),
              let rigidLeft = projectedPoint(rigidReference.localLeft),
              let rigidRight = projectedPoint(rigidReference.localRight),
              let currentTop = projectedPoint(vertices[Self.arKitMouthTopIndex]),
              let currentBottom = projectedPoint(vertices[Self.arKitMouthBottomIndex]) else {
            return nil
        }
        let expression = lipMotionExpression(from: faceAnchor)
        return LipMotionPose(
            rigidCenter: rigidCenter,
            rigidLeft: rigidLeft,
            rigidRight: rigidRight,
            openingDistance: hypot(
                currentBottom.x - currentTop.x,
                currentBottom.y - currentTop.y
            ),
            jawOpen: expression.jawOpen,
            smile: expression.smile,
            pucker: expression.pucker,
            upperRaise: expression.upperRaise,
            lowerDrop: expression.lowerDrop
        )
    }

    private func lipMotionExpression(
        from faceAnchor: ARFaceAnchor
    ) -> (jawOpen: CGFloat, smile: CGFloat, pucker: CGFloat, upperRaise: CGFloat, lowerDrop: CGFloat) {
        let blendShapes = faceAnchor.blendShapes
        let jawOpen = FaceBlendShapeValue.value(.jawOpen, in: blendShapes)
        let smile = max(
            FaceBlendShapeValue.value(.mouthSmileLeft, in: blendShapes),
            FaceBlendShapeValue.value(.mouthSmileRight, in: blendShapes),
            FaceBlendShapeValue.value(.mouthDimpleLeft, in: blendShapes) * 0.55,
            FaceBlendShapeValue.value(.mouthDimpleRight, in: blendShapes) * 0.55
        )
        let pucker = max(
            FaceBlendShapeValue.value(.mouthPucker, in: blendShapes),
            FaceBlendShapeValue.value(.mouthFunnel, in: blendShapes) * 0.72
        )
        let upperRaise = max(
            FaceBlendShapeValue.value(.mouthUpperUpLeft, in: blendShapes),
            FaceBlendShapeValue.value(.mouthUpperUpRight, in: blendShapes)
        )
        let lowerDrop = max(
            FaceBlendShapeValue.value(.mouthLowerDownLeft, in: blendShapes),
            FaceBlendShapeValue.value(.mouthLowerDownRight, in: blendShapes)
        )
        return (
            jawOpen: CGFloat(jawOpen),
            smile: CGFloat(smile),
            pucker: CGFloat(pucker),
            upperRaise: CGFloat(upperRaise),
            lowerDrop: CGFloat(lowerDrop)
        )
    }

    private func rigidLipReference(for faceAnchor: ARFaceAnchor) -> RigidLipReference? {
        let vertices = faceAnchor.geometry.vertices
        let requiredIndex = max(
            Self.arKitMouthLeftIndex,
            Self.arKitMouthRightIndex,
            Self.arKitMouthTopIndex,
            Self.arKitMouthBottomIndex
        )
        guard vertices.count > requiredIndex else {
            return nil
        }

        let candidate = RigidLipReference(
            anchorIdentifier: faceAnchor.identifier,
            localCenter: (
                vertices[Self.arKitMouthLeftIndex] +
                    vertices[Self.arKitMouthRightIndex] +
                    vertices[Self.arKitMouthTopIndex] +
                    vertices[Self.arKitMouthBottomIndex]
            ) * 0.25,
            localLeft: vertices[Self.arKitMouthLeftIndex],
            localRight: vertices[Self.arKitMouthRightIndex]
        )
        motionLock.lock()
        defer { motionLock.unlock() }
        if let existing = rigidLipReference,
           existing.anchorIdentifier == faceAnchor.identifier {
            return existing
        }
        rigidLipReference = candidate
        return candidate
    }

    private func minimumLandmarkSubmitInterval(motionDelta: CGFloat) -> CFTimeInterval {
        let fastMotion = motionDelta > Self.fastLandmarkMotionDelta ||
            motionDelta == .greatestFiniteMagnitude
        switch ProcessInfo.processInfo.thermalState {
        case .critical:
            return fastMotion ? 0.066 : 0.100
        case .serious:
            return fastMotion ? 0.034 : 0.050
        case .fair:
            return fastMotion ? 0.022 : 0.034
        case .nominal:
            return fastMotion ? 0.016 : 0.025
        @unknown default:
            return fastMotion ? 0.025 : 0.040
        }
    }

    private func currentViewport() -> (size: CGSize,
                                       scale: CGFloat,
                                       orientation: UIInterfaceOrientation,
                                       revision: Int) {
        viewportLock.lock()
        let size = cachedViewportSize
        let scale = cachedRenderScale
        let orientation = cachedInterfaceOrientation
        let revision = cachedViewportRevision
        viewportLock.unlock()
        return (size, scale, orientation, revision)
    }

    private func currentTextureGeneration() -> Int {
        textureStateLock.lock()
        let generation = textureGeneration
        textureStateLock.unlock()
        return generation
    }

    private func reserveTextureRequestID() -> Int {
        textureStateLock.lock()
        nextTextureRequestID += 1
        latestTextureRequestID = nextTextureRequestID
        let requestID = nextTextureRequestID
        textureStateLock.unlock()
        return requestID
    }

    private func currentLatestTextureRequestID() -> Int {
        textureStateLock.lock()
        let requestID = latestTextureRequestID
        textureStateLock.unlock()
        return requestID
    }

    private func invalidatePendingTextures() {
        textureStateLock.lock()
        textureGeneration += 1
        pendingTextureRequest = nil
        isLipTextureRefreshPending = true
        lastTextureSubmitTime = nil
        lastTextureMouthOpen = nil
        pendingTextureMouthOpen = nil
        pendingTextureMouthOpenSince = 0
        textureStateLock.unlock()
    }

    private func setSessionActive(_ isActive: Bool) {
        trackingLock.lock()
        isSessionActive = isActive
        if !isActive {
            cameraTrackingIsNormal = false
            trackedFaceAnchorIdentifier = nil
            hasTrackedFaceAnchor = false
        }
        trackingLock.unlock()
    }

    private func currentSessionIsActive() -> Bool {
        trackingLock.lock()
        let isActive = isSessionActive
        trackingLock.unlock()
        return isActive
    }

    private func setCameraTrackingIsNormal(_ isNormal: Bool) {
        trackingLock.lock()
        cameraTrackingIsNormal = isSessionActive && isNormal
        trackingLock.unlock()
    }

    @discardableResult
    private func setTrackedFaceAnchor(_ identifier: UUID?,
                                      isTracked: Bool) -> Bool {
        trackingLock.lock()
        let previousIdentifier = trackedFaceAnchorIdentifier
        let previousTracking = hasTrackedFaceAnchor
        if isTracked,
           isSessionActive,
           cameraTrackingIsNormal,
           let identifier {
            trackedFaceAnchorIdentifier = identifier
            hasTrackedFaceAnchor = true
        } else if identifier == nil || trackedFaceAnchorIdentifier == identifier {
            trackedFaceAnchorIdentifier = nil
            hasTrackedFaceAnchor = false
        }
        let didChange = previousIdentifier != trackedFaceAnchorIdentifier ||
            previousTracking != hasTrackedFaceAnchor
        trackingLock.unlock()
        return didChange
    }

    private func currentTrackedFaceAnchor() -> Bool {
        trackingLock.lock()
        let isTracked = isSessionActive &&
            cameraTrackingIsNormal &&
            hasTrackedFaceAnchor
        trackingLock.unlock()
        return isTracked
    }

    private func currentTrackedFaceAnchorIdentifier() -> UUID? {
        trackingLock.lock()
        let identifier = isSessionActive && cameraTrackingIsNormal && hasTrackedFaceAnchor ?
            trackedFaceAnchorIdentifier :
            nil
        trackingLock.unlock()
        return identifier
    }

    private func setLatestLipMotionSample(_ sample: LipMotionSample?) {
        motionLock.lock()
        if let sample,
           let previous = latestLipMotionSample,
           (previous.trackingEpoch > sample.trackingEpoch ||
            (previous.trackingEpoch == sample.trackingEpoch &&
             previous.sampledAt > sample.sampledAt)) {
            motionLock.unlock()
            return
        }
        latestLipMotionSample = sample
        motionLock.unlock()
    }

    private func currentLipMotionSample() -> LipMotionSample? {
        motionLock.lock()
        let sample = latestLipMotionSample
        motionLock.unlock()

        let sampleAge = sample.map { CACurrentMediaTime() - $0.sampledAt }
        guard let sample,
              let sampleAge,
              sampleAge.isFinite,
              sampleAge >= 0,
              sampleAge <= Self.maxCurrentMotionSampleAge,
              currentLandmarkTrackingEpoch() == sample.trackingEpoch,
              currentTrackedFaceAnchorIdentifier() == sample.anchorIdentifier,
              currentViewport().revision == sample.viewportRevision else {
            return nil
        }
        return sample
    }

    private func currentLipMotionPose() -> LipMotionPose? {
        currentLipMotionSample()?.pose
    }

    private func clearLipMotionSamples(olderThan trackingEpoch: Int) {
        motionLock.lock()
        if latestLipMotionSample.map({ $0.trackingEpoch < trackingEpoch }) ?? false {
            latestLipMotionSample = nil
        }
        if latestRenderMotionSample.map({ $0.trackingEpoch < trackingEpoch }) ?? false {
            latestRenderMotionSample = nil
            latestRenderCenterDelta = nil
            latestRenderPredictionLead = .zero
        }
        motionLock.unlock()
    }

    private func setLatestMeshContour(_ contour: LipContour,
                                      motionReference: LipMotionPose,
                                      capturedAt: CFTimeInterval,
                                      anchorIdentifier: UUID,
                                      viewportRevision: Int,
                                      trackingEpoch: Int,
                                      predictionEligible: Bool) {
        let sample = AcceptedRealContourSample(
            contour: contour,
            capturedAt: capturedAt,
            motionReference: motionReference,
            trackingEpoch: trackingEpoch,
            anchorIdentifier: anchorIdentifier,
            viewportRevision: viewportRevision,
            predictionEligible: predictionEligible
        )
        meshStateLock.lock()
        if let latest = latestRealContourSample,
           latest.trackingEpoch == trackingEpoch,
           latest.anchorIdentifier == anchorIdentifier,
           latest.viewportRevision == viewportRevision,
           capturedAt > latest.capturedAt {
            previousRealContourSample = latest
        } else {
            previousRealContourSample = nil
        }
        latestRealContourSample = sample
        latestMeshContour = contour
        latestMeshContourCaptureTime = capturedAt
        latestMeshContourAcceptedAt = CACurrentMediaTime()
        latestMeshMotionPose = motionReference
        latestMeshAnchorIdentifier = anchorIdentifier
        latestMeshViewportRevision = viewportRevision
        latestMeshTrackingEpoch = trackingEpoch
        meshStateLock.unlock()
    }

    private func installLatestLipTextureIfCurrent(_ texture: LipTexture,
                                                  request: LipTextureRequest) -> Bool {
        let currentTrackingEpoch = currentLandmarkTrackingEpoch()
        let currentAnchorIdentifier = currentTrackedFaceAnchorIdentifier()
        let currentViewportRevision = currentViewport().revision

        textureStateLock.lock()
        guard textureGeneration == request.generation,
              request.requestID == latestTextureRequestID,
              request.requestID > latestInstalledTextureRequestID,
              currentTrackingEpoch == request.trackingEpoch,
              currentAnchorIdentifier == request.anchorIdentifier,
              currentViewportRevision == request.viewportRevision else {
            textureStateLock.unlock()
            return false
        }

        meshStateLock.lock()
        latestLipTexture = texture
        latestLipTextureGeneration = request.generation
        latestLipTextureTrackingEpoch = request.trackingEpoch
        latestLipTextureAnchorIdentifier = request.anchorIdentifier
        latestLipTextureViewportRevision = request.viewportRevision
        meshStateLock.unlock()
        latestInstalledTextureRequestID = request.requestID
        isLipTextureRefreshPending = false
        textureStateLock.unlock()
        return true
    }

    private func currentLipMeshState(
        for renderedAnchorIdentifier: UUID,
        renderMotionSample: LipMotionSample?,
        faceGeometry: ARFaceGeometry,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode
    ) -> LipMeshState? {
        meshStateLock.lock()
        let contour = latestMeshContour
        let contourCaptureTime = latestMeshContourCaptureTime
        let contourAcceptedAt = latestMeshContourAcceptedAt
        let contourMotionPose = latestMeshMotionPose
        let contourAnchorIdentifier = latestMeshAnchorIdentifier
        let contourViewportRevision = latestMeshViewportRevision
        let contourTrackingEpoch = latestMeshTrackingEpoch
        let texture = latestLipTexture
        let textureGeneration = latestLipTextureGeneration
        let textureTrackingEpoch = latestLipTextureTrackingEpoch
        let textureAnchorIdentifier = latestLipTextureAnchorIdentifier
        let textureViewportRevision = latestLipTextureViewportRevision
        let previousRealSample = previousRealContourSample
        let latestRealSample = latestRealContourSample
        meshStateLock.unlock()
        // Rendering uses only the pose produced from this exact face node and
        // SceneKit camera. An asynchronous session pose could move the contour
        // back to another frame during fast motion.
        let currentMotionSample = renderMotionSample
        let currentMotionPose = currentMotionSample?.pose
        let currentTrackingEpoch = currentLandmarkTrackingEpoch()
        let currentAnchorIdentifier = currentTrackedFaceAnchorIdentifier()
        let activeViewport = currentViewport()
        let currentViewportRevision = activeViewport.revision
        let currentGeneration = currentTextureGeneration()

        guard let contour,
              let contourCaptureTime,
              let contourAcceptedAt,
              let texture,
              currentAnchorIdentifier == renderedAnchorIdentifier,
              contourTrackingEpoch == currentTrackingEpoch,
              contourAnchorIdentifier == renderedAnchorIdentifier,
              contourViewportRevision == currentViewportRevision,
              textureGeneration == currentGeneration,
              textureTrackingEpoch == currentTrackingEpoch,
              textureAnchorIdentifier == renderedAnchorIdentifier,
              textureViewportRevision == currentViewportRevision else {
            return nil
        }
        let now = CACurrentMediaTime()
        let contourAge = now - contourCaptureTime
        let noUpdateAge = now - contourAcceptedAt
        guard contourAge.isFinite,
              contourAge >= 0,
              contourAge <= Self.maxRealContourDisplayAge,
              noUpdateAge.isFinite,
              noUpdateAge >= 0 else {
            LipDebugLog.throttled(
                "lip_mesh_real_contour_stale",
                interval: 0.4,
                "lip_mesh hide reason=stale_mediapipe_contour sourceAge=\(String(format: "%.3f", contourAge)) updateAge=\(String(format: "%.3f", noUpdateAge)) max=\(String(format: "%.3f", Self.maxRealContourDisplayAge))"
            )
            return nil
        }
        let freshnessVisibility = contourFreshnessVisibility(
            noUpdateAge: noUpdateAge
        )
        guard freshnessVisibility > 0.001 else {
            LipDebugLog.throttled(
                "lip_mesh_contour_faded",
                interval: 0.4,
                "lip_mesh hide reason=no_fresh_contour updateAge=\(String(format: "%.3f", noUpdateAge))"
            )
            return nil
        }

        let motionDelta = motionDelta(from: contourMotionPose, to: currentMotionPose)
        guard hasCompleteSurfaceCarrier(contour) else {
            LipDebugLog.throttled(
                "lip_mesh_real_contour_no_depth",
                interval: 0.4,
                "lip_mesh hide reason=mediapipe_contour_without_depth"
            )
            return nil
        }

        let displayedContour: LipContour
        let rigidMotionDelta = poseMotionDelta(
            from: contourMotionPose,
            to: currentMotionPose
        )
        if let contourMotionPose,
           let currentMotionPose,
           let sourceContourPose = contour.pose,
           rigidMotionDelta.isFinite {
            let targetContourPose = motionCompensatedPose(
                sourceContourPose,
                from: contourMotionPose,
                to: currentMotionPose
            )
            let compensatedContour = contour.transformed(
                from: sourceContourPose,
                to: targetContourPose
            )
            if compensatedContour.isUsable(in: activeViewport.size) {
                displayedContour = compensatedContour
                LipDebugLog.throttled(
                    "lip_mesh_rigid_compensation",
                    interval: 0.75,
                    "lip_mesh rigid=applied source=same_render_face_node delta=\(String(format: "%.3f", rigidMotionDelta)) age=\(String(format: "%.3f", contourAge))"
                )
            } else {
                LipDebugLog.throttled(
                    "lip_mesh_rigid_compensation_invalid",
                    interval: 0.4,
                    "lip_mesh hide reason=invalid_rigid_compensation delta=\(String(format: "%.3f", rigidMotionDelta))"
                )
                return nil
            }
        } else {
            LipDebugLog.throttled(
                "lip_mesh_missing_render_pose",
                interval: 0.4,
                "lip_mesh hide reason=no_current_rigid_pose age=\(String(format: "%.3f", contourAge))"
            )
            return nil
        }

        let temporallyCompensatedContour: LipContour
        let arExpressionDelta = expressionMotionDelta(
            from: contourMotionPose,
            to: currentMotionPose
        )
        if contour.isStationary && arExpressionDelta < 0.006 {
            // At rest ARKit's per-vertex face mesh and MediaPipe's short-term
            // velocity are both noisier than the real lip motion. Keep only
            // the rigid head pose until either detector hysteresis or a real
            // ARKit mouth-expression delta exits idle mode.
            temporallyCompensatedContour = displayedContour
        } else if arExpressionDelta >= 0.006 {
            // MediaPipe is the shape keyframe, but it is normally 80-150 ms
            // old by the time it reaches this render callback. ARKit's current
            // blendshapes therefore own active mouth motion. Previously a
            // formally valid but nearly static surface warp won this branch
            // first and left the visible contour one detector frame behind.
            let arExpressionContour = arExpressionCompensatedContour(
                capturedContour: contour,
                rigidReferenceContour: displayedContour,
                sourceMotionPose: contourMotionPose,
                targetMotionPose: currentMotionPose,
                viewportSize: activeViewport.size
            )
            if let arExpressionContour {
                temporallyCompensatedContour = arExpressionContour
                LipDebugLog.throttled(
                    "lip_temporal_arkit_expression",
                    interval: 0.75,
                    "lip_temporal route=arkit_expression delta=\(String(format: "%.3f", arExpressionDelta)) age=\(String(format: "%.3f", contourAge))"
                )
            } else {
                // Do not route a rejected active expression through the
                // generic surface carrier: that is the exact high-motion case
                // in which device logs show its bindings becoming outliers.
                temporallyCompensatedContour = expressionPredictedContour(
                    baseContour: displayedContour,
                    previousSample: previousRealSample,
                    latestSample: latestRealSample,
                    targetMotionPose: currentMotionPose,
                    targetFrameTimestamp: currentMotionSample?.frameTimestamp,
                    viewportSize: activeViewport.size
                )
                LipDebugLog.throttled(
                    "lip_temporal_prediction_fallback",
                    interval: 0.75,
                    "lip_temporal route=prediction reason=arkit_topology delta=\(String(format: "%.3f", arExpressionDelta)) age=\(String(format: "%.3f", contourAge))"
                )
            }
        } else if let surfaceWarpedContour = surfaceDeformedContour(
            from: contour,
            rigidReferenceContour: displayedContour,
            faceGeometry: faceGeometry,
            renderer: renderer,
            faceNode: faceNode,
            viewportSize: activeViewport.size
        ) {
            // The surface carrier is useful only as a quiet, local residual.
            // Large expression motion is handled by the branch above.
            temporallyCompensatedContour = surfaceWarpedContour
        } else {
            // ARKit sees no meaningful global expression and the surface
            // residual was not trustworthy. Use only a short, bounded
            // MediaPipe velocity correction; the real contour remains the
            // source of shape and topology.
            temporallyCompensatedContour = expressionPredictedContour(
                baseContour: displayedContour,
                previousSample: previousRealSample,
                latestSample: latestRealSample,
                targetMotionPose: currentMotionPose,
                targetFrameTimestamp: currentMotionSample?.frameTimestamp,
                viewportSize: activeViewport.size
            )
        }

        return LipMeshState(
            // Apply one uniform pose transform to the whole detected contour.
            // Expression prediction below is bounded and topology-checked.
            contour: softenedCupidsBow(in: temporallyCompensatedContour),
            texture: texture,
            contourAge: contourAge,
            motionDelta: motionDelta,
            freshnessVisibility: freshnessVisibility
        )
    }

    private func softenedCupidsBow(in contour: LipContour) -> LipContour {
        let centerIndex = 0
        let leftPeakIndex = 37
        let rightPeakIndex = 267
        guard let pose = contour.pose,
              let centerPoint = contour.meshPointsByIndex[centerIndex],
              let leftPeak = contour.meshPointsByIndex[leftPeakIndex]?.screen,
              let rightPeak = contour.meshPointsByIndex[rightPeakIndex]?.screen,
              let outerOffset = CanonicalLipGeometry.outerLipIndices.firstIndex(
                of: centerIndex
              ) else {
            return contour
        }

        let peakMidpoint = CGPoint(
            x: (leftPeak.x + rightPeak.x) * 0.5,
            y: (leftPeak.y + rightPeak.y) * 0.5
        )
        let requested = CGVector(
            dx: (peakMidpoint.x - centerPoint.screen.x) * 0.28,
            dy: (peakMidpoint.y - centerPoint.screen.y) * 0.28
        )
        let requestedDistance = hypot(requested.dx, requested.dy)
        let maximumDisplacement = min(
            max(pose.width * 0.008, 0.35),
            0.90
        )
        let scale = requestedDistance > maximumDisplacement ?
            maximumDisplacement / requestedDistance : 1
        let softenedPoint = CGPoint(
            x: centerPoint.screen.x + requested.dx * scale,
            y: centerPoint.screen.y + requested.dy * scale
        )
        guard softenedPoint.x.isFinite,
              softenedPoint.y.isFinite else {
            return contour
        }

        var softened = contour
        softened.meshPointsByIndex[centerIndex] = LipMeshPoint(
            screen: softenedPoint,
            normalized: centerPoint.normalized,
            uv: centerPoint.uv
        )
        if softened.outer.indices.contains(outerOffset) {
            softened.outer[outerOffset] = softenedPoint
        }
        return softened
    }

    private func contourFreshnessVisibility(
        noUpdateAge: CFTimeInterval
    ) -> CGFloat {
        let fadeDuration = max(
            Self.contourFreshnessFadeEnd - Self.contourFreshnessFadeStart,
            0.001
        )
        let linearProgress = min(
            max(
                (noUpdateAge - Self.contourFreshnessFadeStart) /
                    fadeDuration,
                0
            ),
            1
        )
        let smoothProgress = linearProgress * linearProgress *
            (3 - 2 * linearProgress)
        return CGFloat(1 - smoothProgress)
    }

    private func arExpressionCompensatedContour(
        capturedContour: LipContour,
        rigidReferenceContour: LipContour,
        sourceMotionPose: LipMotionPose?,
        targetMotionPose: LipMotionPose?,
        viewportSize: CGSize
    ) -> LipContour? {
        guard let sourceMotionPose,
              let targetMotionPose else {
            return nil
        }

        let openingDelta = min(
            max(targetMotionPose.opening - sourceMotionPose.opening, -0.12),
            0.12
        )
        let smileDelta = min(
            max(targetMotionPose.smile - sourceMotionPose.smile, -0.45),
            0.45
        )
        let puckerDelta = min(
            max(targetMotionPose.pucker - sourceMotionPose.pucker, -0.45),
            0.45
        )
        let upperRaiseDelta = min(
            max(targetMotionPose.upperRaise - sourceMotionPose.upperRaise, -0.45),
            0.45
        )
        let lowerDropDelta = min(
            max(targetMotionPose.lowerDrop - sourceMotionPose.lowerDrop, -0.45),
            0.45
        )
        let maximumExpressionDelta = max(
            abs(openingDelta),
            abs(smileDelta),
            abs(puckerDelta),
            abs(upperRaiseDelta),
            abs(lowerDropDelta)
        )
        guard maximumExpressionDelta >= 0.006 else {
            return nil
        }

        for backtrackAmount in [CGFloat(1), 0.75, 0.5, 0.25] {
            let blendedTarget = LipMotionPose(
                center: targetMotionPose.center,
                width: targetMotionPose.width,
                angle: targetMotionPose.angle,
                opening: sourceMotionPose.opening + openingDelta * backtrackAmount,
                jawOpen: sourceMotionPose.jawOpen +
                    (targetMotionPose.jawOpen - sourceMotionPose.jawOpen) *
                    backtrackAmount,
                smile: sourceMotionPose.smile + smileDelta * backtrackAmount,
                pucker: sourceMotionPose.pucker + puckerDelta * backtrackAmount,
                upperRaise: sourceMotionPose.upperRaise + upperRaiseDelta * backtrackAmount,
                lowerDrop: sourceMotionPose.lowerDrop + lowerDropDelta * backtrackAmount
            )
            let candidate = capturedContour.transformed(
                from: sourceMotionPose,
                to: blendedTarget
            )
            guard hasValidPredictedLipTopology(
                candidate,
                relativeTo: rigidReferenceContour,
                viewportSize: viewportSize
            ) else {
                continue
            }

            LipDebugLog.throttled(
                "lip_arkit_expression_fallback",
                interval: 0.75,
                "lip_motion fallback=arkit_expression delta=\(String(format: "%.3f", maximumExpressionDelta)) backtrack=\(String(format: "%.2f", backtrackAmount))"
            )
            return candidate
        }

        LipDebugLog.throttled(
            "lip_arkit_expression_fallback_rejected",
            interval: 0.5,
            "lip_motion reject=arkit_expression reason=topology delta=\(String(format: "%.3f", maximumExpressionDelta))"
        )
        return nil
    }

    private func surfaceDeformedContour(
        from capturedContour: LipContour,
        rigidReferenceContour: LipContour,
        faceGeometry: ARFaceGeometry,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode,
        viewportSize: CGSize
    ) -> LipContour? {
        guard capturedContour.surfaceCarrierSupportsDeformation,
              hasCompleteSurfaceCarrier(capturedContour),
              capturedContour.surfaceVertexCount == faceGeometry.vertices.count,
              capturedContour.surfaceTriangleIndexCount == faceGeometry.triangleIndices.count,
              capturedContour.surfaceTopologySignature == faceSurfaceTopologySignature(
                  vertexCount: faceGeometry.vertices.count,
                  triangleIndices: faceGeometry.triangleIndices
              ),
              let rigidPose = rigidReferenceContour.pose,
              rigidPose.width.isFinite,
              rigidPose.width > 1 else {
            return nil
        }

        let faceVertices = faceGeometry.vertices
        let extendedViewport = CGRect(
            x: -viewportSize.width * 0.25,
            y: -viewportSize.height * 0.25,
            width: viewportSize.width * 1.5,
            height: viewportSize.height * 1.5
        )
        var localResiduals: [Int: CGVector] = [:]
        localResiduals.reserveCapacity(Self.attentionLipIndices.count)

        for landmarkIndex in Self.attentionLipIndices {
            guard let binding = capturedContour.surfaceBindingsByIndex[landmarkIndex],
                  let capturedPoint = capturedContour.meshPointsByIndex[landmarkIndex]?.screen,
                  let rigidPoint = rigidReferenceContour.meshPointsByIndex[landmarkIndex]?.screen,
                  let currentSurfacePoint = boundFaceSurfacePoint(
                      binding,
                      faceVertices: faceVertices
                  ) else {
                return nil
            }
            let referencePoint = binding.referenceScreenPoint
            let localPoint = SCNVector3(
                currentSurfacePoint.x,
                currentSurfacePoint.y,
                currentSurfacePoint.z
            )
            let worldPoint = faceNode.convertPosition(localPoint, to: nil)
            let projected = renderer.projectPoint(worldPoint)
            let currentSurfaceScreenPoint = CGPoint(
                x: CGFloat(projected.x),
                y: CGFloat(projected.y)
            )
            guard referencePoint.x.isFinite,
                  referencePoint.y.isFinite,
                  projected.z.isFinite,
                  projected.z >= 0,
                  projected.z <= 1,
                  currentSurfaceScreenPoint.x.isFinite,
                  currentSurfaceScreenPoint.y.isFinite,
                  extendedViewport.contains(currentSurfaceScreenPoint) else {
                return nil
            }

            // The surface delta contains both head motion and local expression.
            // Rigid head motion is already present in rigidReferenceContour, so
            // keep only the remaining per-landmark expression displacement.
            let surfaceWarpedPoint = CGPoint(
                x: capturedPoint.x + currentSurfaceScreenPoint.x - referencePoint.x,
                y: capturedPoint.y + currentSurfaceScreenPoint.y - referencePoint.y
            )
            let residual = CGVector(
                dx: surfaceWarpedPoint.x - rigidPoint.x,
                dy: surfaceWarpedPoint.y - rigidPoint.y
            )
            let magnitude = hypot(residual.dx, residual.dy)
            guard residual.dx.isFinite,
                  residual.dy.isFinite,
                  magnitude.isFinite else {
                return nil
            }
            localResiduals[landmarkIndex] = residual
        }

        guard localResiduals.count == Self.attentionLipIndices.count else {
            return nil
        }
        // Rigid position is already supplied by the current AR face pose. A
        // coherent residual here is projection/parallax error, not expression;
        // remove it so this optional local warp can never drag the whole mask.
        let translationRemoved = removingCoherentTranslation(from: localResiduals)
        localResiduals = translationRemoved.residuals
        let residualMagnitudes = localResiduals.values.map {
            hypot($0.dx, $0.dy)
        }
        let sortedMagnitudes = residualMagnitudes.sorted()
        let medianMagnitude = sortedMagnitudes[sortedMagnitudes.count / 2]
        let p95Index = min(
            sortedMagnitudes.count - 1,
            Int((Double(sortedMagnitudes.count - 1) * 0.95).rounded(.up))
        )
        let p95Magnitude = sortedMagnitudes[p95Index]
        // This path now handles only quiet local residuals. Device logs show
        // that larger values correlate with weak barycentric bindings and the
        // visible one-frame slip, not with useful mouth deformation.
        let maximumMedianMagnitude = max(1.25, rigidPose.width * 0.020)
        let maximumP95Magnitude = max(2.5, rigidPose.width * 0.040)
        guard medianMagnitude <= maximumMedianMagnitude,
              p95Magnitude <= maximumP95Magnitude else {
            LipDebugLog.throttled(
                "lip_surface_warp_rejected",
                interval: 0.4,
                "lip_surface warp=skipped reason=motion_outlier medianPx=\(debugFloat(medianMagnitude)) p95Px=\(debugFloat(p95Magnitude))"
            )
            return nil
        }

        let regularizedResiduals = spatiallyRegularizedExpressionResiduals(localResiduals)
        let maximumLocalDisplacement = min(
            max(1.5, rigidPose.width * 0.025),
            3.0
        )
        for backtrackAmount in [CGFloat(1), 0.75, 0.5, 0.25] {
            guard let candidate = rigidReferenceContour.offsettingScreenPoints(
                      using: regularizedResiduals,
                      amount: backtrackAmount,
                      maximumDisplacement: maximumLocalDisplacement,
                      preservesSurfaceDeformation: true
                  ),
                  hasValidPredictedLipTopology(
                      candidate,
                      relativeTo: rigidReferenceContour,
                      viewportSize: viewportSize
                  ) else {
                continue
            }

            LipDebugLog.throttled(
                "lip_surface_warp_applied",
                interval: 0.75,
                "lip_surface warp=applied source=arkit_60fps medianPx=\(debugFloat(medianMagnitude)) p95Px=\(debugFloat(p95Magnitude)) backtrack=\(String(format: "%.2f", backtrackAmount))"
            )
            return candidate
        }

        LipDebugLog.throttled(
            "lip_surface_warp_rejected",
            interval: 0.4,
            "lip_surface warp=skipped reason=topology"
        )
        return nil
    }

    private func boundFaceSurfacePoint(
        _ binding: LipSurfaceBinding,
        faceVertices: [SIMD3<Float>]
    ) -> SIMD3<Float>? {
        let firstIndex = Int(binding.vertexIndices.x)
        let secondIndex = Int(binding.vertexIndices.y)
        let thirdIndex = Int(binding.vertexIndices.z)
        let weights = binding.barycentricWeights
        let weightSum = weights.x + weights.y + weights.z
        guard faceVertices.indices.contains(firstIndex),
              faceVertices.indices.contains(secondIndex),
              faceVertices.indices.contains(thirdIndex),
              weights.x.isFinite,
              weights.y.isFinite,
              weights.z.isFinite,
              weightSum.isFinite,
              abs(weightSum - 1) < 0.002 else {
            return nil
        }
        let point = faceVertices[firstIndex] * weights.x +
            faceVertices[secondIndex] * weights.y +
            faceVertices[thirdIndex] * weights.z
        guard point.x.isFinite,
              point.y.isFinite,
              point.z.isFinite else {
            return nil
        }
        return point
    }

    private func expressionPredictedContour(
        baseContour: LipContour,
        previousSample: AcceptedRealContourSample?,
        latestSample: AcceptedRealContourSample?,
        targetMotionPose: LipMotionPose?,
        targetFrameTimestamp: CFTimeInterval?,
        viewportSize: CGSize
    ) -> LipContour {
        guard let previousSample,
              let latestSample,
              previousSample.predictionEligible,
              latestSample.predictionEligible,
              previousSample.trackingEpoch == latestSample.trackingEpoch,
              previousSample.anchorIdentifier == latestSample.anchorIdentifier,
              previousSample.viewportRevision == latestSample.viewportRevision,
              let targetMotionPose,
              let targetFrameTimestamp,
              targetFrameTimestamp.isFinite,
              let previousPose = previousSample.contour.pose,
              let latestPose = latestSample.contour.pose,
              let basePose = baseContour.pose else {
            return baseContour
        }

        let sampleInterval = latestSample.capturedAt - previousSample.capturedAt
        let predictionAge = targetFrameTimestamp - latestSample.capturedAt
        guard sampleInterval.isFinite,
              sampleInterval >= 0.025,
              sampleInterval <= 0.140,
              predictionAge.isFinite,
              predictionAge >= 0,
              predictionAge <= 0.240 else {
            return baseContour
        }

        let widthRatio = latestPose.width / max(previousPose.width, 1)
        let angleDelta = abs(Self.normalizedAngle(latestPose.angle - previousPose.angle))
        guard widthRatio.isFinite,
              widthRatio >= 0.70,
              widthRatio <= 1.42,
              angleDelta <= 0.45,
              Self.attentionLipIndices.allSatisfy({
                  previousSample.contour.meshPointsByIndex[$0] != nil &&
                      latestSample.contour.meshPointsByIndex[$0] != nil &&
                      baseContour.meshPointsByIndex[$0] != nil
              }) else {
            return baseContour
        }

        var normalizedResidualByLandmark: [Int: CGVector] = [:]
        normalizedResidualByLandmark.reserveCapacity(Self.attentionLipIndices.count)

        for landmarkIndex in Self.attentionLipIndices {
            guard let previousPoint = previousSample.contour.meshPointsByIndex[landmarkIndex]?.screen,
                  let latestPoint = latestSample.contour.meshPointsByIndex[landmarkIndex]?.screen else {
                return baseContour
            }
            // Normalize by the rigid AR head pose, not by each contour's own
            // lip pose. This removes head/camera movement while preserving
            // real smile expansion, opening and local lip displacement.
            let previousLocal = normalizedLipPoint(
                previousPoint,
                pose: previousSample.motionReference
            )
            let latestLocal = normalizedLipPoint(
                latestPoint,
                pose: latestSample.motionReference
            )
            let residual = CGVector(
                dx: latestLocal.x - previousLocal.x,
                dy: latestLocal.y - previousLocal.y
            )
            let magnitude = hypot(residual.dx, residual.dy)
            guard residual.dx.isFinite,
                  residual.dy.isFinite,
                  magnitude.isFinite else {
                return baseContour
            }
            normalizedResidualByLandmark[landmarkIndex] = residual
        }

        // Prediction is allowed to change expression only. ARKit owns the
        // global center, so remove detector-wide translation before turning
        // MediaPipe velocity into a future contour.
        normalizedResidualByLandmark = removingCoherentTranslation(
            from: normalizedResidualByLandmark
        ).residuals
        let residualMagnitudes = normalizedResidualByLandmark.values.map {
            hypot($0.dx, $0.dy)
        }
        let largeResidualCount = residualMagnitudes.reduce(into: 0) { count, magnitude in
            if magnitude > 0.20 {
                count += 1
            }
        }
        let sortedMagnitudes = residualMagnitudes.sorted()
        let p95Index = min(
            max(Int(ceil(CGFloat(sortedMagnitudes.count) * 0.95)) - 1, 0),
            max(sortedMagnitudes.count - 1, 0)
        )
        guard !sortedMagnitudes.isEmpty,
              sortedMagnitudes[p95Index] <= 0.16,
              largeResidualCount <= 8 else {
            return baseContour
        }
        let predictedResidualByLandmark = spatiallyRegularizedExpressionResiduals(
            normalizedResidualByLandmark
        )

        let fadeStart: CFTimeInterval = 0.180
        let fadeEnd: CFTimeInterval = 0.240
        let fadeProgress = max(
            0,
            min((predictionAge - fadeStart) / (fadeEnd - fadeStart), 1)
        )
        let smoothFade = 1 - fadeProgress * fadeProgress * (3 - 2 * fadeProgress)
        let observedOpeningDelta = latestSample.motionReference.opening -
            previousSample.motionReference.opening
        let currentOpeningDelta = targetMotionPose.opening -
            latestSample.motionReference.opening
        let directionConfidence: CFTimeInterval
        if abs(observedOpeningDelta) > 0.008,
           observedOpeningDelta * currentOpeningDelta < 0 {
            // ARKit is only a direction veto here; it never supplies shape.
            // Stop extrapolating when opening/closing has already reversed.
            directionConfidence = 0.20
        } else {
            directionConfidence = 1
        }
        let predictionFactor = CGFloat(
            min(
                min(
                    predictionAge,
                    Self.maxExpressionPredictionHorizon
                ) / sampleInterval,
                Self.maxExpressionPredictionFactor
            ) * smoothFade * directionConfidence
        )
        guard predictionFactor.isFinite,
              predictionFactor > 0 else {
            return baseContour
        }

        let cosine = cos(targetMotionPose.angle)
        let sine = sin(targetMotionPose.angle)
        var displacementByLandmark: [Int: CGVector] = [:]
        displacementByLandmark.reserveCapacity(Self.attentionLipIndices.count)
        var maximumRequestedDisplacement: CGFloat = 0
        for landmarkIndex in Self.attentionLipIndices {
            guard let residual = predictedResidualByLandmark[landmarkIndex] else {
                return baseContour
            }
            let localX = residual.dx * targetMotionPose.width * predictionFactor
            let localY = residual.dy * targetMotionPose.width * predictionFactor
            let displacement = CGVector(
                dx: localX * cosine - localY * sine,
                dy: localX * sine + localY * cosine
            )
            let length = hypot(displacement.dx, displacement.dy)
            guard displacement.dx.isFinite,
                  displacement.dy.isFinite,
                  length.isFinite else {
                return baseContour
            }
            displacementByLandmark[landmarkIndex] = displacement
            maximumRequestedDisplacement = max(maximumRequestedDisplacement, length)
        }

        let maximumPointDisplacement = min(
            max(basePose.width * 0.020, 0.75),
            Self.maxExpressionPredictionPixels
        )
        // A small valid correction is preferable to dropping prediction for
        // the entire frame when a thin lip triangle is close to degeneracy.
        // The 0.125 step keeps this a residual correction, never a new shape.
        for backtrackAmount in [CGFloat(1), 0.75, 0.5, 0.25, 0.125] {
            guard let candidate = baseContour.offsettingScreenPoints(
                using: displacementByLandmark,
                amount: backtrackAmount,
                maximumDisplacement: maximumPointDisplacement
            ) else {
                continue
            }
            guard hasValidPredictedLipTopology(
                candidate,
                relativeTo: baseContour,
                viewportSize: viewportSize
            ) else {
                continue
            }
            LipDebugLog.throttled(
                "lip_mesh_expression_prediction",
                interval: 0.75,
                "lip_prediction applied dtMs=\(String(format: "%.1f", sampleInterval * 1000)) ageMs=\(String(format: "%.1f", predictionAge * 1000)) factor=\(String(format: "%.2f", predictionFactor)) maxPx=\(String(format: "%.2f", min(maximumRequestedDisplacement * backtrackAmount, maximumPointDisplacement))) backtrack=\(String(format: "%.2f", backtrackAmount))"
            )
            return candidate
        }

        LipDebugLog.throttled(
            "lip_mesh_expression_prediction_rejected",
            interval: 0.75,
            "lip_prediction skipped reason=topology ageMs=\(String(format: "%.1f", predictionAge * 1000))"
        )
        return baseContour
    }

    private func normalizedLipPoint(_ point: CGPoint, pose: LipMotionPose) -> CGPoint {
        let dx = point.x - pose.center.x
        let dy = point.y - pose.center.y
        let cosine = cos(pose.angle)
        let sine = sin(pose.angle)
        let width = max(pose.width, 1)
        return CGPoint(
            x: (dx * cosine + dy * sine) / width,
            y: (-dx * sine + dy * cosine) / width
        )
    }

    private func spatiallyRegularizedExpressionResiduals(
        _ residuals: [Int: CGVector]
    ) -> [Int: CGVector] {
        var neighbors: [Int: Set<Int>] = [:]
        for triangle in CanonicalLipGeometry.lipMeshTriangles {
            let indices = [triangle.0, triangle.1, triangle.2]
            for index in indices {
                var adjacent = neighbors[index] ?? []
                for neighbor in indices where neighbor != index {
                    adjacent.insert(neighbor)
                }
                neighbors[index] = adjacent
            }
        }

        var regularized: [Int: CGVector] = [:]
        regularized.reserveCapacity(residuals.count)
        for landmarkIndex in Self.attentionLipIndices {
            guard let own = residuals[landmarkIndex] else {
                continue
            }
            let adjacentResiduals = (neighbors[landmarkIndex] ?? []).compactMap {
                residuals[$0]
            }
            guard !adjacentResiduals.isEmpty else {
                regularized[landmarkIndex] = own
                continue
            }
            let sum = adjacentResiduals.reduce(CGVector.zero) { partial, value in
                CGVector(
                    dx: partial.dx + value.dx,
                    dy: partial.dy + value.dy
                )
            }
            let inverseCount = 1 / CGFloat(adjacentResiduals.count)
            let neighborAverage = CGVector(
                dx: sum.dx * inverseCount,
                dy: sum.dy * inverseCount
            )
            regularized[landmarkIndex] = CGVector(
                dx: own.dx * 0.70 + neighborAverage.dx * 0.30,
                dy: own.dy * 0.70 + neighborAverage.dy * 0.30
            )
        }
        return regularized
    }

    private func removingCoherentTranslation(
        from residuals: [Int: CGVector]
    ) -> (residuals: [Int: CGVector], translation: CGVector) {
        guard !residuals.isEmpty else {
            return (residuals, .zero)
        }
        let sortedX = residuals.values.map(\.dx).sorted()
        let sortedY = residuals.values.map(\.dy).sorted()
        let middle = sortedX.count / 2
        let translation = CGVector(
            dx: sortedX[middle],
            dy: sortedY[middle]
        )
        let centered = residuals.mapValues {
            CGVector(
                dx: $0.dx - translation.dx,
                dy: $0.dy - translation.dy
            )
        }
        return (centered, translation)
    }

    private func hasValidPredictedLipTopology(
        _ candidate: LipContour,
        relativeTo reference: LipContour,
        viewportSize: CGSize
    ) -> Bool {
        guard candidate.isUsable(in: viewportSize),
              hasReliableLipTopology(candidate),
              let referencePose = reference.pose,
              Self.attentionLipIndices.allSatisfy({
                  reference.meshPointsByIndex[$0] != nil &&
                      candidate.meshPointsByIndex[$0] != nil
              }) else {
            return false
        }

        let minimumReferenceArea = max(
            Float(0.08),
            Float(referencePose.width * referencePose.width * 0.000_01)
        )
        for triangle in CanonicalLipGeometry.lipMeshTriangles {
            guard let referenceFirst = reference.meshPointsByIndex[triangle.0]?.screen,
                  let referenceSecond = reference.meshPointsByIndex[triangle.1]?.screen,
                  let referenceThird = reference.meshPointsByIndex[triangle.2]?.screen,
                  let candidateFirst = candidate.meshPointsByIndex[triangle.0]?.screen,
                  let candidateSecond = candidate.meshPointsByIndex[triangle.1]?.screen,
                  let candidateThird = candidate.meshPointsByIndex[triangle.2]?.screen else {
                return false
            }
            let referenceArea = Self.signedArea(
                SIMD2<Float>(Float(referenceFirst.x), Float(referenceFirst.y)),
                SIMD2<Float>(Float(referenceSecond.x), Float(referenceSecond.y)),
                SIMD2<Float>(Float(referenceThird.x), Float(referenceThird.y))
            )
            let candidateArea = Self.signedArea(
                SIMD2<Float>(Float(candidateFirst.x), Float(candidateFirst.y)),
                SIMD2<Float>(Float(candidateSecond.x), Float(candidateSecond.y)),
                SIMD2<Float>(Float(candidateThird.x), Float(candidateThird.y))
            )
            guard referenceArea.isFinite,
                  candidateArea.isFinite else {
                return false
            }
            guard abs(referenceArea) > minimumReferenceArea else {
                let maximumCandidateArea = max(
                    minimumReferenceArea * 1.80,
                    abs(referenceArea) * 1.80
                )
                guard abs(candidateArea) <= maximumCandidateArea else {
                    return false
                }
                let resolvableArea: Float = 0.000_1
                if abs(referenceArea) > resolvableArea,
                   abs(candidateArea) > resolvableArea,
                   referenceArea * candidateArea <= 0 {
                    return false
                }
                continue
            }
            let areaRatio = abs(candidateArea) / abs(referenceArea)
            guard referenceArea * candidateArea > 0,
                  areaRatio >= 0.55,
                  areaRatio <= 1.80 else {
                return false
            }
        }
        return true
    }

    private func hasCompleteSurfaceCarrier(_ contour: LipContour) -> Bool {
        guard let vertexCount = contour.surfaceVertexCount,
              vertexCount > 0,
              let triangleIndexCount = contour.surfaceTriangleIndexCount,
              triangleIndexCount >= 3,
              contour.surfaceTopologySignature != nil,
              let carrierCreatedAt = contour.surfaceCarrierCreatedAt,
              carrierCreatedAt.isFinite else {
            return false
        }
        return Self.attentionLipIndices.allSatisfy {
            contour.surfaceBindingsByIndex[$0] != nil
        }
    }

    private func currentLipMeshAvailability() -> (contour: Bool, texture: Bool) {
        meshStateLock.lock()
        let hasContour = latestMeshContour != nil
        let contourAnchorIdentifier = latestMeshAnchorIdentifier
        let contourViewportRevision = latestMeshViewportRevision
        let contourTrackingEpoch = latestMeshTrackingEpoch
        let hasTexture = latestLipTexture != nil
        let textureGeneration = latestLipTextureGeneration
        let textureTrackingEpoch = latestLipTextureTrackingEpoch
        let textureAnchorIdentifier = latestLipTextureAnchorIdentifier
        let textureViewportRevision = latestLipTextureViewportRevision
        meshStateLock.unlock()
        let currentTrackingEpoch = currentLandmarkTrackingEpoch()
        let currentAnchorIdentifier = currentTrackedFaceAnchorIdentifier()
        let currentViewportRevision = currentViewport().revision
        let contourContextIsCurrent =
            contourTrackingEpoch == currentTrackingEpoch &&
            contourAnchorIdentifier == currentAnchorIdentifier &&
            contourViewportRevision == currentViewportRevision
        let textureContextIsCurrent =
            textureGeneration == currentTextureGeneration() &&
            textureTrackingEpoch == currentTrackingEpoch &&
            textureAnchorIdentifier == currentAnchorIdentifier &&
            textureViewportRevision == currentViewportRevision
        return (
            hasContour && contourContextIsCurrent,
            hasTexture && textureContextIsCurrent
        )
    }

    private func clearLipMeshState() {
        meshStateLock.lock()
        latestMeshContour = nil
        latestMeshContourCaptureTime = nil
        latestMeshContourAcceptedAt = nil
        latestMeshMotionPose = nil
        previousRealContourSample = nil
        latestRealContourSample = nil
        latestMeshAnchorIdentifier = nil
        latestMeshViewportRevision = nil
        latestMeshTrackingEpoch = nil
        latestLipTexture = nil
        latestLipTextureGeneration = nil
        latestLipTextureTrackingEpoch = nil
        latestLipTextureAnchorIdentifier = nil
        latestLipTextureViewportRevision = nil
        meshStateLock.unlock()
    }

    private func resetLipTracking(invalidatePipeline: Bool = true,
                                  invalidateTextures: Bool = true) {
        let resetEpoch = invalidatePipeline ? invalidateLandmarkPipeline() : nil
        let motionCutoffEpoch = resetEpoch ?? currentLandmarkTrackingEpoch()
        if invalidateTextures {
            invalidatePendingTextures()
        }
        missedDetectionCount = 0
        smoothedLipContour = nil
        smoothedLipPose = nil
        contourStabilityConfidence = 0
        contourIsStationary = false
        lastAcceptedLandmarkTimestampInMilliseconds = -1
        lastAcceptedMotionPose = nil
        mouthStateLock.lock()
        neutralMouthWidth = nil
        mouthStateLock.unlock()
        clearLipShapeFreshness()
        clearLipMotionSamples(olderThan: motionCutoffEpoch)
        // The procedural texture depends only on style/finish/render scale,
        // not on a face anchor or tracking epoch. Keep it across detector
        // resets so reacquisition never waits for the same raster again.
        lipTextureRenderer.resetTemporalState(preservingProceduralTexture: true)
        clearLipMeshState()
        if let resetEpoch {
            markLandmarkPipelineReady(trackingEpoch: resetEpoch)
        }
    }

    private func resetLipTrackingAsync() {
        let resetEpoch = invalidateLandmarkPipeline()
        invalidatePendingTextures()
        landmarkQueue.async { [weak self] in
            guard let self else {
                return
            }
            self.resetLipTracking(
                invalidatePipeline: false,
                invalidateTextures: false
            )
            self.markLandmarkPipelineReady(trackingEpoch: resetEpoch)
        }
    }

    private func requestRendererClear() {
        rendererStateLock.lock()
        rendererClearIsPending = true
        rendererStateLock.unlock()
    }

    private func applyPendingRendererState() {
        rendererStateLock.lock()
        let lipOpacity = pendingLipOpacity
        let blushStyle = pendingBlushStyle
        let shouldClear = rendererClearIsPending
        pendingLipOpacity = nil
        pendingBlushStyle = nil
        rendererClearIsPending = false
        rendererStateLock.unlock()

        if shouldClear {
            lipMeshRenderer.clearAll()
            blushRenderer.clear()
        }
        if let lipOpacity {
            lipMeshRenderer.updateOpacity(lipOpacity)
        }
        if let blushStyle {
            blushRenderer.updateStyle(
                color: blushStyle.color,
                opacity: blushStyle.opacity
            )
        }
    }

    private func resetAndRequestRendererClear() {
        resetLipTrackingAsync()
        requestRendererClear()
    }
}
