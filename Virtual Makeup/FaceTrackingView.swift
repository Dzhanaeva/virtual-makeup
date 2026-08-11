import ARKit
import CoreImage
import CoreML
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit
import Vision

enum LipFinish: Int, Hashable {
    case matte
    case gloss
    case satin
}

private enum LipDebugLog {
    private static let lock = NSLock()
    private static var lastLoggedAt: [String: CFTimeInterval] = [:]

    static func throttled(_ key: String,
                          interval: CFTimeInterval = 0.5,
                          _ message: @autoclosure () -> String) {

        let now = CACurrentMediaTime()
        lock.lock()
        if let last = lastLoggedAt[key], now - last < interval {
            lock.unlock()
            return
        }
        lastLoggedAt[key] = now
        lock.unlock()
        print(message())

    }
}

private final class FPSMeter {
    private let name: String
    private let lock = NSLock()
    private var frameCount = 0
    private var lastReportTime = CACurrentMediaTime()
    private var accumulatedMilliseconds: Double = 0
    private var measuredSamples = 0

    init(_ name: String) {
        self.name = name
    }

    func tick(workMilliseconds: Double? = nil) {

        let now = CACurrentMediaTime()
        lock.lock()
        frameCount += 1
        if let workMilliseconds {
            accumulatedMilliseconds += workMilliseconds
            measuredSamples += 1
        }

        let elapsed = now - lastReportTime
        guard elapsed >= 1 else {
            lock.unlock()
            return
        }

        let fps = Double(frameCount) / elapsed
        let averageMilliseconds = measuredSamples > 0 ? accumulatedMilliseconds / Double(measuredSamples) : nil
        frameCount = 0
        accumulatedMilliseconds = 0
        measuredSamples = 0
        lastReportTime = now
        lock.unlock()

        if let averageMilliseconds {
            print(String(format: "fps_meter %@ fps=%.1f avg_ms=%.2f", name, fps, averageMilliseconds))
        } else {
            print(String(format: "fps_meter %@ fps=%.1f", name, fps))
        }

    }
}

private struct SemanticLipMask {
    private static let upperLipLabel: Int32 = 12
    private static let lowerLipLabel: Int32 = 13

    let labels: [Int32]
    let width: Int
    let height: Int
    let imageSize: CGSize
    let viewportSize: CGSize

    init?(multiArray: MLMultiArray, imageSize: CGSize, viewportSize: CGSize) {
        let count = multiArray.count
        let shape = multiArray.shape.map(\.intValue).filter { $0 > 1 }
        let inferredWidth: Int
        let inferredHeight: Int
        if shape.count >= 2 {
            inferredHeight = shape[shape.count - 2]
            inferredWidth = shape[shape.count - 1]
        } else {
            let side = Int(Double(count).squareRoot().rounded(.down))
            guard side > 0, side * side <= count else {
                return nil
            }
            inferredWidth = side
            inferredHeight = side
        }

        guard inferredWidth > 1,
              inferredHeight > 1,
              inferredWidth * inferredHeight <= count,
              imageSize.width > 1,
              imageSize.height > 1,
              viewportSize.width > 1,
              viewportSize.height > 1 else {
            return nil
        }

        width = inferredWidth
        height = inferredHeight
        self.imageSize = imageSize
        self.viewportSize = viewportSize

        let labelCount = width * height
        switch multiArray.dataType {
        case .int32:
            let pointer = multiArray.dataPointer.bindMemory(to: Int32.self, capacity: count)
            labels = Array(UnsafeBufferPointer(start: pointer, count: labelCount))
        case .float32:
            let pointer = multiArray.dataPointer.bindMemory(to: Float.self, capacity: count)
            labels = UnsafeBufferPointer(start: pointer, count: labelCount).map { Int32($0.rounded()) }
        case .double:
            let pointer = multiArray.dataPointer.bindMemory(to: Double.self, capacity: count)
            labels = UnsafeBufferPointer(start: pointer, count: labelCount).map { Int32($0.rounded()) }
        default:
            labels = (0..<labelCount).map { Int32(truncating: multiArray[$0]) }
        }
    }

    func lipAlpha(atViewportPoint point: CGPoint) -> Float {
        guard let pixel = imagePixel(forViewportPoint: point) else {
            return 0
        }

        let x = max(0, min(width - 1, Int((pixel.x / imageSize.width) * CGFloat(width))))
        let y = max(0, min(height - 1, Int((pixel.y / imageSize.height) * CGFloat(height))))
        if isLipLabel(label(atX: x, y: y)) {
            return 1
        }

        if hasLipNeighbor(x: x, y: y, radius: 1) {
            return 0.78
        }

        if hasLipNeighbor(x: x, y: y, radius: 2) {
            return 0.36
        }

        return 0
    }

    private func imagePixel(forViewportPoint point: CGPoint) -> CGPoint? {
        let transform = Self.aspectFillTransform(for: imageSize, in: viewportSize)
        let imagePoint = point.applying(transform.inverted())
        guard imagePoint.x >= 0,
              imagePoint.y >= 0,
              imagePoint.x < imageSize.width,
              imagePoint.y < imageSize.height else {
            return nil
        }
        return imagePoint
    }

    private func hasLipNeighbor(x: Int, y: Int, radius: Int) -> Bool {
        let minX = max(0, x - radius)
        let maxX = min(width - 1, x + radius)
        let minY = max(0, y - radius)
        let maxY = min(height - 1, y + radius)
        for sampleY in minY...maxY {
            for sampleX in minX...maxX where isLipLabel(label(atX: sampleX, y: sampleY)) {
                return true
            }
        }
        return false
    }

    private func label(atX x: Int, y: Int) -> Int32 {
        labels[y * width + x]
    }

    private func isLipLabel(_ label: Int32) -> Bool {
        label == Self.upperLipLabel || label == Self.lowerLipLabel
    }

    private static func aspectFillTransform(for imageSize: CGSize, in viewportSize: CGSize) -> CGAffineTransform {
        guard imageSize.width > 0,
              imageSize.height > 0,
              viewportSize.width > 0,
              viewportSize.height > 0 else {
            return .identity
        }

        let scale = max(viewportSize.width / imageSize.width, viewportSize.height / imageSize.height)
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        let offsetX = (viewportSize.width - scaledWidth) * 0.5
        let offsetY = (viewportSize.height - scaledHeight) * 0.5
        return CGAffineTransform(translationX: offsetX, y: offsetY).scaledBy(x: scale, y: scale)
    }
}

private final class LipSemanticSegmenter {
    private let request: VNCoreMLRequest?
    private let lock = NSLock()

    init() {
        guard let modelURL = Bundle.main.url(forResource: "faceParsing", withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: modelURL),
              let visionModel = try? VNCoreMLModel(for: model) else {
            request = nil
            return
        }

        let request = VNCoreMLRequest(model: visionModel)
        request.imageCropAndScaleOption = .scaleFill
        self.request = request
    }

    func makeMask(pixelBuffer: CVPixelBuffer, imageSize: CGSize, viewportSize: CGSize) -> SemanticLipMask? {
        guard let request else {
            return nil
        }

        lock.lock()
        defer {
            lock.unlock()
        }

        do {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up)
            try handler.perform([request])
            guard let observation = request.results?.first as? VNCoreMLFeatureValueObservation,
                  let multiArray = observation.featureValue.multiArrayValue else {
                return nil
            }
            return SemanticLipMask(multiArray: multiArray, imageSize: imageSize, viewportSize: viewportSize)
        } catch {
            LipDebugLog.throttled(
                "lip_semantic_failed",
                interval: 2,
                "lip_semantic failed error=\(error.localizedDescription)"
            )
            return nil
        }
    }
}

struct FaceTrackingView: UIViewRepresentable {
    @Binding var isFaceDetected: Bool
    var lipColor: UIColor
    var lipOpacity: Double
    var lipFinish: LipFinish
    var blushColor: UIColor
    var blushOpacity: Double

    func makeUIView(context: Context) -> ARSCNView {
        let view = ViewportTrackingARSCNView(frame: .zero)
        view.delegate = context.coordinator
        view.session.delegate = context.coordinator
        view.automaticallyUpdatesLighting = false
        view.rendersContinuously = true
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .black
        view.scene = SCNScene()
        view.onLayout = { [weak coordinator = context.coordinator] size, scale, orientation in
            coordinator?.updateViewport(size: size, scale: scale, orientation: orientation)
        }

        context.coordinator.sceneView = view
        context.coordinator.updateLipstick(color: lipColor, opacity: lipOpacity, finish: lipFinish)
        context.coordinator.updateBlush(color: blushColor, opacity: blushOpacity)
        context.coordinator.updateViewport(
            size: view.bounds.size,
            scale: view.traitCollection.displayScale,
            orientation: view.window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .unknown
        )
        context.coordinator.startSession()

        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {
        context.coordinator.updateLipstick(color: lipColor, opacity: lipOpacity, finish: lipFinish)
        context.coordinator.updateBlush(color: blushColor, opacity: blushOpacity)
        context.coordinator.updateViewport(
            size: view.bounds.size,
            scale: view.window?.screen.scale ?? view.traitCollection.displayScale,
            orientation: view.window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .unknown
        )
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(isFaceDetected: $isFaceDetected)
    }

    static func dismantleUIView(_ view: ARSCNView, coordinator: Coordinator) {
        view.session.pause()
        view.delegate = nil
        view.session.delegate = nil
        (view as? ViewportTrackingARSCNView)?.onLayout = nil
        coordinator.stopSession()
        coordinator.sceneView = nil
    }

}

private final class ViewportTrackingARSCNView: ARSCNView {
    var onLayout: ((CGSize, CGFloat, UIInterfaceOrientation) -> Void)?

    override func layoutSubviews() {
        super.layoutSubviews()
        onLayout?(
            bounds.size,
            window?.screen.scale ?? traitCollection.displayScale,
            window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .unknown
        )
    }
}

private enum FaceBlendShapeValue {
    static func value(_ location: ARFaceAnchor.BlendShapeLocation,
                      in blendShapes: [ARFaceAnchor.BlendShapeLocation: NSNumber]) -> Float {
        guard let value = blendShapes[location]?.floatValue,
              value.isFinite else {
            return 0
        }
        return min(max(value, 0), 1)
    }
}

private struct LipMeshPoint {
    let screen: CGPoint
    let normalized: SIMD3<Float>
    let uv: CGPoint
}

private struct LipSurfaceBinding {
    let vertexIndices: SIMD3<Int32>
    let barycentricWeights: SIMD3<Float>
    let projectionError: CGFloat
    let referenceScreenPoint: CGPoint
}

private struct ProjectedFaceSurfaceTriangle {
    let vertexIndices: SIMD3<Int32>
    let first: CGPoint
    let second: CGPoint
    let third: CGPoint
    let cameraDepths: SIMD3<Float>
    let normalizedMouthY: Float
}

private struct FaceSurfaceTriangleKey: Hashable {
    let first: Int32
    let second: Int32
    let third: Int32

    init(_ indices: SIMD3<Int32>) {
        first = indices.x
        second = indices.y
        third = indices.z
    }
}

private struct FaceSurfaceSnapshot {
    let vertexCount: Int
    let triangleIndexCount: Int
    let topologySignature: UInt64
    let sourceVertices: [SIMD3<Float>]
    let mouthFrame: FaceSurfaceMouthFrame
    let triangles: [ProjectedFaceSurfaceTriangle]
    let trianglesByKey: [FaceSurfaceTriangleKey: ProjectedFaceSurfaceTriangle]
}

private struct FaceSurfaceProjectionInput {
    let geometry: ARFaceGeometry
    let anchorTransform: simd_float4x4
    let camera: ARCamera
    let orientation: UIInterfaceOrientation
    let viewportSize: CGSize
    let anchorIdentifier: UUID
}

private struct FaceSurfaceTopologyCache {
    let vertexCount: Int
    let triangleIndexCount: Int
    let topologySignature: UInt64
    let anchorIdentifier: UUID
    let triangles: [SIMD3<Int32>]
}

private struct FaceSurfaceMouthFrame {
    let center: SIMD3<Float>
    let xAxis: SIMD3<Float>
    let downAxis: SIMD3<Float>
    let normalAxis: SIMD3<Float>
    let width: Float
}

private func faceSurfaceTopologySignature(vertexCount: Int,
                                          triangleIndices: [Int16]) -> UInt64 {
    var hash: UInt64 = 1_469_598_103_934_665_603
    hash ^= UInt64(vertexCount)
    hash &*= 1_099_511_628_211
    hash ^= UInt64(triangleIndices.count)
    hash &*= 1_099_511_628_211
    for index in triangleIndices {
        hash ^= UInt64(UInt16(bitPattern: index))
        hash &*= 1_099_511_628_211
    }
    return hash
}

private struct LipContour {
    var outer: [CGPoint]
    var inner: [CGPoint]
    var outer3D: [SIMD3<Float>] = []
    var inner3D: [SIMD3<Float>] = []
    var outerUV: [CGPoint] = []
    var innerUV: [CGPoint] = []
    var meshPointsByIndex: [Int: LipMeshPoint] = [:]
    var surfaceBindingsByIndex: [Int: LipSurfaceBinding] = [:]
    var surfaceVertexCount: Int? = nil
    var surfaceTriangleIndexCount: Int? = nil
    var surfaceTopologySignature: UInt64? = nil
    var surfaceCarrierCreatedAt: CFTimeInterval? = nil
    var surfaceCarrierSupportsDeformation = false
    var faceGeometryPose: FaceGeometryPose? = nil
    // MediaPipe can report a small inner aperture during a closed-mouth smile.
    // Keep the detector value for shape analysis, but let ARKit veto that
    // aperture for the rendered mesh and alpha mask when the lips are closed.
    var renderedInnerOpeningRatio: CGFloat? = nil
    // Set only after rigid head motion has been removed and the residual lip
    // shape has remained inside the detector noise floor for several samples.
    var isStationary = false

    var center: CGPoint? {
        guard !outer.isEmpty else {
            return nil
        }

        let sum = outer.reduce(CGPoint.zero) { partial, point in
            CGPoint(x: partial.x + point.x, y: partial.y + point.y)
        }
        return CGPoint(x: sum.x / CGFloat(outer.count), y: sum.y / CGFloat(outer.count))
    }

    var pose: LipPose? {
        guard outer.count > 10, let center else {
            return nil
        }

        let leftCorner = outer[0]
        let rightCorner = outer[10]
        let dx = rightCorner.x - leftCorner.x
        let dy = rightCorner.y - leftCorner.y
        let width = hypot(dx, dy)
        guard width.isFinite, width > 1 else {
            return nil
        }
        return LipPose(center: center, width: width, angle: atan2(dy, dx))
    }

    var bounds: CGRect? {
        let points = outer + inner
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

    var depthCenter: Float {
        let points = outer3D + inner3D
        guard !points.isEmpty else {
            return 0
        }

        let total = points.reduce(Float(0)) { partial, point in
            partial + point.z
        }
        return total / Float(points.count)
    }

    var innerOpeningRatio: CGFloat {
        guard let pose,
              inner.count >= 4 else {
            return 1
        }
        let cosine = cos(pose.angle)
        let sine = sin(pose.angle)
        let localYValues = inner.map { innerPoint -> CGFloat in
            let dx = innerPoint.x - pose.center.x
            let dy = innerPoint.y - pose.center.y
            return -dx * sine + dy * cosine
        }
        guard let minimumY = localYValues.min(),
              let maximumY = localYValues.max() else {
            return 1
        }
        return max(maximumY - minimumY, 0) / max(pose.width, 1)
    }

    var effectiveInnerOpeningRatio: CGFloat {
        renderedInnerOpeningRatio ?? innerOpeningRatio
    }

    func renderingInnerScreenPoint(_ point: CGPoint) -> CGPoint {
        guard let pose,
              inner.count >= 4 else {
            return point
        }

        let cosine = cos(pose.angle)
        let sine = sin(pose.angle)
        let localValues = inner.map { innerPoint -> (x: CGFloat, y: CGFloat) in
            let dx = innerPoint.x - pose.center.x
            let dy = innerPoint.y - pose.center.y
            return (
                dx * cosine + dy * sine,
                -dx * sine + dy * cosine
            )
        }
        guard let minimumX = localValues.map(\.x).min(),
              let maximumX = localValues.map(\.x).max(),
              let minimumY = localValues.map(\.y).min(),
              let maximumY = localValues.map(\.y).max(),
              maximumX > minimumX else {
            return point
        }
        let openingRatio = effectiveInnerOpeningRatio
        let openingPosition = min(
            max((openingRatio - 0.035) / (0.080 - 0.035), 0),
            1
        )
        let smoothOpening = openingPosition * openingPosition *
            (3 - 2 * openingPosition)
        let seamCenterY = (minimumY + maximumY) * 0.5

        let dx = point.x - pose.center.x
        let dy = point.y - pose.center.y
        let localX = dx * cosine + dy * sine
        let localY = -dx * sine + dy * cosine
        // A small overlap removes the antialiased row at the closed seam. Do
        // not carry that overlap into the mouth corners: their narrow triangle
        // fans otherwise cross during a closed-mouth smile and form spikes.
        let seamCenterX = (minimumX + maximumX) * 0.5
        let seamHalfWidth = max((maximumX - minimumX) * 0.5, 0.001)
        let normalizedHorizontalPosition = min(
            abs(localX - seamCenterX) / seamHalfWidth,
            1
        )
        let cornerFadePosition = min(
            max((normalizedHorizontalPosition - 0.65) / (1 - 0.65), 0),
            1
        )
        let smoothCornerFade = cornerFadePosition * cornerFadePosition *
            (3 - 2 * cornerFadePosition)
        let centerOverlapWeight = 1 - smoothCornerFade
        let closedVerticalScale = -0.35 * centerOverlapWeight
        let verticalScale = closedVerticalScale +
            smoothOpening * (1 - closedVerticalScale)
        let compressedY = seamCenterY + (localY - seamCenterY) * verticalScale
        return CGPoint(
            x: pose.center.x + localX * cosine - compressedY * sine,
            y: pose.center.y + localX * sine + compressedY * cosine
        )
    }

    func isUsable(in viewportSize: CGSize) -> Bool {
        guard let bounds,
              let pose,
              viewportSize.width > 1,
              viewportSize.height > 1 else {
            return false
        }

        let minLipWidth = max(viewportSize.width * 0.08, 28)
        let maxLipWidth = viewportSize.width * 0.72
        // A relaxed mouth can be only 5-6 points high on a 390x844 viewport.
        // Width and topology checks below still reject non-mouth detections.
        let minLipHeight = max(viewportSize.height * 0.006, 5)
        let maxLipHeight = viewportSize.height * 0.22
        let extendedViewport = CGRect(
            x: -viewportSize.width * 0.2,
            y: -viewportSize.height * 0.2,
            width: viewportSize.width * 1.4,
            height: viewportSize.height * 1.4
        )

        return pose.center.x.isFinite &&
            pose.center.y.isFinite &&
            bounds.width >= minLipWidth &&
            bounds.width <= maxLipWidth &&
            bounds.height >= minLipHeight &&
            bounds.height <= maxLipHeight &&
            extendedViewport.contains(pose.center)
    }

    func transformed(from source: LipPose, to target: LipPose) -> LipContour {
        let scale = min(max(target.width / max(source.width, 1), 0.65), 1.45)
        let angleDelta = target.angle - source.angle
        let cosine = cos(angleDelta)
        let sine = sin(angleDelta)

        func transform(_ point: CGPoint) -> CGPoint {
            let x = (point.x - source.center.x) * scale
            let y = (point.y - source.center.y) * scale
            return CGPoint(
                x: target.center.x + x * cosine - y * sine,
                y: target.center.y + x * sine + y * cosine
            )
        }

        return LipContour(
            outer: outer.map(transform),
            inner: inner.map(transform),
            outer3D: outer3D,
            inner3D: inner3D,
            outerUV: outerUV,
            innerUV: innerUV,
            meshPointsByIndex: meshPointsByIndex.mapValues {
                LipMeshPoint(
                    screen: transform($0.screen),
                    normalized: $0.normalized,
                    uv: $0.uv
                )
            },
            surfaceBindingsByIndex: surfaceBindingsByIndex,
            surfaceVertexCount: surfaceVertexCount,
            surfaceTriangleIndexCount: surfaceTriangleIndexCount,
            surfaceTopologySignature: surfaceTopologySignature,
            surfaceCarrierCreatedAt: surfaceCarrierCreatedAt,
            surfaceCarrierSupportsDeformation: surfaceCarrierSupportsDeformation,
            faceGeometryPose: faceGeometryPose,
            renderedInnerOpeningRatio: renderedInnerOpeningRatio,
            isStationary: isStationary
        )
    }

    func transformed(from source: LipMotionPose, to target: LipMotionPose) -> LipContour {
        let scale = min(max(target.width / max(source.width, 1), 0.70), 1.42)
        let sourceCosine = cos(source.angle)
        let sourceSine = sin(source.angle)
        let targetCosine = cos(target.angle)
        let targetSine = sin(target.angle)
        let normalizedOpeningDelta = min(
            max(target.opening - source.opening, -0.22),
            0.22
        )
        let openingDisplacement = normalizedOpeningDelta * target.width
        let smileDelta = min(max(target.smile - source.smile, -0.50), 0.50)
        let puckerDelta = min(max(target.pucker - source.pucker, -0.50), 0.50)
        let upperRaiseDelta = min(
            max(target.upperRaise - source.upperRaise, -0.50),
            0.50
        )
        let lowerDropDelta = min(
            max(target.lowerDrop - source.lowerDrop, -0.50),
            0.50
        )

        func transform(_ point: CGPoint, landmarkIndex: Int?) -> CGPoint {
            let dx = point.x - source.center.x
            let dy = point.y - source.center.y
            let localX = dx * sourceCosine + dy * sourceSine
            let localY = -dx * sourceSine + dy * sourceCosine
            let smileStretch = landmarkIndex.map {
                CGFloat(CanonicalLipGeometry.smileStretchFactor(for: $0))
            } ?? 0
            let expressionXScale = max(
                0.94,
                min(1 + smileDelta * smileStretch - puckerDelta * 0.045, 1.06)
            )
            let x = localX * scale * expressionXScale
            let openingFactor = landmarkIndex.map {
                CanonicalLipGeometry.openingMotionFactor(for: $0)
            } ?? 0
            let upperFactor = landmarkIndex.map {
                CGFloat(CanonicalLipGeometry.upperLipSmileLiftFactor(for: $0))
            } ?? 0
            let lowerFactor = landmarkIndex.map {
                CGFloat(CanonicalLipGeometry.lowerLipDropFactor(for: $0))
            } ?? 0
            let expressionY =
                -upperRaiseDelta * upperFactor * target.width * 0.024 +
                lowerDropDelta * lowerFactor * target.width * 0.018 -
                smileDelta * upperFactor * target.width * 0.014
            let y = localY * scale +
                openingDisplacement * openingFactor +
                expressionY
            return CGPoint(
                x: target.center.x + x * targetCosine - y * targetSine,
                y: target.center.y + x * targetSine + y * targetCosine
            )
        }

        let transformedOuter = outer.enumerated().map { offset, point in
            let index = CanonicalLipGeometry.outerLipIndices.indices.contains(offset) ?
                CanonicalLipGeometry.outerLipIndices[offset] : nil
            return transform(point, landmarkIndex: index)
        }
        let transformedInner = inner.enumerated().map { offset, point in
            let index = CanonicalLipGeometry.innerLipIndices.indices.contains(offset) ?
                CanonicalLipGeometry.innerLipIndices[offset] : nil
            return transform(point, landmarkIndex: index)
        }
        let transformedMeshPoints = meshPointsByIndex.reduce(
            into: [Int: LipMeshPoint]()
        ) { result, entry in
            let (index, point) = entry
            result[index] = LipMeshPoint(
                screen: transform(point.screen, landmarkIndex: index),
                normalized: point.normalized,
                uv: point.uv
            )
        }

        return LipContour(
            outer: transformedOuter,
            inner: transformedInner,
            outer3D: outer3D,
            inner3D: inner3D,
            outerUV: outerUV,
            innerUV: innerUV,
            meshPointsByIndex: transformedMeshPoints,
            surfaceBindingsByIndex: surfaceBindingsByIndex,
            surfaceVertexCount: surfaceVertexCount,
            surfaceTriangleIndexCount: surfaceTriangleIndexCount,
            surfaceTopologySignature: surfaceTopologySignature,
            surfaceCarrierCreatedAt: surfaceCarrierCreatedAt,
            surfaceCarrierSupportsDeformation: surfaceCarrierSupportsDeformation,
            faceGeometryPose: faceGeometryPose,
            renderedInnerOpeningRatio: target.isConfidentlyClosedMouth ?
                0 : renderedInnerOpeningRatio,
            isStationary: isStationary
        )
    }

    func offsettingScreenPoints(
        using displacementByLandmark: [Int: CGVector],
        amount: CGFloat,
        maximumDisplacement: CGFloat,
        preservesSurfaceDeformation: Bool = false
    ) -> LipContour? {
        let clampedAmount = max(0, min(amount, 1))

        func offsetPoint(_ point: LipMeshPoint, landmarkIndex: Int) -> LipMeshPoint? {
            let displacement = displacementByLandmark[landmarkIndex] ?? .zero
            guard displacement.dx.isFinite,
                  displacement.dy.isFinite else {
                return nil
            }
            var applied = CGVector(
                dx: displacement.dx * clampedAmount,
                dy: displacement.dy * clampedAmount
            )
            let length = hypot(applied.dx, applied.dy)
            guard length.isFinite else {
                return nil
            }
            if length > maximumDisplacement,
               maximumDisplacement > 0 {
                let scale = maximumDisplacement / length
                applied = CGVector(
                    dx: applied.dx * scale,
                    dy: applied.dy * scale
                )
            }
            return LipMeshPoint(
                screen: CGPoint(
                    x: point.screen.x + applied.dx,
                    y: point.screen.y + applied.dy
                ),
                normalized: point.normalized,
                uv: point.uv
            )
        }

        var offsetMeshPoints: [Int: LipMeshPoint] = [:]
        offsetMeshPoints.reserveCapacity(meshPointsByIndex.count)
        for landmarkIndex in CanonicalLipGeometry.attentionLipIndices {
            guard let point = meshPointsByIndex[landmarkIndex],
                  let offset = offsetPoint(
                    point,
                    landmarkIndex: landmarkIndex
                  ) else {
                return nil
            }
            offsetMeshPoints[landmarkIndex] = offset
        }

        let offsetOuter = CanonicalLipGeometry.outerLipIndices.compactMap {
            offsetMeshPoints[$0]?.screen
        }
        let offsetInner = CanonicalLipGeometry.innerLipIndices.compactMap {
            offsetMeshPoints[$0]?.screen
        }
        guard offsetOuter.count == CanonicalLipGeometry.outerLipIndices.count,
              offsetInner.count == CanonicalLipGeometry.innerLipIndices.count else {
            return nil
        }

        return LipContour(
            outer: offsetOuter,
            inner: offsetInner,
            outer3D: outer3D,
            inner3D: inner3D,
            outerUV: outerUV,
            innerUV: innerUV,
            meshPointsByIndex: offsetMeshPoints,
            surfaceBindingsByIndex: surfaceBindingsByIndex,
            surfaceVertexCount: surfaceVertexCount,
            surfaceTriangleIndexCount: surfaceTriangleIndexCount,
            surfaceTopologySignature: surfaceTopologySignature,
            surfaceCarrierCreatedAt: surfaceCarrierCreatedAt,
            surfaceCarrierSupportsDeformation: preservesSurfaceDeformation &&
                surfaceCarrierSupportsDeformation,
            faceGeometryPose: faceGeometryPose,
            renderedInnerOpeningRatio: renderedInnerOpeningRatio,
            isStationary: isStationary
        )
    }
}

private struct FaceGeometryPose {
    let matrix: simd_float4x4
    let scale: Float

    init?(transform: TransformMatrix?) {
        guard let transform,
              transform.rows == 4,
              transform.columns == 4 else {
            return nil
        }

        var values = [Float](repeating: 0, count: 16)
        for row in 0..<4 {
            for column in 0..<4 {
                let value = transform.value(atRow: UInt(row), column: UInt(column))
                guard value.isFinite else {
                    return nil
                }
                values[row * 4 + column] = value
            }
        }

        matrix = simd_float4x4(columns: (
            SIMD4<Float>(values[0], values[4], values[8], values[12]),
            SIMD4<Float>(values[1], values[5], values[9], values[13]),
            SIMD4<Float>(values[2], values[6], values[10], values[14]),
            SIMD4<Float>(values[3], values[7], values[11], values[15])
        ))

        let xScale = simd_length(SIMD3<Float>(values[0], values[4], values[8]))
        let yScale = simd_length(SIMD3<Float>(values[1], values[5], values[9]))
        let zScale = simd_length(SIMD3<Float>(values[2], values[6], values[10]))
        scale = max((xScale + yScale + zScale) / 3, 0.0001)
    }

    func relativeCanonicalDepth(for index: Int) -> Float? {
        guard let point = transformedCanonicalPoint(for: index) else {
            return nil
        }

        var totalDepth: Float = 0
        var count: Float = 0
        for lipIndex in CanonicalLipGeometry.allLipIndices {
            guard let lipPoint = transformedCanonicalPoint(for: lipIndex) else {
                continue
            }
            totalDepth += lipPoint.z
            count += 1
        }
        guard count > 0 else {
            return nil
        }
        return (point.z - totalDepth / count) / scale
    }

    private func transformedCanonicalPoint(for index: Int) -> SIMD3<Float>? {
        guard let canonical = CanonicalLipGeometry.canonicalVertex(for: index) else {
            return nil
        }

        let transformed = matrix * SIMD4<Float>(canonical.x, canonical.y, canonical.z, 1)
        guard transformed.w != 0 else {
            return SIMD3<Float>(transformed.x, transformed.y, transformed.z)
        }
        return SIMD3<Float>(
            transformed.x / transformed.w,
            transformed.y / transformed.w,
            transformed.z / transformed.w
        )
    }
}

private struct LipPose {
    let center: CGPoint
    let width: CGFloat
    let angle: CGFloat
}

private enum LipOuterMargin {
    // These fractions preserve the tuned margins around an 80-point mouth,
    // but shrink with the detected lip contour when the face moves away.
    // Absolute upper bounds keep close-up faces from producing oversized
    // textures or SceneKit carrier geometry.
    private static let samplingFraction: CGFloat = 0.075
    private static let carrierFraction: CGFloat = 0.125

    static func sampling(for lipWidth: CGFloat) -> CGFloat {
        min(max(lipWidth, 0) * samplingFraction, 10)
    }

    static func carrier(for lipWidth: CGFloat) -> CGFloat {
        min(max(lipWidth, 0) * carrierFraction, 14)
    }
}

private struct LipMotionPose {
    let center: CGPoint
    let width: CGFloat
    let angle: CGFloat
    let opening: CGFloat
    let jawOpen: CGFloat
    let smile: CGFloat
    let pucker: CGFloat
    let upperRaise: CGFloat
    let lowerDrop: CGFloat

    init(center: CGPoint,
         width: CGFloat,
         angle: CGFloat,
         opening: CGFloat,
         jawOpen: CGFloat = 0,
         smile: CGFloat = 0,
         pucker: CGFloat = 0,
         upperRaise: CGFloat = 0,
         lowerDrop: CGFloat = 0) {
        self.center = center
        self.width = width
        self.angle = angle
        self.opening = opening
        self.jawOpen = jawOpen
        self.smile = smile
        self.pucker = pucker
        self.upperRaise = upperRaise
        self.lowerDrop = lowerDrop
    }

    init?(contourPose: LipPose?) {
        guard let contourPose else {
            return nil
        }
        center = contourPose.center
        width = contourPose.width
        angle = contourPose.angle
        opening = 0
        jawOpen = 0
        smile = 0
        pucker = 0
        upperRaise = 0
        lowerDrop = 0
    }

    init?(left: CGPoint, right: CGPoint, top: CGPoint, bottom: CGPoint) {
        let dx = right.x - left.x
        let dy = right.y - left.y
        let width = hypot(dx, dy)
        guard width.isFinite, width > 1 else {
            return nil
        }

        center = CGPoint(
            x: (left.x + right.x + top.x + bottom.x) * 0.25,
            y: (left.y + right.y + top.y + bottom.y) * 0.25
        )
        self.width = width
        angle = atan2(dy, dx)
        opening = hypot(bottom.x - top.x, bottom.y - top.y) / max(width, 1)
        jawOpen = 0
        smile = 0
        pucker = 0
        upperRaise = 0
        lowerDrop = 0
    }

    init?(left: CGPoint, right: CGPoint, top: CGPoint, bottom: CGPoint, scaleBasis: CGFloat) {
        let dx = right.x - left.x
        let dy = right.y - left.y
        guard scaleBasis.isFinite, scaleBasis > 1 else {
            return nil
        }

        center = CGPoint(
            x: (left.x + right.x + top.x + bottom.x) * 0.25,
            y: (left.y + right.y + top.y + bottom.y) * 0.25
        )
        width = scaleBasis
        angle = atan2(dy, dx)
        opening = hypot(bottom.x - top.x, bottom.y - top.y) / max(scaleBasis, 1)
        jawOpen = 0
        smile = 0
        pucker = 0
        upperRaise = 0
        lowerDrop = 0
    }

    init?(rigidCenter: CGPoint,
          rigidLeft: CGPoint,
          rigidRight: CGPoint,
          openingDistance: CGFloat,
          jawOpen: CGFloat = 0,
          smile: CGFloat = 0,
          pucker: CGFloat = 0,
          upperRaise: CGFloat = 0,
          lowerDrop: CGFloat = 0) {
        let dx = rigidRight.x - rigidLeft.x
        let dy = rigidRight.y - rigidLeft.y
        let width = hypot(dx, dy)
        guard rigidCenter.x.isFinite,
              rigidCenter.y.isFinite,
              width.isFinite,
              width > 1,
              openingDistance.isFinite,
              openingDistance >= 0 else {
            return nil
        }

        center = rigidCenter
        self.width = width
        angle = atan2(dy, dx)
        opening = openingDistance / width
        self.jawOpen = jawOpen
        self.smile = smile
        self.pucker = pucker
        self.upperRaise = upperRaise
        self.lowerDrop = lowerDrop
    }

    var isConfidentlyClosedMouth: Bool {
        // A smile separates ARKit's geometric lip vertices slightly even when
        // there is no real mouth opening. Raise the geometric close threshold
        // with smile intensity and require the independent jaw blendshape to
        // agree before closing the rendered aperture.
        let smileAmount = min(max(smile, 0), 1)
        let geometricCloseThreshold = 0.005 + smileAmount * 0.020
        return jawOpen < 0.055 && opening < geometricCloseThreshold
    }
}

private struct FaceLocalMouthFrame {
    let center: SIMD3<Float>
    let xAxis: SIMD3<Float>
    let downAxis: SIMD3<Float>
    let normalAxis: SIMD3<Float>
    let width: Float
    let smileExpansion: Float
    let openingRatio: Float
    let referenceWidth: Float
    let verticalAlignment: Float
    let jawOpen: Float
    let smileLeft: Float
    let smileRight: Float
    let upperLipRaise: Float
    let lowerLipDrop: Float
    let mouthPucker: Float
    let mouthFunnel: Float

    func translated(by offset: SIMD3<Float>) -> FaceLocalMouthFrame {
        FaceLocalMouthFrame(
            center: center + offset,
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
}

private enum CanonicalLipGeometry {
    static let outerLipIndices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 409, 270, 269, 267, 0, 37, 39, 40, 185
    ]
    static let innerLipIndices = [
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 415, 310, 311, 312, 13, 82, 81, 80, 191
    ]
    static let upperDebugIndices = [
        185, 40, 39, 37, 0, 267, 269, 270, 409,
        408, 304, 303, 302, 11, 72, 73, 74, 184
    ]
    static let attentionLipIndices = [
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
        291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
        308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
        76, 77, 90, 180, 85, 16, 315, 404, 320, 307,
        306, 184, 74, 73, 72, 11, 302, 303, 304, 408,
        62, 96, 89, 179, 86, 15, 316, 403, 319, 325,
        292, 183, 42, 41, 38, 12, 268, 271, 272, 407
    ]
    static let allLipIndices = attentionLipIndices

    private static let outerLipIndexSet = Set(outerLipIndices)
    private static let innerLipIndexSet = Set(innerLipIndices)
    private static let cornerLipIndexSet = Set([61, 291, 78, 308, 76, 306, 62, 292])
    private static let cornerSupportLipIndexSet = Set([185, 40, 191, 80, 184, 74, 183, 42, 409, 270, 415, 310, 408, 304, 407, 272])
    private static let upperOuterLipIndexSet = Set([409, 270, 269, 267, 0, 37, 39, 40, 185])
    private static let upperSupportLipIndexSet = Set([408, 304, 303, 302, 11, 72, 73, 74, 184])
    private static let upperInnerSupportLipIndexSet = Set([183, 42, 41, 38, 12, 268, 271, 272, 407])
    private static let upperInnerLipIndexSet = Set([415, 310, 311, 312, 13, 82, 81, 80, 191])
    private static let lowerOuterLipIndexSet = Set([146, 91, 181, 84, 17, 314, 405, 321, 375])
    private static let lowerSupportLipIndexSet = Set([77, 90, 180, 85, 16, 315, 404, 320, 307])
    private static let lowerInnerSupportLipIndexSet = Set([96, 89, 179, 86, 15, 316, 403, 319, 325])
    private static let lowerInnerLipIndexSet = Set([95, 88, 178, 87, 14, 317, 402, 318, 324])

    static func isOuterLipIndex(_ index: Int) -> Bool {
        outerLipIndexSet.contains(index)
    }

    static func isInnerLipIndex(_ index: Int) -> Bool {
        innerLipIndexSet.contains(index)
    }

    static func expectedSurfaceSide(for index: Int) -> Float {
        if upperOuterLipIndexSet.contains(index) ||
            upperSupportLipIndexSet.contains(index) ||
            upperInnerSupportLipIndexSet.contains(index) ||
            upperInnerLipIndexSet.contains(index) {
            return -1
        }
        if lowerOuterLipIndexSet.contains(index) ||
            lowerSupportLipIndexSet.contains(index) ||
            lowerInnerSupportLipIndexSet.contains(index) ||
            lowerInnerLipIndexSet.contains(index) {
            return 1
        }
        return 0
    }

    static func openingMotionFactor(for index: Int) -> CGFloat {
        // Translate every layer on the same side by the same amount and taper
        // the displacement toward the shared mouth corners. Varying the factor
        // per concentric band shears thin triangles; moving the corner-adjacent
        // vertices at full strength can invert the two corner fans.
        let side = expectedSurfaceSide(for: index)
        guard side != 0,
              let point = canonicalVertexByIndex[index],
              let leftCorner = canonicalVertexByIndex[61],
              let rightCorner = canonicalVertexByIndex[291] else {
            return 0
        }
        let centerX = (leftCorner.x + rightCorner.x) * 0.5
        let halfWidth = max(abs(rightCorner.x - leftCorner.x) * 0.5, 0.000_1)
        let distanceFromCenter = min(abs(point.x - centerX) / halfWidth, 1)
        let cornerTaper = pow(max(1 - distanceFromCenter, 0), 2.5)
        return CGFloat(side * cornerTaper * 0.50)
    }

    static func upperLipSmileLiftFactor(for index: Int) -> Float {
        if upperOuterLipIndexSet.contains(index) {
            return 0.95
        }
        if upperSupportLipIndexSet.contains(index) {
            return 0.76
        }
        if upperInnerSupportLipIndexSet.contains(index) {
            return 0.58
        }
        if index == 13 || index == 12 || index == 11 {
            return 0.54
        }
        return 0
    }

    static func smileStretchFactor(for index: Int) -> Float {
        if cornerLipIndexSet.contains(index) {
            return 0.060
        }
        if cornerSupportLipIndexSet.contains(index) {
            return 0.035
        }
        if upperOuterLipIndexSet.contains(index) || lowerOuterLipIndexSet.contains(index) {
            return 0.018
        }
        return 0
    }

    static func jawOpenDropFactor(for index: Int) -> Float {
        if lowerOuterLipIndexSet.contains(index) {
            return 0.12
        }
        if lowerSupportLipIndexSet.contains(index) {
            return 0.09
        }
        if lowerInnerSupportLipIndexSet.contains(index) {
            return 0.05
        }
        if lowerInnerLipIndexSet.contains(index) {
            return 0.10
        }
        return 0
    }

    static func lowerLipDropFactor(for index: Int) -> Float {
        if lowerOuterLipIndexSet.contains(index) {
            return 1.0
        }
        if lowerSupportLipIndexSet.contains(index) {
            return 0.82
        }
        if lowerInnerSupportLipIndexSet.contains(index) {
            return 0.62
        }
        if index == 14 || index == 15 || index == 16 || index == 17 {
            return 0.70
        }
        return 0
    }

    static func meshExpansionScale(for index: Int) -> CGFloat {
        // MediaPipe owns the visible contour. Expanding individual vertices here
        // makes the rendered boundary drift outside the detected lip edge.
        return 1
    }

    private struct FaceModelLipTopology {
        let surfaceTriangles: [(Int, Int, Int)]
        let innerFillTriangles: [(Int, Int, Int)]
    }

    private static let faceModelTopology: FaceModelLipTopology? = loadFaceModelLipTopology()

    private static let fallbackLipMeshTriangles: [(Int, Int, Int)] = [
        (267, 0, 302),
        (37, 72, 0),
        (11, 302, 0),
        (11, 0, 72),
        (267, 302, 269),
        (37, 39, 72),
        (303, 269, 302),
        (73, 72, 39),
        (269, 303, 270),
        (39, 40, 73),
        (304, 270, 303),
        (74, 73, 40),
        (304, 408, 270),
        (74, 40, 184),
        (409, 270, 408),
        (185, 184, 40),
        (272, 310, 407),
        (42, 183, 80),
        (415, 407, 310),
        (191, 80, 183),
        (307, 375, 306),
        (77, 76, 146),
        (291, 306, 375),
        (61, 146, 76),
        (320, 321, 307),
        (90, 77, 91),
        (375, 307, 321),
        (146, 91, 77),
        (405, 321, 404),
        (181, 180, 91),
        (320, 404, 321),
        (90, 91, 180),
        (17, 314, 16),
        (17, 16, 84),
        (315, 16, 314),
        (85, 84, 16),
        (325, 292, 324),
        (96, 95, 62),
        (308, 324, 292),
        (78, 62, 95),
        (306, 292, 307),
        (76, 77, 62),
        (325, 307, 292),
        (96, 62, 77),
        (302, 268, 303),
        (72, 73, 38),
        (271, 303, 268),
        (41, 38, 73),
        (303, 271, 304),
        (73, 74, 41),
        (272, 304, 271),
        (42, 41, 74),
        (304, 272, 408),
        (74, 184, 42),
        (407, 408, 272),
        (183, 42, 184),
        (319, 320, 325),
        (89, 96, 90),
        (307, 325, 320),
        (77, 90, 96),
        (404, 320, 403),
        (180, 179, 90),
        (319, 403, 320),
        (89, 90, 179),
        (16, 315, 15),
        (16, 15, 85),
        (316, 15, 315),
        (86, 85, 15),
        (15, 316, 14),
        (15, 14, 86),
        (317, 14, 316),
        (87, 86, 14),
        (402, 403, 318),
        (178, 88, 179),
        (319, 318, 403),
        (89, 179, 88),
        (324, 318, 325),
        (95, 96, 88),
        (319, 325, 318),
        (89, 88, 96),
        (271, 311, 272),
        (41, 42, 81),
        (310, 272, 311),
        (80, 81, 42),
        (268, 312, 271),
        (38, 41, 82),
        (311, 271, 312),
        (81, 82, 41),
        (407, 415, 292),
        (183, 62, 191),
        (308, 292, 415),
        (78, 191, 62),
        (11, 12, 302),
        (11, 72, 12),
        (268, 302, 12),
        (38, 12, 72),
        (12, 13, 268),
        (12, 38, 13),
        (312, 268, 13),
        (82, 13, 38),
        (316, 403, 317),
        (86, 87, 179),
        (402, 317, 403),
        (178, 179, 87),
        (315, 404, 316),
        (85, 86, 180),
        (403, 316, 404),
        (179, 180, 86),
        (314, 405, 315),
        (84, 85, 181),
        (404, 315, 405),
        (180, 181, 85),
        (408, 407, 306),
        (184, 76, 183),
        (292, 306, 407),
        (62, 183, 76),
        (408, 306, 409),
        (184, 185, 76),
        (291, 409, 306),
        (61, 76, 185)
    ]

    // This checked-in annulus has exactly the outer and inner 20-point boundary
    // loops. Keep rendering deterministic; the optional OBJ topology must never
    // reintroduce triangles that close the mouth aperture.
    static let lipMeshTriangles: [(Int, Int, Int)] = fallbackLipMeshTriangles

    private static let uvBounds = CGRect(
        x: 0.381974,
        y: 0.262981,
        width: 0.236052,
        height: 0.086615
    )

    private static let uvByIndex: [Int: CGPoint] = [
        0: CGPoint(x: 0.499977, y: 0.347466),
        13: CGPoint(x: 0.500023, y: 0.307652),
        14: CGPoint(x: 0.499977, y: 0.304722),
        17: CGPoint(x: 0.499977, y: 0.262981),
        37: CGPoint(x: 0.471751, y: 0.349596),
        39: CGPoint(x: 0.439785, y: 0.342771),
        40: CGPoint(x: 0.414617, y: 0.333459),
        61: CGPoint(x: 0.381974, y: 0.305289),
        78: CGPoint(x: 0.403629, y: 0.306047),
        80: CGPoint(x: 0.431158, y: 0.307634),
        81: CGPoint(x: 0.452182, y: 0.307634),
        82: CGPoint(x: 0.475387, y: 0.307634),
        84: CGPoint(x: 0.472329, y: 0.263774),
        87: CGPoint(x: 0.473033, y: 0.304722),
        88: CGPoint(x: 0.427942, y: 0.304722),
        91: CGPoint(x: 0.418309, y: 0.279937),
        95: CGPoint(x: 0.413200, y: 0.304600),
        146: CGPoint(x: 0.396100, y: 0.289783),
        178: CGPoint(x: 0.448662, y: 0.304722),
        181: CGPoint(x: 0.444832, y: 0.269206),
        185: CGPoint(x: 0.392400, y: 0.322297),
        191: CGPoint(x: 0.413386, y: 0.307634),
        267: CGPoint(x: 0.528249, y: 0.349596),
        269: CGPoint(x: 0.560215, y: 0.342771),
        270: CGPoint(x: 0.585384, y: 0.333459),
        291: CGPoint(x: 0.618026, y: 0.305289),
        308: CGPoint(x: 0.596371, y: 0.306047),
        310: CGPoint(x: 0.568842, y: 0.307634),
        311: CGPoint(x: 0.547818, y: 0.307634),
        312: CGPoint(x: 0.524613, y: 0.307634),
        314: CGPoint(x: 0.527671, y: 0.263774),
        317: CGPoint(x: 0.526967, y: 0.304722),
        318: CGPoint(x: 0.572058, y: 0.304722),
        321: CGPoint(x: 0.581691, y: 0.279937),
        324: CGPoint(x: 0.586800, y: 0.304600),
        375: CGPoint(x: 0.603900, y: 0.289783),
        402: CGPoint(x: 0.551338, y: 0.304722),
        405: CGPoint(x: 0.555168, y: 0.269206),
        409: CGPoint(x: 0.607600, y: 0.322297),
        415: CGPoint(x: 0.586614, y: 0.307634),
        11: CGPoint(x: 0.500023, y: 0.333766),
        12: CGPoint(x: 0.500016, y: 0.320776),
        15: CGPoint(x: 0.499977, y: 0.294066),
        16: CGPoint(x: 0.499977, y: 0.280615),
        38: CGPoint(x: 0.474155, y: 0.319808),
        41: CGPoint(x: 0.450374, y: 0.319139),
        42: CGPoint(x: 0.428771, y: 0.317309),
        62: CGPoint(x: 0.392389, y: 0.305797),
        72: CGPoint(x: 0.472879, y: 0.333802),
        73: CGPoint(x: 0.446828, y: 0.331473),
        74: CGPoint(x: 0.422762, y: 0.326110),
        76: CGPoint(x: 0.388103, y: 0.306039),
        77: CGPoint(x: 0.403039, y: 0.293460),
        85: CGPoint(x: 0.473087, y: 0.282143),
        86: CGPoint(x: 0.473122, y: 0.295374),
        89: CGPoint(x: 0.426479, y: 0.296460),
        90: CGPoint(x: 0.423162, y: 0.288154),
        96: CGPoint(x: 0.409626, y: 0.298177),
        179: CGPoint(x: 0.448020, y: 0.295368),
        180: CGPoint(x: 0.447112, y: 0.284192),
        183: CGPoint(x: 0.406787, y: 0.314327),
        184: CGPoint(x: 0.400738, y: 0.318931),
        268: CGPoint(x: 0.525850, y: 0.319809),
        271: CGPoint(x: 0.549626, y: 0.319139),
        272: CGPoint(x: 0.571228, y: 0.317308),
        292: CGPoint(x: 0.607591, y: 0.305797),
        302: CGPoint(x: 0.527121, y: 0.333802),
        303: CGPoint(x: 0.553172, y: 0.331473),
        304: CGPoint(x: 0.577238, y: 0.326110),
        306: CGPoint(x: 0.611897, y: 0.306039),
        307: CGPoint(x: 0.596961, y: 0.293460),
        315: CGPoint(x: 0.526913, y: 0.282143),
        316: CGPoint(x: 0.526878, y: 0.295374),
        319: CGPoint(x: 0.573521, y: 0.296460),
        320: CGPoint(x: 0.576838, y: 0.288154),
        325: CGPoint(x: 0.590372, y: 0.298177),
        403: CGPoint(x: 0.551980, y: 0.295368),
        404: CGPoint(x: 0.552888, y: 0.284192),
        407: CGPoint(x: 0.593203, y: 0.314324),
        408: CGPoint(x: 0.599262, y: 0.318931)
    ]

    private static let canonicalVertexByIndex: [Int: SIMD3<Float>] = [
        0: SIMD3<Float>(0.000000, -3.406404, 5.979507),
        13: SIMD3<Float>(0.000000, -3.994436, 5.219482),
        14: SIMD3<Float>(0.000000, -4.542400, 5.404754),
        17: SIMD3<Float>(0.000000, -5.365123, 5.535441),
        37: SIMD3<Float>(-0.711452, -3.329355, 5.877044),
        39: SIMD3<Float>(-1.431615, -3.500953, 5.496189),
        40: SIMD3<Float>(-1.914910, -3.803146, 5.028930),
        61: SIMD3<Float>(-2.456206, -4.342621, 4.283884),
        78: SIMD3<Float>(-2.153084, -4.276322, 4.038093),
        80: SIMD3<Float>(-1.469132, -4.036351, 4.604908),
        81: SIMD3<Float>(-1.024340, -3.989851, 4.926693),
        82: SIMD3<Float>(-0.533422, -3.993222, 5.138202),
        84: SIMD3<Float>(-0.699606, -5.291850, 5.448304),
        87: SIMD3<Float>(-0.583218, -4.517982, 5.339869),
        88: SIMD3<Float>(-1.537170, -4.423206, 4.745470),
        91: SIMD3<Float>(-1.838624, -4.828746, 4.823737),
        95: SIMD3<Float>(-1.826614, -4.399531, 4.399021),
        146: SIMD3<Float>(-2.196121, -4.598322, 4.479786),
        178: SIMD3<Float>(-1.098819, -4.458788, 5.120727),
        181: SIMD3<Float>(-1.325085, -5.106507, 5.205010),
        185: SIMD3<Float>(-2.285339, -4.051196, 4.582438),
        191: SIMD3<Float>(-1.845110, -4.098880, 4.247264),
        267: SIMD3<Float>(0.711452, -3.329355, 5.877044),
        269: SIMD3<Float>(1.431615, -3.500953, 5.496189),
        270: SIMD3<Float>(1.914910, -3.803146, 5.028930),
        291: SIMD3<Float>(2.456206, -4.342621, 4.283884),
        308: SIMD3<Float>(2.153084, -4.276322, 4.038093),
        310: SIMD3<Float>(1.469132, -4.036351, 4.604908),
        311: SIMD3<Float>(1.024340, -3.989851, 4.926693),
        312: SIMD3<Float>(0.533422, -3.993222, 5.138202),
        314: SIMD3<Float>(0.699606, -5.291850, 5.448304),
        317: SIMD3<Float>(0.583218, -4.517982, 5.339869),
        318: SIMD3<Float>(1.537170, -4.423206, 4.745470),
        321: SIMD3<Float>(1.838624, -4.828746, 4.823737),
        324: SIMD3<Float>(1.826614, -4.399531, 4.399021),
        375: SIMD3<Float>(2.196121, -4.598322, 4.479786),
        402: SIMD3<Float>(1.098819, -4.458788, 5.120727),
        405: SIMD3<Float>(1.325085, -5.106507, 5.205010),
        409: SIMD3<Float>(2.285339, -4.051196, 4.582438),
        415: SIMD3<Float>(1.845110, -4.098880, 4.247264),
        11: SIMD3<Float>(0.000000, -3.706811, 5.864924),
        12: SIMD3<Float>(0.000000, -3.918301, 5.569430),
        15: SIMD3<Float>(0.000000, -4.745577, 5.529457),
        16: SIMD3<Float>(0.000000, -5.019567, 5.601448),
        38: SIMD3<Float>(-0.606033, -3.924562, 5.444923),
        41: SIMD3<Float>(-1.131043, -3.973937, 5.189648),
        42: SIMD3<Float>(-1.563548, -4.082763, 4.842263),
        62: SIMD3<Float>(-2.204823, -4.304508, 4.162499),
        72: SIMD3<Float>(-0.672728, -3.688016, 5.737804),
        73: SIMD3<Float>(-1.262560, -3.787691, 5.417779),
        74: SIMD3<Float>(-1.732553, -3.952767, 5.000579),
        76: SIMD3<Float>(-2.321234, -4.329069, 4.258156),
        77: SIMD3<Float>(-2.056846, -4.477671, 4.520883),
        85: SIMD3<Float>(-0.669687, -4.949770, 5.509612),
        86: SIMD3<Float>(-0.630947, -4.695101, 5.449371),
        89: SIMD3<Float>(-1.615600, -4.475942, 4.813632),
        90: SIMD3<Float>(-1.729053, -4.618680, 4.854463),
        96: SIMD3<Float>(-1.929558, -4.411831, 4.497052),
        179: SIMD3<Float>(-1.181124, -4.579996, 5.189564),
        180: SIMD3<Float>(-1.255818, -4.787901, 5.237051),
        183: SIMD3<Float>(-1.953754, -4.183892, 4.431713),
        184: SIMD3<Float>(-2.117802, -4.137093, 4.555096),
        268: SIMD3<Float>(0.606033, -3.924562, 5.444923),
        271: SIMD3<Float>(1.131043, -3.973937, 5.189648),
        272: SIMD3<Float>(1.563548, -4.082763, 4.842263),
        292: SIMD3<Float>(2.204823, -4.304508, 4.162499),
        302: SIMD3<Float>(0.672728, -3.688016, 5.737804),
        303: SIMD3<Float>(1.262560, -3.787691, 5.417779),
        304: SIMD3<Float>(1.732553, -3.952767, 5.000579),
        306: SIMD3<Float>(2.321234, -4.329069, 4.258156),
        307: SIMD3<Float>(2.056846, -4.477671, 4.520883),
        315: SIMD3<Float>(0.669687, -4.949770, 5.509612),
        316: SIMD3<Float>(0.630947, -4.695101, 5.449371),
        319: SIMD3<Float>(1.615600, -4.475942, 4.813632),
        320: SIMD3<Float>(1.729053, -4.618680, 4.854463),
        325: SIMD3<Float>(1.929558, -4.411831, 4.497052),
        403: SIMD3<Float>(1.181124, -4.579996, 5.189564),
        404: SIMD3<Float>(1.255818, -4.787901, 5.237051),
        407: SIMD3<Float>(1.953754, -4.183892, 4.431713),
        408: SIMD3<Float>(2.117802, -4.137093, 4.555096)
    ]
    //здесь правка
    static func normalizedUV(for index: Int) -> CGPoint? {
        guard let uv = uvByIndex[index] else {
            return nil
        }

        let normalizedX = min(
            max((uv.x - uvBounds.minX) / uvBounds.width, 0),
            1
        )
        let normalizedY = min(
            max((uv.y - uvBounds.minY) / uvBounds.height, 0),
            1
        )

        let horizontalPadding: CGFloat = 0.08
        let verticalPadding: CGFloat = 0.12

        return CGPoint(
            x: horizontalPadding +
                normalizedX * (1 - horizontalPadding * 2),
            y: verticalPadding +
                normalizedY * (1 - verticalPadding * 2)
        )
    }

    private static func rawUV(for index: Int) -> CGPoint? {
        uvByIndex[index]
    }

    static func canonicalVertex(for index: Int) -> SIMD3<Float>? {
        canonicalVertexByIndex[index]
    }

    private static func loadFaceModelLipTopology() -> FaceModelLipTopology? {
        guard let url = Bundle.main.url(forResource: "face_model_with_iris", withExtension: "obj"),
              let contents = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }

        var textureCoordinates: [CGPoint] = []
        var faceTextureTriangles: [(Int, Int, Int)] = []

        for rawLine in contents.split(whereSeparator: \.isNewline) {
            let parts = rawLine.split(whereSeparator: { $0 == " " || $0 == "\t" })
            guard let prefix = parts.first else {
                continue
            }

            if prefix == "vt", parts.count >= 3,
               let u = Double(parts[1]),
               let v = Double(parts[2]) {
                textureCoordinates.append(CGPoint(x: CGFloat(u), y: CGFloat(v)))
            } else if prefix == "f", parts.count == 4 {
                var triangle: [Int] = []
                triangle.reserveCapacity(3)
                for part in parts.dropFirst() {
                    let indices = part.split(separator: "/", omittingEmptySubsequences: false)
                    guard indices.count >= 2,
                          let textureIndex = Int(indices[1]) else {
                        triangle.removeAll()
                        break
                    }
                    triangle.append(textureIndex - 1)
                }

                if triangle.count == 3 {
                    faceTextureTriangles.append((triangle[0], triangle[1], triangle[2]))
                }
            }
        }

        guard !textureCoordinates.isEmpty,
              !faceTextureTriangles.isEmpty else {
            return nil
        }

        let lipUVs = allLipIndices.compactMap { index -> (index: Int, uv: CGPoint)? in
            guard let uv = rawUV(for: index) else {
                return nil
            }
            return (index, uv)
        }
        guard lipUVs.count == allLipIndices.count else {
            return nil
        }

        func lipIndex(for textureIndex: Int) -> Int? {
            guard textureCoordinates.indices.contains(textureIndex) else {
                return nil
            }

            let uv = textureCoordinates[textureIndex]
            var bestMatch: Int?
            var bestDistance = CGFloat.greatestFiniteMagnitude
            for candidate in lipUVs {
                let distance = abs(candidate.uv.x - uv.x) + abs(candidate.uv.y - uv.y)
                if distance < bestDistance {
                    bestDistance = distance
                    bestMatch = candidate.index
                }
            }

            return bestDistance < 0.00001 ? bestMatch : nil
        }

        var surfaceTriangles: [(Int, Int, Int)] = []
        var innerFillTriangles: [(Int, Int, Int)] = []
        var seenSurfaceTriangles = Set<String>()
        var seenInnerTriangles = Set<String>()

        func key(_ triangle: (Int, Int, Int)) -> String {
            [triangle.0, triangle.1, triangle.2]
                .sorted()
                .map(String.init)
                .joined(separator: "-")
        }

        for textureTriangle in faceTextureTriangles {
            guard let first = lipIndex(for: textureTriangle.0),
                  let second = lipIndex(for: textureTriangle.1),
                  let third = lipIndex(for: textureTriangle.2) else {
                continue
            }

            let triangle = (first, second, third)
            let triangleKey = key(triangle)
            if innerLipIndexSet.contains(first),
               innerLipIndexSet.contains(second),
               innerLipIndexSet.contains(third) {
                if seenInnerTriangles.insert(triangleKey).inserted {
                    innerFillTriangles.append(triangle)
                }
            } else if seenSurfaceTriangles.insert(triangleKey).inserted {
                surfaceTriangles.append(triangle)
            }
        }

        guard surfaceTriangles.count >= fallbackLipMeshTriangles.count else {
            return nil
        }

        return FaceModelLipTopology(
            surfaceTriangles: surfaceTriangles,
            innerFillTriangles: innerFillTriangles
        )
    }
}

fileprivate final class LipMeshRenderer {
    private static let maxSurfaceCarrierAge: CFTimeInterval = 0.50

    private struct MeshVertex {
        let index: Int
        let screen: CGPoint
        let normalized: SIMD3<Float>
        let uv: CGPoint
    }

    private final class DynamicSurfaceGeometryCache {
        let deviceRegistryID: UInt64
        let landmarkIndices: [Int]
        let triangleCount: Int
        private let buffers: [MTLBuffer]
        private let geometries: [SCNGeometry]
        private var nextSlot = 0

        init?(device: MTLDevice,
              vertices: [MeshVertex],
              indices: [Int32],
              material: SCNMaterial) {
            guard !vertices.isEmpty,
                  !indices.isEmpty,
                  indices.count.isMultiple(of: 3) else {
                return nil
            }
            deviceRegistryID = device.registryID
            landmarkIndices = vertices.map(\.index)
            triangleCount = indices.count / 3

            let textureCoordinates = vertices.map {
                CGPoint(x: $0.uv.x, y: 1 - $0.uv.y)
            }
            let textureSource = SCNGeometrySource(textureCoordinates: textureCoordinates)
            let indexData = Data(
                bytes: indices,
                count: indices.count * MemoryLayout<Int32>.size
            )
            let element = SCNGeometryElement(
                data: indexData,
                primitiveType: .triangles,
                primitiveCount: triangleCount,
                bytesPerIndex: MemoryLayout<Int32>.size
            )

            var buffers: [MTLBuffer] = []
            var geometries: [SCNGeometry] = []
            buffers.reserveCapacity(3)
            geometries.reserveCapacity(3)
            let bufferLength = vertices.count * MemoryLayout<SIMD3<Float>>.stride
            for _ in 0..<3 {
                guard let buffer = device.makeBuffer(
                    length: bufferLength,
                    options: .storageModeShared
                ) else {
                    return nil
                }
                let vertexSource = SCNGeometrySource(
                    buffer: buffer,
                    vertexFormat: .float3,
                    semantic: .vertex,
                    vertexCount: vertices.count,
                    dataOffset: 0,
                    dataStride: MemoryLayout<SIMD3<Float>>.stride
                )
                let geometry = SCNGeometry(
                    sources: [vertexSource, textureSource],
                    elements: [element]
                )
                geometry.materials = [material]
                buffers.append(buffer)
                geometries.append(geometry)
            }
            self.buffers = buffers
            self.geometries = geometries
        }

        func geometry(updating positions: [SIMD3<Float>]) -> SCNGeometry? {
            guard positions.count == landmarkIndices.count,
                  !buffers.isEmpty else {
                return nil
            }
            let slot = nextSlot
            nextSlot = (nextSlot + 1) % buffers.count
            let byteCount = positions.count * MemoryLayout<SIMD3<Float>>.stride
            positions.withUnsafeBytes { bytes in
                guard let source = bytes.baseAddress else {
                    return
                }
                buffers[slot].contents().copyMemory(from: source, byteCount: byteCount)
            }
            return geometries[slot]
        }
    }

    private let lipNode = SCNNode()
    private let debugRootNode = SCNNode()
    private let debugOuterNode = SCNNode()
    private let debugInnerNode = SCNNode()
    private let debugUpperNode = SCNNode()
    private let debugMeshNode = SCNNode()
    private let debugFrameNode = SCNNode()
    private let lipMaterial = SCNMaterial()
    private let debugOuterMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemGreen)
    private let debugInnerMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemCyan)
    private let debugUpperMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemRed)
    private let debugMeshMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemYellow.withAlphaComponent(0.82))
    private let debugFrameMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemPink)
    private var occlusionNode: SCNNode?
    private var lastTextureImage: UIImage?
    private let debugLinesEnabled = false
    private var lastLightingFactor: CGFloat = 1
    private var openMouthGeometryCache: DynamicSurfaceGeometryCache?

    init() {
        lipNode.name = "lipstick.mesh"
        lipNode.isHidden = true
        lipNode.renderingOrder = 100
        debugRootNode.name = "lipstick.debug"
        debugRootNode.renderingOrder = 220
        debugOuterNode.name = "lipstick.debug.outer"
        debugInnerNode.name = "lipstick.debug.inner"
        debugUpperNode.name = "lipstick.debug.upper"
        debugMeshNode.name = "lipstick.debug.mesh"
        debugFrameNode.name = "lipstick.debug.frame"
        debugRootNode.addChildNode(debugMeshNode)
        debugRootNode.addChildNode(debugOuterNode)
        debugRootNode.addChildNode(debugInnerNode)
        debugRootNode.addChildNode(debugUpperNode)
        debugRootNode.addChildNode(debugFrameNode)

        lipMaterial.lightingModel = .constant
        lipMaterial.blendMode = .alpha
        lipMaterial.isDoubleSided = true
        lipMaterial.writesToDepthBuffer = false
        // The lipstick mesh is already clipped to the detected lip annulus.
        // Reading the coarse AR face depth can hide the central lip surface,
        // especially while the mouth is closed.
        lipMaterial.readsFromDepthBuffer = false
        lipMaterial.diffuse.magnificationFilter = .linear
        lipMaterial.diffuse.minificationFilter = .linear
        lipMaterial.diffuse.mipFilter = .none
        lipMaterial.diffuse.wrapS = .clamp
        lipMaterial.diffuse.wrapT = .clamp
        lipMaterial.multiply.contents = UIColor.white
        lipMaterial.multiply.intensity = 1
    }

    func updateOpacity(_ opacity: Double) {
        // Zero is reserved for the before/after original-image toggle. At the
        // visible end, retain a little camera detail instead of becoming a
        // fully opaque vector layer.
        lipMaterial.transparency = CGFloat(max(0, min(opacity, 0.99)))
    }

    func updateLightingFactor(_ factor: CGFloat) {
        let clampedFactor = max(0.45, min(factor, 1))
        guard abs(clampedFactor - lastLightingFactor) > 0.006 else {
            return
        }

        lastLightingFactor = clampedFactor
        lipMaterial.multiply.contents = UIColor(
            red: clampedFactor,
            green: clampedFactor,
            blue: clampedFactor,
            alpha: 1
        )
    }

    func attach(to faceNode: SCNNode) {
        if lipNode.parent !== faceNode {
            lipNode.removeFromParentNode()
            faceNode.addChildNode(lipNode)
        }
        if debugRootNode.parent !== faceNode {
            debugRootNode.removeFromParentNode()
            faceNode.addChildNode(debugRootNode)
        }
    }

    func updateOccluder(faceAnchor: ARFaceAnchor, renderer: SCNSceneRenderer, faceNode: SCNNode) {
        if occlusionNode == nil,
           let device = renderer.device,
           let geometry = ARSCNFaceGeometry(device: device) {
            let material = SCNMaterial()
            material.colorBufferWriteMask = []
            material.writesToDepthBuffer = true
            material.readsFromDepthBuffer = true
            geometry.materials = [material]

            let node = SCNNode(geometry: geometry)
            node.name = "face.depth.occluder"
            node.renderingOrder = -100
            faceNode.addChildNode(node)
            occlusionNode = node
        }

        if let occlusionNode, occlusionNode.parent !== faceNode {
            occlusionNode.removeFromParentNode()
            faceNode.addChildNode(occlusionNode)
        }
        (occlusionNode?.geometry as? ARSCNFaceGeometry)?.update(from: faceAnchor.geometry)
    }

    func render(contour: LipContour,
                texture: LipTexture,
                faceGeometry: ARFaceGeometry,
                mouthFrame: FaceLocalMouthFrame,
                renderer: SCNSceneRenderer,
                faceNode: SCNNode,
                contourAge: CFTimeInterval,
                motionDelta: CGFloat) {
        let correctedMouthFrame = mouthFrame

        guard let geometry = makeGeometry(
            contour: contour,
            texture: texture,
            faceGeometry: faceGeometry,
            mouthFrame: correctedMouthFrame,
            renderer: renderer,
            faceNode: faceNode
        ) else {
            clearLip()
            return
        }

        if lastTextureImage !== texture.image {
            lipMaterial.diffuse.contents = texture.image
            lastTextureImage = texture.image
        }
        lipMaterial.readsFromDepthBuffer = false
        lipNode.geometry = geometry
        lipNode.isHidden = false
        LipDebugLog.throttled(
            "lip_mesh_alignment",
            interval: 0.75,
            "lip_mesh alignment shape=mediapipe_screen_locked depth=ar_surface aperture=mediapipe_inner age=\(Self.fmt(CGFloat(contourAge))) motion=\(Self.fmt(motionDelta))"
        )
        if debugLinesEnabled {
            updateDebugLines(contour: contour, mouthFrame: correctedMouthFrame)
            logProjectionDiagnostics(
                contour: contour,
                mouthFrame: correctedMouthFrame,
                renderer: renderer,
                faceNode: faceNode
            )
        } else {
            clearDebugLines()
            logAttachmentDiagnostics(
                contour: contour,
                contourAge: contourAge,
                motionDelta: motionDelta
            )
        }
    }

    func clearLip() {
        lipNode.geometry = nil
        lipNode.isHidden = true
        clearDebugLines()
    }

    func clearAll() {
        clearLip()
        lastTextureImage = nil
        lipMaterial.diffuse.contents = nil
    }

    private static func makeDebugMaterial(color: UIColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.emission.contents = color
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.readsFromDepthBuffer = false
        material.writesToDepthBuffer = false
        return material
    }

    private func makeGeometry(contour: LipContour,
                              texture: LipTexture,
                              faceGeometry: ARFaceGeometry,
                              mouthFrame: FaceLocalMouthFrame,
                              renderer: SCNSceneRenderer,
                              faceNode: SCNNode) -> SCNGeometry? {
        guard let pose = contour.pose else {
            LipDebugLog.throttled(
                "lip_mesh_nil_pose",
                "lip_mesh makeGeometry=nil reason=no_pose outer=\(contour.outer.count) inner=\(contour.inner.count)"
            )
            return nil
        }
        guard contour.outer.count == Self.lipPointCount,
              contour.inner.count == Self.lipPointCount,
              contour.outer3D.count == Self.lipPointCount,
              contour.inner3D.count == Self.lipPointCount,
              contour.outerUV.count == Self.lipPointCount,
              contour.innerUV.count == Self.lipPointCount else {
            LipDebugLog.throttled(
                "lip_mesh_bad_counts",
                "lip_mesh makeGeometry=nil reason=bad_counts outer=\(contour.outer.count) inner=\(contour.inner.count) outer3D=\(contour.outer3D.count) inner3D=\(contour.inner3D.count) outerUV=\(contour.outerUV.count) innerUV=\(contour.innerUV.count)"
            )
            return nil
        }
        guard mouthFrame.width > 0 else {
            LipDebugLog.throttled(
                "lip_mesh_bad_mouth_frame",
                "lip_mesh makeGeometry=nil reason=bad_mouth_frame width=\(mouthFrame.width)"
            )
            return nil
        }

        guard hasCompatibleDepthCarrier(contour, faceGeometry: faceGeometry) else {
            LipDebugLog.throttled(
                "lip_mesh_no_depth_carrier",
                interval: 0.4,
                "lip_mesh hide reason=no_fresh_ar_depth_carrier shape=mediapipe"
            )
            return nil
        }

        let faceVertices = faceGeometry.vertices
        guard let device = renderer.device else {
            return nil
        }
        let mesh = makeMediaPipeLipMesh(
            contour: contour,
            expandsOuterFeatherBoundary: true
        )
        guard mesh.vertices.count >= 4,
              !mesh.indices.isEmpty else {
            LipDebugLog.throttled(
                "lip_mesh_empty_geometry",
                "lip_mesh makeGeometry=nil reason=mediapipe_mesh"
            )
            return nil
        }
        var vertexByLandmark: [Int: MeshVertex] = [:]
        vertexByLandmark.reserveCapacity(mesh.vertices.count)
        for vertex in mesh.vertices {
            guard vertexByLandmark.updateValue(vertex, forKey: vertex.index) == nil else {
                LipDebugLog.throttled(
                    "lip_mesh_duplicate_landmark",
                    "lip_mesh makeGeometry=nil reason=duplicate_landmark index=\(vertex.index)"
                )
                return nil
            }
        }
        var geometryCache = openMouthGeometryCache
        if geometryCache?.deviceRegistryID != device.registryID {
            guard let newCache = DynamicSurfaceGeometryCache(
                    device: device,
                    vertices: mesh.vertices,
                    indices: mesh.indices,
                    material: lipMaterial
                  ) else {
                LipDebugLog.throttled(
                    "lip_mesh_empty_geometry",
                    "lip_mesh makeGeometry=nil reason=dynamic_geometry_cache"
                )
                return nil
            }
            geometryCache = newCache
            openMouthGeometryCache = newCache
        }
        guard let geometryCache else {
            return nil
        }

        let textureWidth = texture.image.cgImage?.width ?? 0
        let textureHeight = texture.image.cgImage?.height ?? 0
        LipDebugLog.throttled(
            "lip_mesh_geometry",
            interval: 0.75,
            "lip_mesh geometry vertices=\(geometryCache.landmarkIndices.count) triangles=\(geometryCache.triangleCount) buffered=true shape=mediapipe_screen_locked depth=ar_surface texture=\(textureWidth)x\(textureHeight) mouthWidth=\(String(format: "%.4f", mouthFrame.width)) poseWidth=\(String(format: "%.1f", pose.width))"
        )

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(geometryCache.landmarkIndices.count)
        for landmarkIndex in geometryCache.landmarkIndices {
            guard let vertex = vertexByLandmark[landmarkIndex],
                  let binding = contour.surfaceBindingsByIndex[landmarkIndex],
                  let depthReference = faceSurfacePoint(
                    for: binding,
                    faceVertices: faceVertices,
                    outwardReference: mouthFrame.normalAxis,
                    surfaceLift: 0.001_50
                  ),
                  let point = screenLockedScenePoint(
                    depthReference: SCNVector3(
                        depthReference.x,
                        depthReference.y,
                        depthReference.z
                    ),
                    mediaPipeScreenPoint: vertex.screen,
                    renderer: renderer,
                    faceNode: faceNode
                  ) else {
                LipDebugLog.throttled(
                    "lip_mesh_screen_lock_vertex_invalid",
                    interval: 0.4,
                    "lip_mesh hide reason=screen_lock_vertex_invalid index=\(landmarkIndex)"
                )
                return nil
            }
            positions.append(SIMD3<Float>(point.x, point.y, point.z))
        }

        guard hasValidScreenLockedGeometry(
            positions,
            landmarkIndices: geometryCache.landmarkIndices,
            meshVerticesByLandmark: vertexByLandmark,
            mouthFrame: mouthFrame,
            renderer: renderer,
            faceNode: faceNode
        ) else {
            LipDebugLog.throttled(
                "lip_mesh_screen_lock_geometry_invalid",
                interval: 0.4,
                "lip_mesh hide reason=invalid_mediapipe_screen_locked_geometry"
            )
            return nil
        }
        return geometryCache.geometry(updating: positions)
    }

    private func screenLockedScenePoint(
        depthReference: SCNVector3,
        mediaPipeScreenPoint: CGPoint,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode
    ) -> SCNVector3? {
        guard mediaPipeScreenPoint.x.isFinite,
              mediaPipeScreenPoint.y.isFinite else {
            return nil
        }

        let worldDepthReference = faceNode.convertPosition(depthReference, to: nil)
        let projectedDepthReference = renderer.projectPoint(worldDepthReference)
        guard projectedDepthReference.z.isFinite,
              projectedDepthReference.z >= 0,
              projectedDepthReference.z <= 1 else {
            return nil
        }

        // MediaPipe owns the keyframe X/Y shape. Any bounded ARKit delta-warp
        // has already been applied to this screen point; ARFaceGeometry also
        // contributes the camera-space depth used for unprojection.
        let worldTarget = renderer.unprojectPoint(
            SCNVector3(
                Float(mediaPipeScreenPoint.x),
                Float(mediaPipeScreenPoint.y),
                projectedDepthReference.z
            )
        )
        guard worldTarget.x.isFinite,
              worldTarget.y.isFinite,
              worldTarget.z.isFinite else {
            return nil
        }

        let localTarget = faceNode.convertPosition(worldTarget, from: nil)
        guard localTarget.x.isFinite,
              localTarget.y.isFinite,
              localTarget.z.isFinite else {
            return nil
        }
        return localTarget
    }

    private func hasValidScreenLockedGeometry(
        _ positions: [SIMD3<Float>],
        landmarkIndices: [Int],
        meshVerticesByLandmark: [Int: MeshVertex],
        mouthFrame: FaceLocalMouthFrame,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode
    ) -> Bool {
        guard positions.count == CanonicalLipGeometry.attentionLipIndices.count,
              landmarkIndices.count == positions.count,
              meshVerticesByLandmark.count == positions.count,
              mouthFrame.width.isFinite,
              mouthFrame.width > 0 else {
            return false
        }

        var seenLandmarks = Set<Int>()
        seenLandmarks.reserveCapacity(positions.count)
        var maximumReprojectionError: CGFloat = 0
        let maximumLocalDistance = mouthFrame.width * 1.75
        let maximumNormalDistance = mouthFrame.width * 1.25

        for (offset, landmarkIndex) in landmarkIndices.enumerated() {
            guard positions.indices.contains(offset),
                  seenLandmarks.insert(landmarkIndex).inserted,
                  let vertex = meshVerticesByLandmark[landmarkIndex] else {
                return false
            }
            let position = positions[offset]
            guard position.x.isFinite,
                  position.y.isFinite,
                  position.z.isFinite else {
                return false
            }

            let relative = position - mouthFrame.center
            let localDistance = simd_length(relative)
            let normalDistance = abs(simd_dot(relative, mouthFrame.normalAxis))
            guard localDistance.isFinite,
                  normalDistance.isFinite,
                  localDistance <= maximumLocalDistance,
                  normalDistance <= maximumNormalDistance else {
                return false
            }

            let localPoint = SCNVector3(position.x, position.y, position.z)
            let worldPoint = faceNode.convertPosition(localPoint, to: nil)
            let projected = renderer.projectPoint(worldPoint)
            guard projected.x.isFinite,
                  projected.y.isFinite,
                  projected.z.isFinite,
                  projected.z >= 0,
                  projected.z <= 1 else {
                return false
            }

            let reprojectionError = hypot(
                CGFloat(projected.x) - vertex.screen.x,
                CGFloat(projected.y) - vertex.screen.y
            )
            guard reprojectionError.isFinite else {
                return false
            }
            maximumReprojectionError = max(
                maximumReprojectionError,
                reprojectionError
            )
        }

        // project(unproject(MediaPipe X/Y, AR depth)) must remain pixel-tight.
        // A larger delta means coordinate spaces were mixed, so hide instead of
        // substituting a generated mouth shape.
        let maximumAllowedReprojectionError: CGFloat = 1.25
        guard maximumReprojectionError <= maximumAllowedReprojectionError else {
            LipDebugLog.throttled(
                "lip_mesh_screen_lock_reprojection",
                interval: 0.4,
                "lip_mesh hide reason=screen_reprojection_error maxPx=\(Self.fmt(maximumReprojectionError)) allowedPx=\(Self.fmt(maximumAllowedReprojectionError))"
            )
            return false
        }
        LipDebugLog.throttled(
            "lip_mesh_screen_lock_ok",
            interval: 0.75,
            "lip_mesh screenLock=ok maxPx=\(Self.fmt(maximumReprojectionError)) vertices=\(positions.count)"
        )
        return true
    }

    private func hasCompatibleDepthCarrier(_ contour: LipContour,
                                           faceGeometry: ARFaceGeometry) -> Bool {
        let faceVertices = faceGeometry.vertices
        let faceTriangleIndices = faceGeometry.triangleIndices
        let carrierAge = contour.surfaceCarrierCreatedAt.map {
            CACurrentMediaTime() - $0
        } ?? .greatestFiniteMagnitude
        guard contour.surfaceVertexCount == faceVertices.count,
              contour.surfaceTriangleIndexCount == faceTriangleIndices.count,
              carrierAge.isFinite,
              carrierAge >= 0,
              carrierAge <= Self.maxSurfaceCarrierAge,
              contour.surfaceTopologySignature == faceSurfaceTopologySignature(
                vertexCount: faceVertices.count,
                triangleIndices: faceTriangleIndices
              ) else {
            return false
        }
        return CanonicalLipGeometry.attentionLipIndices.allSatisfy {
            contour.surfaceBindingsByIndex[$0] != nil
        }
    }

    private func faceSurfacePoint(for binding: LipSurfaceBinding,
                                  faceVertices: [SIMD3<Float>],
                                  outwardReference: SIMD3<Float>,
                                  surfaceLift: Float = 0.000_70) -> SIMD3<Float>? {
        let firstIndex = Int(binding.vertexIndices.x)
        let secondIndex = Int(binding.vertexIndices.y)
        let thirdIndex = Int(binding.vertexIndices.z)
        guard faceVertices.indices.contains(firstIndex),
              faceVertices.indices.contains(secondIndex),
              faceVertices.indices.contains(thirdIndex) else {
            return nil
        }

        let weights = binding.barycentricWeights
        let weightSum = weights.x + weights.y + weights.z
        guard weights.x.isFinite,
              weights.y.isFinite,
              weights.z.isFinite,
              weightSum.isFinite,
              abs(weightSum - 1) < 0.002 else {
            return nil
        }

        let first = faceVertices[firstIndex]
        let second = faceVertices[secondIndex]
        let third = faceVertices[thirdIndex]
        var point = first * weights.x + second * weights.y + third * weights.z
        var normal = simd_cross(second - first, third - first)
        if simd_length_squared(normal) > 0.000_000_000_1 {
            normal = simd_normalize(normal)
            if simd_dot(normal, outwardReference) < 0 {
                normal = -normal
            }
            point += normal * surfaceLift
        } else {
            point += outwardReference * surfaceLift
        }

        guard point.x.isFinite,
              point.y.isFinite,
              point.z.isFinite else {
            return nil
        }
        return point
    }

    private func updateDebugLines(contour: LipContour, mouthFrame: FaceLocalMouthFrame) {
        guard debugLinesEnabled,
              let pose = contour.pose else {
            clearDebugLines()
            return
        }

        let mesh = makeMediaPipeLipMesh(contour: contour)
        let sceneVertices = mesh.vertices.map {
            scenePoint(for: $0, pose: pose, contour: contour, mouthFrame: mouthFrame)
        }
        var pointByIndex: [Int: SCNVector3] = [:]
        pointByIndex.reserveCapacity(mesh.vertices.count)
        for (offset, vertex) in mesh.vertices.enumerated() {
            guard sceneVertices.indices.contains(offset) else {
                continue
            }
            pointByIndex[vertex.index] = sceneVertices[offset]
        }

        debugOuterNode.geometry = lineLoopGeometry(
            indices: CanonicalLipGeometry.outerLipIndices,
            pointByIndex: pointByIndex,
            material: debugOuterMaterial
        )
        debugInnerNode.geometry = lineLoopGeometry(
            indices: CanonicalLipGeometry.innerLipIndices,
            pointByIndex: pointByIndex,
            material: debugInnerMaterial
        )
        debugUpperNode.geometry = lineStripGeometry(
            indices: CanonicalLipGeometry.upperDebugIndices,
            pointByIndex: pointByIndex,
            offset: mouthFrame.normalAxis * mouthFrame.width * 0.018,
            material: debugUpperMaterial
        )
        debugMeshNode.geometry = meshWireGeometry(
            vertices: sceneVertices,
            triangleIndices: mesh.indices,
            material: debugMeshMaterial
        )
        debugFrameNode.geometry = mouthFrameGeometry(
            mouthFrame,
            material: debugFrameMaterial
        )
        debugRootNode.isHidden = false
    }

    private func clearDebugLines() {
        debugOuterNode.geometry = nil
        debugInnerNode.geometry = nil
        debugUpperNode.geometry = nil
        debugMeshNode.geometry = nil
        debugFrameNode.geometry = nil
        debugRootNode.isHidden = true
    }

    private func lineLoopGeometry(indices: [Int],
                                  pointByIndex: [Int: SCNVector3],
                                  material: SCNMaterial) -> SCNGeometry? {
        let points = indices.compactMap { pointByIndex[$0] }
        guard points.count >= 2 else {
            return nil
        }

        var lineIndices = [Int32]()
        lineIndices.reserveCapacity(points.count * 2)
        for index in points.indices {
            lineIndices.append(Int32(index))
            lineIndices.append(Int32((index + 1) % points.count))
        }

        return lineGeometry(vertices: points, indices: lineIndices, material: material)
    }

    private func lineStripGeometry(indices: [Int],
                                   pointByIndex: [Int: SCNVector3],
                                   offset: SIMD3<Float>,
                                   material: SCNMaterial) -> SCNGeometry? {
        let points = indices.compactMap { index -> SCNVector3? in
            guard let point = pointByIndex[index] else {
                return nil
            }
            return SCNVector3(point.x + offset.x, point.y + offset.y, point.z + offset.z)
        }
        guard points.count >= 2 else {
            return nil
        }

        var lineIndices = [Int32]()
        lineIndices.reserveCapacity((points.count - 1) * 2)
        for index in 0..<(points.count - 1) {
            lineIndices.append(Int32(index))
            lineIndices.append(Int32(index + 1))
        }

        return lineGeometry(vertices: points, indices: lineIndices, material: material)
    }

    private func meshWireGeometry(vertices: [SCNVector3],
                                  triangleIndices: [Int32],
                                  material: SCNMaterial) -> SCNGeometry? {
        guard vertices.count >= 3, triangleIndices.count >= 3 else {
            return nil
        }

        var lineIndices = [Int32]()
        lineIndices.reserveCapacity(triangleIndices.count * 2)
        for offset in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let first = triangleIndices[offset]
            let second = triangleIndices[offset + 1]
            let third = triangleIndices[offset + 2]
            lineIndices.append(contentsOf: [first, second, second, third, third, first])
        }

        return lineGeometry(vertices: vertices, indices: lineIndices, material: material)
    }

    private func mouthFrameGeometry(_ mouthFrame: FaceLocalMouthFrame,
                                    material: SCNMaterial) -> SCNGeometry? {
        let center = SCNVector3(mouthFrame.center.x, mouthFrame.center.y, mouthFrame.center.z)
        let right = mouthFrame.center + mouthFrame.xAxis * mouthFrame.width * 0.42
        let down = mouthFrame.center + mouthFrame.downAxis * mouthFrame.width * 0.30
        let normal = mouthFrame.center + mouthFrame.normalAxis * mouthFrame.width * 0.30
        let vertices = [
            center,
            SCNVector3(right.x, right.y, right.z),
            center,
            SCNVector3(down.x, down.y, down.z),
            center,
            SCNVector3(normal.x, normal.y, normal.z)
        ]
        return lineGeometry(
            vertices: vertices,
            indices: [0, 1, 2, 3, 4, 5],
            material: material
        )
    }

    private func lineGeometry(vertices: [SCNVector3],
                              indices: [Int32],
                              material: SCNMaterial) -> SCNGeometry? {
        guard vertices.count >= 2, indices.count >= 2 else {
            return nil
        }

        let source = SCNGeometrySource(vertices: vertices)
        let element = SCNGeometryElement(indices: indices, primitiveType: .line)
        let geometry = SCNGeometry(sources: [source], elements: [element])
        geometry.materials = [material]
        return geometry
    }

    private func logProjectionDiagnostics(contour: LipContour,
                                          mouthFrame: FaceLocalMouthFrame,
                                          renderer: SCNSceneRenderer,
                                          faceNode: SCNNode) {
        guard let pose = contour.pose else {
            return
        }

        let mesh = makeMediaPipeLipMesh(contour: contour)
        guard !mesh.vertices.isEmpty else {
            return
        }

        var deltasByIndex: [Int: CGPoint] = [:]
        deltasByIndex.reserveCapacity(mesh.vertices.count)
        var allProjected = [CGPoint]()
        allProjected.reserveCapacity(mesh.vertices.count)

        for vertex in mesh.vertices {
            let local = screenAlignedScenePoint(
                for: vertex,
                pose: pose,
                contour: contour,
                mouthFrame: mouthFrame,
                renderer: renderer,
                faceNode: faceNode,
                contourAge: 0,
                motionDelta: 0
            )
            let world = faceNode.convertPosition(local, to: nil)
            let projected = renderer.projectPoint(world)
            let screen = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
            guard screen.x.isFinite, screen.y.isFinite, projected.z.isFinite else {
                continue
            }

            deltasByIndex[vertex.index] = CGPoint(
                x: screen.x - vertex.screen.x,
                y: screen.y - vertex.screen.y
            )
            allProjected.append(screen)
        }

        guard !deltasByIndex.isEmpty else {
            return
        }

        let upper = Self.deltaStats(
            for: CanonicalLipGeometry.upperDebugIndices,
            in: deltasByIndex
        )
        let lower = Self.deltaStats(
            for: [146, 91, 181, 84, 17, 314, 405, 321, 375],
            in: deltasByIndex
        )
        let inner = Self.deltaStats(
            for: CanonicalLipGeometry.innerLipIndices,
            in: deltasByIndex
        )
        let bounds = boundsText(contour.bounds)
        let projectedBounds = Self.bounds(for: allProjected)
        let projectedBoundsText = boundsText(projectedBounds)

        LipDebugLog.throttled(
            "lip_projection_diagnostics",
            interval: 0.22,
            "lip_projection deltaPx upper(avgX:\(Self.fmt(upper.avgX)) avgY:\(Self.fmt(upper.avgY)) max:\(Self.fmt(upper.maxDistance)) n:\(upper.count)) lower(avgX:\(Self.fmt(lower.avgX)) avgY:\(Self.fmt(lower.avgY)) max:\(Self.fmt(lower.maxDistance)) n:\(lower.count)) inner(avgX:\(Self.fmt(inner.avgX)) avgY:\(Self.fmt(inner.avgY)) max:\(Self.fmt(inner.maxDistance)) n:\(inner.count)) mouth(width:\(Self.fmt(CGFloat(mouthFrame.width))) ref:\(Self.fmt(CGFloat(mouthFrame.referenceWidth))) open:\(Self.fmt(CGFloat(mouthFrame.openingRatio))) smile:\(Self.fmt(CGFloat(mouthFrame.smileExpansion))) valign:\(Self.fmt(CGFloat(mouthFrame.verticalAlignment))) contour:\(bounds) projected:\(projectedBoundsText)"
        )
    }

    private func logAttachmentDiagnostics(contour: LipContour,
                                          contourAge: CFTimeInterval,
                                          motionDelta: CGFloat) {
        var totalError: CGFloat = 0
        var maxError: CGFloat = 0
        var count: CGFloat = 0
        var upperTotalError: CGFloat = 0
        var upperCount: CGFloat = 0

        for index in CanonicalLipGeometry.attentionLipIndices {
            guard let binding = contour.surfaceBindingsByIndex[index],
                  binding.projectionError.isFinite else {
                continue
            }

            let error = binding.projectionError
            totalError += error
            maxError = max(maxError, error)
            count += 1
            if CanonicalLipGeometry.upperDebugIndices.contains(index) {
                upperTotalError += error
                upperCount += 1
            }
        }

        guard count > 0 else {
            return
        }

        LipDebugLog.throttled(
            "lip_attachment_error",
            interval: 0.75,
            "lip_attachment carrierError avg=\(Self.fmt(totalError / count)) upper=\(Self.fmt(upperCount > 0 ? upperTotalError / upperCount : 0)) max=\(Self.fmt(maxError)) age=\(Self.fmt(CGFloat(contourAge))) motion=\(Self.fmt(motionDelta))"
        )
    }

    private static func deltaStats(for indices: [Int],
                                   in deltasByIndex: [Int: CGPoint]) -> (avgX: CGFloat, avgY: CGFloat, maxDistance: CGFloat, count: Int) {
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        var maxDistance: CGFloat = 0
        var count = 0

        for index in indices {
            guard let delta = deltasByIndex[index] else {
                continue
            }
            sumX += delta.x
            sumY += delta.y
            maxDistance = max(maxDistance, hypot(delta.x, delta.y))
            count += 1
        }

        guard count > 0 else {
            return (0, 0, 0, 0)
        }
        return (sumX / CGFloat(count), sumY / CGFloat(count), maxDistance, count)
    }

    private static func bounds(for points: [CGPoint]) -> CGRect? {
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

    private static func mouthOpeningRatio(_ contour: LipContour) -> CGFloat {
        guard let outerBounds = bounds(for: contour.outer),
              let innerBounds = bounds(for: contour.inner),
              outerBounds.width > 1 else {
            return 0
        }

        return innerBounds.height / outerBounds.width
    }

    private func boundsText(_ bounds: CGRect?) -> String {
        guard let bounds else {
            return "nil"
        }

        return "x:\(Self.fmt(bounds.minX)) y:\(Self.fmt(bounds.minY)) w:\(Self.fmt(bounds.width)) h:\(Self.fmt(bounds.height))"
    }

    private static func fmt(_ value: CGFloat) -> String {
        String(format: "%.2f", Double(value))
    }

    private func makeMediaPipeLipMesh(
        contour: LipContour,
        expandsOuterFeatherBoundary: Bool = false
    ) -> (vertices: [MeshVertex], indices: [Int32]) {
        var vertices: [MeshVertex] = []
        var indices: [Int32] = []
        var vertexIndexByLandmark: [Int: Int32] = [:]

        func meshVertex(for landmarkIndex: Int) -> MeshVertex? {
            guard let point = contour.meshPointsByIndex[landmarkIndex] else {
                return nil
            }
            let textureUV: CGPoint
            let screenPoint: CGPoint
            if expandsOuterFeatherBoundary,
               CanonicalLipGeometry.isOuterLipIndex(landmarkIndex),
               let pose = contour.pose {
                // The generated texture has a curved, inset alpha contour. Give
                // that feather a small transparent geometric margin so SceneKit
                // cannot clip it back to MediaPipe's 20 straight outer edges.
                let cosine = cos(pose.angle)
                let sine = sin(pose.angle)
                let dx = point.screen.x - pose.center.x
                let dy = point.screen.y - pose.center.y
                let localX = dx * cosine + dy * sine
                let localY = -dx * sine + dy * cosine
                // Keep the transparent carrier comfortably outside the SDF
                // feather. This margin must be uniform: a lower-only radial
                // expansion pulls landmark 17 farther than its neighbours and
                // exposes a V-shaped point at the middle of the lower edge.
                let featherWidth = LipOuterMargin.carrier(
                    for: pose.width
                )
                let radialLength = max(
                    sqrt(localX * localX + localY * localY),
                    0.001
                )

                let expandedX =
                    localX + localX / radialLength * featherWidth
                let expandedY =
                    localY + localY / radialLength * featherWidth

                let outerExpansionX: CGFloat = 1.18
                let outerExpansionY: CGFloat = 1.30

                textureUV = CGPoint(
                    x: 0.5 + (point.uv.x - 0.5) * outerExpansionX,
                    y: 0.5 + (point.uv.y - 0.5) * outerExpansionY
                )
                screenPoint = CGPoint(
                    x: pose.center.x + expandedX * cosine - expandedY * sine,
                    y: pose.center.y + expandedX * sine + expandedY * cosine
                )
            } else if CanonicalLipGeometry.isInnerLipIndex(landmarkIndex) {
                textureUV = point.uv
                screenPoint = contour.renderingInnerScreenPoint(point.screen)
            } else {
                textureUV = point.uv
                screenPoint = point.screen
            }
            return MeshVertex(
                index: landmarkIndex,
                screen: screenPoint,
                normalized: point.normalized,
                uv: textureUV
            )
        }

        func appendOrReuse(_ vertex: MeshVertex) -> Int32 {
            if let existing = vertexIndexByLandmark[vertex.index] {
                return existing
            }

            let newIndex = Int32(vertices.count)
            vertices.append(vertex)
            vertexIndexByLandmark[vertex.index] = newIndex
            return newIndex
        }

        @discardableResult
        func appendTriangleSet(_ triangles: [(Int, Int, Int)]) -> Int {
            var appended = 0
            for triangle in triangles {
                guard let first = meshVertex(for: triangle.0),
                      let second = meshVertex(for: triangle.1),
                      let third = meshVertex(for: triangle.2) else {
                    continue
                }

                indices.append(contentsOf: [
                    appendOrReuse(first),
                    appendOrReuse(second),
                    appendOrReuse(third)
                ])
                appended += 1
            }
            return appended
        }

        let appendedTriangleCount = appendTriangleSet(
            CanonicalLipGeometry.lipMeshTriangles
        )
        guard appendedTriangleCount == CanonicalLipGeometry.lipMeshTriangles.count,
              vertices.count == CanonicalLipGeometry.attentionLipIndices.count,
              indices.count == CanonicalLipGeometry.lipMeshTriangles.count * 3 else {
            return ([], [])
        }
        return (vertices, indices)
    }

    private func scenePoint(for vertex: MeshVertex,
                            pose: LipPose,
                            contour: LipContour,
                            mouthFrame: FaceLocalMouthFrame,
                            appliesExpressionCompensation: Bool = true) -> SCNVector3 {
        let cosine = cos(pose.angle)
        let sine = sin(pose.angle)
        let dx = Float(vertex.screen.x - pose.center.x)
        let dy = Float(vertex.screen.y - pose.center.y)
        var normalizedX = (dx * Float(cosine) + dy * Float(sine)) / Float(max(pose.width, 1))
        let rawNormalizedY = (-dx * Float(sine) + dy * Float(cosine)) / Float(max(pose.width, 1))
        let sideSmile = normalizedX < 0 ? mouthFrame.smileLeft : mouthFrame.smileRight
        let smileDriver = max(mouthFrame.smileExpansion, sideSmile)
        if appliesExpressionCompensation {
            let puckerCompression = max(mouthFrame.mouthPucker, mouthFrame.mouthFunnel * 0.72)
            let puckerFactor = 1 - min(max(puckerCompression, 0), 1) * 0.045
            normalizedX *= (1 + smileDriver * CanonicalLipGeometry.smileStretchFactor(for: vertex.index)) * puckerFactor
        }

        var normalizedY = rawNormalizedY
        if appliesExpressionCompensation {
            let mediaPipeOpening = Float(Self.mouthOpeningRatio(contour))
            let arOpeningTarget = mouthFrame.openingRatio >= 0.08 ? mouthFrame.openingRatio : 0
            let openingShortfall = max(arOpeningTarget - mediaPipeOpening, 0)
            normalizedY += openingShortfall * Float(
                CanonicalLipGeometry.openingMotionFactor(for: vertex.index)
            )
            normalizedY += mouthFrame.jawOpen * CanonicalLipGeometry.jawOpenDropFactor(for: vertex.index) * 0.020
            normalizedY += mouthFrame.lowerLipDrop * CanonicalLipGeometry.lowerLipDropFactor(for: vertex.index) * 0.014
        }
        let depthDelta = max(min(vertex.normalized.z - contour.depthCenter, 0.08), -0.08)
        let canonicalDepthDelta = max(
            min(contour.faceGeometryPose?.relativeCanonicalDepth(for: vertex.index) ?? 0, 0.08),
            -0.08
        )
        let geometryScale = contour.faceGeometryPose.map { min(max($0.scale / 8, 0.45), 1.65) } ?? 1
        // The fallback stays close to the AR mouth frame. Keep only a small,
        // bounded relief so it neither floats in front of the lips nor falls
        // behind the face depth occluder.
        let surfaceOffset = min(max(mouthFrame.width * 0.012, 0.000_45), 0.000_70)
        let requestedDepthOffset = surfaceOffset -
            depthDelta * mouthFrame.width * 0.07 * geometryScale -
            canonicalDepthDelta * mouthFrame.width * 0.03
        let depthOffset = min(max(requestedDepthOffset, 0.000_35), 0.000_80)
        let smileLift: Float
        if appliesExpressionCompensation {
            let openUpperCoverage = min(max((mouthFrame.jawOpen - 0.055) * 2.6, 0), 0.78)
            let raisedUpper = max(mouthFrame.upperLipRaise - 0.035, 0) * 0.9
            let upperDriver = max(raisedUpper, smileDriver * 1.05, openUpperCoverage)
            smileLift = CanonicalLipGeometry.upperLipSmileLiftFactor(for: vertex.index) *
                min(max(upperDriver, 0), 1) *
                mouthFrame.width * 0.027
        } else {
            smileLift = 0
        }

        let local = mouthFrame.center +
            mouthFrame.xAxis * (normalizedX * mouthFrame.width) +
            mouthFrame.downAxis * (normalizedY * mouthFrame.width - smileLift) +
            mouthFrame.normalAxis * depthOffset

        return SCNVector3(local.x, local.y, local.z)
    }

    private func screenAlignedScenePoint(for vertex: MeshVertex,
                                         pose: LipPose,
                                         contour: LipContour,
                                         mouthFrame: FaceLocalMouthFrame,
                                         renderer: SCNSceneRenderer,
                                         faceNode: SCNNode,
                                         contourAge: CFTimeInterval,
                                         motionDelta: CGFloat) -> SCNVector3 {
        // Keep the ARKit lip surface as the primary attachment. MediaPipe can
        // correct projection error while fresh, but it should not pull the mesh
        // away from the face during fast motion or delayed callbacks.
        let depthReference = scenePoint(
            for: vertex,
            pose: pose,
            contour: contour,
            mouthFrame: mouthFrame
        )
        return projectionAlignedScenePoint(
            depthReference: depthReference,
            targetScreenPoint: vertex.screen,
            mouthFrame: mouthFrame,
            renderer: renderer,
            faceNode: faceNode,
            contourAge: contourAge,
            motionDelta: motionDelta,
            maximumCorrectionFraction: 0.22
        )
    }

    private func projectionAlignedScenePoint(depthReference: SCNVector3,
                                             targetScreenPoint: CGPoint,
                                             mouthFrame: FaceLocalMouthFrame,
                                             renderer: SCNSceneRenderer,
                                             faceNode: SCNNode,
                                             contourAge: CFTimeInterval,
                                             motionDelta: CGFloat,
                                             maximumCorrectionFraction: Float) -> SCNVector3 {
        let correctionWeight = projectionCorrectionWeight(
            contourAge: contourAge,
            motionDelta: motionDelta
        )
        guard correctionWeight > 0 else {
            return depthReference
        }

        let worldDepthReference = faceNode.convertPosition(depthReference, to: nil)
        let projectedDepthReference = renderer.projectPoint(worldDepthReference)
        guard projectedDepthReference.z.isFinite,
              projectedDepthReference.z >= 0,
              projectedDepthReference.z <= 1 else {
            return depthReference
        }

        let worldTarget = renderer.unprojectPoint(
            SCNVector3(
                Float(targetScreenPoint.x),
                Float(targetScreenPoint.y),
                projectedDepthReference.z
            )
        )
        guard worldTarget.x.isFinite,
              worldTarget.y.isFinite,
              worldTarget.z.isFinite else {
            return depthReference
        }

        let localTarget = faceNode.convertPosition(worldTarget, from: nil)
        let reference = SIMD3<Float>(depthReference.x, depthReference.y, depthReference.z)
        let target = SIMD3<Float>(localTarget.x, localTarget.y, localTarget.z)
        let delta = target - reference
        // Unprojecting at a camera-space depth introduces a face-normal
        // component at yaw. Applying it would push one side behind the depth
        // occluder and make the other side float. Keep the screen correction in
        // the current mouth tangent plane so the surface lift remains bounded.
        let tangentDelta = mouthFrame.xAxis * simd_dot(delta, mouthFrame.xAxis) +
            mouthFrame.downAxis * simd_dot(delta, mouthFrame.downAxis)
        let tangentLength = simd_length(tangentDelta)
        guard tangentLength.isFinite, tangentLength > 0 else {
            return depthReference
        }

        let maxCorrection = mouthFrame.width * maximumCorrectionFraction
        let clampedDelta = tangentLength > maxCorrection ?
            tangentDelta / tangentLength * maxCorrection :
            tangentDelta
        let corrected = reference + clampedDelta * Float(correctionWeight)
        return SCNVector3(corrected.x, corrected.y, corrected.z)
    }

    private func projectionCorrectionWeight(contourAge: CFTimeInterval,
                                            motionDelta: CGFloat) -> CGFloat {
        guard contourAge.isFinite,
              contourAge <= 0.62 else {
            return 0
        }

        let ageFade = 1 - Self.smoothStep(edge0: 0.045, edge1: 0.30, value: CGFloat(contourAge))
        let finiteMotion = motionDelta.isFinite ? motionDelta : 1
        let motionFade = 1 - Self.smoothStep(edge0: 0.035, edge1: 0.22, value: finiteMotion)
        let freshSignal = max(0, min(ageFade * motionFade, 1))
        return max(0.04, min(1, 0.04 + freshSignal * 0.96))
    }

    private static func smoothStep(edge0: CGFloat, edge1: CGFloat, value: CGFloat) -> CGFloat {
        let t = max(0, min((value - edge0) / max(edge1 - edge0, 0.0001), 1))
        return t * t * (3 - 2 * t)
    }

    private static let lipPointCount = 20
}

private final class BlushRenderer {
    private let rootNode = SCNNode()
    private let leftNode = SCNNode()
    private let rightNode = SCNNode()
    private let leftMaterial = SCNMaterial()
    private let rightMaterial = SCNMaterial()
    private var lastColor: UIColor?
    private var lastOpacity: Double = -1
    private var cheekRegionVertexCount = 0
    private var leftCheekIndices: [Int] = []
    private var rightCheekIndices: [Int] = []

    init() {
        rootNode.name = "blush.root"
        rootNode.renderingOrder = 80
        leftNode.name = "blush.left"
        rightNode.name = "blush.right"
        rootNode.addChildNode(leftNode)
        rootNode.addChildNode(rightNode)

        configureMaterial(leftMaterial)
        configureMaterial(rightMaterial)
        leftNode.geometry = Self.makeBlushGeometry(material: leftMaterial)
        rightNode.geometry = Self.makeBlushGeometry(material: rightMaterial)
    }

    func attach(to faceNode: SCNNode) {
        if rootNode.parent !== faceNode {
            rootNode.removeFromParentNode()
            faceNode.addChildNode(rootNode)
        }
    }

    func updateStyle(color: UIColor, opacity: Double) {
        let clampedOpacity = max(0, min(opacity, 0.85))
        let shouldUpdateTexture = lastColor.map { !Self.sameColor($0, color) } ?? true
        if shouldUpdateTexture {
            let image = Self.makeBlushTexture(color: color)
            leftMaterial.diffuse.contents = image
            rightMaterial.diffuse.contents = image
            lastColor = color
        }

        guard abs(lastOpacity - clampedOpacity) > 0.003 else {
            return
        }
        lastOpacity = clampedOpacity
        leftMaterial.transparency = CGFloat(clampedOpacity)
        rightMaterial.transparency = CGFloat(clampedOpacity)
        rootNode.isHidden = clampedOpacity <= 0.01
    }

    func render(faceAnchor: ARFaceAnchor) {
        guard lastOpacity > 0.01 else {
            rootNode.isHidden = true
            return
        }

        let vertices = faceAnchor.geometry.vertices
        guard let bounds = Self.faceBounds(vertices: vertices) else {
            clear()
            return
        }

        guard let leftGeometry = Self.makeCheekMesh(
            side: -1,
            faceGeometry: faceAnchor.geometry,
            bounds: bounds,
            material: leftMaterial
        ),
              let rightGeometry = Self.makeCheekMesh(
                side: 1,
                faceGeometry: faceAnchor.geometry,
                bounds: bounds,
                material: rightMaterial
              ) else {
            clear()
            return
        }

        rootNode.isHidden = false
        leftNode.simdTransform = matrix_identity_float4x4
        rightNode.simdTransform = matrix_identity_float4x4
        leftNode.geometry = leftGeometry
        rightNode.geometry = rightGeometry
    }

    func clear() {
        rootNode.isHidden = true
    }

    private func configureMaterial(_ material: SCNMaterial) {
        material.lightingModel = .constant
        material.blendMode = .alpha
        material.isDoubleSided = true
        material.writesToDepthBuffer = false
        material.readsFromDepthBuffer = false
        material.diffuse.magnificationFilter = .linear
        material.diffuse.minificationFilter = .linear
        material.transparency = CGFloat(max(lastOpacity, 0))
    }

    private struct CheekPlacement {
        let center: SIMD3<Float>
        let xAxis: SIMD3<Float>
        let yAxis: SIMD3<Float>
        let normal: SIMD3<Float>
        let width: Float
        let height: Float
    }

    private struct FaceBounds {
        let min: SIMD3<Float>
        let max: SIMD3<Float>

        var center: SIMD3<Float> {
            (min + max) * 0.5
        }

        var width: Float {
            max.x - min.x
        }

        var height: Float {
            max.y - min.y
        }

        var depth: Float {
            max.z - min.z
        }
    }

    private func updateCheekRegionsIfNeeded(vertices: [SIMD3<Float>], bounds: FaceBounds) {
        guard cheekRegionVertexCount != vertices.count ||
                leftCheekIndices.isEmpty ||
                rightCheekIndices.isEmpty else {
            return
        }

        cheekRegionVertexCount = vertices.count
        leftCheekIndices = Self.cheekRegionIndices(side: -1, vertices: vertices, bounds: bounds)
        rightCheekIndices = Self.cheekRegionIndices(side: 1, vertices: vertices, bounds: bounds)
    }

    private static func faceBounds(vertices: [SIMD3<Float>]) -> FaceBounds? {
        guard var minPoint = vertices.first,
              var maxPoint = vertices.first else {
            return nil
        }

        for point in vertices.dropFirst() {
            minPoint = simd_min(minPoint, point)
            maxPoint = simd_max(maxPoint, point)
        }

        guard (maxPoint.x - minPoint.x) > 0.02,
              (maxPoint.y - minPoint.y) > 0.02,
              minPoint.x.isFinite,
              minPoint.y.isFinite,
              minPoint.z.isFinite,
              maxPoint.x.isFinite,
              maxPoint.y.isFinite,
              maxPoint.z.isFinite else {
            return nil
        }

        return FaceBounds(min: minPoint, max: maxPoint)
    }

    private static func cheekRegionIndices(side: Float,
                                           vertices: [SIMD3<Float>],
                                           bounds: FaceBounds) -> [Int] {
        var weighted: [(score: Float, index: Int)] = []
        weighted.reserveCapacity(96)

        for (index, point) in vertices.enumerated() {
            guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
                continue
            }

            let cheekX = normalized.x * side
            let cheekY = normalized.y
            let ovalScore = cheekOvalScore(cheekX: cheekX, cheekY: cheekY)
            guard ovalScore > 0 else {
                continue
            }

            weighted.append((ovalScore, index))
        }

        return weighted
            .sorted { $0.score > $1.score }
            .prefix(96)
            .map(\.index)
    }

    private static func cheekPlacement(side: Float,
                                       vertices: [SIMD3<Float>],
                                       indices: [Int],
                                       bounds: FaceBounds) -> CheekPlacement? {
        guard !indices.isEmpty else {
            return nil
        }

        var center = SIMD3<Float>(repeating: 0)
        var weightSum: Float = 0
        for index in indices where index < vertices.count {
            let point = vertices[index]
            guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
                continue
            }

            let weight = max(cheekOvalScore(cheekX: normalized.x * side, cheekY: normalized.y), 0.05)
            center += point * weight
            weightSum += weight
        }

        guard weightSum > 0 else {
            return nil
        }
        center /= max(weightSum, 0.0001)

        let outwardZ = max(bounds.depth * 0.32, 0.018)
        let normal = simd_normalize(SIMD3<Float>(
            side * bounds.width * 0.17,
            bounds.height * 0.02,
            outwardZ
        ))

        var xAxis = SIMD3<Float>(1, 0, 0) - normal * simd_dot(SIMD3<Float>(1, 0, 0), normal)
        if simd_length(xAxis) < 0.001 {
            xAxis = SIMD3<Float>(side, 0, 0)
        } else {
            xAxis = simd_normalize(xAxis)
        }

        var yAxis = SIMD3<Float>(0, 1, 0) - normal * simd_dot(SIMD3<Float>(0, 1, 0), normal)
        yAxis -= xAxis * simd_dot(yAxis, xAxis)
        if simd_length(yAxis) < 0.001 {
            yAxis = simd_normalize(simd_cross(normal, xAxis))
        } else {
            yAxis = simd_normalize(yAxis)
        }

        return CheekPlacement(
            center: center + normal * max(bounds.width * 0.014, 0.0012),
            xAxis: xAxis,
            yAxis: yAxis,
            normal: normal,
            width: max(bounds.width * 0.58, 0.082),
            height: max(bounds.width * 0.36, 0.054)
        )
    }

    private static func normalizedFacePoint(_ point: SIMD3<Float>, bounds: FaceBounds) -> SIMD3<Float>? {
        let width = max(bounds.width, 0.0001)
        let height = max(bounds.height, 0.0001)
        let depth = max(bounds.depth, 0.0001)
        let x = (point.x - bounds.center.x) / (width * 0.5)
        let y = (point.y - bounds.min.y) / height
        let z = (point.z - bounds.min.z) / depth
        guard x.isFinite, y.isFinite, z.isFinite else {
            return nil
        }
        return SIMD3<Float>(x, y, z)
    }

    private static func cheekOvalScore(cheekX: Float, cheekY: Float) -> Float {
        guard cheekX > 0.32,
              cheekX < 0.92,
              cheekY > 0.30,
              cheekY < 0.59 else {
            return 0
        }

        let dx = (cheekX - 0.60) / 0.30
        let dy = (cheekY - 0.45) / 0.15
        let distance = dx * dx + dy * dy
        guard distance < 1 else {
            return 0
        }
        let core = 1 - distance
        return core * core
    }

    private static func makeCheekMesh(side: Float,
                                      faceGeometry: ARFaceGeometry,
                                      bounds: FaceBounds,
                                      material: SCNMaterial) -> SCNGeometry? {
        let vertices = faceGeometry.vertices
        let triangleIndices = faceGeometry.triangleIndices
        guard !vertices.isEmpty,
              triangleIndices.count >= 3 else {
            return nil
        }

        var scnVertices: [SCNVector3] = []
        var textureCoordinates: [CGPoint] = []
        scnVertices.reserveCapacity(vertices.count)
        textureCoordinates.reserveCapacity(vertices.count)

        for point in vertices {
            scnVertices.append(SCNVector3(point.x, point.y, point.z))
            textureCoordinates.append(cheekTextureCoordinate(point, side: side, bounds: bounds))
        }

        var indices: [Int32] = []
        indices.reserveCapacity(triangleIndices.count)
        for triangleStart in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let a = Int(triangleIndices[triangleStart])
            let b = Int(triangleIndices[triangleStart + 1])
            let c = Int(triangleIndices[triangleStart + 2])
            guard a >= 0, b >= 0, c >= 0,
                  a < vertices.count,
                  b < vertices.count,
                  c < vertices.count else {
                continue
            }

            let scoreA = cheekScore(vertices[a], side: side, bounds: bounds)
            let scoreB = cheekScore(vertices[b], side: side, bounds: bounds)
            let scoreC = cheekScore(vertices[c], side: side, bounds: bounds)
            guard min(scoreA, scoreB, scoreC) > 0.018 else {
                continue
            }
            indices.append(contentsOf: [Int32(a), Int32(b), Int32(c)])
        }

        guard indices.count >= 3 else {
            return nil
        }

        let vertexSource = SCNGeometrySource(vertices: scnVertices)
        let textureSource = SCNGeometrySource(textureCoordinates: textureCoordinates)
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        let geometry = SCNGeometry(sources: [vertexSource, textureSource], elements: [element])
        geometry.materials = [material]
        return geometry
    }

    private static func cheekScore(_ point: SIMD3<Float>, side: Float, bounds: FaceBounds) -> Float {
        guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
            return 0
        }
        return cheekOvalScore(cheekX: normalized.x * side, cheekY: normalized.y)
    }

    private static func cheekTextureCoordinate(_ point: SIMD3<Float>,
                                               side: Float,
                                               bounds: FaceBounds) -> CGPoint {
        guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
            return CGPoint(x: 0.5, y: 0.5)
        }

        let cheekX = normalized.x * side
        let cheekY = normalized.y
        let u = 0.5 + CGFloat((cheekX - 0.56) / 0.39) * 0.5
        let v = 0.5 - CGFloat((cheekY - 0.44) / 0.20) * 0.5
        return CGPoint(x: max(0, min(1, u)), y: max(0, min(1, v)))
    }

    private static func makeBlushGeometry(material: SCNMaterial) -> SCNGeometry {
        let columns = 6
        let rows = 5
        var vertices: [SCNVector3] = []
        var textureCoordinates: [CGPoint] = []
        var indices: [Int32] = []

        for row in 0..<rows {
            let v = Float(row) / Float(rows - 1)
            let y = v - 0.5
            for column in 0..<columns {
                let u = Float(column) / Float(columns - 1)
                let x = u - 0.5
                let radial = min(1, (x * x) / 0.25 + (y * y) / 0.18)
                let z = (1 - radial) * 0.0014
                vertices.append(SCNVector3(x, y, z))
                textureCoordinates.append(CGPoint(x: CGFloat(u), y: CGFloat(1 - v)))
            }
        }

        for row in 0..<(rows - 1) {
            for column in 0..<(columns - 1) {
                let a = Int32(row * columns + column)
                let b = Int32(row * columns + column + 1)
                let c = Int32((row + 1) * columns + column)
                let d = Int32((row + 1) * columns + column + 1)
                indices.append(contentsOf: [a, c, b, b, c, d])
            }
        }

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let textureSource = SCNGeometrySource(textureCoordinates: textureCoordinates)
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        let geometry = SCNGeometry(sources: [vertexSource, textureSource], elements: [element])
        geometry.materials = [material]
        return geometry
    }

    private static func sameColor(_ first: UIColor, _ second: UIColor) -> Bool {
        var firstRed: CGFloat = 0
        var firstGreen: CGFloat = 0
        var firstBlue: CGFloat = 0
        var firstAlpha: CGFloat = 0
        var secondRed: CGFloat = 0
        var secondGreen: CGFloat = 0
        var secondBlue: CGFloat = 0
        var secondAlpha: CGFloat = 0
        first.getRed(&firstRed, green: &firstGreen, blue: &firstBlue, alpha: &firstAlpha)
        second.getRed(&secondRed, green: &secondGreen, blue: &secondBlue, alpha: &secondAlpha)
        return abs(firstRed - secondRed) < 0.002 &&
            abs(firstGreen - secondGreen) < 0.002 &&
            abs(firstBlue - secondBlue) < 0.002 &&
            abs(firstAlpha - secondAlpha) < 0.002
    }

    private static func makeBlushTexture(color: UIColor) -> UIImage {
        let size = CGSize(width: 256, height: 180)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            let rect = CGRect(origin: .zero, size: size)
            guard let cgContext = UIGraphicsGetCurrentContext() else {
                return
            }

            var red: CGFloat = 0
            var green: CGFloat = 0
            var blue: CGFloat = 0
            var alpha: CGFloat = 0
            color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
            let center = CGPoint(x: rect.midX, y: rect.midY)
            let colors = [
                UIColor(red: red, green: green, blue: blue, alpha: 0.58).cgColor,
                UIColor(red: red, green: green, blue: blue, alpha: 0.34).cgColor,
                UIColor(red: red, green: green, blue: blue, alpha: 0.13).cgColor,
                UIColor(red: red, green: green, blue: blue, alpha: 0.0).cgColor
            ] as CFArray
            let locations: [CGFloat] = [0.0, 0.48, 0.82, 1.0]
            guard let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: colors,
                locations: locations
            ) else {
                return
            }

            cgContext.saveGState()
            cgContext.translateBy(x: center.x, y: center.y)
            cgContext.scaleBy(x: 1.0, y: 0.62)
            cgContext.translateBy(x: -center.x, y: -center.y)
            cgContext.drawRadialGradient(
                gradient,
                startCenter: center,
                startRadius: 0,
                endCenter: center,
                endRadius: size.width * 0.56,
                options: [.drawsAfterEndLocation]
            )
            cgContext.restoreGState()

            UIColor.white.withAlphaComponent(0.06).setFill()
            context.cgContext.fillEllipse(in: CGRect(x: 82, y: 54, width: 74, height: 38))
        }
    }
}

private struct LipTexture {
    let image: UIImage
}

private struct RGBColor {
    let red: Float
    let green: Float
    let blue: Float
}

private final class MetalLipColorCompositor {
    private struct Vertex {
        let canonicalUV: SIMD2<Float>
        let sourceUV: SIMD2<Float>
    }

    private struct Uniforms {
        var lipstickColor: SIMD4<Float>
        var sourceTexelSize: SIMD2<Float>
        var toneStrength: Float
        var pigmentStrength: Float
        var saturationStrength: Float
        var lightingFactor: Float
        var detailStrength: Float
        var coreCoverage: Float
        var finish: Float
    }

    private struct FeatherUniforms {
        var textureSize: SIMD2<Float>
        var outerPointCount: UInt32
        var innerPointCount: UInt32
        var outerFeatherPixels: Float
        var outerCoreInsetPixels: Float
        var apertureFeatherPixels: Float
        var apertureInsetPixels: Float
    }

    private struct RenderTargets {
        let width: Int
        let height: Int
        let color: MTLTexture
        let output: MTLTexture
    }

    private final class Context {
        let device: MTLDevice
        let commandQueue: MTLCommandQueue
        let textureCache: CVMetalTextureCache
        let renderPipeline: MTLRenderPipelineState
        let compositePipeline: MTLComputePipelineState
        let indexBuffer: MTLBuffer
        let indexCount: Int
        var renderTargets: RenderTargets?

        init(device: MTLDevice,
             commandQueue: MTLCommandQueue,
             textureCache: CVMetalTextureCache,
             renderPipeline: MTLRenderPipelineState,
             compositePipeline: MTLComputePipelineState,
             indexBuffer: MTLBuffer,
             indexCount: Int) {
            self.device = device
            self.commandQueue = commandQueue
            self.textureCache = textureCache
            self.renderPipeline = renderPipeline
            self.compositePipeline = compositePipeline
            self.indexBuffer = indexBuffer
            self.indexCount = indexCount
        }
    }

    private static let shaderSource = #"""
    #include <metal_stdlib>
    using namespace metal;

    struct LipVertex {
        float2 canonicalUV;
        float2 sourceUV;
    };

    struct LipUniforms {
        float4 lipstickColor;
        float2 sourceTexelSize;
        float toneStrength;
        float pigmentStrength;
        float saturationStrength;
        float lightingFactor;
        float detailStrength;
        float coreCoverage;
        float finish;
    };

    struct LipRaster {
        float4 position [[position]];
        float2 sourceUV;
        float2 canonicalUV;
    };

    struct LipFeatherUniforms {
        float2 textureSize;
        uint outerPointCount;
        uint innerPointCount;
        float outerFeatherPixels;
        float outerCoreInsetPixels;
        float apertureFeatherPixels;
        float apertureInsetPixels;
    };

    vertex LipRaster lip_vertex(const device LipVertex *vertices [[buffer(0)]],
                                uint vertexID [[vertex_id]]) {
        LipVertex inputVertex = vertices[vertexID];
        LipRaster output;
        // Canonical v=1 is the top row of the generated CGImage.
        output.position = float4(
            inputVertex.canonicalUV.x * 2.0 - 1.0,
            inputVertex.canonicalUV.y * 2.0 - 1.0,
            0.0,
            1.0
        );
        output.sourceUV = inputVertex.sourceUV;
        output.canonicalUV = inputVertex.canonicalUV;
        return output;
    }

    float lip_luminance(float3 color) {
        return clamp(dot(color, float3(0.299, 0.587, 0.114)), 0.0, 1.0);
    }

    float3 lip_color_with_luminance(float3 color, float targetLuminance) {
        float sourceLuminance = max(lip_luminance(color), 0.0001);
        float3 adjusted = color * (targetLuminance / sourceLuminance);
        float maximum = max(adjusted.r, max(adjusted.g, adjusted.b));
        if (maximum > 1.0) {
            adjusted /= maximum;
        }
        return saturate(adjusted);
    }

    fragment float4 lip_fragment(
        LipRaster input [[stage_in]],
        constant LipUniforms &uniforms [[buffer(0)]],
        texture2d<float> cameraTexture [[texture(0)]]) {
        constexpr sampler cameraSampler(
            coord::normalized,
            address::clamp_to_edge,
            filter::linear
        );

        float2 radius = uniforms.sourceTexelSize * 2.0;
        float3 base = cameraTexture.sample(cameraSampler, input.sourceUV).rgb;
        float3 blurred = base * 4.0;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(radius.x, 0.0)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV - float2(radius.x, 0.0)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(0.0, radius.y)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV - float2(0.0, radius.y)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + radius).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV - radius).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(radius.x, -radius.y)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(-radius.x, radius.y)).rgb;
        blurred /= 12.0;

        // Use the camera's real reflection pattern as the satin highlight.
        // A wider cross-shaped neighbourhood separates an existing bright
        // spot from the surrounding lip tone without inventing its position.
        float2 highlightRadius = uniforms.sourceTexelSize * float2(12.0, 8.0);
        float3 highlightSurround =
            cameraTexture.sample(
                cameraSampler,
                input.sourceUV + float2(highlightRadius.x, 0.0)
            ).rgb;
        highlightSurround += cameraTexture.sample(
            cameraSampler,
            input.sourceUV - float2(highlightRadius.x, 0.0)
        ).rgb;
        highlightSurround += cameraTexture.sample(
            cameraSampler,
            input.sourceUV + float2(0.0, highlightRadius.y)
        ).rgb;
        highlightSurround += cameraTexture.sample(
            cameraSampler,
            input.sourceUV - float2(0.0, highlightRadius.y)
        ).rgb;
        highlightSurround *= 0.25;

        float baseLuminance = lip_luminance(base);
        float blurredLuminance = max(lip_luminance(blurred), 0.055);
        float pigmentLuminance = lip_luminance(uniforms.lipstickColor.rgb);
        float localLighting = smoothstep(0.10, 0.55, blurredLuminance);
        float combinedLighting = clamp(
            mix(uniforms.lightingFactor, localLighting, 0.25),
            0.45,
            1.0
        );
        float exposureScale = pow(combinedLighting, 0.55);
        float scenePigmentLuminance = pigmentLuminance * exposureScale;
        float targetLuminance = mix(
            baseLuminance,
            scenePigmentLuminance,
            uniforms.toneStrength
        );
        float3 tonedPigment = lip_color_with_luminance(
            uniforms.lipstickColor.rgb,
            targetLuminance
        );
        float tonedLuminance = lip_luminance(tonedPigment);
        float3 saturatedPigment = saturate(
            mix(
                float3(tonedLuminance),
                tonedPigment,
                uniforms.saturationStrength
            )
        );
        float logDetail = log2(max(baseLuminance, 0.04)) -
            log2(max(blurredLuminance, 0.04));
        float brightScene = smoothstep(0.78, 1.0, combinedLighting);
        float detailExponent;
        if (uniforms.finish < 0.5) {
            // Matte pigment is diffuse, but not flat. Preserve camera-derived
            // creases more strongly than highlights so the lip volume remains
            // visible without introducing a wet/specular appearance.
            float shadowDetail = min(logDetail, 0.0);
            float highlightDetail = max(logDetail, 0.0);
            float matteShadowStrength = mix(1.15, 0.72, brightScene);
            float matteHighlightStrength = mix(0.65, 0.35, brightScene);
            detailExponent =
                clamp(shadowDetail, -0.14, 0.0) *
                    uniforms.detailStrength * matteShadowStrength +
                clamp(highlightDetail, 0.0, 0.09) *
                    uniforms.detailStrength * matteHighlightStrength;
        } else {
            float adaptiveDetailStrength = uniforms.detailStrength *
                mix(1.0, 0.45, brightScene);
            detailExponent = clamp(logDetail, -0.12, 0.12) *
                adaptiveDetailStrength;
        }
        float detail = exp2(detailExponent);
        float shadow = 1.0 -
            (1.0 - smoothstep(0.08, 0.26, blurredLuminance)) * 0.025;
        float3 pigment = mix(
            base,
            saturatedPigment,
            uniforms.pigmentStrength
        );
        pigment = saturate(pigment * detail * shadow);

        if (uniforms.finish > 1.5) {
            // User controls for the camera-derived satin reflection.
            constexpr float naturalHighlightStrength = 0.9;
            constexpr float naturalHighlightMaximum = 0.20;
            float surroundLuminance = max(
                lip_luminance(highlightSurround),
                0.04
            );
            float highlightLogContrast = log2(
                max(baseLuminance, 0.04) / surroundLuminance
            );
            float relativeHighlight = smoothstep(
                0.035,
                0.22,
                highlightLogContrast
            );
            float highlightBrightness = smoothstep(
                0.18,
                0.68,
                baseLuminance
            );
            float highlightDifference = max(
                baseLuminance - surroundLuminance,
                0.0
            );
            float naturalHighlight = relativeHighlight * highlightBrightness;
            float highlightAmount = min(
                naturalHighlight * (0.07 + highlightDifference * 0.90) *
                    naturalHighlightStrength,
                naturalHighlightMaximum
            );
            pigment = mix(pigment, float3(1.0), highlightAmount);
        }

        float cornerPosition = clamp(abs(input.canonicalUV.x - 0.5) * 2.0, 0.0, 1.0);
        float cornerFade = 1.0 - smoothstep(0.72, 1.0, cornerPosition) * 0.30;
        float nominalAlpha = cornerFade * uniforms.coreCoverage;
        // Store the source-over corrective colour. After SceneKit applies the
        // user opacity, a dark chosen shade remains dark instead of being
        // lifted back toward the bare-lip luminance.
        float3 compositingColor = saturate(
            (pigment - base * (1.0 - nominalAlpha)) / max(nominalAlpha, 0.001)
        );
        return float4(compositingColor, nominalAlpha);
    }

    float lip_distance_to_segment(float2 point, float2 first, float2 second) {
        float2 segment = second - first;
        float denominator = max(dot(segment, segment), 0.0001);
        float position = clamp(dot(point - first, segment) / denominator, 0.0, 1.0);
        return distance(point, first + segment * position);
    }

    kernel void lip_composite(
        texture2d<float, access::read> colorTexture [[texture(0)]],
        texture2d<float, access::write> outputTexture [[texture(1)]],
        const device float2 *outerPoints [[buffer(0)]],
        constant LipFeatherUniforms &uniforms [[buffer(1)]],
        const device float2 *innerPoints [[buffer(2)]],
        uint2 position [[thread_position_in_grid]]) {
        if (position.x >= outputTexture.get_width() ||
            position.y >= outputTexture.get_height()) {
            return;
        }
        float4 color = colorTexture.read(position);
        if (color.a <= 0.0 ||
            uniforms.outerPointCount < 3 ||
            uniforms.innerPointCount < 3) {
            outputTexture.write(float4(0.0), position);
            return;
        }

        float2 pixelPoint = float2(position) + 0.5;
        float minimumDistance = 100000.0;
        bool isInside = false;
        uint previous = uniforms.outerPointCount - 1;
        for (uint index = 0; index < uniforms.outerPointCount; ++index) {
            float2 first = outerPoints[previous];
            float2 second = outerPoints[index];
            minimumDistance = min(
                minimumDistance,
                lip_distance_to_segment(pixelPoint, first, second)
            );

            bool crosses = (first.y > pixelPoint.y) != (second.y > pixelPoint.y);
            if (crosses) {
                float denominator = second.y - first.y;
                float intersectionX = first.x +
                    (pixelPoint.y - first.y) *
                    (second.x - first.x) /
                    (abs(denominator) < 0.0001 ? 0.0001 : denominator);
                if (pixelPoint.x < intersectionX) {
                    isInside = !isInside;
                }
            }
            previous = index;
        }

        float signedDistance = isInside ? -minimumDistance : minimumDistance;
        float outerFeather = 1.0 - smoothstep(
            -uniforms.outerCoreInsetPixels,
            uniforms.outerFeatherPixels,
            signedDistance
        );
        outerFeather = pow(saturate(outerFeather), 1.12);

        float innerMinimumDistance = 100000.0;
        bool isInsideAperture = false;
        uint innerPrevious = uniforms.innerPointCount - 1;
        for (uint index = 0; index < uniforms.innerPointCount; ++index) {
            float2 first = innerPoints[innerPrevious];
            float2 second = innerPoints[index];
            innerMinimumDistance = min(
                innerMinimumDistance,
                lip_distance_to_segment(pixelPoint, first, second)
            );

            bool crosses = (first.y > pixelPoint.y) != (second.y > pixelPoint.y);
            if (crosses) {
                float denominator = second.y - first.y;
                float intersectionX = first.x +
                    (pixelPoint.y - first.y) *
                    (second.x - first.x) /
                    (abs(denominator) < 0.0001 ? 0.0001 : denominator);
                if (pixelPoint.x < intersectionX) {
                    isInsideAperture = !isInsideAperture;
                }
            }
            innerPrevious = index;
        }

        // Pixels in the painted lip ring are outside the aperture and have a
        // positive distance. Retain partial coverage exactly on the mouth seam
        // so a closed mouth does not expose a skin-coloured hairline, then
        // reach full pigment smoothly a few pixels into the lip.
        float apertureSignedDistance = isInsideAperture ?
            -innerMinimumDistance : innerMinimumDistance;
        float apertureFeather = smoothstep(
            -uniforms.apertureInsetPixels,
            uniforms.apertureFeatherPixels,
            apertureSignedDistance
        );
        float alpha = color.a * outerFeather * apertureFeather;
        outputTexture.write(float4(color.rgb * alpha, alpha), position);
    }
    """#

    private let context: Context?
    private let availabilityLock = NSLock()
    private var consecutiveRuntimeFailures = 0
    private var isRuntimeDisabled = false

    var isAvailable: Bool {
        availabilityLock.lock()
        let available = context != nil && !isRuntimeDisabled
        availabilityLock.unlock()
        return available
    }

    init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            context = nil
            LipDebugLog.throttled(
                "lip_metal_unavailable",
                interval: 10,
                "lip_metal unavailable reason=no_device_or_queue"
            )
            return
        }

        var textureCache: CVMetalTextureCache?
        let cacheStatus = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            nil,
            device,
            nil,
            &textureCache
        )
        guard cacheStatus == kCVReturnSuccess,
              let textureCache else {
            context = nil
            LipDebugLog.throttled(
                "lip_metal_unavailable",
                interval: 10,
                "lip_metal unavailable reason=texture_cache status=\(cacheStatus)"
            )
            return
        }

        do {
            let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
            guard let vertexFunction = library.makeFunction(name: "lip_vertex"),
                  let fragmentFunction = library.makeFunction(name: "lip_fragment"),
                  let compositeFunction = library.makeFunction(name: "lip_composite") else {
                context = nil
                LipDebugLog.throttled(
                    "lip_metal_unavailable",
                    interval: 10,
                    "lip_metal unavailable reason=shader_function"
                )
                return
            }

            let renderDescriptor = MTLRenderPipelineDescriptor()
            renderDescriptor.label = "VirtualMakeup.LipColor"
            renderDescriptor.vertexFunction = vertexFunction
            renderDescriptor.fragmentFunction = fragmentFunction
            renderDescriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
            let renderPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)

            let compositePipeline = try device.makeComputePipelineState(function: compositeFunction)
            let indices = Self.makeIndices()
            let builtIndexBuffer: MTLBuffer? = indices.withUnsafeBytes { bytes -> MTLBuffer? in
                guard let baseAddress = bytes.baseAddress else {
                    return nil
                }
                return device.makeBuffer(
                    bytes: baseAddress,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
            guard !indices.isEmpty,
                  let indexBuffer = builtIndexBuffer else {
                context = nil
                LipDebugLog.throttled(
                    "lip_metal_unavailable",
                    interval: 10,
                    "lip_metal unavailable reason=index_buffer"
                )
                return
            }

            context = Context(
                device: device,
                commandQueue: commandQueue,
                textureCache: textureCache,
                renderPipeline: renderPipeline,
                compositePipeline: compositePipeline,
                indexBuffer: indexBuffer,
                indexCount: indices.count
            )
            LipDebugLog.throttled(
                "lip_metal_ready",
                interval: 10,
                "lip_metal ready device=\(device.name) triangles=\(indices.count / 3)"
            )
        } catch {
            context = nil
            LipDebugLog.throttled(
                "lip_metal_unavailable",
                interval: 10,
                "lip_metal unavailable reason=pipeline error=\(error.localizedDescription)"
            )
        }
    }

    func makeTexture(contour: LipContour,
                     pixelBuffer: CVPixelBuffer,
                     imageSize: CGSize,
                     viewportSize: CGSize,
                     pixelWidth: Int,
                     pixelHeight: Int,
                     renderScale: CGFloat,
                     color: RGBColor,
                     lightingFactor: CGFloat,
                     finish: LipFinish) -> LipTexture? {
        guard let context,
              isAvailable,
              pixelWidth > 1,
              pixelHeight > 1,
              CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              let vertices = makeVertices(
                contour: contour,
                imageSize: imageSize,
                viewportSize: viewportSize
              ),
              let outerDistancePoints = makeOuterDistancePoints(
                contour: contour,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
              ),
              let innerDistancePoints = makeInnerDistancePoints(
                contour: contour,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
              ),
              let targets = renderTargets(
                context: context,
                width: pixelWidth,
                height: pixelHeight
              ) else {
            return nil
        }

        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        var cameraTextureReference: CVMetalTexture?
        let textureStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            context.textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            sourceWidth,
            sourceHeight,
            0,
            &cameraTextureReference
        )
        guard textureStatus == kCVReturnSuccess,
              let cameraTextureReference,
              let cameraTexture = CVMetalTextureGetTexture(cameraTextureReference) else {
            recordRuntimeFailure("camera_texture status=\(textureStatus)")
            return nil
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            recordRuntimeFailure("command_buffer")
            return nil
        }
        commandBuffer.label = "VirtualMakeup.LipComposite"

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            recordRuntimeFailure("render_encoder")
            return nil
        }
        renderEncoder.label = "VirtualMakeup.LipShade"
        renderEncoder.setRenderPipelineState(context.renderPipeline)
        vertices.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                renderEncoder.setVertexBytes(baseAddress, length: bytes.count, index: 0)
            }
        }
        var uniforms = Uniforms(
            lipstickColor: SIMD4<Float>(color.red, color.green, color.blue, 1),
            sourceTexelSize: SIMD2<Float>(
                1 / Float(max(sourceWidth, 1)),
                1 / Float(max(sourceHeight, 1))
            ),
            // The camera contributes lightness/detail, never its RGB hue.
            toneStrength: finish == .matte ? 0.92 : 1,
            pigmentStrength: finish == .satin ? 0.42 : 0.9,
            saturationStrength: 1,
            lightingFactor: Float(max(0.5, min(lightingFactor, 1))),
            detailStrength: finish == .matte ? 0.92 : 0.68,
            coreCoverage: 1,
            finish: Float(finish.rawValue)
        )
        renderEncoder.setFragmentBytes(
            &uniforms,
            length: MemoryLayout<Uniforms>.stride,
            index: 0
        )
        renderEncoder.setFragmentTexture(cameraTexture, index: 0)
        renderEncoder.setCullMode(.none)
        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: context.indexCount,
            indexType: .uint16,
            indexBuffer: context.indexBuffer,
            indexBufferOffset: 0
        )
        renderEncoder.endEncoding()

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("compute_encoder")
            return nil
        }
        computeEncoder.label = "VirtualMakeup.LipFeather"
        computeEncoder.setComputePipelineState(context.compositePipeline)
        computeEncoder.setTexture(targets.color, index: 0)
        computeEncoder.setTexture(targets.output, index: 1)
        outerDistancePoints.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                computeEncoder.setBytes(
                    baseAddress,
                    length: bytes.count,
                    index: 0
                )
            }
        }
        innerDistancePoints.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                computeEncoder.setBytes(
                    baseAddress,
                    length: bytes.count,
                    index: 2
                )
            }
        }
        let apertureOpeningPosition = min(
            max((contour.effectiveInnerOpeningRatio - 0.035) / (0.080 - 0.035), 0),
            1
        )
        let smoothApertureOpening = apertureOpeningPosition *
            apertureOpeningPosition *
            (3 - 2 * apertureOpeningPosition)
        let closedApertureFeather: Float = finish == .matte ? 2.0 : 1.0
        let openApertureFeather: Float = finish == .matte ? 6.0 : 3.0
        var featherUniforms = FeatherUniforms(
            textureSize: SIMD2<Float>(Float(pixelWidth), Float(pixelHeight)),
            outerPointCount: UInt32(outerDistancePoints.count),
            innerPointCount: UInt32(innerDistancePoints.count),
            outerFeatherPixels: 3.0,
            outerCoreInsetPixels: 7.0,
            apertureFeatherPixels: closedApertureFeather +
                Float(smoothApertureOpening) *
                (openApertureFeather - closedApertureFeather),
            apertureInsetPixels: Float(18 - smoothApertureOpening * 15)
        )
        computeEncoder.setBytes(
            &featherUniforms,
            length: MemoryLayout<FeatherUniforms>.stride,
            index: 1
        )
        let threadWidth = context.compositePipeline.threadExecutionWidth
        let threadHeight = max(
            1,
            context.compositePipeline.maxTotalThreadsPerThreadgroup / max(threadWidth, 1)
        )
        computeEncoder.dispatchThreads(
            MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threadWidth,
                height: min(threadHeight, 16),
                depth: 1
            )
        )
        computeEncoder.endEncoding()

        let startedAt = CACurrentMediaTime()
        withExtendedLifetime(cameraTextureReference) {
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        guard commandBuffer.status == .completed else {
            recordRuntimeFailure(
                "gpu status=\(commandBuffer.status.rawValue) error=\(commandBuffer.error?.localizedDescription ?? "none")"
            )
            return nil
        }

        var rgba = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
        rgba.withUnsafeMutableBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                return
            }
            targets.output.getBytes(
                baseAddress,
                bytesPerRow: pixelWidth * 4,
                from: MTLRegionMake2D(0, 0, pixelWidth, pixelHeight),
                mipmapLevel: 0
            )
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cgImage = CGImage(
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            recordRuntimeFailure("cgimage")
            return nil
        }

        recordRuntimeSuccess()
        let gpuMilliseconds = commandBuffer.gpuEndTime > commandBuffer.gpuStartTime ?
            (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000 : 0
        let blockingMilliseconds = (CACurrentMediaTime() - startedAt) * 1000
        LipDebugLog.throttled(
            "lip_metal_stats",
            interval: 0.6,
            "lip_metal frame size=\(pixelWidth)x\(pixelHeight) gpuMs=\(String(format: "%.2f", gpuMilliseconds)) waitReadbackMs=\(String(format: "%.2f", blockingMilliseconds)) lighting=\(String(format: "%.3f", lightingFactor)) apertureRaw=\(String(format: "%.3f", contour.innerOpeningRatio)) apertureRendered=\(String(format: "%.3f", contour.effectiveInnerOpeningRatio)) apertureBlend=\(String(format: "%.2f", smoothApertureOpening))"
        )
        return LipTexture(
            image: UIImage(cgImage: cgImage, scale: renderScale, orientation: .up)
        )
    }

    private func makeVertices(contour: LipContour,
                              imageSize: CGSize,
                              viewportSize: CGSize) -> [Vertex]? {
        guard imageSize.width > 1,
              imageSize.height > 1,
              viewportSize.width > 1,
              viewportSize.height > 1 else {
            return nil
        }
        let inverseImageTransform = Self.aspectFillTransform(
            for: imageSize,
            in: viewportSize
        ).inverted()
        var vertices: [Vertex] = []
        vertices.reserveCapacity(CanonicalLipGeometry.attentionLipIndices.count)
        for index in CanonicalLipGeometry.attentionLipIndices {
            guard let point = contour.meshPointsByIndex[index] else {
                return nil
            }

            var renderScreenPoint = point.screen
            var renderCanonicalUV = point.uv

            if CanonicalLipGeometry.isOuterLipIndex(index),
               let pose = contour.pose {
                let cosine = cos(pose.angle)
                let sine = sin(pose.angle)

                let dx = point.screen.x - pose.center.x
                let dy = point.screen.y - pose.center.y

                let localX = dx * cosine + dy * sine
                let localY = -dx * sine + dy * cosine
                // Выводим источник цвета немного за настоящий контур.
                let featherWidth = LipOuterMargin.sampling(
                    for: pose.width
                )
                let radialLength = max(
                    sqrt(localX * localX + localY * localY),
                    0.001
                )

                let expandedX =
                    localX + localX / radialLength * featherWidth
                let expandedY =
                    localY + localY / radialLength * featherWidth

                renderScreenPoint = CGPoint(
                    x: pose.center.x +
                        expandedX * cosine -
                        expandedY * sine,
                    y: pose.center.y +
                        expandedX * sine +
                        expandedY * cosine
                )

                // Выводим цвет в созданный UV-padding.
                let outerExpansionX: CGFloat = 1.18
                let outerExpansionY: CGFloat = 1.30
                renderCanonicalUV = CGPoint(
                    x: 0.5 + (point.uv.x - 0.5) * outerExpansionX,
                    y: 0.5 + (point.uv.y - 0.5) * outerExpansionY
                )
            } else if CanonicalLipGeometry.isInnerLipIndex(index) {
                // Match the SceneKit aperture compression while keeping the
                // canonical UV boundary stable. This changes only which live
                // camera pixels provide colour near a closed mouth seam.
                renderScreenPoint = contour.renderingInnerScreenPoint(
                    point.screen
                )
            }

            let imagePoint = renderScreenPoint.applying(inverseImageTransform)
            let sourceU = Float(imagePoint.x / imageSize.width)
            let sourceV = Float(imagePoint.y / imageSize.height)

            let canonicalU = Float(renderCanonicalUV.x)
            let canonicalV = Float(renderCanonicalUV.y)

            guard sourceU.isFinite,
                  sourceV.isFinite,
                  canonicalU.isFinite,
                  canonicalV.isFinite else {
                return nil
            }
            vertices.append(
                Vertex(
                    canonicalUV: SIMD2<Float>(canonicalU, canonicalV),
                    sourceUV: SIMD2<Float>(
                        min(max(sourceU, 0), 1),
                        min(max(sourceV, 0), 1)
                    )
                )
            )
        }
        return vertices
    }

    private func makeOuterDistancePoints(contour: LipContour,
                                         pixelWidth: Int,
                                         pixelHeight: Int) -> [SIMD2<Float>]? {
        guard contour.outerUV.count == CanonicalLipGeometry.outerLipIndices.count,
              let outerCurve = Self.smoothedClosedBoundary(
                contour.outerUV,
                radialScale: 1.00
              ),
              let outerBounds = Self.bounds(for: outerCurve),
              outerCurve.count >= 4,
              outerBounds.width > 0.000_1,
              outerBounds.height > 0.000_1,
              pixelWidth > 1,
              pixelHeight > 1 else {
            return nil
        }

        // Keep the upper lip exactly on the detected contour. On the lower
        // lip, move only the SDF zero boundary slightly outwards so the real
        // vermilion edge stays in the opaque core instead of the translucent
        // half of the feather. Smooth weighting leaves both mouth corners in
        // place and prevents a visible step between upper and lower halves.
        let lowerCoverageExtensionPixels: CGFloat = 1.5
        let lowerSpan = max(outerBounds.midY - outerBounds.minY, 0.000_1)
        let halfWidth = max(outerBounds.width * 0.5, 0.000_1)
        return outerCurve.map {
            let lowerPosition = min(
                max((outerBounds.midY - $0.y) / lowerSpan, 0),
                1
            )
            // Saturate early across the lower half so the centre and its
            // neighbours receive the same translation. Only the transition
            // into the mouth corners is tapered.
            let lowerMembershipPosition = min(lowerPosition / 0.35, 1)
            let lowerMembership = lowerMembershipPosition *
                lowerMembershipPosition *
                (3 - 2 * lowerMembershipPosition)
            let horizontalPosition = min(
                abs($0.x - outerBounds.midX) / halfWidth,
                1
            )
            let cornerTaperPosition = min(
                max((1 - horizontalPosition) / 0.28, 0),
                1
            )
            let cornerTaper = cornerTaperPosition * cornerTaperPosition *
                (3 - 2 * cornerTaperPosition)
            let lowerWeight = lowerMembership * cornerTaper
            return SIMD2<Float>(
                Float($0.x) * Float(pixelWidth),
                (1 - Float($0.y)) * Float(pixelHeight) +
                    Float(lowerCoverageExtensionPixels * lowerWeight)
            )
        }
    }

    private func makeInnerDistancePoints(contour: LipContour,
                                         pixelWidth: Int,
                                         pixelHeight: Int) -> [SIMD2<Float>]? {
        guard contour.innerUV.count == CanonicalLipGeometry.innerLipIndices.count,
              let innerCurve = Self.smoothedClosedBoundary(
                contour.innerUV,
                radialScale: 1.00
              ),
              innerCurve.count >= 4,
              pixelWidth > 1,
              pixelHeight > 1 else {
            return nil
        }
        return innerCurve.map {
            SIMD2<Float>(
                Float($0.x) * Float(pixelWidth),
                (1 - Float($0.y)) * Float(pixelHeight)
            )
        }
    }

    private static func smoothedClosedBoundary(
        _ controlPoints: [CGPoint],
        radialScale: CGFloat
    ) -> [CGPoint]? {
        guard controlPoints.count >= 4,
              let sourceBounds = bounds(for: controlPoints),
              sourceBounds.width > 0.000_1,
              sourceBounds.height > 0.000_1 else {
            return nil
        }

        var curve = controlPoints
        for _ in 0..<2 {
            var subdivided: [CGPoint] = []
            subdivided.reserveCapacity(curve.count * 2)
            for index in curve.indices {
                let current = curve[index]
                let next = curve[(index + 1) % curve.count]
                subdivided.append(
                    CGPoint(
                        x: current.x + (next.x - current.x) * 0.25,
                        y: current.y + (next.y - current.y) * 0.25
                    )
                )
                subdivided.append(
                    CGPoint(
                        x: current.x + (next.x - current.x) * 0.75,
                        y: current.y + (next.y - current.y) * 0.75
                    )
                )
            }
            curve = subdivided
        }

        guard let curveBounds = bounds(for: curve),
              curveBounds.width > 0.000_1,
              curveBounds.height > 0.000_1 else {
            return nil
        }
        let sourceCenter = CGPoint(x: sourceBounds.midX, y: sourceBounds.midY)
        let curveCenter = CGPoint(x: curveBounds.midX, y: curveBounds.midY)
        let xScale = sourceBounds.width / curveBounds.width * radialScale
        let yScale = sourceBounds.height / curveBounds.height * radialScale
        let result = curve.map {
            CGPoint(
                x: sourceCenter.x + ($0.x - curveCenter.x) * xScale,
                y: sourceCenter.y + ($0.y - curveCenter.y) * yScale
            )
        }
        guard result.allSatisfy({
            $0.x.isFinite && $0.y.isFinite &&
                $0.x >= 0 && $0.x <= 1 &&
                $0.y >= 0 && $0.y <= 1
        }) else {
            return nil
        }
        return result
    }

    private static func bounds(for points: [CGPoint]) -> CGRect? {
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
        return CGRect(
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        )
    }

    private func renderTargets(context: Context,
                               width: Int,
                               height: Int) -> RenderTargets? {
        if let renderTargets = context.renderTargets,
           renderTargets.width == width,
           renderTargets.height == height {
            return renderTargets
        }

        func makeTexture(usage: MTLTextureUsage) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .rgba8Unorm,
                width: width,
                height: height,
                mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = usage
            return context.device.makeTexture(descriptor: descriptor)
        }

        guard let color = makeTexture(usage: [.renderTarget, .shaderRead]),
              let output = makeTexture(usage: [.shaderRead, .shaderWrite]) else {
            recordRuntimeFailure("render_targets")
            return nil
        }
        color.label = "VirtualMakeup.LipColor"
        output.label = "VirtualMakeup.LipOutput"
        let targets = RenderTargets(
            width: width,
            height: height,
            color: color,
            output: output
        )
        context.renderTargets = targets
        return targets
    }

    private func recordRuntimeFailure(_ reason: String) {
        availabilityLock.lock()
        consecutiveRuntimeFailures += 1
        if consecutiveRuntimeFailures >= 3 {
            isRuntimeDisabled = true
        }
        let failures = consecutiveRuntimeFailures
        let disabled = isRuntimeDisabled
        availabilityLock.unlock()
        LipDebugLog.throttled(
            "lip_metal_runtime_failure",
            interval: 0.6,
            "lip_metal failure reason=\(reason) consecutive=\(failures) disabled=\(disabled)"
        )
    }

    private func recordRuntimeSuccess() {
        availabilityLock.lock()
        consecutiveRuntimeFailures = 0
        availabilityLock.unlock()
    }

    private static func makeIndices() -> [UInt16] {
        let indexByLandmark = Dictionary(
            uniqueKeysWithValues: CanonicalLipGeometry.attentionLipIndices
                .enumerated()
                .map { ($0.element, UInt16($0.offset)) }
        )
        var indices: [UInt16] = []
        indices.reserveCapacity(CanonicalLipGeometry.lipMeshTriangles.count * 3)
        for triangle in CanonicalLipGeometry.lipMeshTriangles {
            guard let first = indexByLandmark[triangle.0],
                  let second = indexByLandmark[triangle.1],
                  let third = indexByLandmark[triangle.2] else {
                return []
            }
            indices.append(contentsOf: [first, second, third])
        }
        return indices
    }

    private static func aspectFillTransform(for imageSize: CGSize,
                                            in viewportSize: CGSize) -> CGAffineTransform {
        let scale = max(
            viewportSize.width / imageSize.width,
            viewportSize.height / imageSize.height
        )
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        return CGAffineTransform(
            translationX: (viewportSize.width - scaledWidth) * 0.5,
            y: (viewportSize.height - scaledHeight) * 0.5
        ).scaledBy(x: scale, y: scale)
    }
}

private final class LipTextureRenderer {
    private struct CameraLipSample {
        let base: RGBColor
        let blurred: RGBColor
    }

    private struct MaterialTone {
        static let neutral = MaterialTone(detail: 1, shadow: 1)

        let detail: Float
        let shadow: Float
    }

    private struct PixelMaterialInput {
        let alpha: Float
        let mediaPipeAlpha: Float
        let outerDistance: CGFloat
        let innerDistance: CGFloat
        let lipBase: RGBColor
        let pigment: RGBColor
        let u: CGFloat
        let topV: CGFloat
    }

    private struct ProceduralTextureKey: Equatable {
        let colorSignature: UInt32
        let finish: LipFinish
        let renderScale: Int
        let excludesInnerMouth: Bool
    }

    private let temporalTextureLock = NSLock()
    private var previousToneMap: [MaterialTone]?
    private var previousToneMapWidth = 0
    private var previousToneMapHeight = 0
    private var previousToneColor: RGBColor?
    private var previousToneMouthOpen: Bool?
    private var previousToneAverageLuma: Float?
    private var previousSemanticAlphaMap: [Float]?
    private var previousSemanticAlphaWidth = 0
    private var previousSemanticAlphaHeight = 0
    private var lipstickFinish: LipFinish = .matte
    private var cachedProceduralTextureKey: ProceduralTextureKey?
    private var cachedProceduralTexture: LipTexture?

    private struct CanonicalLipSample {
        let point: CGPoint
        let alpha: Float
        let outerDistance: CGFloat
        let innerDistance: CGFloat
    }

    private struct CanonicalAlphaSample {
        let alpha: Float
        let outerDistance: CGFloat
        let innerDistance: CGFloat
    }

    private struct CanonicalLipSampler {
        private struct UVVertex {
            let uv: CGPoint
            let screen: CGPoint
        }

        private let vertices: [UVVertex]
        private let triangles: [(Int, Int, Int)]
        private let outerBoundarySegments: [(CGPoint, CGPoint)]
        private let innerBoundarySegments: [(CGPoint, CGPoint)]
        private let outerBoundary: [CGPoint]
        private let innerBoundary: [CGPoint]
        private let excludesInnerMouth: Bool

        init?(contour: LipContour, excludesInnerMouth: Bool) {
            guard contour.outer.count == 20,
                  contour.inner.count == 20,
                  contour.outerUV.count == 20,
                  contour.innerUV.count == 20 else {
                return nil
            }

            var builtVertices: [UVVertex] = []
            var builtTriangles: [(Int, Int, Int)] = []
            var vertexIndexByLandmark: [Int: Int] = [:]

            func vertex(for landmarkIndex: Int) -> UVVertex? {
                if let point = contour.meshPointsByIndex[landmarkIndex] {
                    return UVVertex(uv: point.uv, screen: point.screen)
                }

                if let outerOffset = CanonicalLipGeometry.outerLipIndices.firstIndex(of: landmarkIndex),
                   contour.outerUV.indices.contains(outerOffset),
                   contour.outer.indices.contains(outerOffset) {
                    return UVVertex(uv: contour.outerUV[outerOffset], screen: contour.outer[outerOffset])
                }

                if let innerOffset = CanonicalLipGeometry.innerLipIndices.firstIndex(of: landmarkIndex),
                   contour.innerUV.indices.contains(innerOffset),
                   contour.inner.indices.contains(innerOffset) {
                    return UVVertex(uv: contour.innerUV[innerOffset], screen: contour.inner[innerOffset])
                }

                return nil
            }

            func appendOrReuse(_ landmarkIndex: Int, vertex: UVVertex) -> Int {
                if let existing = vertexIndexByLandmark[landmarkIndex] {
                    return existing
                }

                let newIndex = builtVertices.count
                builtVertices.append(vertex)
                vertexIndexByLandmark[landmarkIndex] = newIndex
                return newIndex
            }

            for triangle in CanonicalLipGeometry.lipMeshTriangles {
                guard let first = vertex(for: triangle.0),
                      let second = vertex(for: triangle.1),
                      let third = vertex(for: triangle.2) else {
                    continue
                }

                builtTriangles.append((
                    appendOrReuse(triangle.0, vertex: first),
                    appendOrReuse(triangle.1, vertex: second),
                    appendOrReuse(triangle.2, vertex: third)
                ))
            }

            guard builtTriangles.count == CanonicalLipGeometry.lipMeshTriangles.count,
                  builtVertices.count == CanonicalLipGeometry.attentionLipIndices.count else {
                return nil
            }

            // MediaPipe exposes only twenty points on each visible boundary.
            // Drawing those points as a polygon leaves the cupid's bow and the
            // lower edge visibly faceted. A twice-subdivided Chaikin curve is
            // deterministic, stays close to the control polygon and gives the
            // alpha mask enough samples to form a continuous silhouette.
            guard let builtOuterCurve = Self.smoothedClosedBoundary(
                      contour.outerUV,
                      radialScale: 0.985
                  ),
                  let builtInnerCurve = Self.smoothedClosedBoundary(
                      contour.innerUV,
                      radialScale: 1.015
                  ) else {
                return nil
            }

            vertices = builtVertices
            triangles = builtTriangles
            outerBoundarySegments = Self.closedSegments(for: builtOuterCurve)
            innerBoundarySegments = Self.closedSegments(for: builtInnerCurve)
            outerBoundary = builtOuterCurve
            innerBoundary = builtInnerCurve
            self.excludesInnerMouth = excludesInnerMouth
        }

        func sample(u: CGFloat, topV: CGFloat) -> CanonicalLipSample? {
            guard let alphaSample = alphaSample(u: u, topV: topV),
                  let screen = screenPoint(u: u, topV: topV) else {
                return nil
            }
            return CanonicalLipSample(
                point: screen,
                alpha: alphaSample.alpha,
                outerDistance: alphaSample.outerDistance,
                innerDistance: alphaSample.innerDistance
            )
        }

        func alphaSample(u: CGFloat, topV: CGFloat) -> CanonicalAlphaSample? {
            guard let edgeDistances = edgeDistances(u: u, topV: topV) else {
                return nil
            }

            // Reach zero before the polygon mesh boundary. Otherwise SceneKit
            // exposes its straight outer edge even when the texture itself is
            // filtered linearly.
            let distanceToBoundary = min(edgeDistances.outer, edgeDistances.inner)
            let featherWidth: CGFloat = 4.75 / 192.0
            let smoothCoverage = Self.smoothStep(
                edge0: 0,
                edge1: featherWidth,
                value: distanceToBoundary
            )
            // Keep more of the transition translucent. This produces a soft
            // cosmetic feather instead of a mathematically sharp silhouette.
            let edgeCoverage = pow(smoothCoverage, 1.18)
            return CanonicalAlphaSample(
                alpha: Float(edgeCoverage),
                outerDistance: edgeDistances.outer,
                innerDistance: edgeDistances.inner
            )
        }

        private func screenPoint(u: CGFloat, topV: CGFloat) -> CGPoint? {
            let uv = CGPoint(x: max(0, min(u, 1)), y: 1 - max(0, min(topV, 1)))
            for triangle in triangles {
                let first = vertices[triangle.0]
                let second = vertices[triangle.1]
                let third = vertices[triangle.2]
                guard let barycentric = Self.barycentric(point: uv, a: first.uv, b: second.uv, c: third.uv) else {
                    continue
                }

                let screen = CGPoint(
                    x: first.screen.x * barycentric.u + second.screen.x * barycentric.v + third.screen.x * barycentric.w,
                    y: first.screen.y * barycentric.u + second.screen.y * barycentric.v + third.screen.y * barycentric.w
                )
                return screen
            }

            return nil
        }

        private func edgeDistances(u: CGFloat, topV: CGFloat) -> (outer: CGFloat, inner: CGFloat)? {
            let uv = CGPoint(x: max(0, min(u, 1)), y: 1 - max(0, min(topV, 1)))
            guard Self.pointInPolygon(uv, polygon: outerBoundary),
                  (!excludesInnerMouth || !Self.pointInPolygon(uv, polygon: innerBoundary)) else {
                return nil
            }

            let outerDistanceSquared = outerBoundarySegments.reduce(CGFloat.greatestFiniteMagnitude) { current, segment in
                min(current, Self.squaredDistance(from: uv, to: segment))
            }
            let innerDistanceSquared = innerBoundarySegments.reduce(CGFloat.greatestFiniteMagnitude) { current, segment in
                min(current, Self.squaredDistance(from: uv, to: segment))
            }
            return (sqrt(outerDistanceSquared), sqrt(innerDistanceSquared))
        }

        private static func mix(_ first: CGPoint, _ second: CGPoint, t: CGFloat) -> CGPoint {
            CGPoint(
                x: first.x + (second.x - first.x) * t,
                y: first.y + (second.y - first.y) * t
            )
        }

        private static func smoothedClosedBoundary(
            _ controlPoints: [CGPoint],
            radialScale: CGFloat
        ) -> [CGPoint]? {
            guard controlPoints.count >= 4,
                  let sourceBounds = bounds(for: controlPoints),
                  sourceBounds.width > 0.000_1,
                  sourceBounds.height > 0.000_1 else {
                return nil
            }

            var curve = controlPoints
            for _ in 0..<2 {
                var subdivided: [CGPoint] = []
                subdivided.reserveCapacity(curve.count * 2)
                for index in curve.indices {
                    let current = curve[index]
                    let next = curve[(index + 1) % curve.count]
                    subdivided.append(mix(current, next, t: 0.25))
                    subdivided.append(mix(current, next, t: 0.75))
                }
                curve = subdivided
            }

            guard let curveBounds = bounds(for: curve),
                  curveBounds.width > 0.000_1,
                  curveBounds.height > 0.000_1 else {
                return nil
            }
            let sourceCenter = CGPoint(x: sourceBounds.midX, y: sourceBounds.midY)
            let curveCenter = CGPoint(x: curveBounds.midX, y: curveBounds.midY)
            let xScale = sourceBounds.width / curveBounds.width * radialScale
            let yScale = sourceBounds.height / curveBounds.height * radialScale
            return curve.map {
                CGPoint(
                    x: sourceCenter.x + ($0.x - curveCenter.x) * xScale,
                    y: sourceCenter.y + ($0.y - curveCenter.y) * yScale
                )
            }
        }

        private static func closedSegments(
            for points: [CGPoint]
        ) -> [(CGPoint, CGPoint)] {
            guard points.count > 1 else {
                return []
            }
            return points.indices.map {
                (points[$0], points[($0 + 1) % points.count])
            }
        }

        private static func barycentric(point: CGPoint,
                                        a: CGPoint,
                                        b: CGPoint,
                                        c: CGPoint) -> (u: CGFloat, v: CGFloat, w: CGFloat)? {
            let v0 = CGPoint(x: b.x - a.x, y: b.y - a.y)
            let v1 = CGPoint(x: c.x - a.x, y: c.y - a.y)
            let v2 = CGPoint(x: point.x - a.x, y: point.y - a.y)
            let denominator = v0.x * v1.y - v1.x * v0.y
            guard abs(denominator) > 0.000001 else {
                return nil
            }

            let v = (v2.x * v1.y - v1.x * v2.y) / denominator
            let w = (v0.x * v2.y - v2.x * v0.y) / denominator
            let u = 1 - v - w
            let epsilon: CGFloat = -0.0005
            guard u >= epsilon, v >= epsilon, w >= epsilon else {
                return nil
            }
            return (u, v, w)
        }

        private static func squaredDistance(from point: CGPoint, to segment: (CGPoint, CGPoint)) -> CGFloat {
            let start = segment.0
            let end = segment.1
            let dx = end.x - start.x
            let dy = end.y - start.y
            let lengthSquared = dx * dx + dy * dy
            guard lengthSquared > 0 else {
                let pointDX = point.x - start.x
                let pointDY = point.y - start.y
                return pointDX * pointDX + pointDY * pointDY
            }

            let t = max(0, min(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared, 1))
            let projection = CGPoint(x: start.x + dx * t, y: start.y + dy * t)
            let projectionDX = point.x - projection.x
            let projectionDY = point.y - projection.y
            return projectionDX * projectionDX + projectionDY * projectionDY
        }

        private static func pointInPolygon(_ point: CGPoint, polygon: [CGPoint]) -> Bool {
            guard polygon.count >= 3 else {
                return false
            }

            var isInside = false
            var previousIndex = polygon.count - 1
            for index in polygon.indices {
                let current = polygon[index]
                let previous = polygon[previousIndex]
                let crosses = (current.y > point.y) != (previous.y > point.y)
                if crosses {
                    let denominator = previous.y - current.y
                    if abs(denominator) > 0.000001 {
                        let intersectionX = (previous.x - current.x) *
                            (point.y - current.y) / denominator + current.x
                        if point.x < intersectionX {
                            isInside.toggle()
                        }
                    }
                }
                previousIndex = index
            }

            return isInside
        }

        private static func bounds(for points: [CGPoint]) -> CGRect? {
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

        private static func smoothStep(edge0: CGFloat, edge1: CGFloat, value: CGFloat) -> CGFloat {
            let t = max(0, min((value - edge0) / max(edge1 - edge0, 0.0001), 1))
            return t * t * (3 - 2 * t)
        }
    }

    private let colorLock = NSLock()
    private let metalCompositor = MetalLipColorCompositor()
    private var lipstickColor = RGBColor(red: 0.82, green: 0.08, blue: 0.08)
    private var lipstickStyleSignature: UInt32 = 0

    var supportsRealtimeCompositing: Bool {
        metalCompositor.isAvailable
    }

    @discardableResult
    func updateStyle(color: UIColor, finish: LipFinish) -> Bool {
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

        colorLock.lock()
        let nextColor = RGBColor(red: Float(red), green: Float(green), blue: Float(blue))
        let colorDistance = Self.colorDistance(lipstickColor, nextColor)
        let nextSignature = Self.colorSignature(nextColor)
        let didChange = lipstickStyleSignature != nextSignature || lipstickFinish != finish
        if didChange {
            lipstickColor = nextColor
            lipstickFinish = finish
            lipstickStyleSignature = nextSignature
        }
        colorLock.unlock()

        guard didChange else {
            return false
        }

        resetTemporalState()
        LipDebugLog.throttled(
            "lip_color_update",
            interval: 0.2,
            "lip_color update distance=\(String(format: "%.4f", Double(colorDistance))) finish=\(finish.rawValue) rgb=(\(String(format: "%.3f", Double(nextColor.red))),\(String(format: "%.3f", Double(nextColor.green))),\(String(format: "%.3f", Double(nextColor.blue))))"
        )
        return true
    }

    func resetTemporalState(preservingProceduralTexture: Bool = false) {
        temporalTextureLock.lock()
        previousToneMap = nil
        previousToneMapWidth = 0
        previousToneMapHeight = 0
        previousToneColor = nil
        previousToneMouthOpen = nil
        previousToneAverageLuma = nil
        previousSemanticAlphaMap = nil
        previousSemanticAlphaWidth = 0
        previousSemanticAlphaHeight = 0
        if !preservingProceduralTexture {
            cachedProceduralTextureKey = nil
            cachedProceduralTexture = nil
        }
        temporalTextureLock.unlock()
    }

    func makeTexture(contour: LipContour,
                     pixelBuffer: CVPixelBuffer,
                     imageSize: CGSize,
                     viewportSize: CGSize,
                     renderScale: CGFloat,
                     excludesInnerMouth: Bool,
                     semanticMask: SemanticLipMask?,
                     lightingFactor: CGFloat,
                     lowLatency: Bool = false,
                     motionDelta: CGFloat = 0) -> LipTexture? {
        guard viewportSize.width > 1,
              viewportSize.height > 1,
              imageSize.width > 1,
              imageSize.height > 1 else {
            LipDebugLog.throttled(
                "lip_texture_bad_sizes",
                "lip_texture nil reason=bad_sizes image=\(Int(imageSize.width))x\(Int(imageSize.height)) viewport=\(Int(viewportSize.width))x\(Int(viewportSize.height))"
            )
            return nil
        }

        // Keep the CPU fallback compact. Metal can afford a 1.5x oversampled
        // target, which prevents the curved feather from quantizing into
        // visible steps when the mouth fills much of the screen.
        let pixelWidth = lowLatency ? 128 : 160
        let pixelHeight = lowLatency ? 64 : 80
        let metalPixelWidth = lowLatency ? 192 : 240
        let metalPixelHeight = lowLatency ? 96 : 120

        let style = currentLipstickStyle()

        if metalCompositor.isAvailable {
            if let texture = metalCompositor.makeTexture(
                contour: contour,
                pixelBuffer: pixelBuffer,
                imageSize: imageSize,
                viewportSize: viewportSize,
                pixelWidth: metalPixelWidth,
                pixelHeight: metalPixelHeight,
                renderScale: renderScale,
                color: style.color,
                lightingFactor: lightingFactor,
                finish: style.finish
            ) {
                return texture
            }
            // A transient GPU failure must not immediately start the old
            // 400-580 ms CPU loop and contend with live landmark tracking.
            guard !metalCompositor.isAvailable else {
                return nil
            }
        }

        guard let sampler = CanonicalLipSampler(contour: contour, excludesInnerMouth: excludesInnerMouth) else {
            LipDebugLog.throttled(
                "lip_texture_no_sampler",
                "lip_texture nil reason=no_sampler outer=\(contour.outer.count) inner=\(contour.inner.count) outerUV=\(contour.outerUV.count) innerUV=\(contour.innerUV.count)"
            )
            return nil
        }

        if Self.usesStableProceduralLipstickRenderer {
            return makeStableProceduralTexture(
                sampler: sampler,
                style: style,
                renderScale: renderScale,
                excludesInnerMouth: excludesInnerMouth
            )
        }

        var rgba = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
        var alphaPixels = 0
        var paintedPixels = 0
        var semanticRejectedPixels = 0
        var materialInputs = [PixelMaterialInput?](repeating: nil, count: pixelWidth * pixelHeight)
        var toneMap = [MaterialTone](repeating: .neutral, count: pixelWidth * pixelHeight)
        var activeTonePixels = [Bool](repeating: false, count: pixelWidth * pixelHeight)
        var rawSemanticAlphaMap = [Float](repeating: 1, count: pixelWidth * pixelHeight)
        var cameraLumaSum: Float = 0
        var cameraLumaCount: Float = 0
        let hasSemanticMask = semanticMask != nil
        let inverseImageTransform = Self.aspectFillTransform(for: imageSize, in: viewportSize).inverted()
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let canSampleCamera = pixelFormat == kCVPixelFormatType_32BGRA

        if canSampleCamera {
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        }
        defer {
            if canSampleCamera {
                CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            }
        }

        let baseAddress = canSampleCamera ? CVPixelBufferGetBaseAddress(pixelBuffer) : nil
        let bytesPerRow = canSampleCamera ? CVPixelBufferGetBytesPerRow(pixelBuffer) : 0

        for y in 0..<pixelHeight {
            for x in 0..<pixelWidth {
                let u = (CGFloat(x) + 0.5) / CGFloat(pixelWidth)
                let topV = (CGFloat(y) + 0.5) / CGFloat(pixelHeight)
                guard let sample = sampler.sample(u: u, topV: topV),
                      sample.alpha > 0.01 else {
                    continue
                }
                alphaPixels += 1
                let semanticAlpha = semanticMask?.lipAlpha(atViewportPoint: sample.point) ?? 1
                let mapIndex = y * pixelWidth + x
                rawSemanticAlphaMap[mapIndex] = semanticAlpha
                if hasSemanticMask, semanticAlpha <= 0.01 {
                    semanticRejectedPixels += 1
                }

                let cameraSample: CameraLipSample?
                if let baseAddress {
                    cameraSample = Self.cameraLipSample(
                        atViewportPoint: sample.point,
                        inverseImageTransform: inverseImageTransform,
                        baseAddress: baseAddress,
                        bytesPerRow: bytesPerRow,
                        width: sourceWidth,
                        height: sourceHeight
                    )
                } else {
                    cameraSample = nil
                }

                let edgeNoise = Self.edgeNoise(maskAlpha: sample.alpha, x: x, y: y, frame: .zero)
                let pigmentVariation = Self.stableLipPigmentVariation(u: u, topV: topV)
                let finishAlphaScale: Float = style.finish == .satin ? 0.92 : 0.94
                let finishAlphaCeiling: Float = style.finish == .satin ? (lowLatency ? 0.92 : 0.88) : (lowLatency ? 0.94 : 0.90)
                let trustedCameraSample = semanticAlpha > 0.01 ? cameraSample : nil
                let lipBase = trustedCameraSample?.base ?? Self.stableLipBaseColor(u: u, topV: topV)
                let cosmeticCoverage = Self.cosmeticCoverage(
                    maskAlpha: sample.alpha,
                    u: u,
                    innerDistance: sample.innerDistance,
                    finish: style.finish
                )
                let baseAlpha = min(
                    cosmeticCoverage *
                    finishAlphaScale *
                    edgeNoise *
                    pigmentVariation.alpha *
                    Self.pigmentAlphaScale(
                        base: lipBase,
                        outerDistance: sample.outerDistance,
                        innerDistance: sample.innerDistance
                    ),
                    finishAlphaCeiling
                )
                let alpha = Self.finishAlpha(base: baseAlpha, u: u, topV: topV, finish: style.finish)
                let pigmentRGB = Self.pigmentBlendedColor(
                    lipstick: style.color,
                    variation: pigmentVariation.color
                )
                if let cameraSample = trustedCameraSample {
                    toneMap[mapIndex] = Self.cameraTone(base: cameraSample.base, blurred: cameraSample.blurred)
                    cameraLumaSum += Self.luminance(cameraSample.blurred)
                    cameraLumaCount += 1
                }
                activeTonePixels[mapIndex] = true
                materialInputs[mapIndex] = PixelMaterialInput(
                    alpha: alpha,
                    mediaPipeAlpha: sample.alpha,
                    outerDistance: sample.outerDistance,
                    innerDistance: sample.innerDistance,
                    lipBase: lipBase,
                    pigment: pigmentRGB,
                    u: u,
                    topV: topV
                )
            }
        }

        let stabilizedSemanticAlphaMap = temporallyStabilizedSemanticAlphaMap(
            rawSemanticAlphaMap,
            activePixels: activeTonePixels,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            hasSemanticMask: hasSemanticMask,
            motionDelta: motionDelta
        )
        let averageCameraLuma = cameraLumaCount > 0 ? cameraLumaSum / cameraLumaCount : nil
        let stabilizedToneMap = temporallyStabilizedToneMap(
            toneMap,
            activePixels: activeTonePixels,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            color: style.color,
            excludesInnerMouth: excludesInnerMouth,
            averageCameraLuma: averageCameraLuma,
            lowLatency: lowLatency,
            motionDelta: motionDelta
        )
        for mapIndex in materialInputs.indices {
            guard let input = materialInputs[mapIndex] else {
                continue
            }
            let semanticCoverage = hasSemanticMask ?
                Self.semanticShapeCoverage(
                    semanticAlpha: stabilizedSemanticAlphaMap[mapIndex],
                    mediaPipeAlpha: input.mediaPipeAlpha,
                    outerDistance: input.outerDistance,
                    innerDistance: input.innerDistance
                ) :
                1
            let effectiveAlpha = input.alpha * semanticCoverage
            guard effectiveAlpha > 0.006 else {
                continue
            }
                let materialRGB = Self.materialColor(
                    pigment: input.pigment,
                    base: input.lipBase,
                    tone: stabilizedToneMap[mapIndex],
                    u: input.u,
                topV: input.topV,
                alpha: effectiveAlpha,
                finish: style.finish
            )
            let premultiplied = Self.premultipliedCompositingColor(
                target: materialRGB,
                base: input.lipBase,
                alpha: effectiveAlpha
            )
            let offset = mapIndex * 4
            rgba[offset] = Self.uint8(premultiplied.red)
            rgba[offset + 1] = Self.uint8(premultiplied.green)
            rgba[offset + 2] = Self.uint8(premultiplied.blue)
            rgba[offset + 3] = Self.uint8(effectiveAlpha)
            paintedPixels += 1
        }

        var usedFallback = false
        var fallbackPaintedPixels = 0
        if paintedPixels < max(64, pixelWidth * pixelHeight / 80) {
            let fallback = Self.makeFallbackRGBA(
                sampler: sampler,
                color: style.color,
                finish: style.finish,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
            )
            rgba = fallback.rgba
            usedFallback = true
            fallbackPaintedPixels = fallback.paintedPixels
        }
        // CanonicalAlphaSample already supplies a smooth distance feather.
        // A second CPU blur both delayed the live colour and washed out the
        // captured lip detail we are deliberately preserving.

        LipDebugLog.throttled(
            "lip_texture_stats",
            interval: 0.6,
            "lip_texture stats size=\(pixelWidth)x\(pixelHeight) lowLatency=\(lowLatency) semanticBounded=\(semanticMask != nil) cameraSample=\(baseAddress != nil) alphaPixels=\(alphaPixels) painted=\(paintedPixels) semanticUntrusted=\(semanticRejectedPixels) fallback=\(usedFallback) fallbackPainted=\(fallbackPaintedPixels) image=\(Int(imageSize.width))x\(Int(imageSize.height)) viewport=\(Int(viewportSize.width))x\(Int(viewportSize.height))"
        )

        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cgImage = CGImage(
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            LipDebugLog.throttled(
                "lip_texture_cgimage_failed",
                "lip_texture nil reason=cgimage_failed size=\(pixelWidth)x\(pixelHeight) rgbaBytes=\(rgba.count)"
            )
            return nil
        }

        let texture = LipTexture(image: UIImage(cgImage: cgImage, scale: renderScale, orientation: .up))
        return texture
    }

    private func makeStableProceduralTexture(sampler: CanonicalLipSampler,
                                             style: (color: RGBColor, finish: LipFinish),
                                             renderScale: CGFloat,
                                             excludesInnerMouth: Bool) -> LipTexture? {
        let key = ProceduralTextureKey(
            colorSignature: Self.colorSignature(style.color),
            finish: style.finish,
            renderScale: Int((renderScale * 100).rounded()),
            excludesInnerMouth: excludesInnerMouth
        )

        temporalTextureLock.lock()
        if cachedProceduralTextureKey == key,
           let cachedProceduralTexture {
            temporalTextureLock.unlock()
            return cachedProceduralTexture
        }
        temporalTextureLock.unlock()

        // The visible mouth is normally below 130x60 viewport points. Linear
        // filtering plus the analytic curve keep 256x128 sub-pixel smooth,
        // while avoiding the >1s first raster seen at 384x192 on A14.
        let pixelWidth = 256
        let pixelHeight = 128
        let procedural = Self.makeFallbackRGBA(
            sampler: sampler,
            color: style.color,
            finish: style.finish,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
        )
        // CanonicalAlphaSample already provides a smooth analytic feather.
        // A second 3x3 CPU blur over 384x192 pixels took 3-6 seconds on A14
        // and caused the first texture request after every reset to go stale.
        guard let provider = CGDataProvider(data: Data(procedural.rgba) as CFData),
              let cgImage = CGImage(
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            LipDebugLog.throttled(
                "lip_texture_cgimage_failed",
                "lip_texture nil reason=procedural_cgimage_failed size=\(pixelWidth)x\(pixelHeight) rgbaBytes=\(procedural.rgba.count)"
            )
            return nil
        }

        let texture = LipTexture(image: UIImage(cgImage: cgImage, scale: renderScale, orientation: .up))
        temporalTextureLock.lock()
        cachedProceduralTextureKey = key
        cachedProceduralTexture = texture
        temporalTextureLock.unlock()

        LipDebugLog.throttled(
            "lip_texture_procedural",
            interval: 0.8,
            "lip_texture procedural stableRenderer=true size=\(pixelWidth)x\(pixelHeight) painted=\(procedural.paintedPixels) finish=\(style.finish.rawValue)"
        )
        return texture
    }

    private static func makeFallbackRGBA(sampler: CanonicalLipSampler,
                                         color: RGBColor,
                                         finish: LipFinish,
                                         pixelWidth: Int,
                                         pixelHeight: Int) -> (rgba: [UInt8], paintedPixels: Int) {
        var rgba = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
        var paintedPixels = 0
        for y in 0..<pixelHeight {
            for x in 0..<pixelWidth {
                let u = (CGFloat(x) + 0.5) / CGFloat(pixelWidth)
                let topV = (CGFloat(y) + 0.5) / CGFloat(pixelHeight)
                guard let sample = sampler.alphaSample(u: u, topV: topV),
                      sample.alpha > 0.01 else {
                    continue
                }

                let pigmentVariation = stableLipPigmentVariation(u: u, topV: topV)
                // Coverage is a geometric property. Finish/noise may alter RGB,
                // but must not punch semitransparent holes into the lipstick.
                // The user-facing material opacity remains the sole global
                // intensity control.
                // A tiny deterministic alpha grain breaks the perfectly even
                // vector-like edge without introducing temporal shimmer.
                let edgeGrain = sample.alpha < 0.995 ?
                    0.965 + pigmentVariation.color * 0.035 : 1
                let alpha = sample.alpha * edgeGrain
                let lipBase = stableLipBaseColor(u: u, topV: topV)
                let pigmentRGB = pigmentBlendedColor(
                    lipstick: color,
                    variation: pigmentVariation.color
                )
                let lipRGB = finishColor(
                    pigment: pigmentRGB,
                    lipBase: lipBase,
                    u: u,
                    topV: topV,
                    alpha: alpha,
                    finish: finish
                )
                let offset = (y * pixelWidth + x) * 4
                rgba[offset] = uint8(lipRGB.red * alpha)
                rgba[offset + 1] = uint8(lipRGB.green * alpha)
                rgba[offset + 2] = uint8(lipRGB.blue * alpha)
                rgba[offset + 3] = uint8(alpha)
                paintedPixels += 1
            }
        }
        return (rgba, paintedPixels)
    }

    private func currentLipstickStyle() -> (color: RGBColor, finish: LipFinish) {
        colorLock.lock()
        let color = lipstickColor
        let finish = lipstickFinish
        colorLock.unlock()
        return (color, finish)
    }

    private func averageCoreLipColor(frame: CGRect,
                                     scale: CGFloat,
                                     pixelWidth: Int,
                                     pixelHeight: Int,
                                     outerPath: CGPath,
                                     innerPath: CGPath,
                                     contour: LipContour,
                                     inverseImageTransform: CGAffineTransform,
                                     baseAddress: UnsafeMutableRawPointer,
                                     bytesPerRow: Int,
                                     sourceWidth: Int,
                                     sourceHeight: Int,
                                     featherRadius: CGFloat) -> RGBColor? {
        var red: Float = 0
        var green: Float = 0
        var blue: Float = 0
        var count: Float = 0
        let step = max(Int(scale.rounded()), 1)

        for y in stride(from: 0, to: pixelHeight, by: step * 2) {
            for x in stride(from: 0, to: pixelWidth, by: step * 2) {
                let viewportPoint = CGPoint(
                    x: frame.minX + (CGFloat(x) + 0.5) / scale,
                    y: frame.minY + (CGFloat(y) + 0.5) / scale
                )
                guard Self.isInsideLip(point: viewportPoint, outerPath: outerPath, innerPath: innerPath) else {
                    continue
                }

                let edgeDistance = min(
                    Self.distance(from: viewportPoint, toClosedPolyline: contour.outer),
                    Self.distance(from: viewportPoint, toClosedPolyline: contour.inner)
                )
                guard edgeDistance > featherRadius * 0.85,
                      let pixel = Self.samplePixel(
                        at: viewportPoint.applying(inverseImageTransform),
                        baseAddress: baseAddress,
                        bytesPerRow: bytesPerRow,
                        width: sourceWidth,
                        height: sourceHeight
                      ) else {
                    continue
                }

                red += pixel.red
                green += pixel.green
                blue += pixel.blue
                count += 1
            }
        }

        guard count >= 6 else {
            return nil
        }
        return RGBColor(red: red / count, green: green / count, blue: blue / count)
    }

    private func texturePreservingColor(base: RGBColor, lipstick: RGBColor) -> RGBColor {
        var luminance = max(0, min(base.red * 0.299 + base.green * 0.587 + base.blue * 0.114, 1))
        if Self.isToothLike(base) {
            luminance = min(luminance, 0.50)
        }

        let textureGain = max(0.34, min(0.44 + luminance * 0.98, 1.16))
        let highlight = Self.smoothStep((luminance - 0.62) / 0.26)
        return RGBColor(
            red: min(lipstick.red * textureGain + base.red * highlight * 0.10, 1),
            green: min(lipstick.green * textureGain + base.green * highlight * 0.10, 1),
            blue: min(lipstick.blue * textureGain + base.blue * highlight * 0.10, 1)
        )
    }

    private static func pigmentAlphaScale(base: RGBColor,
                                          outerDistance: CGFloat,
                                          innerDistance: CGFloat) -> Float {
        let luminance = max(0, min(base.red * 0.299 + base.green * 0.587 + base.blue * 0.114, 1))
        let highlight = smoothStep((luminance - 0.62) / 0.28)
        let outerCore = smoothStep((Float(outerDistance) - 0.018) / 0.070)
        let innerCore = smoothStep((Float(innerDistance) - 0.010) / 0.045)
        var scale = 1 - highlight * 0.20 * max(outerCore, innerCore)

        if isToothLike(base) && innerDistance < 0.060 {
            scale *= 0.24
        }

        return max(0.28, min(scale, 1))
    }

    private static func materialColor(pigment: RGBColor,
                                      base: RGBColor,
                                      tone: MaterialTone,
                                      u: CGFloat,
                                      topV: CGFloat,
                                      alpha: Float,
                                      finish: LipFinish) -> RGBColor {
        let sourceLuma = luminance(base)
        let pigmentLuma = luminance(pigment)
        // Preserve the source's local relief, not its absolute brightness.
        // Matching the pigment to sourceLuma made wine/berry shades as bright
        // as bare lips. Dark shades must lower the underlying lip luminance.
        let toneStrength: Float = finish == .matte ? 0.92 : 1
        let targetLuma = clamp01(
            sourceLuma + (pigmentLuma - sourceLuma) * toneStrength
        )
        let lumaPreservingPigment = colorWithLuminance(
            pigment,
            luminance: targetLuma
        )
        let saturatedPigment = colorWithSaturation(
            lumaPreservingPigment,
            strength: finish == .satin ? 1.18 : 1
        )
        let pigmentStrength: Float = finish == .satin ? 0.4 :
            (finish == .matte ? 0.9 : 1)
        let detailStrength = finish == .matte ?
            max(0.90, min(tone.detail, 1.08)) :
            max(0.95, min(tone.detail, 1.05))
        let shadow = max(0.975, min(tone.shadow, 1))
        let shadedPigment = RGBColor(
            red: clamp01((base.red * (1 - pigmentStrength) + saturatedPigment.red * pigmentStrength) * detailStrength * shadow),
            green: clamp01((base.green * (1 - pigmentStrength) + saturatedPigment.green * pigmentStrength) * detailStrength * shadow),
            blue: clamp01((base.blue * (1 - pigmentStrength) + saturatedPigment.blue * pigmentStrength) * detailStrength * shadow)
        )

        if finish == .satin {
            // The fallback also brightens only a highlight already present in
            // the camera detail map. It does not place a procedural reflection.
            let naturalHighlight = smoothStep(
                (detailStrength - 1.005) / 0.045
            )
            let highlightAmount = naturalHighlight * 0.18
            return RGBColor(
                red: clamp01(shadedPigment.red + (1 - shadedPigment.red) * highlightAmount),
                green: clamp01(shadedPigment.green + (1 - shadedPigment.green) * highlightAmount),
                blue: clamp01(shadedPigment.blue + (1 - shadedPigment.blue) * highlightAmount)
            )
        }

        return finishColor(
            pigment: shadedPigment,
            lipBase: shadedPigment,
            u: u,
            topV: topV,
            alpha: alpha,
            finish: finish
        )
    }

    private static func premultipliedCompositingColor(target: RGBColor,
                                                      base: RGBColor,
                                                      alpha: Float) -> RGBColor {
        let alpha = max(0.001, min(alpha, 1))
        func premultiply(_ targetChannel: Float, _ baseChannel: Float) -> Float {
            min(max(targetChannel - baseChannel * (1 - alpha), 0), alpha)
        }
        return RGBColor(
            red: premultiply(target.red, base.red),
            green: premultiply(target.green, base.green),
            blue: premultiply(target.blue, base.blue)
        )
    }

    private static func cameraTone(base: RGBColor, blurred: RGBColor) -> MaterialTone {
        let baseLuma = luminance(base)
        let blurredLuma = max(luminance(blurred), 0.055)
        // Keep a wider camera-detail range for matte lipstick. Individual
        // finishes apply their own narrower bounds in materialColor.
        let detail = max(0.90, min(baseLuma / blurredLuma, 1.08))
        let shadowAmount = smoothStep((0.26 - blurredLuma) / 0.18)
        return MaterialTone(
            detail: detail,
            shadow: 1 - shadowAmount * 0.025
        )
    }

    private static func cosmeticCoverage(maskAlpha: Float,
                                         u: CGFloat,
                                         innerDistance: CGFloat,
                                         finish: LipFinish) -> Float {
        let innerFade = 0.52 + 0.48 * smoothStep(
            (Float(innerDistance) - 0.004) / 0.040
        )
        let cornerPosition = min(abs(Float(u) - 0.5) * 2, 1)
        let cornerFade = 1 - smoothStep((cornerPosition - 0.72) / 0.28) * 0.30
        let coreCoverage: Float = finish == .satin ? 0.76 : 0.99
        return clamp01(maskAlpha * innerFade * cornerFade * coreCoverage)
    }

    private static func colorWithLuminance(_ color: RGBColor,
                                           luminance targetLuminance: Float) -> RGBColor {
        let delta = targetLuminance - luminance(color)
        return clippedLuminanceColor(
            RGBColor(
                red: color.red + delta,
                green: color.green + delta,
                blue: color.blue + delta
            ),
            luminance: targetLuminance
        )
    }

    private static func clippedLuminanceColor(_ color: RGBColor,
                                              luminance targetLuminance: Float) -> RGBColor {
        var red = color.red
        var green = color.green
        var blue = color.blue
        let minimum = min(red, min(green, blue))

        if minimum < 0 {
            let denominator = max(targetLuminance - minimum, 0.000_1)
            red = targetLuminance + (red - targetLuminance) * targetLuminance / denominator
            green = targetLuminance + (green - targetLuminance) * targetLuminance / denominator
            blue = targetLuminance + (blue - targetLuminance) * targetLuminance / denominator
        }
        let maximum = max(red, max(green, blue))
        if maximum > 1 {
            let denominator = max(maximum - targetLuminance, 0.000_1)
            red = targetLuminance + (red - targetLuminance) * (1 - targetLuminance) / denominator
            green = targetLuminance + (green - targetLuminance) * (1 - targetLuminance) / denominator
            blue = targetLuminance + (blue - targetLuminance) * (1 - targetLuminance) / denominator
        }
        return RGBColor(red: clamp01(red), green: clamp01(green), blue: clamp01(blue))
    }

    private static func semanticShapeCoverage(semanticAlpha: Float,
                                              mediaPipeAlpha: Float,
                                              outerDistance: CGFloat,
                                              innerDistance: CGFloat) -> Float {
        let semanticAlpha = clamp01(semanticAlpha)
        guard semanticAlpha < 0.995 else {
            return 1
        }

        let mediaPipeCore = smoothStep((mediaPipeAlpha - 0.56) / 0.34)
        let distanceToBoundary = Float(min(outerDistance, innerDistance))
        let distanceCore = smoothStep((distanceToBoundary - 0.014) / 0.050)
        let corePreservation = max(mediaPipeCore, distanceCore)

        // Semantic masks can flicker or miss small lip creases. Let the neural mask
        // trim uncertain edges, but keep the MediaPipe core painted.
        let coreFloor = corePreservation * 0.92
        return max(semanticAlpha, coreFloor)
    }

    private static func luminance(_ color: RGBColor) -> Float {
        clamp01(color.red * 0.299 + color.green * 0.587 + color.blue * 0.114)
    }

    private static func colorWithSaturation(_ color: RGBColor,
                                            strength: Float) -> RGBColor {
        let gray = luminance(color)
        return RGBColor(
            red: clamp01(gray + (color.red - gray) * strength),
            green: clamp01(gray + (color.green - gray) * strength),
            blue: clamp01(gray + (color.blue - gray) * strength)
        )
    }

    private static func cameraLipSample(atViewportPoint point: CGPoint,
                                        inverseImageTransform: CGAffineTransform,
                                        baseAddress: UnsafeMutableRawPointer,
                                        bytesPerRow: Int,
                                        width: Int,
                                        height: Int) -> CameraLipSample? {
        let imagePoint = point.applying(inverseImageTransform)
        guard let base = samplePixel(
            at: imagePoint,
            baseAddress: baseAddress,
            bytesPerRow: bytesPerRow,
            width: width,
            height: height
        ) else {
            return nil
        }

        let radius = CGFloat(max(2, min(width, height) / 220))
        let offsets = [
            CGPoint.zero,
            CGPoint(x: -radius, y: 0),
            CGPoint(x: radius, y: 0),
            CGPoint(x: 0, y: -radius),
            CGPoint(x: 0, y: radius),
            CGPoint(x: -radius * 0.7, y: -radius * 0.7),
            CGPoint(x: radius * 0.7, y: -radius * 0.7),
            CGPoint(x: -radius * 0.7, y: radius * 0.7),
            CGPoint(x: radius * 0.7, y: radius * 0.7)
        ]

        var red: Float = 0
        var green: Float = 0
        var blue: Float = 0
        var count: Float = 0
        for offset in offsets {
            let samplePoint = CGPoint(x: imagePoint.x + offset.x, y: imagePoint.y + offset.y)
            guard let sample = samplePixel(
                at: samplePoint,
                baseAddress: baseAddress,
                bytesPerRow: bytesPerRow,
                width: width,
                height: height
            ) else {
                continue
            }
            red += sample.red
            green += sample.green
            blue += sample.blue
            count += 1
        }

        guard count > 0 else {
            return CameraLipSample(base: base, blurred: base)
        }

        return CameraLipSample(
            base: base,
            blurred: RGBColor(red: red / count, green: green / count, blue: blue / count)
        )
    }

    private static func stableLipBaseColor(u: CGFloat, topV: CGFloat) -> RGBColor {
        let center = 1 - min(abs(Float(u) - 0.5) * 2, 1)
        let vertical = 1 - min(abs(Float(topV) - 0.52) * 1.75, 1)
        let softWarmth = stableWave(u: u, topV: topV, ux: 7.0, vy: 3.5, phase: 0.25)
        let fineCrease = stableWave(u: u, topV: topV, ux: 33.0, vy: 5.0, phase: 1.7)
        let tone = 0.92 + center * 0.08 + vertical * 0.05 + softWarmth * 0.035 - fineCrease * 0.025

        return RGBColor(
            red: clamp01(0.58 * tone + center * 0.035),
            green: clamp01(0.255 * tone + softWarmth * 0.020),
            blue: clamp01(0.305 * tone + vertical * 0.018)
        )
    }

    private static func pigmentBlendedColor(lipstick: RGBColor,
                                            variation: Float) -> RGBColor {
        // Preserve the preset RGB; this neutral variation supplies only a
        // small amount of stable surface texture.
        let texture = 0.982 + variation * 0.032

        let red = lipstick.red * texture
        let green = lipstick.green * texture
        let blue = lipstick.blue * texture

        return RGBColor(red: clamp01(red), green: clamp01(green), blue: clamp01(blue))
    }

    private static func finishAlpha(base: Float,
                                    u: CGFloat,
                                    topV: CGFloat,
                                    finish: LipFinish) -> Float {
        // Opacity is geometry-driven for satin. A highlight must come from the
        // user's camera image, not from a procedural alpha pattern.
        base
    }

    private static func finishColor(pigment: RGBColor,
                                    lipBase: RGBColor,
                                    u: CGFloat,
                                    topV: CGFloat,
                                    alpha: Float,
                                    finish: LipFinish) -> RGBColor {
        guard finish == .gloss else {
            return pigment
        }

        let wetSpecular = glossSpecular(u: u, topV: topV)
        let wetVeil = glossVeil(u: u, topV: topV)
        let isUpperHighlight = topV < 0.34
        let isLowerHighlight = topV > 0.56
        let highlightBoost: Float = isUpperHighlight ? 1.24 : (isLowerHighlight ? 1.22 : 1.06)
        let specular = min(wetSpecular * highlightBoost, 0.92)
        let veil = min(wetVeil * 0.038, 0.052)
        let minimumHighlightAlpha: Float = (isUpperHighlight || isLowerHighlight) ? 0.58 : 0.36
        let alphaGuard = max(minimumHighlightAlpha, min(alpha, 0.99))

        let glazedRed = clamp01(pigment.red * 1.05 + 0.020)
        let glazedGreen = clamp01(pigment.green * 1.02 + 0.010)
        let glazedBlue = clamp01(pigment.blue * 1.06 + 0.018)
        let red = pigment.red * (1 - veil) + glazedRed * veil
        let green = pigment.green * (1 - veil) + glazedGreen * veil
        let blue = pigment.blue * (1 - veil) + glazedBlue * veil
        let highlightRed: Float = 1
        let highlightGreen: Float = 1
        let highlightBlue: Float = 1

        let highlightBlend = min(specular * alphaGuard * ((isUpperHighlight || isLowerHighlight) ? 1.04 : 0.90), 1)

        return RGBColor(
            red: clamp01(red + (highlightRed - red) * highlightBlend),
            green: clamp01(green + (highlightGreen - green) * highlightBlend),
            blue: clamp01(blue + (highlightBlue - blue) * highlightBlend)
        )
    }

    private static func glossSpecular(u: CGFloat, topV: CGFloat) -> Float {
        let x = Float(u)
        let y = Float(topV)
        let lowerWetSurface = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.51,
            centerY: 0.66 + 0.018 * sin((x - 0.5) * .pi),
            radiusX: 0.30,
            radiusY: 0.060
        )
        let lowerHotBand = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.50,
            centerY: 0.695,
            radiusX: 0.155,
            radiusY: 0.020
        )
        let lowerGlossCore = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.50,
            centerY: 0.695,
            radiusX: 0.125,
            radiusY: 0.009,
            falloff: 1.90
        )
        let lowerGlossSoft = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.50,
            centerY: 0.690,
            radiusX: 0.205,
            radiusY: 0.044,
            falloff: 0.72
        )
        let lowerBrokenLineA = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.435,
            centerY: 0.690,
            radiusX: 0.050,
            radiusY: 0.007,
            falloff: 1.55
        )
        let lowerBrokenLineB = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.515,
            centerY: 0.700,
            radiusX: 0.070,
            radiusY: 0.008,
            falloff: 1.55
        )
        let lowerBrokenLineC = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.590,
            centerY: 0.688,
            radiusX: 0.042,
            radiusY: 0.007,
            falloff: 1.55
        )
        let lowerBrokenLineD = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.485,
            centerY: 0.718,
            radiusX: 0.044,
            radiusY: 0.006,
            falloff: 1.65
        )
        let lowerLeftWet = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.34,
            centerY: 0.64,
            radiusX: 0.13,
            radiusY: 0.054
        )
        let lowerRightWet = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.69,
            centerY: 0.63,
            radiusX: 0.13,
            radiusY: 0.050
        )
        let upperWetArc = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.50,
            centerY: 0.250 - 0.010 * sin((x - 0.5) * .pi),
            radiusX: 0.22,
            radiusY: 0.026,
            falloff: 1.40
        )
        let cupidLeftRidge = rotatedHighlight(
            x: x,
            y: y,
            centerX: 0.462,
            centerY: 0.214,
            radiusMajor: 0.090,
            radiusMinor: 0.007,
            angleRadians: 0.30,
            falloff: 1.85
        )
        let cupidRightRidge = rotatedHighlight(
            x: x,
            y: y,
            centerX: 0.538,
            centerY: 0.214,
            radiusMajor: 0.090,
            radiusMinor: 0.007,
            angleRadians: -0.30,
            falloff: 1.85
        )
        let cupidLeftSoft = rotatedHighlight(
            x: x,
            y: y,
            centerX: 0.462,
            centerY: 0.214,
            radiusMajor: 0.110,
            radiusMinor: 0.027,
            angleRadians: 0.30,
            falloff: 0.74
        )
        let cupidRightSoft = rotatedHighlight(
            x: x,
            y: y,
            centerX: 0.538,
            centerY: 0.214,
            radiusMajor: 0.110,
            radiusMinor: 0.027,
            angleRadians: -0.30,
            falloff: 0.74
        )
        let cupidCenterJoin = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.50,
            centerY: 0.224,
            radiusX: 0.044,
            radiusY: 0.012,
            falloff: 1.55
        )
        let tinyLowerSpark = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.46,
            centerY: 0.72,
            radiusX: 0.055,
            radiusY: 0.020
        )
        let tinyUpperSpark = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.52,
            centerY: 0.220,
            radiusX: 0.090,
            radiusY: 0.034,
            falloff: 1.15
        )
        let lowerStreaks = glossVerticalStreaks(
            x: x,
            y: y,
            centerY: 0.69,
            verticalRadius: 0.070,
            strength: 0.42
        )
        let upperStreaks = glossVerticalStreaks(
            x: x,
            y: y,
            centerY: 0.31,
            verticalRadius: 0.095,
            strength: 0.20
        )
        let innerLineSuppress = 1 - elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.50,
            centerY: 0.50,
            radiusX: 0.50,
            radiusY: 0.070
        ) * 0.68
        let verticalCreaseBreakup = 0.92 +
            stableWave(u: u, topV: topV, ux: 26.0, vy: 1.6, phase: 0.45) * 0.13 +
            stableWave(u: u, topV: topV, ux: 53.0, vy: 2.4, phase: 2.10) * 0.08
        let specular = lowerWetSurface * 0.16 +
            lowerHotBand * 0.22 +
            lowerGlossCore * 0.18 +
            lowerGlossSoft * 0.18 +
            lowerBrokenLineA * 0.34 +
            lowerBrokenLineB * 0.46 +
            lowerBrokenLineC * 0.30 +
            lowerBrokenLineD * 0.24 +
            lowerLeftWet * 0.06 +
            lowerRightWet * 0.06 +
            upperWetArc * 0.03 +
            cupidLeftRidge * 0.40 +
            cupidRightRidge * 0.40 +
            cupidLeftSoft * 0.18 +
            cupidRightSoft * 0.18 +
            cupidCenterJoin * 0.12 +
            lowerStreaks * 0.18 +
            upperStreaks * 0.10 +
            tinyLowerSpark * 0.10 +
            tinyUpperSpark * 0.0
        return min(max(specular * innerLineSuppress * verticalCreaseBreakup, 0), 1)
    }

    private static func glossVeil(u: CGFloat, topV: CGFloat) -> Float {
        let x = Float(u)
        let y = Float(topV)
        let center = max(0, 1 - abs(x - 0.5) / 0.48)
        let lowerVolume = max(0, 1 - abs(y - 0.66) / 0.30)
        let upperVolume = max(0, 1 - abs(y - 0.31) / 0.20)
        let innerLineSuppress = 1 - elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.50,
            centerY: 0.50,
            radiusX: 0.55,
            radiusY: 0.060
        ) * 0.44
        return smoothStep(center) * smoothStep(max(lowerVolume, upperVolume * 0.72)) * innerLineSuppress
    }

    private static func glossVerticalStreaks(x: Float,
                                             y: Float,
                                             centerY: Float,
                                             verticalRadius: Float,
                                             strength: Float) -> Float {
        let verticalEnvelope = max(0, 1 - abs(y - centerY) / max(verticalRadius, 0.001))
        let horizontalEnvelope = max(0, 1 - abs(x - 0.5) / 0.47)
        let stripeA = pow(max(0, sin((x * 56.0 + 0.20) * .pi)), 10)
        let stripeB = pow(max(0, sin((x * 83.0 + 1.35) * .pi)), 14)
        let stripeC = pow(max(0, sin((x * 122.0 + 2.10) * .pi)), 18)
        let broken = 0.70 + sin((x * 7.0 + y * 5.0 + 0.4) * .pi) * 0.18
        let streaks = min(stripeA * 0.62 + stripeB * 0.42 + stripeC * 0.30, 1)
        return streaks *
            pow(verticalEnvelope, 0.72) *
            smoothStep(horizontalEnvelope) *
            max(0.45, broken) *
            strength
    }

    private static func elongatedHighlight(x: Float,
                                           y: Float,
                                           centerX: Float,
                                           centerY: Float,
                                           radiusX: Float,
                                           radiusY: Float,
                                           falloff: Float = 1.20) -> Float {
        let dx = (x - centerX) / max(radiusX, 0.001)
        let dy = (y - centerY) / max(radiusY, 0.001)
        let distance = dx * dx + dy * dy
        return pow(max(0, 1 - distance), falloff)
    }

    private static func rotatedHighlight(x: Float,
                                         y: Float,
                                         centerX: Float,
                                         centerY: Float,
                                         radiusMajor: Float,
                                         radiusMinor: Float,
                                         angleRadians: Float,
                                         falloff: Float) -> Float {
        let dx = x - centerX
        let dy = y - centerY
        let c = cos(angleRadians)
        let s = sin(angleRadians)
        let along = (dx * c + dy * s) / max(radiusMajor, 0.001)
        let across = (-dx * s + dy * c) / max(radiusMinor, 0.001)
        let distance = along * along + across * across
        return pow(max(0, 1 - distance), falloff)
    }

    private static func stableLipPigmentVariation(u: CGFloat, topV: CGFloat) -> (alpha: Float, color: Float) {
        let broad = stableWave(u: u, topV: topV, ux: 5.0, vy: 4.5, phase: 0.8)
        let crease = stableWave(u: u, topV: topV, ux: 42.0, vy: 2.0, phase: 2.4)
        let verticalCrease = stableWave(u: u, topV: topV, ux: 68.0, vy: 0.6, phase: 0.1)
        let color = broad * 0.70 + crease * 0.20 + verticalCrease * 0.10
        let alpha = 0.990 + (broad - 0.5) * 0.010 + (verticalCrease - 0.5) * 0.006
        return (max(0.975, min(alpha, 1.008)), color)
    }

    private static func stableWave(u: CGFloat,
                                   topV: CGFloat,
                                   ux: Double,
                                   vy: Double,
                                   phase: Double) -> Float {
        let value = sin(Double(u) * ux + Double(topV) * vy + phase)
        return Float(value * 0.5 + 0.5)
    }

    private static func isToothLike(_ color: RGBColor) -> Bool {
        let maxChannel = max(color.red, max(color.green, color.blue))
        let minChannel = min(color.red, min(color.green, color.blue))
        let luminance = color.red * 0.299 + color.green * 0.587 + color.blue * 0.114
        let saturation = maxChannel > 0.001 ? (maxChannel - minChannel) / maxChannel : 0
        return luminance > 0.64 && saturation < 0.22
    }

    private static func makeAlphaMask(contour: LipContour,
                                      frame: CGRect,
                                      scale: CGFloat,
                                      pixelWidth: Int,
                                      pixelHeight: Int) -> [UInt8]? {
        var mask = [UInt8](repeating: 0, count: pixelWidth * pixelHeight)
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bytesPerRow = pixelWidth

        let didRender = mask.withUnsafeMutableBytes { rawBuffer -> Bool in
            guard let baseAddress = rawBuffer.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: pixelWidth,
                    height: pixelHeight,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: colorSpace,
                    bitmapInfo: CGImageAlphaInfo.none.rawValue
                  ) else {
                return false
            }

            context.setAllowsAntialiasing(true)
            context.setShouldAntialias(true)
            context.setFillColor(gray: 1, alpha: 1)
            context.addPath(Self.makeLocalPath(contour.outer, frame: frame, scale: scale))
            context.fillPath()
            context.setFillColor(gray: 0, alpha: 1)
            context.addPath(Self.makeLocalPath(contour.inner, frame: frame, scale: scale))
            context.fillPath()
            return true
        }

        return didRender ? mask : nil
    }

    private static func makeLocalPath(_ points: [CGPoint], frame: CGRect, scale: CGFloat) -> CGPath {
        let path = CGMutablePath()
        guard let first = points.first else {
            return path
        }

        func local(_ point: CGPoint) -> CGPoint {
            CGPoint(
                x: (point.x - frame.minX) * scale,
                y: (point.y - frame.minY) * scale
            )
        }

        path.move(to: local(first))
        for point in points.dropFirst() {
            path.addLine(to: local(point))
        }
        path.closeSubpath()
        return path
    }

    private static func edgeNoise(maskAlpha: Float, x: Int, y: Int, frame: CGRect) -> Float {
        guard maskAlpha < 0.96 else {
            return 1
        }

        let fineNoise = valueNoise(
            x: Int(frame.minX.rounded()) + x / 2,
            y: Int(frame.minY.rounded()) + y / 2
        )
        let coarseNoise = valueNoise(
            x: Int(frame.minX.rounded()) + x / 5 + 137,
            y: Int(frame.minY.rounded()) + y / 5 - 71
        )
        let edgeInfluence = pow(max(0, 1 - maskAlpha), 0.58)
        let breakup = 1 - edgeInfluence * (0.010 + fineNoise * 0.038)
        let microVariation = 1 + (coarseNoise - 0.5) * edgeInfluence * 0.024
        return max(0.92, min(1.018, breakup * microVariation))
    }

    private static func edgeSoftenedRGBA(_ rgba: [UInt8],
                                         pixelWidth: Int,
                                         pixelHeight: Int) -> [UInt8] {
        guard pixelWidth > 2, pixelHeight > 2 else {
            return rgba
        }

        var current = rgba
        let weights = [
            (dx: -1, dy: -1, weight: 1),
            (dx: 0, dy: -1, weight: 2),
            (dx: 1, dy: -1, weight: 1),
            (dx: -1, dy: 0, weight: 2),
            (dx: 0, dy: 0, weight: 4),
            (dx: 1, dy: 0, weight: 2),
            (dx: -1, dy: 1, weight: 1),
            (dx: 0, dy: 1, weight: 2),
            (dx: 1, dy: 1, weight: 1)
        ]

        for pass in 0..<2 {
            var output = current
            for y in 1..<(pixelHeight - 1) {
                for x in 1..<(pixelWidth - 1) {
                    let offset = (y * pixelWidth + x) * 4
                    let originalAlpha = current[offset + 3]
                    guard originalAlpha > 0, originalAlpha < 210 else {
                        continue
                    }

                    var red = 0
                    var green = 0
                    var blue = 0
                    var alpha = 0
                    var totalWeight = 0

                    for sample in weights {
                        let sampleOffset = ((y + sample.dy) * pixelWidth + (x + sample.dx)) * 4
                        red += Int(current[sampleOffset]) * sample.weight
                        green += Int(current[sampleOffset + 1]) * sample.weight
                        blue += Int(current[sampleOffset + 2]) * sample.weight
                        alpha += Int(current[sampleOffset + 3]) * sample.weight
                        totalWeight += sample.weight
                    }

                    let edgeBlend = Float(255 - originalAlpha) / 255
                    let normalizedX = Float(x) / Float(max(pixelWidth - 1, 1))
                    let cornerProximity = max(0, 1 - min(normalizedX, 1 - normalizedX) / 0.20)
                    let cornerBlend = smoothStep(cornerProximity)
                    let passBoost = Float(pass) * 0.045
                    let blurBlend = min(max(edgeBlend * 0.92 + cornerBlend * 0.24 + passBoost, 0), 0.74)
                    let blurredAlpha = UInt8(alpha / totalWeight)
                    output[offset] = blendChannel(original: current[offset], blurred: UInt8(red / totalWeight), amount: blurBlend)
                    output[offset + 1] = blendChannel(original: current[offset + 1], blurred: UInt8(green / totalWeight), amount: blurBlend)
                    output[offset + 2] = blendChannel(original: current[offset + 2], blurred: UInt8(blue / totalWeight), amount: blurBlend)
                    output[offset + 3] = blendChannel(original: originalAlpha, blurred: blurredAlpha, amount: blurBlend)
                }
            }
            current = output
        }

        return current
    }

    private static var usesStableProceduralLipstickRenderer: Bool {
        false
    }

    private func temporallyStabilizedSemanticAlphaMap(_ alphaMap: [Float],
                                                      activePixels: [Bool],
                                                      pixelWidth: Int,
                                                      pixelHeight: Int,
                                                      hasSemanticMask: Bool,
                                                      motionDelta: CGFloat) -> [Float] {
        temporalTextureLock.lock()
        defer {
            temporalTextureLock.unlock()
        }

        guard hasSemanticMask else {
            previousSemanticAlphaMap = nil
            previousSemanticAlphaWidth = 0
            previousSemanticAlphaHeight = 0
            return alphaMap
        }

        let shouldReset =
            previousSemanticAlphaMap == nil ||
            previousSemanticAlphaWidth != pixelWidth ||
            previousSemanticAlphaHeight != pixelHeight ||
            previousSemanticAlphaMap?.count != alphaMap.count

        guard !shouldReset,
              let previous = previousSemanticAlphaMap else {
            previousSemanticAlphaMap = alphaMap
            previousSemanticAlphaWidth = pixelWidth
            previousSemanticAlphaHeight = pixelHeight
            return alphaMap
        }

        var output = alphaMap
        let normalizedMotion = Float(max(0, min(motionDelta / 0.22, 1)))
        let growWeight = min(0.88, 0.54 + normalizedMotion * 0.28)
        let shrinkWeight = min(0.72, 0.22 + normalizedMotion * 0.34)

        for index in alphaMap.indices where activePixels[index] {
            let current = alphaMap[index]
            let previousValue = previous[index]
            let weight = current >= previousValue ? growWeight : shrinkWeight
            output[index] = Self.blendFloat(previous: previousValue, current: current, currentWeight: weight)
        }

        previousSemanticAlphaMap = output
        previousSemanticAlphaWidth = pixelWidth
        previousSemanticAlphaHeight = pixelHeight
        return output
    }

    private func temporallyStabilizedToneMap(_ toneMap: [MaterialTone],
                                             activePixels: [Bool],
                                             pixelWidth: Int,
                                             pixelHeight: Int,
                                             color: RGBColor,
                                             excludesInnerMouth: Bool,
                                             averageCameraLuma: Float?,
                                             lowLatency: Bool,
                                             motionDelta: CGFloat) -> [MaterialTone] {
        temporalTextureLock.lock()
        defer {
            temporalTextureLock.unlock()
        }

        let previousLuma = previousToneAverageLuma
        let exposureDelta = averageCameraLuma.flatMap { average in
            previousLuma.map { abs($0 - average) }
        } ?? 0
        let topologyChanged = previousToneMouthOpen.map { $0 != excludesInnerMouth } ?? false
        let shouldReset =
            previousToneMap == nil ||
            previousToneMapWidth != pixelWidth ||
            previousToneMapHeight != pixelHeight ||
            previousToneMap?.count != toneMap.count ||
            previousToneColor.map({ Self.colorDistance($0, color) >= 0.035 }) != false ||
            topologyChanged ||
            exposureDelta > 0.22

        if shouldReset {
            previousToneMap = toneMap
            previousToneMapWidth = pixelWidth
            previousToneMapHeight = pixelHeight
            previousToneColor = color
            previousToneMouthOpen = excludesInnerMouth
            previousToneAverageLuma = averageCameraLuma
            logToneMap(
                toneMap,
                activePixels: activePixels,
                averageCameraLuma: averageCameraLuma,
                motionDelta: motionDelta,
                reset: true
            )
            return toneMap
        }

        guard let previous = previousToneMap else {
            return toneMap
        }

        var output = toneMap
        let normalizedMotion = Float(max(0, min(motionDelta / 0.22, 1)))
        let baseCurrentWeight: Float = lowLatency ? 0.60 : 0.22
        let currentWeight = min(lowLatency ? 0.72 : 0.38, baseCurrentWeight + normalizedMotion * 0.12)
        for index in toneMap.indices {
            guard activePixels[index] else {
                continue
            }

            output[index] = MaterialTone(
                detail: Self.blendFloat(previous: previous[index].detail, current: toneMap[index].detail, currentWeight: currentWeight),
                shadow: Self.blendFloat(previous: previous[index].shadow, current: toneMap[index].shadow, currentWeight: currentWeight)
            )
        }

        previousToneMap = output
        previousToneMapWidth = pixelWidth
        previousToneMapHeight = pixelHeight
        previousToneColor = color
        previousToneMouthOpen = excludesInnerMouth
        previousToneAverageLuma = averageCameraLuma
        logToneMap(
            output,
            activePixels: activePixels,
            averageCameraLuma: averageCameraLuma,
            motionDelta: motionDelta,
            reset: false
        )
        return output
    }

    private func logToneMap(_ toneMap: [MaterialTone],
                            activePixels: [Bool],
                            averageCameraLuma: Float?,
                            motionDelta: CGFloat,
                            reset: Bool) {
        var minDetail: Float = 1
        var maxDetail: Float = 1
        var minShadow: Float = 1
        var maxShadow: Float = 1
        var count = 0
        for index in toneMap.indices where activePixels[index] {
            minDetail = min(minDetail, toneMap[index].detail)
            maxDetail = max(maxDetail, toneMap[index].detail)
            minShadow = min(minShadow, toneMap[index].shadow)
            maxShadow = max(maxShadow, toneMap[index].shadow)
            count += 1
        }
        LipDebugLog.throttled(
            "lip_material_tone",
            interval: 0.6,
            "lip_material tone pixels=\(count) detail=\(String(format: "%.3f", minDetail))...\(String(format: "%.3f", maxDetail)) shadow=\(String(format: "%.3f", minShadow))...\(String(format: "%.3f", maxShadow)) luma=\(String(format: "%.3f", averageCameraLuma ?? -1)) motion=\(String(format: "%.3f", motionDelta)) reset=\(reset)"
        )
    }

    private static func blendChannel(original: UInt8, blurred: UInt8, amount: Float) -> UInt8 {
        let mixed = Float(original) + (Float(blurred) - Float(original)) * amount
        return UInt8(max(0, min(mixed.rounded(), 255)))
    }

    private static func blendFloat(previous: Float, current: Float, currentWeight: Float) -> Float {
        let weight = max(0, min(currentWeight, 1))
        return previous + (current - previous) * weight
    }

    private static func colorDistance(_ first: RGBColor, _ second: RGBColor) -> Float {
        max(abs(first.red - second.red), max(abs(first.green - second.green), abs(first.blue - second.blue)))
    }

    private static func colorSignature(_ color: RGBColor) -> UInt32 {
        let red = UInt32(max(0, min(Int((color.red * 255).rounded()), 255)))
        let green = UInt32(max(0, min(Int((color.green * 255).rounded()), 255)))
        let blue = UInt32(max(0, min(Int((color.blue * 255).rounded()), 255)))
        return (red << 16) | (green << 8) | blue
    }

    private static func makeClosedPath(_ points: [CGPoint]) -> CGPath {
        let path = CGMutablePath()
        guard let first = points.first else {
            return path
        }

        path.move(to: first)
        for point in points.dropFirst() {
            path.addLine(to: point)
        }
        path.closeSubpath()
        return path
    }

    private static func isInsideLip(point: CGPoint, outerPath: CGPath, innerPath: CGPath) -> Bool {
        outerPath.contains(point, using: .winding, transform: .identity) &&
            !innerPath.contains(point, using: .winding, transform: .identity)
    }

    private static func distance(from point: CGPoint, toClosedPolyline points: [CGPoint]) -> CGFloat {
        guard points.count > 1 else {
            return .greatestFiniteMagnitude
        }

        var best = CGFloat.greatestFiniteMagnitude
        for index in 0..<points.count {
            let start = points[index]
            let end = points[(index + 1) % points.count]
            best = min(best, distance(from: point, toSegmentStart: start, end: end))
        }
        return best
    }

    private static func distance(from point: CGPoint, toSegmentStart start: CGPoint, end: CGPoint) -> CGFloat {
        let dx = end.x - start.x
        let dy = end.y - start.y
        let lengthSquared = dx * dx + dy * dy
        guard lengthSquared > 0 else {
            return hypot(point.x - start.x, point.y - start.y)
        }

        let t = max(0, min(((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared, 1))
        let projection = CGPoint(x: start.x + dx * t, y: start.y + dy * t)
        return hypot(point.x - projection.x, point.y - projection.y)
    }

    private static func samplePixel(at imagePoint: CGPoint,
                                    baseAddress: UnsafeMutableRawPointer,
                                    bytesPerRow: Int,
                                    width: Int,
                                    height: Int) -> RGBColor? {
        guard imagePoint.x.isFinite,
              imagePoint.y.isFinite,
              imagePoint.x >= 0,
              imagePoint.y >= 0,
              imagePoint.x < CGFloat(width),
              imagePoint.y < CGFloat(height) else {
            return nil
        }

        let x = min(max(Int(imagePoint.x.rounded(.down)), 0), width - 1)
        let y = min(max(Int(imagePoint.y.rounded(.down)), 0), height - 1)
        let pointer = baseAddress.advanced(by: y * bytesPerRow + x * 4).assumingMemoryBound(to: UInt8.self)
        return RGBColor(
            red: Float(pointer[2]) / 255,
            green: Float(pointer[1]) / 255,
            blue: Float(pointer[0]) / 255
        )
    }

    private static func aspectFillTransform(for imageSize: CGSize, in viewportSize: CGSize) -> CGAffineTransform {
        let scale = max(viewportSize.width / imageSize.width, viewportSize.height / imageSize.height)
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        let tx = (viewportSize.width - scaledWidth) * 0.5
        let ty = (viewportSize.height - scaledHeight) * 0.5
        return CGAffineTransform(translationX: tx, y: ty).scaledBy(x: scale, y: scale)
    }

    private static func softLight(base: Float, blend: Float) -> Float {
        if blend < 0.5 {
            return base - (1 - 2 * blend) * base * (1 - base)
        }
        let d = base < 0.25 ? ((16 * base - 12) * base + 4) * base : sqrt(base)
        return base + (2 * blend - 1) * (d - base)
    }

    private static func smoothStep(_ value: CGFloat) -> Float {
        smoothStep(Float(value))
    }

    private static func smoothStep(_ value: Float) -> Float {
        let x = max(0, min(value, 1))
        return x * x * (3 - 2 * x)
    }

    private static func clamp01(_ value: Float) -> Float {
        max(0, min(value, 1))
    }

    private static func valueNoise(x: Int, y: Int) -> Float {
        var value = UInt32(truncatingIfNeeded: x) &* 374_761_393
        value = value &+ UInt32(truncatingIfNeeded: y) &* 668_265_263
        value = (value ^ (value >> 13)) &* 1_274_126_177
        return Float(value & 0xffff) / Float(0xffff)
    }

    private static func uint8(_ value: Float) -> UInt8 {
        UInt8(max(0, min(value, 1)) * 255)
    }
}

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
    private static let maxMotionCompensationAge: CFTimeInterval = 0.42
    private static let maxRealContourDisplayAge: CFTimeInterval = 0.45
    private static let maxSurfaceCarrierHoldAge: CFTimeInterval = 0.50
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
        sceneView?.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopSession() {
        setSessionActive(false)
        resetLipTrackingAsync()
        requestRendererClear()
        setTrackedFaceAnchor(nil, isTracked: false)
        DispatchQueue.main.async { self.isFaceDetected = false }
    }

    func updateLipstick(color: UIColor, opacity: Double, finish: LipFinish) {
        rendererStateLock.lock()
        pendingLipOpacity = opacity
        rendererStateLock.unlock()
        guard lipTextureRenderer.updateStyle(color: color, finish: finish) else {
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
                    motionDelta: meshState.motionDelta
                )
            } else {
                let availability = currentLipMeshAvailability()
                LipDebugLog.throttled(
                    "lip_scene_skip",
                    interval: 0.6,
                    "lip_scene skip tracked=\(faceAnchor.isTracked) mouthFrame=\(mouthFrame != nil) meshState=\(meshState != nil) hasContour=\(availability.contour) hasTexture=\(availability.texture)"
                )
                lipMeshRenderer.clearLip()
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

        let sourceAgeBeforePreprocessing = CACurrentMediaTime() - context.frameTimestamp
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

        let sourceAgeAfterSurfaceProjection = CACurrentMediaTime() - context.frameTimestamp
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
            "lip_sync submit timestamp=\(context.timestampInMilliseconds) captureToReceiptMs=\(String(format: "%.1f", (context.receivedAt - context.frameTimestamp) * 1000)) captureToSubmitMs=\(String(format: "%.1f", (submittedAt - context.frameTimestamp) * 1000)) input=\(Int(imageSize.width))x\(Int(imageSize.height)) surfaceTriangles=\(faceSurfaceSnapshot.triangles.count) indexed=\(preferredSurfaceTriangles != nil) epoch=\(context.trackingEpoch) viewport=\(context.viewportRevision)"
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
            current: detection.contour
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
        var stableContour: LipContour
        if let previous = smoothedLipContour,
           hasCompleteSurfaceCarrier(previous),
           previous.surfaceVertexCount == frame.faceSurfaceSnapshot.vertexCount,
           previous.surfaceTriangleIndexCount == frame.faceSurfaceSnapshot.triangleIndexCount,
           previous.surfaceTopologySignature == frame.faceSurfaceSnapshot.topologySignature,
           let refreshedBindings = refreshedDepthBindings(
               from: previous,
               for: detection.contour,
               snapshot: frame.faceSurfaceSnapshot
           ) {
            // Keep the same stable AR triangles whenever their current
            // reprojection remains valid. The indexed lookup makes this O(80)
            // instead of searching hundreds of triangles for every landmark.
            var fastBoundContour = smoothedContour
            fastBoundContour.surfaceBindingsByIndex = refreshedBindings
            fastBoundContour.surfaceVertexCount = previous.surfaceVertexCount
            fastBoundContour.surfaceTriangleIndexCount = previous.surfaceTriangleIndexCount
            fastBoundContour.surfaceTopologySignature = previous.surfaceTopologySignature
            fastBoundContour.surfaceCarrierCreatedAt = frame.frameTimestamp
            fastBoundContour.surfaceCarrierSupportsDeformation = true
            stableContour = fastBoundContour
            LipDebugLog.throttled(
                "lip_surface_reuse_depth_carrier",
                interval: 0.4,
                "lip_surface reuse carrier=indexed_delta_warp shape=mediapipe timestamp=\(timestampInMilliseconds)"
            )
        } else if let currentDepthContour = contourBoundToFaceSurface(
            detection.contour,
            publishing: smoothedContour,
            snapshot: frame.faceSurfaceSnapshot,
            carrierCreatedAt: frame.frameTimestamp
        ) {
            // Initial acquisition and large expression changes still use the
            // exhaustive search so the fast path never preserves a bad match.
            stableContour = currentDepthContour
        } else if let previous = smoothedLipContour,
                  hasCompleteSurfaceCarrier(previous),
                  let previousCarrierCreatedAt = previous.surfaceCarrierCreatedAt,
                  CACurrentMediaTime() - previousCarrierCreatedAt <=
                    Self.maxSurfaceCarrierHoldAge {
            // Publishing fresh MediaPipe X/Y must not wait for an exhaustive
            // carrier rebuild. The old barycentric topology still samples the
            // current ARFaceGeometry for depth; a later frame performs the full
            // rebind once the short indexed-snapshot window expires.
            var freshContourWithHeldDepth = smoothedContour
            freshContourWithHeldDepth.surfaceBindingsByIndex = previous.surfaceBindingsByIndex
            freshContourWithHeldDepth.surfaceVertexCount = previous.surfaceVertexCount
            freshContourWithHeldDepth.surfaceTriangleIndexCount = previous.surfaceTriangleIndexCount
            freshContourWithHeldDepth.surfaceTopologySignature = previous.surfaceTopologySignature
            freshContourWithHeldDepth.surfaceCarrierCreatedAt = previousCarrierCreatedAt
            freshContourWithHeldDepth.surfaceCarrierSupportsDeformation = false
            stableContour = freshContourWithHeldDepth
            LipDebugLog.throttled(
                "lip_surface_fast_publish",
                interval: 0.4,
                "lip_surface publish=fresh_mediapipe carrier=held_depth timestamp=\(timestampInMilliseconds)"
            )
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

        stableContour.renderedInnerOpeningRatio =
            frame.motionReference.isConfidentlyClosedMouth ?
                0 : stableContour.innerOpeningRatio
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
            "lip_detection accept lowLatency=\(lowLatencyShapeChange) captureToAcceptMs=\(String(format: "%.1f", (detectionAcceptedAt - frame.frameTimestamp) * 1000)) submitToAcceptMs=\(String(format: "%.1f", (detectionAcceptedAt - frame.submittedAt) * 1000)) \(lipContourDebugSummary(stableContour))"
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
           let previousCarrierCreatedAt = previous.surfaceCarrierCreatedAt,
           CACurrentMediaTime() - previousCarrierCreatedAt <= Self.maxSurfaceCarrierHoldAge,
           previous.surfaceVertexCount == snapshot.vertexCount,
           previous.surfaceTriangleIndexCount == snapshot.triangleIndexCount,
           previous.surfaceTopologySignature == snapshot.topologySignature {
            previousBindings = previous.surfaceBindingsByIndex
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
        // Head/camera motion is already removed by the frozen AR reference.
        // A larger residual is therefore a detector outlier, not real rigid
        // motion. Holding the previous real contour is safer than teleporting.
        if centerDistance > 0.22 || viewportDrift > 0.06 {
            LipDebugLog.throttled(
                "lip_detection_pose_innovation",
                interval: 0.5,
                "lip_detection hold reason=pose_innovation center=\(debugFloat(centerDistance)) widthRatio=\(debugFloat(widthRatio)) angle=\(debugFloat(angleDelta)) viewport=\(debugFloat(viewportDrift))"
            )
            return false
        }
        if angleDelta > 0.25 {
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
        let maximumMeanShift: CGFloat = isNearClosed ? 0.14 : 0.12
        let maximumMedian: CGFloat = isNearClosed ? 0.15 : 0.10
        let maximumP95: CGFloat = isNearClosed ? 0.27 : 0.22
        let maximumLargePointCount = isNearClosed ? 10 : 8
        let isReliable = meanShift <= maximumMeanShift &&
            median <= maximumMedian &&
            p95 <= maximumP95 &&
            largePointCount <= maximumLargePointCount
        if !isReliable {
            LipDebugLog.throttled(
                "lip_detection_shape_innovation",
                interval: 0.5,
                "lip_detection hold reason=shape_innovation mean=\(debugFloat(meanShift)) median=\(debugFloat(median)) p95=\(debugFloat(p95)) large=\(largePointCount)"
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
        // LipContour.transformed applies the same bounds. Keeping a single
        // effective scale range makes the requested and rendered poses agree.
        let scale = min(max(target.width / max(source.width, 1), 0.65), 1.45)
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
        if let retainedContour,
           hasCompleteSurfaceCarrier(retainedContour),
           retainedAge.isFinite,
           retainedAge >= 0,
           retainedAge <= Self.maxSurfaceCarrierHoldAge,
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
            semanticMask: nil,
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
        let requestStillAligned = requestMotionDelta < 0.30
        if let texture,
           isTracked,
           hasFreshShape,
           availability.contour,
           generationOK,
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

        if shouldUseRawContour(previous: previous, current: contour) {
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

    private func shouldUseRawContour(previous: LipContour?, current: LipContour) -> Bool {
        guard let previous,
              let previousBounds = previous.bounds,
              let currentBounds = current.bounds,
              previousBounds.width > 1,
              previousBounds.height > 1 else {
            return true
        }

        let widthChange = abs(currentBounds.width / previousBounds.width - 1)
        let heightChange = abs(currentBounds.height / previousBounds.height - 1)
        let openingChange = abs(mouthOpeningRatio(current) - mouthOpeningRatio(previous))
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
                alpha: 0.82,
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
        let medianActivity = smoothActivity(
            medianResidual,
            quiet: 0.25,
            active: 1.35
        )
        let p95Activity = smoothActivity(
            p95Residual,
            quiet: 0.65,
            active: 3.00
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
                0.20 + activity * 0.66 +
                    (1 - contourStabilityConfidence) * 0.08,
                0.20
            ),
            0.92
        )
        let maximumDeadZone = min(max(currentPose.width * 0.0028, 0.22), 0.38)
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
        let currentTriangleIndices = input.geometry.triangleIndices
        guard carrierAge.isFinite,
              carrierAge >= 0,
              carrierAge <= Self.maxIndexedSurfaceSnapshotAge,
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

        return LipMotionSample(
            pose: pose,
            sampledAt: renderedAt,
            frameTimestamp: renderedAt,
            trackingEpoch: trackingEpoch,
            anchorIdentifier: faceAnchor.identifier,
            viewportRevision: viewport.revision
        )
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
        let contourAge = CACurrentMediaTime() - contourCaptureTime
        guard contourAge.isFinite,
              contourAge >= 0,
              contourAge <= Self.maxRealContourDisplayAge else {
            LipDebugLog.throttled(
                "lip_mesh_real_contour_stale",
                interval: 0.4,
                "lip_mesh hide reason=stale_mediapipe_contour age=\(String(format: "%.3f", contourAge)) max=\(String(format: "%.3f", Self.maxRealContourDisplayAge))"
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
            contour: temporallyCompensatedContour,
            texture: texture,
            contourAge: contourAge,
            motionDelta: motionDelta
        )
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
        var residualMagnitudes: [CGFloat] = []
        localResiduals.reserveCapacity(Self.attentionLipIndices.count)
        residualMagnitudes.reserveCapacity(Self.attentionLipIndices.count)

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
            residualMagnitudes.append(magnitude)
        }

        guard residualMagnitudes.count == Self.attentionLipIndices.count else {
            return nil
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
        var residualMagnitudes: [CGFloat] = []
        residualMagnitudes.reserveCapacity(Self.attentionLipIndices.count)
        var largeResidualCount = 0

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
            residualMagnitudes.append(magnitude)
            if magnitude > 0.20 {
                largeResidualCount += 1
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
                min(predictionAge, 0.045) / sampleInterval,
                1.0
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
            2.0
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
