import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import SceneKit
import simd
import SwiftUI
import UIKit

private enum LipDebugLog {
    private static let lock = NSLock()
    private static var lastLoggedAt: [String: CFTimeInterval] = [:]

    static func throttled(_ key: String,
                          interval: CFTimeInterval = 0.5,
                          _ message: @autoclosure () -> String) {
        #if DEBUG
        let now = CACurrentMediaTime()
        lock.lock()
        if let last = lastLoggedAt[key], now - last < interval {
            lock.unlock()
            return
        }
        lastLoggedAt[key] = now
        lock.unlock()
        print(message())
        #endif
    }
}

struct FaceTrackingView: UIViewRepresentable {
    @Binding var isFaceDetected: Bool
    var lipColor: UIColor

    func makeUIView(context: Context) -> ARSCNView {
        let view = ViewportTrackingARSCNView(frame: .zero)
        view.delegate = context.coordinator
        view.session.delegate = context.coordinator
        view.automaticallyUpdatesLighting = true
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .black
        view.scene = SCNScene()
        view.onLayout = { [weak coordinator = context.coordinator] size, scale in
            coordinator?.updateViewport(size: size, scale: scale)
        }

        context.coordinator.sceneView = view
        context.coordinator.updateColors(lipColor)
        context.coordinator.updateViewport(size: view.bounds.size, scale: view.traitCollection.displayScale)
        context.coordinator.startSession()

        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {
        context.coordinator.updateColors(lipColor)
        context.coordinator.updateViewport(size: view.bounds.size, scale: view.window?.screen.scale ?? view.traitCollection.displayScale)
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(isFaceDetected: $isFaceDetected)
    }

    static func dismantleUIView(_ view: ARSCNView, coordinator: Coordinator) {
        coordinator.stopSession()
        view.session.pause()
    }

}

private final class ViewportTrackingARSCNView: ARSCNView {
    var onLayout: ((CGSize, CGFloat) -> Void)?

    override func layoutSubviews() {
        super.layoutSubviews()
        onLayout?(bounds.size, window?.screen.scale ?? traitCollection.displayScale)
    }
}

private struct LipMeshPoint {
    let screen: CGPoint
    let normalized: SIMD3<Float>
    let uv: CGPoint
}

private struct LipContour {
    var outer: [CGPoint]
    var inner: [CGPoint]
    var outer3D: [SIMD3<Float>] = []
    var inner3D: [SIMD3<Float>] = []
    var outerUV: [CGPoint] = []
    var innerUV: [CGPoint] = []
    var meshPointsByIndex: [Int: LipMeshPoint] = [:]
    var faceGeometryPose: FaceGeometryPose? = nil

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

    func isUsable(in viewportSize: CGSize) -> Bool {
        guard let bounds,
              let pose,
              viewportSize.width > 1,
              viewportSize.height > 1 else {
            return false
        }

        let minLipWidth = max(viewportSize.width * 0.08, 28)
        let maxLipWidth = viewportSize.width * 0.72
        let minLipHeight = max(viewportSize.height * 0.012, 8)
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
            faceGeometryPose: faceGeometryPose
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

private struct LipMotionPose {
    let center: CGPoint
    let width: CGFloat
    let angle: CGFloat
    let opening: CGFloat

    init?(contourPose: LipPose?) {
        guard let contourPose else {
            return nil
        }
        center = contourPose.center
        width = contourPose.width
        angle = contourPose.angle
        opening = 0
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
            verticalAlignment: verticalAlignment
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
    private static let cornerLipIndexSet = Set([61, 291, 78, 308, 76, 306, 62, 292])
    private static let upperOuterLipIndexSet = Set([409, 270, 269, 267, 0, 37, 39, 40, 185])
    private static let upperSupportLipIndexSet = Set([408, 304, 303, 302, 11, 72, 73, 74, 184])
    private static let upperInnerSupportLipIndexSet = Set([183, 42, 41, 38, 12, 268, 271, 272, 407])
    private static let lowerOuterLipIndexSet = Set([146, 91, 181, 84, 17, 314, 405, 321, 375])
    private static let lowerSupportLipIndexSet = Set([77, 90, 180, 85, 16, 315, 404, 320, 307])
    private static let lowerInnerSupportLipIndexSet = Set([96, 89, 179, 86, 15, 316, 403, 319, 325])

    static func isOuterLipIndex(_ index: Int) -> Bool {
        outerLipIndexSet.contains(index)
    }

    static func upperLipSmileLiftFactor(for index: Int) -> Float {
        if upperOuterLipIndexSet.contains(index) {
            return 1.0
        }
        if upperSupportLipIndexSet.contains(index) {
            return 0.82
        }
        if upperInnerSupportLipIndexSet.contains(index) {
            return 0.72
        }
        if index == 13 || index == 12 || index == 11 {
            return 0.68
        }
        return 0
    }

    static func meshExpansionScale(for index: Int) -> CGFloat {
        if cornerLipIndexSet.contains(index) {
            return 1.018
        }
        if upperOuterLipIndexSet.contains(index) {
            return 1.100
        }
        if upperSupportLipIndexSet.contains(index) {
            return 1.075
        }
        if upperInnerSupportLipIndexSet.contains(index) {
            return 1.145
        }
        if lowerOuterLipIndexSet.contains(index) {
            return 1.20
        }
        if lowerSupportLipIndexSet.contains(index) {
            return 1.115
        }
        if lowerInnerSupportLipIndexSet.contains(index) {
            return 1.075
        }
        if outerLipIndexSet.contains(index) {
            return 1.052
        }
        return 1
    }

    static let lipMeshTriangles: [(Int, Int, Int)] = [
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

    static func normalizedUV(for index: Int) -> CGPoint? {
        guard let uv = uvByIndex[index] else {
            return nil
        }

        return CGPoint(
            x: (uv.x - uvBounds.minX) / uvBounds.width,
            y: (uv.y - uvBounds.minY) / uvBounds.height
        )
    }

    static func canonicalVertex(for index: Int) -> SIMD3<Float>? {
        canonicalVertexByIndex[index]
    }
}

fileprivate final class LipMeshRenderer {
    private struct MeshVertex {
        let index: Int
        let screen: CGPoint
        let normalized: SIMD3<Float>
        let uv: CGPoint
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
    private var lastCorrectionX: Float = 0
    private var lastCorrectionY: Float = 0

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
        lipMaterial.readsFromDepthBuffer = false
        lipMaterial.diffuse.magnificationFilter = .linear
        lipMaterial.diffuse.minificationFilter = .linear
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
                mouthFrame: FaceLocalMouthFrame,
                renderer: SCNSceneRenderer,
                faceNode: SCNNode,
                contourAge: CFTimeInterval,
                motionDelta: CGFloat) {
        let correctedMouthFrame = mouthFrame.translated(
            by: projectionAlignmentCorrection(
                contour: contour,
                mouthFrame: mouthFrame,
                renderer: renderer,
                faceNode: faceNode,
                contourAge: contourAge,
                motionDelta: motionDelta
            )
        )

        guard let geometry = makeGeometry(contour: contour, texture: texture, mouthFrame: correctedMouthFrame) else {
            clearLip()
            return
        }

        if lastTextureImage !== texture.image {
            lipMaterial.diffuse.contents = texture.image
            lastTextureImage = texture.image
        }
        geometry.materials = [lipMaterial]
        lipNode.geometry = geometry
        lipNode.isHidden = false
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
                              mouthFrame: FaceLocalMouthFrame) -> SCNGeometry? {
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

        let mesh = makeCanonicalLipMesh(contour: contour, pose: pose)
        guard mesh.vertices.count >= 4, !mesh.indices.isEmpty else {
            LipDebugLog.throttled(
                "lip_mesh_empty_geometry",
                "lip_mesh makeGeometry=nil reason=empty_mesh vertices=\(mesh.vertices.count) indices=\(mesh.indices.count)"
            )
            return nil
        }

        let textureWidth = texture.image.cgImage?.width ?? 0
        let textureHeight = texture.image.cgImage?.height ?? 0
        LipDebugLog.throttled(
            "lip_mesh_geometry",
            interval: 0.75,
            "lip_mesh geometry vertices=\(mesh.vertices.count) triangles=\(mesh.indices.count / 3) texture=\(textureWidth)x\(textureHeight) mouthWidth=\(String(format: "%.4f", mouthFrame.width)) poseWidth=\(String(format: "%.1f", pose.width))"
        )

        let vertices = mesh.vertices.map {
            scenePoint(for: $0, pose: pose, contour: contour, mouthFrame: mouthFrame)
        }
        let textureCoordinates = mesh.vertices.map {
            CGPoint(x: $0.uv.x, y: 1 - $0.uv.y)
        }
        let indices = mesh.indices
        let vertexSource = SCNGeometrySource(vertices: vertices)
        let textureSource = SCNGeometrySource(textureCoordinates: textureCoordinates)
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        return SCNGeometry(sources: [vertexSource, textureSource], elements: [element])
    }

    private func updateDebugLines(contour: LipContour, mouthFrame: FaceLocalMouthFrame) {
        guard debugLinesEnabled,
              let pose = contour.pose else {
            clearDebugLines()
            return
        }

        let mesh = makeCanonicalLipMesh(contour: contour, pose: pose)
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

    private func projectionAlignmentCorrection(contour: LipContour,
                                               mouthFrame: FaceLocalMouthFrame,
                                               renderer: SCNSceneRenderer,
                                               faceNode: SCNNode,
                                               contourAge: CFTimeInterval,
                                               motionDelta: CGFloat) -> SIMD3<Float> {
        guard contourAge < 0.055,
              motionDelta < 0.032 else {
            return mouthFrame.xAxis * lastCorrectionX + mouthFrame.downAxis * lastCorrectionY
        }

        guard let pose = contour.pose else {
            return mouthFrame.xAxis * lastCorrectionX + mouthFrame.downAxis * lastCorrectionY
        }

        let mesh = makeCanonicalLipMesh(contour: contour, pose: pose)
        guard !mesh.vertices.isEmpty else {
            return mouthFrame.xAxis * lastCorrectionX + mouthFrame.downAxis * lastCorrectionY
        }

        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        var count: CGFloat = 0

        for vertex in mesh.vertices {
            let local = scenePoint(for: vertex, pose: pose, contour: contour, mouthFrame: mouthFrame)
            let world = faceNode.convertPosition(local, to: nil)
            let projected = renderer.projectPoint(world)
            let screen = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
            guard screen.x.isFinite,
                  screen.y.isFinite,
                  projected.z.isFinite else {
                continue
            }

            sumX += screen.x - vertex.screen.x
            sumY += screen.y - vertex.screen.y
            count += 1
        }

        guard count > 0 else {
            return mouthFrame.xAxis * lastCorrectionX + mouthFrame.downAxis * lastCorrectionY
        }

        let avgX = sumX / count
        let avgY = sumY / count
        let localUnitsPerPixel = mouthFrame.width / Float(max(pose.width, 1))
        let maxAxisCorrection = mouthFrame.width * 0.16
        let correctionGain: Float = 0.82
        let correctionX = Self.clamped(
            -Float(avgX) * localUnitsPerPixel * correctionGain,
            minValue: -maxAxisCorrection,
            maxValue: maxAxisCorrection
        )
        let correctionY = Self.clamped(
            -Float(avgY) * localUnitsPerPixel * correctionGain,
            minValue: -maxAxisCorrection,
            maxValue: maxAxisCorrection
        )
        lastCorrectionX = lastCorrectionX * 0.38 + correctionX * 0.62
        lastCorrectionY = lastCorrectionY * 0.38 + correctionY * 0.62

        LipDebugLog.throttled(
            "lip_projection_correction",
            interval: 0.22,
            "lip_projection correction age:\(Self.fmt(CGFloat(contourAge))) motion:\(Self.fmt(motionDelta)) avgX:\(Self.fmt(avgX)) avgY:\(Self.fmt(avgY)) localPerPx:\(String(format: "%.6f", Double(localUnitsPerPixel))) corrX:\(String(format: "%.5f", Double(lastCorrectionX))) corrY:\(String(format: "%.5f", Double(lastCorrectionY)))"
        )

        return mouthFrame.xAxis * lastCorrectionX + mouthFrame.downAxis * lastCorrectionY
    }

    private func logProjectionDiagnostics(contour: LipContour,
                                          mouthFrame: FaceLocalMouthFrame,
                                          renderer: SCNSceneRenderer,
                                          faceNode: SCNNode) {
        guard let pose = contour.pose else {
            return
        }

        let mesh = makeCanonicalLipMesh(contour: contour, pose: pose)
        guard !mesh.vertices.isEmpty else {
            return
        }

        var deltasByIndex: [Int: CGPoint] = [:]
        deltasByIndex.reserveCapacity(mesh.vertices.count)
        var allProjected = [CGPoint]()
        allProjected.reserveCapacity(mesh.vertices.count)

        for vertex in mesh.vertices {
            let local = scenePoint(for: vertex, pose: pose, contour: contour, mouthFrame: mouthFrame)
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

    private func boundsText(_ bounds: CGRect?) -> String {
        guard let bounds else {
            return "nil"
        }

        return "x:\(Self.fmt(bounds.minX)) y:\(Self.fmt(bounds.minY)) w:\(Self.fmt(bounds.width)) h:\(Self.fmt(bounds.height))"
    }

    private static func fmt(_ value: CGFloat) -> String {
        String(format: "%.2f", Double(value))
    }

    private static func clamped(_ value: Float, minValue: Float, maxValue: Float) -> Float {
        min(max(value, minValue), maxValue)
    }

    private func makeCanonicalLipMesh(contour: LipContour, pose: LipPose) -> (vertices: [MeshVertex], indices: [Int32]) {
        var vertices: [MeshVertex] = []
        var indices: [Int32] = []
        var vertexIndexByLandmark: [Int: Int32] = [:]

        func expandedPoint(_ point: CGPoint, landmarkIndex: Int) -> CGPoint {
            let scale = CanonicalLipGeometry.meshExpansionScale(for: landmarkIndex)
            return CGPoint(
                x: pose.center.x + (point.x - pose.center.x) * scale,
                y: pose.center.y + (point.y - pose.center.y) * scale
            )
        }

        func meshVertex(for landmarkIndex: Int) -> MeshVertex? {
            if let point = contour.meshPointsByIndex[landmarkIndex] {
                return MeshVertex(
                    index: landmarkIndex,
                    screen: expandedPoint(point.screen, landmarkIndex: landmarkIndex),
                    normalized: point.normalized,
                    uv: point.uv
                )
            }

            if let outerOffset = CanonicalLipGeometry.outerLipIndices.firstIndex(of: landmarkIndex),
               contour.outer.indices.contains(outerOffset),
               contour.outer3D.indices.contains(outerOffset),
               contour.outerUV.indices.contains(outerOffset) {
                return MeshVertex(
                    index: landmarkIndex,
                    screen: expandedPoint(contour.outer[outerOffset], landmarkIndex: landmarkIndex),
                    normalized: contour.outer3D[outerOffset],
                    uv: contour.outerUV[outerOffset]
                )
            }

            if let innerOffset = CanonicalLipGeometry.innerLipIndices.firstIndex(of: landmarkIndex),
               contour.inner.indices.contains(innerOffset),
               contour.inner3D.indices.contains(innerOffset),
               contour.innerUV.indices.contains(innerOffset) {
                return MeshVertex(
                    index: landmarkIndex,
                    screen: contour.inner[innerOffset],
                    normalized: contour.inner3D[innerOffset],
                    uv: contour.innerUV[innerOffset]
                )
            }

            return nil
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

        for triangle in CanonicalLipGeometry.lipMeshTriangles {
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
        }

        if !indices.isEmpty {
            return (vertices, indices)
        }

        func appendFallbackStrip(outerPositions: [Int],
                                 innerPositions: [Int]) {
            guard outerPositions.count == innerPositions.count,
                  outerPositions.count >= 2 else {
                return
            }

            let base = Int32(vertices.count)
            for offset in 0..<outerPositions.count {
                let outerIndex = outerPositions[offset]
                let innerIndex = innerPositions[offset]
                vertices.append(
                    MeshVertex(
                        index: CanonicalLipGeometry.outerLipIndices[outerIndex],
                        screen: expandedPoint(
                            contour.outer[outerIndex],
                            landmarkIndex: CanonicalLipGeometry.outerLipIndices[outerIndex]
                        ),
                        normalized: contour.outer3D[outerIndex],
                        uv: contour.outerUV[outerIndex]
                    )
                )
                vertices.append(
                    MeshVertex(
                        index: CanonicalLipGeometry.innerLipIndices[innerIndex],
                        screen: contour.inner[innerIndex],
                        normalized: contour.inner3D[innerIndex],
                        uv: contour.innerUV[innerIndex]
                    )
                )
            }

            for offset in 0..<(outerPositions.count - 1) {
                let a = base + Int32(offset * 2)
                let b = base + Int32((offset + 1) * 2)
                let c = a + 1
                let d = b + 1
                indices.append(contentsOf: [a, b, c, b, d, c])
            }
        }

        appendFallbackStrip(
            outerPositions: Array(0...10),
            innerPositions: Array(0...10)
        )
        appendFallbackStrip(
            outerPositions: [0] + Array(stride(from: 19, through: 10, by: -1)),
            innerPositions: [0] + Array(stride(from: 19, through: 10, by: -1))
        )

        return (vertices, indices)
    }

    private func scenePoint(for vertex: MeshVertex,
                            pose: LipPose,
                            contour: LipContour,
                            mouthFrame: FaceLocalMouthFrame) -> SCNVector3 {
        let cosine = cos(pose.angle)
        let sine = sin(pose.angle)
        let dx = Float(vertex.screen.x - pose.center.x)
        let dy = Float(vertex.screen.y - pose.center.y)
        let normalizedX = (dx * Float(cosine) + dy * Float(sine)) / Float(max(pose.width, 1))
        let normalizedY = (-dx * Float(sine) + dy * Float(cosine)) / Float(max(pose.width, 1))
        let depthDelta = max(min(vertex.normalized.z - contour.depthCenter, 0.08), -0.08)
        let canonicalDepthDelta = max(
            min(contour.faceGeometryPose?.relativeCanonicalDepth(for: vertex.index) ?? 0, 0.08),
            -0.08
        )
        let geometryScale = contour.faceGeometryPose.map { min(max($0.scale / 8, 0.45), 1.65) } ?? 1
        let surfaceOffset = max(mouthFrame.width * 0.04, 0.003)
        let depthOffset = surfaceOffset -
            depthDelta * mouthFrame.width * 0.45 * geometryScale -
            canonicalDepthDelta * mouthFrame.width * 0.22
        let smileLift = CanonicalLipGeometry.upperLipSmileLiftFactor(for: vertex.index) *
            min(max(mouthFrame.smileExpansion, 0), 1) *
            mouthFrame.width * 0.016

        let local = mouthFrame.center +
            mouthFrame.xAxis * (normalizedX * mouthFrame.width) +
            mouthFrame.downAxis * (normalizedY * mouthFrame.width - smileLift) +
            mouthFrame.normalAxis * depthOffset

        return SCNVector3(local.x, local.y, local.z)
    }

    private static let lipPointCount = 20
}

private struct LipTexture {
    let image: UIImage
}

private struct RGBColor {
    let red: Float
    let green: Float
    let blue: Float
}

private final class LipTextureRenderer {
    private struct CanonicalLipSample {
        let point: CGPoint
        let alpha: Float
    }

    private struct CanonicalAlphaSample {
        let alpha: Float
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

        init?(contour: LipContour) {
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

            func appendFallbackStrip(outerPositions: [Int], innerPositions: [Int]) {
                guard outerPositions.count == innerPositions.count,
                      outerPositions.count >= 2 else {
                    return
                }

                let base = builtVertices.count
                for offset in 0..<outerPositions.count {
                    let outerIndex = outerPositions[offset]
                    let innerIndex = innerPositions[offset]
                    builtVertices.append(UVVertex(uv: contour.outerUV[outerIndex], screen: contour.outer[outerIndex]))
                    builtVertices.append(UVVertex(uv: contour.innerUV[innerIndex], screen: contour.inner[innerIndex]))
                }

                for offset in 0..<(outerPositions.count - 1) {
                    let a = base + offset * 2
                    let b = base + (offset + 1) * 2
                    let c = a + 1
                    let d = b + 1
                    builtTriangles.append((a, b, c))
                    builtTriangles.append((b, d, c))
                }
            }

            if builtTriangles.isEmpty {
                appendFallbackStrip(outerPositions: Array(0...10), innerPositions: Array(0...10))
                appendFallbackStrip(
                    outerPositions: [0] + Array(stride(from: 19, through: 10, by: -1)),
                    innerPositions: [0] + Array(stride(from: 19, through: 10, by: -1))
                )
            }

            var builtOuterBoundary: [(CGPoint, CGPoint)] = []
            var builtInnerBoundary: [(CGPoint, CGPoint)] = []
            func appendBoundary(_ points: [CGPoint], to boundary: inout [(CGPoint, CGPoint)]) {
                guard points.count > 1 else {
                    return
                }

                for index in 0..<(points.count - 1) {
                    boundary.append((points[index], points[index + 1]))
                }
            }

            appendBoundary(Array(0...10).map { contour.outerUV[$0] }, to: &builtOuterBoundary)
            appendBoundary(([0] + Array(stride(from: 19, through: 10, by: -1))).map { contour.outerUV[$0] }, to: &builtOuterBoundary)
            appendBoundary(Array(0...10).map { contour.innerUV[$0] }, to: &builtInnerBoundary)
            appendBoundary(([0] + Array(stride(from: 19, through: 10, by: -1))).map { contour.innerUV[$0] }, to: &builtInnerBoundary)

            vertices = builtVertices
            triangles = builtTriangles
            outerBoundarySegments = builtOuterBoundary
            innerBoundarySegments = builtInnerBoundary
            outerBoundary = contour.outerUV
            innerBoundary = contour.innerUV
        }

        func sample(u: CGFloat, topV: CGFloat) -> CanonicalLipSample? {
            guard let alphaSample = alphaSample(u: u, topV: topV),
                  let screen = screenPoint(u: u, topV: topV) else {
                return nil
            }
            return CanonicalLipSample(point: screen, alpha: alphaSample.alpha)
        }

        func alphaSample(u: CGFloat, topV: CGFloat) -> CanonicalAlphaSample? {
            guard let edgeDistances = edgeDistances(u: u, topV: topV) else {
                return nil
            }

            let outerAlpha = Self.smoothStep(edge0: 0.004, edge1: 0.12, value: edgeDistances.outer)
            let innerEdgeAlpha = Self.smoothStep(edge0: 0.0005, edge1: 0.022, value: edgeDistances.inner)
            let lowerRegion = edgeDistances.uv.y > 0.30
            let edgeFloor: CGFloat = lowerRegion ? 0.62 : 0.46
            let innerFloor: CGFloat = lowerRegion ? 0.74 : 0.64
            let regionBoost: CGFloat = lowerRegion ? 1.24 : 1.12
            let innerAlpha = innerFloor + innerEdgeAlpha * (1 - innerFloor)
            let alpha = min((edgeFloor + outerAlpha * (1 - edgeFloor)) * innerAlpha * regionBoost, 1)
            return CanonicalAlphaSample(alpha: Float(alpha))
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

        private func edgeDistances(u: CGFloat, topV: CGFloat) -> (outer: CGFloat, inner: CGFloat, uv: CGPoint)? {
            let uv = CGPoint(x: max(0, min(u, 1)), y: 1 - max(0, min(topV, 1)))
            guard triangles.contains(where: { triangle in
                Self.barycentric(
                    point: uv,
                    a: vertices[triangle.0].uv,
                    b: vertices[triangle.1].uv,
                    c: vertices[triangle.2].uv
                ) != nil
            }) else {
                return nil
            }
//            guard Self.pointInPolygon(uv, polygon: outerBoundary),
//                  !Self.pointInPolygon(uv, polygon: innerBoundary) else {
//                return nil
//            }

            let outerDistance = outerBoundarySegments.reduce(CGFloat.greatestFiniteMagnitude) { current, segment in
                min(current, Self.distance(from: uv, to: segment))
            }
            let innerDistance = innerBoundarySegments.reduce(CGFloat.greatestFiniteMagnitude) { current, segment in
                min(current, Self.distance(from: uv, to: segment))
            }
            return (outerDistance, innerDistance, uv)
        }

        private static func mix(_ first: CGPoint, _ second: CGPoint, t: CGFloat) -> CGPoint {
            CGPoint(
                x: first.x + (second.x - first.x) * t,
                y: first.y + (second.y - first.y) * t
            )
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

        private static func distance(from point: CGPoint, to segment: (CGPoint, CGPoint)) -> CGFloat {
            let start = segment.0
            let end = segment.1
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

        private static func smoothStep(edge0: CGFloat, edge1: CGFloat, value: CGFloat) -> CGFloat {
            let t = max(0, min((value - edge0) / max(edge1 - edge0, 0.0001), 1))
            return t * t * (3 - 2 * t)
        }
    }

    private let colorLock = NSLock()
    private var lipstickColor = RGBColor(red: 0.82, green: 0.08, blue: 0.08)

    func updateColor(_ color: UIColor) {
        let matteColor = Self.matteLipColor(from: color)
        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        var alpha: CGFloat = 0
        matteColor.getRed(&red, green: &green, blue: &blue, alpha: &alpha)

        colorLock.lock()
        lipstickColor = RGBColor(red: Float(red), green: Float(green), blue: Float(blue))
        colorLock.unlock()
    }

    func makeTexture(contour: LipContour,
                     pixelBuffer: CVPixelBuffer,
                     imageSize: CGSize,
                     viewportSize: CGSize,
                     renderScale: CGFloat,
                     lowLatency: Bool = false) -> LipTexture? {
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

        guard let sampler = CanonicalLipSampler(contour: contour) else {
            LipDebugLog.throttled(
                "lip_texture_no_sampler",
                "lip_texture nil reason=no_sampler outer=\(contour.outer.count) inner=\(contour.inner.count) outerUV=\(contour.outerUV.count) innerUV=\(contour.innerUV.count)"
            )
            return nil
        }

        let pixelWidth = lowLatency ? 192 : 256
        let pixelHeight = lowLatency ? 96 : 128

        let imageTransform = Self.aspectFillTransform(for: imageSize, in: viewportSize)
        let inverseImageTransform = imageTransform.inverted()
        let color = currentLipstickColor()

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)

        var rgba = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
        var alphaPixels = 0
        var sampledPixels = 0
        var outOfImagePixels = 0
        var paintedPixels = 0
        for y in 0..<pixelHeight {
            for x in 0..<pixelWidth {
                let u = (CGFloat(x) + 0.5) / CGFloat(pixelWidth)
                let topV = (CGFloat(y) + 0.5) / CGFloat(pixelHeight)
                guard let sample = sampler.sample(u: u, topV: topV),
                      sample.alpha > 0.01 else {
                    continue
                }
                alphaPixels += 1

                let imagePoint = sample.point.applying(inverseImageTransform)
                if Self.samplePixel(
                    at: imagePoint,
                    baseAddress: baseAddress,
                    bytesPerRow: bytesPerRow,
                    width: sourceWidth,
                    height: sourceHeight
                ) != nil {
                    sampledPixels += 1
                } else {
                    outOfImagePixels += 1
                }

                let edgeNoise = Self.edgeNoise(maskAlpha: sample.alpha, x: x, y: y, frame: .zero)
                let coverageAlpha = max(sample.alpha, lowLatency ? 0.74 : 0.66)
                let alpha = min(coverageAlpha * 0.90 * edgeNoise, lowLatency ? 0.88 : 0.84)
                let lipRGB = color
                let offset = (y * pixelWidth + x) * 4
                rgba[offset] = Self.uint8(lipRGB.red * alpha)
                rgba[offset + 1] = Self.uint8(lipRGB.green * alpha)
                rgba[offset + 2] = Self.uint8(lipRGB.blue * alpha)
                rgba[offset + 3] = Self.uint8(alpha)
                paintedPixels += 1
            }
        }

        var usedFallback = false
        var fallbackPaintedPixels = 0
        if paintedPixels < max(64, pixelWidth * pixelHeight / 80) {
            let fallback = Self.makeFallbackRGBA(
                sampler: sampler,
                color: color,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
            )
            rgba = fallback.rgba
            usedFallback = true
            fallbackPaintedPixels = fallback.paintedPixels
        }
        rgba = Self.edgeSoftenedRGBA(rgba, pixelWidth: pixelWidth, pixelHeight: pixelHeight)

        LipDebugLog.throttled(
            "lip_texture_stats",
            interval: 0.6,
            "lip_texture stats size=\(pixelWidth)x\(pixelHeight) lowLatency=\(lowLatency) alphaPixels=\(alphaPixels) sampled=\(sampledPixels) outOfImage=\(outOfImagePixels) painted=\(paintedPixels) fallback=\(usedFallback) fallbackPainted=\(fallbackPaintedPixels) image=\(Int(imageSize.width))x\(Int(imageSize.height)) viewport=\(Int(viewportSize.width))x\(Int(viewportSize.height))"
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

        return LipTexture(image: UIImage(cgImage: cgImage, scale: renderScale, orientation: .up))
    }

    private static func makeFallbackRGBA(sampler: CanonicalLipSampler,
                                         color: RGBColor,
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

                let edgeNoise = edgeNoise(maskAlpha: sample.alpha, x: x, y: y, frame: .zero)
                let coverageAlpha = max(sample.alpha, 0.68)
                let alpha = min(coverageAlpha * 0.90 * edgeNoise, 0.86)
                let offset = (y * pixelWidth + x) * 4
                rgba[offset] = uint8(color.red * alpha)
                rgba[offset + 1] = uint8(color.green * alpha)
                rgba[offset + 2] = uint8(color.blue * alpha)
                rgba[offset + 3] = uint8(alpha)
                paintedPixels += 1
            }
        }
        return (rgba, paintedPixels)
    }

    private func currentLipstickColor() -> RGBColor {
        colorLock.lock()
        let color = lipstickColor
        colorLock.unlock()
        return color
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

        let textureGain = max(0.36, min(0.48 + luminance * 0.92, 1.14))
        return RGBColor(
            red: min(lipstick.red * textureGain, 1),
            green: min(lipstick.green * textureGain, 1),
            blue: min(lipstick.blue * textureGain, 1)
        )
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
        let breakup = 1 - edgeInfluence * (0.05 + fineNoise * 0.17)
        let microVariation = 1 + (coarseNoise - 0.5) * edgeInfluence * 0.10
        return max(0.70, min(1.06, breakup * microVariation))
    }

    private static func edgeSoftenedRGBA(_ rgba: [UInt8],
                                         pixelWidth: Int,
                                         pixelHeight: Int) -> [UInt8] {
        guard pixelWidth > 2, pixelHeight > 2 else {
            return rgba
        }

        var output = rgba
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

        for y in 1..<(pixelHeight - 1) {
            for x in 1..<(pixelWidth - 1) {
                let offset = (y * pixelWidth + x) * 4
                let originalAlpha = rgba[offset + 3]
                guard originalAlpha > 0, originalAlpha < 245 else {
                    continue
                }

                var red = 0
                var green = 0
                var blue = 0
                var alpha = 0
                var totalWeight = 0

                for sample in weights {
                    let sampleOffset = ((y + sample.dy) * pixelWidth + (x + sample.dx)) * 4
                    red += Int(rgba[sampleOffset]) * sample.weight
                    green += Int(rgba[sampleOffset + 1]) * sample.weight
                    blue += Int(rgba[sampleOffset + 2]) * sample.weight
                    alpha += Int(rgba[sampleOffset + 3]) * sample.weight
                    totalWeight += sample.weight
                }

                let edgeBlend = Float(255 - originalAlpha) / 255
                let blurBlend = min(max(edgeBlend * 0.62, 0), 0.46)
                let blurredAlpha = UInt8(alpha / totalWeight)
                output[offset] = blendChannel(original: rgba[offset], blurred: UInt8(red / totalWeight), amount: blurBlend)
                output[offset + 1] = blendChannel(original: rgba[offset + 1], blurred: UInt8(green / totalWeight), amount: blurBlend)
                output[offset + 2] = blendChannel(original: rgba[offset + 2], blurred: UInt8(blue / totalWeight), amount: blurBlend)
                output[offset + 3] = blendChannel(original: originalAlpha, blurred: blurredAlpha, amount: blurBlend)
            }
        }

        return output
    }

    private static func blendChannel(original: UInt8, blurred: UInt8, amount: Float) -> UInt8 {
        let mixed = Float(original) + (Float(blurred) - Float(original)) * amount
        return UInt8(max(0, min(mixed.rounded(), 255)))
    }

    private static func matteLipColor(from color: UIColor) -> UIColor {
        var hue: CGFloat = 0
        var saturation: CGFloat = 0
        var brightness: CGFloat = 0
        var alpha: CGFloat = 0

        if color.getHue(&hue, saturation: &saturation, brightness: &brightness, alpha: &alpha) {
            return UIColor(
                hue: hue,
                saturation: min(saturation * 1.20 + 0.05, 1),
                brightness: min(brightness * 0.94 + 0.02, 1),
                alpha: 1
            )
        }

        var red: CGFloat = 0
        var green: CGFloat = 0
        var blue: CGFloat = 0
        color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
        return UIColor(
            red: min(red * 1.08, 1),
            green: min(green * 1.08, 1),
            blue: min(blue * 1.08, 1),
            alpha: 1
        )
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
    @Binding private var isFaceDetected: Bool
    weak var sceneView: ARSCNView?

    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let landmarkQueue = DispatchQueue(label: "VirtualMakeup.MediaPipeLandmarks", qos: .userInteractive)
    private let textureQueue = DispatchQueue(label: "VirtualMakeup.LipTexture", qos: .userInteractive)
    private let lipTextureRenderer = LipTextureRenderer()
    private let lipMeshRenderer = LipMeshRenderer()
    private let viewportLock = NSLock()
    private let detectionLock = NSLock()
    private let trackingLock = NSLock()
    private let motionLock = NSLock()
    private let textureStateLock = NSLock()
    private let meshStateLock = NSLock()

    private var faceLandmarker: FaceLandmarker?
    private var cachedViewportSize: CGSize = .zero
    private var cachedRenderScale: CGFloat = 2
    private var isDetectingLandmarks = false
    private var lastLandmarkTimestampInMilliseconds = -1
    private var missedDetectionCount = 0
    private var smoothedLipContour: LipContour?
    private var smoothedLipPose: LipPose?
    private var hasTrackedFaceAnchor = false
    private var latestLipMotionPose: LipMotionPose?
    private var textureGeneration = 0
    private var nextTextureRequestID = 0
    private var latestTextureRequestID = 0
    private var isRenderingTexture = false
    private var pendingTextureRequest: LipTextureRequest?
    private var pendingLiveFrames: [Int: PendingLiveFrame] = [:]
    private var lastTextureSubmitTime: CFTimeInterval?
    private var lastAcceptedLipShapeTime: CFTimeInterval?
    private var lastDisplayedLipTextureTime: CFTimeInterval?
    private var lastAcceptedMotionPose: LipMotionPose?
    private var latestMeshContour: LipContour?
    private var latestMeshContourTime: CFTimeInterval?
    private var latestMeshMotionPose: LipMotionPose?
    private var latestLipTexture: LipTexture?
    private var lastLoggedMediaPipeLandmarkCount: Int?
    private var neutralMouthWidth: Float?

    private static let outerLipIndices = CanonicalLipGeometry.outerLipIndices
    private static let innerLipIndices = CanonicalLipGeometry.innerLipIndices
    private static let attentionLipIndices = CanonicalLipGeometry.attentionLipIndices
    private static let arKitMouthLeftIndex = 249
    private static let arKitMouthRightIndex = 684
    private static let arKitMouthTopIndex = 24
    private static let arKitMouthBottomIndex = 25
    private static let maxMotionCompensationAge: CFTimeInterval = 0.16
    private static let maxDisplayedTextureTrackingAge: CFTimeInterval = 0.36
    private static let maxPendingLiveFrameAge: CFTimeInterval = 0.8
    private static let minTextureRenderInterval: CFTimeInterval = 0.045
    private static let lipMeshVerticalAlignmentOffset: Float = 0.045
    private static let maxLipMeshVerticalAlignment: Float = 0.0022

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

        let configuration = ARFaceTrackingConfiguration()
        configuration.isLightEstimationEnabled = true
        sceneView?.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func stopSession() {
        resetLipTrackingAsync()
        lipMeshRenderer.clearAll()
        setTrackedFaceAnchor(false)
    }

    func updateColors(_ color: UIColor) {
        lipTextureRenderer.updateColor(color)
    }

    func updateViewport(size: CGSize, scale: CGFloat) {
        guard size.width > 1, size.height > 1 else {
            return
        }

        viewportLock.lock()
        cachedViewportSize = size
        cachedRenderScale = scale
        viewportLock.unlock()
    }

    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        guard anchor is ARFaceAnchor else {
            return nil
        }

        setTrackedFaceAnchor(true)
        DispatchQueue.main.async { self.isFaceDetected = true }
        let faceNode = SCNNode()
        lipMeshRenderer.attach(to: faceNode)
        return faceNode
    }

    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        guard let faceAnchor = anchor as? ARFaceAnchor else {
            return
        }

        setTrackedFaceAnchor(faceAnchor.isTracked)
        if !faceAnchor.isTracked {
            resetAndClearOnMain()
        } else {
            lipMeshRenderer.updateOccluder(faceAnchor: faceAnchor, renderer: renderer, faceNode: node)

            if let motionPose = projectedLipMotionPose(from: faceAnchor, renderer: renderer) {
                setLatestLipMotionPose(motionPose)
            }

            let mouthFrame = localMouthFrame(from: faceAnchor)
            let meshState = currentLipMeshState()
            if let mouthFrame,
               let meshState {
                lipMeshRenderer.render(
                contour: meshState.contour,
                texture: meshState.texture,
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

        DispatchQueue.main.async {
            self.isFaceDetected = faceAnchor.isTracked
        }
    }

    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        setTrackedFaceAnchor(false)
        resetAndClearOnMain()
        DispatchQueue.main.async {
            self.isFaceDetected = false
        }
    }

    func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
        if case .normal = camera.trackingState {
            return
        }

        setTrackedFaceAnchor(false)
        resetAndClearOnMain()
        DispatchQueue.main.async {
            self.isFaceDetected = false
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard faceLandmarker != nil, currentTrackedFaceAnchor() else {
            return
        }

        let viewport = currentViewport()
        let viewportSize = viewport.size
        guard viewportSize.width > 1, viewportSize.height > 1 else {
            return
        }

        let timestampInMilliseconds = Int(frame.timestamp * 1000)
        guard beginDetection(timestampInMilliseconds: timestampInMilliseconds) else {
            return
        }

        let pixelBuffer = frame.capturedImage
        landmarkQueue.async { [weak self] in
            guard let self else {
                return
            }

            self.submitLiveStreamFrame(
                pixelBuffer: pixelBuffer,
                timestampInMilliseconds: timestampInMilliseconds,
                viewportSize: viewportSize,
                renderScale: viewport.scale
            )
        }
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

    private struct LipDetection {
        let contour: LipContour
        let pixelBuffer: CVPixelBuffer
        let imageSize: CGSize
    }

    private struct PendingLiveFrame {
        let pixelBuffer: CVPixelBuffer
        let imageSize: CGSize
        let viewportSize: CGSize
        let renderScale: CGFloat
        let generation: Int
        let motionReference: LipMotionPose?
        let createdAt: CFTimeInterval
    }

    private struct LipTextureRequest {
        let contour: LipContour
        let pixelBuffer: CVPixelBuffer
        let imageSize: CGSize
        let viewportSize: CGSize
        let renderScale: CGFloat
        let lowLatency: Bool
        let generation: Int
        let requestID: Int
        let motionReference: LipMotionPose
    }

    private struct LipMeshState {
        let contour: LipContour
        let texture: LipTexture
        let contourAge: CFTimeInterval
        let motionDelta: CGFloat
    }

    private func submitLiveStreamFrame(pixelBuffer: CVPixelBuffer,
                                       timestampInMilliseconds: Int,
                                       viewportSize: CGSize,
                                       renderScale: CGFloat) {
        defer {
            finishDetection()
        }

        guard let faceLandmarker,
              currentTrackedFaceAnchor(),
              let mediaPipeBuffer = makeMediaPipeInputBuffer(from: pixelBuffer) else {
            handleMissedLipDetection()
            return
        }

        let now = CACurrentMediaTime()
        prunePendingLiveFrames(now: now)

        let imageSize = CGSize(
            width: CVPixelBufferGetWidth(mediaPipeBuffer),
            height: CVPixelBufferGetHeight(mediaPipeBuffer)
        )
        pendingLiveFrames[timestampInMilliseconds] = PendingLiveFrame(
            pixelBuffer: mediaPipeBuffer,
            imageSize: imageSize,
            viewportSize: viewportSize,
            renderScale: renderScale,
            generation: currentTextureGeneration(),
            motionReference: currentLipMotionPose(),
            createdAt: now
        )

        do {
            let image = try MPImage(pixelBuffer: mediaPipeBuffer, orientation: .up)
            try faceLandmarker.detectAsync(
                image: image,
                timestampInMilliseconds: timestampInMilliseconds
            )
        } catch {
            pendingLiveFrames.removeValue(forKey: timestampInMilliseconds)
            handleMissedLipDetection()
        }
    }

    func faceLandmarker(_ faceLandmarker: FaceLandmarker,
                        didFinishDetection result: FaceLandmarkerResult?,
                        timestampInMilliseconds: Int,
                        error: Error?) {
        landmarkQueue.async { [weak self] in
            self?.handleLiveStreamResult(
                result,
                timestampInMilliseconds: timestampInMilliseconds,
                error: error
            )
        }
    }

    private func handleLiveStreamResult(_ result: FaceLandmarkerResult?,
                                        timestampInMilliseconds: Int,
                                        error: Error?) {
        guard let frame = pendingLiveFrames.removeValue(forKey: timestampInMilliseconds) else {
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

        missedDetectionCount = 0
        let lowLatencyShapeChange = shouldUseRawContour(
            previous: smoothedLipContour,
            current: detection.contour
        )
        let stableContour = lowLatencyShapeChange ? detection.contour : adaptivelySmoothed(detection.contour)
        let motionReference = frame.motionReference ?? LipMotionPose(contourPose: stableContour.pose)
        guard let motionReference else {
            handleMissedLipDetection()
            return
        }

        smoothedLipContour = stableContour
        smoothedLipPose = stableContour.pose
        lastAcceptedMotionPose = motionReference
        setLatestMeshContour(stableContour, motionReference: motionReference)
        markLipShapeAccepted()
        LipDebugLog.throttled(
            "lip_detection_accept",
            interval: 0.6,
            "lip_detection accept lowLatency=\(lowLatencyShapeChange) \(lipContourDebugSummary(stableContour))"
        )

        guard shouldSubmitTextureRender(
            lowLatency: lowLatencyShapeChange,
            now: CACurrentMediaTime()
        ) else {
            return
        }

        let request = LipTextureRequest(
            contour: stableContour,
            pixelBuffer: detection.pixelBuffer,
            imageSize: detection.imageSize,
            viewportSize: frame.viewportSize,
            renderScale: frame.renderScale,
            lowLatency: lowLatencyShapeChange,
            generation: currentTextureGeneration(),
            requestID: reserveTextureRequestID(),
            motionReference: motionReference
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

        return "\(boundsText) \(poseText) opening=\(debugFloat(mouthOpeningRatio(contour))) outer=\(contour.outer.count) inner=\(contour.inner.count) mesh=\(contour.meshPointsByIndex.count)"
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
            imageSize: frame.imageSize,
            viewportSize: frame.viewportSize
        )
        let inner = mappedLipLandmarks(
            for: Self.innerLipIndices,
            landmarks: landmarks,
            imageSize: frame.imageSize,
            viewportSize: frame.viewportSize
        )
        let meshPoints = mappedLipMeshPoints(
            for: Self.attentionLipIndices,
            landmarks: landmarks,
            imageSize: frame.imageSize,
            viewportSize: frame.viewportSize
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

        let maxWidth = max(max(currentPose.width, expectedPose.width), 1)
        let centerDistance = hypot(
            currentPose.center.x - expectedPose.center.x,
            currentPose.center.y - expectedPose.center.y
        ) / maxWidth
        let widthRatio = currentPose.width / max(expectedPose.width, 1)
        let angleDelta = abs(Self.normalizedAngle(currentPose.angle - expectedPose.angle))
        let viewportDrift = hypot(
            currentPose.center.x - expectedPose.center.x,
            currentPose.center.y - expectedPose.center.y
        ) / max(min(viewportSize.width, viewportSize.height), 1)

        if widthRatio < 0.50 || widthRatio > 1.95 {
            return false
        }
        if centerDistance > 0.92 || viewportDrift > 0.24 {
            return false
        }
        if centerDistance > 0.68 && angleDelta > 0.30 {
            return false
        }
        if angleDelta > 0.90 {
            return false
        }
        return true
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

        return innerBounds.width <= outerBounds.width * 1.08 &&
            innerBounds.height <= outerBounds.height * 1.05
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
        let scale = min(max(target.width / max(source.width, 1), 0.48), 2.25)
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

        let centerDelta = hypot(
            target.center.x - source.center.x,
            target.center.y - source.center.y
        ) / max(source.width, 1)
        let scaleDelta = abs(target.width / max(source.width, 1) - 1)
        let angleDelta = abs(Self.normalizedAngle(target.angle - source.angle))
        let openingDelta = abs(target.opening - source.opening)
        return centerDelta +
            scaleDelta * 0.65 +
            angleDelta * 0.35 +
            openingDelta * 1.45
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

    private func markLipShapeAccepted() {
        motionLock.lock()
        lastAcceptedLipShapeTime = CACurrentMediaTime()
        motionLock.unlock()
    }

    private func markLipTextureDisplayed() {
        motionLock.lock()
        lastDisplayedLipTextureTime = CACurrentMediaTime()
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
        return CACurrentMediaTime() - timestamp <= Self.maxMotionCompensationAge
    }

    private func hasFreshDisplayedLipTexture() -> Bool {
        motionLock.lock()
        let timestamp = lastDisplayedLipTextureTime
        motionLock.unlock()

        guard let timestamp else {
            return false
        }
        return CACurrentMediaTime() - timestamp <= Self.maxDisplayedTextureTrackingAge
    }

    private func handleMissedLipDetection() {
        missedDetectionCount += 1
        guard missedDetectionCount > 3 else {
            return
        }

        resetLipTracking()
        DispatchQueue.main.async { [weak self] in
            self?.lipMeshRenderer.clearAll()
        }
    }

    private func prunePendingLiveFrames(now: CFTimeInterval) {
        pendingLiveFrames = pendingLiveFrames.filter {
            now - $0.value.createdAt <= Self.maxPendingLiveFrameAge
        }

        guard pendingLiveFrames.count > 8 else {
            return
        }

        let oldKeys = pendingLiveFrames
            .sorted { $0.value.createdAt < $1.value.createdAt }
            .prefix(pendingLiveFrames.count - 8)
            .map(\.key)
        for key in oldKeys {
            pendingLiveFrames.removeValue(forKey: key)
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

        if lowLatency {
            lastTextureSubmitTime = now
            return true
        }

        guard let lastTextureSubmitTime else {
            self.lastTextureSubmitTime = now
            return true
        }

        guard now - lastTextureSubmitTime >= Self.minTextureRenderInterval else {
            return false
        }

        self.lastTextureSubmitTime = now
        return true
    }

    private func isTextureRenderInFlight() -> Bool {
        textureStateLock.lock()
        let isInFlight = isRenderingTexture
        textureStateLock.unlock()
        return isInFlight
    }

    private func renderTextureRequest(_ request: LipTextureRequest) {
        let texture = lipTextureRenderer.makeTexture(
            contour: request.contour,
            pixelBuffer: request.pixelBuffer,
            imageSize: request.imageSize,
            viewportSize: request.viewportSize,
            renderScale: request.renderScale,
            lowLatency: request.lowLatency
        )

        let isTracked = currentTrackedFaceAnchor()
        let hasFreshShape = hasFreshLipShape()
        let currentGeneration = currentTextureGeneration()
        let latestRequestID = currentLatestTextureRequestID()
        let generationOK = currentGeneration == request.generation
        let isSuperseded = latestRequestID != request.requestID
        if let texture,
           isTracked,
           hasFreshShape,
           generationOK {
            setLatestLipTexture(texture)
            markLipTextureDisplayed()
            LipDebugLog.throttled(
                "lip_texture_accept",
                interval: 0.6,
                "lip_texture accept requestID=\(request.requestID) latestRequestID=\(latestRequestID) superseded=\(isSuperseded) generation=\(request.generation) lowLatency=\(request.lowLatency)"
            )
            DispatchQueue.main.async { [weak self] in
                self?.isFaceDetected = true
            }
        } else {
            LipDebugLog.throttled(
                "lip_texture_reject",
                interval: 0.6,
                "lip_texture reject hasTexture=\(texture != nil) tracked=\(isTracked) freshShape=\(hasFreshShape) generationOK=\(generationOK) superseded=\(isSuperseded) requestID=\(request.requestID) latestRequestID=\(latestRequestID) requestGeneration=\(request.generation) currentGeneration=\(currentGeneration)"
            )
        }

        textureStateLock.lock()
        if let nextRequest = pendingTextureRequest {
            pendingTextureRequest = nil
            textureStateLock.unlock()
            renderTextureRequest(nextRequest)
        } else {
            isRenderingTexture = false
            textureStateLock.unlock()
        }
    }

    private func makeMediaPipeInputBuffer(from pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let sourceImage = CIImage(cvPixelBuffer: pixelBuffer)
            .oriented(forExifOrientation: Int32(CGImagePropertyOrientation.right.rawValue))
        let normalizedImage = sourceImage.transformed(
            by: CGAffineTransform(
                translationX: -sourceImage.extent.origin.x,
                y: -sourceImage.extent.origin.y
            )
        )
        let width = max(Int(normalizedImage.extent.width.rounded(.up)), 1)
        let height = max(Int(normalizedImage.extent.height.rounded(.up)), 1)

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
            normalizedImage,
            to: outputBuffer,
            bounds: CGRect(x: 0, y: 0, width: width, height: height),
            colorSpace: CGColorSpaceCreateDeviceRGB()
        )
        return outputBuffer
    }

    private func mappedLipLandmarks(for indices: [Int],
                                    landmarks: [NormalizedLandmark],
                                    imageSize: CGSize,
                                    viewportSize: CGSize) -> (points: [CGPoint], points3D: [SIMD3<Float>], uv: [CGPoint]) {
        let transform = aspectFillTransform(for: imageSize, in: viewportSize)
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
            let imagePoint = CGPoint(
                x: CGFloat(landmark.x) * imageSize.width,
                y: CGFloat(landmark.y) * imageSize.height
            )
            let point = imagePoint.applying(transform)
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
                                     imageSize: CGSize,
                                     viewportSize: CGSize) -> [Int: LipMeshPoint] {
        let transform = aspectFillTransform(for: imageSize, in: viewportSize)
        var pointsByIndex: [Int: LipMeshPoint] = [:]
        pointsByIndex.reserveCapacity(indices.count)

        for index in indices {
            guard landmarks.indices.contains(index),
                  let canonicalUV = CanonicalLipGeometry.normalizedUV(for: index) else {
                continue
            }

            let landmark = landmarks[index]
            let imagePoint = CGPoint(
                x: CGFloat(landmark.x) * imageSize.width,
                y: CGFloat(landmark.y) * imageSize.height
            )
            let point = imagePoint.applying(transform)
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

    private func adaptivelySmoothed(_ contour: LipContour) -> LipContour {
        guard var previous = smoothedLipContour,
              previous.outer.count == contour.outer.count,
              previous.inner.count == contour.inner.count else {
            return contour
        }

        if shouldUseRawContour(previous: previous, current: contour) {
            return contour
        }

        if let sourcePose = smoothedLipPose, let targetPose = contour.pose {
            previous = previous.transformed(from: sourcePose, to: targetPose)
        }

        let alpha = smoothingAlpha(for: contour)
        let outer = zip(previous.outer, contour.outer).map {
            smooth(previous: $0.0, current: $0.1, alpha: alpha)
        }
        let inner = zip(previous.inner, contour.inner).map {
            smooth(previous: $0.0, current: $0.1, alpha: alpha)
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
                    screen: smooth(previous: previousPoint.screen, current: currentPoint.screen, alpha: alpha),
                    normalized: previousPoint.normalized + (currentPoint.normalized - previousPoint.normalized) * meshAlpha,
                    uv: currentPoint.uv
                )
            }
        }
        return LipContour(
            outer: outer,
            inner: inner,
            outer3D: outer3D,
            inner3D: inner3D,
            outerUV: contour.outerUV,
            innerUV: contour.innerUV,
            meshPointsByIndex: meshPoints,
            faceGeometryPose: contour.faceGeometryPose ?? previous.faceGeometryPose
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

        return widthChange > 0.065 ||
            heightChange > 0.090 ||
            openingChange > 0.020
    }

    private func mouthOpeningRatio(_ contour: LipContour) -> CGFloat {
        guard let outerBounds = bounds(for: contour.outer),
              let innerBounds = bounds(for: contour.inner),
              outerBounds.width > 1 else {
            return 0
        }

        return innerBounds.height / outerBounds.width
    }

    private func smoothingAlpha(for contour: LipContour) -> CGFloat {
        guard let previousPose = smoothedLipPose,
              let currentPose = contour.pose else {
            return 0.94
        }

        let centerMotion = hypot(
            currentPose.center.x - previousPose.center.x,
            currentPose.center.y - previousPose.center.y
        ) / max(currentPose.width, 1)
        let scaleMotion = abs(currentPose.width - previousPose.width) / max(previousPose.width, 1)
        let angleMotion = abs(currentPose.angle - previousPose.angle)
        let motion = centerMotion * 1.6 + scaleMotion + angleMotion * 0.8
        return min(max(0.94 + motion * 1.6, 0.94), 0.99)
    }

    private func smooth(previous: CGPoint, current: CGPoint, alpha: CGFloat) -> CGPoint {
        CGPoint(
            x: previous.x + (current.x - previous.x) * alpha,
            y: previous.y + (current.y - previous.y) * alpha
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
        if openingRatio < 0.34 {
            if let previousNeutral = neutralMouthWidth {
                neutralMouthWidth = min(previousNeutral * 0.96 + width * 0.04, width)
            } else {
                neutralMouthWidth = width
            }
        }
        let referenceWidth = neutralMouthWidth.map { min(max($0, width * 0.72), width) } ?? width
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

        let verticalAlignment = min(
            referenceWidth * Self.lipMeshVerticalAlignmentOffset,
            Self.maxLipMeshVerticalAlignment
        )
        let center = (left + right + top + bottom) * 0.25 +
            downAxis * verticalAlignment
        guard center.x.isFinite,
              center.y.isFinite,
              center.z.isFinite else {
            return nil
        }

        return FaceLocalMouthFrame(
            center: center,
            xAxis: xAxis,
            downAxis: downAxis,
            normalAxis: normalAxis,
            width: width,
            smileExpansion: smileExpansion,
            openingRatio: openingRatio,
            referenceWidth: referenceWidth,
            verticalAlignment: verticalAlignment
        )
    }

    private func projectedLipMotionPose(from faceAnchor: ARFaceAnchor,
                                        renderer: SCNSceneRenderer) -> LipMotionPose? {
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

        let viewport = currentViewport()
        let extendedViewport = CGRect(
            x: -viewport.size.width * 0.25,
            y: -viewport.size.height * 0.25,
            width: viewport.size.width * 1.5,
            height: viewport.size.height * 1.5
        )

        func projectedPoint(for index: Int) -> CGPoint? {
            let local = vertices[index]
            let world = faceAnchor.transform * SIMD4<Float>(local.x, local.y, local.z, 1)
            let projected = renderer.projectPoint(SCNVector3(world.x, world.y, world.z))
            let point = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
            guard point.x.isFinite,
                  point.y.isFinite,
                  projected.z.isFinite,
                  projected.z >= 0,
                  projected.z <= 1,
                  extendedViewport.contains(point) else {
                return nil
            }
            return point
        }

        let faceScaleBasis = projectedFaceScaleBasis(
            vertices: vertices,
            faceTransform: faceAnchor.transform,
            renderer: renderer,
            extendedViewport: extendedViewport
        )

        guard let left = projectedPoint(for: Self.arKitMouthLeftIndex),
              let right = projectedPoint(for: Self.arKitMouthRightIndex),
              let top = projectedPoint(for: Self.arKitMouthTopIndex),
              let bottom = projectedPoint(for: Self.arKitMouthBottomIndex),
              let faceScaleBasis else {
            return nil
        }

        return LipMotionPose(
            left: left,
            right: right,
            top: top,
            bottom: bottom,
            scaleBasis: faceScaleBasis
        )
    }

    private func projectedFaceScaleBasis(vertices: [vector_float3],
                                         faceTransform: simd_float4x4,
                                         renderer: SCNSceneRenderer,
                                         extendedViewport: CGRect) -> CGFloat? {
        var xValues: [CGFloat] = []
        xValues.reserveCapacity(vertices.count)

        for vertex in vertices {
            let world = faceTransform * SIMD4<Float>(vertex.x, vertex.y, vertex.z, 1)
            let projected = renderer.projectPoint(SCNVector3(world.x, world.y, world.z))
            let point = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
            guard point.x.isFinite,
                  point.y.isFinite,
                  projected.z.isFinite,
                  projected.z >= 0,
                  projected.z <= 1,
                  extendedViewport.contains(point) else {
                continue
            }
            xValues.append(point.x)
        }

        guard xValues.count >= 40 else {
            return nil
        }

        xValues.sort()
        let inset = max(xValues.count / 20, 1)
        let low = xValues[inset]
        let high = xValues[xValues.count - inset - 1]
        let width = high - low
        guard width.isFinite, width > 12 else {
            return nil
        }
        return width
    }

    private func beginDetection(timestampInMilliseconds: Int) -> Bool {
        detectionLock.lock()
        defer {
            detectionLock.unlock()
        }

        guard !isDetectingLandmarks,
              timestampInMilliseconds > lastLandmarkTimestampInMilliseconds else {
            return false
        }

        isDetectingLandmarks = true
        lastLandmarkTimestampInMilliseconds = timestampInMilliseconds
        return true
    }

    private func finishDetection() {
        detectionLock.lock()
        isDetectingLandmarks = false
        detectionLock.unlock()
    }

    private func currentViewport() -> (size: CGSize, scale: CGFloat) {
        viewportLock.lock()
        let size = cachedViewportSize
        let scale = cachedRenderScale
        viewportLock.unlock()
        return (size, scale)
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
        textureStateLock.unlock()
    }

    private func setTrackedFaceAnchor(_ isTracked: Bool) {
        trackingLock.lock()
        hasTrackedFaceAnchor = isTracked
        trackingLock.unlock()
    }

    private func currentTrackedFaceAnchor() -> Bool {
        trackingLock.lock()
        let isTracked = hasTrackedFaceAnchor
        trackingLock.unlock()
        return isTracked
    }

    private func setLatestLipMotionPose(_ pose: LipMotionPose?) {
        motionLock.lock()
        latestLipMotionPose = pose
        motionLock.unlock()
    }

    private func currentLipMotionPose() -> LipMotionPose? {
        motionLock.lock()
        let pose = latestLipMotionPose
        motionLock.unlock()
        return pose
    }

    private func setLatestMeshContour(_ contour: LipContour, motionReference: LipMotionPose) {
        meshStateLock.lock()
        latestMeshContour = contour
        latestMeshContourTime = CACurrentMediaTime()
        latestMeshMotionPose = motionReference
        meshStateLock.unlock()
    }

    private func setLatestLipTexture(_ texture: LipTexture) {
        meshStateLock.lock()
        latestLipTexture = texture
        meshStateLock.unlock()
    }

    private func currentLipMeshState() -> LipMeshState? {
        meshStateLock.lock()
        let contour = latestMeshContour
        let contourTime = latestMeshContourTime
        let contourMotionPose = latestMeshMotionPose
        let texture = latestLipTexture
        let currentMotionPose = latestLipMotionPose
        meshStateLock.unlock()

        guard let contour, let texture else {
            return nil
        }
        return LipMeshState(
            contour: contour,
            texture: texture,
            contourAge: contourTime.map { CACurrentMediaTime() - $0 } ?? .greatestFiniteMagnitude,
            motionDelta: motionDelta(from: contourMotionPose, to: currentMotionPose)
        )
    }

    private func currentLipMeshAvailability() -> (contour: Bool, texture: Bool) {
        meshStateLock.lock()
        let hasContour = latestMeshContour != nil
        let hasTexture = latestLipTexture != nil
        meshStateLock.unlock()
        return (hasContour, hasTexture)
    }

    private func clearLipMeshState() {
        meshStateLock.lock()
        latestMeshContour = nil
        latestMeshContourTime = nil
        latestMeshMotionPose = nil
        latestLipTexture = nil
        meshStateLock.unlock()
    }

    private func resetLipTracking() {
        missedDetectionCount = 0
        smoothedLipContour = nil
        smoothedLipPose = nil
        pendingLiveFrames.removeAll()
        lastTextureSubmitTime = nil
        lastAcceptedMotionPose = nil
        neutralMouthWidth = nil
        clearLipShapeFreshness()
        setLatestLipMotionPose(nil)
        clearLipMeshState()
        invalidatePendingTextures()
    }

    private func resetLipTrackingAsync() {
        landmarkQueue.async { [weak self] in
            self?.resetLipTracking()
        }
    }

    private func resetAndClearOnMain() {
        resetLipTrackingAsync()
        DispatchQueue.main.async {
            self.lipMeshRenderer.clearAll()
        }
    }
}
