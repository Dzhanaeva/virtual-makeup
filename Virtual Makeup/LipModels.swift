import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

enum FaceBlendShapeValue {
    static func value(_ location: ARFaceAnchor.BlendShapeLocation,
                      in blendShapes: [ARFaceAnchor.BlendShapeLocation: NSNumber]) -> Float {
        guard let value = blendShapes[location]?.floatValue,
              value.isFinite else {
            return 0
        }
        return min(max(value, 0), 1)
    }
}

struct LipMeshPoint {
    let screen: CGPoint
    let normalized: SIMD3<Float>
    let uv: CGPoint
}

struct LipSurfaceBinding {
    let vertexIndices: SIMD3<Int32>
    let barycentricWeights: SIMD3<Float>
    let projectionError: CGFloat
    let referenceScreenPoint: CGPoint
}

struct ProjectedFaceSurfaceTriangle {
    let vertexIndices: SIMD3<Int32>
    let first: CGPoint
    let second: CGPoint
    let third: CGPoint
    let cameraDepths: SIMD3<Float>
    let normalizedMouthY: Float
}

struct FaceSurfaceTriangleKey: Hashable {
    let first: Int32
    let second: Int32
    let third: Int32

    init(_ indices: SIMD3<Int32>) {
        first = indices.x
        second = indices.y
        third = indices.z
    }
}

struct FaceSurfaceSnapshot {
    let vertexCount: Int
    let triangleIndexCount: Int
    let topologySignature: UInt64
    let sourceVertices: [SIMD3<Float>]
    let mouthFrame: FaceSurfaceMouthFrame
    let triangles: [ProjectedFaceSurfaceTriangle]
    let trianglesByKey: [FaceSurfaceTriangleKey: ProjectedFaceSurfaceTriangle]
}

struct FaceSurfaceProjectionInput {
    let geometry: ARFaceGeometry
    let anchorTransform: simd_float4x4
    let camera: ARCamera
    let orientation: UIInterfaceOrientation
    let viewportSize: CGSize
    let anchorIdentifier: UUID
}

struct FaceSurfaceTopologyCache {
    let vertexCount: Int
    let triangleIndexCount: Int
    let topologySignature: UInt64
    let anchorIdentifier: UUID
    let triangles: [SIMD3<Int32>]
}

struct FaceSurfaceMouthFrame {
    let center: SIMD3<Float>
    let xAxis: SIMD3<Float>
    let downAxis: SIMD3<Float>
    let normalAxis: SIMD3<Float>
    let width: Float
}

func faceSurfaceTopologySignature(vertexCount: Int,
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

struct LipContour {
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

    var innerApertureVisibility: CGFloat {
        // ARKit explicitly vetoes this value when the mouth is confidently
        // closed. Once that veto is lifted, reveal the MediaPipe aperture
        // early so a newly visible tooth surface is never covered by a stale
        // closed-mouth lipstick mask.
        let openingPosition = min(
            max((effectiveInnerOpeningRatio - 0.018) / (0.055 - 0.018), 0),
            1
        )
        return openingPosition * openingPosition * (3 - 2 * openingPosition)
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
        // ARKit's rigid mouth reference is considerably more stable than a
        // delayed detector keyframe during camera-depth motion. Keep a wide,
        // bounded range so a fast approach does not leave the contour visibly
        // smaller than the face until the next MediaPipe result arrives.
        let scale = min(max(target.width / max(source.width, 1), 0.55), 1.80)
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
        let scale = min(max(target.width / max(source.width, 1), 0.55), 1.80)
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

struct FaceGeometryPose {
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

struct LipPose {
    let center: CGPoint
    let width: CGFloat
    let angle: CGFloat
}

enum LipOuterMargin {
    // These fractions preserve the tuned margins around an 80-point mouth,
    // but shrink with the detected lip contour when the face moves away.
    // Absolute upper bounds keep close-up faces from producing oversized
    // textures or SceneKit carrier geometry.
    private static let samplingFraction: CGFloat = 0.075
    // The carrier needs only a compact strip outside the real lip. Earlier
    // 10-14 point margins were tuned while the alpha map was not applied and
    // made any transparency failure look like a much larger painted lip.
    private static let carrierFraction: CGFloat = 0.030

    static func sampling(for lipWidth: CGFloat) -> CGFloat {
        min(max(lipWidth, 0) * samplingFraction, 10)
    }

    static func carrier(for lipWidth: CGFloat) -> CGFloat {
        let scaledMargin = max(lipWidth, 0) * carrierFraction
        guard scaledMargin > 0 else {
            return 0
        }
        return min(max(scaledMargin, 3.5), 5.0)
    }
}

enum LipOuterFeatherLayout {
    // Both the Metal colour raster and the SceneKit carrier must address the
    // same transparent area of the generated texture. Keep these values in
    // one place so changing one renderer cannot silently clip the other.
    // Distance is measured in canonical texture-height units. Outer UVs are
    // expanded along contour normals, so every carrier vertex lands beyond
    // the alpha tail rather than relying on an approximate radial scale.
    static let canonicalCarrierDistance: CGFloat = 0.052
    static let canonicalTextureAspectRatio: CGFloat = 2.0

    // Soften only a one-pixel rim inside the real lip, then fade through the
    // compact exterior carrier. Spare transparent texels before the carrier
    // boundary prevent texture filtering from exposing a hard polygon edge.
    static let interiorTransitionPixels: Float = 8
    static let exteriorTransitionPixels: Float = 4.0
}

enum LipOuterBoundary {
    /// Builds a constant-distance carrier around the detected lip outline.
    /// Directions come from the fixed canonical contour rather than adjacent
    /// live landmarks. A noisy point can still move itself, but can no longer
    /// rotate the carrier offset of both neighbours and amplify edge jitter.
    static func expandedScreenPoints(
        contour: LipContour,
        distance: CGFloat
    ) -> [Int: CGPoint]? {
        guard let pose = contour.pose,
              distance.isFinite,
              distance >= 0 else {
            return nil
        }

        let indices = CanonicalLipGeometry.outerLipIndices
        var basePoints: [Int: CGPoint] = [:]
        basePoints.reserveCapacity(indices.count)
        for index in indices {
            guard let point = contour.meshPointsByIndex[index]?.screen,
                  point.x.isFinite,
                  point.y.isFinite else {
                return nil
            }
            basePoints[index] = point
        }

        guard let outwardNormals = canonicalOutwardNormals(
            contour: contour,
            aspectRatio: LipOuterFeatherLayout.canonicalTextureAspectRatio
        ) else {
            return nil
        }

        var expanded: [Int: CGPoint] = [:]
        expanded.reserveCapacity(indices.count)
        let cosine = cos(pose.angle)
        let sine = sin(pose.angle)
        for index in indices {
            guard let point = basePoints[index],
                  let normal = outwardNormals[index] else {
                return nil
            }

            // Canonical V points upward, while viewport Y points downward.
            // Rotate the stable local direction with the smoothed lip pose.
            let localX = normal.dx
            let localDown = -normal.dy
            let screenNormalX = localX * cosine - localDown * sine
            let screenNormalY = localX * sine + localDown * cosine

            expanded[index] = CGPoint(
                x: point.x + screenNormalX * distance,
                y: point.y + screenNormalY * distance
            )
        }
        return expanded
    }

    /// Expands the canonical outer polygon by a constant texture-space
    /// distance. Working in aspect-corrected coordinates keeps the feather
    /// equally wide above the cupid's bow and along the mouth corners.
    static func expandedUVPoints(
        contour: LipContour,
        distance: CGFloat,
        aspectRatio: CGFloat
    ) -> [Int: CGPoint]? {
        guard distance.isFinite,
              distance >= 0,
              aspectRatio.isFinite,
              aspectRatio > 0 else {
            return nil
        }

        let indices = CanonicalLipGeometry.outerLipIndices
        guard let outwardNormals = canonicalOutwardNormals(
            contour: contour,
            aspectRatio: aspectRatio
        ) else {
            return nil
        }
        var expanded: [Int: CGPoint] = [:]
        expanded.reserveCapacity(indices.count)
        for index in indices {
            guard let point = contour.meshPointsByIndex[index]?.uv,
                  let normal = outwardNormals[index] else {
                return nil
            }
            let uv = CGPoint(
                x: point.x + normal.dx * distance / aspectRatio,
                y: point.y + normal.dy * distance
            )
            guard uv.x.isFinite,
                  uv.y.isFinite,
                  uv.x >= 0,
                  uv.x <= 1,
                  uv.y >= 0,
                  uv.y <= 1 else {
                return nil
            }
            expanded[index] = uv
        }
        return expanded
    }

    private static func canonicalOutwardNormals(
        contour: LipContour,
        aspectRatio: CGFloat
    ) -> [Int: CGVector]? {
        let indices = CanonicalLipGeometry.outerLipIndices
        var scaledPoints: [Int: CGPoint] = [:]
        scaledPoints.reserveCapacity(indices.count)
        for index in indices {
            guard let uv = contour.meshPointsByIndex[index]?.uv,
                  uv.x.isFinite,
                  uv.y.isFinite else {
                return nil
            }
            scaledPoints[index] = CGPoint(x: uv.x * aspectRatio, y: uv.y)
        }

        let scaledCenter = CGPoint(x: 0.5 * aspectRatio, y: 0.5)
        var normals: [Int: CGVector] = [:]
        normals.reserveCapacity(indices.count)
        for offset in indices.indices {
            let index = indices[offset]
            let previousIndex = indices[(offset - 1 + indices.count) % indices.count]
            let nextIndex = indices[(offset + 1) % indices.count]
            guard let point = scaledPoints[index],
                  let previous = scaledPoints[previousIndex],
                  let next = scaledPoints[nextIndex] else {
                return nil
            }

            let tangentX = next.x - previous.x
            let tangentY = next.y - previous.y
            let tangentLength = hypot(tangentX, tangentY)
            guard tangentLength.isFinite,
                  tangentLength > 0.000_1 else {
                return nil
            }

            var normalX = -tangentY / tangentLength
            var normalY = tangentX / tangentLength
            let radialX = point.x - scaledCenter.x
            let radialY = point.y - scaledCenter.y
            if normalX * radialX + normalY * radialY < 0 {
                normalX = -normalX
                normalY = -normalY
            }
            normals[index] = CGVector(dx: normalX, dy: normalY)
        }
        return normals
    }
}

struct LipMotionPose {
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

struct FaceLocalMouthFrame {
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
