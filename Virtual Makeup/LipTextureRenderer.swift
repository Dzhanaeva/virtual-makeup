import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

final class LipTextureRenderer {
    private struct CameraLipSample {
        let base: RGBColor
        let blurred: RGBColor
        let highlightSurround: RGBColor
        let highlightConfidence: Float
    }

    private struct CameraMaterialSignals {
        static let neutral = CameraMaterialSignals(
            detail: 1,
            lowFrequencyLuminance: 0,
            shineMask: 0
        )

        let detail: Float
        let lowFrequencyLuminance: Float
        let shineMask: Float
    }

    private struct PixelMaterialInput {
        let alpha: Float
        let pigmentDistribution: Float
        let recessShade: Float
        let lipBase: RGBColor
        let highlightSurround: RGBColor
        let pigment: RGBColor
        let topV: CGFloat
        let normalizedLipX: Float
        let lowerLipMembership: Float
        let outerDistance: CGFloat
        let innerDistance: CGFloat
        let opticalSurface: LipOpticalSurface.Sample?
    }

    private struct MaterialAppearance {
        let pigment: RGBColor
        let reflection: RGBColor
        let reflectionCoverage: Float
    }

    private struct ShimmerLayer {
        let color: RGBColor
        let coverage: Float
    }

    private struct ProceduralTextureKey: Equatable {
        let colorSignature: UInt32
        let finish: LipFinish
        let density: LipstickDensity
        let texture: LipstickTexture
        let materialProfile: LipMaterialProfile
        let shadingDebugMode: LipShadingDebugMode
        let renderScale: Int
        let excludesInnerMouth: Bool
        let apertureVisibilityBucket: Int
    }

    private struct LinerMaskKey: Equatable {
        let pixelWidth: Int
        let pixelHeight: Int
        let renderScaleBucket: Int
        let outerUV: [CGPoint]
        let innerUV: [CGPoint]
    }

    private let temporalTextureLock = NSLock()
    private var previousToneMap: [CameraMaterialSignals]?
    private var previousShineConfidence: [Float]?
    private var previousToneMapWidth = 0
    private var previousToneMapHeight = 0
    private var previousToneColor: RGBColor?
    private var previousToneMouthOpen: Bool?
    private var previousToneAverageLuma: Float?
    private var previousToneReferenceLuma: Float?
    private var lipstickFinish: LipFinish = .matte
    private var lipstickDensity: LipstickDensity = .high
    private var lipstickTexture: LipstickTexture = .creamy
    private var lipstickMaterialProfile = LipMaterialProfile.standard(
        finish: .matte,
        density: .high,
        texture: .creamy
    )
    private var lipstickShadingDebugMode: LipShadingDebugMode = .off
    private var cachedProceduralTextureKey: ProceduralTextureKey?
    private var cachedProceduralTexture: LipTexture?
    private var cachedLinerMaskKey: LinerMaskKey?
    private var cachedLinerOpacityImage: UIImage?
    private var previousLinerReferenceLuma: Float?
    private var previousLinerReferenceTime: CFTimeInterval?

    private struct CanonicalLipSample {
        let point: CGPoint
        let alpha: Float
        let recessShade: Float
        let outerDistance: CGFloat
        let innerDistance: CGFloat
    }

    private struct CanonicalAlphaSample {
        let alpha: Float
        let recessShade: Float
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
        private let outerBoundaryBounds: CGRect
        private let innerBoundaryBounds: CGRect
        private let excludesInnerMouth: Bool
        let innerApertureVisibility: CGFloat

        init?(contour: LipContour,
              excludesInnerMouth: Bool,
              expandsCameraCarrier: Bool = false) {
            guard contour.outer.count == 20,
                  contour.inner.count == 20,
                  contour.outerUV.count == 20,
                  contour.innerUV.count == 20 else {
                return nil
            }

            var builtVertices: [UVVertex] = []
            var builtTriangles: [(Int, Int, Int)] = []
            var vertexIndexByLandmark: [Int: Int] = [:]
            let expandedUV = expandsCameraCarrier ?
                LipOuterBoundary.expandedUVPoints(
                    contour: contour,
                    distance: LipOuterFeatherLayout.canonicalCarrierDistance,
                    aspectRatio: LipOuterFeatherLayout.canonicalTextureAspectRatio
                ) ?? [:] : [:]
            let expandedScreen = expandsCameraCarrier ?
                LipOuterBoundary.expandedScreenPoints(
                    contour: contour,
                    distance: LipOuterMargin.sampling(for: contour.pose?.width ?? 0)
                ) ?? [:] : [:]

            func vertex(for landmarkIndex: Int) -> UVVertex? {
                if let point = contour.meshPointsByIndex[landmarkIndex] {
                    if let uv = expandedUV[landmarkIndex],
                       let screen = expandedScreen[landmarkIndex] {
                        return UVVertex(uv: uv, screen: screen)
                    }
                    if CanonicalLipGeometry.isInnerLipIndex(landmarkIndex) {
                        return UVVertex(
                            uv: point.uv,
                            screen: point.screen
                        )
                    }
                    return UVVertex(uv: point.uv, screen: point.screen)
                }

                if let outerOffset = CanonicalLipGeometry.outerLipIndices.firstIndex(of: landmarkIndex),
                   contour.outerUV.indices.contains(outerOffset),
                   contour.outer.indices.contains(outerOffset) {
                    return UVVertex(
                        uv: expandedUV[landmarkIndex] ?? contour.outerUV[outerOffset],
                        screen: expandedScreen[landmarkIndex] ?? contour.outer[outerOffset]
                    )
                }

                if let innerOffset = CanonicalLipGeometry.innerLipIndices.firstIndex(of: landmarkIndex),
                   contour.innerUV.indices.contains(innerOffset),
                   contour.inner.indices.contains(innerOffset) {
                    return UVVertex(
                        uv: contour.innerUV[innerOffset],
                        screen: contour.inner[innerOffset]
                    )
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

            let samplingTriangles = CanonicalLipGeometry.lipMeshTriangles +
                CanonicalLipGeometry.innerFillTriangles
            for triangle in samplingTriangles {
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

            guard builtTriangles.count == samplingTriangles.count,
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
                  ),
                  let builtOuterBounds = Self.bounds(for: builtOuterCurve),
                  let builtInnerBounds = Self.bounds(for: builtInnerCurve),
                  builtOuterBounds.width > 0.000_1 else {
                return nil
            }

            vertices = builtVertices
            triangles = builtTriangles
            outerBoundarySegments = Self.closedSegments(for: builtOuterCurve)
            innerBoundarySegments = Self.closedSegments(for: builtInnerCurve)
            outerBoundary = builtOuterCurve
            innerBoundary = builtInnerCurve
            outerBoundaryBounds = builtOuterBounds
            innerBoundaryBounds = builtInnerBounds
            self.excludesInnerMouth = excludesInnerMouth
            innerApertureVisibility = contour.innerApertureVisibility
        }

        func shineGeometry(
            u: CGFloat,
            topV: CGFloat
        ) -> (normalizedX: Float, lowerLipMembership: Float) {
            let normalizedX = max(
                0,
                min(
                    (u - outerBoundaryBounds.minX) /
                        max(outerBoundaryBounds.width, 0.000_1),
                    1
                )
            )
            let mouthCenterTopV = 1 - innerBoundaryBounds.midY
            let lowerLipMembership = Self.smoothStep(
                edge0: mouthCenterTopV,
                edge1: mouthCenterTopV + 0.025,
                value: topV
            )
            return (Float(normalizedX), Float(lowerLipMembership))
        }

        func sample(u: CGFloat, topV: CGFloat) -> CanonicalLipSample? {
            guard let alphaSample = alphaSample(u: u, topV: topV),
                  let screen = screenPoint(u: u, topV: topV) else {
                return nil
            }
            return CanonicalLipSample(
                point: screen,
                alpha: alphaSample.alpha,
                recessShade: alphaSample.recessShade,
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
            let featherWidth: CGFloat = 4.75 / 192.0
            let outerCoverage = pow(Self.smoothStep(
                edge0: 0,
                edge1: featherWidth,
                value: edgeDistances.outer
            ), 0.96)
            let signedInnerDistance = edgeDistances.isInsideAperture ?
                edgeDistances.inner : -edgeDistances.inner
            let innerCoverage = excludesInnerMouth ? LipInnerBoundaryLayout.coverage(
                signedDistance: signedInnerDistance,
                opening: innerApertureVisibility
            ) : 1
            guard innerCoverage > 0.001 else {
                return nil
            }
            return CanonicalAlphaSample(
                alpha: Float(outerCoverage * innerCoverage),
                recessShade: excludesInnerMouth ? Float(
                    LipInnerBoundaryLayout.recessShade(
                        signedDistance: signedInnerDistance,
                        lipThickness: edgeDistances.outer + edgeDistances.inner,
                        opening: innerApertureVisibility
                    )
                ) : 1,
                outerDistance: edgeDistances.outer,
                innerDistance: edgeDistances.inner
            )
        }

        func screenPoint(u: CGFloat, topV: CGFloat) -> CGPoint? {
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

        private func edgeDistances(
            u: CGFloat,
            topV: CGFloat
        ) -> (outer: CGFloat, inner: CGFloat, isInsideAperture: Bool)? {
            let uv = CGPoint(x: max(0, min(u, 1)), y: 1 - max(0, min(topV, 1)))
            guard Self.pointInPolygon(uv, polygon: outerBoundary) else {
                return nil
            }
            let isInsideAperture = excludesInnerMouth &&
                Self.pointInPolygon(uv, polygon: innerBoundary)

            let outerDistanceSquared = outerBoundarySegments.reduce(CGFloat.greatestFiniteMagnitude) { current, segment in
                min(current, Self.squaredDistance(from: uv, to: segment))
            }
            let innerDistanceSquared = innerBoundarySegments.reduce(CGFloat.greatestFiniteMagnitude) { current, segment in
                min(current, Self.squaredDistance(from: uv, to: segment))
            }
            return (
                sqrt(outerDistanceSquared),
                sqrt(innerDistanceSquared),
                isInsideAperture
            )
        }

        func innerSamplingFringeCoverage(
            u: CGFloat,
            topV: CGFloat
        ) -> CGFloat? {
            guard excludesInnerMouth else {
                return nil
            }
            let uv = CGPoint(
                x: max(0, min(u, 1)),
                y: 1 - max(0, min(topV, 1))
            )
            guard Self.pointInPolygon(uv, polygon: innerBoundary) else {
                return nil
            }
            let innerDistanceSquared = innerBoundarySegments.reduce(
                CGFloat.greatestFiniteMagnitude
            ) { current, segment in
                min(current, Self.squaredDistance(from: uv, to: segment))
            }
            let coverage = LipInnerBoundaryLayout.coverage(
                signedDistance: sqrt(innerDistanceSquared),
                opening: innerApertureVisibility
            )
            return coverage > 0.001 ? coverage : nil
        }

        func shimmerParticleInnerGate(u: CGFloat,
                                      topV: CGFloat,
                                      pixelWidth: Int,
                                      pixelHeight: Int) -> Float {
            guard excludesInnerMouth,
                  innerApertureVisibility > 0.02,
                  innerBoundary.count >= 3 else {
                return 1
            }
            let uv = CGPoint(x: u, y: 1 - topV)
            let opening = Self.smoothStep(
                edge0: 0.02, edge1: 0.25, value: innerApertureVisibility
            )
            guard !Self.pointInPolygon(uv, polygon: innerBoundary) else {
                return Float(1 - opening)
            }
            let point = CGPoint(x: u * CGFloat(pixelWidth),
                                y: topV * CGFloat(pixelHeight))
            let distanceSquared = innerBoundarySegments.reduce(
                CGFloat.greatestFiniteMagnitude
            ) { current, segment in
                let first = CGPoint(x: segment.0.x * CGFloat(pixelWidth),
                                    y: (1 - segment.0.y) * CGFloat(pixelHeight))
                let second = CGPoint(x: segment.1.x * CGFloat(pixelWidth),
                                     y: (1 - segment.1.y) * CGFloat(pixelHeight))
                return min(current, Self.squaredDistance(
                    from: point, to: (first, second)
                ))
            }
            let scale = CGFloat(pixelHeight) /
                LipInnerBoundaryLayout.referenceTextureHeight
            let edgeFade = Self.smoothStep(
                edge0: 2 * scale,
                edge1: 6 * scale,
                value: sqrt(distanceSquared)
            )
            return Float(1 - opening * (1 - edgeFade))
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

    var usesDynamicCameraReflections: Bool {
        let finish = currentLipstickStyle().finish
        return finish == .gloss || finish == .oil || finish == .satin
    }

    @discardableResult
    func updateStyle(color: LipstickColor,
                     finish: LipFinish,
                     density: LipstickDensity,
                     texture: LipstickTexture,
                     materialProfile: LipMaterialProfile,
                     shadingDebugMode: LipShadingDebugMode) -> Bool {
        colorLock.lock()
        let sourceColor = RGBColor(red: color.red, green: color.green, blue: color.blue)
        let sourceLinearLuminance = Self.luminance(
            Self.sRGBToLinear(sourceColor)
        )
        // Every finish now uses the Banuba colour model: catalogue hue and
        // saturation plus the value of each captured camera pixel. Keep the
        // source colour intact so pale shades are not brightened twice.
        let nextColor = sourceColor
        let colorDistance = Self.colorDistance(lipstickColor, nextColor)
        let nextSignature = Self.colorSignature(nextColor)
        let didChange = lipstickStyleSignature != nextSignature ||
            lipstickFinish != finish ||
            lipstickDensity != density ||
            lipstickTexture != texture ||
            lipstickMaterialProfile != materialProfile ||
            lipstickShadingDebugMode != shadingDebugMode
        if didChange {
            lipstickColor = nextColor
            lipstickFinish = finish
            lipstickDensity = density
            lipstickTexture = texture
            lipstickMaterialProfile = materialProfile
            lipstickShadingDebugMode = shadingDebugMode
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
            "lip_color update distance=\(String(format: "%.4f", Double(colorDistance))) finish=\(finish.rawValue) density=\(density.rawValue) texture=\(texture.rawValue) sourceRGB=(\(String(format: "%.3f", Double(sourceColor.red))),\(String(format: "%.3f", Double(sourceColor.green))),\(String(format: "%.3f", Double(sourceColor.blue)))) renderRGB=(\(String(format: "%.3f", Double(nextColor.red))),\(String(format: "%.3f", Double(nextColor.green))),\(String(format: "%.3f", Double(nextColor.blue)))) linearLuma=\(String(format: "%.3f", Double(sourceLinearLuminance)))"
        )
        return true
    }

    func resetTemporalState(preservingProceduralTexture: Bool = false) {
        metalCompositor.resetTemporalState()
        temporalTextureLock.lock()
        previousToneMap = nil
        previousShineConfidence = nil
        previousToneMapWidth = 0
        previousToneMapHeight = 0
        previousToneColor = nil
        previousToneMouthOpen = nil
        previousToneAverageLuma = nil
        previousToneReferenceLuma = nil
        previousLinerReferenceLuma = nil
        previousLinerReferenceTime = nil
        if !preservingProceduralTexture {
            cachedProceduralTextureKey = nil
            cachedProceduralTexture = nil
        }
        temporalTextureLock.unlock()
    }

    // Called on completionQueue with at most one texture request in flight.
    func makeTexture(contour: LipContour,
                     pixelBuffer: CVPixelBuffer,
                     imageToViewportTransform: CGAffineTransform,
                     linerStyle: LinerRenderStyle,
                     imageSize: CGSize,
                     viewportSize: CGSize,
                     renderScale: CGFloat,
                     excludesInnerMouth: Bool,
                     lightingFactor: CGFloat,
                     viewYawDegrees: CGFloat,
                     viewPitchDegrees: CGFloat,
                     lowLatency: Bool = false,
                     motionDelta: CGFloat = 0,
                     completionQueue: DispatchQueue,
                     completion: @escaping (LipTexture?) -> Void) {
        guard viewportSize.width > 1, viewportSize.height > 1,
              imageSize.width > 1, imageSize.height > 1,
              LipCameraFrameMapping.isUsable(imageToViewportTransform) else {
            completion(nil)
            return
        }
        let metalWasAvailable = metalCompositor.isAvailable
        func cpuFallback() -> LipTexture? {
            makeCPUTexture(contour: contour, pixelBuffer: pixelBuffer,
                           imageToViewportTransform: imageToViewportTransform,
                           linerStyle: linerStyle,
                           imageSize: imageSize, viewportSize: viewportSize,
                           renderScale: renderScale, excludesInnerMouth: excludesInnerMouth,
                           lightingFactor: lightingFactor,
                           viewYawDegrees: viewYawDegrees,
                           viewPitchDegrees: viewPitchDegrees,
                           lowLatency: lowLatency,
                           motionDelta: motionDelta, metalWasAvailable: metalWasAvailable)
        }
        guard metalWasAvailable else { completion(cpuFallback()); return }
        let style = currentLipstickStyle()
        let width = lowLatency ? 224 : 256
        let height = lowLatency ? 112 : 128
        metalCompositor.makeTexture(
            contour: contour, pixelBuffer: pixelBuffer,
            imageToViewportTransform: imageToViewportTransform,
            linerStyle: linerStyle,
            imageSize: imageSize,
            viewportSize: viewportSize, pixelWidth: width, pixelHeight: height,
            renderScale: renderScale, color: style.color, lightingFactor: lightingFactor,
            finish: style.finish, materialProfile: style.materialProfile,
            shadingDebugMode: style.shadingDebugMode, lowLatency: lowLatency,
            motionDelta: motionDelta,
            viewYawDegrees: viewYawDegrees,
            viewPitchDegrees: viewPitchDegrees,
            completionQueue: completionQueue
        ) { [self] texture in
            if let texture {
                LipDebugLog.throttled(
                    "lip_texture_path", interval: 1,
                    "lip_texture path=metal fallback=false async=true size=\(width)x\(height)"
                )
                completion(addingLinerMask(to: texture, contour: contour,
                    renderScale: renderScale, enabled: linerStyle.isEnabled))
            } else if !metalCompositor.isAvailable {
                completion(cpuFallback())
            } else {
                // Keep the previous texture through transient GPU failures.
                completion(nil)
            }
        }
    }

    private func makeCPUTexture(contour: LipContour,
                     pixelBuffer: CVPixelBuffer,
                     imageToViewportTransform: CGAffineTransform,
                     linerStyle: LinerRenderStyle,
                     imageSize: CGSize,
                     viewportSize: CGSize,
                     renderScale: CGFloat,
                     excludesInnerMouth: Bool,
                     lightingFactor: CGFloat,
                     viewYawDegrees: CGFloat,
                     viewPitchDegrees: CGFloat,
                     lowLatency: Bool,
                     motionDelta: CGFloat,
                     metalWasAvailable: Bool) -> LipTexture? {
        let pixelWidth = lowLatency ? 128 : 160
        let pixelHeight = lowLatency ? 64 : 80
        let style = currentLipstickStyle()
        guard let sampler = CanonicalLipSampler(contour: contour, excludesInnerMouth: excludesInnerMouth) else {
            LipDebugLog.throttled(
                "lip_texture_no_sampler",
                "lip_texture nil reason=no_sampler outer=\(contour.outer.count) inner=\(contour.inner.count) outerUV=\(contour.outerUV.count) innerUV=\(contour.innerUV.count)"
            )
            return nil
        }
        let opticalSurface = LipOpticalSurface(contour: contour)

        if Self.usesStableProceduralLipstickRenderer {
            LipDebugLog.throttled(
                "lip_texture_path",
                interval: 1.0,
                "lip_texture path=cpu_procedural fallback=true reason=\(metalWasAvailable ? "metal_disabled_after_failure" : "metal_unavailable") size=256x128 lowLatency=\(lowLatency) finish=\(style.finish.rawValue)"
            )
            guard let texture = makeStableProceduralTexture(
                sampler: sampler,
                style: style,
                renderScale: renderScale,
                excludesInnerMouth: excludesInnerMouth
            ) else {
                return nil
            }
            return addingLinerMask(
                to: texture,
                contour: contour,
                renderScale: renderScale,
                enabled: linerStyle.isEnabled
            )
        }

        LipDebugLog.throttled(
            "lip_texture_path",
            interval: 1.0,
            "lip_texture path=cpu_camera fallback=true reason=\(metalWasAvailable ? "metal_disabled_after_failure" : "metal_unavailable") size=\(pixelWidth)x\(pixelHeight) lowLatency=\(lowLatency) finish=\(style.finish.rawValue)"
        )

        var rgba = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
        var alphaPixels = 0
        var paintedPixels = 0
        var materialInputs = [PixelMaterialInput?](repeating: nil, count: pixelWidth * pixelHeight)
        var toneMap = [CameraMaterialSignals](
            repeating: .neutral,
            count: pixelWidth * pixelHeight
        )
        var activeTonePixels = [Bool](repeating: false, count: pixelWidth * pixelHeight)
        var highlightMotionMask: [Float]? =
            LipHighlightMotionDiagnostics.shouldSample(at: CACurrentMediaTime()) ?
                [Float](repeating: 0, count: pixelWidth * pixelHeight) : nil
        var cameraLumaSum: Float = 0
        var cameraLumaCount: Float = 0
        let legacyInverse = Self.aspectFillTransform(for: imageSize,
                                                     in: viewportSize).inverted()
        let usesCameraHighlight = style.finish == .gloss ||
            style.finish == .oil || style.finish == .satin
        let inverseImageTransform = usesCameraHighlight ?
            imageToViewportTransform.inverted() : legacyInverse
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
                let mapIndex = y * pixelWidth + x
                let wetSurface = usesCameraHighlight ?
                    opticalSurface?.sample(at: sample.point) : nil

                let cameraSample: CameraLipSample?
                if let baseAddress {
                    cameraSample = Self.cameraLipSample(
                        atViewportPoint: sample.point,
                        inverseImageTransform: inverseImageTransform,
                        baseAddress: baseAddress,
                        bytesPerRow: bytesPerRow,
                        width: sourceWidth,
                        height: sourceHeight,
                        wetSurface: wetSurface,
                        referenceGloss: style.finish == .gloss,
                        imageToViewportTransform: imageToViewportTransform,
                        mouthWidth: Float(max(contour.pose?.width ?? 1, 1)),
                        profile: style.materialProfile
                    )
                } else {
                    cameraSample = nil
                }

                let edgeNoise = Self.edgeNoise(maskAlpha: sample.alpha, x: x, y: y, frame: .zero)
                let pigmentVariation = Self.stableLipPigmentVariation(u: u, topV: topV)
                let trustedCameraSample = cameraSample
                let lipBase = trustedCameraSample?.base ?? Self.stableLipBaseColor(u: u, topV: topV)
                let pigmentDistribution = style.finish == .tint ?
                    Self.tintDistribution(
                        outerDistance: sample.outerDistance,
                        innerDistance: sample.innerDistance
                    ) : 1
                let baseAlpha = min(
                    sample.alpha *
                    edgeNoise *
                    pigmentVariation.alpha,
                    1
                )
                let mouthCameraCoverage: Float
                if trustedCameraSample != nil,
                   style.shadingDebugMode != .capturedShape {
                    mouthCameraCoverage = Self.innerMouthCameraCoverage(
                        base: lipBase,
                        topV: topV,
                        opening: sampler.innerApertureVisibility
                    )
                } else {
                    mouthCameraCoverage = 1
                }
                let alpha = baseAlpha * mouthCameraCoverage
                let pigmentRGB = Self.sRGBToLinear(style.color)
                if let cameraSample = trustedCameraSample {
                    toneMap[mapIndex] = Self.cameraTone(
                        base: cameraSample.base,
                        blurred: cameraSample.blurred,
                        highlightSurround: cameraSample.highlightSurround,
                        highlightConfidence: usesCameraHighlight ?
                            cameraSample.highlightConfidence : 1,
                        highlightThreshold: usesCameraHighlight ?
                            style.materialProfile.cameraHighlightThreshold : 1,
                        finish: style.finish,
                        microTextureRetention:
                            style.materialProfile.microTextureRetention
                    )
                    let cameraLuminance = Self.luminance(cameraSample.blurred)
                    cameraLumaSum += cameraLuminance
                    cameraLumaCount += 1
                }
                activeTonePixels[mapIndex] = true
                let shineGeometry = sampler.shineGeometry(u: u, topV: topV)
                materialInputs[mapIndex] = PixelMaterialInput(
                    alpha: alpha,
                    pigmentDistribution: pigmentDistribution,
                    recessShade: sample.recessShade,
                    lipBase: lipBase,
                    highlightSurround:
                        trustedCameraSample?.highlightSurround ?? lipBase,
                    pigment: pigmentRGB,
                    topV: topV,
                    normalizedLipX: shineGeometry.normalizedX,
                    lowerLipMembership: shineGeometry.lowerLipMembership,
                    outerDistance: sample.outerDistance,
                    innerDistance: sample.innerDistance,
                    opticalSurface: wetSurface
                )
            }
        }

        let averageCameraLuma = cameraLumaCount > 0 ? cameraLumaSum / cameraLumaCount : nil
        let stabilizedToneMap = temporallyStabilizedToneMap(
            toneMap,
            activePixels: activeTonePixels,
            finish: style.finish,
            materialProfile: style.materialProfile,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            color: style.color,
            excludesInnerMouth: excludesInnerMouth,
            averageCameraLuma: averageCameraLuma,
            lowLatency: lowLatency,
            motionDelta: motionDelta
        )
        let shimmerYawRadians = viewYawDegrees.isFinite ?
            Float(min(max(viewYawDegrees, 0), 65) * .pi / 180) : 0
        let shimmerPitchRadians = viewPitchDegrees.isFinite ?
            Float(min(max(viewPitchDegrees, 0), 65) * .pi / 180) : 0
        for mapIndex in materialInputs.indices {
            guard let input = materialInputs[mapIndex] else {
                continue
            }
            let effectiveAlpha = input.alpha
            guard effectiveAlpha > 0.006 else {
                continue
            }
            let appearance = Self.materialAppearance(
                pigment: input.pigment,
                base: input.lipBase,
                highlightSurround: input.highlightSurround,
                signals: stabilizedToneMap[mapIndex],
                topV: input.topV,
                normalizedLipX: input.normalizedLipX,
                lowerLipMembership: input.lowerLipMembership,
                outerDistance: input.outerDistance,
                innerDistance: input.innerDistance,
                opticalSurface: input.opticalSurface,
                cameraDetailCoherence: style.finish == .gloss ? 1 :
                    max(0.18, 1 - Float(max(0, min(motionDelta / 0.22, 1))) * 0.82),
                apertureClosure: Float(1 - sampler.innerApertureVisibility),
                finish: style.finish,
                materialProfile: style.materialProfile,
                shadingDebugMode: style.shadingDebugMode
            )
            let pigmentCoverage = style.shadingDebugMode == .capturedShape ? 1 :
                style.materialProfile.pigmentCoverage * input.pigmentDistribution
            let reflectionCoverage = appearance.reflectionCoverage
            highlightMotionMask?[mapIndex] = reflectionCoverage
            let baseCoverage = reflectionCoverage +
                pigmentCoverage * (1 - reflectionCoverage)
            // The reflected layer is over the pigment. Its premultiplied
            // contribution does not contain the pigment density. Apply the
            // common lip mask after combining the two material layers.
            let basePremultiplied = RGBColor(
                red: appearance.reflection.red * reflectionCoverage +
                    appearance.pigment.red * pigmentCoverage * (1 - reflectionCoverage),
                green: appearance.reflection.green * reflectionCoverage +
                    appearance.pigment.green * pigmentCoverage * (1 - reflectionCoverage),
                blue: appearance.reflection.blue * reflectionCoverage +
                    appearance.pigment.blue * pigmentCoverage * (1 - reflectionCoverage)
            )
            var materialCoverage = baseCoverage
            var materialPremultiplied = basePremultiplied
            if style.finish == .shimmer && style.shadingDebugMode == .off {
                let layer = Self.shimmerLayer(
                    materialLuminance: Self.luminance(appearance.pigment),
                    reflectionCoverage: reflectionCoverage,
                    u: (Float(mapIndex % pixelWidth) + 0.5) / Float(pixelWidth),
                    v: (Float(mapIndex / pixelWidth) + 0.5) / Float(pixelHeight),
                    viewYawRadians: shimmerYawRadians,
                    viewPitchRadians: shimmerPitchRadians,
                    intensity: style.materialProfile.shimmerIntensity,
                    sampler: sampler,
                    pixelWidth: pixelWidth,
                    pixelHeight: pixelHeight
                )
                materialCoverage = layer.coverage +
                    baseCoverage * (1 - layer.coverage)
                materialPremultiplied = RGBColor(
                    red: layer.color.red * layer.coverage +
                        basePremultiplied.red * (1 - layer.coverage),
                    green: layer.color.green * layer.coverage +
                        basePremultiplied.green * (1 - layer.coverage),
                    blue: layer.color.blue * layer.coverage +
                        basePremultiplied.blue * (1 - layer.coverage)
                )
            }
            let outputAlpha = effectiveAlpha * materialCoverage
            let materialColor = RGBColor(
                red: materialPremultiplied.red / max(materialCoverage, 0.0001),
                green: materialPremultiplied.green / max(materialCoverage, 0.0001),
                blue: materialPremultiplied.blue / max(materialCoverage, 0.0001)
            )
            let linearPremultipliedColor = RGBColor(
                red: materialColor.red * outputAlpha * input.recessShade,
                green: materialColor.green * outputAlpha * input.recessShade,
                blue: materialColor.blue * outputAlpha * input.recessShade
            )
            let storedColor = Self.sRGBPremultipliedColor(
                fromLinearPremultiplied: linearPremultipliedColor,
                alpha: outputAlpha
            )
            let offset = mapIndex * 4
            rgba[offset] = Self.uint8(storedColor.red)
            rgba[offset + 1] = Self.uint8(storedColor.green)
            rgba[offset + 2] = Self.uint8(storedColor.blue)
            rgba[offset + 3] = Self.uint8(outputAlpha)
            paintedPixels += 1
        }

        var usedFallback = false
        var fallbackPaintedPixels = 0
        if paintedPixels < max(64, pixelWidth * pixelHeight / 80) {
            LipDebugLog.throttled(
                "lip_texture_emergency_fallback",
                interval: 1.0,
                "lip_texture emergency_fallback=true reason=low_painted_pixels painted=\(paintedPixels) threshold=\(max(64, pixelWidth * pixelHeight / 80)) size=\(pixelWidth)x\(pixelHeight) finish=\(style.finish.rawValue)"
            )
            let fallback = Self.makeFallbackRGBA(
                sampler: sampler,
                color: style.color,
                finish: style.finish,
                density: style.density,
                texture: style.texture,
                materialProfile: style.materialProfile,
                shadingDebugMode: style.shadingDebugMode,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
            )
            rgba = fallback.rgba
            usedFallback = true
            fallbackPaintedPixels = fallback.paintedPixels
        }
        if let highlightMotionMask {
            LipHighlightMotionDiagnostics.record(
                usedFallback ? [Float](repeating: 0, count: highlightMotionMask.count) :
                    highlightMotionMask,
                width: pixelWidth, height: pixelHeight,
                finish: style.finish, source: "cpu", at: CACurrentMediaTime()
            )
        }
        let innerFeatherPaintedPixels = Self.fillInnerSamplingFeather(
            &rgba,
            sampler: sampler,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight,
            sampledPixels: usedFallback ? nil : activeTonePixels
        )
        // CanonicalAlphaSample already supplies a smooth distance feather.
        // A second CPU blur both delayed the live colour and washed out the
        // captured lip detail we are deliberately preserving.

        LipDebugLog.throttled(
            "lip_texture_stats",
            interval: 0.6,
            "lip_texture stats size=\(pixelWidth)x\(pixelHeight) lowLatency=\(lowLatency) cameraSample=\(baseAddress != nil) alphaPixels=\(alphaPixels) painted=\(paintedPixels) innerFeather=\(innerFeatherPaintedPixels) fallback=\(usedFallback) fallbackPainted=\(fallbackPaintedPixels) image=\(Int(imageSize.width))x\(Int(imageSize.height)) viewport=\(Int(viewportSize.width))x\(Int(viewportSize.height))"
        )

        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cgImage = CGImage(
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB) ??
                    CGColorSpaceCreateDeviceRGB(),
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
        return addingLinerMask(
            to: LipTexture(image: texture.image,
                linerColorImage: linerStyle.isEnabled ? makeCPULinerColorImage(
                    contour: contour, sampler: sampler,
                    imageSize: imageSize, viewportSize: viewportSize,
                    pixelWidth: pixelWidth, pixelHeight: pixelHeight,
                    renderScale: renderScale, style: linerStyle,
                    baseAddress: baseAddress, bytesPerRow: bytesPerRow,
                    sourceWidth: sourceWidth, sourceHeight: sourceHeight) : nil),
            contour: contour,
            renderScale: renderScale,
            enabled: linerStyle.isEnabled
        )
    }

    private func addingLinerMask(to texture: LipTexture,
                                 contour: LipContour,
                                 renderScale: CGFloat,
                                 enabled: Bool) -> LipTexture {
        guard enabled, texture.linerColorImage != nil else { return texture }
        guard let cgImage = texture.image.cgImage else {
            return texture
        }
        let key = LinerMaskKey(
            pixelWidth: cgImage.width,
            pixelHeight: cgImage.height,
            renderScaleBucket: Int((renderScale * 100).rounded()),
            outerUV: contour.outerUV,
            innerUV: contour.innerUV
        )

        temporalTextureLock.lock()
        let cachedImage = cachedLinerMaskKey == key ?
            cachedLinerOpacityImage : nil
        temporalTextureLock.unlock()

        let linerOpacityImage: UIImage
        if let cachedImage {
            linerOpacityImage = cachedImage
        } else {
            guard let renderedImage = Self.makeLinerOpacityImage(
                contour: contour,
                pixelWidth: cgImage.width,
                pixelHeight: cgImage.height,
                renderScale: renderScale
            ) else {
                return texture
            }
            temporalTextureLock.lock()
            cachedLinerMaskKey = key
            cachedLinerOpacityImage = renderedImage
            temporalTextureLock.unlock()
            linerOpacityImage = renderedImage
        }
        return LipTexture(
            image: texture.image,
            opacityImage: texture.opacityImage,
            linerColorImage: texture.linerColorImage,
            linerOpacityImage: linerOpacityImage
        )
    }

    private func makeCPULinerColorImage(
        contour: LipContour,
        sampler: CanonicalLipSampler,
        imageSize: CGSize,
        viewportSize: CGSize,
        pixelWidth: Int,
        pixelHeight: Int,
        renderScale: CGFloat,
        style: LinerRenderStyle,
        baseAddress: UnsafeMutableRawPointer?,
        bytesPerRow: Int,
        sourceWidth: Int,
        sourceHeight: Int
    ) -> UIImage? {
        guard let baseAddress else { return nil }
        let inverse = Self.aspectFillTransform(for: imageSize,
            in: viewportSize).inverted()
        let cameraSampler = CanonicalLipSampler(contour: contour,
            excludesInnerMouth: true, expandsCameraCarrier: true)
        let expandedScreen = LipOuterBoundary.expandedScreenPoints(
            contour: contour,
            distance: LipOuterMargin.sampling(for: contour.pose?.width ?? 0)
        ) ?? [:]
        let sourcePoints = CanonicalLipGeometry.outerLipIndices.enumerated().compactMap {
            offset, index -> CGPoint? in
            guard contour.outer.indices.contains(offset) else { return nil }
            return (expandedScreen[index] ?? contour.outer[offset]).applying(inverse)
        }
        let xs = sourcePoints.map(\.x)
        let ys = sourcePoints.map(\.y)
        let radiusX = min(max(((xs.max() ?? 0) - (xs.min() ?? 0)) * 0.012, 2), 10)
        let radiusY = min(max(((ys.max() ?? 0) - (ys.min() ?? 0)) * 0.032, 2), 10)

        func tone(at point: CGPoint) -> (Float, Float, Float, RGBColor)? {
            let imagePoint = point.applying(inverse)
            func color(_ offset: CGPoint) -> RGBColor? {
                guard let sampled = Self.samplePixel(
                    at: CGPoint(x: imagePoint.x + offset.x,
                                y: imagePoint.y + offset.y),
                    baseAddress: baseAddress, bytesPerRow: bytesPerRow,
                    width: sourceWidth, height: sourceHeight) else { return nil }
                return Self.sRGBToLinear(sampled)
            }
            guard let center = color(.zero) else { return nil }
            let neighbors = [CGPoint(x: radiusX, y: 0),
                CGPoint(x: -radiusX, y: 0),
                CGPoint(x: 0, y: radiusY),
                CGPoint(x: 0, y: -radiusY)]
            let adjacent = neighbors.map { color($0) ?? center }
            let lowColor = RGBColor(
                red: (center.red + adjacent.reduce(Float(0)) { $0 + $1.red }) / 5,
                green: (center.green + adjacent.reduce(Float(0)) { $0 + $1.green }) / 5,
                blue: (center.blue + adjacent.reduce(Float(0)) { $0 + $1.blue }) / 5
            )
            let residual = RGBColor(red: max(center.red - lowColor.red, 0),
                green: max(center.green - lowColor.green, 0),
                blue: max(center.blue - lowColor.blue, 0))
            let highest = max(residual.red, max(residual.green, residual.blue))
            let lowest = min(residual.red, min(residual.green, residual.blue))
            let neutral = Self.smoothStep(
                (lowest / max(highest, 0.0001) - 0.25) / 0.40)
            return (max(Self.luminance(center), 0.0031),
                max(Self.luminance(lowColor), 0.0031), neutral, center)
        }

        var references: [Float] = []
        for topV in [CGFloat(0.24), 0.30, 0.70, 0.76] {
            for u in [CGFloat(0.24), 0.38, 0.50, 0.62, 0.76] {
                if let sample = sampler.sample(u: u, topV: topV),
                   sample.alpha > 0.8,
                   let measured = tone(at: sample.point) {
                    references.append(measured.1)
                }
            }
        }
        references.sort()
        let measuredReference = references.count >= 8 ?
            references[Int(Float(references.count - 1) * 0.65)] : nil
        let now = CACurrentMediaTime()
        temporalTextureLock.lock()
        let reference: Float
        if let measuredReference, measuredReference.isFinite,
           measuredReference > 0.0045 {
            if let previous = previousLinerReferenceLuma,
               let timestamp = previousLinerReferenceTime {
                let elapsed = max(0, min(now - timestamp, 0.25))
                let stops = abs(log2(measuredReference / max(previous, 0.0045)))
                let response: Double = stops > 0.70 ? 8 : 3
                let weight = Float(1 - exp(-elapsed * response))
                reference = previous + (measuredReference - previous) * weight
            } else {
                reference = measuredReference
            }
            previousLinerReferenceLuma = reference
            previousLinerReferenceTime = now
        } else {
            reference = previousLinerReferenceLuma ?? 0
        }
        temporalTextureLock.unlock()

        let pigment = Self.sRGBToLinear(style.color)
        let pigmentY = Self.luminance(pigment)
        let pale = Self.smoothStep((pigmentY - 0.16) / 0.22)
        let lightLimit = style.profile.maximumLightStops * (1 - pale) +
            min(style.profile.maximumLightStops, 0.38) * pale
        var rgba = [UInt8](repeating: 255, count: pixelWidth * pixelHeight * 4)
        for y in 0..<pixelHeight {
            for x in 0..<pixelWidth {
                let u = (CGFloat(x) + 0.5) / CGFloat(pixelWidth)
                let v = (CGFloat(y) + 0.5) / CGFloat(pixelHeight)
                let source = cameraSampler?.screenPoint(u: u, topV: v)
                var shaded = pigment
                if let source, let measured = tone(at: source) {
                    let cameraAwarePigment = Self.banubaBaseColor(
                        cameraLinear: measured.3,
                        pigmentLinear: pigment,
                        showsValueDebug: false
                    )
                    let localReference = reference > 0.0045 ? reference : measured.1
                    let low = min(max(measured.1, localReference * 0.58),
                        localReference * 1.55)
                    let exposure = min(max(log2(low / max(localReference, 0.0045)),
                        -style.profile.maximumShadowStops), lightLimit)
                    let response = exposure < 0 ? style.profile.shadowRetention :
                        style.profile.diffuseLightResponse
                    let detail = min(max(log2(measured.0 / low), -0.24), 0.18)
                    let factor = pow(Float(2), exposure * response +
                        detail * style.profile.detailStrength)
                    shaded = RGBColor(
                        red: Self.clamp01(cameraAwarePigment.red * factor),
                        green: Self.clamp01(cameraAwarePigment.green * factor),
                        blue: Self.clamp01(cameraAwarePigment.blue * factor)
                    )
                    if style.profile.realHighlightStrength > 0 {
                        let observed = Self.smoothStep(
                            (measured.0 - low - 0.03) / 0.13) * measured.2
                        let shine = Self.clamp01(
                            observed * style.profile.realHighlightStrength)
                        shaded = RGBColor(
                            red: shaded.red + (1 - shaded.red) * shine,
                            green: shaded.green + (1 - shaded.green) * shine,
                            blue: shaded.blue + (1 - shaded.blue) * shine
                        )
                    }
                }
                let srgb = Self.linearToSRGB(shaded)
                let offset = (y * pixelWidth + x) * 4
                rgba[offset] = UInt8((Self.clamp01(srgb.red) * 255).rounded())
                rgba[offset + 1] = UInt8((Self.clamp01(srgb.green) * 255).rounded())
                rgba[offset + 2] = UInt8((Self.clamp01(srgb.blue) * 255).rounded())
            }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let image = CGImage(width: pixelWidth, height: pixelHeight,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB) ??
                    CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: true,
                intent: .defaultIntent) else { return nil }
        return UIImage(cgImage: image, scale: renderScale, orientation: .up)
    }

    private enum LinerMaskProfile {
        static let inwardFeatherPixelsAt112: CGFloat = 20
        static let outwardEdgePixelsAt112: CGFloat = 0.8
        static let lowerInwardFeatherPixelsAt112: CGFloat = 9
        static let lowerOutwardEdgePixelsAt112: CGFloat = 2.4
        static let lowerReferenceOutsetPixelsAt112: CGFloat = 2.6
        static let cornerFadePixelsAt112: CGFloat = 14
        // Density remains in LipMeshRenderer's material opacity and UI slider.
    }

    private static let linerSamplesPerSegment = 8

    /// One detected outer reference, with independent inward and outward alpha falloffs.
    private static func makeLinerOpacityImage(contour: LipContour,
                                              pixelWidth: Int,
                                              pixelHeight: Int,
                                              renderScale: CGFloat) -> UIImage? {
        guard let outerSamples = linerReferencePixels(
                contour: contour, pixelWidth: pixelWidth, pixelHeight: pixelHeight
              ),
              let innerSamples = sampleLinerBoundary(
                contour.innerUV, pixelWidth: pixelWidth, pixelHeight: pixelHeight
              ) else {
            return nil
        }

        let outerPath = CGMutablePath()
        outerPath.addLines(between: outerSamples)
        outerPath.closeSubpath()
        let innerPath = CGMutablePath()
        innerPath.addLines(between: innerSamples)
        innerPath.closeSubpath()

        typealias Segment = (start: CGPoint, dx: CGFloat, dy: CGFloat,
                             lengthSquared: CGFloat, arcStart: CGFloat)
        func segments(for samples: [CGPoint]) -> [Segment] {
            var result = [Segment]()
            result.reserveCapacity(samples.count - 1)
            var arcLength: CGFloat = 0
            for (first, second) in zip(samples, samples.dropFirst()) {
                let dx = second.x - first.x
                let dy = second.y - first.y
                let lengthSquared = dx * dx + dy * dy
                guard lengthSquared > 0.000_001 else { continue }
                result.append((first, dx, dy, lengthSquared, arcLength))
                arcLength += sqrt(lengthSquared)
            }
            return result
        }
        func closestPoint(from point: CGPoint,
                          on segments: [Segment]) -> (distanceSquared: CGFloat,
                                                     arcLength: CGFloat) {
            var nearest = CGFloat.infinity
            var nearestArc: CGFloat = 0
            for segment in segments {
                let t = min(max(((point.x - segment.start.x) * segment.dx +
                                 (point.y - segment.start.y) * segment.dy) /
                                segment.lengthSquared, 0), 1)
                let dx = point.x - segment.start.x - t * segment.dx
                let dy = point.y - segment.start.y - t * segment.dy
                let distanceSquared = dx * dx + dy * dy
                if distanceSquared < nearest {
                    nearest = distanceSquared
                    nearestArc = segment.arcStart + t * sqrt(segment.lengthSquared)
                }
            }
            return (nearest, nearestArc)
        }
        let outerSegments = segments(for: outerSamples)
        let innerSegments = segments(for: innerSamples)
        guard let lastOuterSegment = outerSegments.last,
              !innerSegments.isEmpty else { return nil }

        let scale = CGFloat(pixelHeight) / 112
        let inwardFeather = LinerMaskProfile.inwardFeatherPixelsAt112 * scale
        let outwardEdge = LinerMaskProfile.outwardEdgePixelsAt112 * scale
        let lowerInwardFeather =
            LinerMaskProfile.lowerInwardFeatherPixelsAt112 * scale
        let lowerOutwardEdge =
            LinerMaskProfile.lowerOutwardEdgePixelsAt112 * scale
        let maximumOutwardEdge = max(outwardEdge, lowerOutwardEdge)
        let cornerFadeDistance = LinerMaskProfile.cornerFadePixelsAt112 * scale
        let rightCornerArc = outerSegments
            .prefix(10 * linerSamplesPerSegment)
            .reduce(CGFloat.zero) { $0 + sqrt($1.lengthSquared) }
        let totalArc = lastOuterSegment.arcStart +
            sqrt(lastOuterSegment.lengthSquared)
        let xValues = outerSamples.map(\.x)
        let yValues = outerSamples.map(\.y)
        let minX = max(Int(floor((xValues.min() ?? 0) - maximumOutwardEdge - 1)), 0)
        let maxX = min(Int(ceil((xValues.max() ?? 0) + maximumOutwardEdge + 1)), pixelWidth - 1)
        let minY = max(Int(floor((yValues.min() ?? 0) - maximumOutwardEdge - 1)), 0)
        let maxY = min(Int(ceil((yValues.max() ?? 0) + maximumOutwardEdge + 1)), pixelHeight - 1)
        guard minX <= maxX, minY <= maxY else { return nil }

        func smoothStep(_ value: CGFloat) -> CGFloat {
            let t = min(max(value, 0), 1)
            return t * t * (3 - 2 * t)
        }

        var rgba = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
        for y in minY...maxY {
            for x in minX...maxX {
                let point = CGPoint(x: CGFloat(x) + 0.5, y: CGFloat(y) + 0.5)
                let outer = closestPoint(from: point, on: outerSegments)
                let inside = outerPath.contains(point)
                let lowerCornerDistance = min(
                    outer.arcLength, rightCornerArc - outer.arcLength)
                let lowerWeight = outer.arcLength <= rightCornerArc &&
                    rightCornerArc > 0 ? smoothStep(
                        lowerCornerDistance / cornerFadeDistance) : 0
                let currentInwardFeather = inwardFeather +
                    (lowerInwardFeather - inwardFeather) * lowerWeight
                let currentOutwardEdge = outwardEdge +
                    (lowerOutwardEdge - outwardEdge) * lowerWeight
                let innerEdgeFade = currentInwardFeather * 0.25
                let featherWidth = inside ?
                    currentInwardFeather : currentOutwardEdge
                guard outer.distanceSquared.isFinite,
                      outer.distanceSquared < featherWidth * featherWidth,
                      !innerPath.contains(point) else {
                    continue
                }

                let inner = closestPoint(from: point, on: innerSegments)
                guard inner.distanceSquared.isFinite else { continue }
                let edgeFade = smoothStep(
                    sqrt(inner.distanceSquared) / innerEdgeFade
                )
                let contourFade = 1 - smoothStep(
                    sqrt(outer.distanceSquared) / featherWidth
                )
                let distanceToCorner = min(
                    min(outer.arcLength,
                        abs(outer.arcLength - rightCornerArc)),
                    totalArc - outer.arcLength
                )
                let cornerFade = smoothStep(distanceToCorner / cornerFadeDistance)
                let alpha = edgeFade * contourFade * cornerFade
                let byte = UInt8((alpha * 255).rounded())
                let offset = (y * pixelWidth + x) * 4
                rgba[offset] = byte
                rgba[offset + 1] = byte
                rgba[offset + 2] = byte
                rgba[offset + 3] = byte
            }
        }

        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let image = CGImage(
                width: pixelWidth, height: pixelHeight,
                bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue:
                    CGImageAlphaInfo.premultipliedLast.rawValue)
                    .union(.byteOrder32Big),
                provider: provider, decode: nil, shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            return nil
        }
        return UIImage(cgImage: image, scale: renderScale, orientation: .up)
    }

    // The DEBUG overlay samples the upper part of this same full contour.
    static func upperLinerReferencePixels(contour: LipContour,
                                          pixelWidth: Int,
                                          pixelHeight: Int) -> [CGPoint]? {
        guard let allSamples = linerReferencePixels(
            contour: contour, pixelWidth: pixelWidth, pixelHeight: pixelHeight
        ) else {
            return nil
        }
        let first = 10 * linerSamplesPerSegment
        return Array(allSamples[first...])
    }

    private static func linerReferencePixels(contour: LipContour,
                                             pixelWidth: Int,
                                             pixelHeight: Int) -> [CGPoint]? {
        guard var samples = sampleLinerBoundary(contour.outerUV,
                pixelWidth: pixelWidth, pixelHeight: pixelHeight) else {
            return nil
        }
        let lowerSampleCount = 10 * linerSamplesPerSegment
        guard samples.count > lowerSampleCount + 1 else { return nil }
        let scale = CGFloat(pixelHeight) / 112
        let outset = LinerMaskProfile.lowerReferenceOutsetPixelsAt112 * scale
        let outwardSign: CGFloat = samples[0].x >=
            samples[lowerSampleCount].x ? 1 : -1
        let original = samples
        for offset in 1..<lowerSampleCount {
            let tangent = CGPoint(
                x: original[offset + 1].x - original[offset - 1].x,
                y: original[offset + 1].y - original[offset - 1].y)
            let length = hypot(tangent.x, tangent.y)
            guard length > 0.0001 else { continue }
            let cornerDistance = min(offset, lowerSampleCount - offset)
            let fade = min(CGFloat(cornerDistance) /
                CGFloat(2 * linerSamplesPerSegment), 1)
            let weight = fade * fade * (3 - 2 * fade)
            samples[offset] = CGPoint(
                x: original[offset].x +
                    tangent.y / length * outwardSign * outset * weight,
                y: original[offset].y -
                    tangent.x / length * outwardSign * outset * weight
            )
        }
        return samples
    }

    private static func sampleLinerBoundary(_ boundaryUV: [CGPoint],
                                            pixelWidth: Int,
                                            pixelHeight: Int) -> [CGPoint]? {
        guard boundaryUV.count == 20,
              pixelWidth > 1, pixelHeight > 1 else { return nil }
        let points = boundaryUV.map {
            CGPoint(x: $0.x * CGFloat(pixelWidth),
                    y: (1 - $0.y) * CGFloat(pixelHeight))
        }
        guard points.allSatisfy({ $0.x.isFinite && $0.y.isFinite }) else {
            return nil
        }

        let tangentScale: CGFloat = 0.88 / 6
        var samples = [CGPoint]()
        samples.reserveCapacity(points.count * linerSamplesPerSegment + 1)
        for index in points.indices {
            let previous = points[(index - 1 + points.count) % points.count]
            let current = points[index]
            let next = points[(index + 1) % points.count]
            let afterNext = points[(index + 2) % points.count]
            let chord = hypot(next.x - current.x, next.y - current.y)
            let limit = chord * 0.22
            func control(from endpoint: CGPoint, toward candidate: CGPoint) -> CGPoint {
                let dx = candidate.x - endpoint.x
                let dy = candidate.y - endpoint.y
                let length = hypot(dx, dy)
                let fraction = length > limit && length > 0 ? limit / length : 1
                return CGPoint(
                    x: min(max(endpoint.x + dx * fraction,
                               min(current.x, next.x)), max(current.x, next.x)),
                    y: min(max(endpoint.y + dy * fraction,
                               min(current.y, next.y)), max(current.y, next.y))
                )
            }
            let first = control(from: current, toward: CGPoint(
                x: current.x + (next.x - previous.x) * tangentScale,
                y: current.y + (next.y - previous.y) * tangentScale
            ))
            let second = control(from: next, toward: CGPoint(
                x: next.x - (afterNext.x - current.x) * tangentScale,
                y: next.y - (afterNext.y - current.y) * tangentScale
            ))
            for step in 0..<linerSamplesPerSegment {
                let t = CGFloat(step) / CGFloat(linerSamplesPerSegment)
                let complement = 1 - t
                let firstWeight = 3 * complement * complement * t
                let secondWeight = 3 * complement * t * t
                samples.append(CGPoint(
                    x: complement * complement * complement * current.x +
                        firstWeight * first.x + secondWeight * second.x + t * t * t * next.x,
                    y: complement * complement * complement * current.y +
                        firstWeight * first.y + secondWeight * second.y + t * t * t * next.y
                ))
            }
        }
        samples.append(points[0])
        return samples
    }

    private func makeStableProceduralTexture(sampler: CanonicalLipSampler,
                                             style: (color: RGBColor,
                                                     finish: LipFinish,
                                                     density: LipstickDensity,
                                                     texture: LipstickTexture,
                                                     materialProfile: LipMaterialProfile,
                                                     shadingDebugMode: LipShadingDebugMode),
                                             renderScale: CGFloat,
                                             excludesInnerMouth: Bool) -> LipTexture? {
        let key = ProceduralTextureKey(
            colorSignature: Self.colorSignature(style.color),
            finish: style.finish,
            density: style.density,
            texture: style.texture,
            materialProfile: style.materialProfile,
            shadingDebugMode: style.shadingDebugMode,
            renderScale: Int((renderScale * 100).rounded()),
            excludesInnerMouth: excludesInnerMouth,
            apertureVisibilityBucket: Int(
                (sampler.innerApertureVisibility * 20).rounded()
            )
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
            density: style.density,
            texture: style.texture,
            materialProfile: style.materialProfile,
            shadingDebugMode: style.shadingDebugMode,
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
                space: CGColorSpace(name: CGColorSpace.sRGB) ??
                    CGColorSpaceCreateDeviceRGB(),
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
                                         density: LipstickDensity,
                                         texture: LipstickTexture,
                                         materialProfile: LipMaterialProfile,
                                         shadingDebugMode: LipShadingDebugMode,
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
                // Coverage is geometry-driven except for the intentional,
                // shallow transparency at the extreme mouth corners.
                // A tiny deterministic alpha grain breaks the perfectly even
                // vector-like edge without introducing temporal shimmer.
                let edgeGrain = sample.alpha < 0.995 ?
                    0.965 + pigmentVariation.color * 0.035 : 1
                let renderCoverage = materialProfile.pigmentCoverage
                let tintDistribution = finish == .tint ?
                    Self.tintDistribution(
                        outerDistance: sample.outerDistance,
                        innerDistance: sample.innerDistance
                    ) : 1
                let alpha = sample.alpha *
                    edgeGrain *
                    renderCoverage *
                    tintDistribution
                let lipBase = stableLipBaseColor(u: u, topV: topV)
                let baseLipRGB = banubaBaseColor(
                    cameraLinear: lipBase,
                    pigmentLinear: sRGBToLinear(color),
                    showsValueDebug:
                        shadingDebugMode == .capturedShape
                )
                let lipRGB = RGBColor(
                    red: baseLipRGB.red * sample.recessShade,
                    green: baseLipRGB.green * sample.recessShade,
                    blue: baseLipRGB.blue * sample.recessShade
                )
                // Without a camera signal there is no defensible reflection
                // position. Keep fallback pigment diffuse rather than drawing
                // a fixed gloss pattern in canonical UV space.
                let offset = (y * pixelWidth + x) * 4
                let lipSRGB = linearToSRGB(lipRGB)
                rgba[offset] = uint8(lipSRGB.red * alpha)
                rgba[offset + 1] = uint8(lipSRGB.green * alpha)
                rgba[offset + 2] = uint8(lipSRGB.blue * alpha)
                rgba[offset + 3] = uint8(alpha)
                paintedPixels += 1
            }
        }
        paintedPixels += fillInnerSamplingFeather(
            &rgba,
            sampler: sampler,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
        )
        return (rgba, paintedPixels)
    }

    private static func fillInnerSamplingFeather(
        _ rgba: inout [UInt8],
        sampler: CanonicalLipSampler,
        pixelWidth: Int,
        pixelHeight: Int,
        sampledPixels: [Bool]? = nil
    ) -> Int {
        guard pixelWidth > 1,
              pixelHeight > 1,
              rgba.count == pixelWidth * pixelHeight * 4 else {
            return 0
        }

        let source = rgba
        var filledPixels = 0
        for y in 0..<pixelHeight {
            for x in 0..<pixelWidth {
                let targetOffset = (y * pixelWidth + x) * 4
                guard source[targetOffset + 3] == 0 else {
                    continue
                }
                // A sampled tooth pixel can intentionally round to zero alpha.
                // Only recover missing samples, never refill that exclusion.
                if let sampledPixels,
                   sampledPixels.indices.contains(y * pixelWidth + x),
                   sampledPixels[y * pixelWidth + x] {
                    continue
                }
                let u = (CGFloat(x) + 0.5) / CGFloat(pixelWidth)
                let topV = (CGFloat(y) + 0.5) / CGFloat(pixelHeight)
                guard let featherCoverage = sampler.innerSamplingFringeCoverage(
                    u: u,
                    topV: topV
                ),
                featherCoverage > 0 else {
                    continue
                }

                var bestOffset: Int?
                var bestAlpha: UInt8 = 0
                for yOffset in -3...3 {
                    let candidateY = y + yOffset
                    guard candidateY >= 0, candidateY < pixelHeight else {
                        continue
                    }
                    for xOffset in -3...3 {
                        let candidateX = x + xOffset
                        guard candidateX >= 0, candidateX < pixelWidth else {
                            continue
                        }
                        let candidateOffset =
                            (candidateY * pixelWidth + candidateX) * 4
                        let candidateAlpha = source[candidateOffset + 3]
                        if candidateAlpha > bestAlpha {
                            bestAlpha = candidateAlpha
                            bestOffset = candidateOffset
                        }
                    }
                }

                guard let bestOffset, bestAlpha > 0 else {
                    continue
                }
                let coverage = min(max(featherCoverage, 0), 1)
                rgba[targetOffset] = UInt8(
                    (CGFloat(source[bestOffset]) * coverage).rounded()
                )
                rgba[targetOffset + 1] = UInt8(
                    (CGFloat(source[bestOffset + 1]) * coverage).rounded()
                )
                rgba[targetOffset + 2] = UInt8(
                    (CGFloat(source[bestOffset + 2]) * coverage).rounded()
                )
                rgba[targetOffset + 3] = UInt8(
                    (CGFloat(source[bestOffset + 3]) * coverage).rounded()
                )
                filledPixels += 1
            }
        }
        return filledPixels
    }

    private func currentLipstickStyle() -> (color: RGBColor,
                                            finish: LipFinish,
                                            density: LipstickDensity,
                                            texture: LipstickTexture,
                                            materialProfile: LipMaterialProfile,
                                            shadingDebugMode: LipShadingDebugMode) {
        colorLock.lock()
        let color = lipstickColor
        let finish = lipstickFinish
        let density = lipstickDensity
        let texture = lipstickTexture
        let materialProfile = lipstickMaterialProfile
        let shadingDebugMode = lipstickShadingDebugMode
        colorLock.unlock()
        return (
            color,
            finish,
            density,
            texture,
            materialProfile,
            shadingDebugMode
        )
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

    private static func materialAppearance(pigment: RGBColor,
                                      base: RGBColor,
                                      highlightSurround: RGBColor,
                                      signals: CameraMaterialSignals,
                                      topV: CGFloat,
                                      normalizedLipX: Float,
                                      lowerLipMembership: Float,
                                      outerDistance: CGFloat,
                                      innerDistance: CGFloat,
                                      opticalSurface: LipOpticalSurface.Sample?,
                                      cameraDetailCoherence: Float,
                                      apertureClosure: Float,
                                      finish: LipFinish,
                                      materialProfile: LipMaterialProfile,
                                      shadingDebugMode: LipShadingDebugMode) -> MaterialAppearance {
        var cameraAwarePigment = banubaBaseColor(
            cameraLinear: base,
            pigmentLinear: pigment,
            showsValueDebug: shadingDebugMode == .capturedShape
        )
        if shadingDebugMode == .capturedShape {
            return MaterialAppearance(
                pigment: cameraAwarePigment,
                reflection: cameraAwarePigment,
                reflectionCoverage: 0
            )
        }
        if finish == .satin {
            let cameraSRGB = linearToSRGB(base)
            let pigmentSRGB = linearToSRGB(pigment)
            let cameraValue = max(cameraSRGB.red,
                                  max(cameraSRGB.green, cameraSRGB.blue))
            let lipstickValue = max(pigmentSRGB.red,
                                    max(pigmentSRGB.green, pigmentSRGB.blue))
            let rawTargetValue = cameraValue * lipstickValue / 0.85
            if rawTargetValue > 0.82 {
                let targetValue = 0.82 + 0.13 *
                    (1 - exp(-(rawTargetValue - 0.82) / 0.13))
                let pigmentHS = hsvHueAndSaturation(pigmentSRGB)
                cameraAwarePigment = sRGBToLinear(hsvColor(
                    hue: pigmentHS.hue,
                    saturation: pigmentHS.saturation,
                    value: targetValue
                ))
            }
        }
        if finish == .gloss || finish == .oil || finish == .satin {
            let isWet = finish == .gloss || finish == .oil
            let distanceSum = max(Float(outerDistance + innerDistance), 0.0001)
            let surfacePosition = Float(innerDistance) / distanceSum
            let boundaryGate = finish == .gloss ?
                smoothStep((surfacePosition - 0.12) / 0.18) *
                    (1 - smoothStep((surfacePosition - 0.80) / 0.18)) :
                smoothStep((surfacePosition - 0.025) / 0.135) *
                    (1 - smoothStep((surfacePosition - 0.84) / 0.145))
            let observed = signals.shineMask
            let wetMask = smoothStep((observed - materialProfile.shineMaskMinimum) /
                max(materialProfile.shineMaskMaximum - materialProfile.shineMaskMinimum, 0.001))
            if isWet {
                let baseSRGB = linearToSRGB(base)
                let surroundSRGB = linearToSRGB(highlightSurround)
                let cameraValue = max(baseSRGB.red,
                                      max(baseSRGB.green, baseSRGB.blue))
                let surroundValue = max(surroundSRGB.red,
                                        max(surroundSRGB.green, surroundSRGB.blue))
                let diffuseValue = cameraValue +
                    (min(cameraValue, surroundValue) - cameraValue) * wetMask * 0.82
                cameraAwarePigment = banubaBaseColor(
                    cameraValue: diffuseValue,
                    pigmentLinear: pigment,
                    showsValueDebug: false
                )
            }
            let exponent = materialProfile.cameraHighlightExponent *
                materialProfile.highlightConcentration /
                max(materialProfile.cameraRoughnessTuning, 0.35)
            let light = min(pow(wetMask, max(exponent, 0.001)) *
                materialProfile.cameraHighlightGain *
                materialProfile.capturedHighlightStrength * cameraDetailCoherence,
                materialProfile.capturedHighlightMaximum)
            let arc = opticalSurface?.arc ?? normalizedLipX
            let across = opticalSurface?.across ?? (1 - surfacePosition)
            let corner = smoothStep((abs(arc - 0.5) * 2 - 0.62) / 0.34)
            let innerEdge = smoothStep((across - 0.78) / 0.20)
            let seamSuppression: Float = isWet ?
                0.30 + 0.28 * apertureClosure :
                0.74 + 0.18 * apertureClosure
            let guardedLight = light * boundaryGate *
                (1 - corner * 0.82) * (1 - innerEdge * seamSuppression) *
                (finish == .gloss ? lowerLipMembership : 1)
            let layer = reflectionLayer(cameraAwarePigment,
                                        amount: guardedLight,
                                        highlightMaximum: materialProfile.capturedHighlightMaximum,
                                        finish: finish)
            return MaterialAppearance(
                pigment: cameraAwarePigment,
                reflection: layer.color,
                reflectionCoverage: layer.coverage
            )
        }

        let cornerPosition = min(abs(normalizedLipX - 0.5) * 2, 1)
        let cornerRegion = smoothStep((cornerPosition - 0.62) / 0.34)
        let innerSeamRegion = 1 - smoothStep((abs(Float(topV) - 0.50) - 0.018) / 0.085)
        var reflectionCoverage: Float = 0
        var reflection = RGBColor(red: 1, green: 1, blue: 1)
        if finish == .matte || finish == .satin || finish.isWet {
            // Preserve the camera reflection footprint. Finish-specific
            // response changes only its visibility and brightness; it never
            // invents a highlight where the source frame has none.
            let isWet = finish.isWet
            let distanceSum = max(
                Float(outerDistance + innerDistance),
                0.000_1
            )
            // Measure the pixel across the actual lip thickness. Reflections
            // disappear before both the inner wet line and the outer contour.
            let surfacePosition = Float(innerDistance) / distanceSum
            let outerFadeStart: Float
            let outerFadeEnd: Float
            let lowerFadeInStart: Float
            let lowerFadeInEnd: Float
            let lowerFadeOutStart: Float
            let lowerFadeOutEnd: Float
            switch finish {
            case .gloss, .shimmer:
                outerFadeStart = 0.58
                outerFadeEnd = 0.70
                lowerFadeInStart = 0.24
                lowerFadeInEnd = 0.32
                lowerFadeOutStart = 0.68
                lowerFadeOutEnd = 0.76
            case .oil:
                outerFadeStart = 0.60
                outerFadeEnd = 0.73
                lowerFadeInStart = 0.21
                lowerFadeInEnd = 0.29
                lowerFadeOutStart = 0.71
                lowerFadeOutEnd = 0.79
            default:
                outerFadeStart = 0.70
                outerFadeEnd = 0.86
                lowerFadeInStart = 0.14
                lowerFadeInEnd = 0.26
                lowerFadeOutStart = 0.74
                lowerFadeOutEnd = 0.86
            }
            let contourDistanceGate =
                smoothStep((surfacePosition - 0.18) / 0.16) *
                (1 - smoothStep(
                    (surfacePosition - outerFadeStart) /
                        max(outerFadeEnd - outerFadeStart, 0.001)
                ))
            let lowerSurfaceGate =
                smoothStep(
                    (normalizedLipX - lowerFadeInStart) /
                        max(lowerFadeInEnd - lowerFadeInStart, 0.001)
                ) *
                (1 - smoothStep(
                    (normalizedLipX - lowerFadeOutStart) /
                        max(lowerFadeOutEnd - lowerFadeOutStart, 0.001)
                ))
            let cupidStripe = 1 - smoothStep(
                (abs(normalizedLipX - 0.5) - 0.035) / 0.060
            )
            let lipSurfaceGate = contourDistanceGate * (
                cupidStripe * (1 - lowerLipMembership) +
                    lowerSurfaceGate * lowerLipMembership
            )
            let highlightSuppression =
                (1 - cornerRegion * 0.82) *
                (1 - innerSeamRegion * (isWet ? 0.30 : 0.74))
            let highlightAmount: Float
            if isWet {
                let isOil = finish == .oil
                let wetMask = smoothStep(
                    (signals.shineMask - materialProfile.shineMaskMinimum) /
                        max(
                            materialProfile.shineMaskMaximum -
                                materialProfile.shineMaskMinimum,
                            0.001
                        )
                )
                let baseSRGB = linearToSRGB(base)
                let surroundSRGB = linearToSRGB(highlightSurround)
                let cameraValue = max(
                    baseSRGB.red,
                    max(baseSRGB.green, baseSRGB.blue)
                )
                let surroundValue = max(
                    surroundSRGB.red,
                    max(surroundSRGB.green, surroundSRGB.blue)
                )
                let diffuseCameraValue = cameraValue +
                    (min(cameraValue, surroundValue) - cameraValue) *
                    wetMask * 0.82
                cameraAwarePigment = banubaBaseColor(
                    cameraValue: diffuseCameraValue,
                    pigmentLinear: pigment,
                    showsValueDebug: false
                )
                let rawHighlight = pow(
                    wetMask,
                    isOil ? 1.05 : 1.30
                ) * (isOil ? 0.42 : 0.24) *
                    materialProfile.capturedHighlightStrength
                highlightAmount = min(
                    rawHighlight *
                        lipSurfaceGate *
                        highlightSuppression,
                    materialProfile.capturedHighlightMaximum
                )
            } else {
                let satinMask = smoothStep(
                    (signals.shineMask - materialProfile.shineMaskMinimum) /
                        max(
                            materialProfile.shineMaskMaximum -
                                materialProfile.shineMaskMinimum,
                            0.001
                        )
                )
                let softShineStrength: Float = finish == .matte ?
                    0.22 : 0.26
                let rawHighlight = pow(satinMask, 1.20) *
                    softShineStrength * materialProfile.capturedHighlightStrength
                highlightAmount = min(
                    rawHighlight *
                        lipSurfaceGate *
                        highlightSuppression,
                    materialProfile.capturedHighlightMaximum
                )
            }
            (reflection, reflectionCoverage) = reflectionLayer(
                cameraAwarePigment,
                amount: highlightAmount,
                highlightMaximum: materialProfile.capturedHighlightMaximum,
                finish: finish
            )
        }
        return MaterialAppearance(
            pigment: cameraAwarePigment,
            reflection: reflection,
            reflectionCoverage: reflectionCoverage
        )
    }

    /// The mask controls reflection coverage independently of pigment alpha.
    /// Satin keeps the product hue in its softer reflected colour.
    private static func reflectionLayer(
        _ linearColor: RGBColor,
        amount: Float,
        highlightMaximum: Float,
        finish: LipFinish
    ) -> (color: RGBColor, coverage: Float) {
        if finish == .gloss || finish == .oil {
            return (RGBColor(red: 1, green: 1, blue: 1),
                    finish == .gloss ? clamp01(amount * 1.45) :
                        1 - exp(-max(amount, 0)))
        }
        if finish.isWet {
            let coverage = clamp01(
                amount * (finish == .oil ? 2.30 : 1.45)
            )
            return (RGBColor(red: 1, green: 1, blue: 1), coverage)
        }

        let colorSRGB = linearToSRGB(linearColor)
        let hueAndSaturation = hsvHueAndSaturation(colorSRGB)
        let value = max(colorSRGB.red, max(colorSRGB.green, colorSRGB.blue))
        let coverage = finish == .satin ?
            1 - exp(-max(amount, 0) *
                (1 + smoothStep((value - 0.55) / 0.30))) :
            clamp01(amount * (finish == .matte ? 1.30 : 1.40))
        guard coverage > 0.0001 else {
            return (linearColor, 0)
        }
        let satinTarget = sRGBToLinear(hsvColor(
            hue: hueAndSaturation.hue,
            saturation: hueAndSaturation.saturation *
                (1 - coverage * 0.22),
            value: value + (1 - value) * coverage
        ))
        // Choose the reflected colour so full pigment coverage retains the
        // previous soft satin response. Only the shine coverage is inverted.
        let softReflection = RGBColor(
            red: clamp01((satinTarget.red - linearColor.red * (1 - coverage)) / coverage),
            green: clamp01((satinTarget.green - linearColor.green * (1 - coverage)) / coverage),
            blue: clamp01((satinTarget.blue - linearColor.blue * (1 - coverage)) / coverage)
        )
        guard finish == .satin else {
            return (softReflection, coverage)
        }
        // A narrow white core follows only the strongest observed highlights.
        let normalizedShine = amount / max(highlightMaximum, 0.001)
        let whiteCore = 0.92 * smoothStep((normalizedShine - 0.60) / 0.35)
        let combinedCoverage = whiteCore + coverage * (1 - whiteCore)
        let softWeight = coverage * (1 - whiteCore)
        return (RGBColor(
            red: (whiteCore + softReflection.red * softWeight) / combinedCoverage,
            green: (whiteCore + softReflection.green * softWeight) / combinedCoverage,
            blue: (whiteCore + softReflection.blue * softWeight) / combinedCoverage
        ), combinedCoverage)
    }

    /// Adapts Banuba beauty-ios `matt.frag` as the diffuse base for every
    /// finish: retain the
    /// product's HSV hue/saturation, derive HSV value from the camera pixel,
    /// then return to linear RGB for the SceneKit blend.
    private static func banubaBaseColor(
        cameraLinear: RGBColor,
        pigmentLinear: RGBColor,
        showsValueDebug: Bool
    ) -> RGBColor {
        let cameraSRGB = linearToSRGB(cameraLinear)
        let cameraValue = max(
            cameraSRGB.red,
            max(cameraSRGB.green, cameraSRGB.blue)
        )
        return banubaBaseColor(
            cameraValue: cameraValue,
            pigmentLinear: pigmentLinear,
            showsValueDebug: showsValueDebug
        )
    }

    private static func banubaBaseColor(
        cameraValue: Float,
        pigmentLinear: RGBColor,
        showsValueDebug: Bool
    ) -> RGBColor {
        let pigmentSRGB = linearToSRGB(pigmentLinear)
        let pigmentValue = max(
            pigmentSRGB.red,
            max(pigmentSRGB.green, pigmentSRGB.blue)
        )
        let targetValue = clamp01(
            cameraValue * pigmentValue / 0.85
        )
        if showsValueDebug {
            return sRGBToLinear(RGBColor(
                red: targetValue,
                green: targetValue,
                blue: targetValue
            ))
        }

        let hueAndSaturation = hsvHueAndSaturation(pigmentSRGB)
        return sRGBToLinear(hsvColor(
            hue: hueAndSaturation.hue,
            saturation: hueAndSaturation.saturation,
            value: targetValue
        ))
    }

    private static func hsvHueAndSaturation(
        _ color: RGBColor
    ) -> (hue: Float, saturation: Float) {
        let maximum = max(color.red, max(color.green, color.blue))
        let minimum = min(color.red, min(color.green, color.blue))
        let delta = maximum - minimum
        guard delta > 0.000_1, maximum > 0.000_1 else {
            return (0, 0)
        }

        let sector: Float
        if maximum == color.red {
            sector = (color.green - color.blue) / delta
        } else if maximum == color.green {
            sector = 2 + (color.blue - color.red) / delta
        } else {
            sector = 4 + (color.red - color.green) / delta
        }
        var hue = (sector / 6).truncatingRemainder(dividingBy: 1)
        if hue < 0 {
            hue += 1
        }
        return (hue, delta / maximum)
    }

    private static func hsvColor(hue: Float,
                                 saturation: Float,
                                 value: Float) -> RGBColor {
        let normalizedHue = hue - hue.rounded(.down)
        let scaledHue = normalizedHue * 6
        let sector = Int(scaledHue.rounded(.down)) % 6
        let fraction = scaledHue - Float(sector)
        let primary = clamp01(value)
        let p = primary * (1 - saturation)
        let q = primary * (1 - saturation * fraction)
        let t = primary * (1 - saturation * (1 - fraction))

        switch sector {
        case 0:
            return RGBColor(red: primary, green: t, blue: p)
        case 1:
            return RGBColor(red: q, green: primary, blue: p)
        case 2:
            return RGBColor(red: p, green: primary, blue: t)
        case 3:
            return RGBColor(red: p, green: q, blue: primary)
        case 4:
            return RGBColor(red: t, green: p, blue: primary)
        default:
            return RGBColor(red: primary, green: p, blue: q)
        }
    }

    /// Converts the camera's broad lip luminance into an asymmetric exposure
    /// field. Pale opaque pigment keeps almost all captured shadow depth,
    /// while diffuse light remains intentionally shallow for matte products.
    private static func capturedShapeExposure(
        signals: CameraMaterialSignals,
        referenceLuminance: Float?,
        pigmentLuminance: Float,
        materialProfile: LipMaterialProfile
    ) -> Float {
        guard let referenceLuminance,
              referenceLuminance > 0.0045,
              signals.lowFrequencyLuminance > 0 else {
            return 0
        }

        let palePigment = smoothStep(
            (pigmentLuminance - 0.16) / 0.22
        )
        let denseCoverage = smoothStep(
            (materialProfile.pigmentCoverage - 0.68) / 0.24
        )
        let paleDenseAmount = palePigment * denseCoverage
        let capturedExposure = log2(
            max(signals.lowFrequencyLuminance, 0.0031) /
                max(referenceLuminance, 0.0045)
        )

        if capturedExposure < 0 {
            let maximumShadowStops = 0.92 +
                (materialProfile.capturedShadowMaximumStops - 0.92) *
                    paleDenseAmount
            let shadowRetention = materialProfile.shadowRetention +
                (max(materialProfile.shadowRetention, 0.98) -
                    materialProfile.shadowRetention) * paleDenseAmount
            return max(capturedExposure, -maximumShadowStops) *
                shadowRetention
        }

        let maximumDiffuseStops =
            materialProfile.capturedDiffuseLightMaximumStops *
            (1 - paleDenseAmount * 0.28)
        return min(capturedExposure, maximumDiffuseStops) *
            materialProfile.diffuseLightResponse
    }

    private static func premultipliedCompositingColor(
        target: RGBColor,
        base: RGBColor,
        compositingAlpha: Float,
        compensationAlpha: Float,
        luminanceLimit: Float?
    ) -> RGBColor {
        let compositingAlpha = max(0.001, min(compositingAlpha, 1))
        let compensationAlpha = max(0.001, min(compensationAlpha, 1))
        func corrected(_ targetChannel: Float, _ baseChannel: Float) -> Float {
            clamp01(
                (targetChannel - baseChannel * (1 - compensationAlpha)) /
                    compensationAlpha
            )
        }
        var straightColor = RGBColor(
            red: corrected(target.red, base.red),
            green: corrected(target.green, base.green),
            blue: corrected(target.blue, base.blue)
        )
        if let luminanceLimit,
           luminance(straightColor) > luminanceLimit {
            straightColor = colorWithLuminance(
                straightColor,
                luminance: min(max(luminanceLimit, 0), 1)
            )
        }
        return RGBColor(
            red: straightColor.red * compositingAlpha,
            green: straightColor.green * compositingAlpha,
            blue: straightColor.blue * compositingAlpha
        )
    }

    private static func cameraTone(base: RGBColor,
                                   blurred: RGBColor,
                                   highlightSurround: RGBColor,
                                   highlightConfidence: Float,
                                   highlightThreshold: Float,
                                   finish: LipFinish,
                                   microTextureRetention: Float) -> CameraMaterialSignals {
        let baseLuma = luminance(base)
        let blurredLuma = max(luminance(blurred), 0.0045)
        let highlightSurroundLuma = max(
            luminance(highlightSurround),
            0.0045
        )
        // Individual finishes apply their own response and bounds in
        // materialColor; this map contains camera-observed relief only.
        let linearContrast = max(baseLuma / blurredLuma, 0.0001)
        let detail = max(0.90, min(pow(linearContrast, 0.45), 1.08))
        let highlightLogContrast = log2(
            max(baseLuma, 0.0031) / highlightSurroundLuma
        ) * 0.45
        let relativeHighlight = smoothStep(
            (highlightLogContrast - 0.001 * highlightThreshold) /
                max(0.064 * highlightThreshold, 0.0001)
        )
        let highlightBrightness = smoothStep(
            (baseLuma - 0.012) / 0.208
        )
        let residualRed = max(base.red - highlightSurround.red, 0)
        let residualGreen = max(base.green - highlightSurround.green, 0)
        let residualBlue = max(base.blue - highlightSurround.blue, 0)
        let residualMaximum = max(
            residualRed,
            max(residualGreen, residualBlue)
        )
        let residualCommon = min(
            residualRed,
            min(residualGreen, residualBlue)
        )
        // A real specular reflection raises all colour channels together.
        // This rejects bright pigment texture while retaining warm highlights.
        let residualBalance = residualCommon / max(residualMaximum, 0.000_1)
        let illuminantEvidence = smoothStep(
            (residualBalance - 0.18) / 0.44
        )
        let neutralEvidence = 0.30 + illuminantEvidence * 0.70
        let colorEvidence = finish == .satin ?
            max(neutralEvidence, clamp01(microTextureRetention)) :
            neutralEvidence
        return CameraMaterialSignals(
            detail: detail,
            lowFrequencyLuminance: blurredLuma,
            shineMask: relativeHighlight * highlightBrightness *
                colorEvidence * highlightConfidence
        )
    }

    private static func innerMouthCameraCoverage(
        base: RGBColor,
        topV: CGFloat,
        opening: CGFloat
    ) -> Float {
        let srgb = linearToSRGB(base)
        let maximum = max(srgb.red, max(srgb.green, srgb.blue))
        let minimum = min(srgb.red, min(srgb.green, srgb.blue))
        let saturation = maximum > 0.001 ? (maximum - minimum) / maximum : 0
        let luma = srgb.red * 0.299 + srgb.green * 0.587 + srgb.blue * 0.114
        let toothProbability = smoothStep((luma - 0.64) / 0.18) *
            (1 - smoothStep((saturation - 0.16) / 0.14))
        let seamRegion = 1 - smoothStep((abs(Float(topV) - 0.50) - 0.018) / 0.067)
        let toothGuard = toothProbability * seamRegion *
            smoothStep((Float(opening) - 0.24) / 0.22)
        return 1 - toothGuard * 0.98
    }

    /// One at the inner lip boundary and progressively softer toward the
    /// outside contour. Distances make the gradient follow each real lip half
    /// instead of assuming a fixed horizontal band in the texture.
    private static func tintDistribution(outerDistance: CGFloat,
                                         innerDistance: CGFloat) -> Float {
        let distanceSum = max(outerDistance + innerDistance, 0.000_1)
        let innerPosition = Float(outerDistance / distanceSum)
        return 0.58 + smoothStep(innerPosition / 0.86) * 0.42
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

    private static func sRGBToLinear(_ color: RGBColor) -> RGBColor {
        RGBColor(
            red: sRGBToLinear(color.red),
            green: sRGBToLinear(color.green),
            blue: sRGBToLinear(color.blue)
        )
    }

    private static func sRGBToLinear(_ component: Float) -> Float {
        let component = clamp01(component)
        if component <= 0.04045 {
            return component / 12.92
        }
        return pow((component + 0.055) / 1.055, 2.4)
    }

    private static func linearToSRGB(_ color: RGBColor) -> RGBColor {
        RGBColor(
            red: linearToSRGB(color.red),
            green: linearToSRGB(color.green),
            blue: linearToSRGB(color.blue)
        )
    }

    private static func linearToSRGB(_ component: Float) -> Float {
        let component = max(component, 0)
        if component <= 0.003_130_8 {
            return clamp01(component * 12.92)
        }
        return clamp01(1.055 * pow(component, 1 / 2.4) - 0.055)
    }

    private static func sRGBPremultipliedColor(
        fromLinearPremultiplied color: RGBColor,
        alpha: Float
    ) -> RGBColor {
        let alpha = clamp01(alpha)
        guard alpha > 0.0001 else {
            return RGBColor(red: 0, green: 0, blue: 0)
        }
        let straightLinear = RGBColor(
            red: clamp01(color.red / alpha),
            green: clamp01(color.green / alpha),
            blue: clamp01(color.blue / alpha)
        )
        let straightSRGB = linearToSRGB(straightLinear)
        return RGBColor(
            red: straightSRGB.red * alpha,
            green: straightSRGB.green * alpha,
            blue: straightSRGB.blue * alpha
        )
    }

    private static func luminance(_ linearColor: RGBColor) -> Float {
        clamp01(
            linearColor.red * 0.2126 +
                linearColor.green * 0.7152 +
                linearColor.blue * 0.0722
        )
    }

    private static func lightDenseAmount(
        pigment: RGBColor,
        pigmentCoverage: Float
    ) -> Float {
        let palePigment = smoothStep(
            (luminance(pigment) - 0.20) / 0.32
        )
        let denseCoverage = smoothStep(
            (pigmentCoverage - 0.68) / 0.24
        )
        return palePigment * denseCoverage
    }

    private static func percentile(
        _ values: [Float],
        position: Float
    ) -> Float? {
        guard !values.isEmpty else {
            return nil
        }
        let sorted = values.sorted()
        let index = min(
            max(Int((Float(sorted.count - 1) * clamp01(position)).rounded()), 0),
            sorted.count - 1
        )
        return sorted[index]
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
                                        height: Int,
                                        wetSurface: LipOpticalSurface.Sample?,
                                        referenceGloss: Bool,
                                        imageToViewportTransform: CGAffineTransform,
                                        mouthWidth: Float,
                                        profile: LipMaterialProfile) -> CameraLipSample? {
        let imagePoint = point.applying(inverseImageTransform)
        guard let baseSRGB = samplePixel(
            at: imagePoint,
            baseAddress: baseAddress,
            bytesPerRow: bytesPerRow,
            width: width,
            height: height
        ) else {
            return nil
        }
        let base = sRGBToLinear(baseSRGB)

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
            guard let sampleSRGB = samplePixel(
                at: samplePoint,
                baseAddress: baseAddress,
                bytesPerRow: bytesPerRow,
                width: width,
                height: height
            ) else {
                continue
            }
            let sample = sRGBToLinear(sampleSRGB)
            red += sample.red
            green += sample.green
            blue += sample.blue
            count += 1
        }

        guard count > 0 else {
            return CameraLipSample(
                base: base,
                blurred: base,
                highlightSurround: base,
                highlightConfidence: 0
            )
        }

        let highlightRadius = radius * 2.4
        let highlightOffsets = [
            CGPoint(x: -highlightRadius, y: 0),
            CGPoint(x: highlightRadius, y: 0),
            CGPoint(x: 0, y: -highlightRadius * 0.67),
            CGPoint(x: 0, y: highlightRadius * 0.67)
        ]
        var highlightRed: Float = 0
        var highlightGreen: Float = 0
        var highlightBlue: Float = 0
        var highlightCount: Float = 0
        for offset in highlightOffsets {
            let samplePoint = CGPoint(
                x: imagePoint.x + offset.x,
                y: imagePoint.y + offset.y
            )
            guard let sampleSRGB = samplePixel(
                at: samplePoint,
                baseAddress: baseAddress,
                bytesPerRow: bytesPerRow,
                width: width,
                height: height
            ) else {
                continue
            }
            let sample = sRGBToLinear(sampleSRGB)
            highlightRed += sample.red
            highlightGreen += sample.green
            highlightBlue += sample.blue
            highlightCount += 1
        }

        let blurred = RGBColor(
            red: red / count,
            green: green / count,
            blue: blue / count
        )
        var highlightSurround = highlightCount > 0 ?
            RGBColor(
                red: highlightRed / highlightCount,
                green: highlightGreen / highlightCount,
                blue: highlightBlue / highlightCount
            ) : blurred
        var highlightConfidence: Float = 1
        if wetSurface != nil || referenceGloss {
            let sampleScale = profile.cameraSampleScale *
                profile.cameraHighlightSize
            let offsets = [
                CGPoint(x: -CGFloat(12 * sampleScale), y: 0),
                CGPoint(x: CGFloat(12 * sampleScale), y: 0),
                CGPoint(x: 0, y: -CGFloat(8 * sampleScale)),
                CGPoint(x: 0, y: CGFloat(8 * sampleScale))
            ]
            var sum = RGBColor(red: 0, green: 0, blue: 0)
            var validWeight: Float = 0
            for offset in offsets {
                let viewportDelta = SIMD2<Float>(
                    Float(imageToViewportTransform.a * offset.x +
                          imageToViewportTransform.c * offset.y),
                    Float(imageToViewportTransform.b * offset.x +
                          imageToViewportTransform.d * offset.y)
                )
                let geometryWeight: Float
                if let wetSurface, !referenceGloss {
                    let across = wetSurface.across +
                        simd_dot(viewportDelta, wetSurface.transverse) /
                        max(wetSurface.thickness, 1)
                    let cornerClearance = min(wetSurface.arc,
                        1 - wetSurface.arc) * mouthWidth -
                        simd_length(viewportDelta)
                    geometryWeight = smoothStep((across - 0.02) / 0.08) *
                        (1 - smoothStep((across - 0.90) / 0.08)) *
                        smoothStep(cornerClearance / 6)
                } else {
                    geometryWeight = 1
                }
                guard geometryWeight > 0.001,
                    let pixel = samplePixel(
                    at: CGPoint(x: imagePoint.x + offset.x,
                                y: imagePoint.y + offset.y),
                    baseAddress: baseAddress,
                    bytesPerRow: bytesPerRow,
                    width: width,
                    height: height
                ) else { continue }
                let linear = sRGBToLinear(pixel)
                let weight = referenceGloss ? 1 :
                    geometryWeight * (1 - toothProbability(pixel))
                sum = RGBColor(red: sum.red + linear.red * weight,
                               green: sum.green + linear.green * weight,
                               blue: sum.blue + linear.blue * weight)
                validWeight += weight
            }
            highlightSurround = validWeight > 0.001 ?
                RGBColor(red: sum.red / validWeight,
                         green: sum.green / validWeight,
                         blue: sum.blue / validWeight) : base
            highlightConfidence = smoothStep((validWeight - 0.5) / 1.3)
        }

        return CameraLipSample(
            base: base,
            blurred: blurred,
            highlightSurround: highlightSurround,
            highlightConfidence: highlightConfidence
        )
    }

    private static func stableLipBaseColor(u: CGFloat, topV: CGFloat) -> RGBColor {
        let center = 1 - min(abs(Float(u) - 0.5) * 2, 1)
        let vertical = 1 - min(abs(Float(topV) - 0.52) * 1.75, 1)
        let softWarmth = stableWave(u: u, topV: topV, ux: 7.0, vy: 3.5, phase: 0.25)
        let fineCrease = stableWave(u: u, topV: topV, ux: 33.0, vy: 5.0, phase: 1.7)
        let tone = 0.92 + center * 0.08 + vertical * 0.05 + softWarmth * 0.035 - fineCrease * 0.025

        return sRGBToLinear(RGBColor(
            red: clamp01(0.58 * tone + center * 0.035),
            green: clamp01(0.255 * tone + softWarmth * 0.020),
            blue: clamp01(0.305 * tone + vertical * 0.018)
        ))
    }

    private static func pigmentBlendedColor(lipstick: RGBColor,
                                            variation: Float,
                                            texture: LipstickTexture) -> RGBColor {
        let linearLipstick = sRGBToLinear(lipstick)
        guard texture == .mousse else {
            return linearLipstick
        }
        // A fixed UV-space micro-relief gives the fallback a diffuse mousse
        // surface without changing the catalogue hue or shimmering over time.
        let relief = 0.975 + min(max(variation, 0), 1) * 0.050
        return RGBColor(
            red: clamp01(linearLipstick.red * relief),
            green: clamp01(linearLipstick.green * relief),
            blue: clamp01(linearLipstick.blue * relief)
        )
    }

    private static func mixColor(_ first: RGBColor,
                                 _ second: RGBColor,
                                 amount: Float) -> RGBColor {
        let amount = clamp01(amount)
        return RGBColor(
            red: first.red + (second.red - first.red) * amount,
            green: first.green + (second.green - first.green) * amount,
            blue: first.blue + (second.blue - first.blue) * amount
        )
    }

    private static func cornerOpacityScale(u: CGFloat) -> Float {
        let cornerPosition = min(abs(Float(u) - 0.5) * 2, 1)
        let cornerTip = smoothStep((cornerPosition - 0.82) / 0.16)
        return 1 - cornerTip * 0.12
    }

    private static func stableLipPigmentVariation(u: CGFloat, topV: CGFloat) -> (alpha: Float, color: Float) {
        let broad = stableWave(u: u, topV: topV, ux: 5.0, vy: 4.5, phase: 0.8)
        let crease = stableWave(u: u, topV: topV, ux: 42.0, vy: 2.0, phase: 2.4)
        let verticalCrease = stableWave(u: u, topV: topV, ux: 68.0, vy: 0.6, phase: 0.1)
        let color = broad * 0.70 + crease * 0.20 + verticalCrease * 0.10
        let alpha = 0.990 + (broad - 0.5) * 0.010 + (verticalCrease - 0.5) * 0.006
        return (max(0.975, min(alpha, 1.008)), color)
    }

    private static func shimmerHash(x: UInt32, y: UInt32) -> UInt32 {
        var hash = (x &* 0x8da6b343) ^ (y &* 0xd8163841)
        hash ^= hash >> 16
        hash = hash &* 0x7feb352d
        hash ^= hash >> 15
        hash = hash &* 0x846ca68b
        return hash ^ (hash >> 16)
    }

    private static func shimmerNoise(u: Float, v: Float) -> Float {
        let gridX = u * 96
        let gridY = v * 48
        let cellX = UInt32(floor(gridX))
        let cellY = UInt32(floor(gridY))
        let fractionX = gridX - floor(gridX)
        let fractionY = gridY - floor(gridY)
        let weightX = fractionX * fractionX * (3 - 2 * fractionX)
        let weightY = fractionY * fractionY * (3 - 2 * fractionY)
        let first = Float(shimmerHash(x: cellX, y: cellY) & 1023) / 1023
        let second = Float(shimmerHash(x: cellX + 1, y: cellY) & 1023) / 1023
        let third = Float(shimmerHash(x: cellX, y: cellY + 1) & 1023) / 1023
        let fourth = Float(shimmerHash(x: cellX + 1, y: cellY + 1) & 1023) / 1023
        let upper = first + (second - first) * weightX
        let lower = third + (fourth - third) * weightX
        return upper + (lower - upper) * weightY
    }

    private static func shimmerLayer(materialLuminance: Float,
                                      reflectionCoverage: Float,
                                      u: Float, v: Float,
                                      viewYawRadians: Float,
                                      viewPitchRadians: Float,
                                      intensity: Float,
                                      sampler: CanonicalLipSampler,
                                      pixelWidth: Int,
                                      pixelHeight: Int) -> ShimmerLayer {
        let grain = shimmerNoise(u: u, v: v)
        var light = smoothStep((materialLuminance - 0.025) / 0.195)
        let shapeLight = smoothStep((reflectionCoverage - 0.02) / 0.20)
        light = max(light, 0.45 * shapeLight)
        let angle = smoothStep((viewYawRadians + viewPitchRadians * 0.6) / 0.7)
        let colourShift = clamp01(0.45 + (grain - 0.5) * 0.55 +
            0.24 * sin(viewYawRadians * 2.2 + viewPitchRadians * 1.3))
        let pearlColour = mixColor(
            RGBColor(red: 1, green: 0.76, blue: 0.68),
            RGBColor(red: 0.79, green: 0.77, blue: 1),
            amount: colourShift
        )
        let pearlCoverage = clamp01((0.05 + 0.13 * shapeLight) * light *
            (0.45 + 0.55 * grain) * (0.78 + 0.22 * angle) *
            min(max(intensity, 0), 4))

        let gridX = u * 96
        let gridY = v * 48
        let particleHash = shimmerHash(
            x: UInt32(floor(gridX)), y: UInt32(floor(gridY))
        )
        guard particleHash % 8 == 0 else {
            return ShimmerLayer(color: pearlColour, coverage: pearlCoverage)
        }
        let bright = (particleHash >> 11) % 11 == 0
        let dx = gridX - floor(gridX) - 0.5
        let dy = gridY - floor(gridY) - 0.5
        let radius = sqrt(dx * dx + dy * dy)
        let start: Float = bright ? 0.25 : 0.15
        let end: Float = bright ? 0.62 : 0.70
        let particle = 1 - smoothStep((radius - start) / (end - start))
        let reflectedLight = (0.35 + 0.65 * light) *
            (0.70 + 0.30 * shapeLight) * (0.85 + 0.15 * angle)
        let particleStrength = min(
            particle * (bright ? 1.60 : 0.65) * reflectedLight *
                min(max(intensity / 3, 0), 1.33),
            0.92
        ) * sampler.shimmerParticleInnerGate(
            u: CGFloat(u), topV: CGFloat(v),
            pixelWidth: pixelWidth, pixelHeight: pixelHeight
        )
        let particleColour: RGBColor
        switch (particleHash >> 16) & 3 {
        case 0:
            particleColour = RGBColor(red: 1, green: 0.86, blue: 0.72)
        case 1:
            particleColour = RGBColor(red: 1, green: 0.78, blue: 0.90)
        case 2:
            particleColour = RGBColor(red: 0.85, green: 0.80, blue: 1)
        default:
            particleColour = RGBColor(red: 0.78, green: 0.92, blue: 1)
        }
        let reflectionColour = bright ? mixColor(
            particleColour,
            RGBColor(red: 1, green: 1, blue: 1),
            amount: 0.55
        ) : particleColour
        let coverage = particleStrength +
            pearlCoverage * (1 - particleStrength)
        guard coverage > 0.0001 else {
            return ShimmerLayer(color: pearlColour, coverage: 0)
        }
        let pearlWeight = pearlCoverage * (1 - particleStrength)
        return ShimmerLayer(color: RGBColor(
            red: (reflectionColour.red * particleStrength +
                pearlColour.red * pearlWeight) / coverage,
            green: (reflectionColour.green * particleStrength +
                pearlColour.green * pearlWeight) / coverage,
            blue: (reflectionColour.blue * particleStrength +
                pearlColour.blue * pearlWeight) / coverage
        ), coverage: coverage)
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
        // Classification thresholds are perceptual; convert the linear camera
        // sample back to sRGB without moving the shading math out of linear RGB.
        let sRGB = linearToSRGB(color)
        let maxChannel = max(sRGB.red, max(sRGB.green, sRGB.blue))
        let minChannel = min(sRGB.red, min(sRGB.green, sRGB.blue))
        let luminance = sRGB.red * 0.299 + sRGB.green * 0.587 + sRGB.blue * 0.114
        let saturation = maxChannel > 0.001 ? (maxChannel - minChannel) / maxChannel : 0
        return luminance > 0.64 && saturation < 0.22
    }

    private static func toothProbability(_ sRGB: RGBColor) -> Float {
        let maximum = max(sRGB.red, max(sRGB.green, sRGB.blue))
        let minimum = min(sRGB.red, min(sRGB.green, sRGB.blue))
        let saturation = maximum > 0.001 ? (maximum - minimum) / maximum : 0
        let perceptualLuma = sRGB.red * 0.299 + sRGB.green * 0.587 +
            sRGB.blue * 0.114
        return smoothStep((perceptualLuma - 0.64) / 0.18) *
            (1 - smoothStep((saturation - 0.16) / 0.14))
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

    private func temporallyStabilizedReferenceLuminance(
        _ current: Float?
    ) -> Float? {
        guard let current, current.isFinite, current > 0.001 else {
            return nil
        }
        temporalTextureLock.lock()
        defer {
            temporalTextureLock.unlock()
        }

        let next: Float
        if let previous = previousToneReferenceLuma {
            let exposureStops = abs(log2(current / max(previous, 0.001)))
            let currentWeight: Float = exposureStops > 0.70 ? 0.58 : 0.22
            next = previous + (current - previous) * currentWeight
        } else {
            next = current
        }
        previousToneReferenceLuma = next
        return next
    }

    private func temporallyStabilizedToneMap(_ toneMap: [CameraMaterialSignals],
                                             activePixels: [Bool],
                                             finish: LipFinish,
                                             materialProfile: LipMaterialProfile,
                                             pixelWidth: Int,
                                             pixelHeight: Int,
                                             color: RGBColor,
                                             excludesInnerMouth: Bool,
                                             averageCameraLuma: Float?,
                                             lowLatency: Bool,
                                             motionDelta: CGFloat) -> [CameraMaterialSignals] {
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
            previousShineConfidence?.count != toneMap.count ||
            previousToneMapWidth != pixelWidth ||
            previousToneMapHeight != pixelHeight ||
            previousToneMap?.count != toneMap.count ||
            previousToneColor.map({ Self.colorDistance($0, color) >= 0.035 }) != false ||
            topologyChanged ||
            exposureDelta > 0.22

        if shouldReset {
            previousToneMap = toneMap
            let activationFloor = materialProfile.shineMaskMinimum +
                max(materialProfile.shineMaskMaximum -
                    materialProfile.shineMaskMinimum, 0.001) * 0.08
            previousShineConfidence = toneMap.indices.map { index in
                activePixels[index] && toneMap[index].shineMask > activationFloor ?
                    1 : 0
            }
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

        guard let previous = previousToneMap,
              let previousConfidence = previousShineConfidence else {
            return toneMap
        }

        var output = toneMap
        var historyToneMap = toneMap
        var confidenceMap = previousConfidence
        let normalizedMotion = Float(max(0, min(motionDelta / 0.22, 1)))
        let baseCurrentWeight: Float = lowLatency ? 0.60 : 0.22
        let currentWeight = min(lowLatency ? 0.72 : 0.38, baseCurrentWeight + normalizedMotion * 0.12)
        let shineRange = max(materialProfile.shineMaskMaximum -
            materialProfile.shineMaskMinimum, 0.001)
        let activationFloor = materialProfile.shineMaskMinimum +
            shineRange * 0.08
        let activationFull = materialProfile.shineMaskMinimum +
            shineRange * 0.18
        let strongCore = materialProfile.shineMaskMinimum +
            shineRange * 0.45
        for index in toneMap.indices {
            guard activePixels[index] else {
                confidenceMap[index] = 0
                continue
            }

            let centerShine = toneMap[index].shineMask
            let x = index % pixelWidth
            let y = index / pixelWidth
            let left = toneMap[x > 0 ? index - 1 : index].shineMask
            let right = toneMap[x + 1 < pixelWidth ? index + 1 : index].shineMask
            let above = toneMap[y > 0 ? index - pixelWidth : index].shineMask
            let below = toneMap[y + 1 < pixelHeight ? index + pixelWidth : index].shineMask
            // Feather only weak fringe pixels; keep the sampled bright core.
            let crossAverage = (centerShine * 4 + left + right + above + below) * 0.125
            let centerCoverage = Self.smoothStep(
                (centerShine - materialProfile.shineMaskMinimum) / shineRange
            )
            let spatialFringe = 1 - Self.smoothStep((centerCoverage - 0.05) / 0.35)
            let currentShine = centerShine +
                (crossAverage - centerShine) * spatialFringe * 0.45
            let previousShine = previous[index].shineMask
            let shineDelta = abs(currentShine - previousShine)
            let smallHighlightChange = normalizedMotion < 0.35 &&
                shineDelta < 0.12
            let baseShineWeight: Float = currentShine < previousShine ?
                0.90 : (finish == .gloss ? 1 :
                    (lowLatency ? 0.86 : 0.78))
            let responsiveWeight = smallHighlightChange ?
                min(0.22 + 0.23 * Self.smoothStep(shineDelta / 0.12),
                    max(baseShineWeight, currentWeight)) :
                min(0.98,
                    max(baseShineWeight, currentWeight) +
                        normalizedMotion * 0.08)
            let edgeSignal = Self.smoothStep(
                (max(currentShine, previousShine) -
                    materialProfile.shineMaskMinimum) / shineRange
            )
            let edge = 1 - Self.smoothStep((edgeSignal - 0.05) / 0.35)
            var shineCurrentWeight = responsiveWeight +
                (0.16 - responsiveWeight) * edge
            if normalizedMotion < 0.35 &&
                previousConfidence[index] > 0.5 &&
                currentShine < previousShine {
                shineCurrentWeight = min(shineCurrentWeight,
                    previousShine >= strongCore ? 0.25 : 0.16)
            }
            var stabilizedShine = Self.blendFloat(
                previous: previousShine,
                current: currentShine,
                currentWeight: shineCurrentWeight
            )
            if currentShine < 0.003 {
                stabilizedShine *= Self.smoothStep(
                    (stabilizedShine - 0.002) / 0.033
                )
            }
            let evidence = Self.smoothStep(
                (currentShine - activationFloor) /
                    max(activationFull - activationFloor, 0.001)
            )
            let confidence: Float
            if normalizedMotion < 0.35 {
                let previousValue = previousConfidence[index]
                confidence = previousValue + (evidence - previousValue) *
                    (evidence > previousValue ? 0.36 :
                        (previousShine >= strongCore ? 0.30 : 0.10))
            } else {
                confidence = currentShine > activationFloor ? 1 : 0
            }
            confidenceMap[index] = confidence
            let visibility = Self.smoothStep((confidence - 0.35) / 0.40)
            let detail = Self.blendFloat(previous: previous[index].detail,
                current: toneMap[index].detail, currentWeight: currentWeight)
            let luminance = Self.blendFloat(
                previous: previous[index].lowFrequencyLuminance,
                current: toneMap[index].lowFrequencyLuminance,
                currentWeight: currentWeight)
            historyToneMap[index] = CameraMaterialSignals(
                detail: detail, lowFrequencyLuminance: luminance,
                shineMask: stabilizedShine)
            output[index] = CameraMaterialSignals(
                detail: detail, lowFrequencyLuminance: luminance,
                shineMask: stabilizedShine * visibility)
        }

        previousToneMap = historyToneMap
        previousShineConfidence = confidenceMap
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

    private func logToneMap(_ toneMap: [CameraMaterialSignals],
                            activePixels: [Bool],
                            averageCameraLuma: Float?,
                            motionDelta: CGFloat,
                            reset: Bool) {
        var minDetail: Float = 1
        var maxDetail: Float = 1
        var minimumLowFrequencyLuma = Float.greatestFiniteMagnitude
        var maximumLowFrequencyLuma: Float = 0
        var minimumShine = Float.greatestFiniteMagnitude
        var maximumShine: Float = 0
        var count = 0
        for index in toneMap.indices where activePixels[index] {
            minDetail = min(minDetail, toneMap[index].detail)
            maxDetail = max(maxDetail, toneMap[index].detail)
            minimumLowFrequencyLuma = min(
                minimumLowFrequencyLuma,
                toneMap[index].lowFrequencyLuminance
            )
            maximumLowFrequencyLuma = max(
                maximumLowFrequencyLuma,
                toneMap[index].lowFrequencyLuminance
            )
            minimumShine = min(minimumShine, toneMap[index].shineMask)
            maximumShine = max(maximumShine, toneMap[index].shineMask)
            count += 1
        }
        LipDebugLog.throttled(
            "lip_material_tone",
            interval: 0.6,
            "lip_material signals pixels=\(count) detail=\(String(format: "%.3f", minDetail))...\(String(format: "%.3f", maxDetail)) lowLuma=\(String(format: "%.3f", count > 0 ? minimumLowFrequencyLuma : -1))...\(String(format: "%.3f", count > 0 ? maximumLowFrequencyLuma : -1)) shine=\(String(format: "%.3f", count > 0 ? minimumShine : -1))...\(String(format: "%.3f", count > 0 ? maximumShine : -1)) averageLuma=\(String(format: "%.3f", averageCameraLuma ?? -1)) referenceLuma=\(String(format: "%.3f", previousToneReferenceLuma ?? -1)) motion=\(String(format: "%.3f", motionDelta)) reset=\(reset)"
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

    private static func appearanceAdjustedPigmentColor(
        source: RGBColor,
        sourceLinearLuminance: Float,
        materialProfile: LipMaterialProfile
    ) -> RGBColor {
        // A pale physical pigment loses a little lightness over the naturally
        // darker lip substrate. Use linear relative luminance directly so the
        // threshold follows the same colour model as the compositor.
        let lightShadeAmount = smoothStep(
            (sourceLinearLuminance - 0.20) / 0.28
        )
        let scale = 1 -
            materialProfile.paleShadeMaximumLuminanceReduction *
            lightShadeAmount
        let linearSource = sRGBToLinear(source)
        return linearToSRGB(RGBColor(
            red: clamp01(linearSource.red * scale),
            green: clamp01(linearSource.green * scale),
            blue: clamp01(linearSource.blue * scale)
        ))
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
