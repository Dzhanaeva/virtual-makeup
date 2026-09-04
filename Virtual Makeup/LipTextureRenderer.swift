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
    private static let lowDensityGlossNaturalLipWeight: Float = 0.52
    private static let lowDensityGlossCoverage: Float = 0.82

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
        let lipBase: RGBColor
        let compensationBase: RGBColor
        let pigment: RGBColor
        let u: CGFloat
        let topV: CGFloat
    }

    private struct ProceduralTextureKey: Equatable {
        let colorSignature: UInt32
        let finish: LipFinish
        let density: LipstickDensity
        let texture: LipstickTexture
        let renderScale: Int
        let excludesInnerMouth: Bool
        let apertureVisibilityBucket: Int
    }

    private let temporalTextureLock = NSLock()
    private var previousToneMap: [MaterialTone]?
    private var previousToneMapWidth = 0
    private var previousToneMapHeight = 0
    private var previousToneColor: RGBColor?
    private var previousToneMouthOpen: Bool?
    private var previousToneAverageLuma: Float?
    private var lipstickFinish: LipFinish = .matte
    private var lipstickDensity: LipstickDensity = .high
    private var lipstickTexture: LipstickTexture = .creamy
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
        let innerApertureVisibility: CGFloat

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
                    return UVVertex(uv: contour.outerUV[outerOffset], screen: contour.outer[outerOffset])
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
            innerApertureVisibility = contour.innerApertureVisibility
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
            let featherWidth: CGFloat = 4.75 / 192.0
            let outerCoverage = pow(Self.smoothStep(
                edge0: 0,
                edge1: featherWidth,
                value: edgeDistances.outer
            ), 1.18)
            // Never lower alpha on the lip side of the inner contour. A
            // symmetric feather puts a translucent strip directly on the
            // visible mucosal edge and becomes a bright seam when projected at
            // an angle. The aperture-only transition is added below by
            // fillInnerSamplingFeather.
            let innerCoverage: CGFloat = 1
            // Keep more of the transition translucent. This produces a soft
            // cosmetic feather instead of a mathematically sharp silhouette.
            return CanonicalAlphaSample(
                alpha: Float(outerCoverage * innerCoverage),
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
            let innerDistance = sqrt(innerDistanceSquared)
            // Mirror the Metal mask: the edge itself is solid, and only the
            // part inside the aperture fades. Narrow gaps are covered fully;
            // a clearly open mouth retains a transparent centre.
            let opening = min(max(innerApertureVisibility, 0), 1)
            let solidInnerWidth = 0.020 - opening * 0.016
            let transitionWidth = 0.006 - opening * 0.001
            let coverage = 1 - Self.smoothStep(
                edge0: solidInnerWidth,
                edge1: solidInnerWidth + transitionWidth,
                value: innerDistance
            )
            return coverage > 0.001 ? coverage : nil
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
    func updateStyle(color: LipstickColor,
                     finish: LipFinish,
                     density: LipstickDensity,
                     texture: LipstickTexture) -> Bool {
        let oklch = color.oklch
        colorLock.lock()
        let nextColor = RGBColor(red: color.red, green: color.green, blue: color.blue)
        let colorDistance = Self.colorDistance(lipstickColor, nextColor)
        let nextSignature = Self.colorSignature(nextColor)
        let didChange = lipstickStyleSignature != nextSignature ||
            lipstickFinish != finish ||
            lipstickDensity != density ||
            lipstickTexture != texture
        if didChange {
            lipstickColor = nextColor
            lipstickFinish = finish
            lipstickDensity = density
            lipstickTexture = texture
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
            "lip_color update distance=\(String(format: "%.4f", Double(colorDistance))) finish=\(finish.rawValue) density=\(density.rawValue) texture=\(texture.rawValue) rgb=(\(String(format: "%.3f", Double(nextColor.red))),\(String(format: "%.3f", Double(nextColor.green))),\(String(format: "%.3f", Double(nextColor.blue)))) oklch=(\(String(format: "%.3f", Double(oklch.lightness))),\(String(format: "%.3f", Double(oklch.chroma))),\(String(format: "%.1f", Double(oklch.hueDegrees))))"
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

        // Keep the CPU fallback compact. Metal uses a moderately oversampled
        // target so the curved feather does not quantize into visible steps
        // when the mouth fills much of the screen. This remains far below the
        // old 384x192 target that caused multi-second first-frame work on A14.
        let pixelWidth = lowLatency ? 128 : 160
        let pixelHeight = lowLatency ? 64 : 80
        let metalPixelWidth = lowLatency ? 224 : 256
        let metalPixelHeight = lowLatency ? 112 : 128

        let style = currentLipstickStyle()

        let metalWasAvailable = metalCompositor.isAvailable
        if metalWasAvailable {
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
                finish: style.finish,
                density: style.density,
                texture: style.texture
            ) {
                LipDebugLog.throttled(
                    "lip_texture_path",
                    interval: 1.0,
                    "lip_texture path=metal fallback=false size=\(metalPixelWidth)x\(metalPixelHeight) lowLatency=\(lowLatency) finish=\(style.finish.rawValue)"
                )
                return texture
            }
            // A transient GPU failure must not immediately start the old
            // 400-580 ms CPU loop and contend with live landmark tracking.
            guard !metalCompositor.isAvailable else {
                LipDebugLog.throttled(
                    "lip_texture_path_metal_drop",
                    interval: 1.0,
                    "lip_texture path=metal fallback=false action=drop reason=transient_metal_nil size=\(metalPixelWidth)x\(metalPixelHeight) lowLatency=\(lowLatency) finish=\(style.finish.rawValue)"
                )
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
            LipDebugLog.throttled(
                "lip_texture_path",
                interval: 1.0,
                "lip_texture path=cpu_procedural fallback=true reason=\(metalWasAvailable ? "metal_disabled_after_failure" : "metal_unavailable") size=256x128 lowLatency=\(lowLatency) finish=\(style.finish.rawValue)"
            )
            return makeStableProceduralTexture(
                sampler: sampler,
                style: style,
                renderScale: renderScale,
                excludesInnerMouth: excludesInnerMouth
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
        var toneMap = [MaterialTone](repeating: .neutral, count: pixelWidth * pixelHeight)
        var activeTonePixels = [Bool](repeating: false, count: pixelWidth * pixelHeight)
        var cameraLumaSum: Float = 0
        var cameraLumaCount: Float = 0
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
                let mapIndex = y * pixelWidth + x

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
                let trustedCameraSample = cameraSample
                let lipBase = trustedCameraSample?.base ?? Self.stableLipBaseColor(u: u, topV: topV)
                let cosmeticCoverage = Self.cosmeticCoverage(
                    maskAlpha: sample.alpha,
                    u: u,
                    innerDistance: sample.innerDistance,
                    finish: style.finish
                )
                let baseAlpha = min(
                    cosmeticCoverage *
                    edgeNoise *
                    pigmentVariation.alpha *
                    Self.pigmentAlphaScale(
                        base: lipBase,
                        outerDistance: sample.outerDistance,
                        innerDistance: sample.innerDistance
                    ),
                    1
                )
                let alpha = baseAlpha
                let pigmentRGB = Self.pigmentBlendedColor(
                    lipstick: style.color,
                    variation: pigmentVariation.color,
                    texture: style.texture
                )
                if let cameraSample = trustedCameraSample {
                    toneMap[mapIndex] = Self.cameraTone(base: cameraSample.base, blurred: cameraSample.blurred)
                    cameraLumaSum += Self.luminance(cameraSample.blurred)
                    cameraLumaCount += 1
                }
                activeTonePixels[mapIndex] = true
                materialInputs[mapIndex] = PixelMaterialInput(
                    alpha: alpha,
                    lipBase: lipBase,
                    compensationBase: trustedCameraSample?.blurred ?? lipBase,
                    pigment: pigmentRGB,
                    u: u,
                    topV: topV
                )
            }
        }

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
            let effectiveAlpha = input.alpha
            guard effectiveAlpha > 0.006 else {
                continue
            }
            let isLowDensityGloss = style.finish == .gloss &&
                style.density == .low
            let mixedPigment = isLowDensityGloss ?
                Self.mixColor(
                    input.pigment,
                    input.compensationBase,
                    amount: Self.lowDensityGlossNaturalLipWeight
                ) :
                input.pigment
            let materialRGB = Self.materialColor(
                pigment: mixedPigment,
                base: input.lipBase,
                tone: stabilizedToneMap[mapIndex],
                u: input.u,
                topV: input.topV,
                finish: style.finish,
                texture: style.texture
            )
            let densityOpacity = isLowDensityGloss ?
                Self.lowDensityGlossCoverage /
                    LipstickDensity.high.pigmentCoverage :
                style.density.opacityScale
            let materialOpacity = densityOpacity *
                Self.cornerOpacityScale(u: input.u)
            let outputAlpha = effectiveAlpha * materialOpacity
            let correctiveColor: RGBColor
            let outputColorScale: Float
            if isLowDensityGloss {
                // The target RGB already contains the natural lip colour, so
                // compensate directly at the stronger output coverage instead
                // of revealing that colour only through transparency.
                correctiveColor = Self.premultipliedCompositingColor(
                    target: materialRGB,
                    base: input.compensationBase,
                    alpha: outputAlpha
                )
                outputColorScale = 1
            } else {
                correctiveColor = Self.premultipliedCompositingColor(
                    target: materialRGB,
                    base: input.compensationBase,
                    alpha: effectiveAlpha
                )
                outputColorScale = materialOpacity
            }
            let offset = mapIndex * 4
            rgba[offset] = Self.uint8(correctiveColor.red * outputColorScale)
            rgba[offset + 1] = Self.uint8(correctiveColor.green * outputColorScale)
            rgba[offset + 2] = Self.uint8(correctiveColor.blue * outputColorScale)
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
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
            )
            rgba = fallback.rgba
            usedFallback = true
            fallbackPaintedPixels = fallback.paintedPixels
        }
        let innerFeatherPaintedPixels = Self.fillInnerSamplingFeather(
            &rgba,
            sampler: sampler,
            pixelWidth: pixelWidth,
            pixelHeight: pixelHeight
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
        return texture
    }

    private func makeStableProceduralTexture(sampler: CanonicalLipSampler,
                                             style: (color: RGBColor,
                                                     finish: LipFinish,
                                                     density: LipstickDensity,
                                                     texture: LipstickTexture),
                                             renderScale: CGFloat,
                                             excludesInnerMouth: Bool) -> LipTexture? {
        let key = ProceduralTextureKey(
            colorSignature: Self.colorSignature(style.color),
            finish: style.finish,
            density: style.density,
            texture: style.texture,
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
                let isLowDensityGloss = finish == .gloss && density == .low
                let renderCoverage = isLowDensityGloss ?
                    lowDensityGlossCoverage :
                    density.pigmentCoverage
                let alpha = sample.alpha *
                    edgeGrain *
                    renderCoverage *
                    cornerOpacityScale(u: u)
                let lipBase = stableLipBaseColor(u: u, topV: topV)
                let targetPigment = pigmentBlendedColor(
                    lipstick: color,
                    variation: pigmentVariation.color,
                    texture: texture
                )
                let pigmentRGB = isLowDensityGloss ?
                    mixColor(
                        targetPigment,
                        lipBase,
                        amount: lowDensityGlossNaturalLipWeight
                    ) :
                    targetPigment
                let lipRGB = finishColor(
                    pigment: pigmentRGB,
                    lipBase: lipBase,
                    u: u,
                    topV: topV,
                    alpha: alpha,
                    finish: finish,
                    texture: texture
                )
                let offset = (y * pixelWidth + x) * 4
                rgba[offset] = uint8(lipRGB.red * alpha)
                rgba[offset + 1] = uint8(lipRGB.green * alpha)
                rgba[offset + 2] = uint8(lipRGB.blue * alpha)
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
        pixelHeight: Int
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
                                            texture: LipstickTexture) {
        colorLock.lock()
        let color = lipstickColor
        let finish = lipstickFinish
        let density = lipstickDensity
        let texture = lipstickTexture
        colorLock.unlock()
        return (color, finish, density, texture)
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
        if isToothLike(base) && innerDistance < 0.060 {
            return 0.24
        }
        return 1
    }

    private static func materialColor(pigment: RGBColor,
                                      base: RGBColor,
                                      tone: MaterialTone,
                                      u: CGFloat,
                                      topV: CGFloat,
                                      finish: LipFinish,
                                      texture: LipstickTexture) -> RGBColor {
        let cornerPosition = min(abs(Float(u) - 0.5) * 2, 1)
        let cornerRegion = smoothStep((cornerPosition - 0.62) / 0.34)
        let innerSeamRegion = 1 - smoothStep((abs(Float(topV) - 0.50) - 0.018) / 0.085)
        // The preset sRGB is authoritative. Camera samples contribute only
        // relative relief below, never the underlying lip hue or brightness.
        let saturatedPigment = pigment
        let finishDetailStrength: Float
        if finish == .matte {
            finishDetailStrength = max(0.90, min(tone.detail, 1.08))
        } else if tone.detail < 1 {
            let shadowBoost: Float = finish == .satin ? 1.30 : 1.14
            finishDetailStrength = max(0.90, min(1 - (1 - tone.detail) * shadowBoost, 1.05))
        } else {
            let highlightScale: Float = finish == .satin ? 0.92 : 0.96
            finishDetailStrength = max(0.95, min(1 + (tone.detail - 1) * highlightScale, 1.05))
        }
        let detailStrength = 1 +
            (finishDetailStrength - 1) * texture.detailResponse
        let shadedPigment = RGBColor(
            red: clamp01(saturatedPigment.red * detailStrength),
            green: clamp01(saturatedPigment.green * detailStrength),
            blue: clamp01(saturatedPigment.blue * detailStrength)
        )

        let finishedPigment: RGBColor
        if finish == .satin || finish == .gloss {
            // Both finishes follow camera detail. Gloss strengthens only the
            // reflection already present in the captured lip pixels; it does
            // not place white highlight shapes at fixed UV coordinates.
            let isGloss = finish == .gloss
            let rawHighlight = smoothStep(
                (detailStrength - (isGloss ? 1.008 : 1.005)) /
                    (isGloss ? 0.034 : 0.045)
            )
            let highlightSuppression =
                (1 - cornerRegion * 0.82) *
                (1 - innerSeamRegion * (isGloss ? 0.30 : 0.74))
            let highlightAmount: Float
            if isGloss {
                let concentratedHighlight = pow(
                    rawHighlight,
                    texture.highlightConcentration
                )
                let peakLimit = 0.20 * texture.highlightLimitResponse
                highlightAmount = min(
                    (pow(rawHighlight, 1.35) * 0.035 +
                        concentratedHighlight * 0.14) *
                        texture.highlightResponse *
                        highlightSuppression,
                    peakLimit
                )
            } else {
                let baseHighlightStrength: Float = 0.18
                highlightAmount = min(
                    rawHighlight *
                        baseHighlightStrength *
                        texture.highlightResponse *
                        highlightSuppression,
                    baseHighlightStrength * texture.highlightLimitResponse
                )
            }
            finishedPigment = RGBColor(
                red: clamp01(shadedPigment.red + (1 - shadedPigment.red) * highlightAmount),
                green: clamp01(shadedPigment.green + (1 - shadedPigment.green) * highlightAmount),
                blue: clamp01(shadedPigment.blue + (1 - shadedPigment.blue) * highlightAmount)
            )
        } else {
            finishedPigment = shadedPigment
        }

        // Preserve a genuine dark fold only at the extreme corner. Scale all
        // channels together so the selected lipstick hue cannot drift.
        let cornerTip = smoothStep((cornerPosition - 0.86) / 0.14)
        let finishedLuminance = luminance(finishedPigment)
        let capturedCornerLuminance = luminance(base)
        let shadowEvidence = smoothStep(
            (finishedLuminance - capturedCornerLuminance - 0.025) / 0.155
        )
        let retainedLuminance = max(
            capturedCornerLuminance,
            finishedLuminance * 0.58
        )
        let targetLuminance = finishedLuminance +
            (retainedLuminance - finishedLuminance) *
            cornerTip * shadowEvidence * 0.76
        let luminanceScale = targetLuminance / max(finishedLuminance, 0.0001)
        return RGBColor(
            red: clamp01(finishedPigment.red * luminanceScale),
            green: clamp01(finishedPigment.green * luminanceScale),
            blue: clamp01(finishedPigment.blue * luminanceScale)
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
            shadow: 1 - shadowAmount * 0.060
        )
    }

    private static func cosmeticCoverage(maskAlpha: Float,
                                         u: CGFloat,
                                         innerDistance: CGFloat,
                                         finish: LipFinish) -> Float {
        let innerDistance = Float(innerDistance)
        let innerFade: Float
        switch finish {
        case .matte:
            innerFade = 0.44 + 0.56 * smoothStep((innerDistance - 0.004) / 0.040)
        case .satin:
            innerFade = 0.34 + 0.66 * smoothStep((innerDistance - 0.006) / 0.050)
        case .gloss:
            innerFade = 0.40 + 0.60 * smoothStep((innerDistance - 0.004) / 0.044)
        }
        // Build corrective colour against the high-density reference. The
        // selected product density is applied afterwards to premultiplied RGB
        // and alpha together, so lower density genuinely reveals the real lip
        // instead of being cancelled out by colour compensation.
        return clamp01(
            maskAlpha *
                innerFade *
                LipstickDensity.high.pigmentCoverage
        )
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
                                            variation: Float,
                                            texture: LipstickTexture) -> RGBColor {
        guard texture == .mousse else {
            return lipstick
        }
        // A fixed UV-space micro-relief gives the fallback a diffuse mousse
        // surface without changing the catalogue hue or shimmering over time.
        let relief = 0.975 + min(max(variation, 0), 1) * 0.050
        return RGBColor(
            red: clamp01(lipstick.red * relief),
            green: clamp01(lipstick.green * relief),
            blue: clamp01(lipstick.blue * relief)
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

    private static func finishColor(pigment: RGBColor,
                                    lipBase: RGBColor,
                                    u: CGFloat,
                                    topV: CGFloat,
                                    alpha: Float,
                                    finish: LipFinish,
                                    texture: LipstickTexture) -> RGBColor {
        guard finish == .gloss else {
            return pigment
        }

        // This path is used only when camera-aware rendering cannot provide a
        // valid texture. Keep a gentle glaze, but never substitute fixed white
        // spots for reflections that were not observed by the camera.
        let wetVeil = glossVeil(u: u, topV: topV)
        let veilScale: Float
        switch texture {
        case .creamy:
            veilScale = 0.040
        case .mousse:
            veilScale = 0.020
        case .liquid:
            veilScale = 0.052
        }
        let veil = min(wetVeil * veilScale, veilScale)
        let alphaGuard = max(0.40, min(alpha, 0.99))

        let glazedRed = clamp01(pigment.red * 1.05 + 0.020)
        let glazedGreen = clamp01(pigment.green * 1.02 + 0.010)
        let glazedBlue = clamp01(pigment.blue * 1.06 + 0.018)
        let glazeBlend = veil * alphaGuard

        return RGBColor(
            red: clamp01(pigment.red + (glazedRed - pigment.red) * glazeBlend),
            green: clamp01(pigment.green + (glazedGreen - pigment.green) * glazeBlend),
            blue: clamp01(pigment.blue + (glazedBlue - pigment.blue) * glazeBlend)
        )
    }

    private static func cupidHighlightGate(u: CGFloat, topV: CGFloat) -> Float {
        let x = Float(u)
        let y = Float(topV)
        let cupidNotch = exp(-pow((x - 0.50) / 0.038, 2))
        let cupidCurveY: Float = 0.155 + cupidNotch * 0.022
        return irregularLocalHighlight(
            localX: (x - 0.50) / 0.115,
            localY: (y - cupidCurveY) / 0.024,
            falloff: 1.12,
            phase: 0.90
        )
    }

    private static func lowerSatinHighlightGate(u: CGFloat,
                                                 topV: CGFloat) -> Float {
        let x = Float(u)
        let y = Float(topV)
        let localX = (x - 0.50) / 0.34
        let curveY: Float = 0.645 +
            pow(clamp01(abs(localX)), 1.60) * 0.025
        return exp(-pow(localX, 4)) *
            exp(-pow((y - curveY) / 0.105, 2))
    }

    private static func lowerGlossHighlightGate(u: CGFloat,
                                                topV: CGFloat) -> Float {
        let x = Float(u)
        let y = Float(topV)
        let leftHighlight = irregularHighlight(
            x: x,
            y: y,
            centerX: 0.425,
            centerY: 0.685,
            radiusX: 0.073,
            radiusY: 0.050,
            angleRadians: -0.07,
            falloff: 1.16,
            phase: 1.30
        )
        let rightHighlight = irregularHighlight(
            x: x,
            y: y,
            centerX: 0.575,
            centerY: 0.685,
            radiusX: 0.112,
            radiusY: 0.064,
            angleRadians: 0.06,
            falloff: 1.16,
            phase: 3.10
        ) * 0.90
        let centerSeparation = 1 - irregularHighlight(
            x: x,
            y: y,
            centerX: 0.505,
            centerY: 0.685,
            radiusX: 0.019,
            radiusY: 0.068,
            angleRadians: 0,
            falloff: 1.55,
            phase: 0.20
        ) * 0.88
        return max(leftHighlight, rightHighlight) * centerSeparation
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
            radiusY: 0.009,
            falloff: 1.55
        )
        let lowerBrokenLineB = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.515,
            centerY: 0.700,
            radiusX: 0.070,
            radiusY: 0.011,
            falloff: 1.55
        )
        let lowerBrokenLineC = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.590,
            centerY: 0.688,
            radiusX: 0.042,
            radiusY: 0.009,
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
            centerX: 0.425,
            centerY: 0.685,
            radiusX: 0.073,
            radiusY: 0.050,
            falloff: 1.16
        )
        let lowerRightWet = elongatedHighlight(
            x: x,
            y: y,
            centerX: 0.575,
            centerY: 0.685,
            radiusX: 0.112,
            radiusY: 0.064,
            falloff: 1.16
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
            radiusMinor: 0.009,
            angleRadians: 0.30,
            falloff: 1.85
        )
        let cupidRightRidge = rotatedHighlight(
            x: x,
            y: y,
            centerX: 0.538,
            centerY: 0.214,
            radiusMajor: 0.090,
            radiusMinor: 0.009,
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
        let specular = lowerWetSurface * 0.0 +
            lowerHotBand * 0.0 +
            lowerGlossCore * 0.0 +
            lowerGlossSoft * 0.0 +
            lowerBrokenLineA * 0.0 +
            lowerBrokenLineB * 0.0 +
            lowerBrokenLineC * 0.0 +
            lowerBrokenLineD * 0.0 +
            lowerLeftWet * 0.38 +
            lowerRightWet * 0.32 +
            upperWetArc * 0.0 +
            cupidLeftRidge * 0.48 +
            cupidRightRidge * 0.48 +
            cupidLeftSoft * 0.10 +
            cupidRightSoft * 0.10 +
            cupidCenterJoin * 0.0 +
            lowerStreaks * 0.12 +
            upperStreaks * 0.10 +
            tinyLowerSpark * 0.24 +
            tinyUpperSpark * 0.25
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

    private static func irregularHighlight(x: Float,
                                           y: Float,
                                           centerX: Float,
                                           centerY: Float,
                                           radiusX: Float,
                                           radiusY: Float,
                                           angleRadians: Float,
                                           falloff: Float,
                                           phase: Float) -> Float {
        let deltaX = x - centerX
        let deltaY = y - centerY
        let cosine = cos(angleRadians)
        let sine = sin(angleRadians)
        let localX = (deltaX * cosine + deltaY * sine) /
            max(radiusX, 0.001)
        let localY = (-deltaX * sine + deltaY * cosine) /
            max(radiusY, 0.001)
        return irregularLocalHighlight(
            localX: localX,
            localY: localY,
            falloff: falloff,
            phase: phase
        )
    }

    private static func irregularLocalHighlight(localX: Float,
                                                localY: Float,
                                                falloff: Float,
                                                phase: Float) -> Float {
        let distanceSquared = localX * localX + localY * localY
        let edgeWeight = smoothStep((distanceSquared - 0.16) / 0.76)
        let edgeVariation =
            sin(localX * 9.7 + localY * 6.1 + phase) * 0.055 +
            sin(localX * 17.3 - localY * 11.9 + phase * 1.7) * 0.030
        return pow(
            clamp01(1 - distanceSquared + edgeVariation * edgeWeight),
            falloff
        )
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
