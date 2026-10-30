import Foundation
import UIKit
import simd

/// Camera-lit wax pencil, independent of the selected lipstick finish.
struct LinerMaterialProfile: Equatable {
    let pigmentDensity: Float
    let detailStrength: Float
    let diffuseLightResponse: Float
    let shadowRetention: Float
    let maximumLightStops: Float
    let maximumShadowStops: Float
    let realHighlightStrength: Float

    static let wax = LinerMaterialProfile(
        pigmentDensity: 0.86,
        detailStrength: 0.42,
        diffuseLightResponse: 0.56,
        shadowRetention: 0.78,
        maximumLightStops: 0.42,
        maximumShadowStops: 1.10,
        realHighlightStrength: 0
    )
}

struct LinerRenderStyle {
    let color: RGBColor // straight sRGB from the selected UIColor
    let profile: LinerMaterialProfile
    let isEnabled: Bool

    static func snapshot(color: UIColor, opacity: Double) -> LinerRenderStyle {
        func bounded(_ value: Float, fallback: Float) -> Float {
            value.isFinite ? min(max(value, 0), 1) : fallback
        }
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)
        let converted = srgb.flatMap {
            color.cgColor.converted(to: $0, intent: .defaultIntent, options: nil)
        }
        let components = converted?.components ?? []
        let rgb: RGBColor
        if components.count >= 3 {
            rgb = RGBColor(red: Float(components[0]),
                           green: Float(components[1]),
                           blue: Float(components[2]))
        } else {
            var red: CGFloat = 0
            var green: CGFloat = 0
            var blue: CGFloat = 0
            var alpha: CGFloat = 0
            if color.getRed(&red, green: &green, blue: &blue, alpha: &alpha) {
                rgb = RGBColor(red: Float(red), green: Float(green), blue: Float(blue))
            } else {
                rgb = RGBColor(red: 0.47, green: 0.16, blue: 0.19)
            }
        }
        return LinerRenderStyle(
            color: RGBColor(
                red: bounded(rgb.red, fallback: 0.47),
                green: bounded(rgb.green, fallback: 0.16),
                blue: bounded(rgb.blue, fallback: 0.19)
            ),
            profile: .wax,
            isEnabled: opacity.isFinite && opacity > 0
        )
    }
}

enum LipFinish: Int, Hashable, Codable {
    case matte
    case gloss
    case satin
    case oil
    case tint
    case shimmer

    var displayName: String {
        switch self {
        case .matte:
            return "Матовая"
        case .satin:
            return "Сатиновая"
        case .gloss:
            return "Блеск"
        case .oil:
            return "Масло"
        case .tint:
            return "Тинт"
        case .shimmer:
            return "Шиммерная"
        }
    }

    var isWet: Bool {
        self == .gloss || self == .oil || self == .shimmer
    }

    /// Tint has a recognisable stained baseline while product density remains
    /// meaningful. A medium tint lands between medium satin and high matte.
    func pigmentCoverage(for density: LipstickDensity) -> Float {
        guard self == .tint else {
            return density.pigmentCoverage
        }
        let tintBaseline: Float = 0.88
        return density.pigmentCoverage +
            (tintBaseline - density.pigmentCoverage) * 0.65
    }

    /// Wet colour products remain influenced by the wearer's lip tone even
    /// when their catalogue pigment density increases.
    func naturalLipColorWeight(for density: LipstickDensity) -> Float {
        switch self {
        case .gloss, .oil, .shimmer:
            switch density {
            case .low:
                return 0.80
            case .medium:
                return 0.54
            case .high:
                return 0.20
            }
        case .tint:
            return 0.18
        case .matte, .satin:
            return 0
        }
    }
}

enum LipShadingDebugMode: Int, Hashable {
    case off
    case capturedShape
}

/// Catalogue product category supplied independently from its visual formula.
/// Existing product constructors default this field to `lipstick`, preserving
/// the behaviour of cards created before the category was introduced.
enum LipProductType: String, CaseIterable, Hashable, Codable {
    case lipstick
    case oil
    case tint
    case gloss

    var displayName: String {
        switch self {
        case .lipstick:
            return "Помада"
        case .oil:
            return "Масло"
        case .tint:
            return "Тинт"
        case .gloss:
            return "Блеск"
        }
    }
}

/// Coverage supplied by product metadata. It is deliberately independent of
/// colour and finish: two satin lipsticks may have different coverage.
enum LipstickDensity: String, CaseIterable, Hashable, Codable {
    case low
    case medium
    case high

    var displayName: String {
        switch self {
        case .low:
            return "Низкая"
        case .medium:
            return "Средняя"
        case .high:
            return "Высокая"
        }
    }

    /// Physical pigment coverage used by the compositor. High density is the
    /// colour-compensation reference; lower values intentionally reveal more
    /// of the captured natural lip colour.
    var pigmentCoverage: Float {
        switch self {
        case .low:
            return 0.57
        case .medium:
            return 0.68
        case .high:
            return 1
        }
    }

    var opacityScale: Float {
        pigmentCoverage / LipstickDensity.high.pigmentCoverage
    }
}

/// Formula texture supplied by product metadata.
enum LipstickTexture: String, CaseIterable, Hashable, Codable {
    case creamy
    case mousse
    case liquid

    var displayName: String {
        switch self {
        case .creamy:
            return "Кремовая"
        case .mousse:
            return "Муссовая"
        case .liquid:
            return "Жидкая"
        }
    }

    /// Scales camera-derived lip relief around the neutral value of one.
    var detailResponse: Float {
        switch self {
        case .creamy:
            return 0.92
        case .mousse:
            return 0.87
        case .liquid:
            return 0.94
        }
    }

    /// Mousse diffuses existing reflections instead of drawing glossy peaks.
    var highlightResponse: Float {
        switch self {
        case .creamy:
            return 1.02
        case .mousse:
            return 0.4
        case .liquid:
            return 1.65
        }
    }

    var highlightLimitResponse: Float {
        switch self {
        case .creamy:
            return 1
        case .mousse:
            return 0.56
        case .liquid:
            return 1.25
        }
    }

    /// Selects the strongest core of the real camera reflection without
    /// reducing its peak brightness. Liquid keeps a smaller core than creamy.
    var highlightConcentration: Float {
        switch self {
        case .creamy:
            return 2.5
        case .mousse:
            return 1.35
        case .liquid:
            return 1.85
        }
    }
}

/// Resolved material coefficients owned by a catalogue product. Renderers
/// consume this value instead of independently interpreting finish, density,
/// and texture metadata, so a real SKU can be tuned without adding another
/// branch to the real-time compositing path.
struct LipMaterialProfile: Hashable {
    var pigmentCoverage: Float
    var naturalLipColorWeight: Float
    var capturedDetailStrength: Float
    var detailResponse: Float
    var diffuseLightResponse: Float
    var shadowRetention: Float
    var capturedShadowMaximumStops: Float
    var capturedDiffuseLightMaximumStops: Float
    var capturedHighlightStrength: Float
    var capturedHighlightMaximum: Float
    var highlightConcentration: Float
    var shineMaskMinimum: Float
    var shineMaskMaximum: Float
    var roughness: Float
    var specularStrength: Float
    var highlightRetention: Float
    var microTextureRetention: Float
    var wetInnerEdgeStrength: Float
    // IOS_REFERENCE camera-highlight controls. Roughness here is a concentration
    // multiplier with neutral value 1, separate from surface-normal roughness.
    var cameraHighlightExponent: Float
    var cameraHighlightGain: Float
    var cameraSampleScale: Float
    var cameraHighlightSize: Float
    var cameraHighlightThreshold: Float
    var cameraRoughnessTuning: Float
    /// Scales pearlescent optical coverage without changing the contour mask.
    var shimmerIntensity: Float
    var paleShadeMaximumLuminanceReduction: Float

    static func standard(finish: LipFinish,
                         density: LipstickDensity,
                         texture: LipstickTexture) -> LipMaterialProfile {
        let finishDetailStrength: Float
        let finishHighlightStrength: Float
        let finishHighlightMaximum: Float
        let diffuseLightResponse: Float
        let shineMaskMinimum: Float
        let shineMaskMaximum: Float
        let maximumLightShadeReduction: Float
        let capturedDiffuseLightMaximumStops: Float
        let optics: (Float, Float, Float, Float, Float)
        let cameraHighlightExponent: Float
        let cameraHighlightGain: Float

        switch finish {
        case .matte:
            optics = (0.82, 0.07, 0.24, 0.94, 0.03)
            finishDetailStrength = 1.9
            finishHighlightStrength = 0.35
            finishHighlightMaximum = 0.035
            cameraHighlightExponent = 0
            cameraHighlightGain = 0.2
            diffuseLightResponse = 0.45
            shineMaskMinimum = 0.40
            shineMaskMaximum = 0.80
            maximumLightShadeReduction = 0.18
            capturedDiffuseLightMaximumStops = 0.22
        case .satin:
            optics = (0.48, 0.20, 0.68, 0.86, 0.11)
            finishDetailStrength = 1.7
            finishHighlightStrength = 0.75
            finishHighlightMaximum = 0.45
            cameraHighlightExponent = 1.90
            cameraHighlightGain = 0.35
            diffuseLightResponse = 0.72
            shineMaskMinimum = 0.53
            shineMaskMaximum = 0.68
            maximumLightShadeReduction = 0.14
            capturedDiffuseLightMaximumStops = 0.30
        case .gloss:
            optics = (0.20, 0.46, 1.00, 0.72, 0.32)
            finishDetailStrength = 1.6
            finishHighlightStrength = 1.40 * 0.50
            finishHighlightMaximum = 0.40 * 0.50
            cameraHighlightExponent = 1.30
            cameraHighlightGain = 0.24
            diffuseLightResponse = 0.88
            shineMaskMinimum = 0.34
            shineMaskMaximum = 0.78
            maximumLightShadeReduction = 0.08
            capturedDiffuseLightMaximumStops = 0.36
        case .oil:
            optics = (0.15, 0.43, 1.00, 0.85, 0.15)
            finishDetailStrength = 1.9
            finishHighlightStrength = 1.80 * 0.67
            finishHighlightMaximum = 0.60 * 0.80
            cameraHighlightExponent = 1.05
            cameraHighlightGain = 0.76
            diffuseLightResponse = 0.8
            shineMaskMinimum = 0.30
            shineMaskMaximum = 0.72
            maximumLightShadeReduction = 0.08
            capturedDiffuseLightMaximumStops = 0.40
        case .tint:
            optics = (1.00, 0, 0, 0, 0)
            finishDetailStrength = 1.8
            finishHighlightStrength = 0
            finishHighlightMaximum = 0
            cameraHighlightExponent = 0
            cameraHighlightGain = 0
            diffuseLightResponse = 0.62
            shineMaskMinimum = 1
            shineMaskMaximum = 1
            maximumLightShadeReduction = 0.10
            capturedDiffuseLightMaximumStops = 0.24
        case .shimmer:
            optics = (0.20, 0.46, 1.00, 0.72, 0.32)
            finishDetailStrength = 1.6
            finishHighlightStrength = 0.5
            finishHighlightMaximum = 0.3
            cameraHighlightExponent = 0
            cameraHighlightGain = 0
            diffuseLightResponse = 0.88
            shineMaskMinimum = 0.29
            shineMaskMaximum = 0.5
            maximumLightShadeReduction = 0.08
            capturedDiffuseLightMaximumStops = 0.36
        }

        let shadowRetention: Float
        let capturedShadowMaximumStops: Float
        let lightShadeDensityResponse: Float
        switch density {
        case .low:
            shadowRetention = 0.15
            capturedShadowMaximumStops = 1.00
            lightShadeDensityResponse = 0.42
        case .medium:
            shadowRetention = 0.52
            capturedShadowMaximumStops = 1.22
            lightShadeDensityResponse = 0.72
        case .high:
            shadowRetention = 0.72
            capturedShadowMaximumStops = 1.50
            lightShadeDensityResponse = 1
        }

        // Liquid satin should keep a soft satin reflection. Its small texture
        // lift must not inherit the much stronger liquid gloss response.
        let liquidSatin = finish == .satin && texture == .liquid
        let textureHighlightResponse: Float = liquidSatin ? 1.08 : texture.highlightResponse
        let textureHighlightLimit: Float = liquidSatin ? 1.04 : texture.highlightLimitResponse
        let textureHighlightConcentration: Float = liquidSatin ?
            LipstickTexture.creamy.highlightConcentration : texture.highlightConcentration

        return LipMaterialProfile(
            pigmentCoverage: finish.pigmentCoverage(for: density) *
                (finish == .oil ? 0.78 : 1),
            naturalLipColorWeight: finish.naturalLipColorWeight(for: density),
            capturedDetailStrength: finishDetailStrength * texture.detailResponse,
            detailResponse: texture.detailResponse,
            diffuseLightResponse: diffuseLightResponse,
            shadowRetention: shadowRetention,
            capturedShadowMaximumStops: capturedShadowMaximumStops,
            capturedDiffuseLightMaximumStops:
                capturedDiffuseLightMaximumStops,
            capturedHighlightStrength: finishHighlightStrength * textureHighlightResponse,
            capturedHighlightMaximum: finishHighlightMaximum * textureHighlightLimit,
            highlightConcentration: textureHighlightConcentration,
            shineMaskMinimum: shineMaskMinimum,
            shineMaskMaximum: shineMaskMaximum,
            roughness: optics.0,
            specularStrength: optics.1,
            highlightRetention: optics.2,
            microTextureRetention: optics.3,
            wetInnerEdgeStrength: optics.4,
            cameraHighlightExponent: cameraHighlightExponent,
            cameraHighlightGain: cameraHighlightGain,
            cameraSampleScale: 1,
            cameraHighlightSize: 1,
            cameraHighlightThreshold: 1,
            cameraRoughnessTuning: 1,
            shimmerIntensity: finish == .shimmer ? 3.0 : 0,
            paleShadeMaximumLuminanceReduction:
                maximumLightShadeReduction * lightShadeDensityResponse
        )
    }
}

/// Lighting-only surface in the viewport orientation of the captured camera frame.
/// Its height never changes the tracked screen contour or canonical UV map.
struct LipOpticalSurface {
    struct Sample {
        let normal: SIMD3<Float>
        let thickness: Float
        let arc: Float
        let across: Float
        let transverse: SIMD2<Float>
    }

    private let upperOuter: [SIMD2<Float>]
    private let upperInner: [SIMD2<Float>]
    private let lowerOuter: [SIMD2<Float>]
    private let lowerInner: [SIMD2<Float>]

    init?(contour: LipContour) {
        guard contour.outer.count == 20, contour.inner.count == 20 else {
            return nil
        }
        func points(_ source: [CGPoint], _ indices: [Int]) -> [SIMD2<Float>] {
            indices.map { SIMD2(Float(source[$0].x), Float(source[$0].y)) }
        }
        upperOuter = points(contour.outer, [0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
        upperInner = points(contour.inner, [0, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
        lowerOuter = points(contour.outer, Array(0...10))
        lowerInner = points(contour.inner, Array(0...10))
    }

    func sample(at point: CGPoint, side: Float = 0) -> Sample {
        let target = SIMD2(Float(point.x), Float(point.y))
        let upper = nearest(target, outer: upperOuter, inner: upperInner)
        let lower = nearest(target, outer: lowerOuter, inner: lowerInner)
        let useUpper = side < 0 || (side == 0 && upper.distance < lower.distance)
        let chosen = useUpper ? upper : lower
        let outer = useUpper ? upperOuter : lowerOuter
        let inner = useUpper ? upperInner : lowerInner
        let arc = chosen.arc
        let across = chosen.across
        let left = position(arc: max(arc - 0.01, 0), across: across, outer: outer, inner: inner)
        let right = position(arc: min(arc + 0.01, 1), across: across, outer: outer, inner: inner)
        let edge = position(arc: arc, across: max(across - 0.02, 0), outer: outer, inner: inner)
        let seam = position(arc: arc, across: min(across + 0.02, 1), outer: outer, inner: inner)
        let tangent = right - left
        let transverse = seam - edge
        var normal = simd_cross(tangent, transverse)
        if normal.z < 0 { normal = -normal }
        let length = simd_length(normal)
        if !length.isFinite || length < 0.0001 {
            normal = SIMD3(0, 0, 1)
        } else {
            normal /= length
        }
        let pair = interpolated(arc: arc, outer: outer, inner: inner)
        let acrossVector = pair.1 - pair.0
        let transverseLength = simd_length(acrossVector)
        let transverseDirection = transverseLength > 0.0001 && transverseLength.isFinite ?
            acrossVector / transverseLength : SIMD2<Float>(0, useUpper ? 1 : -1)
        return Sample(normal: normal, thickness: simd_distance(pair.0, pair.1),
                      arc: arc, across: across, transverse: transverseDirection)
    }

    private func nearest(_ point: SIMD2<Float>, outer: [SIMD2<Float>],
                         inner: [SIMD2<Float>]) -> (arc: Float, across: Float, distance: Float) {
        var best = (arc: Float(0), across: Float(0), distance: Float.greatestFiniteMagnitude)
        for index in 0..<10 {
            let center0 = (outer[index] + inner[index]) * 0.5
            let center1 = (outer[index + 1] + inner[index + 1]) * 0.5
            let segment = center1 - center0
            let fraction = simd_clamp(simd_dot(point - center0, segment) /
                max(simd_dot(segment, segment), 0.0001), 0, 1)
            let arc = (Float(index) + fraction) / 10
            let pair = interpolated(arc: arc, outer: outer, inner: inner)
            let acrossVector = pair.1 - pair.0
            let across = simd_clamp(simd_dot(point - pair.0, acrossVector) /
                max(simd_dot(acrossVector, acrossVector), 0.0001), 0, 1)
            let projected = simd_mix(pair.0, pair.1, SIMD2(repeating: across))
            let distance = simd_distance(point, projected)
            if distance < best.distance { best = (arc, across, distance) }
        }
        return best
    }

    private func interpolated(arc: Float, outer: [SIMD2<Float>],
                              inner: [SIMD2<Float>]) -> (SIMD2<Float>, SIMD2<Float>) {
        let scaled = min(max(arc, 0), 1) * 10
        let index = min(Int(scaled), 9)
        let fraction = scaled - Float(index)
        return (simd_mix(outer[index], outer[index + 1], SIMD2(repeating: fraction)),
                simd_mix(inner[index], inner[index + 1], SIMD2(repeating: fraction)))
    }

    private func position(arc: Float, across: Float, outer: [SIMD2<Float>],
                          inner: [SIMD2<Float>]) -> SIMD3<Float> {
        let pair = interpolated(arc: arc, outer: outer, inner: inner)
        let screen = simd_mix(pair.0, pair.1, SIMD2(repeating: across))
        let thickness = simd_distance(pair.0, pair.1)
        let arcBulge = pow(max(sin(.pi * arc), 0), 0.65)
        let height = thickness * 0.72 * arcBulge * max(sin(.pi * across), 0)
        return SIMD3(screen.x, -screen.y, height)
    }
}

/// Source colour stored by a product preset as a six-digit sRGB hex value.
struct LipstickColor: Hashable {
    let sourceSRGBHex: UInt32

    var red: Float {
        Float((sourceSRGBHex >> 16) & 0xFF) / 255
    }

    var green: Float {
        Float((sourceSRGBHex >> 8) & 0xFF) / 255
    }

    var blue: Float {
        Float(sourceSRGBHex & 0xFF) / 255
    }

    var uiColor: UIColor {
        UIColor(
            red: CGFloat(red),
            green: CGFloat(green),
            blue: CGFloat(blue),
            alpha: 1
        )
    }
}
