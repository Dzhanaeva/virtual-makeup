import Foundation
import UIKit

enum LipFinish: Int, Hashable, Codable {
    case matte
    case gloss
    case satin
    case oil
    case tint

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
        }
    }

    var isWet: Bool {
        self == .gloss || self == .oil
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
        case .gloss, .oil:
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
            return 0.8
        case .liquid:
            return 0.94
        }
    }

    /// Mousse diffuses existing reflections instead of drawing glossy peaks.
    var highlightResponse: Float {
        switch self {
        case .creamy:
            return 0.5
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
            return 1.15
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

        switch finish {
        case .matte:
            finishDetailStrength = 1.9
            finishHighlightStrength = 0.8
            finishHighlightMaximum = 0.07
            diffuseLightResponse = 0.45
            shineMaskMinimum = 0.40
            shineMaskMaximum = 0.80
            maximumLightShadeReduction = 0.18
            capturedDiffuseLightMaximumStops = 0.22
        case .satin:
            finishDetailStrength = 1.7
            finishHighlightStrength = 1.35
            finishHighlightMaximum = 0.7
            diffuseLightResponse = 0.72
            shineMaskMinimum = 0.37
            shineMaskMaximum = 0.89
            maximumLightShadeReduction = 0.14
            capturedDiffuseLightMaximumStops = 0.30
        case .gloss:
            finishDetailStrength = 1.6
            finishHighlightStrength = 1.4
            finishHighlightMaximum = 0.4
            diffuseLightResponse = 0.88
            shineMaskMinimum = 0.34
            shineMaskMaximum = 0.78
            maximumLightShadeReduction = 0.08
            capturedDiffuseLightMaximumStops = 0.36
        case .oil:
            finishDetailStrength = 1.9
            finishHighlightStrength = 1.4
            finishHighlightMaximum = 0.5
            diffuseLightResponse = 0.8
            shineMaskMinimum = 0.3
            shineMaskMaximum = 0.72
            maximumLightShadeReduction = 0.08
            capturedDiffuseLightMaximumStops = 0.40
        case .tint:
            finishDetailStrength = 1.8
            finishHighlightStrength = 0
            finishHighlightMaximum = 0
            diffuseLightResponse = 0.62
            shineMaskMinimum = 1
            shineMaskMaximum = 1
            maximumLightShadeReduction = 0.10
            capturedDiffuseLightMaximumStops = 0.24
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

        return LipMaterialProfile(
            pigmentCoverage: finish.pigmentCoverage(for: density),
            naturalLipColorWeight: finish.naturalLipColorWeight(for: density),
            capturedDetailStrength: finishDetailStrength * texture.detailResponse,
            detailResponse: texture.detailResponse,
            diffuseLightResponse: diffuseLightResponse,
            shadowRetention: shadowRetention,
            capturedShadowMaximumStops: capturedShadowMaximumStops,
            capturedDiffuseLightMaximumStops:
                capturedDiffuseLightMaximumStops,
            capturedHighlightStrength: finishHighlightStrength * texture.highlightResponse,
            capturedHighlightMaximum: finishHighlightMaximum * texture.highlightLimitResponse,
            highlightConcentration: texture.highlightConcentration,
            shineMaskMinimum: shineMaskMinimum,
            shineMaskMaximum: shineMaskMaximum,
            paleShadeMaximumLuminanceReduction:
                maximumLightShadeReduction * lightShadeDensityResponse
        )
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
