import Foundation
import UIKit

enum LipFinish: Int, Hashable, Codable {
    case matte
    case gloss
    case satin

    var displayName: String {
        switch self {
        case .matte:
            return "Матовая"
        case .satin:
            return "Сатиновая"
        case .gloss:
            return "Блеск"
        }
    }
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
            return 0.54
        case .medium:
            return 0.78
        case .high:
            return 0.92
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
            return 1.32
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

/// Perceptual colour coordinates derived from an sRGB catalogue colour.
/// These values are calculated by the app; they are not catalogue fields.
struct OKLCHColor: Hashable {
    /// Perceptual lightness in `0...1`.
    let lightness: Float
    /// Perceptual chroma. Most sRGB lipstick colours fall below `0.3`.
    let chroma: Float
    /// Hue angle in degrees in `0..<360`.
    let hueDegrees: Float

    fileprivate init(sRGBRed red: Float, green: Float, blue: Float) {
        let linearRed = Self.linearComponent(red)
        let linearGreen = Self.linearComponent(green)
        let linearBlue = Self.linearComponent(blue)

        let l = 0.412_221_470_8 * linearRed +
            0.536_332_536_3 * linearGreen +
            0.051_445_992_9 * linearBlue
        let m = 0.211_903_498_2 * linearRed +
            0.680_699_545_1 * linearGreen +
            0.107_396_956_6 * linearBlue
        let s = 0.088_302_461_9 * linearRed +
            0.281_718_837_6 * linearGreen +
            0.629_978_700_5 * linearBlue

        let cubeRootL = Self.cubeRoot(l)
        let cubeRootM = Self.cubeRoot(m)
        let cubeRootS = Self.cubeRoot(s)
        let okLightness = 0.210_454_255_3 * cubeRootL +
            0.793_617_785 * cubeRootM -
            0.004_072_046_8 * cubeRootS
        let okA = 1.977_998_495_1 * cubeRootL -
            2.428_592_205 * cubeRootM +
            0.450_593_709_9 * cubeRootS
        let okB = 0.025_904_037_1 * cubeRootL +
            0.782_771_766_2 * cubeRootM -
            0.808_675_766 * cubeRootS

        lightness = okLightness
        chroma = hypot(okA, okB)
        let degrees = atan2(okB, okA) * 180 / .pi
        hueDegrees = degrees >= 0 ? degrees : degrees + 360
    }

    private static func linearComponent(_ component: Float) -> Float {
        let component = min(max(component, 0), 1)
        if component <= 0.04045 {
            return component / 12.92
        }
        return pow((component + 0.055) / 1.055, 2.4)
    }

    private static func cubeRoot(_ value: Float) -> Float {
        guard value != 0 else {
            return 0
        }
        return value.sign == .minus ? -pow(-value, 1.0 / 3.0) : pow(value, 1.0 / 3.0)
    }

}

/// Source colour stored by a product preset. OKLCH is derived automatically
/// from the familiar six-digit sRGB hex value.
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

    var oklch: OKLCHColor {
        OKLCHColor(sRGBRed: red, green: green, blue: blue)
    }
}
