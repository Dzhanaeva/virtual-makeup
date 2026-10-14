import Foundation

enum LipInnerBoundaryLayout {
    // The visible lip stays painted through its inner edge. Recess shading
    // remains on the lip; an open mouth receives only subpixel antialiasing.
    static let referenceTextureHeight: CGFloat = 96
    static let mouthSolidPixels: CGFloat = 0.4
    static let mouthTransitionPixels: CGFloat = 0.75
    static let recessLipPixels: CGFloat = 1.40
    static let recessLipFraction: CGFloat = 0.28
    static let openRecessDarkening: CGFloat = 0.14
    static let closedRecessDarkening: CGFloat = 0.05

    /// Positive distance is inside the mouth; negative is on the lip surface.
    static func coverage(
        signedDistance: CGFloat,
        opening: CGFloat
    ) -> CGFloat {
        let distance = signedDistance * referenceTextureHeight
        let openCoverage = 1 - smoothStep(
            (distance - mouthSolidPixels) / mouthTransitionPixels
        )
        // A closed mouth has no transparent hole, regardless of how wide the
        // detector's inner polygon is. Its fold is represented by colour.
        let amount = smoothStep((opening - 0.05) / 0.20)
        return 1 + (openCoverage - 1) * amount
    }

    static func recessShade(
        signedDistance: CGFloat,
        lipThickness: CGFloat,
        opening: CGFloat
    ) -> CGFloat {
        let distance = signedDistance * referenceTextureHeight
        let lipRamp = min(
            recessLipPixels,
            max(lipThickness * referenceTextureHeight * recessLipFraction, 0.001)
        )
        let depth = smoothStep((distance + lipRamp) / (lipRamp + mouthSolidPixels))
        let amount = min(max(opening, 0), 1)
        let darkening = closedRecessDarkening +
            (openRecessDarkening - closedRecessDarkening) * amount
        return 1 - darkening * depth
    }

    private static func smoothStep(_ value: CGFloat) -> CGFloat {
        let t = min(max(value, 0), 1)
        return t * t * (3 - 2 * t)
    }
}

enum LipMouthClosure {
    static func isClosed(arOpening: CGFloat, jawOpen: CGFloat, smile: CGFloat,
                         detectedOpening: CGFloat?, wasClosed: Bool?) -> Bool {
        guard arOpening.isFinite, jawOpen.isFinite, smile.isFinite,
              arOpening >= 0, jawOpen >= 0 else { return false }
        let smileAmount = min(max(smile, 0), 1)
        let closeOpening = 0.018 + smileAmount * 0.018
        let openOpening = 0.032 + smileAmount * 0.020
        let detectorSuggestsOpening = detectedOpening.map {
            $0.isFinite && $0 >= 0.055
        } ?? false
        let corroboratedOpening = detectorSuggestsOpening &&
            (arOpening >= 0.024 + smileAmount * 0.015 || jawOpen >= 0.09)

        if wasClosed == true {
            return arOpening < openOpening && jawOpen < 0.12 &&
                !corroboratedOpening
        }
        let initial = wasClosed == nil
        return arOpening < (initial ? closeOpening + 0.004 : closeOpening) &&
            jawOpen < (initial ? 0.085 : 0.075)
    }
}
