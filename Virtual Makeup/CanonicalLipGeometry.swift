import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

enum CanonicalLipGeometry {
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

    // Complete the inner aperture geometrically and let the texture alpha—not
    // an open mesh edge—decide whether the mouth is visible. When the mouth is
    // open these triangles sample transparent texels. At closure they provide
    // continuous coverage behind the two lip halves, eliminating raster cracks.
    static let innerFillTriangles: [(Int, Int, Int)] = {
        let lower = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        let upper = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        var triangles: [(Int, Int, Int)] = []
        triangles.reserveCapacity(18)
        for segment in 0..<(lower.count - 1) {
            let lowerLeft = lower[segment]
            let lowerRight = lower[segment + 1]
            let upperLeft = upper[segment]
            let upperRight = upper[segment + 1]
            if lowerLeft != upperLeft {
                triangles.append((lowerLeft, lowerRight, upperLeft))
            }
            if lowerRight != upperRight {
                triangles.append((lowerRight, upperRight, upperLeft))
            }
        }
        return triangles
    }()

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
