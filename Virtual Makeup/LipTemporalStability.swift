import CoreGraphics
import Foundation

enum LipTemporalStability {
    enum MotionFrameStep {
        case first
        case repeated
        case advanced(TimeInterval)
        case discontinuous
    }

    static func motionFrameStep(current: TimeInterval?,
                                previous: TimeInterval?) -> MotionFrameStep {
        guard let current, current.isFinite else { return .discontinuous }
        guard let previous else { return .first }
        let interval = current - previous
        guard interval.isFinite else { return .discontinuous }
        if abs(interval) < 0.0005 { return .repeated }
        guard interval >= 0.008, interval <= 0.12 else { return .discontinuous }
        return .advanced(interval)
    }

    static func response(nominalAlpha: CGFloat, elapsed: TimeInterval,
                         nominalInterval: TimeInterval = 1.0 / 30.0) -> CGFloat {
        guard elapsed.isFinite, elapsed > 0, nominalInterval > 0 else { return 0 }
        let boundedAlpha = min(max(nominalAlpha, 0), 1)
        return 1 - pow(1 - boundedAlpha, CGFloat(elapsed / nominalInterval))
    }

    static func activity(displacement: CGFloat, noiseFloor: CGFloat,
                         fullActivityAtNominal: CGFloat, elapsed: TimeInterval,
                         nominalInterval: TimeInterval = 1.0 / 30.0) -> CGFloat {
        guard displacement.isFinite, noiseFloor.isFinite,
              fullActivityAtNominal > noiseFloor,
              nominalInterval > 0 else { return 0 }
        let interval = elapsed.isFinite && elapsed > 0 ?
            max(elapsed, 1.0 / 120.0) : nominalInterval
        // Subtract the per-observation noise floor before dividing by time.
        // Scaling the whole residual by 1/dt would amplify detector jitter at
        // higher callback rates and make an idle mouth appear active.
        let motionRate = max(displacement - noiseFloor, 0) / CGFloat(interval)
        let fullRate = (fullActivityAtNominal - noiseFloor) /
            CGFloat(nominalInterval)
        let t = min(max(motionRate / fullRate, 0), 1)
        return t * t * (3 - 2 * t)
    }

    static func rigidLead(requested: CGVector, previous: CGVector,
                          mouthWidth: CGFloat) -> CGVector {
        let distance = hypot(requested.dx, requested.dy)
        guard distance.isFinite, distance >= 0.01 else { return .zero }
        let direction = CGVector(dx: requested.dx / distance, dy: requested.dy / distance)
        let previousAlongDirection = previous.dx * direction.dx +
            previous.dy * direction.dy
        // A stopped or reversed head must never retain correction in the old direction.
        guard previousAlongDirection >= 0 else { return .zero }
        let previousMagnitude = hypot(previous.dx, previous.dy)
        let alpha: CGFloat = previousMagnitude < 0.01 ? 0.58 : 0.48
        let targetMagnitude = previousAlongDirection +
            (distance - previousAlongDirection) * alpha
        let maximumStep = min(max(mouthWidth * 0.012, 0.65), 1.10)
        let magnitude = min(max(targetMagnitude, 0), previousAlongDirection + maximumStep)
        guard magnitude >= 0.04 else { return .zero }
        return CGVector(dx: direction.dx * magnitude, dy: direction.dy * magnitude)
    }

    static func cameraPlaneOffsets(reference: CGPoint, right: CGPoint,
                                   up: CGPoint, observed: CGPoint,
                                   stepMeters: CGFloat) -> CGPoint? {
        guard stepMeters.isFinite, stepMeters > 0 else { return nil }
        let rx = (right.x - reference.x) / stepMeters
        let ry = (right.y - reference.y) / stepMeters
        let ux = (up.x - reference.x) / stepMeters
        let uy = (up.y - reference.y) / stepMeters
        let determinant = rx * uy - ry * ux
        guard determinant.isFinite, abs(determinant) > 1 else { return nil }
        let dx = observed.x - reference.x
        let dy = observed.y - reference.y
        let result = CGPoint(
            x: (dx * uy - dy * ux) / determinant,
            y: (rx * dy - ry * dx) / determinant
        )
        guard result.x.isFinite, result.y.isFinite else { return nil }
        return result
    }
}
