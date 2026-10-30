import CoreGraphics
import Foundation

enum LipCameraFrameMapping {
    // MediaPipe landmarks use oriented source pixels; ARKit's display transform
    // starts in normalized captured-image space.
    static func imageToViewport(imageSize: CGSize,
                                viewportSize: CGSize,
                                inputToCaptured: CGAffineTransform,
                                capturedToViewport: CGAffineTransform) -> CGAffineTransform {
        CGAffineTransform(scaleX: 1 / imageSize.width,
                          y: 1 / imageSize.height)
            .concatenating(inputToCaptured)
            .concatenating(capturedToViewport)
            .concatenating(CGAffineTransform(scaleX: viewportSize.width,
                                             y: viewportSize.height))
    }

    static func isUsable(_ transform: CGAffineTransform) -> Bool {
        let determinant = transform.a * transform.d - transform.b * transform.c
        return transform.a.isFinite && transform.b.isFinite &&
            transform.c.isFinite && transform.d.isFinite &&
            transform.tx.isFinite && transform.ty.isFinite &&
            determinant.isFinite && abs(determinant) > 0.000_000_001
    }
}

enum LandmarkSubmitPacing {
    static let fairLowMotionInterval: TimeInterval = 0.030

    static func permits(capturedAt: TimeInterval,
                        after lastSubmittedCaptureTime: TimeInterval?,
                        minimumInterval: TimeInterval) -> Bool {
        guard capturedAt.isFinite,
              minimumInterval.isFinite,
              minimumInterval >= 0 else {
            return false
        }
        guard let lastSubmittedCaptureTime else { return true }
        let elapsed = capturedAt - lastSubmittedCaptureTime
        return elapsed.isFinite && elapsed > 0 && elapsed >= minimumInterval
    }

    static func shouldReplacePending(candidateTimestampInMilliseconds: Int,
                                     pendingTimestampInMilliseconds: Int?) -> Bool {
        pendingTimestampInMilliseconds.map {
            candidateTimestampInMilliseconds > $0
        } ?? true
    }
}

enum LipTemporalStability {
    // The existing freshness fade ends at 0.62s. Do not connect observations
    // across a gap after the old mask has disappeared.
    static let temporalGapResetInterval: TimeInterval = 0.65

    struct RecoveryRun {
        private(set) var count = 0
        private var firstTimestamp: TimeInterval?
        private var lastTimestamp: TimeInterval?

        mutating func observe(capturedAt: TimeInterval,
                              agreesWithPrevious: Bool,
                              isBounded: Bool) -> Bool {
            guard capturedAt.isFinite, isBounded else {
                self = RecoveryRun()
                return false
            }
            let continuous = lastTimestamp.map {
                capturedAt > $0 && capturedAt - $0 <= 0.22
            } ?? false
            let withinRun = firstTimestamp.map {
                capturedAt - $0 <= 0.55
            } ?? false
            if continuous && withinRun && agreesWithPrevious {
                count += 1
            } else {
                count = 1
                firstTimestamp = capturedAt
            }
            lastTimestamp = capturedAt
            return count >= 3 &&
                capturedAt - (firstTimestamp ?? capturedAt) >= 0.05 - 0.000_001
        }
    }

    static func recoveryShapesAgree(previous: [CGPoint], current: [CGPoint],
                                    mouthWidth: CGFloat,
                                    centerDistance: CGFloat,
                                    widthRatio: CGFloat) -> Bool {
        guard !previous.isEmpty, previous.count == current.count,
              mouthWidth.isFinite, mouthWidth > 1,
              centerDistance.isFinite, widthRatio.isFinite else { return false }
        let residuals = zip(previous, current).map {
            hypot($0.1.x - $0.0.x, $0.1.y - $0.0.y) / mouthWidth
        }.sorted()
        guard residuals.allSatisfy(\.isFinite) else { return false }
        return centerDistance <= 0.05 &&
            (0.90...1.10).contains(widthRatio) &&
            residuals[residuals.count / 2] <= 0.035 &&
            residuals[Int(CGFloat(residuals.count - 1) * 0.95)] <= 0.075
    }

    static func isTemporalGap(current: TimeInterval, previous: TimeInterval?) -> Bool {
        guard let previous else { return false }
        return !current.isFinite || !previous.isFinite ||
            current <= previous || current - previous > temporalGapResetInterval
    }

    struct ContourMotion {
        let correctionLimits: [CGFloat]
        let localVelocities: [CGVector]
        let trustRemaining: [TimeInterval]
        let coherent: [Bool]
        let confirmed: [Bool]
        let outliers: [Bool]
        let pointActivity: [CGFloat]
        let activity: CGFloat
        let medianResidual: CGFloat
        let p95Residual: CGFloat
        let coherentFraction: CGFloat
        let outlierCount: Int
    }

    // The 80 attention points form four matching 20-point rings. Positions
    // 1...9 are lower-lip points and 11...19 are upper-lip points. Neighbours
    // across the mouth opening must not vote against one another.
    static func assessContourMotion(
        previous: [CGPoint], current: [CGPoint],
        previousLocalVelocities: [CGVector]?,
        previousTrust: [TimeInterval]? = nil,
        previousCoherence: [Bool]? = nil,
        mouthWidth: CGFloat, mouthAngle: CGFloat,
        elapsed: TimeInterval,
        arOpeningDelta: CGFloat, arSmileDelta: CGFloat
    ) -> ContourMotion? {
        guard previous.count == 80, current.count == 80,
              mouthWidth.isFinite, mouthWidth > 1,
              mouthAngle.isFinite,
              elapsed.isFinite, elapsed > 0 else { return nil }
        let scale = min(max(mouthWidth / 80, 0.75), 2.5)
        let interval = max(elapsed, 1.0 / 120.0)
        let cosine = cos(mouthAngle)
        let sine = sin(mouthAngle)
        let movements = zip(previous, current).map {
            CGVector(dx: $0.1.x - $0.0.x, dy: $0.1.y - $0.0.y)
        }
        guard movements.allSatisfy({ $0.dx.isFinite && $0.dy.isFinite }) else {
            return nil
        }
        let magnitudes = movements.map { hypot($0.dx, $0.dy) }
        let sorted = magnitudes.sorted()
        let center = CGPoint(x: (previous[0].x + previous[10].x) * 0.5,
                             y: (previous[0].y + previous[10].y) * 0.5)
        var correctionLimits = [CGFloat](repeating: .greatestFiniteMagnitude, count: 80)
        var localVelocities = [CGVector](repeating: .zero, count: 80)
        var trustRemaining = [TimeInterval](repeating: 0, count: 80)
        var coherent = [Bool](repeating: false, count: 80)
        var confirmed = [Bool](repeating: false, count: 80)
        var outliers = [Bool](repeating: false, count: 80)
        var pointActivity = [CGFloat](repeating: 0, count: 80)
        var trustedMagnitudes: [CGFloat] = []

        for index in 0..<80 {
            let ring = index / 20
            let position = index % 20
            var neighbours: [Int] = []
            if position == 0 {
                neighbours.append(contentsOf: [ring * 20 + 11, ring * 20 + 1])
            } else if position == 10 {
                neighbours.append(contentsOf: [ring * 20 + 9, ring * 20 + 19])
            } else {
                let previousPosition = position == 11 ? 0 : position - 1
                let nextPosition = position == 19 ? 10 : position + 1
                neighbours.append(ring * 20 + previousPosition)
                neighbours.append(ring * 20 + nextPosition)
            }
            for otherRing in 0..<4 where otherRing != ring {
                neighbours.append(otherRing * 20 + position)
            }

            let movement = movements[index]
            let magnitude = magnitudes[index]
            let localDX = movement.dx * cosine + movement.dy * sine
            let localDY = -movement.dx * sine + movement.dy * cosine
            let local = CGVector(
                dx: localDX / mouthWidth / CGFloat(interval),
                dy: localDY / mouthWidth / CGFloat(interval)
            )
            localVelocities[index] = local
            let matchingNeighbours = neighbours.filter {
                motionsAgree(movement, movements[$0], noise: 0.30 * scale)
            }.count
            let hasNeighbourSupport = matchingNeighbours >= 3 &&
                magnitude > 0.30 * scale
            let wasConfirmed: Bool
            if let previousLocalVelocities,
               previousLocalVelocities.count == 80 {
                wasConfirmed = motionsAgree(
                    local, previousLocalVelocities[index], noise: 0.004
                )
            } else {
                wasConfirmed = false
            }
            let openingSupported: Bool
            if position == 0 || position == 10 {
                openingSupported = false
            } else {
                let lipDirection: CGFloat = position < 10 ? 1 : -1
                openingSupported = abs(arOpeningDelta) >= 0.012 &&
                    local.dy * lipDirection * arOpeningDelta > 0 &&
                    abs(localDY) > 0.20 * scale
            }
            let side = (previous[index].x - center.x) * cosine +
                (previous[index].y - center.y) * sine
            let smileSupported = abs(arSmileDelta) >= 0.035 &&
                abs(side) > mouthWidth * 0.18 &&
                local.dx * side * arSmileDelta > 0 &&
                abs(localDX) > 0.20 * scale
            let arSupported = (openingSupported || smileSupported) &&
                matchingNeighbours >= 2
            coherent[index] = hasNeighbourSupport || arSupported
            let settledAfterCoherentMotion = previousCoherence?.count == 80 &&
                previousCoherence?[index] == true && magnitude <= 0.45 * scale &&
                neighbours.filter { magnitudes[$0] <= 0.65 * scale }.count >= 3
            let observedConfirmation = (hasNeighbourSupport && wasConfirmed) ||
                arSupported || settledAfterCoherentMotion
            let priorTrust = previousTrust?.count == 80 ? previousTrust?[index] ?? 0 : 0
            // Finish catching up when a confirmed expression stops. Without
            // this bounded hold, the last part of a smile falls back to idle.
            trustRemaining[index] = observedConfirmation ? 0.08 :
                max(priorTrust - elapsed, 0)

            let neighbourX = neighbours.map { movements[$0].dx }.sorted()
            let neighbourY = neighbours.map { movements[$0].dy }.sorted()
            let reference = CGVector(
                dx: neighbourX[neighbourX.count / 2],
                dy: neighbourY[neighbourY.count / 2]
            )
            let difference = CGVector(
                dx: movement.dx - reference.dx,
                dy: movement.dy - reference.dy
            )
            let differenceMagnitude = hypot(difference.dx, difference.dy)
            let neighbourMagnitude = hypot(reference.dx, reference.dy)
            let isOutlier = !hasNeighbourSupport && !arSupported &&
                magnitude > 1.4 * scale &&
                differenceMagnitude > max(1.4 * scale,
                                          neighbourMagnitude * 0.85)
            outliers[index] = isOutlier
            confirmed[index] = !isOutlier && trustRemaining[index] > 0
            if isOutlier {
                correctionLimits[index] = 1.4 * scale
            }
            let correctedMagnitude = min(magnitude, correctionLimits[index])
            pointActivity[index] = activity(
                displacement: correctedMagnitude,
                noiseFloor: 0.25 * scale,
                fullActivityAtNominal: 1.65 * scale,
                elapsed: interval
            )
            if coherent[index] {
                trustedMagnitudes.append(correctedMagnitude)
            }
        }

        trustedMagnitudes.sort()
        let coherentFraction = CGFloat(coherent.filter { $0 }.count) / 80
        let activityMagnitude = trustedMagnitudes.isEmpty ? 0 :
            trustedMagnitudes[min(trustedMagnitudes.count - 1,
                                  trustedMagnitudes.count * 3 / 4)]
        return ContourMotion(
            correctionLimits: correctionLimits,
            localVelocities: localVelocities,
            trustRemaining: trustRemaining,
            coherent: coherent,
            confirmed: confirmed,
            outliers: outliers,
            pointActivity: pointActivity,
            // A moving lip section can activate the contour; a few agreeing
            // noisy points cannot remove the idle dead zone everywhere.
            activity: min(coherentFraction / 0.4, 1) * activity(
                displacement: activityMagnitude,
                noiseFloor: 0.25 * scale,
                fullActivityAtNominal: 1.65 * scale,
                elapsed: interval
            ),
            medianResidual: sorted[sorted.count / 2],
            p95Residual: sorted[Int(CGFloat(sorted.count - 1) * 0.95)],
            coherentFraction: coherentFraction,
            outlierCount: outliers.filter { $0 }.count
        )
    }

    private static func motionsAgree(_ first: CGVector, _ second: CGVector,
                                     noise: CGFloat) -> Bool {
        let firstLength = hypot(first.dx, first.dy)
        let secondLength = hypot(second.dx, second.dy)
        guard firstLength > noise, secondLength > noise else { return false }
        let ratio = firstLength / secondLength
        return ratio >= 0.5 && ratio <= 2 &&
            (first.dx * second.dx + first.dy * second.dy) /
                (firstLength * secondLength) >= 0.55
    }

    static func pointResponses(motion: ContourMotion?,
                               stability: CGFloat,
                               fastShapeChange: Bool,
                               elapsed: TimeInterval) -> [CGFloat] {
        (0..<80).map { index in
            let localActivity = motion?.pointActivity[index] ?? 0
            let nominalAlpha: CGFloat
            if motion?.outliers[index] == true {
                nominalAlpha = 0.14
            } else if motion?.confirmed[index] == true {
                nominalAlpha = 0.82 + localActivity * 0.14
            } else if motion?.coherent[index] == true {
                nominalAlpha = 0.18 + localActivity *
                    (fastShapeChange ? 0.52 : 0.42)
            } else {
                nominalAlpha = 0.18 + localActivity * 0.12 +
                    (1 - stability) * 0.04
            }
            return response(nominalAlpha: min(nominalAlpha, 0.96),
                            elapsed: elapsed)
        }
    }

    static func updatedStability(previous: CGFloat, activity: CGFloat,
                                 elapsed: TimeInterval) -> CGFloat {
        let target = 1 - min(max(activity, 0), 1)
        let gain = response(nominalAlpha: target < previous ? 0.82 : 0.30,
                            elapsed: elapsed)
        return min(max(previous + (target - previous) * gain, 0), 1)
    }

    static func filteredPoint(previous: CGPoint, current: CGPoint,
                              alpha: CGFloat, deadZone: CGFloat,
                              maximumCorrection: CGFloat = .greatestFiniteMagnitude) -> CGPoint {
        let dx = current.x - previous.x
        let dy = current.y - previous.y
        let distance = hypot(dx, dy)
        guard distance.isFinite, distance > deadZone else { return previous }
        let factor = min(distance - deadZone, maximumCorrection) / distance * alpha
        return CGPoint(x: previous.x + dx * factor,
                       y: previous.y + dy * factor)
    }

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
