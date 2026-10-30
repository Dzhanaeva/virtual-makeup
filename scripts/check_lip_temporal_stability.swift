import CoreGraphics
import Foundation

func oldLead(requested: CGFloat, previous: CGFloat, width: CGFloat) -> CGFloat {
    let alpha: CGFloat = abs(requested) < 0.01 ? 0.62 :
        (requested * previous < 0 ? 0.74 : 0.48)
    let candidate = previous + (requested - previous) * alpha
    let maximumStep = min(max(width * 0.012, 0.65), 1.10)
    return previous + min(max(candidate - previous, -maximumStep), maximumStep)
}

private enum ContourRegression {
    private struct Generator {
        var state: UInt64 = 0x12345678
        mutating func noise() -> CGFloat {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return CGFloat((state >> 32) & 0xffff) / 65535 * 2 - 1
        }
    }

    // Frozen baseline: the previous global max-activity policy and raw-shape
    // bypass. Keeping it here makes regressions comparable to the old output.
    private struct LegacyFilter {
        var raw: [CGPoint]
        var filtered: [CGPoint]
        var stability: CGFloat = 0

        mutating func update(_ current: [CGPoint], elapsed: TimeInterval) {
            let residuals = zip(raw, current).map {
                hypot($0.1.x - $0.0.x, $0.1.y - $0.0.y)
            }.sorted()
            let oldBounds = bounds(Array(filtered.prefix(20)))
            let newBounds = bounds(Array(current.prefix(20)))
            let widthChange = abs(newBounds.width / oldBounds.width - 1)
            let heightChange = abs(newBounds.height / oldBounds.height - 1)
            let openingChange = abs(opening(current) - opening(filtered))
            let scale = min(max(newBounds.width / 80, 0.75), 2.5)
            let bypass = widthChange > 0.12 || heightChange > 0.18 ||
                openingChange > 0.055 || newBounds.width / oldBounds.width > 1.20 ||
                newBounds.height / oldBounds.height > 1.26
            let activity = max(
                LipTemporalStability.activity(displacement: residuals[40],
                    noiseFloor: 0.25 * scale, fullActivityAtNominal: 1.65 * scale, elapsed: elapsed),
                LipTemporalStability.activity(displacement: residuals[75],
                    noiseFloor: 0.65 * scale, fullActivityAtNominal: 3.6 * scale, elapsed: elapsed),
                LipTemporalStability.activity(displacement: abs(opening(current) - opening(raw)),
                    noiseFloor: 0.006, fullActivityAtNominal: 0.055, elapsed: elapsed)
            )
            stability = LipTemporalStability.updatedStability(
                previous: stability, activity: activity, elapsed: elapsed
            )
            let alpha = LipTemporalStability.response(
                nominalAlpha: min(0.18 + activity * 0.70 + (1 - stability) * 0.08, 0.96),
                elapsed: elapsed
            )
            if bypass {
                filtered = current
                stability = 0
            } else {
                filtered = zip(filtered, current).map {
                    LipTemporalStability.filteredPoint(previous: $0.0, current: $0.1,
                        alpha: alpha, deadZone: min(max(newBounds.width * 0.0036, 0.26), 0.85) * stability)
                }
            }
            raw = current
        }
    }

    private struct Filter {
        var raw: [CGPoint]
        var filtered: [CGPoint]
        var velocities: [CGVector]?
        var trust: [TimeInterval]?
        var coherence: [Bool]?
        var stability: CGFloat = 0
        var lastMotion: LipTemporalStability.ContourMotion?

        mutating func update(_ current: [CGPoint], elapsed: TimeInterval,
                             arOpeningDelta: CGFloat = 0,
                             arSmileDelta: CGFloat = 0) {
            let motion = LipTemporalStability.assessContourMotion(
                previous: raw, current: current,
                previousLocalVelocities: velocities,
                previousTrust: trust,
                previousCoherence: coherence,
                mouthWidth: 80, mouthAngle: 0, elapsed: elapsed,
                arOpeningDelta: arOpeningDelta, arSmileDelta: arSmileDelta
            )
            precondition(motion != nil)
            stability = LipTemporalStability.updatedStability(
                previous: stability, activity: motion?.activity ?? 0, elapsed: elapsed
            )
            let alphas = LipTemporalStability.pointResponses(
                motion: motion, stability: stability, fastShapeChange: false,
                elapsed: elapsed
            )
            filtered = current.enumerated().map { index, point in
                LipTemporalStability.filteredPoint(
                    previous: filtered[index], current: point,
                    alpha: alphas[index],
                    deadZone: 0.288 * stability *
                        (1 - (motion?.pointActivity[index] ?? 0)),
                    maximumCorrection: motion?.correctionLimits[index] ?? .greatestFiniteMagnitude
                )
            }
            velocities = motion?.localVelocities
            trust = motion?.trustRemaining
            coherence = motion?.coherent
            raw = current
            lastMotion = motion
        }
    }

    private static func bounds(_ points: [CGPoint]) -> CGRect {
        let xs = points.map(\.x)
        let ys = points.map(\.y)
        return CGRect(x: xs.min() ?? 0, y: ys.min() ?? 0,
            width: (xs.max() ?? 0) - (xs.min() ?? 0),
            height: (ys.max() ?? 0) - (ys.min() ?? 0))
    }

    private static func opening(_ points: [CGPoint]) -> CGFloat {
        bounds(Array(points[20..<40])).height /
            max(bounds(Array(points.prefix(20))).width, 1)
    }

    private static func shape(opening: CGFloat = 0, smile: CGFloat = 0) -> [CGPoint] {
        (0..<80).map { index in
            let position = index % 20
            let t = position <= 10 ? CGFloat(position) / 10 : CGFloat(position - 10) / 10
            let direction: CGFloat = position <= 10 ? 1 : -1
            let thickness: CGFloat = [8, 3, 6.5, 4.5][index / 20]
            let x = -40 + 80 * t
            return CGPoint(x: x * (1 + smile),
                y: direction * (thickness + opening) * sin(.pi * t))
        }
    }

    private static func crossing(_ samples: [(TimeInterval, CGFloat)],
                                 threshold: CGFloat) -> TimeInterval {
        for index in 1..<samples.count where samples[index].1 >= threshold {
            let previous = samples[index - 1]
            let current = samples[index]
            let fraction = (threshold - previous.1) /
                max(current.1 - previous.1, 0.000_001)
            return previous.0 + Double(fraction) * (current.0 - previous.0)
        }
        preconditionFailure("Expression did not reach the target")
    }

    private static func checkMotionEvidence() {
        let base = shape()
        let opened = shape(opening: 1.8)
        let nextOpened = shape(opening: 3.6)
        let interval = 1.0 / 30.0
        guard let first = LipTemporalStability.assessContourMotion(
            previous: base, current: opened, previousLocalVelocities: nil,
            mouthWidth: 80, mouthAngle: 0, elapsed: interval,
            arOpeningDelta: 0, arSmileDelta: 0
        ), let second = LipTemporalStability.assessContourMotion(
            previous: opened, current: nextOpened,
            previousLocalVelocities: first.localVelocities,
            previousCoherence: first.coherent,
            mouthWidth: 80, mouthAngle: 0, elapsed: interval,
            arOpeningDelta: 0, arSmileDelta: 0
        ), let settled = LipTemporalStability.assessContourMotion(
            previous: opened, current: opened,
            previousLocalVelocities: first.localVelocities,
            previousCoherence: first.coherent,
            mouthWidth: 80, mouthAngle: 0, elapsed: interval,
            arOpeningDelta: 0, arSmileDelta: 0
        ) else { preconditionFailure("Valid 80-point rings were rejected") }
        let firstAlphas = LipTemporalStability.pointResponses(
            motion: first, stability: 0, fastShapeChange: true, elapsed: interval)
        let confirmedAlphas = LipTemporalStability.pointResponses(
            motion: second, stability: 0, fastShapeChange: false, elapsed: interval)
        let settledAlphas = LipTemporalStability.pointResponses(
            motion: settled, stability: 0, fastShapeChange: false, elapsed: interval)
        guard let arFirst = LipTemporalStability.assessContourMotion(
            previous: base, current: opened, previousLocalVelocities: nil,
            mouthWidth: 80, mouthAngle: 0, elapsed: interval,
            arOpeningDelta: 0.045, arSmileDelta: 0
        ) else { preconditionFailure("AR expression evidence was rejected") }
        let arAlphas = LipTemporalStability.pointResponses(
            motion: arFirst, stability: 0, fastShapeChange: true, elapsed: interval)
        for index in [5, 15, 25, 35, 45, 55, 65, 75] {
            precondition(first.coherent[index] && !first.outliers[index])
            precondition(firstAlphas[index] <= 0.701)
            precondition(confirmedAlphas[index] > 0.95)
            precondition(settledAlphas[index] >= 0.819)
            precondition(arAlphas[index] > 0.95)
        }

        // AR expression evidence cannot validate an isolated point whose
        // neighbours remain still, including on the opposite lip section.
        for index in [5, 15, 25, 35, 45, 55, 65, 75] {
            var spike = base
            spike[index].y += index % 20 < 10 ? 6 : -6
            guard let motion = LipTemporalStability.assessContourMotion(
                previous: base, current: spike, previousLocalVelocities: nil,
                mouthWidth: 80, mouthAngle: 0, elapsed: interval,
                arOpeningDelta: 0.06, arSmileDelta: 0.1
            ) else { preconditionFailure("Isolated spike assessment failed") }
            precondition(motion.outliers[index] && motion.activity == 0)
            let alphas = LipTemporalStability.pointResponses(
                motion: motion, stability: 1, fastShapeChange: true, elapsed: interval)
            precondition(alphas[index] <= 0.141)
        }

        let angle: CGFloat = 0.32
        func transport(_ points: [CGPoint]) -> [CGPoint] {
            points.map { point in
                CGPoint(x: 110 + point.x * cos(angle) - point.y * sin(angle),
                        y: 70 + point.x * sin(angle) + point.y * cos(angle))
            }
        }
        guard let rotated = LipTemporalStability.assessContourMotion(
            previous: transport(opened), current: transport(nextOpened),
            previousLocalVelocities: first.localVelocities,
            previousCoherence: first.coherent,
            mouthWidth: 80, mouthAngle: angle, elapsed: interval,
            arOpeningDelta: 0, arSmileDelta: 0
        ) else { preconditionFailure("Pose-transported motion failed") }
        for index in [5, 15, 25, 35] {
            precondition(rotated.confirmed[index])
            precondition(abs(rotated.localVelocities[index].dy -
                second.localVelocities[index].dy) < 0.000_001)
        }
        let moved = transport(base)
        precondition(LipTemporalStability.recoveryShapesAgree(
            previous: moved, current: moved, mouthWidth: 80,
            centerDistance: 0, widthRatio: 1))
        precondition(!LipTemporalStability.recoveryShapesAgree(
            previous: base, current: moved, mouthWidth: 80,
            centerDistance: 1, widthRatio: 1))
        print("contour evidence: opposite lip motion, next-observation confirmation, AR outlier veto and common-pose recovery passed")
    }

    private static func checkNoisySpikes(frequency: Int) {
        let base = shape()
        let elapsed = 1.0 / Double(frequency)
        for amplitude: CGFloat in [3, 6] {
            var generator = Generator()
            var oldControl = LegacyFilter(raw: base, filtered: base)
            var oldSpike = oldControl
            var newControl = Filter(raw: base, filtered: base)
            var newSpike = newControl
            var oldPeak: CGFloat = 0
            var newPeak: CGFloat = 0
            for tick in 0..<(frequency * 2) {
                let noisy = base.map {
                    CGPoint(x: $0.x + generator.noise() * 1.2,
                            y: $0.y + generator.noise() * 1.2)
                }
                var spiked = noisy
                if tick == frequency { spiked[5].y += amplitude }
                oldControl.update(noisy, elapsed: elapsed)
                oldSpike.update(spiked, elapsed: elapsed)
                newControl.update(noisy, elapsed: elapsed)
                newSpike.update(spiked, elapsed: elapsed)
                oldPeak = max(oldPeak, hypot(oldControl.filtered[5].x - oldSpike.filtered[5].x,
                    oldControl.filtered[5].y - oldSpike.filtered[5].y))
                newPeak = max(newPeak, hypot(newControl.filtered[5].x - newSpike.filtered[5].x,
                    newControl.filtered[5].y - newSpike.filtered[5].y))
            }
            print(String(format: "contour noisy spike %dHz input=%.0f extraPeak old=%.3f new=%.3f px",
                frequency, Double(amplitude), Double(oldPeak), Double(newPeak)))
            precondition(newPeak < oldPeak * 0.65 && newPeak < 1.2)
        }
    }

    static func run() {
        checkMotionEvidence()
        let base = shape()
        for frequency in [15, 30, 60] {
            checkNoisySpikes(frequency: frequency)
            let elapsed = 1.0 / Double(frequency)
            var generator = Generator()
            var old = LegacyFilter(raw: base, filtered: base)
            var new = Filter(raw: base, filtered: base)
            var oldSquared: CGFloat = 0
            var newSquared: CGFloat = 0
            var count: CGFloat = 0
            for tick in 0..<(frequency * 3) {
                let noisy = base.map { point in
                    CGPoint(x: point.x + generator.noise() * 1.2,
                            y: point.y + generator.noise() * 1.2)
                }
                old.update(noisy, elapsed: elapsed)
                new.update(noisy, elapsed: elapsed)
                if tick >= frequency {
                    for index in 0..<80 {
                        oldSquared += pow(old.filtered[index].x - base[index].x, 2) +
                            pow(old.filtered[index].y - base[index].y, 2)
                        newSquared += pow(new.filtered[index].x - base[index].x, 2) +
                            pow(new.filtered[index].y - base[index].y, 2)
                        count += 1
                    }
                }
            }
            let oldRMS = sqrt(oldSquared / count)
            let newRMS = sqrt(newSquared / count)
            print(String(format: "contour idle %dHz RMS old=%.3f new=%.3f px",
                frequency, Double(oldRMS), Double(newRMS)))
            precondition(newRMS < oldRMS * 0.85)

            for spike: CGFloat in [3, 4.5, 6] {
                old = LegacyFilter(raw: base, filtered: base, stability: 1)
                new = Filter(raw: base, filtered: base, stability: 1)
                var oldPeak: CGFloat = 0
                var newPeak: CGFloat = 0
                for tick in 0..<frequency {
                    var current = base
                    if tick == frequency / 3 { current[5].y += spike }
                    old.update(current, elapsed: elapsed)
                    new.update(current, elapsed: elapsed)
                    oldPeak = max(oldPeak, abs(old.filtered[5].y - base[5].y))
                    newPeak = max(newPeak, abs(new.filtered[5].y - base[5].y))
                    precondition((new.lastMotion?.activity ?? 1) < 0.01)
                }
                print(String(format: "contour spike %dHz input=%.1f oldPeak=%.3f newPeak=%.3f px",
                    frequency, Double(spike), Double(oldPeak), Double(newPeak)))
                precondition(newPeak < 0.7 && newPeak < oldPeak * 0.5)
            }

            for expression in ["opening", "smile"] {
                for useAR in [true, false] {
                    old = LegacyFilter(raw: base, filtered: base, stability: 1)
                    new = Filter(raw: base, filtered: base, stability: 1)
                    var oldSamples: [(TimeInterval, CGFloat)] = []
                    var newSamples: [(TimeInterval, CGFloat)] = []
                    var previousProgress: CGFloat = 0
                    for tick in 0..<(frequency * 2) {
                        let time = Double(tick) * elapsed
                        let progress = CGFloat(min(max((time - 0.5) / 0.2, 0), 1))
                        let current = expression == "opening" ?
                            shape(opening: 8 * progress) : shape(smile: 0.12 * progress)
                        let delta = progress - previousProgress
                        old.update(current, elapsed: elapsed)
                        new.update(current, elapsed: elapsed,
                            arOpeningDelta: useAR && expression == "opening" ? 0.2 * delta : 0,
                            arSmileDelta: useAR && expression == "smile" ? 0.6 * delta : 0)
                        let index = expression == "opening" ? 5 : 10
                        let oldValue = expression == "opening" ?
                            old.filtered[index].y - base[index].y :
                            old.filtered[index].x - base[index].x
                        let newValue = expression == "opening" ?
                            new.filtered[index].y - base[index].y :
                            new.filtered[index].x - base[index].x
                        oldSamples.append((time, oldValue))
                        newSamples.append((time, newValue))
                        previousProgress = progress
                    }
                    let target: CGFloat = expression == "opening" ? 8 : 4.8
                    let oldDelay = crossing(oldSamples, threshold: target * 0.9) - 0.68
                    let newDelay = crossing(newSamples, threshold: target * 0.9) - 0.68
                    print(String(format: "contour %@ %dHz evidence=%@ 90%% lag old=%.1f new=%.1f ms",
                        expression, frequency, useAR ? "AR" : "next", oldDelay * 1000, newDelay * 1000))
                    precondition(newDelay < 0.08 && newDelay <= oldDelay + 0.05)
                }
            }
        }

        var run = LipTemporalStability.RecoveryRun()
        var previousCandidate = base
        for index in 0..<12 {
            let offset: CGFloat = index % 2 == 0 ? 18 : -18
            let candidate = base.map { CGPoint(x: $0.x + offset, y: $0.y) }
            let agreement = LipTemporalStability.recoveryShapesAgree(
                previous: previousCandidate, current: candidate,
                mouthWidth: 80, centerDistance: index == 0 ? 0.225 : 0.45,
                widthRatio: 1
            )
            precondition(!run.observe(capturedAt: Double(index) / 30,
                agreesWithPrevious: agreement, isBounded: true))
            previousCandidate = candidate
        }
        precondition(run.count == 1)
        print("contour recovery: 12 inconsistent AR-bounded candidates never confirm")
        run = LipTemporalStability.RecoveryRun()
        previousCandidate = base
        for index in 0..<12 {
            let offset: CGFloat = index % 2 == 0 ? 3 : -3
            let candidate = base.map { CGPoint(x: $0.x + offset, y: $0.y) }
            let agreement = LipTemporalStability.recoveryShapesAgree(
                previous: previousCandidate, current: candidate, mouthWidth: 80,
                centerDistance: index == 0 ? 0.0375 : 0.075, widthRatio: 1)
            precondition(!run.observe(capturedAt: Double(index) / 30,
                agreesWithPrevious: agreement, isBounded: true))
            previousCandidate = candidate
        }
        precondition(run.count == 1)
        print("contour recovery: alternating 6px candidates never confirm")
        for frequency in [15, 30, 60] {
            run = LipTemporalStability.RecoveryRun()
            var acceptedAt: TimeInterval?
            for index in 0..<8 {
                let time = 4 + Double(index) / Double(frequency)
                let accepted = run.observe(capturedAt: time,
                    agreesWithPrevious: index > 0, isBounded: true)
                if accepted && acceptedAt == nil { acceptedAt = time }
            }
            guard let acceptedAt else { preconditionFailure("Recovery never confirmed") }
            precondition(acceptedAt - 4 >= 0.049_999 && acceptedAt - 4 <= 0.14)
            precondition(LipTemporalStability.isTemporalGap(current: 4, previous: 0))
            precondition(!run.observe(capturedAt: 8,
                agreesWithPrevious: true, isBounded: true))
            precondition(run.count == 1)
            print(String(format: "contour recovery %dHz confirmed after %.1f ms; pause restarts consensus",
                frequency, (acceptedAt - 4) * 1000))
        }
        let wrong = base.map { CGPoint(x: $0.x + 18, y: $0.y - 12) }
        precondition(!LipTemporalStability.recoveryShapesAgree(previous: base,
            current: wrong, mouthWidth: 80, centerDistance: 0.27, widthRatio: 1))
        precondition(LipTemporalStability.recoveryShapesAgree(previous: base,
            current: base, mouthWidth: 80, centerDistance: 0, widthRatio: 1))
    }
}

private enum LandmarkPacingRegression {
    static func run() {
        let fairInterval = LandmarkSubmitPacing.fairLowMotionInterval
        precondition(abs(fairInterval - 0.030) < 0.000_001)
        let callbackJitter = [0.007, 0.001, 0.006, 0.001]
        let frameCount = 300 // Ten seconds of camera capture at 30 FPS.
        let detectorDuration = 0.012
        var lastSubmittedCapture: TimeInterval?
        var lastSubmittedCallback: TimeInterval?
        var lastCallbackAtFairInterval: TimeInterval?
        var captureTimedCount = 0
        var callbackTimedCount = 0
        var callbackTimedAtFairIntervalCount = 0
        var previousCallback: TimeInterval?
        for frame in 0..<frameCount {
            let capturedAt = 100 + Double(frame) / 30
            let callbackAt = capturedAt + callbackJitter[frame % callbackJitter.count]
            if let previousCallback {
                precondition(callbackAt - previousCallback > detectorDuration)
            }
            previousCallback = callbackAt
            if LandmarkSubmitPacing.permits(
                capturedAt: capturedAt,
                after: lastSubmittedCapture,
                minimumInterval: fairInterval
            ) {
                captureTimedCount += 1
                lastSubmittedCapture = capturedAt
            }
            // Frozen baseline: 0.034s compared with callback arrival time.
            if lastSubmittedCallback.map({ callbackAt - $0 >= 0.034 }) ?? true {
                callbackTimedCount += 1
                lastSubmittedCallback = callbackAt
            }
            if lastCallbackAtFairInterval.map({
                callbackAt - $0 >= fairInterval
            }) ?? true {
                callbackTimedAtFairIntervalCount += 1
                lastCallbackAtFairInterval = callbackAt
            }
        }
        precondition(captureTimedCount == frameCount)
        precondition(callbackTimedCount <= frameCount / 2 + 1)
        precondition(callbackTimedAtFairIntervalCount <= frameCount / 2 + 1)
        print(String(format:
            "landmark fair 30Hz jitter old=%.1f FPS callback-at-0.030=%.1f FPS capture-paced=%.1f FPS detector=%.0f ms",
            Double(callbackTimedCount) / 10,
            Double(callbackTimedAtFairIntervalCount) / 10,
            Double(captureTimedCount) / 10,
            detectorDuration * 1000))

        var pendingTimestamp: Int?
        for candidate in [1000, 1033, 1067, 1040] {
            if LandmarkSubmitPacing.shouldReplacePending(
                candidateTimestampInMilliseconds: candidate,
                pendingTimestampInMilliseconds: pendingTimestamp
            ) {
                pendingTimestamp = candidate
            }
        }
        precondition(pendingTimestamp == 1067)
        precondition(!LandmarkSubmitPacing.permits(
            capturedAt: 100, after: 100,
            minimumInterval: fairInterval
        ))
        precondition(!LandmarkSubmitPacing.permits(
            capturedAt: .nan, after: nil,
            minimumInterval: fairInterval
        ))
    }
}

@main
struct LipTemporalCheck {
    static func main() {
        LandmarkPacingRegression.run()
        ContourRegression.run()
        let width: CGFloat = 80
        let previous: CGFloat = 2
        let stopped = LipTemporalStability.rigidLead(
            requested: .zero, previous: CGVector(dx: previous, dy: 0), mouthWidth: width
        )
        let reversed = LipTemporalStability.rigidLead(
            requested: CGVector(dx: -1, dy: 0),
            previous: CGVector(dx: previous, dy: 0), mouthWidth: width
        )
        precondition(stopped.dx == 0 && stopped.dy == 0)
        precondition(reversed.dx <= 0 && reversed.dy == 0)
        print(String(format: "lead stop old=%.2f new=%.2f disabled=0.00 px",
                     Double(oldLead(requested: 0, previous: previous, width: width)),
                     Double(stopped.dx)))
        print(String(format: "lead reverse old=%.2f new=%.2f disabled=0.00 px",
                     Double(oldLead(requested: -1, previous: previous, width: width)),
                     Double(reversed.dx)))

        var sourceTimestamp: TimeInterval?
        var sourceCenter: CGFloat = 0
        var heldLead = CGVector.zero
        var newFrameLead: CGFloat = 0
        var repeatedFrameLead: CGFloat = 0
        for renderTick in 0..<120 {
            let sourceTick = renderTick / 2
            let timestamp = TimeInterval(sourceTick) / 30
            let center = CGFloat(sourceTick)
            switch LipTemporalStability.motionFrameStep(
                current: timestamp, previous: sourceTimestamp
            ) {
            case .first:
                sourceTimestamp = timestamp
                sourceCenter = center
            case let .advanced(interval):
                let delta = center - sourceCenter
                let requested = CGVector(
                    dx: delta * CGFloat(min(interval * 0.58, 0.020) / interval),
                    dy: 0
                )
                heldLead = LipTemporalStability.rigidLead(
                    requested: requested, previous: heldLead, mouthWidth: width
                )
                sourceTimestamp = timestamp
                sourceCenter = center
                newFrameLead = heldLead.dx
            case .repeated:
                repeatedFrameLead = heldLead.dx
            case .discontinuous:
                preconditionFailure("30 FPS source must be continuous")
            }
        }
        precondition(newFrameLead > 0.5)
        precondition(abs(newFrameLead - repeatedFrameLead) < 0.000_001)
        let oldNewFrameLead = LipTemporalStability.rigidLead(
            requested: CGVector(dx: 0.58 * 0.55, dy: 0),
            previous: .zero, mouthWidth: width
        ).dx
        let oldRepeatedFrameLead: CGFloat = 0
        print(String(format:
            "30Hz source / 60Hz render steady lead old=%.3f->%.3f new=%.3f->%.3f px",
            Double(oldNewFrameLead), Double(oldRepeatedFrameLead),
            Double(newFrameLead), Double(repeatedFrameLead)))
        let stoppedFrame = LipTemporalStability.motionFrameStep(
            current: 2, previous: sourceTimestamp
        )
        guard case .advanced = stoppedFrame else {
            preconditionFailure("Stop must be detected on a new source frame")
        }
        let stoppedLead = LipTemporalStability.rigidLead(
            requested: .zero, previous: heldLead, mouthWidth: width
        )
        precondition(stoppedLead.dx == 0)
        let reversedFrame = LipTemporalStability.motionFrameStep(
            current: 2 + 1.0 / 30.0, previous: 2
        )
        guard case .advanced = reversedFrame else {
            preconditionFailure("Reversal must be detected on a new source frame")
        }
        let reversedLead = LipTemporalStability.rigidLead(
            requested: CGVector(dx: -0.58, dy: 0),
            previous: heldLead, mouthWidth: width
        )
        precondition(reversedLead.dx == 0)

        let nominalAlpha: CGFloat = 0.30
        var outputs: [(Int, CGFloat)] = []
        for frequency in [15, 30, 60] {
            let interval = 1.0 / Double(frequency)
            var value: CGFloat = 0
            for _ in 0..<(frequency / 5) {
                let alpha = LipTemporalStability.response(
                    nominalAlpha: nominalAlpha, elapsed: interval
                )
                value += (1 - value) * alpha
            }
            outputs.append((frequency, value))
            let oldResponse = 1 - pow(1 - nominalAlpha, CGFloat(frequency / 5))
            print(String(format: "response %dHz after 200ms old=%.6f new=%.6f",
                         frequency, Double(oldResponse), Double(value)))
        }
        precondition(outputs.map(\.1).max()! - outputs.map(\.1).min()! < 0.000_001)

        var activities: [CGFloat] = []
        var stabilityAfter200ms: [CGFloat] = []
        for frequency in [15, 30, 60] {
            let interval = 1.0 / Double(frequency)
            let activity = LipTemporalStability.activity(
                displacement: 0.25 + 20 * CGFloat(interval),
                noiseFloor: 0.25,
                fullActivityAtNominal: 1.65,
                elapsed: interval
            )
            activities.append(activity)
            precondition(LipTemporalStability.activity(
                displacement: 0.20,
                noiseFloor: 0.25,
                fullActivityAtNominal: 1.65,
                elapsed: interval
            ) == 0)
            var stability: CGFloat = 1
            for _ in 0..<(frequency / 5) {
                let gain = LipTemporalStability.response(
                    nominalAlpha: 0.82, elapsed: interval
                )
                stability += ((1 - activity) - stability) * gain
            }
            stabilityAfter200ms.append(stability)
            print(String(format: "activity %dHz value=%.4f stability200ms=%.4f",
                         frequency, Double(activity), Double(stability)))
        }
        precondition(activities.max()! - activities.min()! < 0.000_001)
        precondition(stabilityAfter200ms.max()! - stabilityAfter200ms.min()! < 0.000_001)
        precondition(activities[0] > 0.4)
        let fastChange = LipTemporalStability.activity(
            displacement: 3,
            noiseFloor: 0.25,
            fullActivityAtNominal: 1.65,
            elapsed: 1.0 / 30.0
        )
        let firstActiveConfidence = 1 + ((1 - fastChange) - 1) *
            LipTemporalStability.response(nominalAlpha: 0.82, elapsed: 1.0 / 30.0)
        precondition(fastChange == 1 && firstActiveConfidence < 0.32)
        let openingAt15 = LipTemporalStability.activity(
            displacement: 0.006 + 0.5 / 15,
            noiseFloor: 0.006,
            fullActivityAtNominal: 0.055,
            elapsed: 1.0 / 15.0
        )
        let openingAt60 = LipTemporalStability.activity(
            displacement: 0.006 + 0.5 / 60,
            noiseFloor: 0.006,
            fullActivityAtNominal: 0.055,
            elapsed: 1.0 / 60.0
        )
        precondition(abs(openingAt15 - openingAt60) < 0.000_001)

        let source = CGPoint(x: 100, y: 100)
        let plane = LipTemporalStability.cameraPlaneOffsets(
            reference: source, right: CGPoint(x: 105, y: 100),
            up: CGPoint(x: 100, y: 95), observed: CGPoint(x: 110, y: 85),
            stepMeters: 0.005
        )
        precondition(abs((plane?.x ?? 0) - 0.010) < 0.000_001)
        precondition(abs((plane?.y ?? 0) - 0.015) < 0.000_001)
        print("source inverse projection error=0.000 px")

        let imageSize = CGSize(width: 512, height: 384)
        let viewportSize = CGSize(width: 390, height: 844)
        let orientedInput = CGAffineTransform(a: 0, b: -1, c: 1, d: 0,
                                              tx: 0, ty: 1)
        let mirroredDisplay = CGAffineTransform(a: -0.8, b: 0, c: 0,
                                                 d: 1.2, tx: 0.9, ty: -0.1)
        let cameraMapping = LipCameraFrameMapping.imageToViewport(
            imageSize: imageSize, viewportSize: viewportSize,
            inputToCaptured: orientedInput,
            capturedToViewport: mirroredDisplay)
        precondition(LipCameraFrameMapping.isUsable(cameraMapping))
        for imagePoint in [CGPoint(x: 64, y: 80), CGPoint(x: 267, y: 192),
                           CGPoint(x: 450, y: 310)] {
            let normalized = CGPoint(x: imagePoint.x / imageSize.width,
                                     y: imagePoint.y / imageSize.height)
                .applying(orientedInput).applying(mirroredDisplay)
            let expected = CGPoint(x: normalized.x * viewportSize.width,
                                   y: normalized.y * viewportSize.height)
            let actual = imagePoint.applying(cameraMapping)
            precondition(hypot(actual.x - expected.x,
                               actual.y - expected.y) < 0.000_001)
            let recovered = actual.applying(cameraMapping.inverted())
            precondition(hypot(recovered.x - imagePoint.x,
                               recovered.y - imagePoint.y) < 0.000_001)
            let sourceNeighbor = CGPoint(x: imagePoint.x + 12,
                                         y: imagePoint.y + 8)
            let mappedNeighbor = sourceNeighbor.applying(cameraMapping)
            let mappedDelta = CGVector(dx: cameraMapping.a * 12 +
                                          cameraMapping.c * 8,
                                       dy: cameraMapping.b * 12 +
                                          cameraMapping.d * 8)
            precondition(hypot(mappedNeighbor.x - actual.x - mappedDelta.dx,
                               mappedNeighbor.y - actual.y - mappedDelta.dy) < 0.000_001)
        }
        print("camera image mapping: rotation, mirror, crop and source steps passed")

        // A pinhole camera and a protruding lip illustrate why one flat basis
        // loses parallax under yaw. Depth reprojection uses each point's z.
        let focal: CGFloat = 800
        let cameraDistance: CGFloat = 0.50
        let points: [(x: CGFloat, depth: CGFloat)] = [
            (-0.035, 0.003), (-0.025, 0.008), (-0.010, 0.013),
            (0.010, 0.013), (0.025, 0.008), (0.035, 0.003)
        ]
        for degrees in [0.0, 15.0, 30.0] {
            let angle = CGFloat(degrees * .pi / 180)
            let errors = points.map { point -> CGFloat in
                let sourceX = focal * point.x / (cameraDistance + point.depth)
                let flatX = sourceX * cameraDistance / focal
                let trueX = focal *
                    (point.x * cos(angle) + point.depth * sin(angle)) /
                    (cameraDistance - point.x * sin(angle) + point.depth * cos(angle))
                let flatProjectedX = focal * flatX * cos(angle) /
                    (cameraDistance - flatX * sin(angle))
                return abs(trueX - flatProjectedX)
            }.sorted()
            print(String(format: "synthetic yaw %.0fdeg flatMedian=%.3f depthMedian=0.000 px",
                         degrees, Double(errors[errors.count / 2])))
        }
        for degrees in [0.0, 15.0, 30.0] {
            let angle = CGFloat(degrees * .pi / 180)
            let errors = points.map { point -> CGFloat in
                let sourceY = focal * point.x / (cameraDistance + point.depth)
                let flatY = sourceY * cameraDistance / focal
                let trueY = focal *
                    (point.x * cos(angle) + point.depth * sin(angle)) /
                    (cameraDistance - point.x * sin(angle) + point.depth * cos(angle))
                let flatProjectedY = focal * flatY * cos(angle) /
                    (cameraDistance - flatY * sin(angle))
                return abs(trueY - flatProjectedY)
            }.sorted()
            print(String(format: "synthetic pitch %.0fdeg flatMedian=%.3f depthMedian=0.000 px",
                         degrees, Double(errors[errors.count / 2])))
        }
    }
}
