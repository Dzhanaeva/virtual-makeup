import CoreGraphics
import Foundation

func oldLead(requested: CGFloat, previous: CGFloat, width: CGFloat) -> CGFloat {
    let alpha: CGFloat = abs(requested) < 0.01 ? 0.62 :
        (requested * previous < 0 ? 0.74 : 0.48)
    let candidate = previous + (requested - previous) * alpha
    let maximumStep = min(max(width * 0.012, 0.65), 1.10)
    return previous + min(max(candidate - previous, -maximumStep), maximumStep)
}

@main
struct LipTemporalCheck {
    static func main() {
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
