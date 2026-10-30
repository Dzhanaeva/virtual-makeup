import Foundation
import QuartzCore
import UIKit
import os

enum LipDebugLog {
    private static let lock = NSLock()
    private static var lastLoggedAt: [String: CFTimeInterval] = [:]

    static func throttled(_ key: String,
                          interval: CFTimeInterval = 0.5,
                          _ message: @autoclosure () -> String) {

        let now = CACurrentMediaTime()
        lock.lock()
        if let last = lastLoggedAt[key], now - last < interval {
            lock.unlock()
            return
        }
        lastLoggedAt[key] = now
        lock.unlock()
        print(message())

    }
}

/// Measures movement of the strongest reflection region in canonical lip pixels.
/// Sampling and logging are bounded so this can run in ordinary builds.
enum LipHighlightMotionDiagnostics {
    private static let lock = NSLock()
    private static var lastSampleAt: CFTimeInterval = 0
    private static var windowStartedAt: CFTimeInterval = 0
    private static var source = ""
    private static var finish: LipFinish?
    private static var size = CGSize.zero
    private static var previousCenter: CGPoint?
    private static var sampleCount = 0
    private static var visibleCount = 0
    private static var moveCount = 0
    private static var moveDistance = 0.0

    static func shouldSample(at now: CFTimeInterval) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard now - lastSampleAt >= 0.10 else { return false }
        lastSampleAt = now
        return true
    }

    static func record(_ mask: [Float], width: Int, height: Int,
                       finish newFinish: LipFinish, source newSource: String,
                       at now: CFTimeInterval) {
        guard width > 0, height > 0, mask.count == width * height else { return }
        let peak = mask.reduce(Float.zero) { current, value in
            value.isFinite ? max(current, value) : current
        }
        var center: CGPoint?
        if peak >= 0.015 {
            let cutoff = max(0.010, peak * 0.65)
            var weightSum = 0.0
            var weightedX = 0.0
            var weightedY = 0.0
            for (index, value) in mask.enumerated() {
                let weight = Double(max(value - cutoff, 0))
                guard weight.isFinite, weight > 0 else { continue }
                weightSum += weight
                weightedX += (Double(index % width) + 0.5) * weight
                weightedY += (Double(index / width) + 0.5) * weight
            }
            if weightSum > 0 {
                center = CGPoint(x: weightedX / weightSum,
                                 y: weightedY / weightSum)
            }
        }

        lock.lock()
        if source != newSource || finish != newFinish ||
            size != CGSize(width: width, height: height) ||
            windowStartedAt == 0 || now - windowStartedAt > 2 {
            source = newSource
            finish = newFinish
            size = CGSize(width: width, height: height)
            previousCenter = nil
            sampleCount = 0
            visibleCount = 0
            moveCount = 0
            moveDistance = 0
            windowStartedAt = now
        }
        sampleCount += 1
        if let center {
            visibleCount += 1
            if let previousCenter {
                let distance = hypot(Double(center.x - previousCenter.x),
                                     Double(center.y - previousCenter.y))
                if distance >= 1.5 {
                    moveCount += 1
                    moveDistance += distance
                }
            }
        }
        previousCenter = center
        let elapsed = now - windowStartedAt
        guard elapsed >= 1 else {
            lock.unlock()
            return
        }
        let samples = sampleCount
        let visible = visibleCount
        let moves = moveCount
        let averageMove = moveCount > 0 ? moveDistance / Double(moveCount) : 0
        windowStartedAt = now
        sampleCount = 0
        visibleCount = 0
        moveCount = 0
        moveDistance = 0
        lock.unlock()

        let centerText = center.map {
            String(format: "%.3f,%.3f", Double($0.x) / Double(width),
                   Double($0.y) / Double(height))
        } ?? "none"
        print(String(format:
            "lip_highlight_motion finish=%@ source=%@ sampleHz=%.1f visible=%d/%d changesPerSec=%.1f avgMovePx=%.2f centerUV=%@ peak=%.3f",
            newFinish.displayName, newSource, Double(samples) / elapsed,
            visible, samples, Double(moves) / elapsed, averageMove,
            centerText, Double(peak)))
    }
}

final class FPSMeter {
    private let name: String
    private let lock = NSLock()
    private var frameCount = 0
    private var lastReportTime = CACurrentMediaTime()
    private var accumulatedMilliseconds: Double = 0
    private var measuredSamples = 0

    init(_ name: String) {
        self.name = name
    }

    func tick(workMilliseconds: Double? = nil) {

        let now = CACurrentMediaTime()
        lock.lock()
        frameCount += 1
        if let workMilliseconds {
            accumulatedMilliseconds += workMilliseconds
            measuredSamples += 1
        }

        let elapsed = now - lastReportTime
        guard elapsed >= 1 else {
            lock.unlock()
            return
        }

        let fps = Double(frameCount) / elapsed
        let averageMilliseconds = measuredSamples > 0 ? accumulatedMilliseconds / Double(measuredSamples) : nil
        frameCount = 0
        accumulatedMilliseconds = 0
        measuredSamples = 0
        lastReportTime = now
        lock.unlock()

        if let averageMilliseconds {
            print(String(format: "fps_meter %@ fps=%.1f avg_ms=%.2f", name, fps, averageMilliseconds))
        } else {
            print(String(format: "fps_meter %@ fps=%.1f", name, fps))
        }

    }
}

// Compact operational telemetry is intentionally available in Release builds.
final class LipPerformanceDiagnostics {
    private struct Snapshot {
        let reason: String
        let timestamp: Date
        let lowPowerMode: Bool
        let thermalState: ProcessInfo.ThermalState
        let requestedCameraFPS: Int
        let requestedRenderFPS: Int
        let cameraFPS: Double
        let renderFPS: Double
        let mediaPipeFPS: Double
        let minimumLandmarkSubmitInterval: CFTimeInterval
        let contourAge: CFTimeInterval?
        let cameraAge: CFTimeInterval?
        let renderAge: CFTimeInterval?
        let mediaPipeAge: CFTimeInterval?
        let landmarkSkips: Int
        let resultRejects: Int
        let contourMisses: Int
        let consecutiveContourMisses: Int
    }

    private let lock = NSLock()
    private let logQueue = DispatchQueue(label: "VirtualMakeup.PerformanceLog", qos: .utility)
    private let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "VirtualMakeup",
        category: "Performance"
    )
    private let buildIdentifier: String = {
        let info = Bundle.main.infoDictionary ?? [:]
        let version = info["CFBundleShortVersionString"] as? String ?? "unknown"
        let build = info["CFBundleVersion"] as? String ?? "unknown"
        return "\(version)(\(build))"
    }()
    private var observers: [NSObjectProtocol] = []
    private var reportTimer: DispatchSourceTimer?
    private var isRunning = false
    private var windowStartedAt = CACurrentMediaTime()
    private var cameraFrames = 0
    private var renderFrames = 0
    private var mediaPipeResults = 0
    private var lastCameraFrameTimestamp: TimeInterval?
    private var lastCameraFrameAt: CFTimeInterval?
    private var lastRenderFrameAt: CFTimeInterval?
    private var lastMediaPipeResultAt: CFTimeInterval?
    private var requestedCameraFPS = 0
    private var requestedRenderFPS = 0
    private var minimumLandmarkSubmitInterval: CFTimeInterval = 0
    private var latestContourCaptureTime: CFTimeInterval?
    private var landmarkSkips = 0
    private var resultRejects = 0
    private var contourMisses = 0
    private var consecutiveContourMisses = 0

    func start(requestedCameraFPS: Int, requestedRenderFPS: Int,
               minimumLandmarkSubmitInterval: CFTimeInterval) {
        lock.lock()
        isRunning = true
        windowStartedAt = CACurrentMediaTime()
        cameraFrames = 0
        renderFrames = 0
        mediaPipeResults = 0
        lastCameraFrameTimestamp = nil
        lastCameraFrameAt = nil
        lastRenderFrameAt = nil
        lastMediaPipeResultAt = nil
        latestContourCaptureTime = nil
        landmarkSkips = 0
        resultRejects = 0
        contourMisses = 0
        consecutiveContourMisses = 0
        self.requestedCameraFPS = requestedCameraFPS
        self.requestedRenderFPS = requestedRenderFPS
        self.minimumLandmarkSubmitInterval = minimumLandmarkSubmitInterval
        let needsObservers = observers.isEmpty
        lock.unlock()

        if needsObservers {
            let center = NotificationCenter.default
            let power = center.addObserver(
                forName: .NSProcessInfoPowerStateDidChange,
                object: nil, queue: nil
            ) { [weak self] _ in self?.report(reason: "power_state_changed") }
            let thermal = center.addObserver(
                forName: ProcessInfo.thermalStateDidChangeNotification,
                object: nil, queue: nil
            ) { [weak self] _ in self?.report(reason: "thermal_state_changed") }
            let background = center.addObserver(
                forName: UIApplication.didEnterBackgroundNotification,
                object: nil, queue: nil
            ) { [weak self] _ in self?.report(reason: "app_background") }
            let active = center.addObserver(
                forName: UIApplication.didBecomeActiveNotification,
                object: nil, queue: nil
            ) { [weak self] _ in self?.report(reason: "app_active") }
            lock.lock()
            observers = [power, thermal, background, active]
            lock.unlock()
        }
        lock.lock()
        if reportTimer == nil {
            let timer = DispatchSource.makeTimerSource(queue: logQueue)
            timer.schedule(deadline: .now() + 5, repeating: 5)
            timer.setEventHandler { [weak self] in
                self?.report(reason: "periodic", resetWindow: true)
            }
            reportTimer = timer
            timer.resume()
        }
        lock.unlock()
        report(reason: "session_start")
    }

    func stop() {
        report(reason: "session_stop")
        lock.lock()
        isRunning = false
        let oldObservers = observers
        observers = []
        let oldTimer = reportTimer
        reportTimer = nil
        lock.unlock()
        oldTimer?.cancel()
        for observer in oldObservers {
            NotificationCenter.default.removeObserver(observer)
        }
    }

    func recordCameraFrame(timestamp: TimeInterval) {
        lock.lock()
        if isRunning, timestamp.isFinite,
           lastCameraFrameTimestamp.map({ timestamp > $0 + 0.0005 }) ?? true {
            cameraFrames += 1
            lastCameraFrameTimestamp = timestamp
            lastCameraFrameAt = CACurrentMediaTime()
        }
        lock.unlock()
    }

    func recordRenderFrame() {
        lock.lock()
        if isRunning {
            renderFrames += 1
            lastRenderFrameAt = CACurrentMediaTime()
        }
        lock.unlock()
    }

    func recordMediaPipeResult() {
        lock.lock()
        if isRunning {
            mediaPipeResults += 1
            lastMediaPipeResultAt = CACurrentMediaTime()
        }
        lock.unlock()
    }

    func recordMinimumLandmarkSubmitInterval(_ interval: CFTimeInterval) {
        lock.lock()
        if isRunning { minimumLandmarkSubmitInterval = interval }
        lock.unlock()
    }

    func recordLandmarkSkip() {
        lock.lock()
        if isRunning { landmarkSkips += 1 }
        lock.unlock()
    }

    func recordResultReject() {
        lock.lock()
        if isRunning { resultRejects += 1 }
        lock.unlock()
    }

    func recordContourMiss() {
        lock.lock()
        if isRunning {
            contourMisses += 1
            consecutiveContourMisses += 1
        }
        lock.unlock()
    }

    func recordContourAccepted(capturedAt: CFTimeInterval) {
        lock.lock()
        if isRunning {
            latestContourCaptureTime = capturedAt
            consecutiveContourMisses = 0
        }
        lock.unlock()
    }

    private func report(reason: String, resetWindow: Bool = false) {
        let now = CACurrentMediaTime()
        lock.lock()
        let snapshot = isRunning ?
            snapshotLocked(reason: reason, now: now, resetWindow: resetWindow) : nil
        lock.unlock()
        if let snapshot { emit(snapshot) }
    }

    private func snapshotLocked(reason: String, now: CFTimeInterval,
                                resetWindow: Bool) -> Snapshot {
        let duration = max(now - windowStartedAt, 0.001)
        let age = latestContourCaptureTime.map { max(now - $0, 0) }
        let snapshot = Snapshot(
            reason: reason,
            timestamp: Date(),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            thermalState: ProcessInfo.processInfo.thermalState,
            requestedCameraFPS: requestedCameraFPS,
            requestedRenderFPS: requestedRenderFPS,
            cameraFPS: Double(cameraFrames) / duration,
            renderFPS: Double(renderFrames) / duration,
            mediaPipeFPS: Double(mediaPipeResults) / duration,
            minimumLandmarkSubmitInterval: minimumLandmarkSubmitInterval,
            contourAge: age,
            cameraAge: lastCameraFrameAt.map { max(now - $0, 0) },
            renderAge: lastRenderFrameAt.map { max(now - $0, 0) },
            mediaPipeAge: lastMediaPipeResultAt.map { max(now - $0, 0) },
            landmarkSkips: landmarkSkips,
            resultRejects: resultRejects,
            contourMisses: contourMisses,
            consecutiveContourMisses: consecutiveContourMisses
        )
        if resetWindow {
            windowStartedAt = now
            cameraFrames = 0
            renderFrames = 0
            mediaPipeResults = 0
        }
        return snapshot
    }

    private func emit(_ snapshot: Snapshot) {
        logQueue.async { [logger, buildIdentifier] in
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            let thermal: String
            switch snapshot.thermalState {
            case .nominal: thermal = "nominal"
            case .fair: thermal = "fair"
            case .serious: thermal = "serious"
            case .critical: thermal = "critical"
            @unknown default: thermal = "unknown"
            }
            let age = snapshot.contourAge.map {
                String(format: "%.1f", $0 * 1000)
            } ?? "none"
            let cameraAge = snapshot.cameraAge.map {
                String(format: "%.1f", $0 * 1000)
            } ?? "none"
            let renderAge = snapshot.renderAge.map {
                String(format: "%.1f", $0 * 1000)
            } ?? "none"
            let mediaPipeAge = snapshot.mediaPipeAge.map {
                String(format: "%.1f", $0 * 1000)
            } ?? "none"
            let line = String(
                format: "lip_performance timestamp=%@ build=%@ reason=%@ lowPower=%@ thermalState=%@ requestedCameraFPS=%d requestedRenderFPS=%d actualCameraFPS=%.1f actualRenderFPS=%.1f actualMediaPipeFPS=%.1f lastSelectedMinimumLandmarkSubmitIntervalMs=%.1f contourAgeMs=%@ cameraAgeMs=%@ renderAgeMs=%@ mediaPipeAgeMs=%@ landmarkSkips=%d resultRejects=%d contourMisses=%d consecutiveContourMisses=%d",
                formatter.string(from: snapshot.timestamp),
                buildIdentifier, snapshot.reason,
                snapshot.lowPowerMode ? "true" : "false", thermal,
                snapshot.requestedCameraFPS, snapshot.requestedRenderFPS,
                snapshot.cameraFPS, snapshot.renderFPS, snapshot.mediaPipeFPS,
                snapshot.minimumLandmarkSubmitInterval * 1000, age,
                cameraAge, renderAge, mediaPipeAge,
                snapshot.landmarkSkips, snapshot.resultRejects,
                snapshot.contourMisses, snapshot.consecutiveContourMisses
            )
            logger.info("\(line, privacy: .public)")
        }
    }
}
