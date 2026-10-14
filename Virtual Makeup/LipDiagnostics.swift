import Foundation
import QuartzCore
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
            lock.lock()
            observers = [power, thermal]
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
        }
        lock.unlock()
    }

    func recordRenderFrame() {
        lock.lock()
        if isRunning { renderFrames += 1 }
        lock.unlock()
    }

    func recordMediaPipeResult() {
        lock.lock()
        if isRunning { mediaPipeResults += 1 }
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
            let line = String(
                format: "lip_performance timestamp=%@ build=%@ reason=%@ lowPower=%@ thermalState=%@ requestedCameraFPS=%d requestedRenderFPS=%d actualCameraFPS=%.1f actualRenderFPS=%.1f actualMediaPipeFPS=%.1f lastSelectedMinimumLandmarkSubmitIntervalMs=%.1f contourAgeMs=%@ landmarkSkips=%d resultRejects=%d contourMisses=%d consecutiveContourMisses=%d",
                formatter.string(from: snapshot.timestamp),
                buildIdentifier, snapshot.reason,
                snapshot.lowPowerMode ? "true" : "false", thermal,
                snapshot.requestedCameraFPS, snapshot.requestedRenderFPS,
                snapshot.cameraFPS, snapshot.renderFPS, snapshot.mediaPipeFPS,
                snapshot.minimumLandmarkSubmitInterval * 1000, age,
                snapshot.landmarkSkips, snapshot.resultRejects,
                snapshot.contourMisses, snapshot.consecutiveContourMisses
            )
            logger.info("\(line, privacy: .public)")
        }
    }
}
