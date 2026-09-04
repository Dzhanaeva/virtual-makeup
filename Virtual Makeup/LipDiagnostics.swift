import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

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
