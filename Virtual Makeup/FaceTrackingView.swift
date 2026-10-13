import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

struct LandmarkDebugMetrics {
    let activity: CGFloat
    let stability: CGFloat
    let alpha: CGFloat
    let deadZone: CGFloat
    let median: CGFloat
    let p95: CGFloat
}

struct FaceTrackingView: UIViewRepresentable {
    @Binding var isFaceDetected: Bool
    var showsLandmarks: Bool
    var lipColor: LipstickColor
    var lipOpacity: Double
    var lipFinish: LipFinish
    var lipDensity: LipstickDensity
    var lipTexture: LipstickTexture
    var lipMaterialProfile: LipMaterialProfile
    var lipShadingDebugMode: LipShadingDebugMode
    var linerColor: UIColor
    var linerOpacity: Double
    var blushColor: UIColor
    var blushOpacity: Double

    func makeUIView(context: Context) -> ARSCNView {
        let view = ViewportTrackingARSCNView(frame: .zero)
        view.delegate = context.coordinator
        view.session.delegate = context.coordinator
        view.automaticallyUpdatesLighting = false
        view.rendersContinuously = true
        view.preferredFramesPerSecond = 60
        view.backgroundColor = .black
        view.scene = SCNScene()
        view.onLayout = { [weak coordinator = context.coordinator] size, scale, orientation in
            coordinator?.updateViewport(size: size, scale: scale, orientation: orientation)
        }

        let landmarkOverlay = LandmarkDebugOverlayView(frame: view.bounds)
        landmarkOverlay.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(landmarkOverlay)

        context.coordinator.sceneView = view
        context.coordinator.configureLandmarkDebugOverlay(
            landmarkOverlay,
            isEnabled: showsLandmarks
        )
        context.coordinator.updateLipRenderingMode(.screenSpace2D)
        context.coordinator.updateLipstick(
            color: lipColor,
            opacity: lipOpacity,
            finish: lipFinish,
            density: lipDensity,
            texture: lipTexture,
            materialProfile: lipMaterialProfile,
            shadingDebugMode: lipShadingDebugMode
        )
        context.coordinator.updateBlush(color: blushColor, opacity: blushOpacity)
        context.coordinator.updateLipLiner(color: linerColor, opacity: linerOpacity)
        context.coordinator.updateViewport(
            size: view.bounds.size,
            scale: view.traitCollection.displayScale,
            orientation: view.window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .unknown
        )
        context.coordinator.startSession()

        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {
        context.coordinator.updateLandmarkDebug(isEnabled: showsLandmarks)
        context.coordinator.updateLipstick(
            color: lipColor,
            opacity: lipOpacity,
            finish: lipFinish,
            density: lipDensity,
            texture: lipTexture,
            materialProfile: lipMaterialProfile,
            shadingDebugMode: lipShadingDebugMode
        )
        context.coordinator.updateBlush(color: blushColor, opacity: blushOpacity)
        context.coordinator.updateLipLiner(color: linerColor, opacity: linerOpacity)
        context.coordinator.updateViewport(
            size: view.bounds.size,
            scale: view.window?.screen.scale ?? view.traitCollection.displayScale,
            orientation: view.window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .unknown
        )
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(isFaceDetected: $isFaceDetected)
    }

    static func dismantleUIView(_ view: ARSCNView, coordinator: Coordinator) {
        coordinator.configureLandmarkDebugOverlay(nil, isEnabled: false)
        view.session.pause()
        view.delegate = nil
        view.session.delegate = nil
        (view as? ViewportTrackingARSCNView)?.onLayout = nil
        coordinator.stopSession()
        coordinator.sceneView = nil
    }

}

final class LandmarkDebugOverlayView: UIView {
    private var facePoints: [CGPoint] = []
    private var lipPoints: [CGPoint] = []
    private var renderedLipPoints: [CGPoint] = []
    private let metricsLabel = UILabel()

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = false
        isAccessibilityElement = false
        accessibilityIdentifier = "mediapipe.landmark.overlay"
        isHidden = true
        configureMetricsLabel()
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        backgroundColor = .clear
        isOpaque = false
        isUserInteractionEnabled = false
        isAccessibilityElement = false
        accessibilityIdentifier = "mediapipe.landmark.overlay"
        isHidden = true
        configureMetricsLabel()
    }

    func updateRaw(facePoints: [CGPoint], lipPoints: [CGPoint]) {
        precondition(Thread.isMainThread)
        self.facePoints = facePoints
        self.lipPoints = lipPoints
        updateVisibility()
        setNeedsDisplay()
    }

    func updateRenderedLipPoints(_ points: [CGPoint]) {
        precondition(Thread.isMainThread)
        renderedLipPoints = points
        updateVisibility()
        setNeedsDisplay()
    }

    func updateMetrics(_ metrics: LandmarkDebugMetrics) {
        precondition(Thread.isMainThread)
        metricsLabel.text = String(
            format: "activity:  %.2f\nstability: %.2f\nalpha:     %.2f\ndeadZone:  %.2f\nmedian:    %.2f\np95:       %.2f",
            Double(metrics.activity),
            Double(metrics.stability),
            Double(metrics.alpha),
            Double(metrics.deadZone),
            Double(metrics.median),
            Double(metrics.p95)
        )
        metricsLabel.isHidden = false
        updateVisibility()
    }

    func clear() {
        precondition(Thread.isMainThread)
        facePoints = []
        lipPoints = []
        renderedLipPoints = []
        metricsLabel.text = nil
        metricsLabel.isHidden = true
        updateVisibility()
        setNeedsDisplay()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        metricsLabel.frame = CGRect(
            x: 12,
            y: safeAreaInsets.top + 54,
            width: 178,
            height: 116
        )
    }

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else {
            return
        }

        context.saveGState()
        context.setBlendMode(.normal)
        draw(
            facePoints,
            radius: 1.55,
            fillColor: UIColor.systemCyan,
            in: context
        )
        draw(
            lipPoints,
            radius: 2.6,
            fillColor: UIColor.systemYellow,
            in: context
        )
        draw(
            renderedLipPoints,
            radius: 1.8,
            fillColor: UIColor.systemPink,
            in: context
        )
        context.restoreGState()
    }

    private func updateVisibility() {
        isHidden = facePoints.isEmpty &&
            lipPoints.isEmpty &&
            renderedLipPoints.isEmpty &&
            metricsLabel.isHidden
    }

    private func configureMetricsLabel() {
        metricsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.68)
        metricsLabel.textColor = .white
        metricsLabel.font = .monospacedSystemFont(ofSize: 12, weight: .semibold)
        metricsLabel.numberOfLines = 6
        metricsLabel.layer.cornerRadius = 10
        metricsLabel.layer.masksToBounds = true
        metricsLabel.textAlignment = .left
        metricsLabel.isHidden = true
        metricsLabel.accessibilityIdentifier = "mediapipe.landmark.metrics"
        addSubview(metricsLabel)
    }

    private func draw(_ points: [CGPoint],
                      radius: CGFloat,
                      fillColor: UIColor,
                      in context: CGContext) {
        let outlineRadius = radius + 0.9
        context.setFillColor(UIColor.black.withAlphaComponent(0.72).cgColor)
        for point in points {
            context.fillEllipse(in: CGRect(
                x: point.x - outlineRadius,
                y: point.y - outlineRadius,
                width: outlineRadius * 2,
                height: outlineRadius * 2
            ))
        }

        context.setFillColor(fillColor.cgColor)
        for point in points {
            context.fillEllipse(in: CGRect(
                x: point.x - radius,
                y: point.y - radius,
                width: radius * 2,
                height: radius * 2
            ))
        }
    }
}

private final class ViewportTrackingARSCNView: ARSCNView {
    var onLayout: ((CGSize, CGFloat, UIInterfaceOrientation) -> Void)?

    override func layoutSubviews() {
        super.layoutSubviews()
        onLayout?(
            bounds.size,
            window?.screen.scale ?? traitCollection.displayScale,
            window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .unknown
        )
    }
}
