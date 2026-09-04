import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

struct FaceTrackingView: UIViewRepresentable {
    @Binding var isFaceDetected: Bool
    var lipColor: LipstickColor
    var lipOpacity: Double
    var lipFinish: LipFinish
    var lipDensity: LipstickDensity
    var lipTexture: LipstickTexture
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

        context.coordinator.sceneView = view
        context.coordinator.updateLipstick(
            color: lipColor,
            opacity: lipOpacity,
            finish: lipFinish,
            density: lipDensity,
            texture: lipTexture
        )
        context.coordinator.updateBlush(color: blushColor, opacity: blushOpacity)
        context.coordinator.updateViewport(
            size: view.bounds.size,
            scale: view.traitCollection.displayScale,
            orientation: view.window?.windowScene?.effectiveGeometry.interfaceOrientation ?? .unknown
        )
        context.coordinator.startSession()

        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {
        context.coordinator.updateLipstick(
            color: lipColor,
            opacity: lipOpacity,
            finish: lipFinish,
            density: lipDensity,
            texture: lipTexture
        )
        context.coordinator.updateBlush(color: blushColor, opacity: blushOpacity)
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
        view.session.pause()
        view.delegate = nil
        view.session.delegate = nil
        (view as? ViewportTrackingARSCNView)?.onLayout = nil
        coordinator.stopSession()
        coordinator.sceneView = nil
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
