import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

final class LipMeshRenderer {
    private struct MeshVertex {
        let index: Int
        let screen: CGPoint
        let normalized: SIMD3<Float>
        let uv: CGPoint
    }

    private final class DynamicSurfaceGeometryCache {
        let deviceRegistryID: UInt64
        let landmarkIndices: [Int]
        let triangleCount: Int
        private let buffers: [MTLBuffer]
        private let geometries: [SCNGeometry]
        private var nextSlot = 0

        init?(device: MTLDevice,
              vertices: [MeshVertex],
              indices: [Int32],
              material: SCNMaterial) {
            guard !vertices.isEmpty,
                  !indices.isEmpty,
                  indices.count.isMultiple(of: 3) else {
                return nil
            }
            deviceRegistryID = device.registryID
            landmarkIndices = vertices.map(\.index)
            triangleCount = indices.count / 3

            let textureCoordinates = vertices.map {
                CGPoint(x: $0.uv.x, y: 1 - $0.uv.y)
            }
            let textureSource = SCNGeometrySource(textureCoordinates: textureCoordinates)
            let indexData = Data(
                bytes: indices,
                count: indices.count * MemoryLayout<Int32>.size
            )
            let element = SCNGeometryElement(
                data: indexData,
                primitiveType: .triangles,
                primitiveCount: triangleCount,
                bytesPerIndex: MemoryLayout<Int32>.size
            )

            var buffers: [MTLBuffer] = []
            var geometries: [SCNGeometry] = []
            buffers.reserveCapacity(3)
            geometries.reserveCapacity(3)
            let bufferLength = vertices.count * MemoryLayout<SIMD3<Float>>.stride
            for _ in 0..<3 {
                guard let buffer = device.makeBuffer(
                    length: bufferLength,
                    options: .storageModeShared
                ) else {
                    return nil
                }
                let vertexSource = SCNGeometrySource(
                    buffer: buffer,
                    vertexFormat: .float3,
                    semantic: .vertex,
                    vertexCount: vertices.count,
                    dataOffset: 0,
                    dataStride: MemoryLayout<SIMD3<Float>>.stride
                )
                let geometry = SCNGeometry(
                    sources: [vertexSource, textureSource],
                    elements: [element]
                )
                geometry.materials = [material]
                buffers.append(buffer)
                geometries.append(geometry)
            }
            self.buffers = buffers
            self.geometries = geometries
        }

        func geometry(updating positions: [SIMD3<Float>]) -> SCNGeometry? {
            guard positions.count == landmarkIndices.count,
                  !buffers.isEmpty else {
                return nil
            }
            let slot = nextSlot
            nextSlot = (nextSlot + 1) % buffers.count
            let byteCount = positions.count * MemoryLayout<SIMD3<Float>>.stride
            positions.withUnsafeBytes { bytes in
                guard let source = bytes.baseAddress else {
                    return
                }
                buffers[slot].contents().copyMemory(from: source, byteCount: byteCount)
            }
            return geometries[slot]
        }
    }

    private let lipNode = SCNNode()
    private let debugRootNode = SCNNode()
    private let debugOuterNode = SCNNode()
    private let debugInnerNode = SCNNode()
    private let debugUpperNode = SCNNode()
    private let debugMeshNode = SCNNode()
    private let debugFrameNode = SCNNode()
    private let lipMaterial = SCNMaterial()
    private let debugOuterMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemGreen)
    private let debugInnerMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemCyan)
    private let debugUpperMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemRed)
    private let debugMeshMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemYellow.withAlphaComponent(0.82))
    private let debugFrameMaterial = LipMeshRenderer.makeDebugMaterial(color: UIColor.systemPink)
    private var occlusionNode: SCNNode?
    private var lastTextureImage: UIImage?
    private let debugLinesEnabled = false
    private var lastLightingFactor: CGFloat = 1
    private var openMouthGeometryCache: DynamicSurfaceGeometryCache?
    private var requestedOpacity: CGFloat = 0.99
    private var maskVisibility: CGFloat = 0
    private var lastMaskVisibilityUpdateTime: CFTimeInterval?
    private var unavailableSince: CFTimeInterval?
    private var unavailableStartVisibility: CGFloat = 0
    private static let maskRecoveryDuration: CFTimeInterval = 0.07
    private static let unavailableFadeDuration: CFTimeInterval = 0.10

    init() {
        lipNode.name = "lipstick.mesh"
        lipNode.isHidden = true
        lipNode.renderingOrder = 100
        debugRootNode.name = "lipstick.debug"
        debugRootNode.renderingOrder = 220
        debugOuterNode.name = "lipstick.debug.outer"
        debugInnerNode.name = "lipstick.debug.inner"
        debugUpperNode.name = "lipstick.debug.upper"
        debugMeshNode.name = "lipstick.debug.mesh"
        debugFrameNode.name = "lipstick.debug.frame"
        debugRootNode.addChildNode(debugMeshNode)
        debugRootNode.addChildNode(debugOuterNode)
        debugRootNode.addChildNode(debugInnerNode)
        debugRootNode.addChildNode(debugUpperNode)
        debugRootNode.addChildNode(debugFrameNode)

        lipMaterial.lightingModel = .constant
        lipMaterial.blendMode = .alpha
        // Per-pixel opacity belongs to SCNMaterial.transparent. The diffuse
        // texture supplies colour; assigning the generated image to both
        // properties lets SceneKit read its alpha channel as the lip mask.
        lipMaterial.transparencyMode = .aOne
        lipMaterial.isDoubleSided = true
        lipMaterial.writesToDepthBuffer = false
        // The lipstick mesh is already clipped to the detected lip annulus.
        // Reading the coarse AR face depth can hide the central lip surface,
        // especially while the mouth is closed.
        lipMaterial.readsFromDepthBuffer = false
        for property in [lipMaterial.diffuse, lipMaterial.transparent] {
            property.mappingChannel = 0
            property.magnificationFilter = .linear
            property.minificationFilter = .linear
            property.mipFilter = .none
            property.wrapS = .clamp
            property.wrapT = .clamp
        }
        // Colour and coverage arrive as separate images. Give each material
        // property only the channels it owns; otherwise alpha can be applied
        // twice and collapse a soft feather into a hard edge.
        lipMaterial.diffuse.textureComponents = [.red, .green, .blue]
        lipMaterial.transparent.textureComponents = .alpha
        lipMaterial.multiply.contents = UIColor.white
        lipMaterial.multiply.intensity = 1
    }

    func updateOpacity(_ opacity: Double) {
        // Zero is reserved for the before/after original-image toggle. At the
        // visible end, retain a little camera detail instead of becoming a
        // fully opaque vector layer.
        requestedOpacity = CGFloat(max(0, min(opacity, 0.99)))
        applyEffectiveOpacity()
    }

    func updateLightingFactor(_ factor: CGFloat) {
        let clampedFactor = max(0.45, min(factor, 1))
        guard abs(clampedFactor - lastLightingFactor) > 0.006 else {
            return
        }

        lastLightingFactor = clampedFactor
        // Preserve the catalogue colour in ordinary and bright conditions.
        // Only the darkest end of the AR light estimate receives a shallow
        // reduction, capped at ten percent and never allowed to brighten RGB.
        let transition = max(0, min((clampedFactor - 0.50) / 0.18, 1))
        let easedTransition = transition * transition * (3 - 2 * transition)
        let displayedFactor = 0.90 + easedTransition * 0.10
        lipMaterial.multiply.contents = UIColor(
            red: displayedFactor,
            green: displayedFactor,
            blue: displayedFactor,
            alpha: 1
        )
    }

    func attach(to faceNode: SCNNode) {
        if lipNode.parent !== faceNode {
            lipNode.removeFromParentNode()
            faceNode.addChildNode(lipNode)
        }
        if debugRootNode.parent !== faceNode {
            debugRootNode.removeFromParentNode()
            faceNode.addChildNode(debugRootNode)
        }
    }

    func updateOccluder(faceAnchor: ARFaceAnchor, renderer: SCNSceneRenderer, faceNode: SCNNode) {
        if occlusionNode == nil,
           let device = renderer.device,
           let geometry = ARSCNFaceGeometry(device: device) {
            let material = SCNMaterial()
            material.colorBufferWriteMask = []
            material.writesToDepthBuffer = true
            material.readsFromDepthBuffer = true
            geometry.materials = [material]

            let node = SCNNode(geometry: geometry)
            node.name = "face.depth.occluder"
            node.renderingOrder = -100
            faceNode.addChildNode(node)
            occlusionNode = node
        }

        if let occlusionNode, occlusionNode.parent !== faceNode {
            occlusionNode.removeFromParentNode()
            faceNode.addChildNode(occlusionNode)
        }
        (occlusionNode?.geometry as? ARSCNFaceGeometry)?.update(from: faceAnchor.geometry)
    }

    func render(contour: LipContour,
                texture: LipTexture,
                faceGeometry: ARFaceGeometry,
                mouthFrame: FaceLocalMouthFrame,
                renderer: SCNSceneRenderer,
                faceNode: SCNNode,
                contourAge: CFTimeInterval,
                motionDelta: CGFloat,
                freshnessVisibility: CGFloat) {
        let correctedMouthFrame = mouthFrame

        guard let geometry = makeGeometry(
            contour: contour,
            texture: texture,
            faceGeometry: faceGeometry,
            mouthFrame: correctedMouthFrame,
            renderer: renderer,
            faceNode: faceNode
        ) else {
            renderUnavailable()
            return
        }

        unavailableSince = nil
        unavailableStartVisibility = 0
        updateMaskVisibility(
            target: max(0, min(freshnessVisibility, 1)),
            now: CACurrentMediaTime()
        )
        if lastTextureImage !== texture.image {
            lipMaterial.diffuse.contents = texture.image
            lipMaterial.transparent.contents = texture.opacityImage
            lastTextureImage = texture.image
        }
        lipMaterial.readsFromDepthBuffer = false
        lipNode.geometry = geometry
        lipNode.isHidden = false
        LipDebugLog.throttled(
            "lip_mesh_alignment",
            interval: 0.75,
            "lip_mesh alignment shape=mediapipe_screen_locked depth=ar_surface aperture=mediapipe_inner age=\(Self.fmt(CGFloat(contourAge))) motion=\(Self.fmt(motionDelta))"
        )
        if debugLinesEnabled {
            updateDebugLines(contour: contour, mouthFrame: correctedMouthFrame)
            logProjectionDiagnostics(
                contour: contour,
                mouthFrame: correctedMouthFrame,
                renderer: renderer,
                faceNode: faceNode
            )
        } else {
            clearDebugLines()
            logAttachmentDiagnostics(
                contour: contour,
                contourAge: contourAge,
                motionDelta: motionDelta
            )
        }
    }

    func renderUnavailable() {
        guard lipNode.geometry != nil,
              !lipNode.isHidden else {
            return
        }

        let now = CACurrentMediaTime()
        if unavailableSince == nil {
            unavailableSince = now
            unavailableStartVisibility = maskVisibility
        }
        let elapsed = max(now - (unavailableSince ?? now), 0)
        let progress = Self.smoothStep(
            edge0: 0,
            edge1: CGFloat(Self.unavailableFadeDuration),
            value: CGFloat(elapsed)
        )
        updateMaskVisibility(
            target: unavailableStartVisibility * (1 - progress),
            now: now
        )
        if maskVisibility <= 0.001 {
            lipNode.geometry = nil
            lipNode.isHidden = true
        }
    }

    func clearLip() {
        lipNode.geometry = nil
        lipNode.isHidden = true
        maskVisibility = 0
        lastMaskVisibilityUpdateTime = nil
        unavailableSince = nil
        unavailableStartVisibility = 0
        applyEffectiveOpacity()
        clearDebugLines()
    }

    func clearAll() {
        clearLip()
        lastTextureImage = nil
        lipMaterial.diffuse.contents = nil
        lipMaterial.transparent.contents = nil
    }

    private func updateMaskVisibility(
        target: CGFloat,
        now: CFTimeInterval
    ) {
        let clampedTarget = max(0, min(target, 1))
        if clampedTarget <= maskVisibility {
            // Do not prolong an obsolete contour: fading out follows freshness
            // immediately. Only recovery is eased to avoid a bright one-frame
            // flash when a new MediaPipe result arrives.
            maskVisibility = clampedTarget
        } else {
            let elapsed = lastMaskVisibilityUpdateTime.map {
                max(now - $0, 0)
            } ?? (1.0 / 60.0)
            let recoveryProgress = min(
                CGFloat(elapsed / Self.maskRecoveryDuration),
                1
            )
            maskVisibility = min(
                clampedTarget,
                maskVisibility + recoveryProgress
            )
        }
        lastMaskVisibilityUpdateTime = now
        applyEffectiveOpacity()
    }

    private func applyEffectiveOpacity() {
        lipMaterial.transparency = requestedOpacity * maskVisibility
    }

    private static func makeDebugMaterial(color: UIColor) -> SCNMaterial {
        let material = SCNMaterial()
        material.diffuse.contents = color
        material.emission.contents = color
        material.lightingModel = .constant
        material.isDoubleSided = true
        material.readsFromDepthBuffer = false
        material.writesToDepthBuffer = false
        return material
    }

    private func makeGeometry(contour: LipContour,
                              texture: LipTexture,
                              faceGeometry: ARFaceGeometry,
                              mouthFrame: FaceLocalMouthFrame,
                              renderer: SCNSceneRenderer,
                              faceNode: SCNNode) -> SCNGeometry? {
        guard let pose = contour.pose else {
            LipDebugLog.throttled(
                "lip_mesh_nil_pose",
                "lip_mesh makeGeometry=nil reason=no_pose outer=\(contour.outer.count) inner=\(contour.inner.count)"
            )
            return nil
        }
        guard contour.outer.count == Self.lipPointCount,
              contour.inner.count == Self.lipPointCount,
              contour.outer3D.count == Self.lipPointCount,
              contour.inner3D.count == Self.lipPointCount,
              contour.outerUV.count == Self.lipPointCount,
              contour.innerUV.count == Self.lipPointCount else {
            LipDebugLog.throttled(
                "lip_mesh_bad_counts",
                "lip_mesh makeGeometry=nil reason=bad_counts outer=\(contour.outer.count) inner=\(contour.inner.count) outer3D=\(contour.outer3D.count) inner3D=\(contour.inner3D.count) outerUV=\(contour.outerUV.count) innerUV=\(contour.innerUV.count)"
            )
            return nil
        }
        guard mouthFrame.width > 0 else {
            LipDebugLog.throttled(
                "lip_mesh_bad_mouth_frame",
                "lip_mesh makeGeometry=nil reason=bad_mouth_frame width=\(mouthFrame.width)"
            )
            return nil
        }

        guard hasCompatibleDepthCarrier(contour, faceGeometry: faceGeometry) else {
            LipDebugLog.throttled(
                "lip_mesh_no_depth_carrier",
                interval: 0.4,
                "lip_mesh hide reason=no_fresh_ar_depth_carrier shape=mediapipe"
            )
            return nil
        }

        let faceVertices = faceGeometry.vertices
        guard let device = renderer.device else {
            return nil
        }
        // Keep enough carrier outside MediaPipe's vermilion edge to display the
        // complete pronounced exterior feather. The alpha tail, rather than a
        // solid mesh edge, controls how much of this extension remains visible.
        let mesh = makeMediaPipeLipMesh(
            contour: contour,
            expandsOuterFeatherBoundary: true
        )
        guard mesh.vertices.count >= 4,
              !mesh.indices.isEmpty else {
            LipDebugLog.throttled(
                "lip_mesh_empty_geometry",
                "lip_mesh makeGeometry=nil reason=mediapipe_mesh"
            )
            return nil
        }
        var vertexByLandmark: [Int: MeshVertex] = [:]
        vertexByLandmark.reserveCapacity(mesh.vertices.count)
        for vertex in mesh.vertices {
            guard vertexByLandmark.updateValue(vertex, forKey: vertex.index) == nil else {
                LipDebugLog.throttled(
                    "lip_mesh_duplicate_landmark",
                    "lip_mesh makeGeometry=nil reason=duplicate_landmark index=\(vertex.index)"
                )
                return nil
            }
        }
        var geometryCache = openMouthGeometryCache
        if geometryCache?.deviceRegistryID != device.registryID {
            guard let newCache = DynamicSurfaceGeometryCache(
                    device: device,
                    vertices: mesh.vertices,
                    indices: mesh.indices,
                    material: lipMaterial
                  ) else {
                LipDebugLog.throttled(
                    "lip_mesh_empty_geometry",
                    "lip_mesh makeGeometry=nil reason=dynamic_geometry_cache"
                )
                return nil
            }
            geometryCache = newCache
            openMouthGeometryCache = newCache
        }
        guard let geometryCache else {
            return nil
        }

        let textureWidth = texture.image.cgImage?.width ?? 0
        let textureHeight = texture.image.cgImage?.height ?? 0
        LipDebugLog.throttled(
            "lip_mesh_geometry",
            interval: 0.75,
            "lip_mesh geometry vertices=\(geometryCache.landmarkIndices.count) triangles=\(geometryCache.triangleCount) buffered=true shape=mediapipe_screen_locked depth=ar_surface texture=\(textureWidth)x\(textureHeight) mouthWidth=\(String(format: "%.4f", mouthFrame.width)) poseWidth=\(String(format: "%.1f", pose.width))"
        )

        var positions: [SIMD3<Float>] = []
        positions.reserveCapacity(geometryCache.landmarkIndices.count)
        for landmarkIndex in geometryCache.landmarkIndices {
            guard let vertex = vertexByLandmark[landmarkIndex],
                  let binding = contour.surfaceBindingsByIndex[landmarkIndex],
                  let depthReference = faceSurfacePoint(
                    for: binding,
                    faceVertices: faceVertices,
                    outwardReference: mouthFrame.normalAxis,
                    surfaceLift: 0.001_50
                  ),
                  let point = screenLockedScenePoint(
                    depthReference: SCNVector3(
                        depthReference.x,
                        depthReference.y,
                        depthReference.z
                    ),
                    mediaPipeScreenPoint: vertex.screen,
                    renderer: renderer,
                    faceNode: faceNode
                  ) else {
                LipDebugLog.throttled(
                    "lip_mesh_screen_lock_vertex_invalid",
                    interval: 0.4,
                    "lip_mesh hide reason=screen_lock_vertex_invalid index=\(landmarkIndex)"
                )
                return nil
            }
            positions.append(SIMD3<Float>(point.x, point.y, point.z))
        }

        guard hasValidScreenLockedGeometry(
            positions,
            landmarkIndices: geometryCache.landmarkIndices,
            meshVerticesByLandmark: vertexByLandmark,
            mouthFrame: mouthFrame,
            renderer: renderer,
            faceNode: faceNode
        ) else {
            LipDebugLog.throttled(
                "lip_mesh_screen_lock_geometry_invalid",
                interval: 0.4,
                "lip_mesh hide reason=invalid_mediapipe_screen_locked_geometry"
            )
            return nil
        }
        return geometryCache.geometry(updating: positions)
    }

    private func screenLockedScenePoint(
        depthReference: SCNVector3,
        mediaPipeScreenPoint: CGPoint,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode
    ) -> SCNVector3? {
        guard mediaPipeScreenPoint.x.isFinite,
              mediaPipeScreenPoint.y.isFinite else {
            return nil
        }

        let worldDepthReference = faceNode.convertPosition(depthReference, to: nil)
        let projectedDepthReference = renderer.projectPoint(worldDepthReference)
        guard projectedDepthReference.z.isFinite,
              projectedDepthReference.z >= 0,
              projectedDepthReference.z <= 1 else {
            return nil
        }

        // MediaPipe owns the keyframe X/Y shape. Any bounded ARKit delta-warp
        // has already been applied to this screen point; ARFaceGeometry also
        // contributes the camera-space depth used for unprojection.
        let worldTarget = renderer.unprojectPoint(
            SCNVector3(
                Float(mediaPipeScreenPoint.x),
                Float(mediaPipeScreenPoint.y),
                projectedDepthReference.z
            )
        )
        guard worldTarget.x.isFinite,
              worldTarget.y.isFinite,
              worldTarget.z.isFinite else {
            return nil
        }

        let localTarget = faceNode.convertPosition(worldTarget, from: nil)
        guard localTarget.x.isFinite,
              localTarget.y.isFinite,
              localTarget.z.isFinite else {
            return nil
        }
        return localTarget
    }

    private func hasValidScreenLockedGeometry(
        _ positions: [SIMD3<Float>],
        landmarkIndices: [Int],
        meshVerticesByLandmark: [Int: MeshVertex],
        mouthFrame: FaceLocalMouthFrame,
        renderer: SCNSceneRenderer,
        faceNode: SCNNode
    ) -> Bool {
        guard positions.count == CanonicalLipGeometry.attentionLipIndices.count,
              landmarkIndices.count == positions.count,
              meshVerticesByLandmark.count == positions.count,
              mouthFrame.width.isFinite,
              mouthFrame.width > 0 else {
            return false
        }

        var seenLandmarks = Set<Int>()
        seenLandmarks.reserveCapacity(positions.count)
        var maximumReprojectionError: CGFloat = 0
        let maximumLocalDistance = mouthFrame.width * 1.75
        let maximumNormalDistance = mouthFrame.width * 1.25

        for (offset, landmarkIndex) in landmarkIndices.enumerated() {
            guard positions.indices.contains(offset),
                  seenLandmarks.insert(landmarkIndex).inserted,
                  let vertex = meshVerticesByLandmark[landmarkIndex] else {
                return false
            }
            let position = positions[offset]
            guard position.x.isFinite,
                  position.y.isFinite,
                  position.z.isFinite else {
                return false
            }

            let relative = position - mouthFrame.center
            let localDistance = simd_length(relative)
            let normalDistance = abs(simd_dot(relative, mouthFrame.normalAxis))
            guard localDistance.isFinite,
                  normalDistance.isFinite,
                  localDistance <= maximumLocalDistance,
                  normalDistance <= maximumNormalDistance else {
                return false
            }

            let localPoint = SCNVector3(position.x, position.y, position.z)
            let worldPoint = faceNode.convertPosition(localPoint, to: nil)
            let projected = renderer.projectPoint(worldPoint)
            guard projected.x.isFinite,
                  projected.y.isFinite,
                  projected.z.isFinite,
                  projected.z >= 0,
                  projected.z <= 1 else {
                return false
            }

            let reprojectionError = hypot(
                CGFloat(projected.x) - vertex.screen.x,
                CGFloat(projected.y) - vertex.screen.y
            )
            guard reprojectionError.isFinite else {
                return false
            }
            maximumReprojectionError = max(
                maximumReprojectionError,
                reprojectionError
            )
        }

        // project(unproject(MediaPipe X/Y, AR depth)) must remain pixel-tight.
        // A larger delta means coordinate spaces were mixed, so hide instead of
        // substituting a generated mouth shape.
        let maximumAllowedReprojectionError: CGFloat = 1.25
        guard maximumReprojectionError <= maximumAllowedReprojectionError else {
            LipDebugLog.throttled(
                "lip_mesh_screen_lock_reprojection",
                interval: 0.4,
                "lip_mesh hide reason=screen_reprojection_error maxPx=\(Self.fmt(maximumReprojectionError)) allowedPx=\(Self.fmt(maximumAllowedReprojectionError))"
            )
            return false
        }
        LipDebugLog.throttled(
            "lip_mesh_screen_lock_ok",
            interval: 0.75,
            "lip_mesh screenLock=ok maxPx=\(Self.fmt(maximumReprojectionError)) vertices=\(positions.count)"
        )
        return true
    }

    private func hasCompatibleDepthCarrier(_ contour: LipContour,
                                           faceGeometry: ARFaceGeometry) -> Bool {
        let faceVertices = faceGeometry.vertices
        let faceTriangleIndices = faceGeometry.triangleIndices
        guard contour.surfaceVertexCount == faceVertices.count,
              contour.surfaceTriangleIndexCount == faceTriangleIndices.count,
              contour.surfaceTopologySignature == faceSurfaceTopologySignature(
                vertexCount: faceVertices.count,
                triangleIndices: faceTriangleIndices
              ) else {
            return false
        }
        // Bindings identify triangles in the current ARFaceGeometry; they do
        // not contain an old depth value. Depth is sampled again from the
        // current vertices below, while MediaPipe still owns visible X/Y.
        // Anchor/epoch/viewport and topology checks already prevent bindings
        // from crossing tracking contexts, so expiring them by wall-clock time
        // only made a valid contour blink every 0.5 seconds.
        return CanonicalLipGeometry.attentionLipIndices.allSatisfy {
            contour.surfaceBindingsByIndex[$0] != nil
        }
    }

    private func faceSurfacePoint(for binding: LipSurfaceBinding,
                                  faceVertices: [SIMD3<Float>],
                                  outwardReference: SIMD3<Float>,
                                  surfaceLift: Float = 0.000_70) -> SIMD3<Float>? {
        let firstIndex = Int(binding.vertexIndices.x)
        let secondIndex = Int(binding.vertexIndices.y)
        let thirdIndex = Int(binding.vertexIndices.z)
        guard faceVertices.indices.contains(firstIndex),
              faceVertices.indices.contains(secondIndex),
              faceVertices.indices.contains(thirdIndex) else {
            return nil
        }

        let weights = binding.barycentricWeights
        let weightSum = weights.x + weights.y + weights.z
        guard weights.x.isFinite,
              weights.y.isFinite,
              weights.z.isFinite,
              weightSum.isFinite,
              abs(weightSum - 1) < 0.002 else {
            return nil
        }

        let first = faceVertices[firstIndex]
        let second = faceVertices[secondIndex]
        let third = faceVertices[thirdIndex]
        var point = first * weights.x + second * weights.y + third * weights.z
        var normal = simd_cross(second - first, third - first)
        if simd_length_squared(normal) > 0.000_000_000_1 {
            normal = simd_normalize(normal)
            if simd_dot(normal, outwardReference) < 0 {
                normal = -normal
            }
            point += normal * surfaceLift
        } else {
            point += outwardReference * surfaceLift
        }

        guard point.x.isFinite,
              point.y.isFinite,
              point.z.isFinite else {
            return nil
        }
        return point
    }

    private func updateDebugLines(contour: LipContour, mouthFrame: FaceLocalMouthFrame) {
        guard debugLinesEnabled,
              let pose = contour.pose else {
            clearDebugLines()
            return
        }

        let mesh = makeMediaPipeLipMesh(contour: contour)
        let sceneVertices = mesh.vertices.map {
            scenePoint(for: $0, pose: pose, contour: contour, mouthFrame: mouthFrame)
        }
        var pointByIndex: [Int: SCNVector3] = [:]
        pointByIndex.reserveCapacity(mesh.vertices.count)
        for (offset, vertex) in mesh.vertices.enumerated() {
            guard sceneVertices.indices.contains(offset) else {
                continue
            }
            pointByIndex[vertex.index] = sceneVertices[offset]
        }

        debugOuterNode.geometry = lineLoopGeometry(
            indices: CanonicalLipGeometry.outerLipIndices,
            pointByIndex: pointByIndex,
            material: debugOuterMaterial
        )
        debugInnerNode.geometry = lineLoopGeometry(
            indices: CanonicalLipGeometry.innerLipIndices,
            pointByIndex: pointByIndex,
            material: debugInnerMaterial
        )
        debugUpperNode.geometry = lineStripGeometry(
            indices: CanonicalLipGeometry.upperDebugIndices,
            pointByIndex: pointByIndex,
            offset: mouthFrame.normalAxis * mouthFrame.width * 0.018,
            material: debugUpperMaterial
        )
        debugMeshNode.geometry = meshWireGeometry(
            vertices: sceneVertices,
            triangleIndices: mesh.indices,
            material: debugMeshMaterial
        )
        debugFrameNode.geometry = mouthFrameGeometry(
            mouthFrame,
            material: debugFrameMaterial
        )
        debugRootNode.isHidden = false
    }

    private func clearDebugLines() {
        debugOuterNode.geometry = nil
        debugInnerNode.geometry = nil
        debugUpperNode.geometry = nil
        debugMeshNode.geometry = nil
        debugFrameNode.geometry = nil
        debugRootNode.isHidden = true
    }

    private func lineLoopGeometry(indices: [Int],
                                  pointByIndex: [Int: SCNVector3],
                                  material: SCNMaterial) -> SCNGeometry? {
        let points = indices.compactMap { pointByIndex[$0] }
        guard points.count >= 2 else {
            return nil
        }

        var lineIndices = [Int32]()
        lineIndices.reserveCapacity(points.count * 2)
        for index in points.indices {
            lineIndices.append(Int32(index))
            lineIndices.append(Int32((index + 1) % points.count))
        }

        return lineGeometry(vertices: points, indices: lineIndices, material: material)
    }

    private func lineStripGeometry(indices: [Int],
                                   pointByIndex: [Int: SCNVector3],
                                   offset: SIMD3<Float>,
                                   material: SCNMaterial) -> SCNGeometry? {
        let points = indices.compactMap { index -> SCNVector3? in
            guard let point = pointByIndex[index] else {
                return nil
            }
            return SCNVector3(point.x + offset.x, point.y + offset.y, point.z + offset.z)
        }
        guard points.count >= 2 else {
            return nil
        }

        var lineIndices = [Int32]()
        lineIndices.reserveCapacity((points.count - 1) * 2)
        for index in 0..<(points.count - 1) {
            lineIndices.append(Int32(index))
            lineIndices.append(Int32(index + 1))
        }

        return lineGeometry(vertices: points, indices: lineIndices, material: material)
    }

    private func meshWireGeometry(vertices: [SCNVector3],
                                  triangleIndices: [Int32],
                                  material: SCNMaterial) -> SCNGeometry? {
        guard vertices.count >= 3, triangleIndices.count >= 3 else {
            return nil
        }

        var lineIndices = [Int32]()
        lineIndices.reserveCapacity(triangleIndices.count * 2)
        for offset in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let first = triangleIndices[offset]
            let second = triangleIndices[offset + 1]
            let third = triangleIndices[offset + 2]
            lineIndices.append(contentsOf: [first, second, second, third, third, first])
        }

        return lineGeometry(vertices: vertices, indices: lineIndices, material: material)
    }

    private func mouthFrameGeometry(_ mouthFrame: FaceLocalMouthFrame,
                                    material: SCNMaterial) -> SCNGeometry? {
        let center = SCNVector3(mouthFrame.center.x, mouthFrame.center.y, mouthFrame.center.z)
        let right = mouthFrame.center + mouthFrame.xAxis * mouthFrame.width * 0.42
        let down = mouthFrame.center + mouthFrame.downAxis * mouthFrame.width * 0.30
        let normal = mouthFrame.center + mouthFrame.normalAxis * mouthFrame.width * 0.30
        let vertices = [
            center,
            SCNVector3(right.x, right.y, right.z),
            center,
            SCNVector3(down.x, down.y, down.z),
            center,
            SCNVector3(normal.x, normal.y, normal.z)
        ]
        return lineGeometry(
            vertices: vertices,
            indices: [0, 1, 2, 3, 4, 5],
            material: material
        )
    }

    private func lineGeometry(vertices: [SCNVector3],
                              indices: [Int32],
                              material: SCNMaterial) -> SCNGeometry? {
        guard vertices.count >= 2, indices.count >= 2 else {
            return nil
        }

        let source = SCNGeometrySource(vertices: vertices)
        let element = SCNGeometryElement(indices: indices, primitiveType: .line)
        let geometry = SCNGeometry(sources: [source], elements: [element])
        geometry.materials = [material]
        return geometry
    }

    private func logProjectionDiagnostics(contour: LipContour,
                                          mouthFrame: FaceLocalMouthFrame,
                                          renderer: SCNSceneRenderer,
                                          faceNode: SCNNode) {
        guard let pose = contour.pose else {
            return
        }

        let mesh = makeMediaPipeLipMesh(contour: contour)
        guard !mesh.vertices.isEmpty else {
            return
        }

        var deltasByIndex: [Int: CGPoint] = [:]
        deltasByIndex.reserveCapacity(mesh.vertices.count)
        var allProjected = [CGPoint]()
        allProjected.reserveCapacity(mesh.vertices.count)

        for vertex in mesh.vertices {
            let local = screenAlignedScenePoint(
                for: vertex,
                pose: pose,
                contour: contour,
                mouthFrame: mouthFrame,
                renderer: renderer,
                faceNode: faceNode,
                contourAge: 0,
                motionDelta: 0
            )
            let world = faceNode.convertPosition(local, to: nil)
            let projected = renderer.projectPoint(world)
            let screen = CGPoint(x: CGFloat(projected.x), y: CGFloat(projected.y))
            guard screen.x.isFinite, screen.y.isFinite, projected.z.isFinite else {
                continue
            }

            deltasByIndex[vertex.index] = CGPoint(
                x: screen.x - vertex.screen.x,
                y: screen.y - vertex.screen.y
            )
            allProjected.append(screen)
        }

        guard !deltasByIndex.isEmpty else {
            return
        }

        let upper = Self.deltaStats(
            for: CanonicalLipGeometry.upperDebugIndices,
            in: deltasByIndex
        )
        let lower = Self.deltaStats(
            for: [146, 91, 181, 84, 17, 314, 405, 321, 375],
            in: deltasByIndex
        )
        let inner = Self.deltaStats(
            for: CanonicalLipGeometry.innerLipIndices,
            in: deltasByIndex
        )
        let bounds = boundsText(contour.bounds)
        let projectedBounds = Self.bounds(for: allProjected)
        let projectedBoundsText = boundsText(projectedBounds)

        LipDebugLog.throttled(
            "lip_projection_diagnostics",
            interval: 0.22,
            "lip_projection deltaPx upper(avgX:\(Self.fmt(upper.avgX)) avgY:\(Self.fmt(upper.avgY)) max:\(Self.fmt(upper.maxDistance)) n:\(upper.count)) lower(avgX:\(Self.fmt(lower.avgX)) avgY:\(Self.fmt(lower.avgY)) max:\(Self.fmt(lower.maxDistance)) n:\(lower.count)) inner(avgX:\(Self.fmt(inner.avgX)) avgY:\(Self.fmt(inner.avgY)) max:\(Self.fmt(inner.maxDistance)) n:\(inner.count)) mouth(width:\(Self.fmt(CGFloat(mouthFrame.width))) ref:\(Self.fmt(CGFloat(mouthFrame.referenceWidth))) open:\(Self.fmt(CGFloat(mouthFrame.openingRatio))) smile:\(Self.fmt(CGFloat(mouthFrame.smileExpansion))) valign:\(Self.fmt(CGFloat(mouthFrame.verticalAlignment))) contour:\(bounds) projected:\(projectedBoundsText)"
        )
    }

    private func logAttachmentDiagnostics(contour: LipContour,
                                          contourAge: CFTimeInterval,
                                          motionDelta: CGFloat) {
        var totalError: CGFloat = 0
        var maxError: CGFloat = 0
        var count: CGFloat = 0
        var upperTotalError: CGFloat = 0
        var upperCount: CGFloat = 0

        for index in CanonicalLipGeometry.attentionLipIndices {
            guard let binding = contour.surfaceBindingsByIndex[index],
                  binding.projectionError.isFinite else {
                continue
            }

            let error = binding.projectionError
            totalError += error
            maxError = max(maxError, error)
            count += 1
            if CanonicalLipGeometry.upperDebugIndices.contains(index) {
                upperTotalError += error
                upperCount += 1
            }
        }

        guard count > 0 else {
            return
        }

        LipDebugLog.throttled(
            "lip_attachment_error",
            interval: 0.75,
            "lip_attachment carrierError avg=\(Self.fmt(totalError / count)) upper=\(Self.fmt(upperCount > 0 ? upperTotalError / upperCount : 0)) max=\(Self.fmt(maxError)) age=\(Self.fmt(CGFloat(contourAge))) motion=\(Self.fmt(motionDelta))"
        )
    }

    private static func deltaStats(for indices: [Int],
                                   in deltasByIndex: [Int: CGPoint]) -> (avgX: CGFloat, avgY: CGFloat, maxDistance: CGFloat, count: Int) {
        var sumX: CGFloat = 0
        var sumY: CGFloat = 0
        var maxDistance: CGFloat = 0
        var count = 0

        for index in indices {
            guard let delta = deltasByIndex[index] else {
                continue
            }
            sumX += delta.x
            sumY += delta.y
            maxDistance = max(maxDistance, hypot(delta.x, delta.y))
            count += 1
        }

        guard count > 0 else {
            return (0, 0, 0, 0)
        }
        return (sumX / CGFloat(count), sumY / CGFloat(count), maxDistance, count)
    }

    private static func bounds(for points: [CGPoint]) -> CGRect? {
        guard let first = points.first else {
            return nil
        }

        var minX = first.x
        var maxX = first.x
        var minY = first.y
        var maxY = first.y

        for point in points.dropFirst() {
            minX = min(minX, point.x)
            maxX = max(maxX, point.x)
            minY = min(minY, point.y)
            maxY = max(maxY, point.y)
        }

        return CGRect(x: minX, y: minY, width: maxX - minX, height: maxY - minY)
    }

    private static func mouthOpeningRatio(_ contour: LipContour) -> CGFloat {
        guard let outerBounds = bounds(for: contour.outer),
              let innerBounds = bounds(for: contour.inner),
              outerBounds.width > 1 else {
            return 0
        }

        return innerBounds.height / outerBounds.width
    }

    private func boundsText(_ bounds: CGRect?) -> String {
        guard let bounds else {
            return "nil"
        }

        return "x:\(Self.fmt(bounds.minX)) y:\(Self.fmt(bounds.minY)) w:\(Self.fmt(bounds.width)) h:\(Self.fmt(bounds.height))"
    }

    private static func fmt(_ value: CGFloat) -> String {
        String(format: "%.2f", Double(value))
    }

    private func makeMediaPipeLipMesh(
        contour: LipContour,
        expandsOuterFeatherBoundary: Bool = false
    ) -> (vertices: [MeshVertex], indices: [Int32]) {
        var vertices: [MeshVertex] = []
        var indices: [Int32] = []
        var vertexIndexByLandmark: [Int: Int32] = [:]
        let expandedOuterScreenPoints: [Int: CGPoint]
        let expandedOuterUVPoints: [Int: CGPoint]
        if expandsOuterFeatherBoundary,
           let pose = contour.pose {
            expandedOuterScreenPoints = LipOuterBoundary.expandedScreenPoints(
                contour: contour,
                distance: LipOuterMargin.carrier(for: pose.width)
            ) ?? [:]
            expandedOuterUVPoints = LipOuterBoundary.expandedUVPoints(
                contour: contour,
                distance: LipOuterFeatherLayout.canonicalCarrierDistance,
                aspectRatio: LipOuterFeatherLayout.canonicalTextureAspectRatio
            ) ?? [:]
        } else {
            expandedOuterScreenPoints = [:]
            expandedOuterUVPoints = [:]
        }

        func meshVertex(for landmarkIndex: Int) -> MeshVertex? {
            guard let point = contour.meshPointsByIndex[landmarkIndex] else {
                return nil
            }
            let textureUV: CGPoint
            let screenPoint: CGPoint
            if expandsOuterFeatherBoundary,
               CanonicalLipGeometry.isOuterLipIndex(landmarkIndex),
               let expandedScreenPoint = expandedOuterScreenPoints[landmarkIndex],
               let expandedUVPoint = expandedOuterUVPoints[landmarkIndex] {
                textureUV = expandedUVPoint
                screenPoint = expandedScreenPoint
            } else if CanonicalLipGeometry.isInnerLipIndex(landmarkIndex) {
                textureUV = point.uv
                screenPoint = point.screen
            } else {
                textureUV = point.uv
                screenPoint = point.screen
            }
            return MeshVertex(
                index: landmarkIndex,
                screen: screenPoint,
                normalized: point.normalized,
                uv: textureUV
            )
        }

        func appendOrReuse(_ vertex: MeshVertex) -> Int32 {
            if let existing = vertexIndexByLandmark[vertex.index] {
                return existing
            }

            let newIndex = Int32(vertices.count)
            vertices.append(vertex)
            vertexIndexByLandmark[vertex.index] = newIndex
            return newIndex
        }

        @discardableResult
        func appendTriangleSet(_ triangles: [(Int, Int, Int)]) -> Int {
            var appended = 0
            for triangle in triangles {
                guard let first = meshVertex(for: triangle.0),
                      let second = meshVertex(for: triangle.1),
                      let third = meshVertex(for: triangle.2) else {
                    continue
                }

                indices.append(contentsOf: [
                    appendOrReuse(first),
                    appendOrReuse(second),
                    appendOrReuse(third)
                ])
                appended += 1
            }
            return appended
        }

        let appendedSurfaceTriangleCount = appendTriangleSet(
            CanonicalLipGeometry.lipMeshTriangles
        )
        let appendedFillTriangleCount = appendTriangleSet(
            CanonicalLipGeometry.innerFillTriangles
        )
        let expectedTriangleCount =
            CanonicalLipGeometry.lipMeshTriangles.count +
            CanonicalLipGeometry.innerFillTriangles.count
        guard appendedSurfaceTriangleCount == CanonicalLipGeometry.lipMeshTriangles.count,
              appendedFillTriangleCount == CanonicalLipGeometry.innerFillTriangles.count,
              vertices.count == CanonicalLipGeometry.attentionLipIndices.count,
              indices.count == expectedTriangleCount * 3 else {
            return ([], [])
        }
        return (vertices, indices)
    }

    private func scenePoint(for vertex: MeshVertex,
                            pose: LipPose,
                            contour: LipContour,
                            mouthFrame: FaceLocalMouthFrame,
                            appliesExpressionCompensation: Bool = true) -> SCNVector3 {
        let cosine = cos(pose.angle)
        let sine = sin(pose.angle)
        let dx = Float(vertex.screen.x - pose.center.x)
        let dy = Float(vertex.screen.y - pose.center.y)
        var normalizedX = (dx * Float(cosine) + dy * Float(sine)) / Float(max(pose.width, 1))
        let rawNormalizedY = (-dx * Float(sine) + dy * Float(cosine)) / Float(max(pose.width, 1))
        let sideSmile = normalizedX < 0 ? mouthFrame.smileLeft : mouthFrame.smileRight
        let smileDriver = max(mouthFrame.smileExpansion, sideSmile)
        if appliesExpressionCompensation {
            let puckerCompression = max(mouthFrame.mouthPucker, mouthFrame.mouthFunnel * 0.72)
            let puckerFactor = 1 - min(max(puckerCompression, 0), 1) * 0.045
            normalizedX *= (1 + smileDriver * CanonicalLipGeometry.smileStretchFactor(for: vertex.index)) * puckerFactor
        }

        var normalizedY = rawNormalizedY
        if appliesExpressionCompensation {
            let mediaPipeOpening = Float(Self.mouthOpeningRatio(contour))
            let arOpeningTarget = mouthFrame.openingRatio >= 0.08 ? mouthFrame.openingRatio : 0
            let openingShortfall = max(arOpeningTarget - mediaPipeOpening, 0)
            normalizedY += openingShortfall * Float(
                CanonicalLipGeometry.openingMotionFactor(for: vertex.index)
            )
            normalizedY += mouthFrame.jawOpen * CanonicalLipGeometry.jawOpenDropFactor(for: vertex.index) * 0.020
            normalizedY += mouthFrame.lowerLipDrop * CanonicalLipGeometry.lowerLipDropFactor(for: vertex.index) * 0.014
        }
        let depthDelta = max(min(vertex.normalized.z - contour.depthCenter, 0.08), -0.08)
        let canonicalDepthDelta = max(
            min(contour.faceGeometryPose?.relativeCanonicalDepth(for: vertex.index) ?? 0, 0.08),
            -0.08
        )
        let geometryScale = contour.faceGeometryPose.map { min(max($0.scale / 8, 0.45), 1.65) } ?? 1
        // The fallback stays close to the AR mouth frame. Keep only a small,
        // bounded relief so it neither floats in front of the lips nor falls
        // behind the face depth occluder.
        let surfaceOffset = min(max(mouthFrame.width * 0.012, 0.000_45), 0.000_70)
        let requestedDepthOffset = surfaceOffset -
            depthDelta * mouthFrame.width * 0.07 * geometryScale -
            canonicalDepthDelta * mouthFrame.width * 0.03
        let depthOffset = min(max(requestedDepthOffset, 0.000_35), 0.000_80)
        let smileLift: Float
        if appliesExpressionCompensation {
            let openUpperCoverage = min(max((mouthFrame.jawOpen - 0.055) * 2.6, 0), 0.78)
            let raisedUpper = max(mouthFrame.upperLipRaise - 0.035, 0) * 0.9
            let upperDriver = max(raisedUpper, smileDriver * 1.05, openUpperCoverage)
            smileLift = CanonicalLipGeometry.upperLipSmileLiftFactor(for: vertex.index) *
                min(max(upperDriver, 0), 1) *
                mouthFrame.width * 0.027
        } else {
            smileLift = 0
        }

        let local = mouthFrame.center +
            mouthFrame.xAxis * (normalizedX * mouthFrame.width) +
            mouthFrame.downAxis * (normalizedY * mouthFrame.width - smileLift) +
            mouthFrame.normalAxis * depthOffset

        return SCNVector3(local.x, local.y, local.z)
    }

    private func screenAlignedScenePoint(for vertex: MeshVertex,
                                         pose: LipPose,
                                         contour: LipContour,
                                         mouthFrame: FaceLocalMouthFrame,
                                         renderer: SCNSceneRenderer,
                                         faceNode: SCNNode,
                                         contourAge: CFTimeInterval,
                                         motionDelta: CGFloat) -> SCNVector3 {
        // Keep the ARKit lip surface as the primary attachment. MediaPipe can
        // correct projection error while fresh, but it should not pull the mesh
        // away from the face during fast motion or delayed callbacks.
        let depthReference = scenePoint(
            for: vertex,
            pose: pose,
            contour: contour,
            mouthFrame: mouthFrame
        )
        return projectionAlignedScenePoint(
            depthReference: depthReference,
            targetScreenPoint: vertex.screen,
            mouthFrame: mouthFrame,
            renderer: renderer,
            faceNode: faceNode,
            contourAge: contourAge,
            motionDelta: motionDelta,
            maximumCorrectionFraction: 0.22
        )
    }

    private func projectionAlignedScenePoint(depthReference: SCNVector3,
                                             targetScreenPoint: CGPoint,
                                             mouthFrame: FaceLocalMouthFrame,
                                             renderer: SCNSceneRenderer,
                                             faceNode: SCNNode,
                                             contourAge: CFTimeInterval,
                                             motionDelta: CGFloat,
                                             maximumCorrectionFraction: Float) -> SCNVector3 {
        let correctionWeight = projectionCorrectionWeight(
            contourAge: contourAge,
            motionDelta: motionDelta
        )
        guard correctionWeight > 0 else {
            return depthReference
        }

        let worldDepthReference = faceNode.convertPosition(depthReference, to: nil)
        let projectedDepthReference = renderer.projectPoint(worldDepthReference)
        guard projectedDepthReference.z.isFinite,
              projectedDepthReference.z >= 0,
              projectedDepthReference.z <= 1 else {
            return depthReference
        }

        let worldTarget = renderer.unprojectPoint(
            SCNVector3(
                Float(targetScreenPoint.x),
                Float(targetScreenPoint.y),
                projectedDepthReference.z
            )
        )
        guard worldTarget.x.isFinite,
              worldTarget.y.isFinite,
              worldTarget.z.isFinite else {
            return depthReference
        }

        let localTarget = faceNode.convertPosition(worldTarget, from: nil)
        let reference = SIMD3<Float>(depthReference.x, depthReference.y, depthReference.z)
        let target = SIMD3<Float>(localTarget.x, localTarget.y, localTarget.z)
        let delta = target - reference
        // Unprojecting at a camera-space depth introduces a face-normal
        // component at yaw. Applying it would push one side behind the depth
        // occluder and make the other side float. Keep the screen correction in
        // the current mouth tangent plane so the surface lift remains bounded.
        let tangentDelta = mouthFrame.xAxis * simd_dot(delta, mouthFrame.xAxis) +
            mouthFrame.downAxis * simd_dot(delta, mouthFrame.downAxis)
        let tangentLength = simd_length(tangentDelta)
        guard tangentLength.isFinite, tangentLength > 0 else {
            return depthReference
        }

        let maxCorrection = mouthFrame.width * maximumCorrectionFraction
        let clampedDelta = tangentLength > maxCorrection ?
            tangentDelta / tangentLength * maxCorrection :
            tangentDelta
        let corrected = reference + clampedDelta * Float(correctionWeight)
        return SCNVector3(corrected.x, corrected.y, corrected.z)
    }

    private func projectionCorrectionWeight(contourAge: CFTimeInterval,
                                            motionDelta: CGFloat) -> CGFloat {
        guard contourAge.isFinite,
              contourAge <= 0.62 else {
            return 0
        }

        let ageFade = 1 - Self.smoothStep(edge0: 0.045, edge1: 0.30, value: CGFloat(contourAge))
        let finiteMotion = motionDelta.isFinite ? motionDelta : 1
        let motionFade = 1 - Self.smoothStep(edge0: 0.035, edge1: 0.22, value: finiteMotion)
        let freshSignal = max(0, min(ageFade * motionFade, 1))
        return max(0.04, min(1, 0.04 + freshSignal * 0.96))
    }

    private static func smoothStep(edge0: CGFloat, edge1: CGFloat, value: CGFloat) -> CGFloat {
        let t = max(0, min((value - edge0) / max(edge1 - edge0, 0.0001), 1))
        return t * t * (3 - 2 * t)
    }

    private static let lipPointCount = 20
}
