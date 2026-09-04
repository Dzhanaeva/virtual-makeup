import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

final class BlushRenderer {
    private let rootNode = SCNNode()
    private let leftNode = SCNNode()
    private let rightNode = SCNNode()
    private let leftMaterial = SCNMaterial()
    private let rightMaterial = SCNMaterial()
    private var lastColor: UIColor?
    private var lastOpacity: Double = -1
    private var cheekRegionVertexCount = 0
    private var leftCheekIndices: [Int] = []
    private var rightCheekIndices: [Int] = []

    init() {
        rootNode.name = "blush.root"
        rootNode.renderingOrder = 80
        leftNode.name = "blush.left"
        rightNode.name = "blush.right"
        rootNode.addChildNode(leftNode)
        rootNode.addChildNode(rightNode)

        configureMaterial(leftMaterial)
        configureMaterial(rightMaterial)
        leftNode.geometry = Self.makeBlushGeometry(material: leftMaterial)
        rightNode.geometry = Self.makeBlushGeometry(material: rightMaterial)
    }

    func attach(to faceNode: SCNNode) {
        if rootNode.parent !== faceNode {
            rootNode.removeFromParentNode()
            faceNode.addChildNode(rootNode)
        }
    }

    func updateStyle(color: UIColor, opacity: Double) {
        let clampedOpacity = max(0, min(opacity, 0.85))
        let shouldUpdateTexture = lastColor.map { !Self.sameColor($0, color) } ?? true
        if shouldUpdateTexture {
            let image = Self.makeBlushTexture(color: color)
            leftMaterial.diffuse.contents = image
            rightMaterial.diffuse.contents = image
            lastColor = color
        }

        guard abs(lastOpacity - clampedOpacity) > 0.003 else {
            return
        }
        lastOpacity = clampedOpacity
        leftMaterial.transparency = CGFloat(clampedOpacity)
        rightMaterial.transparency = CGFloat(clampedOpacity)
        rootNode.isHidden = clampedOpacity <= 0.01
    }

    func render(faceAnchor: ARFaceAnchor) {
        guard lastOpacity > 0.01 else {
            rootNode.isHidden = true
            return
        }

        let vertices = faceAnchor.geometry.vertices
        guard let bounds = Self.faceBounds(vertices: vertices) else {
            clear()
            return
        }

        guard let leftGeometry = Self.makeCheekMesh(
            side: -1,
            faceGeometry: faceAnchor.geometry,
            bounds: bounds,
            material: leftMaterial
        ),
              let rightGeometry = Self.makeCheekMesh(
                side: 1,
                faceGeometry: faceAnchor.geometry,
                bounds: bounds,
                material: rightMaterial
              ) else {
            clear()
            return
        }

        rootNode.isHidden = false
        leftNode.simdTransform = matrix_identity_float4x4
        rightNode.simdTransform = matrix_identity_float4x4
        leftNode.geometry = leftGeometry
        rightNode.geometry = rightGeometry
    }

    func clear() {
        rootNode.isHidden = true
    }

    private func configureMaterial(_ material: SCNMaterial) {
        material.lightingModel = .constant
        material.blendMode = .alpha
        material.isDoubleSided = true
        material.writesToDepthBuffer = false
        material.readsFromDepthBuffer = false
        material.diffuse.magnificationFilter = .linear
        material.diffuse.minificationFilter = .linear
        material.transparency = CGFloat(max(lastOpacity, 0))
    }

    private struct CheekPlacement {
        let center: SIMD3<Float>
        let xAxis: SIMD3<Float>
        let yAxis: SIMD3<Float>
        let normal: SIMD3<Float>
        let width: Float
        let height: Float
    }

    private struct FaceBounds {
        let min: SIMD3<Float>
        let max: SIMD3<Float>

        var center: SIMD3<Float> {
            (min + max) * 0.5
        }

        var width: Float {
            max.x - min.x
        }

        var height: Float {
            max.y - min.y
        }

        var depth: Float {
            max.z - min.z
        }
    }

    private func updateCheekRegionsIfNeeded(vertices: [SIMD3<Float>], bounds: FaceBounds) {
        guard cheekRegionVertexCount != vertices.count ||
                leftCheekIndices.isEmpty ||
                rightCheekIndices.isEmpty else {
            return
        }

        cheekRegionVertexCount = vertices.count
        leftCheekIndices = Self.cheekRegionIndices(side: -1, vertices: vertices, bounds: bounds)
        rightCheekIndices = Self.cheekRegionIndices(side: 1, vertices: vertices, bounds: bounds)
    }

    private static func faceBounds(vertices: [SIMD3<Float>]) -> FaceBounds? {
        guard var minPoint = vertices.first,
              var maxPoint = vertices.first else {
            return nil
        }

        for point in vertices.dropFirst() {
            minPoint = simd_min(minPoint, point)
            maxPoint = simd_max(maxPoint, point)
        }

        guard (maxPoint.x - minPoint.x) > 0.02,
              (maxPoint.y - minPoint.y) > 0.02,
              minPoint.x.isFinite,
              minPoint.y.isFinite,
              minPoint.z.isFinite,
              maxPoint.x.isFinite,
              maxPoint.y.isFinite,
              maxPoint.z.isFinite else {
            return nil
        }

        return FaceBounds(min: minPoint, max: maxPoint)
    }

    private static func cheekRegionIndices(side: Float,
                                           vertices: [SIMD3<Float>],
                                           bounds: FaceBounds) -> [Int] {
        var weighted: [(score: Float, index: Int)] = []
        weighted.reserveCapacity(96)

        for (index, point) in vertices.enumerated() {
            guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
                continue
            }

            let cheekX = normalized.x * side
            let cheekY = normalized.y
            let ovalScore = cheekOvalScore(cheekX: cheekX, cheekY: cheekY)
            guard ovalScore > 0 else {
                continue
            }

            weighted.append((ovalScore, index))
        }

        return weighted
            .sorted { $0.score > $1.score }
            .prefix(96)
            .map(\.index)
    }

    private static func cheekPlacement(side: Float,
                                       vertices: [SIMD3<Float>],
                                       indices: [Int],
                                       bounds: FaceBounds) -> CheekPlacement? {
        guard !indices.isEmpty else {
            return nil
        }

        var center = SIMD3<Float>(repeating: 0)
        var weightSum: Float = 0
        for index in indices where index < vertices.count {
            let point = vertices[index]
            guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
                continue
            }

            let weight = max(cheekOvalScore(cheekX: normalized.x * side, cheekY: normalized.y), 0.05)
            center += point * weight
            weightSum += weight
        }

        guard weightSum > 0 else {
            return nil
        }
        center /= max(weightSum, 0.0001)

        let outwardZ = max(bounds.depth * 0.32, 0.018)
        let normal = simd_normalize(SIMD3<Float>(
            side * bounds.width * 0.17,
            bounds.height * 0.02,
            outwardZ
        ))

        var xAxis = SIMD3<Float>(1, 0, 0) - normal * simd_dot(SIMD3<Float>(1, 0, 0), normal)
        if simd_length(xAxis) < 0.001 {
            xAxis = SIMD3<Float>(side, 0, 0)
        } else {
            xAxis = simd_normalize(xAxis)
        }

        var yAxis = SIMD3<Float>(0, 1, 0) - normal * simd_dot(SIMD3<Float>(0, 1, 0), normal)
        yAxis -= xAxis * simd_dot(yAxis, xAxis)
        if simd_length(yAxis) < 0.001 {
            yAxis = simd_normalize(simd_cross(normal, xAxis))
        } else {
            yAxis = simd_normalize(yAxis)
        }

        return CheekPlacement(
            center: center + normal * max(bounds.width * 0.014, 0.0012),
            xAxis: xAxis,
            yAxis: yAxis,
            normal: normal,
            width: max(bounds.width * 0.58, 0.082),
            height: max(bounds.width * 0.36, 0.054)
        )
    }

    private static func normalizedFacePoint(_ point: SIMD3<Float>, bounds: FaceBounds) -> SIMD3<Float>? {
        let width = max(bounds.width, 0.0001)
        let height = max(bounds.height, 0.0001)
        let depth = max(bounds.depth, 0.0001)
        let x = (point.x - bounds.center.x) / (width * 0.5)
        let y = (point.y - bounds.min.y) / height
        let z = (point.z - bounds.min.z) / depth
        guard x.isFinite, y.isFinite, z.isFinite else {
            return nil
        }
        return SIMD3<Float>(x, y, z)
    }

    private static func cheekOvalScore(cheekX: Float, cheekY: Float) -> Float {
        guard cheekX > 0.32,
              cheekX < 0.92,
              cheekY > 0.30,
              cheekY < 0.59 else {
            return 0
        }

        let dx = (cheekX - 0.60) / 0.30
        let dy = (cheekY - 0.45) / 0.15
        let distance = dx * dx + dy * dy
        guard distance < 1 else {
            return 0
        }
        let core = 1 - distance
        return core * core
    }

    private static func makeCheekMesh(side: Float,
                                      faceGeometry: ARFaceGeometry,
                                      bounds: FaceBounds,
                                      material: SCNMaterial) -> SCNGeometry? {
        let vertices = faceGeometry.vertices
        let triangleIndices = faceGeometry.triangleIndices
        guard !vertices.isEmpty,
              triangleIndices.count >= 3 else {
            return nil
        }

        var scnVertices: [SCNVector3] = []
        var textureCoordinates: [CGPoint] = []
        scnVertices.reserveCapacity(vertices.count)
        textureCoordinates.reserveCapacity(vertices.count)

        for point in vertices {
            scnVertices.append(SCNVector3(point.x, point.y, point.z))
            textureCoordinates.append(cheekTextureCoordinate(point, side: side, bounds: bounds))
        }

        var indices: [Int32] = []
        indices.reserveCapacity(triangleIndices.count)
        for triangleStart in stride(from: 0, to: triangleIndices.count - 2, by: 3) {
            let a = Int(triangleIndices[triangleStart])
            let b = Int(triangleIndices[triangleStart + 1])
            let c = Int(triangleIndices[triangleStart + 2])
            guard a >= 0, b >= 0, c >= 0,
                  a < vertices.count,
                  b < vertices.count,
                  c < vertices.count else {
                continue
            }

            let scoreA = cheekScore(vertices[a], side: side, bounds: bounds)
            let scoreB = cheekScore(vertices[b], side: side, bounds: bounds)
            let scoreC = cheekScore(vertices[c], side: side, bounds: bounds)
            guard min(scoreA, scoreB, scoreC) > 0.018 else {
                continue
            }
            indices.append(contentsOf: [Int32(a), Int32(b), Int32(c)])
        }

        guard indices.count >= 3 else {
            return nil
        }

        let vertexSource = SCNGeometrySource(vertices: scnVertices)
        let textureSource = SCNGeometrySource(textureCoordinates: textureCoordinates)
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        let geometry = SCNGeometry(sources: [vertexSource, textureSource], elements: [element])
        geometry.materials = [material]
        return geometry
    }

    private static func cheekScore(_ point: SIMD3<Float>, side: Float, bounds: FaceBounds) -> Float {
        guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
            return 0
        }
        return cheekOvalScore(cheekX: normalized.x * side, cheekY: normalized.y)
    }

    private static func cheekTextureCoordinate(_ point: SIMD3<Float>,
                                               side: Float,
                                               bounds: FaceBounds) -> CGPoint {
        guard let normalized = normalizedFacePoint(point, bounds: bounds) else {
            return CGPoint(x: 0.5, y: 0.5)
        }

        let cheekX = normalized.x * side
        let cheekY = normalized.y
        let u = 0.5 + CGFloat((cheekX - 0.56) / 0.39) * 0.5
        let v = 0.5 - CGFloat((cheekY - 0.44) / 0.20) * 0.5
        return CGPoint(x: max(0, min(1, u)), y: max(0, min(1, v)))
    }

    private static func makeBlushGeometry(material: SCNMaterial) -> SCNGeometry {
        let columns = 6
        let rows = 5
        var vertices: [SCNVector3] = []
        var textureCoordinates: [CGPoint] = []
        var indices: [Int32] = []

        for row in 0..<rows {
            let v = Float(row) / Float(rows - 1)
            let y = v - 0.5
            for column in 0..<columns {
                let u = Float(column) / Float(columns - 1)
                let x = u - 0.5
                let radial = min(1, (x * x) / 0.25 + (y * y) / 0.18)
                let z = (1 - radial) * 0.0014
                vertices.append(SCNVector3(x, y, z))
                textureCoordinates.append(CGPoint(x: CGFloat(u), y: CGFloat(1 - v)))
            }
        }

        for row in 0..<(rows - 1) {
            for column in 0..<(columns - 1) {
                let a = Int32(row * columns + column)
                let b = Int32(row * columns + column + 1)
                let c = Int32((row + 1) * columns + column)
                let d = Int32((row + 1) * columns + column + 1)
                indices.append(contentsOf: [a, c, b, b, c, d])
            }
        }

        let vertexSource = SCNGeometrySource(vertices: vertices)
        let textureSource = SCNGeometrySource(textureCoordinates: textureCoordinates)
        let indexData = Data(bytes: indices, count: indices.count * MemoryLayout<Int32>.size)
        let element = SCNGeometryElement(
            data: indexData,
            primitiveType: .triangles,
            primitiveCount: indices.count / 3,
            bytesPerIndex: MemoryLayout<Int32>.size
        )
        let geometry = SCNGeometry(sources: [vertexSource, textureSource], elements: [element])
        geometry.materials = [material]
        return geometry
    }

    private static func sameColor(_ first: UIColor, _ second: UIColor) -> Bool {
        var firstRed: CGFloat = 0
        var firstGreen: CGFloat = 0
        var firstBlue: CGFloat = 0
        var firstAlpha: CGFloat = 0
        var secondRed: CGFloat = 0
        var secondGreen: CGFloat = 0
        var secondBlue: CGFloat = 0
        var secondAlpha: CGFloat = 0
        first.getRed(&firstRed, green: &firstGreen, blue: &firstBlue, alpha: &firstAlpha)
        second.getRed(&secondRed, green: &secondGreen, blue: &secondBlue, alpha: &secondAlpha)
        return abs(firstRed - secondRed) < 0.002 &&
            abs(firstGreen - secondGreen) < 0.002 &&
            abs(firstBlue - secondBlue) < 0.002 &&
            abs(firstAlpha - secondAlpha) < 0.002
    }

    private static func makeBlushTexture(color: UIColor) -> UIImage {
        let size = CGSize(width: 256, height: 180)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        return UIGraphicsImageRenderer(size: size, format: format).image { context in
            let rect = CGRect(origin: .zero, size: size)
            guard let cgContext = UIGraphicsGetCurrentContext() else {
                return
            }

            var red: CGFloat = 0
            var green: CGFloat = 0
            var blue: CGFloat = 0
            var alpha: CGFloat = 0
            color.getRed(&red, green: &green, blue: &blue, alpha: &alpha)
            let center = CGPoint(x: rect.midX, y: rect.midY)
            let colors = [
                UIColor(red: red, green: green, blue: blue, alpha: 0.58).cgColor,
                UIColor(red: red, green: green, blue: blue, alpha: 0.34).cgColor,
                UIColor(red: red, green: green, blue: blue, alpha: 0.13).cgColor,
                UIColor(red: red, green: green, blue: blue, alpha: 0.0).cgColor
            ] as CFArray
            let locations: [CGFloat] = [0.0, 0.48, 0.82, 1.0]
            guard let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: colors,
                locations: locations
            ) else {
                return
            }

            cgContext.saveGState()
            cgContext.translateBy(x: center.x, y: center.y)
            cgContext.scaleBy(x: 1.0, y: 0.62)
            cgContext.translateBy(x: -center.x, y: -center.y)
            cgContext.drawRadialGradient(
                gradient,
                startCenter: center,
                startRadius: 0,
                endCenter: center,
                endRadius: size.width * 0.56,
                options: [.drawsAfterEndLocation]
            )
            cgContext.restoreGState()

            UIColor.white.withAlphaComponent(0.06).setFill()
            context.cgContext.fillEllipse(in: CGRect(x: 82, y: 54, width: 74, height: 38))
        }
    }
}
