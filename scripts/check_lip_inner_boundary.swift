import Foundation
import Metal

@main
enum LipInnerBoundaryCheck {
    static func main() throws {
        func require(_ value: Bool, _ message: String) {
            precondition(value, message)
        }
        func coverage(_ distance: CGFloat, opening: CGFloat) -> CGFloat {
            LipInnerBoundaryLayout.coverage(
                signedDistance: distance /
                    LipInnerBoundaryLayout.referenceTextureHeight,
                opening: opening
            )
        }
        func shade(_ distance: CGFloat, opening: CGFloat, thickness: CGFloat) -> CGFloat {
            LipInnerBoundaryLayout.recessShade(
                signedDistance: distance /
                    LipInnerBoundaryLayout.referenceTextureHeight,
                lipThickness: thickness /
                    LipInnerBoundaryLayout.referenceTextureHeight,
                opening: opening
            )
        }

        require(coverage(-10, opening: 1) == 1, "Lip body must keep full coverage")
        require(coverage(-0.01, opening: 1) == 1, "Visible inner lip must remain painted")
        require(coverage(0, opening: 1) == 1, "Wet line must not be underpainted")
        let openSolid = LipInnerBoundaryLayout.mouthSolidPixels
        let openEnd = openSolid + LipInnerBoundaryLayout.mouthTransitionPixels
        require(coverage(openSolid * 0.5, opening: 1) == 1,
                "Visible wet line must stay painted")
        require(coverage(openSolid + 0.1, opening: 1) < 1,
                "Coverage must fade after the wet line")
        require(coverage((openSolid + openEnd) * 0.5, opening: 1) > 0,
                "Open mouth transition must be gradual")
        require(coverage(openEnd + 0.05, opening: 1) == 0,
                "Open mouth must remain clear")
        require(coverage(openEnd + 0.05, opening: 0.25) == 0,
                "A visible opening must not retain closed-seam pigment")
        for distance in [CGFloat(1.5), 4, 12, 20] {
            require(coverage(distance, opening: 0) == 1,
                    "Closed inner polygon must remain filled")
        }
        require(LipMouthClosure.isClosed(
            arOpening: 0.012, jawOpen: 0.03, smile: 0,
            detectedOpening: 0.08, wasClosed: nil
        ), "A detector gap must not open a relaxed mouth")
        require(LipMouthClosure.isClosed(
            arOpening: 0.022, jawOpen: 0.06, smile: 0,
            detectedOpening: 0.08, wasClosed: true
        ), "Closed state must survive small ARKit movement")
        require(!LipMouthClosure.isClosed(
            arOpening: 0.025, jawOpen: 0.10, smile: 0,
            detectedOpening: 0.07, wasClosed: true
        ), "Corroborated opening must reveal the mouth immediately")
        require(!LipMouthClosure.isClosed(
            arOpening: 0.024, jawOpen: 0.06, smile: 0,
            detectedOpening: 0.04, wasClosed: false
        ), "Open state must survive small ARKit movement")
        require(LipMouthClosure.isClosed(
            arOpening: 0.012, jawOpen: 0.03, smile: 0,
            detectedOpening: 0.08, wasClosed: false
        ), "A relaxed mouth must close again")
        require(shade(-3, opening: 1, thickness: 24) == 1,
                "Recess shade must not darken the lip body")
        require(shade(-0.2, opening: 1, thickness: 24) < shade(-1, opening: 1, thickness: 24),
                "Recess shading must stay on the visible lip")
        require(shade(0, opening: 0, thickness: 24) > shade(0, opening: 1, thickness: 24),
                "Closed mouth must not gain a dark stripe")

        var cases: [SIMD4<Float>] = []
        var expected: [SIMD2<Float>] = []
        for opening in [CGFloat(0), 0.10, 0.25, 0.5, 0.75, 1] {
            for thickness in [CGFloat(2), 8, 24] {
                var previous: CGFloat = 1
                for step in -80...160 {
                    let distance = CGFloat(step) / 8
                    let alpha = coverage(distance, opening: opening)
                    require(alpha.isFinite && (0...1).contains(alpha), "Invalid coverage")
                    require(alpha <= previous + 0.000001, "Coverage reverses toward the mouth")
                    previous = alpha
                    for scale: Float in [0.75, 1, 2] {
                        cases.append(SIMD4(Float(distance), Float(thickness), Float(opening), scale))
                        expected.append(SIMD2(
                            Float(alpha),
                            Float(shade(distance, opening: opening, thickness: thickness))
                        ))
                    }
                }
            }
        }

        let source = try String(contentsOfFile: "Virtual Makeup/MetalLipColorCompositor.swift", encoding: .utf8)
        guard let start = source.range(of: "private static let shaderSource = #\"\"\""),
              let end = source.range(of: "\"\"\"#", range: start.upperBound..<source.endIndex) else {
            throw NSError(domain: "LipInnerBoundaryCheck", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Production shader source unavailable"])
        }
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else {
            print("PASS: painted inner edge, recessed lip colour, open-mouth cutout, full closed inner fill; Metal GPU unavailable on this host")
            return
        }
        let shader = String(source[start.upperBound..<end.lowerBound]) + """

        kernel void lip_profile_check(
            const device float4 *cases [[buffer(0)]],
            device float2 *results [[buffer(1)]],
            uint index [[thread_position_in_grid]]) {
            float4 item = cases[index];
            LipFeatherUniforms profile = {};
            profile.innerMouthSolidPixels = \(LipInnerBoundaryLayout.mouthSolidPixels) * item.w;
            profile.innerMouthTransitionPixels = \(LipInnerBoundaryLayout.mouthTransitionPixels) * item.w;
            profile.recessLipPixels = \(LipInnerBoundaryLayout.recessLipPixels) * item.w;
            profile.recessLipFraction = \(LipInnerBoundaryLayout.recessLipFraction);
            profile.openRecessDarkening = \(LipInnerBoundaryLayout.openRecessDarkening);
            profile.closedRecessDarkening = \(LipInnerBoundaryLayout.closedRecessDarkening);
            profile.apertureClosure = 1.0 - item.z;
            results[index] = float2(
                lip_inner_coverage(item.x * item.w, profile),
                lip_inner_recess_shade(item.x * item.w, item.y * item.w, profile)
            );
        }
        """
        let library = try device.makeLibrary(source: shader, options: nil)
        guard let function = library.makeFunction(name: "lip_profile_check"),
              let compositor = library.makeFunction(name: "lip_composite"),
              let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder(),
              let inputs = cases.withUnsafeBytes({ bytes in
                  bytes.baseAddress.flatMap {
                      device.makeBuffer(bytes: $0, length: bytes.count, options: .storageModeShared)
                  }
              }),
              let outputs = device.makeBuffer(length: expected.count * MemoryLayout<SIMD2<Float>>.stride,
                                              options: .storageModeShared) else {
            throw NSError(domain: "LipInnerBoundaryCheck", code: 2)
        }
        _ = try device.makeComputePipelineState(function: compositor)
        let pipeline = try device.makeComputePipelineState(function: function)
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(inputs, offset: 0, index: 0)
        encoder.setBuffer(outputs, offset: 0, index: 1)
        encoder.dispatchThreads(MTLSize(width: expected.count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: pipeline.threadExecutionWidth,
                                                               height: 1, depth: 1))
        encoder.endEncoding()
        commandBuffer.commit()
        // Blocking is confined to this offline check, never the app compositor.
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error { throw error }
        require(commandBuffer.status == .completed, "GPU check did not complete")
        let actual = outputs.contents().bindMemory(to: SIMD2<Float>.self, capacity: expected.count)
        var maximumError: Float = 0
        for index in expected.indices {
            require(actual[index].x.isFinite && actual[index].y.isFinite,
                    "Metal produced non-finite inner profile")
            maximumError = max(
                maximumError,
                abs(actual[index].x - expected[index].x),
                abs(actual[index].y - expected[index].y)
            )
        }
        require(maximumError < 0.00002, "CPU/Metal inner profile mismatch: \(maximumError)")
        print("PASS: painted inner edge, recessed lip colour, open-mouth cutout, full closed inner fill; \(expected.count) CPU/Metal comparisons, max error \(maximumError)")
    }
}
