import ARKit
import CoreImage
import ImageIO
import MediaPipeTasksVision
import Metal
import SceneKit
import simd
import SwiftUI
import UIKit

struct LipTexture {
    let image: UIImage
    let opacityImage: UIImage

    init(image: UIImage, opacityImage: UIImage? = nil) {
        self.image = image
        self.opacityImage = opacityImage ?? image
    }
}

struct RGBColor {
    let red: Float
    let green: Float
    let blue: Float
}

final class MetalLipColorCompositor {
    private struct Vertex {
        let canonicalUV: SIMD2<Float>
        let sourceUV: SIMD2<Float>
    }

    private struct Uniforms {
        var lipstickColor: SIMD4<Float>
        var sourceTexelSize: SIMD2<Float>
        var lightingFactor: Float
        var detailStrength: Float
        var coreCoverage: Float
        var highlightStrength: Float
        var highlightMaximum: Float
        var highlightConcentration: Float
        var finish: Float
        var apertureClosure: Float
    }

    private struct FinishParameters {
        let detailStrength: Float
        let highlightStrength: Float
        let highlightMaximum: Float
    }

    private struct FeatherUniforms {
        var textureSize: SIMD2<Float>
        var outerPointCount: UInt32
        var innerPointCount: UInt32
        var outerFeatherPixels: Float
        var outerInnerFeatherPixels: Float
        var apertureFeatherPixels: Float
        var apertureMouthFeatherPixels: Float
        var apertureInsetPixels: Float
        var apertureClosure: Float
    }

    private struct RenderTargets {
        let width: Int
        let height: Int
        let color: MTLTexture
        let output: MTLTexture
        let horizontalBlur: MTLTexture
        let softenedOutput: MTLTexture
        let blurGate: MTLTexture
    }

    private final class Context {
        let device: MTLDevice
        let commandQueue: MTLCommandQueue
        let textureCache: CVMetalTextureCache
        let renderPipeline: MTLRenderPipelineState
        let compositePipeline: MTLComputePipelineState
        let alphaBlurHorizontalPipeline: MTLComputePipelineState
        let alphaBlurVerticalPipeline: MTLComputePipelineState
        let indexBuffer: MTLBuffer
        let indexCount: Int
        var renderTargets: RenderTargets?

        init(device: MTLDevice,
             commandQueue: MTLCommandQueue,
             textureCache: CVMetalTextureCache,
             renderPipeline: MTLRenderPipelineState,
             compositePipeline: MTLComputePipelineState,
             alphaBlurHorizontalPipeline: MTLComputePipelineState,
             alphaBlurVerticalPipeline: MTLComputePipelineState,
             indexBuffer: MTLBuffer,
             indexCount: Int) {
            self.device = device
            self.commandQueue = commandQueue
            self.textureCache = textureCache
            self.renderPipeline = renderPipeline
            self.compositePipeline = compositePipeline
            self.alphaBlurHorizontalPipeline = alphaBlurHorizontalPipeline
            self.alphaBlurVerticalPipeline = alphaBlurVerticalPipeline
            self.indexBuffer = indexBuffer
            self.indexCount = indexCount
        }
    }

    private static let shaderSource = #"""
    #include <metal_stdlib>
    using namespace metal;

    struct LipVertex {
        float2 canonicalUV;
        float2 sourceUV;
    };

    struct LipUniforms {
        float4 lipstickColor;
        float2 sourceTexelSize;
        float lightingFactor;
        float detailStrength;
        float coreCoverage;
        float highlightStrength;
        float highlightMaximum;
        float highlightConcentration;
        float finish;
        float apertureClosure;
    };

    struct LipRaster {
        float4 position [[position]];
        float2 sourceUV;
        float2 canonicalUV;
    };

    struct LipFeatherUniforms {
        float2 textureSize;
        uint outerPointCount;
        uint innerPointCount;
        float outerFeatherPixels;
        float outerInnerFeatherPixels;
        float apertureFeatherPixels;
        float apertureMouthFeatherPixels;
        float apertureInsetPixels;
        float apertureClosure;
    };

    vertex LipRaster lip_vertex(const device LipVertex *vertices [[buffer(0)]],
                                uint vertexID [[vertex_id]]) {
        LipVertex inputVertex = vertices[vertexID];
        LipRaster output;
        // Canonical v=1 is the top row of the generated CGImage.
        output.position = float4(
            inputVertex.canonicalUV.x * 2.0 - 1.0,
            inputVertex.canonicalUV.y * 2.0 - 1.0,
            0.0,
            1.0
        );
        output.sourceUV = inputVertex.sourceUV;
        output.canonicalUV = inputVertex.canonicalUV;
        return output;
    }

    float lip_luminance(float3 color) {
        return clamp(dot(color, float3(0.299, 0.587, 0.114)), 0.0, 1.0);
    }

    float lip_saturation(float3 color) {
        float maximum = max(color.r, max(color.g, color.b));
        float minimum = min(color.r, min(color.g, color.b));
        return maximum > 0.001 ? (maximum - minimum) / maximum : 0.0;
    }

    float lip_tooth_probability(float3 color) {
        float luminance = lip_luminance(color);
        float saturation = lip_saturation(color);
        float bright = smoothstep(0.64, 0.82, luminance);
        float lowSaturation = 1.0 - smoothstep(0.16, 0.30, saturation);
        return bright * lowSaturation;
    }

    float3 lip_color_with_luminance(float3 color, float targetLuminance) {
        float sourceLuminance = max(lip_luminance(color), 0.0001);
        float3 adjusted = color * (targetLuminance / sourceLuminance);
        float maximum = max(adjusted.r, max(adjusted.g, adjusted.b));
        if (maximum > 1.0) {
            adjusted /= maximum;
        }
        return saturate(adjusted);
    }

    float lip_irregular_glint(float2 local, float falloff, float phase) {
        float distanceSquared = dot(local, local);
        float edgeWeight = smoothstep(0.16, 0.92, distanceSquared);
        float edgeVariation =
            sin(local.x * 9.7 + local.y * 6.1 + phase) * 0.055 +
            sin(local.x * 17.3 - local.y * 11.9 + phase * 1.7) * 0.030;
        return pow(
            saturate(1.0 - distanceSquared + edgeVariation * edgeWeight),
            falloff
        );
    }

    float2 lip_rotated_local(float2 point,
                             float2 center,
                             float2 radii,
                             float angle) {
        float2 delta = point - center;
        float cosine = cos(angle);
        float sine = sin(angle);
        float2 aligned = float2(
            delta.x * cosine + delta.y * sine,
            -delta.x * sine + delta.y * cosine
        );
        return aligned / radii;
    }

    fragment float4 lip_fragment(
        LipRaster input [[stage_in]],
        constant LipUniforms &uniforms [[buffer(0)]],
        texture2d<float> cameraTexture [[texture(0)]]) {
        constexpr sampler cameraSampler(
            coord::normalized,
            address::clamp_to_edge,
            filter::linear
        );

        float2 radius = uniforms.sourceTexelSize * 5.0;
        float3 base = cameraTexture.sample(cameraSampler, input.sourceUV).rgb;
        float3 blurred = base * 4.0;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(radius.x, 0.0)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV - float2(radius.x, 0.0)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(0.0, radius.y)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV - float2(0.0, radius.y)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + radius).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV - radius).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(radius.x, -radius.y)).rgb;
        blurred += cameraTexture.sample(cameraSampler, input.sourceUV + float2(-radius.x, radius.y)).rgb;
        blurred /= 12.0;

        // Use the camera's real reflection pattern as the satin highlight.
        // A wider cross-shaped neighbourhood separates an existing bright
        // spot from the surrounding lip tone without inventing its position.
        float2 highlightRadius = uniforms.sourceTexelSize * float2(12.0, 8.0);
        float3 highlightSurround =
            cameraTexture.sample(
                cameraSampler,
                input.sourceUV + float2(highlightRadius.x, 0.0)
            ).rgb;
        highlightSurround += cameraTexture.sample(
            cameraSampler,
            input.sourceUV - float2(highlightRadius.x, 0.0)
        ).rgb;
        highlightSurround += cameraTexture.sample(
            cameraSampler,
            input.sourceUV + float2(0.0, highlightRadius.y)
        ).rgb;
        highlightSurround += cameraTexture.sample(
            cameraSampler,
            input.sourceUV - float2(0.0, highlightRadius.y)
        ).rgb;
        highlightSurround *= 0.25;

        float baseLuminance = lip_luminance(base);
        float blurredLuminance = max(lip_luminance(blurred), 0.055);
        float localLighting = smoothstep(0.10, 0.55, blurredLuminance);
        float combinedLighting = clamp(
            mix(uniforms.lightingFactor, localLighting, 0.25),
            0.45,
            1.0
        );
        // SceneKit applies the single, deliberately shallow low-light response
        // to both Metal and CPU textures. Keep the generated pigment at the
        // catalogue RGB here so lighting cannot be applied twice.
        float3 saturatedPigment = uniforms.lipstickColor.rgb;
        float logDetail = log2(max(baseLuminance, 0.04)) -
            log2(max(blurredLuminance, 0.04));
        float brightScene = smoothstep(0.78, 1.0, combinedLighting);
        float textureY = 1.0 - input.canonicalUV.y;
        float cornerPosition = clamp(abs(input.canonicalUV.x - 0.5) * 2.0, 0.0, 1.0);
        float cornerRegion = smoothstep(0.62, 0.96, cornerPosition);
        float innerSeamRegion = 1.0 - smoothstep(0.018, 0.085, abs(textureY - 0.50));
        float localShadow = 1.0 - smoothstep(0.16, 0.42, blurredLuminance);
        float apertureClosure = saturate(uniforms.apertureClosure);
        float openMouth = 1.0 - apertureClosure;
        bool isSatin = uniforms.finish > 1.5;
        bool isGloss = uniforms.finish > 0.5 && uniforms.finish < 1.5;
        float lowDensityGloss = isGloss ?
            1.0 - smoothstep(0.60, 0.72, uniforms.coreCoverage) :
            0.0;
        float detailExponent;
        if (uniforms.finish < 0.5) {
            // Matte pigment is diffuse, but not flat. Preserve camera-derived
            // creases more strongly than highlights so the lip volume remains
            // visible without introducing a wet/specular appearance.
            float shadowDetail = min(logDetail, 0.0);
            float highlightDetail = max(logDetail, 0.0);
            float matteShadowStrength = mix(1.15, 0.72, brightScene);
            float matteHighlightStrength = mix(0.65, 0.35, brightScene);
            detailExponent =
                clamp(shadowDetail, -0.14, 0.0) *
                    uniforms.detailStrength * matteShadowStrength +
                clamp(highlightDetail, 0.0, 0.09) *
                    uniforms.detailStrength * matteHighlightStrength;
        } else {
            float adaptiveDetailStrength = uniforms.detailStrength *
                mix(1.0, 0.45, brightScene);
            float shadowDetail = min(logDetail, 0.0);
            float highlightDetail = max(logDetail, 0.0);
            detailExponent =
                clamp(shadowDetail, -0.14, 0.0) *
                    adaptiveDetailStrength * (isSatin ? 1.30 : 1.14) +
                clamp(highlightDetail, 0.0, 0.10) *
                    adaptiveDetailStrength * (isSatin ? 0.92 : 0.96);
        }
        float detail = exp2(detailExponent);
        float3 pigment = saturatedPigment;
        float seamShadow = 1.0 - innerSeamRegion *
            smoothstep(0.30, 0.56, openMouth) *
            (0.020 + localShadow * 0.08);
        pigment = saturate(pigment * detail * seamShadow);

        // A low-density gloss is a tinted transparent medium: mix its RGB
        // with the captured lip colour first, then render that result with a
        // confident mask. This is materially different from merely lowering
        // the opacity of an otherwise opaque lipstick colour.
        float3 naturalLipColor = saturate(
            blurred + (base - blurred) * 0.35
        );
        pigment = mix(
            pigment,
            naturalLipColor,
            lowDensityGloss * 0.6
        );

        float toothGuard = lip_tooth_probability(base) *
            smoothstep(0.24, 0.46, openMouth) *
            innerSeamRegion;

        if (uniforms.highlightStrength > 0.001) {
            // Both finishes inherit the reflection shape from the camera.
            // Gloss extracts only the strongest core of that real reflection
            // and amplifies it, instead of drawing a fixed highlight in UV.
            float surroundLuminance = max(
                lip_luminance(highlightSurround),
                0.04
            );
            float highlightLogContrast = log2(
                max(baseLuminance, 0.04) / surroundLuminance
            );
            float relativeHighlight = smoothstep(
                isGloss ? 0.006 : 0.035,
                isGloss ? 0.11 : 0.22,
                highlightLogContrast
            );
            float highlightBrightness = smoothstep(
                isGloss ? 0.07 : 0.18,
                isGloss ? 0.48 : 0.68,
                baseLuminance
            );
            float highlightDifference = max(
                baseLuminance - surroundLuminance,
                0.0
            );
            float realHighlightSignal = saturate(
                relativeHighlight * highlightBrightness
            );
            float liquidResponse = smoothstep(
                1.20,
                1.80,
                uniforms.highlightConcentration
            );
            float naturalHighlight;
            if (isGloss) {
                // Keep the exact silhouette and position of the reflection
                // captured by the camera. Liquid may concentrate it a little
                // more, but no fixed UV-space highlight is introduced.
                float highlightCore = smoothstep(
                    mix(0.08, 0.16, liquidResponse),
                    mix(0.50, 0.62, liquidResponse),
                    realHighlightSignal
                );
                naturalHighlight = pow(
                    highlightCore,
                    max(uniforms.highlightConcentration, 1.0)
                );
            } else {
                naturalHighlight = pow(
                    realHighlightSignal,
                    max(uniforms.highlightConcentration, 1.0)
                );
            }
            float highlightAmount;
            if (isGloss) {
                float glossPeakLimit = uniforms.highlightMaximum;
                highlightAmount = min(
                    (pow(realHighlightSignal, 1.35) * 0.035 +
                        naturalHighlight * 0.14) *
                        uniforms.highlightStrength,
                    glossPeakLimit
                );
            } else {
                highlightAmount = min(
                    naturalHighlight *
                        (0.07 + highlightDifference * 0.90) *
                        uniforms.highlightStrength,
                    uniforms.highlightMaximum
                );
            }
            highlightAmount *=
                (1.0 - cornerRegion * 0.82) *
                (1.0 - innerSeamRegion *
                    (isGloss ?
                        mix(0.30, 0.58, apertureClosure) :
                        mix(0.74, 0.92, apertureClosure))) *
                (1.0 - toothGuard * 0.96);
            pigment = mix(pigment, float3(1.0), highlightAmount);
        }

        pigment = mix(
            pigment,
            base * (1.0 - localShadow * 0.05),
            toothGuard * 0.94
        );

        // Keep only a real dark fold at the very tip of each mouth corner.
        // The fold changes luminance while retaining the selected lipstick
        // hue, and activates only when the camera actually sees a darker area.
        float cornerTip = smoothstep(0.86, 1.0, cornerPosition);
        float finalPigmentLuminance = lip_luminance(pigment);
        float capturedCornerLuminance = min(baseLuminance, blurredLuminance);
        float cornerShadowEvidence = smoothstep(
            0.025,
            0.18,
            finalPigmentLuminance - capturedCornerLuminance
        );
        float retainedCornerLuminance = max(
            capturedCornerLuminance,
            finalPigmentLuminance * 0.58
        );
        float cornerTargetLuminance = mix(
            finalPigmentLuminance,
            retainedCornerLuminance,
            cornerTip * cornerShadowEvidence * 0.76
        );
        pigment = lip_color_with_luminance(pigment, cornerTargetLuminance);

        // Keep the reduction shallow and confined to the final part of each
        // corner. It is applied after colour compensation below, so it reveals
        // a little of the real lip instead of recalculating a denser corrective
        // colour or creating the broad pale halo of the previous corner fade.
        float cornerOpacity = 1.0 -
            smoothstep(0.82, 0.98, cornerPosition) * 0.12;
        // Restore most of the catalogue saturation without making translucent
        // products look like an opaque sticker. Lower-density formulas retain
        // progressively more of the natural lip colour.
        float outputCoverage = mix(
            uniforms.coreCoverage,
            0.45,
            lowDensityGloss
        );
        float compensationAlpha = mix(
            uniforms.coreCoverage,
            0.92,
            0.55
        );
        compensationAlpha = mix(
            compensationAlpha,
            outputCoverage,
            lowDensityGloss
        );
        // Compensate only the low-frequency lip colour. SceneKit then restores
        // the difference between the real pixel and this local average, so
        // folds and fine texture pass through without shifting the shade's
        // overall brightness toward the natural lip colour.
        float3 correctiveColor = saturate(
            (pigment - blurred * (1.0 - compensationAlpha)) /
                max(compensationAlpha, 0.001)
        );
        return float4(
            correctiveColor,
            outputCoverage * cornerOpacity
        );
    }

    float lip_distance_to_segment(float2 point, float2 first, float2 second) {
        float2 segment = second - first;
        float denominator = max(dot(segment, segment), 0.0001);
        float position = clamp(dot(point - first, segment) / denominator, 0.0, 1.0);
        return distance(point, first + segment * position);
    }

    kernel void lip_composite(
        texture2d<float, access::read> colorTexture [[texture(0)]],
        texture2d<float, access::write> outputTexture [[texture(1)]],
        texture2d<float, access::write> blurGateTexture [[texture(2)]],
        const device float2 *outerPoints [[buffer(0)]],
        constant LipFeatherUniforms &uniforms [[buffer(1)]],
        const device float2 *innerPoints [[buffer(2)]],
        uint2 position [[thread_position_in_grid]]) {
        if (position.x >= outputTexture.get_width() ||
            position.y >= outputTexture.get_height()) {
            return;
        }
        float4 color = colorTexture.read(position);
        if (uniforms.outerPointCount < 3 ||
            uniforms.innerPointCount < 3) {
            outputTexture.write(float4(0.0), position);
            blurGateTexture.write(float4(0.0), position);
            return;
        }

        float2 pixelPoint = float2(position) + 0.5;
        float minimumDistance = 100000.0;
        bool isInside = false;
        uint previous = uniforms.outerPointCount - 1;
        for (uint index = 0; index < uniforms.outerPointCount; ++index) {
            float2 first = outerPoints[previous];
            float2 second = outerPoints[index];
            minimumDistance = min(
                minimumDistance,
                lip_distance_to_segment(pixelPoint, first, second)
            );

            bool crosses = (first.y > pixelPoint.y) != (second.y > pixelPoint.y);
            if (crosses) {
                float denominator = second.y - first.y;
                float intersectionX = first.x +
                    (pixelPoint.y - first.y) *
                    (second.x - first.x) /
                    (abs(denominator) < 0.0001 ? 0.0001 : denominator);
                if (pixelPoint.x < intersectionX) {
                    isInside = !isInside;
                }
            }
            previous = index;
        }

        float signedDistance = isInside ? -minimumDistance : minimumDistance;
        float outerFeatherDistance = max(
            uniforms.outerFeatherPixels,
            0.001
        );
        float innerFeatherDistance = max(
            uniforms.outerInnerFeatherPixels,
            0.001
        );

        // A truly soft raster edge has partial coverage on the contour itself.
        // Confine that change to a tiny inner antialiasing band, then dissolve
        // the same pigment across the exterior carrier. The two smoothstep
        // curves meet at the same coverage with zero slope, avoiding a visible
        // vector-like seam while leaving the rest of the lip untouched.
        const float contourCoverage = 0.68;
        float innerFeatherPosition = saturate(
            (signedDistance + innerFeatherDistance) /
                innerFeatherDistance
        );
        float innerEdgeCoverage = mix(
            1.0,
            contourCoverage,
            smoothstep(0.0, 1.0, innerFeatherPosition)
        );
        float outerEdgeCoverage = contourCoverage *
            (1.0 - smoothstep(
                0.0,
                outerFeatherDistance,
                max(signedDistance, 0.0)
            ));
        float outerFeather = isInside ?
            innerEdgeCoverage :
            outerEdgeCoverage;
        if (signedDistance >= outerFeatherDistance) {
            outerFeather = 0.0;
        }

        float innerMinimumDistance = 100000.0;
        bool isInsideAperture = false;
        uint innerPrevious = uniforms.innerPointCount - 1;
        for (uint index = 0; index < uniforms.innerPointCount; ++index) {
            float2 first = innerPoints[innerPrevious];
            float2 second = innerPoints[index];
            innerMinimumDistance = min(
                innerMinimumDistance,
                lip_distance_to_segment(pixelPoint, first, second)
            );

            bool crosses = (first.y > pixelPoint.y) != (second.y > pixelPoint.y);
            if (crosses) {
                float denominator = second.y - first.y;
                float intersectionX = first.x +
                    (pixelPoint.y - first.y) *
                    (second.x - first.x) /
                    (abs(denominator) < 0.0001 ? 0.0001 : denominator);
                if (pixelPoint.x < intersectionX) {
                    isInsideAperture = !isInsideAperture;
                }
            }
            innerPrevious = index;
        }

        // The lip side of the inner contour must remain fully opaque. The old
        // symmetric smoothstep put its low-alpha midpoint directly on this
        // boundary, so texture filtering stretched it into a pale horizontal
        // seam under head pitch. Put the entire transition inside the real
        // aperture instead. A narrow aperture receives enough solid inward
        // coverage for both sides to meet; as it opens, the solid reach shrinks
        // and leaves a clean transparent centre for teeth and the mouth.
        float apertureFeather = 1.0;
        if (isInsideAperture) {
            float solidInnerCoveragePixels = max(
                uniforms.apertureFeatherPixels,
                0.0
            );
            float innerTransitionPixels = max(
                uniforms.apertureMouthFeatherPixels,
                0.001
            );
            apertureFeather = 1.0 - smoothstep(
                solidInnerCoveragePixels,
                solidInnerCoveragePixels + innerTransitionPixels,
                innerMinimumDistance
            );
        }

        float maskCoverage = outerFeather * apertureFeather;
        float recoverableInnerFeatherPixels =
            max(uniforms.apertureFeatherPixels, 0.0) +
            max(uniforms.apertureMouthFeatherPixels, 0.0);
        bool canRecoverInsideApertureFeather =
            isInsideAperture &&
            innerMinimumDistance <= recoverableInnerFeatherPixels;
        if (color.a <= 0.0 &&
            maskCoverage > 0.001 &&
            (!isInsideAperture || canRecoverInsideApertureFeather)) {
            // The analytic feather can legitimately extend a few texels past
            // the triangulated colour pass. It can also expose a one-texel
            // raster hole when a very thin lip triangle becomes nearly
            // degenerate. Inside the aperture, recovery is limited to the
            // signed-distance feather above; the mask remains authoritative.
            float4 recoveredColor = float4(0.0);
            int2 textureLimit = int2(
                int(outputTexture.get_width()) - 1,
                int(outputTexture.get_height()) - 1
            );
            for (int yOffset = -4; yOffset <= 4; ++yOffset) {
                for (int xOffset = -4; xOffset <= 4; ++xOffset) {
                    int2 candidatePosition = clamp(
                        int2(position) + int2(xOffset, yOffset),
                        int2(0),
                        textureLimit
                    );
                    float4 candidate = colorTexture.read(
                        uint2(candidatePosition)
                    );
                    if (candidate.a > recoveredColor.a) {
                        recoveredColor = candidate;
                    }
                }
            }
            color = recoveredColor;
        }
        float alpha = color.a * maskCoverage;
        float outerGateStart = max(outerFeatherDistance - 1.0, 0.0);
        float outerBlurGate = 1.0 - smoothstep(
            outerGateStart,
            outerFeatherDistance,
            max(signedDistance, 0.0)
        );
        float blurGate = outerBlurGate * apertureFeather;
        // Keep pigment RGB independent from coverage. SceneKit receives this
        // as an opaque diffuse texture and reads `alpha` from a separate
        // transparency texture, so feathering cannot wash out the lip body.
        outputTexture.write(float4(color.rgb, alpha), position);
        blurGateTexture.write(float4(blurGate), position);
    }

    constant float lip_gaussian_weights[13] = {
        0.00735029, 0.01909834, 0.04171460, 0.07659181,
        0.11821653, 0.15338247, 0.16729190, 0.15338247,
        0.11821653, 0.07659181, 0.04171460, 0.01909834,
        0.00735029
    };

    kernel void lip_gaussian_blur_alpha_horizontal(
        texture2d<float, access::read> inputTexture [[texture(0)]],
        texture2d<float, access::write> horizontalBlurTexture [[texture(1)]],
        uint2 position [[thread_position_in_grid]]) {
        if (position.x >= horizontalBlurTexture.get_width() ||
            position.y >= horizontalBlurTexture.get_height()) {
            return;
        }

        int2 textureLimit = int2(
            int(inputTexture.get_width()) - 1,
            int(inputTexture.get_height()) - 1
        );
        float horizontalAlpha = 0.0;
        for (int xOffset = -6; xOffset <= 6; ++xOffset) {
            int2 samplePosition = clamp(
                int2(position) + int2(xOffset, 0),
                int2(0),
                textureLimit
            );
            horizontalAlpha += inputTexture.read(
                uint2(samplePosition)
            ).a * lip_gaussian_weights[xOffset + 6];
        }
        horizontalBlurTexture.write(float4(horizontalAlpha), position);
    }

    kernel void lip_gaussian_blur_alpha_vertical(
        texture2d<float, access::read> horizontalBlurTexture [[texture(0)]],
        texture2d<float, access::read> inputTexture [[texture(1)]],
        texture2d<float, access::write> outputTexture [[texture(2)]],
        texture2d<float, access::read> blurGateTexture [[texture(3)]],
        uint2 position [[thread_position_in_grid]]) {
        if (position.x >= outputTexture.get_width() ||
            position.y >= outputTexture.get_height()) {
            return;
        }

        // The same 13x13 Gaussian is evaluated as two separable 13-tap passes.
        // This preserves the broad photographic falloff while reducing texture
        // reads from 169 to 26 per pixel on older devices such as A14.
        int2 textureLimit = int2(
            int(horizontalBlurTexture.get_width()) - 1,
            int(horizontalBlurTexture.get_height()) - 1
        );
        float blurredAlpha = 0.0;
        for (int yOffset = -6; yOffset <= 6; ++yOffset) {
            int2 samplePosition = clamp(
                int2(position) + int2(0, yOffset),
                int2(0),
                textureLimit
            );
            blurredAlpha += horizontalBlurTexture.read(
                uint2(samplePosition)
            ).r * lip_gaussian_weights[yOffset + 6];
        }

        float4 center = inputTexture.read(position);
        const float gaussianBlurStrength = 1.0;
        float mixedAlpha = mix(
            center.a,
            blurredAlpha,
            saturate(gaussianBlurStrength)
        );
        // Let the Gaussian spread naturally, then cap it with a separate mask
        // that remains zero over the mouth aperture and reaches zero before
        // the outer carrier. This preserves the blur without colour on teeth
        // or a clipped SceneKit silhouette.
        float blurGate = blurGateTexture.read(position).r;
        float softenedAlpha = min(mixedAlpha, blurGate);
        outputTexture.write(float4(center.rgb, softenedAlpha), position);
    }
    """#

    private let context: Context?
    private let availabilityLock = NSLock()
    private var consecutiveRuntimeFailures = 0
    private var isRuntimeDisabled = false

    var isAvailable: Bool {
        availabilityLock.lock()
        let available = context != nil && !isRuntimeDisabled
        availabilityLock.unlock()
        return available
    }

    init() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            context = nil
            LipDebugLog.throttled(
                "lip_metal_unavailable",
                interval: 10,
                "lip_metal unavailable reason=no_device_or_queue"
            )
            return
        }

        var textureCache: CVMetalTextureCache?
        let cacheStatus = CVMetalTextureCacheCreate(
            kCFAllocatorDefault,
            nil,
            device,
            nil,
            &textureCache
        )
        guard cacheStatus == kCVReturnSuccess,
              let textureCache else {
            context = nil
            LipDebugLog.throttled(
                "lip_metal_unavailable",
                interval: 10,
                "lip_metal unavailable reason=texture_cache status=\(cacheStatus)"
            )
            return
        }

        do {
            let library = try device.makeLibrary(source: Self.shaderSource, options: nil)
            guard let vertexFunction = library.makeFunction(name: "lip_vertex"),
                  let fragmentFunction = library.makeFunction(name: "lip_fragment"),
                  let compositeFunction = library.makeFunction(name: "lip_composite"),
                  let alphaBlurHorizontalFunction = library.makeFunction(
                    name: "lip_gaussian_blur_alpha_horizontal"
                  ),
                  let alphaBlurVerticalFunction = library.makeFunction(
                    name: "lip_gaussian_blur_alpha_vertical"
                  ) else {
                context = nil
                LipDebugLog.throttled(
                    "lip_metal_unavailable",
                    interval: 10,
                    "lip_metal unavailable reason=shader_function"
                )
                return
            }

            let renderDescriptor = MTLRenderPipelineDescriptor()
            renderDescriptor.label = "VirtualMakeup.LipColor"
            renderDescriptor.vertexFunction = vertexFunction
            renderDescriptor.fragmentFunction = fragmentFunction
            renderDescriptor.colorAttachments[0].pixelFormat = .rgba8Unorm
            let renderPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)

            let compositePipeline = try device.makeComputePipelineState(function: compositeFunction)
            let alphaBlurHorizontalPipeline = try device.makeComputePipelineState(
                function: alphaBlurHorizontalFunction
            )
            let alphaBlurVerticalPipeline = try device.makeComputePipelineState(
                function: alphaBlurVerticalFunction
            )
            let indices = Self.makeIndices()
            let builtIndexBuffer: MTLBuffer? = indices.withUnsafeBytes { bytes -> MTLBuffer? in
                guard let baseAddress = bytes.baseAddress else {
                    return nil
                }
                return device.makeBuffer(
                    bytes: baseAddress,
                    length: bytes.count,
                    options: .storageModeShared
                )
            }
            guard !indices.isEmpty,
                  let indexBuffer = builtIndexBuffer else {
                context = nil
                LipDebugLog.throttled(
                    "lip_metal_unavailable",
                    interval: 10,
                    "lip_metal unavailable reason=index_buffer"
                )
                return
            }

            context = Context(
                device: device,
                commandQueue: commandQueue,
                textureCache: textureCache,
                renderPipeline: renderPipeline,
                compositePipeline: compositePipeline,
                alphaBlurHorizontalPipeline: alphaBlurHorizontalPipeline,
                alphaBlurVerticalPipeline: alphaBlurVerticalPipeline,
                indexBuffer: indexBuffer,
                indexCount: indices.count
            )
            LipDebugLog.throttled(
                "lip_metal_ready",
                interval: 10,
                "lip_metal ready device=\(device.name) triangles=\(indices.count / 3)"
            )
        } catch {
            context = nil
            LipDebugLog.throttled(
                "lip_metal_unavailable",
                interval: 10,
                "lip_metal unavailable reason=pipeline error=\(error.localizedDescription)"
            )
        }
    }

    private static func parameters(for finish: LipFinish) -> FinishParameters {
        switch finish {
        case .matte:
            return FinishParameters(
                detailStrength: 1.9,
                highlightStrength: 0,
                highlightMaximum: 0
            )
        case .satin:
            return FinishParameters(
                detailStrength: 1.7,
                highlightStrength: 1.16,
                highlightMaximum: 0.48
            )
        case .gloss:
            // Gloss brightness comes from captured reflections. Low-density
            // colour mixing is handled separately in the fragment shader.
            return FinishParameters(
                detailStrength: 1.9,
                highlightStrength: 1.05,
                highlightMaximum: 0.20
            )
        }
    }

    func makeTexture(contour: LipContour,
                     pixelBuffer: CVPixelBuffer,
                     imageSize: CGSize,
                     viewportSize: CGSize,
                     pixelWidth: Int,
                     pixelHeight: Int,
                     renderScale: CGFloat,
                     color: RGBColor,
                     lightingFactor: CGFloat,
                     finish: LipFinish,
                     density: LipstickDensity,
                     texture: LipstickTexture) -> LipTexture? {
        guard let context,
              isAvailable,
              pixelWidth > 1,
              pixelHeight > 1,
              CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA,
              let vertices = makeVertices(
                contour: contour,
                imageSize: imageSize,
                viewportSize: viewportSize
              ),
              let outerDistancePoints = makeOuterDistancePoints(
                contour: contour,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
              ),
              let innerDistancePoints = makeInnerDistancePoints(
                contour: contour,
                pixelWidth: pixelWidth,
                pixelHeight: pixelHeight
              ),
              let targets = renderTargets(
                context: context,
                width: pixelWidth,
                height: pixelHeight
              ) else {
            return nil
        }

        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        var cameraTextureReference: CVMetalTexture?
        let textureStatus = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            context.textureCache,
            pixelBuffer,
            nil,
            .bgra8Unorm,
            sourceWidth,
            sourceHeight,
            0,
            &cameraTextureReference
        )
        guard textureStatus == kCVReturnSuccess,
              let cameraTextureReference,
              let cameraTexture = CVMetalTextureGetTexture(cameraTextureReference) else {
            recordRuntimeFailure("camera_texture status=\(textureStatus)")
            return nil
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            recordRuntimeFailure("command_buffer")
            return nil
        }
        commandBuffer.label = "VirtualMakeup.LipComposite"

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            recordRuntimeFailure("render_encoder")
            return nil
        }
        renderEncoder.label = "VirtualMakeup.LipShade"
        renderEncoder.setRenderPipelineState(context.renderPipeline)
        vertices.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                renderEncoder.setVertexBytes(baseAddress, length: bytes.count, index: 0)
            }
        }
        let smoothApertureOpening = contour.innerApertureVisibility
        let apertureClosure = Float(1 - smoothApertureOpening)
        let finishParameters = Self.parameters(for: finish)
        var uniforms = Uniforms(
            lipstickColor: SIMD4<Float>(color.red, color.green, color.blue, 1),
            sourceTexelSize: SIMD2<Float>(
                1 / Float(max(sourceWidth, 1)),
                1 / Float(max(sourceHeight, 1))
            ),
            // The camera contributes lightness/detail, never its RGB hue.
            lightingFactor: Float(max(0.5, min(lightingFactor, 1))),
            detailStrength:
                finishParameters.detailStrength * texture.detailResponse,
            coreCoverage: density.pigmentCoverage,
            highlightStrength:
                finishParameters.highlightStrength * texture.highlightResponse,
            highlightMaximum:
                finishParameters.highlightMaximum * texture.highlightLimitResponse,
            highlightConcentration: texture.highlightConcentration,
            finish: Float(finish.rawValue),
            apertureClosure: apertureClosure
        )
        renderEncoder.setFragmentBytes(
            &uniforms,
            length: MemoryLayout<Uniforms>.stride,
            index: 0
        )
        renderEncoder.setFragmentTexture(cameraTexture, index: 0)
        renderEncoder.setCullMode(.none)
        renderEncoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: context.indexCount,
            indexType: .uint16,
            indexBuffer: context.indexBuffer,
            indexBufferOffset: 0
        )
        renderEncoder.endEncoding()

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("compute_encoder")
            return nil
        }
        computeEncoder.label = "VirtualMakeup.LipFeather"
        computeEncoder.setComputePipelineState(context.compositePipeline)
        computeEncoder.setTexture(targets.color, index: 0)
        computeEncoder.setTexture(targets.output, index: 1)
        computeEncoder.setTexture(targets.blurGate, index: 2)
        outerDistancePoints.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                computeEncoder.setBytes(
                    baseAddress,
                    length: bytes.count,
                    index: 0
                )
            }
        }
        innerDistancePoints.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                computeEncoder.setBytes(
                    baseAddress,
                    length: bytes.count,
                    index: 2
                )
            }
        }
        let maskPixelScale = Float(pixelHeight) / 96.0
        // Keep the detected inner lip edge in the opaque core. For a closed or
        // very narrow aperture, inward coverage overlaps across the canonical
        // hole and removes the seam. Once the mouth is visibly open, only a
        // short solid rim plus a compact transition remains over the mucosa.
        let apertureOpening = Float(smoothApertureOpening)
        let apertureSolidInnerCoverage: Float =
            (1.45 - apertureOpening * 1.45) * maskPixelScale
        let apertureInnerTransition: Float =
            (0.55 - apertureOpening * 0.20) * maskPixelScale
        var featherUniforms = FeatherUniforms(
            textureSize: SIMD2<Float>(Float(pixelWidth), Float(pixelHeight)),
            outerPointCount: UInt32(outerDistancePoints.count),
            innerPointCount: UInt32(innerDistancePoints.count),
            // Keep the lip interior intact and generate only the translucent
            // tail outside the detected boundary. The SceneKit carrier applies
            // the stricter screen-space cap to that exterior feather.
            outerFeatherPixels:
                LipOuterFeatherLayout.exteriorTransitionPixels * maskPixelScale,
            outerInnerFeatherPixels:
                LipOuterFeatherLayout.interiorTransitionPixels * maskPixelScale,
            apertureFeatherPixels: apertureSolidInnerCoverage,
            apertureMouthFeatherPixels: apertureInnerTransition,
            apertureInsetPixels: 0,
            apertureClosure: apertureClosure
        )
        computeEncoder.setBytes(
            &featherUniforms,
            length: MemoryLayout<FeatherUniforms>.stride,
            index: 1
        )
        let threadWidth = context.compositePipeline.threadExecutionWidth
        let threadHeight = max(
            1,
            context.compositePipeline.maxTotalThreadsPerThreadgroup / max(threadWidth, 1)
        )
        computeEncoder.dispatchThreads(
            MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: threadWidth,
                height: min(threadHeight, 16),
                depth: 1
            )
        )
        computeEncoder.endEncoding()

        guard let horizontalBlurEncoder = commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("alpha_blur_horizontal_encoder")
            return nil
        }
        horizontalBlurEncoder.label = "VirtualMakeup.LipGaussianAlphaHorizontal"
        horizontalBlurEncoder.setComputePipelineState(
            context.alphaBlurHorizontalPipeline
        )
        horizontalBlurEncoder.setTexture(targets.output, index: 0)
        horizontalBlurEncoder.setTexture(targets.horizontalBlur, index: 1)
        let horizontalBlurThreadWidth =
            context.alphaBlurHorizontalPipeline.threadExecutionWidth
        let horizontalBlurThreadHeight = max(
            1,
            context.alphaBlurHorizontalPipeline.maxTotalThreadsPerThreadgroup /
                max(horizontalBlurThreadWidth, 1)
        )
        horizontalBlurEncoder.dispatchThreads(
            MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: horizontalBlurThreadWidth,
                height: min(horizontalBlurThreadHeight, 16),
                depth: 1
            )
        )
        horizontalBlurEncoder.endEncoding()

        guard let verticalBlurEncoder = commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("alpha_blur_vertical_encoder")
            return nil
        }
        verticalBlurEncoder.label = "VirtualMakeup.LipGaussianAlphaVertical"
        verticalBlurEncoder.setComputePipelineState(
            context.alphaBlurVerticalPipeline
        )
        verticalBlurEncoder.setTexture(targets.horizontalBlur, index: 0)
        verticalBlurEncoder.setTexture(targets.output, index: 1)
        verticalBlurEncoder.setTexture(targets.softenedOutput, index: 2)
        verticalBlurEncoder.setTexture(targets.blurGate, index: 3)
        let verticalBlurThreadWidth =
            context.alphaBlurVerticalPipeline.threadExecutionWidth
        let verticalBlurThreadHeight = max(
            1,
            context.alphaBlurVerticalPipeline.maxTotalThreadsPerThreadgroup /
                max(verticalBlurThreadWidth, 1)
        )
        verticalBlurEncoder.dispatchThreads(
            MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: verticalBlurThreadWidth,
                height: min(verticalBlurThreadHeight, 16),
                depth: 1
            )
        )
        verticalBlurEncoder.endEncoding()

        let startedAt = CACurrentMediaTime()
        withExtendedLifetime(cameraTextureReference) {
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        guard commandBuffer.status == .completed else {
            recordRuntimeFailure(
                "gpu status=\(commandBuffer.status.rawValue) error=\(commandBuffer.error?.localizedDescription ?? "none")"
            )
            return nil
        }

        var rgba = [UInt8](repeating: 0, count: pixelWidth * pixelHeight * 4)
        rgba.withUnsafeMutableBytes { bytes in
            guard let baseAddress = bytes.baseAddress else {
                return
            }
            targets.softenedOutput.getBytes(
                baseAddress,
                bytesPerRow: pixelWidth * 4,
                from: MTLRegionMake2D(0, 0, pixelWidth, pixelHeight),
                mipmapLevel: 0
            )
        }
#if DEBUG
        LipDebugLog.throttled(
            "lip_metal_alpha_stats",
            interval: 0.75,
            Self.alphaDiagnostics(rgba)
        )
#endif
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let diffuseCGImage = CGImage(
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB) ??
                    CGColorSpaceCreateDeviceRGB(),
                // The fourth byte belongs exclusively to the opacity image.
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ),
              let opacityCGImage = CGImage(
                width: pixelWidth,
                height: pixelHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: pixelWidth * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB) ??
                    CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
              ) else {
            recordRuntimeFailure("cgimage")
            return nil
        }

        recordRuntimeSuccess()
        let gpuMilliseconds = commandBuffer.gpuEndTime > commandBuffer.gpuStartTime ?
            (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1000 : 0
        let blockingMilliseconds = (CACurrentMediaTime() - startedAt) * 1000
        LipDebugLog.throttled(
            "lip_metal_stats",
            interval: 0.6,
            "lip_metal frame size=\(pixelWidth)x\(pixelHeight) gpuMs=\(String(format: "%.2f", gpuMilliseconds)) waitReadbackMs=\(String(format: "%.2f", blockingMilliseconds)) lighting=\(String(format: "%.3f", lightingFactor)) apertureRaw=\(String(format: "%.3f", contour.innerOpeningRatio)) apertureRendered=\(String(format: "%.3f", contour.effectiveInnerOpeningRatio)) apertureBlend=\(String(format: "%.2f", smoothApertureOpening))"
        )
        return LipTexture(
            image: UIImage(
                cgImage: diffuseCGImage,
                scale: renderScale,
                orientation: .up
            ),
            opacityImage: UIImage(
                cgImage: opacityCGImage,
                scale: renderScale,
                orientation: .up
            )
        )
    }

#if DEBUG
    private static func alphaDiagnostics(_ rgba: [UInt8]) -> String {
        guard rgba.count >= 4 else {
            return "lip_metal alpha pixels=0"
        }

        var zeroCount = 0
        var partialCount = 0
        var opaqueCount = 0
        var minimumPartial = UInt8.max
        var maximumPartial = UInt8.min
        for offset in stride(from: 3, to: rgba.count, by: 4) {
            let alpha = rgba[offset]
            switch alpha {
            case 0:
                zeroCount += 1
            case 255:
                opaqueCount += 1
            default:
                partialCount += 1
                minimumPartial = min(minimumPartial, alpha)
                maximumPartial = max(maximumPartial, alpha)
            }
        }

        let partialRange = partialCount > 0 ?
            "\(minimumPartial)...\(maximumPartial)" : "none"
        return "lip_metal alpha zero=\(zeroCount) partial=\(partialCount) opaque=\(opaqueCount) partialRange=\(partialRange)"
    }
#endif

    private func makeVertices(contour: LipContour,
                              imageSize: CGSize,
                              viewportSize: CGSize) -> [Vertex]? {
        guard imageSize.width > 1,
              imageSize.height > 1,
              viewportSize.width > 1,
              viewportSize.height > 1 else {
            return nil
        }
        let inverseImageTransform = Self.aspectFillTransform(
            for: imageSize,
            in: viewportSize
        ).inverted()
        let expandedOuterScreenPoints: [Int: CGPoint]
        let expandedOuterUVPoints = LipOuterBoundary.expandedUVPoints(
            contour: contour,
            distance: LipOuterFeatherLayout.canonicalCarrierDistance,
            aspectRatio: LipOuterFeatherLayout.canonicalTextureAspectRatio
        ) ?? [:]
        if let pose = contour.pose {
            expandedOuterScreenPoints = LipOuterBoundary.expandedScreenPoints(
                contour: contour,
                distance: LipOuterMargin.sampling(for: pose.width)
            ) ?? [:]
        } else {
            expandedOuterScreenPoints = [:]
        }
        var vertices: [Vertex] = []
        vertices.reserveCapacity(CanonicalLipGeometry.attentionLipIndices.count)
        for index in CanonicalLipGeometry.attentionLipIndices {
            guard let point = contour.meshPointsByIndex[index] else {
                return nil
            }

            var renderScreenPoint = point.screen
            var renderCanonicalUV = point.uv

            if CanonicalLipGeometry.isOuterLipIndex(index),
               let expandedScreenPoint = expandedOuterScreenPoints[index],
               let expandedUVPoint = expandedOuterUVPoints[index] {
                // Sample outside the lip by the same contour-normal rule used
                // by the SceneKit carrier, so source colour and mesh agree.
                renderScreenPoint = expandedScreenPoint

                // Address exactly the same transparent texture carrier that
                // SceneKit displays. Camera sampling remains wider in screen
                // space, but must not use a different canonical UV layout.
                renderCanonicalUV = expandedUVPoint
            } else if CanonicalLipGeometry.isInnerLipIndex(index) {
                renderScreenPoint = point.screen
            }

            let imagePoint = renderScreenPoint.applying(inverseImageTransform)
            let sourceU = Float(imagePoint.x / imageSize.width)
            let sourceV = Float(imagePoint.y / imageSize.height)

            let canonicalU = Float(renderCanonicalUV.x)
            let canonicalV = Float(renderCanonicalUV.y)

            guard sourceU.isFinite,
                  sourceV.isFinite,
                  canonicalU.isFinite,
                  canonicalV.isFinite else {
                return nil
            }
            vertices.append(
                Vertex(
                    canonicalUV: SIMD2<Float>(canonicalU, canonicalV),
                    sourceUV: SIMD2<Float>(
                        min(max(sourceU, 0), 1),
                        min(max(sourceV, 0), 1)
                    )
                )
            )
        }
        return vertices
    }

    private func makeOuterDistancePoints(contour: LipContour,
                                         pixelWidth: Int,
                                         pixelHeight: Int) -> [SIMD2<Float>]? {
        guard contour.outerUV.count == CanonicalLipGeometry.outerLipIndices.count,
              let outerCurve = Self.smoothedClosedBoundary(
                contour.outerUV,
                radialScale: 1.00,
                // The displayed SceneKit carrier ends on the raw MediaPipe
                // segments. Use that identical polygon as the SDF zero edge;
                // otherwise the carrier can clip a separately smoothed fade.
                subdivisionPasses: 0
              ),
              let outerBounds = Self.bounds(for: outerCurve),
              outerCurve.count >= 4,
              outerBounds.width > 0.000_1,
              outerBounds.height > 0.000_1,
              pixelWidth > 1,
              pixelHeight > 1 else {
            return nil
        }

        // Keep the upper lip exactly on the detected contour. On the lower
        // lip, move only the SDF zero boundary slightly outwards so the real
        // vermilion edge stays in the opaque core instead of the translucent
        // half of the feather. Smooth weighting leaves both mouth corners in
        // place and prevents a visible step between upper and lower halves.
        // Do not move the visible lower boundary outside MediaPipe's contour.
        // Dense edge coverage and texture filtering already prevent gaps.
        let lowerCoverageExtensionPixels: CGFloat = 0
        let lowerSpan = max(outerBounds.midY - outerBounds.minY, 0.000_1)
        let halfWidth = max(outerBounds.width * 0.5, 0.000_1)
        return outerCurve.map {
            let lowerPosition = min(
                max((outerBounds.midY - $0.y) / lowerSpan, 0),
                1
            )
            // Saturate early across the lower half so the centre and its
            // neighbours receive the same translation. Only the transition
            // into the mouth corners is tapered.
            let lowerMembershipPosition = min(lowerPosition / 0.35, 1)
            let lowerMembership = lowerMembershipPosition *
                lowerMembershipPosition *
                (3 - 2 * lowerMembershipPosition)
            let horizontalPosition = min(
                abs($0.x - outerBounds.midX) / halfWidth,
                1
            )
            let cornerTaperPosition = min(
                max((1 - horizontalPosition) / 0.28, 0),
                1
            )
            let cornerTaper = cornerTaperPosition * cornerTaperPosition *
                (3 - 2 * cornerTaperPosition)
            let lowerWeight = lowerMembership * cornerTaper
            return SIMD2<Float>(
                Float($0.x) * Float(pixelWidth),
                (1 - Float($0.y)) * Float(pixelHeight) +
                    Float(lowerCoverageExtensionPixels * lowerWeight)
            )
        }
    }

    private func makeInnerDistancePoints(contour: LipContour,
                                         pixelWidth: Int,
                                         pixelHeight: Int) -> [SIMD2<Float>]? {
        guard contour.innerUV.count == CanonicalLipGeometry.innerLipIndices.count,
              let innerCurve = Self.smoothedClosedBoundary(
                contour.innerUV,
                radialScale: 1.00,
                subdivisionPasses: 2
              ),
              innerCurve.count >= 4,
              pixelWidth > 1,
              pixelHeight > 1 else {
            return nil
        }
        return innerCurve.map {
            SIMD2<Float>(
                Float($0.x) * Float(pixelWidth),
                (1 - Float($0.y)) * Float(pixelHeight)
            )
        }
    }

    private static func smoothedClosedBoundary(
        _ controlPoints: [CGPoint],
        radialScale: CGFloat,
        subdivisionPasses: Int
    ) -> [CGPoint]? {
        guard controlPoints.count >= 4,
              let sourceBounds = bounds(for: controlPoints),
              sourceBounds.width > 0.000_1,
              sourceBounds.height > 0.000_1 else {
            return nil
        }

        var curve = controlPoints
        for _ in 0..<max(subdivisionPasses, 0) {
            var subdivided: [CGPoint] = []
            subdivided.reserveCapacity(curve.count * 2)
            for index in curve.indices {
                let current = curve[index]
                let next = curve[(index + 1) % curve.count]
                subdivided.append(
                    CGPoint(
                        x: current.x + (next.x - current.x) * 0.25,
                        y: current.y + (next.y - current.y) * 0.25
                    )
                )
                subdivided.append(
                    CGPoint(
                        x: current.x + (next.x - current.x) * 0.75,
                        y: current.y + (next.y - current.y) * 0.75
                    )
                )
            }
            curve = subdivided
        }

        guard let curveBounds = bounds(for: curve),
              curveBounds.width > 0.000_1,
              curveBounds.height > 0.000_1 else {
            return nil
        }
        let sourceCenter = CGPoint(x: sourceBounds.midX, y: sourceBounds.midY)
        let curveCenter = CGPoint(x: curveBounds.midX, y: curveBounds.midY)
        let xScale = sourceBounds.width / curveBounds.width * radialScale
        let yScale = sourceBounds.height / curveBounds.height * radialScale
        let result = curve.map {
            CGPoint(
                x: sourceCenter.x + ($0.x - curveCenter.x) * xScale,
                y: sourceCenter.y + ($0.y - curveCenter.y) * yScale
            )
        }
        guard result.allSatisfy({
            $0.x.isFinite && $0.y.isFinite &&
                $0.x >= 0 && $0.x <= 1 &&
                $0.y >= 0 && $0.y <= 1
        }) else {
            return nil
        }
        return result
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
        return CGRect(
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        )
    }

    private func renderTargets(context: Context,
                               width: Int,
                               height: Int) -> RenderTargets? {
        if let renderTargets = context.renderTargets,
           renderTargets.width == width,
           renderTargets.height == height {
            return renderTargets
        }

        func makeTexture(
            pixelFormat: MTLPixelFormat = .rgba8Unorm,
            usage: MTLTextureUsage
        ) -> MTLTexture? {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: pixelFormat,
                width: width,
                height: height,
                mipmapped: false
            )
            descriptor.storageMode = .shared
            descriptor.usage = usage
            return context.device.makeTexture(descriptor: descriptor)
        }

        guard let color = makeTexture(usage: [.renderTarget, .shaderRead]),
              let output = makeTexture(usage: [.shaderRead, .shaderWrite]),
              let horizontalBlur = makeTexture(
                pixelFormat: .r16Float,
                usage: [.shaderRead, .shaderWrite]
              ),
              let softenedOutput = makeTexture(
                usage: [.shaderRead, .shaderWrite]
              ),
              let blurGate = makeTexture(
                pixelFormat: .r8Unorm,
                usage: [.shaderRead, .shaderWrite]
              ) else {
            recordRuntimeFailure("render_targets")
            return nil
        }
        color.label = "VirtualMakeup.LipColor"
        output.label = "VirtualMakeup.LipOutput"
        horizontalBlur.label = "VirtualMakeup.LipHorizontalBlur"
        softenedOutput.label = "VirtualMakeup.LipSoftenedOutput"
        blurGate.label = "VirtualMakeup.LipBlurGate"
        let targets = RenderTargets(
            width: width,
            height: height,
            color: color,
            output: output,
            horizontalBlur: horizontalBlur,
            softenedOutput: softenedOutput,
            blurGate: blurGate
        )
        context.renderTargets = targets
        return targets
    }

    private func recordRuntimeFailure(_ reason: String) {
        availabilityLock.lock()
        consecutiveRuntimeFailures += 1
        if consecutiveRuntimeFailures >= 3 {
            isRuntimeDisabled = true
        }
        let failures = consecutiveRuntimeFailures
        let disabled = isRuntimeDisabled
        availabilityLock.unlock()
        LipDebugLog.throttled(
            "lip_metal_runtime_failure",
            interval: 0.6,
            "lip_metal failure reason=\(reason) consecutive=\(failures) disabled=\(disabled)"
        )
    }

    private func recordRuntimeSuccess() {
        availabilityLock.lock()
        consecutiveRuntimeFailures = 0
        availabilityLock.unlock()
    }

    private static func makeIndices() -> [UInt16] {
        let indexByLandmark = Dictionary(
            uniqueKeysWithValues: CanonicalLipGeometry.attentionLipIndices
                .enumerated()
                .map { ($0.element, UInt16($0.offset)) }
        )
        var indices: [UInt16] = []
        indices.reserveCapacity(CanonicalLipGeometry.lipMeshTriangles.count * 3)
        for triangle in CanonicalLipGeometry.lipMeshTriangles {
            guard let first = indexByLandmark[triangle.0],
                  let second = indexByLandmark[triangle.1],
                  let third = indexByLandmark[triangle.2] else {
                return []
            }
            indices.append(contentsOf: [first, second, third])
        }
        return indices
    }

    private static func aspectFillTransform(for imageSize: CGSize,
                                            in viewportSize: CGSize) -> CGAffineTransform {
        let scale = max(
            viewportSize.width / imageSize.width,
            viewportSize.height / imageSize.height
        )
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        return CGAffineTransform(
            translationX: (viewportSize.width - scaledWidth) * 0.5,
            y: (viewportSize.height - scaledHeight) * 0.5
        ).scaledBy(x: scale, y: scale)
    }
}
