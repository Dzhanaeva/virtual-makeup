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
    let linerOpacityImage: UIImage?

    init(image: UIImage,
         opacityImage: UIImage? = nil,
         linerOpacityImage: UIImage? = nil) {
        self.image = image
        self.opacityImage = opacityImage ?? image
        self.linerOpacityImage = linerOpacityImage
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
        var diffuseLightResponse: Float
        var coreCoverage: Float
        var highlightStrength: Float
        var highlightMaximum: Float
        var highlightConcentration: Float
        var finish: Float
        var apertureClosure: Float
        var naturalLipColorWeight: Float
        var shadowRetention: Float
        var shineMaskMinimum: Float
        var shineMaskMaximum: Float
        var referenceLipLuminance: Float
        var capturedShadowMaximumStops: Float
        var capturedDiffuseLightMaximumStops: Float
        var shadingDebugMode: Float
    }

    private struct ShineTemporalUniforms {
        var riseCurrentWeight: Float
        var releaseCurrentWeight: Float
        var historyValid: Float
        var padding: Float = 0
    }

    private struct ShineGeometryUniforms {
        var textureSize: SIMD2<Float>
        var outerPointCount: UInt32
        var innerPointCount: UInt32
        var finish: Float
        var padding: Float = 0
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
        var finish: Float
    }

    private struct RenderTargets {
        let width: Int
        let height: Int
        let color: MTLTexture
        let currentShine: MTLTexture
        let stabilizedShine: MTLTexture
        let gatedShine: MTLTexture
        let shineHistory: MTLTexture
        let shadedColor: MTLTexture
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
        let shineTemporalPipeline: MTLComputePipelineState
        let shineGeometryPipeline: MTLComputePipelineState
        let shineApplyPipeline: MTLComputePipelineState
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
             shineTemporalPipeline: MTLComputePipelineState,
             shineGeometryPipeline: MTLComputePipelineState,
             shineApplyPipeline: MTLComputePipelineState,
             compositePipeline: MTLComputePipelineState,
             alphaBlurHorizontalPipeline: MTLComputePipelineState,
             alphaBlurVerticalPipeline: MTLComputePipelineState,
             indexBuffer: MTLBuffer,
             indexCount: Int) {
            self.device = device
            self.commandQueue = commandQueue
            self.textureCache = textureCache
            self.renderPipeline = renderPipeline
            self.shineTemporalPipeline = shineTemporalPipeline
            self.shineGeometryPipeline = shineGeometryPipeline
            self.shineApplyPipeline = shineApplyPipeline
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
        float diffuseLightResponse;
        float coreCoverage;
        float highlightStrength;
        float highlightMaximum;
        float highlightConcentration;
        float finish;
        float apertureClosure;
        float naturalLipColorWeight;
        float shadowRetention;
        float shineMaskMinimum;
        float shineMaskMaximum;
        float referenceLipLuminance;
        float capturedShadowMaximumStops;
        float capturedDiffuseLightMaximumStops;
        float shadingDebugMode;
    };

    struct LipShineTemporalUniforms {
        float riseCurrentWeight;
        float releaseCurrentWeight;
        float historyValid;
        float padding;
    };

    struct LipShineGeometryUniforms {
        float2 textureSize;
        uint outerPointCount;
        uint innerPointCount;
        float finish;
        float padding;
    };

    struct LipRaster {
        float4 position [[position]];
        float2 sourceUV;
        float2 canonicalUV;
    };

    struct LipFragmentOutput {
        float4 material [[color(0)]];
        float shine [[color(1)]];
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
        float finish;
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

    float3 lip_srgb_to_linear(float3 color) {
        color = saturate(color);
        float3 low = color / 12.92;
        float3 high = pow(
            (color + 0.055) / 1.055,
            float3(2.4)
        );
        return select(low, high, color > 0.04045);
    }

    float3 lip_linear_to_srgb(float3 color) {
        color = max(color, 0.0);
        float3 low = color * 12.92;
        float3 high = 1.055 * pow(
            color,
            float3(1.0 / 2.4)
        ) - 0.055;
        return saturate(select(low, high, color > 0.0031308));
    }

    float lip_luminance(float3 linearColor) {
        return clamp(
            dot(linearColor, float3(0.2126, 0.7152, 0.0722)),
            0.0,
            1.0
        );
    }

    float lip_perceptual_luma(float3 srgbColor) {
        return clamp(
            dot(srgbColor, float3(0.299, 0.587, 0.114)),
            0.0,
            1.0
        );
    }

    float lip_saturation(float3 color) {
        float maximum = max(color.r, max(color.g, color.b));
        float minimum = min(color.r, min(color.g, color.b));
        return maximum > 0.001 ? (maximum - minimum) / maximum : 0.0;
    }

    float2 lip_hsv_hue_saturation(float3 color) {
        float maximum = max(color.r, max(color.g, color.b));
        float minimum = min(color.r, min(color.g, color.b));
        float delta = maximum - minimum;
        float2 result = float2(0.0);
        if (delta > 0.0001 && maximum > 0.0001) {
            result.y = delta / maximum;
            if (maximum == color.r) {
                result.x = (color.g - color.b) / delta;
            } else if (maximum == color.g) {
                result.x = 2.0 + (color.b - color.r) / delta;
            } else {
                result.x = 4.0 + (color.r - color.g) / delta;
            }
            result.x = fract(result.x / 6.0);
        }
        return result;
    }

    float3 lip_hsv_to_rgb(float hue, float saturation, float value) {
        float3 hueOffsets = float3(1.0, 2.0 / 3.0, 1.0 / 3.0);
        float3 channels = clamp(
            abs(fract(hueOffsets + hue) * 6.0 - 3.0) - 1.0,
            0.0,
            1.0
        );
        return value * mix(float3(1.0), channels, saturation);
    }

    float lip_tooth_probability(float3 srgbColor) {
        // Tooth classification is perceptual rather than a lighting operation,
        // so preserve its camera-space thresholds while shading stays linear.
        float luminance = lip_perceptual_luma(srgbColor);
        float saturation = lip_saturation(srgbColor);
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

    float lip_captured_shape_exposure(
        float localLuminance,
        float pigmentLuminance,
        constant LipUniforms &uniforms
    ) {
        if (uniforms.referenceLipLuminance <= 0.0045 ||
            localLuminance <= 0.0) {
            return 0.0;
        }

        float palePigment = smoothstep(0.16, 0.38, pigmentLuminance);
        float denseCoverage = smoothstep(0.68, 0.92, uniforms.coreCoverage);
        float paleDenseAmount = palePigment * denseCoverage;
        float capturedExposure = log2(
            max(localLuminance, 0.0031) /
            max(uniforms.referenceLipLuminance, 0.0045)
        );
        if (capturedExposure < 0.0) {
            float maximumShadowStops = mix(
                0.92,
                uniforms.capturedShadowMaximumStops,
                paleDenseAmount
            );
            float shadowRetention = mix(
                saturate(uniforms.shadowRetention),
                max(saturate(uniforms.shadowRetention), 0.98),
                paleDenseAmount
            );
            return max(capturedExposure, -maximumShadowStops) *
                shadowRetention;
        }

        float maximumDiffuseStops =
            uniforms.capturedDiffuseLightMaximumStops *
            mix(1.0, 0.72, paleDenseAmount);
        return min(capturedExposure, maximumDiffuseStops) *
            uniforms.diffuseLightResponse;
    }

    fragment LipFragmentOutput lip_fragment(
        LipRaster input [[stage_in]],
        constant LipUniforms &uniforms [[buffer(0)]],
        texture2d<float> cameraTexture [[texture(0)]]) {
        constexpr sampler cameraSampler(
            coord::normalized,
            address::clamp_to_edge,
            filter::linear
        );

        float3 baseSRGB = cameraTexture.sample(
            cameraSampler,
            input.sourceUV
        ).rgb;
        float3 base = lip_srgb_to_linear(baseSRGB);
        // Adapted from Banuba beauty-ios matt.frag (MIT) as the diffuse base
        // of every finish: product hue/saturation plus per-pixel camera value.
        // HSV stays in display RGB, then returns to linear RGB for blending.
        float cameraValue = max(
            baseSRGB.r,
            max(baseSRGB.g, baseSRGB.b)
        );
        float3 lipstickSRGB = saturate(uniforms.lipstickColor.rgb);
        float lipstickValue = max(
            lipstickSRGB.r,
            max(lipstickSRGB.g, lipstickSRGB.b)
        );
        float targetValue = saturate(
            cameraValue * lipstickValue / 0.85
        );
        float2 lipstickHS = lip_hsv_hue_saturation(lipstickSRGB);
        float3 cameraAwarePigment = lip_srgb_to_linear(
            lip_hsv_to_rgb(
                lipstickHS.x,
                lipstickHS.y,
                targetValue
            )
        );
        if (uniforms.shadingDebugMode > 0.5) {
            LipFragmentOutput debugOutput;
            debugOutput.material = float4(
                lip_srgb_to_linear(float3(targetValue)),
                1.0
            );
            debugOutput.shine = 0.0;
            return debugOutput;
        }
        // Use the camera's real reflection pattern as the finish highlight.
        // A wider cross-shaped neighbourhood separates an existing bright
        // spot from the surrounding lip tone without inventing its position.
        float2 highlightRadius = uniforms.sourceTexelSize * float2(12.0, 8.0);
        float3 highlightSurround = lip_srgb_to_linear(
            cameraTexture.sample(
                cameraSampler,
                input.sourceUV + float2(highlightRadius.x, 0.0)
            ).rgb
        );
        highlightSurround += lip_srgb_to_linear(cameraTexture.sample(
            cameraSampler,
            input.sourceUV - float2(highlightRadius.x, 0.0)
        ).rgb);
        highlightSurround += lip_srgb_to_linear(cameraTexture.sample(
            cameraSampler,
            input.sourceUV + float2(0.0, highlightRadius.y)
        ).rgb);
        highlightSurround += lip_srgb_to_linear(cameraTexture.sample(
            cameraSampler,
            input.sourceUV - float2(0.0, highlightRadius.y)
        ).rgb);
        highlightSurround *= 0.25;

        float baseLuminance = lip_luminance(base);
        float mediumLuminance = max(
            lip_luminance(highlightSurround),
            0.0045
        );
        float textureY = 1.0 - input.canonicalUV.y;
        float cornerPosition = clamp(abs(input.canonicalUV.x - 0.5) * 2.0, 0.0, 1.0);
        float cornerRegion = smoothstep(0.62, 0.96, cornerPosition);
        float innerSeamRegion = 1.0 - smoothstep(0.018, 0.085, abs(textureY - 0.50));
        float apertureClosure = saturate(uniforms.apertureClosure);
        float openMouth = 1.0 - apertureClosure;
        bool isMatte = uniforms.finish < 0.5;
        bool isGloss = uniforms.finish > 0.5 && uniforms.finish < 1.5;
        bool isOil = uniforms.finish > 2.5 && uniforms.finish < 3.5;
        bool isWet = isGloss || isOil;
        float toothGuard = lip_tooth_probability(baseSRGB) *
            smoothstep(0.24, 0.46, openMouth) *
            innerSeamRegion;

        float currentShine = 0.0;
        if (uniforms.highlightStrength > 0.001) {
            // Use only the camera's real high-frequency reflection. Brightness
            // is amplified later without generating a synthetic lower-lip
            // lobe or increasing the spatial footprint of this signal.
            float highlightLogContrast = log2(
                max(baseLuminance, 0.0031) / mediumLuminance
            ) * 0.45;
            float relativeHighlight = smoothstep(
                0.001,
                0.065,
                highlightLogContrast
            );
            float highlightBrightness = smoothstep(
                0.012,
                0.220,
                baseLuminance
            );
            float3 highlightResidual = max(
                base - highlightSurround,
                float3(0.0)
            );
            float residualMaximum = max(
                highlightResidual.r,
                max(highlightResidual.g, highlightResidual.b)
            );
            float residualCommon = min(
                highlightResidual.r,
                min(highlightResidual.g, highlightResidual.b)
            );
            // A dielectric reflection raises all colour channels toward the
            // illuminant. This rejects bright red lip texture that has local
            // luminance contrast but is not a real specular reflection.
            float residualBalance = residualCommon /
                max(residualMaximum, 0.0001);
            float illuminantEvidence = smoothstep(
                0.18,
                0.62,
                residualBalance
            );
            float cameraShineMask = saturate(
                relativeHighlight * highlightBrightness *
                    mix(0.30, 1.0, illuminantEvidence)
            );

            if (isWet) {
                float wetMask = smoothstep(
                    uniforms.shineMaskMinimum,
                    uniforms.shineMaskMaximum,
                    cameraShineMask
                );
                // Separate the observed reflection from the diffuse lipstick
                // value before adding the white wet-specular layer below.
                // Otherwise the same camera highlight appears once as a
                // coloured bright patch and again as a white reflection.
                float3 highlightSurroundSRGB = lip_linear_to_srgb(
                    highlightSurround
                );
                float highlightSurroundValue = max(
                    highlightSurroundSRGB.r,
                    max(
                        highlightSurroundSRGB.g,
                        highlightSurroundSRGB.b
                    )
                );
                float diffuseCameraValue = mix(
                    cameraValue,
                    min(cameraValue, highlightSurroundValue),
                    wetMask * 0.82
                );
                float diffuseTargetValue = saturate(
                    diffuseCameraValue * lipstickValue / 0.85
                );
                cameraAwarePigment = lip_srgb_to_linear(
                    lip_hsv_to_rgb(
                        lipstickHS.x,
                        lipstickHS.y,
                        diffuseTargetValue
                    )
                );
                float wetExponent = isOil ? 1.05 : 1.30;
                float wetStrength = isOil ? 0.42 : 0.24;
                currentShine = min(
                    pow(wetMask, wetExponent) * wetStrength *
                        uniforms.highlightStrength,
                    uniforms.highlightMaximum
                );
            } else {
                float satinMask = smoothstep(
                    uniforms.shineMaskMinimum,
                    uniforms.shineMaskMaximum,
                    cameraShineMask
                );
                float softShineStrength = isMatte ? 0.22 : 0.26;
                currentShine = min(
                    pow(satinMask, 1.20) * softShineStrength *
                        uniforms.highlightStrength,
                    uniforms.highlightMaximum
                );
            }
            // Exact inner/outer-boundary exclusion happens in a following
            // signed-distance pass using this frame's detected contours.
            currentShine *=
                (1.0 - cornerRegion * 0.82) *
                (1.0 - innerSeamRegion *
                    (isWet ?
                        mix(0.30, 0.58, apertureClosure) :
                        mix(0.74, 0.92, apertureClosure))) *
                (1.0 - toothGuard * 0.96);
        }

        // Density is the only diffuse blending difference between products.
        // The camera already owns the light and shadow in cameraAwarePigment;
        // applying corner fades or inverse-alpha correction here would expose
        // bright seams and flatten that captured value field.
        LipFragmentOutput output;
        output.material = float4(
            cameraAwarePigment,
            uniforms.coreCoverage
        );
        output.shine = currentShine;
        return output;
    }

    kernel void lip_temporal_shine(
        texture2d<float, access::read> currentTexture [[texture(0)]],
        texture2d<float, access::read> historyTexture [[texture(1)]],
        texture2d<float, access::write> outputTexture [[texture(2)]],
        constant LipShineTemporalUniforms &uniforms [[buffer(0)]],
        uint2 position [[thread_position_in_grid]]) {
        if (position.x >= outputTexture.get_width() ||
            position.y >= outputTexture.get_height()) {
            return;
        }

        float current = saturate(currentTexture.read(position).r);
        float stabilized = current;
        if (uniforms.historyValid > 0.5) {
            float previous = saturate(historyTexture.read(position).r);
            float delta = abs(current - previous);
            float responsiveWeight = mix(
                current < previous ?
                    uniforms.releaseCurrentWeight :
                    uniforms.riseCurrentWeight,
                0.94,
                smoothstep(0.08, 0.34, delta)
            );
            stabilized = mix(previous, current, responsiveWeight);
            // A disappearing reflection must not leave a floating bright trail
            // on the canonical lip texture after the head starts moving.
            if (current < 0.003 && stabilized < 0.035) {
                stabilized = 0.0;
            }
        }
        outputTexture.write(float4(stabilized), position);
    }

    kernel void lip_apply_shine(
        texture2d<float, access::read> materialTexture [[texture(0)]],
        texture2d<float, access::read> shineTexture [[texture(1)]],
        texture2d<float, access::write> outputTexture [[texture(2)]],
        constant float &finish [[buffer(0)]],
        uint2 position [[thread_position_in_grid]]) {
        if (position.x >= outputTexture.get_width() ||
            position.y >= outputTexture.get_height()) {
            return;
        }
        float4 material = materialTexture.read(position);
        float shine = saturate(shineTexture.read(position).r);
        bool isGloss = finish > 0.5 && finish < 1.5;
        bool isOil = finish > 2.5 && finish < 3.5;
        if (isGloss || isOil) {
            // Wet specular reflection is achromatic. Keep its compact mask but
            // drive the core visibly toward white; oil is the brighter finish.
            float whiteAmount = saturate(
                shine * (isOil ? 2.30 : 1.45)
            );
            material.rgb = mix(material.rgb, float3(1.0), whiteAmount);
        } else {
            // Satin and subtle matte shine raise value softly while retaining
            // most of the product saturation.
            float3 materialSRGB = lip_linear_to_srgb(material.rgb);
            float2 materialHS = lip_hsv_hue_saturation(materialSRGB);
            float materialValue = max(
                materialSRGB.r,
                max(materialSRGB.g, materialSRGB.b)
            );
            float visibleShine = saturate(
                shine * (finish < 0.5 ? 1.30 : 1.40)
            );
            material.rgb = lip_srgb_to_linear(
                lip_hsv_to_rgb(
                    materialHS.x,
                    materialHS.y * (1.0 - visibleShine * 0.22),
                    mix(materialValue, 1.0, visibleShine)
                )
            );
        }
        outputTexture.write(material, position);
    }

    float lip_distance_to_segment(float2 point, float2 first, float2 second) {
        float2 segment = second - first;
        float denominator = max(dot(segment, segment), 0.0001);
        float position = clamp(dot(point - first, segment) / denominator, 0.0, 1.0);
        return distance(point, first + segment * position);
    }

    kernel void lip_gate_shine(
        texture2d<float, access::read> inputTexture [[texture(0)]],
        texture2d<float, access::write> outputTexture [[texture(1)]],
        const device float2 *outerPoints [[buffer(0)]],
        constant LipShineGeometryUniforms &uniforms [[buffer(1)]],
        const device float2 *innerPoints [[buffer(2)]],
        uint2 position [[thread_position_in_grid]]) {
        if (position.x >= outputTexture.get_width() ||
            position.y >= outputTexture.get_height()) {
            return;
        }
        if (uniforms.outerPointCount < 3 ||
            uniforms.innerPointCount < 3) {
            outputTexture.write(float4(0.0), position);
            return;
        }

        float inputShine = saturate(inputTexture.read(position).r);
        if (inputShine <= 0.0001) {
            outputTexture.write(float4(0.0), position);
            return;
        }

        float2 pixelPoint = float2(position) + 0.5;
        float outerMinimumDistance = 100000.0;
        float outerMinimumX = 100000.0;
        float outerMaximumX = -100000.0;
        bool isInsideOuter = false;
        uint outerPrevious = uniforms.outerPointCount - 1;
        for (uint index = 0; index < uniforms.outerPointCount; ++index) {
            float2 first = outerPoints[outerPrevious];
            float2 second = outerPoints[index];
            outerMinimumDistance = min(
                outerMinimumDistance,
                lip_distance_to_segment(pixelPoint, first, second)
            );
            outerMinimumX = min(outerMinimumX, second.x);
            outerMaximumX = max(outerMaximumX, second.x);
            bool crosses =
                (first.y > pixelPoint.y) != (second.y > pixelPoint.y);
            if (crosses) {
                float denominator = second.y - first.y;
                float intersectionX = first.x +
                    (pixelPoint.y - first.y) *
                    (second.x - first.x) /
                    (abs(denominator) < 0.0001 ? 0.0001 : denominator);
                if (pixelPoint.x < intersectionX) {
                    isInsideOuter = !isInsideOuter;
                }
            }
            outerPrevious = index;
        }

        float innerMinimumDistance = 100000.0;
        float innerMinimumY = 100000.0;
        float innerMaximumY = -100000.0;
        bool isInsideAperture = false;
        uint innerPrevious = uniforms.innerPointCount - 1;
        for (uint index = 0; index < uniforms.innerPointCount; ++index) {
            float2 first = innerPoints[innerPrevious];
            float2 second = innerPoints[index];
            innerMinimumDistance = min(
                innerMinimumDistance,
                lip_distance_to_segment(pixelPoint, first, second)
            );
            innerMinimumY = min(innerMinimumY, second.y);
            innerMaximumY = max(innerMaximumY, second.y);
            bool crosses =
                (first.y > pixelPoint.y) != (second.y > pixelPoint.y);
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

        if (!isInsideOuter || isInsideAperture) {
            outputTexture.write(float4(0.0), position);
            return;
        }

        // Measure position across the actual lip thickness. Zero is the inner
        // contour and one is the outer contour, regardless of lip shape.
        float surfacePosition = innerMinimumDistance / max(
            innerMinimumDistance + outerMinimumDistance,
            0.001
        );
        bool isGloss = uniforms.finish > 0.5 && uniforms.finish < 1.5;
        bool isOil = uniforms.finish > 2.5 && uniforms.finish < 3.5;
        float outerFadeStart = isGloss ? 0.58 : (isOil ? 0.60 : 0.70);
        float outerFadeEnd = isGloss ? 0.70 : (isOil ? 0.73 : 0.86);
        float contourDistanceGate =
            smoothstep(0.18, 0.34, surfacePosition) *
            (1.0 - smoothstep(
                outerFadeStart,
                outerFadeEnd,
                surfacePosition
            ));

        float normalizedX = saturate(
            (pixelPoint.x - outerMinimumX) /
                max(outerMaximumX - outerMinimumX, 0.001)
        );
        float lowerFadeInStart = isGloss ? 0.24 : (isOil ? 0.21 : 0.14);
        float lowerFadeInEnd = isGloss ? 0.32 : (isOil ? 0.29 : 0.26);
        float lowerFadeOutStart = isGloss ? 0.68 : (isOil ? 0.71 : 0.74);
        float lowerFadeOutEnd = isGloss ? 0.76 : (isOil ? 0.79 : 0.86);
        float lowerHorizontalGate =
            smoothstep(lowerFadeInStart, lowerFadeInEnd, normalizedX) *
            (1.0 - smoothstep(
                lowerFadeOutStart,
                lowerFadeOutEnd,
                normalizedX
            ));
        float cupidStripe = 1.0 - smoothstep(
            0.035,
            0.095,
            abs(normalizedX - 0.5)
        );
        float mouthCenterY = (innerMinimumY + innerMaximumY) * 0.5;
        float lowerLip = smoothstep(
            mouthCenterY - 0.5,
            mouthCenterY + max(uniforms.textureSize.y * 0.025, 1.0),
            pixelPoint.y
        );
        float surfaceRegionGate = mix(
            cupidStripe,
            lowerHorizontalGate,
            lowerLip
        );
        float geometryGate = saturate(
            contourDistanceGate * surfaceRegionGate
        );
        outputTexture.write(float4(inputShine * geometryGate), position);
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
        const float contourCoverage = 0.76;
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
        if (uniforms.finish > 3.5) {
            // A tint stains the moist inner lip most strongly, then loses
            // pigment continuously toward the outer vermilion contour.
            float tintInnerPosition = minimumDistance /
                max(minimumDistance + innerMinimumDistance, 0.001);
            float tintDistribution = mix(
                0.58,
                1.0,
                smoothstep(0.0, 0.86, tintInnerPosition)
            );
            maskCoverage *= tintDistribution;
        }
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
        // This is the only colour-space exit from the linear Metal pipeline.
        // The 8-bit readback is tagged sRGB when it becomes a CGImage.
        outputTexture.write(
            float4(lip_linear_to_srgb(center.rgb), softenedAlpha),
            position
        );
    }
    """#

    private let context: Context?
    private let availabilityLock = NSLock()
    private let temporalStateLock = NSLock()
    private var consecutiveRuntimeFailures = 0
    private var isRuntimeDisabled = false
    private var smoothedReferenceLipLuminance: Float?
    private var isShineHistoryValid = false
    private var temporalGeneration: UInt64 = 0

    var isAvailable: Bool {
        availabilityLock.lock()
        let available = context != nil && !isRuntimeDisabled
        availabilityLock.unlock()
        return available
    }

    func resetTemporalState() {
        temporalStateLock.lock()
        temporalGeneration &+= 1
        smoothedReferenceLipLuminance = nil
        isShineHistoryValid = false
        temporalStateLock.unlock()
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
                  let shineTemporalFunction = library.makeFunction(
                    name: "lip_temporal_shine"
                  ),
                  let shineGeometryFunction = library.makeFunction(
                    name: "lip_gate_shine"
                  ),
                  let shineApplyFunction = library.makeFunction(
                    name: "lip_apply_shine"
                  ),
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
            renderDescriptor.colorAttachments[0].pixelFormat = .rgba16Float
            renderDescriptor.colorAttachments[1].pixelFormat = .r16Float
            let renderPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)

            let shineTemporalPipeline = try device.makeComputePipelineState(
                function: shineTemporalFunction
            )
            let shineGeometryPipeline = try device.makeComputePipelineState(
                function: shineGeometryFunction
            )
            let shineApplyPipeline = try device.makeComputePipelineState(
                function: shineApplyFunction
            )
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
                shineTemporalPipeline: shineTemporalPipeline,
                shineGeometryPipeline: shineGeometryPipeline,
                shineApplyPipeline: shineApplyPipeline,
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
                     materialProfile: LipMaterialProfile,
                     shadingDebugMode: LipShadingDebugMode,
                     lowLatency: Bool,
                     motionDelta: CGFloat,
                     completionQueue: DispatchQueue,
                     completion: @escaping (LipTexture?) -> Void) {
        temporalStateLock.lock()
        let generation = temporalGeneration
        temporalStateLock.unlock()
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
            completion(nil)
            return
        }

        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let referenceLipLuminance = stabilizedReferenceLipLuminance(
            Self.referenceLipLuminance(
                pixelBuffer: pixelBuffer,
                vertices: vertices
            ), generation: generation
        ) ?? 0
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
            completion(nil)
            return
        }

        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else {
            recordRuntimeFailure("command_buffer")
            completion(nil)
            return
        }
        commandBuffer.label = "VirtualMakeup.LipComposite"

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = targets.color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        pass.colorAttachments[1].texture = targets.currentShine
        pass.colorAttachments[1].loadAction = .clear
        pass.colorAttachments[1].storeAction = .store
        pass.colorAttachments[1].clearColor = MTLClearColorMake(0, 0, 0, 0)
        guard let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            recordRuntimeFailure("render_encoder")
            completion(nil)
            return
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
        var uniforms = Uniforms(
            lipstickColor: SIMD4<Float>(color.red, color.green, color.blue, 1),
            sourceTexelSize: SIMD2<Float>(
                1 / Float(max(sourceWidth, 1)),
                1 / Float(max(sourceHeight, 1))
            ),
            // The camera contributes lightness/detail, never its RGB hue.
            lightingFactor: Float(max(0.5, min(lightingFactor, 1))),
            detailStrength: materialProfile.capturedDetailStrength,
            diffuseLightResponse: materialProfile.diffuseLightResponse,
            coreCoverage: materialProfile.pigmentCoverage,
            highlightStrength: materialProfile.capturedHighlightStrength,
            highlightMaximum: materialProfile.capturedHighlightMaximum,
            highlightConcentration: materialProfile.highlightConcentration,
            finish: Float(finish.rawValue),
            apertureClosure: apertureClosure,
            naturalLipColorWeight: materialProfile.naturalLipColorWeight,
            shadowRetention: materialProfile.shadowRetention,
            shineMaskMinimum: materialProfile.shineMaskMinimum,
            shineMaskMaximum: materialProfile.shineMaskMaximum,
            referenceLipLuminance: referenceLipLuminance,
            capturedShadowMaximumStops:
                materialProfile.capturedShadowMaximumStops,
            capturedDiffuseLightMaximumStops:
                materialProfile.capturedDiffuseLightMaximumStops,
            shadingDebugMode: Float(shadingDebugMode.rawValue)
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

        guard let shineTemporalEncoder = commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("shine_temporal_encoder")
            completion(nil)
            return
        }
        shineTemporalEncoder.label = "VirtualMakeup.LipShineTemporal"
        shineTemporalEncoder.setComputePipelineState(
            context.shineTemporalPipeline
        )
        shineTemporalEncoder.setTexture(targets.currentShine, index: 0)
        shineTemporalEncoder.setTexture(targets.shineHistory, index: 1)
        shineTemporalEncoder.setTexture(targets.stabilizedShine, index: 2)
        let normalizedMotion = Float(max(0, min(motionDelta / 0.22, 1)))
        var shineTemporalUniforms = ShineTemporalUniforms(
            riseCurrentWeight: min(
                0.94,
                (lowLatency ? 0.86 : 0.78) + normalizedMotion * 0.12
            ),
            releaseCurrentWeight: min(
                0.98,
                0.90 + normalizedMotion * 0.08
            ),
            historyValid: shineHistoryIsValid() ? 1 : 0
        )
        shineTemporalEncoder.setBytes(
            &shineTemporalUniforms,
            length: MemoryLayout<ShineTemporalUniforms>.stride,
            index: 0
        )
        let shineTemporalThreadWidth =
            context.shineTemporalPipeline.threadExecutionWidth
        let shineTemporalThreadHeight = max(
            1,
            context.shineTemporalPipeline.maxTotalThreadsPerThreadgroup /
                max(shineTemporalThreadWidth, 1)
        )
        shineTemporalEncoder.dispatchThreads(
            MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: shineTemporalThreadWidth,
                height: min(shineTemporalThreadHeight, 16),
                depth: 1
            )
        )
        shineTemporalEncoder.endEncoding()

        guard let shineGeometryEncoder =
                commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("shine_geometry_encoder")
            completion(nil)
            return
        }
        shineGeometryEncoder.label = "VirtualMakeup.LipShineGeometry"
        shineGeometryEncoder.setComputePipelineState(
            context.shineGeometryPipeline
        )
        shineGeometryEncoder.setTexture(targets.stabilizedShine, index: 0)
        shineGeometryEncoder.setTexture(targets.gatedShine, index: 1)
        outerDistancePoints.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                shineGeometryEncoder.setBytes(
                    baseAddress,
                    length: bytes.count,
                    index: 0
                )
            }
        }
        innerDistancePoints.withUnsafeBytes { bytes in
            if let baseAddress = bytes.baseAddress {
                shineGeometryEncoder.setBytes(
                    baseAddress,
                    length: bytes.count,
                    index: 2
                )
            }
        }
        var shineGeometryUniforms = ShineGeometryUniforms(
            textureSize: SIMD2<Float>(Float(pixelWidth), Float(pixelHeight)),
            outerPointCount: UInt32(outerDistancePoints.count),
            innerPointCount: UInt32(innerDistancePoints.count),
            finish: Float(finish.rawValue)
        )
        shineGeometryEncoder.setBytes(
            &shineGeometryUniforms,
            length: MemoryLayout<ShineGeometryUniforms>.stride,
            index: 1
        )
        let shineGeometryThreadWidth =
            context.shineGeometryPipeline.threadExecutionWidth
        let shineGeometryThreadHeight = max(
            1,
            context.shineGeometryPipeline.maxTotalThreadsPerThreadgroup /
                max(shineGeometryThreadWidth, 1)
        )
        shineGeometryEncoder.dispatchThreads(
            MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: shineGeometryThreadWidth,
                height: min(shineGeometryThreadHeight, 16),
                depth: 1
            )
        )
        shineGeometryEncoder.endEncoding()

        guard let shineApplyEncoder = commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("shine_apply_encoder")
            completion(nil)
            return
        }
        shineApplyEncoder.label = "VirtualMakeup.LipShineApply"
        shineApplyEncoder.setComputePipelineState(context.shineApplyPipeline)
        shineApplyEncoder.setTexture(targets.color, index: 0)
        shineApplyEncoder.setTexture(targets.gatedShine, index: 1)
        shineApplyEncoder.setTexture(targets.shadedColor, index: 2)
        var shineFinish = Float(finish.rawValue)
        shineApplyEncoder.setBytes(
            &shineFinish,
            length: MemoryLayout<Float>.stride,
            index: 0
        )
        let shineApplyThreadWidth = context.shineApplyPipeline.threadExecutionWidth
        let shineApplyThreadHeight = max(
            1,
            context.shineApplyPipeline.maxTotalThreadsPerThreadgroup /
                max(shineApplyThreadWidth, 1)
        )
        shineApplyEncoder.dispatchThreads(
            MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: shineApplyThreadWidth,
                height: min(shineApplyThreadHeight, 16),
                depth: 1
            )
        )
        shineApplyEncoder.endEncoding()

        guard let shineHistoryEncoder = commandBuffer.makeBlitCommandEncoder() else {
            recordRuntimeFailure("shine_history_encoder")
            completion(nil)
            return
        }
        shineHistoryEncoder.label = "VirtualMakeup.LipShineHistory"
        shineHistoryEncoder.copy(
            from: targets.gatedShine,
            sourceSlice: 0,
            sourceLevel: 0,
            sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
            sourceSize: MTLSize(width: pixelWidth, height: pixelHeight, depth: 1),
            to: targets.shineHistory,
            destinationSlice: 0,
            destinationLevel: 0,
            destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0)
        )
        shineHistoryEncoder.endEncoding()

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            recordRuntimeFailure("compute_encoder")
            completion(nil)
            return
        }
        computeEncoder.label = "VirtualMakeup.LipFeather"
        computeEncoder.setComputePipelineState(context.compositePipeline)
        computeEncoder.setTexture(targets.shadedColor, index: 0)
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
        let apertureSolidInnerCoverage = Float(
            LipInnerBoundaryLayout.solidCoveragePixels(
                opening: smoothApertureOpening
            )
        ) * maskPixelScale
        let apertureInnerTransition = Float(
            LipInnerBoundaryLayout.transitionPixels(
                opening: smoothApertureOpening
            )
        ) * maskPixelScale
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
            apertureClosure: apertureClosure,
            finish: Float(finish.rawValue)
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
            completion(nil)
            return
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
            completion(nil)
            return
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
        let riseWeight = shineTemporalUniforms.riseCurrentWeight
        let releaseWeight = shineTemporalUniforms.releaseCurrentWeight
        // One request owns the targets until completion (including readback).
        // Keep both the CV wrapper and its source alive for the entire GPU job.
        commandBuffer.addCompletedHandler { [self, cameraTextureReference, pixelBuffer] completedBuffer in
            completionQueue.async { [self] in
                withExtendedLifetime((cameraTextureReference, pixelBuffer)) {
                    let completedAt = CACurrentMediaTime()
                    guard completedBuffer.status == .completed else {
                        self.recordRuntimeFailure(
                            "gpu status=\(completedBuffer.status.rawValue) error=\(completedBuffer.error?.localizedDescription ?? "none")"
                        )
                        completion(nil)
                        return
                    }
                    self.temporalStateLock.lock()
                    let temporalContextIsCurrent = self.temporalGeneration == generation
                    if temporalContextIsCurrent { self.isShineHistoryValid = true }
                    self.temporalStateLock.unlock()
                    guard temporalContextIsCurrent else { completion(nil); return }

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
                        self.recordRuntimeFailure("cgimage")
                        completion(nil)
                        return
                    }

                    self.recordRuntimeSuccess()
                    let gpuMilliseconds = completedBuffer.gpuEndTime > completedBuffer.gpuStartTime ?
                        (completedBuffer.gpuEndTime - completedBuffer.gpuStartTime) * 1000 : 0
                    let readbackMilliseconds = (CACurrentMediaTime() - completedAt) * 1000
                    LipDebugLog.throttled(
                        "lip_metal_stats",
                        interval: 0.6,
                        "lip_metal frame size=\(pixelWidth)x\(pixelHeight) gpuMs=\(String(format: "%.2f", gpuMilliseconds)) completionMs=\(String(format: "%.2f", (completedAt - startedAt) * 1000)) readbackMs=\(String(format: "%.2f", readbackMilliseconds)) async=true lighting=\(String(format: "%.3f", lightingFactor)) referenceLuma=\(String(format: "%.3f", referenceLipLuminance)) shineWeight=\(String(format: "%.2f", riseWeight))/\(String(format: "%.2f", releaseWeight)) motion=\(String(format: "%.3f", motionDelta)) apertureRaw=\(String(format: "%.3f", contour.innerOpeningRatio)) apertureRendered=\(String(format: "%.3f", contour.effectiveInnerOpeningRatio)) apertureBlend=\(String(format: "%.2f", smoothApertureOpening))"
                    )
                    completion(LipTexture(
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
                    ))
                }
            }
        }
        commandBuffer.commit()
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

        guard let color = makeTexture(
                pixelFormat: .rgba16Float,
                usage: [.renderTarget, .shaderRead]
              ),
              let currentShine = makeTexture(
                pixelFormat: .r16Float,
                usage: [.renderTarget, .shaderRead]
              ),
              let stabilizedShine = makeTexture(
                pixelFormat: .r16Float,
                usage: [.shaderRead, .shaderWrite]
              ),
              let gatedShine = makeTexture(
                pixelFormat: .r16Float,
                usage: [.shaderRead, .shaderWrite]
              ),
              let shineHistory = makeTexture(
                pixelFormat: .r16Float,
                usage: [.shaderRead]
              ),
              let shadedColor = makeTexture(
                pixelFormat: .rgba16Float,
                usage: [.shaderRead, .shaderWrite]
              ),
              let output = makeTexture(
                pixelFormat: .rgba16Float,
                usage: [.shaderRead, .shaderWrite]
              ),
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
        currentShine.label = "VirtualMakeup.LipCurrentShine"
        stabilizedShine.label = "VirtualMakeup.LipStabilizedShine"
        gatedShine.label = "VirtualMakeup.LipGatedShine"
        shineHistory.label = "VirtualMakeup.LipShineHistory"
        shadedColor.label = "VirtualMakeup.LipShadedColor"
        output.label = "VirtualMakeup.LipOutput"
        horizontalBlur.label = "VirtualMakeup.LipHorizontalBlur"
        softenedOutput.label = "VirtualMakeup.LipSoftenedOutput"
        blurGate.label = "VirtualMakeup.LipBlurGate"
        let targets = RenderTargets(
            width: width,
            height: height,
            color: color,
            currentShine: currentShine,
            stabilizedShine: stabilizedShine,
            gatedShine: gatedShine,
            shineHistory: shineHistory,
            shadedColor: shadedColor,
            output: output,
            horizontalBlur: horizontalBlur,
            softenedOutput: softenedOutput,
            blurGate: blurGate
        )
        context.renderTargets = targets
        invalidateShineHistory()
        return targets
    }

    private static func referenceLipLuminance(
        pixelBuffer: CVPixelBuffer,
        vertices: [Vertex]
    ) -> Float? {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) ==
                kCVPixelFormatType_32BGRA,
              vertices.count == CanonicalLipGeometry.attentionLipIndices.count,
              CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly) ==
                kCVReturnSuccess else {
            return nil
        }
        defer {
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        var luminances: [Float] = []
        luminances.reserveCapacity(vertices.count / 2)

        for (offset, landmarkIndex) in
            CanonicalLipGeometry.attentionLipIndices.enumerated() {
            // Outer vertices sample the feather carrier and inner vertices can
            // touch the mouth aperture. The two support rings are the most
            // stable representation of the actual lip surface.
            guard !CanonicalLipGeometry.isOuterLipIndex(landmarkIndex),
                  !CanonicalLipGeometry.isInnerLipIndex(landmarkIndex) else {
                continue
            }
            let sourceUV = vertices[offset].sourceUV
            let x = min(
                max(Int((sourceUV.x * Float(width)).rounded(.down)), 0),
                width - 1
            )
            let y = min(
                max(Int((sourceUV.y * Float(height)).rounded(.down)), 0),
                height - 1
            )
            let pixel = baseAddress
                .advanced(by: y * bytesPerRow + x * 4)
                .assumingMemoryBound(to: UInt8.self)
            let red = Float(pixel[2]) / 255
            let green = Float(pixel[1]) / 255
            let blue = Float(pixel[0]) / 255
            let perceptualLuma = red * 0.299 + green * 0.587 + blue * 0.114
            let maximum = max(red, max(green, blue))
            let minimum = min(red, min(green, blue))
            let saturation = maximum > 0.001 ?
                (maximum - minimum) / maximum : 0
            guard !(perceptualLuma > 0.64 && saturation < 0.22) else {
                continue
            }

            func linear(_ component: Float) -> Float {
                component <= 0.04045 ?
                    component / 12.92 :
                    pow((component + 0.055) / 1.055, 2.4)
            }
            let luminance = linear(red) * 0.2126 +
                linear(green) * 0.7152 +
                linear(blue) * 0.0722
            if luminance.isFinite, luminance > 0.001 {
                luminances.append(luminance)
            }
        }

        guard luminances.count >= 8 else {
            return nil
        }
        luminances.sort()
        let percentileIndex = min(
            max(Int((Float(luminances.count - 1) * 0.65).rounded()), 0),
            luminances.count - 1
        )
        return luminances[percentileIndex]
    }

    private func stabilizedReferenceLipLuminance(_ current: Float?, generation: UInt64) -> Float? {
        guard let current, current.isFinite, current > 0.001 else {
            return nil
        }
        temporalStateLock.lock()
        defer {
            temporalStateLock.unlock()
        }

        guard temporalGeneration == generation else { return nil }
        let next: Float
        if let previous = smoothedReferenceLipLuminance {
            let exposureStops = abs(log2(current / max(previous, 0.001)))
            let currentWeight: Float = exposureStops > 0.70 ? 0.58 : 0.22
            next = previous + (current - previous) * currentWeight
        } else {
            next = current
        }
        smoothedReferenceLipLuminance = next
        return next
    }

    private func shineHistoryIsValid() -> Bool {
        temporalStateLock.lock()
        let isValid = isShineHistoryValid
        temporalStateLock.unlock()
        return isValid
    }

    private func invalidateShineHistory() {
        temporalStateLock.lock()
        isShineHistoryValid = false
        temporalStateLock.unlock()
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
        let samplingTriangles = CanonicalLipGeometry.lipMeshTriangles +
            CanonicalLipGeometry.innerFillTriangles
        var indices: [UInt16] = []
        indices.reserveCapacity(samplingTriangles.count * 3)
        // Rasterize the aperture carrier as well as the visible lip annulus.
        // Its alpha stays controlled by the signed-distance mask, but exposed
        // wet-line pixels now have direct camera-aware colour instead of a
        // fragile nearest-neighbour recovery from the lip surface.
        for triangle in samplingTriangles {
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
