#include "newtype/render/PassDenoiser.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/render/Shading.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

void RelaxDenoiser::compilePrepass(Device& device) {
	//==========================================================================
	// ReLAX Prepass: Poisson 8-tap spatial blur on noisy input
	// Ported from RELAX_PrePass.cs.hlsl (NRD v4.17)
	//
	// Edge-stopped blur to reduce noise before temporal accumulation.
	// Applies separate blur radii for diffuse and specular channels.
	//==========================================================================
	_relaxPrepass = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat io_Diff,
		ImageFloat io_Spec,
		ImageFloat gIn_Normal_Roughness,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_Tiles
		) noexcept {
		set_block_size(16u, 16u, 1u);
		set_name("relax_prepass");
		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);

		UInt rectW = cast<uint>(c.gRectSizeX);
		UInt rectH = cast<uint>(c.gRectSizeY);
		
		// Tile-based early out
		UInt2 tilePos = pixelPos >> 4u;
		//Float isSky = gIn_Tiles.read(tilePos).x;
		$if(Expr{ gIn_Tiles.read(tilePos).x != 0.0f } | pixelPos.x >= rectW | pixelPos.y >= rectH) {
			$return();
		};

		Float2 rectSizeInv = make_float2(c.gRectSizeInvX, c.gRectSizeInvY);
		Float denoisingRange = c.gDenoisingRange;
		Float centerViewZ = luisa::compute::abs(gIn_ViewZ.read(pixelPos).x);
		$if(centerViewZ > denoisingRange) { $return(); };

		// Read center data
		Float4 centerDiff = io_Diff.read(pixelPos);
		Float4 centerSpec = io_Spec.read(pixelPos);
		Float4 centerNR = gIn_Normal_Roughness.read(pixelPos);

		// Checkerboard bilateral resolve (NRD PrePass approach):
		// Fill inactive pixels from active horizontal neighbors using depth weights.
		// MUST happen before the blur to prevent zero contamination.
		UInt cbField = c.gDiffCheckerboard;
		Bool cbActive = (cbField == 0u) | (((pixelPos.x + pixelPos.y) & 1u) == (cbField & 1u));
		$if(!cbActive) {
			UInt nbLx = ite(pixelPos.x > 0u, pixelPos.x - 1u, pixelPos.x + 1u);
			UInt nbRx = ite(pixelPos.x < rectW - 1u, pixelPos.x + 1u, pixelPos.x - 1u);
			Float vzL = luisa::compute::abs(gIn_ViewZ.read(make_uint2(nbLx, pixelPos.y)).x);
			Float vzR = luisa::compute::abs(gIn_ViewZ.read(make_uint2(nbRx, pixelPos.y)).x);
			// NRD bilateral weight: LinearStep(0.03, 0.0, |z - zc| / max(z, zc))
			Float maxZ_L = luisa::compute::max(vzL, centerViewZ);
			Float maxZ_R = luisa::compute::max(vzR, centerViewZ);
			Float wL = luisa::compute::saturate(1.0f - luisa::compute::abs(vzL - centerViewZ) / (maxZ_L + 1e-6f) * 33.3f);
			Float wR = luisa::compute::saturate(1.0f - luisa::compute::abs(vzR - centerViewZ) / (maxZ_R + 1e-6f) * 33.3f);
			Float ws = wL + wR;
			$if(ws > 0.0f) {
				Float invWs = 1.0f / ws;
				Float4 dL = io_Diff.read(make_uint2(nbLx, pixelPos.y));
				Float4 dR = io_Diff.read(make_uint2(nbRx, pixelPos.y));
				centerDiff = (dL * wL + dR * wR) * invWs;
				Float4 sL = io_Spec.read(make_uint2(nbLx, pixelPos.y));
				Float4 sR = io_Spec.read(make_uint2(nbRx, pixelPos.y));
				centerSpec = (sL * wL + sR * wR) * invWs;
			};
		};
		Float3 centerNormal = luisa::compute::normalize(centerNR.xyz() * 2.0f - 1.0f);
		Float centerRoughness = luisa::compute::fract(centerNR.w);  // unpack roughness from packed matID+roughness

		// Center world position
		Float2 clipXY = (make_float2(pixelPos) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
		Float3 frustumFwd = c.gFrustumForward.xyz();
		Float3 frustumRight = c.gFrustumRight.xyz();
		Float3 frustumUp = c.gFrustumUp.xyz();
		Float3 rayDir = frustumFwd + frustumRight * clipXY.x - frustumUp * clipXY.y;
		Float viewZc = centerViewZ / luisa::compute::length(rayDir);
		Float3 centerWorldPos = viewZc * rayDir;

		Float unproject = c.gUnproject;

		// View vector for specular
		Float3 V = luisa::compute::normalize(-centerWorldPos);

		// Specular dominant direction for lobe-aware filtering
		Float3 dominantDir = luisa::compute::reflect(-V, centerNormal);
		Float3 specLobeDir = luisa::compute::normalize(
			lerp(centerNormal, dominantDir, centerRoughness * (0.7f + 0.3f * centerRoughness)));
		Float NoLobeDir = luisa::compute::abs(luisa::compute::dot(centerNormal, specLobeDir));

		// Spec magic curve: NRD _NRD_GetSpecMagicCurve(R, 0.25)
		//   f = (1 - exp2(-200·R²)) · R^0.25
		Float _smcR = luisa::compute::saturate(centerRoughness);
		Float specMagicCurve = (1.0f - luisa::compute::exp2(-200.0f * _smcR * _smcR))
		                     * luisa::compute::sqrt(luisa::compute::sqrt(_smcR));

		// Hit distance factors
		Float diffHitDist = ite(centerDiff.w == 0.0f, 1.0f, centerDiff.w);
		Float specHitDist = ite(centerSpec.w == 0.0f, 1.0f, centerSpec.w);
		Float frustumSize = unproject * luisa::compute::abs(centerViewZ)
			* luisa::compute::min(cast<Float>(rectW), cast<Float>(rectH));
		Float diffHitDistFactor = diffHitDist / (diffHitDist + frustumSize);
		Float specHitDistFactor = (specHitDist * NoLobeDir) / (specHitDist * NoLobeDir + frustumSize);

		// World-space blur radii
		Float diffBlurRadius = c.gDiffBlurRadius * diffHitDistFactor;
		Float specBlurRadius = c.gSpecBlurRadius * specHitDistFactor * specMagicCurve;

		// NRD PrePass: cap spec blur radius to the actual GGX lobe footprint to
		// avoid over-blurring sharp reflections before TA (RELAX_PrePass.cs.hlsl:281-285).
		// ImportanceSampling::GetSpecularLobeTanHalfAngle default percentOfVolume = 0.75
		// (NOT gLobeAngleFraction, which is for edge-stopping normal weight — this is
		// "how much of the lobe footprint the pre-blur is allowed to remove").
		Float lobeCapPercentOfVolume = 0.75f;
		Float lobeTanHalf = centerRoughness * centerRoughness * lobeCapPercentOfVolume
		                  / (1.0f - lobeCapPercentOfVolume + 1e-6f);
		Float lobeRadiusWorld = specHitDist * NoLobeDir * lobeTanHalf;
		Float viewZAtHit = luisa::compute::abs(centerViewZ) + specHitDist * NoLobeDir;
		Float pixelWorldSizeAtHit = unproject * luisa::compute::max(viewZAtHit, 1e-6f);
		Float specLobeMinBlurRadius = lobeRadiusWorld / pixelWorldSizeAtHit;
		specBlurRadius = luisa::compute::min(specBlurRadius, specLobeMinBlurRadius);

		// Accumulation for diffuse
		Float diffSumW = def(1.0f);
		Float3 diffSum = centerDiff.xyz();
		Float diffHdSumW = def(1.0f);
		Float diffHdSum = centerDiff.w;

		// Accumulation for specular
		Float specSumW = def(1.0f);
		Float3 specSum = centerSpec.xyz();
		Float specHdSumW = def(1.0f);
		Float specHdSum = centerSpec.w;

		Float diffBlurActive = ite(c.gDiffBlurRadius > 0.0f, 1.0f, 0.0f);
		Float specBlurActive = ite(c.gSpecBlurRadius > 0.0f, 1.0f, 0.0f);

		$if(diffBlurActive + specBlurActive > 0.0f) {
			Float depthThreshold = c.gDepthThreshold * centerViewZ;

			Float diffNormalWeightParam = 1.0f / luisa::compute::max(
				luisa::compute::atan(0.25f * c.gLobeAngleFraction), 1.5f / 255.0f);

			// NRD roughness weight params: a = 1/lerp(sensitivity, 1, saturate(R·fraction))
			// with NRD_ROUGHNESS_SENSITIVITY = 0.01. Used in SmoothStep(1, 0, |ΔR|·a) form.
			Float roughSensitivity = 0.01f;
			Float roughA = 1.0f / luisa::compute::lerp(roughSensitivity, 1.0f,
				luisa::compute::saturate(centerRoughness * c.gRoughnessFraction));

			// NRD: GetNormalWeightParam2(R, 0.5 * gLobeAngleFraction) — roughness-dependent
			// cone, narrow at low R (mirror), wide at high R (rough).
			Float specAngleFraction = 0.5f * c.gLobeAngleFraction;
			Float specTanHalf = centerRoughness * centerRoughness * specAngleFraction
			                  / (1.0f - specAngleFraction + 1e-6f);
			Float specNormWParam = 1.0f / luisa::compute::max(
				luisa::compute::atan(specTanHalf), 1.5f / 255.0f);

			Float pixelRadius = luisa::compute::max(diffBlurRadius, specBlurRadius);

			// NRD PrePass: per-frame Poisson rotation (NRD_FRAME mode in RELAX_PrePass.cs.hlsl:80).
			// Same rotation for all pixels in a given frame, varying frame-to-frame.
			// Reduces directional noise patterns without per-pixel divergence.
			Float rotAngle = cast<Float>(c.gFrameIndex) * 2.39996323f;  // golden angle
			Float rotCos = luisa::compute::cos(rotAngle);
			Float rotSin = luisa::compute::sin(rotAngle);

			// Poisson tap helper lambda - inlined 8 times
			auto prepassSample = [&](Float2 pOff) noexcept {
				// Rotate offset by per-frame rotator
				Float2 rOff = make_float2(
					pOff.x * rotCos - pOff.y * rotSin,
					pOff.x * rotSin + pOff.y * rotCos);
				Int2 samplePosInt = make_int2(pixelPos) +
					make_int2(cast<int>(rOff.x * pixelRadius),
						cast<int>(rOff.y * pixelRadius));
				samplePosInt = clamp(samplePosInt, make_int2(0),
					make_int2(cast<int>(rectW) - 1, cast<int>(rectH) - 1));
				UInt2 sp = make_uint2(samplePosInt);

				Float sampleViewZ = luisa::compute::abs(gIn_ViewZ.read(sp).x);
				$if(sampleViewZ > denoisingRange) { $return(); };

				Float4 sampleNR = gIn_Normal_Roughness.read(sp);
				Float3 sampleNormal = luisa::compute::normalize(sampleNR.xyz() * 2.0f - 1.0f);
				Float sampleRoughness = luisa::compute::fract(sampleNR.w);

				Float2 sampleClipXY = (make_float2(sp) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
				Float3 sampleRayDir = frustumFwd + frustumRight * sampleClipXY.x - frustumUp * sampleClipXY.y;
				Float sampleViewZc = sampleViewZ / luisa::compute::length(sampleRayDir);
				Float3 sampleWorldPos = sampleViewZc * sampleRayDir;

				Float planeDist = luisa::compute::abs(
					luisa::compute::dot(sampleWorldPos - centerWorldPos, centerNormal));
				Float geometryW = luisa::compute::saturate(
					1.0f - planeDist / luisa::compute::max(depthThreshold, 1e-6f));

				Float offsetLen = luisa::compute::length(pOff);
				Float gaussianW = luisa::compute::exp(-offsetLen * offsetLen * 2.0f);

				Float baseW = geometryW * gaussianW;
				$if(baseW < 1e-4f) { $return(); };

				Float angle = luisa::compute::acos(
					luisa::compute::clamp(luisa::compute::dot(centerNormal, sampleNormal), -1.0f, 1.0f));
				Float diffNormalW = luisa::compute::clamp(
					1.0f - luisa::compute::smoothstep(0.0f, 1.0f, angle * diffNormalWeightParam),
					0.0f, 1.0f);
				Float specNormalW = luisa::compute::clamp(
					1.0f - luisa::compute::smoothstep(0.0f, 1.0f, angle * specNormWParam),
					0.0f, 1.0f);

				Float4 sampleDiff = io_Diff.read(sp);
				Float4 sampleSpec = io_Spec.read(sp);
				Float sampleDiffHd = ite(sampleDiff.w == 0.0f, 1.0f, sampleDiff.w);
				Float sampleSpecHd = ite(sampleSpec.w == 0.0f, 1.0f, sampleSpec.w);

				Float diffHdW = luisa::compute::exp(
					-luisa::compute::abs(diffHitDist - sampleDiffHd) / (diffHitDist + 1e-6f) * 4.0f);
				Float specHdW = luisa::compute::exp(
					-luisa::compute::abs(specHitDist - sampleSpecHd) / (specHitDist + 1e-6f) * 4.0f);

				Float dw = baseW * diffNormalW * diffHdW;
				$if(dw > 1e-4f) {
					diffSumW = diffSumW + dw;
					diffSum = diffSum + dw * sampleDiff.xyz();
					diffHdSumW = diffHdSumW + dw;
					diffHdSum = diffHdSum + dw * sampleDiffHd;
				};

				// NRD ComputeNonExponentialWeight: SmoothStep(1, 0, |ΔR|·a) = 1 - d·d·(3-2d)
				// where d = saturate(|ΔR|·a). Cubic shape — more permissive at small ΔR than linear.
				Float _dR = luisa::compute::saturate(
					luisa::compute::abs(centerRoughness - sampleRoughness) * roughA);
				Float roughnessW = 1.0f - _dR * _dR * (3.0f - 2.0f * _dR);
				Float sw = baseW * specNormalW * specHdW * roughnessW;

				// NRD PrePass: contact-dimming weight (RELAX_PrePass.cs.hlsl:358-362).
				// Samples close to reflection contacts (h << d) get reduced weight on
				// smooth surfaces to preserve sharp contact reflections. Rough surfaces
				// (centerRoughness > 0.5) are exempt via LinearStep(0.5, 1.0, roughness).
				Float contactDist = luisa::compute::length(sampleWorldPos - centerWorldPos);
				Float sampleHitT = sampleSpec.w;
				Float t = sampleHitT / (centerSpec.w + contactDist + 1e-6f);
				Float roughnessFactor = luisa::compute::saturate(2.0f * centerRoughness - 1.0f);
				Float contactW = lerp(luisa::compute::saturate(t), 1.0f, roughnessFactor);
				sw = sw * contactW;

				$if(sw > 1e-4f) {
					specSumW = specSumW + sw;
					specSum = specSum + sw * sampleSpec.xyz();
					specHdSumW = specHdSumW + sw;
					specHdSum = specHdSum + sw * sampleSpecHd;
				};
			};

			prepassSample(luisa::make_float2(-0.470980f, 0.254864f));
			prepassSample(luisa::make_float2(-0.480085f, -0.545432f));
			prepassSample(luisa::make_float2(0.361964f, -0.397249f));
			prepassSample(luisa::make_float2(0.429670f, 0.376037f));
			prepassSample(luisa::make_float2(-0.183193f, 0.823649f));
			prepassSample(luisa::make_float2(0.264636f, -0.851386f));
			prepassSample(luisa::make_float2(-0.753498f, -0.107800f));
			prepassSample(luisa::make_float2(0.850542f, 0.042284f));
		};

		// Normalize and write
		Float invDiffW = 1.0f / diffSumW;
		Float invDiffHdW = 1.0f / luisa::compute::max(diffHdSumW, 1e-6f);
		io_Diff.write(pixelPos, make_float4(diffSum * invDiffW, diffHdSum * invDiffHdW));

		Float invSpecW = 1.0f / specSumW;
		Float invSpecHdW = 1.0f / luisa::compute::max(specHdSumW, 1e-6f);
		io_Spec.write(pixelPos, make_float4(specSum * invSpecW, specHdSum * invSpecHdW));
	});
}
}