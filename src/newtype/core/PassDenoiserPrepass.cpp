#include "newtype/render/PassDenoiser.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/render/Shading.h"
#include "newtype/util/Rng.h"

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
		// Compile-time projection tag (§9 perf fix): C++ constant — every
		// projection condition below folds in DXC, so the perspective build
		// keeps the pre-projection instruction sequence.
		const uint bakedProjection = _bakedProjection;

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

		// NRD PrePass clamps the specular hit distance to the denoising range
		// before use (RELAX_PrePass.cs.hlsl:266).
		centerSpec.w = luisa::compute::clamp(centerSpec.w, 0.0f, denoisingRange);

		// Center world position — NRD GetCurrentWorldPosFromClipSpaceXY form
		// (RELAX_Common.hlsli:75-80): viewZ * (fwd + right·cx ± up·cy). NRD's
		// minus sign assumes its Y-DOWN uv; this engine's texel uv is Y-UP
		// (row 0 = screen bottom, the same space the TA reconstructs in), so
		// the faithful engine form is +up·cy with cy = uv·2−1. The former −up
		// mirrored every reconstructed position about the camera's horizontal
		// axis (latent: internally-consistent plane gates masked it).
		Float2 clipXY = (make_float2(pixelPos) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
		Float3 frustumFwd = c.gFrustumForward.xyz();
		Float3 frustumRight = c.gFrustumRight.xyz();
		Float3 frustumUp = c.gFrustumUp.xyz();
		// Point-origin projections: viewZ * pixel-direction(uv) (§5.4).
		Float3 reconDir = def((frustumFwd + frustumRight * clipXY.x + frustumUp * clipXY.y));
		if (bakedProjection != 0u) {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
		    reconDir = util::eval_point_origin_direction(clipXY,
		        pc.gFrustumRight.xyz(), pc.gFrustumUp.xyz(), pc.gFrustumForward.xyz(),
		        bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
		}
		Float3 centerWorldPos = centerViewZ * reconDir;

		Float unproject = c.gUnproject;

		// View vector for specular
		Float3 V = luisa::compute::normalize(-centerWorldPos);
		Float NoV = luisa::compute::abs(luisa::compute::dot(centerNormal, V));

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

		// Accumulation for specular — RGB is a weighted blur; the hit distance is
		// a MIN over a fixed 1px ring (see the min block below). The former
		// weighted-average .w systematically OVERSHOT along hit-distance
		// discontinuities (highlight edges), displacing the virtual-motion ray
		// beyond the true previous highlight position (RC2); the intermediate
		// NRD-style wide stochastic min temporally flickered at the same
		// discontinuities (RC2 follow-up, report §12 R2).
		Float specSumW = def(1.0f);
		Float3 specSum = centerSpec.xyz();
		// NRD: specularHitT = (.a == 0) ? gDenoisingRange : .a; minHitT init from it
		Float specularHitT = ite(centerSpec.w == 0.0f, denoisingRange, centerSpec.w);
		Float minHitT = specularHitT;
		// NRD: specMinHitDistanceWeight = (.a == 0) ? 1.0 : gMinHitDistanceWeight * smc
		Float specMinHitDistanceWeight = ite(centerSpec.w == 0.0f, 1.0f,
			c.gMinHitDistanceWeight * specMagicCurve);

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
				if (bakedProjection == 1u | bakedProjection == 2u) {
					Int wfull = cast<int>(rectW);
					Int wx = samplePosInt.x - wfull * (samplePosInt.x / wfull);
					wx = wx + ite(wx < 0, wfull, 0);
					samplePosInt.x = wx;
					samplePosInt.y = clamp(samplePosInt.y, 0, cast<int>(rectH) - 1);
				} else {
					samplePosInt = clamp(samplePosInt, make_int2(0),
						make_int2(cast<int>(rectW) - 1, cast<int>(rectH) - 1));
				}
				UInt2 sp = make_uint2(samplePosInt);

				Float sampleViewZ = luisa::compute::abs(gIn_ViewZ.read(sp).x);
				$if(sampleViewZ > denoisingRange) { $return(); };

				Float4 sampleNR = gIn_Normal_Roughness.read(sp);
				Float3 sampleNormal = luisa::compute::normalize(sampleNR.xyz() * 2.0f - 1.0f);
				Float sampleRoughness = luisa::compute::fract(sampleNR.w);

				Float2 sampleClipXY = (make_float2(sp) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
				Float3 reconDir = def((frustumFwd + frustumRight * sampleClipXY.x + frustumUp * sampleClipXY.y));
				if (bakedProjection != 0u) {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
				    reconDir = util::eval_point_origin_direction(sampleClipXY,
				        pc.gFrustumRight.xyz(), pc.gFrustumUp.xyz(), pc.gFrustumForward.xyz(),
				        bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
				}
				Float3 sampleWorldPos = sampleViewZ * reconDir;

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

				Float diffHdW = luisa::compute::exp(
					-luisa::compute::abs(diffHitDist - sampleDiffHd) / (diffHitDist + 1e-6f) * 4.0f);

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

				// (Hit-distance min moved out of this lambda: fixed 1px ring with
				// deterministic gates — see the min block after the blur loop.)

				// NRD RGB hit-distance weight (RELAX_PrePass.cs.hlsl:355 + Common.hlsli:529-532):
				// lerp(specMinHitDistanceWeight, 1, ComputeExponentialWeight(sample.a, 9, -9*center.a))
				// = lerp(floor, 1, 1/(t²+t+1)) with t = 27·|sample.a - center.a|
				// (ExpApprox(-3·|·|), params a = 1/(1/9) = 9, scale 3).
				Float hdwT = 27.0f * luisa::compute::abs(sampleSpec.w - centerSpec.w);
				Float specHdW = luisa::compute::lerp(specMinHitDistanceWeight, 1.0f,
					1.0f / (hdwT * hdwT + hdwT + 1.0f));
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

		// Hit-distance min: FIXED 1px ring, deterministic gates. NRD samples the
		// min over the full rotated blur radius with per-frame stochastic gating
		// (RELAX_PrePass.cs.hlsl:351-375). At a hit-distance step (a reflected
		// light edge) that reach spreads the near-side distance a full blur
		// radius into the field, and the per-frame rotation + gate coin-flips
		// make the step boundary flicker — TA feeds this field straight into the
		// virtual-motion ray, so the VMB fetch position jumps several px per
		// frame at every reflection edge, a band history clamping cannot average
		// out (report §12 R2: the ring band around the fixture reflection).
		$if(specBlurActive > 0.0f) {
			Float minDepthThreshold = c.gDepthThreshold * centerViewZ;
			Float minCenterMat = luisa::compute::floor(centerNR.w);
			auto minTap = [&](int dx, int dy) noexcept {
				Int2 mp = clamp(make_int2(pixelPos) + make_int2(dx, dy),
					make_int2(0), make_int2(cast<int>(rectW) - 1, cast<int>(rectH) - 1));
				UInt2 mpp = make_uint2(mp);
				Float mZ = luisa::compute::abs(gIn_ViewZ.read(mpp).x);
				$if(mZ > 0.0f & mZ <= denoisingRange) {
					Float4 mNR = gIn_Normal_Roughness.read(mpp);
					Float4 mSpec = io_Spec.read(mpp);
					$if(mSpec.w > 0.0f
						& luisa::compute::max(luisa::compute::floor(mNR.w), c.gSpecMinMaterial)
							== luisa::compute::max(minCenterMat, c.gSpecMinMaterial)) {
					Float2 mClip = (make_float2(mpp) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
					// The tap's own world position at its own depth — the former
					// code reused the CENTER's scaled depth along the neighbor ray,
					// so the plane gate tested a phantom point (NRD PrePass:341
					// reconstructs each sample at sampleViewZ).
					Float3 mWorld = mZ
						* (frustumFwd + frustumRight * mClip.x + frustumUp * mClip.y);
						Float mPlane = luisa::compute::abs(
							luisa::compute::dot(mWorld - centerWorldPos, centerNormal));
						$if(mPlane < minDepthThreshold) {
							minHitT = luisa::compute::min(minHitT, mSpec.w);
						};
					};
				};
			};
			minTap(1, 0);
			minTap(-1, 0);
			minTap(0, 1);
			minTap(0, -1);
			minTap(1, 1);
			minTap(-1, -1);
			minTap(1, -1);
			minTap(-1, 1);
		};

		// Normalize and write
		Float invDiffW = 1.0f / diffSumW;
		Float invDiffHdW = 1.0f / luisa::compute::max(diffHdSumW, 1e-6f);
		io_Diff.write(pixelPos, make_float4(diffSum * invDiffW, diffHdSum * invDiffHdW));

		Float invSpecW = 1.0f / specSumW;
		// Specular .w is the fixed-ring min (NRD RELAX_PrePass.cs.hlsl:375 analog).
		Float specHitTOut = ite(minHitT >= denoisingRange, 0.0f, minHitT);
		io_Spec.write(pixelPos, make_float4(specSum * invSpecW, specHitTOut));
	});
}
}