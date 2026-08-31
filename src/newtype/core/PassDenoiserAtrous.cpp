#include "newtype/render/PassDenoiser.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Rng.h"
#include "newtype/render/Shading.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

void RelaxDenoiser::compileAtrous(Device& device) {
	//================================================================================================================================================
		// ReLAX Step 7: Atrous Spatial Filter
		// Ported from RELAX_Atrous.cs.hlsl (NRD v4.17)
		//
		// 3x3 Gaussian cross-bilateral filter with configurable step size.
		// Edge-stopping: depth (plane distance), normal (angle), luminance (exp).
		// Variance propagated in .w channel. History length in .w on last pass.
		//==========================================================================
	_relaxAtrous = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat out_Diff,
		ImageFloat in_Diff,
		ImageFloat gIn_Normal_Roughness,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_HistoryLength,
		UInt stepSize,
		UInt isLastPass,
		ImageFloat gIn_Tiles,
		// Specular (9-10): pass-through
		ImageFloat out_Spec,
		ImageFloat in_Spec
		) noexcept {
		set_block_size(8u, 16u, 1u);
		set_name("atrous");

		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);

		UInt rectW = cast<uint>(c.gRectSizeX);
		UInt rectH = cast<uint>(c.gRectSizeY);
		Float2 rectSizeInv = make_float2(c.gRectSizeInvX, c.gRectSizeInvY);
		Float denoisingRange = c.gDenoisingRange;

		// Tile-based early out
		UInt2 tilePos = pixelPos >> 4u;
		//Float isSky = gIn_Tiles.read(tilePos).x;
		$if(Expr{ gIn_Tiles.read(tilePos).x != 0.0f } | pixelPos.x >= rectW | pixelPos.y >= rectH) {
			$return();
		};

		// Early out if beyond denoising range
		Float centerViewZ = luisa::compute::abs(gIn_ViewZ.read(pixelPos).x);
		$if(centerViewZ > denoisingRange) {
			$return();
		};

		// Read center data
		Float4 centerNR = gIn_Normal_Roughness.read(pixelPos);
		Float3 centerNormal = luisa::compute::normalize(centerNR.xyz() * 2.0f - 1.0f);
		Float centerPackedNR   = centerNR.w;
		Float centerMaterialID = luisa::compute::floor(centerPackedNR);
		Float centerRoughness  = centerPackedNR - centerMaterialID;

		Float4 _hlRead2 = gIn_HistoryLength.read(pixelPos);
		Float historyLength = 255.0f * _hlRead2.x;
		Float specConf = _hlRead2.y;

		// Center diffuse + variance
		Float4 centerDiff = in_Diff.read(pixelPos);
		Float centerLum = luminance(centerDiff.xyz());
		Float centerVar = centerDiff.w;
		// Center specular
		Float4 centerSpec = in_Spec.read(pixelPos);
		Float centerSpecLum = luminance(centerSpec.xyz());

		// NRD RELAX_Atrous.cs.hlsl:120 — diffuse normal weight param uses lobeFraction
		// directly (no relaxation). The invented normalRelaxFactor below used to widen
		// the acceptance cone by gNormalEdgeStoppingRelaxation, letting cross-curvature
		// neighbors leak noise into the diffuse filter on metal under rotation.
		Float lobeFraction = c.gLobeAngleFraction / luisa::compute::sqrt(cast<Float>(stepSize));
		lobeFraction = lerp(0.99f, lobeFraction, luisa::compute::saturate(historyLength / 5.0f));

		Float normalWeightParam = 1.0f / luisa::compute::max(
			luisa::compute::atan(lobeFraction), 1.5f / 255.0f);

		// Roughness weight params (NRD: GetRoughnessWeightParams).
		// ComputeWeight(sampleRoughness, a, -b) simplifies to saturate(1 - abs(Δr) * a).
		Float roughSensitivity = 0.01f;
		Float roughA = 1.0f / luisa::compute::lerp(roughSensitivity, 1.0f,
			luisa::compute::saturate(centerRoughness * c.gRoughnessFraction));

		// Specular normal weight params (NRD: two-path � simplified + view-dependent)
		// Simplified: uses diffuse lobe fraction for roughness-disabled path
		Float specNormalWeightParamSimplified = 1.0f / luisa::compute::max(
			luisa::compute::atan(lobeFraction), 1.5f / 255.0f);
		// Full: roughness-based cone angle with view-dependent weight
		Float specLobeFraction = c.gLobeAngleFraction;
		Float specRelaxation = luisa::compute::saturate(historyLength / 5.0f);
		specRelaxation = specRelaxation * lerp(1.0f, specConf, c.gNormalEdgeStoppingRelaxation);
		Float specF = 0.9f + 0.1f * specRelaxation;
		Float specTanHalf = centerRoughness * centerRoughness * specLobeFraction / (1.0f - specLobeFraction + 1e-6f);
		Float specAngleParam = luisa::compute::atan(specTanHalf);
		specAngleParam = specAngleParam * (10.0f - 9.0f * specRelaxation);
		specAngleParam = specAngleParam + c.gSpecLobeAngleSlack;
		specAngleParam = luisa::compute::min(specAngleParam, 1.5707963f);
		Float invSpecAngleParam = 1.0f / luisa::compute::max(specAngleParam, 1e-6f);

		// Luminance weight inverse sigma.
		// NRD RELAX_Atrous.cs.hlsl:104 — no history-length relaxation on luminance,
		// only on normal weights (diffuseLobeAngleFraction :50). Earlier code here
		// multiplied phiInv by lerp(0.2, 1.0, histLen/5); that was invented and
		// disabled luminance edge-stopping during rotation when histLen drops,
		// leaking noisy firefly neighbors into mirror surfaces.
		Float phiInv = 1.0f / luisa::compute::max(1e-4f,
			c.gDiffPhiLuminance * luisa::compute::sqrt(luisa::compute::max(centerVar, 0.0f)));

		// Specular luminance weight (NRD RELAX_Atrous.cs.hlsl:56)
		Float specPhiInv = 1.0f / luisa::compute::max(1e-4f,
			c.gSpecPhiLuminance * luisa::compute::sqrt(luisa::compute::max(centerSpec.w, 0.0f)));

		// Luminance edge-stopping relaxation (only for early atrous passes)
		Float specLumRelax = ite(cast<Float>(stepSize) <= 4.0f,
			lerp(1.0f, specConf, c.gLuminanceEdgeStoppingRelaxation), 1.0f);

		// Center world position (perspective)
		Float2 clipXY = (make_float2(pixelPos) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
		Float3 frustumFwd = c.gFrustumForward.xyz();
		Float3 frustumRight = c.gFrustumRight.xyz();
		Float3 frustumUp = c.gFrustumUp.xyz();
		Float3 rayDir = frustumFwd + frustumRight * clipXY.x - frustumUp * clipXY.y;
		Float viewZc = centerViewZ / luisa::compute::length(rayDir);
		Float3 centerWorldPos = viewZc * rayDir;
		Float3 centerV = -centerWorldPos / centerViewZ;

		// NRD RELAX_Atrous.cs.hlsl:133 — depth threshold is gDepthThreshold * centerViewZ.
		// Invented stepSize*3 floor removed: it widened the binary plane-distance test
		// at large step sizes, letting non-coplanar neighbors pass and leak noise into
		// the diffuse filter under rotation.
		Float depthThreshold = c.gDepthThreshold * centerViewZ;
		//Float invDepthThreshold = 1.0f / luisa::compute::max(depthThreshold, 1e-6f);

		// Accumulation: start with center pixel (Gaussian weight 0.44198^2)
		Float centerW = 0.44198f * 0.44198f;
		Float sumW = centerW;
		Float4 sw4 = make_float4(sumW);
		sw4.w = sw4.w * sw4.w;
		Float4 sumDiff = centerDiff * sw4;
		Float4 sumSpec = centerSpec * sw4;
		Float sumSpecW = centerW;

		// Random offset for large step sizes to reduce ringing
		Int2 offset = def(make_int2(0));
		$if(stepSize > 4u) {
			UInt h = pixelPos.x * 747558u + pixelPos.y * 19349669u + c.gFrameIndex * 2654435769u;
			h = h ^ (h >> 13u);
			h = h * 1274126177u;
			h = h ^ (h >> 16u);
			Float rx = cast<Float>(h & 0xFFFFu) / 65535.0f - 0.5f;
			h = h * 1103515245u + 12345u;
			Float ry = cast<Float>(h & 0xFFFFu) / 65535.0f - 0.5f;
			offset = make_int2(
				cast<int>(cast<Float>(stepSize) * 0.5f * rx),
				cast<int>(cast<Float>(stepSize) * 0.5f * ry)
			);
		};

		// Gaussian 3x3 kernel weights
		Float kw0 = 0.44198f;
		Float kw1 = 0.27901f;
		// Loop-invariant int bounds; XIR has no LICM pass, so hoist explicitly.
		Int rectWint = cast<int>(rectW) - 1;
		Int rectHint = cast<int>(rectH) - 1;

		// === Normal Atrous 3x3 filter ===
		// 3x3 neighborhood loop
		$for(j, -1, 2) {
			$for(i, -1, 2) {
				$if(i == 0 & j == 0) {
					$continue;
				};

				Int2 samplePosInt = make_int2(pixelPos) + offset +
					make_int2(i * cast<int>(stepSize),
						j * cast<int>(stepSize));

				samplePosInt = clamp(samplePosInt, make_int2(0), make_int2(rectWint, rectHint));
				UInt2 sp = make_uint2(samplePosInt);

				Float sampleViewZ = luisa::compute::abs(gIn_ViewZ.read(sp).x);
				$if(sampleViewZ > denoisingRange) {
					$continue;
				};

				Float4 sampleNR = gIn_Normal_Roughness.read(sp);
				Float3 sampleNormal = luisa::compute::normalize(sampleNR.xyz() * 2.0f - 1.0f);
				Float sampleMaterialID = luisa::compute::floor(sampleNR.w);
				Float sampleRoughness  = sampleNR.w - sampleMaterialID;

				Float2 sampleClipXY = (make_float2(sp) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
				Float3 sampleRayDir = frustumFwd + frustumRight * sampleClipXY.x - frustumUp * sampleClipXY.y;
				Float sampleViewZc = sampleViewZ / luisa::compute::length(sampleRayDir);
				Float3 sampleWorldPos = sampleViewZc * sampleRayDir;

				Float planeDist = luisa::compute::abs(
					luisa::compute::dot(sampleWorldPos - centerWorldPos, centerNormal));
				Float geometryW = ite(planeDist < depthThreshold, 1.0f, 0.0f);

				Float kernelW = ite(abs(i) == 0, kw0, kw1) * ite(abs(j) == 0, kw0, kw1);
				Float w = geometryW * kernelW;

				$if(w < 1e-4f) {
					$continue;
				};

				// Normal dot product (shared by diffuse + specular)
				Float cosN = luisa::compute::max(luisa::compute::dot(centerNormal, sampleNormal), -1.0f);
				Float angle = luisa::compute::acos(cosN);
				Float normalW = luisa::compute::clamp(
					1.0f - luisa::compute::smoothstep(0.0f, 1.0f, luisa::compute::abs(angle * normalWeightParam)),
					0.0f, 1.0f);
				Float diffMaterialW = ite(
					luisa::compute::max(centerMaterialID, c.gDiffMinMaterial) == luisa::compute::max(sampleMaterialID, c.gDiffMinMaterial),
					1.0f, 0.0f);
				Float wDiffuse = w * normalW * diffMaterialW;

				// Diffuse path (separate early-out)
				$if(wDiffuse > 1e-4f) {
					Float4 sampleDiff = in_Diff.read(sp);
					Float sampleLum = luminance(sampleDiff.xyz());
					Float lumW = luisa::compute::abs(centerLum - sampleLum) * phiInv;
					lumW = luisa::compute::min(lumW, c.gDiffMaxLuminanceRelativeDifference);
					wDiffuse = wDiffuse * luisa::compute::exp(-lumW);

					$if(wDiffuse > 1e-4f) {
						sumW = sumW + wDiffuse;
						Float4 dw4 = make_float4(wDiffuse);
						dw4.w = dw4.w * dw4.w;
						sumDiff = sumDiff + dw4 * sampleDiff;
					};
				};

				// Specular path (uniform $if to skip unused weight computation)
				Float wSpecPreLum = def(0.0f);
				Float materialW = ite(
					luisa::compute::max(centerMaterialID, c.gSpecMinMaterial) == luisa::compute::max(sampleMaterialID, c.gSpecMinMaterial),
					1.0f, 0.0f);
				$if(c.gRoughnessEdgeStoppingEnabled != 0u) {
					// NRD: relax view-direction rejection by mixing in centerWorldPos
					Float3 sampleV = -luisa::compute::normalize(sampleWorldPos + c.gRoughnessEdgeStoppingRelaxation * centerWorldPos);
					Float cosaV = luisa::compute::dot(centerV, sampleV);
					Float cosa = luisa::compute::min(cosN, cosaV);
					Float specA = luisa::compute::acos(luisa::compute::max(cosa, 0.0f));
					Float specT = luisa::compute::saturate(specA * invSpecAngleParam);
					Float specSmoothed = specT * specT * (3.0f - 2.0f * specT);
					Float normalWSpecular = luisa::compute::saturate(1.0f - specSmoothed * specF);
					// Roughness weight: NRD ComputeNonExponentialWeight = SmoothStep(1, 0, |ΔR|·a).
					Float _dR_spec = luisa::compute::saturate(
						luisa::compute::abs(sampleRoughness - centerRoughness) * roughA);
					Float roughnessWSpecular = 1.0f - _dR_spec * _dR_spec * (3.0f - 2.0f * _dR_spec);
					wSpecPreLum = w * normalWSpecular * roughnessWSpecular * materialW;
				} $else {
					Float normalWSpecSimplified = luisa::compute::clamp(
						1.0f - luisa::compute::smoothstep(0.0f, 1.0f, luisa::compute::abs(angle * specNormalWeightParamSimplified)),
						0.0f, 1.0f);
					wSpecPreLum = w * normalWSpecSimplified * materialW;
				};

				$if(wSpecPreLum > 1e-4f) {
					Float4 sampleSpec = in_Spec.read(sp);
					Float sampleSpecLum = luminance(sampleSpec.xyz());
					Float specLumW = luisa::compute::abs(centerSpecLum - sampleSpecLum) * specPhiInv;
					specLumW = luisa::compute::min(specLumW, c.gSpecMaxLuminanceRelativeDifference);
					specLumW *= specLumRelax;
					Float specW = wSpecPreLum * luisa::compute::exp(-specLumW);

					$if(specW > 1e-4f) {
						Float4 spw4 = make_float4(specW);
						spw4.w = spw4.w * spw4.w;
						sumSpec = sumSpec + spw4 * sampleSpec;
						sumSpecW = sumSpecW + specW;
					};
				};
			};
		};

		// Normalize diffuse
		Float sumWSafe = luisa::compute::max(sumW, 1e-6f);
		Float invW = 1.0f / sumWSafe;
		Float4 filtered = make_float4(
			sumDiff.x * invW,
			sumDiff.y * invW,
			sumDiff.z * invW,
			sumDiff.w / (sumWSafe * sumWSafe)
		);

		Float currHistoryLength = luisa::compute::max(historyLength - 1.0f, 0.0f);
		$if(isLastPass != 0u) {
			filtered.w = currHistoryLength;
		};

		out_Diff.write(pixelPos, filtered);

		// Normalize specular
		Float sumSpecWSafe = luisa::compute::max(sumSpecW, 1e-6f);
		Float specInvW = 1.0f / sumSpecWSafe;
		Float4 filteredSpec = make_float4(
			sumSpec.x * specInvW,
			sumSpec.y * specInvW,
			sumSpec.z * specInvW,
			sumSpec.w / (sumSpecWSafe * sumSpecWSafe)
		);
		$if(isLastPass != 0u) {
			filteredSpec.w = currHistoryLength;
		};
		out_Spec.write(pixelPos, filteredSpec);
	});
}
void RelaxDenoiser::compileAtrousSmem(Device& device) {
	_relaxAtrousSmem = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat out_Diff,
		ImageFloat in_Diff,
		ImageFloat gIn_Normal_Roughness,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_HistoryLength,
		UInt stepSize,
		UInt isLastPass,
		ImageFloat gIn_Tiles,
		// Specular (9-10): pass-through
		ImageFloat out_Spec,
		ImageFloat in_Spec
		) noexcept {
		set_block_size(8u, 8u, 1u);
		set_name("atrous_smem");

		constexpr uint B = 8u;
		constexpr uint BORDER = 2u;
		constexpr uint T = B + 2u * BORDER;  // 12
		constexpr uint TILE_TOTAL = T * T;   // 144
		constexpr uint BLOCK_TOTAL = B * B;  // 64

		Shared<float> nrTile(TILE_TOTAL * 4u);
		Shared<float> vzTile(TILE_TOTAL);
		Shared<float> diffTile(TILE_TOTAL * 4u);
		Shared<float> specTile(TILE_TOTAL * 4u);

		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);

		UInt rectW = cast<uint>(c.gRectSizeX);
		UInt rectH = cast<uint>(c.gRectSizeY);
		Float2 rectSizeInv = make_float2(c.gRectSizeInvX, c.gRectSizeInvY);
		Float denoisingRange = c.gDenoisingRange;

		UInt2 gid = block_id().xy();
		UInt flat_tid = pixelPos.x % B + (pixelPos.y % B) * B;

		// Cooperative preload: 64 threads load 144 entries in 2 stages
		$for(stage, 3u) {
			UInt si = flat_tid + stage * BLOCK_TOTAL;
			$if(si < TILE_TOTAL) {
				UInt ty = si / T;
				UInt tx = si - ty * T;
				Int gxx = cast<Int>(gid.x * B + tx) - cast<Int>(BORDER);
				Int gyy = cast<Int>(gid.y * B + ty) - cast<Int>(BORDER);
				UInt2 sp = make_uint2(
					cast<UInt>(max(gxx, 0)),
					cast<UInt>(max(gyy, 0)));
				sp = min(sp, make_uint2(rectW - 1u, rectH - 1u));
				Float4 nr = gIn_Normal_Roughness.read(sp);
				nrTile.write(si * 4u + 0u, nr.x);
				nrTile.write(si * 4u + 1u, nr.y);
				nrTile.write(si * 4u + 2u, nr.z);
				nrTile.write(si * 4u + 3u, nr.w);
				Float vz = gIn_ViewZ.read(sp).x;
				vzTile.write(si, vz);
				Float4 d = in_Diff.read(sp);
				diffTile.write(si * 4u + 0u, d.x);
				diffTile.write(si * 4u + 1u, d.y);
				diffTile.write(si * 4u + 2u, d.z);
				diffTile.write(si * 4u + 3u, d.w);
				Float4 s = in_Spec.read(sp);
				specTile.write(si * 4u + 0u, s.x);
				specTile.write(si * 4u + 1u, s.y);
				specTile.write(si * 4u + 2u, s.z);
				specTile.write(si * 4u + 3u, s.w);
			};
		};
		sync_block();

		// Tile-based early out
		UInt2 tilePos = pixelPos >> 4u;
		$if(Expr{ gIn_Tiles.read(tilePos).x != 0.0f } | pixelPos.x >= rectW | pixelPos.y >= rectH) {
			$return();
		};

		// Read center data from smem
		Int cx = cast<Int>(pixelPos.x) - cast<Int>(gid.x * B) + cast<Int>(BORDER);
		Int cy = cast<Int>(pixelPos.y) - cast<Int>(gid.y * B) + cast<Int>(BORDER);
		UInt cIdx = cast<UInt>(cx) + cast<UInt>(cy) * T;

		Float centerViewZ = luisa::compute::abs(vzTile.read(cIdx));
		$if(centerViewZ > denoisingRange) {
			$return();
		};

		Float4 centerNR = make_float4(
			nrTile.read(cIdx * 4u + 0u), nrTile.read(cIdx * 4u + 1u),
			nrTile.read(cIdx * 4u + 2u), nrTile.read(cIdx * 4u + 3u));
		Float3 centerNormal = luisa::compute::normalize(centerNR.xyz() * 2.0f - 1.0f);
		Float centerPackedNR   = centerNR.w;
		Float centerMaterialID = luisa::compute::floor(centerPackedNR);
		Float centerRoughness  = centerPackedNR - centerMaterialID;

		Float4 _hlRead2 = gIn_HistoryLength.read(pixelPos);
		Float historyLength = 255.0f * _hlRead2.x;
		Float specConf = _hlRead2.y;

		// Center diffuse + variance (from smem)
		Float4 centerDiff = make_float4(
			diffTile.read(cIdx * 4u + 0u), diffTile.read(cIdx * 4u + 1u),
			diffTile.read(cIdx * 4u + 2u), diffTile.read(cIdx * 4u + 3u));
		Float centerLum = luminance(centerDiff.xyz());
		Float centerVar = centerDiff.w;
		// Center specular (from smem)
		Float4 centerSpec = make_float4(
			specTile.read(cIdx * 4u + 0u), specTile.read(cIdx * 4u + 1u),
			specTile.read(cIdx * 4u + 2u), specTile.read(cIdx * 4u + 3u));
		Float centerSpecLum = luminance(centerSpec.xyz());

		// === Spatial Variance Estimation (5x5) for early-history pixels ===
		// NRD RELAX_AtrousSmem.cs.hlsl:375-482. When historyLength < gHistoryThreshold,
		// per-pixel temporal variance in .w is unreliable (not enough samples yet).
		// Recompute variance spatially from a 25-tap cross-bilateral neighborhood.
		// Reads come from smem (no extra bandwidth). Center is included in the loop;
		// its self-match weight is 1.0.
		$if(historyLength < c.gHistoryThreshold) {
			Float sveNormalWeightParam = 1.0f / luisa::compute::max(
				luisa::compute::atan(c.gLobeAngleFraction), 1.5f / 255.0f);

			Float sumDiffW   = def(0.0f);
			Float3 sumDiffRad = def(make_float3(0.0f));
			Float sumDiffM1  = def(0.0f);
			Float sumDiffM2  = def(0.0f);
			Float sumSpecW   = def(0.0f);
			Float3 sumSpecRad = def(make_float3(0.0f));
			Float sumSpecM1  = def(0.0f);
			Float sumSpecM2  = def(0.0f);

			$for(sdy, -2, 3) {
				$for(sdx, -2, 3) {
					Int ssx = cx + sdx;
					Int ssy = cy + sdy;
					UInt ssIdx = cast<UInt>(ssx) + cast<UInt>(ssy) * T;

					Float4 sampleNR = make_float4(
						nrTile.read(ssIdx * 4u + 0u), nrTile.read(ssIdx * 4u + 1u),
						nrTile.read(ssIdx * 4u + 2u), nrTile.read(ssIdx * 4u + 3u));
					Float3 sampleNormal = luisa::compute::normalize(sampleNR.xyz() * 2.0f - 1.0f);
					Float sampleMaterialID = luisa::compute::floor(sampleNR.w);

					Float cosN = luisa::compute::clamp(
						luisa::compute::dot(centerNormal, sampleNormal), -1.0f, 1.0f);
					Float angle = luisa::compute::acos(cosN);
					Float normalW = luisa::compute::clamp(
						1.0f - luisa::compute::smoothstep(0.0f, 1.0f,
							luisa::compute::abs(angle * sveNormalWeightParam)),
						0.0f, 1.0f);

					// Diffuse accumulation (material-gated)
					Bool diffMatOK = luisa::compute::max(centerMaterialID, c.gDiffMinMaterial)
						== luisa::compute::max(sampleMaterialID, c.gDiffMinMaterial);
					Float dw = normalW * ite(diffMatOK, 1.0f, 0.0f);
					Float4 sampleDiff = make_float4(
						diffTile.read(ssIdx * 4u + 0u), diffTile.read(ssIdx * 4u + 1u),
						diffTile.read(ssIdx * 4u + 2u), diffTile.read(ssIdx * 4u + 3u));
					Float sampleDiffLum = luminance(sampleDiff.xyz());
					sumDiffW   = sumDiffW   + dw;
					sumDiffRad = sumDiffRad + dw * sampleDiff.xyz();
					sumDiffM1  = sumDiffM1  + dw * sampleDiffLum;
					sumDiffM2  = sumDiffM2  + dw * sampleDiff.w;

					// Specular accumulation (material-gated)
					Bool specMatOK = luisa::compute::max(centerMaterialID, c.gSpecMinMaterial)
						== luisa::compute::max(sampleMaterialID, c.gSpecMinMaterial);
					Float sw = normalW * ite(specMatOK, 1.0f, 0.0f);
					Float4 sampleSpec = make_float4(
						specTile.read(ssIdx * 4u + 0u), specTile.read(ssIdx * 4u + 1u),
						specTile.read(ssIdx * 4u + 2u), specTile.read(ssIdx * 4u + 3u));
					Float sampleSpecLum = luminance(sampleSpec.xyz());
					sumSpecW   = sumSpecW   + sw;
					sumSpecRad = sumSpecRad + sw * sampleSpec.xyz();
					sumSpecM1  = sumSpecM1  + sw * sampleSpecLum;
					sumSpecM2  = sumSpecM2  + sw * sampleSpec.w;
				};
			};

			// NRD: boost widens variance box for very early history
			Float sveBoost = luisa::compute::max(1.0f, 4.0f / (historyLength + 1.0f));
			Float currHistoryLength = luisa::compute::max(historyLength - 1.0f, 0.0f);

			Float invDiffW = 1.0f / luisa::compute::max(sumDiffW, 1e-6f);
			Float3 avgDiffRad = sumDiffRad * invDiffW;
			Float avgDiffM1 = sumDiffM1 * invDiffW;
			Float avgDiffM2 = sumDiffM2 * invDiffW;
			Float diffVar = luisa::compute::max(0.0f,
				avgDiffM2 - avgDiffM1 * avgDiffM1) * sveBoost;
			Float4 diffOut = make_float4(avgDiffRad, diffVar);
			$if(isLastPass != 0u) {
				diffOut.w = currHistoryLength;
			};
			out_Diff.write(pixelPos, diffOut);

			Float invSpecW = 1.0f / luisa::compute::max(sumSpecW, 1e-6f);
			Float3 avgSpecRad = sumSpecRad * invSpecW;
			Float avgSpecM1 = sumSpecM1 * invSpecW;
			Float avgSpecM2 = sumSpecM2 * invSpecW;
			Float specVar = luisa::compute::max(0.0f,
				avgSpecM2 - avgSpecM1 * avgSpecM1) * sveBoost;
			Float4 specOut = make_float4(avgSpecRad, specVar);
			$if(isLastPass != 0u) {
				specOut.w = currHistoryLength;
			};
			out_Spec.write(pixelPos, specOut);

			$return();
		};

		// NRD RELAX_Atrous.cs.hlsl:120 — diffuse normal weight param uses lobeFraction
		// directly (no relaxation), matching _relaxAtrous above. The invented
		// normalRelaxFactor widened the acceptance cone by
		// gNormalEdgeStoppingRelaxation, letting cross-curvature neighbors leak
		// noise into the diffuse filter on metal under rotation.
		Float lobeFraction = c.gLobeAngleFraction / luisa::compute::sqrt(cast<Float>(stepSize));
		lobeFraction = lerp(0.99f, lobeFraction, luisa::compute::saturate(historyLength / 5.0f));

		Float normalWeightParam = 1.0f / luisa::compute::max(
			luisa::compute::atan(lobeFraction), 1.5f / 255.0f);

		// Roughness weight params (NRD: GetRoughnessWeightParams).
		Float roughSensitivity = 0.01f;
		Float roughA = 1.0f / luisa::compute::lerp(roughSensitivity, 1.0f,
			luisa::compute::saturate(centerRoughness * c.gRoughnessFraction));

		Float specNormalWeightParamSimplified = 1.0f / luisa::compute::max(
			luisa::compute::atan(lobeFraction), 1.5f / 255.0f);
		Float specLobeFraction = c.gLobeAngleFraction;
		Float specRelaxation = luisa::compute::saturate(historyLength / 5.0f);
		specRelaxation = specRelaxation * lerp(1.0f, specConf, c.gNormalEdgeStoppingRelaxation);
		Float specF = 0.9f + 0.1f * specRelaxation;
		Float specTanHalf = centerRoughness * centerRoughness * specLobeFraction / (1.0f - specLobeFraction + 1e-6f);
		Float specAngleParam = luisa::compute::atan(specTanHalf);
		specAngleParam = specAngleParam * (10.0f - 9.0f * specRelaxation);
		specAngleParam = specAngleParam + c.gSpecLobeAngleSlack;
		specAngleParam = luisa::compute::min(specAngleParam, 1.5707963f);
		Float invSpecAngleParam = 1.0f / luisa::compute::max(specAngleParam, 1e-6f);

		// Luminance weight inverse sigma (relaxed for short-history pixels)
		Float phiInv = 1.0f / luisa::compute::max(1e-4f,
			c.gDiffPhiLuminance * luisa::compute::sqrt(luisa::compute::max(centerVar, 0.0f)));

		// Specular luminance weight
		Float specPhiInv = 1.0f / luisa::compute::max(1e-4f,
			c.gSpecPhiLuminance * luisa::compute::sqrt(luisa::compute::max(centerSpec.w, 0.0f)));

		Float specLumRelax = ite(cast<Float>(stepSize) <= 4.0f,
			lerp(1.0f, specConf, c.gLuminanceEdgeStoppingRelaxation), 1.0f);

		// Center world position (perspective)
		Float2 clipXY = (make_float2(pixelPos) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
		Float3 frustumFwd = c.gFrustumForward.xyz();
		Float3 frustumRight = c.gFrustumRight.xyz();
		Float3 frustumUp = c.gFrustumUp.xyz();
		Float3 rayDir = frustumFwd + frustumRight * clipXY.x - frustumUp * clipXY.y;
		Float viewZc = centerViewZ / luisa::compute::length(rayDir);
		Float3 centerWorldPos = viewZc * rayDir;
		Float3 centerV = -centerWorldPos / centerViewZ;

		// NRD RELAX_Atrous.cs.hlsl:133 — depth threshold is gDepthThreshold * centerViewZ
		// (matches _relaxAtrous; the invented stepSize*3 floor widened the binary
		// plane-distance test at large step sizes, letting non-coplanar neighbors
		// pass and leak noise under rotation).
		Float depthThreshold = c.gDepthThreshold * centerViewZ;
		//Float invDepthThreshold = 1.0f / luisa::compute::max(depthThreshold, 1e-6f);

		// Accumulation: start with center pixel (Gaussian weight 0.44198^2)
		Float centerW = 0.44198f * 0.44198f;
		Float sumW = centerW;
		Float4 sw4 = make_float4(sumW);
		sw4.w = sw4.w * sw4.w;
		Float4 sumDiff = centerDiff * sw4;
		Float4 sumSpec = centerSpec * sw4;
		Float sumSpecW = centerW;

		// Random offset for large step sizes
		Int2 offset = def(make_int2(0));
		$if(stepSize > 4u) {
			UInt h = pixelPos.x * 747558u + pixelPos.y * 19349669u + c.gFrameIndex * 2654435769u;
			h = h ^ (h >> 13u);
			h = h * 1274126177u;
			h = h ^ (h >> 16u);
			Float rx = cast<Float>(h & 0xFFFFu) / 65535.0f - 0.5f;
			h = h * 1103515245u + 12345u;
			Float ry = cast<Float>(h & 0xFFFFu) / 65535.0f - 0.5f;
			offset = make_int2(
				cast<int>(cast<Float>(stepSize) * 0.5f * rx),
				cast<int>(cast<Float>(stepSize) * 0.5f * ry)
			);
		};

		// Gaussian 3x3 kernel weights
		Float kw0 = 0.44198f;
		Float kw1 = 0.27901f;
		// Loop-invariant int bounds; XIR has no LICM pass, so hoist explicitly.
		Int rectWint = cast<int>(rectW) - 1;
		Int rectHint = cast<int>(rectH) - 1;

		// 3x3 neighborhood loop (reads NR+ViewZ from smem)
		$for(j, -1, 2) {
			$for(i, -1, 2) {
				$if(i == 0 & j == 0) {
					$continue;
				};

				Int2 samplePosInt = make_int2(pixelPos) + offset +
					make_int2(i * cast<int>(stepSize),
						j * cast<int>(stepSize));

				samplePosInt = clamp(samplePosInt, make_int2(0), make_int2(rectWint, rectHint));
				UInt2 sp = make_uint2(samplePosInt);

				// Read from smem
				Int sx = cast<Int>(sp.x) - cast<Int>(gid.x * B) + cast<Int>(BORDER);
				Int sy = cast<Int>(sp.y) - cast<Int>(gid.y * B) + cast<Int>(BORDER);
				// Clamp smem coords for out-of-tile neighbors (can happen at edges)
				sx = clamp(sx, 0, cast<Int>(T - 1u));
				sy = clamp(sy, 0, cast<Int>(T - 1u));
				UInt sIdx = cast<UInt>(sx) + cast<UInt>(sy) * T;

				Float sampleViewZ = luisa::compute::abs(vzTile.read(sIdx));
				$if(sampleViewZ > denoisingRange) {
					$continue;
				};

				Float4 sampleNR = make_float4(
					nrTile.read(sIdx * 4u + 0u), nrTile.read(sIdx * 4u + 1u),
					nrTile.read(sIdx * 4u + 2u), nrTile.read(sIdx * 4u + 3u));
				Float3 sampleNormal = luisa::compute::normalize(sampleNR.xyz() * 2.0f - 1.0f);
				Float sampleMaterialID = luisa::compute::floor(sampleNR.w);
				Float sampleRoughness  = sampleNR.w - sampleMaterialID;

				Float2 sampleClipXY = (make_float2(sp) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
				Float3 sampleRayDir = frustumFwd + frustumRight * sampleClipXY.x - frustumUp * sampleClipXY.y;
				Float sampleViewZc = sampleViewZ / luisa::compute::length(sampleRayDir);
				Float3 sampleWorldPos = sampleViewZc * sampleRayDir;

				Float planeDist = luisa::compute::abs(
					luisa::compute::dot(sampleWorldPos - centerWorldPos, centerNormal));
				Float geometryW = ite(planeDist < depthThreshold, 1.0f, 0.0f);

				Float kernelW = ite(abs(i) == 0, kw0, kw1) * ite(abs(j) == 0, kw0, kw1);
				Float w = geometryW * kernelW;

				$if(w < 1e-4f) {
					$continue;
				};

				// Normal dot product (shared by diffuse + specular)
				Float cosN = luisa::compute::max(luisa::compute::dot(centerNormal, sampleNormal), -1.0f);
				Float angle = luisa::compute::acos(cosN);
				Float normalW = luisa::compute::clamp(
					1.0f - luisa::compute::smoothstep(0.0f, 1.0f, luisa::compute::abs(angle * normalWeightParam)),
					0.0f, 1.0f);
				Float diffMaterialW = ite(
					luisa::compute::max(centerMaterialID, c.gDiffMinMaterial) == luisa::compute::max(sampleMaterialID, c.gDiffMinMaterial),
					1.0f, 0.0f);
				Float wDiffuse = w * normalW * diffMaterialW;

				// Diffuse path
				$if(wDiffuse > 1e-4f) {
					Float4 sampleDiff = make_float4(
						diffTile.read(sIdx * 4u + 0u), diffTile.read(sIdx * 4u + 1u),
						diffTile.read(sIdx * 4u + 2u), diffTile.read(sIdx * 4u + 3u));
					Float sampleLum = luminance(sampleDiff.xyz());
					Float lumW = luisa::compute::abs(centerLum - sampleLum) * phiInv;
					lumW = luisa::compute::min(lumW, c.gDiffMaxLuminanceRelativeDifference);
					wDiffuse = wDiffuse * luisa::compute::exp(-lumW);

					$if(wDiffuse > 1e-4f) {
						sumW = sumW + wDiffuse;
						Float4 dw4 = make_float4(wDiffuse);
						dw4.w = dw4.w * dw4.w;
						sumDiff = sumDiff + dw4 * sampleDiff;
					};
				};

				// Specular path
				Float wSpecPreLum = def(0.0f);
				Float materialW = ite(
					luisa::compute::max(centerMaterialID, c.gSpecMinMaterial) == luisa::compute::max(sampleMaterialID, c.gSpecMinMaterial),
					1.0f, 0.0f);
				$if(c.gRoughnessEdgeStoppingEnabled != 0u) {
					// NRD: relax view-direction rejection by mixing in centerWorldPos
					Float3 sampleV = -luisa::compute::normalize(sampleWorldPos + c.gRoughnessEdgeStoppingRelaxation * centerWorldPos);
					Float cosaV = luisa::compute::dot(centerV, sampleV);
					Float cosa = luisa::compute::min(cosN, cosaV);
					Float specA = luisa::compute::acos(luisa::compute::max(cosa, 0.0f));
					Float specT = luisa::compute::saturate(specA * invSpecAngleParam);
					Float specSmoothed = specT * specT * (3.0f - 2.0f * specT);
					Float normalWSpecular = luisa::compute::saturate(1.0f - specSmoothed * specF);
					// Roughness weight: NRD ComputeNonExponentialWeight = SmoothStep(1, 0, |ΔR|·a).
					Float _dR_spec = luisa::compute::saturate(
						luisa::compute::abs(sampleRoughness - centerRoughness) * roughA);
					Float roughnessWSpecular = 1.0f - _dR_spec * _dR_spec * (3.0f - 2.0f * _dR_spec);
					wSpecPreLum = w * normalWSpecular * roughnessWSpecular * materialW;
				} $else {
					Float normalWSpecSimplified = luisa::compute::clamp(
						1.0f - luisa::compute::smoothstep(0.0f, 1.0f, luisa::compute::abs(angle * specNormalWeightParamSimplified)),
						0.0f, 1.0f);
					wSpecPreLum = w * normalWSpecSimplified * materialW;
				};

				$if(wSpecPreLum > 1e-4f) {
					Float4 sampleSpec = make_float4(
						specTile.read(sIdx * 4u + 0u), specTile.read(sIdx * 4u + 1u),
						specTile.read(sIdx * 4u + 2u), specTile.read(sIdx * 4u + 3u));
					Float sampleSpecLum = luminance(sampleSpec.xyz());
					Float specLumW = luisa::compute::abs(centerSpecLum - sampleSpecLum) * specPhiInv;
					specLumW = luisa::compute::min(specLumW, c.gSpecMaxLuminanceRelativeDifference);
					specLumW *= specLumRelax;
					Float specW = wSpecPreLum * luisa::compute::exp(-specLumW);

					$if(specW > 1e-4f) {
						Float4 spw4 = make_float4(specW);
						spw4.w = spw4.w * spw4.w;
						sumSpec = sumSpec + spw4 * sampleSpec;
						sumSpecW = sumSpecW + specW;
					};
				};
			};
		};

		// Normalize diffuse
		Float sumWSafe = luisa::compute::max(sumW, 1e-6f);
		Float invW = 1.0f / sumWSafe;
		Float4 filtered = make_float4(
			sumDiff.x * invW,
			sumDiff.y * invW,
			sumDiff.z * invW,
			sumDiff.w / (sumWSafe * sumWSafe)
		);

		Float currHistoryLength = luisa::compute::max(historyLength - 1.0f, 0.0f);
		$if(isLastPass != 0u) {
			filtered.w = currHistoryLength;
		};

		out_Diff.write(pixelPos, filtered);

		// Normalize specular
		Float sumSpecWSafe = luisa::compute::max(sumSpecW, 1e-6f);
		Float specInvW = 1.0f / sumSpecWSafe;
		Float4 filteredSpec = make_float4(
			sumSpec.x * specInvW,
			sumSpec.y * specInvW,
			sumSpec.z * specInvW,
			sumSpec.w / (sumSpecWSafe * sumSpecWSafe)
		);
		$if(isLastPass != 0u) {
			filteredSpec.w = currHistoryLength;
		};
		out_Spec.write(pixelPos, filteredSpec);
	});
}



}