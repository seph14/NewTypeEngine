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

// NRD RELAX_NORMAL_ULP: 8-bit normal encoding precision threshold.
// Below this angle, normal differences are considered encoding noise.
constexpr float RELAX_NORMAL_ULP = 1.5f / 255.0f;

void RelaxDenoiser::compileHitAndTemporal(Device& device) {
	_relaxHitDistReconstruct = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat gIn_SpecInput,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_Normal_Roughness,
		ImageFloat out_SpecHitDist
		) noexcept {
		// NRD RELAX_HitDistReconstruction.cs.hlsl pattern: 8x8 block + 1px halo smem preload.
		// Replaces 8 direct reads × 3 textures per pixel with smem reads (~6x reduction).
		constexpr uint B = 8u;
		constexpr uint BORDER = 1u;
		constexpr uint T = B + 2u * BORDER;      // 10
		constexpr uint TILE_TOTAL = T * T;       // 100
		constexpr uint BLOCK_TOTAL = B * B;      // 64

		set_block_size(B, B, 1u);
		set_name("hit_dist_reconstruct");

		UInt2 pixelPos = dispatch_id().xy();
		UInt2 resolution = dispatch_size().xy();

		// NRD: skip reconstruction when checkerboard is active (Relax.cpp:200).
		// Pass-through raw hit dist — smem preload and bilateral filter are skipped entirely.
		auto c = consts.read(0u);
		$if(c.gDiffCheckerboard != 0u) {
			$if(all(pixelPos < resolution)) {
				out_SpecHitDist.write(pixelPos, make_float4(gIn_SpecInput.read(pixelPos).w));
			};
			$return();
		};

		UInt2 gid = block_id().xy();
		UInt flat_tid = (pixelPos.x % B) + (pixelPos.y % B) * B;

		// Smem tiles: NR (4 floats), ViewZ (1), HitDist (1) per cell.
		Shared<float> nrTile(TILE_TOTAL * 4u);
		Shared<float> vzTile(TILE_TOTAL);
		Shared<float> hdTile(TILE_TOTAL);

		// Cooperative preload: 64 threads load 100 cells in 2 stages.
		// Edge-clamp globalPos to screen bounds (NRD Preload pattern).
		$for(stage, 2u) {
			UInt si = flat_tid + stage * BLOCK_TOTAL;
			$if(si < TILE_TOTAL) {
				UInt ty = si / T;
				UInt tx = si - ty * T;
				Int gxx = cast<Int>(gid.x * B + tx) - cast<Int>(BORDER);
				Int gyy = cast<Int>(gid.y * B + ty) - cast<Int>(BORDER);
				UInt2 sp = make_uint2(
					cast<UInt>(luisa::compute::max(gxx, 0)),
					cast<UInt>(luisa::compute::max(gyy, 0)));
				sp = min(sp, resolution - 1u);

				Float4 nr = gIn_Normal_Roughness.read(sp);
				nrTile.write(si * 4u + 0u, nr.x);
				nrTile.write(si * 4u + 1u, nr.y);
				nrTile.write(si * 4u + 2u, nr.z);
				nrTile.write(si * 4u + 3u, nr.w);
				vzTile.write(si, gIn_ViewZ.read(sp).x);
				hdTile.write(si, gIn_SpecInput.read(sp).w);
			};
		};
		sync_block();

		$if(any(pixelPos >= resolution)) { $return(); };

		// Smem coords of center
		Int cx = cast<Int>(pixelPos.x) - cast<Int>(gid.x * B) + cast<Int>(BORDER);
		Int cy = cast<Int>(pixelPos.y) - cast<Int>(gid.y * B) + cast<Int>(BORDER);

		UInt cIdx = cast<UInt>(cx) + cast<UInt>(cy) * T;
		Float centerHitDist = hdTile.read(cIdx);
		Float centerViewZ = vzTile.read(cIdx);
		Float4 centerNR = make_float4(
			nrTile.read(cIdx * 4u + 0u), nrTile.read(cIdx * 4u + 1u),
			nrTile.read(cIdx * 4u + 2u), nrTile.read(cIdx * 4u + 3u));
		Float3 centerNormal = luisa::compute::normalize(centerNR.xyz());

		// Gaussian 3x3 weights
		Float wSum = def(0.0f);
		Float hdSum = def(0.0f);

		$for(dy, -1, 2) {
			$for(dx, -1, 2) {
				Int sx = cx + dx;
				Int sy = cy + dy;
				UInt sIdx = cast<UInt>(sx) + cast<UInt>(sy) * T;

				Float sampleHitDist = hdTile.read(sIdx);
				Float sampleViewZ = vzTile.read(sIdx);
				Float4 sampleNR = make_float4(
					nrTile.read(sIdx * 4u + 0u), nrTile.read(sIdx * 4u + 1u),
					nrTile.read(sIdx * 4u + 2u), nrTile.read(sIdx * 4u + 3u));
				Float3 sampleNormal = luisa::compute::normalize(sampleNR.xyz());

				// Spatial weight: Gaussian 3x3
				Float spatialW = ite(dx == 0 & dy == 0, 0.44198f,
					ite(luisa::compute::abs(dx) + luisa::compute::abs(dy) == 1,
						0.27901f, 0.06198f));

				// Depth weight: plane-distance
				Float depthW = def(1.0f);
				$if(luisa::compute::abs(centerViewZ) > 1e-6f) {
					depthW = luisa::compute::exp(-luisa::compute::min(Expr{
						luisa::compute::abs(centerViewZ - sampleViewZ)
						/ (luisa::compute::abs(centerViewZ) * 0.01f + 1e-6f)
					}, 10.0f));
				};

				// Normal weight
				Float normalW = luisa::compute::clamp(dot(centerNormal, sampleNormal), 0.0f, 1.0f);
				normalW = ite(normalW < 0.8f, 0.0f, normalW);

				Float w = spatialW * depthW * normalW;
				wSum = wSum + w;
				hdSum = hdSum + w * sampleHitDist;
			};
		};

		Float smoothedHitDist = ite(wSum > 1e-6f, hdSum / wSum, centerHitDist);
		out_SpecHitDist.write(pixelPos, make_float4(smoothedHitDist));
	});

	//==========================================================================
	_relaxTemporalAccumulation = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_Normal_Roughness,
		ImageFloat gIn_Diff,
		ImageFloat gbuf_bary_motion,
		ImageUInt gbuf_vis,
		ImageFloat out_Diff,
		ImageFloat out_DiffFast,
		ImageFloat out_HistoryLength,
		ImageFloat out_Normal_Roughness,
		ImageFloat out_ViewZ,
		ImageFloat gIn_Tiles,
		ImageFloat gPrev_ViewZ,
		ImageFloat gPrev_Normal_Roughness,
		ImageFloat gPrev_DiffHistory,
		ImageFloat gPrev_DiffFastHistory,
		ImageFloat gPrev_HistoryLength,
		// Specular (16-21)
		ImageFloat gIn_Spec,
		ImageFloat out_Spec,
		ImageFloat out_SpecFast,
		ImageFloat gPrev_SpecHistory,
		ImageFloat gPrev_SpecFastHistory,
		ImageFloat gPrev_ViewZ2,
		// Virtual motion (22-24)
		ImageFloat gPrev_SpecHitDist,
		ImageFloat out_SpecHitDist,
		ImageFloat gIn_SpecHitDistSmoothed
		) noexcept {
		set_block_size(8u, 16u, 1u);
		set_name("temporal_accumulation");

		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);

		// Resolution as UInt for bounds checks, Float for UV math
		UInt  rectW = cast<uint>(c.gRectSizeX);
		UInt  rectH = cast<uint>(c.gRectSizeY);
		Float2 rectSizeInv = make_float2(c.gRectSizeInvX, c.gRectSizeInvY);
		Float2 rectSizePrev = make_float2(c.gRectSizePrevX, c.gRectSizePrevY);
		Float  denoisingRange = c.gDenoisingRange;
		UInt   resetHistory = c.gResetHistory;

		//----------------------------------------------------------------------
		// Shared-memory preload for 3x3 current-frame normal averaging
		// (NRD RELAX_TemporalAccumulation.cs.hlsl:22-23, 355-368 pattern).
		// Block is 8x16 (not 8x8), so tile is Tx=10 x Ty=18 with 1px halo =
		// 180 cells. 128 threads cooperatively load in 2 stages.
		// Replaces 8 direct gIn_Normal_Roughness reads per pixel with smem reads.
		// gIn_Spec.w (hitDist) stays direct — single channel, separate texture.
		//----------------------------------------------------------------------
		constexpr uint Bx = 8u;
		constexpr uint By = 16u;
		constexpr uint BORDER = 1u;
		constexpr uint Tx = Bx + 2u * BORDER;      // 10
		constexpr uint Ty = By + 2u * BORDER;      // 18
		constexpr uint TILE_TOTAL = Tx * Ty;       // 180
		constexpr uint BLOCK_TOTAL = Bx * By;      // 128

		UInt2 gid = block_id().xy();
		UInt flat_tid = (pixelPos.x % Bx) + (pixelPos.y % By) * Bx;

		Shared<float> normalRoughnessTile(TILE_TOTAL * 4u);

		$for(stage, 2u) {
			UInt si = flat_tid + stage * BLOCK_TOTAL;
			$if(si < TILE_TOTAL) {
				UInt ty = si / Tx;
				UInt tx = si - ty * Tx;
				Int gxx = cast<Int>(gid.x * Bx + tx) - cast<Int>(BORDER);
				Int gyy = cast<Int>(gid.y * By + ty) - cast<Int>(BORDER);
				UInt2 sp = make_uint2(
					cast<UInt>(luisa::compute::max(gxx, 0)),
					cast<UInt>(luisa::compute::max(gyy, 0)));
				sp = min(sp, make_uint2(rectW - 1u, rectH - 1u));
				Float4 nr = gIn_Normal_Roughness.read(sp);
				normalRoughnessTile.write(si * 4u + 0u, nr.x);
				normalRoughnessTile.write(si * 4u + 1u, nr.y);
				normalRoughnessTile.write(si * 4u + 2u, nr.z);
				normalRoughnessTile.write(si * 4u + 3u, nr.w);
			};
		};
		sync_block();

		// Tile-based early out
		UInt2 tilePos = pixelPos >> 4u;
		//Float isSky = gIn_Tiles.read(tilePos).x;
		$if(Expr{ gIn_Tiles.read(tilePos).x != 0.0f } | pixelPos.x >= rectW | pixelPos.y >= rectH) {
			// Sky/out-of-rect tile: history must read as "unrejectable background".
			// viewZ above denoisingRange (NOT 0) so downstream depth/plane tests
			// reject these texels, and matID 255 fails every material gate —
			// zero-viewZ/matID-0 history leaked into silhouette pixels via TA
			// bilinear, HistoryFix and the SVE 5x5 (jittering black edge pixels).
			Float _z = 0.0f;
			out_ViewZ.write(pixelPos, make_float4(3.402823466e38f));
			out_Diff.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_DiffFast.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_HistoryLength.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_Normal_Roughness.write(pixelPos, make_float4(0.5f, 0.5f, 1.0f, 255.0f));
			out_Spec.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_SpecFast.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_SpecHitDist.write(pixelPos, make_float4(_z));
			$return();
		};

		// Read current G-buffer
		Float currentViewZ = luisa::compute::abs(gIn_ViewZ.read(pixelPos).x);

		// Noisy input (read before sky early-out so we can pass-through)
		Float3 diffuseIllumination = gIn_Diff.read(pixelPos).xyz();
		Float4 specInput = gIn_Spec.read(pixelPos);
		Float3 specIllumination = specInput.xyz();
		Float specHitDist = gIn_SpecHitDistSmoothed.read(pixelPos).x; // bilateral-smoothed hit distance

		// Checkerboard bilateral resolve (NRD PrePass approach):
		// Fill inactive pixels from active neighbors using depth weights.
		// This ensures TA always receives valid input, avoiding zero-input
		// artifacts and the need for neighbor proxies in the boost logic.
		UInt cbField = c.gDiffCheckerboard;
		Bool cbActive = (cbField == 0u) | (((pixelPos.x + pixelPos.y) & 1u) == (cbField & 1u));
		$if(!cbActive) {
			UInt nbLx = ite(pixelPos.x > 0u, pixelPos.x - 1u, pixelPos.x + 1u);
			UInt nbRx = ite(pixelPos.x < rectW - 1u, pixelPos.x + 1u, pixelPos.x - 1u);
			Float vzL = luisa::compute::abs(gIn_ViewZ.read(make_uint2(nbLx, pixelPos.y)).x);
			Float vzR = luisa::compute::abs(gIn_ViewZ.read(make_uint2(nbRx, pixelPos.y)).x);
			Float dThresh = currentViewZ * 0.02f;
			Float wL = ite(luisa::compute::abs(vzL - currentViewZ) < dThresh, 1.0f, 0.0f);
			Float wR = ite(luisa::compute::abs(vzR - currentViewZ) < dThresh, 1.0f, 0.0f);
			Float ws = wL + wR;
			$if(ws > 0.0f) {
				Float invWs = 1.0f / ws;
				Float3 dL = gIn_Diff.read(make_uint2(nbLx, pixelPos.y)).xyz();
				Float3 dR = gIn_Diff.read(make_uint2(nbRx, pixelPos.y)).xyz();
				diffuseIllumination = (dL * wL + dR * wR) * invWs;
				Float3 sL = gIn_Spec.read(make_uint2(nbLx, pixelPos.y)).xyz();
				Float3 sR = gIn_Spec.read(make_uint2(nbRx, pixelPos.y)).xyz();
				specIllumination = (sL * wL + sR * wR) * invWs;
				Float hdL = gIn_SpecHitDistSmoothed.read(make_uint2(nbLx, pixelPos.y)).x;
				Float hdR = gIn_SpecHitDistSmoothed.read(make_uint2(nbRx, pixelPos.y)).x;
				specHitDist = (hdL * wL + hdR * wR) * invWs;
			};
		};

		// Sky / out-of-range pixels: write zeros and exit
		// Sky pixels don't need accumulation — bright HDR values persist
		$if(currentViewZ > denoisingRange) {
			Float _z = 0.0f; Float _inv255 = 1.0f / 255.0f;
			out_ViewZ.write(pixelPos, make_float4(currentViewZ));
			out_Diff.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_DiffFast.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_HistoryLength.write(pixelPos, make_float4(_inv255, _z, _z, _z));
			// matID 255 sentinel (matches prefilter) so material gates reject sky history
			out_Normal_Roughness.write(pixelPos, make_float4(0.5f, 0.5f, 1.0f, 255.0f));
			out_Spec.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_SpecFast.write(pixelPos, make_float4(_z, _z, _z, _z));
			out_SpecHitDist.write(pixelPos, make_float4(_z));
			$return();
		};

		Float4 currentNormalRoughness = gIn_Normal_Roughness.read(pixelPos);
		Float3 currentNormal = luisa::compute::normalize(currentNormalRoughness.xyz());
		// Encoded [-1,1] -> [0,1] form, used by both NR-packing write sites below.
		Float3 currentNormalEncoded = currentNormal * 0.5f + 0.5f;
		// .w packs matID (integer) + roughness (fraction). Unpack both.
		Float  currentPackedNR   = currentNormalRoughness.w;
		Float  currentMaterialID = luisa::compute::floor(currentPackedNR);
		Float  currentRoughness  = currentPackedNR - currentMaterialID;

		// 3x3 normal average for modified roughness (NRD: GetModifiedRoughnessFromNormalVariance)
		// When normals disagree (edges), increase effective roughness -> relax specular tracking
		Float3 currentNormalAvg = currentNormal;
		Float minHitDist3x3 = ite(specHitDist > 0.0f, specHitDist, 1e6f);
		// Cache neighbor normals at (x+1,y) and (x,y+1) for curvature block
		// (saves 2 redundant smem reads in virtual-motion setup).
		Float3 n10_cached = def(make_float3(0.0f, 0.0f, 1.0f));
		Float3 n01_cached = def(make_float3(0.0f, 0.0f, 1.0f));
		// rectW/rectH int bounds are loop-invariant; XIR has no LICM pass.
		Int rectWint = cast<int>(rectW) - 1;
		Int rectHint = cast<int>(rectH) - 1;
		// Center smem coord (matches _relaxHitDistReconstruct:87-89 pattern, adapted for 8x16 block).
		Int cx = cast<Int>(pixelPos.x) - cast<Int>(gid.x * Bx) + cast<Int>(BORDER);
		Int cy = cast<Int>(pixelPos.y) - cast<Int>(gid.y * By) + cast<Int>(BORDER);
		$for(dy, -1, 2) {
			$for(dx, -1, 2) {
				$if(dx == 0 & dy == 0) { $continue; };
				UInt sx = cast<UInt>(cx + dx);
				UInt sy = cast<UInt>(cy + dy);
				UInt sIdx = sx + sy * Tx;
				Float4 nrSample = make_float4(
					normalRoughnessTile.read(sIdx * 4u + 0u),
					normalRoughnessTile.read(sIdx * 4u + 1u),
					normalRoughnessTile.read(sIdx * 4u + 2u),
					normalRoughnessTile.read(sIdx * 4u + 3u));
				Float3 nSample = luisa::compute::normalize(nrSample.xyz());
				currentNormalAvg = currentNormalAvg + nSample;
				// Stash nSample for the (1,0) and (0,1) iterations so the curvature
				// block can reuse them instead of re-reading smem.
				n10_cached = ite(dx == 1 & dy == 0, nSample, n10_cached);
				n01_cached = ite(dx == 0 & dy == 1, nSample, n01_cached);
				// gIn_Spec.w (hit distance) stays direct — single channel, separate texture.
				UInt2 nPos = make_uint2(
					Expr{ cast<uint>(clamp(cast<int>(pixelPos.x) + dx, 0, rectWint)) },
					Expr{ cast<uint>(clamp(cast<int>(pixelPos.y) + dy, 0, rectHint)) });
				Float sampleHitDist = gIn_Spec.read(nPos).w;
				$if(sampleHitDist > 0.0f) {
					minHitDist3x3 = luisa::compute::min(minHitDist3x3, sampleHitDist);
				};
			};
		};
		currentNormalAvg = currentNormalAvg / 9.0f;
		Float normalVariance = 1.0f - luisa::compute::dot(currentNormalAvg, currentNormalAvg);
		Float currentRoughnessModified = luisa::compute::min(
			currentRoughness + luisa::compute::sqrt(luisa::compute::max(normalVariance, 0.0f)) * 0.5f,
			1.0f);

		// 2nd moment of noisy luminance
		Float lum = luminance(diffuseIllumination);
		Float diffuse2ndMoment = lum * lum;

		// Compute prevUV from motion vectors (NDC -> UV)
		Float2 pixelUv = (make_float2(pixelPos) + 0.5f) * rectSizeInv;
		Float2 mv_raw = gbuf_bary_motion.read(pixelPos).zw();
		// Same convention as DI/GI temporal reuse: motion is in screen-NDC (Y-down),
		// so prevUV = pixelUV + motion * 0.5 (no Y negation)
		Float2 prevUV = pixelUv + mv_raw * 0.5f;

		// World position and view vector (needed for NoV disocclusion + virtual motion)
		Float2 clipXY = pixelUv * 2.0f - 1.0f;
		Float3 currentWorldPos = c.gFrustumForward.xyz() * currentViewZ
			+ c.gFrustumRight.xyz() * clipXY.x
			+ c.gFrustumUp.xyz() * clipXY.y;
		Float3 V = luisa::compute::normalize(-currentWorldPos);
		Float NoV = luisa::compute::abs(luisa::compute::dot(currentNormal, V));

		// Out-of-screen early out: pass through noisy input as-is
		// (no prev-frame data available; HistoryFix will fill from neighbors)
		$if(prevUV.x < 0.0f | prevUV.x > 1.0f | prevUV.y < 0.0f | prevUV.y > 1.0f) {
			out_Diff.write(pixelPos, make_float4(diffuseIllumination, diffuse2ndMoment));
			out_DiffFast.write(pixelPos, make_float4(diffuseIllumination, 0.0f));
			Float _oneOver255 = 1.0f / 255.0f;
			out_HistoryLength.write(pixelPos, make_float4(_oneOver255, 0.0f, 0.0f, 0.0f));
			out_Normal_Roughness.write(pixelPos, make_float4(currentNormalEncoded, currentPackedNR));
			Float specLum = luminance(specIllumination);
			out_ViewZ.write(pixelPos, make_float4(currentViewZ));
			out_Spec.write(pixelPos, make_float4(specIllumination, specLum * specLum));
			out_SpecFast.write(pixelPos, make_float4(specIllumination, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			$return();
		};

		// === 4x4 depth grid for bicubic validation (NRD-aligned) ===
			Float2 prevPixelPosBase = prevUV * rectSizePrev;
			Float2 flooredBase = luisa::compute::floor(prevPixelPosBase - 0.5f);
			UInt  oxBase = cast<uint>(luisa::compute::max(flooredBase.x, 0.0f));
			UInt  oyBase = cast<uint>(luisa::compute::max(flooredBase.y, 0.0f));
			UInt maxOxBase = cast<uint>(rectSizePrev.x) - 2u;
			UInt maxOyBase = cast<uint>(rectSizePrev.y) - 2u;
			oxBase = clamp(oxBase, 0u, maxOxBase);
			oyBase = clamp(oyBase, 0u, maxOyBase);

			// Disocclusion threshold with NoV slope scaling
			Float disocclusionThreshold = c.gDisocclusionThreshold;
			Float pixelSize = c.gUnproject * currentViewZ;
			Float frustumSize = pixelSize * luisa::compute::min(rectSizePrev.x, rectSizePrev.y);
			Float parallaxInPixels = luisa::compute::length(mv_raw * make_float2(rectSizePrev) * 0.5f);
			Float slopeScale = 1.0f / luisa::compute::lerp(
				luisa::compute::lerp(0.05f, 1.0f, NoV),
				1.0f,
				luisa::compute::saturate(parallaxInPixels / c.gDisocclusionParallaxDenominator));
			Float baseThresh = luisa::compute::saturate(disocclusionThreshold * slopeScale) * frustumSize;
			Float threshold = luisa::compute::max(baseThresh - 1e-4f, 0.0f);

			// Compute prev-frame view-Z of the current world point. currentWorldPos
			// is reconstructed in current-camera-centered coords, so adding
			// gCameraDelta (= cam_pos - prev_cam_pos) converts to prev-camera-centered
			// coords, then dot with prevFrustumForward extracts the prev view-Z.
			// This handles BOTH camera translation AND rotation — the previous
			// `currentViewZ + dot(gCameraDelta, gFrustumForward) * edgeGating` form
			// was a translation-only approximation that produced false-positive
			// disocclusions under camera rotation (especially at grazing angle),
			// causing mass history resets and one-frame flashes during fast rotation.
			Float3 prevCamRelativePos = currentWorldPos + c.gCameraDelta.xyz();
			Float depthRef = luisa::compute::dot(prevCamRelativePos, c.gPrevFrustumForward.xyz());

			// Grid positions: bilinear origin +/- 1, clamped to screen
			Int obx = cast<int>(oxBase);
			Int oby = cast<int>(oyBase);
			Int rsw = cast<int>(rectSizePrev.x) - 1;
			Int rsh = cast<int>(rectSizePrev.y) - 1;
			UInt gc0 = cast<uint>(luisa::compute::max(obx - 1, 0));
			UInt gc1 = oxBase;
			UInt gc2 = oxBase + 1u;
			UInt gc3 = cast<uint>(luisa::compute::min(obx + 2, rsw));
			UInt gr0 = cast<uint>(luisa::compute::max(oby - 1, 0));
			UInt gr1 = oyBase;
			UInt gr2 = oyBase + 1u;
			UInt gr3 = cast<uint>(luisa::compute::min(oby + 2, rsh));

			// Read 4x4 depth grid (16 values)
			Float4 depthR0 = make_float4(
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc0, gr0)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc1, gr0)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc2, gr0)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc3, gr0)).x));
			Float4 depthR1 = make_float4(
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc0, gr1)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc1, gr1)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc2, gr1)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc3, gr1)).x));
			Float4 depthR2 = make_float4(
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc0, gr2)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc1, gr2)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc2, gr2)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc3, gr2)).x));
			Float4 depthR3 = make_float4(
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc0, gr3)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc1, gr3)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc2, gr3)).x),
				luisa::compute::abs(gPrev_ViewZ.read(make_uint2(gc3, gr3)).x));

			// Augment threshold with local depth range across the 4x4 grid.
			// NRD's default gDisocclusionThreshold is calibrated for typical scenes; large
			// grazing-angle surfaces (e.g. floor) have naturally large depth gradients across
			// the 4x4 footprint that exceed what NRD's slopeScale accounts for. Without this
			// augmentation, the per-tap validity mask flips frame-to-frame at grazing+rotation,
			// causing the bicubic<->bilinear choice and bilinear tap weights to flap in lockstep
			// (visible as a vertical zigzag on mid-roughness surfaces during fast rotation).
			// Half the observed range accepts taps within the local surface neighborhood while
			// still rejecting cross-surface matches (which typically exceed half the range).
			Float4 dRowMin = make_float4(
				luisa::compute::min(luisa::compute::min(depthR0.x, depthR0.y), luisa::compute::min(depthR0.z, depthR0.w)),
				luisa::compute::min(luisa::compute::min(depthR1.x, depthR1.y), luisa::compute::min(depthR1.z, depthR1.w)),
				luisa::compute::min(luisa::compute::min(depthR2.x, depthR2.y), luisa::compute::min(depthR2.z, depthR2.w)),
				luisa::compute::min(luisa::compute::min(depthR3.x, depthR3.y), luisa::compute::min(depthR3.z, depthR3.w)));
			Float4 dRowMax = make_float4(
				luisa::compute::max(luisa::compute::max(depthR0.x, depthR0.y), luisa::compute::max(depthR0.z, depthR0.w)),
				luisa::compute::max(luisa::compute::max(depthR1.x, depthR1.y), luisa::compute::max(depthR1.z, depthR1.w)),
				luisa::compute::max(luisa::compute::max(depthR2.x, depthR2.y), luisa::compute::max(depthR2.z, depthR2.w)),
				luisa::compute::max(luisa::compute::max(depthR3.x, depthR3.y), luisa::compute::max(depthR3.z, depthR3.w)));
			Float gridMin = luisa::compute::min(luisa::compute::min(dRowMin.x, dRowMin.y), luisa::compute::min(dRowMin.z, dRowMin.w));
			Float gridMax = luisa::compute::max(luisa::compute::max(dRowMax.x, dRowMax.y), luisa::compute::max(dRowMax.z, dRowMax.w));
			threshold = luisa::compute::max(threshold, gridMax - gridMin);

			// Per-tap depth validity (16 taps, 1.0 if within threshold)
			Float4 depthRef4 = make_float4(depthRef);
			Float4 dvR0 = ite(luisa::compute::abs(depthR0 - depthRef4) < threshold, 1.0f, 0.0f);
			Float4 dvR1 = ite(luisa::compute::abs(depthR1 - depthRef4) < threshold, 1.0f, 0.0f);
			Float4 dvR2 = ite(luisa::compute::abs(depthR2 - depthRef4) < threshold, 1.0f, 0.0f);
			Float4 dvR3 = ite(luisa::compute::abs(depthR3 - depthRef4) < threshold, 1.0f, 0.0f);

			// Zero corners for bicubic validity (positions 0,0 / 3,0 / 0,3 / 3,3)
			Float4 bvR0 = make_float4(0.0f, dvR0.y, dvR0.z, 0.0f);
			Float4 bvR1 = dvR1;
			Float4 bvR2 = dvR2;
			Float4 bvR3 = make_float4(0.0f, dvR3.y, dvR3.z, 0.0f);
			Float bicubicValidCount = dot(bvR0 + bvR1 + bvR2 + bvR3, make_float4(1.0f));
			Bool bicubicFootprintValid = bicubicValidCount > 11.5f;

			// Bilinear taps: center 2x2 = depthR1.yz, depthR2.yz
			Float4 bilinearTapDepthValid = make_float4(dvR1.y, dvR1.z, dvR2.y, dvR2.z);

			// Center normal for backface rejection (NRD: bilinear-filtered at center)
			Float4 cnr00 = gPrev_Normal_Roughness.read(make_uint2(gc1, gr1));
			Float4 cnr10 = gPrev_Normal_Roughness.read(make_uint2(gc2, gr1));
			Float4 cnr01 = gPrev_Normal_Roughness.read(make_uint2(gc1, gr2));
			Float4 cnr11 = gPrev_Normal_Roughness.read(make_uint2(gc2, gr2));
			Float3 prevNormalPackedAvg = (cnr00.xyz() + cnr10.xyz() + cnr01.xyz() + cnr11.xyz()) * 0.25f;
			Float3 prevNormalFlat = luisa::compute::normalize(prevNormalPackedAvg * 2.0f - 1.0f);
			Float3 prevNormalRotated = transform_normal(c.gWorldPrevToWorld, prevNormalFlat);
			Bool backfaceValid = luisa::compute::dot(currentNormal, prevNormalRotated) >= 0.0f;

			$if(!backfaceValid) {
				bicubicFootprintValid = false;
				bilinearTapDepthValid = make_float4(0.0f);
			};

			// === Material-ID mask for bicubic fetch (prevents history mixing across boundaries) ===
			// 8 additional reads for non-center taps; center 2x2 reuses cnr00..11.w
			Float mv_10 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc1, gr0)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_20 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc2, gr0)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_01 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc0, gr1)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_11 = ite(luisa::compute::abs(luisa::compute::floor(cnr00.w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_21 = ite(luisa::compute::abs(luisa::compute::floor(cnr10.w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_31 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc3, gr1)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_02 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc0, gr2)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_12 = ite(luisa::compute::abs(luisa::compute::floor(cnr01.w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_22 = ite(luisa::compute::abs(luisa::compute::floor(cnr11.w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_32 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc3, gr2)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_13 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc1, gr3)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			Float mv_23 = ite(luisa::compute::abs(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc2, gr3)).w) - currentMaterialID) < 0.5f, 1.0f, 0.0f);
			// Material-weighted count: drop bicubic at material boundaries, fall back to bilinear
			Float bicubicMatCount = mv_10 + mv_20 + mv_01 + mv_11 + mv_21 + mv_31
				+ mv_02 + mv_12 + mv_22 + mv_32 + mv_13 + mv_23;
			$if(bicubicMatCount < 11.5f) {
				bicubicFootprintValid = false;
			};

			// === Jittered position for bilinear fallback ===
			Float2 prevPixelPosFloat = prevUV * rectSizePrev;
			UInt sr_seed = util::xxhash32(make_uint3(pixelPos.x | (pixelPos.y << 16u), c.gFrameIndex, 42u));
			Float mvLen = luisa::compute::length(mv_raw);
			// Cap jitter amplitude at 0.3 (empirically tuned). Full ±0.5px jitter at high
			// motion amplifies per-tap depth-mask flickering into visible noise on smooth mid-roughness
			// surfaces. 0.3 retains the banding-dither function for disocclusion boundaries
			// while avoiding the motion-correlated zigzag during fast rotation.
			Float jitterScale = luisa::compute::lerp(0.1f, 0.3f, luisa::compute::saturate(mvLen * 200.0f));
			Float jitter_x = util::uniform_uint_to_float(sr_seed) * 0.5f * jitterScale;
			Float jitter_y = util::uniform_uint_to_float(util::xxhash32(make_uint2(sr_seed, 1u))) * 0.5f * jitterScale;
			prevPixelPosFloat = prevPixelPosFloat + make_float2(jitter_x, jitter_y);
			Float2 floored = luisa::compute::floor(prevPixelPosFloat - 0.5f);
			UInt  ox = cast<uint>(luisa::compute::max(floored.x, 0.0f));
			UInt  oy = cast<uint>(luisa::compute::max(floored.y, 0.0f));
			UInt maxOx = cast<uint>(rectSizePrev.x) - 2u;
			UInt maxOy = cast<uint>(rectSizePrev.y) - 2u;
			ox = clamp(ox, 0u, maxOx);
			oy = clamp(oy, 0u, maxOy);
			Float2 bilinearWeights = luisa::compute::fract(prevPixelPosFloat - 0.5f);

			UInt2 p00 = make_uint2(ox, oy);
			UInt2 p10 = make_uint2(ox + 1u, oy);
			UInt2 p01 = make_uint2(ox, oy + 1u);
			UInt2 p11 = make_uint2(ox + 1u, oy + 1u);

			// Glass flag matching at jittered bilinear taps
			Float4 nr00 = gPrev_Normal_Roughness.read(p00);
			Float4 nr10 = gPrev_Normal_Roughness.read(p10);
			Float4 nr01 = gPrev_Normal_Roughness.read(p01);
			Float4 nr11 = gPrev_Normal_Roughness.read(p11);
			Float4 prevMatID = make_float4(
				luisa::compute::floor(nr00.w),
				luisa::compute::floor(nr10.w),
				luisa::compute::floor(nr01.w),
				luisa::compute::floor(nr11.w));
			Float4 gv = ite(luisa::compute::abs(prevMatID - currentMaterialID) < 0.5f, 1.0f, 0.0f);

			// Bilinear custom weights with per-tap validity
			Float bw_x = bilinearWeights.x;
			Float bw_y = bilinearWeights.y;
			Float4 cw = make_float4(
				(1.0f - bw_x) * (1.0f - bw_y),
				bw_x * (1.0f - bw_y),
				(1.0f - bw_x) * bw_y,
				bw_x * bw_y
			) * bilinearTapDepthValid * gv;
			Float sumW = dot(cw, make_float4(1.0f));

			// 3-tier fallback logic
			Bool reprojectionFound = (sumW > c.gReprojectionConfidence);
			Bool useBicubic = bicubicFootprintValid & reprojectionFound;
			$if(resetHistory != 0u) {
				reprojectionFound = false;
				useBicubic = false;
			};

			// === Diffuse history fetch (bicubic or bilinear) ===
			Float4 prevDiff = def(make_float4(0.0f));
			Float3 prevDiffFast = def(make_float3(0.0f));
			Float historyLength = 0.0f;

			$if(useBicubic) {
				// === Catmull-Rom weights for bicubic fetch ===
				Float2 fracBase = luisa::compute::fract(prevPixelPosBase - 0.5f);
				Float fx = fracBase.x;
				Float fy = fracBase.y;
				Float wx0 = fx * (fx * (-0.5f * fx + 1.0f) - 0.5f);
				Float wx1 = fx * (fx * (1.5f * fx - 2.5f)) + 1.0f;
				Float wx2 = fx * (fx * (-1.5f * fx + 2.0f) + 0.5f);
				Float wx3 = fx * (fx * (0.5f * fx - 0.5f));
				Float wy0 = fy * (fy * (-0.5f * fy + 1.0f) - 0.5f);
				Float wy1 = fy * (fy * (1.5f * fy - 2.5f)) + 1.0f;
				Float wy2 = fy * (fy * (-1.5f * fy + 2.0f) + 0.5f);
				Float wy3 = fy * (fy * (0.5f * fy - 0.5f));

				// 12-tap no-corners Catmull-Rom weights
				Float w_10 = wx1 * wy0 * mv_10;
				Float w_20 = wx2 * wy0 * mv_20;
				Float w_01 = wx0 * wy1 * mv_01;
				Float w_11 = wx1 * wy1 * mv_11;
				Float w_21 = wx2 * wy1 * mv_21;
				Float w_31 = wx3 * wy1 * mv_31;
				Float w_02 = wx0 * wy2 * mv_02;
				Float w_12 = wx1 * wy2 * mv_12;
				Float w_22 = wx2 * wy2 * mv_22;
				Float w_32 = wx3 * wy2 * mv_32;
				Float w_13 = wx1 * wy3 * mv_13;
				Float w_23 = wx2 * wy3 * mv_23;
				Float bicubicWSum = w_10 + w_20 + w_01 + w_11 + w_21 + w_31
					+ w_02 + w_12 + w_22 + w_32 + w_13 + w_23;
				// 12-tap Catmull-Rom fetch at unjittered grid positions
				Float4 d_10 = gPrev_DiffHistory.read(make_uint2(gc1, gr0));
				Float4 d_20 = gPrev_DiffHistory.read(make_uint2(gc2, gr0));
				Float4 d_01 = gPrev_DiffHistory.read(make_uint2(gc0, gr1));
				Float4 d_11 = gPrev_DiffHistory.read(make_uint2(gc1, gr1));
				Float4 d_21 = gPrev_DiffHistory.read(make_uint2(gc2, gr1));
				Float4 d_31 = gPrev_DiffHistory.read(make_uint2(gc3, gr1));
				Float4 d_02 = gPrev_DiffHistory.read(make_uint2(gc0, gr2));
				Float4 d_12 = gPrev_DiffHistory.read(make_uint2(gc1, gr2));
				Float4 d_22 = gPrev_DiffHistory.read(make_uint2(gc2, gr2));
				Float4 d_32 = gPrev_DiffHistory.read(make_uint2(gc3, gr2));
				Float4 d_13 = gPrev_DiffHistory.read(make_uint2(gc1, gr3));
				Float4 d_23 = gPrev_DiffHistory.read(make_uint2(gc2, gr3));
				Float4 bicubicDiff = d_10 * w_10 + d_20 * w_20
					+ d_01 * w_01 + d_11 * w_11 + d_21 * w_21 + d_31 * w_31
					+ d_02 * w_02 + d_12 * w_12 + d_22 * w_22 + d_32 * w_32
					+ d_13 * w_13 + d_23 * w_23;
				prevDiff = bicubicDiff / bicubicWSum;

				// Fast diffuse
				Float3 df_10 = gPrev_DiffFastHistory.read(make_uint2(gc1, gr0)).xyz();
				Float3 df_20 = gPrev_DiffFastHistory.read(make_uint2(gc2, gr0)).xyz();
				Float3 df_01 = gPrev_DiffFastHistory.read(make_uint2(gc0, gr1)).xyz();
				Float3 df_11 = gPrev_DiffFastHistory.read(make_uint2(gc1, gr1)).xyz();
				Float3 df_21 = gPrev_DiffFastHistory.read(make_uint2(gc2, gr1)).xyz();
				Float3 df_31 = gPrev_DiffFastHistory.read(make_uint2(gc3, gr1)).xyz();
				Float3 df_02 = gPrev_DiffFastHistory.read(make_uint2(gc0, gr2)).xyz();
				Float3 df_12 = gPrev_DiffFastHistory.read(make_uint2(gc1, gr2)).xyz();
				Float3 df_22 = gPrev_DiffFastHistory.read(make_uint2(gc2, gr2)).xyz();
				Float3 df_32 = gPrev_DiffFastHistory.read(make_uint2(gc3, gr2)).xyz();
				Float3 df_13 = gPrev_DiffFastHistory.read(make_uint2(gc1, gr3)).xyz();
				Float3 df_23 = gPrev_DiffFastHistory.read(make_uint2(gc2, gr3)).xyz();
				Float3 bicubicDiffFast = df_10 * w_10 + df_20 * w_20
					+ df_01 * w_01 + df_11 * w_11 + df_21 * w_21 + df_31 * w_31
					+ df_02 * w_02 + df_12 * w_12 + df_22 * w_22 + df_32 * w_32
					+ df_13 * w_13 + df_23 * w_23;
				prevDiffFast = bicubicDiffFast / bicubicWSum;

			} $else {
				// Bilinear fallback at jittered positions
				Float4 pd00 = gPrev_DiffHistory.read(p00);
				Float4 pd10 = gPrev_DiffHistory.read(p10);
				Float4 pd01 = gPrev_DiffHistory.read(p01);
				Float4 pd11 = gPrev_DiffHistory.read(p11);
				$if(reprojectionFound) {
					prevDiff = (pd00 * cw.x + pd10 * cw.y + pd01 * cw.z + pd11 * cw.w) / sumW;
				};

				Float4 pf00 = gPrev_DiffFastHistory.read(p00);
				Float4 pf10 = gPrev_DiffFastHistory.read(p10);
				Float4 pf01 = gPrev_DiffFastHistory.read(p01);
				Float4 pf11 = gPrev_DiffFastHistory.read(p11);
				$if(reprojectionFound) {
					prevDiffFast = (pf00 * cw.x + pf10 * cw.y + pf01 * cw.z + pf11 * cw.w).xyz() / sumW;
				};

			};

			// History length: always bilinear (NRD: low-frequency signal)
			Float hl00 = gPrev_HistoryLength.read(p00).x;
			Float hl10 = gPrev_HistoryLength.read(p10).x;
			Float hl01 = gPrev_HistoryLength.read(p01).x;
			Float hl11 = gPrev_HistoryLength.read(p11).x;
			$if(reprojectionFound) {
				historyLength = (hl00 * cw.x + hl10 * cw.y + hl01 * cw.z + hl11 * cw.w) / sumW;
			};

			prevDiff = luisa::compute::max(prevDiff, make_float4(0.0f));
			prevDiffFast = luisa::compute::max(prevDiffFast, make_float3(0.0f));
			historyLength = historyLength * 255.0f + 1.0f;

			// Shorten history if partial reprojection
			Float footprintQuality = ite(useBicubic, 1.0f, ite(reprojectionFound, sumW, 0.0f));

		// NoV-based size scaling (NRD TA lines 557-563): prevent momentary stretching
		// when viewing angle changes. sizeQuality < 1 means the footprint shrunk, reducing history.
		Float3 prevWorldPos = currentWorldPos + c.gCameraDelta.xyz();
		Float3 Vprev = luisa::compute::normalize(-prevWorldPos);
		Float NoVprev = luisa::compute::abs(luisa::compute::dot(currentNormal, Vprev));
		Float sizeQuality = (NoVprev + 1e-3f) / (NoV + 1e-3f);
		sizeQuality = sizeQuality * sizeQuality;
		sizeQuality = sizeQuality * sizeQuality; // ^4
		footprintQuality = footprintQuality * luisa::compute::lerp(0.1f, 1.0f,
			luisa::compute::saturate(sizeQuality));

		$if(footprintQuality < 1.0f) {
			historyLength = historyLength * luisa::compute::sqrt(luisa::compute::max(footprintQuality, 0.0f));
			historyLength = luisa::compute::max(historyLength, 1.0f);
		};

		// Handle reset
		$if(resetHistory != 0u) {
			historyLength = 1.0f;
		};

		// Cap history length
		Float maxAccum = 1.0f + c.gDiffMaxAccumulatedFrameNum;
		historyLength = luisa::compute::min(historyLength, maxAccum);

		// Dual-rate EMA (time-based steady-state alpha)
		Float diffAlpha = ite(reprojectionFound,
			luisa::compute::max(c.gDiffAlphaSteady, 1.0f / historyLength), 1.0f);
		Float diffAlphaFast = ite(reprojectionFound,
			luisa::compute::max(c.gDiffAlphaFastSteady, 1.0f / historyLength), 1.0f);

		// Reduce alpha for inactive checkerboard pixels (NRD approach)
		Float cbResolveSpeed = 0.5f;
		$if(!cbActive & historyLength > 1.0f) {
			diffAlpha *= (1.0f - cbResolveSpeed);
			diffAlphaFast *= (1.0f - cbResolveSpeed);
		};

		// Accumulate (all pixels blend, inactive with reduced alpha)
		Float4 resultDiff = lerp(prevDiff, make_float4(diffuseIllumination, diffuse2ndMoment), diffAlpha);
		Float3 resultFast = lerp(prevDiffFast, diffuseIllumination, diffAlphaFast);

		// Pack normal+roughness for next frame
		Float4 packedNR = make_float4(currentNormalEncoded, currentPackedNR);

		// Write outputs
		out_Diff.write(pixelPos, resultDiff);
		out_DiffFast.write(pixelPos, make_float4(resultFast, 0.0f));
		out_Normal_Roughness.write(pixelPos, packedNR);
		out_ViewZ.write(pixelPos, make_float4(currentViewZ));

		// ====== Specular temporal accumulation (virtual motion) ======
		Float specLum = luminance(specIllumination);
		Float spec2ndMoment = specLum * specLum;

		// Specular history fetch uses full bilinear weights (no checkerboard filter).
		// The binary checkerboard filter rejects the center tap (~90% weight) for active
		// pixels, leaving only right+bottom neighbors (~10%). This creates a systematic
		// directional bias that manifests as "flowing water" on metallic surfaces.
		// The TA bilateral fill already provides reasonable data at inactive pixels.
		Float4 specCw = cw;
		Float specSumW = sumW;

					// --- Surface-motion specular history fetch (bicubic or bilinear) ---
			Float4 prevSpecSurface = def(make_float4(0.0f));
			Float3 prevSpecFastSurface = def(make_float3(0.0f));
			// NRD TA:217 - slow hit-dist history at SMB UV, bilinear-blended with the
			// same weights as the spec history. Promoted to outer scope so the
			// hit-dist EMA after the alpha computation can read it.
			Float prevReflectionHitTSMB = def(0.0f);

			// Surface-motion specular: always bilinear (NRD uses virtual motion for spec)
			{
				Float4 specCw = cw;
				Float specSumW = sumW;
				Float4 ps00 = gPrev_SpecHistory.read(p00);
				Float4 ps10 = gPrev_SpecHistory.read(p10);
				Float4 ps01 = gPrev_SpecHistory.read(p01);
				Float4 ps11 = gPrev_SpecHistory.read(p11);
				$if(reprojectionFound & specSumW > .1f) {
					prevSpecSurface = (ps00 * specCw.x + ps10 * specCw.y + ps01 * specCw.z + ps11 * specCw.w) / specSumW;
				};

				Float4 psf00 = gPrev_SpecFastHistory.read(p00);
				Float4 psf10 = gPrev_SpecFastHistory.read(p10);
				Float4 psf01 = gPrev_SpecFastHistory.read(p01);
				Float4 psf11 = gPrev_SpecFastHistory.read(p11);
				$if(reprojectionFound & specSumW > .1f) {
					prevSpecFastSurface = (psf00 * specCw.x + psf10 * specCw.y + psf01 * specCw.z + psf11 * specCw.w).xyz() / specSumW;
				};

				// NRD TA:217 - slow hit-dist history at SMB UV, same bilinear weights.
				// Used by the hit-dist EMA downstream (Gap B) so out_SpecHitDist carries
				// an EMA'd value instead of single-frame bilateral-smoothed current.
				Float4 psh00 = gPrev_SpecHitDist.read(p00);
				Float4 psh10 = gPrev_SpecHitDist.read(p10);
				Float4 psh01 = gPrev_SpecHitDist.read(p01);
				Float4 psh11 = gPrev_SpecHitDist.read(p11);
				$if(reprojectionFound & specSumW > .1f) {
					prevReflectionHitTSMB = (psh00.x * specCw.x + psh10.x * specCw.y
						+ psh01.x * specCw.z + psh11.x * specCw.w) / specSumW;
				};
			}
			prevSpecSurface = luisa::compute::max(prevSpecSurface, make_float4(0.0f));
			prevSpecFastSurface = luisa::compute::max(prevSpecFastSurface, make_float3(0.0f));

		// --- Virtual motion for specular ---
		// Estimate where the specular reflection was in the previous frame using thin-lens + curvature.
		Float4 prevSpecVirtual = def(make_float4(0.0f));
		Float3 prevSpecFastVirtual = def(make_float3(0.0f));
		// NRD TA:835 reads prevSpecularIlluminationAnd2ndMomentVMBResponsive.a
		// as hitDistForTrackingPrev for Stage 2 virtual motion. Preserve the
		// fast-history hitDist from the virtual bilinear fetch.
		Float prevSpecFastVirtual_w = def(0.0f);
		// NRD TA:979 - slow hit-dist history at VMB UV. Promoted to outer scope so
		// the hit-dist EMA after the alpha computation can read it. Stays 0 when VMB
		// reprojection fails; downstream EMA collapses to specHitDist via specVMBAlpha=1.
		Float prevReflectionHitTVMB = def(0.0f);
		Bool virtualReprojectionFound = false;
		Float dominanceFactor = def(0.0f);
		Float2 prevUV_vmb = def(make_float2(0.0f));

		Float curvature = def(0.0f);
		// Promoted from inside the virtual-motion $if so Stage 2 hit-distance
		// confidence (which lives in $if(virtualReprojectionFound), outside the
		// virtual-motion $if) can read them. All assigned inside the gate.
		Float3 dominantDir = def(make_float3(0.0f, 0.0f, 1.0f));
		Float3 tangent = def(make_float3(0.0f));
		Float3 bitangent = def(make_float3(0.0f));
		Float silhouetteF = def(1.0f);
		Float3 virtualWorldPos = def(make_float3(0.0f));
		// Always attempt VMB (NRD has no hard gate). The closenessToSurface blend
		// further down handles short-hit-distance case; virtualReprojectionFound
		// stays false for invalid VMB (UV OOB or acceleration > cap).
		$if(minHitDist3x3 >= 0.0f) {
			Float a = 0.298475f * luisa::compute::log(
				39.4115f - 39.0029f * currentRoughnessModified);
			Float dominanceFactorInner = luisa::compute::saturate(
				luisa::compute::pow(luisa::compute::saturate(1.0f - NoV), 10.8649f)
				* (1.0f - a) + a);
			Float3 reflDir = luisa::compute::reflect(-V, currentNormal);
			dominantDir = luisa::compute::normalize(lerp(reflDir, currentNormal, dominanceFactorInner));
			dominanceFactor = dominanceFactorInner;

			// --- Curvature estimation along motion direction ---
			{
				Float2 deltaUv = mv_raw * make_float2(rectSizePrev) * 0.5f;
				Float2 absDelta = luisa::compute::abs(deltaUv) + 1.0f / 256.0f;
				Float2 w = absDelta / (absDelta.x + absDelta.y);

				// Neighbor at (x+1, y)
				UInt2 pos10 = make_uint2(luisa::compute::min(pixelPos.x + 1u, rectW - 1u), pixelPos.y);
				Float3 n10 = n10_cached;
				Float viewZ10 = luisa::compute::abs(gIn_ViewZ.read(pos10).x);
				Float2 uv10 = (make_float2(pos10) + 0.5f) * rectSizeInv;
				Float2 clipXY10 = uv10 * 2.0f - 1.0f;
				Float3 x10 = c.gFrustumForward.xyz() * viewZ10
					+ c.gFrustumRight.xyz() * clipXY10.x
					+ c.gFrustumUp.xyz() * clipXY10.y;

				// Neighbor at (x, y+1)
				UInt2 pos01 = make_uint2(pixelPos.x, luisa::compute::min(pixelPos.y + 1u, rectH - 1u));
				Float3 n01 = n01_cached;
				Float viewZ01 = luisa::compute::abs(gIn_ViewZ.read(pos01).x);
				Float2 uv01 = (make_float2(pos01) + 0.5f) * rectSizeInv;
				Float2 clipXY01 = uv01 * 2.0f - 1.0f;
				Float3 x01 = c.gFrustumForward.xyz() * viewZ01
					+ c.gFrustumRight.xyz() * clipXY01.x
					+ c.gFrustumUp.xyz() * clipXY01.y;

				Float3 xNeighbor = x10 * w.x + x01 * w.y;
				Float3 nNeighbor = luisa::compute::normalize(n10 * w.x + n01 * w.y);
				Float3 edge = xNeighbor - currentWorldPos;
				Float edgeLenSq = luisa::compute::dot(edge, edge);
				$if(edgeLenSq > 1e-10f) {
					curvature = luisa::compute::dot(nNeighbor - currentNormal, edge) / edgeLenSq;
				};
				// Dampen negative curvature to prevent excessive virtual motion
				$if(curvature < 0.0f) {
					curvature = curvature * 0.5f;
				};
			}

			// --- Thin-lens virtual motion ---
			tangent = luisa::compute::normalize(
				ite(luisa::compute::abs(currentNormal.x) > 0.999f,
					luisa::make_float3(0.0f, 1.0f, 0.0f),
					luisa::compute::cross(currentNormal, luisa::make_float3(1.0f, 0.0f, 0.0f))));
			bitangent = luisa::compute::cross(currentNormal, tangent);

			Float3 reflectionRay = dominantDir * specHitDist;
			Float Ox = luisa::compute::dot(tangent, reflectionRay);
			Float Oy = luisa::compute::dot(bitangent, reflectionRay);
			Float Oz = luisa::compute::dot(currentNormal, reflectionRay);
			Oz = -Oz; // NRD sign convention: O.z negated (reflection points outward)

			// Thin-lens: I = O / (2*curvature*Oz - 1)
			Float magDenom = 2.0f * curvature * Oz - 1.0f;
			Float mag = 1.0f / magDenom;

			// Reduce magnification at grazing angles (silhouette protection)
			// NoV already computed above for dominant factor
			silhouetteF = 1.0f / (1.0f + luisa::compute::length(currentWorldPos)
				* luisa::compute::saturate(1.0f - NoV)
				* luisa::compute::max(curvature, 0.0f));
			mag = mag * silhouetteF;

			Float Ix = Ox * mag; Float Iy = Oy * mag; Float Iz = Oz * mag;
			Float3 I_world = tangent * Ix + bitangent * Iy + currentNormal * Iz;

			// Virtual world position: prev camera pos + virtual offset along V
			Float3 prevWorldPos = currentWorldPos + c.gCameraDelta.xyz();
			Float Il = luisa::compute::length(I_world);
			Float magSign = ite(mag >= 0.0f, 1.0f, -1.0f);
			virtualWorldPos = prevWorldPos + V * Il * magSign;

			// Blend toward surface motion for close reflections (hitDist -> 0)
			Float closeness = luisa::compute::saturate(
				dominanceFactorInner * Il / luisa::compute::max(specHitDist, 1e-4f));
			virtualWorldPos = lerp(prevWorldPos, virtualWorldPos, closeness);

			// --- Project virtual position to previous frame UV ---
			Float4 virtualClipPos = c.gWorldToClipPrev * make_float4(virtualWorldPos, 1.0f);
			Float invW = 1.0f / luisa::compute::max(luisa::compute::abs(virtualClipPos.w), 1e-6f);
			Float2 prevUV_vmb = virtualClipPos.xy() * invW * luisa::make_float2(0.5f, -0.5f)
				+ luisa::make_float2(0.5f, 0.5f);

			Bool virtualValid = prevUV_vmb.x >= 0.0f & prevUV_vmb.x <= 1.0f
				& prevUV_vmb.y >= 0.0f & prevUV_vmb.y <= 1.0f;

			// Clamp virtual motion acceleration
			Float surfaceParallax = luisa::compute::length(
				(prevUV - pixelUv) * rectSizePrev);
			Float virtualParallax = luisa::compute::length(
				(prevUV_vmb - pixelUv) * rectSizePrev);
			Float accelRatio = virtualParallax / luisa::compute::max(surfaceParallax, 1.0f);
			$if(accelRatio > c.gMaxAllowedVirtualMotionAcceleration) {
				virtualValid = false;
			};

			$if(virtualValid) {
				Float2 prevPixelPosVMB = prevUV_vmb * rectSizePrev;
				Float2 flooredVMB = luisa::compute::floor(prevPixelPosVMB - 0.5f);
				UInt oxVMB = cast<uint>(luisa::compute::max(flooredVMB.x, 0.0f));
				UInt oyVMB = cast<uint>(luisa::compute::max(flooredVMB.y, 0.0f));
				oxVMB = clamp(oxVMB, 0u, cast<uint>(rectSizePrev.x) - 2u);
				oyVMB = clamp(oyVMB, 0u, cast<uint>(rectSizePrev.y) - 2u);

				UInt2 vp00 = make_uint2(oxVMB, oyVMB);
				UInt2 vp10 = make_uint2(oxVMB + 1u, oyVMB);
				UInt2 vp01 = make_uint2(oxVMB, oyVMB + 1u);
				UInt2 vp11 = make_uint2(oxVMB + 1u, oyVMB + 1u);

				// Depth validation at virtual UV (more relaxed threshold)
				Float vz00 = luisa::compute::abs(gPrev_ViewZ2.read(vp00).x);
				Float vz10 = luisa::compute::abs(gPrev_ViewZ2.read(vp10).x);
				Float vz01 = luisa::compute::abs(gPrev_ViewZ2.read(vp01).x);
				Float vz11 = luisa::compute::abs(gPrev_ViewZ2.read(vp11).x);

				Float2 bWeights = luisa::compute::fract(prevPixelPosVMB - 0.5f);
				Float4 vcw = make_float4(
					(1.0f - bWeights.x) * (1.0f - bWeights.y),
					bWeights.x * (1.0f - bWeights.y),
					(1.0f - bWeights.x) * bWeights.y,
					bWeights.x * bWeights.y);

				Float depthThresh = threshold;
				Float4 vdz = make_float4(
					luisa::compute::abs(vz00 - depthRef),
					luisa::compute::abs(vz10 - depthRef),
					luisa::compute::abs(vz01 - depthRef),
					luisa::compute::abs(vz11 - depthRef));
				Float4 vValidMask = make_float4(
					ite(vz00 < denoisingRange & vdz.x < depthThresh, 1.0f, 0.0f),
					ite(vz10 < denoisingRange & vdz.y < depthThresh, 1.0f, 0.0f),
					ite(vz01 < denoisingRange & vdz.z < depthThresh, 1.0f, 0.0f),
					ite(vz11 < denoisingRange & vdz.w < depthThresh, 1.0f, 0.0f));
				vcw = vcw * vValidMask;
				Float vSumW = dot(vcw, make_float4(1.0f));

				// Partial-footprint VMB: accept if depth-valid weight above confidence
				// threshold (gReprojectionConfidence default 0.1). Strict all()-tap test
				// (NRD :351) creates a hard silhouette-parallel boundary on metallic
				// spheres: VMB hard-rejected where any tap falls off the depth edge,
				// falling back to slow SMB-only EMA. The two regimes converge to
				// different stable values - visible as a black ring at the VMB
				// accept/reject boundary. Lenient acceptance degrades VMB weight
				// smoothly via vcw * vValidMask + downstream confidence factors.
				$if(vSumW > c.gReprojectionConfidence) {
					Float4 vps00 = gPrev_SpecHistory.read(vp00);
					Float4 vps10 = gPrev_SpecHistory.read(vp10);
					Float4 vps01 = gPrev_SpecHistory.read(vp01);
					Float4 vps11 = gPrev_SpecHistory.read(vp11);
					// No checkerboard filter: use full virtual motion bilinear weights
					Float4 vSpecCw = vcw;
					Float vSpecSumW = vSumW;
					$if(vSpecSumW > c.gReprojectionConfidence) {
						prevSpecVirtual = (vps00 * vSpecCw.x + vps10 * vSpecCw.y + vps01 * vSpecCw.z + vps11 * vSpecCw.w) / vSpecSumW;
						prevSpecVirtual = luisa::compute::max(prevSpecVirtual, make_float4(0.0f));

						Float4 vpsf00 = gPrev_SpecFastHistory.read(vp00);
						Float4 vpsf10 = gPrev_SpecFastHistory.read(vp10);
						Float4 vpsf01 = gPrev_SpecFastHistory.read(vp01);
						Float4 vpsf11 = gPrev_SpecFastHistory.read(vp11);
						Float4 vpsfSum = vpsf00 * vSpecCw.x + vpsf10 * vSpecCw.y + vpsf01 * vSpecCw.z + vpsf11 * vSpecCw.w;
						prevSpecFastVirtual = vpsfSum.xyz() / vSpecSumW;
						prevSpecFastVirtual = luisa::compute::max(prevSpecFastVirtual, make_float3(0.0f));
						prevSpecFastVirtual_w = vpsfSum.w / vSpecSumW;
					};

					virtualReprojectionFound = true;
				};
			};
		};

		// === Specular virtual motion confidence (NRD TA lines 699-850) ===
		// Lobe half-angle for SMB/VMB confidence
		Float lobeHalfAngle = luisa::compute::max(
			luisa::compute::atan(currentRoughnessModified * currentRoughnessModified * 0.75f
				/ (1.0f - 0.75f + 1e-6f)),
			1.5f / 255.0f);

		// SpecMagicCurve: NRD _NRD_GetSpecMagicCurve(R, 0.25)
		//   f = (1 - exp2(-200·R²)) · R^0.25
		// R^0.25 via two sqrts (cheaper than pow, bit-exact for power-of-2).
		Float _smcR = luisa::compute::saturate(currentRoughnessModified);
		Float SMC = (1.0f - luisa::compute::exp2(-200.0f * _smcR * _smcR))
		          * luisa::compute::sqrt(luisa::compute::sqrt(_smcR));
		// NRD virtual-motion confidence (RELAX_TemporalAccumulation.cs.hlsl:780-863)
		// specVMBConfidence gates how strongly VMB history drives the spec blend,
		// vmbWeightMultiplier collects all virtualHistoryAmount reductions so they
		// can be applied at the single virtualHistoryAmount init site.
		Float specVMBConfidence = def(0.0f);
		Float vmbWeightMultiplier = def(1.0f);
		// Promoted from $if(virtualReprojectionFound) so the alpha computation
		// (outside that $if) can read the Stage 1+2 hit-distance confidence.
		Float virtualHistoryHitDistConfidence = def(1.0f);

		$if(virtualReprojectionFound) {
			// Virtual UV center (reused by hit-distance + back-look blocks)
			UInt bxhd = cast<uint>(clamp(cast<int>(prevUV_vmb.x * c.gResourceSizeX), 0, cast<int>(c.gResourceSizeX) - 1));
			UInt byhd = cast<uint>(clamp(cast<int>(prevUV_vmb.y * c.gResourceSizeY), 0, cast<int>(c.gResourceSizeY) - 1));

			// === Immediate normal + roughness gates (NRD TA:783-801) ===
			// Catches fast virtual-motion landings on a wrong surface — the failure
			// mode that manifests as edge jitter on small / high-curvature objects.
			Float4 prevNormalRoughnessVMB_packed = gPrev_Normal_Roughness.read(make_uint2(bxhd, byhd));
			Float3 prevNormalVMB = luisa::compute::normalize(prevNormalRoughnessVMB_packed.xyz() * 2.0f - 1.0f);
			Float  prevRoughnessVMB = prevNormalRoughnessVMB_packed.w;
			Float3 prevNormalVMB_rotated = transform_normal(c.gWorldPrevToWorld, prevNormalVMB);

			// uvDiff + curvature angle (NRD TA:783-789)
			Float2 uvDiff = prevUV_vmb - prevUV;
			Float uvDiffLengthInPixels = luisa::compute::length(uvDiff * rectSizePrev);
			Float tanCurvature = luisa::compute::abs(curvature * pixelSize);
			tanCurvature *= luisa::compute::max(uvDiffLengthInPixels / luisa::compute::max(NoV, 0.01f), 1.0f);
			Float curvatureAngle = luisa::compute::atan(tanCurvature);

			// Immediate encoding-aware normal weight (NRD TA:791-794)
			Float cosa_n = luisa::compute::clamp(luisa::compute::dot(currentNormal, prevNormalVMB_rotated), 0.0f, 1.0f);
			Float angle_n = luisa::compute::acos(cosa_n);
			Float normalWeight = luisa::compute::smoothstep(0.0f, 1.0f,
				1.0f - (angle_n - curvatureAngle - RELAX_NORMAL_ULP) / luisa::compute::max(lobeHalfAngle, 1e-6f));
			normalWeight = luisa::compute::smoothstep(0.05f, 0.95f, normalWeight);
			Float jitterFriendlyNormalW = luisa::compute::lerp(
				1.0f - luisa::compute::saturate(uvDiffLengthInPixels), 1.0f, normalWeight);

			// Immediate encoding-aware roughness weight (NRD TA:796-800)
			// Relaxed roughness params: returns (a, b=m*a) so weight(x²) = smoothstep(1,0,|x²*a - b|)
			Float mrr = currentRoughnessModified * currentRoughnessModified;
			Float trr = luisa::compute::lerp(mrr, currentRoughnessModified,
				luisa::compute::saturate(c.gRoughnessFraction));
			constexpr float ROUGHNESS_SENSITIVITY = 0.01f;
			Float rwp_a = 1.0f / luisa::compute::lerp(ROUGHNESS_SENSITIVITY, 1.0f, luisa::compute::saturate(trr));
			Float rwp_b = mrr * rwp_a;
			Float virtualRoughnessWeight = luisa::compute::smoothstep(1.0f, 0.0f,
				luisa::compute::abs(prevRoughnessVMB * prevRoughnessVMB * rwp_a - rwp_b));
			virtualRoughnessWeight = luisa::compute::lerp(
				1.0f - luisa::compute::saturate(uvDiffLengthInPixels), 1.0f, virtualRoughnessWeight);

			vmbWeightMultiplier = vmbWeightMultiplier * jitterFriendlyNormalW * virtualRoughnessWeight;
			specVMBConfidence = virtualRoughnessWeight * 0.9f + 0.1f;

			// === Hit distance confidence (NRD TA:720-831) — preserved ===
			prevReflectionHitTVMB = gPrev_SpecHitDist.read(make_uint2(bxhd, byhd)).x;
			prevReflectionHitTVMB = luisa::compute::max(prevReflectionHitTVMB, 0.001f);

			Float hitDistC = luisa::compute::lerp(minHitDist3x3, prevReflectionHitTVMB, SMC);
			Float hitDist1 = hitDistC / (2.0f * curvature * hitDistC + 1.0f);
			Float hitDist2 = prevReflectionHitTVMB / (2.0f * curvature * prevReflectionHitTVMB + 1.0f);
			Float maxDist = luisa::compute::max(hitDist1, hitDist2);
			Float dHitT = luisa::compute::abs(hitDist1 - hitDist2);
			Float dHitTMultiplier = luisa::compute::lerp(20.0f, 0.0f, SMC);
			// Stage 1: thin-lens hit-distance confidence (NRD TA:822-831).
			virtualHistoryHitDistConfidence = 1.0f - luisa::compute::saturate(
				dHitTMultiplier * dHitT / (currentViewZ + maxDist));
			virtualHistoryHitDistConfidence = luisa::compute::lerp(virtualHistoryHitDistConfidence, 1.0f, SMC);

			// Stage 2: lobe-radius vs delta-parallax (NRD TA:833-849). Re-derive the
			// virtual position with prev frame's hit-distance. If that UV is far from
			// the current virtual UV (beyond lobe footprint), virtual motion landed
			// wrong — reduce confidence. Critical for narrow metallic lobes.
			//
			// NRD TA:835 reads prevSpecularIlluminationAnd2ndMomentVMBResponsive.a
			// (the FAST 1-frame-old min-3x3 hitDist from gPrev_SpecFast.a) here,
			// NOT the slow EMA'd gPrev_SpecHitDist. Slow lags by many frames
			// during rotation, giving wrong prevVirtualWorldPos and wrong
			// virtualHistoryHitDistConfidence on narrow metallic lobes.
			Float hitDistForTrackingPrev = luisa::compute::max(prevSpecFastVirtual_w, 0.001f);
			Float3 prevReflectionRay = dominantDir * hitDistForTrackingPrev;
			Float prevOx = luisa::compute::dot(tangent, prevReflectionRay);
			Float prevOy = luisa::compute::dot(bitangent, prevReflectionRay);
			Float prevOz = -luisa::compute::dot(currentNormal, prevReflectionRay);
			Float prevMagDenom = 2.0f * curvature * prevOz - 1.0f;
			Float prevMag = (1.0f / prevMagDenom) * silhouetteF;
			Float3 prevI_world = tangent * (prevOx * prevMag)
			                   + bitangent * (prevOy * prevMag)
			                   + currentNormal * (prevOz * prevMag);
			Float prevIl = luisa::compute::length(prevI_world);
			Float prevMagSign = ite(prevMag >= 0.0f, 1.0f, -1.0f);
			Float prevCloseness = luisa::compute::saturate(
				dominanceFactor * prevIl / luisa::compute::max(hitDistForTrackingPrev, 1e-4f));
			Float3 prevVirtualWorldPos = lerp(prevWorldPos,
				prevWorldPos + V * prevIl * prevMagSign, prevCloseness);

			Float4 prevVirtualClipPos = c.gWorldToClipPrev * make_float4(prevVirtualWorldPos, 1.0f);
			Float prevInvW = 1.0f / luisa::compute::max(luisa::compute::abs(prevVirtualClipPos.w), 1e-6f);
			Float2 prevUVVMBTest = prevVirtualClipPos.xy() * prevInvW * luisa::make_float2(0.5f, -0.5f)
			                     + luisa::make_float2(0.5f, 0.5f);

			// Lobe radius in pixels at the virtual distance (NRD uses raw roughness
			// here, not currentRoughnessModified).
			Float percentOfVolume = 0.6f;
			Float lobeTanHalfAngle = currentRoughness * currentRoughness * percentOfVolume
			                       / (1.0f - percentOfVolume + 1e-6f);
			lobeTanHalfAngle = luisa::compute::max(lobeTanHalfAngle, 0.5f * rectSizeInv.x);
			Float pixelWorldSizeAtVirtual = c.gUnproject * luisa::compute::max(
				luisa::compute::length(virtualWorldPos),
				luisa::compute::length(prevVirtualWorldPos));
			Float unproj1 = luisa::compute::min(specHitDist, hitDistForTrackingPrev)
			              / luisa::compute::max(pixelWorldSizeAtVirtual, 1e-6f);
			Float lobeRadiusInPixels = lobeTanHalfAngle * unproj1;

			Float deltaParallaxInPixels = luisa::compute::length(
				(prevUVVMBTest - prevUV_vmb) * make_float2(rectSizePrev));

			// NRD: SmoothStep(R+0.25, 0, delta) — 1 when delta=0, 0 when delta=R+0.25.
			virtualHistoryHitDistConfidence = virtualHistoryHitDistConfidence
			    * (1.0f - luisa::compute::smoothstep(0.0f, lobeRadiusInPixels + 0.25f, deltaParallaxInPixels));

			// === Back-look loop (NRD TA:803-820) — rewritten ===
			// Compares prevVMB normal (rotated) against back-look normals, scaled by
			// curvatureAngle*(n+1) (NRD), plus roughness back-look.
			Float2 uvDiffNorm = uvDiff / luisa::compute::max(luisa::compute::length(uvDiff), 1e-6f);
			Float2 stepSize = uvDiffNorm / rectSizePrev
				* (luisa::compute::saturate(uvDiffLengthInPixels / 0.1f) + uvDiffLengthInPixels / 2.0f);

			Float prevPrevNormalWeight = def(1.0f);
			Float backRoughnessWeight = def(1.0f);

			$for(n, 1, 3) {
				Float2 backUV = prevUV_vmb + cast<Float>(n) * stepSize;
				$if(backUV.x >= 0.0f & backUV.x <= 1.0f & backUV.y >= 0.0f & backUV.y <= 1.0f) {
					UInt bx = cast<uint>(clamp(cast<int>(backUV.x * c.gResourceSizeX), 0, cast<int>(c.gResourceSizeX) - 1));
					UInt by = cast<uint>(clamp(cast<int>(backUV.y * c.gResourceSizeY), 0, cast<int>(c.gResourceSizeY) - 1));
					Float4 pnr = gPrev_Normal_Roughness.read(make_uint2(bx, by));
					Float3 backNormal = luisa::compute::normalize(pnr.xyz() * 2.0f - 1.0f);
					Float3 backNormalRotated = transform_normal(c.gWorldPrevToWorld, backNormal);
					Float backRoughness = luisa::compute::fract(pnr.w);

					// Encoding-aware normal weight between prev VMB and back-look normals
					Float cosa_b = luisa::compute::clamp(luisa::compute::dot(prevNormalVMB_rotated, backNormalRotated), 0.0f, 1.0f);
					Float angle_b = luisa::compute::acos(cosa_b);
					Float lobeScale = curvatureAngle * (1.0f + cast<Float>(n));
					Float nm = luisa::compute::smoothstep(0.0f, 1.0f,
						1.0f - (angle_b - lobeScale - RELAX_NORMAL_ULP) / luisa::compute::max(lobeHalfAngle, 1e-6f));
					nm = luisa::compute::smoothstep(0.05f, 0.95f, nm);
					prevPrevNormalWeight = prevPrevNormalWeight * nm;

					// Roughness back-look weight (NRD TA:805)
					Float rwn = luisa::compute::smoothstep(1.0f, 0.0f,
						luisa::compute::abs(backRoughness * backRoughness * rwp_a - rwp_b));
					backRoughnessWeight = backRoughnessWeight * rwn;
				};
			};

			// NRD TA:815-816
			vmbWeightMultiplier = vmbWeightMultiplier * (0.33f + 0.67f * prevPrevNormalWeight);
			specVMBConfidence = specVMBConfidence * (0.33f + 0.67f * prevPrevNormalWeight);

			// NRD TA:818-820 (orthogonal mode guard omitted — engine is perspective only)
			vmbWeightMultiplier = vmbWeightMultiplier * (backRoughnessWeight * 0.9f + 0.1f);

			// Note: hit-dist confidence (virtualHistoryHitDistConfidence, Stage 1+2)
			// is NOT applied to specVMBConfidence here — NRD keeps them separate.
			// The hit-dist factor enters only via specVMBResponsiveAlpha below so
			// the slow EMA path stays at clean specVMBConfidence.
		};

		// SMB confidence: NRD encoding-aware normal weight between V and Vprev,
		// modulated by framerate scale (NRD RELAX TA:852-853). Restores angular
		// sensitivity so SMB fallback isn't fully trusted when the view vector
		// has rotated — important on small objects where SMB reprojection itself
		// is unstable.
		Float smbCosA = luisa::compute::clamp(luisa::compute::dot(V, Vprev), 0.0f, 1.0f);
		Float smbAngle = luisa::compute::acos(smbCosA);
		Float smbMaxAngle = lobeHalfAngle * NoV / luisa::compute::max(c.gFramerateScale, 1e-6f);
		Float smbNormalWeight = luisa::compute::smoothstep(0.0f, 1.0f,
			1.0f - smbAngle / luisa::compute::max(smbMaxAngle, 1e-6f));
		smbNormalWeight = luisa::compute::smoothstep(0.05f, 0.95f, smbNormalWeight);
		Float specSMBConfidence = ite(reprojectionFound, smbNormalWeight, 0.0f);
		Float virtualHistoryAmount = dominanceFactor
			* ite(virtualReprojectionFound, 1.0f, 0.0f)
			* vmbWeightMultiplier;
		virtualHistoryAmount = virtualHistoryAmount * luisa::compute::saturate(
			specVMBConfidence / (specSMBConfidence + 1e-6f));

		// NRD TA:878-884 — separate slow vs responsive, SMB vs VMB alphas.
		// virtualHistoryHitDistConfidence (Stage 1+2) enters only via
		// specVMBResponsiveAlpha — the slow path stays at clean specVMBConfidence.
		Float specSMBAlpha = luisa::compute::max(1.0f - specSMBConfidence,
			luisa::compute::max(c.gSpecAlphaSteady, 1.0f / historyLength));
		Float specSMBResponsiveAlpha = specSMBAlpha; // NRD: SMB uses same alpha for slow + responsive
		Float specVMBAlpha = luisa::compute::max(1.0f - specVMBConfidence,
			luisa::compute::max(c.gSpecAlphaSteady, 1.0f / historyLength));
		Float specVMBResponsiveAlpha = luisa::compute::max(
			1.0f - specVMBConfidence * virtualHistoryHitDistConfidence,
			luisa::compute::max(c.gSpecAlphaFastSteady, 1.0f / historyLength));

		// NRD TA:864-868, 887-892 — checkerboard adjustment applied to each alpha
		// separately, gated on its respective reprojection flag. Previously lumped
		// into one specAlphaAdj post-lerp; the lumped form is incompatible with the
		// two-EMA-then-blend restructure below.
		$if(!cbActive & parallaxInPixels < 0.5f) {
			Float smbCbGate = ite(reprojectionFound,        1.0f, 0.0f);
			Float vmbCbGate = ite(virtualReprojectionFound, 1.0f, 0.0f);
			specSMBAlpha           *= 1.0f - cbResolveSpeed * smbCbGate;
			specSMBResponsiveAlpha *= 1.0f - cbResolveSpeed * smbCbGate;
			specVMBAlpha           *= 1.0f - cbResolveSpeed * vmbCbGate;
			specVMBResponsiveAlpha *= 1.0f - cbResolveSpeed * vmbCbGate;
		};

		// NRD TA:871-910 — EMA SMB and VMB paths SEPARATELY, then blend by VHA.
		// Previous form (lerp histories, then EMA with lerped alpha) is algebraically
		// non-equivalent when alphaSMB != alphaVMB, which is exactly the rotation
		// case for narrow-lobe metals where VMB confidence fluctuates. Matches NRD
		// so unstable VMB samples don't leak through during rotation. When
		// virtualReprojectionFound=false, virtualHistoryAmount=0 (set above) so the
		// VMB side drops out and the result is the SMB path alone.
		Float3 accSpecIllumSMB = lerp(prevSpecSurface.xyz(), specIllumination, specSMBAlpha);
		Float  accSpecM2SMB    = lerp(prevSpecSurface.w,     spec2ndMoment,    specSMBAlpha);
		Float3 accSpecFastSMB  = lerp(prevSpecFastSurface,   specIllumination, specSMBResponsiveAlpha);

		Float3 accSpecIllumVMB = lerp(prevSpecVirtual.xyz(), specIllumination, specVMBAlpha);
		Float  accSpecM2VMB    = lerp(prevSpecVirtual.w,     spec2ndMoment,    specVMBAlpha);
		Float3 accSpecFastVMB  = lerp(prevSpecFastVirtual,   specIllumination, specVMBResponsiveAlpha);

		Float3 resultSpecIllum = lerp(accSpecIllumSMB, accSpecIllumVMB, virtualHistoryAmount);
		Float  resultSpecM2    = lerp(accSpecM2SMB,    accSpecM2VMB,    virtualHistoryAmount);
		Float3 resultSpecFast  = lerp(accSpecFastSMB,  accSpecFastVMB,  virtualHistoryAmount);

		Float4 resultSpec      = make_float4(resultSpecIllum, resultSpecM2);

		// NRD TA:873, 897, 905, 933 - EMA hit-dist per path then blend by VHA.
		// out_SpecHitDist stores the SLOW accumulated hit-dist, not bilateral-smoothed
		// current. Next frame's Stage 1+2 virtual motion reads this as
		// hitDistForTrackingPrev; an un-EMA'd value oscillates with single noisy GI
		// samples on narrow-lobe metals, making virtual UV jitter frame-to-frame.
		// The max(..., 0.1) floor ensures hit-dist tracks current reflection depth
		// even when illumination alpha has converged small (NRD TA:873, 897).
		Float accHitDistSMB = lerp(prevReflectionHitTSMB, specHitDist,
			luisa::compute::max(specSMBAlpha, 0.1f));
		Float accHitDistVMB = lerp(prevReflectionHitTVMB, specHitDist,
			luisa::compute::max(specVMBAlpha, 0.1f));
		Float accumulatedReflectionHitT = lerp(accHitDistSMB, accHitDistVMB, virtualHistoryAmount);

		// specularHistoryConfidence needed before out_Spec.write so the NRD
		// TA:927 variance boost below can use it. Moved up from old line ~1135.
		Float specularHistoryConfidence = lerp(specSMBConfidence, specVMBConfidence, virtualHistoryAmount);

		// NRD TA:927 - if zero specular sample (color = 0), artificially add
		// variance for pixels with low reprojection confidence. Low-R metal
		// pixels that miss the narrow GGX lobe leave 2nd moment = 0; without
		// this boost Atrous has no variance signal and can't filter aggressively.
		Float boosted2ndMoment = c.gSpecVarianceBoost * (1.0f - specularHistoryConfidence);
		resultSpec.w = ite(resultSpec.w == 0.0f, boosted2ndMoment, resultSpec.w);

		out_Spec.write(pixelPos, resultSpec);
		// NRD RELAX_TemporalAccumulation.cs.hlsl:931 writes hitDist (= min-3x3
		// from current input, NOT the EMA'd slow history) to specFast.a. Next
		// frame's TA reads this as hitDistForTrackingPrev for Stage 2 virtual
		// motion. Sentinel 1e6f from line 268 maps to 0 to match NRD's
		// `minHitDist3x3 == NRD_INF ? 0.0 : minHitDist3x3`.
		Float specFastHitDist = ite(minHitDist3x3 >= 1e6f, 0.0f, minHitDist3x3);
		out_SpecFast.write(pixelPos, make_float4(resultSpecFast, specFastHitDist));
		out_SpecHitDist.write(pixelPos, make_float4(accumulatedReflectionHitT));

		out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
	});

}

}