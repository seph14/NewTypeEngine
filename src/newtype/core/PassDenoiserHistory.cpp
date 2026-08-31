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

void RelaxDenoiser::compileHistory(Device& device) {
	//==========================================================================
	// ReLAX Step 5: HistoryFix
	// Ported from RELAX_HistoryFix.cs.hlsl (NRD v4.17)
	//
	// 5x5 sparse cross-bilateral filter for disoccluded pixels.
	// Only processes pixels with historyLength <= gHistoryFixFrameNum.
	// Stride adapts to disocclusion age: shorter history -> wider search.
	//
	// Simplifications: no specular, no material ID, no SH, perspective only.
	//==========================================================================
	_relaxHistoryFix = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat io_Diff,
		ImageFloat io_DiffFast,
		ImageFloat io_HistoryLength,
		ImageFloat gIn_Normal_Roughness,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_Tiles,
		// Specular (7-8): pass-through
		ImageFloat io_Spec,
		ImageFloat io_SpecFast
		) noexcept {
		set_block_size(8u, 8u, 1u);
		set_name("history_fix");

		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);

		UInt  rectW = cast<uint>(c.gRectSizeX);
		UInt  rectH = cast<uint>(c.gRectSizeY);
		Float2 rectSizeInv = make_float2(c.gRectSizeInvX, c.gRectSizeInvY);
		Float  denoisingRange = c.gDenoisingRange;
		// Loop-invariant int bounds; XIR has no LICM pass, so hoist explicitly.
		Int rectWint = cast<int>(rectW) - 1;
		Int rectHint = cast<int>(rectH) - 1;

		// Tile-based early out
		UInt2 tilePos = pixelPos >> 4u;
		//Float isSky = gIn_Tiles.read(tilePos).x;
		$if(Expr{ gIn_Tiles.read(tilePos).x != 0.0f } | pixelPos.x >= rectW | pixelPos.y >= rectH) {
			$return();
		};

		// Read center data
		Float centerViewZ = luisa::compute::abs(gIn_ViewZ.read(pixelPos).x);
		$if(centerViewZ > denoisingRange) {
			$return();
		};

		// History length (un-normalize from R8 storage)
		Float historyLength = 255.0f * io_HistoryLength.read(pixelPos).x;

		// Early out if not disoccluded (historyLength > fixFrameNum) or HistoryFix disabled (fixFrameNum < 1)
		Float fixFrameNum = c.gHistoryFixFrameNum;
		$if(historyLength > fixFrameNum | fixFrameNum < 1.0f) {
			$return();
		};

		// Center normal (packed 0..1 -> -1..1)
		Float4 centerNR = gIn_Normal_Roughness.read(pixelPos);
		Float3 centerNormal = luisa::compute::normalize(centerNR.xyz() * 2.0f - 1.0f);
		Float centerMaterialID = luisa::compute::floor(centerNR.w);  // unpack matID from packed matID+roughness

		// Center world position (perspective: gOrthoMode = 0)
		Float2 clipXY = (make_float2(pixelPos) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
		Float3 frustumFwd = c.gFrustumForward.xyz();
		Float3 frustumRight = c.gFrustumRight.xyz();
		Float3 frustumUp = c.gFrustumUp.xyz();
		Float3 rayDir = frustumFwd + frustumRight * clipXY.x - frustumUp * clipXY.y;
		Float centerViewZc = centerViewZ / luisa::compute::length(rayDir);
		Float3 centerWorldPos = centerViewZc * rayDir;

		// Depth threshold (scaled by viewZ for perspective)
		Float depthThreshold = c.gDepthThreshold * centerViewZc;

		// NRD HistoryFix specular weight params: GetNormalWeightParams_ATrous with
		// fixed hl=5, specConf=1, relax=0 (matches NRD's HistoryFix constant args).
		// This makes specular disocclusion fill use BRDF-appropriate sharpness
		// rather than the diffuse normal power.
		Float centerRoughness  = centerNR.w - centerMaterialID;
		Float3 centerV = -luisa::compute::normalize(centerWorldPos);
		Float specLobeFraction = c.gLobeAngleFraction;
		Float specTanHalf = centerRoughness * centerRoughness * specLobeFraction
		                  / (1.0f - specLobeFraction + 1e-6f);
		Float specAngleParam = luisa::compute::atan(specTanHalf);
		specAngleParam = luisa::compute::min(specAngleParam + c.gSpecLobeAngleSlack, 1.5707963f);
		Float2 specNormalWeightParams = make_float2(specAngleParam, 1.0f);

		// Read center diffuse + fast diffuse + specular
		Float4 centerDiff = io_Diff.read(pixelPos);
		Float3 centerDiffFast = io_DiffFast.read(pixelPos).xyz();
		Float4 centerSpec = io_Spec.read(pixelPos);
		Float3 centerSpecFast = io_SpecFast.read(pixelPos).xyz();

		// Accumulators (start with center, weight=1)
		Float4 diffSum = centerDiff;
		Float3 diffFastSum = centerDiffFast;
		Float4 specSum = centerSpec;
		Float3 specFastSum = centerSpecFast;
		Float  wSum = 1.0f;
		Float  wSpecSum = 1.0f;

		// Compute stride: shorter history -> wider search
		Float baseStride = c.gHistoryFixBasePixelStride;
		Float r = luisa::compute::round(baseStride / (1.0f + historyLength));

		// 5x5 sparse cross-bilateral loop
		$for(j, -2, 3) {
			$for(i, -2, 3) {
				$if(i == 0 & j == 0) {
					$continue;
				};

				// Sample position with stride, clamped to screen bounds
				Int2 offset = make_int2(i, j) * cast<int>(r);
				UInt2 samplePos = make_uint2(clamp(
					make_int2(pixelPos) + offset,
					make_int2(0), make_int2(rectWint, rectHint)));

				// Read sample data
				Float sampleViewZ = luisa::compute::abs(gIn_ViewZ.read(samplePos).x);
				$if(sampleViewZ > denoisingRange) {
					$continue;
				};

				Float4 sampleNR = gIn_Normal_Roughness.read(samplePos);
				Float3 sampleNormal = luisa::compute::normalize(
					sampleNR.xyz() * 2.0f - 1.0f
				);

				// Sample world position
				Float2 sampleClipXY = (make_float2(samplePos) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
				Float3 sampleRayDir = frustumFwd + frustumRight * sampleClipXY.x - frustumUp * sampleClipXY.y;
				Float sampleViewZc = sampleViewZ / luisa::compute::length(sampleRayDir);
				Float3 sampleWorldPos = sampleViewZc * sampleRayDir;

				// Geometry weight: binary plane distance test
				Float planeDist = luisa::compute::abs(
					luisa::compute::dot(sampleWorldPos - centerWorldPos, centerNormal)
				);
				Float geometryW = ite(planeDist < depthThreshold, 1.0f, 0.0f);

				// Diffuse normal weight (existing pow(N·N, power) formula)
				Float normalPower = luisa::compute::max(c.gHistoryFixEdgeStoppingNormalPower, 0.01f);
				Float diffNormalW = luisa::compute::pow(
					luisa::compute::max(luisa::compute::dot(centerNormal, sampleNormal), 0.01f),
					normalPower);

				// Specular normal weight (NRD GetSpecularNormalWeight_ATrous) — view-vector aware.
				// Rejects samples from dissimilar viewing angles even when normals match,
				// preserving sharp reflections on smooth metals during disocclusion fill.
				Float3 sampleV = -luisa::compute::normalize(
					sampleWorldPos + c.gRoughnessEdgeStoppingRelaxation * centerWorldPos);
				Float cosaN = luisa::compute::dot(centerNormal, sampleNormal);
				Float cosaV = luisa::compute::dot(centerV, sampleV);
				Float cosa  = luisa::compute::min(cosaN, cosaV);
				Float specA = luisa::compute::acos(luisa::compute::clamp(cosa, -1.0f, 1.0f));
				Float specT = luisa::compute::saturate(
					specA / luisa::compute::max(specNormalWeightParams.x, 1e-6f));
				Float specSmoothed = specT * specT * (3.0f - 2.0f * specT);
				Float specNormalW = luisa::compute::saturate(
					1.0f - specSmoothed * specNormalWeightParams.y);

				// Material gates (NRD CompareMaterials): max(m0, minm) == max(m, minm)
				Float sampleMaterialID = luisa::compute::floor(sampleNR.w);
				Bool diffMatOK = luisa::compute::max(centerMaterialID, c.gDiffMinMaterial)
				              == luisa::compute::max(sampleMaterialID, c.gDiffMinMaterial);
				Bool specMatOK = luisa::compute::max(centerMaterialID, c.gSpecMinMaterial)
				              == luisa::compute::max(sampleMaterialID, c.gSpecMinMaterial);

				Float wDiff = geometryW * diffNormalW * ite(diffMatOK, 1.0f, 0.0f);
				Float wSpec = geometryW * specNormalW * ite(specMatOK, 1.0f, 0.0f);

				// Accumulate with separate weights
				$if(wDiff > 1e-4f) {
					Float4 sampleDiff = io_Diff.read(samplePos);
					Float3 sampleDiffFast = io_DiffFast.read(samplePos).xyz();
					diffSum = diffSum + sampleDiff * wDiff;
					diffFastSum = diffFastSum + sampleDiffFast * wDiff;
					wSum = wSum + wDiff;
				};
				$if(wSpec > 1e-4f) {
					Float4 sampleSpec = io_Spec.read(samplePos);
					Float3 sampleSpecFast = io_SpecFast.read(samplePos).xyz();
					specSum = specSum + sampleSpec * wSpec;
					specFastSum = specFastSum + sampleSpecFast * wSpec;
					wSpecSum = wSpecSum + wSpec;
				};
			};
		};

		// Normalize and write back (separate diffuse/specular normalization)
		Float invDiffW = 1.0f / luisa::compute::max(wSum, 1e-6f);
		io_Diff.write(pixelPos, diffSum * invDiffW);
		io_DiffFast.write(pixelPos, make_float4(diffFastSum * invDiffW, 0.0f));

		Float invSpecW = 1.0f / luisa::compute::max(wSpecSum, 1e-6f);
		io_Spec.write(pixelPos, specSum * invSpecW);
		io_SpecFast.write(pixelPos, make_float4(specFastSum * invSpecW, 0.0f));
	});

	//==========================================================================
	// ReLAX Step 6: HistoryClamping
	// Ported from RELAX_HistoryClamping.cs.hlsl (NRD v4.17)
	//
	// YCoCg color-box clamping of slow history against responsive statistics.
	// Anti-lag acceleration + history reset for lagging pixels.
	// Uses Shared<float> arrays (not Shared<float4>).
	//==========================================================================
	_relaxHistoryClamping = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat io_Diff,
		ImageFloat io_DiffFast,
		ImageFloat io_HistoryLength,
		ImageFloat gIn_DiffNoisy,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_Tiles,
		// Specular (7-9): pass-through
		ImageFloat io_Spec,
		ImageFloat io_SpecFast,
		ImageFloat gIn_SpecNoisy
		) noexcept {
		set_block_size(8u, 8u, 1u);
		set_name("history_clamping");

		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);

		UInt  rectW = cast<uint>(c.gRectSizeX);
		UInt  rectH = cast<uint>(c.gRectSizeY);
		Float2 rectSizeInv = make_float2(c.gRectSizeInvX, c.gRectSizeInvY);
		Float  denoisingRange = c.gDenoisingRange;
		// Loop-invariant int bounds; XIR has no LICM pass, so hoist explicitly.
		Int rectWint = cast<int>(rectW);
		Int rectHint = cast<int>(rectH);

		//----------------------------------------------------------------------
		// Shared-memory preload (NRD RELAX_HistoryClamping.cs.hlsl:22-53 pattern).
		// 8x8 block + 2px halo (5x5 footprint) = 12x12 tile = 144 cells.
		// 64 threads cooperatively load in 3 stages (last stage has 16 idle).
		// Replaces 25 direct reads × 5 textures per pixel with smem reads.
		//----------------------------------------------------------------------
		constexpr uint B = 8u;
		constexpr uint BORDER = 2u;
		constexpr uint T = B + 2u * BORDER;      // 12
		constexpr uint TILE_TOTAL = T * T;       // 144
		constexpr uint BLOCK_TOTAL = B * B;      // 64

		UInt2 gid = block_id().xy();
		UInt flat_tid = (pixelPos.x % B) + (pixelPos.y % B) * B;

		// Tiles: ViewZ (1 float), DiffFast/DiffNoisy/SpecFast/SpecNoisy (XYZ only, 3 floats each).
		// Center reads use direct textures (need .w channels; single tap — not the bottleneck).
		Shared<float> viewZTile(TILE_TOTAL);
		Shared<float> diffFastTile(TILE_TOTAL * 3u);
		Shared<float> diffNoisyTile(TILE_TOTAL * 3u);
		Shared<float> specFastTile(TILE_TOTAL * 3u);
		Shared<float> specNoisyTile(TILE_TOTAL * 3u);

		$for(stage, 3u) {
			UInt si = flat_tid + stage * BLOCK_TOTAL;
			$if(si < TILE_TOTAL) {
				UInt ty = si / T;
				UInt tx = si - ty * T;
				Int gxx = cast<Int>(gid.x * B + tx) - cast<Int>(BORDER);
				Int gyy = cast<Int>(gid.y * B + ty) - cast<Int>(BORDER);
				UInt2 sp = make_uint2(
					cast<UInt>(luisa::compute::max(gxx, 0)),
					cast<UInt>(luisa::compute::max(gyy, 0)));
				sp = min(sp, make_uint2(rectW - 1u, rectH - 1u));

				viewZTile.write(si, gIn_ViewZ.read(sp).x);

				Float3 df = io_DiffFast.read(sp).xyz();
				diffFastTile.write(si * 3u + 0u, df.x);
				diffFastTile.write(si * 3u + 1u, df.y);
				diffFastTile.write(si * 3u + 2u, df.z);

				Float3 dn = gIn_DiffNoisy.read(sp).xyz();
				diffNoisyTile.write(si * 3u + 0u, dn.x);
				diffNoisyTile.write(si * 3u + 1u, dn.y);
				diffNoisyTile.write(si * 3u + 2u, dn.z);

				Float3 sf = io_SpecFast.read(sp).xyz();
				specFastTile.write(si * 3u + 0u, sf.x);
				specFastTile.write(si * 3u + 1u, sf.y);
				specFastTile.write(si * 3u + 2u, sf.z);

				Float3 sn = gIn_SpecNoisy.read(sp).xyz();
				specNoisyTile.write(si * 3u + 0u, sn.x);
				specNoisyTile.write(si * 3u + 1u, sn.y);
				specNoisyTile.write(si * 3u + 2u, sn.z);
			};
		};
		sync_block();

		// Tile-based early out (after preload — matches AtrousSmem pattern)
		UInt2 tilePos = pixelPos >> 4u;
		$if(Expr{ gIn_Tiles.read(tilePos).x != 0.0f } | pixelPos.x >= rectW | pixelPos.y >= rectH) { $return(); };

		// Center smem coord (for center viewZ read + 5x5 loop indexing)
		Int cx = cast<Int>(pixelPos.x) - cast<Int>(gid.x * B) + cast<Int>(BORDER);
		Int cy = cast<Int>(pixelPos.y) - cast<Int>(gid.y * B) + cast<Int>(BORDER);
		UInt centerSIdx = cast<UInt>(cx) + cast<UInt>(cy) * T;

		// Read center ViewZ from smem; reject sky/out-of-range
		Float centerViewZ = luisa::compute::abs(viewZTile.read(centerSIdx));
		$if(centerViewZ > denoisingRange | centerViewZ == 0.0f) { $return(); };

		Float4 _hlRead = io_HistoryLength.read(pixelPos);
		Float historyLength = 255.0f * _hlRead.x;
		Float specConf = _hlRead.y;

		// Center fast history in YCoCg (read directly, no shared memory)
		Float4 centerFast = io_DiffFast.read(pixelPos);
		// Center specular data
		Float4 centerSpec = io_Spec.read(pixelPos);
		// Float4 read so the .w (fast-history hitDist from TA:931) survives
		// the YCoCg clamping below and can be passed through unchanged.
		Float4 _centerSpecFastFull = io_SpecFast.read(pixelPos);
		Float3 centerSpecFast = _centerSpecFastFull.xyz();
		Float  centerSpecFastHitDist = _centerSpecFastFull.w;
		Float centerSpecLum = luminance(centerSpec.xyz());
		Float cfR = centerFast.x; Float cfG = centerFast.y; Float cfB = centerFast.z;
		Float cyY = cfR * 0.25f + cfG * 0.5f + cfB * 0.25f;
		Float cyCo = cfR * 0.5f + cfB * -0.5f;
		Float cyCg = cfR * -0.25f + cfG * 0.5f + cfB * -0.25f;

		// Center noisy for firefly rejection
		Float3 centerNoisy = gIn_DiffNoisy.read(pixelPos).xyz();
		//Float3 lw = luisa::make_float3(0.2126f, 0.7152f, 0.0722f);
		Float fnR = centerNoisy.x; Float fnG = centerNoisy.y; Float fnB = centerNoisy.z;
		Float fnLum = luminance(centerNoisy);// make_float3(fnR, fnG, fnB));

		//----------------------------------------------------------------------
		// 5x5 statistics (smem reads — direct texture reads replaced)
		//----------------------------------------------------------------------
		Float3 rfm = def(make_float3(0.0f));
		Float3 rsm = def(make_float3(0.0f));
		Float3 nfm = def(make_float3(0.0f));
		Float nsmLum = def(0.0f);
		// Specular YCoCg statistics (separate from diffuse)
		Float3 srfm = def(make_float3(0.0f));
		Float3 srsm = def(make_float3(0.0f));
		Float3 snfm = def(make_float3(0.0f));
		Float snsmLum = def(0.0f);
		Float sumW = def(0.0f);
#if NT_DEBUG_VIZ
		Float sumW_boundsOnly = def(0.0f);
#endif

		$for(dy, -2, 3) {
			$for(dx, -2, 3) {
				// InBounds check against rect (preserves original semantics —
				// smem edge-clamps but we still need to know which cells were real).
				Int nx = cast<Int>(pixelPos.x) + dx;
				Int ny = cast<Int>(pixelPos.y) + dy;
				Bool inBounds = nx >= 0 & nx < rectWint & ny >= 0 & ny < rectHint;

				UInt sx = cast<UInt>(cx + dx);
				UInt sy = cast<UInt>(cy + dy);
				UInt sIdx = sx + sy * T;

				Float vz = luisa::compute::abs(viewZTile.read(sIdx));

				$if(inBounds & vz < denoisingRange & vz > 0.0f) {
					Float3 resp = make_float3(
						diffFastTile.read(sIdx * 3u + 0u),
						diffFastTile.read(sIdx * 3u + 1u),
						diffFastTile.read(sIdx * 3u + 2u));
					Float3 ycocg = make_float3(dot(resp, luisa::make_float3(0.25f, 0.5f, 0.25f)),
						dot(resp, luisa::make_float3(0.5f, 0.0f, -0.5f)),
						dot(resp, luisa::make_float3(-0.25f, 0.5f, -0.25f)));
					rfm = rfm + ycocg;
					rsm = rsm + ycocg * ycocg;

					Float3 noisy = make_float3(
						diffNoisyTile.read(sIdx * 3u + 0u),
						diffNoisyTile.read(sIdx * 3u + 1u),
						diffNoisyTile.read(sIdx * 3u + 2u));
					Float nlum = luminance(noisy);
					nfm = nfm + noisy;
					nsmLum = nsmLum + nlum * nlum;

					// Specular YCoCg statistics
					Float3 sResp = make_float3(
						specFastTile.read(sIdx * 3u + 0u),
						specFastTile.read(sIdx * 3u + 1u),
						specFastTile.read(sIdx * 3u + 2u));
					Float3 sycocg = make_float3(dot(sResp, luisa::make_float3(0.25f, 0.5f, 0.25f)),
						dot(sResp, luisa::make_float3(0.5f, 0.0f, -0.5f)),
						dot(sResp, luisa::make_float3(-0.25f, 0.5f, -0.25f)));
					srfm = srfm + sycocg;
					srsm = srsm + sycocg * sycocg;

					Float3 sNoisy = make_float3(
						specNoisyTile.read(sIdx * 3u + 0u),
						specNoisyTile.read(sIdx * 3u + 1u),
						specNoisyTile.read(sIdx * 3u + 2u));
					Float snlum = luminance(sNoisy);
					snfm = snfm + sNoisy;
					snsmLum = snsmLum + snlum * snlum;

					sumW = sumW + 1.0f;
				};
#if NT_DEBUG_VIZ
				sumW_boundsOnly = sumW_boundsOnly + ite(inBounds, 1.0f, 0.0f);
#endif
			};
		};

		// Background pixels fail the inBounds/viewZ test on all 25 taps, so sumW can be 0
		Float invW = 1.0f / luisa::compute::max(sumW, 1e-6f);
		Float3 rm = rfm * invW;
		Float3 rs = rsm * invW;
		Float3 nm = nfm * invW;
		Float nsL = nsmLum * invW;

		// Specular mean/sigma
		Float3 srm = srfm * invW;
		Float3 srs = srsm * invW;
		Float3 snm = snfm * invW;

		// Sigma in YCoCg (with minimum floor to prevent over-tight clamping boxes)
		Float minSigma = 0.01f;
		Float3 sigma = luisa::compute::max(make_float3(minSigma), 
					luisa::compute::sqrt(luisa::compute::max(make_float3(0.0f), rs - rm * rm)));

		// Variance boost for early frames (NRD: wider clamping box when history is short)
		Float varianceBoost = luisa::compute::max(1.0f, luisa::compute::sqrt(4.0f / (historyLength + 1.0f)));
		sigma = sigma * varianceBoost;

		Float ss = c.gFastHistoryClampingSigmaScale;
		Float3 cmin = rm - ss * sigma;
		Float3 cmax = rm + ss * sigma;

		// Expand box with center pixel
		Float3 centerYCgCo = make_float3(cyY, cyCo, cyCg);
		cmin = luisa::compute::min(cmin, centerYCgCo);
		cmax = luisa::compute::max(cmax, centerYCgCo);

		//----------------------------------------------------------------------
		// Clamp slow history in YCoCg
		//----------------------------------------------------------------------
		Float4 diff2nd = io_Diff.read(pixelPos);
		Float dr = diff2nd.x; Float dg = diff2nd.y; Float db = diff2nd.z;
		Float dy = dr * 0.25f + dg * 0.5f + db * 0.25f;
		Float dco = dr * 0.5f + db * -0.5f;
		Float dcg = dr * -0.25f + dg * 0.5f + db * -0.25f;

		// NOTE: Do NOT expand box with slow history (NRD does not do this).
		// Expanding with slow history means clamp(slow, cmin, cmax) is always
		// a no-op since slow is already inside the box. This defeats clamping
		// entirely and causes ghost images around bright emissive geometries.

		Float cy1 = dy; Float cy2 = dco; Float cy3 = dcg;
		Bool shouldClamp = c.gDiffMaxFastAccumulatedFrameNum < c.gDiffMaxAccumulatedFrameNum;
		$if(shouldClamp & c.gDisableClamp == 0u) {
			Float3 clamped = luisa::compute::clamp(make_float3(dy, dco, dcg), cmin, cmax);
			cy1 = clamped.x; cy2 = clamped.y; cy3 = clamped.z;
		};

		// YCoCg -> RGB
		Float cr = cy1 + cy2 - cy3;
		Float cg = cy1 + cy3;
		Float cb = cy1 - cy2 - cy3;

		// Center responsive in RGB
		Float rcR = cyY + cyCo - cyCg;
		Float rcG = cyY + cyCg;
		Float rcB = cyY - cyCo - cyCg;

		// Outputs (scalars to avoid Float4 member issues)
		Float oR = cr; Float oG = cg; Float oB = cb; Float oA = diff2nd.w;
		Float oFR = rcR; Float oFG = rcG; Float oFB = rcB;

		// NRD: If history length <= gHistoryFixFrameNum, recently disoccluded.
		// Replace slow history output with fast history center to avoid
		// stale bright values from emissive geometries bleeding through.
		$if(historyLength <= c.gHistoryFixFrameNum) {
			oR = rcR; oG = rcG; oB = rcB;
		};

		//----------------------------------------------------------------------
		// Clamping factor
		//----------------------------------------------------------------------
		Float cfDelta = cy1 - dy;
		Float cfDenom = cyY - dy;
		Float cf = def(0.0f);
		$if(luisa::compute::abs(cfDelta) < 1.0e-6f | luisa::compute::abs(cfDenom) < 1.0e-6f) {
			cf = 0.0f;
		} $else{
			cf = luisa::compute::saturate(cfDelta / cfDenom);
		};
		$if(historyLength <= c.gHistoryFixFrameNum) { cf = 1.0f; };

		//----------------------------------------------------------------------
		// Anti-lag acceleration
		//----------------------------------------------------------------------
		Float adR = luisa::compute::abs(rcR - dr);
		Float adG = luisa::compute::abs(rcG - dg);
		Float adB = luisa::compute::abs(rcB - db);
		Float hdl = 10.0f * c.gHistoryAccelerationAmount * luminance(make_float3(adR, adG, adB));// (adR* lw.x + adG * lw.y + adB * lw.z);
		hdl = hdl * cf;
		hdl = hdl * ite(historyLength <= c.gHistoryFixFrameNum, 0.0f, 1.0f);
		hdl = ite(c.gDisableAntilag != 0u, 0.0f, hdl);

		Float3 cdVec = nm - make_float3(rcR, rcG, rcB);
		Float cdl = luminance(luisa::compute::abs(cdVec));// *lw.x + luisa::compute::abs(cdG) * lw.y + luisa::compute::abs(cdB) * lw.z;

		Float3 caDiff = ite(cdl != 0.0f, cdVec * (hdl / cdl), make_float3(0.0f));

		Float caL = luminance(luisa::compute::abs(caDiff));
		Float caRatio = ite(caL == 0.0f, 0.0f, cdl / caL);
		caDiff = caDiff * ite(caRatio < 1.0f, caRatio, 1.0f);
		caDiff = ite(caRatio <= 0.0f, make_float3(0.0f), caDiff);

		// Clamp anti-lag correction lower bound only (NRD has no upper clamp).
		Float3 oCorr = luisa::compute::max(make_float3(oR, oG, oB) + caDiff, make_float3(0.0f));
		oR = oCorr.x; oG = oCorr.y; oB = oCorr.z;

		//----------------------------------------------------------------------
		// Anti-lag reset
		//----------------------------------------------------------------------
		Float dL = luminance(make_float3(dr, dg, db));// dr * lw.x + dg * lw.y + db * lw.z;
		Float niL = luminance(nm);
		Float tSig = c.gHistoryResetTemporalSigmaScale * luisa::compute::sqrt(luisa::compute::max(0.0f, nsL - niL * niL));
		Float sSig = c.gHistoryResetSpatialSigmaScale * sigma.x;
		Float ra = 0.5f * c.gHistoryResetAmount *
			luisa::compute::max(0.0f, luisa::compute::abs(dL - niL) - sSig - tSig) /
			(1.0e-6f + luisa::compute::max(dL, niL) + sSig + tSig);
		ra = luisa::compute::saturate(ra);
		ra = ra * ite(historyLength <= c.gHistoryFixFrameNum, 0.0f, 1.0f);
		ra = ite(c.gDisableAntilag != 0u, 0.0f, ra);

		// Anti-lag reset toward center noisy (NRD: resets both slow AND fast)
		oR = lerp(oR, fnR, ra); oG = lerp(oG, fnG, ra); oB = lerp(oB, fnB, ra);
		oFR = lerp(oFR, fnR, ra); oFG = lerp(oFG, fnG, ra); oFB = lerp(oFB, fnB, ra);

		//----------------------------------------------------------------------
		// 2nd moment correction
		//----------------------------------------------------------------------
		Float outL = luminance(make_float3(oR, oG, oB));// oR* lw.x + oG * lw.y + oB * lw.z;
		Float mc = outL * outL - dL * dL;
		Float c2nd = luisa::compute::max(0.0f, oA + mc);

		// Clamp outputs to non-negative
		oR = luisa::compute::max(oR, 0.0f); oG = luisa::compute::max(oG, 0.0f); oB = luisa::compute::max(oB, 0.0f);
		oFR = luisa::compute::max(oFR, 0.0f); oFG = luisa::compute::max(oFG, 0.0f); oFB = luisa::compute::max(oFB, 0.0f);
		Float outL2 = luminance(make_float3(oR, oG, oB));// oR * lw.x + oG * lw.y + oB * lw.z;
		c2nd = luisa::compute::max(c2nd, luisa::compute::max(outL2 * outL2 * 0.01f, 0.01f));

		// NRD passthrough (RELAX_HistoryClamping.cs.hlsl:353) — history length written
		// through unchanged. antilagFactor kept for debug viz only (mode 7).
		Float antilagFactor = luisa::compute::saturate(ra * c.gAntilagYoungHistoryScale);
		Float hlOut = historyLength;

		//----------------------------------------------------------------------
		// Debug visualization (gDebugViz != 0)
		//----------------------------------------------------------------------
#if NT_DEBUG_VIZ
		UInt debugViz = c.gDebugViz;
		$if(debugViz != 0u) {
			Float vizR = def(0.0f); Float vizG = def(0.0f); Float vizB = def(0.0f);

			$if(debugViz == 1u) {
				Float t = luisa::compute::saturate(historyLength / 30.0f);
				vizR = specConf;// 1.0f - t;
				vizG = ite(t < 0.5f, t * 2.0f, 2.0f - t * 2.0f);
				vizB = t;
			}
			$elif(debugViz == 2u) {
				vizR = sumW / 25.0f;
				vizG = sumW / 25.0f;
				vizB = sumW / 25.0f;
				$if(sumW < 22.0f) { vizR = 1.0f; vizG = 0.0f; vizB = 0.0f; };
			}
			$elif(debugViz == 3u) {
				vizR = 1.0f - cf;
				vizG = cf;
				vizB = 1.0f;
			}
			$elif(debugViz == 4u) {
				Float antiLagLum =
					luminance(luisa::compute::abs(caDiff));
				/*luisa::compute::abs(caR)* lw.x
				+ luisa::compute::abs(caG) * lw.y
				+ luisa::compute::abs(caB) * lw.z;*/
				vizR = luisa::compute::min(antiLagLum * 10.0f, 1.0f);
				vizG = luisa::compute::min(antiLagLum * 10.0f, 1.0f);
				vizB = vizR;
			}
			$elif(debugViz == 5u) {
				Float centerVZ = luisa::compute::abs(gIn_ViewZ.read(pixelPos).x);
				Float vzNorm = luisa::compute::saturate(centerVZ / 20.0f);
				vizR = vzNorm; vizG = vzNorm; vizB = vzNorm;
				$if(centerVZ > denoisingRange) { vizR = 1.0f; vizG = 0.0f; vizB = 0.0f; };
				$if(centerVZ == 0.0f) { vizR = 0.0f; vizG = 0.0f; vizB = 1.0f; };
			}
			$elif(debugViz == 6u) {
				vizR = sumW_boundsOnly / 25.0f;
				vizG = sumW_boundsOnly / 25.0f;
				vizB = sumW_boundsOnly / 25.0f;
				$if(sumW_boundsOnly < 22.0f) { vizR = 1.0f; vizG = 0.0f; vizB = 0.0f; };
			}
			$elif(debugViz == 7u) {
				// Anti-lag reset amount ra: bright = high reset, dark = no reset
				vizR = luisa::compute::saturate(ra * 5.0f);
				vizG = ra;
				vizB = luisa::compute::saturate(antilagFactor * 2.0f);
			};

			io_Diff.write(pixelPos, make_float4(vizR, vizG, vizB, c2nd));
			io_DiffFast.write(pixelPos, make_float4(vizR, vizG, vizB, 0.0f));
			io_HistoryLength.write(pixelPos, make_float4(hlOut / 255.0f, specConf, 0.0f, 0.0f));
		} $else{
#endif
			io_Diff.write(pixelPos, make_float4(oR, oG, oB, c2nd));
			io_DiffFast.write(pixelPos, make_float4(oFR, oFG, oFB, 0.0f));
			io_HistoryLength.write(pixelPos, make_float4(hlOut / 255.0f, specConf, 0.0f, 0.0f));
#if NT_DEBUG_VIZ
		};
#endif

		// ====== Specular YCoCg clamping (own statistics) ======
		Float sR = centerSpec.x; Float sG = centerSpec.y; Float sB = centerSpec.z;
		Float syY = sR * 0.25f + sG * 0.5f + sB * 0.25f;
		Float syCo = sR * 0.5f + sB * -0.5f;
		Float syCg = sR * -0.25f + sG * 0.5f + sB * -0.25f;

		// Compute specular sigma and box from specular neighborhood stats.
		// NRD: no variance boost on specular — widening the clamp box here leaks fireflies through.
		Float3 ssigma = luisa::compute::max(make_float3(minSigma), luisa::compute::sqrt(luisa::compute::max(make_float3(0.0f), srs - srm * srm)));

		Float sss = c.gFastHistoryClampingSigmaScale;
		Float3 scmin = srm - sss * ssigma;
		Float3 scmax = srm + sss * ssigma;

		// Expand box with FAST specular center (NRD: uses responsive center, not slow history)
		Float3 sfExp = make_float3(
			dot(centerSpecFast, luisa::make_float3(0.25f, 0.5f, 0.25f)),
			dot(centerSpecFast, luisa::make_float3(0.5f, 0.0f, -0.5f)),
			dot(centerSpecFast, luisa::make_float3(-0.25f, 0.5f, -0.25f)));
		scmin = luisa::compute::min(scmin, sfExp);
		scmax = luisa::compute::max(scmax, sfExp);

		Float scy1 = syY; Float scy2 = syCo; Float scy3 = syCg;
		Bool specShouldClamp = c.gSpecMaxFastAccumulatedFrameNum < c.gSpecMaxAccumulatedFrameNum;
		$if(specShouldClamp& c.gDisableClamp == 0u) {
			Float3 sClamped = luisa::compute::clamp(make_float3(syY, syCo, syCg), scmin, scmax);
			scy1 = sClamped.x; scy2 = sClamped.y; scy3 = sClamped.z;
		};

		// YCoCg -> RGB
		Float soR = scy1 + scy2 - scy3;
		Float soG = scy1 + scy3;
		Float soB = scy1 - scy2 - scy3;
		soR = luisa::compute::max(soR, 0.0f); soG = luisa::compute::max(soG, 0.0f); soB = luisa::compute::max(soB, 0.0f);

		// Specular fast: clamp using specular box (reuse sfExp = YCoCg of centerSpecFast)
		Float sfY = sfExp.x; Float sfCo = sfExp.y; Float sfCg = sfExp.z;
		$if(specShouldClamp& c.gDisableClamp == 0u) {
			Float3 sfClamped = luisa::compute::clamp(sfExp, scmin, scmax);
			sfY = sfClamped.x; sfCo = sfClamped.y; sfCg = sfClamped.z;
		};
		Float sfr = sfY + sfCo - sfCg;
		Float sfg = sfY + sfCg;
		Float sfb = sfY - sfCo - sfCg;
		sfr = luisa::compute::max(sfr, 0.0f); sfg = luisa::compute::max(sfg, 0.0f); sfb = luisa::compute::max(sfb, 0.0f);


		// NRD: If recently disoccluded, replace slow specular with fast specular center
		$if(historyLength <= c.gHistoryFixFrameNum) {
			soR = sfr; soG = sfg; soB = sfb;
		};

		// ====== Specular anti-lag acceleration (NRD: 0.33x scale) ======
		// NRD: specular clamping factor from specular YCoCg stats (not diffuse)
		Float specCF = ite(luisa::compute::abs(scy1 - syY) < 1.0e-6f, 0.0f,
			luisa::compute::saturate((scy1 - syY) / (sfY - syY)));
		$if(historyLength <= c.gHistoryFixFrameNum) { specCF = 1.0f; };
		Float sadR = luisa::compute::abs(sfr - sR);
		Float sadG = luisa::compute::abs(sfg - sG);
		Float sadB = luisa::compute::abs(sfb - sB);
		Float shdl = 3.33f * c.gHistoryAccelerationAmount * (luminance(make_float3(sadR, sadG, sadB)));
		shdl = shdl * specCF;
		shdl = shdl * ite(historyLength <= c.gHistoryFixFrameNum, 0.0f, 1.0f);
		shdl = ite(c.gDisableAntilag != 0u, 0.0f, shdl);

		Float3 scdVec = snm - make_float3(sfr, sfg, sfb);
		Float scdl = luminance(luisa::compute::abs(scdVec));
		//luisa::compute::abs(scdR) * lw.x + luisa::compute::abs(scdG) * lw.y + luisa::compute::abs(scdB) * lw.z;

		Float3 scaDiff = ite(scdl != 0.0f, scdVec * (shdl / scdl), make_float3(0.0f));

		// Overshoot prevention
		Float scaL = luminance(luisa::compute::abs(scaDiff));
		Float scaRatio = ite(scaL == 0.0f, 0.0f, scdl / scaL);
		scaDiff = scaDiff * ite(scaRatio < 1.0f, scaRatio, 1.0f);
		scaDiff = ite(scaRatio <= 0.0f, make_float3(0.0f), scaDiff);

		Float3 soCorr = luisa::compute::max(make_float3(soR, soG, soB) + scaDiff, make_float3(0.0f));
		//soR = soCorr.x; soG = soCorr.y; soB = soCorr.z;

		// Specular history reset (NRD: 0.5x � specular has extra rejection heuristics)
		Float sL = luminance(make_float3(sR, sG, sB));
		Float snL = luminance(snm);
		Float sNoisyTemporalSig = c.gHistoryResetTemporalSigmaScale *
			luisa::compute::sqrt(luisa::compute::max(0.0f,
				snsmLum * invW - snL * snL));
		Float sNoisySpatialSig = c.gHistoryResetSpatialSigmaScale * ssigma.x;
		Float specRA = 0.5f * c.gHistoryResetAmount *
			luisa::compute::max(0.0f, luisa::compute::abs(sL - snL) - sNoisySpatialSig - sNoisyTemporalSig) /
			(1.0e-6f + luisa::compute::max(sL, snL) + sNoisySpatialSig + sNoisyTemporalSig);
		specRA = luisa::compute::saturate(specRA);
		specRA = specRA * ite(historyLength <= c.gHistoryFixFrameNum, 0.0f, 1.0f);
		specRA = ite(c.gDisableAntilag != 0u, 0.0f, specRA);
		// Reset specular toward noisy input
		Float3 sNoisyCenter = gIn_SpecNoisy.read(pixelPos).xyz();
		soCorr = lerp(soCorr, sNoisyCenter, specRA);
		//soR = lerp(soR, sNoisyCenter.x, specRA);
		//soG = lerp(soG, sNoisyCenter.y, specRA);
		//soB = lerp(soB, sNoisyCenter.z, specRA);
		sfr = lerp(sfr, sNoisyCenter.x, specRA);
		sfg = lerp(sfg, sNoisyCenter.y, specRA);
		sfb = lerp(sfb, sNoisyCenter.z, specRA);

		// Specular 2nd moment correction after anti-lag
		// NRD RELAX_HistoryClamping.cs.hlsl:235-236 - just max(0, m + correction).
		// Previous code added an extra max(sOutL²·0.01, 0.01) floor that NRD does
		// NOT have; it prevented variance collapse on metals where NRD would
		// settle, leaving a permanent slight noise bias.
		Float sOutL = luminance(soCorr);// make_float3(soR, soG, soB));
		Float sInL = luminance(make_float3(sR, sG, sB));
		Float smc = sOutL * sOutL - sInL * sInL;
		Float sc2nd = luisa::compute::max(0.0f, centerSpec.w + smc);

		io_Spec.write(pixelPos, make_float4(soCorr, sc2nd));// make_float4(soR, soG, soB, sc2nd));
		// NRD: accelerate fast specular too
		Float3 sfCorr = luisa::compute::max(make_float3(sfr, sfg, sfb) + scaDiff, make_float3(0.0f));
		// Preserve .a (fast hitDist) from input — NRD RELAX_HistoryClamping.cs.hlsl:240
		// carries outSpecularResponsive.a through unchanged. Zeroing it (as before)
		// starves next frame's Stage 2 virtual motion of a valid hitDistForTrackingPrev.
		io_SpecFast.write(pixelPos, make_float4(sfCorr, centerSpecFastHitDist));
	});
}

}