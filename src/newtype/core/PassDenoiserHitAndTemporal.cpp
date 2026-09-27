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

// CDSL-workaround: the compiled normalize() intrinsic in this kernel returns
// ~2x the unit vector (in-block V measured |1.96|, direction exact; explicit
// rsqrt(dot) form measured |0.95-1.0|). Every direction vector below goes
// through this helper instead of luisa::compute::normalize.
[[nodiscard]] inline Float3 norm3(Float3 v) noexcept {
	return v * luisa::compute::rsqrt(luisa::compute::dot(v, v));
}

void RelaxDenoiser::compileHitAndTemporal(Device& device) {
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
				// Split-kernel VMB prepass outputs (e55). Info2 is read
				// unconditionally (dominanceFactor/hitDist/flag feed
				// out-of-gate math); Info + FetchA/B/C are read only inside
				// $if(virtualReprojectionFound).
				ImageFloat vmbIn_Info,
				ImageFloat vmbIn_Info2,
				ImageFloat vmbIn_FetchA,
				ImageFloat vmbIn_FetchB,
				ImageFloat vmbIn_FetchC
			) noexcept {
		set_block_size(8u, 16u, 1u);
		set_name("temporal_accumulation_v15");

		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);
		// Compile-time projection tag (§9 perf fix): C++ constant — every
		// projection condition in this kernel folds in DXC, so the
		// perspective build keeps the pre-projection instruction sequence.
		const uint bakedProjection = _bakedProjection;

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
		// (The min-3x3 hitDist no longer needs reads here at all: the VMB
		// prepass computes the identical value — same gIn_Spec binding, same
		// checkerboard fill, same >0 neighbor filter, same INF->0 mapping —
		// and transports it through Info2.z, fp32-exact.)
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

		// R3 wave/smem pass: vector tile — one float4 element per cell (same
		// 180-cell footprint, same values); 1 vector store/load per cell
		// instead of 4 scalar ops, and consecutive-float4 access is bank-
		// conflict-free by construction (the scalar ×4 layout paid 2-way
		// conflicts on the wrap groups of the 10/18-stride tap reads).
		Shared<float4> normalRoughnessTile(TILE_TOTAL);

		//----------------------------------------------------------------------
		// Block-uniform sky-tile early out BEFORE the smem preload (NRD
		// PRELOAD_INTO_SMEM_WITH_TILE_CHECK pattern, Common.hlsli:128-145):
		// an 8x16 block lies inside a single 16x16 tile (x spans 8 pixels at
		// an 8-aligned origin, y spans 16 at a 16-aligned origin), so the
		// branch cannot diverge within the block — the cooperative preload
		// and its sync_block are skipped entirely for all-sky tiles. Only the
		// sky half of the former combined gate may move up here; the
		// per-pixel bounds half must stay BELOW the preload/sync (edge
		// blocks are not block-uniform and must not skip the barrier).
		// Sentinel writes are the block at the bottom of this patch.
		//----------------------------------------------------------------------
		$if(Expr{ gIn_Tiles.read(pixelPos >> 4u).x != 0.0f }) {
			// Sky tile: history must read as "unrejectable background".
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
				normalRoughnessTile.write(si, nr);
			};
		};
		sync_block();

		// Out-of-rect threads (bounds half of the former combined gate — kept
		// after the preload for barrier uniformity; same sentinel writes).
		$if(pixelPos.x >= rectW | pixelPos.y >= rectH) {
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
		// NRD specularIllumination.a: post-PrePass hit distance (stochastic min).
		// The former 3x3-bilateral "smoothed" hit-dist pass was a local invention
		// (NRD's HitDistReconstruction is zero-fill, default OFF) and is retired —
		// virtual motion must be min-biased, not mean-biased (RC2).
		Float specHitDist = specInput.w;

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
				Float hdL = gIn_Spec.read(make_uint2(nbLx, pixelPos.y)).w;
				Float hdR = gIn_Spec.read(make_uint2(nbRx, pixelPos.y)).w;
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
		Float3 currentNormal = norm3(currentNormalRoughness.xyz());
		// Encoded [-1,1] -> [0,1] form, used by both NR-packing write sites below.
		Float3 currentNormalEncoded = currentNormal * 0.5f + 0.5f;
		// .w packs matID (integer) + roughness (fraction). Unpack both.
		Float  currentPackedNR   = currentNormalRoughness.w;
		Float  currentMaterialID = luisa::compute::floor(currentPackedNR);
		Float  currentRoughness  = currentPackedNR - currentMaterialID;

		// 3x3 normal average for modified roughness (NRD: GetModifiedRoughnessFromNormalVariance)
		// When normals disagree (edges), increase effective roughness -> relax specular tracking
		Float3 currentNormalAvg = currentNormal;
		// Center smem coord (matches _relaxHitDistReconstruct:87-89 pattern, adapted for 8x16 block).
		Int cx = cast<Int>(pixelPos.x) - cast<Int>(gid.x * Bx) + cast<Int>(BORDER);
		Int cy = cast<Int>(pixelPos.y) - cast<Int>(gid.y * By) + cast<Int>(BORDER);
		$for(dy, -1, 2) {
			$for(dx, -1, 2) {
				$if(dx == 0 & dy == 0) { $continue; };
				UInt sx = cast<UInt>(cx + dx);
				UInt sy = cast<UInt>(cy + dy);
				UInt sIdx = sx + sy * Tx;
				Float4 nrSample = normalRoughnessTile.read(sIdx);
				Float3 nSample = norm3(nrSample.xyz());
				currentNormalAvg = currentNormalAvg + nSample;
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
		// Point-origin projections (§4 seams & poles): the equirect/cylindrical
		// u axis wraps — fold prevUV.x into [0,1) so the u seam reprojects
		// through the opposite edge instead of resetting history every frame.
		// `projection` is the BAKED literal (§9 perf fix): a literal-valued
		// UInt — every condition below folds in DXC while keeping the
		// original runtime-code operator forms.
		const uint projection = _bakedProjection;
		const bool wrapU = projection == 1u || projection == 2u;
		if (wrapU) {
			prevUV.x = luisa::compute::fract(prevUV.x);
		}

		// World position and view vector (needed for NoV disocclusion + virtual motion)
		// UV CONVENTION (proven by the viz epoch marker landing at screen BOTTOM):
		// the render target's row 0 is the screen BOTTOM - pixel uv is Y-UP. So
		// clipXY = uv*2-1 IS the NDC xy directly, and the reconstruction is
		// viewZ * (fwd + right*cx + up*cy). COROLLARY: every matrix projection in
		// this kernel maps clip -> uv with clip.xy*(0.5,+0.5)+0.5 (NOT NRD's -0.5
		// form, which assumes y-down uv and vertically mirrors the result - the
		// original source of the VMB band: prevUV_vmb was the vertical mirror of
		// the correct fetch position, so tap tests only passed near the
		// mirror-symmetric screen row).
		// (clipXY = uv*2-1 kept inline at the reconstruction below.)
		// Position reconstruction MUST scale the lateral terms by viewZ: the
		// frustum constants are ray-direction basis vectors (right = right*aspect*
		// tan, up = up*tan), so the true point is z*(fwd + right*cx + up*cy) —
		// the exact inverse of Camera::generate_ray. The former z-less form
		// (fwd*z + right*cx + up*cy) reconstructs an off-ray point whose
		// projection collapses radially toward the principal point:
		// uv' = 0.5 + (uv-0.5)/viewZ — the VMB fetch-UV "collapse" measured all
		// week (mode 27: 0.5+0.155*(px-0.5); slope 1/z = 0.1-0.35 over the
		// sheet's depth range; band at uv.y ~ 0.5, the law's only fixed point).
		// Verified in isolation by tools/vmb_repro (DSL == CPU == analytic law;
		// any on-ray offset leaves the collapsed uv unchanged — hence no
		// hitDist/curvature/codegen workaround could move it).
		Float3 currentWorldPos = def(make_float3(0.0f));
		if (projection == 0u) {
			currentWorldPos = currentViewZ * (c.gFrustumForward.xyz()
				+ c.gFrustumRight.xyz() * (pixelUv.x * 2.0f - 1.0f)
				+ c.gFrustumUp.xyz() * (pixelUv.y * 2.0f - 1.0f));
		} else {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
			// Point-origin projections: viewZ is the ray parameter t along the
			// pixel's OWN unit direction, so worldPos = viewZ * dir(uv) is exact
			// (same relative-position semantics as the frustum form above).
			currentWorldPos = currentViewZ * util::eval_point_origin_direction(
				pixelUv * 2.0f - 1.0f,
				pc.gFrustumRight.xyz(), pc.gFrustumUp.xyz(), pc.gFrustumForward.xyz(),
				bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
		}
		// (The rsqrt form is kept from the e37-era "normalize() returns 2x"
		// workaround — disproven by tools/vmb_repro ([6] both forms unit) —
		// but harmless and marginally cheaper.)
		Float3 V = (-currentWorldPos) * luisa::compute::rsqrt(
			luisa::compute::dot(currentWorldPos, currentWorldPos));
		// CDSL-workaround #3: gPrevFrustum* struct-member accesses miscompile
		// when read inside the tap-test context (measured reading the
		// gCamPosCur slot's content — gPrevFrustumUp ≈ C_cur·1.1 — which shears
		// every tap position vertically ∝ clipY, zero at the center row: THE
		// rim band). Hoisted into top-level locals, where member access is
		// verified correct (end-of-kernel dump matches the host bytes).
		Float3 prevFwdV = c.gPrevFrustumForward.xyz();
		Float3 prevRightV = c.gPrevFrustumRight.xyz();
		Float3 prevUpV = c.gPrevFrustumUp.xyz();
		Float3 camPosPrevV = c.gCamPosCur.xyz() - c.gCameraDelta.xyz();
		Float3 camPosCurV = c.gCamPosCur.xyz();
		// CDSL-workaround #4: the matrix operator* itself miscompiles when it
		// sits inside the deep virtual-motion $if — the projected VMB fetch
		// collapses toward the screen center (in-block prevUV_vmb ≈ (0.5, 0.5)
		// was measured twice: the night-session dump and the e49 mode-27
		// probe). Copy the matrix into a top-level local so no struct read is
		// rematerialized in-block; the three in-block projections (curvature
		// guard uv1/uv2 and the virtual UV) use the local.
		auto prevClipMat = c.gWorldToClipPrev;
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

			// NRD TA:475-479 — two-sided surface parallax (NRD Common.hlsli:339-343).
			// Term1 (project the current surface point into the previous frame, vs
			// the motion-reprojected uv) is the translation-sensitive
			// motion-vector-consistency residual — ≈0 for consistent motion
			// vectors. Term2 = the surface's screen-space motion — the
			// rotation-sensitive complement (NRD derives the same split by
			// projecting prev-frame CAMERA-RELATIVE coords with the current
			// matrix; here the motion vectors already carry the full
			// rotation+translation displacement). Max ≈ full surface motion, as
			// NRD; Min ≈ 0 under rotation.
			// prevCamRelativePos is a CURRENT-frame camera-relative point; world
			// conversions add gCamPosCur (see the projections below).
			Float3 prevCamRelativePos = currentWorldPos;
			Float2 uvReprojPrev = def(make_float2(0.0f));
			if (projection == 0u) {
				Float4 pcPrev = c.gWorldToClipPrev * make_float4(prevCamRelativePos + c.gCamPosCur.xyz(), 1.0f);
				uvReprojPrev = pcPrev.xy() * (1.0f / pcPrev.w) * luisa::make_float2(0.5f, 0.5f)
					+ luisa::make_float2(0.5f, 0.5f);
			} else {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
				// Point-origin: analytic prev-UV of the world point (direction
				// from the PREV camera position, prev basis).
				Float3 wpAbs = prevCamRelativePos + c.gCamPosCur.xyz();
				uvReprojPrev = util::project_point_origin_direction(
					normalize(wpAbs - (c.gCamPosCur.xyz() - c.gCameraDelta.xyz())),
					pc.gPrevFrustumRight.xyz(), pc.gPrevFrustumUp.xyz(), pc.gPrevFrustumForward.xyz(),
					bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y) * 0.5f + 0.5f;
				if (wrapU) { uvReprojPrev.x = luisa::compute::fract(uvReprojPrev.x); }
			}
			Float smbParallaxInPixels1 = luisa::compute::length((uvReprojPrev - prevUV) * rectSizePrev);
			Float smbParallaxInPixels2 = luisa::compute::length((prevUV - pixelUv) * rectSizePrev);
			Float smbParallaxInPixelsMax = luisa::compute::max(smbParallaxInPixels1, smbParallaxInPixels2);
			Float smbParallaxInPixelsMin = luisa::compute::min(smbParallaxInPixels1, smbParallaxInPixels2);

			// Disocclusion threshold with NoV slope scaling (NRD TA:112-117)
			Float disocclusionThreshold = c.gDisocclusionThreshold;
			Float pixelSize = c.gUnproject * currentViewZ;
			Float frustumSize = pixelSize * luisa::compute::min(rectSizePrev.x, rectSizePrev.y);
			Float slopeScale = 1.0f / luisa::compute::lerp(
				luisa::compute::lerp(0.05f, 1.0f, NoV),
				1.0f,
				luisa::compute::saturate(smbParallaxInPixelsMax / c.gDisocclusionParallaxDenominator));
			Float baseThresh = luisa::compute::saturate(disocclusionThreshold * slopeScale) * frustumSize;
			Float threshold = luisa::compute::max(baseThresh - 1e-4f, 0.0f);

			// Compute prev-frame view-Z of the current world point. currentWorldPos
			// is reconstructed in current-camera-centered coords, so adding
			// gCameraDelta (= cam_pos − prev_cam_pos, world axes) converts the
			// camera-relative current position to PREV-camera-relative coords:
			// dot(X_rel + C_cur − C_prev, fwd_prev) = the point's previous view-Z.
			// Handles BOTH camera translation AND rotation — the previous
			// `currentViewZ + dot(gCameraDelta, gFrustumForward) * edgeGating` form
			// was a translation-only approximation that produced false-positive
			// disocclusions under camera rotation (especially at grazing angle),
			// causing mass history resets and one-frame flashes during fast rotation.
			Float depthRef = def(0.0f);
			if (projection == 0u) {
				depthRef = luisa::compute::dot(
					prevCamRelativePos + c.gCameraDelta.xyz(), c.gPrevFrustumForward.xyz());
			} else {
				// Point-origin: the surface's previous view-Z is its distance
				// along the PREV pixel direction it reprojects to (prevUV),
				// matching the per-pixel-ray semantics of gbufDepth.
				auto pc = consts.read(1u); // pano state: element 1 (aliased members)
				Float3 wpAbs = prevCamRelativePos + c.gCamPosCur.xyz();
				Float3 prevDir = util::eval_point_origin_direction(
					prevUV * 2.0f - 1.0f,
					pc.gPrevFrustumRight.xyz(), pc.gPrevFrustumUp.xyz(), pc.gPrevFrustumForward.xyz(),
					bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
				depthRef = luisa::compute::dot(wpAbs - (c.gCamPosCur.xyz() - c.gCameraDelta.xyz()),
				                               prevDir);
			}

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
			// Equirect/cylindrical: wrap the 4x4 grid's u columns across the
			// pano seam (the clamped values above would duplicate the edge).
			if (wrapU) {
				UInt wPrev = cast<uint>(rectSizePrev.x);
				gc0 = (oxBase + wPrev - 1u) % wPrev;
				gc2 = (oxBase + 1u) % wPrev;
				gc3 = (oxBase + 2u) % wPrev;
			}

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
			Float3 prevNormalFlat = norm3(prevNormalPackedAvg * 2.0f - 1.0f);
			Float3 prevNormalRotated = transform_normal(c.gWorldPrevToWorld, prevNormalFlat);
			Bool backfaceValid = luisa::compute::dot(currentNormal, prevNormalRotated) >= 0.0f;

			$if(!backfaceValid) {
				bicubicFootprintValid = false;
				bilinearTapDepthValid = make_float4(0.0f);
			};

			// === Material-ID mask for bicubic fetch (prevents history mixing across boundaries) ===
			// NRD CompareMaterials: max(m, minMaterial) == max(m0, minMaterial) — the
			// clamp turns the gate into a no-op for IDs below minMaterial (both default
			// 4.0), so it only separates *real* material boundaries. Raw-equality
			// comparison would spuriously reject at every material edge (§8.7).
			// NRD TA:130 uses min(gSpecMinMaterial, gDiffMinMaterial) for the shared SMB footprint.
			Float smbMinMaterial = luisa::compute::min(c.gDiffMinMaterial, c.gSpecMinMaterial);
			Float curMatClamped = luisa::compute::max(currentMaterialID, smbMinMaterial);
			// 8 additional reads for non-center taps; center 2x2 reuses cnr00..11.w
			Float mv_10 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc1, gr0)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_20 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc2, gr0)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_01 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc0, gr1)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_11 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(cnr00.w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_21 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(cnr10.w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_31 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc3, gr1)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_02 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc0, gr2)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_12 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(cnr01.w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_22 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(cnr11.w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_32 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc3, gr2)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_13 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc1, gr3)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			Float mv_23 = ite(luisa::compute::abs(luisa::compute::max(luisa::compute::floor(gPrev_Normal_Roughness.read(make_uint2(gc2, gr3)).w), smbMinMaterial) - curMatClamped) < 0.5f, 1.0f, 0.0f);
			// Material-weighted count: drop bicubic at material boundaries, fall back to bilinear
			Float bicubicMatCount = mv_10 + mv_20 + mv_01 + mv_11 + mv_21 + mv_31
				+ mv_02 + mv_12 + mv_22 + mv_32 + mv_13 + mv_23;
			$if(bicubicMatCount < 11.5f) {
				bicubicFootprintValid = false;
			};

			// === Bilinear fallback position (unjittered) ===
			// NRD never jitters the reprojection position. The former per-pixel
			// hash jitter dithered disocclusion boundaries but made per-tap
			// validity, footprintQuality and historyLength noisy — the faint
			// diagonal banding inside the bright region (RC5.1, removed).
			Float2 bilinearWeights = luisa::compute::fract(prevPixelPosBase - 0.5f);

			UInt oxNext = oxBase + 1u;
			UInt oxPrevW = cast<uint>(rectSizePrev.x);
			if (wrapU) {
				oxNext = oxNext % oxPrevW; // seam wrap for the +1 column
			} else {
				oxNext = cast<uint>(luisa::compute::min(oxNext, oxPrevW - 1u));
			}
			UInt2 p00 = make_uint2(oxBase, oyBase);
			UInt2 p10 = make_uint2(oxNext, oyBase);
			UInt2 p01 = make_uint2(oxBase, oyBase + 1u);
			UInt2 p11 = make_uint2(oxNext, oyBase + 1u);

			// Material mask at bilinear taps (NRD CompareMaterials clamp, §8.7)
			Float4 nr00 = gPrev_Normal_Roughness.read(p00);
			Float4 nr10 = gPrev_Normal_Roughness.read(p10);
			Float4 nr01 = gPrev_Normal_Roughness.read(p01);
			Float4 nr11 = gPrev_Normal_Roughness.read(p11);
			Float4 prevMatIDClamped = luisa::compute::max(
				make_float4(
					luisa::compute::floor(nr00.w),
					luisa::compute::floor(nr10.w),
					luisa::compute::floor(nr01.w),
					luisa::compute::floor(nr11.w)),
				smbMinMaterial);
			Float4 gv = ite(luisa::compute::abs(prevMatIDClamped - curMatClamped) < 0.5f, 1.0f, 0.0f);

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
		Float3 prevWorldPos = currentWorldPos;
		// NRD TA:558: Vprev = -normalize(prevWorldPos - gCameraDelta_NRD) = the view
		// vector from the PREVIOUS camera. NRD's delta is C_prev - C_cur, the
		// engine's is C_cur - C_prev, so the engine form is -normalize(prevWorldPos
		// + gCameraDelta) with prevWorldPos + delta = P - C_prev. Using the current
		// V silently disables the view-rotation term of specSMBConfidence below.
		Float3 Vprev = norm3(-(prevWorldPos + c.gCameraDelta.xyz()));
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
		// NRD defaults when the VMB fetch didn't happen: prevNormal = currentNormal,
		// prevRoughness = 0 (RELAX_TA:299-300). Bilinear-sampled when it did (N5).
		Float4 prevNormalRoughnessVMBPacked = def(make_float4(0.5f, 0.5f, 1.0f, 0.0f));
		Float prevRoughnessVMB = def(0.0f);
		Bool virtualReprojectionFound = false;
		Float dominanceFactor = def(0.0f);      // modified roughness (virtualHistoryAmount, NRD TA:774)
		Float dominanceFactorRaw = def(0.0f);   // raw roughness (GetXvirtual offsets, NRD TA:738) — recomputed below, not transported
		Float2 prevUV_vmb = def(make_float2(0.0f));

		Float curvature = def(0.0f);
		// Promoted from inside the virtual-motion $if so Stage 2 hit-distance
		// confidence (which lives in $if(virtualReprojectionFound), outside the
		// virtual-motion $if) can read them. All assigned inside the gate.
		Float3 dominantDir = def(make_float3(0.0f, 0.0f, 1.0f));
		Float3 tangent = def(make_float3(0.0f));
		Float3 bitangent = def(make_float3(0.0f));
		Float silhouetteF = def(1.0f);
		// virtualWorldPos no longer exists in TA; the prepass outputs its
		// LENGTH (the only downstream consumer) via vmbIn_Info.w (vwpLength).
		// Always attempt VMB (NRD has no hard gate). virtualReprojectionFound
		// stays false for invalid VMB (UV OOB or any invalid footprint tap).
		// ===== Virtual motion for specular — SPLIT-KERNEL (e55) =====
		// The entire VMB production chain (dominance/curvature/offset/projection/
		// tap validation/history fetch) moved to _relaxSpecVmbPrepass. The TA
		// kernel's compiled code provably computed different values for the same
		// variable at different use sites (e49-e54: fetch UV collapsed to
		// 0.5 + 0.15*(px-0.5); hitDist read 4.87 m at one site and >=35 m at the
		// offset site; |dominantDir| ~2.5), and five workaround classes failed to
		// change it (hoisting, local matrix, flattening to top level, liveness
		// probes, inline re-spell). The prepass kernel is small enough to compile
		// correctly; TA consumes its outputs as plain textures — the same shape
		// as the (always-correct) SMB path above.
		// Info2 is the only unconditional read: its found flag, dominanceFactor
		// and hitDist feed math outside the found-gate (hitDist reaches
		// out_SpecFast.w and the viz encodings). Info (uv/curvature/vwpLength)
		// and the three Fetch textures are consumed only inside the gate, so
		// they are read there — failed-VMB pixels (sky tiles exit earlier;
		// diffuse/disoccluded footprints) skip 4 texture loads. The def()
		// defaults above reproduce the not-found path bit-exactly:
		// virtualHistoryAmount = dominanceFactor * 0 zeroes the VMB side of
		// every downstream blend (all lerp operands stay finite), so skipping
		// the loads cannot change a single output bit.
		Float4 vmbInfo2 = vmbIn_Info2.read(pixelPos);
		virtualReprojectionFound = vmbInfo2.x > 0.5f;
		dominanceFactor = vmbInfo2.y;
		// NRD TA:650 — hitDist is the min-biased distance over the 3x3
		// neighborhood of the raw (post-PrePass) input, INF -> 0. The prepass
		// computes the identical value (same gIn_Spec binding, same checkerboard
		// fill, same >0 neighbor filter over edge-clamped cells, same final
		// mapping; min is exact so the fold order is irrelevant) and transports
		// it here fp32-exact through RGBA32F Info2.z. Removes TA's 8 direct
		// gIn_Spec reads per pixel — upstream NRD also sources this from the
		// smem-preloaded spec tile (its TA loads gIn_Spec exactly 2x/pixel).
		Float hitDist = vmbInfo2.z;
		prevRoughnessVMB = vmbInfo2.w;
		// dominanceFactorRaw recomputed here instead of transported (the slot
		// now carries hitDist): upstream has no transported raw factor — NRD
		// evaluates GetSpecularDominantDirection inside TA (TA:738/773-775),
		// and the recompute is bit-exact by construction: a pure function of
		// currentRoughness and NoV, both top-level locals here with the same
		// expressions as the prepass (same texel, same V/NoV ops). The lambda
		// below is byte-identical to the prepass's — the same op sequence on
		// the same inputs (the silhouetteF recompute above is the precedent).
		auto evalDominanceFactor = [&](Float roughnessIn) -> Float {
			Float aDom = 0.298475f * luisa::compute::log(39.4115f - 39.0029f * roughnessIn);
			return luisa::compute::saturate(
				luisa::compute::pow(luisa::compute::saturate(1.0f - NoV), 10.8649f)
				* (1.0f - aDom) + aDom);
		};
		dominanceFactorRaw = evalDominanceFactor(currentRoughness);
		// Tangent-frame basis for Stage 2's previous-frame offset (NRD basis);
		// recomputed here — pure function of the normal.
		tangent = norm3(
			ite(luisa::compute::abs(currentNormal.x) > 0.999f,
				luisa::make_float3(0.0f, 1.0f, 0.0f),
				luisa::compute::cross(currentNormal, luisa::make_float3(1.0f, 0.0f, 0.0f))));
		bitangent = luisa::compute::cross(currentNormal, tangent);

#if NT_DEBUG_VIZ
		// Mode 27 passthrough: the prepass overwrote its FetchA output with the
		// IN-SITU offset-ingredient dump (e54 encoding) when viz 27 is active.
		$if(c.gDebugViz == 27u) {
			Float3 viz27 = vmbIn_FetchA.read(pixelPos).xyz();
			out_Spec.write(pixelPos, make_float4(viz27, 1.0f));
			out_SpecFast.write(pixelPos, make_float4(viz27, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, 0.0f, 0.0f, 0.0f));
			out_ViewZ.write(pixelPos, make_float4(currentViewZ));
			out_Normal_Roughness.write(pixelPos, make_float4(currentNormalEncoded, currentPackedNR));
			$return();
		};
#endif

				// the whole virtual-motion path now lives at top level)

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
			// VMB state fetch — plain loads into promoted locals, the same shape
			// as the SMB history fetches above (texture reads inside this gate
			// are already proven safe: the back-look loop below has always read
			// gPrev_Normal_Roughness here).
			Float4 vmbInfo = vmbIn_Info.read(pixelPos);
			prevUV_vmb = vmbInfo.xy();
			curvature = vmbInfo.z;
			Float vwpLength = vmbInfo.w;
			// silhouetteF recomputed from the transported (post-negative-zeroing)
			// curvature — bit-identical to the prepass expression: the identical
			// op sequence on identical inputs (currentWorldPos/NoV are top-level
			// locals here, curvature is the prepass's final value).
			silhouetteF = 1.0f / (1.0f + luisa::compute::length(currentWorldPos)
				* luisa::compute::saturate(1.0f - NoV)
				* luisa::compute::max(curvature, 0.0f));
			prevSpecVirtual = luisa::compute::max(vmbIn_FetchA.read(pixelPos), make_float4(0.0f));
			Float4 vmbFetchB = vmbIn_FetchB.read(pixelPos);
			prevSpecFastVirtual = vmbFetchB.xyz();
			prevSpecFastVirtual_w = vmbFetchB.w;
			// FetchC.w carries prevReflectionHitTVMB (bilinear slow hit-T at the
			// virtual UV); its out-of-gate consumer (accHitDistVMB below) is
			// blended by virtualHistoryAmount = 0 when not found, so the def(0)
			// default reproduces the not-found path exactly.
			Float4 vmbFetchC = vmbIn_FetchC.read(pixelPos);
			prevNormalRoughnessVMBPacked = vmbFetchC;
			prevReflectionHitTVMB = vmbFetchC.w;

			// === Immediate normal + roughness gates (NRD TA:783-801) ===
			// Catches fast virtual-motion landings on a wrong surface — the failure
			// mode that manifests as edge jitter on small / high-curvature objects.
			// prevNormal/prevRoughness come from the bilinear VMB fetch above (N5).
			Float3 prevNormalVMB = norm3(prevNormalRoughnessVMBPacked.xyz() * 2.0f - 1.0f);
			Float3 prevNormalVMB_rotated = transform_normal(c.gWorldPrevToWorld, prevNormalVMB);

			// uvDiff + curvature angle (NRD TA:783-789)
			Float2 uvDiff = prevUV_vmb - prevUV;
			Float uvDiffLengthInPixels = luisa::compute::length(uvDiff * rectSizePrev);
			Float tanCurvature = luisa::compute::abs(curvature * pixelSize);
			tanCurvature *= luisa::compute::max(uvDiffLengthInPixels / luisa::compute::max(NoV, 0.01f), 1.0f);
			Float curvatureAngle = luisa::compute::atan(tanCurvature);

			// Back-facing VMB gate (NRD TA:781): prev VMB normal must not face away
			// from the averaged current normal (§8.2).
			vmbWeightMultiplier = vmbWeightMultiplier
				* ite(luisa::compute::dot(prevNormalVMB_rotated, currentNormalAvg) > 0.0f, 1.0f, 0.0f);

			// Immediate encoding-aware normal weight (NRD TA:791-794)
			Float cosa_n = luisa::compute::clamp(luisa::compute::dot(currentNormal, prevNormalVMB_rotated), 0.0f, 1.0f);
			Float angle_n = luisa::compute::acos(cosa_n);
			Float normalWeight = luisa::compute::smoothstep(0.0f, 1.0f,
				1.0f - (angle_n - curvatureAngle - RELAX_NORMAL_ULP) / luisa::compute::max(lobeHalfAngle, 1e-6f));
			normalWeight = luisa::compute::smoothstep(0.05f, 0.95f, normalWeight);
			Float jitterFriendlyNormalW = luisa::compute::lerp(
				1.0f - luisa::compute::saturate(uvDiffLengthInPixels), 1.0f, normalWeight);

			// Immediate encoding-aware roughness weight (NRD TA:796-800 + Common.hlsli:513-522)
			// GetRelaxedRoughnessWeightParams(m = RAW R², fraction):
			//   a = 1 / lerp(sensitivity, 1, lerp(m*m, m, fraction)); b = m * a
			// The former port used modified roughness AND lerp(m, R, f), making the
			// gate ~4.3x looser at R=0.3 (accepted prevR range [0,0.47] vs NRD's
			// [0.245,0.347]) — wrong-roughness VMB history leaked through (RC3).
			Float rrw_m = currentRoughness * currentRoughness;
			Float rrw_t = luisa::compute::lerp(rrw_m * rrw_m, rrw_m,
				luisa::compute::saturate(c.gRoughnessFraction));
			constexpr float ROUGHNESS_SENSITIVITY = 0.01f;
			Float rwp_a = 1.0f / luisa::compute::lerp(ROUGHNESS_SENSITIVITY, 1.0f, luisa::compute::saturate(rrw_t));
			Float rwp_b = rrw_m * rwp_a;
			Float virtualRoughnessWeight = luisa::compute::smoothstep(1.0f, 0.0f,
				luisa::compute::abs(prevRoughnessVMB * prevRoughnessVMB * rwp_a - rwp_b));
			virtualRoughnessWeight = luisa::compute::lerp(
				1.0f - luisa::compute::saturate(uvDiffLengthInPixels), 1.0f, virtualRoughnessWeight);

			vmbWeightMultiplier = vmbWeightMultiplier * jitterFriendlyNormalW * virtualRoughnessWeight;
			specVMBConfidence = virtualRoughnessWeight * 0.9f + 0.1f;

			// === Hit distance confidence (NRD TA:720-831) ===
			// prevReflectionHitTVMB comes bilinear-sampled from the VMB fetch above.
			// Stage-1 base is the RAW current hit distance (NRD TA:824 uses
			// specularIllumination.a, not the min-3x3).
			Float hitDistC = luisa::compute::lerp(specHitDist, prevReflectionHitTVMB, SMC);
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
			// dominantDir here is the RAW-roughness dominant direction (NRD
			// GetXvirtual semantics) and the offset carries the dominance factor.
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
			Float3 prevVirtualWorldPos = prevWorldPos
				+ V * (dominanceFactorRaw * prevIl) * prevMagSign;

			Float2 prevUVVMBTest = def(make_float2(0.0f));
			if (projection == 0u) {
				Float4 prevVirtualClipPos = c.gWorldToClipPrev * make_float4(prevVirtualWorldPos + c.gCamPosCur.xyz(), 1.0f);
				Float prevInvW = 1.0f / luisa::compute::max(luisa::compute::abs(prevVirtualClipPos.w), 1e-6f);
				prevUVVMBTest = prevVirtualClipPos.xy() * prevInvW * luisa::make_float2(0.5f, 0.5f)
				              + luisa::make_float2(0.5f, 0.5f);
			} else {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
				prevUVVMBTest = util::project_point_origin_direction(
					normalize(prevVirtualWorldPos),
					pc.gPrevFrustumRight.xyz(), pc.gPrevFrustumUp.xyz(), pc.gPrevFrustumForward.xyz(),
					bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y) * 0.5f + 0.5f;
				if (wrapU) { prevUVVMBTest.x = luisa::compute::fract(prevUVVMBTest.x); }
			}

			// Lobe radius in pixels at the virtual distance (NRD uses raw roughness
			// here, not currentRoughnessModified).
			Float percentOfVolume = 0.6f;
			Float lobeTanHalfAngle = currentRoughness * currentRoughness * percentOfVolume
			                       / (1.0f - percentOfVolume + 1e-6f);
			lobeTanHalfAngle = luisa::compute::max(lobeTanHalfAngle, 0.5f * rectSizeInv.x);
			Float pixelWorldSizeAtVirtual = c.gUnproject * luisa::compute::max(
				vwpLength,
				luisa::compute::length(prevVirtualWorldPos));
			Float unproj1 = luisa::compute::min(hitDist, hitDistForTrackingPrev)
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
					Float3 backNormal = norm3(pnr.xyz() * 2.0f - 1.0f);
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

		// NRD TA:855-884 — separate slow vs responsive, SMB vs VMB alphas.
		// - specSMBResponsiveAlpha floors with the FAST steady alpha (NRD:
		//   max(specSMBAlpha, 1/(1+specHistoryResponsiveFrames))); pinning it to
		//   the slow alpha made the fast (clamp-reference) history carry the same
		//   lag as the slow one, contaminating the HistoryClamping box (N3).
		// - specVMBHitTAlpha = responsive base (includes
		//   virtualHistoryHitDistConfidence) floored with the SLOW steady alpha
		//   (NRD TA:880,884) — the hit-dist EMA must track the current reflection
		//   depth faster exactly when hit-dist confidence is low (§8.1).
		Float specSMBAlpha = luisa::compute::max(1.0f - specSMBConfidence,
			luisa::compute::max(c.gSpecAlphaSteady, 1.0f / historyLength));
		Float specSMBResponsiveAlpha = luisa::compute::max(specSMBAlpha,
			luisa::compute::max(c.gSpecAlphaFastSteady, 1.0f / historyLength));
		Float specVMBAlpha = luisa::compute::max(1.0f - specVMBConfidence,
			luisa::compute::max(c.gSpecAlphaSteady, 1.0f / historyLength));
		Float specVMBResponsiveAlpha = luisa::compute::max(
			1.0f - specVMBConfidence * virtualHistoryHitDistConfidence,
			luisa::compute::max(c.gSpecAlphaFastSteady, 1.0f / historyLength));
		Float specVMBHitTAlpha = luisa::compute::max(
			1.0f - specVMBConfidence * virtualHistoryHitDistConfidence,
			luisa::compute::max(c.gSpecAlphaSteady, 1.0f / historyLength));

		// NRD TA:864-868, 887-892 — checkerboard adjustment applied to each alpha
		// separately, gated on its respective reprojection flag. Previously lumped
		// into one specAlphaAdj post-lerp; the lumped form is incompatible with the
		// two-EMA-then-blend restructure below.
		$if(!cbActive & smbParallaxInPixelsMax < 0.5f) {
			Float smbCbGate = ite(reprojectionFound,        1.0f, 0.0f);
			Float vmbCbGate = ite(virtualReprojectionFound, 1.0f, 0.0f);
			specSMBAlpha           *= 1.0f - cbResolveSpeed * smbCbGate;
			specSMBResponsiveAlpha *= 1.0f - cbResolveSpeed * smbCbGate;
			specVMBAlpha           *= 1.0f - cbResolveSpeed * vmbCbGate;
			specVMBResponsiveAlpha *= 1.0f - cbResolveSpeed * vmbCbGate;
			specVMBHitTAlpha       *= 1.0f - cbResolveSpeed * vmbCbGate;
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

		// NRD TA:873, 897, 905 - EMA hit-dist per path then blend by VHA.
		// out_SpecHitDist stores the SLOW accumulated hit-dist. NRD blends toward
		// the RAW current hit distance (specularIllumination.w) with
		// max(alpha, 0.1); the VMB path uses specVMBHitTAlpha (includes
		// virtualHistoryHitDistConfidence, §8.1).
		Float accHitDistSMB = lerp(prevReflectionHitTSMB, specHitDist,
			luisa::compute::max(specSMBAlpha, 0.1f));
		Float accHitDistVMB = lerp(prevReflectionHitTVMB, specHitDist,
			luisa::compute::max(specVMBHitTAlpha, 0.1f));
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

#if NT_DEBUG_VIZ
		// Debug viz mode 13 ("HistClamp Debug Viz" slider): specular temporal
		// state. R = virtualHistoryAmount (0 = SMB history, 1 = VMB history),
		// G = curvature sign/magnitude (0.5 = flat, black = negative/concave,
		// white = positive/convex), B = min-3x3 hit distance ramp (white ≈ 5m).
		// Shows on the conductor via the spec channel (spec remod ≈ 1); poisons
		// one frame of spec history — toggle off to recover.
		// Displayed via _relaxSpecVizBlit (RAW TA output + build-epoch marker
		// painted on the final framebuffer — see PassDenoiserUtility.cpp).
		$if(c.gDebugViz == 13u) {
			// curvature re-read from Info (it is only unpacked inside the
			// found-gate on the hot path).
			Float vizCurv = luisa::compute::clamp(vmbIn_Info.read(pixelPos).z * 0.5f + 0.5f, 0.0f, 1.0f);
			Float vizHD = luisa::compute::saturate(hitDist * 0.2f);
			Float3 viz = make_float3(virtualHistoryAmount, vizCurv, vizHD);
			out_Spec.write(pixelPos, make_float4(viz, 1.0f));
			out_SpecFast.write(pixelPos, make_float4(viz, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
			$return();
		};
		// Mode 14: the SMB/VMB regime flags. R = VMB footprint valid (all 4 taps),
		// G = SMB reprojection found, B = specVMBConfidence. R+G = yellow (both).
		// A red/green boundary that tracks the highlight edge is the tracking-
		// regime discontinuity behind the rim band (report §12).
		$if(c.gDebugViz == 14u) {
			Float3 viz = make_float3(
				ite(virtualReprojectionFound, 1.0f, 0.0f),
				ite(reprojectionFound, 1.0f, 0.0f),
				specVMBConfidence);
			out_Spec.write(pixelPos, make_float4(viz, 1.0f));
			out_SpecFast.write(pixelPos, make_float4(viz, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
			$return();
		};
		// Mode 26: gPrevFrustum health from inside the kernel:
		//   R = sat(gPrevFrustumUp.y * 0.5)   (healthy 0.3485 -> 0.174; C_cur.y -> 1.0)
		//   G = sat(gPrevFrustumUp.z * 0.5)   (healthy -0.2238 -> 0.0;   C_cur.z -> 1.0)
		//   B = sat(gPrevFrustumForward.z * 0.5) (healthy -0.8414 -> 0.0)
		$if(c.gDebugViz == 26u) {
			Float3 viz = make_float3(
				luisa::compute::saturate(c.gPrevFrustumUp.y * 0.5f),
				luisa::compute::saturate(c.gPrevFrustumUp.z * 0.5f),
				luisa::compute::saturate(c.gPrevFrustumForward.z * 0.5f));
			out_Spec.write(pixelPos, make_float4(viz, 1.0f));
			out_SpecFast.write(pixelPos, make_float4(viz, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
			$return();
		};
		// Mode 25: gCamPosCur health. Expected camera (0.036, 2.948, 5.993), |C| = 6.63:
		//   R = saturate(|gCamPosCur.xyz| * 0.15)  — expect ≈ 0.995
		//   G = saturate(gCamPosCur.y * 0.3)       — expect ≈ 0.884
		//   B = saturate(|gCameraDelta.xyz| * 10)  — expect 0 at rest
		$if(c.gDebugViz == 25u) {
			Float3 viz = make_float3(
				luisa::compute::saturate(luisa::compute::length(c.gCamPosCur.xyz()) * 0.15f),
				luisa::compute::saturate(c.gCamPosCur.y * 0.3f),
				luisa::compute::saturate(luisa::compute::length(c.gCameraDelta.xyz()) * 10.0f));
			out_Spec.write(pixelPos, make_float4(viz, 1.0f));
			out_SpecFast.write(pixelPos, make_float4(viz, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
			$return();
		};
		// Mode 24: prev-matrix health. uvReprojPrev (line ~311) = proj with
		// gWorldToClipPrev of the current surface point. At rest with healthy
		// prev==cur it MUST equal pixelUv exactly:
		//   R = uvReprojPrev.y (must equal the row uv.y)
		//   G = uvReprojPrev.x
		//   B = saturate(|uvReprojPrev - pixelUv| * 10) — 0 = prev matrix healthy
		$if(c.gDebugViz == 24u) {
			Float3 viz = make_float3(
				saturate(uvReprojPrev.y),
				saturate(uvReprojPrev.x),
				saturate(luisa::compute::length(uvReprojPrev - pixelUv) * 10.0f));
			out_Spec.write(pixelPos, make_float4(viz, 1.0f));
			out_SpecFast.write(pixelPos, make_float4(viz, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
			$return();
		};
		// Mode 15: VMB failure isolation (epoch-5 encoding).
		//   R = 1 iff VMB found (all taps valid), else 0;
		//   G = CONVENTION DISCRIMINATOR: saturate(0.5 + (uvProjY - pixelUv.y)),
		//       where uvProjY is the projected uv.y of the current pixel's own
		//       world position (current matrix, current projection formula).
		//       G = mid-gray (0.5) everywhere <=> the (reconstruction-sign,
		//       projection-formula) pair is consistent with the engine matrices.
		//       A gradient deflected away from mid-gray, mirrored about the
		//       center row, means the projection formula's Y sign is wrong for
		//       this reconstruction sign.
		//   B = the center tap's plane distance in meters (0.1 = 10cm step).
		$if(c.gDebugViz == 15u) {
			// Epoch-12: SYNTHETIC-POINT ratio probe. Build two points purely from
			// the frustum CONSTANTS: camPos + fwd*10 + right*(|gFrustumRight|)*10
			// and camPos + fwd*10 + up*(|gFrustumUp|)*10. If the constants and
			// view_proj agree, they project to uv = (0.75, 0.5) exactly. The
			// measured uv is the constants-vs-matrix tan ratio:
			// ratio_x = (B_LUTinv - 0.5)*2, ratio_y = (G_LUTinv - 0.5)*2.
			Float ah = luisa::compute::length(c.gFrustumRight.xyz());
			Float hy = luisa::compute::length(c.gFrustumUp.xyz());
			Float3 testRX = c.gCamPosCur.xyz() + c.gFrustumForward.xyz() * 10.0f + c.gFrustumRight.xyz() * 10.0f;
			Float3 testUY = c.gCamPosCur.xyz() + c.gFrustumForward.xyz() * 10.0f + c.gFrustumUp.xyz() * 10.0f;
			// Epoch-13: SAME-PIXEL-RAY probe at two distances. The pixel ray
			// direction is rebuilt from the frustum basis with the pixel ndc;
			// points at 1x and 4x the view depth both lie on that ray, so both
			// must project back to the pixel exactly. G = uvProj.y at 1x (vs the
			// row: equal => ray-consistent), B = uvProj.y at 4x (equal to G =>
			// direction-consistent; different => the depth enters the direction).
			Float2 pndc = pixelUv * 2.0f - 1.0f;
			Float3 pdir = c.gFrustumForward.xyz() + c.gFrustumRight.xyz() * pndc.x + c.gFrustumUp.xyz() * pndc.y;
			Float3 w1 = c.gCamPosCur.xyz() + pdir * currentViewZ;
			Float3 w4 = c.gCamPosCur.xyz() + pdir * currentViewZ * 4.0f;
			// Epoch-15: FETCH-UV + PLANE dump. G = the actual fetch row
			// (prevUV_vmb.y raw: should EQUAL the row uv.y at rest — if it
			// differs, the virtual-point projection is broken; if it equals
			// uv.y but found=false, the TAP tests are the failing stage).
			// B = the center-tap plane distance (0.2m scale; ~0 expected).
			Float3 viz = make_float3(
				ite(virtualReprojectionFound, 1.0f, 0.0f),
				saturate(vmbIn_Info.read(pixelPos).y),  // G = the fetch row (prevUV_vmb.y): must equal the row uv.y
				0.0f);  // B: plane distance now lives in the prepass (mode 27 dump)
			out_Spec.write(pixelPos, make_float4(viz, 1.0f));
			out_SpecFast.write(pixelPos, make_float4(viz, 0.0f));
			out_SpecHitDist.write(pixelPos, make_float4(specHitDist));
			out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
			$return();
		};
#endif

		out_Spec.write(pixelPos, resultSpec);
		// NRD RELAX_TemporalAccumulation.cs.hlsl:931 writes hitDist (= min-3x3
		// from current input, INF -> 0) to specFast.a. Next frame's TA reads this
		// as hitDistForTrackingPrev for Stage 2 virtual motion.
		out_SpecFast.write(pixelPos, make_float4(resultSpecFast, hitDist));
		out_SpecHitDist.write(pixelPos, make_float4(accumulatedReflectionHitT));

		out_HistoryLength.write(pixelPos, make_float4(historyLength / 255.0f, specularHistoryConfidence, 0.0f, 0.0f));
	// ==================================================================
	// _relaxSpecVmbPrepass (e55) — the specular virtual-motion production
	// chain split out of the TA mega-kernel. The TA kernel's compiled code
	// provably computed different values for the same variable at different
	// use sites (fetch UV collapsed toward screen center, hitDist 4.87 m vs
	// >=35 m at the offset site, |dominantDir| ~2.5 — see doc section on
	// e49-e54); five in-kernel workaround classes failed. This kernel is
	// small/shallow enough to compile correctly; TA consumes the outputs
	// below as plain textures, exactly like the (always-correct) SMB path.
	// ==================================================================
	_relaxSpecVmbPrepass = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat gIn_ViewZ,
		ImageFloat gIn_Normal_Roughness,
		ImageFloat gbuf_bary_motion,
		ImageFloat gIn_Spec,
		ImageFloat gPrev_SpecHistory,
		ImageFloat gPrev_SpecFastHistory,
		ImageFloat gPrev_SpecHitDist,
		ImageFloat gPrev_Normal_Roughness,
		ImageFloat gPrev_ViewZ2,
		ImageFloat gIn_Tiles,
		ImageFloat out_Info,
		ImageFloat out_Info2,
		ImageFloat out_FetchA,
		ImageFloat out_FetchB,
		ImageFloat out_FetchC
		) noexcept {
		set_block_size(8u, 8u, 1u);
		set_name("spec_vmb_prepass_v1");

		UInt2 pixelPos = dispatch_id().xy();
		auto c = consts.read(0u);
		// Compile-time projection tag (§9 perf fix) — see the TA kernel above.
		const uint bakedProjection = _bakedProjection;
		UInt rectW = cast<uint>(c.gRectSizeX);
		UInt rectH = cast<uint>(c.gRectSizeY);
		Float2 rectSizeInv = make_float2(c.gRectSizeInvX, c.gRectSizeInvY);
		Float2 rectSizePrev = make_float2(c.gRectSizePrevX, c.gRectSizePrevY);
		Float denoisingRange = c.gDenoisingRange;

		// Sky-tile early out: TA exits at its own tile gate before reading any
		// VMB output, so these pixels' outputs are dead — write the not-found
		// defaults and skip the input reads entirely. Block-uniform (an 8x8
		// block lies entirely inside one 16x16 tile), so the sync_block() in
		// the smem preload below is never divergent.
		$if(Expr{ gIn_Tiles.read(pixelPos >> 4u).x != 0.0f }) {
			out_Info.write(pixelPos, make_float4(0.0f));
			out_Info2.write(pixelPos, make_float4(0.0f));
			$return();
		};

		//----------------------------------------------------------------------
		// Shared-memory preload for the 3x3 scan (NR smem tile + hit-dist tile),
		// same pattern as the TA kernel / the removed HitDistReconstruct pass:
		// 8x8 block + 1px halo = 10x10 cells, cooperatively loaded by 64
		// threads in 2 stages. Replaces 8 direct gIn_Normal_Roughness + 8
		// gIn_Spec reads per pixel (~6x fewer global loads). Tile cells are
		// edge-clamped exactly like the former per-tap clamp() calls, so the
		// sampled values are bit-identical. Placed before the (divergent)
		// sky-pixel early-out so the barrier is uniform, as in TA.
		//----------------------------------------------------------------------
		constexpr uint B = 8u;
		constexpr uint T = B + 2u;           // 10
		constexpr uint TILE_TOTAL = T * T;   // 100
		constexpr uint BLOCK_TOTAL = B * B;  // 64

		UInt2 gid = block_id().xy();
		UInt flat_tid = (pixelPos.x % B) + (pixelPos.y % B) * B;

		// R3 wave/smem pass: vector nr tile (see the TA kernel above — same
		// footprint/values, 1 vector op per cell instead of 4, conflict-free).
		Shared<float4> nrTile(TILE_TOTAL);
		Shared<float> hdTile(TILE_TOTAL);

		$for(stage, 2u) {
			UInt si = flat_tid + stage * BLOCK_TOTAL;
			$if(si < TILE_TOTAL) {
				UInt ty = si / T;
				UInt tx = si - ty * T;
				Int gxx = cast<Int>(gid.x * B + tx) - cast<Int>(1u);
				Int gyy = cast<Int>(gid.y * B + ty) - cast<Int>(1u);
				UInt2 sp = make_uint2(
					cast<UInt>(luisa::compute::max(gxx, 0)),
					cast<UInt>(luisa::compute::max(gyy, 0)));
				sp = min(sp, make_uint2(rectW - 1u, rectH - 1u));
				Float4 nr = gIn_Normal_Roughness.read(sp);
				nrTile.write(si, nr);
				hdTile.write(si, gIn_Spec.read(sp).w);
			};
		};
		sync_block();

		Float currentViewZ = luisa::compute::abs(gIn_ViewZ.read(pixelPos).x);
		Float specHitDist = gIn_Spec.read(pixelPos).w;

		// Checkerboard hit-dist fill (replicates the TA input stage).
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
				Float hdL = gIn_Spec.read(make_uint2(nbLx, pixelPos.y)).w;
				Float hdR = gIn_Spec.read(make_uint2(nbRx, pixelPos.y)).w;
				specHitDist = (hdL * wL + hdR * wR) * invWs;
			};
		};

		// Sky: defaults + exit (Fetch outputs stay unwritten — TA reads them
		// only inside its found-gate, and found defaults to false here).
		$if(currentViewZ > denoisingRange) {
			out_Info.write(pixelPos, make_float4(0.0f));
			out_Info2.write(pixelPos, make_float4(0.0f));
			$return();
		};

		Int cx = cast<Int>(pixelPos.x) - cast<Int>(gid.x * B) + 1;
		Int cy = cast<Int>(pixelPos.y) - cast<Int>(gid.y * B) + 1;
		UInt cIdx = cast<UInt>(cx) + cast<UInt>(cy) * T;

		Float4 currentNormalRoughness = nrTile.read(cIdx);
		Float3 currentNormal = norm3(currentNormalRoughness.xyz());
		Float currentPackedNR = currentNormalRoughness.w;
		Float currentMaterialID = luisa::compute::floor(currentPackedNR);
		Float currentRoughness = currentPackedNR - currentMaterialID;

		// 3x3: modified-roughness normal average + min hit-dist + curvature
		// neighbor normals (smem reads).
		Float3 currentNormalAvg = currentNormal;
		Float minHitDist3x3 = ite(specHitDist > 0.0f & specHitDist < denoisingRange, specHitDist, 1e6f);
		$for(dy, -1, 2) {
			$for(dx, -1, 2) {
				$if(dx == 0 & dy == 0) { $continue; };
				UInt sIdx = cast<UInt>(cx + dx) + cast<UInt>(cy + dy) * T;
				Float3 nSample = norm3(nrTile.read(sIdx).xyz());
				currentNormalAvg = currentNormalAvg + nSample;
				Float sampleHitDist = hdTile.read(sIdx);
				$if(sampleHitDist > 0.0f) {
					minHitDist3x3 = luisa::compute::min(minHitDist3x3, sampleHitDist);
				};
			};
		};
		currentNormalAvg = currentNormalAvg / 9.0f;
		// Curvature neighbors (x+1,y) / (x,y+1) straight from the tile — same
		// clamped cells the former loop-cached ite() selects produced.
		UInt i10 = cast<UInt>(cx) + 1u + cast<UInt>(cy) * T;
		UInt i01 = cast<UInt>(cx) + (cast<UInt>(cy) + 1u) * T;
		Float3 n10_cached = norm3(nrTile.read(i10).xyz());
		Float3 n01_cached = norm3(nrTile.read(i01).xyz());
		Float normalVariance = 1.0f - luisa::compute::dot(currentNormalAvg, currentNormalAvg);
		Float currentRoughnessModified = luisa::compute::min(
			currentRoughness + luisa::compute::sqrt(luisa::compute::max(normalVariance, 0.0f)) * 0.5f,
			1.0f);
		Float hitDist = ite(minHitDist3x3 >= denoisingRange, 0.0f, minHitDist3x3);

		Float2 pixelUv = (make_float2(pixelPos) + 0.5f) * rectSizeInv;
		Float2 mv_raw = gbuf_bary_motion.read(pixelPos).zw();
		Float2 prevUV = pixelUv + mv_raw * 0.5f;
		// Equirect/cylindrical u wraps (see TA) — fold into [0,1).
		if (bakedProjection == 1u | bakedProjection == 2u) {
			prevUV.x = luisa::compute::fract(prevUV.x);
		}
		// UV CONVENTION: pixel uv is Y-UP; clipXY = uv*2-1 is NDC directly (see TA).
		// Lateral terms carry *viewZ (true point = z*(fwd + right*cx + up*cy),
		// the inverse of Camera::generate_ray) — the z-less form projected to
		// uv' = 0.5 + (uv-0.5)/z, the fetch-UV collapse behind the rim band.
		Float3 currentWorldPos = def(make_float3(0.0f));
		if (bakedProjection == 0u) {
			currentWorldPos = currentViewZ * (c.gFrustumForward.xyz()
				+ c.gFrustumRight.xyz() * (pixelUv.x * 2.0f - 1.0f)
				+ c.gFrustumUp.xyz() * (pixelUv.y * 2.0f - 1.0f));
		} else {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
			currentWorldPos = currentViewZ * util::eval_point_origin_direction(
				pixelUv * 2.0f - 1.0f,
				pc.gFrustumRight.xyz(), pc.gFrustumUp.xyz(), pc.gFrustumForward.xyz(),
				bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
		}
		Float3 V = (-currentWorldPos) * luisa::compute::rsqrt(
			luisa::compute::dot(currentWorldPos, currentWorldPos));
		Float NoV = luisa::compute::abs(luisa::compute::dot(currentNormal, V));

		// Hoisted struct reads + matrix local (CDSL workarounds #3/#4).
		Float3 prevFwdV = c.gPrevFrustumForward.xyz();
		Float3 prevRightV = c.gPrevFrustumRight.xyz();
		Float3 prevUpV = c.gPrevFrustumUp.xyz();
		Float3 camPosPrevV = c.gCamPosCur.xyz() - c.gCameraDelta.xyz();
		Float3 camPosCurV = c.gCamPosCur.xyz();
		auto prevClipMat = c.gWorldToClipPrev;

		Float3 prevCamRelativePos = currentWorldPos;
		Float2 uvReprojPrev = def(make_float2(0.0f));
		if (bakedProjection == 0u) {
			Float4 pcPrev = c.gWorldToClipPrev * make_float4(prevCamRelativePos + c.gCamPosCur.xyz(), 1.0f);
			uvReprojPrev = pcPrev.xy() * (1.0f / pcPrev.w) * luisa::make_float2(0.5f, 0.5f)
				+ luisa::make_float2(0.5f, 0.5f);
		} else {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
			Float3 wpAbs = prevCamRelativePos + c.gCamPosCur.xyz();
			uvReprojPrev = util::project_point_origin_direction(
				normalize(wpAbs - (c.gCamPosCur.xyz() - c.gCameraDelta.xyz())),
				pc.gPrevFrustumRight.xyz(), pc.gPrevFrustumUp.xyz(), pc.gPrevFrustumForward.xyz(),
				bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y) * 0.5f + 0.5f;
			if (bakedProjection == 1u | bakedProjection == 2u) {
				uvReprojPrev.x = luisa::compute::fract(uvReprojPrev.x);
			}
		}
		Float smbParallaxInPixels1 = luisa::compute::length((uvReprojPrev - prevUV) * rectSizePrev);
		Float smbParallaxInPixels2 = luisa::compute::length((prevUV - pixelUv) * rectSizePrev);
		Float smbParallaxInPixelsMax = luisa::compute::max(smbParallaxInPixels1, smbParallaxInPixels2);
		Float smbParallaxInPixelsMin = luisa::compute::min(smbParallaxInPixels1, smbParallaxInPixels2);
		Float pixelSize = c.gUnproject * currentViewZ;
		Float frustumSize = pixelSize * luisa::compute::min(rectSizePrev.x, rectSizePrev.y);

		// VMB state (mirrors the TA promoted locals)
		Float4 prevSpecVirtual = def(make_float4(0.0f));
		Float3 prevSpecFastVirtual = def(make_float3(0.0f));
		Float prevSpecFastVirtual_w = def(0.0f);
		Float prevReflectionHitTVMB = def(0.0f);
		Float4 prevNormalRoughnessVMBPacked = def(make_float4(0.5f, 0.5f, 1.0f, 0.0f));
		Float prevRoughnessVMB = def(0.0f);
		Bool virtualReprojectionFound = false;
		Float dominanceFactor = def(0.0f);
		Float dominanceFactorRaw = def(0.0f);
		Float2 prevUV_vmb = def(make_float2(0.0f));
		Float curvature = def(0.0f);
		Float3 dominantDir = def(make_float3(0.0f, 0.0f, 1.0f));
		Float3 tangent = def(make_float3(0.0f));
		Float3 bitangent = def(make_float3(0.0f));
		Float3 virtualWorldPos = def(make_float3(0.0f));

		// ===== Virtual motion for specular — computed at TOP LEVEL =====
		// CDSL-backend hazard (measured, mode-27 probe e49): every value produced
		// or consumed inside the former deeply-nested $if(minHitDist3x3 ->
		// $if(virtualValid) -> $if(allVMBTapsValid) block miscomputed at those
		// program points while the identical math at top level stayed correct:
		// the fetch UV collapsed toward screen center (prevUV_vmb ≈ 0.5 +
		// 0.16·(pixel−0.5)), the in-block curvature read ~50x its true value
		// (offset inflated 4-8x — the same 4x measured on e37), the tap depth
		// reads returned the pixel's own texel, and the tap reconstruction sheared
		// ∝ clipY along −C_cur (the rim band). Hoisting individual reads did NOT
		// survive compilation (rematerialized in-block). The SMB fetch above has
		// always been healthy at this depth — the VMB path now uses the same
		// structure: branchless math at top level, ONE shallow $if for the fetch.
		// (The former gate was vacuous anyway: minHitDist3x3 = ite(>0, s, 1e6) is
		// never negative; it is folded into vmbFetch below.)
		// Two dominant-direction evaluations (NRD distinction):
			// - RAW roughness drives GetXvirtual (the virtual motion position,
			//   RELAX_TA:731/738/836) — feeding modified roughness here lets
			//   normal-variance noise on curved edges perturb the VMB UV itself.
			// - MODIFIED roughness drives virtualHistoryAmount only (RELAX_TA:774).
			auto evalDominanceFactor = [&](Float roughnessIn) -> Float {
				Float aDom = 0.298475f * luisa::compute::log(39.4115f - 39.0029f * roughnessIn);
				return luisa::compute::saturate(
					luisa::compute::pow(luisa::compute::saturate(1.0f - NoV), 10.8649f)
					* (1.0f - aDom) + aDom);
			};
				dominanceFactorRaw = evalDominanceFactor(currentRoughness);
				Float3 reflDir = luisa::compute::reflect(-V, currentNormal);
				// NRD _NRD_GetSpecularDominantDirection (NRD.hlsli:447-453):
				// D = normalize(lerp(N, R, dominantFactor)) — factor 1 = mirror
				// direction, 0 = normal. The former lerp(reflDir, N, f) inverted the
				// blend: at R=0.3 (f≈0.99) the "dominant" direction pointed along N,
				// corrupting Oz and with it the curvature magnification on the fold.
				dominantDir = norm3(lerp(currentNormal, reflDir, dominanceFactorRaw));
				dominanceFactor = evalDominanceFactor(currentRoughnessModified);

			// --- Curvature estimation along motion direction (NRD TA:652-736) ---
			{
				// Unit motion direction in pixels (direction feeds only the w mix)
				Float2 deltaUvPx = mv_raw * make_float2(rectSizePrev) * 0.5f;
				Float2 deltaDir = deltaUvPx / luisa::compute::max(luisa::compute::length(deltaUvPx), 1e-6f);
				Float2 absDelta = luisa::compute::abs(deltaDir) + 1.0f / 256.0f;
				Float2 w = absDelta / (absDelta.x + absDelta.y);

				// Neighbor positions via LINE-PLANE INTERSECTION with the center
				// tangent plane (NRD TA:663-681), not the raw neighbor view-Z
				// reconstruction: a constant ~1px baseline that ignores depth
				// discontinuities behind the pixel (§8.4).
				Float3 n10 = n10_cached;
				Float3 n01 = n01_cached;

				Float2 uv10 = (make_float2(pixelPos) + make_float2(1.0f, 0.0f) + 0.5f) * rectSizeInv;
				Float2 clipXY10 = uv10 * 2.0f - 1.0f;
				Float3 rd10 = def(c.gFrustumForward.xyz() + c.gFrustumRight.xyz() * clipXY10.x
					+ c.gFrustumUp.xyz() * clipXY10.y);
				if (bakedProjection != 0u) {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
					rd10 = util::eval_point_origin_direction(clipXY10,
						pc.gFrustumRight.xyz(), pc.gFrustumUp.xyz(), pc.gFrustumForward.xyz(),
						bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
				}
				Float d10 = luisa::compute::dot(currentNormal, rd10);
				d10 = ite(d10 >= 0.0f, luisa::compute::max(d10, 1e-6f), luisa::compute::min(d10, -1e-6f));
				Float3 x10 = rd10 * (luisa::compute::dot(currentWorldPos, currentNormal) / d10);

				Float2 uv01 = (make_float2(pixelPos) + make_float2(0.0f, 1.0f) + 0.5f) * rectSizeInv;
				Float2 clipXY01 = uv01 * 2.0f - 1.0f;
				Float3 rd01 = def(c.gFrustumForward.xyz() + c.gFrustumRight.xyz() * clipXY01.x
					+ c.gFrustumUp.xyz() * clipXY01.y);
				if (bakedProjection != 0u) {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
					rd01 = util::eval_point_origin_direction(clipXY01,
						pc.gFrustumRight.xyz(), pc.gFrustumUp.xyz(), pc.gFrustumForward.xyz(),
						bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
				}
				Float d01 = luisa::compute::dot(currentNormal, rd01);
				d01 = ite(d01 >= 0.0f, luisa::compute::max(d01, 1e-6f), luisa::compute::min(d01, -1e-6f));
				Float3 x01 = rd01 * (luisa::compute::dot(currentWorldPos, currentNormal) / d01);

				Float3 xNeighbor = x10 * w.x + x01 * w.y;
				Float3 nNeighbor = norm3(n10 * w.x + n01 * w.y);

				// High-parallax same-surface replacement (NRD TA:691-721): with
				// real pixel motion, look ahead along the motion direction at a
				// snapped pixel center and replace the curvature neighborhood if
				// it lies on the same tangent plane — flattens the curvature
				// estimate on high motion (NRD's own rotation-robustness measure).
				{
					// Sequence::Bayer4x4 equivalent (4x4 ordered dither, temporally shifted)
					Float bayerShift = cast<Float>(c.gFrameIndex & 3u);
					Float2 bp = floor(make_float2(pixelPos) + make_float2(bayerShift, bayerShift));
					Float2 bpHalf = bp * 0.5f;
					Float bayerA = luisa::compute::fract(bpHalf.x * 0.5f + bpHalf.y * bpHalf.y * 0.75f);
					Float bayerB = luisa::compute::fract(bp.x * 0.5f + bp.y * bp.y * 0.75f);
					Float dither = bayerA * 0.25f + bayerB;

				Float edgeFix = 1.0f - luisa::compute::pow(NoV, 5.0f);
				Float deltaUvLenFixed = smbParallaxInPixelsMin; // min: not needed for camera-attached
				deltaUvLenFixed *= 1.0f + edgeFix * (1.0f + c.gFramerateScale * dither);
				Float2 motionUvHigh = pixelUv + deltaUvLenFixed * deltaDir * rectSizeInv;
				Float2 rectSizeF = make_float2(cast<Float>(rectW), cast<Float>(rectH));
				motionUvHigh = (floor(motionUvHigh * rectSizeF) + 0.5f) * rectSizeInv; // snap to pixel center

					Bool uvHighInScreen = motionUvHigh.x >= 0.0f & motionUvHigh.x <= 1.0f
						& motionUvHigh.y >= 0.0f & motionUvHigh.y <= 1.0f;
					$if(deltaUvLenFixed > 1.0f & uvHighInScreen) {
						UInt2 posHigh = make_uint2(
							cast<UInt>(clamp(cast<int>(motionUvHigh.x * cast<Float>(rectW)), 0, cast<int>(rectW) - 1)),
							cast<UInt>(clamp(cast<int>(motionUvHigh.y * cast<Float>(rectH)), 0, cast<int>(rectH) - 1)));
					Float zHigh = luisa::compute::abs(gIn_ViewZ.read(posHigh).x);
					Float3 nHigh = norm3(gIn_Normal_Roughness.read(posHigh).xyz());
					Float2 clipHigh = (make_float2(posHigh) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
					Float3 reconHigh = def(c.gFrustumForward.xyz()
						+ c.gFrustumRight.xyz() * clipHigh.x
						+ c.gFrustumUp.xyz() * clipHigh.y);
					if (bakedProjection != 0u) {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
						reconHigh = util::eval_point_origin_direction(clipHigh,
							pc.gFrustumRight.xyz(), pc.gFrustumUp.xyz(), pc.gFrustumForward.xyz(),
							bakedProjection, pc.gCameraDelta.x, pc.gCameraDelta.y);
					}
					Float3 xHigh = zHigh * reconHigh;

						// Same-surface test (NRD GetGeometryWeightParams, threshold 0.04)
						constexpr float CURVATURE_HIGH_PARALLAX_THRESH = 0.04f;
						Float gNorm = CURVATURE_HIGH_PARALLAX_THRESH * frustumSize;
						Float ga = 1.0f / gNorm;
						Float gb = luisa::compute::dot(currentNormal, currentWorldPos) * ga;
						Float NoX = luisa::compute::dot(currentNormal, xHigh);
						Float wHigh = luisa::compute::smoothstep(1.0f, 0.0f,
							luisa::compute::abs(NoX * ga - gb));
						Bool sameSurface = (wHigh > 0.5f) & (zHigh < denoisingRange);
						xNeighbor = ite(sameSurface, xHigh, xNeighbor);
						nNeighbor = ite(sameSurface, nHigh, nNeighbor);
					};
				}

				Float3 edge = xNeighbor - currentWorldPos;
				Float edgeLenSq = luisa::compute::dot(edge, edge);
				$if(edgeLenSq > 1e-10f) {
					curvature = luisa::compute::dot(nNeighbor - currentNormal, edge) / edgeLenSq;
				};
			}

			// --- Thin-lens virtual motion (NRD GetXvirtual, Common.hlsli:407-445) ---
			tangent = norm3(
				ite(luisa::compute::abs(currentNormal.x) > 0.999f,
					luisa::make_float3(0.0f, 1.0f, 0.0f),
					luisa::compute::cross(currentNormal, luisa::make_float3(1.0f, 0.0f, 0.0f))));
			bitangent = luisa::compute::cross(currentNormal, tangent);

			// O = basis(rotation of D_raw * hitDist); O.z negated (NRD sign convention).
			// hitDist is the MIN-biased distance (NRD TA:650) — a mean over a
			// hit-distance step overshoots the virtual ray along highlight edges.
			Float3 reflectionRay = dominantDir * hitDist;
			Float Ox = luisa::compute::dot(tangent, reflectionRay);
			Float Oy = luisa::compute::dot(bitangent, reflectionRay);
			Float Oz = -luisa::compute::dot(currentNormal, reflectionRay);

			// Full NRD offset: V * D.w * sign(mag) where D.w = dominance * |I|
			// (elongated). The former code scaled the offset by the closeness
			// factor and dropped the dominance term — under-moving every virtual
			// position (§8.3 + N1).
			auto virtualOffset = [&](Float curvIn) -> Float3 {
				Float magL = 1.0f / (2.0f * curvIn * Oz - 1.0f);
				Float silFL = 1.0f / (1.0f + luisa::compute::length(currentWorldPos)
					* luisa::compute::saturate(1.0f - NoV)
					* luisa::compute::max(curvIn, 0.0f));
				magL *= silFL;
				Float3 I_l = tangent * (Ox * magL) + bitangent * (Oy * magL) + currentNormal * (Oz * magL);
				Float signL = ite(magL >= 0.0f, 1.0f, -1.0f);
				return V * (dominanceFactorRaw * luisa::compute::length(I_l)) * signL;
			};

			// NRD TA:728-735 — negative curvature (focusing) is zeroed unless the
			// virtual UV stays within ACCEL * smbParallaxInPixelsMax + 1px of the
			// surface-motion position. Applied to negative curvature ONLY:
			// legitimate positive-curvature tracking (a highlight sweeping fast
			// across a convex sheet) is not killed.
			$if(curvature < 0.0f) {
				Float4 uv1Clip = prevClipMat * make_float4(prevCamRelativePos + camPosCurV + virtualOffset(curvature), 1.0f);
				Float2 uv1 = uv1Clip.xy() * (1.0f / uv1Clip.w) * luisa::make_float2(0.5f, 0.5f)
					+ luisa::make_float2(0.5f, 0.5f);
				Float4 uv2Clip = prevClipMat * make_float4(prevCamRelativePos + camPosCurV, 1.0f);
				Float2 uv2 = uv2Clip.xy() * (1.0f / uv2Clip.w) * luisa::make_float2(0.5f, 0.5f)
					+ luisa::make_float2(0.5f, 0.5f);
				Float accel = luisa::compute::length((uv1 - uv2) * rectSizePrev);
				Bool withinCap = accel < c.gMaxAllowedVirtualMotionAcceleration * smbParallaxInPixelsMax
					+ rectSizeInv.x;
				$if(!withinCap) { curvature = 0.0f; };
			};

			// Virtual world position: surface-motion base + full offset along V.
			// (NRD's closenessToSurface blends the base between Xprev and X; for a
			// static world these coincide, so the blend degenerates — §8.3.
			// silhouetteF — the prepass's own copy — folded into magL2/silFL2
			// below; TA recomputes it from the transported curvature.)
			// CDSL-workaround #5 (e53): the OFFSET VALUE is corrupted at this use
			// site while the identical lambda/probe reads are healthy elsewhere
			// (e52: the projected virtual point exploded to |P| ≥ 17 m with a
			// fold-flipping direction; e51: curvature/anchor/basis all healthy at
			// a probe site). The corruption follows the USE SITE, not the nesting
			// depth — so the offset is re-spelled inline here (fresh codegen at
			// this program point). Keep byte-identical math to virtualOffset().
			{
				Float3 reflRay2 = dominantDir * hitDist;
				Float Ox2 = luisa::compute::dot(tangent, reflRay2);
				Float Oy2 = luisa::compute::dot(bitangent, reflRay2);
				Float Oz2 = -luisa::compute::dot(currentNormal, reflRay2);
				Float magL2 = 1.0f / (2.0f * curvature * Oz2 - 1.0f);
				Float silFL2 = 1.0f / (1.0f + luisa::compute::length(currentWorldPos)
					* luisa::compute::saturate(1.0f - NoV)
					* luisa::compute::max(curvature, 0.0f));
				magL2 *= silFL2;
				Float3 Il2 = tangent * (Ox2 * magL2) + bitangent * (Oy2 * magL2)
					+ currentNormal * (Oz2 * magL2);
				Float signL2 = ite(magL2 >= 0.0f, 1.0f, -1.0f);
				// CDSL-workaround #6 (e57): the vector*scalar form above bound the
				// WRONG 3-wide value at this site — the compiled offset behaved as
				// dominantDir·|Il| (the reflection direction, off-ray) instead of
				// V·|Il|·sign (on-ray). Componentwise products bind different ops;
				// sign folded into the scalar.
				Float offScale2 = signL2 * dominanceFactorRaw * luisa::compute::length(Il2);
				Float3 off2 = make_float3(
					V.x * offScale2,
					V.y * offScale2,
					V.z * offScale2);
				// CDSL-workaround #7 (e59): the sum through the prevCamRelativePos
				// ALIAS miscomputed at this site (e56: the projected input was a
				// saturated dominantDir-like diagonal while e58 proved off2 itself
				// is healthy on-ray). Build the sum componentwise from the original
				// currentWorldPos — mathematically identical (the alias is a copy).
				virtualWorldPos = make_float3(
					currentWorldPos.x + off2.x,
					currentWorldPos.y + off2.y,
					currentWorldPos.z + off2.z);

				// CDSL-workaround #8 (e60→e62): the projection FOLDED INTO THIS
				// BLOCK as ONE EXPRESSION with no named intermediate for the
				// point. Rationale: e59's mode-15 measured the fetch UV collapsed
				// to ≈ projecting THE CAMERA POSITION (uv ≈ (0.50, 0.52) = uv(C))
				// even though off2 was proven on-ray — i.e. the assigned
				// `virtualWorldPos` read back as its def() (0,0,0) at the
				// consumption site. A single expression tree gives the
				// variable no lifetime across statements to break.
				Float4 virtualClipPos = c.gWorldToClipPrev * make_float4(
					currentWorldPos.x + off2.x + c.gCamPosCur.x,
					currentWorldPos.y + off2.y + c.gCamPosCur.y,
					currentWorldPos.z + off2.z + c.gCamPosCur.z,
					1.0f);
				Float invW = 1.0f / luisa::compute::max(
					luisa::compute::abs(virtualClipPos.w), 1e-6f);
				prevUV_vmb = virtualClipPos.xy() * invW * luisa::make_float2(0.5f, 0.5f)
					+ luisa::make_float2(0.5f, 0.5f);
#if NT_DEBUG_VIZ
				// e61 MATRIX SANITY — project 3 synthetic points through
				// c.gWorldToClipPrev IN THIS BLOCK (pixel-independent constants):
				//   P0 = C + fwd*10              → uv MUST be (0.5, 0.5)
				//   P1 = P0 + rightVec*10        → uv.x MUST be 1.0
				//   P2 = P0 + upVec*10           → uv.y MUST be 1.0
				//   R = uv(P0).x (0.5)   G = uv(P1).x (1.0)   B = uv(P2).y (1.0)
				// Anything else = the device-side matrix content/offset is wrong.
				$if(c.gDebugViz == 27u) {
					Float3 P0 = c.gCamPosCur.xyz() + c.gFrustumForward.xyz() * 10.0f;
					Float3 P1 = P0 + c.gFrustumRight.xyz() * 10.0f;
					Float3 P2 = P0 + c.gFrustumUp.xyz() * 10.0f;
					Float4 c0 = c.gWorldToClipPrev * make_float4(P0, 1.0f);
					Float4 c1 = c.gWorldToClipPrev * make_float4(P1, 1.0f);
					Float4 c2 = c.gWorldToClipPrev * make_float4(P2, 1.0f);
					Float2 uv0 = c0.xy() * (1.0f / c0.w) * 0.5f + 0.5f;
					Float2 uv1 = c1.xy() * (1.0f / c1.w) * 0.5f + 0.5f;
					Float2 uv2 = c2.xy() * (1.0f / c2.w) * 0.5f + 0.5f;
					Float3 viz27 = make_float3(uv0.x, uv1.x, uv2.y);
					out_FetchA.write(pixelPos, make_float4(viz27, 1.0f));
					out_Info.write(pixelPos, make_float4(0.0f));
					out_Info2.write(pixelPos, make_float4(0.0f));
					out_FetchB.write(pixelPos, make_float4(0.0f));
					out_FetchC.write(pixelPos, make_float4(0.0f));
					$return();
				};
#endif
				};

			// Non-perspective: VMB virtual points project through the pinhole
			// matrix path below; rather than re-derive it analytically inside
			// this CDSL-sensitive block (see the workaround log), specular
			// falls back to surface-motion reprojection (NRD's own fallback).
			Bool virtualValid = prevUV_vmb.x >= 0.0f & prevUV_vmb.x <= 1.0f
				& prevUV_vmb.y >= 0.0f & prevUV_vmb.y <= 1.0f;
			if (bakedProjection != 0u) {
			// Pano state: element 1 of the constants buffer (aliased members)
			auto pc = consts.read(1u);
				virtualValid = false; // VMB gated off under non-perspective
			}

			// Tap stage: computed UNCONDITIONALLY at top level (all reads use
			// clamped coordinates; validity flags merge into vmbFetch below) —
			// nothing value-bearing may be produced inside a nested block here.
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

				Float2 bWeights = luisa::compute::fract(prevPixelPosVMB - 0.5f);
				Float4 vcw = make_float4(
					(1.0f - bWeights.x) * (1.0f - bWeights.y),
					bWeights.x * (1.0f - bWeights.y),
					(1.0f - bWeights.x) * bWeights.y,
					bWeights.x * bWeights.y);

				// NRD loadVirtualMotionBasedPrevData (:271-292): per-tap validity is a
				// PLANE-DISTANCE test of the reconstructed tap position against
				// disocclusionThreshold * viewZ plus the CompareMaterials gate. The tap
				// position is reconstructed with the PREV frustum basis anchored at
				// C_prev = (gCamPosCur − gCameraDelta), world axes both sides — the
				// plane distance is then exact for any rigid camera motion. The up
				// term carries the engine-wide "−up·clipY" convention (Y-down uv ->
				// clipY = −NDC_y); the former "+" mirrored the taps vertically and
				// made co-planar taps fail off the center row (the rim band).
				Float vmbThresh = luisa::compute::max(c.gDisocclusionThreshold * currentViewZ - 1e-4f, 0.0f);
				Float curMatClampedVMB = luisa::compute::max(currentMaterialID, c.gSpecMinMaterial);

				Float vz00 = luisa::compute::abs(gPrev_ViewZ2.read(vp00).x);
				Float vz10 = luisa::compute::abs(gPrev_ViewZ2.read(vp10).x);
				Float vz01 = luisa::compute::abs(gPrev_ViewZ2.read(vp01).x);
				Float vz11 = luisa::compute::abs(gPrev_ViewZ2.read(vp11).x);

				Float4 vnr00 = gPrev_Normal_Roughness.read(vp00);
				Float4 vnr10 = gPrev_Normal_Roughness.read(vp10);
				Float4 vnr01 = gPrev_Normal_Roughness.read(vp01);
				Float4 vnr11 = gPrev_Normal_Roughness.read(vp11);


				auto tapPlaneDistValid = [&](Float vzIn, Float4 vnrIn, UInt2 vpIn) {
					Float2 clipTap = (make_float2(vpIn) + 0.5f) * rectSizeInv * 2.0f - 1.0f;
					// PREV frustum basis + prev depth = the tap point relative to the
					// PREVIOUS camera; adding (gCamPosCur − gCameraDelta) = C_prev
					// gives the tap's WORLD point. prevCamRelativePos (+gCamPosCur)
					// is the current surface point in world — the plane distance is
					// then exact for any rigid camera motion. Lateral terms carry
					// the tap's OWN viewZ (same true-point form as currentWorldPos).
					Float3 posTap = vzIn * (prevFwdV
						+ prevRightV * clipTap.x
						+ prevUpV * clipTap.y)
						+ camPosPrevV;
					Float planeDist = luisa::compute::abs(
						luisa::compute::dot(prevCamRelativePos + camPosCurV - posTap, currentNormal));
					return vzIn < denoisingRange
						& planeDist < vmbThresh
						& luisa::compute::abs(luisa::compute::max(luisa::compute::floor(vnrIn.w), c.gSpecMinMaterial) - curMatClampedVMB) < 0.5f;
				};

				Bool vValid00 = tapPlaneDistValid(vz00, vnr00, vp00);
				Bool vValid10 = tapPlaneDistValid(vz10, vnr10, vp10);
				Bool vValid01 = tapPlaneDistValid(vz01, vnr01, vp01);
				Bool vValid11 = tapPlaneDistValid(vz11, vnr11, vp11);

				// NRD TA:351 — the VMB footprint is valid ONLY if ALL 4 taps are
				// valid. Partial footprints fetch history across hit-distance /
				// luminance discontinuities (the old highlight position) with
				// near-full local weight — the displaced-copy chain of the zigzag
				// (RC1). The old "black ring at the accept/reject boundary" is
				// handled by NRD's own smooth confidence factors instead.
				Bool allVMBTapsValid = vValid00 & vValid10 & vValid01 & vValid11;
				Bool vmbFetch = (minHitDist3x3 >= 0.0f) & virtualValid & allVMBTapsValid;
				virtualReprojectionFound = vmbFetch;


				$if(vmbFetch) {
					// Plain bilinear fetch — all taps valid, weights sum to 1.
					Float4 vps00 = gPrev_SpecHistory.read(vp00);
					Float4 vps10 = gPrev_SpecHistory.read(vp10);
					Float4 vps01 = gPrev_SpecHistory.read(vp01);
					Float4 vps11 = gPrev_SpecHistory.read(vp11);
					prevSpecVirtual = vps00 * vcw.x + vps10 * vcw.y + vps01 * vcw.z + vps11 * vcw.w;
					prevSpecVirtual = luisa::compute::max(prevSpecVirtual, make_float4(0.0f));

					Float4 vpsf00 = gPrev_SpecFastHistory.read(vp00);
					Float4 vpsf10 = gPrev_SpecFastHistory.read(vp10);
					Float4 vpsf01 = gPrev_SpecFastHistory.read(vp01);
					Float4 vpsf11 = gPrev_SpecFastHistory.read(vp11);
					Float4 vpsfSum = vpsf00 * vcw.x + vpsf10 * vcw.y + vpsf01 * vcw.z + vpsf11 * vcw.w;
					prevSpecFastVirtual = luisa::compute::max(vpsfSum.xyz(), make_float3(0.0f));
					prevSpecFastVirtual_w = vpsfSum.w;

					// NRD TA:340-346 — normal/roughness for the VMB gates and the
					// slow hit-dist history are sampled BILINEARLY at the virtual UV
					// (the former nearest-tap read added per-pixel gate noise on the
					// 1px VMB boundaries, N5). Roughness is blended per-tap in
					// unpacked form — the packed matID.roughness float must not be
					// filtered across integer boundaries.
					prevNormalRoughnessVMBPacked = vnr00 * vcw.x + vnr10 * vcw.y + vnr01 * vcw.z + vnr11 * vcw.w;
					prevRoughnessVMB = luisa::compute::fract(vnr00.w) * vcw.x
						+ luisa::compute::fract(vnr10.w) * vcw.y
						+ luisa::compute::fract(vnr01.w) * vcw.z
						+ luisa::compute::fract(vnr11.w) * vcw.w;
					prevReflectionHitTVMB = luisa::compute::max(
						gPrev_SpecHitDist.read(vp00).x * vcw.x + gPrev_SpecHitDist.read(vp10).x * vcw.y
						+ gPrev_SpecHitDist.read(vp01).x * vcw.z + gPrev_SpecHitDist.read(vp11).x * vcw.w,
						0.001f);

					// Fetch outputs are written ONLY for found pixels — TA reads
					// them only inside its found-gate (its def() defaults reproduce
					// the not-found path bit-exactly), so skipped writes are never
					// observed. Saves 48B/px of write traffic on the (often
					// majority) failed-VMB pixels.
					out_FetchA.write(pixelPos, prevSpecVirtual);
					out_FetchB.write(pixelPos, make_float4(prevSpecFastVirtual, prevSpecFastVirtual_w));
					// FetchC.w = prevReflectionHitTVMB (the former Info3.x slot).
					out_FetchC.write(pixelPos, make_float4(
						prevNormalRoughnessVMBPacked.xyz(), prevReflectionHitTVMB));
				};
				// (former $if(virtualValid) / $if(minHitDist3x3) closers removed —

		// --- Outputs (TA consumes these as plain textures) ---
		// Info/Info2 are always written; .w = |virtualWorldPos| (vwpLength),
		// consumed by TA's Stage-2 lobe-radius test. silhouetteF is no longer
		// transported — TA recomputes it bit-exactly from curvature + its own
		// currentWorldPos/NoV. Info2.z likewise transports the prepass's
		// hitDist (the min-3x3 value TA's own 3x3 scan used to recompute via
		// 8 direct gIn_Spec reads — identical by construction, see TA);
		// dominanceFactorRaw is NOT transported — TA recomputes it from its
		// own currentRoughness/NoV (upstream-faithful: NRD evaluates the
		// dominant factor inside TA), and it stays a prepass local feeding
		// dominantDir and the virtual-offset lambdas below.
		out_Info.write(pixelPos, make_float4(prevUV_vmb.x, prevUV_vmb.y, curvature,
			luisa::compute::length(virtualWorldPos)));
		out_Info2.write(pixelPos, make_float4(
			ite(virtualReprojectionFound, 1.0f, 0.0f), dominanceFactor, hitDist, prevRoughnessVMB));
	});

	});

}

}
