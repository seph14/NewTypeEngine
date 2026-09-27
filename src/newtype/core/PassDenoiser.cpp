#include "newtype/render/PassDenoiser.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Rng.h"
#include "newtype/render/Shading.h"
#include "cinder/Log.h"
#include "newtype/util/UiHelper.h"
#include "cinder/CinderImGui.h"
#include "newtype/util/Profiler.h"
#include "cinder/Utilities.h"
#include "newtype/util/TypeConv.h"
#include "newtype/util/AccumulationTime.h"
#include "cinder/app/App.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

//==========================================================================
// RelaxDenoiser::compile
//==========================================================================

void RelaxDenoiser::compile(Device& device, const SurfaceResolverPoly& resolver) {
	compilePrefilterAndClassifyTiles(device, resolver);
	compilePrepass(device);
	compileHitAndTemporal(device);
	compileHistory(device);
	compileAtrous(device);
	compileAtrousSmem(device);
	compileUtility(device);	_relaxMotionCopy = device.compile<2>(
		[](ImageFloat raster_depth, ImageUInt gbuf_vis, ImageFloat gbuf_bary_motion) noexcept {
		set_name("relax_motion_copy");
		UInt2 coord = dispatch_id().xy();

		UInt4 vis = gbuf_vis.read(coord);
		Bool is_point = ((vis.y >> 30u) & 1u) > 0u;
		$if(!is_point) { $return(); };

		Float packed = raster_depth.read(coord).y;
		Float motion_x = 0.0f;
		Float motion_y = 0.0f;
		$if(packed != 0.0f) {
			UInt bits = as<uint>(packed);
			motion_x = (cast<float>(bits >> 16u) / 65535.0f) * 2.0f - 1.0f;
			motion_y = (cast<float>(bits & 0xFFFFu) / 65535.0f) * 2.0f - 1.0f;
		};

		Float4 bary = gbuf_bary_motion.read(coord);
		bary.z = motion_x;
		bary.w = motion_y;
		gbuf_bary_motion.write(coord, bary);
	});
}

void RelaxDenoiser::recompileCallables(Device& device, const SurfaceResolverPoly& resolver) {
	compilePrefilterAndClassifyTiles(device, resolver);
}

void RelaxDenoiser::recompileProjectionKernels(Device& device) {
	// The projection-sensitive kernels bake _bakedProjection as a C++
	// constant at trace time (§9 perf investigation: perspective must keep
	// the pre-projection instruction sequence). These are the resolver-free
	// kernels recompileCallables deliberately skips — pure functions of the
	// baked tag, so a projection flip rebuilds exactly these. Revisited
	// projections hit the shader disk cache.
	compilePrepass(device);
	compileHitAndTemporal(device);
	compileHistory(device);
	compileAtrous(device);
	compileAtrousSmem(device);
}

//==========================================================================
// RelaxDenoiser::createImages
//==========================================================================

void RelaxDenoiser::createImages(Device& device, uint width, uint height) {
	for (int i = 0; i < 2; i++) {
		_relaxDiffHistory[i]       = device.create_image<float>(PixelStorage::HALF4, width, height);
		_relaxDiffFastHistory[i]   = device.create_image<float>(PixelStorage::HALF4, width, height);
		_relaxSpecHistory[i]       = device.create_image<float>(PixelStorage::HALF4, width, height);
		_relaxSpecFastHistory[i]   = device.create_image<float>(PixelStorage::HALF4, width, height);
		_relaxNormalRoughnessPrev[i] = device.create_image<float>(PixelStorage::HALF4, width, height);
		_relaxViewZPrev[i]         = device.create_image<float>(PixelStorage::HALF1, width, height);
		// RG16F: .x = historyLength, .y = specularHistoryConfidence. TA writes both
		// (PassDenoiserHitAndTemporal out_HistoryLength); atrous/HistoryClamping read
		// .y as specConf. Was HALF1 — the .y write was silently truncated to 0,
		// pinning all specConf-driven relaxation at zero.
		_relaxHistoryLengthPrev[i] = device.create_image<float>(PixelStorage::HALF2, width, height);
		_relaxSpecHitDistPrev[i]  = device.create_image<float>(PixelStorage::HALF1, width, height);

		_relaxDiffHistory[i].set_name("relax_diff_hist" + ci::toString(i));
		_relaxDiffFastHistory[i].set_name("relax_diff_fast" + ci::toString(i));
		_relaxSpecHistory[i].set_name("relax_spec_hist" + ci::toString(i));
		_relaxSpecFastHistory[i].set_name("relax_spec_fast" + ci::toString(i));
		_relaxNormalRoughnessPrev[i].set_name("relax_norh_prev" + ci::toString(i));
		_relaxViewZPrev[i].set_name("relax_view_prev" + ci::toString(i));
		_relaxHistoryLengthPrev[i].set_name("relax_hist_prev" + ci::toString(i));
		_relaxSpecHitDistPrev[i].set_name("relax_specd_prev" + ci::toString(i));
	}

	uint tileW = (width + 15u) / 16u;
	uint tileH = (height + 15u) / 16u;
	_relaxTiles = device.create_image<float>(PixelStorage::BYTE2, tileW, tileH);
	_relaxTiles.set_name("relax_tiles");

	for (int i = 0; i < 2; i++) {
		_relaxDiff[i]  = device.create_image<float>(PixelStorage::HALF4, width, height);
		_relaxSpec[i]  = device.create_image<float>(PixelStorage::HALF4, width, height);

		_relaxDiff[i].set_name("relax_diff" + ci::toString(i));
		_relaxSpec[i].set_name("relax_spec" + ci::toString(i));
	}

	_relaxDiffFast      = device.create_image<float>(PixelStorage::HALF4, width, height);
	_relaxSpecFast      = device.create_image<float>(PixelStorage::HALF4, width, height);
	_relaxHistoryLength = device.create_image<float>(PixelStorage::HALF1, width, height);

	// Split-kernel VMB prepass outputs (single-buffered, same-frame consumers).
	// FLOAT4 (not HALF): prevUV_vmb/curvature feed threshold tests where fp16
	// quantization would add gate noise on top of the compiler-hazard history;
	// Info2.z transports the min-3x3 hitDist, which must stay fp32-exact so TA's
	// output matches its former direct min computation bit-for-bit.
	// FetchA/B/C are written/read only for pixels whose VMB footprint test
	// passed (TA reads them inside its found-gate) — failed-VMB pixels carry
	// no fetch traffic at all. FetchA/B/C are HALF4, which halves the
	// found-pixel traffic at upstream-equivalent precision (NRD keeps these
	// quantities in RGBA16F/R16F histories).
	constexpr PixelStorage vmbFetchStorage = PixelStorage::HALF4;
	_relaxVmbInfo   = device.create_image<float>(PixelStorage::FLOAT4, width, height);
	_relaxVmbInfo2  = device.create_image<float>(PixelStorage::FLOAT4, width, height);
	_relaxVmbFetchA = device.create_image<float>(vmbFetchStorage, width, height);
	_relaxVmbFetchB = device.create_image<float>(vmbFetchStorage, width, height);
	_relaxVmbFetchC = device.create_image<float>(vmbFetchStorage, width, height);
	_relaxVmbInfo.set_name("relax_vmb_info");
	_relaxVmbInfo2.set_name("relax_vmb_info2");
	_relaxVmbFetchA.set_name("relax_vmb_fetch_a");
	_relaxVmbFetchB.set_name("relax_vmb_fetch_b");
	_relaxVmbFetchC.set_name("relax_vmb_fetch_c");

	_relaxHeap = device.create_bindless_array(10u);
	_relaxConstantsBuf = device.create_buffer<RelaxConstants>(2u); // element 1 = pano state (aliased members)
}

//==========================================================================
// RelaxDenoiser::zeroInitImages
//==========================================================================

void RelaxDenoiser::zeroInitImages(uint width, uint height) {
	auto& stream = Renderer::stream();
	// Batch all clears into a single command list to amortize stream submission
	// overhead (was 19 separate stream << dispatches).
	auto cl = CommandList::create();
	for (int i = 0; i < 2; i++) {
		cl << _clearImageShader(_relaxDiffHistory[i], _relaxDiffHistory[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxDiffFastHistory[i], _relaxDiffFastHistory[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxSpecHistory[i], _relaxSpecHistory[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxSpecFastHistory[i], _relaxSpecFastHistory[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxNormalRoughnessPrev[i], _relaxNormalRoughnessPrev[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxViewZPrev[i], _relaxViewZPrev[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxHistoryLengthPrev[i], _relaxHistoryLengthPrev[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxSpecHitDistPrev[i], _relaxSpecHitDistPrev[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxDiff[i], _relaxDiff[i]).dispatch(width, height);
		cl << _clearImageShader(_relaxSpec[i], _relaxSpec[i]).dispatch(width, height);
	}
	uint tileW = (width + 15u) / 16u;
	uint tileH = (height + 15u) / 16u;
	cl << _clearImageShader(_relaxTiles, _relaxTiles).dispatch(tileW, tileH);
	stream << cl.commit() << synchronize();
}

void RelaxDenoiser::release() {
	for (int i = 0; i < 2; i++) {
		_relaxDiffHistory[i].release();
		_relaxDiffFastHistory[i].release();
		_relaxSpecHistory[i].release();
		_relaxSpecFastHistory[i].release();
		_relaxNormalRoughnessPrev[i].release();
		_relaxViewZPrev[i].release();
		_relaxHistoryLengthPrev[i].release();
		_relaxSpecHitDistPrev[i].release();
		_relaxDiff[i].release();
		_relaxSpec[i].release();
	}

	_relaxTiles.release();
	_relaxDiffFast.release();
	_relaxSpecFast.release();
	_relaxHistoryLength.release();
	_relaxVmbInfo.release();
	_relaxVmbInfo2.release();
	_relaxVmbFetchA.release();
	_relaxVmbFetchB.release();
	_relaxVmbFetchC.release();
}

//==========================================================================
// RelaxDenoiser::_populateRelaxConstantsStruct
//==========================================================================

void RelaxDenoiser::_populateRelaxConstantsStruct(
	RelaxConstants& consts,
	const util::CameraData& cam,
	uint width, uint height,
	uint frameCount, uint cbField,
	bool accumReset,
	float dt)
{
	// FPS-aware accumulation: overwrite raw frame-count fields in _relaxSettings
	// in place so both the consts assignments below and the EMA tau computation
	// stay consistent with the live-FPS-derived values. Mirrors RTXDI FullSample
	// (RTXDI/Samples/FullSample/Source/UserInterface.cpp:1469-1481).
	if (_relaxSettings.accumulationTimeEnabled) {
		float fps = util::clampSmoothedFps(
			ci::app::getWindow()->getApp()->getAverageFps());
		uint32_t slow = util::computeAccumulatedFrames(_relaxSettings.accumulationTime, fps);
		uint32_t fast = util::computeFastAccumulatedFrames(slow);
		_relaxSettings.diffuseMaxAccumulatedFrameNum    = slow;
		_relaxSettings.diffuseMaxFastAccumulatedFrameNum = fast;
		_relaxSettings.specMaxAccumulatedFrameNum       = slow;
		_relaxSettings.specMaxFastAccumulatedFrameNum   = fast;
	}

	float fov_rad = glm::radians(cam.fov);
	float halfTan = glm::tan(fov_rad * 0.5f);
	consts.gFrustumForward = luisa::make_float4(cam.front, 0.0f);
	consts.gFrustumRight   = luisa::make_float4(cam.right * cam.aspect * halfTan, 0.0f);
	consts.gFrustumUp      = luisa::make_float4(cam.up * halfTan, 0.0f);
	consts.gPrevFrustumForward = luisa::make_float4(cam.prev_front, 0.0f);
	consts.gPrevFrustumRight   = luisa::make_float4(cam.prev_right * cam.aspect * halfTan, 0.0f);
	consts.gPrevFrustumUp      = luisa::make_float4(cam.prev_up * halfTan, 0.0f);
	consts.gWorldToClip     = cam.view_proj;
	// NRD frame contract: the denoiser reconstructs CAMERA-RELATIVE positions
	// (viewZ along the current frustum basis) while these matrices are
	// WORLD-space. The bridge is explicit in the shaders:
	// - projecting a camera-relative point P (virtual motion, guard, Stage 2):
	//   use (P + gCamPosCur) with the world matrices —
	//   prevClip(W) = prev_view_proj · (X_rel + C_cur), exact for rotation AND
	//   translation (the former composed-matrix attempt rotated the world-axes
	//   offset X_rel by the camera basis — garbage projections, mode 15 R = 0);
	// - reconstructing previous-frame tap positions (prev frustum basis, prev
	//   camera at origin): add (gCamPosCur − gCameraDelta) = C_prev;
	// - gCameraDelta stays the WORLD-axes translation C_cur − C_prev (NRD's
	//   semantic), used for the depth reference and the tap position.
	consts.gWorldToClipPrev = cam.prev_view_proj;
	consts.gCameraDelta = luisa::make_float4(cam.position - cam.prev_position, 0.0f);
	consts.gCamPosCur = luisa::make_float4(cam.position, 0.0f);
	// Normals are stored in WORLD axes (denoiseNormal / history), and NRD's
	// camera-relative world keeps the app-world AXES between frames (only the
	// origin moves) — gWorldPrevToWorld exists for animated objects and is
	// identity for static scenes (NRD sample passes identity). The former
	// B_curᵀ·B_prev camera rotation wrongly rotated world-space prev normals in
	// the VMB normal gates by the full per-frame camera rotation.
	consts.gWorldPrevToWorld = luisa::float4x4(
		luisa::make_float4(1.f, 0.f, 0.f, 0.f), luisa::make_float4(0.f, 1.f, 0.f, 0.f),
		luisa::make_float4(0.f, 0.f, 1.f, 0.f), luisa::make_float4(0.f, 0.f, 0.f, 1.f));
	consts.gMvScale = luisa::make_float4(0.5f, 0.5f, 0.0f, 0.0f);
	consts.gJitterX = cam.jitter.x;
	consts.gJitterY = cam.jitter.y;
	consts.gRectSizeX = (int32_t)width;
	consts.gRectSizeY = (int32_t)height;
	consts.gRectSizeInvX = 1.0f / width;
	consts.gRectSizeInvY = 1.0f / height;
	consts.gRectSizePrevX = (float)width;
	consts.gRectSizePrevY = (float)height;
	consts.gResourceSizeInvX = 1.0f / width;
	consts.gResourceSizeInvY = 1.0f / height;
	consts.gResourceSizeInvPrevX = 1.0f / width;
	consts.gResourceSizeInvPrevY = 1.0f / height;
	consts.gResourceSizeX = (float)width;
	consts.gResourceSizeY = (float)height;
	consts.gResolutionScaleX = 1.0f;
	consts.gResolutionScaleY = 1.0f;
	consts.gRectOriginX = 0u;
	consts.gRectOriginY = 0u;
	consts.gUnproject = 2.0f * halfTan / (float)height;
	consts.gViewZScale = 1.0f;
	consts.gDiffMaxAccumulatedFrameNum     = (float)_relaxSettings.diffuseMaxAccumulatedFrameNum;
	consts.gDiffMaxFastAccumulatedFrameNum  = (float)_relaxSettings.diffuseMaxFastAccumulatedFrameNum;
	consts.gSpecMaxAccumulatedFrameNum      = (float)_relaxSettings.specMaxAccumulatedFrameNum;
	consts.gSpecMaxFastAccumulatedFrameNum  = (float)_relaxSettings.specMaxFastAccumulatedFrameNum;
	consts.gDisocclusionThreshold        = _relaxSettings.disocclusionThreshold;
	consts.gDisocclusionThresholdAlternate = _relaxSettings.disocclusionThresholdAlternate;
	consts.gSpecPhiLuminance   = _relaxSettings.specPhiLuminance;
	consts.gDiffPhiLuminance   = _relaxSettings.diffPhiLuminance;
	consts.gDiffMaxLuminanceRelativeDifference = -logf(std::max(std::min(_relaxSettings.diffuseMinLuminanceWeight, 1.0f), 1e-6f));
	consts.gSpecMaxLuminanceRelativeDifference = -logf(std::max(std::min(_relaxSettings.specularMinLuminanceWeight, 1.0f), 1e-6f));
	consts.gRoughnessFraction  = _relaxSettings.roughnessFraction;
	consts.gDepthThreshold     = _relaxSettings.depthThreshold;
	consts.gLobeAngleFraction  = _relaxSettings.lobeAngleFraction;
	consts.gMinHitDistanceWeight = _relaxSettings.minHitDistanceWeight;
	consts.gRoughnessEdgeStoppingEnabled = _relaxSettings.enableRoughnessEdgeStopping ? 1u : 0u;
	consts.gDiffBlurRadius     = _relaxSettings.diffusePrepassBlurRadius;
	consts.gSpecBlurRadius     = _relaxSettings.specularPrepassBlurRadius;
	consts.gSpecVarianceBoost  = _relaxSettings.specularVarianceBoost;
	consts.gSpecLobeAngleSlack = glm::radians(_relaxSettings.specularLobeAngleSlack);
	consts.gLuminanceEdgeStoppingRelaxation = _relaxSettings.luminanceEdgeStoppingRelaxation;
	consts.gRoughnessEdgeStoppingRelaxation = _relaxSettings.roughnessEdgeStoppingRelaxation;
	consts.gNormalEdgeStoppingRelaxation = _relaxSettings.normalEdgeStoppingRelaxation;
	consts.gConfidenceDrivenRelaxationMultiplier = _relaxSettings.confidenceDrivenRelaxationMultiplier;
	consts.gConfidenceDrivenLuminanceEdgeStoppingRelaxation = _relaxSettings.confidenceDrivenLuminanceEdgeStoppingRelaxation;
	consts.gConfidenceDrivenNormalEdgeStoppingRelaxation = _relaxSettings.confidenceDrivenNormalEdgeStoppingRelaxation;
	consts.gMinSpecHitDistForVirtualMotion       = 0.1f;
	// NRD NRD_MAX_ALLOWED_VIRTUAL_MOTION_ACCELERATION (Common.hlsli:55) = 5.0
	// ("keep relatively high to avoid ruining concave mirrors (was 15)").
	consts.gMaxAllowedVirtualMotionAcceleration   = 5.0f;
	consts.gDisocclusionParallaxDenominator       = _relaxSettings.disocclusionParallaxDenominator;
	consts.gDiffMinMaterial                    = _relaxSettings.diffMinMaterial;
	consts.gSpecMinMaterial                    = _relaxSettings.specMinMaterial;
	consts.gHistoryFixFrameNum               = (float)_relaxSettings.historyFixFrameNum;
	consts.gHistoryFixBasePixelStride        = _relaxSettings.historyFixBasePixelStride;
	consts.gHistoryFixEdgeStoppingNormalPower = _relaxSettings.historyFixEdgeStoppingNormalPower;
	consts.gFastHistoryClampingSigmaScale = _relaxSettings.fastHistoryClampingSigmaScale;
	consts.gHistoryAccelerationAmount      = _relaxSettings.antilagAccelerationAmount;
	consts.gHistoryResetTemporalSigmaScale = _relaxSettings.antilagTemporalSigmaScale;
	consts.gHistoryResetSpatialSigmaScale  = _relaxSettings.antilagSpatialSigmaScale;
	consts.gHistoryResetAmount             = _relaxSettings.antilagResetAmount;
	consts.gDenoisingRange = _relaxSettings.denoisingRange;
	consts.gHistoryThreshold = (float)_relaxSettings.spatialVarianceEstimationHistoryThreshold;
	consts.gOrthoMode    = 0.0f;
	consts.gFramerateScale = 60.0f * dt;

	// --- Non-perspective projection state (§5.4): raw bases + per-type params.
	// Rides ELEMENT 1 of _relaxConstantsBuf through aliased members (§9 perf
	// fix — see PassDenoiser.h). The unproject factor for the point-origin projections is
	// the per-projection angular pixel size (max of the horizontal/vertical
	// extents) — a constant per frame, unlike the fov-derived perspective
	// value; it overwrites the existing gUnproject member (no size change). ---
	const float fisheyeHalfFovRad = glm::radians(cam.fisheye_fov) * 0.5f;
	_relaxPanoConstsCpu = render::RelaxConstants{};
	_relaxPanoConstsCpu.gFrustumRight  = luisa::make_float4(cam.right, 0.0f);
	_relaxPanoConstsCpu.gFrustumUp     = luisa::make_float4(cam.up, 0.0f);
	_relaxPanoConstsCpu.gFrustumForward = luisa::make_float4(cam.front, 0.0f);
	_relaxPanoConstsCpu.gPrevFrustumRight = luisa::make_float4(cam.prev_right, 0.0f);
	_relaxPanoConstsCpu.gPrevFrustumUp    = luisa::make_float4(cam.prev_up, 0.0f);
	_relaxPanoConstsCpu.gPrevFrustumForward = luisa::make_float4(cam.prev_front, 0.0f);
	_relaxPanoConstsCpu.gCameraDelta = luisa::make_float4(fisheyeHalfFovRad, halfTan, 0.0f, 0.0f);
	switch (cam.projection) {
		case 1u: // equirect: horizontal 2pi/W, vertical pi/H
			consts.gUnproject = std::max(6.2831853f / (float)width,
			                             3.14159265f / (float)height);
			break;
		case 2u: // cylindrical: horizontal 2pi/W, vertical 2*tan(fov/2)/H
			consts.gUnproject = std::max(6.2831853f / (float)width,
			                             2.0f * halfTan / (float)height);
			break;
		case 3u: // fisheye: angular pixel = (fov/2)/(min dimension/2)
			consts.gUnproject = fisheyeHalfFovRad /
			                    (0.5f * (float)std::min(width, height));
			break;
		default:
			break;
	}

	// Time-based steady-state alphas: 1 - exp(-dt / tau)
	// where tau = N / 60, preserving behavior at 60fps
	float diffTauSlow = (float)_relaxSettings.diffuseMaxAccumulatedFrameNum / 60.0f;
	float diffTauFast = (float)_relaxSettings.diffuseMaxFastAccumulatedFrameNum / 60.0f;
	float specTauSlow = (float)_relaxSettings.specMaxAccumulatedFrameNum / 60.0f;
	float specTauFast = (float)_relaxSettings.specMaxFastAccumulatedFrameNum / 60.0f;
	consts.gDiffAlphaSteady     = 1.0f - expf(-dt / diffTauSlow);
	consts.gDiffAlphaFastSteady = 1.0f - expf(-dt / diffTauFast);
	consts.gSpecAlphaSteady     = 1.0f - expf(-dt / specTauSlow);
	consts.gSpecAlphaFastSteady = 1.0f - expf(-dt / specTauFast);
	consts.gAntilagYoungHistoryScale = _relaxSettings.antilagYoungHistoryScale;
	consts.gFrameIndex   = frameCount;
	consts.gResetHistory = (_relaxFrameIdx == 0 || accumReset) ? 1u : 0u;
	consts.gDiffCheckerboard    = cbField;
	consts.gSpecCheckerboard    = cbField;
	consts.gHasHistoryConfidence    = 0u;
	consts.gHasDisocclusionThresholdMix = 0u;
#if NT_DEBUG_VIZ
	consts.gDisableAntilag = _relaxDisableAntilag ? 1u : 0u;
	consts.gDisableClamp   = _relaxDisableClamp   ? 1u : 0u;
#else
	consts.gDisableAntilag = 0u;
	consts.gDisableClamp   = 0u;
#endif
	consts.gIsLastPass = 0u;
#if NT_DEBUG_VIZ
	consts.gDebugViz = (uint)_relaxHistClampDebugViz;
#else
	consts.gDebugViz = 0u;
#endif
}

//==========================================================================
// RelaxDenoiser::_populateRelaxConstants
//==========================================================================

void RelaxDenoiser::_populateRelaxConstants(
	CommandList& cmdlist,
	const util::CameraData& cam,
	uint width, uint height,
	uint frameCount, uint cbField,
	bool accumReset,
	float dt)
{
	// Populates the member staging struct (not a local) — the host pointer
	// must outlive CommandList submission.
	_relaxConstantsCpu = render::RelaxConstants{};
	_populateRelaxConstantsStruct(_relaxConstantsCpu, cam, width, height, frameCount, cbField, accumReset, dt);
	// Two-element upload: [0] = denoiser constants, [1] = pano state (§9
	// perf fix — the pano kernels read element 1 through aliased members;
	// perspective kernels never touch it, and their shader hashes stay
	// identical to the pre-projection build). The member array outlives
	// CommandList submission.
	_relaxConstantsUpload[0] = _relaxConstantsCpu;
	_relaxConstantsUpload[1] = _relaxPanoConstsCpu;
	cmdlist << _relaxConstantsBuf.copy_from(&_relaxConstantsUpload[0]);
}

//==========================================================================
// RelaxDenoiser::renderClassifyTiles (perf R2 item 9)
//==========================================================================

void RelaxDenoiser::renderClassifyTiles(CommandList& cmdlist, const FrameContext& ctx) {
	if (!_enabled) return;
	cmdlist << _relaxClassifyTiles(_relaxTiles, ctx.gbufDepth,
			_relaxSettings.denoisingRange)
		.dispatch(((ctx.width + 15u) / 16u) * 8u, ((ctx.height + 15u) / 16u) * 4u);
	_tilesClassifiedThisFrame = true;
}

//==========================================================================
// RelaxDenoiser::renderPrefilter
//==========================================================================

void RelaxDenoiser::renderPrefilter(CommandList& cmdlist, const FrameContext& ctx) {
	if (!_enabled) return;
	auto& profiler = util::Profiler::instance();
	util::CpuScopedTimer _cpu_ReLAX_prefilter("ReLAX/prefilter");
	profiler.set_pass("ReLAX/prefilter");
	cmdlist << _denoisePreFilterShader(
			ctx.denoiseAlbedo, ctx.denoiseSpecFactor, ctx.denoiseNormal,
			ctx.gbufDepth, ctx.gbufVis, ctx.gbufBaryMotion, _relaxTiles,
			ctx.camera, ctx.geometry.instance_buffer(),
			ctx.geometry.instance_transform_buffer(),
			ctx.materialPool.buffer(), ctx.geometry.vertex_bindless(),
			ctx.materialPool.textures()
#if NT_ENABLE_PROCEDURAL
			, *ctx.procBindless
#endif
		).dispatch(ctx.width, ctx.height);
}

//==========================================================================
// RelaxDenoiser::render
//==========================================================================

bool RelaxDenoiser::render(Stream& stream, const FrameContext& ctx,
                            const Image<float>& renderTarget, bool debugTagNone,
                            const Image<float>* rasterDepth) {
	if (!_enabled) return false;
	// Projection gate: the point-origin projections (equirect / cylindrical /
	// fisheye) run the analytic reconstruction paths added in Phase 2
	// (eval_point_origin_direction + analytic reprojection + u-wrap). The room
	// rig (4) still needs per-face frustum constants — denoising it stays
	// deferred to the per-face dispatch evaluation (report §5.2); offline
	// progressive accumulation remains its export path.
	if (ctx.camera.projection == static_cast<uint32_t>(util::CameraProjection::RoomRig)) {
		static bool sWarnedRig = false;
		if (!sWarnedRig) {
			sWarnedRig = true;
			CI_LOG_W("Denoiser: room-rig projection active - per-face denoising is "
			         "not implemented (report Phase 2 option); rendering raw "
			         "(use progressive accumulation for clean stills)");
		}
		return false;
	}
	_frameCount = ctx.frameCount;
	_width = ctx.width;
	_height = ctx.height;
	_accumReset = ctx.accumReset;

	auto& profiler = util::Profiler::instance();
	const auto& envmap = ctx.lightSampler.envmap_image();
	uint env_width = ctx.lightSampler.env_width();
	uint env_height = ctx.lightSampler.env_height();
	auto& env_rotation = ctx.lightSampler.env_rotation_buffer();
	float env_exposure = ctx.lightSampler.env_exposure();

#if NT_DEBUG_VIZ
	if (_relaxHistClampDebugViz == 20) {
		stream << _compositeBlitShader(renderTarget, ctx.accumBuffer, ctx.gbufDepth,
				envmap, ctx.camera, env_width, env_height,
				env_rotation, env_exposure,
				ctx.denoiseAlbedo, ctx.denoiseSpecFactor, ctx.specularBuffer,
				ctx.solidBgEnabled ? 1u : 0u, ctx.solidBgColor)
			.dispatch(_width, _height);
		return true;
	}
	if (_relaxHistClampDebugViz == 21) {
		// Raw specular hit-distance field (pre-denoiser, pre-PrePass). The RGB
		// can be clean while .w churns — this is the field TA steers the
		// virtual-motion ray with (report §12 R2).
		stream << _relaxRawHitDistViz(renderTarget, ctx.specularBuffer)
			.dispatch(_width, _height);
		return true;
	}
#endif

	uint prevIdx = _relaxFrameIdx & 1u;
	uint curIdx  = 1u - prevIdx;

	// Batch: constants upload + classify + prepass (NRD order) + temporal
	// accumulation + history fix + history clamping. The constants
	// upload used to be its own stream submission before the setup batch —
	// folding it in front of its first reader saves one submission per frame
	// (each stream << costs a CommandAllocator with pre-allocated GPU
	// buffers). Noisy inputs are read straight from shade's clamped accum/spec
	// buffers — the clamp-blit conversion passes were folded into the shade
	// writes. The former 3x3 hit-dist smoothing pass was removed (RC2): NRD's
	// reconstruction pass is zero-fill and OFF by default; TA consumes the
	// PrePass stochastic-min .w directly.
	{
		util::CpuScopedTimer _cpu_ReLAX_setup("ReLAX/setup");
		auto cl = CommandList::create();
		_populateRelaxConstants(cl, ctx.camera, _width, _height, _frameCount, ctx.cbField, _accumReset, ctx.deltaTime);
		// ClassifyTiles normally runs with the prefilter batch (Pass 1.6,
		// renderClassifyTiles) so prefilter's sky gate sees this frame's
		// tiles. Re-dispatch here only when that batch was skipped (debug-viz
		// early returns) — the prepass below is the first tile consumer.
		if (!_tilesClassifiedThisFrame) {
			profiler.set_pass("ReLAX/classify");
			cl << _relaxClassifyTiles(_relaxTiles, ctx.gbufDepth,
					_relaxSettings.denoisingRange)
				.dispatch(((_width + 15u) / 16u) * 8u, ((_height + 15u) / 16u) * 4u);
		}
		_tilesClassifiedThisFrame = false;
		profiler.set_pass("ReLAX/prepass");
		cl << _relaxPrepass(_relaxConstantsBuf, ctx.accumBuffer, ctx.specularBuffer,
				ctx.denoiseNormal, ctx.gbufDepth, _relaxTiles).dispatch(_width, _height);
		if (rasterDepth) {
			cl << _relaxMotionCopy(*rasterDepth, ctx.gbufVis, ctx.gbufBaryMotion)
				.dispatch(_width, _height);
		}

		// Split-kernel specular VMB prepass (e55): produces the virtual-motion
		// fetch/state that TA consumes as textures. Same bindings as the TA
		// spec channels (prev-frame histories + this frame's filtered input);
		// tiles give it the same sky-tile early-out TA has.
		profiler.set_pass("ReLAX/VmbPrepass");
		cl << _relaxSpecVmbPrepass(_relaxConstantsBuf,
				ctx.gbufDepth, ctx.denoiseNormal, ctx.gbufBaryMotion, ctx.specularBuffer,
				_relaxSpecHistory[prevIdx], _relaxSpecFastHistory[prevIdx],
				_relaxSpecHitDistPrev[prevIdx], _relaxNormalRoughnessPrev[prevIdx],
				_relaxViewZPrev[prevIdx], _relaxTiles,
				_relaxVmbInfo, _relaxVmbInfo2,
				_relaxVmbFetchA, _relaxVmbFetchB, _relaxVmbFetchC
			).dispatch(_width, _height);

		{
			util::CpuScopedTimer _cpu_ReLAX_TA_Clamp("ReLAX/TA+Clamp");
			profiler.set_pass("ReLAX/temporal");
			cl << _relaxTemporalAccumulation(
				_relaxConstantsBuf, ctx.gbufDepth, ctx.denoiseNormal, ctx.accumBuffer,
				ctx.gbufBaryMotion, ctx.gbufVis,
				_relaxDiffHistory[curIdx], _relaxDiffFastHistory[curIdx],
				_relaxHistoryLengthPrev[curIdx], _relaxNormalRoughnessPrev[curIdx],
				_relaxViewZPrev[curIdx], _relaxTiles,
				_relaxViewZPrev[prevIdx], _relaxNormalRoughnessPrev[prevIdx],
				_relaxDiffHistory[prevIdx], _relaxDiffFastHistory[prevIdx],
				_relaxHistoryLengthPrev[prevIdx],
				ctx.specularBuffer, _relaxSpecHistory[curIdx], _relaxSpecFastHistory[curIdx],
				_relaxSpecHistory[prevIdx], _relaxSpecFastHistory[prevIdx],
				_relaxViewZPrev[prevIdx], _relaxSpecHitDistPrev[prevIdx],
				_relaxSpecHitDistPrev[curIdx],
				_relaxVmbInfo, _relaxVmbInfo2,
				_relaxVmbFetchA, _relaxVmbFetchB, _relaxVmbFetchC
			).dispatch(_width, _height);

			profiler.set_pass("ReLAX/HistoryFix");
			cl << _relaxHistoryFix(_relaxConstantsBuf,
				_relaxDiffHistory[curIdx], _relaxDiffFastHistory[curIdx],
				_relaxHistoryLengthPrev[curIdx], _relaxNormalRoughnessPrev[curIdx],
				_relaxViewZPrev[curIdx], _relaxTiles,
				_relaxSpecHistory[curIdx], _relaxSpecFastHistory[curIdx]
			).dispatch(_width, _height);

#if NT_DEBUG_VIZ
			if (_relaxDebugStage == 0 || _relaxDebugStage >= 3) {
#endif
				profiler.set_pass("ReLAX/HistoryClamp");
				cl << _relaxHistoryClamping(_relaxConstantsBuf,
					_relaxDiffHistory[curIdx], _relaxDiffFastHistory[curIdx],
					_relaxHistoryLengthPrev[curIdx], ctx.accumBuffer,
					_relaxViewZPrev[curIdx], _relaxTiles,
					_relaxSpecHistory[curIdx], _relaxSpecFastHistory[curIdx],
					ctx.specularBuffer
				).dispatch(_width, _height);
#if NT_DEBUG_VIZ
			}
#endif

			stream << cl.commit();
		}
	}

#if NT_DEBUG_VIZ
	bool taDebugViz = _relaxHistClampDebugViz == 10;
	if (taDebugViz) {
		stream << _debugTaBlitShader(renderTarget, _relaxDiffHistory[curIdx],
				ctx.gbufDepth, envmap, ctx.camera, env_width, env_height,
				env_rotation, env_exposure,
				ctx.denoiseAlbedo, ctx.denoiseSpecFactor, _relaxSpecHistory[curIdx],
				ctx.solidBgEnabled ? 1u : 0u, ctx.solidBgColor
			).dispatch(_width, _height);
	}
#endif
	{
		// Anti-firefly reads from history directly, avoiding a redundant copy.
		// When disabled, copy history to ping-pong slot 0 for atrous input.
		// Anti-firefly/histbld + atrous iterations share a single CommandList:
		// same-CL UAV barriers order the write→read between slot 0 writes here
		// and slot 0 reads in atrous iteration 0. Saves one stream submission.
		uint atrousStartIdx = 0u;
		auto cl = CommandList::create();
		if (_relaxSettings.enableAntiFirefly) {
			util::CpuScopedTimer _cpu_ReLAX_antifirefly("ReLAX/antifirefly");
			profiler.set_pass("ReLAX/antifirefly");
			cl << _relaxAntiFirefly(_relaxConstantsBuf,
					_relaxDiff[0], _relaxDiffHistory[curIdx], _relaxTiles,
					_relaxSpec[0], _relaxSpecHistory[curIdx], _relaxTiles,
					_relaxNormalRoughnessPrev[curIdx])
				.dispatch(_width, _height);
		} else {
			util::CpuScopedTimer _cpu_ReLAX_histbld("ReLAX/histbld");
			profiler.set_pass("ReLAX/histbld diff");
			cl << _relaxDiffHistory[curIdx].copy_to(_relaxDiff[0]);
			profiler.set_pass("ReLAX/histbld spec");
			cl << _relaxSpecHistory[curIdx].copy_to(_relaxSpec[0]);
		}

#if NT_DEBUG_VIZ
		if (_relaxHistClampDebugViz == 11) {
			stream << cl.commit();
			stream << _compositeBlitShader(renderTarget, _relaxDiff[0], ctx.gbufDepth,
					envmap, ctx.camera, env_width, env_height,
				env_rotation, env_exposure,
				ctx.denoiseAlbedo, ctx.denoiseSpecFactor, _relaxSpec[0],
				ctx.solidBgEnabled ? 1u : 0u, ctx.solidBgColor)
				.dispatch(_width, _height);
			return true;
		}
#endif

#if NT_DEBUG_VIZ
		if (_relaxHistClampDebugViz == 12) {
			uint afIdx = atrousStartIdx;
			stream << cl.commit();
			stream << _compositeBlitShader(renderTarget, _relaxDiff[afIdx], ctx.gbufDepth,
					envmap, ctx.camera, env_width, env_height,
				env_rotation, env_exposure,
				ctx.denoiseAlbedo, ctx.denoiseSpecFactor, _relaxSpec[afIdx],
				ctx.solidBgEnabled ? 1u : 0u, ctx.solidBgColor)
				.dispatch(_width, _height);
			return true;
		}
#endif

		// Atrous iterations: isLastPass passed as direct shader arg, no per-iteration
		// copy_from needed. Reduces buffer uploads and CommandAllocator overhead.
		// Batched into the shared CommandList: the DX backend's EnhancedBarrierTracker
		// inserts automatic UAV barriers between dispatches that read+write the same
		// ping-pong slot (_relaxDiff[0]<->_relaxDiff[1], _relaxSpec[0]<->_relaxSpec[1]),
		// so per-iteration ordering is preserved without per-iteration stream submissions.
		uint atrousIter = _relaxSettings.atrousIterationNum;
		{
			util::CpuScopedTimer _cpu_ReLAX_atrous("ReLAX/atrous");
			profiler.set_pass("ReLAX/atrous");
			for (uint pass = 0; pass < atrousIter; pass++) {
				uint stepSize = 1u << pass;
				uint readIdx  = (pass + atrousStartIdx) & 1u;
				uint writeIdx = 1u - readIdx;
				uint isLastPass = (pass == atrousIter - 1u) ? 1u : 0u;
				// smem variant's 2px halo only covers taps within ±2 px of the
				// block — use it for steps 1-2 (exactly in-tile), direct variant
				// for step >= 4 (smem coords would clamp to the tile edge and
				// silently read the wrong neighbors). Equirect/cylindrical seam
				// wraps disqualify the smem variant at every step: a u-wrapped
				// tap lands across the whole image, outside any block halo, and
				// the smem coord clamp would read the wrong texel while the
				// world position still uses the wrapped UV.
				const bool wrapUProjection =
					ctx.camera.projection == static_cast<uint32_t>(util::CameraProjection::Equirect) ||
					ctx.camera.projection == static_cast<uint32_t>(util::CameraProjection::Cylindrical);
				auto& atrousShader =
					(stepSize <= 2u && !wrapUProjection) ? _relaxAtrousSmem : _relaxAtrous;
				cl << atrousShader(_relaxConstantsBuf,
						_relaxDiff[writeIdx], _relaxDiff[readIdx],
						_relaxNormalRoughnessPrev[curIdx], _relaxViewZPrev[curIdx],
						_relaxHistoryLengthPrev[curIdx], stepSize, isLastPass, _relaxTiles,
						_relaxSpec[writeIdx], _relaxSpec[readIdx]
					).dispatch(_width, _height);
			}
		}

		// Composite rides the same CommandList as the atrous iterations —
		// same-stream FIFO + automatic UAV barriers order it behind the last
		// atrous write (was a separate stream submission).
		uint atrousResultIdx = (atrousIter + atrousStartIdx) & 1u;
#if NT_DEBUG_VIZ
		if (_relaxHistClampDebugViz >= 13 && _relaxHistClampDebugViz <= 40) {
			// Spec-temporal viz modes: blit the RAW TA output (_relaxSpecHistory
			// is where TA writes the viz encoding). The former wiring blitted
			// _relaxSpec[atrousResultIdx] — five atrous blur iterations smeared
			// the debug channels (ramps smoothed, markers dissolved), making
			// every 13/14/15 capture approximate. The new blit also paints the
			// build-epoch marker (top-left corner) on the FINAL framebuffer,
			// where no early-out or blur can bury it.
			// (Range extended to 19 for modes 16-18: outside it the probe
			// encodings fell through to the scene composite and every decode
			// of them was garbage.)
			cl << _relaxSpecVizBlit(renderTarget, _relaxSpecHistory[curIdx])
				.dispatch(_width, _height);
		} else
#endif
		if (debugTagNone) {
			util::CpuScopedTimer _cpu_ReLAX_compose("ReLAX/compose");
			profiler.set_pass("ReLAX/compose");
			cl << _compositeBlitShader(renderTarget, _relaxDiff[atrousResultIdx],
					ctx.gbufDepth, envmap, ctx.camera, env_width, env_height,
				env_rotation, env_exposure,
				ctx.denoiseAlbedo, ctx.denoiseSpecFactor, _relaxSpec[atrousResultIdx],
				ctx.solidBgEnabled ? 1u : 0u, ctx.solidBgColor)
				.dispatch(_width, _height);
		}
		stream << cl.commit();
	}
	return true;
}
//==========================================================================
// RelaxDenoiser::drawUi
//==========================================================================

void RelaxDenoiser::drawUi() {
	if (ImGui::CollapsingHeader("ReLAX Denoiser")) {
		ImGui::Checkbox("Enable Denoiser", &_enabled);
		if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Master toggle for the ReLAX denoiser.");
		if (_enabled) {
			ImGui::Checkbox("FPS-aware accumulation", &_relaxSettings.accumulationTimeEnabled);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("When ON, the four *MaxAccumulatedFrameNum sliders below are derived each frame\nfrom accumulationTime * live FPS (RTXDI pattern). Wall-clock tau stays constant across framerates.");
			if (_relaxSettings.accumulationTimeEnabled) {
				ImGui::SameLine();
				ImGui::PushItemWidth(140.f);
				ImGui::SliderFloat("Accum time (s)", &_relaxSettings.accumulationTime, 0.05f, 1.0f, "%.3f");
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Target temporal-blur tau in seconds. 0.334 = 20 frames @ 60 fps, 30 @ 90 fps.");
				ImGui::PopItemWidth();
			}
			ImGui::Separator();

			ImGui::BeginDisabled(_relaxSettings.accumulationTimeEnabled);
			ImGui::SliderInt("Diff Max Accum", reinterpret_cast<int*>(&_relaxSettings.diffuseMaxAccumulatedFrameNum), 1, 60);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more stable, slower to react to lighting changes. Lower = more responsive, noisier.");
			ImGui::SliderInt("Diff Max Fast Accum", reinterpret_cast<int*>(&_relaxSettings.diffuseMaxFastAccumulatedFrameNum), 1, 20);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more averaging before clamping (smoother, more lag). Lower = more responsive.");
			ImGui::EndDisabled();
			ImGui::SliderFloat("Diff Phi Luminance", &_relaxSettings.diffPhiLuminance, 0.0f, 10.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = sharper diffuse edges (more noise). Lower = more blur across bright/dark.");
			ImGui::SliderFloat("Spec Phi Luminance", &_relaxSettings.specPhiLuminance, 0.0f, 10.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = sharper specular edges (preserves highlights, more noise). Lower = more blur across bright/dark.");
			ImGui::SliderFloat("Depth Threshold", &_relaxSettings.depthThreshold, 0.0f, 0.1f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more blur across depth edges. Lower = sharper depth separation (more noise).");
			ImGui::SliderInt("Atrous Iterations", reinterpret_cast<int*>(&_relaxSettings.atrousIterationNum), 0, 8);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = wider blur reach (smoother, slower). Lower = tighter blur (more noise).");
			ImGui::SliderFloat("Anti-lag Amount", &_relaxSettings.antilagAccelerationAmount, 0.0f, 1.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = faster reaction to lighting changes (less lag, more noise). Lower = more stable.");
			ImGui::SliderFloat("Antilag History Scale", &_relaxSettings.antilagYoungHistoryScale, 0.0f, 5.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = stronger anti-lag on young histories (faster settle, possible flicker). Lower = calmer post-reset.");
			ImGui::SliderFloat("Antilag Temporal Sigma", &_relaxSettings.antilagTemporalSigmaScale, 0.0f, 2.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = fewer history resets (more lag, more stable). Lower = more resets (faster reaction, possible flicker).");
			if (ImGui::CollapsingHeader("Disocclusion / Anti-lag Advanced")) {
				ImGui::SliderFloat("Antilag Reset Amount", &_relaxSettings.antilagResetAmount, 0.0f, 1.0f);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("How much history to reset when divergence detected. Higher = faster kill of disocclusion bright bias (more flicker). NRD default 0.5.");
				ImGui::SliderFloat("Antilag Spatial Sigma", &_relaxSettings.antilagSpatialSigmaScale, 0.5f, 8.0f);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Spatial smoothness during anti-lag reset. Lower = sharper (less bleed, noisier). NRD default 4.5; lower to ~2.5 if bright bleeds across edges.");
				ImGui::SliderInt("History Fix Frame Num", reinterpret_cast<int*>(&_relaxSettings.historyFixFrameNum), 0, 10);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Frames after disocclusion during which neighbor-fill runs. Higher = more frames of fill (slower re-converge but stable). NRD default 3.");
				ImGui::SliderFloat("History Fix Pixel Stride", &_relaxSettings.historyFixBasePixelStride, 4.0f, 40.0f);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Search radius (px) for similar neighbor in disocclusion fill. Higher = finds better match in textures (risk of bleed). NRD default 14.");
				ImGui::SliderFloat("Diff Min Material", &_relaxSettings.diffMinMaterial, 0.0f, 16.0f);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Material confidence threshold for diffuse. Higher = more aggressive reset on low-M reservoirs (faster bright-bias kill). NRD default 4.0.");
				ImGui::SliderFloat("Spec Min Material", &_relaxSettings.specMinMaterial, 0.0f, 16.0f);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Material confidence threshold for specular. Higher = stricter. NRD default 4.0.");
				ImGui::SliderFloat("Disocclusion Threshold", &_relaxSettings.disocclusionThreshold, 0.001f, 0.2f);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Depth sensitivity for disocclusion detection. Higher = more pixels marked disoccluded (more fill, possible soft edges).");
			}

			ImGui::Checkbox("Enable Anti-Firefly", &_relaxSettings.enableAntiFirefly);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("ON = suppress isolated bright pixels (recommended for metal). OFF = preserve highlights but risk blocky clusters.");
			ImGui::SliderFloat("Diff Prepass Blur", &_relaxSettings.diffusePrepassBlurRadius, 0.0f, 100.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more pre-blur (cleaner, softer). Lower = sharper but noisier. 0 disables.");
			ImGui::SliderFloat("Spec Prepass Blur", &_relaxSettings.specularPrepassBlurRadius, 0.0f, 100.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more pre-blur for specular highlights. Lower = sharper highlights, more noise.");
			ImGui::SliderFloat("Lum Relaxation", &_relaxSettings.luminanceEdgeStoppingRelaxation, 0.0f, 1.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more blur in low-confidence areas (cleaner, softer). Lower = sharper edges.");
			ImGui::SliderFloat("Normal Relaxation", &_relaxSettings.normalEdgeStoppingRelaxation, 0.0f, 1.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more blur across normal edges. Lower = sharper geometry edges.");
			ImGui::SliderFloat("Roughness Relaxation", &_relaxSettings.roughnessEdgeStoppingRelaxation, 0.0f, 1.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more blur across roughness changes. Lower = sharper roughness transitions.");
			ImGui::SliderFloat("Roughness Fraction", &_relaxSettings.roughnessFraction, 0.0f, 1.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = stricter separation across roughness (sharper, noisier). Lower = more blur across roughness.");
			ImGui::SliderFloat("Spec Lobe Slack (deg)", &_relaxSettings.specularLobeAngleSlack, 0.0f, 1.0f);
			if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = let highlights survive more normal variation (smoother, may leak). Lower = sharper highlight containment.");
			if (ImGui::CollapsingHeader("Specular")) {
				ImGui::BeginDisabled(_relaxSettings.accumulationTimeEnabled);
				ImGui::SliderInt("Spec Max Accum", reinterpret_cast<int*>(&_relaxSettings.specMaxAccumulatedFrameNum), 1, 60);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more stable highlights, slower to react. Lower = more responsive to view changes (noisier).");
				ImGui::SliderInt("Spec Max Fast Accum", reinterpret_cast<int*>(&_relaxSettings.specMaxFastAccumulatedFrameNum), 1, 20);
				if (ImGui::IsItemHovered()) ImGui::SetItemTooltip("Higher = more averaging for specular clamping (smoother, more lag). Lower = more responsive.");
				ImGui::EndDisabled();
			}
#if NT_DEBUG_VIZ
			ImGui::SliderInt("Debug Stage (0=all, 1=TA only, 2=+HistFix, 3=+HistClamp)", &_relaxDebugStage, 0, 3);
			ImGui::Checkbox("Disable Anti-lag", &_relaxDisableAntilag);
			ImGui::Checkbox("Disable YCoCg Clamp", &_relaxDisableClamp);
			ImGui::SliderInt("HistClamp Debug Viz (0=off, 10=TA, 13/14/15=spec temporal, 21=raw hitT)", &_relaxHistClampDebugViz, 0, 21);
#endif
		}
	}
}

//==========================================================================
// Config serialization
//==========================================================================
void RelaxDenoiser::toJson(ci::Json& j) const {
	const auto& s = _relaxSettings;
	j = ci::Json{
		{"enabled",                             _enabled},
		{"diffuseMaxAccumulatedFrameNum",       s.diffuseMaxAccumulatedFrameNum},
		{"diffuseMaxFastAccumulatedFrameNum",    s.diffuseMaxFastAccumulatedFrameNum},
		{"specMaxAccumulatedFrameNum",           s.specMaxAccumulatedFrameNum},
		{"specMaxFastAccumulatedFrameNum",        s.specMaxFastAccumulatedFrameNum},
		{"disocclusionThreshold",                s.disocclusionThreshold},
		{"disocclusionThresholdAlternate",       s.disocclusionThresholdAlternate},
		{"historyFixFrameNum",                   s.historyFixFrameNum},
		{"historyFixBasePixelStride",            s.historyFixBasePixelStride},
		{"historyFixEdgeStoppingNormalPower",    s.historyFixEdgeStoppingNormalPower},
		{"fastHistoryClampingSigmaScale",        s.fastHistoryClampingSigmaScale},
		{"specPhiLuminance",                     s.specPhiLuminance},
		{"diffPhiLuminance",                     s.diffPhiLuminance},
		{"lobeAngleFraction",                    s.lobeAngleFraction},
		{"roughnessFraction",                    s.roughnessFraction},
		{"depthThreshold",                       s.depthThreshold},
		{"minHitDistanceWeight",                  s.minHitDistanceWeight},
		{"enableRoughnessEdgeStopping",           s.enableRoughnessEdgeStopping},
		{"luminanceEdgeStoppingRelaxation",       s.luminanceEdgeStoppingRelaxation},
		{"normalEdgeStoppingRelaxation",          s.normalEdgeStoppingRelaxation},
		{"roughnessEdgeStoppingRelaxation",       s.roughnessEdgeStoppingRelaxation},
		{"diffusePrepassBlurRadius",              s.diffusePrepassBlurRadius},
		{"specularPrepassBlurRadius",             s.specularPrepassBlurRadius},
		{"spatialVarianceEstimationHistoryThreshold", s.spatialVarianceEstimationHistoryThreshold},
		{"atrousIterationNum",                    s.atrousIterationNum},
		{"enableAntiFirefly",                     s.enableAntiFirefly},
		{"antilagAccelerationAmount",             s.antilagAccelerationAmount},
		{"antilagSpatialSigmaScale",              s.antilagSpatialSigmaScale},
		{"antilagTemporalSigmaScale",             s.antilagTemporalSigmaScale},
		{"antilagResetAmount",                    s.antilagResetAmount},
		{"specularVarianceBoost",                 s.specularVarianceBoost},
		{"specularLobeAngleSlack",                s.specularLobeAngleSlack},
		{"diffuseMinLuminanceWeight",             s.diffuseMinLuminanceWeight},
		{"specularMinLuminanceWeight",            s.specularMinLuminanceWeight},
		{"denoisingRange",                        s.denoisingRange},
		{"reprojectionConfidence",                s.reprojectionConfidence},
		{"disocclusionParallaxDenominator",       s.disocclusionParallaxDenominator},
		{"diffMinMaterial",                       s.diffMinMaterial},
		{"specMinMaterial",                       s.specMinMaterial},
		{"accumulationTimeEnabled",               s.accumulationTimeEnabled},
		{"accumulationTime",                      s.accumulationTime},
		{"histClampDebugViz",                     _relaxHistClampDebugViz},
		{"debugStage",                            _relaxDebugStage},
	};
}

void RelaxDenoiser::fromJson(const ci::Json& j) {
	if (!j.is_object()) return;
	_enabled = j.value("enabled", _enabled);
	// Debug-viz mode persists across launches: DSL kernels compile at startup,
	// so a fresh instance starts with the slider's previous value — and the
	// corner epoch marker is visible from the first frame without UI driving.
	_relaxHistClampDebugViz = j.value("histClampDebugViz", _relaxHistClampDebugViz);
	_relaxDebugStage = j.value("debugStage", _relaxDebugStage);
	auto& s = _relaxSettings;
	s.diffuseMaxAccumulatedFrameNum        = j.value("diffuseMaxAccumulatedFrameNum",       s.diffuseMaxAccumulatedFrameNum);
	s.diffuseMaxFastAccumulatedFrameNum     = j.value("diffuseMaxFastAccumulatedFrameNum",   s.diffuseMaxFastAccumulatedFrameNum);
	s.specMaxAccumulatedFrameNum            = j.value("specMaxAccumulatedFrameNum",          s.specMaxAccumulatedFrameNum);
	s.specMaxFastAccumulatedFrameNum        = j.value("specMaxFastAccumulatedFrameNum",       s.specMaxFastAccumulatedFrameNum);
	s.disocclusionThreshold                 = j.value("disocclusionThreshold",               s.disocclusionThreshold);
	s.disocclusionThresholdAlternate        = j.value("disocclusionThresholdAlternate",      s.disocclusionThresholdAlternate);
	s.historyFixFrameNum                    = j.value("historyFixFrameNum",                  s.historyFixFrameNum);
	s.reprojectionConfidence               = j.value("reprojectionConfidence",               s.reprojectionConfidence);
	s.historyFixBasePixelStride             = j.value("historyFixBasePixelStride",           s.historyFixBasePixelStride);
	s.historyFixEdgeStoppingNormalPower     = j.value("historyFixEdgeStoppingNormalPower",   s.historyFixEdgeStoppingNormalPower);
	s.fastHistoryClampingSigmaScale         = j.value("fastHistoryClampingSigmaScale",       s.fastHistoryClampingSigmaScale);
	s.specPhiLuminance                      = j.value("specPhiLuminance",                    s.specPhiLuminance);
	s.diffPhiLuminance                      = j.value("diffPhiLuminance",                    s.diffPhiLuminance);
	s.lobeAngleFraction                     = j.value("lobeAngleFraction",                   s.lobeAngleFraction);
	s.roughnessFraction                     = j.value("roughnessFraction",                   s.roughnessFraction);
	s.depthThreshold                        = j.value("depthThreshold",                      s.depthThreshold);
	s.minHitDistanceWeight                  = j.value("minHitDistanceWeight",                 s.minHitDistanceWeight);
	s.enableRoughnessEdgeStopping           = j.value("enableRoughnessEdgeStopping",          s.enableRoughnessEdgeStopping);
	s.luminanceEdgeStoppingRelaxation       = j.value("luminanceEdgeStoppingRelaxation",      s.luminanceEdgeStoppingRelaxation);
	s.normalEdgeStoppingRelaxation          = j.value("normalEdgeStoppingRelaxation",         s.normalEdgeStoppingRelaxation);
	s.roughnessEdgeStoppingRelaxation       = j.value("roughnessEdgeStoppingRelaxation",      s.roughnessEdgeStoppingRelaxation);
	s.diffusePrepassBlurRadius              = j.value("diffusePrepassBlurRadius",             s.diffusePrepassBlurRadius);
	s.specularPrepassBlurRadius             = j.value("specularPrepassBlurRadius",            s.specularPrepassBlurRadius);
	s.spatialVarianceEstimationHistoryThreshold = j.value("spatialVarianceEstimationHistoryThreshold", s.spatialVarianceEstimationHistoryThreshold);
	s.atrousIterationNum                    = j.value("atrousIterationNum",                  s.atrousIterationNum);
	s.enableAntiFirefly                     = j.value("enableAntiFirefly",                   s.enableAntiFirefly);
	s.antilagAccelerationAmount             = j.value("antilagAccelerationAmount",           s.antilagAccelerationAmount);
	s.antilagSpatialSigmaScale              = j.value("antilagSpatialSigmaScale",            s.antilagSpatialSigmaScale);
	s.antilagTemporalSigmaScale             = j.value("antilagTemporalSigmaScale",           s.antilagTemporalSigmaScale);
	s.antilagResetAmount                    = j.value("antilagResetAmount",                  s.antilagResetAmount);
	s.specularVarianceBoost                 = j.value("specularVarianceBoost",               s.specularVarianceBoost);
	s.specularLobeAngleSlack                = j.value("specularLobeAngleSlack",              s.specularLobeAngleSlack);
	s.diffuseMinLuminanceWeight             = j.value("diffuseMinLuminanceWeight",           s.diffuseMinLuminanceWeight);
	s.specularMinLuminanceWeight            = j.value("specularMinLuminanceWeight",          s.specularMinLuminanceWeight);
	s.denoisingRange                        = j.value("denoisingRange",                      s.denoisingRange);
	s.disocclusionParallaxDenominator       = j.value("disocclusionParallaxDenominator",     s.disocclusionParallaxDenominator);
	s.diffMinMaterial                       = j.value("diffMinMaterial",                      s.diffMinMaterial);
	s.specMinMaterial                       = j.value("specMinMaterial",                      s.specMinMaterial);
	s.accumulationTimeEnabled               = j.value("accumulationTimeEnabled",              s.accumulationTimeEnabled);
	s.accumulationTime                      = j.value("accumulationTime",                     s.accumulationTime);
}

} // namespace newtype::core
