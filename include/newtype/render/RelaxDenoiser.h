#pragma once
#include <luisa/luisa-compute.h>

namespace newtype {
namespace render {

//==============================================================================
// RelaxConstants — GPU-side constants uploaded each frame (LUISA_STRUCT)
//==============================================================================
// Ported from NRD v4.17 RELAX_SHARED_CONSTANTS (RELAX_Config.hlsli).
// Simplifications for our case:
//   - No SH mode (NRD_MODE = RADIANCE)
//   - No checkerboard (CheckerboardMode::OFF)
//   - No material ID comparison
//   - No confidence inputs
//   - Diffuse-only initially (NRD_SPEC=0)

struct RelaxConstants {
    // --- Transform matrices (row-major float4x4) ---
    luisa::float4x4 gWorldToClip;
    luisa::float4x4 gWorldToClipPrev;
    luisa::float4x4 gWorldToViewPrev;
    luisa::float4x4 gWorldPrevToWorld;

    // --- Rotator (stochastic sampling rotation) ---
    luisa::float4 gRotatorPre;             // cos/sin pair for spatial jitter

    // --- Current frame frustum ---
    luisa::float4 gFrustumRight;
    luisa::float4 gFrustumUp;
    luisa::float4 gFrustumForward;

    // --- Previous frame frustum ---
    luisa::float4 gPrevFrustumRight;
    luisa::float4 gPrevFrustumUp;
    luisa::float4 gPrevFrustumForward;

    // --- Camera motion ---
    luisa::float4 gCameraDelta;            // world-space camera translation
    luisa::float4 gMvScale;                // motion vector scale (xyz), unused(w)

    // --- Jitter ---
    float  gJitterX;
    float  gJitterY;
    // pad to float2 alignment

    // --- Resolution / rect ---
    float  gResolutionScaleX;
    float  gResolutionScaleY;
    float  gRectOffsetX;
    float  gRectOffsetY;
    float  gResourceSizeInvX;
    float  gResourceSizeInvY;
    float  gResourceSizeX;
    float  gResourceSizeY;
    float  gRectSizeInvX;
    float  gRectSizeInvY;
    float  gRectSizePrevX;
    float  gRectSizePrevY;
    float  gResourceSizeInvPrevX;
    float  gResourceSizeInvPrevY;

    // --- Printf / debug (unused, kept for struct compatibility) ---
    uint   gPrintfAtX;
    uint   gPrintfAtY;
    uint   gRectOriginX;
    uint   gRectOriginY;

    int    gRectSizeX;
    int    gRectSizeY;

    // --- Accumulation limits ---
    float  gSpecMaxAccumulatedFrameNum;     // 0 for diffuse-only
    float  gSpecMaxFastAccumulatedFrameNum; // 0 for diffuse-only
    float  gDiffMaxAccumulatedFrameNum;     // default 30
    float  gDiffMaxFastAccumulatedFrameNum; // default 6

    // --- Disocclusion thresholds ---
    float  gDisocclusionThreshold;
    float  gDisocclusionThresholdAlternate;

    // --- Material / strand (unused in diffuse-only) ---
    float  gCameraAttachedReflectionMaterialID;
    float  gStrandMaterialID;
    float  gStrandThickness;

    // --- Edge-stopping: roughness ---
    float  gRoughnessFraction;              // default 0.15

    // --- Specular-only (kept for struct layout, set to 0) ---
    float  gSpecVarianceBoost;

    // --- Debug ---
    float  gSplitScreen;

    // --- Blur radii (prepass, optional) ---
    float  gDiffBlurRadius;
    float  gSpecBlurRadius;

    // --- Edge-stopping: depth ---
    float  gDepthThreshold;                 // default 0.003

    // --- Edge-stopping: lobe angle ---
    float  gLobeAngleFraction;              // default 0.5
    float  gSpecLobeAngleSlack;

    // --- History fix ---
    float  gHistoryFixEdgeStoppingNormalPower; // default 8.0
    float  gRoughnessEdgeStoppingRelaxation;
    float  gNormalEdgeStoppingRelaxation;

    // --- History clamping ---
    float  gFastHistoryClampingSigmaScale;  // default 2.0

    // --- Anti-lag ---
    float  gHistoryAccelerationAmount;      // default 0.3
    float  gHistoryResetTemporalSigmaScale; // default 0.5
    float  gHistoryResetSpatialSigmaScale;  // default 4.5
    float  gHistoryResetAmount;             // default 0.5

    // --- Reprojection ---
    float  gReprojectionConfidence;         // default 0.1 - bilinear tap validity threshold

    // --- Denoising range ---
    float  gDenoisingRange;                 // sky threshold (e.g. 1e6)

    // --- Luminance edge-stopping ---
    float  gSpecPhiLuminance;
    float  gDiffPhiLuminance;               // default 2.0
    float  gDiffMaxLuminanceRelativeDifference;
    float  gSpecMaxLuminanceRelativeDifference;
    float  gLuminanceEdgeStoppingRelaxation;

    // --- Confidence (unused) ---
    float  gConfidenceDrivenRelaxationMultiplier;
    float  gConfidenceDrivenLuminanceEdgeStoppingRelaxation;
    float  gConfidenceDrivenNormalEdgeStoppingRelaxation;

    // --- Misc ---
    float  gDebug;
    float  gOrthoMode;
    float  gUnproject;                      // pixel-to-world scale
    float  gFramerateScale;                 // 60/fps
    float  gCheckerboardResolveAccumSpeed;
    float  gJitterDelta;

    // --- Time-based temporal accumulation alphas (pre-computed on CPU) ---
    float  gDiffAlphaSteady;       // 1 - exp(-dt / (diffMaxAccum / 60))
    float  gDiffAlphaFastSteady;   // 1 - exp(-dt / (diffMaxFast / 60))
    float  gSpecAlphaSteady;       // 1 - exp(-dt / (specMaxAccum / 60))
    float  gSpecAlphaFastSteady;   // 1 - exp(-dt / (specMaxFast / 60))
    float  gAntilagYoungHistoryScale; // default 0.3 — scale for young-history anti-lag

    // --- History fix params ---
    float  gHistoryFixFrameNum;             // default 3
    float  gHistoryFixBasePixelStride;      // default 14
    float  gHistoryFixAlternatePixelStride;
    float  gHistoryFixAlternatePixelStrideMaterialID;
    float  gHistoryThreshold;
    float  gViewZScale;
    float  gMinHitDistanceWeight;            // default 0.1

    // --- Virtual motion for specular ---
    float  gMinSpecHitDistForVirtualMotion;  // default 0.1 — minimum hit distance to activate virtual motion
    float  gMaxAllowedVirtualMotionAcceleration; // default 10.0 — cap on virtual motion parallax vs surface motion
    float  gDisocclusionParallaxDenominator;    // default 30.0 — NRD: parallax / this scales disocclusion threshold

    // --- Material thresholds ---
    float  gDiffMinMaterial;
    float  gSpecMinMaterial;

    // --- Toggles ---
    uint   gRoughnessEdgeStoppingEnabled;   // 1
    uint   gFrameIndex;
    uint   gDiffCheckerboard;               // 0
    uint   gSpecCheckerboard;               // 0
    uint   gHasHistoryConfidence;           // 0
    uint   gHasDisocclusionThresholdMix;    // 0
    uint   gResetHistory;                   // 1 on first frame / accum reset
    uint   gDisableAntilag;                 // debug: disable anti-lag accel + reset
    uint   gDisableClamp;                   // debug: disable YCoCg clamping
    uint   gIsLastPass;                     // 1 on last Atrous iteration (write history length)
    uint   gDebugViz;                       // HistoryClamping debug viz: 0=off, 1=histLen, 2=sumW, 3=clampFactor, 4=antilagDelta

};

} // namespace render
} // namespace newtype

// Register RelaxConstants as a LuisaCompute DSL struct — must be outside namespace
LUISA_STRUCT(newtype::render::RelaxConstants,
    gWorldToClip, gWorldToClipPrev, gWorldToViewPrev, gWorldPrevToWorld,
    gRotatorPre,
    gFrustumRight, gFrustumUp, gFrustumForward,
    gPrevFrustumRight, gPrevFrustumUp, gPrevFrustumForward,
    gCameraDelta, gMvScale,
    gJitterX, gJitterY,
    gResolutionScaleX, gResolutionScaleY, gRectOffsetX, gRectOffsetY,
    gResourceSizeInvX, gResourceSizeInvY, gResourceSizeX, gResourceSizeY,
    gRectSizeInvX, gRectSizeInvY, gRectSizePrevX, gRectSizePrevY,
    gResourceSizeInvPrevX, gResourceSizeInvPrevY,
    gPrintfAtX, gPrintfAtY, gRectOriginX, gRectOriginY,
    gRectSizeX, gRectSizeY,
    gSpecMaxAccumulatedFrameNum, gSpecMaxFastAccumulatedFrameNum,
    gDiffMaxAccumulatedFrameNum, gDiffMaxFastAccumulatedFrameNum,
    gDisocclusionThreshold, gDisocclusionThresholdAlternate,
    gCameraAttachedReflectionMaterialID, gStrandMaterialID, gStrandThickness,
    gRoughnessFraction, gSpecVarianceBoost, gSplitScreen,
    gDiffBlurRadius, gSpecBlurRadius,
    gDepthThreshold, gLobeAngleFraction, gSpecLobeAngleSlack,
    gHistoryFixEdgeStoppingNormalPower, gRoughnessEdgeStoppingRelaxation, gNormalEdgeStoppingRelaxation,
    gFastHistoryClampingSigmaScale,
    gHistoryAccelerationAmount, gHistoryResetTemporalSigmaScale, gHistoryResetSpatialSigmaScale, gHistoryResetAmount,
    gReprojectionConfidence,
    gDenoisingRange,
    gSpecPhiLuminance, gDiffPhiLuminance,
    gDiffMaxLuminanceRelativeDifference, gSpecMaxLuminanceRelativeDifference, gLuminanceEdgeStoppingRelaxation,
    gConfidenceDrivenRelaxationMultiplier, gConfidenceDrivenLuminanceEdgeStoppingRelaxation, gConfidenceDrivenNormalEdgeStoppingRelaxation,
    gDebug, gOrthoMode, gUnproject, gFramerateScale,
    gCheckerboardResolveAccumSpeed, gJitterDelta,
    gDiffAlphaSteady, gDiffAlphaFastSteady, gSpecAlphaSteady, gSpecAlphaFastSteady,
    gAntilagYoungHistoryScale,
    gHistoryFixFrameNum, gHistoryFixBasePixelStride,
    gHistoryFixAlternatePixelStride, gHistoryFixAlternatePixelStrideMaterialID,
    gHistoryThreshold, gViewZScale, gMinHitDistanceWeight,
    gMinSpecHitDistForVirtualMotion, gMaxAllowedVirtualMotionAcceleration,
    gDisocclusionParallaxDenominator,
    gDiffMinMaterial, gSpecMinMaterial,
    gRoughnessEdgeStoppingEnabled, gFrameIndex,
    gDiffCheckerboard, gSpecCheckerboard,
    gHasHistoryConfidence, gHasDisocclusionThresholdMix, gResetHistory,
    gDisableAntilag, gDisableClamp,
    gIsLastPass, gDebugViz
) {};

namespace newtype {
namespace render {

//==============================================================================
// RelaxSettings — CPU-side tunable parameters
//==============================================================================
struct RelaxSettings {
    // Temporal accumulation
    uint32_t diffuseMaxAccumulatedFrameNum  = 30;
    uint32_t diffuseMaxFastAccumulatedFrameNum = 6;
    uint32_t specMaxAccumulatedFrameNum     = 30;   // unused in diffuse-only
    uint32_t specMaxFastAccumulatedFrameNum = 6;    // unused in diffuse-only

    // Disocclusion
    float disocclusionThreshold             = 0.01f;
    float disocclusionThresholdAlternate    = 0.05f;
    float disocclusionParallaxDenominator   = 30.0f;

    // History fix
    uint32_t historyFixFrameNum             = 3;
    float    historyFixBasePixelStride      = 14.0f;
    float    historyFixEdgeStoppingNormalPower = 8.0f;

    // History clamping
    float fastHistoryClampingSigmaScale     = 2.0f;

    // Edge-stopping
    float specPhiLuminance                  = 1.0f;
    float diffPhiLuminance                  = 1.0f;  // RTXDI FullSample override for ReSTIR input (UserInterface.cpp:~1470); NRD default is 2.0
    float lobeAngleFraction                 = 0.5f;
    float roughnessFraction                 = 0.15f;
    float depthThreshold                    = 0.003f;
    float minHitDistanceWeight              = 0.1f;
    bool  enableRoughnessEdgeStopping       = true;

    // Edge-stopping relaxation
    float luminanceEdgeStoppingRelaxation   = 0.5f;
    float normalEdgeStoppingRelaxation      = 0.3f;
    float roughnessEdgeStoppingRelaxation   = 1.0f;

    // Confidence-driven relaxation (requires confidence texture, currently unused)
    float confidenceDrivenRelaxationMultiplier             = 0.0f;
    float confidenceDrivenLuminanceEdgeStoppingRelaxation  = 0.0f;
    float confidenceDrivenNormalEdgeStoppingRelaxation     = 0.0f;

    // Prepass
    float diffusePrepassBlurRadius          = 30.0f;
    float specularPrepassBlurRadius         = 50.0f;

    // Spatial variance estimation
    uint32_t spatialVarianceEstimationHistoryThreshold = 1;  // RTXDI FullSample override: SVE after 1 frame stabilizes young penumbra history

    // Atrous
    uint32_t atrousIterationNum             = 5;

    // Anti-firefly (RCRS). NRD RELAX default is OFF (NRDSettings.h:448); the
    // NRD-Sample also runs with it off. Was ON since 2026-06 to stop blocky
    // clusters on high-spec metal (isolated env fireflies blowing up Atrous
    // variance) — if those clusters return, re-enable this first.
    bool  enableAntiFirefly                 = true;   // RTXDI FullSample ships RCRS on for ReSTIR + HDR input

    // Anti-lag
    float antilagAccelerationAmount         = 0.3f;
    float antilagSpatialSigmaScale          = 4.5f;
    float antilagTemporalSigmaScale         = 0.5f;  // NRD default (NRDSettings.h:355)
    float antilagResetAmount                = 0.5f;
    float antilagYoungHistoryScale          = 2.0f;

    // Reprojection
    float reprojectionConfidence               = 0.1f;

    // Specular
    // NRD default is 0.0f (off). Was 1.0f to give Atrous a variance signal on
    // low-R metal pixels that miss the narrow GGX lobe (2nd moment = 0); at 1.0
    // it also keeps injecting variance into converged specular, over-blurring
    // reflections.
    float specularVarianceBoost             = 0.0f;
    float specularLobeAngleSlack            = 0.15f; // degrees, converted to radians for GPU

    // Minimum luminance weight (converted to MaxLuminanceRelativeDifference via -log)
    float diffuseMinLuminanceWeight         = 0.0f;
    float specularMinLuminanceWeight        = 0.0f;

    // Denoising range
    float denoisingRange                    = 500000.0f;

    // Material edge-stopping (NRD defaults: 4.0 / 4.0 — triggers history reset on
    // low-confidence/low-M reservoirs, which is exactly what disocclusion produces.
    // Set to 0 to disable material-based reset.)
    float diffMinMaterial                   = 4.0f;
    float specMinMaterial                   = 4.0f;

    // FPS-aware accumulation (RTXDI FullSample pattern).
    // When enabled, the four *MaxAccumulatedFrameNum fields above are recomputed
    // each frame from `accumulationTime * smoothedFps` (clamped to [0, 63]) and
    // `max(slow/6, 2)` for fast. Defaults OFF to preserve bit-identical behavior
    // for configs that don't set the flag.
    bool  accumulationTimeEnabled           = false;
    float accumulationTime                  = 0.334f; // seconds; RTXDI sample default (20 frames @ 60 fps)
};

} // namespace render
} // namespace newtype
