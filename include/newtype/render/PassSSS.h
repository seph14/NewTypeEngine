#pragma once

#include <luisa/luisa-compute.h>
#include "newtype/core/Config.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/SurfaceResolver.h"
#include "newtype/render/LightSampler.h"
#include "newtype/core/BindingGroups.h"
#include "newtype/scene/Geometry.h"
#include "newtype/util/Camera.h"
#include "newtype/core/FrameContext.h"

namespace newtype {
namespace core {

using namespace newtype::render;

//==============================================================================
// SSSParams — runtime-tunable SSS probe parameters uploaded per frame
//==============================================================================
// PRIMITIVES ONLY — LUISA_STRUCT rejects DSL wrapper types (Float/UInt/etc).
// Mirrors the DI/GI/Denoiser params-buffer pattern; staging struct must be a
// class member because cmdlist << copy_from(&struct) is deferred (see
// session-2026-06-26-pass-params-buffer.md).
struct SSSParams {
    float radiusQuantile;        // default 0.5 (median) — see PassSSS.cpp::createImages for rationale
    float scatterDistanceScale;  // Multiplier on Burley s (default 1.0)
    float nLocalFloor;           // Reject grazing exits (default 1e-3, Bug 4)
};

//==============================================================================
// SSS Probe Shader Type Alias
//==============================================================================
// Mirrors PassDI::renderGBuffer dispatch shape (full-res, 16x16 block).
// Reads G-Buffer (depth/vis/bary) + scene/light resources; writes per-channel
// demodulated SSS radiance to a single HALF4 image.
using SSSProbeShaderType = luisa::compute::Shader<2,
    luisa::compute::Buffer<SSSParams>,                  // 0: sss_params
    luisa::compute::Image<float>,                       // 1: sss_radiance (HALF4, output)
    luisa::compute::Image<float>,                       // 2: gbuf_depth
    luisa::compute::Image<uint>,                        // 3: gbuf_vis
    luisa::compute::Image<float>,                       // 4: gbuf_bary_motion (reads .xy() for bary)
    luisa::compute::uint,                               // 5: frame_count
    luisa::compute::Accel,                              // 6: TLAS (probe + shadow rays)
    newtype::util::CameraData,                          // 7: camera
    SceneGeometryResources,                             // 8: instance/transform/material buffers
    luisa::compute::BindlessArray,                      // 9: vertex_bindless
    luisa::compute::BindlessArray,                      // 10: tex_bindless (material textures)
    LightSamplingResources                              // 11: triangle lights + alias table + counts
#if NT_ENABLE_PROCEDURAL
    ,
    luisa::compute::BindlessArray                       // 12: procedural bindless (instances, positions, indices, AABBs, normals)
#endif
>;

//==============================================================================
// PassSSS — single-shot ray-traced SSS probe
//==============================================================================
// Phase 1: Burley 2015 BSSRDF + single exit-point probe + single NEE candidate.
// Channel-per-frame strobing (1/3) with x3 scale — converges via ReLAX temporal
// accumulation. No ReSTIR resampling. See plan `resilient-squishing-avalanche.md`.
class PassSSS {
public:
    explicit PassSSS() noexcept = default;
    PassSSS(PassSSS&&) noexcept = default;
    PassSSS(const PassSSS&) = delete;
    PassSSS& operator=(const PassSSS&) = delete;
    PassSSS& operator=(PassSSS&&) = delete;
    ~PassSSS() noexcept = default;

    void compile(luisa::compute::Device& device,
                 newtype::scene::Geometry& geom,
                 const SurfaceResolverPoly& resolver);
    void createImages(luisa::compute::Device& device, uint width, uint height);
    void release();
    void renderProbe(luisa::compute::CommandList& cmdlist, const FrameContext& ctx);

    void drawUi();

    [[nodiscard]] luisa::compute::Image<float>& sssRadiance() noexcept { return _sssRadiance; }

private:
    scene::Geometry*           _geom = nullptr;
    SSSProbeShaderType         _probeShader;
    luisa::compute::Image<float>   _sssRadiance;     // HALF4, simultaneous_access=true
    luisa::compute::Buffer<SSSParams> _sssParamsBuf;
    SSSParams                  _sssParamsCpu{};      // CLASS MEMBER (cmdlist deferred)
    bool                       _paramDirty = true;

    void _populateParams(luisa::compute::CommandList& cmdlist) noexcept;
};

} // namespace core
} // namespace newtype

// Register SSSParams as a LuisaCompute DSL struct - must be at global scope.
LUISA_STRUCT(newtype::core::SSSParams,
    radiusQuantile,
    scatterDistanceScale,
    nLocalFloor
) {};
