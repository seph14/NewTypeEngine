#pragma once
#include "newtype/core/Config.h"
#include <luisa/luisa-compute.h>

#include <luisa/luisa-compute.h>

namespace newtype {
namespace scene  { class Geometry; }
namespace render { class MaterialPool; class LightSampler; }
namespace util   { struct CameraData; }

namespace core {

struct FrameContext {
    // G-Buffer images (Pipeline-owned, shared across passes)
    luisa::compute::Image<float>& gbufDepth;
    luisa::compute::Image<uint>&  gbufVis;
    luisa::compute::Image<float>& gbufBaryMotion; // RGBA16F: RG=barycentrics, BA=motion vectors
    luisa::compute::Image<float>& glassThroughput; // RGBA16F: RGB=attenuation*(1-F), A=Fresnel reflectivity

    // Camera
    const util::CameraData& camera;

    // Scene references
    scene::Geometry&      geometry;
    render::MaterialPool& materialPool;
    render::LightSampler& lightSampler;

    // Frame state
    uint frameCount;
    uint width;
    uint height;
    uint cbField;        // checkerboard: 0=off, 1 or 2 = active field
    bool accumReset;
    float deltaTime;     // frame delta time in seconds (clamped 1/90 .. 1/30)

    // Seed image (shared random state for candidate/initial passes)
    luisa::compute::Image<uint>& seedImage;

    // Accumulation outputs (Shade writes, Denoiser reads)
    luisa::compute::Image<float>& accumBuffer;
    luisa::compute::Image<float>& specularBuffer;

    // Denoiser auxiliary images (PreFilter writes)
    luisa::compute::Image<float>& denoiseAlbedo;
    luisa::compute::Image<float>& denoiseSpecFactor;
    luisa::compute::Image<float>& denoiseNormal;

    // Previous frame G-Buffer (disocclusion detection)
    luisa::compute::Image<float>& gbufDepthPrev;
    luisa::compute::Image<uint>&  gbufVisPrev;
    luisa::compute::Image<float>& denoiseNormalPrev;

#if NT_ENABLE_BSSRDF
    // SSS radiance buffer (written by PassSSS probe, read by shade).
    // HALF4 simultaneous-access image.
    luisa::compute::Image<float>& sssRadiance;
#endif

#if NT_ENABLE_PROCEDURAL
    // Procedural bindless array (instances, positions, indices, AABBs, normals)
    const luisa::compute::BindlessArray* procBindless = nullptr;
#endif

    // Solid background (display only, does not affect lighting)
    bool solidBgEnabled = false;
    luisa::float3 solidBgColor = luisa::make_float3(0.02f);
};

} // namespace core
} // namespace newtype
