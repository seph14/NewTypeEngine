#pragma once

#include "newtype/render/Sampling.h"
#include "newtype/render/BSDF.h"
#include "newtype/render/LightSampler.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/util/Vertex.h"
#include "newtype/util/Camera.h"
#include "newtype/util/MoveOnlyAny.h"
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Production Path Tracer with NEE + MIS
//==============================================================================

/**
 * @brief Production path tracer with Next Event Estimation and MIS
 *
 * This is the main rendering engine for NewTypeEngine, implementing:
 * - Next Event Estimation (direct light sampling)
 * - Multiple Importance Sampling (BSDF + light sampling)
 * - PBR materials via MaterialPool
 * - GGX microfacet BSDF
 * - Russian roulette termination
 * - Temporal accumulation
 *
 * @note This is a static shader (not hot-reloadable). For experimentation,
 *       use the PathTracerShader in runtime_shaders/.
 */
class PathTracer {
public:
    //==========================================================================
    // Configuration
    //==========================================================================

    struct Config {
        uint maxDepth = 5;          // Maximum path length
        uint rrDepth = 3;           // Russian roulette start depth
        float rrThreshold = 0.95f;  // Russian roulette survival threshold
        bool enableNEE = true;      // Enable Next Event Estimation
        bool enableMIS = true;      // Enable Multiple Importance Sampling
    };

    //==========================================================================
    // Type Definitions
    //==========================================================================

    // Shader kernel type (for use with ShaderManager or direct dispatch)
    using ShaderType = Shader2D<
        luisa::compute::Image<float>,           // output (accumulation buffer)
        luisa::compute::Image<uint>,            // seed_image (random number state)
        luisa::compute::Accel,                  // accel (scene acceleration structure)
        luisa::compute::Buffer<util::Vertex>,   // mesh_vertices (vertex attributes, now using compact 32-byte Vertex)
        luisa::compute::Buffer<uint>,                 // material_indices (per-primitive material)
        luisa::compute::Buffer<MaterialData>,         // materials (material data from MaterialPool)
        float,                                    // light_select_pmf (PMF for light selection)
        luisa::compute::Buffer<EmissiveTriangle>,     // emissive_triangles (for NEE)
        uint,                                     // emissive_count (number of emissive triangles)
        util::CameraData,                         // camera
        uint,                                     // frame_index
        uint                                      // max_bounces
    >;

    //==========================================================================
    // Construction
    //==========================================================================

    /**
     * @brief Create path tracer
     *
     * @param device LuisaCompute device
     * @param config Path tracer configuration
     */
    explicit PathTracer(Device& device, const Config& config = Config{});

    ~PathTracer();

    // Non-copyable, non-movable
    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;
    PathTracer(PathTracer&&) = delete;
    PathTracer& operator=(PathTracer&&) = delete;

    //==========================================================================
    // Shader Compilation
    //==========================================================================

    /**
     * @brief Compile the path tracer kernel
     *
     * Returns a MoveOnlyAny containing the compiled shader.
     * Use with ShaderManager or dispatch directly.
     */
    [[nodiscard]] newtype::core::MoveOnlyAny compile(Device& device);

    //==========================================================================
    // Configuration Access
    //==========================================================================

    [[nodiscard]] const Config& config() const noexcept { return mConfig; }
    [[nodiscard]] uint maxDepth() const noexcept { return mConfig.maxDepth; }
    [[nodiscard]] bool enableNEE() const noexcept { return mConfig.enableNEE; }
    [[nodiscard]] bool enableMIS() const noexcept { return mConfig.enableMIS; }

private:
    Config mConfig;
};

//==============================================================================
// Inline Helper Functions for Shader Use
//==============================================================================

/**
 * @brief sRGB to linear conversion (for textures)
 *
 * Textures are typically stored in sRGB space, but rendering requires
 * linear values for correct lighting computation.
 */
inline Float srgb_to_linear(Float srgb) noexcept {
    return ite(srgb <= 0.04045f,
        srgb / 12.92f,
        pow((srgb + 0.055f) / 1.055f, 2.4f));
}

inline Float3 srgb_to_linear(Float3 srgb) noexcept {
    return ite(srgb.x <= 0.04045f,
        srgb / 12.92f,
        pow((srgb + 0.055f) / 1.055f, 2.4f));
}

} // namespace newtype::render
