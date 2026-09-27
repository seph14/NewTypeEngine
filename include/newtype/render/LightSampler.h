//
// Created by Claude on 2026/03/25.
//

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/runtime/image.h>
#include "MaterialPool.h"
#include "cinder/Surface.h"

namespace newtype {

// Forward declarations
namespace scene {
class Geometry;
class MeshShape;
class LightShape;
} // namespace scene

namespace render {

using namespace luisa;
using compute::Device;
using compute::Stream;
using compute::Buffer;
using compute::Image;

//==============================================================================
// Alias Table Entry (for O(1) light sampling)
//==============================================================================

/**
 * @brief Single entry in the alias table
 *
 * The alias method allows O(1) sampling from a discrete distribution
 * with arbitrary weights. Each entry stores either a real item or
 * an alias pointer to another item.
 *
 * Based on Vose's alias method algorithm.
 */
struct AliasEntry {
    uint  alias_index;    // Index of alias triangle (or itself if no alias)
    float pdf;            // Probability for this entry (normalized to n/sum)
    uint  triangle_index; // Actual triangle index in global space
    uint  padding;        // Pad to 16 bytes
};
static_assert(sizeof(AliasEntry) == 16u);

//==============================================================================
// Emissive Triangle Record
//==============================================================================

/**
 * @brief CPU-side record of an emissive triangle
 *
 * Stored during build phase for generating the alias table.
 */
struct EmissiveTriangleRecord {
    const scene::MeshShape *shape;     // Pointer to the shape (for vertex access)
    uint          instance_id;   // Which instance contains this triangle
    uint          primitive_id;  // Which triangle within the instance
    uint          triangle_index;// Global triangle index
    float         area;          // Triangle area (in world space)
    luisa::float3 emission;      // Emissive color (cached for CPU-side power recalculation)
    float         power;         // Luminance * area
    uint          material_index;// Material pool index for GPU emission lookup
};

//==============================================================================
// Environment Light
//==============================================================================

/**
 * @brief Environment map light source
 *
 * Provides infinite area light sampling via 2D CDFs.
 * Uses hierarchical sampling: marginal CDF for rows, conditional CDF for columns.
 */
class EnvironmentLight {
    friend class LightSampler;
public:
    struct CDFs {
        Image<float> marginal;     // 1D CDF for y-axis (height x 1)
        Image<float> conditional; // 2D CDF for x-axis (height x width)
        Image<float> pdf;         // PDF values (height x width)
        float integral;           // Total integral (for normalization)
        float width_inv;
        float height_inv;
    };

private:
    Device &_device;
    CDFs _cdfs;
    uint _width, _height;
    bool _built = false;

    // Original envmap (for radiance lookup)
    Image<float> _envmap;

    // GPU CDF buffers (efficient binary search)
    luisa::compute::Buffer<float> _cdfs_buffer_marginal;     // (height+1) floats
    luisa::compute::Buffer<float> _cdfs_buffer_conditional;  // height*(width+1) floats

    // --- Rotation / Exposure / Sky parameters ---
    float _yaw = 0.0f;             // radians, horizontal rotation around Y
    float _elevation = 0.698f;     // ~40 degrees, sun elevation for procedural sky
    float _exposure = 1.0f;        // radiance multiplier (not baked into CDFs)
    luisa::float3 _sunDirection{luisa::make_float3(0.0f, 1.0f, 0.0f)};
    luisa::float3x3 _rotationMatrix{luisa::make_float3x3(1.0f, 0.0f, 0.0f,
                                                          0.0f, 1.0f, 0.0f,
                                                          0.0f, 0.0f, 1.0f)};
    luisa::compute::Buffer<luisa::float3x3> _rotationBuffer;
    bool _rotationDirty = true;

    // Procedural sky parameters
    float _turbidity = 2.5f;
    float _sunIntensity = 1.5f;
    luisa::float3 _groundAlbedo{luisa::make_float3(0.3f)};
    bool _isProcedural = false;

    void _update_rotation() noexcept;

public:
    explicit EnvironmentLight(Device &device) noexcept;
    ~EnvironmentLight() = default;

    /**
     * @brief Build environment light from HDR image
     */
    void build(Image<float> &&envmap, Stream &stream) noexcept;

    /**
     * @brief Build CDFs from CPU-side pixel data
     */
    void build_cdfs(const float* rgba_data, Stream &stream) noexcept;

    /**
     * @brief Generate Rayleigh/Mie procedural sky
     */
    void generate_procedural_sky(Stream &stream, uint width = 512u, uint height = 256u) noexcept;

    /**
     * @brief Regenerate procedural sky with current parameters
     */
    void regenerate_procedural_sky(Stream &stream) noexcept;

    /// Rotation / Exposure API
    void set_yaw(float yaw_rad) noexcept;
    void set_elevation(float elev_rad) noexcept;
    void set_exposure(float exp) noexcept { _exposure = exp; }
    void set_turbidity(float t) noexcept { _turbidity = t; }
    void set_sun_intensity(float i) noexcept { _sunIntensity = i; }
    void set_ground_albedo(luisa::float3 a) noexcept { _groundAlbedo = a; }

    // Uploads the rotation matrix when yaw/elevation changed. Returns true
    // when a new matrix was queued — consumers with rotation-dependent cached
    // GPU state (env presampled candidates) must order behind the upload.
    bool update_rotation_buffer(Stream &stream) noexcept;

    /// Accessors
    [[nodiscard]] bool has_envmap() const noexcept { return _built; }
    [[nodiscard]] const Image<float>& envmap() const noexcept { return _envmap; }
    [[nodiscard]] const CDFs& cdfs() const noexcept { return _cdfs; }
    [[nodiscard]] float yaw() const noexcept { return _yaw; }
    [[nodiscard]] float elevation() const noexcept { return _elevation; }
    [[nodiscard]] float exposure() const noexcept { return _exposure; }
    [[nodiscard]] float turbidity() const noexcept { return _turbidity; }
    [[nodiscard]] const luisa::float3& sun_direction() const noexcept { return _sunDirection; }
    [[nodiscard]] const luisa::float3x3& rotation_matrix() const noexcept { return _rotationMatrix; }
    [[nodiscard]] const luisa::compute::Buffer<luisa::float3x3>& rotation_buffer() const noexcept { return _rotationBuffer; }
    [[nodiscard]] bool is_procedural() const noexcept { return _isProcedural; }
};

//==============================================================================
// LightSampler
//==============================================================================

/**
 * @brief Light sampling system for ReSTIR DI
 *
 * Provides O(1) light sampling via alias table for emissive triangles
 * and environment map sampling via hierarchical CDFs.
 *
 * This is the foundation for ReSTIR DI — initial candidate sampling
 * and PDF evaluation both depend on this system.
 *
 * Usage:
 * ```cpp
 * LightSampler sampler(device);
 * sampler.build(stream, geometry, material_pool);
 *
 * // In shader:
 * auto [triangle_id, uv, pdf] = sampler.sample_light(rng.next_float2());
 * auto light_emission = sampler.get_emission(triangle_id, uv);
 * ```
 */
class LightSampler {
public:
    //==========================================================================
    // GPU Sampling Data (DSL-accessible)
    //==========================================================================

    /**
     * @brief Compact triangle data for GPU
     *
     * Stores the minimum information needed to evaluate a light:
     * - Instance ID (for transform lookup)
     * - Primitive ID (for triangle indexing)
     * - Emission color
     */
    struct TriangleLight {
        uint          instance_id;    // 4  @ 0
        uint          primitive_id;   // 4  @ 4
        float         area;           // 4  @ 8
        float         pdf;            // 4  @ 12  (power / total_power, precomputed)
        // Precomputed world-space geometric normal, normalize(cross(v1-v0, v2-v0)),
        // refreshed alongside the vertex buffer in update_transforms. Replaces the
        // per-eval cross+normalize in DI/GI/shade light evaluation. Stored as 3 raw
        // floats (not luisa::float3) to keep the struct size predictable.
        float         nx;             // 4  @ 16
        float         ny;             // 4  @ 20
        float         nz;             // 4  @ 24
        // Baked emissive color (copy of MaterialData::emission). Refreshed in
        // _upload_to_gpu / update_weights / update_transforms; lets every light-eval
        // site skip the 192-byte MaterialData read it previously paid for .emission.
        float         ex;             // 4  @ 28
        float         ey;             // 4  @ 32
        float         ez;             // 4  @ 36
    };

    /**
     * @brief Vertex positions for triangle light sampling
     *
     * Stores world-space vertex positions for barycentric sampling.
     * Parallel to TriangleLight buffer (same index).
     */
    struct TriangleVertexData {
        luisa::float3 v0;  // First vertex position (world space)
        luisa::float3 v1;  // Second vertex position (world space)
        luisa::float3 v2;  // Third vertex position (world space)
        //uint            _padding;  // Pad to 16-byte boundary
    };

    /**
     * @brief Light sampling result
     *
     * Returned by the GPU sampling function.
     */
    struct Sample {
        uint          triangle_index;// Index into triangle_lights buffer
        luisa::float2 uv;            // Barycentric coordinates on triangle
        float         pdf;           // Selection PDF
    };

private:
    Device &_device;

    // Emissive triangles
    luisa::vector<EmissiveTriangleRecord> _emissive_tris_cpu;
    Buffer<TriangleLight>       _triangle_lights;
    Buffer<TriangleVertexData>  _triangle_vertices;  // World-space vertex positions
    Buffer<AliasEntry>          _alias_table;

    // Persistent staging buffers (avoid per-frame heap allocation in update_transforms)
    luisa::vector<TriangleVertexData> _vertex_staging;
    luisa::vector<TriangleLight> _light_staging;

    // Reverse mapping: material_index -> [indices into _emissive_tris_cpu]
    luisa::unordered_map<uint, luisa::vector<uint>> _materialToRecords;

    uint _total_emissive_count = 0u;
    float _total_emissive_count_float = 0.0f;  // Float version for DSL
    float _total_power = 0.0f;
    float _total_power_inv = 0.0f;
    float _emissive_count_inv = 0.0f;           // 1/N for uniform light sampling
    bool _uniform_sampling = false;              // Uniform vs power-weighted alias table

    // Instance → emissive triangle base index lookup
    // Maps instance_id → first emissive triangle index in flat buffer (~0u if not a light)
    Buffer<uint> _instanceToLightBase;
    uint _instanceCount = 0u;

    // Environment light
    EnvironmentLight _env_light;

    // GPU buffers for envmap CDFs
    Buffer<float> _env_marginal_buffer;
    Buffer<float> _env_conditional_buffer;

public:
    //==========================================================================
    // Construction
    //==========================================================================

    explicit LightSampler(Device &device) noexcept;
    ~LightSampler() = default;

    // Non-copyable
    LightSampler(const LightSampler&) = delete;
    LightSampler& operator=(const LightSampler&) = delete;

    //==========================================================================
    // Building
    //==========================================================================

    /**
     * @brief Build light sampling structures from scene
     *
     * Scans geometry for emissive triangles and builds alias table.
     *
     * @param stream Command stream for uploads
     * @param geometry Scene geometry (provides light_shapes)
     * @param material_pool Material pool for emission data
     */
    void build(Stream &stream, const scene::Geometry &geometry,
               const MaterialPool &material_pool) noexcept;

    /**
     * @brief Build with optional environment map
     */
    void build(Stream &stream, const scene::Geometry &geometry,
               const MaterialPool &material_pool,
               luisa::compute::Image<float> &&envmap) noexcept;

    /**
     * @brief Build envmap from CPU-side pixel data (creates GPU image + CDFs)
     */
    void build_envmap(uint width, uint height, const float* rgba_data, Stream &stream) noexcept;

    /**
     * @brief Build envmap from a Cinder Surface32f (HDR)
     */
    void build_envmap(const ci::Surface32f &surface, Stream &stream) noexcept;

    /**
     * @brief Rebuild for dynamic scenes
     *
     * Call this when emissive properties change (not needed for transform-only changes).
     */
    void rebuild(Stream &stream, const scene::Geometry &geometry,
                 const MaterialPool &material_pool) noexcept;

    /**
     * @brief Update alias table weights when only emission changed (no geometry scan)
     *
     * Recomputes power for all cached records from current material emission,
     * rebuilds alias table, and re-uploads GPU buffers.
     */
    void update_weights(Stream &stream, const MaterialPool &material_pool,
                        const scene::Geometry &geometry) noexcept;

    /**
     * @brief Update light vertex positions (and optionally areas) when transforms change
     *
     * Always re-uploads world-space vertex positions from current shape transforms.
     * If scale_changed, also recomputes triangle areas, power weights, and rebuilds
     * the alias table.
     *
     * @param scale_changed True if any light shape had a scale change
     */
    void update_transforms(Stream &stream, const scene::Geometry &geometry,
                           const MaterialPool &material_pool,
                           bool scale_changed) noexcept;

    //==========================================================================
    // GPU Sampling Interface
    //==========================================================================

    /**
     * @brief Sample a light (GPU-side)
     *
     * Returns a triangle light sampled proportionally to power.
     *
     * @param u Random number [0, 1)
     * @return std::pair<triangle_index (UInt), pdf (Float)>
     *
     * The returned pdf is the selected triangle's power/total — the same
     * value stored in TriangleLight::pdf — i.e. the pdf consumers must
     * divide the contribution by (matches tri_light.pdf / area usage in
     * the DI/GI/SHARC kernels).
     */
    [[nodiscard]] auto sample_light(const Float &u) const noexcept;

    /**
     * @brief Sample light with 3 random numbers (selection + surface point)
     *
     * u.x drives the alias-table selection, u.y/u.z sample the triangle
     * surface (sqrt trick, same convention as the DI presample kernel).
     * @return std::tuple<triangle_index (UInt), uv (Float2), pdf (Float)>
     */
    [[nodiscard]] auto sample_light_with_uv(const Float3 &u) const noexcept;

    /**
     * @brief Get PDF of selecting a triangle light
     *
     * @param triangle_index Index returned by sample_light()
     * @return Selection PDF (power/total, not including emission term)
     */
    [[nodiscard]] Float light_pdf(const UInt &triangle_index) const noexcept;

    /**
     * @brief Sample environment map (GPU-side)
     */
    [[nodiscard]] auto sample_env(const Float2 &u) const noexcept;

    /**
     * @brief Get environment map PDF
     */
    [[nodiscard]] auto env_pdf(const Float3 &direction) const noexcept;

    /**
     * @brief Get environment map radiance
     */
    [[nodiscard]] auto eval_env(const Float3 &direction) const noexcept;

    //==========================================================================
    // CPU-side Queries
    //==========================================================================

    /**
     * @brief Pure-CPU Vose alias table construction (unit-testable core)
     *
     * Builds an O(1)-sampling alias table realizing exactly the distribution
     * weights[i] / sum(weights). Worklists carry residual (not-yet-placed)
     * weights so each item's bucket is configured once with its remaining
     * mass — the property every consumer relies on when it divides by
     * TriangleLight::pdf (power/total).
     *
     * @param weights Non-negative per-entry weights (hidden lights use 0)
     * @param count   Entry count (may be 0 → empty table)
     * @param uniform_sampling Ignore weights, equal probability per entry
     */
    [[nodiscard]] static luisa::vector<AliasEntry> build_alias_table_cpu(
        const float *weights, size_t count, bool uniform_sampling) noexcept;

    [[nodiscard]] uint emissive_triangle_count() const noexcept { return _total_emissive_count; }
    [[nodiscard]] float total_power() const noexcept { return _total_power; }
    [[nodiscard]] float total_power_inv() const noexcept { return _total_power_inv; }
    [[nodiscard]] float emissive_count_inv() const noexcept { return _emissive_count_inv; }
    [[nodiscard]] bool has_environment() const noexcept { return _env_light.has_envmap(); }

    void set_uniform_sampling(bool enabled) noexcept { _uniform_sampling = enabled; }

    // --- Envmap rotation / exposure / sky ---
    void set_env_yaw(float yaw_rad) noexcept { _env_light.set_yaw(yaw_rad); }
    void set_env_elevation(float elev_rad) noexcept { _env_light.set_elevation(elev_rad); }
    void set_env_exposure(float exp) noexcept { _env_light.set_exposure(exp); }
    void set_env_turbidity(float t) noexcept { _env_light.set_turbidity(t); }
    void generate_procedural_sky(Stream &stream) noexcept { _env_light.generate_procedural_sky(stream); }
    void regenerate_procedural_sky(Stream &stream) noexcept { _env_light.regenerate_procedural_sky(stream); }
    // Returns true when update_rotation_buffer queued a new matrix this call.
    bool update_env_rotation(Stream &stream) noexcept { return _env_light.update_rotation_buffer(stream); }

    //==========================================================================
    // GPU Resource Access (for shader binding)
    //==========================================================================

    [[nodiscard]] const Buffer<TriangleLight>& triangle_buffer() const noexcept { return _triangle_lights; }
    [[nodiscard]] const Buffer<TriangleVertexData>& vertex_buffer() const noexcept { return _triangle_vertices; }
    [[nodiscard]] const Buffer<AliasEntry>& alias_table() const noexcept { return _alias_table; }
    [[nodiscard]] const Buffer<uint>& instance_to_light_base() const noexcept { return _instanceToLightBase; }
    [[nodiscard]] const EnvironmentLight& env_light() const noexcept { return _env_light; }
    [[nodiscard]] const Buffer<float>& env_marginal() const noexcept { return _env_marginal_buffer; }
    [[nodiscard]] const Buffer<float>& env_conditional() const noexcept { return _env_conditional_buffer; }
    [[nodiscard]] const luisa::compute::Buffer<float>& env_cdf_marginal() const noexcept { return _env_light._cdfs_buffer_marginal; }
    [[nodiscard]] const luisa::compute::Buffer<float>& env_cdf_conditional() const noexcept { return _env_light._cdfs_buffer_conditional; }
    [[nodiscard]] const Image<float>& envmap_image() const noexcept { return _env_light._envmap; }
    [[nodiscard]] float env_integral() const noexcept { return _env_light._cdfs.integral; }
    [[nodiscard]] uint env_width() const noexcept { return _env_light._width; }
    [[nodiscard]] uint env_height() const noexcept { return _env_light._height; }
    [[nodiscard]] const luisa::compute::Buffer<luisa::float3x3>& env_rotation_buffer() const noexcept { return _env_light.rotation_buffer(); }
    [[nodiscard]] const luisa::float3x3& env_rotation_matrix() const noexcept { return _env_light.rotation_matrix(); }
    [[nodiscard]] float env_exposure() const noexcept { return _env_light.exposure(); }
    [[nodiscard]] float env_yaw() const noexcept { return _env_light.yaw(); }
    [[nodiscard]] float env_elevation() const noexcept { return _env_light.elevation(); }
    [[nodiscard]] float env_turbidity() const noexcept { return _env_light.turbidity(); }
    [[nodiscard]] bool has_procedural_sky() const noexcept { return _env_light.is_procedural(); }

private:
    //==========================================================================
    // Internal Helpers
    //==========================================================================

    /**
     * @brief Scan geometry for emissive triangles
     */
    void _collect_emissive_tris(
        const scene::Geometry &geometry,
        const MaterialPool &material_pool) noexcept;

    /**
     * @brief Build alias table from collected triangles
     */
    void _build_alias_table(Stream &stream) noexcept;

    /**
     * @brief Upload light data to GPU
     */
    void _upload_to_gpu(Stream &stream) noexcept;

    /**
     * @brief Calculate triangle area
     */
    [[nodiscard]] float _triangle_area(
        const scene::MeshShape &shape,
        uint primitive_id) const noexcept;

    /**
     * @brief Get triangle vertex positions (world space)
     */
    void _get_triangle_vertices(
        const scene::MeshShape &shape,
        uint primitive_id,
        luisa::float3 &v0,
        luisa::float3 &v1,
        luisa::float3 &v2) const noexcept;

    /**
     * @brief Sample point on triangle (helper for ReSTIR)
     */
    [[nodiscard]] static Float3 _sample_triangle(
        const Float3 &p0, const Float3 &p1, const Float3 &p2,
        const Float2 &u) noexcept;
};

//==============================================================================
// Legacy/Compatibility Types
//==============================================================================

/**
 * @brief Result of sampling a light source (legacy compatibility)
 *
 * Simplified to minimal for testing - just PDF for now.
 * @deprecated Use LightSampler::Sample instead
 */
struct LightSampleResult {
    float pdf{0.0f};
};

/**
 * @brief Emissive triangle data for light sampling (legacy compatibility)
 *
 * @deprecated Use LightSampler::TriangleLight instead
 */
struct EmissiveTriangle {
    luisa::uint v0{0u}, v1{0u}, v2{0u};
    luisa::float3 emission{0.0f};
};

} // namespace render
} // namespace newtype

//==============================================================================
// DSL Struct Registration
//==============================================================================

LUISA_STRUCT(newtype::render::LightSampler::TriangleLight,
              instance_id, primitive_id, area, pdf, nx, ny, nz, ex, ey, ez) {

    [[nodiscard]] auto is_valid() const noexcept {
        return instance_id != ~0u;
    }

    // Compose the precomputed world-space light normal on demand.
    [[nodiscard]] auto normal() const noexcept {
        return luisa::compute::make_float3(nx, ny, nz);
    }

    // Compose the baked emissive color on demand.
    [[nodiscard]] auto emission() const noexcept {
        return luisa::compute::make_float3(ex, ey, ez);
    }
};

LUISA_STRUCT(newtype::render::LightSampler::TriangleVertexData,
              v0, v1, v2/*, _padding*/) { };

LUISA_STRUCT(newtype::render::LightSampler::Sample,
              triangle_index, uv, pdf) {};

LUISA_STRUCT(newtype::render::AliasEntry,
              alias_index, pdf, triangle_index, padding) {};

LUISA_STRUCT(newtype::render::LightSampleResult, pdf) {};
LUISA_STRUCT(newtype::render::EmissiveTriangle, v0, v1, v2, emission) {};
