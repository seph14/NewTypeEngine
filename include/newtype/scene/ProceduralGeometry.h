#pragma once

#include <newtype/core/Config.h>

#if !NT_ENABLE_PROCEDURAL
// Empty stub — all code using ProceduralGeometry is gated by NT_ENABLE_PROCEDURAL
#else

#include <luisa/luisa-compute.h>
#include <luisa/runtime/rtx/accel.h>
#include "cinder/TriMesh.h"

// Forward declare struct at global scope for LUISA_STRUCT
namespace newtype::scene {
struct ProcInstanceData;
struct ProcDeformState;
struct ProcMeshMeta;
class Geometry;
}

/// Per-instance deformation state for type=3 (deformable static mesh)
struct newtype::scene::ProcDeformState {
    luisa::float4 params[4]; // 64 bytes generic user data
};
LUISA_STRUCT(newtype::scene::ProcDeformState, params) {};

/// Per-topology metadata for GPU lookup (VAT + deform)
struct newtype::scene::ProcMeshMeta {
    uint all_data_base;   // offset into _vatAllPositions/_vatAllNormals (VAT only, 0 for deform)
    uint vertex_count;
    uint uv_base;         // offset into _procUVs
};
LUISA_STRUCT(newtype::scene::ProcMeshMeta, all_data_base, vertex_count, uv_base) {};

/// Per procedural instance data (GPU layout, 48 bytes, 16-byte aligned)
struct newtype::scene::ProcInstanceData {
    uint  type;              // 0=VAT mesh, 1=sphere, 2=cube, 3=deformable static
    uint  material_layers;   // Same 4x8bit packing as mesh instances
    uint  mesh_id;           // VAT: mesh index; Builtin: unused
    uint  packed_offsets;    // VAT: hi16=index_offset, lo16=position_offset

    float param;             // VAT: current_frame; Cube: half_extent; Sphere: unused
    uint  vertex_count;      // VAT only
    uint  tri_count;         // VAT only
    uint  frame_count;       // VAT only

    luisa::float4 rotation;  // Quaternion (x,y,z,w) — identity = (0,0,0,1)
};

// LUISA_STRUCT registration (must be at global scope)
LUISA_STRUCT(newtype::scene::ProcInstanceData, 
             type, material_layers, mesh_id, packed_offsets,
             param, vertex_count, tri_count, frame_count,
             rotation) {};

/// Pack two 16-bit offsets into a single uint (CPU-side helper)
[[nodiscard]] inline uint pack_proc_offsets(uint index_offset, uint position_offset) noexcept {
    return (index_offset << 16u) | (position_offset & 0xFFFFu);
}

/// Double-sided flag packed into ProcInstanceData.type upper bits
static constexpr uint kProcDoubleSided = 1u << 31u;

namespace newtype::scene {

using namespace luisa;
using compute::Accel;
using compute::Buffer;
using compute::Device;
using compute::ProceduralPrimitive;
using compute::Stream;

using ProcGeomPtr = luisa::unique_ptr<class ProceduralGeometry>;

/**
 * @brief Procedural primitive geometry system
 *
 * Manages procedural primitives via LuisaCompute's ProceduralPrimitive API.
 * Each instance is represented by an AABB in a shared BLAS. During ray
 * traversal, the `on_procedural_candidate` handler performs custom
 * intersection (e.g., VAT mesh ray-triangle test with frame interpolation).
 *
 * Lifecycle:
 *   1. add_vat_mesh() — register VAT topology (positions, normals, indices)
 *   2. add_instance() — add a procedural instance referencing a VAT mesh
 *   3. build() — create BLAS, upload AABBs, build acceleration structure
 *   4. update() — per-frame AABB update + BLAS refit
 */
class ProceduralGeometry {
public:
    using AABB = compute::AABB;

private:
    Device& _device;

    // BLAS for all procedural instances (one ProceduralPrimitive)
    ProceduralPrimitive _blas;
    Buffer<AABB> _aabbBuffer;
    Buffer<ProcInstanceData> _instanceBuffer;
    uint _instanceCount = 0;

    // CPU mirror of instance data
    luisa::vector<ProcInstanceData> _instances_cpu;

    // Dirty tracking for partial uploads
    luisa::vector<uint> _dirty_proc_indices;
    luisa::vector<uint> _dirtyDeformIndices;
    luisa::vector<uint> _dirtyVatIndices;
    bool _has_vat = false;  // Cached: true if any VAT instance exists
    bool _has_deform = false; // Cached: true if any type=3 instance exists

    // Per-instance animation state (CPU only, VAT only)
    struct AnimState {
        float fps         = 24.0f;
        float time_offset = 0.0f;
        float speed       = 1.0f;
        bool  looping     = true;
    };
    luisa::vector<AnimState> _animState;

    // CPU mirror of deform state (indexed by deform instance count, not global instance idx)
    luisa::vector<ProcDeformState> _deformStateCPU;

    // Per-instance CPU AABBs for builtins (indexed by instance_id)
    luisa::vector<compute::AABB> _builtinAabbs;

    // VAT mesh registry (shared topology across instances)
    struct VATMeshRegistry {
        luisa::vector<luisa::float3> positions_cpu; // kept until build()
        luisa::vector<luisa::float3> normals_cpu;
        luisa::vector<luisa::float2> uvs_cpu;       // [vertex_count] static UVs
        luisa::vector<compute::Triangle> indices_cpu;
        uint vertex_count;
        uint frame_count;
        uint tri_count;
        uint position_base;  // Global offset into combined buffers (set during build)
        uint normal_base;
        uint index_base;
        uint uv_base;        // Global offset into _procUVs (set during build)
        uint all_data_base;  // Global offset into _vatAllPositions/_vatAllNormals (set during build)
    };
    luisa::vector<VATMeshRegistry> _vatMeshes;

    // Deformable static mesh registry (type=3)
    struct DeformMeshRegistry {
        luisa::vector<luisa::float3> positions_cpu;  // [vertex_count] — base mesh
        luisa::vector<luisa::float3> normals_cpu;
        luisa::vector<luisa::float2> uvs_cpu;        // [vertex_count] static UVs
        luisa::vector<compute::Triangle> indices_cpu;
        uint vertex_count, tri_count;
        uint position_base; // Global offset into combined VAT buffers (set during build)
        uint index_base;
        uint deform_base_pos_offset; // Offset into _deformBasePositions
        uint uv_base;                // Global offset into _procUVs (set during build)
    };
    luisa::vector<DeformMeshRegistry> _deformMeshes;

    Buffer<ProcDeformState> _deformStateBuffer;
    Buffer<luisa::float3> _deformBasePositions;  // Undeformed mesh per topology
    Buffer<luisa::float3> _deformBaseNormals;    // Undeformed normals per topology
    Buffer<luisa::float2> _procUVs;              // Per-topology static UVs (update-only)
    Buffer<uint> _deformBaseOffsets;              // mesh_id → offset in _deformBasePositions
    compute::Shader<1,
        Buffer<luisa::float4>,              // rw: positions (.xyz=pos, .w=uv_u)
        Buffer<luisa::float4>,              // rw: normals   (.xyz=norm, .w=uv_v)
        Buffer<AABB>,                        // w: aabb_buffer
        Buffer<ProcInstanceData>,            // r: instances
        Buffer<ProcDeformState>,             // r: states
        Buffer<luisa::float3>,               // r: base_positions
        Buffer<luisa::float3>,               // r: base_normals
        Buffer<uint>,                        // r: base_offsets
        Buffer<compute::Triangle>,           // r: indices
        Buffer<luisa::float2>,               // r: static_uvs
        Buffer<ProcMeshMeta>                 // r: mesh_meta (uv_base lookup)
    > _deformShader;
    uint _deformInstanceCount = 0u;
    uint _deformStartIdx = 0u;               // first type=3 instance after sort
    luisa::vector<uint> _originalToSorted;   // pre-sort idx → post-sort idx
    luisa::string       _deformShaderId;     // overload deform shader with id for ShaderManager
    uint                _deformShaderBlock;  // overload deform shader with custom block size

    // Source data (all VAT frames, read-only after build)
    Buffer<luisa::float4> _vatAllPositions;  // all frames, .w = uv_u
    Buffer<luisa::float4> _vatAllNormals;    // all frames, .w = uv_v

    // Active data (per-instance, single frame — both VAT + deform)
    Buffer<luisa::float4> _vatPositions;     // .xyz = pos, .w = uv_u
    Buffer<luisa::float4> _vatNormals;       // .xyz = norm, .w = uv_v
    Buffer<compute::Triangle> _vatIndices;

    // Per-topology metadata (GPU-accessible)
    Buffer<ProcMeshMeta> _procMeshMeta;

    // VAT interpolation shader (writes active pos/norm from all-frames + computes AABB)
    Buffer<uint> _vatDispatchBuf;             // VAT instance indices for sparse dispatch
    compute::Shader<2,
        Buffer<uint>,                        // r: dispatch_indices (x-dim lookup)
        Buffer<luisa::float4>,              // rw: active_pos
        Buffer<luisa::float4>,              // rw: active_norm
        Buffer<AABB>,                        // rw: aabbs
        Buffer<ProcInstanceData>,            // r: instances
        Buffer<luisa::float4>,              // r: all_pos (source)
        Buffer<luisa::float4>,              // r: all_norm (source)
        Buffer<ProcMeshMeta>                 // r: mesh_meta
    > _vatInterpolateShader;

    bool _built = false;

public:
    explicit ProceduralGeometry(Device& device) noexcept;
    ~ProceduralGeometry() = default;

    static ProcGeomPtr create(Device& device) noexcept {
        return luisa::make_unique<ProceduralGeometry>(device);
    }

    /// Register a VAT mesh topology. Returns mesh_id for use with add_instance().
    uint add_vat_mesh(
        luisa::span<const luisa::float3> positions,  // [frame_count * vertex_count]
        luisa::span<const luisa::float3> normals,    // [frame_count * vertex_count]
        luisa::span<const compute::Triangle> indices,
        uint vertex_count, uint frame_count,
        luisa::span<const luisa::float2> uvs = {}) noexcept;

    /// Add a procedural instance referencing a VAT mesh.
    /// Optional: fps, time_offset (seconds), speed (multiplier), double_sided.
    /// Returns instance index.
    uint add_instance(uint mesh_id, uint32_t material_layers,
                      float fps = 24.0f, float time_offset = 0.0f,
                      float speed = 1.0f, bool double_sided = false) noexcept;

    /// Add a procedural sphere instance.
    /// Returns instance index.
    uint add_sphere(luisa::float3 center, float radius,
                    uint32_t material_layers,
                    luisa::float4 rotation = luisa::make_float4(0.f, 0.f, 0.f, 1.f),
                    bool double_sided = false) noexcept;

    /// Add a procedural cube instance (uniform half-extent).
    /// Returns instance index.
    uint add_cube(luisa::float3 center, float half_extent,
                  uint32_t material_layers,
                  luisa::float4 rotation = luisa::make_float4(0.f, 0.f, 0.f, 1.f),
                  bool double_sided = false) noexcept;

    /// Convenience: load a single .vat file and register as a VAT mesh.
    /// Returns (mesh_id, vertex_count, frame_count, tri_count).
    struct VATLoadResult { uint mesh_id; uint vertex_count; uint frame_count; uint tri_count; };
    VATLoadResult add_vat_from_file(const std::filesystem::path& path) noexcept;

    /// Override animation state for a specific instance.
    void set_anim_state(uint instance_idx, float fps, float time_offset = 0.0f,
                        float speed = 1.0f, bool looping = true) noexcept;

    /// Build BLAS, upload all data, compile shaders. Call once.
    void build(Stream& stream) noexcept;

    /// Per-frame update: advance animation, recompute AABBs, refit BLAS.
    /// Returns true if any data changed.
    bool update(Stream& stream, float time) noexcept;

    /// Register a static mesh topology for deformable instances. Returns mesh_id.
    uint add_static_mesh(
        luisa::span<const luisa::float3> positions,
        luisa::span<const luisa::float3> normals,
        luisa::span<const compute::Triangle> indices,
        uint vertex_count,
        luisa::span<const luisa::float2> uvs = {}) noexcept;

    /// Convenience: register a static mesh from a Cinder TriMesh. Returns mesh_id.
    uint add_static_mesh(ci::TriMesh& triMesh) noexcept;

    /// Add N deformable instances of a static mesh (type=3).
    /// Returns first instance index.
    uint add_deformable_instances(uint mesh_id, uint count,
                                   uint32_t material_layers,
                                   luisa::span<const ProcDeformState> initial_states = {},
                                   bool double_sided = false) noexcept;

    /// Override deformation state for a specific instance.
    void set_deform_state(uint instance_idx, const ProcDeformState& state) noexcept;
    /// Override deformation shader with a ID in ShaderManager
    void set_deform_shader_id(luisa::string_view shaderId, uint blockSize) noexcept;

    // --- Accessors ---

    [[nodiscard]] ProceduralPrimitive& blas() noexcept { return _blas; }
    [[nodiscard]] Buffer<AABB>& aabb_buffer() noexcept { return _aabbBuffer; }
    [[nodiscard]] Buffer<ProcInstanceData>& instance_buffer() noexcept { return _instanceBuffer; }
    [[nodiscard]] Buffer<luisa::float4>& vat_positions() noexcept { return _vatPositions; }
    [[nodiscard]] Buffer<luisa::float4>& vat_normals() noexcept { return _vatNormals; }
    [[nodiscard]] Buffer<compute::Triangle>& vat_indices() noexcept { return _vatIndices; }
    [[nodiscard]] Buffer<ProcDeformState>& deform_state_buffer() noexcept { return _deformStateBuffer; }
    // Note: _procUVs and _procMeshMeta are NOT exposed via accessors — they are
    // internal to the VAT interpolate + deform shaders and passed as direct
    // kernel parameters via the shader dispatch site, not through bindless slots.
    [[nodiscard]] const luisa::vector<ProcInstanceData>& instances_cpu() const noexcept { return _instances_cpu; }
    [[nodiscard]] uint instance_count() const noexcept { return _instanceCount; }
    [[nodiscard]] bool has_instances() const noexcept { return _instanceCount > 0u; }
    [[nodiscard]] uint deform_start_idx() const noexcept { return _deformStartIdx; }
    [[nodiscard]] uint vat_count() const noexcept { return static_cast<uint>(_vatMeshes.size()); }
    [[nodiscard]] uint deform_count() const noexcept { return static_cast<uint>(_deformMeshes.size()); }
};

} // namespace newtype::scene

#endif // NT_ENABLE_PROCEDURAL
