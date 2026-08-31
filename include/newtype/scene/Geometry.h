//
// Created by Claude on 2026/03/24.
//

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/runtime/rtx/accel.h>
#include "newtype/scene/Shape.h"
#include "newtype/scene/Transform.h"
#include "newtype/scene/Interaction.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/DeformableMesh.h"

namespace newtype::scene {

// Forward declarations
class LightShape;
class ProceduralGeometry;

// Stable shape handle — survives removal/reordering via slot map indirection
using ShapeId = uint;
static constexpr ShapeId kInvalidShapeId = ~0u;

} // namespace newtype::scene

namespace newtype::render {
class MaterialPool;
class LightSampler;
}

namespace newtype::scene {

using namespace luisa;
using compute::Accel;
using compute::Buffer;
using compute::Device;
using compute::Expr;
using compute::Float4x4;
using compute::Mesh;
using compute::Ray;
using compute::Stream;
using compute::Var;

// Forward declarations
using TriangleHit = luisa::compute::TriangleHit;

/**
 * @brief Slot map entry: maps stable ShapeId → current TLAS instance index
 */
struct ShapeSlot {
    uint tlas_index;   // Dense TLAS instance index (changes on swap-and-pop)
    bool alive;        // false = slot was freed
};

/**
 * @brief Mesh geometry with buffer IDs
 *
 * Stores the LuisaCompute Mesh resource and the bindless
 * array base index for GPU access.
 */
/*
struct MeshGeometry {
    Mesh *resource;           // LuisaCompute Mesh (BLAS)
    uint buffer_id_base;      // Bindless array base index
};*/

/**
 * @brief Prototype mesh data (shared BLAS for instancing)
 *
 * A prototype is a mesh that is NOT directly added to the TLAS.
 * Instead, multiple lightweight instances reference the same BLAS.
 * The prototype owns the vertex/triangle buffers and BLAS.
 */
struct PrototypeData {
    luisa::unique_ptr<MeshShape> shape;
    uint vertex_bindless_slot   = ~0u;
    uint triangle_bindless_slot = ~0u;
};

/**
 * @brief Per-instance data
 *
 * CPU-side data for each TLAS instance.
 * Regular instances own their shape via unique_ptr.
 * Prototype instances reference a prototype's shape via non-owning pointer.
 */
struct InstanceData {
    luisa::unique_ptr<MeshShape> shape;  // Owning pointer (regular shapes)
    MeshShape *shape_ref = nullptr;      // Non-owning reference (prototype instances)

    // Transform owned by the pipeline (owning add_shape overload). Registered
    // in _instanced_transforms unconditionally and polled every frame, so
    // later mutation propagates on the next update(). Borrowed transforms
    // (raw-pointer overload) stay owned by the caller instead.
    luisa::unique_ptr<Transform> _owned_transform;

    uint instance_id;                   // ShapeId (stable, NOT tlas index)
    uint properties;                    // Property flags
    float shadow_terminator;            // Shadow terminator factor
    float intersection_offset;
    bool visible = true;                // Visibility state (TLAS visibility_mask)

    // Per-instance data (authoritative — used by _update_instance_buffer and TLAS ops)
    luisa::float4x4 _transform       = luisa::make_float4x4(1.0f);
    uint            _bindless_vert    = ~0u;
    uint            _bindless_tri     = ~0u;
    uint32_t        _material_layers  = 0u;

    /// Get shape pointer (owned or referenced)
    [[nodiscard]] MeshShape *get_shape() const noexcept {
        return shape ? shape.get() : shape_ref;
    }
    /// True if this instance owns its shape (not a prototype instance)
    [[nodiscard]] bool owns_shape() const noexcept { return shape != nullptr; }
};

class Geometry;
typedef luisa::unique_ptr<Geometry> GeomPtr;

/**
 * @brief Geometry system
 *
 * Manages all scene geometry:
 * - TLAS (top-level acceleration structure)
 * - Shape instances with transform hierarchy
 * - Transform updates (dirty flag tracking via TransformTree)
 * - Ray tracing queries
 */
class Geometry {
public:
    using SurfaceCandidate = compute::SurfaceCandidate;

private:
    Device          &_device;
    Accel           _tlas;
    TransformTree   _transform_tree;   // Hierarchical transforms

    // Instance tracking
    luisa::vector<InstanceData> _instances;
    luisa::vector<luisa::uint4> _instance_buffer_cpu;  // Packed instance data for GPU
    Buffer<luisa::uint4>        _instance_buffer;      // GPU buffer

    // Instance transform buffer (for normal transformation in shaders)
    luisa::vector<luisa::float4x4> _instance_transform_cpu;  // 4×4 transforms for GPU
    Buffer<luisa::float4x4>      _instance_transform_buffer; // GPU buffer
    // Previous-frame transforms for object motion vectors: snapshotted from
    // the GPU current buffer before dirty transforms upload, so the G-buffer
    // can reproject through the prior instance transform (moving geometry
    // with a static camera otherwise produces zero motion).
    Buffer<luisa::float4x4>      _instance_transform_prev_buffer;

    // Instanced transforms for updates (pairs instance_id with transform node)
    luisa::vector<InstancedTransform> _instanced_transforms;

    // Slot map: stable ShapeId ↔ dense TLAS index mapping
    luisa::vector<ShapeSlot> _slots;          // ShapeId → TLAS index
    luisa::vector<uint>      _tlas_to_shape;  // TLAS index → ShapeId (reverse)
    luisa::vector<uint>      _free_slots;     // Reusable ShapeIds from removals
    bool _built = false;                      // True after build() called

    // Dirty tracking for per-frame update coalescing
    bool _visibility_dirty = false;
    bool _transform_dirty = false;
    bool _instance_props_dirty = false;       // material layers / bindless slots changed
    bool _instance_props_structure_changed = false; // full CPU rebuild needed (add/remove)
    bool _instance_transforms_dirty = false;  // any transform changed
    luisa::vector<uint> _dirty_transform_indices; // which instance indices changed since last upload
    // True after a dirty upload left prev != curr: motion has stopped, so the
    // next clean frame must re-snapshot prev ← curr once, or the G-buffer
    // keeps emitting the last motion step forever (ghost flow after stopping).
    bool _transform_prev_stale = false;
    bool _lights_dirty = false;                // topology changed (add/remove light) → full rebuild
    bool _lights_visibility_dirty = false;     // light visibility toggled → update_weights only
    bool _tlas_needs_rebuild = false;   // True when topology changes (add/remove)
    bool _bindless_update_needed = false; // True when bindless array was recreated
    bool _pending_blas_build = false;    // True when new shapes need BLAS build

    // Light transform dirty tracking (set when light shape transforms change)
    bool _lightTransformDirty = false;   // Light vertex positions need re-upload
    bool _lightScaleDirty = false;       // Light areas/powers need recompute

    // Glass visibility tracking (recomputed on scene topology / visibility changes)
    const newtype::render::MaterialPool* _material_pool = nullptr;
    bool _has_visible_glass = false;
    bool _has_visible_glass_dirty = false;  // set by set_visibility/set_material_layers; recompute deferred to update()

    // Active-Subsurface tracking. True when any visible instance has a layer-0
    // material of type Subsurface (6). Gates the SSS probe dispatch so scenes
    // without SSS pay zero cost. Recomputed alongside glass detection.
    bool _has_active_subsurface = false;
    bool _has_active_subsurface_dirty = false;

    // Shadow-ray mode selection: true if any visible instance has a material
    // that requires transparent/cutout shadow handling (type 3/5/11 or alphacut>0).
    // When false, the shade shader can use intersect_any (any-hit) instead of
    // the full closest-hit transparent shadow loop.
    bool _has_transparent_shadow_casters = false;

    // Scene bounds
    luisa::float3 _world_min;
    luisa::float3 _world_max;
    uint _triangle_count{0};

    // Light shape indices into _instances (for emissive triangle enumeration)
    luisa::vector<uint> _light_instance_indices;

    // O(1) lookup: is tlas_index a light instance?
    luisa::vector<bool> _is_light_instance;

    // Prototype meshes (shared BLAS for instancing)
    luisa::vector<PrototypeData> _prototypes;

    // Vertex bindless array for per-instance vertex access in shaders
    compute::BindlessArray _vertex_bindless;

    // Procedural primitive geometry (non-owning, set by Pipeline)
    ProceduralGeometry* _procGeom = nullptr;

private:
    void _update_instance_props_buffer() noexcept;
    void _recompute_has_visible_glass() noexcept;

    /// Shared body of the two public add_shape overloads. `borrowed` is the
    /// raw pointer actually used for the initial matrix (owned.get() when the
    /// owning overload was called); `owned` is stored on the instance and
    /// forces unconditional registration in _instanced_transforms.
    ShapeId _add_shape_impl(luisa::unique_ptr<MeshShape> shape, const Transform *borrowed,
                            luisa::unique_ptr<Transform> owned,
                            float shadow_terminator,
                            float intersection_offset) noexcept;

    /// Recreate _vertex_bindless from scratch and re-assign every prototype's
    /// and instance's bindless slots. Caller is responsible for flushing via
    /// `_vertex_bindless.update()` (build) or setting `_bindless_update_needed`
    /// for deferred flush in update(). Used by build(), add_shape post-build,
    /// and add_prototype post-build so the three paths cannot drift.
    void _rebuild_bindless_array() noexcept;

public:
    explicit Geometry(Device &device) noexcept;
    static GeomPtr create(Device& device) noexcept {
        return luisa::make_unique<Geometry>(device);
    }

    /// Add a shape to the scene (transfers shape ownership). Returns stable ShapeId.
    /// Can be called before or after build().
    /// `transform` is borrowed: the caller must keep non-static transforms alive
    /// for the shape's lifetime (they are polled every frame).
    ShapeId add_shape(luisa::unique_ptr<MeshShape> shape, const Transform *transform,
                      float shadow_terminator = 0.0f,
                      float intersection_offset = 0.0f) noexcept;

    /// Owning overload: the pipeline stores the transform and polls it every
    /// frame — mutate later via get_transform(id)->set_*() / set_matrix().
    /// nullptr defaults to an identity StaticTransform.
    ShapeId add_shape(luisa::unique_ptr<MeshShape> shape,
                      luisa::unique_ptr<Transform> transform = nullptr,
                      float shadow_terminator = 0.0f,
                      float intersection_offset = 0.0f) noexcept;

    /// Build TLAS from all added shapes. Must be called once before first render.
    void build(Stream &stream) noexcept;

    /// Per-frame update: flush dirty transforms, visibility, deformable meshes.
    /// Call before render each frame.
    bool update(Stream &stream, bool requireTLASRefit, float time) noexcept;

    // --- Runtime manipulation (O(1), call before update()) ---

    /// Update transform for a shape. Fast refit, no TLAS rebuild.
    /// changeHint: pass Change::Affine for rigid transforms to skip LightSampler area rebuild.
    void set_transform(ShapeId id, const float4x4 &matrix,
                       Change changeHint = Change::Scale) noexcept;

    /// Toggle visibility. Fast refit (RT cores skip invisible instances).
    void set_visibility(ShapeId id, bool visible) noexcept;

    /// Update material layers. Requires instance buffer re-upload.
    void set_material_layers(ShapeId id, uint32_t layers) noexcept;

    /// Mark glass-detection for recompute next update(). Needed when a material's
    /// type/override/alphacut is mutated in-place via MaterialPool (UI/JSON path),
    /// bypassing set_visibility/set_material_layers.
    void mark_has_visible_glass_dirty() noexcept { _has_visible_glass_dirty = true; }

    /// Mark active-Subsurface detection for recompute next update(). Same triggers
    /// as glass — material type mutation via MaterialPool bypasses set_*_layers.
    void mark_has_active_subsurface_dirty() noexcept { _has_active_subsurface_dirty = true; }

    /// Remove shape, freeing memory. Swap-and-pop keeps TLAS dense.
    /// Other ShapeIds remain valid. Triggers full TLAS rebuild.
    bool remove_shape(ShapeId id) noexcept;

    // --- Prototype Instancing (shared BLAS) ---

    /// Register a prototype mesh (NOT added to TLAS).
    /// Returns stable ShapeId for use with add_instance().
    /// Must be called before build().
    ShapeId add_prototype(luisa::unique_ptr<MeshShape> prototype) noexcept;

    /// Add a lightweight TLAS instance referencing a prototype's BLAS.
    /// Shares the prototype's vertex/triangle buffers and bindless slots.
    /// Returns stable ShapeId.
    ShapeId add_instance(ShapeId prototype_id, const float4x4 &transform) noexcept;

    /// Batch: add N instances of a prototype with different transforms.
    /// Appends ShapeIds to out_ids.
    void add_instances(ShapeId prototype_id,
                       luisa::span<const float4x4> transforms,
                       luisa::vector<ShapeId> &out_ids) noexcept;

    /// Get prototype MeshShape by ShapeId (returns nullptr if not a prototype)
    [[nodiscard]] MeshShape *get_prototype(ShapeId prototype_id) const noexcept;

    // --- Accessors ---

    /// Get underlying MeshShape (for direct property access)
    [[nodiscard]] MeshShape* get_shape(ShapeId id) noexcept;

    /// Get the transform update route for a shape: the pipeline-owned transform
    /// if the owning add_shape overload was used, else the shape's internal
    /// StaticTransform. Mutations apply on the next update(). (For prototype
    /// instances this returns the shared prototype transform, which is not
    /// polled — use set_transform for those.)
    [[nodiscard]] Transform* get_transform(ShapeId id) noexcept;

    /// Get DeformableMesh if shape is deformable, else nullptr
    [[nodiscard]] DeformableMesh* get_deformable(ShapeId id) noexcept;

    /// Check if ShapeId is valid (alive)
    [[nodiscard]] bool is_valid(ShapeId id) const noexcept;

    /// Check if shape is visible
    [[nodiscard]] bool is_visible(ShapeId id) const noexcept;

    /// True after build() has been called. Pipeline uses this to warn when
    /// callers mutate the scene in ways that are silently ignored or partially
    /// applied post-build (e.g. setProceduralGeometry, custom resolver reg).
    [[nodiscard]] bool is_built() const noexcept { return _built; }
    // Lookup by dense TLAS index — used by LightSampler to mask hidden lights.
    [[nodiscard]] bool is_instance_visible(uint tlas_idx) const noexcept {
        return tlas_idx < _instances.size() && _instances[tlas_idx].visible;
    }

    // --- Dirty state (used by Pipeline::update) ---

    [[nodiscard]] bool props_dirty() const noexcept { return _instance_props_dirty; }
    [[nodiscard]] bool transforms_dirty() const noexcept { return _instance_transforms_dirty; }
    // True when prev-transform buffer needs one final re-sync after motion stopped
    [[nodiscard]] bool transform_prev_stale() const noexcept { return _transform_prev_stale; }
    [[nodiscard]] bool lights_need_rebuild() const noexcept { return _lights_dirty; }
    void clear_lights_dirty() noexcept { _lights_dirty = false; }
    [[nodiscard]] bool lights_visibility_dirty() const noexcept { return _lights_visibility_dirty; }
    void clear_lights_visibility_dirty() noexcept { _lights_visibility_dirty = false; }

    [[nodiscard]] bool light_transforms_dirty() const noexcept { return _lightTransformDirty; }
    [[nodiscard]] bool light_scale_dirty() const noexcept { return _lightScaleDirty; }
    void clear_light_transforms_dirty() noexcept { _lightTransformDirty = false; _lightScaleDirty = false; }

    void upload_instance_props(Stream &stream) noexcept;
    void upload_dirty_transforms(Stream &stream) noexcept;

    /// Accessors
    [[nodiscard]] const Accel& tlas() const noexcept { return _tlas; }
    [[nodiscard]] Accel& tlas() noexcept { return _tlas; }
    [[nodiscard]] const Buffer<luisa::uint4>& instance_buffer() const noexcept { return _instance_buffer; }
    [[nodiscard]] Buffer<luisa::uint4>& instance_buffer() noexcept { return _instance_buffer; }

    /// Access instance transform buffer (for normal transformation in shaders)
    [[nodiscard]] const Buffer<luisa::float4x4>& instance_transform_buffer() const noexcept { return _instance_transform_buffer; }
    [[nodiscard]] Buffer<luisa::float4x4>& instance_transform_buffer() noexcept { return _instance_transform_buffer; }
    [[nodiscard]] const Buffer<luisa::float4x4>& instance_transform_prev_buffer() const noexcept { return _instance_transform_prev_buffer; }
    [[nodiscard]] const luisa::vector<InstanceData>& instances() const noexcept { return _instances; }
    [[nodiscard]] const luisa::vector<uint>& light_indices() const noexcept { return _light_instance_indices; }
    [[nodiscard]] auto world_min() const noexcept { return _world_min; }
    [[nodiscard]] auto world_max() const noexcept { return _world_max; }
    [[nodiscard]] auto triangle_count() const noexcept { return _triangle_count; }

    /// Check if geometry has any emissive lights
    [[nodiscard]] bool has_lights() const noexcept { return !_light_instance_indices.empty(); }

    /// Check if any visible instance uses a glass (Dielectric/ThinDielectric) material
    [[nodiscard]] bool has_visible_glass() const noexcept { return _has_visible_glass; }

    /// Check if any visible instance has a layer-0 Subsurface material (type 6).
    /// Gates the SSS probe dispatch — false means the probe is skipped entirely.
    [[nodiscard]] bool has_active_subsurface() const noexcept { return _has_active_subsurface; }

    /// Check if any visible instance has a material requiring transparent/cutout
    /// shadow handling. When false, shade shadows can use the any-hit fast path.
    [[nodiscard]] bool has_transparent_shadow_casters() const noexcept { return _has_transparent_shadow_casters; }

    /// Set material pool reference (called by Pipeline before build)
    void set_material_pool(const newtype::render::MaterialPool* pool) noexcept { _material_pool = pool; }

    /// Get vertex bindless array for shader-side vertex/triangle buffer access
    [[nodiscard]] const compute::BindlessArray& vertex_bindless() const noexcept { return _vertex_bindless; }
    [[nodiscard]] compute::BindlessArray& vertex_bindless() noexcept { return _vertex_bindless; }

    /// Set procedural geometry (called by Pipeline after build)
    void set_procedural_geometry(ProceduralGeometry* procGeom) noexcept { _procGeom = procGeom; }
    [[nodiscard]] ProceduralGeometry* procedural_geometry() const noexcept { return _procGeom; }

    /// Hit processing methods are implemented in DSL kernels (PathTracer, etc.)
    /// The instance_buffer provides the data needed for shader-side hit processing.
};

} // namespace newtype::scene
