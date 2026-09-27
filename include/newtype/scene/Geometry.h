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

// Sentinel for add_instance material_layers: inherit the prototype's layers
// (the pre-B1 behavior). Any other value is the instance's own 4 × 8-bit
// layer pack, independent of the prototype and of sibling instances.
static constexpr uint32_t kInheritMaterialLayers = 0xFFFFFFFFu;

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
    bool prototype;    // true = the id names a prototype; tlas_index stores
                       // the prototype index, NOT a TLAS row
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

    // Transform owned by the pipeline (owning add_shape / add_instance
    // overloads). Registered in _instanced_transforms unconditionally and
    // polled every frame, so later mutation propagates on the next update().
    // Borrowed transforms (raw-pointer overloads) stay owned by the caller.
    luisa::unique_ptr<Transform> _owned_transform;
    // Borrowed transform for prototype instances (raw-pointer add_instance
    // overload). Regular shapes do not store theirs: the caller keeps it
    // alive and mutation is polled through _instanced_transforms directly.
    const Transform *_borrowed_transform = nullptr;

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

    // Per-instance custom data (track B2): 4 float4 (64 B) readable by custom
    // material callables via instance_params(tex, instance_index, i). Rows
    // default to zero and live on the instance (so swap-and-pop row moves
    // carry them, like _material_layers); the GPU buffer materializes at the
    // pool only after the first set_instance_user_param (host-authored).
    luisa::float4   _user_params[4]   = {};

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
 * - Shape instances with transform hierarchy (parent/child links on
 *   Transform; dirty flags propagate down the chain)
 * - Transform updates (per-frame polling of registered transforms)
 * - Ray tracing queries
 */
class Geometry {
public:
    using SurfaceCandidate = compute::SurfaceCandidate;

private:
    Device          &_device;
    Accel           _tlas;

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

    // TLAS indices polled for shape-internal transform changes (owning
    // instances only — prototype instances share the prototype's transform,
    // which is not authoritative per instance). Updated on remove/reindex
    // alongside _instanced_transforms.
    luisa::vector<uint> _shape_transform_instances;

    // TLAS indices of deformable meshes, polled per frame in update() step 1.
    // Deformable meshes always own their shape (prototype instances share a
    // static BLAS and can never register); deformable() is fixed at
    // construction, so registration at add time is complete. Maintained on
    // remove/reindex exactly like _shape_transform_instances — a per-frame
    // sweep over ALL instances costs O(N) on scenes whose instances are all
    // prototypes (the tetcage path: ~20 ms/frame at 1.6M instances in Debug).
    luisa::vector<uint> _deformable_instances;

    // Slot map: stable ShapeId ↔ dense TLAS index mapping
    luisa::vector<ShapeSlot> _slots;          // ShapeId → TLAS index
    luisa::vector<uint>      _tlas_to_shape;  // TLAS index → ShapeId (reverse)
    luisa::vector<uint>      _free_slots;     // Reusable ShapeIds from removals
    bool _built = false;                      // True after build() called

    // Dirty tracking for per-frame update coalescing
    bool _visibility_dirty = false;
    bool _transform_dirty = false;    bool _instance_props_dirty = false;       // material layers / bindless slots changed
    bool _instance_props_structure_changed = false; // full CPU rebuild needed (add/remove)
    // Per-instance user-params row upload pending (value edit, or a row
    // reshuffle AFTER the pool-side buffer materialized — before that, rows
    // ride lazily in InstanceData at zero GPU cost).
    bool _instance_params_dirty = false;
    bool _instance_transforms_dirty = false;  // any transform changed
    luisa::vector<uint> _dirty_transform_indices; // which instance indices changed since last upload
    // Indices dirty at the PREVIOUS upload: an instance that stops moving
    // needs one final prev ← curr refresh on the next frame (its prev still
    // lags curr by one step); the full-buffer copy used to fix this as a side
    // effect every moving frame. See upload_dirty_transforms.
    luisa::vector<uint> _last_dirty_transform_indices;
    // True after a dirty upload left prev != curr: motion has stopped, so the
    // next clean frame must re-snapshot prev ← curr once, or the G-buffer
    // keeps emitting the last motion step forever (ghost flow after stopping).
    bool _transform_prev_stale = false;
    bool _lights_dirty = false;                // topology changed (add/remove light) → full rebuild
    bool _lights_visibility_dirty = false;     // light visibility toggled → update_weights only
    bool _tlas_needs_rebuild = false;   // True when topology changes (add/remove)
    bool _bindless_update_needed = false; // True when bindless array was recreated
    bool _pending_blas_build = false;    // True when new shapes need BLAS build

    // GPU-owned transform rows (device-side TLAS transform source, see
    // docs/TLAS-instance-transform-updates.md Part 2). A registered row's
    // current AND previous matrices are written exclusively by a device
    // kernel (e.g. the TetCage solve); CPU uploads skip those rows and
    // set_transform on them is dropped. v1: one contiguous range (the LC
    // API carries a single {first, count} source per build).
    luisa::vector<std::pair<uint, uint>> _gpuTransformRanges; // [first, count)
    uint _topologyGeneration = 0u;  // bumped by post-build add/remove (row reshuffle)
    bool _gpuTransformsDirty = false; // device writer ran; next build carries the source

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

    /// True when tlas_idx belongs to a registered GPU-owned transform range.
    [[nodiscard]] bool _row_is_gpu_owned(uint tlas_idx) const noexcept;

    /// Bump _topologyGeneration and drop registrations (post-build add/remove
    /// reshuffles dense rows; the device writer must re-resolve + re-register).
    void _invalidate_gpu_transform_rows(const char *why) noexcept;

    /// Shared body of the two public add_shape overloads. `borrowed` is the
    /// raw pointer actually used for the initial matrix (owned.get() when the
    /// owning overload was called); `owned` is stored on the instance and
    /// forces unconditional registration in _instanced_transforms.
    ShapeId _add_shape_impl(luisa::unique_ptr<MeshShape> shape, const Transform *borrowed,
                            luisa::unique_ptr<Transform> owned,
                            float shadow_terminator,
                            float intersection_offset) noexcept;

    /// Propagate an instance's new world matrix into every CPU/GPU mirror:
    /// InstanceData::_transform, the CPU transform array + dirty index, the
    /// TLAS-on-update slot, and the light-transform flags. Shared by the
    /// per-frame transform polls and set_instance_transform.
    void _apply_instance_transform(uint tlas_idx, const float4x4 &matrix,
                                   Change change) noexcept;

    /// Shared body of the add_instance overloads. `initial_world` seeds the
    /// instance matrix; when a transform is supplied (borrowed or owned) and
    /// the chain is not fully static, it is registered in
    /// _instanced_transforms and polled every frame like shape transforms.
    /// `material_layers` = kInheritMaterialLayers copies the prototype's
    /// layers; any other value becomes this instance's own layer pack
    /// (validated against the material pool when one is set).
    ShapeId _add_instance_impl(ShapeId prototype_id, const float4x4 &initial_world,
                               const Transform *borrowed,
                               luisa::unique_ptr<Transform> owned,
                               uint32_t material_layers = kInheritMaterialLayers) noexcept;

    /// Warn when any active layer index is out of material-pool range (the
    /// B9 check shared by set_material_layers and the add_instance path).
    void _validate_material_layers(uint32_t layers, const char *ctx) const noexcept;

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
    void set_instance_transform(ShapeId id, const float4x4 &matrix,
                                Change changeHint = Change::Scale) noexcept;

    /// Toggle visibility. Fast refit (RT cores skip invisible instances).
    void set_visibility(ShapeId id, bool visible) noexcept;

    /// Toggle camera-path visibility only (PROPERTY_INVISIBLE_TO_CAMERA).
    /// Camera rays (G-buffer primary, mirror reflections, glass tint replay)
    /// pass through the shape; light sampling, shadow, and GI rays still hit
    /// it. TLAS visibility and light sampling power are untouched — use this
    /// for lights that should illuminate without rendering. Re-uploads the
    /// instance props buffer on the next update().
    void set_camera_visibility(ShapeId id, bool camera_visible) noexcept;

    /// Camera-path visibility state (see set_camera_visibility).
    [[nodiscard]] bool is_camera_visible(ShapeId id) const noexcept;

    /// Update material layers. Requires instance buffer re-upload. Works on
    /// prototype instances (the instance's own layer pack changes; the
    /// prototype and siblings are untouched) and on owning shapes (shape is
    /// kept in sync). Marks the light table dirty — emission is baked per
    /// instance into the LightSampler triangle records.
    void set_material_layers(ShapeId id, uint32_t layers) noexcept;

    /// Per-instance material layers (4 × 8-bit indices; layer 0 = base).
    /// Returns 0 for invalid ids and prototype ids (prototypes have no
    /// TLAS row; read the prototype shape's layers via get_prototype()).
    [[nodiscard]] uint32_t material_layers(ShapeId id) const noexcept;

    // --- Per-instance custom data (track B2; feeds MaterialCallables) ---

    /// Write one float4 slot (0..3) of an instance's 64 B user-params row
    /// (readable shader-side via instance_params). Materializes the pool-side
    /// GPU buffer on the next update — the first authoring makes rows
    /// resident for EVERY instance (zeros elsewhere). Works on prototype
    /// instances and owning shapes; false (no-op) for invalid/prototype ids.
    bool set_instance_user_param(ShapeId id, uint slot, luisa::float4 value) noexcept;

    /// Read back one slot (zeros when never authored; w=0 for invalid ids).
    [[nodiscard]] luisa::float4 instance_user_param(ShapeId id, uint slot) const noexcept;

    /// True when params rows must (re)upload this frame (value edit or row
    /// reshuffle after the buffer materialized). Pipeline flushes via
    /// upload_instance_params alongside upload_instance_props.
    [[nodiscard]] bool instance_params_dirty() const noexcept { return _instance_params_dirty; }

    /// Serialize the dense TLAS-ordered rows (4 float4 per instance) into the
    /// pool's reserved-slot buffer. No-op unless instance_params_dirty().
    void upload_instance_params(newtype::render::MaterialPool& pool,
                                Stream &stream) noexcept;

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

    /// Release the CPU vertex/triangle mirrors of meshes no runtime consumer
    /// needs (A1 system-RAM cut; opt-in, call after buildScene()). Skips
    /// deformables (the CPU cache is the animation source), light-flagged
    /// shapes, and any shape whose own or per-instance layers reference an
    /// emissive material (the LightSampler recomputes triangle areas from
    /// CPU data on rebuild). Note: a mesh swapped to an emissive material
    /// afterwards cannot join the light table (zero area) — keep such meshes
    /// out of this call.
    void unload_static_cpu_data() noexcept;

    // --- GPU-owned transform rows (device-side TLAS transform source) ---

    /// Dense TLAS row of a shape (~0u for invalid ids and prototype ids).
    /// Rows reshuffle on remove_shape swap-and-pop — re-resolve after any
    /// topology change (compare topology_generation()).
    [[nodiscard]] uint tlas_index_of(ShapeId id) const noexcept;

    /// Hand a set of TLAS rows to a device writer: their current+previous
    /// matrices leave CPU ownership (set_transform on them is dropped; CPU
    /// uploads skip them), and after each device write the caller invokes
    /// notify_gpu_transforms_dirty() so the next TLAS build copies the rows
    /// straight from _instance_transform_buffer on the device (LC fork API
    /// Accel::set_transform_buffer_on_update — no host round-trip).
    /// v1 constraint: the ids must resolve to ONE contiguous row run;
    /// returns false (without registering) for holes, duplicates, or ids
    /// that are not TLAS instances. A topology change (post-build
    /// add/remove) clears the registration — re-register on a generation
    /// bump.
    bool register_gpu_transform_rows(luisa::span<const ShapeId> ids) noexcept;

    /// Mark the registered rows as rewritten by the device writer this
    /// frame; Geometry::update wires the buffer into the next TLAS build.
    void notify_gpu_transforms_dirty() noexcept { _gpuTransformsDirty = true; }

    /// Monotonic counter bumped by every post-build add_instance/add_shape/
    /// remove_shape (dense-row reshuffle). Device writers compare it to
    /// detect stale row registrations.
    [[nodiscard]] uint topology_generation() const noexcept { return _topologyGeneration; }

    /// True while a GPU transform registration is active.
    [[nodiscard]] bool has_gpu_transform_rows() const noexcept {
        return !_gpuTransformRanges.empty();
    }

    // --- Prototype Instancing (shared BLAS) ---

    /// Register a prototype mesh (NOT added to TLAS).
    /// Returns stable ShapeId for use with add_instance().
    /// Must be called before build().
    ShapeId add_prototype(luisa::unique_ptr<MeshShape> prototype) noexcept;

    /// Add a lightweight TLAS instance referencing a prototype's BLAS.
    /// Shares the prototype's vertex/triangle buffers and bindless slots.
    /// `material_layers` (B1): per-instance 4 × 8-bit layer pack — instances
    /// of one prototype can render different materials. Default inherits the
    /// prototype's layers. Returns stable ShapeId.
    ShapeId add_instance(ShapeId prototype_id, const float4x4 &transform,
                         uint32_t material_layers = kInheritMaterialLayers) noexcept;

    /// Add a prototype instance with a borrowed transform. The caller must
    /// keep the transform alive for the instance's lifetime. When the
    /// transform (or any ancestor in its parent chain) is not fully static,
    /// it is registered and polled every frame — mutate it directly (e.g.
    /// set_local_position / set_position) and the instance follows.
    ShapeId add_instance(ShapeId prototype_id, const Transform *transform,
                         uint32_t material_layers = kInheritMaterialLayers) noexcept;

    /// Owning overload: the pipeline stores the transform and polls it
    /// every frame — mutate later via get_transform(id)->set_*().
    ShapeId add_instance(ShapeId prototype_id, luisa::unique_ptr<Transform> transform,
                         uint32_t material_layers = kInheritMaterialLayers) noexcept;

    /// Batch: add N instances of a prototype with different transforms.
    /// Appends ShapeIds to out_ids.
    void add_instances(ShapeId prototype_id,
                       luisa::span<const float4x4> transforms,
                       luisa::vector<ShapeId> &out_ids) noexcept;

    /// Batch with per-instance material layers (parallel to `transforms`;
    /// entries may be kInheritMaterialLayers to inherit that instance).
    /// Appends ShapeIds to out_ids.
    void add_instances(ShapeId prototype_id,
                       luisa::span<const float4x4> transforms,
                       luisa::span<const uint32_t> material_layers,
                       luisa::vector<ShapeId> &out_ids) noexcept;

    /// Batch with borrowed transforms (same registration rule as the
    /// single add_instance(Transform*) overload).
    void add_instances(ShapeId prototype_id,
                       luisa::span<const Transform *const> transforms,
                       luisa::vector<ShapeId> &out_ids) noexcept;

    /// Get prototype MeshShape by ShapeId (returns nullptr if not a prototype)
    [[nodiscard]] MeshShape *get_prototype(ShapeId prototype_id) const noexcept;

    // --- Accessors ---

    /// Get underlying MeshShape (for direct property access)
    [[nodiscard]] MeshShape* get_shape(ShapeId id) noexcept;

    /// Get the transform update route for a shape: the pipeline-owned
    /// transform if an owning add_shape/add_instance overload was used, the
    /// stored borrowed transform for prototype instances, else the shape's
    /// internal StaticTransform. Mutations apply on the next update().
    /// Returns nullptr for matrix-only prototype instances (no retained
    /// transform — use set_transform for those).
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
