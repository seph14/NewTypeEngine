#pragma once

#include <luisa/luisa-compute.h>
#include "newtype/scene/Geometry.h"

namespace newtype::scene {

/**
 * @brief Instanced mesh: N TLAS instances sharing one prototype BLAS
 *
 * Efficient instancing for ~hundreds of identical meshes with per-instance
 * transforms. One BLAS, one vertex buffer, N TLAS entries.
 *
 * Transform updates:
 * - CPU: set_transforms(span<float4x4>) — direct batch update
 * - GPU: write to transform_buffer() from compute, then apply_gpu_transforms()
 *
 * Meshes created via Pipeline::createInstancedMesh are flushed automatically
 * by Pipeline::update() (flush_transform_updates); meshes created directly
 * against a Geometry must be flushed manually before Geometry::update().
 *
 * Usage:
 *   auto proto = pipeline.addPrototype(std::move(sphere));
 *   auto instMesh = pipeline.createInstancedMesh(proto, 400);
 *   luisa::vector<ShapeId> ids;
 *   pipeline.addPrototypeInstances(proto, transforms, ids);
 *   for (uint i = 0; i < ids.size(); ++i) instMesh->set_instance_id(i, ids[i]);
 *   // Per-frame:
 *   instMesh->set_transforms(newTransforms);
 */
class InstancedMesh {
public:
    InstancedMesh(luisa::compute::Device &device,
                  MeshShape *prototype,
                  uint instance_count) noexcept;

    static luisa::unique_ptr<InstancedMesh> create(
        luisa::compute::Device &device,
        MeshShape *prototype,
        uint instance_count) noexcept {
        return luisa::make_unique<InstancedMesh>(device, prototype, instance_count);
    }

    // --- Batch transform update (CPU array, marks dirty) ---
    void set_transforms(luisa::span<const luisa::float4x4> transforms) noexcept;

    // --- Set a single instance transform ---
    void set_transform(uint index, const luisa::float4x4 &transform) noexcept;

    /// Parent transform for the whole batch (optional). When set, the
    /// per-instance matrices are interpreted as LOCAL to the parent:
    /// flush_transform_updates() composes parent->matrix() * local for each
    /// instance. Animate the parent directly (set_local_* / set_*) and the
    /// entire batch follows. Caller keeps the parent transform alive.
    void setParentTransform(Transform *parent) noexcept { _parentTransform = parent; }
    [[nodiscard]] Transform *parentTransform() const noexcept { return _parentTransform; }

    // --- GPU transform buffer (write from compute shader) ---
    [[nodiscard]] luisa::compute::Buffer<luisa::float4x4> &transform_buffer() noexcept { return _gpuTransforms; }
    [[nodiscard]] const luisa::compute::Buffer<luisa::float4x4> &transform_buffer() const noexcept { return _gpuTransforms; }

    /// Read back GPU buffer to CPU mirror, marks dirty. Call after compute shader finishes.
    void apply_gpu_transforms(luisa::compute::Stream &stream) noexcept;

    // --- Per-instance material layers (applied on the next flush) ---

    /// Queue a material-layers swap (4 × 8-bit indices) for one instance.
    /// Applied via Geometry::set_material_layers on the next
    /// flush_transform_updates (automatic for meshes created via
    /// Pipeline::createInstancedMesh).
    void set_material(uint index, uint32_t layers) noexcept;

    /// Queue material layers for every instance at once (span size must
    /// equal count(); entries may be kInheritMaterialLayers to leave that
    /// instance unchanged).
    void set_materials(luisa::span<const uint32_t> layers) noexcept;

    // --- Per-instance custom data (track B2; applied on the next flush) ---

    /// Queue one float4 of an instance's 64 B user-params row (slot 0..3,
    /// readable shader-side via instance_params). Applied via
    /// Geometry::set_instance_user_param on the next flush_transform_updates
    /// (independent of transform dirty, like set_material).
    void set_user_param(uint index, uint slot, luisa::float4 value) noexcept;

    // --- Apply dirty transforms to TLAS via Geometry ---
    // Automatic for meshes created via Pipeline::createInstancedMesh (called
    // by Pipeline::update before Geometry::update); call manually otherwise.
    void flush_transform_updates(Geometry &geom) noexcept;

    // --- Accessors ---
    [[nodiscard]] uint count() const noexcept { return _count; }
    [[nodiscard]] MeshShape *prototype() const noexcept { return _prototype; }
    [[nodiscard]] ShapeId instanceId(uint index) const noexcept { return _instanceIds[index]; }
    [[nodiscard]] bool dirty() const noexcept { return _dirty; }

    /// Wire the ShapeIds returned by addPrototypeInstances to this batch's
    /// slots (one call per instance) — the ids drive flush_transform_updates.
    void set_instance_id(uint index, ShapeId id) noexcept { _instanceIds[index] = id; }

private:
    luisa::compute::Device &_device;
    MeshShape *_prototype;                              // Non-owning
    uint _count;

    luisa::vector<luisa::float4x4> _cpuTransforms;     // CPU mirror
    luisa::compute::Buffer<luisa::float4x4> _gpuTransforms; // GPU buffer
    luisa::vector<ShapeId> _instanceIds;               // Per-instance ShapeIds
    // Queued per-instance layer swaps (kInheritMaterialLayers = nothing
    // pending); applied and cleared in flush_transform_updates.
    luisa::vector<uint32_t> _pendingMaterials;
    // Queued per-instance user-param writes (track B2); applied and cleared
    // in flush_transform_updates alongside _pendingMaterials.
    struct PendingUserParam { uint index; uint slot; luisa::float4 value; };
    luisa::vector<PendingUserParam> _pendingUserParams;
    Transform *_parentTransform = nullptr;             // Optional batch parent (non-owning)
    bool _dirty = false;
};

} // namespace newtype::scene
