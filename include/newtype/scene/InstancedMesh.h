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
 * - CPU: setTransforms(span<float4x4>) — direct batch update
 * - GPU: write to transformBuffer() from compute, then applyGpuTransforms()
 *
 * Usage:
 *   auto proto = pipeline.addPrototype(std::move(sphere));
 *   auto instMesh = InstancedMesh::create(device, pipeline.getPrototype(proto), 400);
 *   pipeline.addPrototypeInstances(proto, transforms, ids);
 *   // Per-frame:
 *   instMesh->setTransforms(newTransforms);
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
    void setTransforms(luisa::span<const luisa::float4x4> transforms) noexcept;

    // --- Set a single instance transform ---
    void setTransform(uint index, const luisa::float4x4 &transform) noexcept;

    // --- GPU transform buffer (write from compute shader) ---
    [[nodiscard]] luisa::compute::Buffer<luisa::float4x4> &transformBuffer() noexcept { return _gpuTransforms; }
    [[nodiscard]] const luisa::compute::Buffer<luisa::float4x4> &transformBuffer() const noexcept { return _gpuTransforms; }

    /// Read back GPU buffer to CPU mirror, marks dirty. Call after compute shader finishes.
    void applyGpuTransforms(luisa::compute::Stream &stream) noexcept;

    // --- Apply dirty transforms to TLAS via Geometry ---
    // Called internally by Geometry::update() or manually
    void flushTransformUpdates(Geometry &geom) noexcept;

    // --- Accessors ---
    [[nodiscard]] uint count() const noexcept { return _count; }
    [[nodiscard]] MeshShape *prototype() const noexcept { return _prototype; }
    [[nodiscard]] ShapeId instanceId(uint index) const noexcept { return _instanceIds[index]; }
    [[nodiscard]] bool dirty() const noexcept { return _dirty; }

    /// Set the ShapeId for an instance (called by Geometry during add_instance)
    void setInstanceId(uint index, ShapeId id) noexcept { _instanceIds[index] = id; }

private:
    luisa::compute::Device &_device;
    MeshShape *_prototype;                              // Non-owning
    uint _count;

    luisa::vector<luisa::float4x4> _cpuTransforms;     // CPU mirror
    luisa::compute::Buffer<luisa::float4x4> _gpuTransforms; // GPU buffer
    luisa::vector<ShapeId> _instanceIds;                // Per-instance ShapeIds
    bool _dirty = false;
};

} // namespace newtype::scene
