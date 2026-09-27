//
// Created by Claude on 2026/03/24.
//

#pragma once

#include <luisa/luisa-compute.h>
#include <span>
#include "newtype/scene/MeshShape.h"

namespace newtype::scene {

using namespace luisa;
using compute::Buffer;
using compute::Device;
using compute::Float;
using compute::Mesh;
using compute::Shader;
using compute::Stream;
using compute::UInt;

class DeformableMesh;
typedef luisa::unique_ptr<DeformableMesh> DeformMeshPtr;

/**
 * @brief Deformable mesh with double-buffered vertex updates
 *
 * Extends MeshShape to support dynamic vertex deformation (e.g., skinning,
 * procedural animation, physics) with zero-stall updates via double buffering.
 *
 * The class inherits from MeshShape, so it can be used anywhere a MeshShape
 * is expected. The deformable() method returns true, and Geometry will
 * automatically handle BLAS updates for this type.
 *
 * Usage:
 * ```cpp
 * auto mesh = make_unique<DeformableMesh>(device, material_id);
 * mesh->load_from(triMesh);
 * mesh->build(stream);
 *
 * // Set deformation shader
 * auto shader = device.compile<1>([&](BufferVar<Vertex> v, UInt n, Float t) {
 *     set_block_size(256u);
 *     UInt i = dispatch_x();
 *     $if (i < n) {
 *         Var<Vertex> vert = v.read(i);
 *         vert.pz += sin(vert.px * 2.0f + t) * 0.1f;
 *         v.write(i, vert);
 *     };
 * });
 * mesh->set_deformation_shader(std::move(shader));
 *
 * geometry->add_shape(mesh.get(), transform);
 *
 * // Each frame:
 * geometry->update(stream, time);
 * ```
 */
class DeformableMesh : public MeshShape {
public:
    using Vertex = MeshShape::Vertex;

    /// Deformation shader function signature
    //using DeformationShader = Shader<1, Buffer<Vertex>, UInt, Float>;

private:
    // Double buffering for zero-stall updates (separate from base MeshShape buffers)
    struct DeformableFrame {
        Buffer<GpuVertex> vertex_buffer;
        luisa::unique_ptr<Mesh> blas;
    };

    std::array<DeformableFrame, 2> _frames;
    uint _current_frame = 0;
    bool _need_upload = false, _require_double_buffer;
    // Set by update_cpu() / _upload_and_rebuild() so the next update(stream)
    // returns true and Geometry propagates the new BLAS / vertex buffer to the
    // TLAS instance and bindless array. Without this, Geometry's deformable
    // branch sees update(stream)==false and skips the TLAS/bindless refresh,
    // leaving them pointing at the previous frame's resources.
    bool _cpu_updated = false;

public:
    DeformableMesh(Device &device, bool requireDoubleBuffer = true, uint material_id = 0) noexcept;
    ~DeformableMesh() = default;

    static DeformMeshPtr create(Device& device, bool requireDoubleBuffer = true, uint material_id = 0) noexcept {
        return luisa::make_unique<DeformableMesh>(device, requireDoubleBuffer, material_id);
    }

    // Shape interface override
    [[nodiscard]] bool deformable() const noexcept override { return true; }

    /// Override: returns the current frame's animated vertex buffer (not the static original)
    [[nodiscard]] const Buffer<GpuVertex>& vertex_buffer() const noexcept override {
        return _frames[_current_frame].vertex_buffer;
    }

    /// Get the vertex buffers for GPU update
    [[nodiscard]] const Buffer<GpuVertex>& curr_vertex_buffer() const noexcept {
        return _frames[_current_frame].vertex_buffer;
    }

    [[nodiscard]] const Buffer<GpuVertex>& next_vertex_buffer() const noexcept {
        return _frames[_require_double_buffer ? (1 - _current_frame) : _current_frame].vertex_buffer;
    }

    /// Get the active deformable BLAS (override from MeshShape::mesh_resource())
    [[nodiscard]] Mesh* mesh_resource() noexcept override {
        return _frames[_current_frame].blas.get(); 
    }

    [[nodiscard]] const Mesh* mesh_resource() const noexcept override {
        return _frames[_current_frame].blas.get();
    }

    void build(luisa::compute::Stream& stream) noexcept override;
    void mark_buffer_dirty() { _need_upload = true; }
    
    /**
     * @brief Update vertices and swap double buffers
     *
     * This performs the vertex update and prepares the next frame's buffers
     * while the current frame's BLAS may still be in use by the GPU.
     *
     * @param stream Command stream for uploads/compute dispatch
     * @return The Mesh* BLAS that should be used in this frame's TLAS
     */
    const bool update(Stream &stream ) noexcept;

    /**
     * @brief CPU-side vertex update (callback-based)
     *
     * Use this when you don't have a GPU deformation shader.
     * The callback receives the vertex array and should modify positions.
     *
     * Example:
     * ```cpp
     * mesh->update_cpu([](span<Vertex> verts, float time) {
     *     for (auto &v : verts) {
     *         v.py += sin(v.px + time) * 0.1f;
     *     }
     * }, stream, time);
     * ```
     *
     * @param cpu_deform_fn Function to modify vertices on CPU
     * @param stream Command stream for upload
     * @param time Current time
     * @return The Mesh* BLAS for this frame
     */
    template<typename Fn>
    const Mesh* update_cpu(Fn &&cpu_deform_fn, Stream &stream, float time) noexcept {
        // Apply CPU deformation to cache
        cpu_deform_fn(std::span<Vertex>(_vertices), time);

        // Upload to next buffer and rebuild BLAS
        return _upload_and_rebuild(stream);
    }

private:
    // Upload vertex cache to GPU and rebuild BLAS
    const Mesh* _upload_and_rebuild(Stream &stream) noexcept;
};

/**
 * @brief Helper to create a DeformableMesh
 */
inline luisa::unique_ptr<DeformableMesh> make_deformable_mesh(
    Device &device, uint material_id = 0) noexcept {
    // Explicit bool: the ctor's second parameter is requireDoubleBuffer —
    // passing material_id positionally bound it to the bool (material stayed
    // 0; caught by the TransformTreeTest churn material self-check).
    return luisa::make_unique<DeformableMesh>(device, true, material_id);
}

} // namespace newtype::scene
