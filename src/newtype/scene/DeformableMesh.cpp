//
// Created by Claude on 2026/03/24.
//

#include "newtype/scene/DeformableMesh.h"
#include "cinder/Log.h"

namespace newtype::scene {

using namespace luisa;

//==============================================================================
// DeformableMesh
//==============================================================================

DeformableMesh::DeformableMesh(Device &device, bool requireDoubleBuffer, uint material_id) noexcept
    : MeshShape(device, material_id), _require_double_buffer(requireDoubleBuffer) { }

void DeformableMesh::build(Stream& stream) noexcept {
    if (_vertices.empty() || _triangles.empty())
        return;

    // No rest-pose GPU buffer (A1): the base MeshShape _vertex_buffer copy
    // was written once here and never read again — the frame buffers below
    // carry the data (vertex_buffer() returns the current frame), and the
    // original stays available CPU-side in _vertices. Saves one full vertex
    // buffer per deformable.

    // Create triangle buffer
    _triangle_buffer = _device.create_buffer<Triangle>(_numTriangle);
    stream << _triangle_buffer.copy_from(_triangles.data());

    // Create double-buffered vertex buffers and BLAS instances
    int num = _require_double_buffer ? 2 : 1;
    for (int i = 0; i < num; ++i) {
        _frames[i].vertex_buffer = _device.create_buffer<GpuVertex>(_numVertices);
        util::upload_vertex_buffer(_frames[i].vertex_buffer,
                                   std::span<const Vertex>{_vertices}, stream);

        // Create LuisaCompute Mesh (BLAS)
        auto mesh = _device.create_mesh(
            _frames[i].vertex_buffer,
            _triangle_buffer
        );
        _frames[i].blas = luisa::make_unique<Mesh>(std::move(mesh));

        // build acceleration structure
        stream << _frames[i].blas->build();
    }

    _built = true;
}

const Mesh* DeformableMesh::_upload_and_rebuild(Stream &stream) noexcept {
    // Determine next frame index
    uint next_frame = _require_double_buffer ? (1 - _current_frame) : _current_frame;

    // Copy CPU cache to GPU (A2: packs through the active GPU layout)
    util::upload_vertex_buffer(_frames[next_frame].vertex_buffer,
                               std::span<const Vertex>{_vertices}, stream);

    // Rebuild BLAS
    stream << _frames[next_frame].blas->build();

    // Swap to next frame
    _current_frame = next_frame;

    // Signal Geometry that TLAS/bindless need to refresh to the new frame.
    _cpu_updated = true;

    return _frames[_current_frame].blas.get();
}

const bool DeformableMesh::update(Stream &stream) noexcept {
    // Initialize buffers on first call
    if (!_frames[0].blas) {
        CI_LOG_E("DeformableMesh wasn't built");
        return false;
    }

    // CPU-side update path: update_cpu()/_upload_and_rebuild() already did the
    // vertex upload + BLAS rebuild + frame swap. Just propagate to Geometry.
    if (_cpu_updated) {
        _cpu_updated = false;
        return true;
    }

    // If buffer is updated via shader, rebuild BLAS
    if (_need_upload) {
        // Rebuild BLAS
        uint next_frame = _require_double_buffer ? (1 - _current_frame) : _current_frame;
        stream << _frames[next_frame].blas->build();
        _current_frame = next_frame;

        _need_upload = false;
        return true;
    }

    return false;
}

} // namespace newtype::scene
