#include "newtype/scene/InstancedMesh.h"

namespace newtype::scene {

using namespace luisa;

//==============================================================================
// InstancedMesh
//==============================================================================

InstancedMesh::InstancedMesh(compute::Device &device,
                              MeshShape *prototype,
                              uint instance_count) noexcept
    : _device(device),
      _prototype(prototype),
      _count(instance_count),
      _gpuTransforms(device.create_buffer<float4x4>(instance_count)),
      _instanceIds(instance_count, kInvalidShapeId) {

    // Initialize CPU transforms to identity
    _cpuTransforms.resize(instance_count);
    for (uint i = 0u; i < instance_count; ++i) {
        _cpuTransforms[i] = make_float4x4(1.0f);
    }
}

void InstancedMesh::setTransforms(luisa::span<const float4x4> transforms) noexcept {
    if (transforms.size() != _count) return;
    for (uint i = 0u; i < _count; ++i) {
        _cpuTransforms[i] = transforms[i];
    }
    _dirty = true;
}

void InstancedMesh::setTransform(uint index, const float4x4 &transform) noexcept {
    if (index >= _count) return;
    _cpuTransforms[index] = transform;
    _dirty = true;
}

void InstancedMesh::applyGpuTransforms(compute::Stream &stream) noexcept {
    stream << _gpuTransforms.copy_to(luisa::span{_cpuTransforms});
    _dirty = true;
}

void InstancedMesh::flushTransformUpdates(Geometry &geom) noexcept {
    if (!_dirty) return;
    for (uint i = 0u; i < _count; ++i) {
        if (_instanceIds[i] != kInvalidShapeId) {
            geom.set_transform(_instanceIds[i], _cpuTransforms[i]);
        }
    }
    _dirty = false;
}

} // namespace newtype::scene
