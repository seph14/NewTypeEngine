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
      _instanceIds(instance_count, kInvalidShapeId),
      _pendingMaterials(instance_count, kInheritMaterialLayers) {

    // Initialize CPU transforms to identity
    _cpuTransforms.resize(instance_count);
    for (uint i = 0u; i < instance_count; ++i) {
        _cpuTransforms[i] = make_float4x4(1.0f);
    }
}

void InstancedMesh::set_transforms(luisa::span<const float4x4> transforms) noexcept {
    if (transforms.size() != _count) return;
    for (uint i = 0u; i < _count; ++i) {
        _cpuTransforms[i] = transforms[i];
    }
    _dirty = true;
}

void InstancedMesh::set_transform(uint index, const float4x4 &transform) noexcept {
    if (index >= _count) return;
    _cpuTransforms[index] = transform;
    _dirty = true;
}

void InstancedMesh::apply_gpu_transforms(luisa::compute::Stream &stream) noexcept {
    stream << _gpuTransforms.copy_to(luisa::span{_cpuTransforms});
    _dirty = true;
}

void InstancedMesh::set_material(uint index, uint32_t layers) noexcept {
    if (index >= _count) return;
    _pendingMaterials[index] = layers;
}

void InstancedMesh::set_materials(luisa::span<const uint32_t> layers) noexcept {
    if (layers.size() != _count) return;
    for (uint i = 0u; i < _count; ++i)
        _pendingMaterials[i] = layers[i];
}

void InstancedMesh::set_user_param(uint index, uint slot, float4 value) noexcept {
    if (index >= _count || slot >= 4u) return;
    _pendingUserParams.push_back({index, slot, value});
}

void InstancedMesh::flush_transform_updates(Geometry &geom) noexcept {
    // Queued material swaps apply even when transforms are clean (an
    // idle batch still supports per-instance material animation).
    for (uint i = 0u; i < _count; ++i) {
        if (_instanceIds[i] != kInvalidShapeId &&
            _pendingMaterials[i] != kInheritMaterialLayers) {
            geom.set_material_layers(_instanceIds[i], _pendingMaterials[i]);
            _pendingMaterials[i] = kInheritMaterialLayers;
        }
    }
    // Queued user-param writes (track B2) — same idle-batch rule. The queue
    // is append-only: repeated writes to one slot replay in order, so the
    // last write wins exactly like the direct Geometry API.
    for (const auto &p : _pendingUserParams) {
        if (_instanceIds[p.index] != kInvalidShapeId)
            geom.set_instance_user_param(_instanceIds[p.index], p.slot, p.value);
    }
    _pendingUserParams.clear();
    if (!_dirty) return;
    // With a parent transform, the batch matrices are local-to-parent:
    // world = parent_world * local.
    const float4x4 parent_world = _parentTransform ? _parentTransform->matrix()
                                                   : make_float4x4(1.0f);
    for (uint i = 0u; i < _count; ++i) {
        if (_instanceIds[i] != kInvalidShapeId) {
            geom.set_instance_transform(_instanceIds[i], _parentTransform
                                            ? parent_world * _cpuTransforms[i]
                                            : _cpuTransforms[i]);
        }
    }
    _dirty = false;
}

} // namespace newtype::scene
