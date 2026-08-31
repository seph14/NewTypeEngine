//
// Created by Claude on 2026/03/24.
//

#include "newtype/scene/Geometry.h"
#include "newtype/scene/LightShape.h"
#include "newtype/scene/DeformableMesh.h"
#include "newtype/core/Config.h"
#include "newtype/render/MaterialPool.h"
#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#endif
#include "newtype/util/Profiler.h"
#include <luisa/dsl/syntax.h>
#include "cinder/Log.h"
#include <algorithm>

namespace newtype::scene {

using namespace luisa;
using compute::Expr;
using compute::Float3;
using compute::Var;

//==============================================================================
// Geometry
//==============================================================================

Geometry::Geometry(Device &device) noexcept
    : _device(device),
      _tlas(device.create_accel()),
      _vertex_bindless(device.create_bindless_array()),
      _world_max(std::numeric_limits<float>::lowest()),
      _world_min(std::numeric_limits<float>::max()) { }

ShapeId Geometry::add_shape(luisa::unique_ptr<MeshShape> shape, const Transform *transform,
                             float shadow_terminator,
                             float intersection_offset) noexcept {
    return _add_shape_impl(std::move(shape), transform, nullptr, shadow_terminator, intersection_offset);
}

ShapeId Geometry::add_shape(luisa::unique_ptr<MeshShape> shape, luisa::unique_ptr<Transform> transform,
                             float shadow_terminator,
                             float intersection_offset) noexcept {
    if (!transform) transform = StaticTransform::create();
    return _add_shape_impl(std::move(shape), transform.get(), std::move(transform),
                           shadow_terminator, intersection_offset);
}

ShapeId Geometry::_add_shape_impl(luisa::unique_ptr<MeshShape> shape, const Transform *transform,
                                  luisa::unique_ptr<Transform> owned,
                                  float shadow_terminator,
                                  float intersection_offset) noexcept {
    if (!shape) return kInvalidShapeId;

    // Capture before `owned` is moved into the instance data below.
    const bool has_owned_transform = owned != nullptr;

    // Allocate ShapeId (reuse free slot or append)
    ShapeId shape_id;
    if (!_free_slots.empty()) {
        shape_id = _free_slots.back();
        _free_slots.pop_back();
    } else {
        shape_id = static_cast<ShapeId>(_slots.size());
        _slots.emplace_back();
    }

    // Dense TLAS index
    uint tlas_index = static_cast<uint>(_instances.size());

    // Set transform on the shape
    shape->set_transform(transform->matrix());

    // Create instance data
    InstanceData data;
    uint properties = shape->properties();
    uint tri_count  = shape->triangle_count();

    // Track light shapes (before moving)
    if (properties & PROPERTY_HAS_LIGHT) {
        _light_instance_indices.push_back(tlas_index);
    }

    // Maintain O(1) light lookup
    if (tlas_index >= _is_light_instance.size())
        _is_light_instance.resize(tlas_index + 1u, false);
    _is_light_instance[tlas_index] = (properties & PROPERTY_HAS_LIGHT) != 0;

    data._transform         = transform->matrix();
    data._material_layers   = shape->material_layers();
    data.shape              = std::move(shape);
    data._owned_transform   = std::move(owned);
    data.instance_id        = shape_id;
    data.properties         = properties;
    data.shadow_terminator  = shadow_terminator;
    data.intersection_offset= intersection_offset;

    _instances.push_back(std::move(data));

    // Update slot map
    _slots[shape_id] = { tlas_index, true };
    _tlas_to_shape.push_back(shape_id);

    // Track instanced transforms if dynamic. Owned transforms are registered
    // unconditionally — the pipeline guarantees their lifetime, so polling a
    // (currently) static one is safe and later mutation propagates.
    if (has_owned_transform || !transform->is_static()) {
        _transform_tree.push(transform);
        auto [node, is_static] = _transform_tree.leaf(transform);
        _transform_tree.pop(transform);
        _instanced_transforms.emplace_back(node, tlas_index);
    }

    // Update triangle count
    _triangle_count += tri_count;
    // Note: prototype instances share the prototype's triangles, so we don't
    // double-count for prototype instances. The prototype itself is NOT in
    // _instances, so this is correct for regular add_shape.

    // If already built, handle post-build addition (CPU-side setup only;
    // stream-dependent ops deferred to update())
    if (_built) {
        auto &inst = _instances[tlas_index];

        // Mark BLAS build as pending (needs stream, handled in update())
        _pending_blas_build = true;

        // Recreate bindless array with the new buffer included. Helper also
        // refreshes prototype-instance slots (which point at prototype slots
        // that may have shifted in the recreated array).
        _rebuild_bindless_array();
        _bindless_update_needed = true;

        // Add to TLAS (CPU-side registration; build deferred to update())
        const compute::Mesh *mesh = inst.get_shape()->mesh_resource();
        if (mesh) {
            _tlas.emplace_back(*mesh, inst._transform);
        } else {
            CI_LOG_W("Geometry::add_shape post-build: ShapeId=" << shape_id
                << " has null mesh_resource() — TLAS entry skipped, instance "
                << "will not be visible in ray traversal.");
        }

        _instance_props_dirty = true;
        _instance_props_structure_changed = true;
        _instance_transforms_dirty = true;
        _dirty_transform_indices.clear();
        _tlas_needs_rebuild = true;

        CI_LOG_I("Post-build add: ShapeId=" << shape_id
            << ", tlas_index=" << tlas_index
            << ", total_instances=" << _instances.size());
    }

    return shape_id;
}

void Geometry::_rebuild_bindless_array() noexcept {
    // Recreate the bindless array from scratch and reassign slots for every
    // prototype and owning instance. Prototype instances copy their prototype's
    // freshly-assigned slots. Used by build() and the post-build paths of
    // add_shape()/add_prototype() so all three stay in sync.
    uint total = static_cast<uint>(_instances.size()) + static_cast<uint>(_prototypes.size());
    compute::BindlessArray new_bindless = _device.create_bindless_array(total * 2u);
    uint slot = 0u;
    for (auto &proto : _prototypes) {
        new_bindless.emplace_on_update(slot, proto.shape->vertex_buffer());
        proto.vertex_bindless_slot = slot;
        slot++;
        new_bindless.emplace_on_update(slot, proto.shape->triangle_buffer());
        proto.triangle_bindless_slot = slot;
        slot++;
    }
    for (auto &i : _instances) {
        if (i.owns_shape()) {
            new_bindless.emplace_on_update(slot, i.get_shape()->vertex_buffer());
            i.get_shape()->set_vertex_bindless_slot(slot);
            i._bindless_vert = slot;
            slot++;
            new_bindless.emplace_on_update(slot, i.get_shape()->triangle_buffer());
            i.get_shape()->set_triangle_bindless_slot(slot);
            i._bindless_tri = slot;
            slot++;
        } else {
            for (const auto &proto : _prototypes) {
                if (proto.shape.get() == i.shape_ref) {
                    i._bindless_vert = proto.vertex_bindless_slot;
                    i._bindless_tri  = proto.triangle_bindless_slot;
                    break;
                }
            }
        }
    }
    _vertex_bindless = std::move(new_bindless);
}

void Geometry::_update_instance_props_buffer() noexcept {
    _instance_buffer_cpu.clear();
    _instance_buffer_cpu.reserve(_instances.size());

    _has_visible_glass = false;
    _has_transparent_shadow_casters = false;
    _has_active_subsurface = false;

    for (const auto &inst : _instances) {
        luisa::uint4 encoded;
        encoded.x = inst.properties;
        encoded.y = inst._material_layers;
        encoded.z = inst._bindless_vert;
        encoded.w = inst._bindless_tri;
        _instance_buffer_cpu.push_back(encoded);

        if (inst.visible && _material_pool) {
            for (int layer = 0; layer < 4; ++layer) {
                uint mat_idx = (inst._material_layers >> (layer * 8)) & 0xFFu;
                if (mat_idx == 0xFFu) continue;
                const auto &md = _material_pool->getMaterial(mat_idx).data;
                if (!_has_visible_glass && (md.type == 3u || md.type == 11u ||
                    static_cast<uint>(md.bsdf_type_override) == 3u ||
                    static_cast<uint>(md.bsdf_type_override) == 11u))
                    _has_visible_glass = true;
                if (!_has_transparent_shadow_casters &&
                    (md.type == 3u || md.type == 5u || md.type == 11u ||
                        static_cast<uint>(md.bsdf_type_override) == 3u ||
                        static_cast<uint>(md.bsdf_type_override) == 5u ||
                        static_cast<uint>(md.bsdf_type_override) == 11u || md.alphacut > 0.f))
                    _has_transparent_shadow_casters = true;
                // Active-Subsurface: matches PassSSS probe gate, which reads
                // only the layer-0 material. flatness check is left to the
                // kernel so runtime UI tweaks to flatness don't require a
                // rescan.
                if (!_has_active_subsurface && layer == 0 && md.type == 6u)
                    _has_active_subsurface = true;
                if (_has_visible_glass && _has_transparent_shadow_casters && _has_active_subsurface) break;
            }
        }
    }

#if NT_ENABLE_PROCEDURAL
    // Procedural instances currently have no SSS path (PassSSS early-outs on
    // is_procedural), so we only scan for glass/transparent here.
    if (_material_pool && _procGeom && !(_has_visible_glass && _has_transparent_shadow_casters)) {
        for (const auto &inst : _procGeom->instances_cpu()) {
            for (int layer = 0; layer < 4; ++layer) {
                uint mat_idx = (inst.material_layers >> (layer * 8)) & 0xFFu;
                if (mat_idx == 0xFFu) continue;
                const auto &md = _material_pool->getMaterial(mat_idx).data;
                if (!_has_visible_glass && (md.type == 3u || md.type == 11u))
                    _has_visible_glass = true;
                if (!_has_transparent_shadow_casters &&
                    (md.type == 3u || md.type == 5u || md.type == 11u || md.alphacut > 0.f))
                    _has_transparent_shadow_casters = true;
                if (_has_visible_glass && _has_transparent_shadow_casters) break;
            }
            if (_has_visible_glass && _has_transparent_shadow_casters) break;
        }
    }
#endif

    if (!_instance_buffer || _instance_buffer.size() != _instance_buffer_cpu.size())
        _instance_buffer = _device.create_buffer<luisa::uint4>(_instance_buffer_cpu.size());

    // _has_visible_glass / _has_active_subsurface were just recomputed inline
    // above; clear deferred flags to avoid a redundant scan in update().
    _has_visible_glass_dirty = false;
    _has_active_subsurface_dirty = false;
}

void Geometry::_recompute_has_visible_glass() noexcept {
    _has_visible_glass = false;
    _has_transparent_shadow_casters = false;
    _has_active_subsurface = false;
    if (!_material_pool) return;
    for (const auto &inst : _instances) {
        if (!inst.visible) continue;
        for (int layer = 0; layer < 4; ++layer) {
            uint mat_idx = (inst._material_layers >> (layer * 8)) & 0xFFu;
            if (mat_idx == 0xFFu) continue;
            const auto &md = _material_pool->getMaterial(mat_idx).data;
            if (!_has_visible_glass && (md.type == 3u || md.type == 11u))
                _has_visible_glass = true;
            if (!_has_transparent_shadow_casters &&
                (md.type == 3u || md.type == 5u || md.type == 11u || md.alphacut > 0.f))
                _has_transparent_shadow_casters = true;
            if (!_has_active_subsurface && layer == 0 && md.type == 6u)
                _has_active_subsurface = true;
            if (_has_visible_glass && _has_transparent_shadow_casters && _has_active_subsurface) break;
        }
        if (_has_visible_glass && _has_transparent_shadow_casters && _has_active_subsurface) return;
    }
    // Also check procedural instances (glass/transparent only — no SSS path)
#if NT_ENABLE_PROCEDURAL
    if (_procGeom) {
        for (const auto &inst : _procGeom->instances_cpu()) {
            for (int layer = 0; layer < 4; ++layer) {
                uint mat_idx = (inst.material_layers >> (layer * 8)) & 0xFFu;
                if (mat_idx == 0xFFu) continue;
                const auto &md = _material_pool->getMaterial(mat_idx).data;
                if (!_has_visible_glass && (md.type == 3u || md.type == 11u))
                    _has_visible_glass = true;
                if (!_has_transparent_shadow_casters &&
                    (md.type == 3u || md.type == 5u || md.type == 11u || md.alphacut > 0.f))
                    _has_transparent_shadow_casters = true;
                if (_has_visible_glass && _has_transparent_shadow_casters) break;
            }
            if (_has_visible_glass && _has_transparent_shadow_casters) return;
        }
    }
#endif
}

void Geometry::build(Stream &stream) noexcept {
    // First, build all prototype BLAS
    for (auto &proto : _prototypes) {
        if (!proto.shape->built()) {
            proto.shape->build(stream);
        }
    }

    // Build all regular mesh BLAS
    for (auto &inst : _instances) {
        if (inst.owns_shape() && !inst.get_shape()->built()) {
            inst.get_shape()->build(stream);
        }
    }

    // Create vertex bindless array and register all buffers
    // Prototypes get their own slots; prototype instances share them
    _rebuild_bindless_array();
    stream << _vertex_bindless.update();

    // Update instance buffer
    _update_instance_props_buffer();
    stream << _instance_buffer.copy_from(_instance_buffer_cpu.data());

    // Populate transform CPU buffer and upload
    _instance_transform_cpu.clear();
    _instance_transform_cpu.reserve(_instances.size());
    for (const auto &inst : _instances)
        _instance_transform_cpu.push_back(inst._transform);
    if (!_instance_transform_buffer || _instance_transform_buffer.size() != _instance_transform_cpu.size())
        _instance_transform_buffer = _device.create_buffer<luisa::float4x4>(_instance_transform_cpu.size());
    stream << _instance_transform_buffer.copy_from(_instance_transform_cpu.data());
    // Topology (re)build: prev = curr so instances get zero object motion on
    // the first frame; dense indices reshuffle on add/remove, so carrying old
    // per-slot transforms would reproject through the wrong instance.
    if (!_instance_transform_prev_buffer || _instance_transform_prev_buffer.size() != _instance_transform_cpu.size())
        _instance_transform_prev_buffer = _device.create_buffer<luisa::float4x4>(_instance_transform_cpu.size());
    stream << _instance_transform_buffer.copy_to(_instance_transform_prev_buffer);
    _transform_prev_stale = false;

    // Build TLAS with all mesh instances (regular + prototype instances)
    for (const auto &inst : _instances) {
        MeshShape *shape = inst.get_shape();
        const compute::Mesh *mesh_to_use = shape->mesh_resource();

        if (mesh_to_use) {
            _tlas.emplace_back(*mesh_to_use, inst._transform);
        }
    }

    // Ensure O(1) light lookup is sized correctly
    _is_light_instance.resize(_instances.size(), false);
    for (uint i = 0u; i < _instances.size(); ++i)
        _is_light_instance[i] = (_instances[i].properties & PROPERTY_HAS_LIGHT) != 0;

    stream << _tlas.build();

#if NT_ENABLE_PROCEDURAL
    // Register procedural primitive BLAS in TLAS (identity transform)
    if (_procGeom && _procGeom->has_instances()) {
        uint proc_tlas_index = static_cast<uint>(_instances.size());
        _tlas.emplace_back(_procGeom->blas());
        stream << _tlas.build();
        CI_LOG_I("ProceduralGeometry BLAS registered as TLAS instance " << proc_tlas_index
            << " (" << _procGeom->instance_count() << " AABBs)");
    }
#endif

    _built = true;
}

bool Geometry::update(Stream &stream, bool requireTLASRefit, float time) noexcept {
    bool needs_update = requireTLASRefit, requireSync = false;

    // Build any pending BLAS (from post-build additions)
    if (_pending_blas_build) {
        for (auto &proto : _prototypes) {
            if (!proto.shape->built()) {
                proto.shape->build(stream);
            }
        }
        for (auto &inst : _instances) {
            if (inst.owns_shape() && !inst.get_shape()->built()) {
                inst.get_shape()->build(stream);
            }
        }
        _pending_blas_build = false;
        needs_update = true;
    }

    // Flush pending bindless array update (from post-build additions/removals)
    if (_bindless_update_needed) {
        stream << _vertex_bindless.update() << compute::synchronize();
        requireSync = true;
        _bindless_update_needed = false;
    }

    // 1. Deformable meshes (skip prototype instances — they share the prototype's BLAS)
    for (auto &inst : _instances) {
        if (inst.owns_shape() && inst.get_shape()->deformable()) {
            auto *deformable = static_cast<DeformableMesh*>(inst.get_shape());
            if (deformable->update(stream)) {
                uint tlas_idx = _slots[inst.instance_id].tlas_index;
                _tlas.set_mesh(tlas_idx, *deformable->mesh_resource());
                // Update bindless array to point to the new frame's vertex buffer
                _vertex_bindless.emplace_on_update(
                    deformable->vertex_bindless_slot(),
                    deformable->vertex_buffer());
                _bindless_update_needed = true;
                needs_update = true;
            }
        }
    }

    // Flush bindless updates from deformable mesh buffer swaps
    if (_bindless_update_needed) {
        stream << _vertex_bindless.update();
        _bindless_update_needed = false;
    }

    // 2. Animated transforms
    bool transforms_dirty = false;
    for (auto &inst_transform : _instanced_transforms) {
        if (inst_transform.is_dirty()) {
            float4x4 new_matrix = inst_transform.matrix();
            uint tlas_idx = static_cast<uint>(inst_transform.get_instance_id());
            _instances[tlas_idx]._transform = new_matrix;
            // Size guard: post-build additions leave _instance_transform_cpu
            // undersized until upload_dirty_transforms() resizes it.
            if (tlas_idx < _instance_transform_cpu.size()) {
                _instance_transform_cpu[tlas_idx] = new_matrix;
                _dirty_transform_indices.push_back(tlas_idx);
            }
            //if (_instances[tlas_idx].owns_shape())
                _instances[tlas_idx].get_shape()->set_transform(new_matrix);
            _tlas.set_transform_on_update(inst_transform.get_instance_id(), new_matrix);

            // Track light transform changes for LightSampler re-upload (O(1))
            if (tlas_idx < _is_light_instance.size() && _is_light_instance[tlas_idx]) {
                _lightTransformDirty = true;
                auto ch = inst_transform.node->transform()->change();
                if (ch == Change::Scale) _lightScaleDirty = true;
            }

            const_cast<Transform*>(inst_transform.node->transform())->clear_dirty();
            needs_update = true;
            transforms_dirty = true;
        }
    }

    // 2a. Shape-internal transform updates (getShape(id)->set_transform(m) or
    // getShapeTransform(id)->set_*() on the shape's internal transform). The
    // GPU state (TLAS matrix, instance transform buffers) lives here, not in
    // the shape, so the shape's dirty flag must be polled and propagated like
    // the animated path above. The animated loop re-marks the shape-internal
    // transform via set_transform() after propagating, and Geometry::set_transform
    // marks it after its immediate propagation — the matrix comparison makes
    // those no-ops, and clear_dirty() keeps the flag from staying set forever.
    // Prototype instances are skipped: shape_ref is shared with the prototype
    // and all sibling instances, so the internal transform is not authoritative.
    for (uint tlas_idx = 0u; tlas_idx < _instances.size(); ++tlas_idx) {
        auto &inst = _instances[tlas_idx];
        if (!inst.owns_shape()) continue;
        Transform *shape_transform = inst.get_shape()->transform();
        if (!shape_transform->is_dirty()) continue;
        float4x4 new_matrix = shape_transform->matrix();
        bool changed = any(new_matrix.cols[0] != inst._transform.cols[0]) ||
                       any(new_matrix.cols[1] != inst._transform.cols[1]) ||
                       any(new_matrix.cols[2] != inst._transform.cols[2]) ||
                       any(new_matrix.cols[3] != inst._transform.cols[3]);
        if (changed) {
            inst._transform = new_matrix;
            if (tlas_idx < _instance_transform_cpu.size()) {
                _instance_transform_cpu[tlas_idx] = new_matrix;
                _dirty_transform_indices.push_back(tlas_idx);
            }
            _tlas.set_transform_on_update(tlas_idx, new_matrix);

            // Track light transform changes for LightSampler re-upload (O(1))
            if (tlas_idx < _is_light_instance.size() && _is_light_instance[tlas_idx]) {
                _lightTransformDirty = true;
                if (shape_transform->change() == Change::Scale)
                    _lightScaleDirty = true;
            }
            needs_update = true;
            transforms_dirty = true;
        }
        shape_transform->clear_dirty();
    }
    if (transforms_dirty) _instance_transforms_dirty = true;

    // 3. Visibility + transform changes (already queued via set_*_on_update)
    if (_visibility_dirty || _transform_dirty) {
        needs_update = true;
        _visibility_dirty = false;
        _transform_dirty = false;
    }

    // 3a. Deferred material-flag recompute (set_visibility / set_material_layers
    // mark dirty; the value is read only at render-time, so safe to recompute here).
    // Single recompute handles glass + transparent + subsurface in one scan.
    if (_has_visible_glass_dirty || _has_active_subsurface_dirty) {
        _recompute_has_visible_glass();
        _has_visible_glass_dirty = false;
        _has_active_subsurface_dirty = false;
    }

    // 4. Build TLAS
    if (_tlas_needs_rebuild) {
        // Full rebuild (topology changed from add/remove)
        stream << _tlas.build(); 
        _tlas_needs_rebuild = false;
        needs_update = true;  // Ensure caller knows something changed
    } else if (needs_update) {
        // Fast refit
        stream << _tlas.build(Accel::BuildRequest::PREFER_UPDATE);  
    }

    return needs_update || requireSync;
}

//==============================================================================
// Prototype Instancing (Shared BLAS)
//==============================================================================

ShapeId Geometry::add_prototype(luisa::unique_ptr<MeshShape> prototype) noexcept {
    if (!prototype) return kInvalidShapeId;

    // Allocate ShapeId for the prototype
    ShapeId proto_id;
    if (!_free_slots.empty()) {
        proto_id = _free_slots.back();
        _free_slots.pop_back();
    } else {
        proto_id = static_cast<ShapeId>(_slots.size());
        _slots.emplace_back();
    }

    // Store prototype with its ShapeId for later lookup
    uint proto_idx = static_cast<uint>(_prototypes.size());
    PrototypeData pd;
    pd.shape = std::move(prototype);
    _prototypes.push_back(std::move(pd));

    // Mark slot as alive (tlas_index stores prototype index, not TLAS index)
    _slots[proto_id] = { proto_idx, true };

    // If the scene is already built, the prototype's BLAS must be built on the
    // next update() pass and the bindless array must be extended to cover the
    // new prototype's vertex/triangle buffers. Without this, add_instance()
    // calls referencing this prototype would copy ~0u bindless slots and the
    // TLAS would reference an unbuilt BLAS — producing silent UB that manifests
    // as driver stalls / garbage renders. Mirrors add_shape's post-build path.
    if (_built) {
        if (!_prototypes.back().shape->mesh_resource()) {
            CI_LOG_W("Geometry::add_prototype post-build: ShapeId=" << proto_id
                << " has null mesh_resource() — instances referencing this "
                "prototype will not traverse correctly.");
        }
        _pending_blas_build = true;
        _rebuild_bindless_array();
        _bindless_update_needed = true;
        // Refresh instance props so any pre-existing prototype instances pick
        // up potentially-shifted slot indices in the recreated bindless array.
        _instance_props_dirty = true;
    }

    CI_LOG_I("Prototype registered: ShapeId=" << proto_id
        << ", proto_idx=" << proto_idx
        << ", triangles=" << _prototypes.back().shape->triangle_count()
        << ", total_prototypes=" << _prototypes.size()
        << (_built ? " (post-build)" : ""));

    return proto_id;
}

ShapeId Geometry::add_instance(ShapeId prototype_id, const float4x4 &transform) noexcept {
    if (prototype_id >= _slots.size() || !_slots[prototype_id].alive)
        return kInvalidShapeId;

    // Lookup prototype via slot map (tlas_index stores proto_idx for prototypes)
    uint proto_idx = _slots[prototype_id].tlas_index;
    if (proto_idx >= _prototypes.size()) {
        CI_LOG_E("add_instance: ShapeId " << prototype_id << " is not a prototype");
        return kInvalidShapeId;
    }

    MeshShape *proto_shape = _prototypes[proto_idx].shape.get();
    if (!proto_shape) {
        CI_LOG_E("add_instance: null prototype shape");
        return kInvalidShapeId;
    }

    // Allocate ShapeId for this instance
    ShapeId instance_id;
    if (!_free_slots.empty()) {
        instance_id = _free_slots.back();
        _free_slots.pop_back();
    } else {
        instance_id = static_cast<ShapeId>(_slots.size());
        _slots.emplace_back();
    }

    // Dense TLAS index
    uint tlas_index = static_cast<uint>(_instances.size());

    // Create lightweight InstanceData (no owning shape)
    InstanceData data;
    data.shape              = nullptr;                   // No owning shape
    data.shape_ref          = proto_shape;                // Reference to prototype
    data.instance_id        = instance_id;
    data.properties         = proto_shape->properties();
    data.shadow_terminator  = 0.0f;
    data.intersection_offset= 0.0f;
    data._transform         = transform;
    data._material_layers   = proto_shape->material_layers();
    data._bindless_vert     = _prototypes[proto_idx].vertex_bindless_slot;
    data._bindless_tri      = _prototypes[proto_idx].triangle_bindless_slot;

    // Track light shapes
    if (data.properties & PROPERTY_HAS_LIGHT) {
        _light_instance_indices.push_back(tlas_index);
        _lights_dirty = true;
    }

    // Maintain O(1) light lookup
    if (tlas_index >= _is_light_instance.size())
        _is_light_instance.resize(tlas_index + 1u, false);
    _is_light_instance[tlas_index] = (data.properties & PROPERTY_HAS_LIGHT) != 0;

    _instances.push_back(std::move(data));

    // Update slot map
    _slots[instance_id] = { tlas_index, true };
    _tlas_to_shape.push_back(instance_id);

    // Add to TLAS — only for post-build additions (pre-build handled by build())
    if (_built) {
        const compute::Mesh *mesh = proto_shape->mesh_resource();
        if (mesh) {
            _tlas.emplace_back(*mesh, transform);
        } else {
            CI_LOG_W("Geometry::add_instance post-build: prototype has null "
                "mesh_resource() — TLAS entry skipped, instance will not be "
                "visible in ray traversal (proto ShapeId=" << prototype_id
                << ", inst ShapeId=" << instance_id << ").");
        }
        _instance_props_dirty = true;
        _instance_props_structure_changed = true;
        _instance_transforms_dirty = true;
        _dirty_transform_indices.clear();
        _tlas_needs_rebuild = true;
    }

    return instance_id;
}

void Geometry::add_instances(ShapeId prototype_id,
                             luisa::span<const float4x4> transforms,
                             luisa::vector<ShapeId> &out_ids) noexcept {
    out_ids.reserve(out_ids.size() + transforms.size());
    for (auto &xform : transforms) {
        auto id = add_instance(prototype_id, xform);
        out_ids.push_back(id);
    }
}

MeshShape *Geometry::get_prototype(ShapeId prototype_id) const noexcept {
    if (prototype_id >= _slots.size() || !_slots[prototype_id].alive)
        return nullptr;
    uint proto_idx = _slots[prototype_id].tlas_index;
    if (proto_idx >= _prototypes.size()) return nullptr;
    return _prototypes[proto_idx].shape.get();
}

//==============================================================================
// Runtime Manipulation
//==============================================================================

void Geometry::set_transform(ShapeId id, const float4x4 &matrix,
                             Change changeHint) noexcept {
    if (!is_valid(id)) return;
    if (!_built) {
        CI_LOG_W("Geometry::set_transform called before build() — transform "
            "will be applied when build() runs, but TLAS ops are queued on "
            "an unbuilt accel (ShapeId=" << id << ").");
    }
    uint tlas_idx = _slots[id].tlas_index;
    auto &inst = _instances[tlas_idx];

    inst._transform = matrix;
    if (tlas_idx < _instance_transform_cpu.size()) {
        _instance_transform_cpu[tlas_idx] = matrix;
        _dirty_transform_indices.push_back(tlas_idx);
    }
    _instance_transforms_dirty = true;
    if (inst.owns_shape()) inst.get_shape()->set_transform(matrix);

    _tlas.set_transform_on_update(tlas_idx, matrix);
    _transform_dirty = true;

    // Track light transform changes for LightSampler re-upload (O(1))
    if (tlas_idx < _is_light_instance.size() && _is_light_instance[tlas_idx]) {
        _lightTransformDirty = true;
        if (changeHint == Change::Scale)
            _lightScaleDirty = true;
    }
}

void Geometry::set_visibility(ShapeId id, bool visible) noexcept {
    if (!is_valid(id)) return;
    if (!_built) {
        CI_LOG_W("Geometry::set_visibility called before build() — visibility "
            "state is recorded but TLAS set_visibility_on_update is queued on "
            "an unbuilt accel (ShapeId=" << id << ").");
    }
    uint tlas_idx = _slots[id].tlas_index;
    if (_instances[tlas_idx].visible == visible) return;
    _instances[tlas_idx].visible = visible;
    _tlas.set_visibility_on_update(tlas_idx, visible ? 0xFFu : 0x00u);
    _visibility_dirty = true;

    // If this is a light shape, light sampler weights need refreshing
    // (alias table must exclude/include its triangles). Visibility toggles
    // don't change topology, so update_weights is sufficient — no full rebuild.
    if (_instances[tlas_idx].properties & PROPERTY_HAS_LIGHT) {
        _lights_visibility_dirty = true;
    }

    _has_visible_glass_dirty = true;
    _has_active_subsurface_dirty = true;
}

void Geometry::set_material_layers(ShapeId id, uint32_t layers) noexcept {
    if (!is_valid(id)) return;
    if (!_built) {
        CI_LOG_W("Geometry::set_material_layers called before build() — value "
            "is recorded on the instance; build() will pick it up, but any "
            "out-of-range layer indices won't be validated until shade "
            "(ShapeId=" << id << ").");
    }
    uint tlas_idx = _slots[id].tlas_index;
    auto &inst = _instances[tlas_idx];
    if (inst._material_layers == layers) return;

    // B9: validate layer indices against material pool if available
    if (_material_pool) {
        for (int layer = 0; layer < 4; ++layer) {
            uint mat_idx = (layers >> (layer * 8)) & 0xFFu;
            if (mat_idx == 0xFFu) continue;  // sentinel: layer unused
            if (mat_idx >= _material_pool->count()) {
                CI_LOG_W("Geometry::set_material_layers: layer " << layer
                    << " references material index " << mat_idx
                    << " which is out of range (pool size "
                    << _material_pool->count() << ") — shade will read garbage "
                    "data (ShapeId=" << id << ").");
            }
        }
    }

    inst._material_layers = layers;
    if (inst.owns_shape()) inst.get_shape()->set_material_layers(layers);
    if (tlas_idx < _instance_buffer_cpu.size())
        _instance_buffer_cpu[tlas_idx].y = layers;
    _instance_props_dirty = true;
    _has_visible_glass_dirty = true;
    _has_active_subsurface_dirty = true;
}

bool Geometry::remove_shape(ShapeId id) noexcept {
    if (!is_valid(id)) return false;

    if (!_built) {
        CI_LOG_W("Geometry::remove_shape called before build() — TLAS has no "
            "entries yet; _tlas.pop_back() will be called on an empty accel "
            "(ShapeId=" << id << ").");
    }

    uint tlas_idx = _slots[id].tlas_index;
    uint last_idx = static_cast<uint>(_instances.size()) - 1u;

    // Subtract triangle count (only for owned shapes, not prototype instances)
    if (_instances[tlas_idx].owns_shape()) {
        _triangle_count -= _instances[tlas_idx].get_shape()->triangle_count();
    }

    // Remove the removed instance's animated transform entry (if any)
    _instanced_transforms.erase(
        std::remove_if(_instanced_transforms.begin(), _instanced_transforms.end(),
            [tlas_idx](const InstancedTransform &it) {
                return it.get_instance_id() == tlas_idx;
            }),
        _instanced_transforms.end());

    // Remove from light indices if it was a light
    if (_instances[tlas_idx].properties & PROPERTY_HAS_LIGHT) {
        _light_instance_indices.erase(
            std::remove(_light_instance_indices.begin(), _light_instance_indices.end(), tlas_idx),
            _light_instance_indices.end());
        _lights_dirty = true;
    }

    if (tlas_idx != last_idx) {
        // Swap with last instance
        uint moved_shape_id = _tlas_to_shape[last_idx];

        // Move instance data
        _instances[tlas_idx] = std::move(_instances[last_idx]);

        // Update TLAS: set moved mesh at the new position
        auto &moved = _instances[tlas_idx];
        MeshShape *moved_shape = moved.get_shape();
        const compute::Mesh *mesh = moved_shape->mesh_resource();
        if (mesh) {
            _tlas.set(tlas_idx, *mesh, moved._transform);
        }

        // Bindless slots are stable global indices — no recalculation needed.
        // The moved instance keeps its existing _bindless_vert/_bindless_tri.

        // Update slot map for moved instance
        _slots[moved_shape_id].tlas_index = tlas_idx;
        _tlas_to_shape[tlas_idx] = moved_shape_id;

        // Update the moved instance's animated transform entry (last_idx → tlas_idx)
        for (auto &it : _instanced_transforms) {
            if (it.get_instance_id() == last_idx) {
                it.instance_id = tlas_idx;
                break;
            }
        }

        // Update light_instance_indices: moved light last_idx → tlas_idx
        for (auto &li : _light_instance_indices) {
            if (li == last_idx) li = tlas_idx;
        }

        // Update O(1) light lookup for swapped instance
        _is_light_instance[tlas_idx] = _is_light_instance[last_idx];
        _is_light_instance[last_idx] = false;
    }

    // Pop last from containers
    _tlas.pop_back();
    _instances.pop_back();
    _tlas_to_shape.pop_back();
    _is_light_instance.pop_back();

    // Free the slot
    _slots[id].alive = false;
    _free_slots.push_back(id);

    // Mark dirty
    _instance_props_dirty = true;
    _instance_props_structure_changed = true;
    _instance_transforms_dirty = true;
    _dirty_transform_indices.clear();
    _tlas_needs_rebuild = true;

    CI_LOG_I("Removed ShapeId=" << id << ", remaining_instances=" << _instances.size());
    return true;
}

//==============================================================================
// Accessors
//==============================================================================

MeshShape* Geometry::get_shape(ShapeId id) noexcept {
    if (!is_valid(id)) return nullptr;
    return _instances[_slots[id].tlas_index].get_shape();
}

Transform* Geometry::get_transform(ShapeId id) noexcept {
    if (!is_valid(id)) return nullptr;
    auto &inst = _instances[_slots[id].tlas_index];
    return inst._owned_transform ? inst._owned_transform.get()
                                 : inst.get_shape()->transform();
}

DeformableMesh* Geometry::get_deformable(ShapeId id) noexcept {
    MeshShape *shape = get_shape(id);
    if (!shape || !shape->deformable()) return nullptr;
    return static_cast<DeformableMesh*>(shape);
}

bool Geometry::is_valid(ShapeId id) const noexcept {
    return id < _slots.size() && _slots[id].alive;
}

bool Geometry::is_visible(ShapeId id) const noexcept {
    if (!is_valid(id)) return false;
    return _instances[_slots[id].tlas_index].visible;
}

void Geometry::upload_instance_props(Stream &stream) noexcept {
    if (_instance_props_structure_changed) {
        _update_instance_props_buffer();
        _instance_props_structure_changed = false;
    }
    // Ensure buffer exists and is correct size
    if (!_instance_buffer || _instance_buffer.size() != _instance_buffer_cpu.size())
        _instance_buffer = _device.create_buffer<luisa::uint4>(_instance_buffer_cpu.size());
    stream << _instance_buffer.copy_from(_instance_buffer_cpu.data());
    _instance_props_dirty = false;
}

void Geometry::upload_dirty_transforms(Stream &stream) noexcept {
    if (_instance_transforms_dirty) {
        // Topology changed since the GPU buffers were sized (post-build
        // add_shape/add_instance/remove_shape set _instance_transforms_dirty
        // but clear _dirty_transform_indices, so the per-index path below
        // would upload nothing while the buffers hold stale/undersized data).
        // Resize the CPU mirror (full repopulate — swap-and-pop on removal
        // reshuffles dense indices) and recreate the GPU buffers with a full
        // upload, mirroring build().
        if (_instance_transform_cpu.size() != _instances.size()) {
            _dirty_transform_indices.clear();
            _instance_transforms_dirty = false;
            _transform_prev_stale = false;
            if (_instances.empty()) return;
            _instance_transform_cpu.resize(_instances.size());
            for (uint i = 0u; i < _instances.size(); ++i)
                _instance_transform_cpu[i] = _instances[i]._transform;
            _instance_transform_buffer = _device.create_buffer<luisa::float4x4>(_instance_transform_cpu.size());
            _instance_transform_prev_buffer = _device.create_buffer<luisa::float4x4>(_instance_transform_cpu.size());
            stream << _instance_transform_buffer.copy_from(_instance_transform_cpu.data());
            // prev = curr: new/swapped instances get zero object motion rather
            // than reprojection through a reshuffled slot (same rationale as
            // build()).
            stream << _instance_transform_buffer.copy_to(_instance_transform_prev_buffer);
            return;
        }
        // Snapshot the GPU current buffer into prev BEFORE the dirty uploads:
        // at this point it still holds last frame's transforms. Static instances
        // get prev == curr (zero object motion) automatically. This full-buffer
        // device copy is 64B per instance.
        if (_instance_transform_prev_buffer &&
            _instance_transform_prev_buffer.size() == _instance_transform_buffer.size()) {
            stream << _instance_transform_buffer.copy_to(_instance_transform_prev_buffer);
        }
        for (uint idx : _dirty_transform_indices) {
            stream << _instance_transform_buffer.view(idx, 1u).copy_from(
                &_instance_transform_cpu[idx]);
        }
        _dirty_transform_indices.clear();
        _instance_transforms_dirty = false;
        // curr advanced this frame; once motion stops, prev needs one re-sync.
        _transform_prev_stale = true;
    } else if (_transform_prev_stale) {
        // Motion stopped: bring prev back in line with curr so object motion
        // returns to zero instead of persisting as phantom flow.
        if (_instance_transform_prev_buffer &&
            _instance_transform_prev_buffer.size() == _instance_transform_buffer.size()) {
            stream << _instance_transform_buffer.copy_to(_instance_transform_prev_buffer);
        }
        _transform_prev_stale = false;
    }
}

// Note: Hit processing methods are defined inline in Geometry.h as they
// use LuisaCompute DSL constructs that require JIT compilation.

} // namespace newtype::scene
