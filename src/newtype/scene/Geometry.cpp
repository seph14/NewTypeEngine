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
#include "cinder/CinderAssert.h"
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
    // Capture the raw pointer BEFORE the move: parameter initializers are
    // indeterminately sequenced and MSVC evaluates them right-to-left, so an
    // inline "transform.get(), std::move(transform)" pair hands the callee a
    // null raw pointer once `owned` has been move-initialized first.
    const Transform *raw = transform.get();
    return _add_shape_impl(std::move(shape), raw, std::move(transform),
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
    _slots[shape_id] = { tlas_index, true, false };
    _tlas_to_shape.push_back(shape_id);

    // Track instanced transforms if dynamic. Owned transforms are registered
    // unconditionally — the pipeline guarantees their lifetime, so polling a
    // (currently) static one is safe and later mutation propagates. For
    // borrowed transforms, is_static() is chain-aware: a static transform
    // under an animated parent is registered so parent motion re-flushes it.
    if (has_owned_transform || !transform->is_static()) {
        _instanced_transforms.emplace_back(transform, tlas_index);
    }

    // Also poll the shape-internal transform (getShape(id)->set_transform or
    // get_transform(id)->set_*() on it): the GPU state lives in Geometry, not
    // in the shape, so its dirty flag is propagated like the registered
    // transforms above. Regular add_shape instances always own their shape.
    _shape_transform_instances.push_back(tlas_index);

    // Per-frame deformable poll (update() step 1): deformable meshes always
    // own their shape, and deformable() is fixed at construction — register
    // once here (pre- and post-build adds take this same path). Read the
    // LIVE instance: `data` is moved-from above and owning adds leave its
    // shape_ref null.
    if (_instances[tlas_index].get_shape()->deformable())
        _deformable_instances.push_back(tlas_index);

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

        // Add to TLAS (CPU-side registration; build deferred to update()).
        // Pass the recorded visibility so a set_visibility() between add and
        // the deferred rebuild is not lost.
        const compute::Mesh *mesh = inst.get_shape()->mesh_resource();
        if (mesh) {
            _tlas.emplace_back(*mesh, inst._transform, inst.visible ? 0xFFu : 0x00u);
        } else {
            CI_LOG_W("Geometry::add_shape post-build: ShapeId=" << shape_id
                << " has null mesh_resource() - TLAS entry skipped, instance "
                << "will not be visible in ray traversal.");
        }

        _instance_props_dirty = true;
        _instance_props_structure_changed = true;
        _instance_transforms_dirty = true;
        _dirty_transform_indices.clear();
        _tlas_needs_rebuild = true;
        _invalidate_gpu_transform_rows("post-build add_shape");

        CI_LOG_D("Post-build add: ShapeId=" << shape_id
            << ", tlas_index=" << tlas_index
            << ", total_instances=" << static_cast<uint>(_instances.size()));
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

    // VRAM accounting (A2 gate): distinct GPU vertex buffers (prototypes +
    // owning instances share nothing; prototype instances share the
    // prototype's buffer). Reported per layout for the before/after gate.
    {
        size_t vertex_bytes = 0u, triangle_bytes = 0u;
        auto accumulate = [&](const MeshShape *s) noexcept {
            vertex_bytes += s->vertex_buffer().size() * sizeof(MeshShape::GpuVertex);
            triangle_bytes += s->triangle_buffer().size() * sizeof(Triangle);
        };
        for (auto &proto : _prototypes) accumulate(proto.shape.get());
        for (auto &inst : _instances)
            if (inst.owns_shape()) accumulate(inst.get_shape());
        CI_LOG_I("Geometry::build: GPU vertex buffers " << (vertex_bytes >> 10u)
            << " KB (" << sizeof(MeshShape::GpuVertex) << " B/vertex, layout "
            << NT_VERTEX_LAYOUT << "), triangles " << (triangle_bytes >> 10u) << " KB");
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

    // Build TLAS with all mesh instances (regular + prototype instances).
    // The recorded instance visibility is applied here — set_visibility()
    // calls made before build() rely on this (e.g. VATMesh hides all
    // topologies except the first at registration time).
    for (const auto &inst : _instances) {
        MeshShape *shape = inst.get_shape();
        const compute::Mesh *mesh_to_use = shape->mesh_resource();

        if (mesh_to_use) {
            _tlas.emplace_back(*mesh_to_use, inst._transform,
                               inst.visible ? 0xFFu : 0x00u);
        }
    }

    // Ensure O(1) light lookup is sized correctly
    _is_light_instance.resize(_instances.size(), false);
    for (uint i = 0u; i < _instances.size(); ++i)
        _is_light_instance[i] = (_instances[i].properties & PROPERTY_HAS_LIGHT) != 0;

    // Re-derive the deformable poll list from ground truth (one O(N) pass;
    // add_shape maintains it incrementally, this self-heals any future add
    // path that forgets to register).
    _deformable_instances.clear();
    for (uint i = 0u; i < _instances.size(); ++i)
        if (_instances[i].get_shape()->deformable())
            _deformable_instances.push_back(i);

    stream << _tlas.build();

#if NT_ENABLE_PROCEDURAL
    // Register procedural primitive BLAS in TLAS (identity transform)
    if (_procGeom && _procGeom->has_instances()) {
        uint proc_tlas_index = static_cast<uint>(_instances.size());
        _tlas.emplace_back(_procGeom->blas());
        stream << _tlas.build();
        CI_LOG_D("ProceduralGeometry BLAS registered as TLAS instance " << proc_tlas_index
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

    // Flush pending bindless array update (from post-build additions/removals).
    // Stream-ordered on its own; render-stream consumers are ordered by the
    // requireSync -> _geomUpdateEvent timeline signal, so no CPU stall here.
    if (_bindless_update_needed) {
        stream << _vertex_bindless.update();
        requireSync = true;
        _bindless_update_needed = false;
    }

    // 1. Deformable meshes, polled through the registered indices only
    // (add_shape registers; prototype instances share a static BLAS and can
    // never be deformable). A full _instances sweep here is O(N) per frame
    // even when nothing is deformable — ~20 ms/frame at 1.6M prototype
    // instances in a Debug build.
    for (uint tlas_idx : _deformable_instances) {
        auto *deformable = static_cast<DeformableMesh *>(
            _instances[tlas_idx].get_shape());
        if (deformable->update(stream)) {
            _tlas.set_mesh(tlas_idx, *deformable->mesh_resource());
            // Update bindless array to point to the new frame's vertex buffer
            _vertex_bindless.emplace_on_update(
                deformable->vertex_bindless_slot(),
                deformable->vertex_buffer());
            _bindless_update_needed = true;
            needs_update = true;
        }
    }

    // Flush bindless updates from deformable mesh buffer swaps
    if (_bindless_update_needed) {
        stream << _vertex_bindless.update();
        _bindless_update_needed = false;
    }

    // 2. Animated transforms. A parent mutation propagates the dirty flag
    // down to all descendants, so checking each registered leaf is enough.
    // Registered external transforms are applied first and mirror their
    // matrix into the shape's internal transform WITHOUT re-dirtying it, so
    // the shape-internal poll below only fires on direct user sets.
    bool transforms_dirty = false;
    for (auto &inst_transform : _instanced_transforms) {
        if (inst_transform.is_dirty()) {
            uint tlas_idx = static_cast<uint>(inst_transform.get_instance_id());
            const float4x4 new_matrix = inst_transform.matrix();
            _apply_instance_transform(tlas_idx, new_matrix,
                                      inst_transform.transform->change());
            // Keep the shape's CPU-side transform() view in sync. Owning
            // instances only: prototype instances share the prototype's
            // transform, which no single instance may clobber.
            if (_instances[tlas_idx].owns_shape())
                _instances[tlas_idx].get_shape()->store_transform(new_matrix);

            // Self-only clear: ancestors keep their own flags so their
            // registered instances are still processed this frame; the
            // sweep below clears the chains afterwards.
            const_cast<Transform*>(inst_transform.transform)->clear_dirty();
            needs_update = true;
            transforms_dirty = true;
        }
    }
    // Clear ancestor dirty state (group-only transforms without shapes of
    // their own are never polled directly and would stay dirty forever).
    for (auto &inst_transform : _instanced_transforms) {
        const_cast<Transform*>(inst_transform.transform)->clear_dirty_chain();
    }

    // 2a. Shape-internal transform updates (getShape(id)->set_transform(m)
    // or getShapeTransform(id)->set_*() on the shape's internal transform).
    // Polled over the registered owning instances only — no full instance
    // scan; prototype instances share the prototype's transform, which is
    // not authoritative per instance. The matrix comparison absorbs
    // redundant marks (e.g. the add-time seed writing the same matrix).
    for (uint tlas_idx : _shape_transform_instances) {
        auto &inst = _instances[tlas_idx];
        Transform *shape_transform = inst.get_shape()->transform();
        if (!shape_transform->is_dirty()) continue;
        const float4x4 new_matrix = shape_transform->matrix();
        bool changed = any(new_matrix.cols[0] != inst._transform.cols[0]) ||
                       any(new_matrix.cols[1] != inst._transform.cols[1]) ||
                       any(new_matrix.cols[2] != inst._transform.cols[2]) ||
                       any(new_matrix.cols[3] != inst._transform.cols[3]);
        if (changed) {
            _apply_instance_transform(tlas_idx, new_matrix,
                                      shape_transform->change());
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

    // 3b. GPU-owned transform rows: a device writer (e.g. the TetCage solve)
    // rewrote the registered rows of _instance_transform_buffer this frame.
    // Hand the TLAS build a device-side copy source so the matrices reach the
    // instance descriptors without a host round-trip (LC fork API; the copy
    // kernel writes only the transform rows p0-p2 and runs after the set
    // kernel, so CPU modifications to other rows compose). Deferred while a
    // topology rebuild or a transform-buffer resize is pending: the source
    // buffer would be replaced later inside this same Pipeline::update
    // (upload_dirty_transforms), and the rebuild repopulates rows anyway —
    // the writer re-registers and the GPU path resumes next frame.
    if (_gpuTransformsDirty) {
        const bool buffersInSync = _instance_transform_buffer &&
            _instance_transform_cpu.size() == _instances.size() &&
            _instance_transform_buffer.size() == _instances.size();
        if (_gpuTransformRanges.empty()) {
            // Registration lost (topology) and not re-established — drop the
            // flag so we don't force refits forever.
            _gpuTransformsDirty = false;
        } else if (!_tlas_needs_rebuild && buffersInSync) {
            const auto [first, count] = _gpuTransformRanges.front();
            _tlas.set_transform_buffer_on_update(
                first, _instance_transform_buffer.view(first, count));
            _gpuTransformsDirty = false;
            needs_update = true;
        }
        // else: keep the flag and retry next frame (rebuild settles first)
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
    _slots[proto_id] = { proto_idx, true, true };

    // If the scene is already built, the prototype's BLAS must be built on the
    // next update() pass and the bindless array must be extended to cover the
    // new prototype's vertex/triangle buffers. Without this, add_instance()
    // calls referencing this prototype would copy ~0u bindless slots and the
    // TLAS would reference an unbuilt BLAS — producing silent UB that manifests
    // as driver stalls / garbage renders. Mirrors add_shape's post-build path.
    if (_built) {
        if (!_prototypes.back().shape->mesh_resource()) {
            CI_LOG_W("Geometry::add_prototype post-build: ShapeId=" << proto_id
                << " has null mesh_resource() - instances referencing this "
                "prototype will not traverse correctly.");
        }
        _pending_blas_build = true;
        _rebuild_bindless_array();
        _bindless_update_needed = true;
        // Refresh instance props so any pre-existing prototype instances pick
        // up potentially-shifted slot indices in the recreated bindless array.
        _instance_props_dirty = true;
    }

    CI_LOG_D("Prototype registered: ShapeId=" << proto_id
        << ", proto_idx=" << proto_idx
        << ", triangles=" << _prototypes.back().shape->triangle_count()
        << ", total_prototypes=" << _prototypes.size()
        << (_built ? " (post-build)" : ""));

    return proto_id;
}

ShapeId Geometry::add_instance(ShapeId prototype_id, const float4x4 &transform,
                               uint32_t material_layers) noexcept {
    return _add_instance_impl(prototype_id, transform, nullptr, nullptr, material_layers);
}

ShapeId Geometry::add_instance(ShapeId prototype_id, const Transform *transform,
                               uint32_t material_layers) noexcept {
    if (transform == nullptr) return kInvalidShapeId;
    return _add_instance_impl(prototype_id, transform->matrix(), transform, nullptr, material_layers);
}

ShapeId Geometry::add_instance(ShapeId prototype_id, luisa::unique_ptr<Transform> transform,
                               uint32_t material_layers) noexcept {
    if (!transform) return kInvalidShapeId;
    return _add_instance_impl(prototype_id, transform->matrix(), nullptr, std::move(transform),
                              material_layers);
}

ShapeId Geometry::_add_instance_impl(ShapeId prototype_id, const float4x4 &initial_world,
                                     const Transform *borrowed,
                                     luisa::unique_ptr<Transform> owned,
                                     uint32_t material_layers) noexcept {
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
    data._transform         = initial_world;
    // Per-instance material layers (B1): an explicit value becomes this
    // instance's own layer pack; the sentinel inherits the prototype's.
    if (material_layers == kInheritMaterialLayers) {
        data._material_layers = proto_shape->material_layers();
    } else {
        if (!_built)
            CI_LOG_W("Geometry::add_instance: per-instance material layers set "
                "before build() - out-of-range layer indices won't be validated "
                "until shade (proto ShapeId=" << prototype_id << ").");
        _validate_material_layers(material_layers, "add_instance");
        data._material_layers = material_layers;
    }
    data._bindless_vert     = _prototypes[proto_idx].vertex_bindless_slot;
    data._bindless_tri      = _prototypes[proto_idx].triangle_bindless_slot;

    // Retained transform (polled route). Borrowed transforms are stored so
    // get_transform() can hand them back; owned ones force registration.
    data._borrowed_transform = borrowed;
    data._owned_transform    = std::move(owned);
    const Transform *xform  = data._owned_transform ? data._owned_transform.get() : borrowed;

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
    _slots[instance_id] = { tlas_index, true, false };
    _tlas_to_shape.push_back(instance_id);

    // Register for per-frame polling (same rule as add_shape; is_static()
    // is chain-aware so a static transform under an animated parent polls).
    if (xform != nullptr && (data._owned_transform || !xform->is_static())) {
        _instanced_transforms.emplace_back(xform, tlas_index);
    }

    // Add to TLAS — only for post-build additions (pre-build handled by build())
    if (_built) {
        const compute::Mesh *mesh = proto_shape->mesh_resource();
        if (mesh) {
            _tlas.emplace_back(*mesh, initial_world, data.visible ? 0xFFu : 0x00u);
        } else {
            CI_LOG_W("Geometry::add_instance post-build: prototype has null "
                "mesh_resource() - TLAS entry skipped, instance will not be "
                "visible in ray traversal (proto ShapeId=" << prototype_id
                << ", inst ShapeId=" << instance_id << ").");
        }
        _instance_props_dirty = true;
        _instance_props_structure_changed = true;
        _instance_transforms_dirty = true;
        _dirty_transform_indices.clear();
        _tlas_needs_rebuild = true;
        _invalidate_gpu_transform_rows("post-build add_instance");
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

void Geometry::add_instances(ShapeId prototype_id,
                             luisa::span<const float4x4> transforms,
                             luisa::span<const uint32_t> material_layers,
                             luisa::vector<ShapeId> &out_ids) noexcept {
    if (!material_layers.empty() && material_layers.size() != transforms.size()) {
        CI_LOG_E("Geometry::add_instances: material_layers size "
            << material_layers.size() << " != transforms size "
            << transforms.size() << " - batch ignored");
        return;
    }
    out_ids.reserve(out_ids.size() + transforms.size());
    for (uint i = 0u; i < transforms.size(); ++i) {
        auto id = add_instance(prototype_id, transforms[i],
            material_layers.empty() ? kInheritMaterialLayers : material_layers[i]);
        out_ids.push_back(id);
    }
}

void Geometry::add_instances(ShapeId prototype_id,
                             luisa::span<const Transform *const> transforms,
                             luisa::vector<ShapeId> &out_ids) noexcept {
    out_ids.reserve(out_ids.size() + transforms.size());
    for (auto *xform : transforms) {
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

void Geometry::_apply_instance_transform(uint tlas_idx, const float4x4 &matrix,
                                         Change change) noexcept {
    auto &inst = _instances[tlas_idx];
    inst._transform = matrix;
    // Size guard: post-build additions leave _instance_transform_cpu
    // undersized until upload_dirty_transforms() resizes it.
    if (tlas_idx < _instance_transform_cpu.size()) {
        _instance_transform_cpu[tlas_idx] = matrix;
        _dirty_transform_indices.push_back(tlas_idx);
    }
    _tlas.set_transform_on_update(tlas_idx, matrix);

    // Track light transform changes for LightSampler re-upload (O(1))
    if (tlas_idx < _is_light_instance.size() && _is_light_instance[tlas_idx]) {
        _lightTransformDirty = true;
        if (change == Change::Scale)
            _lightScaleDirty = true;
    }
}

void Geometry::set_instance_transform(ShapeId id, const float4x4 &matrix,
                                      Change changeHint) noexcept {
    if (!is_valid(id)) return;
    if (!_built) {
        CI_LOG_W("Geometry::set_instance_transform called before build() - transform "
            "will be applied when build() runs, but TLAS ops are queued on "
            "an unbuilt accel (ShapeId=" << id << ").");
    }
    uint tlas_idx = _slots[id].tlas_index;
    auto &inst = _instances[tlas_idx];

    if (_row_is_gpu_owned(tlas_idx)) {
        CI_LOG_W("Geometry::set_instance_transform on GPU-owned row " << tlas_idx
            << " dropped - a device kernel owns the matrix; re-register or "
               "use the CPU path (ShapeId=" << id << ")");
        return;
    }

    _apply_instance_transform(tlas_idx, matrix, changeHint);
    _instance_transforms_dirty = true;
    if (inst.owns_shape()) inst.get_shape()->store_transform(matrix);
    _transform_dirty = true;
}

void Geometry::set_visibility(ShapeId id, bool visible) noexcept {
    if (!is_valid(id)) return;
    if (!_built) {
        // Recorded on the instance; build() applies it to the TLAS mask when
        // the accel is constructed.
        CI_LOG_D("Geometry::set_visibility called before build() - recorded, "
            "applied at TLAS build (ShapeId=" << id << ").");
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

void Geometry::set_camera_visibility(ShapeId id, bool camera_visible) noexcept {
    if (!is_valid(id)) return;
    uint tlas_idx = _slots[id].tlas_index;
    auto &inst = _instances[tlas_idx];
    bool invisible = !camera_visible;
    bool currently_invisible = (inst.properties & PROPERTY_INVISIBLE_TO_CAMERA) != 0u;
    if (currently_invisible == invisible) return;

    inst.properties = invisible
        ? (inst.properties | PROPERTY_INVISIBLE_TO_CAMERA)
        : (inst.properties & ~PROPERTY_INVISIBLE_TO_CAMERA);
    if (auto *shape = inst.get_shape()) {
        shape->set_property_flag(PROPERTY_INVISIBLE_TO_CAMERA, invisible);
    }

    if (tlas_idx < _instance_buffer_cpu.size())
        _instance_buffer_cpu[tlas_idx].x = inst.properties;

    // Camera-path skip is read from the instance props buffer (.x) by the
    // G-buffer / mirror / glass-tint shaders. No TLAS mask or LightSampler
    // change: the shape keeps its visibility and light sampling power.
    _instance_props_dirty = true;
}

bool Geometry::is_camera_visible(ShapeId id) const noexcept {
    return is_valid(id) &&
        (_instances[_slots[id].tlas_index].properties & PROPERTY_INVISIBLE_TO_CAMERA) == 0u;
}

void Geometry::set_material_layers(ShapeId id, uint32_t layers) noexcept {
    if (!is_valid(id)) return;
    if (_slots[id].prototype) {
        CI_LOG_W("Geometry::set_material_layers: ShapeId " << id
            << " names a prototype (no TLAS instance of its own) - ignored");
        return;
    }
    if (!_built) {
        CI_LOG_W("Geometry::set_material_layers called before build() - value "
            "is recorded on the instance; build() will pick it up, but any "
            "out-of-range layer indices won't be validated until shade "
            "(ShapeId=" << id << ").");
    }
    uint tlas_idx = _slots[id].tlas_index;
    auto &inst = _instances[tlas_idx];
    if (inst._material_layers == layers) return;

    // B9: validate layer indices against material pool if available
    _validate_material_layers(layers, "set_material_layers");

    inst._material_layers = layers;
    if (inst.owns_shape()) inst.get_shape()->set_material_layers(layers);
    if (tlas_idx < _instance_buffer_cpu.size())
        _instance_buffer_cpu[tlas_idx].y = layers;
    _instance_props_dirty = true;
    _has_visible_glass_dirty = true;
    _has_active_subsurface_dirty = true;
    // Emission is baked per instance into the LightSampler's triangle
    // records — a layers change on any instance requires the table rebuild.
    // Conservative (any layers change dirties lights): rebuild cost is
    // edit-time only, never per-frame.
    _lights_dirty = true;
}

uint32_t Geometry::material_layers(ShapeId id) const noexcept {
    if (!is_valid(id) || _slots[id].prototype) return 0u;
    return _instances[_slots[id].tlas_index]._material_layers;
}

bool Geometry::set_instance_user_param(ShapeId id, uint slot, luisa::float4 value) noexcept {
    if (slot >= 4u) {
        CI_LOG_W("Geometry::set_instance_user_param: slot " << slot
            << " out of range (0..3) - ignored (ShapeId=" << id << ")");
        return false;
    }
    if (!is_valid(id) || _slots[id].prototype) {
        CI_LOG_W("Geometry::set_instance_user_param: ShapeId " << id
            << " is not a TLAS instance (invalid or prototype) - ignored");
        return false;
    }
    uint tlas_idx = _slots[id].tlas_index;
    _instances[tlas_idx]._user_params[slot] = value;
    _instance_params_dirty = true;
    return true;
}

luisa::float4 Geometry::instance_user_param(ShapeId id, uint slot) const noexcept {
    if (!is_valid(id) || _slots[id].prototype || slot >= 4u)
        return luisa::float4(0.0f);
    return _instances[_slots[id].tlas_index]._user_params[slot];
}

void Geometry::_validate_material_layers(uint32_t layers, const char *ctx) const noexcept {
    if (!_material_pool) return;
    for (int layer = 0; layer < 4; ++layer) {
        uint mat_idx = (layers >> (layer * 8)) & 0xFFu;
        if (mat_idx == 0xFFu) continue;  // sentinel: layer unused
        if (mat_idx >= _material_pool->count()) {
            CI_LOG_W("Geometry::" << ctx << ": layer " << layer
                << " references material index " << mat_idx
                << " which is out of range (pool size "
                << _material_pool->count() << ") - shade will read garbage "
                "data.");
        }
    }
}

bool Geometry::remove_shape(ShapeId id) noexcept {
    if (!is_valid(id)) return false;

    if (!_built) {
        CI_LOG_W("Geometry::remove_shape called before build() - TLAS has no "
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
    _shape_transform_instances.erase(
        std::remove(_shape_transform_instances.begin(),
                    _shape_transform_instances.end(), tlas_idx),
        _shape_transform_instances.end());
    _deformable_instances.erase(
        std::remove(_deformable_instances.begin(),
                    _deformable_instances.end(), tlas_idx),
        _deformable_instances.end());

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
        for (auto &idx : _shape_transform_instances) {
            if (idx == last_idx) {
                idx = tlas_idx;
                break;
            }
        }
        for (auto &idx : _deformable_instances) {
            if (idx == last_idx) {
                idx = tlas_idx;
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
    _invalidate_gpu_transform_rows("remove_shape swap-and-pop");

#ifndef NDEBUG
    // Registry invariants after the swap-and-pop fixups: in-range, unique,
    // each entry actually deformable. A missed remap would silently freeze
    // the wrong instance's per-frame deform poll (TransformTreeTest's churn
    // knob drives this path; remove is rare so the O(k²) check is free).
    for (size_t i = 0; i < _deformable_instances.size(); ++i) {
        const uint di = _deformable_instances[i];
        CI_ASSERT_MSG(di < _instances.size(),
            "remove_shape: deformable registry index out of range");
        CI_ASSERT_MSG(_instances[di].get_shape()->deformable(),
            "remove_shape: deformable registry points at a non-deformable "
            "instance (missed swap-and-pop remap)");
        for (size_t j = i + 1; j < _deformable_instances.size(); ++j)
            CI_ASSERT_MSG(di != _deformable_instances[j],
                "remove_shape: duplicate deformable registry index");
    }
#endif

    CI_LOG_D("Removed ShapeId=" << id << ", remaining_instances=" << static_cast<uint>(_instances.size()));
    return true;
}

//==============================================================================
// GPU-owned transform rows (device-side TLAS transform source)
//==============================================================================

bool Geometry::_row_is_gpu_owned(uint tlas_idx) const noexcept {
    for (const auto &[first, count] : _gpuTransformRanges)
        if (tlas_idx >= first && tlas_idx < first + count)
            return true;
    return false;
}

void Geometry::_invalidate_gpu_transform_rows(const char *why) noexcept {
    if (!_gpuTransformRanges.empty()) {
        CI_LOG_W("Geometry: GPU transform rows invalidated (" << why
            << ") - dense rows reshuffled; the device writer must re-resolve "
               "and re-register before its next write");
        _gpuTransformRanges.clear();
    }
    ++_topologyGeneration;
}

uint Geometry::tlas_index_of(ShapeId id) const noexcept {
    if (id >= _slots.size() || !_slots[id].alive || _slots[id].prototype)
        return ~0u;
    uint row = _slots[id].tlas_index;
    return row < _instances.size() ? row : ~0u;
}

bool Geometry::register_gpu_transform_rows(luisa::span<const ShapeId> ids) noexcept {
    _gpuTransformRanges.clear();
    if (ids.empty()) {
        CI_LOG_W("Geometry::register_gpu_transform_rows: empty id list");
        return false;
    }
    luisa::vector<uint> rows;
    rows.reserve(ids.size());
    for (ShapeId id : ids) {
        uint row = tlas_index_of(id);
        if (row == ~0u) {
            CI_LOG_W("Geometry::register_gpu_transform_rows: ShapeId " << id
                << " does not resolve to a TLAS row (prototype or dead slot)");
            return false;
        }
        rows.push_back(row);
    }
    std::sort(rows.begin(), rows.end());
    const auto uniqueEnd = std::unique(rows.begin(), rows.end());
    if (uniqueEnd != rows.end()) {
        CI_LOG_W("Geometry::register_gpu_transform_rows: duplicate ids - rejected");
        return false;
    }
    for (size_t i = 1; i < rows.size(); ++i) {
        if (rows[i] != rows.front() + i) {
            CI_LOG_W("Geometry::register_gpu_transform_rows: rows ["
                << rows.front() << ".." << rows.back()
                << "] are not one contiguous run - the v1 device-copy API "
                   "takes a single {first, count} range; rejected");
            return false;
        }
    }
    _gpuTransformRanges.emplace_back(rows.front(),
                                     static_cast<uint>(rows.size()));
    CI_LOG_I("Geometry: " << rows.size() << " TLAS rows ["
        << rows.front() << ", " << rows.front() + rows.size()
        << ") now GPU-owned (device writer holds curr+prev)");
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
    if (inst._owned_transform) return inst._owned_transform.get();
    if (inst._borrowed_transform) return const_cast<Transform*>(inst._borrowed_transform);
    // Matrix-only prototype instances share the prototype's internal
    // transform with all siblings — not authoritative, report none.
    if (!inst.owns_shape()) return nullptr;
    return inst.get_shape()->transform();
}

DeformableMesh* Geometry::get_deformable(ShapeId id) noexcept {
    MeshShape *shape = get_shape(id);
    if (!shape || !shape->deformable()) return nullptr;
    return static_cast<DeformableMesh*>(shape);
}

bool Geometry::is_valid(ShapeId id) const noexcept {
    return id < _slots.size() && _slots[id].alive;
}

void Geometry::unload_static_cpu_data() noexcept {
    if (!_built) {
        CI_LOG_W("Geometry::unload_static_cpu_data called before build() - ignored");
        return;
    }
    const auto layers_emissive = [&](uint32_t layers) noexcept {
        if (!_material_pool) return false;
        for (int layer = 0; layer < 4; ++layer) {
            uint mat_idx = (layers >> (layer * 8)) & 0xFFu;
            if (mat_idx == 0xFFu) continue;
            const auto &em = _material_pool->getMaterial(mat_idx).data.emission;
            if (dot(em, make_float3(0.2126f, 0.7152f, 0.0722f)) > 0.001f) return true;
        }
        return false;
    };

    size_t freed = 0u, count = 0u;

    // Owning shapes: unload when non-deformable, non-light, non-emissive.
    for (const auto &inst : _instances) {
        if (!inst.owns_shape()) continue;
        MeshShape *shape = inst.shape.get();
        if (!shape->built() || shape->deformable() || !shape->has_cpu_data()) continue;
        if (shape->properties() & PROPERTY_HAS_LIGHT) continue;
        if (layers_emissive(inst._material_layers)) continue;
        freed += shape->vertices().size() * sizeof(MeshShape::Vertex) +
                 shape->triangles().size() * sizeof(Triangle);
        shape->unloadCPUData();
        ++count;
    }

    // Prototypes: the CPU data serves LightSampler areas for EVERY instance
    // (per-instance layers included), so unload only when no instance —
    // including per-instance emissive overrides — needs it.
    luisa::vector<bool> keep(_prototypes.size(), false);
    for (const auto &inst : _instances) {
        if (inst.owns_shape() || !layers_emissive(inst._material_layers)) continue;
        for (uint p = 0u; p < _prototypes.size(); ++p)
            if (_prototypes[p].shape.get() == inst.shape_ref) { keep[p] = true; break; }
    }
    for (uint p = 0u; p < _prototypes.size(); ++p) {
        MeshShape *shape = _prototypes[p].shape.get();
        if (keep[p] || !shape->built() || shape->deformable() ||
            !shape->has_cpu_data() || (shape->properties() & PROPERTY_HAS_LIGHT))
            continue;
        freed += shape->vertices().size() * sizeof(MeshShape::Vertex) +
                 shape->triangles().size() * sizeof(Triangle);
        shape->unloadCPUData();
        ++count;
    }

    CI_LOG_I("Geometry::unload_static_cpu_data: released ~" << (freed >> 10u)
        << " KB of CPU mesh data across " << count << " meshes");
}

bool Geometry::is_visible(ShapeId id) const noexcept {
    if (!is_valid(id)) return false;
    return _instances[_slots[id].tlas_index].visible;
}

void Geometry::upload_instance_props(Stream &stream) noexcept {
    if (_instance_props_structure_changed) {
        _update_instance_props_buffer();
        _instance_props_structure_changed = false;
        // Dense rows reshuffled (swap-and-pop / append): user-params rows
        // follow the same ordering once the pool-side buffer exists. Before
        // the first authoring the buffer never materialized, so scenes that
        // don't use per-instance data stay at zero cost.
        if (_material_pool && _material_pool->instanceParamsResident())
            _instance_params_dirty = true;
    }
    // Ensure buffer exists and is correct size
    if (!_instance_buffer || _instance_buffer.size() != _instance_buffer_cpu.size())
        _instance_buffer = _device.create_buffer<luisa::uint4>(_instance_buffer_cpu.size());
    stream << _instance_buffer.copy_from(_instance_buffer_cpu.data());
    _instance_props_dirty = false;
}

void Geometry::upload_instance_params(
    newtype::render::MaterialPool &pool, Stream &stream) noexcept {
    if (!_instance_params_dirty || _instances.empty()) {
        _instance_params_dirty = false;
        return;
    }
    // Serialize the dense TLAS-ordered rows (4 float4 per instance).
    luisa::vector<luisa::float4> rows;
    rows.reserve(_instances.size() * 4u);
    for (const auto &inst : _instances)
        for (uint slot = 0u; slot < 4u; slot++)
            rows.push_back(inst._user_params[slot]);
    pool.uploadInstanceParams(stream, rows);
    _instance_params_dirty = false;
}

void Geometry::upload_dirty_transforms(Stream &stream) noexcept {
    if (_instance_transforms_dirty) {
        // Topology changed since the GPU buffers were sized (post-build
        // add_shape/add_instance/remove_shape set _instance_transforms_dirty
        // but clear _dirty_transform_indices, so the per-index path below
        // would upload nothing while the buffers hold stale/undersized data).
        // Resize the CPU mirror (full repopulate — swap-and-pop on removal
        // reshuffles dense indices) and recreate the GPU buffers with a full
        // upload, mirroring build(). GPU-owned rows cannot be active here:
        // every path that desynchronizes the sizes bumps the topology
        // generation, which clears the registration.
        if (_instance_transform_cpu.size() != _instances.size()) {
            _dirty_transform_indices.clear();
            _last_dirty_transform_indices.clear();
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
        // Snapshot prev BEFORE the dirty uploads: at this point curr still
        // holds last frame's transforms. Only indices whose curr is about to
        // change — this frame's AND last frame's movers (the latter get their
        // stop-frame prev fix here; the full-buffer copy that used to run
        // every moving frame repaired them as a side effect) — need the
        // refresh; static instances already have prev == curr. 64B device
        // copy per index, no host staging.
        const bool prevValid = _instance_transform_prev_buffer &&
            _instance_transform_prev_buffer.size() == _instance_transform_buffer.size();
        const size_t n = _instance_transform_cpu.size();
        // With GPU-owned rows active, never whole-buffer-upload curr or
        // whole-buffer-copy prev — that would clobber rows a device kernel
        // wrote this frame (TetCage solve owns both curr and prev there).
        // Per-index uploads naturally skip them (dirty rows are never
        // GPU-owned; owned rows in _last_dirty are stale handoff stragglers
        // filtered below).
        const bool hasGpuRows = has_gpu_transform_rows();
        const bool many = !hasGpuRows && _dirty_transform_indices.size() * 4u >= n;
        const bool lastMany = !hasGpuRows && _last_dirty_transform_indices.size() * 4u >= n;
        if (prevValid) {
            if (many || lastMany) {
                stream << _instance_transform_buffer.copy_to(_instance_transform_prev_buffer);
            } else {
                for (uint idx : _last_dirty_transform_indices)
                    if (!_row_is_gpu_owned(idx))
                        stream << _instance_transform_prev_buffer.view(idx, 1u).copy_from(
                            _instance_transform_buffer.view(idx, 1u));
                for (uint idx : _dirty_transform_indices)
                    if (!_row_is_gpu_owned(idx))
                        stream << _instance_transform_prev_buffer.view(idx, 1u).copy_from(
                            _instance_transform_buffer.view(idx, 1u));
            }
        }
        if (many) {
            // Single full-buffer host upload — _instance_transform_cpu is
            // maintained by set_transform/animated-update paths.
            stream << _instance_transform_buffer.copy_from(_instance_transform_cpu.data());
        } else {
            for (uint idx : _dirty_transform_indices) {
                stream << _instance_transform_buffer.view(idx, 1u).copy_from(
                    &_instance_transform_cpu[idx]);
            }
        }
        _last_dirty_transform_indices = _dirty_transform_indices;
        _dirty_transform_indices.clear();
        _instance_transforms_dirty = false;
        // curr advanced this frame; once motion stops, prev needs one re-sync.
        _transform_prev_stale = true;
    } else if (_transform_prev_stale) {
        // Motion stopped: bring prev back in line with curr so object motion
        // returns to zero instead of persisting as phantom flow. Per-index
        // over the stopped movers when GPU rows are active (their prev is
        // maintained by the device writer; a full copy would freeze its
        // motion for a frame).
        if (_instance_transform_prev_buffer &&
            _instance_transform_prev_buffer.size() == _instance_transform_buffer.size()) {
            if (has_gpu_transform_rows()) {
                for (uint idx : _last_dirty_transform_indices)
                    if (!_row_is_gpu_owned(idx))
                        stream << _instance_transform_prev_buffer.view(idx, 1u).copy_from(
                            _instance_transform_buffer.view(idx, 1u));
            } else {
                stream << _instance_transform_buffer.copy_to(_instance_transform_prev_buffer);
            }
        }
        _transform_prev_stale = false;
        _last_dirty_transform_indices.clear();
    }
}

// Note: Hit processing methods are defined inline in Geometry.h as they
// use LuisaCompute DSL constructs that require JIT compilation.

} // namespace newtype::scene
