//
// Created by Claude on 2026/03/23.
//

#pragma once

#include <luisa/luisa-compute.h>
#include "cinder/gl/gl.h"

namespace newtype::scene {

using namespace luisa;
using compute::float4x4;

class StaticTransform;
typedef luisa::unique_ptr<StaticTransform>   StaticTransPtr;
class AnimatedTransform;
typedef luisa::unique_ptr<AnimatedTransform> AnimTransPtr;

/**
 * @brief Granularity of transform change
 *
 * Used by LightSampler to skip area/power recomputation when only
 * translation or rotation changed (area preserved under rigid transforms).
 */
enum class Change : uint8_t {
    None   = 0,  // No change since last clear
    Affine = 1,  // Translation/rotation only — area preserved
    Scale  = 2,  // Scale changed — area may differ
};

/**
 * @brief Transform base with dirty tracking and hierarchy support
 *
 * A Transform stores its LOCAL matrix (relative to its parent, see
 * set_parent) and computes the WORLD matrix on demand by concatenating the
 * parent chain: world = parent_world * local. The composed world matrix is
 * memoized and invalidated down the chain whenever any ancestor changes.
 *
 * Space conventions (existing API stays world-space):
 *   - matrix()/set_matrix()/set_position()/... operate in WORLD space.
 *     For root transforms world == local, so behavior and cost are
 *     unchanged from the pre-hierarchy API.
 *   - local_matrix()/set_local_*()/local_position()/... operate relative
 *     to the parent and are the O(1) fast path for animation.
 *
 * Dirty propagation: mutating a transform marks itself AND all descendants
 * dirty (with Change escalation), so instances registered anywhere in a
 * subtree re-flush on the next Geometry::update() when an ancestor moves.
 *
 * Lifetime: parents do not own children. The caller keeps transforms alive
 * (unique_ptr or borrowed raw pointer, same rule as Geometry::add_shape).
 * Destroying a transform detaches its children (they become roots) and a
 * destroying parent marks surviving children dirty so dependent instances
 * re-flush. Transforms are not copyable — parenting links would silently
 * diverge from the parent's child list.
 */
class Transform {
protected:
    bool _dirty = false;       // Set to true when transform changes
    Change _change = Change::None;

private:
    Transform *_parent = nullptr;          // Non-owning link up the tree
    luisa::vector<Transform *> _children;  // Non-owning links down the tree
    mutable float4x4 _world_cache = make_float4x4(1.0f);
    mutable bool _world_stale = true;

public:
    Transform() noexcept = default;
    virtual ~Transform() noexcept;

    Transform(const Transform &) = delete;
    Transform &operator=(const Transform &) = delete;
    Transform(Transform &&) = delete;
    Transform &operator=(Transform &&) = delete;

    // --- Hierarchy ---

    /// Reparent this transform (detaches from the current parent). Cycles
    /// are rejected with a warning. Marks the subtree dirty so registered
    /// instances re-flush. Note: instances added with a fully-static
    /// borrowed transform are not polled — parent those before add_shape,
    /// or use the owning overload.
    void set_parent(Transform *parent) noexcept;

    [[nodiscard]] Transform *parent() const noexcept { return _parent; }
    [[nodiscard]] bool is_root() const noexcept { return _parent == nullptr; }
    [[nodiscard]] auto child_count() const noexcept { return _children.size(); }

    // --- Classification (chain-aware) ---

    /// True when neither this transform nor any ancestor can change over
    /// time. Geometry uses this to decide whether an instance must be
    /// polled every frame (a static transform under an animated parent is
    /// NOT static as far as its world matrix is concerned).
    [[nodiscard]] bool is_static() const noexcept {
        return _is_self_static() && (_parent == nullptr || _parent->is_static());
    }

    /// True when the composed world matrix is the identity.
    [[nodiscard]] bool is_identity() const noexcept {
        return _is_self_identity() && (_parent == nullptr || _parent->is_identity());
    }

    // --- Matrices ---

    /// World matrix: parent chain composed with the local matrix.
    /// Memoized until any ancestor or this transform changes.
    [[nodiscard]] float4x4 matrix() const noexcept {
        if (_world_stale) {
            _world_cache = _parent ? _parent->matrix() * local_matrix() : local_matrix();
            _world_stale = false;
        }
        return _world_cache;
    }

    /// Local matrix relative to the parent (the stored representation).
    [[nodiscard]] virtual float4x4 local_matrix() const noexcept = 0;

    // --- Dirty tracking ---

    /// Check if this transform needs update (dirty flag)
    [[nodiscard]] bool is_dirty() const noexcept { return _dirty; }

    /// What kind of change occurred since last clear
    [[nodiscard]] Change change() const noexcept { return _change; }

    /// Clear dirty flag and change tracking (self only; called after the
    /// instance's TLAS update)
    void clear_dirty() noexcept { _dirty = false; _change = Change::None; }

    /// Clear this transform and all its ancestors (post-poll sweep so
    /// group-only transforms without shapes do not stay dirty forever).
    void clear_dirty_chain() noexcept {
        for (Transform *t = this; t != nullptr; t = t->_parent) {
            t->clear_dirty();
        }
    }

    /// Mark as dirty and propagate to all descendants (call when the local
    /// transform changes after _change has been set).
    void mark_dirty() noexcept;

protected:
    [[nodiscard]] virtual bool _is_self_static() const noexcept = 0;
    [[nodiscard]] virtual bool _is_self_identity() const noexcept = 0;

private:
    void _propagate_dirty(Change c) noexcept;
};

/**
 * @brief Static transform (constant local matrix)
 *
 * Simple wrapper around a constant float4x4 matrix stored as the LOCAL
 * matrix. Used for objects that never move.
 */
class StaticTransform : public Transform {
private:
    float4x4 _matrix;   // LOCAL matrix (relative to parent)

    /// Store a local matrix + change classification in one step (so dirty
    /// propagation sees the final Change value).
    void _store_local(const float4x4 &m, Change c) noexcept;

    /// Store a world-space matrix: back-solves the local matrix through
    /// the parent's inverse (identity when unparented). Falls back to
    /// treating the input as local if the parent matrix is singular.
    void _store_world(const float4x4 &world, Change c) noexcept;

public:
    explicit StaticTransform(const float4x4 &local = make_float4x4(1.0f)) noexcept
        : _matrix(local) {}

    static StaticTransPtr create(const float4x4& local = make_float4x4(1.0f)) noexcept {
        return luisa::make_unique<StaticTransform>(local);
    }

    [[nodiscard]] float4x4 local_matrix() const noexcept override { return _matrix; }

    [[nodiscard]] bool _is_self_static() const noexcept override { return true; }

    [[nodiscard]] bool _is_self_identity() const noexcept override {
        return _matrix[0][0] == 1.0f && _matrix[0][1] == 0.0f &&
               _matrix[0][2] == 0.0f && _matrix[0][3] == 0.0f &&
               _matrix[1][0] == 0.0f && _matrix[1][1] == 1.0f &&
               _matrix[1][2] == 0.0f && _matrix[1][3] == 0.0f &&
               _matrix[2][0] == 0.0f && _matrix[2][1] == 0.0f &&
               _matrix[2][2] == 1.0f && _matrix[2][3] == 0.0f &&
               _matrix[3][0] == 0.0f && _matrix[3][1] == 0.0f &&
               _matrix[3][2] == 0.0f && _matrix[3][3] == 1.0f;
    }

    // --- Local-space setters (fast path, operate on the stored matrix) ---

    /// Set the LOCAL matrix (marks dirty)
    void set_local_matrix(const float4x4 &matrix) noexcept {
        _store_local(matrix, Change::Scale);
    }

    /// default r should be make_float4(0.0f, 0.0f, 1.0f, 0.0f)
    void set_local_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s);
    void set_local_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s);

    // change only one comp
    void set_local_position(const luisa::float3& p);
    void set_local_rotation(const luisa::float4& r);
    void set_local_scale   (const luisa::float3& s);
    void set_local_position(const ci::vec3& p);
    void set_local_rotation(const ci::quat& r);
    void set_local_scale   (const ci::vec3& s);

    /// Decompose the LOCAL matrix
    void decompose_local(ci::vec3& p, ci::quat& r, ci::vec3& s);

    // --- World-space setters (existing names; back-solve local when parented) ---

    /// Set the WORLD matrix (marks dirty). When parented, the local matrix
    /// is back-solved via the parent's inverse so the world matches exactly.
    void set_matrix(const float4x4 &world) noexcept { _store_world(world, Change::Scale); }

    /// Set the WORLD matrix from ci::mat4 (glm::mat4) - Cinder compatibility
    void set_matrix(const ci::mat4 &world) noexcept;

    void set_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s);
    void set_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s);

    // change only one comp (world space)
    void set_position(const luisa::float3& p);
    void set_rotation(const luisa::float4& r);
    void set_scale   (const luisa::float3& s);
    void set_position(const ci::vec3& p);
    void set_rotation(const ci::quat& r);
    void set_scale   (const ci::vec3& s);

    /// Decompose the WORLD matrix
    void decompose(ci::vec3& p, ci::quat& r, ci::vec3& s);

    /// Set LOCAL matrix from ci::mat4 (glm::mat4) - Cinder compatibility (marks dirty)
    void set_local_matrix(const ci::mat4 &matrix) noexcept;
};

/**
 * @brief Animated transform with cached local TRS components
 *
 * Stores local position, rotation, scale separately for O(1) change
 * detection. The local matrix is lazily rebuilt from the cached TRS — no
 * glm::decompose on every setter. Registered in Geometry::_instanced_transforms
 * (is_static() == false) so dirty propagation works automatically through
 * Geometry::update().
 *
 * Space conventions: local_position()/set_local_*() are the O(1) animation
 * path. The world-space position()/rotation()/scale()/set_*() accessors are
 * direct field access when unparented and decompose/back-solve through the
 * parent chain otherwise.
 */
class AnimatedTransform : public Transform {
private:
    luisa::float3 _position{0.0f, 0.0f, 0.0f};   // LOCAL TRS
    ci::quat      _rotation{1.0f, 0.0f, 0.0f, 0.0f};
    luisa::float3 _scale{1.0f, 1.0f, 1.0f};
    mutable float4x4 _matrix = make_float4x4(1.0f);  // local matrix cache
    mutable bool     _matrixDirty = false;

    void _rebuild_matrix() const noexcept;

    /// Replace the local TRS from a matrix (single decompose, no marking)
    void _assign_local_from_matrix(const float4x4 &m) noexcept;

    /// Decompose the WORLD matrix (cached TRS are local, so a parented
    /// transform must decompose the composed matrix).
    void _decompose_world(ci::vec3& p, ci::quat& r, ci::vec3& s) const noexcept;

    /// Store a world-space TRS: back-solves the local matrix through the
    /// parent's inverse and re-decomposes it into the cached local TRS.
    void _store_world_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s,
                          Change c) noexcept;

public:
    AnimatedTransform() = default;
    explicit AnimatedTransform(const float4x4 &m) noexcept;
    explicit AnimatedTransform(const ci::mat4 &m) noexcept;

    static AnimTransPtr create() noexcept {
        return luisa::make_unique<AnimatedTransform>();
    }
    static AnimTransPtr create(const float4x4 &m) noexcept {
        return luisa::make_unique<AnimatedTransform>(m);
    }
    static AnimTransPtr create(const ci::mat4 &m) noexcept {
        return luisa::make_unique<AnimatedTransform>(m);
    }

    [[nodiscard]] bool _is_self_static() const noexcept override { return false; }
    [[nodiscard]] bool _is_self_identity() const noexcept override;

    [[nodiscard]] float4x4 local_matrix() const noexcept override;

    // --- Local TRS accessors (O(1), direct field access) ---

    [[nodiscard]] const luisa::float3& local_position() const noexcept { return _position; }
    [[nodiscard]] const ci::quat&      local_rotation() const noexcept { return _rotation; }
    [[nodiscard]] const luisa::float3& local_scale()    const noexcept { return _scale; }

    // Local setters — O(1) change detection via old vs new comparison
    void set_local_position(const luisa::float3& p) noexcept;
    void set_local_position(const ci::vec3& p) noexcept;
    void set_local_rotation(const ci::quat& r) noexcept;
    void set_local_rotation(const luisa::float4& r) noexcept;
    void set_local_scale(const luisa::float3& s) noexcept;
    void set_local_scale(const ci::vec3& s) noexcept;
    void set_local_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) noexcept;
    void set_local_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) noexcept;

    /// Replace the LOCAL TRS from a matrix (decomposes once)
    void set_local_matrix(const float4x4 &m) noexcept;

    // --- World TRS accessors (existing names; by value) ---

    /// WORLD position. Direct field access when unparented; decomposes the
    /// composed world matrix when parented.
    [[nodiscard]] luisa::float3 position() const noexcept;
    [[nodiscard]] ci::quat      rotation() const noexcept;
    [[nodiscard]] luisa::float3 scale()    const noexcept;

    // World setters — O(1) when unparented; back-solve through the parent
    // inverse when parented (decompose round-trip).
    void set_position(const luisa::float3& p) noexcept;
    void set_position(const ci::vec3& p) noexcept;
    void set_rotation(const ci::quat& r) noexcept;
    void set_rotation(const luisa::float4& r) noexcept;
    void set_scale(const luisa::float3& s) noexcept;
    void set_scale(const ci::vec3& s) noexcept;
    void set_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) noexcept;
    void set_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) noexcept;
};

/**
 * @brief Per-instance transform with acceleration structure index
 *
 * Pairs a (possibly null) transform with the corresponding TLAS instance
 * ID. Polled every frame by Geometry::update(): when the transform (or any
 * of its ancestors, via dirty propagation) changed, the composed world
 * matrix is flushed to the TLAS and the instance transform buffers.
 */
struct InstancedTransform {
    const Transform *transform;
    size_t instance_id;

    InstancedTransform(const Transform *t, size_t inst) noexcept
        : transform(t), instance_id(inst) {}

    [[nodiscard]] auto get_instance_id() const noexcept { return instance_id; }
    [[nodiscard]] float4x4 matrix() const noexcept {
        return transform == nullptr ? make_float4x4(1.0f) : transform->matrix();
    }
    /// Check if this instance needs update. Parent mutations propagate the
    /// dirty flag down to descendants, so checking the leaf is sufficient.
    [[nodiscard]] bool is_dirty() const noexcept {
        return transform != nullptr && transform->is_dirty();
    }
};


} // namespace newtype::scene
