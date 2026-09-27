//
// Created by Claude on 2026/03/23.
//

#include "newtype/scene/Transform.h"
#include "newtype/util/TypeConv.h"
#include "cinder/Log.h"
#include <glm/gtx/matrix_decompose.hpp>
#include <algorithm>
#include <cmath>

namespace newtype::scene {

//==============================================================================
// Transform: hierarchy + dirty propagation
//==============================================================================

Transform::~Transform() noexcept {
    // Unlink from parent (parents never own children).
    if (_parent != nullptr) {
        auto &siblings = _parent->_children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), this),
                       siblings.end());
    }
    // Detach children: they survive as roots. Their world matrices change,
    // so mark their subtrees dirty to re-flush any registered instances.
    for (Transform *child : _children) {
        child->_parent = nullptr;
        child->_world_stale = true;
        child->mark_dirty();
    }
}

void Transform::set_parent(Transform *parent) noexcept {
    if (parent == this) {
        CI_LOG_W("Transform::set_parent: transform cannot be its own parent");
        return;
    }
    if (parent == _parent) return;
    if (parent != nullptr) {
        // Reject cycles: walk the prospective ancestor chain.
        for (auto *ancestor = parent; ancestor != nullptr; ancestor = ancestor->_parent) {
            if (ancestor == this) {
                CI_LOG_W("Transform::set_parent: cycle rejected "
                         "(transform is an ancestor of the new parent)");
                return;
            }
        }
    }
    if (_parent != nullptr) {
        auto &siblings = _parent->_children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), this),
                       siblings.end());
    }
    _parent = parent;
    if (_parent != nullptr) _parent->_children.push_back(this);
    // The composed world matrix changed: re-mark this subtree so every
    // registered instance below re-flushes on the next update.
    mark_dirty();
}

void Transform::mark_dirty() noexcept {
    _dirty = true;
    _world_stale = true;
    _propagate_dirty(_change);
}

void Transform::_propagate_dirty(Change c) noexcept {
    for (Transform *child : _children) {
        child->_dirty = true;
        if (c > child->_change) child->_change = c;
        child->_world_stale = true;
        child->_propagate_dirty(c);
    }
}

namespace {

/// Back-solve a world matrix into parent-local space. Returns the input
/// unchanged when unparented, or when the parent matrix is singular (e.g.
/// zero scale) — in that case the caller's intent is unknowable, so the
/// value is stored as local rather than as NaNs (warns once).
float4x4 world_to_local(const Transform *self, const float4x4 &world) noexcept {
    const Transform *parent = self->parent();
    if (parent == nullptr) return world;
    float4x4 inv = luisa::inverse(parent->matrix());
    const bool finite = std::isfinite(inv.cols[0].x) && std::isfinite(inv.cols[1].y) &&
                        std::isfinite(inv.cols[2].z) && std::isfinite(inv.cols[3].w);
    if (!finite) {
        static bool warned = false;
        if (!warned) {
            CI_LOG_W("Transform: singular parent matrix cannot back-solve a "
                     "world-space input - storing it as local instead");
            warned = true;
        }
        return world;
    }
    return inv * world;
}

} // namespace

//==============================================================================
// StaticTransform
//==============================================================================

void StaticTransform::_store_local(const float4x4 &m, Change c) noexcept {
    _matrix = m;
    _change = c;
    mark_dirty();
}

void StaticTransform::_store_world(const float4x4 &world, Change c) noexcept {
    _store_local(world_to_local(this, world), c);
}

void StaticTransform::set_matrix(const ci::mat4 &world) noexcept {
    // ci::mat4 (glm, column-access) and float4x4 share the column-major
    // element convention — a straight copy is exact.
    _store_world(tolc(world), Change::Scale);
}

void StaticTransform::set_local_matrix(const ci::mat4 &matrix) noexcept {
    set_local_matrix(tolc(matrix));
}

void StaticTransform::set_local_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose_local(pos, rot, scl);

    _store_local(luisa::translation(t) *
              luisa::rotation(normalize(r.xyz()), radians(r.w)) *
              luisa::scaling(s),
              glm::length2(scl - toci(s)) > .0001f ? Change::Scale : Change::Affine);
}

void StaticTransform::set_local_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose_local(pos, rot, scl);

    _store_local(luisa::translation(tolc(t)) *
        luisa::rotation (normalize(make_float3(r.x,r.y,r.z)), r.w) *
        luisa::scaling  (tolc(s)),
        glm::length2(scl - s) > .0001f ? Change::Scale : Change::Affine);
}

void StaticTransform::set_local_position(const luisa::float3& p) {
    set_local_position(toci(p));
}

void StaticTransform::set_local_rotation(const luisa::float4& r) {
    set_local_rotation(glm::quat(r.x, r.y, r.z, r.w));
}

void StaticTransform::set_local_scale(const luisa::float3& s) {
    set_local_scale(toci(s));
}

void StaticTransform::decompose_local(ci::vec3& p, ci::quat& r, ci::vec3& s) {
    auto tmp = toci(_matrix);

    glm::vec3 t, skew;
    glm::vec4 perspective;
    glm::decompose(tmp, s, r, p, skew, perspective);
}

void StaticTransform::set_local_position(const ci::vec3& p) {
    glm::vec3 s, pos;
    glm::quat rot;
    decompose_local(pos, rot, s);

    _store_local(tolc(glm::translate(p)
            * glm::mat4_cast(rot)
            * glm::scale(s)), Change::Affine);
}

void StaticTransform::set_local_rotation(const ci::quat& r) {
    glm::vec3 s, pos;
    glm::quat rot;
    decompose_local(pos, rot, s);

    _store_local(tolc(glm::translate(pos)
            * glm::mat4_cast(r)
            * glm::scale(s)), Change::Affine);
}

void StaticTransform::set_local_scale(const ci::vec3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose_local(pos, rot, scl);

    _store_local(tolc  (glm::translate(pos)
        * glm::mat4_cast(rot)
        * glm::scale(s)), Change::Scale);
}

// --- World-space setters (back-solve local through the parent inverse) ---

void StaticTransform::set_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose(pos, rot, scl);

    auto world = luisa::translation(t) *
              luisa::rotation(normalize(r.xyz()), radians(r.w)) *
              luisa::scaling(s);
    _store_world(world, glm::length2(scl - toci(s)) > .0001f ? Change::Scale : Change::Affine);
}

void StaticTransform::set_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose(pos, rot, scl);

    auto world = luisa::translation(tolc(t)) *
        luisa::rotation (normalize(make_float3(r.x,r.y,r.z)), r.w) *
        luisa::scaling  (tolc(s));
    _store_world(world, glm::length2(scl - s) > .0001f ? Change::Scale : Change::Affine);
}

void StaticTransform::set_position(const luisa::float3& p) {
    set_position(toci(p));
}

void StaticTransform::set_rotation(const luisa::float4& r) {
    set_rotation(glm::quat(r.x, r.y, r.z, r.w));
}

void StaticTransform::set_scale(const luisa::float3& s) {
    set_scale(toci(s));
}

void StaticTransform::decompose(ci::vec3& p, ci::quat& r, ci::vec3& s) {
    auto tmp = toci(matrix());   // world matrix (parent chain composed)

    glm::vec3 t, skew;
    glm::vec4 perspective;
    glm::decompose(tmp, s, r, p, skew, perspective);
}

void StaticTransform::set_position(const ci::vec3& p) {
    glm::vec3 s, pos;
    glm::quat rot;
    decompose(pos, rot, s);

    _store_world(tolc(glm::translate(p)
            * glm::mat4_cast(rot)
            * glm::scale(s)), Change::Affine);
}

void StaticTransform::set_rotation(const ci::quat& r) {
    glm::vec3 s, pos;
    glm::quat rot;
    decompose(pos, rot, s);

    _store_world(tolc(glm::translate(pos)
            * glm::mat4_cast(r)
            * glm::scale(s)), Change::Affine);
}

void StaticTransform::set_scale(const ci::vec3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose(pos, rot, scl);

    _store_world(tolc  (glm::translate(pos)
        * glm::mat4_cast(rot)
        * glm::scale(s)), Change::Scale);
}

//==============================================================================
// AnimatedTransform
//==============================================================================

void AnimatedTransform::_rebuild_matrix() const noexcept {
    auto t = glm::translate(ci::vec3(_position.x, _position.y, _position.z));
    auto r = glm::mat4_cast(_rotation);
    auto s = glm::scale(ci::vec3(_scale.x, _scale.y, _scale.z));
    _matrix = tolc(t * r * s);
    _matrixDirty = false;
}

/// Replace the local TRS from a matrix (single decompose, no dirty marking)
void AnimatedTransform::_assign_local_from_matrix(const float4x4 &m) noexcept {
    _matrix = m;
    _matrixDirty = false;
    ci::vec3 p, s;
    ci::quat r;
    glm::vec3 skew;
    glm::vec4 persp;
    glm::decompose(toci(m), s, r, p, skew, persp);
    _position = make_float3(p.x, p.y, p.z);
    _rotation = r;
    _scale = make_float3(s.x, s.y, s.z);
}

AnimatedTransform::AnimatedTransform(const float4x4 &m) noexcept {
    _assign_local_from_matrix(m);
}

AnimatedTransform::AnimatedTransform(const ci::mat4 &m) noexcept
    : AnimatedTransform(tolc(m)) {}

bool AnimatedTransform::_is_self_identity() const noexcept {
    return _position.x == 0.0f && _position.y == 0.0f && _position.z == 0.0f &&
           _rotation.w == 1.0f && _rotation.x == 0.0f && _rotation.y == 0.0f && _rotation.z == 0.0f &&
           _scale.x == 1.0f && _scale.y == 1.0f && _scale.z == 1.0f;
}

float4x4 AnimatedTransform::local_matrix() const noexcept {
    if (_matrixDirty) _rebuild_matrix();
    return _matrix;
}

void AnimatedTransform::_decompose_world(ci::vec3& p, ci::quat& r, ci::vec3& s) const noexcept {
    if (parent() == nullptr) {
        p = ci::vec3(_position.x, _position.y, _position.z);
        r = _rotation;
        s = ci::vec3(_scale.x, _scale.y, _scale.z);
        return;
    }
    glm::vec3 t, scl, skew;
    glm::vec4 persp;
    glm::decompose(toci(matrix()), scl, r, p, skew, persp);
    s = scl;
}

void AnimatedTransform::_store_world_trs(const ci::vec3& t, const ci::quat& r,
                                         const ci::vec3& s, Change c) noexcept {
    if (parent() == nullptr) {
        _position = make_float3(t.x, t.y, t.z);
        _rotation = r;
        _scale = make_float3(s.x, s.y, s.z);
    } else {
        auto world = tolc(glm::translate(t) * glm::mat4_cast(r) * glm::scale(s));
        _assign_local_from_matrix(world_to_local(this, world));
    }
    _matrixDirty = true;
    _change = c;
    mark_dirty();
}

// --- World TRS accessors (by value; decompose when parented) ---

luisa::float3 AnimatedTransform::position() const noexcept {
    if (parent() == nullptr) return _position;
    ci::vec3 p; ci::quat r; ci::vec3 s;
    _decompose_world(p, r, s);
    return make_float3(p.x, p.y, p.z);
}

ci::quat AnimatedTransform::rotation() const noexcept {
    if (parent() == nullptr) return _rotation;
    ci::vec3 p; ci::quat r; ci::vec3 s;
    _decompose_world(p, r, s);
    return r;
}

luisa::float3 AnimatedTransform::scale() const noexcept {
    if (parent() == nullptr) return _scale;
    ci::vec3 p; ci::quat r; ci::vec3 s;
    _decompose_world(p, r, s);
    return make_float3(s.x, s.y, s.z);
}

// --- World setters (O(1) when root, back-solve when parented) ---

void AnimatedTransform::set_position(const luisa::float3& p) noexcept {
    if (parent() == nullptr) {
        if (_position.x == p.x && _position.y == p.y && _position.z == p.z) return;
        _position = p;
        _matrixDirty = true;
        _change = Change::Affine;
        mark_dirty();
        return;
    }
    ci::vec3 wp; ci::quat wr; ci::vec3 ws;
    _decompose_world(wp, wr, ws);
    _store_world_trs(toci(p), wr, ws, Change::Affine);
}

void AnimatedTransform::set_position(const ci::vec3& p) noexcept {
    set_position(make_float3(p.x, p.y, p.z));
}

void AnimatedTransform::set_rotation(const ci::quat& r) noexcept {
    if (parent() == nullptr) {
        if (_rotation.x == r.x && _rotation.y == r.y && _rotation.z == r.z && _rotation.w == r.w) return;
        _rotation = r;
        _matrixDirty = true;
        _change = Change::Affine;
        mark_dirty();
        return;
    }
    ci::vec3 wp; ci::quat wr; ci::vec3 ws;
    _decompose_world(wp, wr, ws);
    _store_world_trs(wp, r, ws, Change::Affine);
}

void AnimatedTransform::set_rotation(const luisa::float4& r) noexcept {
    set_rotation(ci::quat(r.x, r.y, r.z, r.w));
}

void AnimatedTransform::set_scale(const luisa::float3& s) noexcept {
    if (parent() == nullptr) {
        if (_scale.x == s.x && _scale.y == s.y && _scale.z == s.z) return;
        _change = Change::Scale;
        _scale = s;
        _matrixDirty = true;
        mark_dirty();
        return;
    }
    ci::vec3 wp; ci::quat wr; ci::vec3 ws;
    _decompose_world(wp, wr, ws);
    _store_world_trs(wp, wr, toci(s), Change::Scale);
}

void AnimatedTransform::set_scale(const ci::vec3& s) noexcept {
    set_scale(make_float3(s.x, s.y, s.z));
}

void AnimatedTransform::set_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) noexcept {
    set_trs(toci(t), ci::quat(r.x, r.y, r.z, r.w), toci(s));
}

void AnimatedTransform::set_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) noexcept {
    if (parent() == nullptr) {
        bool scaleChanged = _scale.x != s.x || _scale.y != s.y || _scale.z != s.z;
        _position = make_float3(t.x, t.y, t.z);
        _rotation = r;
        _scale = make_float3(s.x, s.y, s.z);
        _matrixDirty = true;
        _change = scaleChanged ? Change::Scale : Change::Affine;
        mark_dirty();
        return;
    }
    ci::vec3 wp; ci::quat wr; ci::vec3 ws;
    _decompose_world(wp, wr, ws);
    _store_world_trs(t, r, s, glm::length2(ws - s) > .0001f ? Change::Scale : Change::Affine);
}

// --- Local setters (O(1) compare-and-set on the cached TRS) ---

void AnimatedTransform::set_local_position(const luisa::float3& p) noexcept {
    if (_position.x == p.x && _position.y == p.y && _position.z == p.z) return;
    _position = p;
    _matrixDirty = true;
    _change = Change::Affine;
    mark_dirty();
}

void AnimatedTransform::set_local_position(const ci::vec3& p) noexcept {
    set_local_position(make_float3(p.x, p.y, p.z));
}

void AnimatedTransform::set_local_rotation(const ci::quat& r) noexcept {
    if (_rotation.x == r.x && _rotation.y == r.y && _rotation.z == r.z && _rotation.w == r.w) return;
    _rotation = r;
    _matrixDirty = true;
    _change = Change::Affine;
    mark_dirty();
}

void AnimatedTransform::set_local_rotation(const luisa::float4& r) noexcept {
    set_local_rotation(ci::quat(r.x, r.y, r.z, r.w));
}

void AnimatedTransform::set_local_scale(const luisa::float3& s) noexcept {
    if (_scale.x == s.x && _scale.y == s.y && _scale.z == s.z) return;
    _change = Change::Scale;
    _scale = s;
    _matrixDirty = true;
    mark_dirty();
}

void AnimatedTransform::set_local_scale(const ci::vec3& s) noexcept {
    set_local_scale(make_float3(s.x, s.y, s.z));
}

void AnimatedTransform::set_local_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) noexcept {
    set_local_trs(toci(t), ci::quat(r.x, r.y, r.z, r.w), toci(s));
}

void AnimatedTransform::set_local_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) noexcept {
    bool scaleChanged = _scale.x != s.x || _scale.y != s.y || _scale.z != s.z;
    _position = make_float3(t.x, t.y, t.z);
    _rotation = r;
    _scale = make_float3(s.x, s.y, s.z);
    _matrixDirty = true;
    _change = scaleChanged ? Change::Scale : Change::Affine;
    mark_dirty();
}

void AnimatedTransform::set_local_matrix(const float4x4 &m) noexcept {
    _assign_local_from_matrix(m);
    _change = Change::Scale;
    mark_dirty();
}

} // namespace newtype::scene
