//
// Created by Claude on 2026/03/23.
//

#include "newtype/scene/Transform.h"
#include "newtype/util/TypeConv.h"
#include <glm/gtx/matrix_decompose.hpp>

namespace newtype::scene {

//==============================================================================
// StaticTransform
//==============================================================================

void StaticTransform::set_matrix(const ci::mat4 &ci_mat) noexcept {
    // Convert ci::mat4 (glm::mat4, column-major) to luisa::float4x4 (row-major)
    // GLM stores as columns: mat4[col][row]
    // luisa::float4x4 stores as rows
    _matrix = make_float4x4(
        make_float4(ci_mat[0][0], ci_mat[1][0], ci_mat[2][0], ci_mat[3][0]),  // row 0
        make_float4(ci_mat[0][1], ci_mat[1][1], ci_mat[2][1], ci_mat[3][1]),  // row 1
        make_float4(ci_mat[0][2], ci_mat[1][2], ci_mat[2][2], ci_mat[3][2]),  // row 2
        make_float4(ci_mat[0][3], ci_mat[1][3], ci_mat[2][3], ci_mat[3][3])   // row 3
    );
    _change = Change::Scale;
    mark_dirty();
}

void StaticTransform::set_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose(pos, rot, scl);

    _matrix = luisa::translation(t) *
              luisa::rotation(normalize(r.xyz()), radians(r.w)) *
              luisa::scaling(s);
    _change = glm::length2(scl - toci(s)) > .0001f ? Change::Scale : Change::Affine;
    mark_dirty();
}

void StaticTransform::set_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose(pos, rot, scl);

    _matrix = luisa::translation(tolc(t)) *
        luisa::rotation (normalize(make_float3(r.x,r.y,r.z)), r.w) *
        luisa::scaling  (tolc(s));
    _change = glm::length2(scl - s) > .0001f ? Change::Scale : Change::Affine;
    mark_dirty();
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
    auto tmp = toci(_matrix);

    glm::vec3 t, skew;
    glm::vec4 perspective;
    glm::decompose(tmp, s, r, p, skew, perspective);
}

void StaticTransform::set_position(const ci::vec3& p) {
    glm::vec3 s, pos;
    glm::quat rot;
    decompose(pos, rot, s);

    _matrix = tolc(glm::translate(p)
            * glm::mat4_cast(rot)
            * glm::scale(s));
    _change = Change::Affine;
    mark_dirty();
}

void StaticTransform::set_rotation(const ci::quat& r) {
    glm::vec3 s, pos;
    glm::quat rot;
    decompose(pos, rot, s);

    _matrix = tolc(glm::translate(pos)
            * glm::mat4_cast(r)
            * glm::scale(s));
    _change = Change::Affine;
    mark_dirty();
}

void StaticTransform::set_scale(const ci::vec3& s) {
    glm::vec3 scl, pos;
    glm::quat rot;
    decompose(pos, rot, scl);

    _matrix = tolc  (glm::translate(pos)
        * glm::mat4_cast(rot)
        * glm::scale(s));
    _change = Change::Scale;
    mark_dirty();
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

AnimatedTransform::AnimatedTransform(const float4x4 &m) noexcept
    : _matrix(m) {
    ci::vec3 p, scl, skew;
    ci::vec4 persp;
    ci::quat rot;
    glm::decompose(toci(m), scl, rot, p, skew, persp);
    _position = make_float3(p.x, p.y, p.z);
    _rotation = rot;
    _scale = make_float3(scl.x, scl.y, scl.z);
}

AnimatedTransform::AnimatedTransform(const ci::mat4 &m) noexcept
    : AnimatedTransform(tolc(m)) {}

bool AnimatedTransform::is_identity() const noexcept {
    return _position.x == 0.0f && _position.y == 0.0f && _position.z == 0.0f &&
           _rotation.w == 1.0f && _rotation.x == 0.0f && _rotation.y == 0.0f && _rotation.z == 0.0f &&
           _scale.x == 1.0f && _scale.y == 1.0f && _scale.z == 1.0f;
}

float4x4 AnimatedTransform::matrix() const noexcept {
    if (_matrixDirty) _rebuild_matrix();
    return _matrix;
}

void AnimatedTransform::set_position(const luisa::float3& p) noexcept {
    if (_position.x == p.x && _position.y == p.y && _position.z == p.z) return;
    _position = p;
    _matrixDirty = true;
    _change = Change::Affine;
    mark_dirty();
}

void AnimatedTransform::set_position(const ci::vec3& p) noexcept {
    set_position(make_float3(p.x, p.y, p.z));
}

void AnimatedTransform::set_rotation(const ci::quat& r) noexcept {
    if (_rotation.x == r.x && _rotation.y == r.y && _rotation.z == r.z && _rotation.w == r.w) return;
    _rotation = r;
    _matrixDirty = true;
    _change = Change::Affine;
    mark_dirty();
}

void AnimatedTransform::set_rotation(const luisa::float4& r) noexcept {
    set_rotation(ci::quat(r.x, r.y, r.z, r.w));
}

void AnimatedTransform::set_scale(const luisa::float3& s) noexcept {
    if (_scale.x == s.x && _scale.y == s.y && _scale.z == s.z) return;
    _change = Change::Scale;
    _scale = s;
    _matrixDirty = true;
    mark_dirty();
}

void AnimatedTransform::set_scale(const ci::vec3& s) noexcept {
    set_scale(make_float3(s.x, s.y, s.z));
}

void AnimatedTransform::set_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s) noexcept {
    bool scaleChanged = _scale.x != s.x || _scale.y != s.y || _scale.z != s.z;
    _position = t;
    _rotation = ci::quat(r.x, r.y, r.z, r.w);
    _scale = s;
    _matrixDirty = true;
    _change = scaleChanged ? Change::Scale : Change::Affine;
    mark_dirty();
}

void AnimatedTransform::set_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s) noexcept {
    bool scaleChanged = _scale.x != s.x || _scale.y != s.y || _scale.z != s.z;
    _position = make_float3(t.x, t.y, t.z);
    _rotation = r;
    _scale = make_float3(s.x, s.y, s.z);
    _matrixDirty = true;
    _change = scaleChanged ? Change::Scale : Change::Affine;
    mark_dirty();
}

//==============================================================================
// TransformTree::Node
//==============================================================================

float4x4 TransformTree::Node::matrix() const noexcept {
    if (_transform == nullptr) return make_float4x4(1.0f);
    auto local = _transform->matrix();
    if (_parent == nullptr) return local;
    auto parent_matrix = _parent->matrix();
    return parent_matrix * local;
}

//==============================================================================
// TransformTree
//==============================================================================

void TransformTree::push(const Transform *t) noexcept {
    auto parent = _node_stack.empty() ? nullptr : _node_stack.back();
    auto is_static = _static_stack.empty() ? true :
                    (_static_stack.back() && t->is_static());
    _nodes.emplace_back    (luisa::make_unique<Node>(parent, t));
    _node_stack.push_back  (_nodes.back().get());
    _static_stack.push_back(is_static);
}

void TransformTree::pop(const Transform *t) noexcept {
    _node_stack.pop_back  ();
    _static_stack.pop_back();
}

std::pair<const TransformTree::Node *, bool>
TransformTree::leaf(const Transform *t) noexcept {
    auto node = _node_stack.empty() ? nullptr : _node_stack.back();
    auto is_static = _static_stack.empty() ? true : _static_stack.back();
    return {node, is_static};
}

bool TransformTree::is_dirty() const noexcept {
    // Check if any transform in current chain is dirty
    for (auto *node : _node_stack) {
        if (node->transform() && node->transform()->is_dirty()) {
            return true;
        }
    }
    return false;
}

//==============================================================================
// InstancedTransform
//==============================================================================

bool InstancedTransform::is_dirty() const noexcept {
    if (node == nullptr || node->transform() == nullptr) return false;
    return node->transform()->is_dirty();
}

} // namespace newtype::scene
