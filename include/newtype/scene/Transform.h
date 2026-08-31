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
 * @brief Transform interface with dirty flag tracking
 *
 * Supports static transforms (never changes) and animated transforms
 * (time-varying matrices). The dirty flag indicates when the transform
 * has been modified and needs TLAS update.
 */
class Transform {
protected:
    bool _dirty = false;       // Set to true when transform changes
    Change _change = Change::None;

public:
    virtual ~Transform() = default;

    /// Check if transform is static (does not change over time)
    [[nodiscard]] virtual bool is_static() const noexcept = 0;

    /// Check if transform is identity (no transformation)
    [[nodiscard]] virtual bool is_identity() const noexcept = 0;

    /// Get transform matrix at given time
    [[nodiscard]] virtual float4x4 matrix() const noexcept = 0;

    /// Check if transform needs update (dirty flag)
    [[nodiscard]] bool is_dirty() const noexcept { return _dirty; }

    /// What kind of change occurred since last clear
    [[nodiscard]] Change change() const noexcept { return _change; }

    /// Clear dirty flag and change tracking (called after TLAS update)
    void clear_dirty() noexcept { _dirty = false; _change = Change::None; }

    /// Mark as dirty (call when transform changes)
    void mark_dirty() noexcept { _dirty = true; }
};

/**
 * @brief Static transform (constant matrix)
 *
 * Simple wrapper around a constant float4x4 matrix.
 * Used for objects that never move.
 */
class StaticTransform : public Transform {
private:
    float4x4 _matrix;

public:
    explicit StaticTransform(const float4x4 &matrix = make_float4x4(1.0f)) noexcept
        : _matrix(matrix) {}

    static StaticTransPtr create(const float4x4& matrix = make_float4x4(1.0f)) noexcept {
        return luisa::make_unique<StaticTransform>(matrix);
    }

    [[nodiscard]] bool is_static() const noexcept override { return true; }

    [[nodiscard]] bool is_identity() const noexcept override {
        return _matrix[0][0] == 1.0f && _matrix[0][1] == 0.0f &&
               _matrix[0][2] == 0.0f && _matrix[0][3] == 0.0f &&
               _matrix[1][0] == 0.0f && _matrix[1][1] == 1.0f &&
               _matrix[1][2] == 0.0f && _matrix[1][3] == 0.0f &&
               _matrix[2][0] == 0.0f && _matrix[2][1] == 0.0f &&
               _matrix[2][2] == 1.0f && _matrix[2][3] == 0.0f &&
               _matrix[3][0] == 0.0f && _matrix[3][1] == 0.0f &&
               _matrix[3][2] == 0.0f && _matrix[3][3] == 1.0f;
    }

    [[nodiscard]] float4x4 matrix() const noexcept override {
        return _matrix;
    }

    /// Set matrix from float4x4 (marks dirty)
    void set_matrix(const float4x4 &matrix) noexcept {
        _matrix = matrix;
        _change = Change::Scale;
        mark_dirty();
    }

    /// default r should be make_float4(0.0f, 0.0f, 1.0f, 0.0f)
    void set_trs(const luisa::float3& t, const luisa::float4& r, const luisa::float3& s);
    void set_trs(const ci::vec3& t, const ci::quat& r, const ci::vec3& s);

    // change only one comp
    void set_position(const luisa::float3& p);
    void set_rotation(const luisa::float4& r);
    void set_scale   (const luisa::float3& s);
    void set_position(const ci::vec3& p);
    void set_rotation(const ci::quat& r);
    void set_scale   (const ci::vec3& s);

    void decompose(ci::vec3& p, ci::quat& r, ci::vec3& s);

    /// Set matrix from ci::mat4 (glm::mat4) - Cinder compatibility (marks dirty)
    void set_matrix(const ci::mat4 &matrix) noexcept;
};

/**
 * @brief Animated transform with cached TRS components
 *
 * Stores position, rotation, scale separately for O(1) change detection.
 * Matrix is lazily rebuilt from cached TRS — no glm::decompose on every setter.
 * Registered in Geometry::_instanced_transforms (is_static() == false) so
 * dirty propagation works automatically through Geometry::update().
 */
class AnimatedTransform : public Transform {
private:
    luisa::float3 _position{0.0f, 0.0f, 0.0f};
    ci::quat      _rotation{1.0f, 0.0f, 0.0f, 0.0f};
    luisa::float3 _scale{1.0f, 1.0f, 1.0f};
    mutable float4x4 _matrix = make_float4x4(1.0f);
    mutable bool     _matrixDirty = false;

    void _rebuild_matrix() const noexcept;

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

    [[nodiscard]] bool is_static() const noexcept override { return false; }
    [[nodiscard]] bool is_identity() const noexcept override;

    [[nodiscard]] float4x4 matrix() const noexcept override;

    // Component accessors
    [[nodiscard]] const luisa::float3& position() const noexcept { return _position; }
    [[nodiscard]] const ci::quat&      rotation() const noexcept { return _rotation; }
    [[nodiscard]] const luisa::float3& scale()    const noexcept { return _scale; }

    // Setters — O(1) change detection via old vs new comparison
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
 * @brief Hierarchical transform tree
 *
 * Maintains a stack of transforms for scene graph traversal.
 * Each node has a parent and computes its world-space matrix
 * by concatenating parent transforms.
 *
 * The tree tracks which nodes need updates via dirty flags.
 */
class TransformTree {
public:
    class Node {
    private:
        const Node *_parent;
        const Transform *_transform;

    public:
        Node(const Node *parent, const Transform *t) noexcept
            : _parent(parent), _transform(t) {}

        [[nodiscard]] auto transform() const noexcept { return _transform; }
        [[nodiscard]] float4x4 matrix() const noexcept;
    };

private:
    luisa::vector<luisa::unique_ptr<Node>> _nodes;
    luisa::vector<const Node *> _node_stack;
    luisa::vector<bool> _static_stack;

public:
    TransformTree() = default;

    [[nodiscard]] auto size() const noexcept { return _nodes.size(); }
    [[nodiscard]] auto empty() const noexcept { return _nodes.empty(); }

    /// Push transform onto stack (scene graph traversal)
    void push(const Transform *t) noexcept;

    /// Pop transform from stack
    void pop(const Transform *t) noexcept;

    /// Get leaf node and whether the entire chain is static
    [[nodiscard]] std::pair<const Node *, bool /* is_static */>
        leaf(const Transform *t) noexcept;

    /// Check if any node in current chain is dirty
    [[nodiscard]] bool is_dirty() const noexcept;
};

/**
 * @brief Per-instance transform with acceleration structure index
 *
 * Stores a transform tree node and the corresponding TLAS instance ID.
 * Used for efficient per-frame transform updates.
 */
struct InstancedTransform {
    const TransformTree::Node *node;
    size_t instance_id;

    InstancedTransform(const TransformTree::Node *n, size_t inst) noexcept
        : node(n), instance_id(inst) {}

    [[nodiscard]] auto get_instance_id() const noexcept { return instance_id; }
    [[nodiscard]] auto matrix() const noexcept {
        return node == nullptr ? make_float4x4(1.0f) : node->matrix();
    }

    /// Check if this instance needs update
    [[nodiscard]] bool is_dirty() const noexcept;
};


} // namespace newtype::scene
