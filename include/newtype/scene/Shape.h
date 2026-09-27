//
// Created by Claude on 2026/03/23.
//

#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/dsl/syntax.h>
#include "newtype/util/Vertex.h"

namespace newtype::scene {

using namespace luisa;
using compute::AccelOption;
using compute::Expr;
using compute::Float;
using compute::Float2;
using compute::Float3;
using compute::UInt;
using compute::Triangle;

using namespace luisa;
using compute::AccelOption;
using compute::Triangle;

/**
 * @brief View into mesh geometry data
 *
 * Non-owning reference to vertex and triangle arrays.
 * Used for mesh upload and caching.
 */
struct MeshView {
    luisa::span<const util::Vertex> vertices;
    luisa::span<const Triangle> triangles;
};

/**
 * @brief Shape property flags
 *
 * Bit flags stored in Shape::Handle for shader-side queries.
 */
enum ShapeProperty : uint {
    PROPERTY_HAS_VERTEX_NORMAL   = 1u << 0u,  // Mesh has per-vertex normals
    PROPERTY_HAS_VERTEX_UV       = 1u << 1u,  // Mesh has texture coordinates
    PROPERTY_HAS_VERTEX_TANGENT  = 1u << 7u,  // Mesh has per-vertex tangents
    PROPERTY_HAS_SURFACE         = 1u << 2u,  // Shape has material
    PROPERTY_HAS_LIGHT           = 1u << 3u,  // Shape is emissive
    PROPERTY_HAS_MEDIUM          = 1u << 4u,  // Shape has participating medium
    PROPERTY_MAYBE_NON_OPAQUE    = 1u << 5u,  // Shape may have alpha testing
    PROPERTY_IS_DYNAMIC          = 1u << 6u,  // Shape transforms per-frame
    PROPERTY_DOUBLE_SIDED        = 1u << 8u,  // Use face normals, render both sides
    // Illuminates without rendering: camera-path rays (G-buffer primary,
    // mirror reflections, glass tint replay) pass through the shape, while
    // light sampling / shadow / GI rays still hit it. Used for light shapes
    // that should not appear as visible geometry.
    // NOTE: bit 9 is the LAST free bit — the 10-bit property field of
    // Shape::Handle is now full; further flags need a wider field.
    PROPERTY_INVISIBLE_TO_CAMERA = 1u << 9u,
};

/**
 * @brief Shape base class
 *
 * Abstract interface for all renderable geometry types:
 * - Triangle meshes
 * - Curves (future)
 * - Subdivision surfaces (future)
 */
class Shape {
public:
    class Handle;  // Forward declaration

protected:
    uint _properties = 0;  // Property flags

public:
    virtual ~Shape() = default;

    /// Check if shape is a triangle mesh
    [[nodiscard]] virtual bool is_mesh() const noexcept { return false; }

    /// Get mesh view (only valid if is_mesh() == true)
    [[nodiscard]] virtual MeshView mesh() const noexcept {
        return MeshView{{}, {}};
    }

    /// Get property flags
    [[nodiscard]] uint properties() const noexcept { return _properties; }

    /// Set/clear a property flag at runtime (e.g. PROPERTY_INVISIBLE_TO_CAMERA).
    /// CPU-side mirror only; the GPU-visible instance buffer is owned by Geometry.
    void set_property_flag(uint flag, bool value) noexcept {
        _properties = value ? (_properties | flag) : (_properties & ~flag);
    }

    /// Check specific properties
    [[nodiscard]] bool has_vertex_normal() const noexcept {
        return (_properties & PROPERTY_HAS_VERTEX_NORMAL) != 0;
    }
    [[nodiscard]] bool has_vertex_uv() const noexcept {
        return (_properties & PROPERTY_HAS_VERTEX_UV) != 0;
    }
    [[nodiscard]] bool is_dynamic() const noexcept {
        return (_properties & PROPERTY_IS_DYNAMIC) != 0;
    }

    /// Check if shape is deformable (vertex changes per-frame)
    [[nodiscard]] virtual bool deformable() const noexcept { return false; }
};

/**
 * @brief GPU-accessible shape handle
 *
 * Packed encoding of all shader-accessible shape data.
 * Passed by value in kernels for efficient access.
 *
 * Encoding format (uint4):
 * - x: buffer_base | (properties << 22)
 * - y: (surface_tag << 20) | light_tag
 * - z: (medium_tag << 20) | triangle_count
 * - w: (shadow_terminator << 16) | intersection_offset
 *
 * Buffer layout (bindless array indices):
 * - buffer_base + 0: vertex buffer
 * - buffer_base + 1: triangle buffer
 * - buffer_base + 2: alias table (for light sampling, future)
 * - buffer_base + 3: PDF buffer (for light sampling, future)
 */
class Shape::Handle {
private:
    static constexpr auto property_flag_bits = 10u;
    static constexpr auto property_flag_mask = (1u << property_flag_bits) - 1u;
    static constexpr auto buffer_base_max = (1u << (32u - property_flag_bits)) - 1u;

    static constexpr auto light_tag_bits = 12u;
    static constexpr auto surface_tag_bits = 12u;
    static constexpr auto medium_tag_bits = 32u - light_tag_bits - surface_tag_bits;

    static constexpr auto surface_tag_offset = light_tag_bits;
    static constexpr auto medium_tag_offset = surface_tag_offset + surface_tag_bits;

    static constexpr auto vertex_buffer_id_offset = 0u;
    static constexpr auto triangle_buffer_id_offset = 1u;
    static constexpr auto alias_table_buffer_id_offset = 2u;
    static constexpr auto pdf_buffer_id_offset = 3u;

private:
    UInt _buffer_base;
    UInt _properties;
    UInt _surface_tag;
    UInt _light_tag;
    UInt _medium_tag;
    UInt _triangle_count;
    Float _shadow_terminator;
    Float _intersection_offset;

private:
    Handle(Expr<uint> buffer_base, Expr<uint> properties,
           Expr<uint> surface_tag, Expr<uint> light_tag, Expr<uint> medium_tag,
           Expr<uint> triangle_count,
           Expr<float> shadow_terminator, Expr<float> intersection_offset) noexcept
        : _buffer_base(buffer_base), _properties(properties),
          _surface_tag(surface_tag), _light_tag(light_tag), _medium_tag(medium_tag),
          _triangle_count(triangle_count),
          _shadow_terminator(shadow_terminator),
          _intersection_offset(intersection_offset) {}

public:
    Handle() noexcept = default;

    /// Encode CPU data into packed uint4
    [[nodiscard]] static luisa::uint4 encode(
        uint buffer_base, uint properties,
        uint surface_tag, uint light_tag, uint medium_tag,
        uint triangle_count,
        float shadow_terminator, float intersection_offset) noexcept;

    /// Decode packed uint4 into Handle
    [[nodiscard]] static Shape::Handle decode(Expr<luisa::uint4> compressed) noexcept;

public: // Shader accessors
    [[nodiscard]] auto geometry_buffer_base() const noexcept { return _buffer_base; }
    [[nodiscard]] auto property_flags() const noexcept { return _properties; }
    [[nodiscard]] auto vertex_buffer_id() const noexcept {
        return geometry_buffer_base() + vertex_buffer_id_offset;
    }
    [[nodiscard]] auto triangle_buffer_id() const noexcept {
        return geometry_buffer_base() + triangle_buffer_id_offset;
    }
    [[nodiscard]] auto triangle_count() const noexcept { return _triangle_count; }

    [[nodiscard]] auto surface_tag() const noexcept { return _surface_tag; }
    [[nodiscard]] auto light_tag() const noexcept { return _light_tag; }
    [[nodiscard]] auto medium_tag() const noexcept { return _medium_tag; }

    [[nodiscard]] auto test_property_flag(uint flag) const noexcept {
        return (property_flags() & flag) != 0u;
    }
    [[nodiscard]] auto has_vertex_normal() const noexcept {
        return test_property_flag(PROPERTY_HAS_VERTEX_NORMAL);
    }
    [[nodiscard]] auto has_vertex_uv() const noexcept {
        return test_property_flag(PROPERTY_HAS_VERTEX_UV);
    }
    [[nodiscard]] auto has_light() const noexcept {
        return test_property_flag(PROPERTY_HAS_LIGHT);
    }
    [[nodiscard]] auto has_surface() const noexcept {
        return test_property_flag(PROPERTY_HAS_SURFACE);
    }
    [[nodiscard]] auto maybe_non_opaque() const noexcept {
        return test_property_flag(PROPERTY_MAYBE_NON_OPAQUE);
    }

    [[nodiscard]] auto shadow_terminator_factor() const noexcept {
        return _shadow_terminator;
    }
    [[nodiscard]] auto intersection_offset_factor() const noexcept {
        return _intersection_offset;
    }
};

} // namespace newtype::scene

LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(newtype::scene::Shape::Handle)
