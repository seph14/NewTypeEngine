//
// Created by Claude on 2026/03/23.
//

#pragma once

#include <luisa/dsl/rtx/ray.h>
#include <luisa/dsl/func.h>
#include "newtype/util/Vertex.h"
#include "newtype/scene/Shape.h"

namespace newtype::scene {

using namespace luisa;
using compute::Bool;
using compute::Expr;
using compute::Float2;
using compute::Float3;
using compute::Float4x4;
using compute::Ray;
using compute::UInt;
using compute::Var;

/**
 * @brief Basic geometric attributes at intersection point
 *
 * Contains position, normal, and area for light sampling
 * and visibility testing.
 */
struct GeometryAttribute {
    Float3 p;      // World-space position
    Float3 n;      // Geometric normal (face normal)
    Float area;    // Triangle area (for light sampling)
};

/**
 * @brief Full shading attributes at intersection point
 *
 * Contains all attributes needed for BSDF evaluation:
 * - Geometric attributes
 * - Shading normal (interpolated vertex normals)
 * - Texture coordinates
 * - Partial derivatives for texture filtering
 */
struct ShadingAttribute {
    GeometryAttribute g;
    Float3 ps;       // Shading point (same as g.p for flat shading)
    Float3 ns;       // Shading normal (interpolated)
    Float3 dpdu;     // Tangent vector (partial derivative wrt u)
    Float3 dpdv;     // Bitangent vector (partial derivative wrt v)
    Float2 uv;       // Texture coordinates
};

/**
 * @brief Ray-surface intersection data
 *
 * Created by ray tracing queries and consumed by
 * path tracing kernels for shading.
 */
class Interaction {
private:
    Shape::Handle _shape;
    Float3 _pg;          // Geometric point (world-space hit position)
    Float3 _ng;          // Geometric normal
    Float2 _uv;          // Texture coordinates
    Float3 _ps;          // Shading point
    Float3 _ns;          // Shading normal
    Float3 _dpdu;        // Tangent
    Float3 _dpdv;        // Bitangent
    UInt _inst_id;       // Instance ID (in TLAS)
    UInt _prim_id;       // Primitive ID (triangle index)
    Float _prim_area;    // Primitive area (for light sampling)
    Bool _back_facing;   // True if ray hit back face
    bool _has_uv;        // True if texture coordinates available
    bool _has_shading;   // True if shading normal available

public:
    // Default: miss
    Interaction() noexcept
        : _inst_id(~0u), _prim_id(~0u),
          _has_uv(false), _has_shading(false) {}

    // Environment hit
    explicit Interaction(Expr<luisa::float2> uv) noexcept
        : _uv(uv), _inst_id(~0u), _prim_id(~0u),
          _has_uv(true), _has_shading(false) {}

    // Simple geometry hit (no attributes)
    Interaction(Expr<luisa::float3> p) noexcept
        : _pg(p), _ng(p), _inst_id(~0u), _prim_id(~0u),
          _has_uv(false), _has_shading(false) {}

    // Full geometry hit (from Shape::Handle)
    Interaction(Shape::Handle shape, Expr<uint> inst_id,
                Expr<uint> prim_id, Expr<float> prim_area,
                Expr<luisa::float3> p, Expr<luisa::float3> ng,
                Expr<bool> back_facing) noexcept
        : _shape(std::move(shape)),
          _pg(p), _ng(ng), _ps(p), _ns(ng),
          _inst_id(inst_id), _prim_id(prim_id),
          _prim_area(prim_area), _back_facing(back_facing),
          _has_uv(false), _has_shading(false) {}

    // Full shading hit (with all attributes)
    Interaction(Shape::Handle shape, Expr<uint> inst_id,
                Expr<uint> prim_id, Expr<float> prim_area,
                Expr<luisa::float3> pg, Expr<luisa::float3> ng, Expr<luisa::float2> uv,
                Expr<luisa::float3> ps, Expr<luisa::float3> ns,
                Expr<luisa::float3> dpdu, Expr<luisa::float3> dpdv,
                Expr<bool> back_facing) noexcept
        : _shape(std::move(shape)),
          _pg(pg), _ng(ng), _uv(uv),
          _ps(ps), _ns(ns), _dpdu(dpdu), _dpdv(dpdv),
          _inst_id(inst_id), _prim_id(prim_id),
          _prim_area(prim_area), _back_facing(back_facing),
          _has_uv(true), _has_shading(true) {}

    // Construct from ShadingAttribute
    Interaction(Shape::Handle shape, Expr<uint> inst_id,
                Expr<uint> prim_id, const ShadingAttribute &attrib,
                Expr<bool> back_facing) noexcept
        : _shape(std::move(shape)),
          _pg(attrib.g.p), _ng(attrib.g.n), _uv(attrib.uv),
          _ps(attrib.ps), _ns(attrib.ns), _dpdu(attrib.dpdu), _dpdv(attrib.dpdv),
          _inst_id(inst_id), _prim_id(prim_id),
          _prim_area(attrib.g.area), _back_facing(back_facing),
          _has_uv(true), _has_shading(true) {}

public: // Accessors
    [[nodiscard]] auto p() const noexcept { return _pg; }
    [[nodiscard]] auto p_shading() const noexcept { return _ps; }
    [[nodiscard]] auto ng() const noexcept { return _ng; }
    [[nodiscard]] auto ns() const noexcept { return _ns; }
    [[nodiscard]] auto uv() const noexcept { return _uv; }
    [[nodiscard]] auto instance_id() const noexcept { return _inst_id; }
    [[nodiscard]] auto triangle_id() const noexcept { return _prim_id; }
    [[nodiscard]] auto triangle_area() const noexcept { return _prim_area; }
    [[nodiscard]] auto valid() const noexcept { return _inst_id != ~0u; }
    [[nodiscard]] auto has_uv() const noexcept { return _has_uv; }
    [[nodiscard]] auto has_shading() const noexcept { return _has_shading; }
    [[nodiscard]] auto back_facing() const noexcept { return _back_facing; }
    [[nodiscard]] auto &shape() const noexcept { return _shape; }

public: // Ray spawning
    static constexpr auto default_t_max = std::numeric_limits<float>::max();

    [[nodiscard]] Var<Ray> spawn_ray(
        Expr<luisa::float3> wi,
        Expr<float> t_max = default_t_max) const noexcept;

    [[nodiscard]] Var<Ray> spawn_ray_to(Expr<luisa::float3> p) const noexcept;
};

} // namespace newtype::scene

// Disable DSL address-of operator for these types
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(newtype::scene::GeometryAttribute)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(newtype::scene::ShadingAttribute)
LUISA_DISABLE_DSL_ADDRESS_OF_OPERATOR(newtype::scene::Interaction)
