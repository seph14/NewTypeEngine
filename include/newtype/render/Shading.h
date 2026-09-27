#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "newtype/util/Vertex.h"
#include "newtype/render/BSDF.h"
#include "newtype/render/SurfaceData.h"
#include "newtype/render/MaterialPool.h"   // LUISA_STRUCT(MaterialData) + get_albedo() etc.
#include "newtype/render/SurfaceResolver.h"
#if NT_ENABLE_PROCEDURAL
#include "newtype/render/ProceduralTrace.h"
#include "newtype/render/ProcBindlessSlots.h"
#endif

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

/// Instance buffer property flag for double-sided geometry.
/// Must match PROPERTY_DOUBLE_SIDED in newtype/scene/Shape.h.
static constexpr uint kDoubleSidedFlag = 1u << 8u;

/// Instance buffer property flag for camera-invisible geometry (e.g. a light
/// shape that illuminates without rendering). Camera-path traces — G-buffer
/// primary rays, shade mirror reflections, the glass tint replay — skip these
/// instances; shadow / GI / light-sampling rays are unaffected.
/// Must match PROPERTY_INVISIBLE_TO_CAMERA in newtype/scene/Shape.h.
static constexpr uint kCameraInvisibleFlag = 1u << 9u;

/// Returns the effective BSDF type from a MaterialData.
/// For built-in types (0-12), returns material.type directly.
/// For custom callables (type >= 13), returns bsdf_type_override if set (> 0).
[[nodiscard]] inline UInt get_effective_bsdf_type(
    const Var<MaterialData> &material) noexcept {
    UInt result;
    $if(material.bsdf_type_override > 0.5f) {
        result = cast<UInt>(material.bsdf_type_override);
    }
    $else {
        result = material.type;
    };
    return result;
}

/// Interleaved gradient noise (Jiménez 2014), STATIC in frame count.
/// Glass-blend rolls seed from this — a rotating/per-frame seed would flip the
/// stored-surface identity every frame and reset ReLAX history to salt-and-
/// pepper (same contract as the dispersion d-line fix, RC1). No frame term.
[[nodiscard]] inline Float ign_static(UInt2 pixel) noexcept {
    Float2 p = make_float2(cast<float>(pixel.x), cast<float>(pixel.y));
    return fract(52.9829189f * fract(dot(p, make_float2(0.06711056f, 0.00583715f))));
}

/// Stored-surface consumer reclass for callable-driven glass blending
/// (docs/glass_blend_plan.md). The G-Buffer PSR stochastically rolls blendable
/// dielectrics per pixel: rolled-GLASS pixels trace through and store the
/// background (glass bit set); rolled-OPAQUE pixels store the blendable
/// surface itself — whose re-resolved bsdf_type is still 3/11 via
/// bsdf_type_override. Every such consumer must reclass those pixels to a
/// shadable opaque type (1) or they render black (delta BSDF, no NEE).
/// Safe for legacy content: a stored 3/11 surface with a clear glass bit only
/// exists for rolled-opaque blend pixels (real glass always accumulates
/// fresnel > 0; the false-hit passthrough stores the non-glass background).
inline void reclass_blend_rolled_opaque(SurfaceData &s, Expr<bool> gbuf_is_glass) noexcept {
    Bool rolled = ((s.bsdf_type == 3u) | (s.bsdf_type == 11u)) & !gbuf_is_glass;
    s.bsdf_type = ite(rolled, 1u, s.bsdf_type);
    // Blendables author roughness 0 for the GLASS side; the opaque roll must
    // not inherit it — the shade's delta-mirror branch (roughness <
    // kMinRoughness on non-glass) would render the diffuse side as a black
    // mirror. Floor only the rolled pixels (legacy surfaces never reclass).
    s.roughness = ite(rolled, max(s.roughness, kMinRoughness), s.roughness);
    // Same for dielectric specular_trans (1 on glass-authored materials): it
    // routes the lobe list's energy into the transmission lobe, leaving the
    // opaque roll zero diffuse — black. The glass side never reads it.
    s.specular_trans = ite(rolled, 0.0f, s.specular_trans);
}

/// Callable signature for custom material surface resolution.
/// Defined at namespace scope so DLL code can reference it.
/// Only valid inside device.compile() contexts.
/// UV is read from s.uv inside the callable (set by resolve_surface / build_procedural_surface_base).
using SurfaceResolveFn = std::function<SurfaceData(
    SurfaceData, Var<MaterialData>, Float2, Float3, Float,
    const BindlessVar&, UInt, UInt)>;

/// Read a callable's runtime tuning params (docs/resolver_params_abi_plan.md).
/// The params buffer occupies the reserved texture-bindless slot
/// (MaterialPool::kResolverParamsBindlessSlot) the resolver already receives
/// as `tex`; `base` is the callable's float4 base (host-assigned at
/// registration — the ABI v2 paramFn's return value, captured by value in the
/// resolver lambda). Values never enter the AST: editing them host-side is a
/// buffer upload with no shader recompile. `i` indexes float4s within the
/// callable's block (scalar j is component j%4 of i=j/4).
[[nodiscard]] inline auto resolver_params(const BindlessVar& tex, UInt base, UInt i) noexcept {
    return tex.buffer<luisa::float4>(MaterialPool::kResolverParamsBindlessSlot).read(base + i);
}

/// Read one float4 of an instance's per-instance custom data row (track B2,
/// docs/vertex-packing-instancing-plan.md §4). Rows live at the reserved
/// texture-bindless slot 1 and cover every TLAS instance once the app has
/// authored any row (Pipeline::setInstanceUserData materializes them);
/// before that the slot holds a small zero-filled placeholder — callables
/// should read it only for scenes that author per-instance data. Mesh
/// surfaces only: `s.instance_index` on a procedural surface is the
/// procedural index, not a TLAS row. Zero cost unless a callable reads it.
[[nodiscard]] inline auto instance_params(const BindlessVar& tex, UInt instance_index, UInt i) noexcept {
    return tex.buffer<luisa::float4>(MaterialPool::kInstanceParamsBindlessSlot)
        .read(instance_index * MaterialPool::kInstanceParamsPerInstance + i);
}

/// Register a custom material callable from any callable (lambda, function ptr, etc).
/// Wraps into CustomSurfaceResolver<F> and registers with the Polymorphic container.
/// MUST be called before buildScene().
template<typename F>
uint registerCustomCallable(SurfaceResolverPoly& poly, const std::string& name, F&& resolve) {
    auto tag = poly.create<CustomSurfaceResolver<std::decay_t<F>>>(std::forward<F>(resolve));
    return static_cast<uint>(tag);
}

/// Compute luminance of a color vector (Rec. 709 weights)
[[nodiscard]] inline Float luminance(Float3 c) noexcept {
    return dot(c, luisa::make_float3(0.2126f, 0.7152f, 0.0722f));
}

/// Encode a unit normal to 2-component octahedral representation.
/// Maps the unit sphere to [-1,1]^2 square. Precision is sufficient for HALF storage.
/// Only callable inside device.compile() contexts (uses DSL $if).
[[nodiscard]] inline Float2 oct_encode(Float3 n) noexcept {
    Float denom = abs(n.x) + abs(n.y) + abs(n.z);
    Float2 enc = make_float2(n.x, n.y) / max(denom, 1e-10f);
    enc = ite(n.z < 0.0f,
        (make_float2(1.0f) - abs(enc)) * make_float2(
            ite(n.x >= 0.0f, 1.0f, -1.0f),
            ite(n.y >= 0.0f, 1.0f, -1.0f)),
        enc);
    return enc;
}

/// Decode 2-component octahedral representation back to unit normal.
/// Only callable inside device.compile() contexts (uses DSL $if).
[[nodiscard]] inline Float3 oct_decode(Float2 enc) noexcept {
    Float3 n = make_float3(enc.x, enc.y, 1.0f - abs(enc.x) - abs(enc.y));
    $if(n.z < 0.0f) {
        n = make_float3(
            (1.0f - abs(enc.y)) * ite(enc.x >= 0.0f, 1.0f, -1.0f),
            (1.0f - abs(enc.x)) * ite(enc.y >= 0.0f, 1.0f, -1.0f),
            n.z);
    };
    return normalize(n);
}

/// Reconstruct shading normal from G-Buffer hit data via bindless vertex access.
/// Caller reads instance_buffer once and passes the pre-extracted bindless slots.
/// Inline — BindlessVar is non-copyable so must be passed by reference.
[[nodiscard]] inline Float3 reconstruct_normal(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary) noexcept {
    using Triangle = luisa::compute::Triangle;
    using Vertex   = newtype::util::ActiveVertex;   // GPU layout (A2)

    auto tri = vertex_bindless.buffer<Triangle>(tri_slot).read(prim_id);
    auto v0  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i0);
    auto v1  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i1);
    auto v2  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i2);

    return normalize(
        v0->normal() * (1.0f - bary.x - bary.y)+
        v1->normal() * bary.x +
        v2->normal() * bary.y);
}

/// Reconstruct geometric face normal from triangle edges (no vertex normal interpolation).
/// Used for double-sided geometry where face normals are more reliable than
/// interpolated vertex normals on backface hits.
[[nodiscard]] inline Float3 reconstruct_face_normal(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id) noexcept {
    using Triangle = luisa::compute::Triangle;
    using Vertex   = newtype::util::ActiveVertex;   // GPU layout (A2)

    auto tri = vertex_bindless.buffer<Triangle>(tri_slot).read(prim_id);
    auto v0  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i0);
    auto v1  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i1);
    auto v2  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i2);

    return normalize(cross(v1->position() - v0->position(), v2->position() - v0->position()));
}

/// Reconstruct object-space hit position from bindless vertex access.
/// Uses exact vertex positions + barycentric interpolation — avoids
/// precision loss from the ray equation (origin + t * dir).
///
/// Returns OBJECT-SPACE position. Callers that need world space must
/// apply the instance transform: `(xform * make_float4(p, 1)).xyz()`.
/// resolve_surface_* does this internally and exposes world-space
/// SurfaceData.position; direct callers must do it themselves.
[[nodiscard]] inline Float3 reconstruct_object_position(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary) noexcept {
    using Triangle = luisa::compute::Triangle;
    using Vertex   = newtype::util::ActiveVertex;   // GPU layout (A2)

    auto tri = vertex_bindless.buffer<Triangle>(tri_slot).read(prim_id);
    auto v0  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i0);
    auto v1  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i1);
    auto v2  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i2);

    return v0->position() * (1.0f - bary.x - bary.y) +
           v1->position() * bary.x +
           v2->position() * bary.y;
}

/// Reconstruct interpolated tangent frame from bindless vertex access.
/// Returns float4: xyz = tangent direction, w = bitangent handedness.
[[nodiscard]] inline Float4 reconstruct_tangent(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary) noexcept {
    using Triangle = luisa::compute::Triangle;
    using Vertex   = newtype::util::ActiveVertex;   // GPU layout (A2)

    auto tri = vertex_bindless.buffer<Triangle>(tri_slot).read(prim_id);
    auto v0  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i0);
    auto v1  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i1);
    auto v2  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i2);

    Float4 t = v0->tangent() * (1.0f - bary.x - bary.y) +
               v1->tangent() * bary.x +
               v2->tangent() * bary.y;
    return make_float4(normalize(t.xyz()), t.w);
}

//==============================================================================
// Fused mesh-triangle fetch — single set of bindless reads (1 triangle + 3
// vertices = 4 reads) shared by all downstream reconstruct_* overloads below.
// Mirrors the read_procedural_triangle precedent. Calling reconstruct_*
// individually on the same triangle costs 4 reads each (16 reads for the
// full set); going through MeshTriVerts collapses that to 4.
//==============================================================================
struct MeshTriVerts {
    Var<luisa::compute::Triangle> tri;
    Var<newtype::util::ActiveVertex> v0;
    Var<newtype::util::ActiveVertex> v1;
    Var<newtype::util::ActiveVertex> v2;
};

[[nodiscard]] inline MeshTriVerts read_mesh_triangle(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id) noexcept {
    using Triangle = luisa::compute::Triangle;
    using Vertex   = newtype::util::ActiveVertex;   // GPU layout (A2)
    auto tri = vertex_bindless.buffer<Triangle>(tri_slot).read(prim_id);
    auto v0  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i0);
    auto v1  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i1);
    auto v2  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i2);
    return MeshTriVerts{tri, v0, v1, v2};
}

[[nodiscard]] inline Float3 reconstruct_normal(const MeshTriVerts& tv, Float2 bary) noexcept {
    return normalize(
        tv.v0->normal() * (1.0f - bary.x - bary.y) +
        tv.v1->normal() * bary.x +
        tv.v2->normal() * bary.y);
}

[[nodiscard]] inline Float3 reconstruct_face_normal(const MeshTriVerts& tv) noexcept {
    return normalize(cross(tv.v1->position() - tv.v0->position(),
                           tv.v2->position() - tv.v0->position()));
}

[[nodiscard]] inline Float3 reconstruct_object_position(const MeshTriVerts& tv, Float2 bary) noexcept {
    return tv.v0->position() * (1.0f - bary.x - bary.y) +
           tv.v1->position() * bary.x +
           tv.v2->position() * bary.y;
}

[[nodiscard]] inline Float4 reconstruct_tangent(const MeshTriVerts& tv, Float2 bary) noexcept {
    Float4 t = tv.v0->tangent() * (1.0f - bary.x - bary.y) +
               tv.v1->tangent() * bary.x +
               tv.v2->tangent() * bary.y;
    return make_float4(normalize(t.xyz()), t.w);
}

[[nodiscard]] inline Float2 reconstruct_uv(const MeshTriVerts& tv, Float2 bary) noexcept {
    return tv.v0->uv() * (1.0f - bary.x - bary.y) +
           tv.v1->uv() * bary.x +
           tv.v2->uv() * bary.y;
}

/// Transform object-space direction to world space using 4×4 instance transform.
/// Extracts the 3×3 portion and multiplies (w=0 ignores translation).
/// LuisaCompute float4x4 is row-major (xform[row][col]), so gather columns across rows.
[[nodiscard]] inline Float3 transform_normal(
    const Float4x4& xform, Float3 n) noexcept {
    Float3 col0 = make_float3(xform[0][0], xform[1][0], xform[2][0]);
    Float3 col1 = make_float3(xform[0][1], xform[1][1], xform[2][1]);
    Float3 col2 = make_float3(xform[0][2], xform[1][2], xform[2][2]);
    return normalize(col0 * n.x + col1 * n.y + col2 * n.z);
}

/// Transform all surface normals/tangent from object space to world space,
/// then flip ns and geo_ns to face the incoming direction wo.
inline void transform_surface_normals(
    SurfaceData& surface, const Float4x4& xform, const Float3& wo) noexcept {
    surface.ns      = transform_normal(xform, surface.ns);
    surface.geo_ns  = transform_normal(xform, surface.geo_ns);
    surface.tangent = normalize(transform_normal(xform, surface.tangent));
    // Don't flip geo_ns — it must remain the true geometric normal for shadow
    // ray offsets and entering/exiting checks. Only ns is flipped for shading.
    surface.ns      = ite(dot(wo, surface.ns) < 0.f, -surface.ns, surface.ns);
}

//==============================================================================
// NRD material demodulation factors (NRD.hlsli:735-746)
//
// Bounded factors in [0.02, 1] for dividing denoiser input (and multiplying
// the denoised result back). Unlike raw albedo, they never blow up 1/x: the
// specular factor follows the grazing-whitening of Fresnel, so white-at-
// grazing conductor highlights and sheen fuzz lobes (which are NOT
// albedo-proportional) divide by a near-white factor instead of 1/albedo.
// Fenv is the "Ray Tracing Gems" ch.32 environment-term fit (NRD.hlsli:529).
//==============================================================================
inline void nrd_material_factors(
    const Float3& N, const Float3& V, const Float3& albedo,
    const Float3& rf0, const Float& roughness,
    Float3& diff_factor, Float3& spec_factor) noexcept {
    constexpr float kMaterialFactorMinScale = 0.02f; // NRD: smaller may destabilize
    constexpr float kRoughnessFactorMinScale = 0.1f; // NRD: smaller biases

    Float NoV = abs(dot(N, V));
    Float m = luisa::compute::saturate(roughness * roughness);

    // X = (1, NoV, NoV^2, NoV^3); Y = (1, m, m^2, m^3) — matrix-vector
    // products from the fit written out as explicit row dots.
    Float2 x2 = make_float2(1.0f, NoV);
    Float3 x3a = make_float3(1.0f, NoV, NoV * NoV);          // X.xyw
    Float3 x3b = make_float3(1.0f, NoV * NoV, NoV * NoV * NoV); // X.xzw
    Float2 y2 = make_float2(1.0f, m);
    Float3 y3 = make_float3(1.0f, m, m * m);                 // Y.xyw

    // bias = dot(M1*X.xy, Y.xy) / max(dot(M2*X.xyw, Y.xyw), eps)
    Float2 m1v = make_float2(
        0.99044f * x2.x + -1.28514f * x2.y,
        1.29678f * x2.x + -0.755907f * x2.y);
    Float3 m2v = make_float3(
        1.0f * x3a.x + 2.92338f * x3a.y + 59.4188f * x3a.z,
        20.3225f * x3a.x + -27.0302f * x3a.y + 222.592f * x3a.z,
        121.563f * x3a.x + 626.13f * x3a.y + 316.627f * x3a.z);
    Float bias = dot(m1v, y2) / max(dot(m2v, y3), 1e-6f);

    // scale = dot(M3*X.xy, Y.xy) / max(dot(M4*X.xzw, Y.xyw), eps)
    Float2 m3v = make_float2(
        0.0365463f * x2.x + 3.32707f * x2.y,
        9.0632f * x2.x + -9.04756f * x2.y);
    Float3 m4v = make_float3(
        1.0f * x3b.x + 3.59685f * x3b.y + -1.36772f * x3b.z,
        9.04401f * x3b.x + -16.3174f * x3b.y + 9.22949f * x3b.z,
        5.56589f * x3b.x + 19.7886f * x3b.y + -20.2123f * x3b.z);
    Float scale = dot(m3v, y2) / max(dot(m4v, y3), 1e-6f);

    Float3 fenv = luisa::compute::saturate(rf0 * scale + bias);

    Float3 one = make_float3(1.0f);
    Float3 minF = make_float3(kMaterialFactorMinScale);
    diff_factor = lerp(minF, one, (1.0f - fenv) * albedo);
    Float3 sf = fenv * lerp(make_float3(kRoughnessFactorMinScale), one, roughness);
    spec_factor = lerp(minF, one, sf);
}

#if NT_ENABLE_PROCEDURAL

// Minimal inline helpers for procedural reconstruction (avoids pulling full ProceduralTrace.h)
/*namespace newtype::render::proc_helpers {
[[nodiscard]] inline UInt unpack_index_offset(UInt packed) noexcept { return packed >> 16u; }
[[nodiscard]] inline UInt unpack_position_offset(UInt packed) noexcept { return packed & 0xFFFFu; }
}*/ // namespace proc_helpers

/// Reconstruct world-space position from procedural triangle mesh (VAT or deform).
/// After pre-interpolation, single-frame read from active buffer.
[[nodiscard]] inline Float3 reconstruct_procedural_position(
    const BindlessVar& proc_bindless,
    Var<scene::ProcInstanceData> inst,
    UInt local_tri, Float2 bary) noexcept {
    using Triangle = luisa::compute::Triangle;
    UInt idx_base = unpack_index_offset(inst.packed_offsets);
    UInt pos_base = unpack_position_offset(inst.packed_offsets);
    auto tri = proc_bindless.buffer<Triangle>(kSlot_ProcIndices).read(idx_base + local_tri);

    Float3 p0 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i0).xyz();
    Float3 p1 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i1).xyz();
    Float3 p2 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i2).xyz();
    return p0 * (1.0f - bary.x - bary.y) + p1 * bary.x + p2 * bary.y;
}

/// Reconstruct shading normal from procedural triangle mesh (VAT or deform).
/// After pre-interpolation, single-frame read from active buffer.
[[nodiscard]] inline Float3 reconstruct_procedural_normal(
    const BindlessVar& proc_bindless,
    Var<scene::ProcInstanceData> inst,
    UInt local_tri, Float2 bary) noexcept {
    using Triangle = luisa::compute::Triangle;
    UInt idx_base = unpack_index_offset(inst.packed_offsets);
    UInt pos_base = unpack_position_offset(inst.packed_offsets);
    auto tri = proc_bindless.buffer<Triangle>(kSlot_ProcIndices).read(idx_base + local_tri);

    Float3 n0 = proc_bindless.buffer<luisa::float4>(kSlot_ProcNormals).read(pos_base + tri.i0).xyz();
    Float3 n1 = proc_bindless.buffer<luisa::float4>(kSlot_ProcNormals).read(pos_base + tri.i1).xyz();
    Float3 n2 = proc_bindless.buffer<luisa::float4>(kSlot_ProcNormals).read(pos_base + tri.i2).xyz();
    return normalize(n0 * (1.0f - bary.x - bary.y) + n1 * bary.x + n2 * bary.y);
}

/// Reconstruct interpolated UV from procedural triangle mesh (VAT or deform).
/// UVs are packed in .w channels: position.w = uv_u, normal.w = uv_v.
/// LuisaCompute's float3 is 16-byte aligned, so the .w slot is free real estate
/// and we deliberately do NOT use a dedicated UV buffer.
[[nodiscard]] inline Float2 reconstruct_procedural_uv(
    const BindlessVar& proc_bindless,
    Var<scene::ProcInstanceData> inst,
    UInt local_tri, Float2 bary) noexcept {
    using Triangle = luisa::compute::Triangle;
    UInt pos_base = unpack_position_offset(inst.packed_offsets);
    UInt idx_base = unpack_index_offset(inst.packed_offsets);
    auto tri = proc_bindless.buffer<Triangle>(kSlot_ProcIndices).read(idx_base + local_tri);

    // UVs packed in .w channels: position.w = uv_u, normal.w = uv_v
    Float u0 = proc_bindless.buffer<float4>(kSlot_ProcPositions).read(pos_base + tri.i0).w;
    Float u1 = proc_bindless.buffer<float4>(kSlot_ProcPositions).read(pos_base + tri.i1).w;
    Float u2 = proc_bindless.buffer<float4>(kSlot_ProcPositions).read(pos_base + tri.i2).w;
    Float v0 = proc_bindless.buffer<float4>(kSlot_ProcNormals).read(pos_base + tri.i0).w;
    Float v1 = proc_bindless.buffer<float4>(kSlot_ProcNormals).read(pos_base + tri.i1).w;
    Float v2 = proc_bindless.buffer<float4>(kSlot_ProcNormals).read(pos_base + tri.i2).w;

    Float w0 = 1.0f - bary.x - bary.y;
    Float u = u0 * w0 + u1 * bary.x + u2 * bary.y;
    Float v = v0 * w0 + v1 * bary.x + v2 * bary.y;
    return make_float2(u, v);
}

/// Fused single-pass triangle read for VAT/deform procedural geometry.
/// Reads each bindless entry EXACTLY ONCE and returns positions, normals,
/// and per-vertex UVs together. Use this when multiple attributes are needed
/// (avoids the duplicate reads that calling reconstruct_procedural_normal +
/// reconstruct_proc_face_normal + reconstruct_procedural_uv in sequence would cause).
/// UVs are unpacked from .w channels (see reconstruct_procedural_uv).
struct ProceduralTriData {
    Float3 p0, p1, p2;    // vertex positions (.xyz)
    Float3 n0, n1, n2;    // vertex normals  (.xyz)
    Float2 uv0, uv1, uv2; // (pos.w, norm.w) per vertex
};
[[nodiscard]] inline ProceduralTriData read_procedural_triangle(
    const BindlessVar& proc_bindless,
    Var<scene::ProcInstanceData> inst,
    UInt local_tri) noexcept {
    using Triangle = luisa::compute::Triangle;
    UInt idx_base = unpack_index_offset(inst.packed_offsets);
    UInt pos_base = unpack_position_offset(inst.packed_offsets);
    auto tri = proc_bindless.buffer<Triangle>(kSlot_ProcIndices).read(idx_base + local_tri);

    auto pa = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i0);
    auto pb = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i1);
    auto pc = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i2);
    auto na = proc_bindless.buffer<luisa::float4>(kSlot_ProcNormals).read(pos_base + tri.i0);
    auto nb = proc_bindless.buffer<luisa::float4>(kSlot_ProcNormals).read(pos_base + tri.i1);
    auto nc = proc_bindless.buffer<luisa::float4>(kSlot_ProcNormals).read(pos_base + tri.i2);

    ProceduralTriData t;
    t.p0 = pa.xyz(); t.p1 = pb.xyz(); t.p2 = pc.xyz();
    t.n0 = na.xyz(); t.n1 = nb.xyz(); t.n2 = nc.xyz();
    t.uv0 = make_float2(pa.w, na.w);
    t.uv1 = make_float2(pb.w, nb.w);
    t.uv2 = make_float2(pc.w, nc.w);
    return t;
}

/// Procedural equivalent of reconstruct_unjittered_bary (Shading.h:691).
/// Reprojects the unjittered camera ray onto the procedural triangle plane and
/// solves for barycentric coordinates. Falls back to jit_bary when the unjit
/// ray grazes the triangle (silhouette / edge pixels) — same validity gates
/// as the mesh version.
///
/// Used to stabilize texture sampling on procedural geometry: the G-Buffer
/// stores jittered bary (Halton sub-pixel jitter), and without this recompute
/// UVs walk across texels frame-to-frame and shimmer under the denoiser's
/// demod/remod.
///
/// @param inst_id     procedural instance id (proc AABB index from hit.prim)
/// @param prim_id     packed prim_id stored in vis.y — bit 29 set, low bits = local_tri
/// @param jit_bary    jittered barycentric from gbuf_bary_motion (fallback)
/// @param cam_origin  unjittered camera ray origin
/// @param unjit_dir   unjittered camera ray direction (normalized)
[[nodiscard]] inline Float2 reconstruct_unjittered_bary_procedural(
    const BindlessVar& proc_bindless,
    UInt inst_id, UInt prim_id,
    Float2 jit_bary,
    Float3 cam_origin, Float3 unjit_dir) noexcept {

    Var<scene::ProcInstanceData> inst =
        proc_bindless.buffer<scene::ProcInstanceData>(kSlot_ProcInstances).read(inst_id);
    UInt local_tri = prim_id & 0x1FFFFFFFu;

    ProceduralTriData t = read_procedural_triangle(proc_bindless, inst, local_tri);
    Float3 p0 = t.p0, p1 = t.p1, p2 = t.p2;

    Float3 e01 = p1 - p0;
    Float3 e02 = p2 - p0;
    Float3 N   = cross(e01, e02);

    Float  denom      = dot(unjit_dir, N);
    Bool   denom_safe = abs(denom) > 1e-6f;
    Float  safe_denom = ite(denom_safe, denom, 1.0f);
    Float  t_hit      = dot(p0 - cam_origin, N) / safe_denom;
    Float3 P          = cam_origin + t_hit * unjit_dir;

    Float3 v0P = P - p0;
    Float  d11 = dot(e01, e01);
    Float  d12 = dot(e01, e02);
    Float  d22 = dot(e02, e02);
    Float  d31 = dot(v0P, e01);
    Float  d32 = dot(v0P, e02);
    Float  dd  = d11 * d22 - d12 * d12;
    Bool   dd_safe = abs(dd) > 1e-10f;
    Float  safe_dd = ite(dd_safe, dd, 1.0f);
    Float  b1 = (d22 * d31 - d12 * d32) / safe_dd;
    Float  b2 = (d11 * d32 - d12 * d31) / safe_dd;
    Float  b0 = 1.0f - b1 - b2;

    Bool bary_in_tri = (b0 >= -0.02f) & (b0 <= 1.02f)
                     & (b1 >= -0.02f) & (b1 <= 1.02f)
                     & (b2 >= -0.02f) & (b2 <= 1.02f);
    Bool use_unjit = denom_safe & dd_safe & bary_in_tri;

    return ite(use_unjit, make_float2(b1, b2), jit_bary);
}

/// Reconstruct geometric face normal from procedural triangle positions.
/// Single-frame read — no interpolation needed after pre-interpolation.
[[nodiscard]] inline Float3 reconstruct_proc_face_normal(
    const BindlessVar& proc_bindless,
    Var<scene::ProcInstanceData> inst,
    UInt local_tri) noexcept {
    using Triangle = luisa::compute::Triangle;
    UInt idx_base = unpack_index_offset(inst.packed_offsets);
    UInt pos_base = unpack_position_offset(inst.packed_offsets);
    auto tri = proc_bindless.buffer<Triangle>(kSlot_ProcIndices).read(idx_base + local_tri);

    Float3 p0 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i0).xyz();
    Float3 p1 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i1).xyz();
    Float3 p2 = proc_bindless.buffer<luisa::float4>(kSlot_ProcPositions).read(pos_base + tri.i2).xyz();
    return normalize(cross(p1 - p0, p2 - p0));
}
#endif

//==============================================================================
// BSDF-flavored p_hat / direct-illuminance helpers
//
// These overloads take a pre-built MaterialBSDF const& and delegate directly to
// bsdf.evaluate*() — no rebuild from scalars. Use these whenever a SurfaceData
// is available (so make_bsdf() can produce the layered/composed LobeList path).
// Required for correct target_pdf under vertical layering: the scalar rebuild
// below calls make_material_bsdf() (Phase-1 single-layer path) and never sees
// the coat/fuzz fields, producing a target_pdf that disagrees with the shade
// path's bsdf.evaluate() — see plan resilient-herding-kahn.md.
//
// Glass gate reads bsdf.bsdf_type (the BSDF carries the same value as
// SurfaceData::bsdf_type — make_bsdf() propagates it verbatim).
//==============================================================================

[[nodiscard]] inline Float3 evaluate_direct_illuminance_with_geometry(
    const MaterialBSDF& bsdf,
    Float3 emission,
    Float3 light_dir, Float dist_sq,
    Float cos_shading, Float cos_light,
    Float3 ns, Float3 wo) noexcept {
    Float3 result = def(make_float3(0.0f));
    $if(bsdf.bsdf_type != 3u) {
        Float3 brdf = bsdf.evaluate(wo, light_dir, ns);
        result = emission * brdf * cos_shading * cos_light / dist_sq;
    };
    return result;
}

[[nodiscard]] inline Float3 evaluate_direct_illuminance(
    const MaterialBSDF& bsdf,
    Float3 emission,
    Float3 light_normal,
    Float3 light_point, Float3 world_pos, Float3 ns,
    Float3 wo) noexcept {
    Float3 to_light    = light_point - world_pos;
    Float  dist_sq     = max(dot(to_light, to_light), 1e-6f);
    Float3 light_dir   = to_light * rsqrt(dist_sq);
    Float  cos_shading = max(0.0f, dot(ns, light_dir));
    Float  cos_light   = max(0.0f, dot(light_normal, -light_dir));
    return evaluate_direct_illuminance_with_geometry(
        bsdf, emission, light_dir, dist_sq, cos_shading, cos_light, ns, wo);
}

[[nodiscard]] inline Float3 evaluate_env_contribution(
    const MaterialBSDF& bsdf,
    Float3 env_radiance, Float3 wi, Float3 ns, Float3 wo) noexcept {
    Float cos_theta = max(0.0f, dot(ns, wi));
    Float3 result = def(make_float3(0.0f));
    $if(bsdf.bsdf_type != 3u) {
        Float3 brdf = bsdf.evaluate(wo, wi, ns);
        result = env_radiance * brdf * cos_theta;
    };
    return result;
}

[[nodiscard]] inline Float evaluate_p_hat(
    const MaterialBSDF& bsdf,
    Float3 emission,
    Float3 light_normal,
    Float3 light_point, Float3 world_pos, Float3 ns,
    Float3 wo) noexcept {
    return max(luminance(evaluate_direct_illuminance(
        bsdf, emission, light_normal, light_point, world_pos, ns, wo)), 0.0f);
}

[[nodiscard]] inline Float evaluate_p_hat_with_geometry(
    const MaterialBSDF& bsdf,
    Float3 emission,
    Float3 light_dir, Float dist_sq,
    Float cos_shading, Float cos_light,
    Float3 ns, Float3 wo) noexcept {
    return max(luminance(evaluate_direct_illuminance_with_geometry(
        bsdf, emission, light_dir, dist_sq, cos_shading, cos_light, ns, wo)), 0.0f);
}

[[nodiscard]] inline Float evaluate_p_hat_env(
    const MaterialBSDF& bsdf,
    Float3 env_radiance, Float3 wi, Float3 ns, Float3 wo) noexcept {
    return max(luminance(evaluate_env_contribution(
        bsdf, env_radiance, wi, ns, wo)), 0.0f);
}

//==============================================================================
// Legacy scalar p_hat / direct-illuminance helpers
//
// Rebuild a MaterialBSDF from scalar surface fields and call build_lobe_list()
// (Phase-1 single-layer path). Use ONLY when no SurfaceData is available
// (e.g. x2-NEE in gi_bounce, mirror_nee_at_x2, raster shade path). For
// primary-surface evaluation, prefer the MaterialBSDF const& overloads above
// so coat/fuzz/composed-LobeList propagate into target_pdf.
//==============================================================================

/// Compute unshadowed direct illumination with diffuse/specular separated.
/// Returns diffuse contribution; specular via out_param.
[[nodiscard]] inline Float3 evaluate_direct_illuminance_split(
    Float3 emission,
    Float3 light_normal,
    Float3 light_point, Float3 world_pos, Float3 ns,
    Float3 wo,
    Float3 base_color, Float roughness, Float metallic, Float ior,
    UInt material_type,
    Float3& out_specular,
    Float sheen_val = 0.f, Float sheen_tint_val = 0.f,
    Float clearcoat_val = 0.f, Float clearcoat_gloss_val = 0.5f,
    Float iridescence_val = 0.f,
    Float iridescence_ior_val = 1.3f,
    Float iridescence_thickness_val = 0.f,
    Float anisotropic_val = 0.f,
    Float anisotropic_rot_val = 0.f,
    Float3 tangent_dir = luisa::compute::make_float3(1.f, 0.f, 0.f),
    Float bitangent_sign = 1.f,
    Float3 conductor_eta = luisa::compute::make_float3(1.f, 1.f, 1.f),
    Float3 conductor_k = luisa::compute::make_float3(0.f, 0.f, 0.f)) noexcept {
    Float3 to_light    = light_point - world_pos;
    Float  dist_sq     = max(dot(to_light, to_light), 1e-6f);
    Float3 light_dir   = to_light * rsqrt(dist_sq);

    Float cos_shading  = max(0.0f, dot(ns, light_dir));
    Float cos_light    = max(0.0f, dot(light_normal, -light_dir));

    Float3 result = def(make_float3(0.0f));
    out_specular = def(make_float3(0.0f));
    $if(material_type != 3u) {
        MaterialBSDF bsdf = make_material_bsdf(
            base_color, roughness, metallic, ior,
            sheen_val, sheen_tint_val,
            clearcoat_val, clearcoat_gloss_val,
            iridescence_val, iridescence_ior_val, iridescence_thickness_val,
            anisotropic_val, anisotropic_rot_val,
            tangent_dir, bitangent_sign,
            conductor_eta, conductor_k,
            ns);
        Float3 diff_brdf, spec_brdf;
        bsdf.evaluate_split(wo, light_dir, ns, diff_brdf, spec_brdf);
        Float3 geom = emission * cos_shading * cos_light / dist_sq;
        result = geom * diff_brdf;
        out_specular = geom * spec_brdf;
    };
    return result;
}

/// Lean variant of evaluate_direct_illuminance that skips the internal
/// `to_light`/`dist_sq`/`light_dir`/`cos_shading`/`cos_light` computation.
/// Callers that need those terms anyway (e.g. for BRDF-PDF MIS blending in
/// ReSTIR candidate loops) compute them once and pass them in, avoiding
/// duplicate vector math per candidate.
[[nodiscard]] inline Float3 evaluate_direct_illuminance_with_geometry(
    Float3 emission,
    Float3 light_dir, Float dist_sq,
    Float cos_shading, Float cos_light,
    Float3 ns, Float3 wo,
    Float3 base_color, Float roughness, Float metallic, Float ior,
    UInt material_type,
    Float sheen_val = 0.f, Float sheen_tint_val = 0.f,
    Float clearcoat_val = 0.f, Float clearcoat_gloss_val = 0.5f,
    Float iridescence_val = 0.f,
    Float iridescence_ior_val = 1.3f,
    Float iridescence_thickness_val = 0.f,
    Float anisotropic_val = 0.f,
    Float anisotropic_rot_val = 0.f,
    Float3 tangent_dir = luisa::compute::make_float3(1.f, 0.f, 0.f),
    Float bitangent_sign = 1.f,
    // Complex-IOR conductor Fresnel (defaults preserve legacy Schlick behavior):
    // conductor_eta uses the same overloaded slot as MaterialData::attenuation.
    Float3 conductor_eta = luisa::compute::make_float3(1.f, 1.f, 1.f),
    Float3 conductor_k = luisa::compute::make_float3(0.f, 0.f, 0.f)) noexcept {
    Float3 result = def(make_float3(0.0f));
    $if(material_type != 3u) {
        MaterialBSDF bsdf = make_material_bsdf(
            base_color, roughness, metallic, ior,
            sheen_val, sheen_tint_val,
            clearcoat_val, clearcoat_gloss_val,
            iridescence_val, iridescence_ior_val, iridescence_thickness_val,
            anisotropic_val, anisotropic_rot_val,
            tangent_dir, bitangent_sign,
            conductor_eta, conductor_k,
            ns);
        Float3 brdf = bsdf.evaluate(wo, light_dir, ns);
        result = emission * brdf * cos_shading * cos_light / dist_sq;
    };
    return result;
}

/// p_hat variant that accepts pre-computed geometry terms. Used by the
/// ReSTIR candidate loop where the same geometry is needed for BRDF-PDF MIS
/// blending — computing it once instead of twice per candidate.
[[nodiscard]] inline Float evaluate_p_hat_with_geometry(
    Float3 emission,
    Float3 light_dir, Float dist_sq,
    Float cos_shading, Float cos_light,
    Float3 ns, Float3 wo,
    Float3 base_color, Float roughness, Float metallic, Float ior,
    UInt material_type,
    Float sheen_val = 0.f, Float sheen_tint_val = 0.f,
    Float clearcoat_val = 0.f, Float clearcoat_gloss_val = 0.5f,
    Float iridescence_val = 0.f,
    Float iridescence_ior_val = 1.3f,
    Float iridescence_thickness_val = 0.f,
    Float anisotropic_val = 0.f,
    Float anisotropic_rot_val = 0.f,
    Float3 tangent_dir = luisa::compute::make_float3(1.f, 0.f, 0.f),
    Float bitangent_sign = 1.f,
    Float3 conductor_eta = luisa::compute::make_float3(1.f, 1.f, 1.f),
    Float3 conductor_k = luisa::compute::make_float3(0.f, 0.f, 0.f)) noexcept {
    return max(luminance(evaluate_direct_illuminance_with_geometry(
        emission, light_dir, dist_sq, cos_shading, cos_light,
        ns, wo, base_color, roughness, metallic, ior, material_type,
        sheen_val, sheen_tint_val, clearcoat_val, clearcoat_gloss_val,
        iridescence_val, iridescence_ior_val, iridescence_thickness_val,
        anisotropic_val, anisotropic_rot_val, tangent_dir, bitangent_sign,
        conductor_eta, conductor_k)), 0.0f);
}

/// Compute unshadowed direct illumination from environment light (Float3 RGB).
/// No cos_light/dist_sq terms — environment light is infinitely far away.
[[nodiscard]] inline Float3 evaluate_env_contribution(
    Float3 env_radiance, Float3 wi, Float3 ns, Float3 wo,
    Float3 base_color, Float roughness, Float metallic, Float ior,
    UInt material_type,
    Float sheen_val = 0.f, Float sheen_tint_val = 0.f,
    Float clearcoat_val = 0.f, Float clearcoat_gloss_val = 0.5f,
    Float iridescence_val = 0.f,
    Float iridescence_ior_val = 1.3f,
    Float iridescence_thickness_val = 0.f,
    Float anisotropic_val = 0.f,
    Float anisotropic_rot_val = 0.f,
    Float3 tangent_dir = luisa::compute::make_float3(1.f, 0.f, 0.f),
    Float bitangent_sign = 1.f,
    Float3 conductor_eta = luisa::compute::make_float3(1.f, 1.f, 1.f),
    Float3 conductor_k = luisa::compute::make_float3(0.f, 0.f, 0.f)) noexcept {
    Float cos_theta = max(0.0f, dot(ns, wi));
    Float3 result = def(make_float3(0.0f));
    $if(material_type != 3u) {
        MaterialBSDF bsdf = make_material_bsdf(
            base_color, roughness, metallic, ior,
            sheen_val, sheen_tint_val,
            clearcoat_val, clearcoat_gloss_val,
            iridescence_val, iridescence_ior_val, iridescence_thickness_val,
            anisotropic_val, anisotropic_rot_val,
            tangent_dir, bitangent_sign,
            conductor_eta, conductor_k,
            ns);
        Float3 brdf = bsdf.evaluate(wo, wi, ns);
        result = env_radiance * brdf * cos_theta;
    };
    return result;
}

/// Evaluate p_hat (scalar luminance) for environment light RIS weights.
[[nodiscard]] inline Float evaluate_p_hat_env(
    Float3 env_radiance, Float3 wi, Float3 ns, Float3 wo,
    Float3 base_color, Float roughness, Float metallic, Float ior,
    UInt material_type,
    Float sheen_val = 0.f, Float sheen_tint_val = 0.f,
    Float clearcoat_val = 0.f, Float clearcoat_gloss_val = 0.5f,
    Float iridescence_val = 0.f,
    Float iridescence_ior_val = 1.3f,
    Float iridescence_thickness_val = 0.f,
    Float anisotropic_val = 0.f,
    Float anisotropic_rot_val = 0.f,
    Float3 tangent_dir = luisa::compute::make_float3(1.f, 0.f, 0.f),
    Float bitangent_sign = 1.f,
    Float3 conductor_eta = luisa::compute::make_float3(1.f, 1.f, 1.f),
    Float3 conductor_k = luisa::compute::make_float3(0.f, 0.f, 0.f)) noexcept {
    return max(luminance(evaluate_env_contribution(
        env_radiance, wi, ns, wo,
        base_color, roughness, metallic, ior, material_type,
        sheen_val, sheen_tint_val, clearcoat_val, clearcoat_gloss_val,
        iridescence_val, iridescence_ior_val, iridescence_thickness_val,
        anisotropic_val, anisotropic_rot_val, tangent_dir, bitangent_sign,
        conductor_eta, conductor_k)), 0.0f);
}

/// Convert equirectangular UV to world direction (with envmap rotation).
[[nodiscard]] inline Float3 envmap_uv_to_direction(Float u, Float v, UInt width, UInt height,
    Float3x3 env_rotation) noexcept {
    Float theta = 3.14159265359f * (cast<float>(cast<uint>(v * cast<float>(height))) + 0.5f) / cast<float>(height);
    Float phi   = 2.0f * 3.14159265359f * (cast<float>(cast<uint>(u * cast<float>(width))) + 0.5f) / cast<float>(width);
    Float sin_theta = sin(theta);
    Float3 local_dir = make_float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
    return env_rotation * local_dir;
}

/// Convert world direction to equirectangular UV (with inverse envmap rotation).
[[nodiscard]] inline Float2 direction_to_envmap_uv(Float3 dir, UInt width, UInt height,
    Float3x3 env_rotation) noexcept {
    Float3 local_dir = transpose(env_rotation) * dir;
    Float u = atan2(local_dir.z, local_dir.x) * (1.0f / (2.0f * 3.14159265359f)) + 0.5f;
    Float v = acos(clamp(local_dir.y, -1.0f, 1.0f)) * (1.0f / 3.14159265359f);
    return make_float2(u, v);
}

/// Sample environment map direction using hierarchical CDF (binary search).
/// Returns (direction, uv.x, uv.y, pdf_solid_angle). Direction is in world space.
[[nodiscard]] inline auto sample_envmap_cdf(
    Float2 u, const BufferVar<float> &marginal_cdf, const BufferVar<float> &conditional_cdf,
    UInt width, UInt height, Float integral, Float3x3 env_rotation) noexcept {
    // Binary search marginal CDF for row
    Float u_marginal = u.x * integral;
    UInt lo = def(0u);
    UInt hi = height;
    $while(lo < hi) {
        UInt mid = (lo + hi) >> 1u;
        $if(marginal_cdf.read(mid + 1u) < u_marginal) {
            lo = mid + 1u;
        } $else {
            hi = mid;
        };
    };
    UInt row = min(lo, height - 1u);

    // Binary search conditional CDF for column
    UInt row_offset = row * (width + 1u);
    Float row_total = conditional_cdf.read(row_offset + width);
    Float u_cond = u.y * row_total;
    lo = def(0u);
    hi = width;
    $while(lo < hi) {
        UInt mid = (lo + hi) >> 1u;
        $if(conditional_cdf.read(row_offset + mid + 1u) < u_cond) {
            lo = mid + 1u;
        } $else {
            hi = mid;
        };
    };
    UInt col = min(lo, width - 1u);

    // Direction from equirectangular coordinates (envmap-local space)
    Float theta = (cast<float>(row) + 0.5f) * 3.14159265359f / cast<float>(height);
    Float phi   = (cast<float>(col) + 0.5f) * 2.0f * 3.14159265359f / cast<float>(width);
    Float sin_theta = sin(theta);
    Float3 local_dir = make_float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));

    // Apply rotation to get world-space direction
    Float3 dir = env_rotation * local_dir;

    // UV for storing in reservoir (envmap-local, rotation-independent)
    Float env_u = (cast<float>(col) + 0.5f) / cast<float>(width);
    Float env_v = (cast<float>(row) + 0.5f) / cast<float>(height);

    // PDF: marginal * conditional * (W*H) / (2*pi^2 * sin_theta)
    // Rotation has Jacobian=1, so PDF is the same in world space
    Float texel_weight = conditional_cdf.read(row_offset + col + 1u)
        - conditional_cdf.read(row_offset + col);
    Float marginal_pdf = row_total / max(integral, 1e-10f);
    Float conditional_pdf = texel_weight / max(row_total, 1e-10f);
    Float pdf = marginal_pdf * conditional_pdf * cast<float>(width * height)
        / max(2.0f * 9.86960440109f * sin_theta, 1e-10f);

    return std::make_tuple(dir, env_u, env_v, pdf);
}

/// Evaluate envmap solid-angle PDF for a given world-space direction.
[[nodiscard]] inline Float eval_envmap_pdf(
    Float3 dir, const BufferVar<float> &marginal_cdf, const BufferVar<float> &conditional_cdf,
    UInt width, UInt height, Float integral, Float3x3 env_rotation) noexcept {
    Float3 local_dir = transpose(env_rotation) * dir;
    Float u = atan2(local_dir.z, local_dir.x) * (1.0f / (2.0f * 3.14159265359f)) + 0.5f;
    Float v = acos(clamp(local_dir.y, -1.0f, 1.0f)) * (1.0f / 3.14159265359f);
    UInt col = min(cast<uint>(u * cast<float>(width)), width - 1u);
    UInt row = min(cast<uint>(v * cast<float>(height)), height - 1u);
    UInt row_offset = row * (width + 1u);
    Float row_total = conditional_cdf.read(row_offset + width);
    Float texel_weight = conditional_cdf.read(row_offset + col + 1u)
        - conditional_cdf.read(row_offset + col);
    Float marginal_pdf = row_total / max(integral, 1e-10f);
    Float conditional_pdf = texel_weight / max(row_total, 1e-10f);
    Float sin_theta = sqrt(max(1.0f - local_dir.y * local_dir.y, 1e-10f));
    Float pdf = marginal_pdf * conditional_pdf * cast<float>(width * height)
        / max(2.0f * 9.86960440109f * sin_theta, 1e-10f);
    return pdf;
}

/// Evaluate envmap radiance from world-space direction via texture lookup.
/// Applies rotation + exposure.
[[nodiscard]] inline Float3 eval_envmap_radiance(
    Float3 dir, const ImageVar<float> &envmap, UInt width, UInt height,
    Float3x3 env_rotation, Float env_exposure) noexcept {
    Float3 local_dir = transpose(env_rotation) * dir;
    Float u = atan2(local_dir.z, local_dir.x) * (1.0f / (2.0f * 3.14159265359f)) + 0.5f;
    Float v = acos(clamp(local_dir.y, -1.0f, 1.0f)) * (1.0f / 3.14159265359f);
    UInt col = min(cast<uint>(u * cast<float>(width)), width - 1u);
    UInt row = min(cast<uint>(v * cast<float>(height)), height - 1u);
    return envmap.read(make_uint2(col, row)).xyz() * env_exposure;
}

/// Evaluate envmap radiance from stored UV (for shade/reuse passes).
/// Applies exposure only (UVs are envmap-local, no rotation needed).
[[nodiscard]] inline Float3 eval_envmap_from_uv(
    Float env_u, Float env_v, const ImageVar<float> &envmap,
    UInt width, UInt height, Float env_exposure) noexcept {
    UInt col = min(cast<uint>(env_u * cast<float>(width)), width - 1u);
    UInt row = min(cast<uint>(env_v * cast<float>(height)), height - 1u);
    return envmap.read(make_uint2(col, row)).xyz() * env_exposure;
}

/// Reconstruct world-space direction from stored UV (for reuse passes).
[[nodiscard]] inline Float3 uv_to_direction(Float env_u, Float env_v, UInt width, UInt height,
    Float3x3 env_rotation) noexcept {
    UInt col = min(cast<uint>(env_u * cast<float>(width)), width - 1u);
    UInt row = min(cast<uint>(env_v * cast<float>(height)), height - 1u);
    Float theta = (cast<float>(row) + 0.5f) * 3.14159265359f / cast<float>(height);
    Float phi   = (cast<float>(col) + 0.5f) * 2.0f * 3.14159265359f / cast<float>(width);
    Float sin_theta = sin(theta);
    Float3 local_dir = make_float3(sin_theta * cos(phi), cos(theta), sin_theta * sin(phi));
    return env_rotation * local_dir;
}

//==============================================================================
// Texture Sampling Helpers
//==============================================================================

/// Reconstruct interpolated UV coordinates from bindless vertex access.
[[nodiscard]] inline Float2 reconstruct_uv(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary) noexcept {
    using Triangle = luisa::compute::Triangle;
    using Vertex   = newtype::util::ActiveVertex;   // GPU layout (A2)

    auto tri = vertex_bindless.buffer<Triangle>(tri_slot).read(prim_id);
    auto v0  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i0);
    auto v1  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i1);
    auto v2  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i2);

    return v0->uv() * (1.0f - bary.x - bary.y) +
           v1->uv() * bary.x +
           v2->uv() * bary.y;
}

/// Result of `reconstruct_unjittered_hit`. When `valid` is false, `bary`
/// falls back to the caller-provided jittered bary; `position` and `ray_t`
/// are not meaningful and must not be consumed without a validity gate.
struct UnjitHit {
    Float2 bary;
    Float3 position;
    Float  ray_t;
    Bool   valid;
};

/// Compute where an unjittered pixel-center ray would hit the same triangle
/// the jittered ray hit. Returns barycentric coordinates, hit position, and
/// ray parameter t along `unjit_dir`. Used for stable texture sampling and
/// stable glass PSR refraction (albedo/normal/refraction/absorption land on
/// the same texel each frame regardless of Halton jitter).
///
/// Falls back to jit_bary when the unjittered ray grazes the triangle plane
/// or lands outside the triangle (silhouette / edge pixels). `position` and
/// `ray_t` are still computed but only meaningful when `valid` is true.
[[nodiscard]] inline UnjitHit reconstruct_unjittered_hit(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id,
    Float2 jit_bary,
    Float3 cam_origin, Float3 unjit_dir) noexcept {
    using Triangle = luisa::compute::Triangle;
    using Vertex   = newtype::util::ActiveVertex;   // GPU layout (A2)

    auto tri = vertex_bindless.buffer<Triangle>(tri_slot).read(prim_id);
    auto v0  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i0);
    auto v1  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i1);
    auto v2  = vertex_bindless.buffer<Vertex>(vertex_slot).read(tri.i2);

    Float3 p0 = v0->position();
    Float3 p1 = v1->position();
    Float3 p2 = v2->position();

    Float3 e01 = p1 - p0;
    Float3 e02 = p2 - p0;
    Float3 N   = cross(e01, e02);

    // Ray-plane intersection: t such that O + t*D lies in the triangle plane.
    Float  denom      = dot(unjit_dir, N);
    Bool   denom_safe = abs(denom) > 1e-6f;
    Float  safe_denom = ite(denom_safe, denom, 1.0f);
    Float  t          = dot(p0 - cam_origin, N) / safe_denom;
    Float3 P          = cam_origin + t * unjit_dir;

    // Convert P to barycentric via the standard 2x2 solve.
    Float3 v0P = P - p0;
    Float  d11 = dot(e01, e01);
    Float  d12 = dot(e01, e02);
    Float  d22 = dot(e02, e02);
    Float  d31 = dot(v0P, e01);
    Float  d32 = dot(v0P, e02);
    Float  dd  = d11 * d22 - d12 * d12;
    Bool   dd_safe = abs(dd) > 1e-10f;
    Float  safe_dd = ite(dd_safe, dd, 1.0f);
    Float  b1 = (d22 * d31 - d12 * d32) / safe_dd;
    Float  b2 = (d11 * d32 - d12 * d31) / safe_dd;
    Float  b0 = 1.0f - b1 - b2;

    // Reject if the unjittered ray lands outside the triangle (silhouette).
    Bool bary_in_tri = (b0 >= -0.02f) & (b0 <= 1.02f)
                     & (b1 >= -0.02f) & (b1 <= 1.02f)
                     & (b2 >= -0.02f) & (b2 <= 1.02f);
    Bool use_unjit = denom_safe & dd_safe & bary_in_tri;

    UnjitHit result;
    result.bary     = ite(use_unjit, make_float2(b1, b2), jit_bary);
    result.position = P;
    result.ray_t    = t;
    result.valid    = use_unjit;
    return result;
}

/// Convenience wrapper returning only the stable barycentric coordinates.
/// Existing callers (shade, prefilter) that don't need position / ray_t.
[[nodiscard]] inline Float2 reconstruct_unjittered_bary(
    const BindlessVar& vertex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id,
    Float2 jit_bary,
    Float3 cam_origin, Float3 unjit_dir) noexcept {
    return reconstruct_unjittered_hit(
        vertex_bindless, vertex_slot, tri_slot, prim_id,
        jit_bary, cam_origin, unjit_dir).bary;
}

/// MeshTriVerts overload — reuses a pre-fetched triangle instead of issuing
/// another 4 bindless reads. Pair with resolve_surface_from_instance_verts
/// so the unjittered-bary recompute and the surface resolve share one fetch.
[[nodiscard]] inline UnjitHit reconstruct_unjittered_hit(
    const MeshTriVerts& tv,
    Float2 jit_bary,
    Float3 cam_origin, Float3 unjit_dir) noexcept {

    Float3 p0 = tv.v0->position();
    Float3 p1 = tv.v1->position();
    Float3 p2 = tv.v2->position();

    Float3 e01 = p1 - p0;
    Float3 e02 = p2 - p0;
    Float3 N   = cross(e01, e02);

    // Ray-plane intersection: t such that O + t*D lies in the triangle plane.
    Float  denom      = dot(unjit_dir, N);
    Bool   denom_safe = abs(denom) > 1e-6f;
    Float  safe_denom = ite(denom_safe, denom, 1.0f);
    Float  t          = dot(p0 - cam_origin, N) / safe_denom;
    Float3 P          = cam_origin + t * unjit_dir;

    // Convert P to barycentric via the standard 2x2 solve.
    Float3 v0P = P - p0;
    Float  d11 = dot(e01, e01);
    Float  d12 = dot(e01, e02);
    Float  d22 = dot(e02, e02);
    Float  d31 = dot(v0P, e01);
    Float  d32 = dot(v0P, e02);
    Float  dd  = d11 * d22 - d12 * d12;
    Bool   dd_safe = abs(dd) > 1e-10f;
    Float  safe_dd = ite(dd_safe, dd, 1.0f);
    Float  b1 = (d22 * d31 - d12 * d32) / safe_dd;
    Float  b2 = (d11 * d32 - d12 * d31) / safe_dd;
    Float  b0 = 1.0f - b1 - b2;

    // Reject if the unjittered ray lands outside the triangle (silhouette).
    Bool bary_in_tri = (b0 >= -0.02f) & (b0 <= 1.02f)
                     & (b1 >= -0.02f) & (b1 <= 1.02f)
                     & (b2 >= -0.02f) & (b2 <= 1.02f);
    Bool use_unjit = denom_safe & dd_safe & bary_in_tri;

    UnjitHit result;
    result.bary     = ite(use_unjit, make_float2(b1, b2), jit_bary);
    result.position = P;
    result.ray_t    = t;
    result.valid    = use_unjit;
    return result;
}

/// Convenience wrapper (MeshTriVerts overload).
[[nodiscard]] inline Float2 reconstruct_unjittered_bary(
    const MeshTriVerts& tv,
    Float2 jit_bary,
    Float3 cam_origin, Float3 unjit_dir) noexcept {
    return reconstruct_unjittered_hit(tv, jit_bary, cam_origin, unjit_dir).bary;
}

/// Deterministic alpha cutout test: returns true if the surface is cut out
/// (ray should pass through). Samples the albedo texture's alpha channel
/// and compares against the material's alphacut threshold.
[[nodiscard]] inline Bool is_alpha_cutout(
    const Var<MaterialData>& mat,
    const BindlessVar& vertex_bindless,
    const BindlessVar& tex_bindless,
    UInt4 inst_data,
    UInt prim,
    Float2 bary) noexcept {
    Bool cutout = def(false);
    $if(mat.alphacut > 0.0f) {
        Float2 uv = reconstruct_uv(vertex_bindless, inst_data.z, inst_data.w, prim, bary);
        Float pixel_alpha;
        using luisa::compute::cast;
        UInt albedo_slot = ite(mat.albedoTexIdx >= 0, cast<uint>(mat.albedoTexIdx), 0u);
        $if(mat.albedoTexIdx >= 0) {
            pixel_alpha = mat.albedo.w * tex_bindless->tex2d(albedo_slot).sample(uv).w;
        } $else {
            pixel_alpha = mat.albedo.w;
        };
        cutout = pixel_alpha < mat.alphacut;
    };
    return cutout;
}

/// Perturb geometric normal using a tangent-space normal map.
/// Falls back to the geometric normal if the normal texture index is -1
/// or if the tangent frame is degenerate.
[[nodiscard]] inline Float3 perturb_normal(
    Float3 geo_ns, Float3 tangent, Float tangent_w,
    Expr<int> normalTexIdx,
    Float2 uv,
    const BindlessVar& tex_bindless) noexcept {

    Float3 result = geo_ns;
    $if(normalTexIdx >= 0) {
        // Check tangent validity
        Float tangent_len = length(tangent);
        $if(tangent_len > 1e-6f) {
            Float3 t = tangent * (1.0f / tangent_len);
            Float3 b = cross(geo_ns, t) * tangent_w;
            Float3 n = geo_ns;

            // Sample normal map and decode from [0,1] to [-1,1]
            // Use safe index — GPU SIMD may speculatively evaluate even inside $if
            UInt safe_idx = ite(normalTexIdx >= 0, cast<uint>(normalTexIdx), 0u);
            Float4 n_sample = tex_bindless->tex2d(safe_idx).sample(uv);
            Float3 tsn = make_float3(n_sample.xy() * 2.0f - 1.0f, n_sample.z);

            // Reconstruct Z for BC5/2-channel normal maps
            Float z_sq = max(1.0f - tsn.x * tsn.x - tsn.y * tsn.y, 0.0f);
            tsn = make_float3(tsn.x, tsn.y, sqrt(z_sq));

            // Transform tangent-space normal to world space via TBN
            result = normalize(t * tsn.x + b * tsn.y + n * tsn.z);
        };
    };
    return result;
}

/// Core resolution body over a pre-fetched triangle (read_mesh_triangle).
/// All vertex-dependent attributes (uv / position / normals / tangent) are
/// reconstructed from that single fetch. The slot-based entry points below
/// funnel through here so a surface resolve costs exactly 1 triangle +
/// 3 vertex reads instead of one 4-read fetch per attribute.
[[nodiscard]] inline SurfaceData resolve_surface_verts(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& tex_bindless,
    const MeshTriVerts& tv,
    Float2 bary,
    Var<MaterialData> material,
    Float3 wo,
    Float4x4 instance_transform,
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u,
    UInt instance_flags = 0u,
    UInt instance_index = 0u) noexcept {

    SurfaceData s;

    // --- Interpolate UV from barycentric coordinates ---
    Float2 uv = reconstruct_uv(tv, bary);

    // --- Sample textures (fall back to flat constants when texIdx == -1) ---
    // Use ite() to compute safe uint indices — never pass cast<uint>(-1) = 0xFFFFFFFF
    // to tex2d(). GPU SIMD can speculatively evaluate bindless accesses even inside
    // $if() branches, so the index must always be valid to avoid a fault.
    UInt albedo_slot  = ite(material.albedoTexIdx  >= 0, cast<uint>(material.albedoTexIdx),  0u);
    UInt emissive_slot= ite(material.emissiveTexIdx >= 0, cast<uint>(material.emissiveTexIdx), 0u);
    UInt rma_slot     = ite(material.rmaTexIdx      >= 0, cast<uint>(material.rmaTexIdx),      0u);
    {
        Float3 albedo_result;
        Float albedo_alpha_result;
        $if(material.albedoTexIdx >= 0) {
            Float4 tex_sample = tex_bindless->tex2d(albedo_slot).sample(uv);
            albedo_result = material.albedo.xyz() * tex_sample.xyz();
            albedo_alpha_result = material.albedo.w * tex_sample.w;
        } $else {
            albedo_result = material.albedo.xyz();
            albedo_alpha_result = material.albedo.w;
        };
        s.albedo = albedo_result;
        s.albedo_alpha = albedo_alpha_result;
    }
    {
        Float3 emission_result;
        $if(material.emissiveTexIdx >= 0) {
            emission_result = material.emission * tex_bindless->tex2d(emissive_slot).sample(uv).xyz();
        } $else {
            emission_result = material.emission;
        };
        s.emission = emission_result;
    }
    {
        // Gate the fetch on rmaTexIdx (matches albedo/emissive above) — the
        // previous unconditional sample cost a wasted bilinear fetch on
        // every untextured surface resolve.
        Float roughness_result;
        Float metallic_result;
        Float ao_result;
        $if(material.rmaTexIdx >= 0) {
            Float4 rma_sample = tex_bindless->tex2d(rma_slot).sample(uv);
            roughness_result = rma_sample.x;
            metallic_result  = rma_sample.y;
            ao_result        = rma_sample.z;
        } $else {
            roughness_result = material.roughness;
            metallic_result  = material.metallic;
            ao_result        = 1.0f;
        };
        s.roughness = roughness_result;
        s.metallic  = metallic_result;
        s.ao        = ao_result;
    }
    {
        // Iridescence thickness map — R = factor mask (multiplies the
        // iridescence strength), G = thickness mix parameter:
        //   thickness = mix(thickness_min, thickness_max, G)
        // Active only when a texture is bound AND thickness_max >= 0 (the -1
        // sentinel keeps untextured materials on the flat thickness).
        UInt irid_slot = ite(material.iridescenceTexIdx >= 0,
                             cast<uint>(material.iridescenceTexIdx), 0u);
        Float iridescence_result;
        Float thickness_result;
        $if(material.iridescenceTexIdx >= 0) {
            Float4 irid_sample = tex_bindless->tex2d(irid_slot).sample(uv);
            iridescence_result = material.iridescence * irid_sample.x;
            thickness_result = ite(material.iridescence_thickness_max >= 0.0f,
                lerp(material.iridescence_thickness,
                     material.iridescence_thickness_max, irid_sample.y),
                material.iridescence_thickness);
        } $else {
            iridescence_result = material.iridescence;
            thickness_result = material.iridescence_thickness;
        };
        s.iridescence           = iridescence_result;
        s.iridescence_thickness = thickness_result;
    }

    // --- Copy non-textured parameters ---
    s.ior                   = material.ior;
    s.alphacut              = material.alphacut;
    s.sheen                 = material.sheen;
    s.sheen_tint            = material.sheen_tint;
    s.clearcoat             = material.clearcoat;
    s.clearcoat_gloss       = material.clearcoat_gloss;
    s.iridescence_ior       = material.iridescence_ior;
    s.anisotropic           = material.anisotropic;
    s.anisotropic_rot       = material.anisotropic_rot;
    s.attenuation           = material.attenuation;
    s.attenuation_distance  = material.attenuation_distance;
    s.conductor_k           = material.conductor_k;
    s.specular_tint         = material.specular_tint;
    s.specular_trans        = material.specular_trans;
    s.dispersion            = material.dispersion;
    s.flatness              = material.flatness;
    s.diffuse_trans         = material.diffuse_trans;
    s.fabric                = material.fabric;
    // Glass-blend default (1 = pure glass). Set BEFORE the callable dispatch
    // so custom resolvers can lower it; built-in glass never touches it.
    s.glass_blend           = 1.0f;
    s.material_type         = material.type;
    s.bsdf_type             = get_effective_bsdf_type(material);

    // --- Reconstruct position (object → world via instance transform) ---
    Float3 obj_pos = reconstruct_object_position(tv, bary);
    s.position = (instance_transform * make_float4(obj_pos, 1.0f)).xyz();

    // --- Reconstruct geometry ---
    Bool is_double_sided = (instance_flags & kDoubleSidedFlag) != 0u;
    s.geo_ns = ite(is_double_sided,
        reconstruct_face_normal(tv),
        reconstruct_normal(tv, bary));
    Float4 t_frame = reconstruct_tangent(tv, bary);
    s.tangent   = make_float3(normalize(t_frame.xyz()));
    s.tangent_w = t_frame.w;

    // --- Normal map perturbation ---
    s.ns = perturb_normal(s.geo_ns, s.tangent, s.tangent_w,
                          material.normalTexIdx, uv, tex_bindless);

    // --- Flip normal to face viewer ---
    s.ns = ite(dot(wo, s.ns) < 0.0f, -s.ns, s.ns);

    // --- UV (for downstream custom callables) ---
    s.uv = uv;

    // --- Instance identity (track B2): visible to custom callables below. ---
    s.instance_index = instance_index;

    // --- Material Callable Dispatch (Polymorphic) ---
    // Per-type resolvers transform surface parameters (procedural effects).
    // Built-in types (0-13): IdentitySurfaceResolver (no-op).
    // Custom types (14+): user-defined CustomSurfaceResolver<F>.
    // Polymorphic::dispatch() generates a $switch at AST build time.
    polymorphic.dispatch(material.type, [&](const SurfaceResolver* resolver) {
        resolver->resolve(s, material, screen_uv, wo, time,
            tex_bindless, screen_w, screen_h);
    });

    // --- Transform object-space fields to world space ---
    // Position is already world (above). Normals/tangent still object-space
    // from the bindless reads + normal map; fold the transform in here so
    // callers no longer need a post-call transform_surface_normals().
    transform_surface_normals(s, instance_transform, wo);

#if NT_ENABLE_MS_GGX
    // --- Kulla-Conty MS-GGX invariants (conductors, roughness-gated) ---
    // E(mu_o), E_avg and F_avg depend only on material + wo — hoisted here so
    // NEE/reuse loops pay one E fit per surface resolution instead of per
    // evaluate call. f_ms is assembled in MaterialBSDF's spec blocks (only the
    // E_i fit remains per-call). Same alpha math as the spec blocks.
    // See docs/ms_ggx_compensation.md.
    $if(s.metallic > 0.5f) {
        Float aspect_ms = sqrt(max(1.f - s.anisotropic * 0.9f, 0.1f));
        Float2 alpha_ms = roughness_to_alpha(make_float2(
            max(s.roughness / aspect_ms, 1e-4f),
            max(s.roughness * aspect_ms, 1e-4f)));
        Float a_ms = sqrt(alpha_ms.x * alpha_ms.y);
        $if(a_ms > 0.0225f) {
            // Closed-form complex F0 |(eta+ik)-1|^2/|(eta+ik)+1|^2; legacy
            // metals use albedo as F0. Then Schlick /21 average.
            Bool use_complex_ms = any(s.conductor_k != 0.f);
            Float3 F0_ms = ite(use_complex_ms,
                (sqr(s.attenuation - 1.f) + sqr(s.conductor_k))
                    / (sqr(s.attenuation + 1.f) + sqr(s.conductor_k)),
                s.albedo);
            s.ms_f_avg = schlick_f_avg(F0_ms);
            s.ms_e_o   = ggx_directional_albedo_fit(abs(dot(wo, s.ns)), a_ms);
            s.ms_e_avg = ggx_avg_albedo_fit(a_ms);
        };
    };
#endif

    return s;
}

//==============================================================================
// Slim resolve for shadow walks (perf review 2026-09 item 6).
//
// trace_shadow's transparent path (shared walk + dispersive sub-walks)
// consumes exactly four material scalars — ior, dispersion, attenuation,
// attenuation_distance — plus the interpolated geometric normal. The full
// resolve additionally samples up to 4 textures, reconstructs
// tangent/UV/position, perturbs the normal map, and computes MS-GGX
// invariants, none of which the walk reads. For built-in material types
// (< 14) the resolver is the identity no-op, so those scalars are exact
// copies of the MaterialData fields and the slim path skips everything
// else. Custom resolvers (>= 14) may drive ior/attenuation procedurally,
// so they fall back to the full resolve (sharing the same fused triangle
// fetch). Bit-identical to the former resolve_surface + unfused
// reconstruct_normal pair by construction (same math, same source values).
//==============================================================================
struct ShadowSurfaceData {
    Float3 geo_ns;                 // world-space interpolated geometric normal
    Float   ior;
    Float   dispersion;
    Float3 attenuation;
    Float   attenuation_distance;
    Float   glass_blend;           // 1 = pure glass (see SurfaceData.glass_blend)
};

[[nodiscard]] inline ShadowSurfaceData resolve_shadow_surface(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& vertex_bindless,
    const BindlessVar& tex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary,
    Var<MaterialData> material,
    Float3 wo,
    Float4x4 instance_transform,
    UInt instance_index = 0u) noexcept {
    // One fused fetch (1 triangle + 3 vertices) serves both the normal and
    // the custom-resolver fallback; the former call sites paid this twice
    // (once inside resolve_surface, once via the unfused reconstruct_normal).
    MeshTriVerts tv = read_mesh_triangle(vertex_bindless, vertex_slot, tri_slot, prim_id);
    ShadowSurfaceData s;
    s.geo_ns = transform_normal(instance_transform, reconstruct_normal(tv, bary));
    $if(material.type >= 14u) {
        SurfaceData full = resolve_surface_verts(
            polymorphic, tex_bindless, tv, bary, material, wo,
            instance_transform, 0.0f, make_float2(0.0f), 0u, 0u, 0u, instance_index);
        s.ior                  = full.ior;
        s.dispersion           = full.dispersion;
        s.attenuation          = full.attenuation;
        s.attenuation_distance = full.attenuation_distance;
        s.glass_blend          = full.glass_blend;
    } $else {
        s.ior                  = material.ior;
        s.dispersion           = material.dispersion;
        s.attenuation          = material.attenuation;
        s.attenuation_distance = material.attenuation_distance;
        s.glass_blend          = 1.0f;
    };
    return s;
}

/// Resolve all surface parameters from material data + texture sampling.
///
/// This is the central function that connects the texture bindless array
/// to the material system. When all texture indices are -1, it returns
/// values identical to the flat MaterialData constants.
///
/// @param vertex_bindless  Bindless array for vertex buffers
/// @param tex_bindless     Bindless array for material textures
/// @param vertex_slot      Instance's vertex buffer bindless slot
/// @param tri_slot         Instance's triangle buffer bindless slot
/// @param prim_id          Primitive (triangle) index
/// @param bary             Barycentric coordinates
/// @param material         Material data from GPU buffer
/// @param wo               Outgoing view direction (for normal flipping)
/// @param instance_transform Object-to-world transform for this instance
///                           (scene.instance_transform_buffer.read(inst_id)).
///                           Applied to position and normals so the returned
///                           SurfaceData is fully in world space.
/// @param time             Time for animated effects (default 0)
/// @param screen_uv        Screen-space UV [0,1] for screen-space effects (default 0)
/// @param screen_w         Screen width (default 0)
/// @param screen_h         Screen height (default 0)
/// @param instance_index   TLAS instance row (track B2; default 0 = unknown)
[[nodiscard]] inline SurfaceData resolve_surface(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& vertex_bindless,
    const BindlessVar& tex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary,
    Var<MaterialData> material,
    Float3 wo,
    Float4x4 instance_transform,
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u,
    UInt instance_flags = 0u,
    UInt instance_index = 0u) noexcept {

    MeshTriVerts tv = read_mesh_triangle(vertex_bindless, vertex_slot, tri_slot, prim_id);
    return resolve_surface_verts(polymorphic, tex_bindless, tv, bary, material, wo,
        instance_transform, time, screen_uv, screen_w, screen_h, instance_flags, instance_index);
}

//==============================================================================
// Multi-Layer Surface Resolution
//==============================================================================

/// Resolve surface with weighted multi-layer blending.
///
/// Layer 0 is always active with implicit weight 1.0.
/// Layers 1-3 are skipped if their index is 0xFF.
/// Layer weight comes from frac(material.meta).
/// Emission is additive (not weight-normalized).
/// Normal map and material_type come from layer 0.
[[nodiscard]] inline SurfaceData resolve_surface_multilayer(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& vertex_bindless,
    const BindlessVar& tex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary,
    UInt material_layers,
    const BufferVar<MaterialData>& material_buffer,
    Float3 wo,
    Float4x4 instance_transform,
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u,
    UInt instance_flags = 0u,
    UInt instance_index = 0u) noexcept {

    // --- Accumulators for weighted blend ---
    Float3 accum_albedo   = def(make_float3(0.0f));
    Float3 accum_emission = def(make_float3(0.0f));
    Float  accum_roughness       = def(0.0f);
    Float  accum_metallic        = def(0.0f);
    Float  accum_ior             = def(0.0f);
    Float  accum_albedo_alpha    = def(0.0f);
    Float  accum_ao              = def(0.0f);
    Float  accum_sheen           = def(0.0f);
    Float  accum_sheen_tint      = def(0.0f);
    Float  accum_clearcoat       = def(0.0f);
    Float  accum_clearcoat_gloss = def(0.0f);
    Float  accum_iridescence           = def(0.0f);
    Float  accum_iridescence_ior       = def(0.0f);
    Float  accum_iridescence_thickness = def(0.0f);
    Float  accum_anisotropic           = def(0.0f);
    Float  accum_anisotropic_rot       = def(0.0f);
    Float3 accum_attenuation     = def(make_float3(0.0f));
    Float  accum_attenuation_distance  = def(0.0f);
    // Complex-IOR imaginary part (linear blend — physically wrong for stacked
    // conductors but no shipped scene stacks conductor as layer ≥1; see plan §4).
    Float3 accum_conductor_k     = def(make_float3(0.0f));
    Float  accum_specular_tint  = def(0.0f);
    Float  accum_specular_trans = def(0.0f);
    Float  accum_dispersion     = def(0.0f);
    Float  accum_flatness       = def(0.0f);
    Float  accum_diffuse_trans  = def(0.0f);
    Float  accum_fabric         = def(0.0f);
    Float  accum_glass_blend    = def(0.0f);
    Float  total_weight         = def(0.0f);

    // Layer 0's classification (determines downstream BSDF dispatch)
    UInt   layer0_type     = def(0u);
    UInt   layer0_bsdf_type = def(0u);

    // Single fused fetch shared by all layer resolutions + the geometry tail.
    MeshTriVerts tv = read_mesh_triangle(vertex_bindless, vertex_slot, tri_slot, prim_id);

    for (uint layer = 0; layer < 4; ++layer) {
        UInt mat_idx = (material_layers >> (layer * 8u)) & 0xFFu;
        $if(mat_idx != 0xFFu) {
            Var<MaterialData> layer_mat = material_buffer.read(mat_idx);
            Float weight = ite(layer == 0u, 1.0f, fract(layer_mat.meta));

            SurfaceData layer_s = resolve_surface_verts(
                polymorphic, tex_bindless, tv, bary,
                layer_mat, wo, instance_transform, time, screen_uv, screen_w, screen_h,
                0u, instance_index);

            accum_albedo               += layer_s.albedo * weight;
            accum_emission             += layer_s.emission * weight;
            accum_roughness            += layer_s.roughness * weight;
            accum_metallic             += layer_s.metallic * weight;
            accum_ior                  += layer_s.ior * weight;
            accum_albedo_alpha         += layer_s.albedo_alpha * weight;
            accum_ao                   += layer_s.ao * weight;
            accum_sheen                += layer_s.sheen * weight;
            accum_sheen_tint           += layer_s.sheen_tint * weight;
            accum_clearcoat            += layer_s.clearcoat * weight;
            accum_clearcoat_gloss      += layer_s.clearcoat_gloss * weight;
            accum_iridescence          += layer_s.iridescence * weight;
            accum_iridescence_ior      += layer_s.iridescence_ior * weight;
            accum_iridescence_thickness+= layer_s.iridescence_thickness * weight;
            accum_anisotropic          += layer_s.anisotropic * weight;
            accum_anisotropic_rot      += layer_s.anisotropic_rot * weight;
            accum_attenuation          += layer_s.attenuation * weight;
            accum_attenuation_distance += layer_s.attenuation_distance * weight;
            accum_conductor_k          += layer_s.conductor_k * weight;
            accum_specular_tint        += layer_s.specular_tint * weight;
            accum_specular_trans       += layer_s.specular_trans * weight;
            accum_dispersion           += layer_s.dispersion * weight;
            accum_flatness             += layer_s.flatness * weight;
            accum_diffuse_trans        += layer_s.diffuse_trans * weight;
            accum_fabric               += layer_s.fabric * weight;
            accum_glass_blend          += layer_s.glass_blend * weight;
            total_weight               += weight;

            // Capture layer 0's type
            $if(layer == 0u) {
                layer0_type      = layer_s.material_type;
                layer0_bsdf_type = layer_s.bsdf_type;
            };
        };
    }

    // --- Normalize (except emission which is additive) ---
    Float inv_w = 1.0f / max(total_weight, 1e-6f);

    SurfaceData s;
    s.albedo                = accum_albedo * inv_w;
    s.emission              = accum_emission;  // additive, not normalized
    s.roughness             = accum_roughness * inv_w;
    s.metallic              = accum_metallic * inv_w;
    s.ior                   = accum_ior * inv_w;
    s.albedo_alpha          = accum_albedo_alpha * inv_w;
    s.ao                    = accum_ao * inv_w;
    s.sheen                 = accum_sheen * inv_w;
    s.sheen_tint            = accum_sheen_tint * inv_w;
    s.clearcoat             = accum_clearcoat * inv_w;
    s.clearcoat_gloss       = accum_clearcoat_gloss * inv_w;
    s.iridescence           = accum_iridescence * inv_w;
    s.iridescence_ior       = accum_iridescence_ior * inv_w;
    s.iridescence_thickness = accum_iridescence_thickness * inv_w;
    s.anisotropic           = accum_anisotropic * inv_w;
    s.anisotropic_rot       = accum_anisotropic_rot * inv_w;
    s.attenuation           = accum_attenuation * inv_w;
    s.attenuation_distance  = accum_attenuation_distance * inv_w;
    s.conductor_k           = accum_conductor_k * inv_w;
    s.specular_tint         = accum_specular_tint * inv_w;
    s.specular_trans        = accum_specular_trans * inv_w;
    s.dispersion            = accum_dispersion * inv_w;
    s.flatness              = accum_flatness * inv_w;
    s.diffuse_trans         = accum_diffuse_trans * inv_w;
    s.fabric                = accum_fabric * inv_w;
    s.glass_blend           = accum_glass_blend * inv_w;

    // --- Geometry from layer 0 (shared across all layers) ---
    // resolve_surface() already reconstructed geometry per-layer,
    // but for multi-layer we use layer 0's geometry consistently.
    // Since all layers share the same vertex/triangle data and bary coords,
    // the geometry is identical — just use the last computed values.
    // We reconstruct once explicitly to be clear:
    Float3 ml_obj_pos = reconstruct_object_position(tv, bary);
    s.position = (instance_transform * make_float4(ml_obj_pos, 1.0f)).xyz();
    Bool ml_double_sided = (instance_flags & kDoubleSidedFlag) != 0u;
    s.geo_ns   = ite(ml_double_sided,
        reconstruct_face_normal(tv),
        reconstruct_normal(tv, bary));
    Float4 t_frame = reconstruct_tangent(tv, bary);
    s.tangent   = make_float3(normalize(t_frame.xyz()));
    s.tangent_w = t_frame.w;

    // Normal map and alphacut from layer 0's material
    Var<MaterialData> mat0 = material_buffer.read(material_layers & 0xFFu);
    s.alphacut = mat0.alphacut;
    Float2 uv = reconstruct_uv(tv, bary);
    s.ns = perturb_normal(s.geo_ns, s.tangent, s.tangent_w,
                          mat0.normalTexIdx, uv, tex_bindless);

    // Flip normal to face viewer
    s.ns = ite(dot(wo, s.ns) < 0.0f, -s.ns, s.ns);

    // Classification from layer 0
    s.material_type = layer0_type;
    s.bsdf_type     = layer0_bsdf_type;

    // --- Transform object-space normals/tangent to world space ---
    transform_surface_normals(s, instance_transform, wo);

    return s;
}

//==============================================================================
// Procedural Normal Reconstruction — lightweight normal for edge-stopping checks
//==============================================================================

#if NT_ENABLE_PROCEDURAL
/// Reconstruct geometric normal for procedural geometry without bindless vertex access.
/// Uses VAT normals with frame interpolation for type 0, sphere/cube analytics otherwise.
[[nodiscard]] inline Float3 reconstruct_procedural_normal(
    const BindlessVar& proc_bindless,
    UInt inst_id, UInt prim_id, Float3 hit_pos, Float2 bary) noexcept {

    Var<scene::ProcInstanceData> proc_inst = proc_bindless.buffer<scene::ProcInstanceData>(kSlot_ProcInstances).read(inst_id);
    UInt local_tri = prim_id & 0x1FFFFFFFu;
    Float3 ns;
    $if((proc_inst.type & 0xFu) == 1u) {
        Var<compute::AABB> aabb = proc_bindless.buffer<compute::AABB>(kSlot_ProcAABBs).read(inst_id);
        Float3 center = (aabb->min() + aabb->max()) * 0.5f;
        ns = sphere_normal(hit_pos, center, proc_inst.rotation);
    }
    $elif((proc_inst.type & 0xFu) == 2u) {
        ns = cube_normal(local_tri, proc_inst.rotation);
    }
    $else {
        // VAT (type 0) and deformable (type 3) — unified single-frame read
        ns = reconstruct_procedural_normal(proc_bindless, proc_inst, local_tri, bary);
    };

    // Double-sided: override with flat face normal for triangle-based types
    $if((proc_inst.type & kProcDoubleSided) != 0u) {
        ns = reconstruct_proc_face_normal(proc_bindless, proc_inst, local_tri);
    };

    return ns;
}
#endif // NT_ENABLE_PROCEDURAL

//==============================================================================
// Procedural Surface Resolution — build SurfaceData without bindless vertices
//==============================================================================

#if NT_ENABLE_PROCEDURAL
/// Shared SurfaceData construction for procedural geometry.
/// Reads material from buffer + reconstructs normals. Does NOT dispatch custom callables.
[[nodiscard]] inline SurfaceData build_procedural_surface_base(
    const BindlessVar& proc_bindless,
    UInt inst_id, UInt prim_id, Float3 hit_pos, Float3 wo,
    const BufferVar<MaterialData>& material_buffer,
    Float2 bary = make_float2(0.5f)) noexcept {

    Var<scene::ProcInstanceData> proc_inst = proc_bindless.buffer<scene::ProcInstanceData>(kSlot_ProcInstances).read(inst_id);
    Var<MaterialData> pmat = material_buffer.read(Expr{proc_inst.material_layers & 0xFFu});

    SurfaceData s;
    s.albedo              = pmat.albedo.xyz();
    s.albedo_alpha        = pmat.albedo.w;
    s.emission            = pmat.emission;
    s.roughness           = pmat.roughness;
    s.metallic            = pmat.metallic;
    s.ior                 = pmat.ior;
    s.alphacut            = pmat.alphacut;
    s.sheen               = pmat.sheen;
    s.sheen_tint          = pmat.sheen_tint;
    s.clearcoat           = pmat.clearcoat;
    s.clearcoat_gloss     = pmat.clearcoat_gloss;
    s.iridescence         = pmat.iridescence;
    s.iridescence_ior     = pmat.iridescence_ior;
    s.iridescence_thickness = pmat.iridescence_thickness;
    s.anisotropic         = pmat.anisotropic;
    s.anisotropic_rot     = pmat.anisotropic_rot;
    s.attenuation         = pmat.attenuation;
    s.attenuation_distance = pmat.attenuation_distance;
    s.specular_tint       = pmat.specular_tint;
    s.specular_trans      = pmat.specular_trans;
    s.dispersion          = pmat.dispersion;
    s.flatness            = pmat.flatness;
    s.fabric              = pmat.fabric;
    s.diffuse_trans       = 0.f;
    s.ao                  = 1.f;
    s.material_type       = pmat.type;
    s.bsdf_type           = ite(cast<UInt>(cast<uint>(pmat.bsdf_type_override)) > 0u,
                                cast<UInt>(cast<uint>(pmat.bsdf_type_override)), pmat.type);
    s.position            = hit_pos;

    Float3 tangent   = def(compute::make_float3(1.0f, 0.0f, 0.0f));
    Float  tangent_w = def(1.0f);

    UInt   local_tri = prim_id & 0x1FFFFFFFu;
    Float3 ns        = def(compute::make_float3(0.0f, 1.0f, 0.0f));
    Float2 uv        = def(make_float2(0.f, 0.f));

    $if((proc_inst.type & 0xFu) == 1u) {
        // Sphere: analytic normal from AABB center; no UV
        Var<compute::AABB> aabb = proc_bindless.buffer<compute::AABB>(kSlot_ProcAABBs).read(inst_id);
        Float3 center = (aabb->min() + aabb->max()) * 0.5f;
        ns = sphere_normal(hit_pos, center, proc_inst.rotation);
    }
    $elif((proc_inst.type & 0xFu) == 2u) {
        // Cube: analytic per-face normal; no UV
        ns = cube_normal(local_tri, proc_inst.rotation);
    }
    $else {
        // VAT (type 0) / deform (type 3): single fused read for pos + normal + UV.
        // Replaces what used to be 3 separate helper calls (smooth normal,
        // face normal, UV) that each re-read the same triangle and vertex data.
        ProceduralTriData t = read_procedural_triangle(proc_bindless, proc_inst, local_tri);
        Float w0 = 1.0f - bary.x - bary.y;
        ns = normalize(t.n0 * w0 + t.n1 * bary.x + t.n2 * bary.y);
        uv = t.uv0 * w0 + t.uv1 * bary.x + t.uv2 * bary.y;

        // Double-sided: override smooth normal with flat face normal from the same fused read
        $if((proc_inst.type & kProcDoubleSided) != 0u) {
            ns = normalize(cross(t.p1 - t.p0, t.p2 - t.p0));
        };
    };

    Float3 geo_ns = ns;
    ns = ite(dot(wo, ns) < 0.f, -ns, ns);

    s.ns       = ns;
    s.geo_ns   = geo_ns;
    s.tangent  = tangent;
    s.tangent_w = tangent_w;
    s.uv       = uv;

    return s;
}

/// With custom material callable dispatch (same as mesh path).
[[nodiscard]] inline SurfaceData resolve_procedural_surface(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& proc_bindless,
    UInt inst_id, UInt prim_id, Float3 hit_pos, Float3 wo,
    const BufferVar<MaterialData>& material_buffer,
    Float2 bary = make_float2(0.5f),
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u) noexcept {

    SurfaceData s = build_procedural_surface_base(
        proc_bindless, inst_id, prim_id, hit_pos, wo, material_buffer, bary);

    // Re-read material for dispatch (build_procedural_surface_base already consumed
    // its own copy; MaterialData is small and we need it for the dispatch index).
    Var<scene::ProcInstanceData> proc_inst = proc_bindless.buffer<scene::ProcInstanceData>(kSlot_ProcInstances).read(inst_id);
    Var<MaterialData> pmat = material_buffer.read(Expr{proc_inst.material_layers & 0xFFu});

    // Track B2: instance identity for callables. Procedural surfaces carry
    // the PROCEDURAL instance index (an AABB-row id, not a TLAS row) —
    // hash-style variation works uniformly; instance_params() rows are
    // mesh-only.
    s.instance_index = inst_id;

    polymorphic.dispatch(pmat.type, [&](const SurfaceResolver* resolver) {
        resolver->resolve(s, pmat, screen_uv, wo, time,
            proc_bindless, screen_w, screen_h);
    });

    return s;
}

/// Without polymorphic dispatch — for paths that only need base material (denoiser prefilter).
[[nodiscard]] inline SurfaceData resolve_procedural_surface(
    const BindlessVar& proc_bindless,
    UInt inst_id, UInt prim_id, Float3 hit_pos, Float3 wo,
    const BufferVar<MaterialData>& material_buffer,
    Float2 bary = make_float2(0.5f)) noexcept {

    return build_procedural_surface_base(
        proc_bindless, inst_id, prim_id, hit_pos, wo, material_buffer, bary);
}

/// Full surface resolution with texture sampling for procedural geometry.
/// UV comes from s.uv (populated by build_procedural_surface_base from the
/// packed .w channels of position+normal buffers — no separate UV buffer).
[[nodiscard]] inline SurfaceData resolve_procedural_surface_textured(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& proc_bindless,
    const BindlessVar& tex_bindless,
    UInt inst_id, UInt prim_id, Float3 hit_pos, Float3 wo,
    const BufferVar<MaterialData>& material_buffer,
    Float2 bary = make_float2(0.5f),
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u) noexcept {

    // Build base surface — single fused read for proc triangle data, populates s.uv.
    SurfaceData s = build_procedural_surface_base(
        proc_bindless, inst_id, prim_id, hit_pos, wo, material_buffer, bary);

    // Re-read material for texture indices + dispatch (small, kept for clarity).
    Var<scene::ProcInstanceData> proc_inst =
        proc_bindless.buffer<scene::ProcInstanceData>(kSlot_ProcInstances).read(inst_id);
    Var<MaterialData> pmat = material_buffer.read(Expr{proc_inst.material_layers & 0xFFu});

    Float2 uv = s.uv;

    // --- Texture sampling (same pattern as resolve_surface) ---
    UInt albedo_slot   = ite(pmat.albedoTexIdx  >= 0, cast<uint>(pmat.albedoTexIdx),  0u);
    UInt emissive_slot = ite(pmat.emissiveTexIdx >= 0, cast<uint>(pmat.emissiveTexIdx), 0u);
    UInt rma_slot      = ite(pmat.rmaTexIdx      >= 0, cast<uint>(pmat.rmaTexIdx),      0u);

    {
        Float3 albedo_result;
        Float albedo_alpha_result;
        $if(pmat.albedoTexIdx >= 0) {
            Float4 tex_sample = tex_bindless->tex2d(albedo_slot).sample(uv);
            albedo_result = pmat.albedo.xyz() * tex_sample.xyz();
            albedo_alpha_result = pmat.albedo.w * tex_sample.w;
        } $else {
            albedo_result = pmat.albedo.xyz();
            albedo_alpha_result = pmat.albedo.w;
        };
        s.albedo = albedo_result;
        s.albedo_alpha = albedo_alpha_result;
    }
    {
        Float3 emission_result;
        $if(pmat.emissiveTexIdx >= 0) {
            emission_result = pmat.emission * tex_bindless->tex2d(emissive_slot).sample(uv).xyz();
        } $else {
            emission_result = pmat.emission;
        };
        s.emission = emission_result;
    }
    {
        $if(pmat.rmaTexIdx >= 0) {
            Float4 rma_sample = tex_bindless->tex2d(rma_slot).sample(uv);
            s.roughness = rma_sample.x;
            s.metallic  = rma_sample.y;
            s.ao        = rma_sample.z;
        };
    }
    {
        // Iridescence thickness map (same convention as resolve_surface_verts)
        UInt irid_slot = ite(pmat.iridescenceTexIdx >= 0,
                             cast<uint>(pmat.iridescenceTexIdx), 0u);
        $if(pmat.iridescenceTexIdx >= 0) {
            Float4 irid_sample = tex_bindless->tex2d(irid_slot).sample(uv);
            s.iridescence = pmat.iridescence * irid_sample.x;
            s.iridescence_thickness = ite(pmat.iridescence_thickness_max >= 0.0f,
                lerp(pmat.iridescence_thickness,
                     pmat.iridescence_thickness_max, irid_sample.y),
                pmat.iridescence_thickness);
        };
    }

    // Normal map perturbation
    s.ns = perturb_normal(s.geo_ns, s.tangent, s.tangent_w,
                          pmat.normalTexIdx, uv, tex_bindless);
    s.ns = ite(dot(wo, s.ns) < 0.0f, -s.ns, s.ns);

    // Track B2: instance identity for callables (procedural index — see the
    // non-textured resolve_procedural_surface above).
    s.instance_index = inst_id;

    // Polymorphic dispatch — UV already in s.uv
    polymorphic.dispatch(pmat.type, [&](const SurfaceResolver* resolver) {
        resolver->resolve(s, pmat, screen_uv, wo, time,
            tex_bindless, screen_w, screen_h);
    });

    return s;
}
#endif // NT_ENABLE_PROCEDURAL

//==============================================================================
// Vertical Layering Composition 
//==============================================================================

/// Resolve surface with vertical layer composition (coat / base / fuzz roles).
///
/// Type-driven role classification (replaces parameter blend):
///   coat  : Clearcoat (7) at any layer; Dielectric (3) at layer ≥ 1.
///           Single coat slot — highest-weight candidate wins.
///   fuzz  : Sheen (8). Single fuzz slot — highest-weight wins.
///   base  : everything else (Diffuse/Conductor/Plastic/Fabric/Subsurface/
///           ThinDielectric/Unlit/Emissive, OR Dielectric at layer 0).
///           Layer 0 is always dominant (weight 1.0 > frac(meta)).
///   modifier (Anisotropy 9/Iridescence 10): transformative — composed into
///           the base's fields right after classification (§2b): strength
///           blended by layer weight, highest-weight layer of each kind wins;
///           the iridescence film's IOR/thickness replace the base's.
///
/// Composition (layered slot layout):
///   slot 0 : coat Clearcoat (additive)            [weight 0 if no coat]
///   slot 1 : base's slot 0 (SpecularMetal or DeltaDielectric)
///   slot 2 : base's slot 1 (Diffuse)
///   slot 3 : base's slot 2 (Fabric)
///   slot 4 : base's slot 3 (Subsurface)
///   slot 5 : base's slot 4 (Transmission)
///   slot 6 : base's slot 5 (Sheen additive)
///   slot 7 : base's slot 6 (Clearcoat additive — base's own)
///   slot 8 : fuzz Sheen (additive)                [weight 0 if no fuzz]
///   count  : 9 (fixed; unused slots are zero-weighted from reset)
///
/// Coat attenuates base lobes by (1 - F_coat); base's own additive lobes
/// (Sheen, Clearcoat) are also attenuated. Fuzz is on top — not attenuated.
///
/// F_coat = fresnel_dielectric(cos_theta_o, 1.0, coat.ior). At grazing angles
/// F_coat → 1, giving the coat precedence over the base — the Phase 2 win
/// "metal + clearcoat now actually does something".
[[nodiscard]] inline SurfaceData resolve_surface_layered(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& vertex_bindless,
    const BindlessVar& tex_bindless,
    UInt vertex_slot, UInt tri_slot, UInt prim_id, Float2 bary,
    UInt material_layers,
    const BufferVar<MaterialData>& material_buffer,
    Float3 wo,
    Float4x4 instance_transform,
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u,
    UInt instance_flags = 0u,
    UInt instance_index = 0u) noexcept {

    // ------------------------------------------------------------------
    // 1. Resolve layer 0 → base_s. Geometry and base material come from here.
    // ------------------------------------------------------------------
    UInt layer0_idx = material_layers & 0xFFu;
    Var<MaterialData> layer0_mat = material_buffer.read(layer0_idx);
    MeshTriVerts tv = read_mesh_triangle(vertex_bindless, vertex_slot, tri_slot, prim_id);
    SurfaceData base_s = resolve_surface_verts(
        polymorphic, tex_bindless, tv, bary,
        layer0_mat, wo, instance_transform, time, screen_uv, screen_w, screen_h,
        instance_flags, instance_index);

    // ------------------------------------------------------------------
    // 2. Loop layers 1-3: classify coat / fuzz candidates, pick highest weight.
    // ------------------------------------------------------------------
    UInt  coat_idx     = def(0xFFu);
    Float coat_weight  = def(0.f);
    UInt  coat_mtype   = def(0u);
    UInt  fuzz_idx     = def(0xFFu);
    Float fuzz_weight  = def(0.f);
    // Track fuzz params (read inside the loop, applied after)
    Float fuzz_albedo_x = def(1.f), fuzz_albedo_y = def(1.f), fuzz_albedo_z = def(1.f);
    Float fuzz_sheen_v  = def(0.f);
    Float fuzz_sheen_t  = def(0.f);
    Float fuzz_sheen_rough = def(0.5f);
    // Track coat params
    Float coat_cc_v     = def(0.f);
    Float coat_cc_g     = def(0.5f);
    Float coat_ior_v    = def(1.5f);
    Float coat_rough_v  = def(0.f);
    Float coat_att_x    = def(1.f), coat_att_y = def(1.f), coat_att_z = def(1.f);
    Int   coat_normal_tex = def(-1);   // coat layer's own normal map (Clearcoat)

    // Modifier params (Anisotropy 9 / Iridescence 10): transformative — the
    // highest-weight layer of each kind wins and is composed into the base's
    // fields after the loop (section 2b). Not additive lobes: they occupy no
    // lobe-list slots; the base's anisotropy/iridescence fields are what
    // MaterialBSDF reads.
    UInt  mod_aniso_idx     = def(0xFFu);
    Float mod_aniso_w       = def(-1.f);
    Float mod_aniso_v       = def(0.f);
    Float mod_aniso_rot     = def(0.f);
    UInt  mod_irid_idx      = def(0xFFu);
    Float mod_irid_w        = def(-1.f);
    Float mod_irid_v        = def(0.f);
    Float mod_irid_ior      = def(1.3f);
    Float mod_irid_thick    = def(0.f);
    Float mod_irid_thick_max= def(-1.f);
    Int   mod_irid_tex      = def(-1);

    for (uint layer = 1; layer < 4; ++layer) {
        UInt mat_idx = (material_layers >> (layer * 8u)) & 0xFFu;
        $if(mat_idx != 0xFFu) {
            Var<MaterialData> layer_mat = material_buffer.read(mat_idx);
            // Phase 2 fix: meta defaults to 0 for most materials — treat unset
            // meta as full weight (1.0) so users don't have to set it explicitly.
            Float raw_meta_w = fract(layer_mat.meta);
            Float weight = ite(raw_meta_w > 0.f, raw_meta_w, 1.0f);
            UInt mtype = layer_mat.type;

            // Coat: Clearcoat (7) at any layer ≥1, or Dielectric (3) at layer ≥1
            Bool is_coat_clearcoat = (mtype == 7u);
            Bool is_coat_dielectric = (mtype == 3u);
            Bool is_fuzz = (mtype == 8u);
            Bool is_mod_aniso = (mtype == 9u);
            Bool is_mod_irid  = (mtype == 10u);

            $if(is_coat_clearcoat | is_coat_dielectric) {
                coat_idx    = layer;
                coat_weight = weight;
                coat_mtype  = mtype;
                coat_cc_v   = layer_mat.clearcoat;
                coat_cc_g   = layer_mat.clearcoat_gloss;
                coat_ior_v  = layer_mat.ior;
                coat_rough_v= layer_mat.roughness;
                coat_att_x  = layer_mat.attenuation.x;
                coat_att_y  = layer_mat.attenuation.y;
                coat_att_z  = layer_mat.attenuation.z;
                coat_normal_tex = layer_mat.normalTexIdx;
            };
            $if(is_fuzz) {
                fuzz_idx      = layer;
                fuzz_weight   = weight;
                fuzz_albedo_x = layer_mat.albedo.x;
                fuzz_albedo_y = layer_mat.albedo.y;
                fuzz_albedo_z = layer_mat.albedo.z;
                fuzz_sheen_v  = layer_mat.sheen;
                fuzz_sheen_t  = layer_mat.sheen_tint;
                fuzz_sheen_rough = layer_mat.roughness;
            };
            $if(is_mod_aniso & (weight > mod_aniso_w)) {
                mod_aniso_idx = layer;
                mod_aniso_w   = weight;
                mod_aniso_v   = layer_mat.anisotropic;
                mod_aniso_rot = layer_mat.anisotropic_rot;
            };
            $if(is_mod_irid & (weight > mod_irid_w)) {
                mod_irid_idx      = layer;
                mod_irid_w        = weight;
                mod_irid_v        = layer_mat.iridescence;
                mod_irid_ior      = layer_mat.iridescence_ior;
                mod_irid_thick    = layer_mat.iridescence_thickness;
                mod_irid_thick_max= layer_mat.iridescence_thickness_max;
                mod_irid_tex      = layer_mat.iridescenceTexIdx;
            };
        };
    }

    // ------------------------------------------------------------------
    // 2b. Compose modifier layers into the base. Anisotropy/Iridescence are
    //     transformative modifiers, not additive lobes: blend the matching
    //     base fields (strength by layer weight; the film's IOR/thickness
    //     replace — the film being added is the modifier's, texture-mapped
    //     when it carries a thickness map). Applied BEFORE the coat/fuzz
    //     short-circuit so modifier-only stacks still take effect through
    //     the single-layer path. No modifier layers → zero field changes
    //     (bit-identical). Note: the MS-GGX invariants hoisted inside
    //     resolve_surface_verts keep the base's pre-modifier anisotropy —
    //     acceptable for a compensation fit, not an energy term.
    // ------------------------------------------------------------------
    $if(mod_aniso_idx != 0xFFu) {
        base_s.anisotropic     = lerp(base_s.anisotropic,     mod_aniso_v,   mod_aniso_w);
        base_s.anisotropic_rot = lerp(base_s.anisotropic_rot, mod_aniso_rot, mod_aniso_w);
    };
    $if(mod_irid_idx != 0xFFu) {
        Float mod_strength  = mod_irid_v;
        Float mod_thickness = mod_irid_thick;
        $if(mod_irid_tex >= 0) {
            // Safe index even for speculative evaluation (perturb_normal
            // convention); same R/G thickness-map layout as the base path.
            UInt irid_slot = ite(mod_irid_tex >= 0, cast<uint>(mod_irid_tex), 0u);
            Float4 irid_sample = tex_bindless->tex2d(irid_slot).sample(base_s.uv);
            mod_strength  *= irid_sample.x;
            mod_thickness  = ite(mod_irid_thick_max >= 0.f,
                lerp(mod_irid_thick, mod_irid_thick_max, irid_sample.y),
                mod_thickness);
        };
        base_s.iridescence           = lerp(base_s.iridescence, mod_strength, mod_irid_w);
        base_s.iridescence_ior       = mod_irid_ior;
        base_s.iridescence_thickness = mod_thickness;
    };

    // ------------------------------------------------------------------
    // 3. Short-circuit: if no coat AND no fuzz candidate was found in
    // layers 1-3 (sentinel 0xFF slots, or materials outside the
    // coat/fuzz/modifier roles which the loop ignores), the layering is a
    // no-op beyond any §2b modifier composition. Return base_s with
    // composed_lobe_list_count left at 0 so make_bsdf() runs the standard
    // single-layer path — bit-identical to non-layered instances (when no
    // modifiers fired), and skips all of: base_bsdf.make_bsdf(), F_coat,
    // composed-list build, and coat/fuzz field forwarding.
    //
    // This is the main perf recover from the lobe-list layering fix
    // (commit 4633f7a): instances that pay the layered dispatch but have
    // no actual coat/fuzz no longer pay the 9-slot evaluate cost.
    // ------------------------------------------------------------------
    Bool has_coat = coat_idx != 0xFFu;
    Bool has_fuzz = fuzz_idx != 0xFFu;
    Bool has_any_layer = has_coat | has_fuzz;

    // Coat shading normal: the coat layer's own normal map (Clearcoat-type
    // layers carry normalTexIdx), perturbed from geo_ns over the base tangent
    // frame — already world space here (resolve_surface_verts ran
    // transform_surface_normals). An independent second normal lets the coat
    // sparkle (orange peel) over a differently-mapped base. No coat normal
    // texture → ns, bit-identical to the legacy shared-normal behavior.
    Float3 coat_ns_v = base_s.ns;
    $if(has_coat) {
        Float3 coat_n = perturb_normal(base_s.geo_ns, base_s.tangent, base_s.tangent_w,
                                       coat_normal_tex, base_s.uv, tex_bindless);
        coat_n = ite(dot(wo, coat_n) < 0.0f, -coat_n, coat_n);
        coat_ns_v = ite(coat_normal_tex >= 0, coat_n, base_s.ns);
    };

    SurfaceData s = base_s;  // composed_lobe_list_count defaults to 0
    $if(has_any_layer) {
        // Build base's lobe_list (single-layer Phase 1 path) so we can copy it.
        // Deferred to here so no-layer pixels skip make_bsdf()/build_lobe_list().
        MaterialBSDF base_bsdf = base_s.make_bsdf();

        // ------------------------------------------------------------------
        // 4. Compute F12 (air→coat) and F23 (coat→base) Fresnel terms.
        //    base_scale = (1-F12)·(1-F23) when NT_ENABLE_TWO_INTERFACE_FRESNEL,
        //    otherwise (1-F12) only (legacy single-interface path, bit-identical).
        //    Both interfaces live on the COAT surface — evaluate at the coat
        //    normal (= ns when the coat has no normal map of its own).
        // ------------------------------------------------------------------
        Float cos_theta_o = max(dot(wo, coat_ns_v), 0.f);
        Float F12 = def(0.f);
        Float F23 = def(0.f);
        $if(has_coat) {
            F12 = fresnel_dielectric(cos_theta_o, 1.0f, coat_ior_v);
            #if NT_ENABLE_TWO_INTERFACE_FRESNEL
            // F23 = Fresnel at coat → base interface, for DIELECTRIC bases only,
            // evaluated at the refracted in-coat angle (Snell), NOT the air-side
            // cosine. The coat→base interface is only ever hit from inside the
            // coat, at sin_coat = sin(air)/η_coat; evaluating it at the air-side
            // angle made sin_t = η_coat/η_base · sin(air) and TIR'd (F23=1,
            // base zeroed) for view angles > arcsin(η_base/η_coat) ≈ 41.8°
            // whenever η_base < η_coat (diffuse ior 1.0 under coat 1.5 went
            // black). See docs/coat_fresnel_f23.md §2.
            //
            // Conductor bases skip (1-F23) entirely: the coat→conductor interface
            // IS the metal's own reflection, already evaluated inside the base's
            // MicrofacetBSDF::evaluate (fresnel_conductor on the same η/k).
            // Subtracting (1-F23) counts that interface twice — light "transmitted"
            // into an opaque metal is absorbed, never returned by R3 — and crushes
            // the metal ~13x (gold: (1-0.93) ≈ 0.07). Correct single-pass composite
            // for a metal base: R = F12 + (1-F12)·F_metal. See
            // docs/coat_fresnel_f23.md §2.1 amendment.
            Bool base_is_conductor = dot(base_s.conductor_k, base_s.conductor_k) > 0.f;
            $if(!base_is_conductor) {
                // Refract the view ray into the coat first: sin_t inside
                // fresnel_dielectric then equals sin(air)/η_base < 1 — TIR from
                // air-side incidence is impossible for any η_base ≥ 1.
                Float sin_coat = sqrt(max(0.f, 1.f - sqr(cos_theta_o))) / coat_ior_v;
                Float cos_coat = sqrt(max(0.f, 1.f - sqr(sin_coat)));
                F23 = fresnel_dielectric(cos_coat, coat_ior_v, base_s.ior);
            };
            #endif
        };
        // Fuzz-layer energy budget: the composed sheen slot 8 takes
        // 0.25*fuzz_weight*fuzz_sheen_v out of the base it covers (defaults 0
        // when no fuzz layer → scale 1). Same convention as the single-layer
        // sheen budget in build_lobe_list_for_layer. evaluate()/evaluate_split
        // re-derive the equivalent factor from lobe_list.weights[8] for the
        // decoupled specular eval weight.
        Float fuzz_budget = 1.f - 0.25f * fuzz_weight * fuzz_sheen_v;
#if NT_ENABLE_TWO_INTERFACE_FRESNEL
        Float base_scale = (1.f - F12) * (1.f - F23) * fuzz_budget;
#else
        Float base_scale = (1.f - F12) * fuzz_budget;
#endif

        // ------------------------------------------------------------------
        // 5. Build composed LobeList.
        // ------------------------------------------------------------------
        LobeListData composed;
        for (uint i = 0u; i < LobeList::kMaxLobes; ++i) {
            composed.type_flags[i] = pack_lobe(static_cast<uint>(LobeType::Diffuse), 0u);
            composed.weights[i]    = 0.f;
        }

        // Slot 0: coat Clearcoat (additive). Weight = coat_weight * clearcoat_strength.
        // Drop F_coat multiplier — ClearcoatBSDF already has its own internal Schlick
        // Fresnel (R0=0.04 at IOR 1.5), so external F_coat was double-counting and
        // made the coat effect invisible at normal incidence (~0.0016 effective weight).
        // Grazing-angle behavior still emerges naturally from ClearcoatBSDF's internal Fresnel.
        //
        // Dielectric coats (coat_bsdf_type == 3) don't populate the `clearcoat` field
        // (it's a Clearcoat-only param), so coat_cc_v is 0 and the legacy weight formula
        // produces a no-op slot 0. Under NT_ENABLE_TWO_INTERFACE_FRESNEL, use coat_weight
        // directly for Dielectric coats so the slot-0 MicrofacetBSDF dispatch (see
        // BSDF.cpp::evaluate) actually fires with the coat's ior + roughness.
        $if(has_coat) {
            composed.type_flags[0] = pack_lobe(static_cast<uint>(LobeType::Clearcoat),
                                               kLobeIsAdditive | kLobeIsReflection);
            #if NT_ENABLE_TWO_INTERFACE_FRESNEL
            Float slot0_w = ite(coat_mtype == 3u, coat_weight, coat_weight * coat_cc_v);
            composed.weights[0] = slot0_w;
            #else
            composed.weights[0] = coat_weight * coat_cc_v;
            #endif
        };

        // Slots 1-7: copy base's lobes (slot 0 → 1, 1 → 2, ..., 6 → 7), scale by base_scale.
        // base_scale = (1 - F_coat) attenuates base at grazing angles when coat is present.
        composed.type_flags[1] = base_bsdf.lobe_list.type_flags[0];
        composed.weights[1]    = base_bsdf.lobe_list.weights[0] * base_scale;
        composed.type_flags[2] = base_bsdf.lobe_list.type_flags[1];
        composed.weights[2]    = base_bsdf.lobe_list.weights[1] * base_scale;
        composed.type_flags[3] = base_bsdf.lobe_list.type_flags[2];
        composed.weights[3]    = base_bsdf.lobe_list.weights[2] * base_scale;
        composed.type_flags[4] = base_bsdf.lobe_list.type_flags[3];
        composed.weights[4]    = base_bsdf.lobe_list.weights[3] * base_scale;
        composed.type_flags[5] = base_bsdf.lobe_list.type_flags[4];
        composed.weights[5]    = base_bsdf.lobe_list.weights[4] * base_scale;
        composed.type_flags[6] = base_bsdf.lobe_list.type_flags[5];
        composed.weights[6]    = base_bsdf.lobe_list.weights[5] * base_scale;
        composed.type_flags[7] = base_bsdf.lobe_list.type_flags[6];
        composed.weights[7]    = base_bsdf.lobe_list.weights[6] * base_scale;

        // Slot 8: fuzz Sheen (Charlie lobe). Weight = 0.25 * fuzz_weight *
        // sheen (Disney 0.25 scale, matching the coat slots); the same
        // fraction was taken out of the base via fuzz_budget above.
        $if(has_fuzz) {
            composed.type_flags[8] = pack_lobe(static_cast<uint>(LobeType::Sheen),
                                               kLobeIsReflection);
            composed.weights[8]    = 0.25f * fuzz_weight * fuzz_sheen_v;
        };

        composed.count = 9u;

        // ------------------------------------------------------------------
        // 6. Compose final SurfaceData. Geometry/material params from base_s.
        //    Coat/fuzz fields populated when present. composed_lobe_list rides along.
        // ------------------------------------------------------------------
        s.composed_lobe_list       = composed;
        s.composed_lobe_list_count = 9u;

        // Coat/fuzz transmittance for the Burley BSSRDF probe add (shade
        // shader): the probe is the direct SSS model, so it carries the same
        // base_scale the composed lobes get — no SSS-specific exemption.
        s.sss_probe_scale          = base_scale;

        // Coat params (read by slot 0 dispatch in evaluate/sample/pdf)
        $if(has_coat) {
            s.coat_bsdf_type           = coat_mtype;
            s.coat_clearcoat_val       = coat_cc_v;
            s.coat_clearcoat_gloss_val = coat_cc_g;
            s.coat_ior                 = coat_ior_v;
            s.coat_roughness           = coat_rough_v;
            s.coat_attenuation         = make_float3(coat_att_x, coat_att_y, coat_att_z);
            s.coat_ns                  = coat_ns_v;
            // Forward F12 / F23 so evaluate / evaluate_split don't recompute.
            s.coat_F12                 = F12;
            s.coat_F23                 = F23;
        };
        // Fuzz params (read by slot 8 dispatch)
        $if(has_fuzz) {
            s.fuzz_albedo        = make_float3(fuzz_albedo_x, fuzz_albedo_y, fuzz_albedo_z);
            s.fuzz_sheen_val     = fuzz_sheen_v;
            s.fuzz_sheen_tint_val= fuzz_sheen_t;
            s.fuzz_sheen_roughness_val = fuzz_sheen_rough;
        };
    };

    return s;
}

//==============================================================================
// Convenience Wrapper — Resolve surface from instance data
//==============================================================================

/// Resolve surface from instance data, reading material from buffer.
/// Multi-layer branch: when any of layers 1-3 are active (material_layers >> 8 != 0),
/// uses weighted blending via resolve_surface_multilayer() (toggle-off) or
/// vertical layer composition via resolve_surface_layered() (toggle-on).
/// Single-layer (default): zero overhead — same path as before.
[[nodiscard]] inline SurfaceData resolve_surface_from_instance(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& vertex_bindless,
    const BindlessVar& tex_bindless,
    UInt4 inst_data,
    UInt prim_id, Float2 bary,
    const BufferVar<MaterialData>& material_buffer,
    Float3 wo,
    Float4x4 instance_transform,
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u,
    UInt instance_index = 0u) noexcept {

    UInt material_layers = inst_data.y;
    UInt inst_flags = inst_data.x;
    Bool multi = (material_layers >> 8u) != 0u;

    SurfaceData surface;
    $if(multi) {
        // route multi-layer scenes through vertical layer composition.
        // Fall back to resolve_surface_multilayer only when toggle is off.
        surface = resolve_surface_layered(
            polymorphic,
            vertex_bindless, tex_bindless,
            inst_data.z, inst_data.w, prim_id, bary,
            material_layers, material_buffer,
            wo, instance_transform, time, screen_uv, screen_w, screen_h,
            inst_flags, instance_index);
    } $else {
        Var<MaterialData> material = material_buffer.read(Expr{material_layers & 0xFFu});
        surface = resolve_surface(polymorphic,
            vertex_bindless, tex_bindless,
            inst_data.z, inst_data.w, prim_id, bary,
            material, wo, instance_transform, time, screen_uv, screen_w, screen_h,
            inst_flags, instance_index);
    };
    return surface;
}

/// resolve_surface_from_instance variant taking a pre-fetched triangle
/// (read_mesh_triangle). Use at call sites that also need vertex data before
/// the resolve (e.g. reconstruct_unjittered_bary) so both share one fetch.
/// The multi-layer branch re-fetches internally — multi-layer instances are
/// the rare case and keep the slot-based path.
[[nodiscard]] inline SurfaceData resolve_surface_from_instance_verts(
    const SurfaceResolverPoly& polymorphic,
    const BindlessVar& vertex_bindless,
    const BindlessVar& tex_bindless,
    const MeshTriVerts& tv,
    UInt4 inst_data,
    UInt prim_id, Float2 bary,
    const BufferVar<MaterialData>& material_buffer,
    Float3 wo,
    Float4x4 instance_transform,
    Float time = 0.0f,
    Float2 screen_uv = make_float2(0.0f),
    UInt screen_w = 0u,
    UInt screen_h = 0u,
    UInt instance_index = 0u) noexcept {

    UInt material_layers = inst_data.y;
    UInt inst_flags = inst_data.x;
    Bool multi = (material_layers >> 8u) != 0u;

    SurfaceData surface;
    $if(multi) {
        surface = resolve_surface_layered(
            polymorphic,
            vertex_bindless, tex_bindless,
            inst_data.z, inst_data.w, prim_id, bary,
            material_layers, material_buffer,
            wo, instance_transform, time, screen_uv, screen_w, screen_h,
            inst_flags, instance_index);
    } $else {
        Var<MaterialData> material = material_buffer.read(Expr{material_layers & 0xFFu});
        surface = resolve_surface_verts(
            polymorphic, tex_bindless, tv, bary,
            material, wo, instance_transform, time, screen_uv, screen_w, screen_h,
            inst_flags, instance_index);
    };
    return surface;
}

} // namespace newtype::render
