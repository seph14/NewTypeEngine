#pragma once
//
// Created by Mike Smith on 2022/11/8.
// Edited by Seph Li on 2026/03/23.
//

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>
#include <luisa/dsl/syntax.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/stream.h>
#include "newtype/core/Config.h"

namespace newtype::util {
    using namespace luisa;

    template<typename T>
        requires std::same_as<luisa::compute::expr_value_t<T>, luisa::float3>
    [[nodiscard]] inline auto oct_encode(T n_in) noexcept {
        auto n = select(normalize(n_in), make_float3(0.f, 0.f, 1.f), all(n_in == 0.f));
        constexpr auto oct_wrap = [](auto v) noexcept {
            return (1.f - abs(v.yx())) * select(make_float2(-1.f), make_float2(1.f), v >= 0.0f);
        };
        auto abs_n = abs(n);
        auto p = n.xy() * (1.f / (abs_n.x + abs_n.y + abs_n.z));
        p = select(oct_wrap(p), p, n.z >= 0.f);// in [-1, 1]
        auto u = make_uint2(clamp(round((p * .5f + .5f) * 65535.f), 0.f, 65535.f));
        return u.x | (u.y << 16u);
    };

    template<typename T>
        requires std::same_as<luisa::compute::expr_value_t<T>, uint>
    [[nodiscard]] inline auto oct_decode(T u) noexcept {
        auto p = make_float2(make_uint2(u & 0xffffu, u >> 16u)) * ((1.f / 65535.f) * 2.f) - 1.f;
        auto abs_p = abs(p);
        auto n = make_float3(p, 1.f - abs_p.x - abs_p.y);
        auto t = clamp(n.z, -1.f, 0.f);
        auto xy = sign(n.xy()) * t + n.xy();
        return make_float3(xy, n.z);
    }

    struct alignas(16) Vertex {
        float px;
        float py;
        float pz;
        float nx;
        float ny;
        float nz;
        float tx;
        float ty;
        float tz;
        float tw;    // bitangent handedness (-1 or +1)
        float u;
        float v;

        [[nodiscard]] static auto encode(luisa::float3 p, luisa::float3 n,
                                          luisa::float4 t, luisa::float2 uv) noexcept {
            return Vertex{ p.x, p.y, p.z, n.x, n.y, n.z,
                           t.x, t.y, t.z, t.w, uv.x, uv.y };
        };
        // Layout-generic pack surface (A2): the fp32 layout IS the authoring
        // type, so pack is the identity (pack_vertices' non-template
        // if-constexpr needs both branches to parse).
        [[nodiscard]] static Vertex pack(const Vertex &v) noexcept { return v; }
        [[nodiscard]] static Vertex pack(luisa::float3 p, luisa::float3 n,
                                         luisa::float4 t, luisa::float2 uv) noexcept {
            return encode(p, n, t, uv);
        }
        [[nodiscard]] auto position() const noexcept { return luisa::make_float3(px, py, pz); }
        [[nodiscard]] auto normal() const noexcept { return luisa::make_float3(nx, ny, nz); }
        [[nodiscard]] auto tangent() const noexcept { return luisa::make_float4(tx, ty, tz, tw); }
        [[nodiscard]] auto uv() const noexcept { return luisa::make_float2(u, v); }
    };

    static_assert(sizeof(Vertex) == 48u);

    //==========================================================================
    // GPU packed vertex layout — 32 B (perf review R2 item 15)
    //
    // DORMANT (measured): the full engine switch to this layout measured
    // -2.2..-2.5% fps on cornell (validation build, warm, 1440x1440, the
    // only uncapped bench scene) — the per-read decode ALU (oct normal +
    // 4x fp16 tangent per vertex) outweighs the 144->96 B fetch cut on
    // L2-resident bench meshes. DI/G-Buffer alone DID improve (-1.2..-2.8%),
    // so the fetch-side win is real; revisit for 4K / render-scale or
    // out-of-L2 scenes, where bandwidth dominates ALU. Type + pack/unpack
    // helpers stay for that revisit (the item-2 knob precedent); nothing
    // consumes them today.
    //
    // The 48 B fp32 AoS Vertex remains the CPU authoring type (Physics, VAT,
    // tests author plain fields) AND the GPU bindless/BLAS layout. Had this
    // shipped, GpuVertex would have ridden the vertex buffers with:
    //   - positions fp32 at offset 0 (the BLAS reads them there; a 32 B
    //     stride is legal — D3D12 requires stride % component size == 0);
    //   - UVs fp32 (texture filtering identical);
    //   - the normal octahedral-encoded into one uint (~1e-5 angular);
    //   - the tangent fp16-packed (xyz + handedness; +/-1 is fp16-exact).
    //==========================================================================
    struct GpuVertex {
        float    px, py, pz;   // fp32 positions (AS build reads offset 0)
        uint32_t oct_n;        // octahedral-encoded vertex normal
        uint32_t tan_xy;       // fp16(tx) | fp16(ty) << 16
        uint32_t tan_zw;       // fp16(tz) | fp16(tw) << 16
        float    u, v;         // fp32 UV

        [[nodiscard]] static GpuVertex pack(float3 p, float3 n,
                                            float4 t, float2 uv) noexcept;
        // Convenience overload: pack an authoring Vertex verbatim.
        [[nodiscard]] static GpuVertex pack(const Vertex &v) noexcept {
            return pack(make_float3(v.px, v.py, v.pz),
                        make_float3(v.nx, v.ny, v.nz),
                        make_float4(v.tx, v.ty, v.tz, v.tw),
                        make_float2(v.u, v.v));
        }
    };

    static_assert(sizeof(GpuVertex) == 32u);

    //==========================================================================
    // Cheap-decode packed layouts (track A2, docs/vertex-packing-instancing-
    // plan.md §5). Same 32 B footprint as GpuVertex, but the normal/tangent
    // decode is shift/mask/mad (~3-5 ops/component) instead of GpuVertex's
    // software-fp16 chain (~15 ops/component) — targeting exactly the decode
    // ALU that made item 15a lose. Selected compile-time via NT_VERTEX_LAYOUT
    // (Config.h); `ActiveVertex` below is the single alias the whole engine
    // binds GPU buffers, reads, and writer kernels through — no per-mesh
    // divergence, no runtime branching.
    //
    // Layouts (fp32 positions at offset 0 in all — the BLAS reads them there;
    // D3D12 needs stride % component size == 0, satisfied at 32/40 B):
    //   PackedVertex32: pos(12) + snorm10x3 normal(4) + snorm10x3 tangent
    //                   with 2-bit handedness(4) + fp32 UV(8) + pad(4)
    //                   96 B/triangle (vs 144 fp32).
    //   PackedVertex40: pos(12) + snorm10x3 normal(4) + fp32 tangent(16) +
    //                   fp32 UV(8) — the fallback rung if snorm tangents
    //                   show in capture A/B.
    // Decode does NOT renormalize: every consumer already normalizes after
    // barycentric interpolation (reconstruct_normal/tangent, PassGI/PassSSS
    // inline reads). UVs stay fp32 (tiling UVs exceed [0,1]).
    //==========================================================================
    struct PackedVertex32 {
        float    px, py, pz;   // fp32 positions (AS build reads offset 0)
        uint32_t nrm;          // snorm10x3 vertex normal (2-bit w unused)
        uint32_t tan;          // snorm10x3 tangent.xyz, w2 = handedness (1 = negative)
        float    u, v;         // fp32 UV
        uint32_t pad;          // structure padding, zero

        [[nodiscard]] static PackedVertex32 pack(float3 p, float3 n,
                                                 float4 t, float2 uv) noexcept;
        [[nodiscard]] static PackedVertex32 pack(const Vertex &v) noexcept {
            return pack(float3{v.px, v.py, v.pz}, float3{v.nx, v.ny, v.nz},
                        float4{v.tx, v.ty, v.tz, v.tw}, float2{v.u, v.v});
        }
    };
    static_assert(sizeof(PackedVertex32) == 32u);

    struct PackedVertex40 {
        float    px, py, pz;   // fp32 positions (AS build reads offset 0)
        uint32_t nrm;          // snorm10x3 vertex normal (2-bit w unused)
        float    tx, ty, tz;   // fp32 tangent
        float    tw;           // bitangent handedness (-1 or +1)
        float    u, v;         // fp32 UV

        [[nodiscard]] static PackedVertex40 pack(float3 p, float3 n,
                                                 float4 t, float2 uv) noexcept;
        [[nodiscard]] static PackedVertex40 pack(const Vertex &v) noexcept {
            return pack(float3{v.px, v.py, v.pz}, float3{v.nx, v.ny, v.nz},
                        float4{v.tx, v.ty, v.tz, v.tw}, float2{v.u, v.v});
        }
    };
    static_assert(sizeof(PackedVertex40) == 40u);

    // --- snorm 10-10-10-2 codecs -------------------------------------------
    // Symmetric encoding: code = round(clamp(c, -1, 1) * 511) in signed
    // 10-bit two's complement; decode = sign_extend(code) / 511. The 2-bit
    // w field carries the tangent handedness (t.w < 0 → 1; exact, ±1 needs
    // no quantization).
    [[nodiscard]] inline uint32_t snorm10x3_pack_host(float3 v, uint32_t w2) noexcept {
        auto q = [](float c) noexcept {
            float r = std::nearbyint(std::clamp(c, -1.f, 1.f) * 511.f);
            return static_cast<uint32_t>(static_cast<int32_t>(r)) & 0x3FFu;
        };
        return q(v.x) | (q(v.y) << 10u) | (q(v.z) << 20u) | ((w2 & 0x3u) << 30u);
    }
    [[nodiscard]] inline float3 snorm10x3_unpack_host(uint32_t code) noexcept {
        auto d = [](uint32_t c10) noexcept {
            auto s = static_cast<int32_t>(c10 << 22u) >> 22;   // sign-extend
            return static_cast<float>(s) * (1.f / 511.f);
        };
        return float3{d(code), d(code >> 10u), d(code >> 20u)};
    }

    inline PackedVertex32 PackedVertex32::pack(float3 p, float3 n,
                                               float4 t, float2 uv) noexcept {
        PackedVertex32 r;
        r.px = p.x;  r.py = p.y;  r.pz = p.z;
        r.nrm = snorm10x3_pack_host(n, 0u);
        r.tan = snorm10x3_pack_host(float3{t.x, t.y, t.z}, t.w < 0.f ? 1u : 0u);
        r.u = uv.x;  r.v = uv.y;
        r.pad = 0u;
        return r;
    }

    inline PackedVertex40 PackedVertex40::pack(float3 p, float3 n,
                                               float4 t, float2 uv) noexcept {
        PackedVertex40 r;
        r.px = p.x;  r.py = p.y;  r.pz = p.z;
        r.nrm = snorm10x3_pack_host(n, 0u);
        r.tx = t.x;  r.ty = t.y;  r.tz = t.z;  r.tw = t.w;
        r.u = uv.x;  r.v = uv.y;
        return r;
    }

    // --- Active layout alias (NT_VERTEX_LAYOUT, Config.h) -------------------
#if NT_VERTEX_LAYOUT == 1
    using ActiveVertex = PackedVertex32;
#elif NT_VERTEX_LAYOUT == 2
    using ActiveVertex = PackedVertex40;
#else
    using ActiveVertex = Vertex;
#endif
    inline constexpr bool kVertexLayoutPacked =
        NT_VERTEX_LAYOUT == 1 || NT_VERTEX_LAYOUT == 2;

    /// Stage CPU-authored vertices into the active GPU layout (identity at
    /// fp32 — the vector copies through; packed layouts encode per vertex).
    [[nodiscard]] inline std::vector<ActiveVertex> pack_vertices(
        std::span<const Vertex> verts) noexcept {
        if constexpr (std::same_as<ActiveVertex, Vertex>) {
            return std::vector<ActiveVertex>(verts.begin(), verts.end());
        } else {
            std::vector<ActiveVertex> out(verts.size());
            for (size_t i = 0; i < verts.size(); ++i)
                out[i] = ActiveVertex::pack(verts[i]);
            return out;
        }
    }

    /// Upload CPU-authored vertices into an ActiveVertex GPU buffer.
    /// Zero-overhead identity at fp32 (direct copy_from); pack-staged for
    /// the packed layouts. Single upload path for static meshes, deformable
    /// CPU updates, and legacy util::Mesh.
    inline void upload_vertex_buffer(luisa::compute::Buffer<ActiveVertex> &buffer,
                                     std::span<const Vertex> verts,
                                     luisa::compute::Stream &stream) noexcept {
        if constexpr (std::same_as<ActiveVertex, Vertex>) {
            stream << buffer.copy_from(verts.data());
        } else {
            const std::vector<ActiveVertex> staged = pack_vertices(verts);
            stream << buffer.copy_from(staged.data());
        }
    }


    // Host fp16 pack (round-to-nearest-even) — transcription of the GPU
    // pack_half below; used by GpuVertex::pack on the CPU upload path.
    [[nodiscard]] inline uint16_t pack_f16_host(float value) noexcept {
        uint32_t bits = std::bit_cast<uint32_t>(value);
        uint32_t sign = (bits >> 16u) & 0x8000u;
        uint32_t exponent = (bits >> 23u) & 0xFFu;
        uint32_t mantissa = bits & 0x7FFFFFu;

        int32_t e = static_cast<int32_t>(exponent) - 112;

        uint32_t nan_or_inf = 0x7C00u | ((mantissa != 0u) ? (0x0200u | (mantissa >> 13u)) : 0u);

        uint32_t normal = (static_cast<uint32_t>(e) << 10u) | (mantissa >> 13u);
        uint32_t rem = mantissa & 0x1FFFu;
        bool round_up = (rem > 0x1000u) || ((rem == 0x1000u) && ((normal & 1u) != 0u));
        normal = normal + (round_up ? 1u : 0u);

        uint32_t mantissa24 = mantissa | 0x800000u;
        uint32_t shift = std::min(static_cast<uint32_t>(14 - e), 24u);
        uint32_t sub = mantissa24 >> shift;
        uint32_t rem_sub = mantissa24 & ((1u << shift) - 1u);
        uint32_t half_ulp = 1u << (shift - 1u);
        bool round_up_sub = (rem_sub > half_ulp) || ((rem_sub == half_ulp) && ((sub & 1u) != 0u));
        sub = sub + (round_up_sub ? 1u : 0u);

        uint32_t result = (exponent == 255u) ? nan_or_inf
            : (e >= 31) ? 0x7C00u
            : (e <= 0) ? ((e < -10) ? 0u : sub)
            : normal;
        return static_cast<uint16_t>(sign | result);
    }

    // Host twin of the DSL oct_encode (same sentinel + rounding): normalize
    // (zero-length -> +Z sentinel), octahedral map, two clamped 16-bit
    // fixed-point components. std::nearbyint is round-to-nearest-even under
    // the default FP mode, matching HLSL round().
    [[nodiscard]] inline uint oct_encode_host(float3 n_in) noexcept {
        bool zero = n_in.x == 0.f && n_in.y == 0.f && n_in.z == 0.f;
        float3 n = zero ? float3{0.f, 0.f, 1.f}
                        : n_in * (1.f / std::sqrt(n_in.x * n_in.x +
                                                  n_in.y * n_in.y +
                                                  n_in.z * n_in.z));
        auto oct_wrap = [](float2 v) noexcept {
            return float2{1.f - std::abs(v.y), 1.f - std::abs(v.x)} *
                   float2{(v.x >= 0.f) ? 1.f : -1.f, (v.y >= 0.f) ? 1.f : -1.f};
        };
        float3 abs_n{std::abs(n.x), std::abs(n.y), std::abs(n.z)};
        float2 p = float2{n.x, n.y} * (1.f / (abs_n.x + abs_n.y + abs_n.z));
        p = (n.z >= 0.f) ? p : oct_wrap(p);
        auto quant = [](float c) noexcept {
            float q = std::nearbyint((c * .5f + .5f) * 65535.f);
            return static_cast<uint>(std::clamp(q, 0.f, 65535.f));
        };
        return quant(p.x) | (quant(p.y) << 16u);
    }

    inline GpuVertex GpuVertex::pack(float3 p, float3 n, float4 t, float2 uv) noexcept {
        GpuVertex g;
        g.px = p.x;  g.py = p.y;  g.pz = p.z;
        g.oct_n = oct_encode_host(n);
        g.tan_xy = static_cast<uint32_t>(pack_f16_host(t.x)) |
                   (static_cast<uint32_t>(pack_f16_host(t.y)) << 16u);
        g.tan_zw = static_cast<uint32_t>(pack_f16_host(t.z)) |
                   (static_cast<uint32_t>(pack_f16_host(t.w)) << 16u);
        g.u = uv.x;  g.v = uv.y;
        return g;
    }

    //==========================================================================
    // Import-time weld (perf review 2026-09 item 15 follow-up, track A1)
    //
    // Merges vertices whose 12 floats are BIT-IDENTICAL — the duplicates
    // indexed importers emit at attribute-index seams (same position, normal,
    // tangent AND uv, split only because the source format indexed them
    // separately). First-occurrence order is preserved and merged values are
    // untouched, so rendering is bit-identical by construction while the
    // vertex buffer (and BLAS) shrinks. Returns the number of removed
    // vertices; `triangles` elements need mutable i0/i1/i2 members.
    //==========================================================================
    namespace weld_detail {
        struct VertexBitsHash {
            size_t operator()(const Vertex &v) const noexcept {
                const auto *b = reinterpret_cast<const unsigned char *>(&v);
                size_t h = 1469598103934665603ull;   // FNV-1a 64
                for (size_t i = 0; i < sizeof(Vertex); ++i) {
                    h ^= b[i];
                    h *= 1099511628211ull;
                }
                return h;
            }
        };
        struct VertexBitsEq {
            bool operator()(const Vertex &a, const Vertex &b) const noexcept {
                // Vertex is 12 packed floats (static_assert below) — no
                // padding participates in the comparison.
                return std::memcmp(&a, &b, sizeof(Vertex)) == 0;
            }
        };
    }// namespace weld_detail

    template<typename VertexContainer, typename TriangleContainer>
    inline size_t weld_vertices(VertexContainer &vertices,
                                TriangleContainer &triangles) noexcept {
        if (vertices.empty() || triangles.empty()) return 0u;
        std::unordered_map<Vertex, uint,
                           weld_detail::VertexBitsHash,
                           weld_detail::VertexBitsEq> remap;
        remap.reserve(vertices.size() * 2u);
        std::vector<uint> new_index(vertices.size());
        VertexContainer unique;
        unique.reserve(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            auto [it, inserted] = remap.emplace(vertices[i],
                                                static_cast<uint>(unique.size()));
            if (inserted) unique.push_back(vertices[i]);
            new_index[i] = it->second;
        }
        const size_t removed = vertices.size() - unique.size();
        if (removed == 0u) return 0u;
        for (auto &t : triangles) {
            t.i0 = new_index[t.i0];
            t.i1 = new_index[t.i1];
            t.i2 = new_index[t.i2];
        }
        vertices = std::move(unique);
        return removed;
    }

}// namespace luisa::render

// GPU-side fp16 conversions for GpuVertex (round-to-nearest-even, exact).
// Layering note: duplicated from render/Sharc.h's sharc_pack_f16 /
// sharc_unpack_f16 (util cannot include render; Sharc's copies stay put —
// they are parity-harness-covered and must not silently change).
namespace newtype::util {
    using namespace luisa;
    using namespace luisa::compute;

    [[nodiscard]] inline UInt pack_half(Expr<float> value) noexcept {
        UInt bits = value.bitcast<uint>();
        UInt sign = (bits >> 16u) & 0x8000u;
        UInt exponent = (bits >> 23u) & 0xFFu;
        UInt mantissa = bits & 0x7FFFFFu;

        Int e = cast<Int>(exponent) - 112;
        UInt nan_or_inf = 0x7C00u | ite(mantissa != 0u, 0x0200u | (mantissa >> 13u), 0u);
        UInt normal = (cast<UInt>(e) << 10u) | (mantissa >> 13u);
        UInt rem = mantissa & 0x1FFFu;
        Bool round_up = (rem > 0x1000u) | ((rem == 0x1000u) & ((normal & 1u) != 0u));
        normal = normal + ite(round_up, 1u, 0u);

        UInt mantissa24 = mantissa | 0x800000u;
        UInt shift = min(cast<UInt>(14 - e), 24u);
        UInt sub = mantissa24 >> shift;
        UInt rem_sub = mantissa24 & ((1u << shift) - 1u);
        UInt half_ulp = 1u << (shift - 1u);
        Bool round_up_sub = (rem_sub > half_ulp) | ((rem_sub == half_ulp) & ((sub & 1u) != 0u));
        sub = sub + ite(round_up_sub, 1u, 0u);

        UInt result = ite(exponent == 255u, nan_or_inf,
                          ite(e >= 31, 0x7C00u,
                              ite(e <= 0, ite(e < -10, 0u, sub),
                                  normal)));
        return sign | result;
    }

    [[nodiscard]] inline Float unpack_half(Expr<uint> value) noexcept {
        UInt h = value & 0xFFFFu;
        UInt sign_bits = (h & 0x8000u) << 16u;
        UInt exponent = (h >> 10u) & 0x1Fu;
        UInt mantissa = h & 0x03FFu;

        UInt normal_bits = ite(exponent >= 31u, 0x7F800000u | (mantissa << 13u),
                               ((exponent + 112u) << 23u) | (mantissa << 13u));
        Float normal_f = (sign_bits | normal_bits).bitcast<float>();

        Float sub = cast<Float>(mantissa) * 5.9604644775390625e-08f; // 2^-24
        sub = ite(sign_bits != 0u, -sub, sub);

        return ite(exponent == 0u, sub, normal_f);
    }

    [[nodiscard]] inline UInt pack_half2(Expr<float2> v) noexcept {
        return pack_half(v.x) | (pack_half(v.y) << 16u);
    }

    //==========================================================================
    // snorm 10-10-10-2 GPU codecs (host twins above; track A2). Decode is
    // shift-sign-extend-multiply (~3-4 ops/component); encode is one round +
    // clamp + pack. Used by the LUISA_STRUCT accessors below (readers) and
    // the vertex_set_* writer helpers (VAT/Physics per-frame re-encode — a
    // few ops, unlike the fp16 RNE chain that helped kill item 15a).
    //==========================================================================
    [[nodiscard]] inline UInt snorm10x3_pack(Expr<float3> v, Expr<uint> w2) noexcept {
        auto q = [](Expr<float> c) noexcept {
            return cast<UInt>(cast<Int>(round(clamp(c, -1.f, 1.f) * 511.f))) & 0x3FFu;
        };
        return q(v.x) | (q(v.y) << 10u) | (q(v.z) << 20u) | ((w2 & 0x3u) << 30u);
    }
    [[nodiscard]] inline Float3 snorm10x3_unpack(Expr<uint> code) noexcept {
        auto d = [](Expr<uint> c) noexcept {
            // Sign-extend the 10-bit field (<< 22 puts bit 9 into the sign;
            // >> 22 on a signed Int is an arithmetic shift in HLSL/DXIL).
            return cast<Float>((cast<Int>(c) << 22) >> 22) * (1.f / 511.f);
        };
        return make_float3(d(code), d(code >> 10u), d(code >> 20u));
    }

    /// Layout-generic normal writer for GPU vertex writers. Positions are
    /// fp32 in every layout (assign .px/.py/.pz directly); normals/tangents
    /// need re-encoding under the packed layouts.
    template<typename V>
        requires std::same_as<V, Vertex> || std::same_as<V, PackedVertex32> ||
                 std::same_as<V, PackedVertex40>
    inline void vertex_set_normal(Var<V> &v, Expr<float3> n) noexcept {
        if constexpr (std::same_as<V, Vertex>) {
            v.nx = n.x;  v.ny = n.y;  v.nz = n.z;
        } else {
            v.nrm = snorm10x3_pack(n, 0u);
        }
    }

    /// Layout-generic tangent writer (direction + handedness separately —
    /// handedness is exact (±1) in every layout).
    template<typename V>
        requires std::same_as<V, Vertex> || std::same_as<V, PackedVertex32> ||
                 std::same_as<V, PackedVertex40>
    inline void vertex_set_tangent(Var<V> &v, Expr<float3> t, Expr<float> w) noexcept {
        if constexpr (std::same_as<V, PackedVertex32>) {
            v.tan = snorm10x3_pack(t, ite(w < 0.f, 1u, 0u));
        } else {
            v.tx = t.x;  v.ty = t.y;  v.tz = t.z;  v.tw = w;
        }
    }
}// namespace newtype::util

// clang-format off
LUISA_STRUCT(newtype::util::GpuVertex, px, py, pz, oct_n, tan_xy, tan_zw, u, v) {
    [[nodiscard]] auto position() const noexcept { return luisa::compute::make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return newtype::util::oct_decode(oct_n); }
    [[nodiscard]] auto tangent() const noexcept {
        return luisa::compute::make_float4(newtype::util::unpack_half(tan_xy),
                                           newtype::util::unpack_half(tan_xy >> 16u),
                                           newtype::util::unpack_half(tan_zw),
                                           newtype::util::unpack_half(tan_zw >> 16u));
    }
    [[nodiscard]] auto uv() const noexcept { return luisa::compute::make_float2(u, v); }
};

// Track A2 packed layouts (snorm10 codecs — see the types above). Accessors
// mirror util::Vertex so every reader (Shading.h reconstructors, PassGI/SSS
// raw reads) is layout-agnostic; decode does NOT renormalize.
LUISA_STRUCT(newtype::util::PackedVertex32, px, py, pz, nrm, tan, u, v, pad) {
    [[nodiscard]] auto position() const noexcept { return luisa::compute::make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return newtype::util::snorm10x3_unpack(nrm); }
    [[nodiscard]] auto tangent() const noexcept {
        return luisa::compute::make_float4(
            newtype::util::snorm10x3_unpack(tan),
            luisa::compute::ite((tan >> 30u) != 0u, -1.f, 1.f));
    }
    [[nodiscard]] auto uv() const noexcept { return luisa::compute::make_float2(u, v); }
};

LUISA_STRUCT(newtype::util::PackedVertex40, px, py, pz, nrm, tx, ty, tz, tw, u, v) {
    [[nodiscard]] auto position() const noexcept { return luisa::compute::make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return newtype::util::snorm10x3_unpack(nrm); }
    [[nodiscard]] auto tangent() const noexcept { return luisa::compute::make_float4(tx, ty, tz, tw); }
    [[nodiscard]] auto uv() const noexcept { return luisa::compute::make_float2(u, v); }
};
// clang-format on

// clang-format off
LUISA_STRUCT(newtype::util::Vertex, px, py, pz, nx, ny, nz, tx, ty, tz, tw, u, v) {
    [[nodiscard]] auto position() const noexcept { return luisa::compute::make_float3(px, py, pz); }
    [[nodiscard]] auto normal() const noexcept { return luisa::compute::make_float3(nx, ny, nz); }
    [[nodiscard]] auto tangent() const noexcept { return luisa::compute::make_float4(tx, ty, tz, tw); }
    [[nodiscard]] auto uv() const noexcept { return luisa::compute::make_float2(u, v); }
};
// clang-format on
