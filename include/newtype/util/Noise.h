#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/syntax.h>

namespace newtype::util {
    using luisa::compute::Expr;
    using luisa::compute::Float;
    using luisa::compute::Float2;
    using luisa::compute::Float3;
    using luisa::compute::Float4;

    [[nodiscard]] Float random(Expr<luisa::float2> p) noexcept;
    [[nodiscard]] Float hash(Expr<luisa::float3> p) noexcept;
    [[nodiscard]] Float4 fbmNoise(Expr<luisa::float3> x) noexcept;

    [[nodiscard]] Float4 permute(Expr<luisa::float4> x) noexcept;
    [[nodiscard]] Float4 taylorInvSqrt(Expr<luisa::float4> r) noexcept;

    [[nodiscard]] Float snoise(Expr<luisa::float3> r) noexcept;
    [[nodiscard]] Float noise(Expr<luisa::float3> r) noexcept;
    [[nodiscard]] Float noise(Expr<luisa::float2> r) noexcept;

    [[nodiscard]] Float3 snoiseFlt3(Expr<luisa::float3> r) noexcept;
    [[nodiscard]] Float3 curlNoise(Expr<luisa::float3> r) noexcept;
}