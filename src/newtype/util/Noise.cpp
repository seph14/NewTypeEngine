#include "newtype/util/Noise.h"

namespace newtype::util {
    using luisa::compute::Callable;
    using luisa::compute::def;
    using namespace luisa::compute;

    Float hash(Expr<luisa::float3> p) noexcept {
        auto pp = make_float3(
            dot(p, make_float3(127.1f, 311.7f, 74.7f)), 
            dot(p, make_float3(269.5f, 183.3f, 246.1f)), 
            dot(p, make_float3(113.5f, 271.9f, 124.6f))
        );
        return -1.f + 2.f * fract(p.x * p.y * p.z * (p.x + p.y + p.z));
    }

    Float random(Expr<luisa::float2> p) noexcept {
        auto constexpr dt = make_float2(12.9898f, 78.233f);
        return fract(sin(dot(p.xy(), dt)) * 43758.5453123f);
    }

    Float4 fbmNoise(Expr<luisa::float3> x) noexcept {
        auto p = floor(x);
        auto w = fract(x);
        // quintic interpolation
        auto u = w * w * w * (w * (w * 6.f - 15.f) + 10.f);
        auto du = 30.f * w * w * (w * (w - 2.f) + 1.f);

        auto a = hash(p);
        auto b = hash(p + make_float3(1.f, 0.f, 0.f));
        auto c = hash(p + make_float3(0.f, 1.f, 0.f));
        auto d = hash(p + make_float3(1.f, 1.f, 0.f));
        auto e = hash(p + make_float3(0.f, 0.f, 1.f));
        auto f = hash(p + make_float3(1.f, 0.f, 1.f));
        auto g = hash(p + make_float3(0.f, 1.f, 1.f));
        auto h = hash(p + make_float3(1.f, 1.f, 1.f));
        auto k0 = a;
        auto k1 = b - a;
        auto k2 = c - a;
        auto k3 = e - a;
        auto k4 = a - b - c + d;
        auto k5 = a - c - e + g;
        auto k6 = a - b - e + f;
        auto k7 = -a + b + c - d + e - f - g + h;
        return make_float4(
            k0 + k1 * u.x + k2 * u.y + k3 * u.z + k4 * u.x * u.y + k5 * u.y * u.z + k6 * u.z * u.x + k7 * u.x * u.y * u.z,
            du * make_float3(
                k1 + k4 * u.y + k6 * u.z + k7 * u.y * u.z,
                k2 + k5 * u.z + k4 * u.x + k7 * u.z * u.x,
                k3 + k6 * u.x + k5 * u.y + k7 * u.x * u.y));
    }

    Float4 permute(Expr<luisa::float4> x) noexcept {
        return mod(((x * 34.f) + 1.f) * x, 289.f);
    }

    Float4 taylorInvSqrt(Expr<luisa::float4> r) noexcept {
        return 1.79284291400159f - 0.85373472095314f * r;
    }

    Float snoise(Expr<luisa::float3> v) noexcept {
        auto constexpr C = make_float2(1.f / 6.f, 1.f / 3.f);
        auto constexpr D = make_float4(0.f, 0.5f, 1.f, 2.f);

        auto i = floor(v + dot(v, C.yyy()));
        auto x0 = v - i + dot(i, C.xxx());
        auto g = step(x0.yzx(), x0.xyz());
        auto l = 1.f - g;
        auto i1 = min(g.xyz(), l.zxy());
        auto i2 = max(g.xyz(), l.zxy());
        auto x1 = x0 - i1 + 1.f * C.xxx();
        auto x2 = x0 - i2 + 2.f * C.xxx();
        auto x3 = x0 - 1.f + 3.f * C.xxx();
        i = mod(i, 289.f);
        auto p = permute(permute(permute(
            i.z + make_float4(0.f, i1.z, i2.z, 1.f))
            + i.y + make_float4(0.f, i1.y, i2.y, 1.f))
            + i.x + make_float4(0.f, i1.x, i2.x, 1.f));

        auto  ns = D.wyz() / 7.f - D.xzx(); // N = 7
        auto j = p - 49.f * floor(p * ns.z * ns.z);  //  mod(p,N*N)
        auto x_ = floor(j * ns.z);
        auto y_ = floor(j - 7.f * x_);    // mod(j,N)
        auto x = x_ * ns.x + ns.yyyy();
        auto y = y_ * ns.x + ns.yyyy();
        auto h = 1.f - abs(x) - abs(y);
        auto b0 = make_float4(x.xy(), y.xy());
        auto b1 = make_float4(x.zw(), y.zw());
        auto s0 = floor(b0) * 2.f + 1.f;
        auto s1 = floor(b1) * 2.f + 1.f;
        auto sh = - step(h, make_float4(0.f));
        auto a0 = b0.xzyw() + s0.xzyw() * sh.xxyy();
        auto a1 = b1.xzyw() + s1.xzyw() * sh.zzww();
        auto p0 = make_float3(a0.xy(), h.x);
        auto p1 = make_float3(a0.zw(), h.y);
        auto p2 = make_float3(a1.xy(), h.z);
        auto p3 = make_float3(a1.zw(), h.w);
        auto norm = taylorInvSqrt(make_float4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
        p0 *= norm.x;
        p1 *= norm.y;
        p2 *= norm.z;
        p3 *= norm.w;
        auto m = max(0.6f - make_float4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.f);
        m = m * m;
        return 42.f * dot(m * m, 
            make_float4(dot(p0, x0), dot(p1, x1),
                        dot(p2, x2), dot(p3, x3)));
    }

    Float noise(Expr<luisa::float3> p) noexcept {
        auto ip = floor(p);
        auto pp = p - ip;
        auto constexpr s = make_float3(7.f, 157.f, 113.f);
        auto h = make_float4(0.f, s.yz(), s.y + s.z) + dot(ip, s);
        pp = pp * pp * (3.f - 2.f * pp);
        h = lerp(fract(sin(h) * 43758.5f), fract(sin(h + s.x) * 43758.5f), p.x);
        return lerp(lerp(h.x, h.y, p.y), lerp(h.z, h.w, p.y), p.z);
    }

    Float noise(Expr<luisa::float2> p) noexcept {
        auto ip = floor(p);
        auto u  = fract(p);
        u = u * u * (3.f - 2.f * u);

        auto res = lerp(
            lerp(random(ip),
                random(ip + make_float2(1.f, 0.f)),
                u.x),
            lerp(random(ip + make_float2(0.f, 1.f)),
                random(ip + make_float2(1.f, 1.f)),
                u.x),
            u.y);
        return res * res;
    }

    Float3 snoiseFlt3(Expr<luisa::float3> x) noexcept {
        auto s  = snoise(x);
        auto s1 = snoise(make_float3(x.y - 19.1f, x.z + 33.4f, x.x + 47.2f));
        auto s2 = snoise(make_float3(x.z + 74.2f, x.x - 124.5f, x.y + 99.4f));
        return make_float3(s,s1,s2);
    }

    Float3 curlNoise(Expr<luisa::float3> p) noexcept {
        auto constexpr e = .1f;
        auto constexpr divisor = 1.f / (2.f * e);

        auto constexpr dx = make_float3(e, 0.f, 0.f);
        auto constexpr dy = make_float3(0.f, e, 0.f);
        auto constexpr dz = make_float3(0.f, 0.f, e);

        auto p_x0 = snoiseFlt3(p - dx);
        auto p_x1 = snoiseFlt3(p + dx);
        auto p_y0 = snoiseFlt3(p - dy);
        auto p_y1 = snoiseFlt3(p + dy);
        auto p_z0 = snoiseFlt3(p - dz);
        auto p_z1 = snoiseFlt3(p + dz);
        auto x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
        auto y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
        auto z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;
        return normalize(make_float3(x, y, z) * divisor);
    }
}