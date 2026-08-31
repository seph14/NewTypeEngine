#pragma once
#include <luisa/luisa-compute.h>
#include <luisa/dsl/syntax.h>

namespace newtype::util {
    using luisa::compute::Expr;
    using luisa::compute::Float;
    using luisa::compute::Float3;
    using luisa::compute::Float4;
    using luisa::compute::Callable;
    using luisa::compute::def;

    inline Float4 quatMul(Expr<luisa::float4> q1, Expr<luisa::float4> q2) noexcept {
        static Callable impl = [](Float4 q1, Float4 q2) noexcept {
            return luisa::compute::make_float4(
                q2.xyz() * q1.w + q1.xyz() * q2.w + luisa::compute::cross(q1.xyz(), q2.xyz()),
                q1.w * q2.w - luisa::compute::dot(q1.xyz(), q2.xyz())
            );
        };
        return impl(q1, q2);
    }

    inline Float3 quatRotVector(Expr<luisa::float3> v, Expr<luisa::float4> r) noexcept {
        static Callable impl = [](Float3 v, Float4 r) noexcept {
            auto r_c = r * def(luisa::make_float4(-1.f, -1.f, -1.f, 1.f));
            return quatMul(r, quatMul(luisa::compute::make_float4(v, 0.f), r_c)).xyz();
        };
        return impl(v,r);
    }

    inline Float4 quatRotAngleAxis(Expr<float> angle, Expr<luisa::float3> axis) noexcept {
        static Callable impl = [](Float angle, Float3 axis) noexcept {
            auto sn = sin(angle * .5f);
            auto cs = cos(angle * .5f);
            return luisa::compute::make_float4(axis * sn, cs);
        };
        return impl(angle, axis);
    }

    inline Float4 quatFromTo(Expr<luisa::float3> v1, Expr<luisa::float3> v2) noexcept {
        static Callable impl = [](Float3 v1, Float3 v2) noexcept {
            Float4 q;
            auto d = dot(v1, v2);
            $if (d < -0.999999f) {
                auto right = def(luisa::make_float3(1.f, 0.f, 0.f));
                auto up = def(luisa::make_float3(0.f, 1.f, 0.f));
                auto tmp = cross(right, v1);
                $if(length(tmp) < 0.000001f) {
                    tmp = cross(up, v1);
                };
                q = quatRotAngleAxis(3.141592653589793f, normalize(tmp));
            }
            $elif (d > 0.999999f) {
                q = def(luisa::make_float4(0.f, 0.f, 0.f, 1.f));;
            }
            $else{
                q = normalize(make_float4(cross(v1, v2), 1.f + d));
            };

            return q;
        };

        return impl(v1, v2);
    }

    inline Float4 quatConj(Expr<luisa::float4> q) noexcept {
        static Callable impl = [](Float4 q) noexcept {
            return luisa::compute::make_float4(-q.x, -q.y, -q.z, q.w);
        };
        return impl(q);
    }

    inline Float4 quatInverse(Expr<luisa::float4> q) noexcept {
        static Callable impl = [](Float4 q) noexcept {
            auto conj = quatConj(q);
            return conj / dot(q, q);
        };
        return impl(q);
    }

    inline Float4 quatDiff(Expr<luisa::float4> q1, Expr<luisa::float4> q2) noexcept {
        static Callable impl = [](Float4 q1, Float4 q2) noexcept {
            return q2 * quatInverse(q1);
        };
        return impl(q1, q2);

    }

    inline Float4 quatSlerp(Expr<luisa::float4> a, Expr<luisa::float4> b, Expr<float> t) noexcept {
        static Callable impl = [](Float4 a, Float4 b, Float t) noexcept {
            Float4 res = luisa::compute::make_float4(0.f,0.f,0.f,1.f);
            // if either input is zero, return the other.
            $if (luisa::compute::length_squared(a) < 1e-6f) {
                $if(luisa::compute::length_squared(b) < 1e-6f) {
                    res = def(luisa::make_float4(0.f, 0.f, 0.f, 1.f));
                }
                $else {
                    res = b;
                };
            }
            $elif(luisa::compute::length_squared(b) < 1e-6f) {
                res = a;
            }
            $else{
                auto cosHalfAngle = a.w * b.w + dot(a.xyz(), b.xyz());
                $if(cosHalfAngle >= 1.f | cosHalfAngle <= -1.f) {
                    res = a;
                } $else{
                    $if(cosHalfAngle < 0.f) {
                        b *= -1.f;
                        cosHalfAngle = -cosHalfAngle;
                    };

                    Float blendA;
                    Float blendB;
                    $if(cosHalfAngle < 0.99f) {
                        // do proper slerp for big angles
                        auto halfAngle = acos(cosHalfAngle);
                        auto sinHalfAngle = sin(halfAngle);
                        auto oneOverSinHalfAngle = 1.f / sinHalfAngle;
                        blendA = sin(halfAngle * (1.f - t)) * oneOverSinHalfAngle;
                        blendB = sin(halfAngle * t) * oneOverSinHalfAngle;
                    }
                    $else{
                        // do lerp if angle is really small.
                        blendA = 1.f - t;
                        blendB = t;
                    };

                    auto result = luisa::compute::make_float4(
                        blendA * a.xyz() + blendB * b.xyz(), blendA * a.w + blendB * b.w);
                    auto len2 = luisa::compute::length_squared(result);
                    $if(len2 > 0.f) {
                        res = result / sqrt(len2);
                    };
                };
            };

            return res;
        };

        return impl(a, b, t);
    }

    inline Float4 quatLookAt(Expr<luisa::float3> forward, Expr<luisa::float3> up) noexcept {
        static Callable impl = [](Float3 forward, Float3 up) noexcept {
            Float4 q = luisa::compute::make_float4(0.f, 0.f, 0.f, 1.f);
            auto right = normalize(.00001f + cross(forward, up));
            up = normalize(.00001f + cross(forward, right));

            auto m00 = right.x;
            auto m01 = right.y;
            auto m02 = right.z;
            auto m10 = up.x;
            auto m11 = up.y;
            auto m12 = up.z;
            auto m20 = forward.x;
            auto m21 = forward.y;
            auto m22 = forward.z;

            auto num8 = (m00 + m11) + m22;
            $if (num8 > 0.f) {
                auto num = sqrt(num8 + 1.f);
                q.w = num * 0.5f;
                num = 0.5f / num;
                q.x = (m12 - m21) * num;
                q.y = (m20 - m02) * num;
                q.z = (m01 - m10) * num;
            }
            $elif ((m00 >= m11) & (m00 >= m22)) {
                auto num7 = sqrt(((1.f + m00) - m11) - m22);
                auto num4 = 0.5f / num7;
                q.x = 0.5f * num7;
                q.y = (m01 + m10) * num4;
                q.z = (m02 + m20) * num4;
                q.w = (m12 - m21) * num4;
            }
            $else{
                $if(m11 > m22) {
                    auto num6 = sqrt(((1.f + m11) - m00) - m22);
                    auto num3 = 0.5f / num6;
                    q.x = (m10 + m01) * num3;
                    q.y = 0.5f * num6;
                    q.z = (m21 + m12) * num3;
                    q.w = (m20 - m02) * num3;
                }
                $else {
                    auto num5 = sqrt(((1.f + m22) - m00) - m11);
                    auto num2 = 0.5f / num5;
                    q.x = (m20 + m02) * num2;
                    q.y = (m21 + m12) * num2;
                    q.z = 0.5f * num5;
                    q.w = (m01 - m10) * num2;
                };
            };

            return q;
        };

        return impl(forward, up);
    }
}