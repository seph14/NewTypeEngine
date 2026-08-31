#pragma once

#include <luisa/luisa-compute.h>
#include "cinder/gl/gl.h"

namespace newtype {
	// Luisa → Cinder
	[[nodiscard]] inline auto toci(const luisa::float2& v) { return ci::vec2(v.x, v.y); }
	[[nodiscard]] inline auto toci(const luisa::float3& v) { return ci::vec3(v.x, v.y, v.z); }
	[[nodiscard]] inline auto toci(const luisa::float4& v) { return ci::vec4(v.x, v.y, v.z, v.w); }
    [[nodiscard]] inline auto toci(const luisa::float4x4& m) {
        ci::mat4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i][j] = m[i][j];
            }
        }
        return result;
    }

	// Cinder → Luisa
	[[nodiscard]] inline auto tolc(const ci::mat4& m) {
        luisa::float4x4 result;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                result[i][j] = m[i][j];
            }
        }
        return result;
	}
    [[nodiscard]] inline auto tolc(const ci::vec2& v) {
        return luisa::make_float2(v.x, v.y);
    }
    [[nodiscard]] inline auto tolc(const ci::vec3& v) {
        return luisa::make_float3(v.x, v.y, v.z);
    }
    [[nodiscard]] inline auto tolc(const ci::vec4& v) {
        return luisa::make_float4(v.x, v.y, v.z, v.w);
    }
    [[nodiscard]] inline auto tolc(const ci::quat& q) {
        return luisa::make_float4(q.x, q.y, q.z, q.w);
    }
    [[nodiscard]] inline auto tolc(const ci::Color8u& c) {
        return luisa::make_ubyte3(c.r, c.g, c.b);
    }
    [[nodiscard]] inline auto tolc(const ci::ColorA8u& c) {
        return luisa::make_ubyte4(c.r, c.g, c.b, c.a);
    }
    [[nodiscard]] inline auto tolc(const ci::Color& c) {
        return luisa::make_float3(c.r, c.g, c.b);
    }
    [[nodiscard]] inline auto tolc(const ci::ColorA& c) {
        return luisa::make_float4(c.r, c.g, c.b, c.a);
    }
}
