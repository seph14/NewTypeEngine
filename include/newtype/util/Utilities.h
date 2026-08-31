#pragma once
#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype {
	namespace util {
		// ui helpers
		void setHDRColorTint(luisa::float3& hdr, const luisa::float3& c);
		void setHDRColorIntensity(luisa::float3& hdr, float intensity);

	}
}