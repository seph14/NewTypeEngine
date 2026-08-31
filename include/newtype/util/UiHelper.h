#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype {
namespace util {
	bool ui_color(std::string_view name, luisa::float3& color);
	bool ui_color(std::string_view name, luisa::float4& color);
	bool ui_color_hdr(std::string_view name, luisa::float3& color);

	template<uint dim>
	bool ui_dragFlt(std::string_view name, luisa::Vector<float,dim>& vec);
	template<uint dim>
	bool ui_dragInt(std::string_view name, luisa::Vector<int, dim>& vec);
	template<uint dim>
	bool ui_dragUInt(std::string_view name, luisa::Vector<uint32_t, dim>& vec);
}
}