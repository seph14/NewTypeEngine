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

	// Show a LuisaCompute device image as an ImGui image widget. DX12 present
	// path only: the image resource gets a per-call SRV in CinderImGui's
	// shader-visible descriptor heap and is drawn zero-copy by ImGui's own
	// pipeline. V is flipped internally (Luisa row 0 = image bottom). A size
	// component <= 0 means "auto" (fit available width / keep pixel aspect).
	// Sampling rides the same present-time contract as DxPresent's background
	// blit (Luisa stream synchronized before present) — do not point it at
	// images written by an unsynchronized async queue. One Image<float>
	// overload covers byte/half/float storage: the DXGI format comes from the
	// image's runtime PixelStorage, not the C++ element type. Returns false
	// (with placeholder text) on the GL present path, a null image, or a
	// pixel format without a DXGI counterpart.
	bool ui_image(luisa::compute::Image<float>& tex,
	              luisa::float2 size = luisa::float2(0.f), bool border = false);
	bool ui_image_button(const char* str_id, luisa::compute::Image<float>& tex,
	                     luisa::float2 size = luisa::float2(0.f));
}
}