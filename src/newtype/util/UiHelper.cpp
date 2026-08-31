#include "newtype/util/UiHelper.h"
#include "cinder/CinderImGui.h"

using namespace std;

namespace newtype {
namespace util {
	bool ui_color(std::string_view name, luisa::float3& color) {
		return ImGui::ColorEdit3(name.data(), &color.x);
	}

	bool ui_color(std::string_view name, luisa::float4& color) {
		return ImGui::ColorEdit4(name.data(), &color.x);
	}

	template<uint dim>
	bool ui_dragFlt(std::string_view name, luisa::Vector<float, dim>& vec) {
		if (dim == 2) return ImGui::DragFloat2(name.data(), &vec[0]);
		if (dim == 3) return ImGui::DragFloat3(name.data(), &vec[0]);
		if (dim == 4) return ImGui::DragFloat4(name.data(), &vec[0]);
		return false;
	}

	template<uint dim>
	bool ui_dragInt(std::string_view name, luisa::Vector<int, dim>& vec) {
		if (dim == 2) return ImGui::DragInt2(name.data(), &vec[0]);
		if (dim == 3) return ImGui::DragInt3(name.data(), &vec[0]);
		if (dim == 4) return ImGui::DragInt4(name.data(), &vec[0]);
		return false;
	}

	template<uint dim>
	bool ui_dragUInt(std::string_view name, luisa::Vector<uint32_t, dim>& vec) {
		if (dim == 2) return ImGui::DragInt2(name.data(), &vec[0], 1.f, 0, 65535);
		if (dim == 3) return ImGui::DragInt3(name.data(), &vec[0], 1.f, 0, 65535);
		if (dim == 4) return ImGui::DragInt4(name.data(), &vec[0], 1.f, 0, 65535);
		return false;
	}

	bool ui_color_hdr(std::string_view name, luisa::float3& color) {
		ImGui::PushID(name.data());
		// Decompose HDR color into normalized tint + intensity. A black color
		// carries no hue: fall back to a white tint so the swatch previews
		// what the Intensity slider produces, and a freshly picked hue starts
		// at unit intensity instead of quantizing to ~0.
		float maxc		= glm::max(glm::max(color.x, color.y), color.z);
		float intensity = glm::max(.01f, maxc);
		auto  tint		= maxc > 1e-6f
			? luisa::clamp(luisa::float3(color / intensity), 0.f, 1.f)
			: luisa::float3(1.f);

		bool rgbChanged = ImGui::ColorEdit3(name.data(), &tint.x);
		bool intChanged = ImGui::DragFloat("Intensity", &intensity, 0.1f, 0.01f, 800.f, "%.2f");

		if (rgbChanged && maxc <= 1e-6f) {
			color = tint;
		} else if (rgbChanged) {
			auto  oldTint= color / intensity;
			float oldLum = glm::max(.01f, glm::max(glm::max(oldTint.x, oldTint.y), oldTint.z));
			float newLum = glm::max(.01f, glm::max(glm::max(tint.x, tint.y), tint.z));
			color = tint * oldLum / newLum * intensity;
		} else if (intChanged)
			color = tint * intensity;

		ImGui::PopID();
		return rgbChanged || intChanged;
	}
}
}