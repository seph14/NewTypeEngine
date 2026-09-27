#include "newtype/util/UiHelper.h"
#include "cinder/CinderImGui.h"
#include "cinder/Log.h"
#include "newtype/core/Renderer.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>

using namespace std;

namespace newtype {
namespace util {

namespace {

// CinderImGui's DX12 backend owns a 64-slot shader-visible SRV heap:
// slot 0 = font atlas (legacy single descriptor), slot 1 = the per-frame
// background blit (DxPresent), slots 2..63 are free for user textures.
constexpr UINT kFirstUserSlot = 2u;
constexpr UINT kHeapSlots     = 64u;

UINT next_srv_slot() {
    // Round-robin without per-frame reset: descriptors are only dereferenced
    // by ImGui_ImplDX12_RenderDrawData during the frame they were written, so
    // overwriting a previous frame's slot is harmless.
    static UINT s_slot = kFirstUserSlot;
    if (s_slot >= kHeapSlots) {
        static bool s_warned = false;
        if (!s_warned) {
            s_warned = true;
            CI_LOG_W("UiHelper: more than " << (kHeapSlots - kFirstUserSlot)
                     << " ui_image calls in one frame wrap the ImGui SRV heap");
        }
        s_slot = kFirstUserSlot;
    }
    return s_slot++;
}

DXGI_FORMAT to_dxgi_format(luisa::compute::PixelFormat fmt) {
    using PF = luisa::compute::PixelFormat;
    switch (fmt) {
        case PF::R8UNorm:          return DXGI_FORMAT_R8_UNORM;
        case PF::RG8UNorm:         return DXGI_FORMAT_R8G8_UNORM;
        case PF::RGBA8UNorm:       return DXGI_FORMAT_R8G8B8A8_UNORM;
        case PF::RGBA8SRGB:        return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        case PF::R16UNorm:         return DXGI_FORMAT_R16_UNORM;
        case PF::RG16UNorm:        return DXGI_FORMAT_R16G16_UNORM;
        case PF::RGBA16UNorm:      return DXGI_FORMAT_R16G16B16A16_UNORM;
        case PF::R16F:             return DXGI_FORMAT_R16_FLOAT;
        case PF::RG16F:            return DXGI_FORMAT_R16G16_FLOAT;
        case PF::RGBA16F:          return DXGI_FORMAT_R16G16B16A16_FLOAT;
        case PF::R32F:             return DXGI_FORMAT_R32_FLOAT;
        case PF::RG32F:            return DXGI_FORMAT_R32G32_FLOAT;
        case PF::RGBA32F:          return DXGI_FORMAT_R32G32B32A32_FLOAT;
        case PF::R11G11B10F:       return DXGI_FORMAT_R11G11B10_FLOAT;
        case PF::R10G10B10A2UNorm: return DXGI_FORMAT_R10G10B10A2_UNORM;
        default:                   return DXGI_FORMAT_UNKNOWN;
    }
}

// Writes an SRV for the Luisa image into the next ImGui heap slot and returns
// the GPU descriptor handle as an ImTextureRef (the imgui DX12 backend's
// texture-identifier contract — imgui_impl_dx12 binds it as a root descriptor
// table). Caller gates on IsUsingD3D12() / valid image / known format.
ImTextureRef make_texture_ref(luisa::compute::Image<float>& tex) {
    ID3D12DescriptorHeap* heap = ImGui::GetD3D12SrvHeap();
    auto* device = static_cast<ID3D12Device*>(core::Renderer::device().native_handle());
    DXGI_FORMAT fmt = to_dxgi_format(tex.format());

    const UINT inc = device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT slot = next_srv_slot();

    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += static_cast<SIZE_T>(inc) * slot;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += static_cast<UINT64>(inc) * slot;

    // Same present-time contract as DxPresent's background blit: the Luisa
    // stream is synchronized before present, so the texture is effectively
    // COMMON and promotes to SRV read without a barrier. Per-call recreation
    // also makes this immune to image re-creation on resize.
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format                    = fmt;
    srv.ViewDimension             = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping   = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels       = 1u;
    srv.Texture2D.MostDetailedMip = 0u;
    device->CreateShaderResourceView(
        static_cast<ID3D12Resource*>(tex.native_handle()), &srv, cpu);

    return ImTextureRef(ImTextureID(static_cast<intptr_t>(gpu.ptr)));
}

// uv flip shared by both widgets: Luisa row 0 is the image bottom, the ImGui
// quad scans uv0 at the top-left.
constexpr ImVec2 kUv0 = ImVec2(0.f, 1.f);
constexpr ImVec2 kUv1 = ImVec2(1.f, 0.f);

ImVec2 resolve_size(const luisa::float2& size, const luisa::uint2& px) {
    float w = size.x, h = size.y;
    if (w <= 0.f) w = ImGui::GetContentRegionAvail().x;
    if (h <= 0.f) h = w * static_cast<float>(px.y) / static_cast<float>(max(px.x, 1u));
    return ImVec2(w, h);
}

} // namespace

bool ui_image(luisa::compute::Image<float>& tex, luisa::float2 size, bool border) {
    if (!ImGui::IsUsingD3D12()) {
        ImGui::TextDisabled("%s", "ui_image: DX12 present path required");
        return false;
    }
    if (!tex) {
        ImGui::TextDisabled("%s", "ui_image: null image");
        return false;
    }
    if (to_dxgi_format(tex.format()) == DXGI_FORMAT_UNKNOWN) {
        static bool s_warned = false;
        if (!s_warned) {
            s_warned = true;
            CI_LOG_W("UiHelper: pixel format has no DXGI counterpart for ui_image");
        }
        ImGui::TextDisabled("%s", "ui_image: unsupported pixel format");
        return false;
    }

    ImVec2 sz = resolve_size(size, tex.size());
    if (border) {
        ImGui::PushStyleVar(ImGuiStyleVar_ImageBorderSize, 1.f);
        ImGui::Image(make_texture_ref(tex), sz, kUv0, kUv1);
        ImGui::PopStyleVar();
    } else {
        ImGui::Image(make_texture_ref(tex), sz, kUv0, kUv1);
    }
    return true;
}

bool ui_image_button(const char* str_id, luisa::compute::Image<float>& tex, luisa::float2 size) {
    if (!ImGui::IsUsingD3D12()) {
        ImGui::TextDisabled("%s", "ui_image_button: DX12 present path required");
        return false;
    }
    if (!tex) {
        ImGui::TextDisabled("%s", "ui_image_button: null image");
        return false;
    }
    if (to_dxgi_format(tex.format()) == DXGI_FORMAT_UNKNOWN) {
        ImGui::TextDisabled("%s", "ui_image_button: unsupported pixel format");
        return false;
    }
    return ImGui::ImageButton(str_id, make_texture_ref(tex),
                              resolve_size(size, tex.size()), kUv0, kUv1);
}

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