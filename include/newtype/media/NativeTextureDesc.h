#pragma once

// Vendored binary-compatible copy of LuisaCompute's lc::dx::NativeTextureDesc.
// Upstream: LuisaCompute/src/backends/dx/DXApi/ext.h:33-39 (private backend header, not exported).
//
// We can't include the upstream struct because its header is private to the DX
// backend. LuisaCompute's public NativeResourceExt::create_native_image<T> takes
// `void* custom_data` and the DX backend reinterprets it as `NativeTextureDesc const*`.
// We mirror the layout exactly; the static_asserts below fail loudly on upstream
// layout drift (e.g. field reorder, new field, pragma pack change).
//
// On a LuisaCompute submodule bump, re-diff against ext.h and update this file.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <d3d12.h>
#include <dxgiformat.h>

namespace newtype::media {

struct NativeTextureDesc {
    D3D12_RESOURCE_STATES initState;      // canonical state for Luisa barrier planner
    DXGI_FORMAT           custom_format;  // DXGI_FORMAT_UNKNOWN lets Luisa derive from PixelStorage
    bool                  allowUav;       // false for video read-only textures
};

static_assert(sizeof(NativeTextureDesc) == 12,
    "NativeTextureDesc layout drift — re-diff against lc::dx::NativeTextureDesc in LuisaCompute/src/backends/dx/DXApi/ext.h");
static_assert(alignof(NativeTextureDesc) == 4,
    "NativeTextureDesc alignment drift — re-diff against upstream");

} // namespace newtype::media
