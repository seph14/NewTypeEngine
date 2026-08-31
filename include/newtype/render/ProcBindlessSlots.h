#pragma once
#include <cstdint>

namespace newtype::render {

constexpr uint kSlot_ProcInstances = 0u;
constexpr uint kSlot_ProcPositions = 1u;
constexpr uint kSlot_ProcIndices   = 2u;
constexpr uint kSlot_ProcAABBs     = 3u;
constexpr uint kSlot_ProcNormals      = 4u;
constexpr uint kSlot_ProcDeformState  = 5u;
// Notes: UVs are packed into position.w / normal.w (free real estate — float3 is
// 16-byte aligned in LuisaCompute). No dedicated _procUVs bindless slot needed.
// _procMeshMeta is also unnecessary on the bindless path — it's passed directly
// as a kernel parameter to the VAT interpolate + deform shaders.
constexpr uint kProcSlotCount         = 6u;

} // namespace newtype::render
