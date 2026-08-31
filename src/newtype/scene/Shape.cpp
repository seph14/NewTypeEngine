//
// Created by Claude on 2026/03/23.
//

#include "newtype/scene/Shape.h"
#include <algorithm>

namespace newtype::scene {

//==============================================================================
// Shape::Handle
//==============================================================================

uint4 Shape::Handle::encode(
    uint buffer_base, uint properties,
    uint surface_tag, uint light_tag, uint medium_tag,
    uint triangle_count,
    float shadow_terminator, float intersection_offset) noexcept
{
    uint4 encoded;

    // x: buffer_base (lower 22 bits) | properties (upper 10 bits)
    encoded.x = buffer_base | (properties << 22u);

    // y: surface_tag (upper 20 bits) | light_tag (lower 12 bits)
    encoded.y = (surface_tag << 12u) | ((light_tag & 0xFFFu));

    // z: medium_tag (upper 12 bits) | triangle_count (lower 20 bits)
    encoded.z = (medium_tag << 20u) | (triangle_count & 0xFFFFFu);

    // w: shadow_terminator (upper 16 bits) | intersection_offset (lower 16 bits)
    auto shadow_term = static_cast<uint>(std::clamp(shadow_terminator, 0.f, 1.f) * 65535.f);
    auto intersect_off = static_cast<uint>(std::clamp(intersection_offset, 0.f, 1.f) * 65535.f);
    encoded.w = (shadow_term << 16u) | intersect_off;

    return encoded;
}

Shape::Handle Shape::Handle::decode(Expr<uint4> compressed) noexcept {
    // Extract packed components
    auto x = compressed.x;
    auto y = compressed.y;
    auto z = compressed.z;
    auto w = compressed.w;

    // Decode x: buffer_base (lower 22 bits) | properties (upper 10 bits)
    auto buffer_base = x & 0x3FFFFFu;  // 22 bits
    auto properties = x >> 22u;         // 10 bits

    // Decode y: surface_tag (upper 20 bits) | light_tag (lower 12 bits)
    auto light_tag = y & 0xFFFu;        // 12 bits
    auto surface_tag = y >> 12u;        // 20 bits

    // Decode z: medium_tag (upper 12 bits) | triangle_count (lower 20 bits)
    auto triangle_count = z & 0xFFFFFu;  // 20 bits
    auto medium_tag = z >> 20u;          // 12 bits

    // Decode w: shadow_terminator (upper 16 bits) | intersection_offset (lower 16 bits)
    auto shadow_terminator = cast<float>(w >> 16u) / 65535.f;
    auto intersection_offset = cast<float>(w & 0xFFFFu) / 65535.f;

    return Handle(buffer_base, properties, surface_tag, light_tag, medium_tag,
                  triangle_count, shadow_terminator, intersection_offset);
}

} // namespace newtype::scene
