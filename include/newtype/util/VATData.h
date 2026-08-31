#pragma once

#include <luisa/luisa-compute.h>

namespace newtype::util {

/**
 * @brief Parsed data from a single .vat file
 *
 * Contains all vertex animation data read from a VAT binary file.
 * Positions and normals are stored as flattened arrays:
 * positions[frame_count * vertex_count + vertex_index]
 */
struct VATData {
    uint32_t vertex_count = 0;
    uint32_t frame_count = 0;
    uint32_t index_count = 0;

    luisa::vector<uint32_t>      indices;    // [index_count]
    luisa::vector<float>         texcoords;   // [vertex_count * 2] - may be empty if skipped
    luisa::vector<luisa::float3> positions;   // [frame_count * vertex_count]
    luisa::vector<luisa::float3> normals;     // [frame_count * vertex_count]

    [[nodiscard]] bool is_valid() const noexcept {
        return vertex_count > 0 && frame_count > 0;
    }
};

} // namespace newtype::util
