#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
//https://o-l-l-i.github.io/lut-maker/
namespace newtype::util {

/// Parsed 3D LUT from a .cube file.
struct LutData {
    uint32_t             size  = 0u;                  // LUT_3D_SIZE (e.g. 33)
    std::vector<float>   data;                        // R,G,B triples, size^3 * 3 entries
    std::string          title;                       // TITLE field (optional)
};

/// Parse a standard .cube 3D LUT file.
/// Returns a LutData with size, RGB data, and optional title.
/// Throws std::runtime_error on parse failure.
[[nodiscard]] LutData loadCubeLut(const std::filesystem::path& path);

} // namespace newtype::core
