#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>
//https://o-ll-l-i.github.io/lut-maker/
namespace newtype::util {

/// Parsed 3D LUT from a .cube file.
struct LutData {
    uint32_t             size  = 0u;                  // LUT_3D_SIZE (e.g. 33)
    std::vector<float>   data;                        // R,G,B triples, size^3 * 3 entries
    std::string          title;                       // TITLE field (optional)
};

/// Parse a standard .cube 3D LUT from any file path (absolute or cwd-relative).
/// Used for runtime-chosen LUTs (config / ImGui file dialog) that stay external.
/// Throws std::runtime_error on parse failure.
[[nodiscard]] LutData loadCubeLut(const std::filesystem::path& path);

/// Editor twin: resolve via Cinder's asset system. Call sites keep the literal
/// form `loadAsset("tonemap/x.cube")` so the Bundler can rewrite them into
/// loadResource(RES_NAME) with the LUT embedded in the exe.
/// Throws std::runtime_error on parse failure.
[[nodiscard]] LutData loadAsset(const std::filesystem::path& assetPath);

/// Production twin: leading params are the CINDER_RESOURCE macro expansion.
/// Throws std::runtime_error on parse failure.
[[nodiscard]] LutData loadResource(
    const std::filesystem::path& resourcePath,
    int mswID,
    const std::string& mswType);

} // namespace newtype::util
