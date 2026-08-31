#pragma once

#include "newtype/util/VATData.h"
#include <filesystem>

namespace newtype::scene {

/**
 * @brief VAT file loader - parses single .vat binary files
 *
 * Provides static methods to load VAT (Vertex Animation Texture) files
 * and return the parsed data as util::VATData.
 */
class VATLoader {
public:
    /**
     * @brief Load a single .vat file
     * @param path Path to the .vat file
     * @return VATData containing parsed data, or empty if loading failed
     */
    static util::VATData load(const std::filesystem::path& path) noexcept;

    /**
     * @brief Load a single .vat file with optional texcoord skipping
     * @param path Path to the .vat file
     * @param skip_texcoords If true, texcoords are not read (saves memory when not needed)
     * @return VATData containing parsed data, or empty if loading failed
     */
    static util::VATData load(const std::filesystem::path& path, bool skip_texcoords) noexcept;

    /**
     * @brief Validate VAT file version without full parse
     * @param path Path to the .vat file
     * @return true if version is supported (version 0)
     */
    static bool validate_version(const std::filesystem::path& path) noexcept;
};

} // namespace newtype::scene
