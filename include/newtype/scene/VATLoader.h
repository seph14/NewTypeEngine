#pragma once

#include "newtype/util/VATData.h"
#include "cinder/DataSource.h"
#include <filesystem>
#include <vector>

namespace newtype::scene {

/**
 * @brief VAT file loader - parses .vat binary files
 *
 * Handles both on-disk layouts:
 *  - V0: one topology per file (<base><N>.vat sequences)
 *  - V1: all topologies packed into a single file behind a
 *    topology header table (<base>.vat)
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
     * @brief Load from any Cinder data source (file, asset, or embedded resource)
     *
     * Lets call sites feed `ci::app::loadAsset("models/x.vat")` or
     * `ci::app::loadResource(RES_NAME)` directly, so single-file VAT loads are
     * Bundler-convertible. Folder-based sequences (VATMesh::load_folder) stay
     * path-based — dynamic enumeration cannot be embedded.
     *
     * @param source Data source of the .vat content
     * @param skip_texcoords If true, texcoords are not read (default false)
     * @return VATData containing parsed data, or empty if loading failed
     */
    static util::VATData load(const ci::DataSourceRef& source, bool skip_texcoords = false) noexcept;

    /**
     * @brief Load every topology from a .vat file
     *
     * V0 files hold a single topology and yield a one-element vector;
     * V1 packed files yield one VATData per topology in table order.
     *
     * @param path Path to the .vat file
     * @return Vector with one VATData per topology, or empty if loading failed
     */
    static std::vector<util::VATData> load_all(const std::filesystem::path& path) noexcept;

    /**
     * @brief Load every topology from a .vat file with optional texcoord skipping
     * @param path Path to the .vat file
     * @param skip_texcoords If true, texcoords are not read (saves memory when not needed)
     * @return Vector with one VATData per topology, or empty if loading failed
     */
    static std::vector<util::VATData> load_all(const std::filesystem::path& path,
                                               bool skip_texcoords) noexcept;

    /**
     * @brief Load every topology from any Cinder data source
     * @param source Data source of the .vat content
     * @param skip_texcoords If true, texcoords are not read (default false)
     * @return Vector with one VATData per topology, or empty if loading failed
     */
    static std::vector<util::VATData> load_all(const ci::DataSourceRef& source,
                                               bool skip_texcoords = false) noexcept;

    /**
     * @brief Validate VAT file version without full parse
     * @param path Path to the .vat file
     * @return true if version is supported (version 0 or 1)
     */
    static bool validate_version(const std::filesystem::path& path) noexcept;
};

} // namespace newtype::scene
