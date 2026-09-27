#include "newtype/scene/VATLoader.h"
#include "cinder/Log.h"
#include "cinder/app/App.h"
#include <filesystem>

namespace newtype::scene {

using namespace luisa;

//==============================================================================
// Internal helpers
//==============================================================================

namespace {

// V1 packed header-table entry: bounds + counts for one topology
struct TopologyHeaderV1 {
    ci::vec3 center, extend;
    uint32_t index_count  = 0;
    uint32_t vertex_count = 0;
    uint32_t frame_count  = 0;
};

// Read a frame-major block of `count` float3s (raw little-endian floats)
luisa::vector<luisa::float3> read_float3_block(const ci::IStreamRef& stream,
                                               uint32_t count, bool fix_nan) noexcept {
    std::vector<ci::vec3> raw(count);
    stream->readData(raw.data(), count * 3 * sizeof(float));
    luisa::vector<luisa::float3> out(count);
    for (uint32_t i = 0; i < count; ++i) {
        out[i] = luisa::make_float3(raw[i].x, raw[i].y, raw[i].z);
        if (fix_nan && any(isnan(out[i])))
            out[i] = luisa::make_float3(0.f, 1.f, 0.f);
    }
    return out;
}

// Parse one V0 topology; the version field has already been consumed
util::VATData parse_v0(const ci::IStreamRef& stream, bool skip_texcoords) noexcept {
    util::VATData result;

    // Bounding box (skip — not needed)
    ci::vec3 center, extend;
    stream->readData(&center.x, 3 * sizeof(float));
    stream->readData(&extend.x, 3 * sizeof(float));

    // Indices
    stream->readLittle(&result.index_count);
    result.indices.resize(result.index_count);
    stream->readData(result.indices.data(), result.index_count * sizeof(uint32_t));

    // Vertex count + texcoords
    stream->readLittle(&result.vertex_count);
    if (result.vertex_count == 0)
        return util::VATData{};
    if (!skip_texcoords) {
        result.texcoords.resize(result.vertex_count * 2u);
        stream->readData(result.texcoords.data(), 2 * result.vertex_count * sizeof(float));
    } else {
        stream->seekRelative(2 * result.vertex_count * sizeof(float));
    }

    // Frame data (frame count is derived in V0)
    uint32_t total_vert_floats = 0;  // frameCount * vertexCount * 3
    stream->readLittle(&total_vert_floats);
    uint32_t total_vertices = total_vert_floats / 3u;
    result.frame_count = total_vertices / result.vertex_count;

    result.positions = read_float3_block(stream, total_vertices, /*fix_nan=*/false);
    result.normals   = read_float3_block(stream, total_vertices, /*fix_nan=*/true);

    return result;
}

} // namespace

//==============================================================================
// Version Validation
//==============================================================================

bool VATLoader::validate_version(const std::filesystem::path& path) noexcept {
    auto target = ci::loadFile(path);
    if (!target) {
        CI_LOG_E("VATLoader: Failed to open " << path.string());
        return false;
    }

    auto stream = target->createStream();
    uint32_t version = 0;
    stream->readLittle(&version);

    if (version != 0 && version != 1) {
        CI_LOG_E("VATLoader: Unsupported version " << version << " in " << path.string());
        return false;
    }

    return true;
}

//==============================================================================
// Loading
//==============================================================================

util::VATData VATLoader::load(const std::filesystem::path& path) noexcept {
    return load(path, false);
}

util::VATData VATLoader::load(const std::filesystem::path& path, bool skip_texcoords) noexcept {
    try {
        return load(ci::loadFile(path), skip_texcoords);
    } catch (const std::exception& ex) {
        CI_LOG_E("VATLoader: Failed to open " << path.string() << " - " << ex.what());
        return util::VATData{};
    }
}

util::VATData VATLoader::load(const ci::DataSourceRef& source, bool skip_texcoords) noexcept {
    auto all = load_all(source, skip_texcoords);
    if (all.empty())
        return util::VATData{};
    return std::move(all.front());
}

std::vector<util::VATData> VATLoader::load_all(const std::filesystem::path& path) noexcept {
    return load_all(path, false);
}

std::vector<util::VATData> VATLoader::load_all(const std::filesystem::path& path,
                                               bool skip_texcoords) noexcept {
    try {
        return load_all(ci::loadFile(path), skip_texcoords);
    } catch (const std::exception& ex) {
        CI_LOG_E("VATLoader: Failed to open " << path.string() << " - " << ex.what());
        return {};
    }
}

std::vector<util::VATData> VATLoader::load_all(const ci::DataSourceRef& source,
                                               bool skip_texcoords) noexcept {
    std::vector<util::VATData> results;

    if (!source) {
        CI_LOG_E("VATLoader: Invalid data source");
        return results;
    }

    const auto name = source->getFilePath().string();
    auto stream = source->createStream();
    if (!stream) {
        // ci::loadFile() hands back a source for missing files; its stream is
        // null. Without this guard the first readLittle dereferences null.
        CI_LOG_E("VATLoader: Cannot open stream for " << name);
        return results;
    }

    // Version
    uint32_t version = 0;
    stream->readLittle(&version);

    if (version == 0) {
        // Single topology per file
        auto data = parse_v0(stream, skip_texcoords);
        if (data.is_valid())
            results.push_back(std::move(data));
    } else if (version == 1) {
        // Packed multi-topology: header table first, then the payloads in table order
        uint32_t topology_count = 0;
        stream->readLittle(&topology_count);
        if (topology_count == 0) {
            CI_LOG_E("VATLoader: Empty topology table in " << name);
            return results;
        }

        std::vector<TopologyHeaderV1> headers(topology_count);
        for (auto& header : headers) {
            stream->readData(&header.center.x, 3 * sizeof(float));
            stream->readData(&header.extend.x, 3 * sizeof(float));
            stream->readLittle(&header.index_count);
            stream->readLittle(&header.vertex_count);
            stream->readLittle(&header.frame_count);
        }

        results.reserve(topology_count);
        for (uint32_t t = 0; t < topology_count; ++t) {
            const auto& header = headers[t];
            if (header.vertex_count == 0 || header.frame_count == 0) {
                CI_LOG_E("VATLoader: Invalid topology " << t << " in " << name);
                return {};
            }

            util::VATData data;
            data.index_count  = header.index_count;
            data.vertex_count = header.vertex_count;
            data.frame_count  = header.frame_count;

            data.indices.resize(header.index_count);
            stream->readData(data.indices.data(), header.index_count * sizeof(uint32_t));

            if (!skip_texcoords) {
                data.texcoords.resize(header.vertex_count * 2u);
                stream->readData(data.texcoords.data(), 2 * header.vertex_count * sizeof(float));
            } else {
                stream->seekRelative(2 * header.vertex_count * sizeof(float));
            }

            const uint32_t total_vertices = header.vertex_count * header.frame_count;
            data.positions = read_float3_block(stream, total_vertices, /*fix_nan=*/false);
            data.normals   = read_float3_block(stream, total_vertices, /*fix_nan=*/true);

            results.push_back(std::move(data));
        }
    } else {
        CI_LOG_E("VATLoader: Unsupported version " << version << " in " << name);
        return results;
    }

    if (stream->tell() != stream->size()) {
        CI_LOG_E("VATLoader: Error reading file: " << name);
        return {};  // Return empty on error
    }

    for (const auto& r : results) {
        CI_LOG_D("VATLoader: Loaded " << name
                 << " - verts:" << r.vertex_count
                 << " frames:" << r.frame_count
                 << " indices:" << r.index_count);
    }

    return results;
}

} // namespace newtype::scene
