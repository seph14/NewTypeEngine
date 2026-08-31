#include "newtype/scene/VATLoader.h"
#include "cinder/Log.h"
#include "cinder/app/App.h"
#include <filesystem>

namespace newtype::scene {

using namespace luisa;

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

    if (version != 0) {
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
    util::VATData result;

    auto target = ci::loadFile(path);
    if (!target) {
        CI_LOG_E("VATLoader: Failed to open " << path.string());
        return result;
    }

    auto stream = target->createStream();

    // Version
    uint32_t version = 0;
    stream->readLittle(&version);
    if (version != 0) {
        CI_LOG_E("VATLoader: Unsupported version " << version << " in " << path.string());
        return result;
    }

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
    if (!skip_texcoords) {
        result.texcoords.resize(result.vertex_count * 2u);
        stream->readData(result.texcoords.data(), 2 * result.vertex_count * sizeof(float));
    } else {
        // Skip texcoords
        stream->seekRelative(2 * result.vertex_count * sizeof(float));
    }

    // Frame data
    uint32_t total_vert_floats = 0;  // frameCount * vertexCount * 3
    stream->readLittle(&total_vert_floats);
    uint32_t total_vertices = total_vert_floats / 3u;
    result.frame_count = total_vertices / result.vertex_count;

    // Read raw float data and reconstruct into luisa::float3
    std::vector<ci::vec3> verts, normals;
    verts.resize  (total_vertices);
    normals.resize(total_vertices);
    stream->readData(verts.data(),   total_vert_floats * sizeof(float));
    stream->readData(normals.data(), total_vert_floats * sizeof(float));

    result.positions.resize(total_vertices);
    result.normals.resize(total_vertices);
    for (uint32_t i = 0; i < total_vertices; ++i) {
        result.positions[i] = luisa::make_float3(verts[i].x, verts[i].y, verts[i].z);
        auto n = luisa::make_float3(normals[i].x, normals[i].y, normals[i].z);
        if (any(isnan(n))) n = luisa::make_float3(0.f, 1.f, 0.f);
        result.normals[i] = n;
    }

    if (stream->tell() != stream->size()) {
        CI_LOG_E("VATLoader: Error reading file: " << path.string());
        return util::VATData{};  // Return empty on error
    }

    CI_LOG_I("VATLoader: Loaded " << path.filename().string()
             << " - verts:" << result.vertex_count
             << " frames:" << result.frame_count
             << " indices:" << result.index_count);

    return result;
}

} // namespace newtype::scene
