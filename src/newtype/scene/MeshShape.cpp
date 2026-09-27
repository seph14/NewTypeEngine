//
// Created by Claude on 2026/03/24.
//

#include "newtype/scene/MeshShape.h"
#include "cinder/Log.h"

namespace newtype::scene {

//==============================================================================
// MeshShape
//==============================================================================

bool MeshShape::load_from(ci::TriMesh &triMesh) noexcept {
    _vertices.clear();
    _triangles.clear();

    const auto& vertexIndices = triMesh.getIndices();
    // Require indexed vertices (simpler and matches existing Mesh.cpp)
    if (vertexIndices.empty()) 
        return false;

    if (!triMesh.hasNormals()) 
        triMesh.recalculateNormals(/*smooth=*/true, /*weighted=*/true);
    // Generate tangents if not present (requires UVs)
    if (!triMesh.hasTangents() && triMesh.hasTexCoords0())
        triMesh.recalculateTangents();
    // Derive bitangents for handedness computation
    if (triMesh.hasTangents() && !triMesh.hasBitangents())
        triMesh.recalculateBitangents();

    const ci::vec3* positions = triMesh.getPositions<3>();
    const std::vector<ci::vec3>& normals = triMesh.getNormals();
    const std::vector<ci::vec3>& tangents = triMesh.getTangents();
    const std::vector<ci::vec3>& bitangents = triMesh.getBitangents();
    const ci::vec2* texCoords = triMesh.hasTexCoords0() ? triMesh.getTexCoords0<2>() : nullptr;

    // Check if mesh has required attributes
    bool has_uvs = texCoords != nullptr;
    bool has_tangents = triMesh.hasTangents();
    bool has_bitangents = triMesh.hasBitangents();
    // make sure normals always exist
    _properties |= PROPERTY_HAS_VERTEX_NORMAL;
    if (has_uvs)
        _properties |= PROPERTY_HAS_VERTEX_UV;
    if (has_tangents)
        _properties |= PROPERTY_HAS_VERTEX_TANGENT;
    
    _numTriangle = vertexIndices.size() / 3;
    _triangles.reserve(_numTriangle);

    for (size_t i = 0; i < vertexIndices.size(); i += 3) {
        _triangles.push_back(Triangle{
            static_cast<uint>(vertexIndices[i + 0]),
            static_cast<uint>(vertexIndices[i + 1]),
            static_cast<uint>(vertexIndices[i + 2])
        });
    }

    // Build vertex array
    _numVertices = triMesh.getNumVertices();
    _vertices.reserve(_numVertices);

    for (size_t i = 0; i < _numVertices; ++i) {
        auto& p  = positions[i];
        auto& n  = normals[i];
        auto  uv = has_uvs ? texCoords[i] : ci::vec2(0, 0);

        // Tangent + handedness
        luisa::float4 t(1.f, 0.f, 0.f, 1.f);
        if (has_tangents) {
            auto& tn = tangents[i];
            t = luisa::make_float4(tn.x, tn.y, tn.z, 1.f);
            // Compute handedness from bitangent direction
            if (has_bitangents) {
                auto& bt = bitangents[i];
                ci::vec3 cross_nb = ci::cross(n, tn);
                float sign = ci::dot(cross_nb, bt) >= 0.f ? 1.f : -1.f;
                t.w = sign;
            }
        }

        _vertices.push_back(Vertex::encode(
            luisa::make_float3(p.x, p.y, p.z),
            luisa::make_float3(n.x, n.y, n.z),
            t,
            luisa::make_float2(uv.x, uv.y)
        ));
    }
    
    _has_cpu_data = true;

    // A1: weld bit-identical duplicate vertices (importer index seams) —
    // bit-identical rendering, smaller vertex buffer + BLAS.
    if (const size_t removed = util::weld_vertices(_vertices, _triangles); removed > 0u) {
        _numVertices = static_cast<uint32_t>(_vertices.size());
        CI_LOG_I("MeshShape::load_from: welded " << removed
            << " duplicate vertices (" << _numVertices << " remain)");
    }
    return true;
}

bool MeshShape::load_from(const ci::geom::Source& geom) noexcept {
    ci::TriMesh triMesh(geom);  // Use constructor instead of append()
    return load_from(triMesh);
}

void MeshShape::unloadCPUData() noexcept {
    if (_has_cpu_data) {
        _vertices.clear();
        _triangles.clear();
        _has_cpu_data = false;
    }
}

void MeshShape::set_data(luisa::span<const Vertex> vertices,
                         luisa::span<const Triangle> triangles) noexcept {
    _vertices.assign (vertices.begin(), vertices.end());
    _triangles.assign(triangles.begin(), triangles.end());
    
    _numVertices    = static_cast<uint32_t>(_vertices.size());
    _numTriangle    = static_cast<uint32_t>(_triangles.size());
    _has_cpu_data   = true;

    // Update properties based on data
    _properties |= PROPERTY_HAS_VERTEX_NORMAL;
    _properties |= PROPERTY_HAS_VERTEX_UV;
    _properties |= PROPERTY_HAS_VERTEX_TANGENT;
}

void MeshShape::build(luisa::compute::Stream &stream) noexcept {
    if (_vertices.empty() || _triangles.empty()) 
        return;

    // Create vertex buffer (A2: ActiveVertex layout — fp32 identity or
    // snorm10-packed; upload stages through the CPU authoring cache).
    _vertex_buffer = _device.create_buffer<GpuVertex>(_numVertices);
    util::upload_vertex_buffer(_vertex_buffer,
                               std::span<const Vertex>{_vertices}, stream);

    // Create triangle buffer
    _triangle_buffer = _device.create_buffer<Triangle>(_numTriangle);
    stream << _triangle_buffer.copy_from(_triangles.data());

    // Create LuisaCompute Mesh (BLAS)
    _mesh = luisa::make_unique<Mesh>(_device.create_mesh(
        _vertex_buffer, _triangle_buffer));

    // Build acceleration structure
    stream << _mesh->build();

    _built = true;
}

} // namespace newtype::scene
