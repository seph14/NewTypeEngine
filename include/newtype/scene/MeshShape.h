//
// Created by Claude on 2026/03/24.
//

#pragma once

#include <luisa/luisa-compute.h>
#include "newtype/scene/Shape.h"
#include "newtype/scene/Transform.h"
#include "cinder/TriMesh.h"

namespace newtype::scene {

using namespace luisa;
using compute::Buffer;
using compute::Device;
using compute::Mesh;
using compute::Triangle;

class MeshShape;
typedef luisa::unique_ptr<MeshShape> MeshShapePtr;

/**
 * @brief Triangle mesh shape
 *
 * Concrete Shape implementation that wraps a LuisaCompute Mesh
 * and stores vertex/triangle data. Replaces the old util::Mesh.
 *
 * Features:
 * - Loads from ci::TriMesh (preserves existing functionality)
 * - Stores material ID per-mesh
 * - Has transform with dirty flag for dynamic updates
 * - Caches BLAS for reuse
 */
class MeshShape : public Shape {
public:
    using Vertex = util::Vertex;  // 48 bytes — see util::Vertex static_assert

protected:
    Device &_device;
    luisa::unique_ptr<Transform> _transform;
    uint32_t _material_layers;    // 4 × 8-bit material indices (layer 0 = base, layers 1-3 = modifiers)

    // Geometry data
    luisa::vector<Vertex>   _vertices;
    luisa::vector<Triangle> _triangles;
    Buffer<Vertex>          _vertex_buffer;
    Buffer<Triangle>        _triangle_buffer;

    // LuisaCompute resources
    luisa::unique_ptr<Mesh> _mesh;

    // Bindless array slots (set by Geometry during build)
    uint _vertex_bindless_slot = ~0u;
    uint _triangle_bindless_slot = ~0u;

    // Build state
    bool _built = false, _has_cpu_data = false;
    uint32_t _numTriangle, _numVertices;

public:
    MeshShape(Device &device, uint material_id = 0) noexcept
        : _device(device),
          _material_layers(0xFFFFFF00u | (material_id & 0xFFu)),
          _transform(luisa::make_unique<StaticTransform>()),
          _numTriangle(0), _numVertices(0) {}

    static MeshShapePtr create(Device& device, uint material_id = 0) noexcept {
        return luisa::make_unique<MeshShape>(device, material_id);    }

    virtual ~MeshShape() = default;

    /// Load from Cinder TriMesh (preserves existing util::Mesh functionality)
    bool load_from(ci::TriMesh &triMesh) noexcept;

    /// Load from Cinder geom::Source
    bool load_from(const ci::geom::Source& geom) noexcept;

    /// Direct vertex/triangle data upload
    void set_data(luisa::span<const Vertex> vertices,
                  luisa::span<const Triangle> triangles) noexcept;

    // Unload cpu side data 
    void unloadCPUData() noexcept;
    [[nodiscard]] bool has_cpu_data() const noexcept { return _has_cpu_data; }

    // Shape interface
    [[nodiscard]] bool is_mesh() const noexcept override { return true; }
    [[nodiscard]] MeshView mesh() const noexcept override {
        return MeshView{_vertices, _triangles};
    }

    /// Get base material ID (layer 0) — backward compatible
    [[nodiscard]] uint material_id() const noexcept { return _material_layers & 0xFFu; }
    /// Set base material ID (layer 0)
    void set_material_id(uint id) noexcept {
        _material_layers = (_material_layers & ~0xFFu) | (id & 0xFFu);
    }

    /// Get full 4-layer packed uint32
    [[nodiscard]] uint32_t material_layers() const noexcept { return _material_layers; }
    /// Set full 4-layer packed uint32
    void set_material_layers(uint32_t layers) noexcept { _material_layers = layers; }

    /// Get material index for a specific layer (0-3)
    [[nodiscard]] uint8_t layer_material(uint i) const noexcept {
        return static_cast<uint8_t>((_material_layers >> (i * 8u)) & 0xFFu);
    }
    /// Set material index for a specific layer (0-3)
    void set_layer(uint i, uint8_t idx) noexcept {
        _material_layers = (_material_layers & ~(0xFFu << (i * 8u)))
                         | (static_cast<uint32_t>(idx) << (i * 8u));
    }
    /// Check if a specific layer has a material (index != 0xFF)
    [[nodiscard]] bool has_layer(uint i) const noexcept {
        return layer_material(i) != 0xFFu;
    }
    /// Set multiple layers at once. Inactive layers default to 0xFF.
    void set_layers(uint8_t base, uint8_t l1 = 0xFF, uint8_t l2 = 0xFF, uint8_t l3 = 0xFF) noexcept {
        _material_layers = static_cast<uint32_t>(base)
                         | (static_cast<uint32_t>(l1) << 8u)
                         | (static_cast<uint32_t>(l2) << 16u)
                         | (static_cast<uint32_t>(l3) << 24u);
    }
    /// Get base material index (layer 0)
    [[nodiscard]] uint8_t base_material() const noexcept { return layer_material(0u); }

    /// Get transform
    [[nodiscard]] Transform* transform() noexcept { return _transform.get(); }
    [[nodiscard]] const Transform* transform() const noexcept { return _transform.get(); }

    /// Set transform matrix (marks dirty — applied on the next pipeline update(),
    /// which polls this shape's dirty flag and propagates to the TLAS/instance
    /// transform buffers). For an immediate update outside the frame loop, use
    /// Pipeline::setShapeTransform instead.
    void set_transform(const float4x4 &matrix) noexcept {
        auto static_transform = static_cast<StaticTransform*>(_transform.get());
        static_transform->set_matrix(matrix);
    }

    /// Check if mesh needs TLAS update
    [[nodiscard]] bool is_dirty() const noexcept { return _transform->is_dirty(); }

    /// Double-sided rendering (use face normals instead of interpolated vertex normals)
    void set_double_sided(bool v) noexcept {
        if (v) _properties |= PROPERTY_DOUBLE_SIDED;
        else   _properties &= ~PROPERTY_DOUBLE_SIDED;
    }
    [[nodiscard]] bool is_double_sided() const noexcept {
        return (_properties & PROPERTY_DOUBLE_SIDED) != 0u;
    }

    /// Get LuisaCompute Mesh (BLAS)
    [[nodiscard]] virtual Mesh* mesh_resource() noexcept { return _mesh.get(); }
    [[nodiscard]] virtual const Mesh* mesh_resource() const noexcept { return _mesh.get(); }

    /// Get vertex/triangle buffers
    [[nodiscard]] virtual const Buffer<Vertex>& vertex_buffer() const noexcept { return _vertex_buffer; }
    [[nodiscard]] const Buffer<Triangle>& triangle_buffer() const noexcept { return _triangle_buffer; }

    /// Bindless array slot accessors (set by Geometry during build)
    [[nodiscard]] uint vertex_bindless_slot() const noexcept { return _vertex_bindless_slot; }
    [[nodiscard]] uint triangle_bindless_slot() const noexcept { return _triangle_bindless_slot; }
    void set_vertex_bindless_slot(uint slot) noexcept { _vertex_bindless_slot = slot; }
    void set_triangle_bindless_slot(uint slot) noexcept { _triangle_bindless_slot = slot; }

    /// Get CPU-side vertex data (for deformable meshes)
    [[nodiscard]] const luisa::vector<Vertex>& vertices() const noexcept { return _vertices; }

    /// Get CPU-side triangle data (mirrors vertices() for physics/external access)
    [[nodiscard]] const luisa::vector<Triangle>& triangles() const noexcept { return _triangles; }

    /// Build BLAS
    virtual void build(luisa::compute::Stream &stream) noexcept;

    /// Check if BLAS has been built
    [[nodiscard]] bool built() const noexcept { return _built; }

    /// Triangle count
    [[nodiscard]] uint32_t triangle_count() const noexcept { return _numTriangle; }
    /// Vertex count
    [[nodiscard]] uint32_t vertex_count() const noexcept { return _numVertices; }
};

/**
 * @brief Helper to create MeshShape from Cinder geometry
 */
inline luisa::unique_ptr<MeshShape> make_mesh(
    Device &device, ci::TriMesh &triMesh, uint material_id = 0) noexcept
{
    auto mesh = luisa::make_unique<MeshShape>(device, material_id);
    mesh->load_from(triMesh);
    return mesh;
}

} // namespace newtype::scene
