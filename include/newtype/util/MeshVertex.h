#pragma once
#include <luisa/luisa-compute.h>
#include <luisa/dsl/struct.h>

namespace newtype::util {

/**
 * @brief Extended vertex structure for ray tracing with full attribute support
 *
 * Matches common mesh formats for production rendering:
 * - Positions for geometry
 * - Normals for smooth shading and lighting
 * - Tangents (xyz) + bitangent sign (w) for normal mapping
 * - Texture coordinates for material sampling
 */
struct MeshVertex {
	luisa::float3 position;   // Vertex position in object space
	luisa::float3 normal;      // Vertex normal (smooth/flat shading)
	luisa::float4 tangent;     // xyz=tangent direction, w=bitangent handedness (-1 or +1)
	luisa::float2 texCoord;    // Primary UV set for texture sampling
};

} // namespace newtype::util

// Register MeshVertex with LuisaCompute DSL
LUISA_STRUCT(newtype::util::MeshVertex, position, normal, tangent, texCoord) {};
