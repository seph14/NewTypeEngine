#pragma once

#include <luisa/luisa-compute.h>
#include "newtype/render/MaterialPool.h"
#include "newtype/render/LightSampler.h"

namespace newtype {
namespace core {

//==========================================================================
// Binding Groups for invariant GPU resources
//==========================================================================

// Environment light resources (invariant after envmap setup)
struct EnvLightResources {
    const luisa::compute::Image<float>& envmap;
    const luisa::compute::Buffer<float>& env_marginal_cdf;
    const luisa::compute::Buffer<float>& env_conditional_cdf;
    float env_integral;
    luisa::uint env_width;
    luisa::uint env_height;
    luisa::float3x3 env_rotation;  // direct value (read from buffer on CPU side)
};

// Light sampling data (invariant after buildScene)
struct LightSamplingResources {
    const luisa::compute::Buffer<render::LightSampler::TriangleLight>& triangle_lights;
    const luisa::compute::Buffer<render::LightSampler::TriangleVertexData>& triangle_vertices;
    const luisa::compute::Buffer<render::AliasEntry>& alias_table;
    luisa::uint emissive_count;
    float total_power_inv;
    float emissive_count_inv;
    const luisa::compute::Buffer<luisa::uint>& instance_to_light_base;
};

// Scene geometry buffers (invariant after buildScene)
struct SceneGeometryResources {
    const luisa::compute::Buffer<luisa::uint4>& instance_buffer;
    const luisa::compute::Buffer<luisa::float4x4>& instance_transform_buffer;
    const luisa::compute::Buffer<luisa::float4x4>& instance_transform_prev_buffer;
    const luisa::compute::Buffer<render::MaterialData>& material_buffer;
    // Packed {roughness, luminance(F0), luminance(albedo)} per material for the
    // ReSTIR GI similarity gates (see MaterialSimilarity.h) — 12B vs 176B reads.
    const luisa::compute::Buffer<luisa::float3>& sim_key_buffer;
};

} // namespace core
} // namespace newtype

// Binding group registrations (global namespace)
LUISA_BINDING_GROUP(newtype::core::EnvLightResources,
                    envmap, env_marginal_cdf, env_conditional_cdf,
                    env_integral, env_width, env_height, env_rotation) {};

LUISA_BINDING_GROUP(newtype::core::LightSamplingResources,
                    triangle_lights, triangle_vertices, alias_table,
                    emissive_count, total_power_inv, emissive_count_inv,
                    instance_to_light_base) {};

LUISA_BINDING_GROUP(newtype::core::SceneGeometryResources,
                    instance_buffer, instance_transform_buffer, instance_transform_prev_buffer,
                    material_buffer, sim_key_buffer) {};
