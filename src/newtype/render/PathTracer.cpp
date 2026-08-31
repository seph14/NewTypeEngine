#include "newtype/render/PathTracer.h"
#include "newtype/util/MoveOnlyAny.h"
#include <luisa/dsl/sugar.h>
#include <limits>
#include <cmath>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;
using namespace newtype::core;
using namespace newtype::util;

//==============================================================================
// PathTracer Implementation
//==============================================================================

PathTracer::PathTracer(Device& device, const Config& config)
    : mConfig(config) {

    LUISA_INFO("PathTracer created: maxDepth={}, NEE={}, MIS={}",
        config.maxDepth, config.enableNEE, config.enableMIS);
}

PathTracer::~PathTracer() = default;

//==============================================================================
// Compile Path Tracer Kernel
//==============================================================================

MoveOnlyAny PathTracer::compile(Device& device) {

    //==========================================================================
    // STEP 2: MATERIAL + NEE (Next Event Estimation)
    //
    // Add direct light sampling for faster convergence.
    // No MIS yet.
    //==========================================================================

    // Linear congruential generator for random numbers
    Callable<float(uint&)> lcg = [](UInt& state) noexcept {
        constexpr uint a = 1664525u;
        constexpr uint c = 1013904223u;
        state = a * state + c;
        return cast<float>(state) / cast<float>(std::numeric_limits<uint>::max());
    };

    // Sample uniform triangle (local version for kernel)
    Callable<float3(float2)> sample_uniform_triangle = [](Float2 u) noexcept {
        Float2 uv = ite(
            u.x < u.y,
            make_float2(0.5f * u.x, -0.5f * u.x + u.y),
            make_float2(-0.5f * u.y + u.x, 0.5f * u.y));
        return make_float3(uv, 1.0f - uv.x - uv.y);
    };

    // Cosine-weighted hemisphere sampling
    Callable<float3(float2)> cosine_sample_hemisphere = [](Float2 u) noexcept {
        Float r = sqrt(u.x);
        Float theta = 2.0f * pi * u.y;
        return make_float3(r * cos(theta), r * sin(theta), sqrt(1.0f - u.x));
    };

    // Orthonormal basis from normal
    Callable<float3x3(float3)> make_onb = [](Float3 normal) noexcept {
        Float3 b1 = ite(abs(normal.y) < 0.999f,
            normalize(cross(normal, make_float3(0.0f, 1.0f, 0.0f))),
            make_float3(1.0f, 0.0f, 0.0f));
        Float3 b2 = cross(normal, b1);
        return make_float3x3(b1, b2, normal);
    };

    // Cosine hemisphere PDF
    Callable<float(float)> cosine_hemisphere_pdf = [](Float cos_theta) noexcept {
        return cos_theta * inv_pi;
    };

    // Path tracer kernel - STEP 2: Material + NEE
    Kernel2D path_tracer = [&](
        ImageFloat output,
        ImageUInt seed_image,
        AccelVar accel,
        BufferVar<util::Vertex> vertices,
        BufferVar<uint> material_indices,
        BufferVar<MaterialData> materials,
        Float light_select_pmf,                 // Unused in Step 1
        BufferVar<EmissiveTriangle> emissives,  // Unused in Step 1
        UInt emissive_count,                    // Unused in Step 1
        Var<util::CameraData> camera,
        UInt frame_index,
        UInt max_bounces) noexcept {

        set_block_size(16u, 16u, 1u);

        UInt2 coord         = dispatch_id().xy();
        Float2 resolution   = make_float2(dispatch_size().xy());

        // Initialize random seed on first frame
        $if(frame_index == 0u) {
            seed_image.write(coord, make_uint4(coord.x * 1234567u + coord.y));
        };

        UInt seed = seed_image.read(coord).x;

        // Generate camera ray with jittered sampling
        Float2 uv = (make_float2(coord) + make_float2(lcg(seed), lcg(seed))) / resolution;
        Float2 ndc = uv * 2.0f - 1.0f;
        Float aspect = resolution.x / resolution.y;
        Float2 pixel_ndc = ndc * make_float2(aspect, 1.0f);

        // Generate ray from camera
        Var<Ray> ray = camera->generate_ray(pixel_ndc);

        // Path tracing loop
        Float3 radiance = def(make_float3(0.0f));
        Float3 throughput = def(make_float3(1.0f));

        $for(bounce, max_bounces) {
            // Trace ray
            Var<TriangleHit> hit = accel.intersect(ray, {});

            $if(hit->miss()) {
                $break;
            };

            // Get hit information
            Var<util::Vertex> v0 = vertices.read(hit.prim * 3u + 0u);
            Var<util::Vertex> v1 = vertices.read(hit.prim * 3u + 1u);
            Var<util::Vertex> v2 = vertices.read(hit.prim * 3u + 2u);

            // Interpolate vertex attributes
            Float3 p = hit->triangle_interpolate(v0->position(), v1->position(), v2->position());
            Float3 n = normalize(hit->triangle_interpolate(v0->normal(), v1->normal(), v2->normal()));

            // Get material from material pool
            UInt matIdx = material_indices.read(hit.prim);
            Var<MaterialData> material = materials.read(matIdx);

            Float3 albedo = material->albedo.xyz();
            Float3 emission = material->emission;

            // Add emission if hitting light
            $if(emission.x > 0.0f | emission.y > 0.0f | emission.z > 0.0f) {
                radiance += throughput * emission;
                $break;
            };

            // ===== NEXT EVENT ESTIMATION (Direct Light Sampling) =====
            $if(emissive_count > 0u) {
                // Sample a light uniformly
                Float u_light = lcg(seed);
                UInt light_idx = clamp(cast<uint>(u_light * cast<float>(emissive_count)), 0u, emissive_count - 1u);

                // Get emissive triangle data
                Var<EmissiveTriangle> tri = emissives.read(light_idx);

                // Sample point on triangle
                Float2 u_tri = make_float2(lcg(seed), lcg(seed));
                Float3 bary = sample_uniform_triangle(u_tri);

                // Get vertex positions for the light triangle
                Var<util::Vertex> lv0 = vertices.read(tri->v0);
                Var<util::Vertex> lv1 = vertices.read(tri->v1);
                Var<util::Vertex> lv2 = vertices.read(tri->v2);

                // Compute light sample point
                Float3 light_p = bary.x * lv0->position() + bary.y * lv1->position() + bary.z * lv2->position();

                // Direction to light
                Float3 to_light = light_p - p;
                Float light_dist = length(to_light);
                Float3 light_dir = to_light / light_dist;

                // Cast shadow ray
                Float3 shadow_origin = p + n * 1e-4f;
                Var<Ray> shadow_ray = make_ray(shadow_origin, light_dir);
                Var<TriangleHit> shadow_hit = accel.intersect(shadow_ray, {});

                // Check if light is visible
                $if(shadow_hit->miss() | shadow_hit->distance() >= light_dist - 1e-3f) {
                    // BSDF evaluation (Lambertian)
                    Float bsdf_cos_theta = max(dot(light_dir, n), 0.0f);
                    Float3 bsdf_eval = albedo * inv_pi;
                    Float bsdf_pdf = bsdf_cos_theta * inv_pi;

                    // Light PDF (uniform over area and lights)
                    Float light_area = length(cross(lv1->position() - lv0->position(), lv2->position() - lv0->position())) * 0.5f;
                    Float light_pdf = 1.0f / (light_area * cast<float>(emissive_count));

                    // MIS weight (power heuristic)
                    Float mis_weight = (light_pdf * light_pdf) / (light_pdf * light_pdf + bsdf_pdf * bsdf_pdf);

                    // Add direct lighting with MIS
                    radiance += throughput * tri->emission * bsdf_eval * bsdf_cos_theta * mis_weight / light_pdf;
                };
            };

            // ===== BSDF SAMPLING =====
            Float2 u_bsdf = make_float2(lcg(seed), lcg(seed));
            Float3 wi_local = cosine_sample_hemisphere(u_bsdf);
            Float3x3 tnb = make_onb(n);
            Float3 wi = tnb * wi_local;
            Float bsdf_pdf = cosine_hemisphere_pdf(abs(dot(wi, n)));

            // Evaluate BSDF (Lambertian)
            Float3 bsdf_eval = albedo * inv_pi;

            // Update throughput
            Float cos_theta = abs(dot(wi, n));
            throughput *= bsdf_eval * cos_theta / bsdf_pdf;

            // Russian roulette
            UInt rr_start = 3u;
            $if(bounce + 1u >= rr_start) {
                Float p_survive = max(throughput.x, max(throughput.y, throughput.z));
                $if(lcg(seed) > p_survive) { $break; };
                throughput /= p_survive;
            };

            // Setup next ray
            Float3 new_origin = p + n * 1e-4f;
            ray = make_ray(new_origin, wi);
        };

        // Accumulate over frames
        $if(frame_index == 0u) {
            output.write(coord, make_float4(radiance, 1.0f));
        }
        $else {
            Float4 prev = output.read(coord);
            Float count = prev.w + 1.0f;
            Float3 avg = lerp(prev.xyz(), radiance, 1.0f / count);
            output.write(coord, make_float4(avg, count));
        };

        seed_image.write(coord, make_uint4(seed));
    };

    // Compile kernel - store with static lifetime
    static auto compiled = device.compile(path_tracer);
    return MoveOnlyAny(&compiled);
}

} // namespace newtype::render
