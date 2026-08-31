#include "newtype/core/Pipeline.h"
#include "newtype/util/Rng.h"
#include "newtype/render/Shading.h"
#include "cinder/Log.h"
#include "newtype/util/UiHelper.h"
#include "cinder/CinderImGui.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>

namespace newtype::core {
	using namespace luisa;
	using namespace luisa::compute;
	using namespace newtype::scene;
	using namespace newtype::render;

	void Pipeline::_compileDenoiserShaders() {
		auto& device = Renderer::device();

        //==========================================================================
    // Denoise PreFilter: G-Buffer → clean albedo + world normal
    //==========================================================================
        _denoisePreFilterShader = device.compile<2>([&](
            ImageFloat albedo_output,
            ImageFloat normal_output,
            ImageFloat gbuf_depth,
            ImageUInt  gbuf_vis,
            ImageFloat gbuf_bary,
            Var<util::CameraData> camera,
            BufferVar<luisa::uint4> instance_buffer,
            BufferVar<MaterialData> material_buffer,
            BindlessVar vertex_bindless
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 coord = dispatch_id().xy();
            UInt2 resolution = dispatch_size().xy();

            Float  depth = gbuf_depth.read(coord).x;
            UInt4  vis = gbuf_vis.read(coord);
            UInt   inst_id = vis.x;
            UInt   prim_id = vis.y;

            Float3 albedo = def(luisa::make_float3(0.0f));
            Float3 normal = def(luisa::make_float3(0.0f, 0.0f, 1.0f));
            Float  has_emission = def(0.0f); // 0 = no emission, 1 = emissive

            $if(inst_id != ~0u) {
                Float2 bary = gbuf_bary.read(coord).xy();

                // Reconstruct normal via bindless vertex access
                UInt4 inst_data = instance_buffer.read(inst_id);
                Float3 ns = reconstruct_normal(vertex_bindless,
                    inst_data.z, inst_data.w, prim_id, bary);

                // Flip normal toward camera (same as shade pass)
                Float2 ndc = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
                auto   ray = camera->generate_ray(ndc);
                Float3 wo = -normalize(ray->direction());
                ns = ite(dot(wo, ns) < 0.f, -ns, ns);

                // Read material albedo (base color without lighting)
                Var<MaterialData> material = material_buffer.read(Expr{ inst_data.y & 0xFFu });

                albedo = material.albedo.xyz();
                normal = ns;

                // Mark emissive surfaces (emission luminance > threshold)
                Float em_lum = luminance(material.emission);
                has_emission = ite(em_lum > 0.01f, 1.0f, 0.0f);
            } $else {
                // Background pixels: skip denoising entirely
                has_emission = 1.0f;
            };

            // .w = emissive flag (1.0 if surface has emission, 0.0 otherwise)
            albedo_output.write(coord, make_float4(albedo, has_emission));
            normal_output.write(coord, make_float4(normal, 0.0f));
        });

        //==========================================================================
        // SVGF: Albedo Demodulate — divide shade output by albedo → lighting-only
        //==========================================================================
        _svgfDemodShader = device.compile<2>([&](
            ImageFloat output,
            ImageFloat input,
            ImageFloat albedo_img
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 coord = dispatch_id().xy();

            Float3 color = input.read(coord).xyz();
            Float4 albedo_data = albedo_img.read(coord);
            Float3 albedo = albedo_data.xyz();
            Float  is_emissive = albedo_data.w; // 1.0 if surface has emission

            Float  max_albedo = luisa::compute::max(albedo.x, luisa::compute::max(albedo.y, albedo.z));

            // For emissive surfaces: pass through as-is (emission is not albedo-scaled)
            // For non-emissive: divide by albedo to get pure lighting
            Float3 lighting = ite(Expr{ is_emissive > 0.5f | max_albedo < 0.01f },
                color,
                //ite(max_albedo < 0.01f, color, color / (albedo + 0.1f))
                color / (albedo + 0.1f)
            );
            //      color / luisa::compute::max(albedo, 0.001f)));
           // Clamp to prevent fp16 overflow in HALF4 buffers
            output.write(coord, make_float4(clamp(lighting, -100.0f, 100.0f), 1.0f));
        });

        //==========================================================================
        // SVGF: Temporal Accumulation — EMA blend lighting + luminance moments
        //==========================================================================
        _svgfTemporalShader = device.compile<2>([&](
            ImageFloat output_history,
            ImageFloat output_moments,
            ImageFloat input_lighting,
            ImageFloat history_read,
            ImageFloat moments_read,
            ImageFloat gbuf_motion,
            ImageFloat gbuf_depth,
            ImageUInt  gbuf_vis,
            ImageFloat gbuf_depth_prev,
            ImageUInt  gbuf_vis_prev,
            ImageFloat normal_img,
            ImageFloat normal_prev_img,
            UInt       reset_accum,
            Float      alpha,
            Float      alphaMoments,
            ImageFloat albedo_img
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 coord = dispatch_id().xy();
            UInt2 resolution = dispatch_size().xy();
            Float2 resF = make_float2(resolution);

            Float3 cur_lighting = input_lighting.read(coord).xyz();
            Float  cur_depth = gbuf_depth.read(coord).x;
            UInt4  cur_vis = gbuf_vis.read(coord);
            Float3 cur_normal = normal_img.read(coord).xyz();
            UInt   cur_inst = cur_vis.x;  // instance ID for tap rejection

            // Approximate fwidth of depth using neighbor reads
            Int ix = cast<Int>(coord.x);
            Int iy = cast<Int>(coord.y);
            Float zL = gbuf_depth.read(make_uint2(cast<uint>(luisa::compute::max(ix - 1, 0)), coord.y)).x;
            Float zR = gbuf_depth.read(make_uint2(cast<uint>(luisa::compute::min(ix + 1, cast<Int>(resolution.x) - 1)), coord.y)).x;
            Float zB = gbuf_depth.read(make_uint2(coord.x, cast<uint>(luisa::compute::max(iy - 1, 0)))).x;
            Float zT = gbuf_depth.read(make_uint2(coord.x, cast<uint>(luisa::compute::min(iy + 1, cast<Int>(resolution.y) - 1)))).x;
            Float fwidthZ = luisa::compute::max(abs(zR - zL), abs(zT - zB)) * 0.5f;

            // Approximate fwidth of normal using neighbor reads
            Float3 nL = normal_img.read(make_uint2(cast<uint>(luisa::compute::max(ix - 1, 0)), coord.y)).xyz();
            Float3 nR = normal_img.read(make_uint2(cast<uint>(luisa::compute::min(ix + 1, cast<Int>(resolution.x) - 1)), coord.y)).xyz();
            Float3 nB = normal_img.read(make_uint2(coord.x, cast<uint>(luisa::compute::max(iy - 1, 0)))).xyz();
            Float3 nT = normal_img.read(make_uint2(coord.x, cast<uint>(luisa::compute::min(iy + 1, cast<Int>(resolution.y) - 1)))).xyz();
            Float fwidthN = luisa::compute::max(
                luisa::compute::length(nR - nL),
                luisa::compute::length(nT - nB)) * 0.5f;

            // Reproject: compute sub-pixel previous position
            Float2 motion = gbuf_motion.read(coord).xy();
            Float2 posPrev = make_float2(coord) + motion * resF;
            Float fracX = luisa::compute::fract(posPrev.x);
            Float fracY = luisa::compute::fract(posPrev.y);
            Int2 basePrev = make_int2(floor(posPrev));

            // Bilinear 4-tap offsets
            luisa::int2 bOffset[4]{
                {0, 0}, {1, 0},
                {0, 1}, {1, 1}
            };
            Constant<luisa::int2> bOffsets(bOffset, 4);

            // Bilinear weights (computed from fractional position)
            Float bW0 = (1.0f - fracX) * (1.0f - fracY);
            Float bW1 = fracX * (1.0f - fracY);
            Float bW2 = (1.0f - fracX) * fracY;
            Float bW3 = fracX * fracY;

            // Per-tap validation and bilinear accumulation
            Bool valid = false;
            Float4 prevIllum = def(make_float4(0.0f));
            Float2 prevMoments = def(make_float2(0.0f));
            Float  sumW = def(0.0f);
            Float  prevAge = def(0.0f);

            $if(reset_accum == 0u) {
                $for(tapIdx, 4) {
                    Int2 tapCoord = basePrev + bOffsets.read(tapIdx);
                    $if(tapCoord.x >= 0 & tapCoord.x < cast<Int>(resolution.x)
                        & tapCoord.y >= 0 & tapCoord.y < cast<Int>(resolution.y)) {
                        UInt2 tc = make_uint2(cast<uint>(tapCoord.x), cast<uint>(tapCoord.y));

                        Float tapDepth = gbuf_depth_prev.read(tc).x;
                        Float3 tapNormal = normal_prev_img.read(tc).xyz();
                        UInt   tapInst = gbuf_vis_prev.read(tc).x;

                        // Depth validation: |dz| / (fwidthZ + 1e-2) < 10
                        Bool depthOk = abs(tapDepth - cur_depth) / (fwidthZ + 1e-2f) < 10.0f;
                        // Normal validation: distance / (fwidthN + 1e-2) < 16
                        Bool normalOk = luisa::compute::length(tapNormal - cur_normal) / (fwidthN + 1e-2f) < 16.0f;
                        // Instance validation: reject cross-surface history
                        Bool instOk = (tapInst == cur_inst);

                        $if(depthOk & normalOk & instOk) {
                            Float w = ite(tapIdx == 0u, bW0,
                                ite(tapIdx == 1u, bW1,
                                    ite(tapIdx == 2u, bW2, bW3)));
                            Float4 tapHist = history_read.read(tc);
                            prevIllum = prevIllum + w * tapHist;
                            prevMoments = prevMoments + w * moments_read.read(tc).xy();
                            prevAge = prevAge + w * tapHist.w;
                            sumW = sumW + w;
                            valid = true;
                        };
                    };
                };

                // Normalize accumulated bilinear result
                $if(valid & sumW >= 0.01f) {
                    Float invW = 1.0f / sumW;
                    prevIllum = prevIllum * invW;
                    prevMoments = prevMoments * invW;
                    prevAge = prevAge * invW;
                }
                $else{
                    // Bilinear taps all failed — cross-bilateral fallback
                    // Search 5x5 around reprojected position for valid history samples
                    Float  cbCount = def(0.0f);
                    Float4 cbIllum = def(make_float4(0.0f));
                    Float2 cbMoments = def(make_float2(0.0f));
                    Float  cbAge = def(0.0f);

                    luisa::int2 cbOffsets[25]{
                        {-2,-2},{-1,-2},{0,-2},{1,-2},{2,-2},
                        {-2,-1},{-1,-1},{0,-1},{1,-1},{2,-1},
                        {-2, 0},{-1, 0},{0, 0},{1, 0},{2, 0},
                        {-2, 1},{-1, 1},{0, 1},{1, 1},{2, 1},
                        {-2, 2},{-1, 2},{0, 2},{1, 2},{2, 2}
                    };
                    Constant<luisa::int2> cbOff(cbOffsets, 25);

                    $for(ci, 25u) {
                        Int2 cbCoord = basePrev + cbOff.read(ci);
                        $if(cbCoord.x >= 0 & cbCoord.x < cast<Int>(resolution.x)
                           & cbCoord.y >= 0 & cbCoord.y < cast<Int>(resolution.y)) {
                            UInt2 cc = make_uint2(cast<uint>(cbCoord.x), cast<uint>(cbCoord.y));
                            Float cbDepth = gbuf_depth_prev.read(cc).x;
                            Float3 cbNormal = normal_prev_img.read(cc).xyz();
                            UInt   cbInst = gbuf_vis_prev.read(cc).x;

                            Bool depthOk = abs(cbDepth - cur_depth) / (fwidthZ + 1e-2f) < 10.0f;
                            Bool normalOk = luisa::compute::length(cbNormal - cur_normal) / (fwidthN + 1e-2f) < 16.0f;
                            Bool instOk = (cbInst == cur_inst);

                            $if(depthOk & normalOk & instOk) {
                                Float4 cbHist = history_read.read(cc);
                                cbIllum = cbIllum + cbHist;
                                cbMoments = cbMoments + moments_read.read(cc).xy();
                                cbAge = cbAge + cbHist.w;
                                cbCount = cbCount + 1.0f;
                            };
                        };
                    };

                    $if(cbCount > 0.0f) {
                        Float invC = 1.0f / cbCount;
                        prevIllum = cbIllum * invC;
                        prevMoments = cbMoments * invC;
                        // Reset age for cross-bilateral samples (disoccluded region —
                        // averaging ages from random valid neighbors carries bright history)
                        prevAge = 1.0f;
                        valid = true;
                    }
                    $else {
                        valid = false;
                        prevIllum = make_float4(0.0f);
                        prevMoments = make_float2(0.0f);
                        prevAge = 0.0f;
                    };
                };
            };

            // Compute current luminance
            Float cur_lum = luminance(cur_lighting);

            // History length: increment if valid, reset otherwise
            Float age = luisa::compute::min(ite(valid, prevAge + 1.0f, 1.0f), 32.0f);

            // History-length adaptive alpha: max(alpha, 1/age) for faster early convergence
            Float effAlpha = ite(valid, luisa::compute::max(alpha, 1.0f / age), 1.0f);
            Float effAlphaMoments = ite(valid, luisa::compute::max(alphaMoments, 1.0f / age), 1.0f);

            // Emissive surfaces: force alpha=1.0 (no temporal accumulation — emission is deterministic)
            Float is_emissive = albedo_img.read(coord).w;
            effAlpha = ite(is_emissive > 0.5f, 1.0f, effAlpha);
            effAlphaMoments = ite(is_emissive > 0.5f, 1.0f, effAlphaMoments);

            Float3 prevColor = prevIllum.xyz();
            // NaN guard
            $if(luisa::compute::isnan(prevColor.x) | luisa::compute::isnan(prevColor.y)
                | luisa::compute::isnan(prevColor.z)) {
                prevColor = cur_lighting;
                age = 1.0f;
            };
            Float pmx = prevMoments.x;
            Float pmy = prevMoments.y;
            $if(luisa::compute::isnan(pmx) | luisa::compute::isnan(pmy)) {
                pmx = 0.0f;
                pmy = 0.0f;
            };

            // Temporal integration with separate alpha
            Float3 blended = lerp(cur_lighting, prevColor, 1.0f - effAlpha);
            Float4 out_hist = make_float4(blended, age);

            Float mu1 = lerp(cur_lum, pmx, 1.0f - effAlphaMoments);
            Float mu2 = lerp(cur_lum * cur_lum, pmy, 1.0f - effAlphaMoments);
            Float4 out_mom = make_float4(mu1, mu2, 0.0f, 0.0f);

            // Clamp to safe range to prevent fp16 overflow downstream
            out_hist = make_float4(clamp(out_hist.xyz(), -100.0f, 100.0f), out_hist.w);
            out_mom = make_float4(clamp(out_mom.xy(), 0.0f, 1e4f), out_mom.zw());

            output_history.write(coord, out_hist);
            output_moments.write(coord, out_mom);
        });

        //==========================================================================
        // SVGF: Variance Estimation — moments → variance + 3x3 Gaussian prefilter
        //==========================================================================
        _svgfVarianceShader = device.compile<2>([&](
            ImageFloat output_variance,
            ImageFloat input_history,
            ImageFloat input_moments,
            ImageFloat gbuf_depth,
            ImageUInt  gbuf_vis
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 coord = dispatch_id().xy();
            UInt2 resolution = dispatch_size().xy();

            Float age = input_history.read(coord).w;
            Float mu1 = input_moments.read(coord).x;
            Float mu2 = input_moments.read(coord).y;
            Float var = luisa::compute::max(mu2 - mu1 * mu1, 0.0f);
            // Variance boost for early frames (more aggressive atrous filtering)
            var *= 4.0f / luisa::compute::max(age, 1.0f);

            Float  c_depth = gbuf_depth.read(coord).x;
            UInt4  c_vis = gbuf_vis.read(coord);

            // If history is short (< 4 frames), use spatial variance fallback
            $if(age < 4.0f) {
                Float3 c_light = input_history.read(coord).xyz();
                Float3 sum_l = def(c_light);
                Float  sum_l2 = def(luminance(c_light) * luminance(c_light));
                Float  sum_w = def(1.0f);

                // 3x3 neighborhood spatial variance
                $for(dy, -1, 2) {
                    $for(dx, -1, 2) {
                        $if(dx == 0 & dy == 0) { $continue; };
                        Int2 nc = make_int2(cast<int>(coord.x) + dx, cast<int>(coord.y) + dy);
                        $if(nc.x >= 0 & nc.x < cast<int>(resolution.x)
                            & nc.y >= 0 & nc.y < cast<int>(resolution.y)) {
                            UInt2 ncoord = make_uint2(cast<uint>(nc.x), cast<uint>(nc.y));
                            UInt4 n_vis = gbuf_vis.read(ncoord);
                            // Only include same instance
                            $if(n_vis.x == c_vis.x) {
                                Float3 n_light = input_history.read(ncoord).xyz();
                                Float  n_lum = luminance(n_light);
                                sum_l = sum_l + n_light;
                                sum_l2 = sum_l2 + n_lum * n_lum;
                                sum_w = sum_w + 1.0f;
                            };
                        };
                    };
                };
                Float spatial_mu1 = luminance(sum_l) / sum_w;
                Float spatial_mu2 = sum_l2 / sum_w;
                var = luisa::compute::max(spatial_mu2 - spatial_mu1 * spatial_mu1, 0.0f);
            };

            // 3x3 Gaussian prefilter on variance (edge-stopped by depth + instance)
            // Gaussian weights: [1/16, 1/4, 3/8, 1/4, 1/16] in each axis
            // Simplified: use [1, 2, 1] / 4 in each axis for the 3x3
            Float sum_v = def(var * 4.0f);   // center weight = 4
            Float sum_w = def(4.0f);

            luisa::int2 gauss_offsets[8]{
                {-1,-1}, {-1,0}, {-1,1},
                { 0,-1},         { 0,1},
                { 1,-1}, { 1,0}, { 1,1}
            };
            // Gaussian 3x3 weights for 8 neighbors
            float gauss_w[8]{ 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 1.0f, 2.0f, 1.0f };
            Constant<luisa::int2> g_offsets(gauss_offsets, 8);
            Constant<float> g_weights(gauss_w, 8);

            $for(i, 8) {
                Int2 dd = g_offsets.read(i);
                Int nx = cast<Int>(coord.x) + dd.x;
                Int ny = cast<Int>(coord.y) + dd.y;
                $if(nx >= 0 & nx < cast<Int>(resolution.x)
                    & ny >= 0 & ny < cast<Int>(resolution.y)) {
                    UInt2 nc = make_uint2(cast<uint>(nx), cast<uint>(ny));
                    UInt4 n_vis = gbuf_vis.read(nc);
                    $if(n_vis.x == c_vis.x) {
                        Float n_mu1 = input_moments.read(nc).x;
                        Float n_mu2 = input_moments.read(nc).y;
                        Float n_var = luisa::compute::max(n_mu2 - n_mu1 * n_mu1, 0.0f);
                        Float w = g_weights.read(i);
                        sum_v = sum_v + w * n_var;
                        sum_w = sum_w + w;
                    };
                };
            };

            Float filtered_var = luisa::compute::min(sum_v / sum_w, 1e4f);
            $if(luisa::compute::isnan(filtered_var)) { filtered_var = 0.0f; };
            output_variance.write(coord, make_float4(filtered_var, 0.0f, 0.0f, 0.0f));
        });

        //==========================================================================
        // SVGF: Depth Gradient Precomputation — forward differences
        //==========================================================================
        _svgfDepthGradShader = device.compile<2>([&](
            ImageFloat output_grad, ImageFloat gbuf_depth
            ) noexcept {
            //set_block_size(16u, 16u, 1u);
            UInt2 coord = dispatch_id().xy();
            UInt2 resolution = dispatch_size().xy();
            /*
            $autodiff {
                Float zx = gbuf_depth.read(coord).x;
                Float zy = gbuf_depth.read(coord).x;

                requires_grad(zx, zy);
                Float z  = zx + zy;
                backward (z);

                output_grad.write(coord, make_float4(grad(zx), grad(zy), 0.0f, 0.0f));
            };
            */
            Float z = gbuf_depth.read(coord).x;
            // Forward differences (clamp to edge)
            UInt2 rc = make_uint2(luisa::compute::min(coord.x + 1u, resolution.x - 1u), coord.y);
            UInt2 uc = make_uint2(coord.x, luisa::compute::min(coord.y + 1u, resolution.y - 1u));
            Float zr = gbuf_depth.read(rc).x;
            Float zu = gbuf_depth.read(uc).x;

            Float ddx = zr - z;
            Float ddy = zu - z;
            output_grad.write(coord, make_float4(ddx, ddy, 0.0f, 0.0f));
        });

        //==========================================================================
        // SVGF: Atrous Wavelet — Falcor-style separable 5x5 with variance pre-blur
        //==========================================================================
        {
            // 5x5 kernel offsets (24 neighbors, excluding center)
            luisa::int2 ao[24]{
                {-2,-2},{-2,-1},{-2,0},{-2,1},{-2,2},
                {-1,-2},{-1,-1},{-1,0},{-1,1},{-1,2},
                { 0,-2},{ 0,-1},        { 0, 1},{ 0, 2},
                { 1,-2},{ 1,-1},{ 1,0},{ 1,1},{ 1,2},
                { 2,-2},{ 2,-1},{ 2,0},{ 2,1},{ 2,2}
            };
            // Falcor separable weights [1, 2/3, 1/6] per axis
            // kernel[i][j] = kernelWeight[|i|] * kernelWeight[|j|]
            float kw[3]{ 1.0f, 2.0f / 3.0f, 1.0f / 6.0f };
            float aw[24];
            for (int i = 0; i < 24; i++) {
                aw[i] = kw[std::abs(ao[i].x)] * kw[std::abs(ao[i].y)];
            }
            Constant<luisa::int2> atrous_offsets(ao, 24);
            Constant<float> atrous_weights(aw, 24);

            // Center weight = 1.0 (bypasses edge-stopping, per Falcor SVGF reference)
            float center_weight_data[1]{ 1.0f };
            Constant<float> center_weight(center_weight_data, 1);

            // 3x3 variance pre-blur offsets (8 neighbors + center)
            luisa::int2 vb[8]{
                {-1,-1},{-1,0},{-1,1},
                { 0,-1},        { 0,1},
                { 1,-1},{ 1,0},{ 1,1}
            };
            // Bilateral weights for 3x3: center=1/4, edge=1/8, corner=1/16
            float vw[8]{ 1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f,
                          1.0f / 8.0f,            1.0f / 8.0f,
                          1.0f / 16.0f, 1.0f / 8.0f, 1.0f / 16.0f };
            Constant<luisa::int2> var_blur_offsets(vb, 8);
            Constant<float> var_blur_weights(vw, 8);
            float var_blur_center[1]{ 1.0f / 4.0f };
            Constant<float> var_blur_center_w(var_blur_center, 1);

            _svgfAtrousShader = device.compile<2>([&](
                ImageFloat output_img,
                ImageFloat input_img,
                ImageFloat normal_img,
                ImageFloat gbuf_depth,
                ImageFloat depth_grad,
                ImageUInt  gbuf_vis,
                ImageFloat variance_img,
                UInt       stride
                ) noexcept {
                set_block_size(16u, 16u, 1u);
                UInt2 coord = dispatch_id().xy();
                UInt2 resolution = dispatch_size().xy();

                // Center pixel data
                Float4 c_data = input_img.read(coord);
                Float3 c_color = c_data.xyz();
                Float  c_var = c_data.w;
                Float3 c_normal = normal_img.read(coord).xyz();
                Float  c_depth = gbuf_depth.read(coord).x;
                Float2 c_grad = depth_grad.read(coord).xy();
                UInt   c_inst = gbuf_vis.read(coord).x;

                // Inline 3x3 variance pre-blur (Falcor computeVarianceCenter)
                // Blur variance from the dedicated variance texture before using it
                // for edge-stopping weights — gives more stable sigma estimation
                Float blurred_var = var_blur_center_w.read(0u) * variance_img.read(coord).x;
                Float vb_sum = var_blur_center_w.read(0u);
                $for(vi, 8) {
                    Int2 voff = var_blur_offsets.read(vi);
                    Int vnx = cast<Int>(coord.x) + voff.x;
                    Int vny = cast<Int>(coord.y) + voff.y;
                    $if(vnx >= 0 & vnx < cast<Int>(resolution.x)
                        & vny >= 0 & vny < cast<Int>(resolution.y)) {
                        UInt2 vnc = make_uint2(cast<uint>(vnx), cast<uint>(vny));
                        blurred_var = blurred_var + var_blur_weights.read(vi) * variance_img.read(vnc).x;
                        vb_sum = vb_sum + var_blur_weights.read(vi);
                    };
                };
                blurred_var = blurred_var / luisa::compute::max(vb_sum, 1e-6f);
                Float c_vimg = luisa::compute::max(blurred_var, c_var);

                // Center contribution
                Float cw = center_weight.read(0u);
                Float3 sum_color = cw * c_color;
                Float  sum_var = cw * cw * c_var;
                Float  sum_weight = cw;

                $for(i, 24) {
                    Int2 dd = atrous_offsets.read(i) * cast<Int>(stride);
                    Int nx = cast<Int>(coord.x) + dd.x;
                    Int ny = cast<Int>(coord.y) + dd.y;

                    $if(nx >= 0 & nx < cast<Int>(resolution.x)
                        & ny >= 0 & ny < cast<Int>(resolution.y)) {
                        UInt2 nc = make_uint2(cast<uint>(nx), cast<uint>(ny));

                        Float4 n_data = input_img.read(nc);
                        Float3 n_color = n_data.xyz();
                        Float  n_var = n_data.w;
                        Float3 n_normal = normal_img.read(nc).xyz();
                        Float  n_depth = gbuf_depth.read(nc).x;
                        UInt   n_inst = gbuf_vis.read(nc).x;

                        // Instance boundary reject
                        $if(n_inst == c_inst) {

                            // Normal weight: pow(max(dot, 0), sigma_n)
                            Float w_n = pow(luisa::compute::max(
                                luisa::compute::dot(c_normal, n_normal), 0.0f), _svgfSigmaN);

                            // Gradient-based depth weight
                            Float depth_diff = abs(c_depth - n_depth);
                            Float grad_scale = luisa::compute::max(
                                abs(c_grad.x * cast<float>(dd.x)) + abs(c_grad.y * cast<float>(dd.y)),
                                1e-8f);
                            Float w_z = exp(-depth_diff / (_svgfSigmaZ * grad_scale + 1e-6f));

                            // Variance-guided luminance weight (using pre-blurred variance)
                            Float l_diff = abs(luminance(c_color) - luminance(n_color));
                            Float sigma = luisa::compute::max(sqrt(luisa::compute::max(c_vimg, 0.0f)), 1e-6f);
                            Float w_l = exp(-l_diff / (_svgfSigmaL * sigma + 1e-6f));

                            Float kh = atrous_weights.read(i);
                            Float w = kh * w_n * w_z * w_l;

                            sum_color = sum_color + w * n_color;
                            sum_var = sum_var + (w * w) * n_var;
                            sum_weight = sum_weight + w;
                        };
                    };
                };

                Float inv_w = 1.0f / luisa::compute::max(sum_weight, 1e-6f);
                Float3 out_color = sum_color * inv_w;
                Float  out_var = luisa::compute::min(sum_var * inv_w * inv_w, 1e4f);

                // Clamp color to fp16-safe range (output goes to HALF4)
                out_color = clamp(out_color, -100.0f, 100.0f);

                // NaN guard: reset to center pixel if anything went wrong
                $if(luisa::compute::isnan(out_color.x) | luisa::compute::isnan(out_color.y)
                    | luisa::compute::isnan(out_color.z) | luisa::compute::isnan(out_var)) {
                    out_color = clamp(c_color, -100.0f, 100.0f);
                    out_var = luisa::compute::min(c_var, 1e4f);
                };

                output_img.write(coord, make_float4(out_color, out_var));
            });
        }

        //==========================================================================
        // SVGF: Albedo Remodulate — filtered lighting × albedo → display
        //==========================================================================
        _svgfRemodShader = device.compile<2>([&](
            ImageFloat output,
            ImageFloat input,
            ImageFloat albedo_img
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2  coord = dispatch_id().xy();
            Float3 lighting = input.read(coord).xyz();
            Float4 albedo_data = albedo_img.read(coord);
            Float3 albedo = albedo_data.xyz();
            Float  is_emissive = albedo_data.w; // 1.0 if surface has emission

            // Emissive surfaces: pass through (emission was not demodulated)
            // Non-emissive: multiply by albedo to reconstruct final color
            Float max_albedo = luisa::compute::max(albedo.x, luisa::compute::max(albedo.y, albedo.z));
            Float3 result = ite(is_emissive > 0.5f,
                lighting,
                ite(max_albedo < 0.01f, lighting, lighting * albedo));

            // Clamp to prevent extreme values from spreading through temporal accumulation
            result = clamp(result, 0.0f, 100.0f);

            $if(
                luisa::compute::isnan(result.x) |
                luisa::compute::isnan(result.y) |
                luisa::compute::isnan(result.z)) {
                output.write(coord, make_float4(albedo, 1.0f));
            }
            $else{
                output.write(coord, make_float4(result, 1.0f));
            };
        });

        //==========================================================================
        // SVGF: Feedback Blit — copy atrous result back to history (preserving age)
        //==========================================================================
        _svgfFeedbackBlit = device.compile<2>([&](
            ImageFloat history_img,    // FLOAT4, read+write in-place
            ImageFloat atrous_output   // HALF4
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2 coord = dispatch_id().xy();

            Float3 filtered = atrous_output.read(coord).xyz();
            Float4 hist = history_img.read(coord);

            // NaN guard: if filtered result is bad, keep existing history color
            $if(luisa::compute::isnan(filtered.x) | luisa::compute::isnan(filtered.y)
                | luisa::compute::isnan(filtered.z)) {
                filtered = hist.xyz();
            };

            // Replace color (.xyz) with filtered result, preserve age (.w)
            history_img.write(coord, make_float4(filtered, hist.w));
        });

        //==========================================================================
        // SVGF: History Pack — FLOAT4 history + HALF4 variance → HALF4 pingpong
        //==========================================================================
        _svgfHistoryPackShader = device.compile<2>([&](
            ImageFloat output,     // HALF4 pingpong (.xyz=color, .w=variance)
            ImageFloat history,    // FLOAT4 history
            ImageFloat variance    // HALF4 variance (.x)
            ) noexcept {
            set_block_size(16u, 16u, 1u);
            UInt2  coord = dispatch_id().xy();

            Float3 color = clamp(history.read(coord).xyz(), -100.0f, 100.0f);
            Float  var = luisa::compute::min(variance.read(coord).x, 1e4f);

            // NaN guard
            $if(luisa::compute::isnan(color.x) | luisa::compute::isnan(color.y)
                | luisa::compute::isnan(color.z) | luisa::compute::isnan(var)) {
                color = make_float3(0.0f);
                var = 0.0f;
            };

            output.write(coord, make_float4(color, var));
        });
	}
}