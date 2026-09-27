#include "newtype/feature/FxaaFeature.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Profiler.h"
#include "cinder/CinderImGui.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::feature {
    using namespace luisa;
    using namespace luisa::compute;

void FxaaFeature::onInit(Device& device) {
    // FXAA 3.11 console variant (proven compact port, e.g. three.js/geeks3d),
    // transliterated into the DSL. Works in pixel space; taps are manual
    // bilinear gathers (house idiom — images are read(), never sampled).
    auto kernel = Kernel2D([&](
        ImageFloat output, ImageFloat input,
        Float span_max, Float reduce_mul, Float reduce_min
        ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 res_u = dispatch_size().xy();
        $if(any(coord >= res_u)) { $return(); };

        Float2 res    = make_float2(cast<float>(res_u.x), cast<float>(res_u.y));
        Float2 inv_vp = 1.0f / res;
        UInt2  max_xy = res_u - 1u;

        // Manual bilinear tap at a continuous pixel-space position, clamped to
        // the image rect (border taps clamp instead of wrapping).
        auto tap = [&](Float2 pixel) -> Float4 {
            Float2 p = pixel - 0.5f;
            Float2 f = fract(p);
            Int2 i0 = make_int2(floor(p));
            Int2 i1 = i0 + 1;
            Int2 ib = make_int2(cast<int>(max_xy.x), cast<int>(max_xy.y));
            Int2 c00 = clamp(i0, make_int2(0), ib);
            Int2 c10 = clamp(i1, make_int2(0), ib);
            Int2 c01 = make_int2(c00.x, c10.y);
            Int2 c11 = make_int2(c10.x, c00.y);
            Float4 s00 = input.read(make_uint2(c00));
            Float4 s10 = input.read(make_uint2(c10));
            Float4 s01 = input.read(make_uint2(c01));
            Float4 s11 = input.read(make_uint2(c11));
            return lerp(lerp(s00, s10, f.x), lerp(s01, s11, f.x), f.y);
        };
        auto tap_uv = [&](Float2 uv) -> Float4 { return tap(uv * res); };

        Float3 luma_w = make_float3(0.299f, 0.587f, 0.114f);
        auto lum = [&](Float3 rgb) -> Float { return dot(rgb, luma_w); };

        Float2 pc = make_float2(coord) + 0.5f;
        Float2 uv = pc * inv_vp;

        Float4 rgbM4 = tap(pc);
        Float3 rgbM  = rgbM4.xyz();
        Float3 rgbNW = tap(pc + make_float2(-1.0f, -1.0f)).xyz();
        Float3 rgbNE = tap(pc + make_float2( 1.0f, -1.0f)).xyz();
        Float3 rgbSW = tap(pc + make_float2(-1.0f,  1.0f)).xyz();
        Float3 rgbSE = tap(pc + make_float2( 1.0f,  1.0f)).xyz();

        Float lumaNW = lum(rgbNW);
        Float lumaNE = lum(rgbNE);
        Float lumaSW = lum(rgbSW);
        Float lumaSE = lum(rgbSE);
        Float lumaM  = lum(rgbM);
        Float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
        Float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

        // Edge direction from the corner luma cross-differences (pixel space).
        Float dirx = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
        Float diry =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
        // Dir reduce: scale the threshold with the local corner contrast, floored
        // so flat regions never self-trigger.
        Float dir_reduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25f * reduce_mul), reduce_min);
        Float rcp_dir_min = 1.0f / (min(abs(dirx), abs(diry)) + dir_reduce);
        Float2 dir_px = min(make_float2(span_max), max(make_float2(-span_max),
                            make_float2(dirx, diry) * rcp_dir_min));
        Float2 dir_uv = dir_px * inv_vp;

        // Directional 3-then-2-tap blur along the edge; the wider 4-tap blend
        // is rejected when its luma escapes the input range (alias-guard).
        Float3 rgbA = 0.5f * (tap_uv(uv + dir_uv * (1.0f / 3.0f - 0.5f)).xyz()
                            + tap_uv(uv + dir_uv * (2.0f / 3.0f - 0.5f)).xyz());
        Float3 rgbB = rgbA * 0.5f + 0.25f * (tap_uv(uv + dir_uv * -0.5f).xyz()
                                           + tap_uv(uv + dir_uv *  0.5f).xyz());
        Float lumaB = lum(rgbB);
        Float3 out_rgb = ite((lumaB < lumaMin) | (lumaB > lumaMax), rgbA, rgbB);

        output.write(coord, make_float4(out_rgb, rgbM4.w));
    });

    _shader = device.compile(kernel);
    _compiled = true;
}

void FxaaFeature::onResize(Device& device, uint width, uint height) {
    _width = width;
    _height = height;
}

void FxaaFeature::onExecute(Stream& stream, const core::FeatureContext& ctx) {
    // Post-tonemap contract: only the LDR display target is a valid input.
    // In HDR float display mode the pipeline never tonemaps, so skip (running
    // FXAA on linear HDR would blur before the tonemapper curves it).
    if (!ctx.pipeline.lastFrameTonemapped()) return;
    // Upscaler guard + compiled/enabled in one predicate (must match
    // wantsTonemapTempRoute — see FxaaFeature.h).
    if (!willRun(ctx.pipeline)) return;
    uint w = ctx.width(), h = ctx.height();

    auto& temp = ctx.pipeline.requestTempImage(
        PixelStorage::BYTE4, w, h, "fxaa_temp");
    auto& profiler = util::Profiler::instance();

    if (!ctx.pipeline.tonemapRoutedToTemp()) {
        // No-route frames (feature toggles, HDR float display, additional
        // AfterToneMap features): tonemap wrote the display target, so the
        // temp needs the usual full-screen copy (BYTE4 matches display
        // storage for copy_to).
        stream << ctx.renderTarget.copy_to(temp);
    }
    // Routed frames (perf R2 item 13): tonemap already wrote "fxaa_temp" —
    // read it and write the display target directly, no copy.
    profiler.set_pass("PostFX/FXAA");
    stream << _shader(ctx.renderTarget, temp, spanMax, reduceMul, reduceMin)
              .dispatch(w, h);
}

void FxaaFeature::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    bool en = j.value("enabled", enabled());
    setEnabled(en);
    spanMax   = j.value("spanMax",   spanMax);
    reduceMul = j.value("reduceMul", reduceMul);
    reduceMin = j.value("reduceMin", reduceMin);
}

nlohmann::json FxaaFeature::toJson() const {
    nlohmann::json file;
    file["enabled"]  = enabled();
    file["spanMax"]  = spanMax;
    file["reduceMul"] = reduceMul;
    file["reduceMin"] = reduceMin;
    return file;
}

void FxaaFeature::drawUi() {
    if (ImGui::CollapsingHeader("FXAA (post-tonemap AA)")) {
        ImGui::ScopedId scpId("fFXAA");
        bool en = enabled();
        if (ImGui::Checkbox("Enable", &en)) setEnabled(en);
        if (enabled()) {
            ImGui::SliderFloat("Span Max", &spanMax, 1.0f, 16.0f, "%.1f px");
            ImGui::SliderFloat("Reduce Mul", &reduceMul, 0.0f, 0.5f, "%.4f",
                               ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Reduce Min", &reduceMin, 0.0f, 0.125f, "%.4f",
                               ImGuiSliderFlags_Logarithmic);
        }
    }
}

} // namespace newtype::feature
