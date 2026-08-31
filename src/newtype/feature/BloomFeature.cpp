#include "newtype/feature/BloomFeature.h"
#include "newtype/core/FeatureContext.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Profiler.h"
#include "cinder/CinderImGui.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include <algorithm>

namespace newtype::feature {
    using namespace luisa;
    using namespace luisa::compute;

void BloomFeature::onInit(Device& device) {
    // === Bright pass: full-res input (FLOAT4) → 1/2-res mips[0] (HALF4) ===
    // 2x2 box downsample + Karis-style smoothstep threshold on Rec.709 luminance.
    auto bright_kernel = Kernel2D([&](
        ImageFloat output, ImageFloat input,
        Float threshold, Float knee
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 out_size = dispatch_size().xy();
        $if(coord.x >= out_size.x | coord.y >= out_size.y) { $return(); };

        UInt2 src_size = out_size * 2u;
        UInt2 max_xy = src_size - 1u;
        UInt2 src = coord * 2u;

        UInt2 p00 = min(src,                       max_xy);
        UInt2 p10 = min(src + make_uint2(1u, 0u),  max_xy);
        UInt2 p01 = min(src + make_uint2(0u, 1u),  max_xy);
        UInt2 p11 = min(src + make_uint2(1u, 1u),  max_xy);

        Float3 c00 = input.read(p00).xyz();
        Float3 c10 = input.read(p10).xyz();
        Float3 c01 = input.read(p01).xyz();
        Float3 c11 = input.read(p11).xyz();
        Float3 color = (c00 + c10 + c01 + c11) * 0.25f;

        Float lum = dot(color, make_float3(0.2126f, 0.7152f, 0.0722f));
        Float soft = smoothstep(threshold, threshold + knee, lum);

        Float3 bright = color * soft;
        // Clamp to HALF16 max for safety against extreme HDR fireflies.
        bright = min(bright, make_float3(65000.0f));
        output.write(coord, make_float4(bright, 1.0f));
    });
    _bright = device.compile(bright_kernel);

    // === Downsample: mips[i] → mips[i+1] (4x4 binomial, single-pass) ===
    // Weights = outer product of [1,3,3,1]/8, total /64. Each dst taps a 4x4 block in src
    // starting at (2*coord - 1), giving 2x anti-aliasing plus mild noise smoothing.
    auto down_kernel = Kernel2D([&](
        ImageFloat output, ImageFloat input
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 out_size = dispatch_size().xy();
        $if(coord.x >= out_size.x | coord.y >= out_size.y) { $return(); };

        UInt2 src_size = out_size * 2u;
        Int2 max_xy = make_int2(cast<int>(src_size.x) - 1, cast<int>(src_size.y) - 1);
        Int2 base = make_int2(cast<int>(coord.x) * 2 - 1, cast<int>(coord.y) * 2 - 1);

        Float3 sum = def(make_float3(0.0f));

        auto tap = [&](Int dx, Int dy, Float w) noexcept {
            Int2 p = make_int2(base.x + dx, base.y + dy);
            p = clamp(p, make_int2(0), max_xy);
            Float3 s = input.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).xyz();
            sum = sum + s * w;
        };

        // 4x4 binomial — row weights [1,3,3,1] × col weights [1,3,3,1] / 64
        tap(0, 0, 1.0f / 64.0f); tap(1, 0, 3.0f / 64.0f); tap(2, 0, 3.0f / 64.0f); tap(3, 0, 1.0f / 64.0f);
        tap(0, 1, 3.0f / 64.0f); tap(1, 1, 9.0f / 64.0f); tap(2, 1, 9.0f / 64.0f); tap(3, 1, 3.0f / 64.0f);
        tap(0, 2, 3.0f / 64.0f); tap(1, 2, 9.0f / 64.0f); tap(2, 2, 9.0f / 64.0f); tap(3, 2, 3.0f / 64.0f);
        tap(0, 3, 1.0f / 64.0f); tap(1, 3, 3.0f / 64.0f); tap(2, 3, 3.0f / 64.0f); tap(3, 3, 1.0f / 64.0f);

        output.write(coord, make_float4(sum, 1.0f));
    });
    _downsample = device.compile(down_kernel);

    // === Karis downsample: same 4x4 binomial taps but luminance-weighted ===
    // Used ONLY for mips[0] → mips[1] (first pass after bright). Weights each tap by
    // w / (1 + lum) so a single HDR firefly (e.g. direct-visible local light sample
    // that the denoiser didn't fully smooth) can't dominate the average. Subsequent
    // passes use plain `_downsample` — outliers have already been averaged out by then.
    auto down_karis_kernel = Kernel2D([&](
        ImageFloat output, ImageFloat input
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 out_size = dispatch_size().xy();
        $if(coord.x >= out_size.x | coord.y >= out_size.y) { $return(); };

        UInt2 src_size = out_size * 2u;
        Int2 max_xy = make_int2(cast<int>(src_size.x) - 1, cast<int>(src_size.y) - 1);
        Int2 base = make_int2(cast<int>(coord.x) * 2 - 1, cast<int>(coord.y) * 2 - 1);

        Float3 sum = def(make_float3(0.0f));
        Float weight_sum = def(0.0f);

        auto tap = [&](Int dx, Int dy, Float binomial_w) noexcept {
            Int2 p = make_int2(base.x + dx, base.y + dy);
            p = clamp(p, make_int2(0), max_xy);
            // Clamp input to non-negative — ReLAX denoiser can produce small negative values
            // around bright edges (anti-ringing overshoot). Without this, lum < 0 makes
            // 1+lum < 1, which inverts the Karis weight and produces negative bloom that
            // darkens the scene in the composite pass.
            Float3 s = max(input.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).xyz(),
                           make_float3(0.0f));
            Float lum = dot(s, make_float3(0.2126f, 0.7152f, 0.0722f));
            Float w = binomial_w / (1.0f + lum);
            sum = sum + s * w;
            weight_sum = weight_sum + w;
        };

        // Same 4x4 binomial base weights (un-normalized — Karis normalization handles it).
        tap(0, 0, 1.0f); tap(1, 0, 3.0f); tap(2, 0, 3.0f); tap(3, 0, 1.0f);
        tap(0, 1, 3.0f); tap(1, 1, 9.0f); tap(2, 1, 9.0f); tap(3, 1, 3.0f);
        tap(0, 2, 3.0f); tap(1, 2, 9.0f); tap(2, 2, 9.0f); tap(3, 2, 3.0f);
        tap(0, 3, 1.0f); tap(1, 3, 3.0f); tap(2, 3, 3.0f); tap(3, 3, 1.0f);

        Float3 result = sum / max(weight_sum, 1e-4f);
        output.write(coord, make_float4(result, 1.0f));
    });
    _downsampleKaris = device.compile(down_karis_kernel);

    // === Upsample: lerp(lower_existing, higher_upsampled, lerp_factor) ===
    // 3x3 binomial tent on higher mip (weights [1,2,1]² /16). In-place on `lower`/output.
    auto up_kernel = Kernel2D([&](
        ImageFloat output, ImageFloat lower, ImageFloat higher, Float lerp_factor
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 out_size = dispatch_size().xy();
        $if(coord.x >= out_size.x | coord.y >= out_size.y) { $return(); };

        // Higher mip size is ceil(out_size / 2). We compute hp = coord / 2 then tap 3x3 around it.
        UInt2 h_size = (out_size + 1u) / 2u;
        UInt2 h_max = h_size - 1u;
        Int2 hp = make_int2(cast<int>(coord.x) / 2, cast<int>(coord.y) / 2);

        Float3 sum = def(make_float3(0.0f));

        auto tap = [&](Int dx, Int dy, Float w) noexcept {
            Int2 p = make_int2(hp.x + dx, hp.y + dy);
            p = clamp(p, make_int2(0), make_int2(cast<int>(h_max.x), cast<int>(h_max.y)));
            Float3 s = higher.read(make_uint2(cast<uint>(p.x), cast<uint>(p.y))).xyz();
            sum = sum + s * w;
        };

        // 3x3 binomial — outer product of [1,2,1]/4, total /16
        tap(-1, -1, 1.0f / 16.0f); tap(0, -1, 2.0f / 16.0f); tap(1, -1, 1.0f / 16.0f);
        tap(-1,  0, 2.0f / 16.0f); tap(0,  0, 4.0f / 16.0f); tap(1,  0, 2.0f / 16.0f);
        tap(-1,  1, 1.0f / 16.0f); tap(0,  1, 2.0f / 16.0f); tap(1,  1, 1.0f / 16.0f);

        Float3 higher_up = sum;
        Float3 lower_existing = lower.read(coord).xyz();
        Float3 result = lerp(lower_existing, higher_up, lerp_factor);
        output.write(coord, make_float4(result, 1.0f));
    });
    _upsample = device.compile(up_kernel);

    // === Composite: renderTarget += mips[0] * strength (additive, in-place) ===
    // mips[0] is at 1/2-res, so we bilinearly upsample to full-res here. This is the
    // final tent-upsample step of the standard UnrealBloom pipeline folded into the
    // additive composite to save a dispatch.
    auto comp_kernel = Kernel2D([&](
        ImageFloat output, ImageFloat bloom, Float strength
    ) noexcept {
        set_block_size(16u, 16u, 1u);
        UInt2 coord = dispatch_id().xy();
        UInt2 out_size = dispatch_size().xy();
        $if(coord.x >= out_size.x | coord.y >= out_size.y) { $return(); };

        UInt2 bloom_size = out_size / 2u;
        Int2 bloom_max = make_int2(cast<int>(bloom_size.x) - 1, cast<int>(bloom_size.y) - 1);

        // Map full-res pixel to bloom-space (centered): full_coord / 2 - 0.5
        Float2 bf = (make_float2(coord) - 0.5f) * 0.5f;
        Int2 b0 = make_int2(cast<int>(floor(bf.x)), cast<int>(floor(bf.y)));
        Float2 f = luisa::compute::fract(bf);

        Int2 b00 = clamp(b0,                   make_int2(0), bloom_max);
        Int2 b10 = clamp(b0 + make_int2(1, 0), make_int2(0), bloom_max);
        Int2 b01 = clamp(b0 + make_int2(0, 1), make_int2(0), bloom_max);
        Int2 b11 = clamp(b0 + make_int2(1, 1), make_int2(0), bloom_max);

        Float3 s00 = bloom.read(make_uint2(b00)).xyz();
        Float3 s10 = bloom.read(make_uint2(b10)).xyz();
        Float3 s01 = bloom.read(make_uint2(b01)).xyz();
        Float3 s11 = bloom.read(make_uint2(b11)).xyz();

        Float3 a = lerp(s00, s10, f.x);
        Float3 b = lerp(s01, s11, f.x);
        Float3 bloom_color = lerp(a, b, f.y);

        Float4 scene = output.read(coord);
        Float3 result = scene.xyz() + bloom_color * strength;
        output.write(coord, make_float4(result, scene.w));
    });
    _composite = device.compile(comp_kernel);

    _compiled = true;
}

void BloomFeature::onResize(Device& device, uint width, uint height) {
    _fullW = width;
    _fullH = height;

    // Cap pyramid so smallest mip is at least 4 pixels — below that, the 4x4 / 3x3
    // kernels degenerate. Also clamp to user-requested mipLevels.
    uint min_dim = min(width, height);
    uint max_mips = 0u;
    while (min_dim >= 4u && max_mips < mipLevels) {
        min_dim >>= 1;
        max_mips++;
    }
    _effectiveMips = max(max_mips, 1u);

    _mips.clear();
    for (uint i = 0u; i < _effectiveMips; ++i) {
        uint w = max(1u, width >> (i + 1u));
        uint h = max(1u, height >> (i + 1u));
        _mips.push_back(device.create_image<float>(PixelStorage::HALF4, w, h));
    }
}

void BloomFeature::onExecute(Stream& stream, const core::FeatureContext& ctx) {
    if (!_compiled || !enabled()) return;
    if (_mips.empty()) return;

    auto& profiler = util::Profiler::instance(); 
    auto cl = CommandList::create();

    // Pass 1: Bright pass — full-res render target → mips[0] (1/2-res)
    float knee = std::max(threshold * 0.5f, 1e-3f);
    cl << _bright(_mips[0], ctx.renderTarget, threshold, knee)
                  .dispatch(_mips[0].size().x, _mips[0].size().y);

    // Pass 2: Downsample chain — first pass uses Karis luminance-weighted rejection
    // to kill PT fireflies on direct-visible lights; subsequent passes use plain binomial.
    for (uint i = 1u; i < _effectiveMips; ++i) {
        auto& dst = _mips[i];
        if (i == 1u) {
            cl << _downsampleKaris(dst, _mips[0u])
                  .dispatch(dst.size().x, dst.size().y);
        } else {
            cl << _downsample(dst, _mips[i - 1u])
                  .dispatch(dst.size().x, dst.size().y);
        }
    }

    // Pass 3: Upsample chain (mips[N-1] folds back into mips[0])
    float lerp_factor = std::clamp(radius, 0.0f, 1.0f);
    for (uint i = _effectiveMips - 1u; i >= 1u; --i) {
        auto& dst = _mips[i - 1u];
        cl << _upsample(dst, dst, _mips[i], lerp_factor)
              .dispatch(dst.size().x, dst.size().y);
    }

    // Pass 4: Composite — additive blend to render target
    profiler.set_pass("PostEffects/Bloom");
    cl << _composite(ctx.renderTarget, _mips[0], strength)
          .dispatch(ctx.width(), ctx.height());
    stream << cl.commit();
}

void BloomFeature::load(const nlohmann::json& j) {
    if (!j.is_object()) return;
    bool en = j.value("enabled", false);
    setEnabled(en);
    strength  = j.value("strength",  strength);
    radius    = j.value("radius",    radius);
    threshold = j.value("threshold", threshold);
    mipLevels = j.value("mips",      mipLevels);
}

nlohmann::json BloomFeature::toJson() const {
    nlohmann::json file;
    file["enabled"]   = enabled();
    file["strength"]  = strength;
    file["radius"]    = radius;
    file["threshold"] = threshold;
    file["mips"]      = mipLevels;
    return file;
}

void BloomFeature::drawUi() {
    if (ImGui::CollapsingHeader("Bloom")) {
        ImGui::ScopedId scpId("fBloom");
        bool en = enabled();
        if (ImGui::Checkbox("Enable", &en)) setEnabled(en);
        if (enabled()) {
            ImGui::SliderFloat("Strength",  &strength,  0.0f, 4.0f);
            ImGui::SliderFloat("Radius",    &radius,    0.0f, 1.0f);
            ImGui::SliderFloat("Threshold", &threshold, 0.0f, 16.0f);
            int mips = static_cast<int>(mipLevels);
            if (ImGui::SliderInt("Mips", &mips, 1, 8)) {
                mipLevels = static_cast<uint>(mips);
                // Re-trigger mip rebuild on next resize; or rebuild immediately if dimensions known.
                if (_fullW > 0u && _fullH > 0u) {
                    onResize(core::Renderer::device(), _fullW, _fullH);
                }
            }
        }
    }
}

} // namespace newtype::feature
