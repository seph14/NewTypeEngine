#include "newtype/render/PassDenoiser.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/util/Rng.h"
#include "newtype/render/Shading.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

void RelaxDenoiser::compileUtility(Device& device) {
	//==========================================================================
	// ReLAX Anti-firefly: RCRS 3x3 min/max luminance clamping
	//==========================================================================
	_relaxAntiFirefly = device.compile<2>([&](
		BufferVar<RelaxConstants> consts,
		ImageFloat out_Diff,
		ImageFloat in_Diff,
		ImageFloat gIn_Tiles,
		// Specular (4-6): pass-through
		ImageFloat out_Spec,
		ImageFloat in_Spec,
		ImageFloat gIn_Tiles2,
		ImageFloat gIn_Normal_Roughness
		) noexcept {
		set_block_size(8u, 8u, 1u);
		set_name("anti_firefly");

		UInt2 pixelPos = dispatch_id().xy();
		UInt2 resolution = dispatch_size().xy();
		auto c = consts.read(0u);

		$if(any(pixelPos >= resolution)) {
			$return();
		};

		// Tile-based early out (sky)
		UInt2 tilePos = pixelPos >> 4u;
		//Float isSky = gIn_Tiles.read(tilePos).x;
		$if(gIn_Tiles.read(tilePos).x != 0.0f) {
			out_Diff.write(pixelPos, in_Diff.read(pixelPos));
			$return();
		};

		// Read center
		Float4 centerDiff = in_Diff.read(pixelPos);
		Float centerLum = luminance(centerDiff.xyz());
		Float4 centerSpec = in_Spec.read(pixelPos);
		Float centerSpecLum = luminance(centerSpec.xyz());

		// Center material ID (packed in normal_roughness.w as floor(.w))
		Float centerMaterialID = luisa::compute::floor(gIn_Normal_Roughness.read(pixelPos).w);

		// Track min/max luminance and their colors in 3x3 neighborhood
		Float minLum = 1e30f;
		Float maxLum = -1e30f;
		Float3 minColor = def(make_float3(0.0f));
		Float3 maxColor = def(make_float3(0.0f));
		Float minSpecLum = 1e30f;
		Float maxSpecLum = -1e30f;
		Float3 minSpecColor = def(make_float3(0.0f));
		Float3 maxSpecColor = def(make_float3(0.0f));

		$for(dy, -1, 2) {
			$for(dx, -1, 2) {
				$if(dx == 0 & dy == 0) {
					$continue;
				};

				Int2 sp = make_int2(pixelPos) + make_int2(dx, dy);
				sp = clamp(sp, make_int2(0), make_int2(cast<int>(resolution.x) - 1, cast<int>(resolution.y) - 1));
				UInt2 sup = make_uint2(sp);

				Float4 sampleDiff = in_Diff.read(sup);
				Float sampleLum = luminance(sampleDiff.xyz());
				Float4 sampleSpec = in_Spec.read(sup);
				Float sampleSpecLum = luminance(sampleSpec.xyz());

				// Material-ID gate (NRD CompareMaterials): skip neighbors on
				// different materials to prevent firefly bleed across boundaries.
				Float sampleMaterialID = luisa::compute::floor(gIn_Normal_Roughness.read(sup).w);
				Bool sameMaterialDiff = luisa::compute::max(centerMaterialID, c.gDiffMinMaterial)
				                      == luisa::compute::max(sampleMaterialID, c.gDiffMinMaterial);
				Bool sameMaterialSpec = luisa::compute::max(centerMaterialID, c.gSpecMinMaterial)
				                      == luisa::compute::max(sampleMaterialID, c.gSpecMinMaterial);

				$if(sameMaterialDiff) {
					$if(sampleLum < minLum) {
						minLum = sampleLum;
						minColor = sampleDiff.xyz();
					};
					$if(sampleLum > maxLum) {
						maxLum = sampleLum;
						maxColor = sampleDiff.xyz();
					};
				};
				$if(sameMaterialSpec) {
					$if(sampleSpecLum < minSpecLum) {
						minSpecLum = sampleSpecLum;
						minSpecColor = sampleSpec.xyz();
					};
					$if(sampleSpecLum > maxSpecLum) {
						maxSpecLum = sampleSpecLum;
						maxSpecColor = sampleSpec.xyz();
					};
				};
			};
		};

		// RCRS: clamp center to neighborhood min/max range
		Float3 result = centerDiff.xyz();
		result = ite(centerLum > maxLum, maxColor, result);
		result = ite(centerLum < minLum, minColor, result);

		// Preserve 2nd moment
		out_Diff.write(pixelPos, make_float4(result, centerDiff.w));

		// Specular RCRS: clamp center to neighborhood min/max range.
		// Rejects isolated fireflies on dark conductor surfaces while preserving
		// legitimate metal highlights (which have similar-brightness neighbors).
		Float3 specResult = centerSpec.xyz();
		specResult = ite(centerSpecLum > maxSpecLum, maxSpecColor, specResult);
		specResult = ite(centerSpecLum < minSpecLum, minSpecColor, specResult);
		out_Spec.write(pixelPos, make_float4(specResult, centerSpec.w));
	});


	//==========================================================================
	// Utility shaders used by the denoiser internally
	//==========================================================================
	_clearImageShader = device.compile<2>([&](
		ImageFloat output,
		ImageFloat /*unused*/) noexcept {
		set_name("clear_image");
		UInt2 coord = dispatch_id().xy();
		output.write(coord, make_float4(0.0f));
	});

	_blitShader = device.compile<2>([&](
		ImageFloat out_frame,
		ImageFloat input) noexcept {
		set_name("blit");
		UInt2 coord = dispatch_id().xy();
		out_frame.write(coord, input.read(coord));
	});

	_compositeBlitShader = device.compile<2>([&](
		ImageFloat out_frame,
		ImageFloat denoised_diff,
		ImageFloat gbuf_depth,
		ImageFloat envmap_img,
		Var<util::CameraData> camera,
		UInt env_width,
		UInt env_height,
		BufferVar<float3x3> env_rotation_buffer,
		Float env_exp,
		ImageFloat albedo_img,
		ImageFloat spec_factor_img,
		ImageFloat denoised_spec,
		UInt solid_bg_enabled,
		Float3 solid_bg_color
		) noexcept {
		set_name("composite_blit");
		UInt2 coord = dispatch_id().xy();
		UInt2 resolution = dispatch_size().xy();
		Float depth = gbuf_depth.read(coord).x;
		$if(depth > 1e10f) {
			Float3 sky_color = def(make_float3(0.0f));
			$if(solid_bg_enabled != 0u) {
				sky_color = solid_bg_color;
			} $else {
				Float2 ndc = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera.jitter;
				auto ray = camera->generate_ray(ndc);
				Float3x3 env_rot = env_rotation_buffer.read(0u);
				sky_color = eval_envmap_radiance(normalize(ray->direction()), envmap_img,
					env_width, env_height, env_rot, env_exp);
			};
			out_frame.write(coord, make_float4(sky_color, 1.0f));
		} $else{
			Float4 diff = denoised_diff.read(coord);
			Float4 spec = denoised_spec.read(coord);
			Float4 albedo = albedo_img.read(coord);
			Bool isEmissive = albedo.w > 0.5f;
			Float3 remod_diff = ite(isEmissive, make_float3(1.0f), albedo.xyz());
			Float3 remod_spec = spec_factor_img.read(coord).xyz();
			Float3 denoised = diff.xyz() * remod_diff + spec.xyz() * remod_spec;
			Float3 combined = ite(isEmissive, albedo.xyz(), denoised);
			out_frame.write(coord, make_float4(combined, 1.0f));
		};
	});

#if NT_DEBUG_VIZ
	_debugTaBlitShader = device.compile<2>([&](
		ImageFloat out_frame,
		ImageFloat denoised_diff,
		ImageFloat gbuf_depth,
		ImageFloat envmap_img,
		Var<util::CameraData> camera,
		UInt env_width,
		UInt env_height,
		BufferVar<float3x3> env_rotation_buffer,
		Float env_exp,
		ImageFloat albedo_img,
		ImageFloat spec_factor_img,
		ImageFloat spec_img,
		UInt solid_bg_enabled,
		Float3 solid_bg_color
		) noexcept {
		set_name("debug_ta_blit");
		UInt2 coord = dispatch_id().xy();
		UInt2 resolution = dispatch_size().xy();
		Float3x3 env_rot = env_rotation_buffer.read(0u);
		Float depth = gbuf_depth.read(coord).x;
		Float3 diff = denoised_diff.read(coord).xyz();
		Float3 spec = spec_img.read(coord).xyz();
			Float4 albedo = albedo_img.read(coord);
			Float3 remod_diff = ite(albedo.w > 0.5f, make_float3(1.0f), albedo.xyz());
			Float3 remod_spec = spec_factor_img.read(coord).xyz();
		$if(depth > 1e10f) {
			Float3 sky_color = def(make_float3(0.0f));
			$if(solid_bg_enabled != 0u) {
				sky_color = solid_bg_color;
			} $else {
				Float2 ndc = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera.jitter;
				auto ray = camera->generate_ray(ndc);
				sky_color = eval_envmap_radiance(normalize(ray->direction()), envmap_img,
				env_width, env_height, env_rot, env_exp);
			};
			out_frame.write(coord, make_float4(sky_color, 1.0f));
		} $else{
			Float3 combined = diff * remod_diff + spec * remod_spec;
			out_frame.write(coord, make_float4(combined, 1.0f));
		};
	});
#endif // NT_DEBUG_VIZ
}

}