#include "newtype/render/PassDenoiser.h"
#include "newtype/core/FrameContext.h"
#include "newtype/core/Renderer.h"
#include "newtype/render/Shading.h"
#include "newtype/core/Config.h"
#if NT_ENABLE_PROCEDURAL
#include "newtype/render/ProceduralTrace.h"
#endif

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

namespace newtype::core {
using namespace luisa;
using namespace luisa::compute;
using namespace newtype::scene;
using namespace newtype::render;

void RelaxDenoiser::compilePrefilterAndClassifyTiles(Device& device, const SurfaceResolverPoly& resolver){
	//==========================================================================
	// Denoise PreFilter: G-Buffer -> clean albedo + world normal
	//==========================================================================
	_denoisePreFilterShader = device.compile<2>([&](
		ImageFloat				albedo_output,
		ImageFloat				spec_factor_output,
		ImageFloat				normal_output,
		ImageFloat				gbuf_depth,
		ImageUInt				gbuf_vis,
		ImageFloat				gbuf_bary_motion,
		Var<util::CameraData>	camera,
		BufferVar<luisa::uint4> instance_buffer,
		BufferVar<luisa::float4x4> instance_transform_buffer,
		BufferVar<MaterialData> material_buffer,
		BindlessVar				vertex_bindless,
		BindlessVar				tex_bindless
#if NT_ENABLE_PROCEDURAL
		, BindlessVar proc_bindless
	#endif
		) noexcept {
		set_block_size(16u, 16u, 1u);
		set_name("denoise_pre_filter");
		UInt2 coord = dispatch_id().xy();
		UInt2 resolution = dispatch_size().xy();

		// Guard against partial-block threads
		$if(any(coord >= resolution)) { $return(); };

		//Float  depth = gbuf_depth.read(coord).x;
		UInt4  vis = gbuf_vis.read(coord);
		UInt   inst_id = vis.x;
		UInt   prim_id = vis.y & 0x3FFFFFFFu;
		Bool   isGlassPixel = (vis.y >> 31u) != 0u;
		Bool   is_point = ((vis.y >> 30u) & 1u) > 0u;
		Bool   is_procedural = ((vis.y >> 29u) & 1u) > 0u;

		Float3 albedo = def(luisa::make_float3(0.0f));
		Float3 normal = def(luisa::make_float3(0.0f, 0.0f, 1.0f));
		Float  has_emission = def(0.0f);
		Float  metallic_val = def(0.0f);
		Float  matIdVal = def(255.0f);
		Float  roughness_val = def(0.0f);  // sky sentinel: matID=255, roughness=0
		Float3 spec_factor = def(luisa::make_float3(1.0f));

		$if(inst_id != ~0u) {
			$if(is_point) {
				// Point cloud: read material directly, decode oct-normal from bary_motion.zw
				Var<MaterialData> pmat = material_buffer.read(Expr{ inst_id });
				Float2 oct_n = gbuf_bary_motion.read(coord).zw();
				normal = render::oct_decode(oct_n);
				albedo = pmat.albedo.xyz();
				metallic_val = pmat.metallic;
				roughness_val = pmat.roughness;
				UInt4 pc_inst = instance_buffer.read(inst_id);
				matIdVal = cast<Float>(pc_inst.y & 0xFFu);
				// Match the shade pass point demod: NRD factors from oct normal + view ray
				Float2 ndc_pt = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
				auto ray_pt = camera->generate_ray(Expr{ ndc_pt });
				Float3 wo_pt = -normalize(ray_pt->direction());
				Float3 rf0_pt = lerp(make_float3(0.04f), pmat.albedo.xyz(), pmat.metallic);
				nrd_material_factors(normal, wo_pt, pmat.albedo.xyz(), rf0_pt,
					pmat.roughness, albedo, spec_factor);
			}
#if NT_ENABLE_PROCEDURAL
			$elif(is_procedural) {
				// Procedural: resolve surface through callable pipeline, reconstruct normal
				Float  depth_val = gbuf_depth.read(coord).x;
				Float2 ndc_dn = (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f;
				auto   ray_dn = camera->generate_ray(ndc_dn);
				Float3 hit_pos = ray_dn->origin() + ray_dn->direction() * depth_val;
				Float3 wo_dn = -normalize(ray_dn->direction());

				// Read jit bary from G-Buffer, then recompute at unjittered pixel
				// center for stable texture sampling (same rationale as mesh branch).
				Float2 bary_dn = gbuf_bary_motion.read(coord).xy();
				auto   ray_dn_unjit = camera->generate_ray(Expr{ ndc_dn - camera->jitter });
				$if(!isGlassPixel) {
					bary_dn = reconstruct_unjittered_bary_procedural(
						proc_bindless, inst_id, prim_id, bary_dn,
						ray_dn_unjit->origin(), ray_dn_unjit->direction());
				};

				SurfaceData surface = resolve_procedural_surface_textured(
					resolver, proc_bindless, tex_bindless,
					inst_id, prim_id, hit_pos, wo_dn,
					material_buffer, bary_dn, 0.0f,
					Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) },
					resolution.x, resolution.y);

				albedo = surface.albedo;
				metallic_val = surface.metallic;
				roughness_val = surface.roughness;
				Float3 rf0_pn = lerp(make_float3(0.04f), surface.albedo, surface.metallic);
				Float3 diff_factor_pn = def(make_float3(1.0f));
				nrd_material_factors(surface.ns, wo_dn, surface.albedo, rf0_pn,
					surface.roughness, diff_factor_pn, spec_factor);

				Var<scene::ProcInstanceData> proc_inst = proc_bindless.buffer<scene::ProcInstanceData>(render::kSlot_ProcInstances).read(inst_id);
				matIdVal = cast<Float>(proc_inst.material_layers & 0xFFu);

				// Unlit materials must be flagged as emissive so the composition pass
				// bypasses the denoised result (matches shade's albedo.w=1.0 encoding).
				// Without this, inactive checkerboard pixels retain the prefilter's
				// albedo.w=0 from this pass and get treated as diffuse, producing
				// albedo*albedo instead of albedo on alternating frames → flicker.
				Bool is_unlit = surface.bsdf_type == 12u;
				Float em_lum = luminance(surface.emission);
				has_emission = ite(is_unlit | em_lum > 0.01f, 1.0f, 0.0f);
				albedo = ite(is_unlit, surface.albedo,
					ite(em_lum > 0.01f, surface.emission, diff_factor_pn));

				normal = surface.ns;
			}
#endif
			$else {
				Float2 bary = gbuf_bary_motion.read(coord).xy();
				UInt4 inst_data = instance_buffer.read(inst_id);
				matIdVal = cast<Float>(inst_data.y & 0xFFu);

				auto   ray = camera->generate_ray(Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f });
				Float3 wo = -normalize(ray->direction());

				// Recompute bary at unjittered pixel center for stable albedo
				// (NRD/ReLAX assumes albedo is jitter-free for demod/remod).
				auto   ray_unjit = camera->generate_ray(Expr{
					(make_float2(coord) + 0.5f) / make_float2(resolution) * 2.0f - 1.0f - camera->jitter
				});
				// Single fused triangle fetch shared by the unjittered-bary
				// recompute and the surface resolve below.
				MeshTriVerts tri_verts = read_mesh_triangle(
					vertex_bindless, inst_data.z, inst_data.w, prim_id);
				$if(!isGlassPixel) {
					bary = reconstruct_unjittered_bary(
						tri_verts, bary,
						ray_unjit->origin(), ray_unjit->direction());
				};

				SurfaceData surface = resolve_surface_from_instance_verts(
					resolver, vertex_bindless, tex_bindless,
					tri_verts, inst_data, prim_id, bary,
					material_buffer, wo,
					instance_transform_buffer.read(inst_id),
					0.0f, Expr{ (make_float2(coord) + 0.5f) / make_float2(resolution) },
					resolution.x, resolution.y);

				normal = surface.ns;

				// NRD MaterialFactors (unjittered surface — see Shading.h).
				// Stored rgb replaces raw albedo as the diffuse remod factor.
				Float3 rf0_m = lerp(make_float3(0.04f), surface.albedo, surface.metallic);
				Float3 diff_factor_m = def(make_float3(1.0f));
				nrd_material_factors(surface.ns, wo, surface.albedo, rf0_m,
					surface.roughness, diff_factor_m, spec_factor);

				// Unlit materials must be flagged as emissive so the composition pass
				// bypasses the denoised result (matches shade's albedo.w=1.0 encoding).
				// Without this, inactive checkerboard pixels retain the prefilter's
				// albedo.w=0 from this pass and get treated as diffuse, producing
				// albedo*albedo instead of albedo on alternating frames → flicker.
				Bool is_unlit = surface.bsdf_type == 12u;
				Float em_lum = luminance(surface.emission);
				has_emission = ite(is_unlit | em_lum > 0.01f, 1.0f, 0.0f);
				albedo = ite(is_unlit, surface.albedo,
					ite(em_lum > 0.01f, surface.emission, diff_factor_m));
				metallic_val = surface.metallic;
				roughness_val = surface.roughness;
			};
		} $else{
			has_emission = 1.0f;
		};

			Float packed_w = ite(has_emission > 0.5f, 1.0f, metallic_val * 0.49f);
			albedo_output.write(coord, make_float4(albedo, packed_w));
			spec_factor_output.write(coord, make_float4(spec_factor, 0.0f));
		// Pack matID (integer 0-255) + roughness (fractional 0-1) into .w.
		// Decode downstream: matID = floor(w), roughness = fract(w).
		Float packed_nr_w = ite(inst_id != ~0u, matIdVal + roughness_val, 255.0f);
		normal_output.write(coord, make_float4(normal, packed_nr_w));
	});

	//==========================================================================
	// ReLAX Step 3: ClassifyTiles
	//==========================================================================
	_relaxClassifyTiles = device.compile<2>([&](
		ImageFloat tile_output,
		ImageFloat gIn_ViewZ,
		BufferVar<RelaxConstants> consts
		) noexcept {
		set_block_size(8u, 4u, 1u);
		set_name("classify_tiles");

		UInt2 threadPos = thread_id().xy();
		UInt2 tilePos = block_id().xy();
		UInt  threadIdx = thread_id().x + thread_id().y * 8u;

		Shared<uint> s_isSky(1u);

		$if(threadIdx == 0u) {
			s_isSky.write(0u, 0u);
		};
		sync_block();

		UInt2 pixelPos = tilePos * 16u + make_uint2(threadPos.x * 2u, threadPos.y * 4u);
		UInt  isSky = def(0u);

		Float denoisingRange = consts.read(0u).gDenoisingRange;

		$for(i, 2u) {
			$for(j, 4u) {
				//Expr<UInt2> pos = pixelPos + make_uint2(cast<uint>(i), cast<uint>(j));
				Float viewZ = luisa::compute::abs(
					gIn_ViewZ.read(Expr{ pixelPos + make_uint2(cast<uint>(i), cast<uint>(j)) }).x);
				isSky += ite(viewZ > denoisingRange, 1u, 0u);
			};
		};

		s_isSky.atomic(0u).fetch_add(isSky);
		sync_block();

		$if(threadIdx == 0u) {
			Float result = ite(s_isSky.read(0u) == 256u, 1.0f, 0.0f);
			tile_output.write(tilePos, make_float4(result, 0.0f, 0.0f, 0.0f));
		};
	});
}
};