//==============================================================================
// TetCageScene — tetrahedral-cage wind spike (Stage 2 validation scene)
// (src/tests/TetCageScene.cpp, docs/tetrahedral-cage-prototype.md)
//
// Renders N animated copies of a caged VAT mesh (default: ginkgo) through
// scene::TetCageGeometry — static shared per-tet piece BLASes, one TLAS
// instance per tet per copy. The wind is scene property here: two
// ShaderManager-registered shaders carry the analytic bend (evaluated at
// rest positions, per-copy phase 0.3 · copyIndex) and are handed over via
// set_deform_shader_id():
//   - "tetcage_cage_wind" animates the cage verts on the GPU; the TetSolve
//     kernel derives the per-tet affines on the device and the TLAS copies
//     the matrices during its build (no readback — watertight either way).
//   - with NT_ENABLE_PROCEDURAL, "tetcage_proc_wind" drives a type-3
//     ProceduralGeometry instance of the same mesh (full per-vertex wind,
//     same formula) at the identity transform — toggling "Cage copies"
//     against "Procedural reference" diffs the piecewise-linear cage
//     approximation against the exact deformation.
//
// Copy count: NT_TETCAGE_COPIES environment variable (default 6) — set before
// launch; instances are created at build time. NT_TETCAGE_ASSET (default
// "ginkgo/ginkgo0") picks the content under assets/models/ — any .tetcage +
// matching .vat pair (e.g. "northwall/NorthWall_t5" with NT_TETCAGE_VAT_TOPO=5
// for the packed V1 file). NT_TETCAGE_PROC_COPIES (default 1) adds that many
// type-3 procedural instances for the update-cost comparison (they render at
// the identity transform — the procedural path has no per-instance
// translation). NT_TETCAGE_CPU_PATH=1 forces the readback path (wind readback
// + CPU solve + per-instance setShapeTransform) — the A/B validation baseline
// against the default GPU solve path (wind + TetSolve kernels, matrices reach
// the TLAS from the device; see docs/TLAS-instance-transform-updates.md).
//
// Contents: ground plane, the copy row, one local area light. The "Engine"
// window carries the wind controls + cage stats (update time, instance/VRAM
// estimates vs the procedural path).
//==============================================================================

#include "cinder/Log.h"
#include "cinder/app/App.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"
#include "cinder/CinderImGui.h"

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/core/ShaderManager.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/LightShape.h"
#include "newtype/scene/TetCageGeometry.h"
#include "newtype/render/Material.h"
#include "newtype/util/Camera.h"
#include "newtype/util/TypeConv.h"

#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#include "newtype/scene/VATLoader.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <filesystem>

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;

namespace {

namespace fs = std::filesystem;

inline constexpr const char* kCageWindShader = "tetcage_cage_wind";
inline constexpr const char* kProcWindShader = "tetcage_proc_wind";
inline constexpr uint kCageWindBlock = 256u;  // cage-wind y-dim (vert stride)

//==============================================================================
// Wind shader registrations — the analytic bend moved out of the engine.
// They live in their own namespace with ONLY the LuisaCompute
// using-directives: the scene TU's cinder/glm usings would otherwise hijack
// the DSL math names (dot/cross/length/min/...).
//==============================================================================
namespace wind_shaders {
// ONLY luisa::compute here — the host luisa namespace carries conflicting
// math overloads (length, ...) for the DSL names used below.
using namespace luisa::compute;

/// Cage wind: x = copy (reads its state), y grid-strides the cage verts.
/// Signature contract: TetCageGeometry.h (set_deform_shader_id).
void register_cage(core::ShaderManager& sm, uint cageVerts) {
    if (sm.hasShader(kCageWindShader)) return;
    sm.registerShader<2>(kCageWindShader,
        [cageVerts](compute::BufferVar<luisa::float3> anim_cage,
                    compute::BufferVar<luisa::float3> rest_cage,
                    compute::BufferVar<scene::TetCageDeformState> states) noexcept {
            // Keep the DSL math names despite the TU's glm/luisa usings.
            using luisa::compute::sin;
            using luisa::compute::cast;
            using luisa::compute::make_float3;
            const UInt copy = dispatch_id().x;
            Var<scene::TetCageDeformState> st = states.read(copy);
            const Float t = st.params[0].x;
            const Float s = st.params[0].y;
            const Float f = st.params[0].z;
            const Float phase = cast<Float>(copy) * 0.3f;
            $for(v, dispatch_id().y, cageVerts, kCageWindBlock) {
                Float3 r = rest_cage.read(v);
                Float bend = r.y * s * sin(t * f + r.x * 0.5f + phase);
                anim_cage.write(copy * cageVerts + v,
                                make_float3(r.x + bend, r.y, r.z + bend * 0.3f));
            };
        });
}

#if NT_ENABLE_PROCEDURAL
/// Procedural type-3 wind — the engine's former built-in deform shader,
/// verbatim. One thread per instance (blockSize 1 keeps the AABB/normals
/// accumulation race-free); deform-only geometry ⇒ deform start and vat
/// count are 0. Signature contract: ProceduralGeometry.h
/// (set_deform_shader_id).
void register_proc(core::ShaderManager& sm) {
    if (sm.hasShader(kProcWindShader)) return;
    sm.registerShader<2>(kProcWindShader,
        [](compute::BufferVar<luisa::float4> positions,
           compute::BufferVar<luisa::float4> normals,
           compute::BufferVar<compute::AABB> aabbs,
           compute::BufferVar<scene::ProcInstanceData> instances,
           compute::BufferVar<scene::ProcDeformState> states,
           compute::BufferVar<luisa::float3> base_positions,
           compute::BufferVar<luisa::float3> base_normals,
           compute::BufferVar<uint> base_offsets,
           compute::BufferVar<compute::Triangle> indices,
           compute::BufferVar<luisa::float2> static_uvs,
           compute::BufferVar<scene::ProcMeshMeta> mesh_meta) noexcept {
            // Keep the DSL math names despite the TU's glm/luisa usings.
            using luisa::compute::sin;
            using luisa::compute::min;
            using luisa::compute::max;
            using luisa::compute::dot;
            using luisa::compute::cross;
            using luisa::compute::length;
            using luisa::compute::normalize;
            using luisa::compute::acos;
            using luisa::compute::clamp;
            using luisa::compute::cast;
            using luisa::compute::make_float3;
            using luisa::compute::make_float4;
            UInt idx = dispatch_id().x;
            Var<scene::ProcInstanceData> inst = instances.read(idx);

            Var<scene::ProcDeformState> state = states.read(idx);
            Float time_val = state.params[0].x;
            Float strength = state.params[0].y;
            Float freq = state.params[0].z;

            UInt base_offset = base_offsets.read(inst.mesh_id);
            Var<scene::ProcMeshMeta> meta = mesh_meta.read(inst.mesh_id);
            UInt frame_idx = cast<UInt>(inst.param);
            UInt pos_base = inst.packed_offsets & 0xFFFFu;
            UInt idx_base = inst.packed_offsets >> 16u;

            Float3 aabb_min = make_float3(1e10f);
            Float3 aabb_max = make_float3(-1e10f);

            // --- Pass 1: Deform vertices, pack uv_u into position.w ---
            $for(v, inst.vertex_count) {
                Float3 base_pos = base_positions.read(base_offset + v);
                Float bend = base_pos.y * strength *
                    sin(time_val * freq + base_pos.x * 0.5f + frame_idx * 0.3f);
                Float3 deformed = make_float3(
                    base_pos.x + bend,
                    base_pos.y,
                    base_pos.z + bend * 0.3f);
                Float2 uv = static_uvs.read(meta.uv_base + v);
                positions.write(pos_base + v, make_float4(deformed, uv.x));
                aabb_min = min(aabb_min, deformed);
                aabb_max = max(aabb_max, deformed);
            };

            // --- Pass 2: Zero normals for accumulation ---
            $for(v, inst.vertex_count) {
                normals.write(pos_base + v, make_float4(0.0f));
            };

            // --- Pass 3: Accumulate angle-weighted face normals ---
            $for(t, inst.tri_count) {
                auto tri = indices.read(idx_base + t);
                Float3 p0 = positions.read(pos_base + tri.i0).xyz();
                Float3 p1 = positions.read(pos_base + tri.i1).xyz();
                Float3 p2 = positions.read(pos_base + tri.i2).xyz();
                Float3 e1 = p1 - p0;
                Float3 e2 = p2 - p0;
                Float3 fn = cross(e1, e2);
                Float area = length(fn);
                $if(area > 1e-8f) {
                    Float3 n = fn / area;
                    // Angle at p0
                    Float cos_a = dot(normalize(e1), normalize(e2));
                    Float3 prev0 = normals.read(pos_base + tri.i0).xyz();
                    normals.write(pos_base + tri.i0,
                        make_float4(prev0 + n * acos(clamp(cos_a, -1.0f, 1.0f)), 0.0f));
                    // Angle at p1
                    Float3 e1b = p0 - p1;
                    Float3 e2b = p2 - p1;
                    Float cos_b = dot(normalize(e1b), normalize(e2b));
                    Float3 prev1 = normals.read(pos_base + tri.i1).xyz();
                    normals.write(pos_base + tri.i1,
                        make_float4(prev1 + n * acos(clamp(cos_b, -1.0f, 1.0f)), 0.0f));
                    // Angle at p2
                    Float3 e1c = p0 - p2;
                    Float3 e2c = p1 - p2;
                    Float cos_c = dot(normalize(e1c), normalize(e2c));
                    Float3 prev2 = normals.read(pos_base + tri.i2).xyz();
                    normals.write(pos_base + tri.i2,
                        make_float4(prev2 + n * acos(clamp(cos_c, -1.0f, 1.0f)), 0.0f));
                };
            };

            // --- Pass 4: Normalize + pack uv_v into normal.w ---
            $for(v, inst.vertex_count) {
                Float3 n = normals.read(pos_base + v).xyz();
                Float2 uv = static_uvs.read(meta.uv_base + v);
                normals.write(pos_base + v, make_float4(normalize(n), uv.y));
            };

            // params[0].w < 0 collapses the AABB: the instance stops
            // generating candidates (per-instance visibility toggle — a
            // procedural primitive is a single TLAS instance, so there is no
            // per-AABB visibility mask).
            Var<AABB> aabb;
            $if (state.params[0].w < 0.0f) {
                aabb.packed_min = make_float3(1e30f);
                aabb.packed_max = make_float3(-1e30f);
            } $else {
                aabb.packed_min = { aabb_min.x, aabb_min.y, aabb_min.z };
                aabb.packed_max = { aabb_max.x, aabb_max.y, aabb_max.z };
            };
            aabbs.write(idx, aabb);
        });
}
#endif
} // namespace wind_shaders

class TetCageScene : public nt::test::TestScene {
public:
    const char* name() const override { return "tetcage"; }

    void build(core::Pipeline& pipeline) override {
        auto& device = core::Renderer::device();
        auto& stream = core::Renderer::stream();
        _pipeline = &pipeline;

        auto envLong = [](const char* name, long lo, long hi, long dflt) -> long {
            if (const char* env = std::getenv(name)) {
                long v = std::strtol(env, nullptr, 10);
                if (v >= lo && v <= hi) return v;
                CI_LOG_W("--scene tetcage: ignoring out-of-range " << name << "=" << env);
            }
            return dflt;
        };
        std::string asset = "ginkgo/ginkgo0";
        if (const char* env = std::getenv("NT_TETCAGE_ASSET")) asset = env;

        const fs::path cagePath = app::getAssetPath("models/" + asset + ".tetcage");
        if (!fs::exists(cagePath)) {
            CI_LOG_E("--scene tetcage: " << (asset + ".tetcage") << " not found under "
                << cagePath.parent_path().string()
                << " - build it with tools/tet_cage/run.bat <source.(vat|obj)>");
            return;
        }

        //======================================================================
        // Materials
        //======================================================================
        auto leafMatIdx = pipeline.addMaterial("tetcage_leaf",
            render::make_diffuse(luisa::make_float3(0.22f, 0.42f, 0.15f)));
        auto groundMatIdx = pipeline.addMaterial("tetcage_ground",
            render::make_diffuse(luisa::make_float3(0.32f)));
        _litMatIdx = pipeline.addMaterial("tetcage_light",
            render::make_emissive(luisa::make_float3(90.f)));

        //======================================================================
        // Cage — shared static piece BLASes + one TLAS instance per tet/copy
        //======================================================================
        _cage = scene::TetCageGeometry::create(device);
        if (!_cage->load(cagePath)) {
            _cage.reset();
            return;
        }

        // Cage wind shader — the analytic bend, registered with the
        // ShaderManager and handed to the cage via set_deform_shader_id()
        // (signature contract: TetCageGeometry.h).
        {
            auto& sm = core::ShaderManager::instance();
            wind_shaders::register_cage(sm, _cage->cage_vertex_count());
            _cage->set_deform_shader_id(kCageWindShader, kCageWindBlock);
        }

        uint copyCount = 1024;// static_cast<uint>(envLong("NT_TETCAGE_COPIES", 1, 1024, 6));
        // NT_TETCAGE_CAGE_OFF=1: start hidden + settled (proc-only timing runs;
        // the TLAS still carries the instances — RT cores skip masked ones).
        // NT_TETCAGE_PROC_SHOW=1 / NT_TETCAGE_PAUSE=1: A/B diff captures —
        // frozen wind at t=0 is deterministic and identical in both paths.
        // NT_TETCAGE_DOUBLE_SIDED=0: single-sided meshes — the engine shades
        // double-sided instances with FLAT face normals (PassDI/Shading rule),
        // so smooth vertex normals only show when this is off (backsides then
        // shade dark/miss).
        const bool cageOff = std::getenv("NT_TETCAGE_CAGE_OFF") != nullptr;
        _doubleSided = envLong("NT_TETCAGE_DOUBLE_SIDED", 0, 1, 1) != 0;
        _cageVisible = !cageOff;
        _procVisible = std::getenv("NT_TETCAGE_PROC_SHOW") != nullptr;
        _playing = std::getenv("NT_TETCAGE_PAUSE") == nullptr;
        _freezeAfter = static_cast<float>(
            envLong("NT_TETCAGE_FREEZE_AFTER", 0, 600, 0));
        // NT_TETCAGE_BAKE_WIND=<sec>: seed the deformed pose through the
        // INITIAL instance upload (bake_initial_pose) and never write instance
        // transforms again — separates affine/serialization errors from
        // per-frame upload-churn artifacts.
        if (const char* bake = std::getenv("NT_TETCAGE_BAKE_WIND")) {
            _bakeAt = std::max(0.f, static_cast<float>(atof(bake)));
            _playing = false;
            _windTime = _bakeAt;
        }

        // Copy 0 at identity (pairs with the procedural reference); the row
        // spreads along +x from it. NT_TETCAGE_TILT rotates copy 0 about z
        // (normal-following diagnostic: shading must respond to the tilt).
        float tiltDeg = 0.f;
        if (const char* t = std::getenv("NT_TETCAGE_TILT"))
            tiltDeg = std::max(-180.f, std::min(180.f, static_cast<float>(atof(t))));
        const glm::mat4 tilt = glm::rotate(glm::mat4(1.f), glm::radians(tiltDeg), vec3(0.f, 0.f, 1.f));
        const luisa::float3 bmin = _cage->rest_min(), bmax = _cage->rest_max();
        const luisa::float3 ext = bmax - bmin;
        const float span = std::max(std::max(ext.x, ext.y), ext.z);
        const float spacing = span * 1.5f + 0.2f;
        for (uint i = 0u; i < copyCount; ++i)
            _cage->add_copy(tolc(glm::translate(vec3(
                static_cast<float>(i%100u) * spacing, 0.f,
                static_cast<float>(i/100u) * spacing)) *
                (i == 0u ? tilt : glm::mat4(1.f))));
        _rowExtent = static_cast<float>(100u - 1u) * spacing;
        float _colExtent = static_cast<float>(copyCount / 100u - 1u) * spacing;

        if (_bakeAt >= 0.f) {
            push_cage_states(_bakeAt, _strength);
            _cage->bake_initial_pose();
        }
        _cage->build(pipeline, stream, leafMatIdx, _doubleSided);
        if (cageOff) _cage->set_visible(pipeline, false);

        //======================================================================
        // Ground (top surface just under the cage copies' rest bounds)
        //======================================================================
        {
            auto groundMesh = scene::MeshShape::create(device, groundMatIdx);
            groundMesh->load_from(geom::Cube().size(vec3(
                _rowExtent + 4.f * span + 2.f, .1f, _colExtent + 4.f * span + 2.f)));
            groundMesh->build(stream);
            _groundTrans = scene::StaticTransform::create(
                tolc(glm::translate(vec3(_rowExtent * .5f, bmin.y - .07f,
                    _colExtent * .5f))));
            _groundId = pipeline.addShape(std::move(groundMesh), _groundTrans.get());
        }

        //======================================================================
        // Local area light above the row center
        //======================================================================
        {
            TriMesh lightMesh = ObjLoader(app::loadAsset("models/arealight.obj"));
            auto light = scene::make_light(device, lightMesh, _litMatIdx);
            _lightTrans = scene::StaticTransform::create(
                tolc(glm::translate(vec3(_rowExtent * .5f,
                                         bmin.y + 2.5f * span,
                                         bmin.z + 2.5f * span)) *
                     glm::scale(vec3(1.5f))));
            _lightId = pipeline.addLightShape(std::move(light), _lightTrans.get());
        }

        //======================================================================
        // Procedural reference (type-3) — same mesh, same wind, full
        // per-vertex deformation at the identity transform (= cage copy 0).
        // Hand-rolled here (the engine class no longer wires this): VAT load
        // with base-name fallback, static mesh, N deformable instances, the
        // "tetcage_proc_wind" deform shader, build, pipeline attach.
        // The engine flag is off by default; the controls hide in that build.
        //======================================================================
#if NT_ENABLE_PROCEDURAL
        {
            const uint topoIdx = static_cast<uint>(envLong("NT_TETCAGE_VAT_TOPO", 0, 64, 0));
            const uint procCopies = static_cast<uint>(envLong("NT_TETCAGE_PROC_COPIES", 0, 256, 1));
            // Derive from the (existing) cage path — getAssetPath returns EMPTY
            // for missing files, which would defeat the base-name fallback.
            fs::path vatPath = cagePath;
            vatPath.replace_extension(".vat");
            // Per-topology assets exist as <base>_tN.<ext>, but packed
            // multi-topology VATs live at the base name (NorthWall_t5.vat →
            // packed NorthWall.vat). Resolve to the base name when the
            // literal path is absent.
            if (!fs::exists(vatPath)) {
                std::string stem = vatPath.stem().string();
                const auto cut = stem.rfind('_');
                if (cut != std::string::npos) {
                    fs::path packed = vatPath.parent_path() /
                        (stem.substr(0, cut) + vatPath.extension().string());
                    if (fs::exists(packed)) vatPath = packed;
                }
            }

            if (procCopies == 0u) {
                CI_LOG_I("--scene tetcage: no procedural reference requested (copies = 0)");
            } else if (!fs::exists(vatPath)) {
                CI_LOG_E("--scene tetcage: cannot load reference VAT "
                    << vatPath.string());
            } else {
                // Procedural wind shader (signature contract:
                // ProceduralGeometry.h), then the reference geometry itself.
                auto& sm = core::ShaderManager::instance();
                wind_shaders::register_proc(sm);

                auto vats = scene::VATLoader::load_all(vatPath, /*skip_texcoords=*/false);
                if (vats.empty()) {
                    CI_LOG_E("--scene tetcage: cannot load reference VAT "
                        << vatPath.string());
                } else {
                    const uint topo = std::min(topoIdx,
                        static_cast<uint>(vats.size()) - 1u);
                    auto& vat = vats[topo];

                    auto proc = scene::ProceduralGeometry::create(device);
                    const uint triCount = vat.index_count / 3u;
                    luisa::vector<compute::Triangle> tris(triCount);
                    for (uint t = 0u; t < triCount; ++t)
                        tris[t] = compute::Triangle{
                            vat.indices[t * 3u], vat.indices[t * 3u + 1u], vat.indices[t * 3u + 2u]};
                    luisa::vector<luisa::float2> uvs;
                    if (!vat.texcoords.empty()) {
                        uvs.resize(vat.vertex_count);
                        for (uint32_t v = 0u; v < vat.vertex_count; ++v)
                            uvs[v] = luisa::make_float2(
                                vat.texcoords[v * 2u], vat.texcoords[v * 2u + 1u]);
                    }
                    const uint meshId = proc->add_static_mesh(
                        luisa::span<const luisa::float3>{vat.positions.data(), vat.vertex_count},
                        luisa::span<const luisa::float3>{vat.normals.data(), vat.vertex_count},
                        luisa::span<const compute::Triangle>{tris.data(), tris.size()},
                        vat.vertex_count,
                        uvs.empty() ? luisa::span<const luisa::float2>{} :
                                      luisa::span<const luisa::float2>{uvs.data(), uvs.size()});

                    proc->set_deform_shader_id(kProcWindShader, 1u);
                    _refFirstInstance = proc->add_deformable_instances(meshId, procCopies,
                        leafMatIdx | (0xFFu << 8u) | (0xFFu << 16u) | (0xFFu << 24u),
                        {}, _doubleSided);
                    _procCopies = procCopies;
                    _procVertexCount = vat.vertex_count;
                    _procTriCount = triCount;
                    proc->build(stream);
                    pipeline.setProceduralGeometry(std::move(proc));
                    _refProc = pipeline.proceduralGeom();
                    CI_LOG_I("--scene tetcage: " << procCopies
                        << " procedural type-3 reference instance(s) attached from "
                        << vatPath.string() << " (topo " << topo << ": "
                        << vat.vertex_count << " verts, " << triCount << " tris)");
                }
            }
        }
#endif
    }

    // Runs before Pipeline::update() — cage matrices must be queued first so
    // the geometry update flushes + refits the TLAS in the same frame.
    void update(float, float dt, core::Pipeline& pipeline) override {
        if (_playing) _windTime += dt * _windSpeed;

        // NT_TETCAGE_FREEZE_AFTER: snap to the exact freeze time, write that
        // pose once (both paths), then stop updating entirely — no further
        // instance uploads. Separates affine-continuity errors (visible in
        // the frozen pose) from upload-churn artifacts.
        if (_freezeAfter > 0.f && _windTime >= _freezeAfter) {
            if (_frozen) { settle_cage_once(pipeline); return; }
            _windTime = _freezeAfter;
            _frozen = true;
        }

        // Baked mode: the deformed pose lives in the initial instance
        // matrices; never write cage transforms again. The procedural
        // reference gets one deform-state write so it shows the same pose.
        if (_bakeAt >= 0.f) {
            if (!_bakeProcSet) {
                _bakeProcSet = true;
                push_proc_states(_bakeAt, /*visible=*/true);
            }
            return;
        }

        if (_cage) {
            if (_cageVisible && _strength != 0.f) {
                _cageSettled = false;
                _cageSettlePending = true;
                push_cage_states(_windTime, _strength);
                _cage->update(pipeline);
            } else if (!_cageSettled) {
                // Wind off / hidden: one solve at strength 0 settles every
                // copy to its rest matrices, then the cage idles.
                _cageSettled = true;
                _cageSettlePending = true;
                push_cage_states(0.f, 0.f);
                _cage->update(pipeline);
            } else {
                // First idle frame after the last animated write: converge
                // prev ← curr on the GPU path (the CPU path's stop-frame fix
                // is Geometry-internal) so motion vectors return to zero.
                settle_cage_once(pipeline);
            }
        }
        // Same analytic wind for the reference: params[0] = (time, strength,
        // freq, visibility) — a negative w collapses the instance AABBs.
        push_proc_states(_windTime, _procVisible);
    }

    void settle_cage_once(core::Pipeline& pipeline) noexcept {
        if (!_cageSettlePending) return;
        _cageSettlePending = false;
        if (_cage) _cage->settle(pipeline);
    }

    void drawUi() override {
        if (ImGui::CollapsingHeader("Tet Cage Wind", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Playing", &_playing);
            ImGui::SameLine();
            ImGui::SliderFloat("Speed", &_windSpeed, 0.f, 4.f, "%.2f");
            ImGui::SliderFloat("Wind time", &_windTime, 0.f, 120.f, "%.1f s");
            ImGui::SliderFloat("Strength", &_strength, 0.f, 2.f, "%.2f");
            ImGui::SliderFloat("Frequency", &_freq, 0.f, 6.f, "%.2f");

            bool cageVis = _cageVisible;
            if (ImGui::Checkbox("Cage copies", &cageVis)) {
                _cageVisible = cageVis;
                if (_cage && _pipeline) _cage->set_visible(*_pipeline, cageVis);
            }
#if NT_ENABLE_PROCEDURAL
            if (_procCopies > 0u) {
                ImGui::Checkbox("Procedural reference (type-3)", &_procVisible);
                ImGui::TextDisabled("%u instance(s), %u verts / %u tris each — copy 0 sits",
                    _procCopies, _procVertexCount, _procTriCount);
                ImGui::TextDisabled("exactly on reference #0: toggle the boxes for the diff");
            }
            ImGui::TextDisabled("double-sided shading uses flat face normals (engine rule);");
            ImGui::TextDisabled("NT_TETCAGE_DOUBLE_SIDED=0 shows smooth normals but dark backsides");
#endif
            if (_cage) {
                ImGui::Text("cage: %u tets / %u tris, %u copies = %u instances",
                    _cage->tet_count(), _cage->piece_triangle_count(),
                    _cage->copy_count(), _cage->instance_count());
                ImGui::Text("update: %.3f ms/frame (%s)", _cage->last_update_ms(),
                    _cage->gpu_path_active()
                        ? "GPU: wind + solve dispatch, no readback"
                        : "CPU: wind dispatch + readback + solve + flush; "
                          "NT_TETCAGE_CPU_PATH=1 forces this");

                // Per-frame animated-state VRAM: cage = instance descriptors
                // (transform 64B + prev 64B + props 16B); procedural = one
                // active pos+norm vertex pair (2x16B) per instance plus the
                // shared base pair. Static soup/BLASes excluded from both.
                const double cageMb = _cage->instance_count() * 144.0 / (1024.0 * 1024.0);
                ImGui::Text("animated-state est: cage %.2f MB", cageMb);
#if NT_ENABLE_PROCEDURAL
                if (_procVertexCount > 0u) {
                    const double procMb = (2.0 * _procCopies * _procVertexCount + 2.0 * _procVertexCount) * 16.0 / (1024.0 * 1024.0);
                    ImGui::Text("animated-state est: proc  %.2f MB (%u copy[ies])",
                        procMb, _procCopies);
                }
#endif
            }
        }
    }

    void applyCamera(newtype::util::Camera& camera) override {
        if (!_cage) return;
        // Frame copy 0 (the procedural-diff target) large in frame; the rest
        // of the row recedes to the right.
        const luisa::float3 bmin = _cage->rest_min(), bmax = _cage->rest_max();
        const luisa::float3 ctr = (bmin + bmax) * 0.5f;
        const luisa::float3 ext = bmax - bmin;
        const float span = std::max(std::max(ext.x, ext.y), ext.z);
        camera.ciCam().lookAt(
            vec3(ctr.x + span * .25f, ctr.y + span * .55f, ctr.z + span * 2.1f),
            vec3(ctr.x, ctr.y, ctr.z));
    }

private:
    // TestScene has no pipeline handle in drawUi(); build() runs on the same
    // app object that drives update()/drawUi(), so stashing the pointer for
    // visibility toggles is safe for this scene's lifetime.
    core::Pipeline* _pipeline = nullptr;

    scene::TetCagePtr         _cage;
    scene::StaticTransPtr     _groundTrans, _lightTrans;   // polled by the pipeline — keep alive
    scene::ShapeId            _groundId, _lightId;
    uint                      _litMatIdx = 0u;
    float                     _rowExtent = 0.f;

    // Wind state — single source shared by the cage and procedural paths.
    float _windTime  = 0.f;
    float _windSpeed = 1.f;
    float _strength  = 0.35f;
    float _freq      = 1.5f;
    float _freezeAfter = 0.f;   // NT_TETCAGE_FREEZE_AFTER: snap wind time and
    bool  _frozen      = false; // stop updating (upload-churn diagnosis; 0=off)
    float _bakeAt      = -1.f;  // NT_TETCAGE_BAKE_WIND: <0 = off
    bool  _bakeProcSet = false; // one-shot deform-state write when baked
    bool  _playing   = true;
    bool  _cageVisible  = true;
    bool  _cageSettled  = true;   // rest-flush done — update() idles until wind returns
    bool  _cageSettlePending = false; // one prev←curr convergence after the last animated write
    bool  _procVisible  = false;
    bool  _doubleSided  = true;   // NT_TETCAGE_DOUBLE_SIDED=0 → smooth normals

    // Procedural reference stats (UI) + per-frame state push (the pipeline
    // owns the ProceduralGeometry object; we keep the raw handle).
    uint _procCopies = 0u;
    uint _procVertexCount = 0u;
    uint _procTriCount = 0u;
#if NT_ENABLE_PROCEDURAL
    scene::ProceduralGeometry* _refProc = nullptr;
    uint _refFirstInstance = 0u;
#endif

    void push_cage_states(float time, float strength) noexcept {
        scene::TetCageDeformState st{};
        st.params[0] = luisa::make_float4(time, strength, _freq, 0.f);
        _cage->set_deform_state_all(st);
    }

#if NT_ENABLE_PROCEDURAL
    void push_proc_states(float time, bool visible) noexcept {
        if (!_refProc || _procCopies == 0u) return;
        scene::ProcDeformState st{};
        st.params[0] = luisa::make_float4(time, _strength, _freq, visible ? 0.f : -1.f);
        for (uint i = 0u; i < _procCopies; ++i)
            _refProc->set_deform_state(_refFirstInstance + i, st);
    }
#else
    void push_proc_states(float, bool) noexcept {}
#endif
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createTetCageScene() { return std::make_unique<TetCageScene>(); }
} // namespace newtype::test
