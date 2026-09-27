//==============================================================================
// TransformTreeTest — scene-graph hierarchy verification scene
// (src/tests/TransformTreeTest.cpp)
//
// Exercises the transform tree end to end:
//   _pivot (animated root, rotates)
//     ├── _arm (animated child, fixed local offset)
//     │     ├── _sat  (STATIC borrowed transform under animated ancestors —
//     │     │          verifies chain-aware registration: parent-only motion
//     │     │          must still re-flush this instance)
//     │     └── _moon (animated child, O(1) set_local_position animation)
//     └── _protoInst (prototype instance added with a borrowed transform,
//                    parented into the tree through addPrototypeInstance)
//
// Also periodically drives a WORLD-space setter on a parented transform
// (set_position back-solves through the parent inverse) and logs a one-shot
// dirty-propagation sanity check on the first frame.
//
// GPU transform rows (docs/TLAS-instance-transform-updates.md Part 2/4): a
// ring of NT_XFORM_GPU_COUNT (default 48) prototype cubes orbits above the
// rig, animated entirely by a device kernel that writes the engine
// instance-transform buffer (curr+prev); the TLAS copies the registered
// rows on the device. Shares the TLAS with the CPU-animated rig — the
// mixed-scene guard coverage (per-index uploads/prev snapshots must skip
// the GPU rows). NT_XFORM_CPU_PATH=1 runs the identical animation through
// CPU setShapeTransform (A/B baseline); NT_XFORM_FREEZE_AFTER=<sec> snaps
// to an exact pose and freezes (pixel-diff captures).
//
// NT_XFORM_DEFORM_CHURN=<n> (default off): post-build add/remove cycling of
// two animated DeformableMeshes — every n frames the oldest is removeShape'd
// and a fresh one added. Exercises Geometry's deformable-instance registry
// (per-frame poll list) through the swap-and-pop reindex, including the
// registry entry remap when the swapped tail instance is the other deformable
// (the two churn meshes occupy the highest instance indices). Also bumps the
// topology generation every cycle, driving the GPU ring's re-registration
// watch. Cycles are appended to xform_selfcheck.log.
//
// Per-instance materials (B1): the prototype paths above carry per-instance
// material coverage — the ring cubes take a gray/red/gold palette at birth
// (one BLAS, three materials), the tree-parented prototype instance overrides
// its prototype's gold with red at birth, an emissive prototype pair places
// one inheriting instance beside one warm-tint override (LightSampler reads
// per-instance layers), and every 3 s ring instance 0 rotates through the
// palette via setShapeMaterial (instance-row re-upload + light rebuild).
// A first-frame self-check verifies the rows took; churn cycles log the
// surviving mesh's layers to prove swap-and-pop preserves them.
//==============================================================================

#include "cinder/app/App.h"
#include "cinder/Log.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"
#include "cinder/CinderImGui.h"
#include <fstream>

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/DeformableMesh.h"
#include "newtype/scene/LightShape.h"
#include "newtype/render/Material.h"
#include "newtype/render/MetalData.h"
#include "newtype/util/TypeConv.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <span>

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;
using namespace ci::app;

namespace {

// Ring animation constants — used verbatim by both the GPU kernel and the
// CPU A/B path so they produce the same matrices (op-for-op mirrors; only
// libm-vs-GPU transcendental ulps may differ).
constexpr float kRingRadius  = 1.5f;   // orbit radius
constexpr float kRingCenterY = 1.1f;   // orbit height above the rig
constexpr float kRingScale   = 0.11f;  // cube half-extent scale
constexpr float kRingOrbitW  = 0.5f;   // orbital angular velocity
constexpr float kRingBobAmp  = 0.35f;  // vertical bob amplitude
constexpr float kRingBobW    = 1.3f;   // vertical bob frequency

// Single-material layer pack (layer 0 only, layers 1-3 unused) — the same
// packing MeshShape's constructor uses.
constexpr uint32_t layer0(uint mat) noexcept { return 0xFFFFFF00u | (mat & 0xFFu); }

// CPU mirror of the GPU ring kernel's matrix (column-major, same op order).
luisa::float4x4 ring_matrix_cpu(uint i, uint count, float time) noexcept {
    const float angle = static_cast<float>(i) * (6.2831853f / static_cast<float>(count)) + time * kRingOrbitW;
    const float bob = kRingBobAmp * std::sin(time * kRingBobW + static_cast<float>(i) * 0.7f);
    const float spin = time * (1.0f + 0.25f * std::cos(static_cast<float>(i) * 1.9f));
    const float cs = std::cos(spin), ss = std::sin(spin);
    const float s = kRingScale;
    luisa::float4x4 m;
    m.cols[0] = luisa::make_float4(cs * s, 0.f, -ss * s, 0.f);
    m.cols[1] = luisa::make_float4(0.f, s, 0.f, 0.f);
    m.cols[2] = luisa::make_float4(ss * s, 0.f, cs * s, 0.f);
    m.cols[3] = luisa::make_float4(std::cos(angle) * kRingRadius,
                                   kRingCenterY + bob,
                                   std::sin(angle) * kRingRadius, 1.f);
    return m;
}

class TransformTreeScene : public nt::test::TestScene {
public:
    const char* name() const override { return "xform"; }

    void build(core::Pipeline& pipeline) override {
        auto& device = core::Renderer::device();
        auto& stream = core::Renderer::stream();

        //======================================================================
        // Materials
        //======================================================================
        auto grayMatIdx  = pipeline.addMaterial("xform_gray",
            render::make_diffuse(luisa::make_float3(.6f)));
        auto redMatIdx   = pipeline.addMaterial("xform_red",
            render::make_diffuse(luisa::make_float3(.8f, .15f, .1f)));
        auto goldMatIdx  = pipeline.addMaterial("xform_gold",
            render::make_conductor_metal(render::MetalPreset::Gold));
        auto cyanMatIdx  = pipeline.addMaterial("xform_light",
            render::make_emissive(luisa::make_float3(10.f)));
        auto warmMatIdx  = pipeline.addMaterial("xform_light_warm",
            render::make_emissive(luisa::make_float3(9.f, 2.2f, 0.35f)));

        // Stashed for update()-time use (build()'s locals are gone by then).
        _matGray = grayMatIdx;  _matRed = redMatIdx;  _matGold = goldMatIdx;
        _matCyan = cyanMatIdx;  _matWarm = warmMatIdx;

        //======================================================================
        // Floor + ceiling light (static, world-space)
        //======================================================================
        {
            scene::StaticTransform floorTrans(tolc(
                glm::translate(vec3(0.f, -1.9f, 0.f))));

            auto floor = scene::MeshShape::create(device, grayMatIdx);
            floor->load_from(geom::Cube().size(vec3(6.f, .1f, 6.f)));
            floor->build(stream);
            (void)pipeline.addShape(std::move(floor), &floorTrans);

            TriMesh lightMesh = ObjLoader(app::loadAsset("models/arealight.obj"));
            auto light = scene::make_light(device, lightMesh, cyanMatIdx);
            auto lightTrans = scene::StaticTransform::create(tolc(
                glm::translate(vec3(0.f, 2.2f, 0.f)) * glm::scale(vec3(1.2f))));
            (void)pipeline.addLightShape(std::move(light), lightTrans.get());
        }

        //======================================================================
        // Hierarchy rig
        //======================================================================
        _pivot = scene::AnimatedTransform::create(tolc(glm::translate(vec3(0.f, 0.4f, 0.f))));
        _arm   = scene::AnimatedTransform::create(tolc(glm::translate(vec3(0.55f, 0.f, 0.f))));
        _sat   = scene::StaticTransform::create(tolc(glm::translate(vec3(0.45f, 0.35f, 0.f))));
        _moon  = scene::AnimatedTransform::create(tolc(glm::translate(vec3(0.f, -0.35f, 0.f))));
        _arm->set_parent(_pivot.get());
        _sat->set_parent(_arm.get());   // static leaf under animated ancestors
        _moon->set_parent(_arm.get());

        // Pivot hub
        {
            auto hub = scene::MeshShape::create(device, goldMatIdx);
            hub->load_from(geom::Sphere().radius(0.12f));
            hub->build(stream);
            (void)pipeline.addShape(std::move(hub), _pivot.get());
        }
        // Arm bar
        {
            auto bar = scene::MeshShape::create(device, grayMatIdx);
            bar->load_from(geom::Cube().size(vec3(1.1f, 0.06f, 0.06f)));
            bar->build(stream);
            (void)pipeline.addShape(std::move(bar), _arm.get());
        }
        // Satellite: STATIC borrowed transform — moves only because its
        // ancestors do. If registration were not chain-aware, this instance
        // would freeze at its add-time matrix.
        {
            auto sat = scene::MeshShape::create(device, goldMatIdx);
            sat->load_from(geom::Sphere().radius(0.09f));
            sat->build(stream);
            (void)pipeline.addShape(std::move(sat), _sat.get());
        }
        // Moon: animated locally with the O(1) set_local_* path
        {
            auto moon = scene::MeshShape::create(device, redMatIdx);
            moon->load_from(ObjLoader(app::loadAsset("models/obj1.obj")));
            moon->build(stream);
            (void)pipeline.addShape(std::move(moon), _moon.get());
        }

        //======================================================================
        // Prototype instance parented into the tree
        // B1: born with its own material (red) — the gold prototype and any
        // sibling instances are unaffected.
        //======================================================================
        {
            auto protoMesh = scene::MeshShape::create(device, goldMatIdx);
            protoMesh->load_from(geom::Sphere().radius(0.07f));
            protoMesh->build(stream);
            _proto = pipeline.addPrototype(std::move(protoMesh));

            _protoInst = scene::AnimatedTransform::create(tolc(
                glm::translate(vec3(0.9f, 0.7f, 0.f))));
            _protoInst->set_parent(_pivot.get());
            _protoInstId = pipeline.addPrototypeInstance(_proto, _protoInst.get(),
                                                         layer0(redMatIdx));
        }

        //======================================================================
        // B1: emissive prototype pair — the left instance inherits the
        // prototype's white emission, the right one overrides to warm at
        // birth. Both must glow with their OWN tint and light the scene
        // (LightSampler reads per-instance layers).
        //======================================================================
        {
            auto emisMesh = scene::MeshShape::create(device, cyanMatIdx);
            emisMesh->load_from(geom::Sphere().radius(0.09f));
            emisMesh->build(stream);
            auto emisProto = pipeline.addPrototype(std::move(emisMesh));

            const float s = 2.2f;
            auto place = [=](float x) {
                return tolc(glm::translate(vec3(x, 0.28f, -0.35f)) * glm::scale(vec3(s)));
            };
            _emisA = pipeline.addPrototypeInstance(emisProto, place(-1.95f));
            _emisB = pipeline.addPrototypeInstance(emisProto, place(-1.05f),
                                                   layer0(warmMatIdx));
        }

        //======================================================================
        // GPU transform rows — orbiting cube ring animated by a device kernel
        // (docs/TLAS-instance-transform-updates.md). NT_XFORM_GPU_COUNT=0
        // disables; NT_XFORM_CPU_PATH=1 animates the same ring through CPU
        // setShapeTransform (A/B baseline).
        //======================================================================
        if (const char* env = std::getenv("NT_XFORM_FREEZE_AFTER"))
            _ringFreezeAfter = std::max(0.f, static_cast<float>(atof(env)));
        // Deformable churn (see the file header): two DeformableMeshes cycled
        // post-build every n frames. Material index stashed for update()-time
        // adds (build()'s locals are gone by then).
        _churnMat = redMatIdx;
        if (const char* env = std::getenv("NT_XFORM_DEFORM_CHURN")) {
            long n = std::strtol(env, nullptr, 10);
            if (n < 5 || n > 10000) {
                CI_LOG_W("--scene xform: ignoring out-of-range NT_XFORM_DEFORM_CHURN=" << env);
            } else {
                _churnEvery = static_cast<uint32_t>(n);
            }
        }
        long ringCount = 48;
        if (const char* env = std::getenv("NT_XFORM_GPU_COUNT")) {
            ringCount = std::strtol(env, nullptr, 10);
            if (ringCount < 0 || ringCount > 4096) {
                CI_LOG_W("--scene xform: ignoring out-of-range NT_XFORM_GPU_COUNT=" << env);
                ringCount = 48;
            }
        }
        if (ringCount > 0) {
            _ringCount = static_cast<uint>(ringCount);
            _ringCpuPath = std::getenv("NT_XFORM_CPU_PATH") != nullptr;

            auto ringMesh = scene::MeshShape::create(device, goldMatIdx);
            ringMesh->load_from(geom::Cube().size(vec3(1.f)));
            ringMesh->build(stream);
            auto ringProto = pipeline.addPrototype(std::move(ringMesh));

            const luisa::float4x4 identity = luisa::make_float4x4(1.f);
            // B1: per-instance palette — one shared BLAS, three materials
            // (gray/red/gold by index).
            const uint32_t palette[3] = {
                layer0(grayMatIdx), layer0(redMatIdx), layer0(goldMatIdx) };
            _ringIds.reserve(_ringCount);
            for (uint i = 0u; i < _ringCount; ++i)
                _ringIds.push_back(pipeline.addPrototypeInstance(
                    ringProto, identity, palette[i % 3u]));

            // B2: author per-instance custom data on two ring instances
            // (readable shader-side via instance_params); verified by the
            // one-shot update() self-check below.
            if (_ringCount >= 2u) {
                pipeline.setInstanceUserData(_ringIds[0], 0u,
                    luisa::float4(0.125f, 0.25f, 0.5f, 1.f));
                pipeline.setInstanceUserData(_ringIds[1], 3u,
                    luisa::float4(7.f, 8.f, 9.f, 10.f));
            }

            if (!_ringCpuPath) {
                const float invCount = 6.2831853f / static_cast<float>(_ringCount);
                _ringShader = device.compile<1>(
                    [invCount](UInt first_row, Float time,
                               BufferVar<luisa::float4x4> transforms,
                               BufferVar<luisa::float4x4> transforms_prev) noexcept {
                        // Keep the DSL math names despite the TU's glm usings.
                        using luisa::compute::sin;
                        using luisa::compute::cos;
                        using luisa::compute::cast;
                        using luisa::compute::make_float3;
                        using luisa::compute::make_float4;
                        using luisa::compute::make_float4x4;
                        const UInt i = dispatch_id().x;
                        const Float angle = cast<Float>(i) * invCount + time * kRingOrbitW;
                        const Float bob = kRingBobAmp * sin(time * kRingBobW + cast<Float>(i) * 0.7f);
                        const Float spin = time * (1.0f + 0.25f * cos(cast<Float>(i) * 1.9f));
                        const Float cs = cos(spin);
                        const Float ss = sin(spin);
                        const Float s = kRingScale;
                        const Float3 col3 = make_float3(
                            cos(angle) * kRingRadius,
                            kRingCenterY + bob,
                            sin(angle) * kRingRadius);
                        const UInt row = first_row + i;
                        // prev ← old curr (motion vectors), then the new pose —
                        // the row belongs to this kernel until re-registration.
                        transforms_prev.write(row, transforms.read(row));
                        transforms.write(row, make_float4x4(
                            make_float4(cs * s, 0.0f, -ss * s, 0.0f),
                            make_float4(0.0f, s, 0.0f, 0.0f),
                            make_float4(ss * s, 0.0f, cs * s, 0.0f),
                            make_float4(col3, 1.0f)));
                    });

                if (!register_ring_rows(pipeline)) {
                    _ringCpuPath = true; // rows not contiguous — CPU fallback
                }
            }

            // Headless-friendly registration report (see the self-check log
            // at the top of update()).
            if (std::ofstream out{"xform_selfcheck.log", std::ios::app}) {
                out << (_ringIds.empty() ? "SKIP" :
                        _ringCpuPath ? "CPU " : "GPU ")
                    << "ring rows=" << _ringIds.size() << std::endl;
            }
            CI_LOG_I("TransformTreeTest: GPU transform ring - "
                << _ringCount << " instances, "
                << (_ringCpuPath ? "CPU setShapeTransform path"
                                 : "GPU kernel path (device-written TLAS rows)"));
        }
    }

    void update(float time, float dt, core::Pipeline& pipeline) override {
        if (!_pivot) return;

        // Deformable churn first: its post-build add/remove bumps the
        // topology generation the ring watch below consumes this same frame.
        update_churn(pipeline, time);

        // Freeze mode (NT_XFORM_FREEZE_AFTER): drive the rig from the ring's
        // snapped clock too, so the WHOLE scene freezes at an exact,
        // run-independent pose for A/B pixel diffs. Normal runs keep wall time.
        update_ring(pipeline, dt);
        const float t = _ringFreezeAfter > 0.f ? _ringTime : time;

        // Mutate ONLY the root: every descendant instance (including the
        // static satellite and the prototype instance) must re-flush.
        _pivot->set_local_rotation(glm::angleAxis(t * 0.6f, vec3(0.f, 1.f, 0.f)));

        // O(1) local animation on a child
        _moon->set_local_position(vec3(0.f, -0.35f + 0.12f * glm::sin(t * 2.f), 0.f));

        // Local animation on the parented prototype instance
        _protoInst->set_local_position(vec3(
            0.9f * glm::cos(t * 0.8f), 0.25f * glm::sin(t * 1.3f),
            0.9f * glm::sin(t * 0.8f)));

        // B1: every 3 s, rotate ring instance 0 through the palette via the
        // runtime setter — exercises the per-instance row re-upload and the
        // light-table dirty path (first tick re-sets its birth value; the
        // set is a no-op then).
        if (!_ringIds.empty()) {
            int mslot = static_cast<int>(t / 3.f) % 3;
            if (mslot != _ringMatSlot) {
                _ringMatSlot = mslot;
                const uint32_t pal[3] = {
                    layer0(_matGray), layer0(_matRed), layer0(_matGold) };
                pipeline.setShapeMaterial(_ringIds.front(),
                                          pal[static_cast<uint>(mslot)]);
            }
        }

        // Every 3 s, exercise a WORLD-space setter on a parented transform:
        // back-solves local through the (rotating) parent inverse, so the
        // satellite snaps to a fixed world point, then rides its parents
        // again as they keep rotating.
        int slot = static_cast<int>(t / 3.f) % 2;
        if (slot != _satSlot) {
            _satSlot = slot;
            _sat->set_position(vec3(slot ? 1.2f : -1.2f, slot ? 1.5f : 0.9f, 0.f));
        }

        // One-shot sanity checks (run before Pipeline::update consumes the
        // dirty flags): parent-only mutation must dirty the whole subtree,
        // and a parented transform's world matrix must differ from local.
        // The result is also written to xform_selfcheck.log next to the exe
        // so the check can be verified without a console (GUI subsystem).
        static bool checked = false;
        if (!checked) {
            checked = true;
            const bool sat_dirty = _sat->is_dirty();
            const bool proto_dirty = _protoInst->is_dirty();
            // Translation column: pivot+arm offsets guarantee world != local.
            const bool world_differs =
                any(_sat->matrix().cols[3] != _sat->local_matrix().cols[3]);
            const bool ok = sat_dirty && proto_dirty && world_differs;
            if (ok) {
                CI_LOG_I("TransformTreeTest: parent->child dirty propagation OK, "
                         "world != local for parented transform OK");
            } else {
                CI_LOG_E("TransformTreeTest FAILED: sat_dirty=" << sat_dirty
                    << " proto_dirty=" << proto_dirty
                    << " world_differs=" << world_differs);
            }
            if (std::ofstream out{"xform_selfcheck.log", std::ios::app}) {
                out << (ok ? "PASS" : "FAIL")
                    << " sat_dirty=" << sat_dirty
                    << " proto_dirty=" << proto_dirty
                    << " world_differs=" << world_differs << std::endl;
            }
        }

        // One-shot B1 self-check: the per-instance material rows took at
        // birth and are readable back through Geometry.
        static bool matChecked = false;
        if (!matChecked) {
            matChecked = true;
            const scene::Geometry *geom = pipeline.geometry();
            const uint32_t red = layer0(_matRed);
            bool ok = _protoInstId != scene::kInvalidShapeId
                && geom->material_layers(_protoInstId) == red
                && _emisA != scene::kInvalidShapeId
                && geom->material_layers(_emisA) == layer0(_matCyan)
                && _emisB != scene::kInvalidShapeId
                && geom->material_layers(_emisB) == layer0(_matWarm);
            if (_ringIds.size() >= 2u) {
                ok = ok && geom->material_layers(_ringIds[0]) == layer0(_matGray)
                          && geom->material_layers(_ringIds[1]) == red
                          && geom->material_layers(_ringIds[2]) == layer0(_matGold);
            }
            if (ok) {
                CI_LOG_I("TransformTreeTest: per-instance material rows OK "
                    "(protoInst=red, ring palette, emissive pair inherit/override)");
            } else {
                CI_LOG_E("TransformTreeTest FAILED: per-instance material rows - "
                    "protoInst=0x" << std::hex << geom->material_layers(_protoInstId)
                    << " emisA=0x" << geom->material_layers(_emisA)
                    << " emisB=0x" << geom->material_layers(_emisB) << std::dec);
            }
            if (std::ofstream out{"xform_selfcheck.log", std::ios::app}) {
                out << (ok ? "PASS" : "FAIL") << " per_instance_materials"
                    << " proto=0x" << std::hex << geom->material_layers(_protoInstId)
                    << " emisA=0x" << geom->material_layers(_emisA)
                    << " emisB=0x" << geom->material_layers(_emisB) << std::dec
                    << std::endl;
            }
        }

        // One-shot B2 self-check: authored per-instance user-params rows read
        // back exactly, untouched rows/slots read zeros, and invalid ids are
        // rejected (getter zero, setter false).
        static bool paramChecked = false;
        if (!paramChecked) {
            paramChecked = true;
            scene::Geometry *geom = pipeline.geometry();
            const luisa::float4 a = geom->instance_user_param(_ringIds[0], 0u);
            const luisa::float4 b = geom->instance_user_param(_ringIds[1], 3u);
            const luisa::float4 untouched = geom->instance_user_param(_ringIds[1], 0u);
            const luisa::float4 invalid = geom->instance_user_param(
                scene::kInvalidShapeId, 0u);
            const bool rejected = !geom->set_instance_user_param(
                scene::kInvalidShapeId, 0u, luisa::float4(1.f));
            bool ok = _ringCount >= 2u
                && all(a == luisa::float4(0.125f, 0.25f, 0.5f, 1.f))
                && all(b == luisa::float4(7.f, 8.f, 9.f, 10.f))
                && all(untouched == luisa::float4(0.f))
                && all(invalid == luisa::float4(0.f))
                && rejected;
            if (ok) {
                CI_LOG_I("TransformTreeTest: per-instance user params OK "
                    "(authored rows read back, untouched slots zero, invalid id rejected)");
            } else {
                CI_LOG_E("TransformTreeTest FAILED: per-instance user params - "
                    "a=(" << a.x << "," << a.y << "," << a.z << "," << a.w << ") "
                    "b=(" << b.x << "," << b.y << "," << b.z << "," << b.w << ") "
                    "untouched=(" << untouched.x << "," << untouched.y << ","
                    << untouched.z << "," << untouched.w << ") rejected=" << rejected);
            }
            if (std::ofstream out{"xform_selfcheck.log", std::ios::app}) {
                out << (ok ? "PASS" : "FAIL") << " per_instance_user_data"
                    << " ring0=(" << a.x << "," << a.y << "," << a.z << "," << a.w << ")"
                    << " ring1s3=(" << b.x << "," << b.y << "," << b.z << "," << b.w << ")"
                    << std::endl;
            }
        }

        // One-shot A1 check: release static CPU mirrors once the scene is
        // built and checked. By construction the churn deformables (seeded
        // above), the light, and the emissive prototype (per-instance warm
        // override) keep their CPU data; everything static releases. Later
        // churn cycles + material swaps must still run clean.
        static bool cpuUnloaded = false;
        if (!cpuUnloaded) {
            cpuUnloaded = true;
            pipeline.unloadStaticMeshCPUData();
        }
    }

    void drawUi() override {
        if (_ringCount == 0u) return;
        if (!ImGui::CollapsingHeader("GPU Transform Rows")) return;
        ImGui::Text("%s: %u rows%s, %.3f ms/frame",
            _ringCpuPath ? "CPU setShapeTransform" : "GPU kernel (device-written TLAS rows)",
            _ringCount,
            _ringFirstRow == ~0u ? "" : " (contiguous run)",
            _ringLastMs);
        ImGui::TextDisabled("A/B: relaunch with NT_XFORM_CPU_PATH=1 (baseline)");
        ImGui::TextDisabled("freeze-diff: NT_XFORM_FREEZE_AFTER=<sec>; count: NT_XFORM_GPU_COUNT");
    }

    void applyCamera(newtype::util::Camera& camera) override {
        camera.ciCam().lookAt(vec3(3.4f, 2.6f, 4.8f), vec3(0.f, 0.8f, 0.f));
    }

private:
    /// (Re-)register the ring's TLAS rows; resolves the contiguous run fresh
    /// (a post-build add/remove anywhere in the scene reshuffles rows).
    bool register_ring_rows(core::Pipeline& pipeline) noexcept {
        if (!pipeline.registerGpuTransformRows(
                luisa::span<const scene::ShapeId>{_ringIds.data(), _ringIds.size()})) {
            _ringFirstRow = ~0u;
            _ringGpuActive = false;
            return false;
        }
        _ringFirstRow = pipeline.geometryTlasRow(_ringIds.front());
        _ringTopologyGen = pipeline.topologyGeneration();
        _ringGpuActive = true;
        return true;
    }

    void update_ring(core::Pipeline& pipeline, float dt) noexcept {
        if (_ringIds.empty()) return;
        if (!_ringFrozen) _ringTime += dt;

        // Freeze: snap to the exact time, write that pose once, then idle
        // (one prev←curr settle on the first frozen frame — GPU path only).
        if (_ringFreezeAfter > 0.f && _ringTime >= _ringFreezeAfter) {
            if (_ringFrozen) {
                if (_ringSettlePending) {
                    _ringSettlePending = false;
                    if (_ringGpuActive) dispatch_ring(pipeline);
                }
                return;
            }
            _ringTime = _ringFreezeAfter;
            _ringFrozen = true;
        }

        const auto t0 = std::chrono::steady_clock::now();
        if (_ringCpuPath) {
            for (uint i = 0u; i < _ringIds.size(); ++i)
                pipeline.setShapeTransform(_ringIds[i], ring_matrix_cpu(i, _ringCount, _ringTime),
                                           scene::Change::Affine);
        } else {
            // Topology watch: re-resolve rows after any post-build add/remove.
            if (!_ringGpuActive || pipeline.topologyGeneration() != _ringTopologyGen) {
                if (!register_ring_rows(pipeline)) {
                    CI_LOG_W("TransformTreeTest: ring rows no longer contiguous - "
                             "switching the ring to CPU setShapeTransform");
                    _ringCpuPath = true;
                }
            }
            if (_ringGpuActive) {
                dispatch_ring(pipeline);
                _ringSettlePending = true;
            } else {
                for (uint i = 0u; i < _ringIds.size(); ++i)
                    pipeline.setShapeTransform(_ringIds[i], ring_matrix_cpu(i, _ringCount, _ringTime),
                                               scene::Change::Affine);
            }
        }
        _ringLastMs = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
    }

    void dispatch_ring(core::Pipeline& pipeline) noexcept {
        scene::Geometry* geom = pipeline.geometry();
        pipeline.computeStream() << _ringShader(
            _ringFirstRow, _ringTime,
            geom->instance_transform_buffer(),
            geom->instance_transform_prev_buffer())
            .dispatch(_ringCount);
        pipeline.notifyGpuTransformsDirty();
    }

    //======================================================================
    // Deformable churn (NT_XFORM_DEFORM_CHURN) — see the file header
    //======================================================================

    // The pipeline owns the meshes; the raw pointer only drives update_cpu,
    // and the StaticTransform must outlive its borrowed registration.
    struct ChurnMesh {
        scene::ShapeId id = scene::kInvalidShapeId;
        scene::DeformableMesh* mesh = nullptr;
        scene::StaticTransPtr trans;
        float phase = 0.f;
    };
    luisa::vector<ChurnMesh> _churnLive;
    uint32_t _churnTick = 0u;
    uint32_t _churnEvery = 0u;
    uint32_t _churnAdds = 0u, _churnRemoves = 0u;
    uint _churnMat = 0u;

    /// Fresh 9×9 wave patch as a DeformableMesh, added POST-build.
    scene::ShapeId add_churn_mesh(core::Pipeline& pipeline, uint slot) noexcept {
        auto& device = core::Renderer::device();
        auto mesh = scene::make_deformable_mesh(device, _churnMat);
        auto& stream = pipeline.computeStream();

        constexpr uint kN = 9u;             // grid resolution per side
        constexpr float kHalf = 0.35f;      // half-extent
        luisa::vector<scene::MeshShape::Vertex> verts;
        luisa::vector<compute::Triangle> tris;
        verts.reserve(kN * kN);
        for (uint iy = 0u; iy < kN; ++iy)
            for (uint ix = 0u; ix < kN; ++ix) {
                const float u = static_cast<float>(ix) / (kN - 1u);
                const float v = static_cast<float>(iy) / (kN - 1u);
                verts.push_back(scene::MeshShape::Vertex::encode(
                    luisa::make_float3(-kHalf + 2.f * kHalf * u, 0.f,
                                       -kHalf + 2.f * kHalf * v),
                    luisa::make_float3(0.f, 1.f, 0.f),
                    luisa::make_float4(1.f, 0.f, 0.f, 1.f),
                    luisa::make_float2(u, v)));
            }
        for (uint iy = 0u; iy + 1u < kN; ++iy)
            for (uint ix = 0u; ix + 1u < kN; ++ix) {
                const uint a = iy * kN + ix, b = a + 1u,
                            c = a + kN, d = c + 1u;
                tris.push_back(compute::Triangle{a, c, b});
                tris.push_back(compute::Triangle{b, c, d});
            }
        mesh->set_data(luisa::span<const scene::MeshShape::Vertex>{verts},
                       luisa::span<const compute::Triangle>{tris});
        // Build BEFORE addShape: the post-build add path only appends a TLAS
        // entry when mesh_resource() is non-null — an unbuilt deformable
        // would leave _tlas shorter than _instances and misalign every row.
        mesh->build(stream);

        ChurnMesh rec;
        rec.trans = scene::StaticTransform::create(tolc(glm::translate(vec3(
            -1.9f, 0.1f, -0.6f + 0.9f * static_cast<float>(slot)))));
        rec.phase = 0.7f * static_cast<float>(_churnAdds + 1u);
        rec.mesh = mesh.get();
        rec.id = pipeline.addShape(std::move(mesh), rec.trans.get());
        if (rec.id == scene::kInvalidShapeId) {
            CI_LOG_E("TransformTreeTest churn: post-build addShape failed");
            return scene::kInvalidShapeId;
        }
        const scene::ShapeId id = rec.id;
        ++_churnAdds;
        _churnLive.push_back(std::move(rec));
        // B2: a user-params row on every churn mesh so the churn log can
        // verify rows survive remove_shape's swap-and-pop row move (the
        // survivor's InstanceData — params included — is the moved element).
        pipeline.setInstanceUserData(id, 1u,
            luisa::float4(0.5f, 1.f, 2.f, 3.f));
        deform_churn_mesh(_churnLive.back(), stream, 0.f); // seed the first BLAS
        return id;
    }

    /// Analytic wave: y = A·sin(kx·x + w·t + phase)·cos(kz·z); normals from
    /// the analytic gradient. Mutates the packed CPU vertex cache in place.
    void deform_churn_mesh(ChurnMesh& rec, luisa::compute::Stream& stream,
                           float time) noexcept {
        constexpr float kAmp = 0.09f, kKx = 9.f, kKz = 6.f, kW = 2.2f;
        const float phase = rec.phase;
        rec.mesh->update_cpu(
            [phase](std::span<scene::MeshShape::Vertex> verts, float t) noexcept {
                for (auto &vert : verts) {
                    const float sx = std::sin(kKx * vert.px + kW * t + phase);
                    const float cz = std::cos(kKz * vert.pz + phase);
                    const float y = kAmp * sx * cz;
                    // dy/dx = kAmp·kKx·cos(...)·cz; dy/dz = -kAmp·kKz·sx·sin(...)
                    const float dydx = kAmp * kKx * std::cos(kKx * vert.px + kW * t + phase) * cz;
                    const float dydz = -kAmp * kKz * sx * std::sin(kKz * vert.pz + phase);
                    const float inv = 1.f / std::sqrt(dydx * dydx + dydz * dydz + 1.f);
                    vert.py = y;
                    vert.nx = -dydx * inv;
                    vert.ny = inv;
                    vert.nz = -dydz * inv;
                }
            }, stream, time);
    }

    void update_churn(core::Pipeline& pipeline, float time) noexcept {
        if (_churnEvery == 0u) return;
        auto& stream = pipeline.computeStream();

        // Seed two live meshes on the first update (post-build adds —
        // _pending_blas_build + _tlas_needs_rebuild + GPU-row invalidation).
        if (_churnLive.empty()) {
            add_churn_mesh(pipeline, 0u);
            add_churn_mesh(pipeline, 1u);
            CI_LOG_I("TransformTreeTest: deformable churn ON - cycling every "
                << _churnEvery << " frames (2 live)");
            if (std::ofstream out{"xform_selfcheck.log", std::ios::app})
                out << "CHURN start every=" << _churnEvery << std::endl;
        }

        for (auto &rec : _churnLive)
            deform_churn_mesh(rec, stream, time);

        if (++_churnTick < _churnEvery) return;
        _churnTick = 0u;

        // Cycle: drop the oldest, add a fresh one. The two churn meshes hold
        // the highest instance indices, so removeShape's swap-and-pop moves
        // the OTHER deformable across the freed index — the registry remap
        // path (asserted inside Geometry::remove_shape in Debug builds).
        if (_churnLive.size() >= 2u) {
            const scene::ShapeId gone = _churnLive.front().id;
            _churnLive.erase(_churnLive.begin());
            ++_churnRemoves;
            if (!pipeline.removeShape(gone))
                CI_LOG_E("TransformTreeTest churn: removeShape(" << gone << ") failed");
        }
        add_churn_mesh(pipeline, static_cast<uint>(_churnAdds & 1u));
        if (std::ofstream out{"xform_selfcheck.log", std::ios::app}) {
            // B1: the surviving churn mesh was moved by swap-and-pop — its
            // per-instance material row must have moved with it.
            const uint32_t survivor_mats =
                pipeline.geometry()->material_layers(_churnLive.front().id);
            // B2: same check for the per-instance user-params row (slot 1,
            // authored at add_churn_mesh birth).
            const luisa::float4 survivor_params =
                pipeline.geometry()->instance_user_param(_churnLive.front().id, 1u);
            const bool params_ok =
                all(survivor_params == luisa::float4(0.5f, 1.f, 2.f, 3.f));
            out << "CHURN cycle add=" << _churnAdds
                << " rem=" << _churnRemoves << " live=" << _churnLive.size()
                << " survivor_mats=0x" << std::hex << survivor_mats
                << " (expect 0x" << layer0(_churnMat) << ")"
                << " survivor_params=" << (params_ok ? "OK" : "LOST")
                << std::dec << std::endl;
        }
    }

private:
    // Transforms are polled by the pipeline every frame — they must stay
    // alive for the lifetime of the scene (borrowed pointers are registered).
    nt::scene::AnimTransPtr _pivot, _arm, _moon, _protoInst;
    nt::scene::StaticTransPtr _sat;
    core::Pipeline::PrototypeHandle _proto{};
    int _satSlot = -1;

    // B1 per-instance material coverage (see the file header).
    uint _matGray = 0u, _matRed = 0u, _matGold = 0u;
    uint _matCyan = 0u, _matWarm = 0u;
    scene::ShapeId _protoInstId = scene::kInvalidShapeId;
    scene::ShapeId _emisA = scene::kInvalidShapeId, _emisB = scene::kInvalidShapeId;
    int _ringMatSlot = -1;

    // GPU transform-row ring (see the file header). _ringFirstRow is the
    // contiguous TLAS run base; thread i of the kernel writes row first+i.
    luisa::vector<scene::ShapeId> _ringIds;
    uint _ringCount = 0u;
    uint _ringFirstRow = ~0u;
    uint _ringTopologyGen = ~0u;
    bool _ringGpuActive = false;
    bool _ringCpuPath = false;
    bool _ringFrozen = false;
    bool _ringSettlePending = false;
    float _ringFreezeAfter = 0.f;
    float _ringTime = 0.f;
    float _ringLastMs = 0.f;
    luisa::compute::Shader<1,
        luisa::uint,                              // first TLAS row
        float,                                    // time
        luisa::compute::Buffer<luisa::float4x4>,  // rw: transforms (curr)
        luisa::compute::Buffer<luisa::float4x4>   // rw: transforms (prev)
    > _ringShader;
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createTransformTreeScene() { return std::make_unique<TransformTreeScene>(); }
} // namespace newtype::test
