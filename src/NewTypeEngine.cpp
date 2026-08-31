#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "newtype/util/Mesh.h"
#include "newtype/render/TextureConverter.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/LightSampler.h"
#include "newtype/render/MetalData.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"
#include "cinder/CameraUi.h"
#include "cinder/Log.h"
#include "cinder/CinderImGui.h"
#include "cinder/Utilities.h"

#include "newtype/util/Rand.h"
#include "newtype/util/Simplex.h"
#include "newtype/NewType.h"
#include "newtype/scene/Geometry.h"
#include "newtype/scene/LightShape.h"
#include "newtype/core/Pipeline.h"
#include "newtype/scene/VATMesh.h"
#include "newtype/util/Utilities.h"
#include "newtype/feature/DoFFeature.h"
#include "newtype/feature/MotionBlurFeature.h"
#include "newtype/feature/BloomFeature.h"
#include "newtype/feature/ChromaticAberrationFeature.h"

#include "tests/TestScenes.h"

#if NT_ENABLE_MEDIA_PLAYER
#include "newtype/media/VideoPlayer.h"
#endif

#if NT_ENABLE_PROCEDURAL
#include "newtype/scene/ProceduralGeometry.h"
#endif

#if NT_ALLOW_RASTER_FEATURES
#include "newtype/util/Rng.h"
#include "newtype/feature/PointCloud.h"
#include "newtype/feature/Trail.h"
#endif

#include "newtype/util/Noise.h"
#include "newtype/timeline/Timeline.h"
#include "newtype/util/SoundController.h"

#include <filesystem>

#define PHYSICS_TEST 0

#if PHYSICS_TEST
#include "newtype/physics/Physics.h"
#endif

using namespace luisa;
using namespace luisa::compute;

using namespace nt;
using namespace ci;
using namespace ci::app;

class NewTypeEngine : public App {
public:
    void setup() override;
    void update() override;
    void draw() override;
    void cleanup() override;
    void keyDown(KeyEvent event) override;
    void mouseDown(MouseEvent event) override;
    void mouseDrag(MouseEvent event) override;
    void mouseWheel(MouseEvent event) override;
    void resize() override;

private:
    // Renderer for LuisaCompute to Cinder integration
    core::RenderPtr renderer;

    // Camera
    util::CamPtr mCamera;
    // Pipeline (owns scene, G-Buffer, shaders, accumulation)
    core::PipelinePtr mPipeline;

#if NT_ENABLE_PROCEDURAL
    scene::ProcGeomPtr mProcMesh;
#endif

    feature::DoFFeature* _dof = nullptr;
    feature::MotionBlurFeature* _mb = nullptr;
    feature::BloomFeature* _bloom = nullptr;
    feature::ChromaticAberrationFeature* _ca = nullptr;

#if NT_ALLOW_RASTER_FEATURES
    feature::PointCloud* _pc = nullptr;
    feature::Trail* _trail = nullptr;
#endif

    float mTime = 0.0f, mDt = 0.f;
    bool  mTick = false, mDrawUi = true;

    // Timeline
    std::unique_ptr<timeline::Timeline> mTimeline;

#if NT_ENABLE_MEDIA_PLAYER
    // Phase 6: continuous video playback. Decoder + bridge run per render
    // thread update; the wrapped Luisa image is registered as a bindless slot
    // in MaterialPool on setup() for future mesh binding.
    std::unique_ptr<media::VideoPlayer> mVideoPlayer;
    int                                  mVideoBindlessSlot = -1;
    uint64_t                             mVideoFrameCount   = 0u;
#endif
    std::unordered_map<std::string, scene::ShapeId> _shapeNameToId;

#if PHYSICS_TEST
    // Physics (LCS solver) — multi-body capable
    luisa::unique_ptr<physics::Physics> mPhysics;
#endif

    // Sample scene (src/tests/), selected at startup via --scene <name>
    test::ScenePtr mScene;
    std::string    mSceneName;

    // engine funcs
    void initGeometries();
    void initFeatures();
    void updatePipeline();
    void render();
    void drawUi();

    void saveScene();
    void loadScene();

    void wireTimelineCallbacks();

};

class CheckerboardResolver : public render::SurfaceResolver {
public:
    void resolve(render::SurfaceData& s,
        const Var<render::MaterialData>& material,
        Float2 screen_uv, Float3 wo, Float time,
        const BindlessVar& tex_bindless,
        UInt screen_w, UInt screen_h) const noexcept override {
        // Procedural checkerboard pattern modulating albedo (UV from s.uv)
        Float2 uv = s.uv;
        Float scale = 10.0f;
        Float2 grid = compute::fract(uv * scale);
        Float checker = ite((grid.x < 0.5f) ^ (grid.y < 0.5f), 1.0f, 0.0f);
        s.albedo = lerp(s.albedo, s.emission, checker);
        //s.emission = def(make_float3(0.f));
        //s.albedo_alpha = checker;
        //s.alphacut = .5f;
    }
};

//==============================================================================
// NewTypeEngine Implementation
//==============================================================================

void NewTypeEngine::initGeometries() {
    auto& device = core::Renderer::device();
    auto& stream = core::Renderer::stream();

    // Get material pool first so we can use its stream for texture upload
    // (DX12 requires image upload and bindless update on the same stream)
    auto matPool = mPipeline->material();


    //==========================================================================
    // Sample scene (src/tests/) — selected at startup via --scene <name>
    //==========================================================================
    mScene = test::createScene(mSceneName);
    if (!mScene) {
        CI_LOG_W("Unknown scene '" << mSceneName << "', falling back to '"
                 << test::kDefaultScene << "'");
        mSceneName = test::kDefaultScene;
        mScene = test::createScene(mSceneName);
    }

#if NT_ENABLE_MEDIA_PLAYER
    mScene->set_external_albedo_slot(mVideoBindlessSlot);
#endif

    mScene->build(*mPipeline);

    // Scene material indices for the experimental blocks below
#if NT_ALLOW_RASTER_FEATURES || NT_ENABLE_PROCEDURAL
    uint redMatIdx     = mScene->material("red");
    uint greenMatIdx   = mScene->material("green");
    uint checkerMatIdx = mScene->material("checker");
    uint whiteMatIdx   = mScene->material("white");
#endif

#if NT_ALLOW_RASTER_FEATURES
    const uint num = 512 * 1024;// 128;
    auto cloud = std::make_unique<feature::PointCloud>(
        device, num, greenMatIdx,
        luisa::float3{ -1.f }, luisa::float3{ 1.f }, 128u);
    //cloud->setMesh(geom::Teapot());
    _pc = cloud.get();
    vector<luisa::float4> pos(num);
    vector<luisa::float4> vel(num);
    for (uint32_t i = 0; i < num; i++) {
        pos[i] = luisa::make_float4(
            util::randFloat(-.5f, .5f),
            util::randFloat(-.5f, .5f),
            util::randFloat(-.5f, .5f),
            util::randFloat(.0005f, .0045f)
        );
        vel[i] = luisa::make_float4(0.f, 0.f, 0.f, .6f);  // w = alpha (1.0 = opaque)
    }
    cloud->upload_positions(stream, pos, vel, num);
    cloud->setBlendMode(feature::RasterBase::BlendMode::Transparent);
    mPipeline->addFeature(std::move(cloud));  // BEFORE buildScene

    // Create trail feature: 1024 max segments, material index 3
    auto trail = std::make_unique<feature::Trail>(
        device, 1024u, redMatIdx,
        luisa::float3{ -1.f }, // voxel bounds min
        luisa::float3{ 1.f },  // voxel bounds max
        128u                   // voxel resolution
    );

    // Build point data on CPU — N points define N-1 segments
    luisa::vector<feature::TrailPoint> points;

    for (int i = 0; i <= 100; i++) {  // 101 points → 100 segments
        float t = float(i) / 100.f;
        float x = cos(t * 6.f) * .6f;
        float z = sin(t * 6.f) * .6f;
        float y = (t - .5f) * 1.6f;
        float w = 0.08f * (1.f - t);  // taper from thick to thin
        points.push_back({luisa::make_float4(x, y, z, w)});
    }

    // Upload and register (must happen BEFORE buildScene)
    _trail = trail.get();
    trail->uploadPoints(stream, points, points.size());
    mPipeline->addFeature(std::move(trail));
#endif

#if NT_ENABLE_PROCEDURAL
    mProcMesh = scene::ProceduralGeometry::create(device);

    // --- Test: Sphere procedural primitives ---
    {
        // Two test spheres
        mProcMesh->add_sphere(luisa::make_float3(0.8f, 0.0f, 0.5f), 0.08f, checkerMatIdx);
        mProcMesh->add_cube(luisa::make_float3(0.5f, 0.3f, 0.5f), 0.08f, redMatIdx);
    }

    {
        auto mesh = TriMesh(ObjLoader(app::loadAsset("models/lantern.obj")) >> geom::Scale(5.f));
        auto deformId = mProcMesh->add_static_mesh(mesh);
        mProcMesh->add_deformable_instances(deformId, 1, checkerMatIdx);
    }

    mProcMesh->build(stream);
    mPipeline->setProceduralGeometry(std::move(mProcMesh));
#endif

    //==========================================================================
    // Physics — pinned cloth test (LCS solver)
    //==========================================================================
#if PHYSICS_TEST
    {
        auto clothMatIdx = mPipeline->addMaterial(
            "physics_cloth", render::make_diffuse(luisa::float3(0.8f, 0.2f, 0.2f)));

        // Same 32x2 grid topology as the old PhysicsBridge: a 1x1 plane in XZ
        // (half-size 0.5) with 31 subdivisions per side -> 32x32 verts.
        ci::TriMesh clothTri(ci::geom::Plane()
                                 .size(glm::vec2(1.f))
                                 .subdivisions(glm::vec2(31.f, 31.f)));

        mPhysics = luisa::make_unique<physics::Physics>();
        auto* clothMesh = mPhysics->add_body(
            clothTri,
            physics::ClothBodyConfig{
                .stretch_model          = physics::ClothStretchModel::FEM_BW98,
                .bending_model          = physics::ClothBendingModel::QuadraticBending,
                .thickness              = 0.001f,
                .youngs_modulus         = 1e5f,
                .poisson_ratio          = 0.3f,
                .translation            = luisa::float3(0.f, 3.8f, 0.25f),
                .pin_methods            = {physics::FixedPinMethod::LeftBack,
                                           physics::FixedPinMethod::RightBack},
                .name                   = "physics_cloth",
            },
            clothMatIdx);

        // Register with the scene — Pipeline takes ownership. Physics keeps a
        // non-owning pointer for per-frame updates.
        scene::StaticTransform clothIdentity;
        mPipeline->addShape(
            luisa::unique_ptr<scene::DeformableMesh>(clothMesh),
            &clothIdentity);
        mPhysics->prepare();
    }
#endif
}

void NewTypeEngine::wireTimelineCallbacks() {
}

void NewTypeEngine::saveScene() {
    nlohmann::json file;
    file["camera"] = mCamera->toJson();
#if NT_ALLOW_RASTER_FEATURES
    if(_pc) file["pc"] = _pc->toJson();
#endif

    try {
        auto path = app::getAssetPath("") / "scene.json";
        ci::writeJson(path, file);
        CI_LOG_I("Saved scene to " << path);
    } catch (const std::exception& e) {
        CI_LOG_E("Failed to save scene: " << e.what());
    }
}

void NewTypeEngine::loadScene() {
    auto path = app::getAssetPath("") / "scene.json";
    if (fs::exists(path)) {
        try {
            auto config = ci::loadJson(path);
            mCamera->load(config["camera"]);
            CI_LOG_I("Loaded config from " << path);
        } catch (const std::exception& e) {
            CI_LOG_E("Failed to load config: " << e.what());
        }
    }
}

void NewTypeEngine::initFeatures() {
    //==========================================================================
    // Post-processing features (register after buildScene)
    //==========================================================================
    {
        auto dof = std::make_unique<feature::DoFFeature>();
        _dof = dof.get();
        _dof->setEnabled(false);
        mPipeline->addFeature(std::move(dof));

        auto mb = std::make_unique<feature::MotionBlurFeature>();
        _mb = mb.get();
        _mb->setEnabled(false);
        mPipeline->addFeature(std::move(mb));

        // Bloom + Chromatic Aberration run AFTER DoF/MB at the same AfterGlassTint point
        // (FIFO order): they operate on the denoised HDR with bokeh + motion already baked in.
        auto bloom = std::make_unique<feature::BloomFeature>();
        _bloom = bloom.get();
        _bloom->setEnabled(false);
        mPipeline->addFeature(std::move(bloom));

        auto ca = std::make_unique<feature::ChromaticAberrationFeature>();
        _ca = ca.get();
        _ca->setEnabled(false);
        mPipeline->addFeature(std::move(ca));
    }
}

void NewTypeEngine::updatePipeline() {
    // Timeline evaluates here — callbacks mutate transforms/materials before GPU upload
    if (mTimeline)
        mTimeline->evaluate(mDt);

    // Sample scene per-frame animation (animated transforms are polled by the pipeline)
    if (mScene) mScene->update(mTime, mDt);

#if NT_ENABLE_MEDIA_PLAYER
    if (mVideoPlayer) {
        if (mVideoPlayer->update()) ++mVideoFrameCount;
    }
#endif
}

void NewTypeEngine::render() {
    auto& profiler = util::Profiler::instance();
    profiler.begin_profiling();

#if NT_ALLOW_RASTER_FEATURES
    if (_pc && mTick) {
        util::CpuScopedTimer _cpu_PC_update("PC/update");
        profiler.set_pass("PC/update");
        mPipeline->computeStream() << core::ShaderManager::instance().shader<1,
            compute::Buffer<luisa::float4>,
            compute::Buffer<luisa::float4>,
            uint, float, float>(
                "PointCloudUpdate",
                _pc->pos_buffer(),
                _pc->vel_buffer(),
                _pc->particle_count(), mTime, mDt
            ).dispatch(_pc->particle_count());
    }
#endif

#if PHYSICS_TEST
    if (mTick && mPhysics) {
        mPhysics->step();
        // Geometry::update rebuilds dirty deformable BLAS on computeStream;
        // wait for physics-stream vertex writes to complete first.
        mPhysics->wait_for_step(mPipeline->computeStream());
    }
#endif

    mPipeline->update(mTime, mDt);

    mPipeline->beginFrame(*renderer);

    // Submit current frame's GPU work (no sync at end — deferred to next beginFrame)
    mPipeline->render(*renderer, mCamera->cam(), core::Pipeline::DebugTag::None, mDt);

    // Display previous frame (already synced in beginFrame)
    renderer->endFrame();
    profiler.end_profiling();

}

void NewTypeEngine::drawUi() {
    if (!mDrawUi) return;

    // ImGui — CPU-only, overlaps with prev frame GPU tail
    {
        ImGui::ScopedWindow scpWin("Engine", true);
        ImGui::SetWindowSize(ivec2(320, 500));
        ImGui::SetWindowPos(ivec2(5, 5));
        mPipeline->drawUi();
        util::SoundController::get().drawUi();
        if (renderer->recorder())
            renderer->recorder()->drawUi();
#if PHYSICS_TEST
        if (mPhysics) mPhysics->drawUi();
#endif
        if (mTimeline) {
            if(mTimeline->drawUi())
                wireTimelineCallbacks();
        }

        if (mScene) mScene->drawUi();
#if NT_ENABLE_MEDIA_PLAYER
        if (mVideoPlayer) {
            if (ImGui::CollapsingHeader("Video")) {
                bool playing = mVideoPlayer->is_playing();
                if (ImGui::Checkbox("Playing", &playing))
                    playing ? mVideoPlayer->play() : mVideoPlayer->pause();
                bool loop = mVideoPlayer->is_looping();
                if (ImGui::Checkbox("Loop", &loop)) mVideoPlayer->set_looping(loop);
                if (ImGui::Button("Rewind")) mVideoPlayer->rewind();
                ImGui::SameLine();
                if (mVideoPlayer->duration_sec() > 0.0) {
                    float t = static_cast<float>(mVideoPlayer->current_time_sec());
                    if (ImGui::SliderFloat("Seek", &t, 0.f,
                                           static_cast<float>(mVideoPlayer->duration_sec()),
                                           "%.1f s"))
                        mVideoPlayer->seek(static_cast<double>(t));
                }
            }
        }
#endif
    }
    {
        ImGui::ScopedWindow scpWin("Render", true);
        ImGui::SetWindowSize(ivec2(320, 500));
        ImGui::SetWindowPos(ivec2(330, 5));

        mPipeline->material()->drawUi();
        if (_dof) _dof->drawUi();
        if (_mb) _mb->drawUi();
        if (_bloom) _bloom->drawUi();
        if (_ca) _ca->drawUi(); 
#if NT_ALLOW_RASTER_FEATURES
        if (_pc) _pc->drawUi();
        if (_trail) _trail->drawUi();
#endif
    }
    // Timeline UI (now creates its own windows)
    if (mTimeline)
        mTimeline->drawTimeline(mTimeline->currentTime(), 500.f, 1280.f);
}

void NewTypeEngine::setup() {
    // Create renderer — change Backend::CUDA to Backend::DirectX to test DX12
    constexpr core::Backend kBackend = core::Backend::DirectX;
    renderer = core::Renderer::create(
        getWindowWidth(), getWindowHeight(), kBackend, false);
    renderer->initialize(core::TextureType::Int8);

    gl::enableVerticalSync(false);
    app::setFrameRate(300.f);

    ImGui::Initialize();
    mCamera = util::Camera::create(getWindow());

    //util::SoundController::get().loadConfig(ci::fs::path("audio/sounds.json"));

    //==========================================================================
    // Create Pipeline (compiles shaders, creates G-Buffer + accumulation images)
    //==========================================================================
    mPipeline = core::Pipeline::create(*renderer);

#if NT_ENABLE_MEDIA_PLAYER
    {
        namespace fs = std::filesystem;
        // replace this with actual video file path
        fs::path videoPath = app::getAssetPath("videos/test.mp4");
        if (fs::exists(videoPath)) {
            mVideoPlayer = std::make_unique<media::VideoPlayer>();
            auto* luDevice = static_cast<ID3D12Device*>(core::Renderer::device().native_handle());
            if (mVideoPlayer->open(videoPath, core::Renderer::device(), luDevice)) {
                auto matPool = mPipeline->material();
                mVideoBindlessSlot = static_cast<int>(
                    matPool->register_external_image(mVideoPlayer->image()));
                CI_LOG_I("[Phase6] video player ready, bindless slot=" << mVideoBindlessSlot);
            } else {
                mVideoPlayer.reset();
            }
        } else {
            CI_LOG_W("[Phase6] assets/video/test.mp4 not found — video player disabled");
        }
    }
#endif

    //==========================================================================
    // Sample scene selection: --scene cornell|material|room (default "material")
    //==========================================================================
    mSceneName = test::kDefaultScene;
    const auto& cmdArgs = getCommandLineArgs();
    for (size_t i = 0; i + 1 < cmdArgs.size(); ++i) {
        if (cmdArgs[i] == "--scene" || cmdArgs[i] == "-scene") {
            mSceneName = cmdArgs[i + 1];
            break;
        }
    }

    initGeometries();
    getWindow()->setTitle("NewTypeEngine - realtime pathtracing with ReSTIR [" + mSceneName + "]");

    mPipeline->addEnvMap(Surface32f::create(
        loadImage(app::loadAsset("textures/colorful_studio_4k.hdr"))));
    // Build TLAS + LightSampler
    mPipeline->buildScene();
    initFeatures();
    loadScene();

    //==========================================================================
    // Timeline
    //==========================================================================
    mTimeline = std::make_unique<timeline::Timeline>();
    auto tlPath = app::getAssetPath("timeline.json");
    if (fs::exists(tlPath)) {
        mTimeline->load(tlPath);
        wireTimelineCallbacks();
    }

    // Load saved config (parameters, materials, feature toggles)
    mPipeline->loadConfig(app::getAssetPath("config.json"));
    //mCamera->camUi().disable();
    
    // point cloud move shader
#if NT_ALLOW_RASTER_FEATURES
    auto& sm = newtype::core::ShaderManager::instance();
    sm.registerShader<1>("PointCloudUpdate", [&](
        compute::BufferVar<luisa::float4> pos_buffer,
        compute::BufferVar<luisa::float4> vel_buffer,
        compute::UInt vertex_count,
        compute::Float time,
        compute::Float dt
        ) noexcept {

        set_block_size(256u);
        UInt i = dispatch_x();

        $if(i >= vertex_count) { $return(); };
        auto pos = pos_buffer.read(i);
        Float prev_alpha = vel_buffer.read(i).w;
        auto vel = util::curlNoise(pos.xyz() * 2.6f + .02f * time) * .0035f * 60.f;
        auto tmp = pos.xyz() + vel * dt;

        $if(any(tmp < def(make_float3(-1.f))) | any(tmp > def(make_float3(1.f)))) {
            auto uvec3 = make_uint3(
                cast<UInt>(10000.f + 10000.f * tmp.x + time) + i,
                cast<UInt>(10000.f + 10000.f * tmp.y + time) + i,
                cast<UInt>(10000.f + 10000.f * tmp.z + time) + i);
            auto rvec3 = util::pcg3d( uvec3 );
            tmp.x = 2.f * util::uniform_uint_to_float(rvec3.x) - 1.f;
            tmp.y = 2.f * util::uniform_uint_to_float(rvec3.y) - 1.f;
            tmp.z = 2.f * util::uniform_uint_to_float(rvec3.z) - 1.f;
            vel = make_float3(0.f);
        };

        //auto tmp = clamp(pos.xyz() + vel * .0035f, make_float3(-1.f), make_float3(1.f));
        pos_buffer.write(i, make_float4(tmp, pos.w));
        vel_buffer.write(i, make_float4(vel, prev_alpha));
    });
#endif

    mTime = app::getElapsedSeconds();
}

void NewTypeEngine::cleanup() {
    mScene.reset();
    mPipeline.release();
    renderer.reset();
}

void NewTypeEngine::update() {
    // Advance time every frame regardless of mTick so the denoiser (and any
    // other dt-dependent subsystem) always sees a valid delta.
    float currTime = app::getElapsedSeconds();
    mDt = glm::clamp(currTime - mTime, 1.f / 90.f, 1.f / 30.f);

    if (mTick) {
        updatePipeline();
    }

    mTime = currTime;

    mCamera->update();
    util::SoundController::get().update();
    drawUi();
}

void NewTypeEngine::draw() {
    gl::clear(Color(0.1f, 0.1f, 0.15f));
    render();
    gl::draw(renderer->currentTexture(), getWindowBounds());

    // Capture frame for recording/screenshot
    if (mTick)
        renderer->recorder()->update(renderer->captureFbo());
}

void NewTypeEngine::keyDown(KeyEvent event) {
    if (event.getChar() == 'f')
        setFullScreen(!isFullScreen());
    else if (event.isControlDown() && event.getCode() == KeyEvent::KEY_s)
        saveScene();
    else if (event.getCode() == KeyEvent::KEY_ESCAPE)
        quit();
    else if (event.getCode() == KeyEvent::KEY_SPACE)
        mTick = !mTick;
    else if (event.getCode() == KeyEvent::KEY_s)
        renderer->recorder()->saveScreenshot();
    else if (event.getCode() == KeyEvent::KEY_u)
        mDrawUi = !mDrawUi;
    else if (event.getCode() == KeyEvent::KEY_c)
        mCamera->camUi().enable(!mCamera->camUi().isEnabled());
}

void NewTypeEngine::resize() {
    if (renderer) {
        renderer->resize(getWindowWidth(), getWindowHeight());
    }
    mCamera->ciCam().setAspectRatio(getWindowAspectRatio());

    mPipeline->resize(getWindowWidth(), getWindowHeight());
}

void NewTypeEngine::mouseDown(MouseEvent event) {
    mCamera->mouseDown(event);
}

void NewTypeEngine::mouseDrag(MouseEvent event) {
    mCamera->mouseDrag(event);
}

void NewTypeEngine::mouseWheel(MouseEvent event) {
    mCamera->mouseWheel(event);
}

CINDER_APP(NewTypeEngine, RendererGl, [](App::Settings* settings) {
    settings->setWindowSize (1920, 1080);
    settings->setResizable  (true);
    settings->setTitle      ("NewTypeEngine - realtime pathtracing with ReSTIR");
    settings->setConsoleWindowEnabled();
})
