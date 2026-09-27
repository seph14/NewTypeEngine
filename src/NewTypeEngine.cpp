#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/app/RendererD3d12.h"
#include "cinder/app/Platform.h"
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
#include "newtype/feature/FxaaFeature.h"
#include "newtype/util/UiHelper.h"
#include "newtype/util/JsonBackup.h"
#if NT_ENABLE_EDITOR
#include "newtype/feature/Gizmo.h"
#endif

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
#include "newtype/audio/SpatialAudio.h"

#include <filesystem>
#include <fstream>
#include <algorithm>

// Legacy compile-time demo branches (off). Sample scenes live in src/tests/
// and are selected at runtime with `--scene <cornell|material|room>`.
#define DEMO 0
#define NT_ALEX 0

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
    // Engine teardown, safe to run exactly once. Called from the window-close
    // signal (while Cinder's D3D12 renderer is still valid — the WM_CLOSE
    // path kills the renderer impl BEFORE Cinder's cleanup phase) and again
    // from cleanup() on the quit() path.
    void shutdownEngine();
    bool _engineDown = false;
    void keyDown(KeyEvent event) override;
    void mouseDown(MouseEvent event) override;
    void mouseDrag(MouseEvent event) override;
    void mouseWheel(MouseEvent event) override;
    void resize() override;

private:
    // Renderer for LuisaCompute to Cinder integration
    core::RenderPtr renderer;
    // Non-null when launched with --dx: present through Cinder's D3D12
    // renderer instead of the GL interop path (see the CINDER_APP settings)
    ci::app::RendererD3d12Ref mDx12Renderer;

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
    feature::FxaaFeature* _fxaa = nullptr;
#if NT_ENABLE_EDITOR
    feature::GizmoFeature* _gizmo = nullptr;
#endif


    // Post-processing features
#if DEMO
    scene::VATMeshPtr mVatMesh;
    std::vector<scene::StaticTransPtr> _transforms;
    std::vector<scene::AnimTransPtr> _lanternTrans;
    luisa::vector<scene::ShapeId> _flockIDs, _lanternIDs;
    struct FlockData {
        ci::vec3 p;
        float    leadership;
        ci::vec3 v;
        float    crowd;
    };
    luisa::vector<FlockData> _flockData, _lanternData;
    ci::AxisAlignedBox mCardBnd;
    bool mFlock = false;
#endif

#if NT_ALLOW_RASTER_FEATURES
    feature::PointCloud* _pc = nullptr;
    feature::Trail* _trail = nullptr;
    core::ShaderHandle<1,
        compute::Buffer<luisa::float4>,
        compute::Buffer<luisa::float4>,
        uint, float, float> _pcUpdateShader;
#endif

    float mTime = 0.0f, mDt = 0.f;
    bool  mTick = false, mDrawUi = true;

#if NT_DEBUG_VIZ
    // ui_image smoke test (DX texture in ImGui): gradient + checker stripe.
    ImgFlt _uiTestImg;
    bool   _uiTestImgInit = false;
    void   _ensureUiTestImage();
#endif

    // Perf harness (perf review R2 item 1): --perf <frames> [--perf-out
    // <path>] [--perf-warmup <n>] [--perf-orbit] — automated timed run
    // exporting per-pass GPU ms (StatsExt; validation build), CPU pass ms
    // and fps to CSV, then auto-quits. HUD is forced off for comparable
    // numbers.
    bool        mPerfMode = false;
    bool        mPerfOrbit = false;
    bool        mPerfShot = false;
    // Deferred --resolver-param sets (applied after buildScene).
    std::vector<std::tuple<std::string, uint, float>> mPendingResolverParams;
    uint32_t    mPerfWarmup = 60u;
    uint32_t    mPerfFrames = 300u;
    uint32_t    mPerfFrame = 0u;
    std::string mPerfOut = "perf_run.csv";
    std::unordered_map<std::string, double> mPerfGpuSum, mPerfCpuSum;
    std::vector<double> mPerfFpsSamples;
    double      mPerfGpuTotalSum = 0.0, mPerfCpuTotalSum = 0.0;
    float       mPerfOrbitAngle = 0.f;
    glm::vec3   mPerfOrbitEye0{0.f};
    void        _perfCollectFrame();
    void        _perfFinish();

    // Non-perspective projection harness (docs/non_perspective_camera_report.md):
    // --projection <perspective|equirect|cylindrical|fisheye|roomrig>
    // [--res WxH] [--rig-preset <4wall|4wall_floor|cube>] [--rig-face-res N]
    // [--fisheye-fov deg] [--accum N] [--shot path.png]
    // --accum enables progressive accumulation; --shot saves there (render
    // resolution) after N accumulated frames and quits.
    std::string             mProjectionArg;
    std::string             mRigPresetArg = "4wall";
    uint32_t                mRigFaceRes = 1024u;
    float                   mFisheyeFov = 180.0f;
    std::optional<luisa::uint2> mResOverrideArg;
    uint32_t                mAccumFrames = 0u;
    std::string             mShotPath;
    void                    _applyProjection(util::CameraProjection p,
                                             std::optional<luisa::uint2> resOverride);

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

    // The material pool's stream is reused for texture uploads so load-time
    // copies share one D3D12 queue with the compression pass in createMaterial.
    // Reusing it is an optimization, not a correctness requirement: uploads on
    // any stream are synchronized before the texture is used, and cross-stream
    // use is safe (LC barriers restore COMMON after every dispatch).
    auto matPool = mPipeline->material();

    //==========================================================================
    // Scene geometry
    //==========================================================================
#if DEMO
    auto cardGeom = TriMesh::create(ObjLoader(app::loadAsset("models/card.obj")));
    auto cardBnd  = cardGeom->calcBoundingBox();
    mCardBnd = cardBnd;

    // first card need separate texture
    auto mainMat = render::make_diffuse();
    render::MaterialTextures textures;
    textures.albedo = render::TextureConverter::loadFile(
        app::getAssetPath("textures/logo_tex.jpg"), device, &matPool->stream(), true);
    render::TextureCompressionSettings compression;
    compression.enableCompression = true;  // Master toggle
    auto mainMatIdx = matPool->createMaterial("custom", mainMat, std::move(textures), compression);

    // cards as separate mesh
    auto mainTrans = scene::StaticTransform::create();
    auto mainCardMesh = scene::MeshShape::create(device, mainMatIdx);
    mainCardMesh->load_from(*cardGeom);
    mainCardMesh->build(stream);
    auto mainCardID = mPipeline->addShape(std::move(mainCardMesh), mainTrans.get());

    _shapeNameToId["main"] = mainCardID;
    _transforms.push_back(std::move(mainTrans));
  
    for (uint32_t i = 0; i < 8; i++) {
        uint matID;
        std::string cardname = "card_" + toString(i);
        if (i == 0) {
            matID = mPipeline->addMaterial(cardname + "_mat",
                render::make_conductor());
        } else if (i == 1) {
            matID = mPipeline->addMaterial(cardname + "_mat",
                render::make_fabric());
        } else if (i == 2) {
            matID = mPipeline->addMaterial(cardname + "_mat",
                render::make_plastic());
        } else if (i == 3) {
            matID = mPipeline->addMaterial(cardname + "_mat",
                render::make_dielectric());
        } else if (i == 4) {
            matID = mPipeline->addMaterial(cardname + "_mat",
                render::make_subsurface());
        } else if (i == 5) {
            // conductor + iridescence
            auto mat = render::make_conductor();
            mat.iridescence = 1.f;
            mat.iridescence_ior = 3.6f;
            mat.iridescence_thickness = 12.f;
            matID = mPipeline->addMaterial(cardname + "_mat", mat);
        } else if (i == 6) {
            // plastic + sheen 
            auto mat = render::make_plastic();
            mat.sheen = 1.f;
            mat.sheen_tint = .9f;
            matID = mPipeline->addMaterial(cardname + "_mat", mat);
        } else if (i == 7) {
            // subsurface + iridence
            auto mat = render::make_subsurface();
            mat.iridescence = 1.f;
            mat.iridescence_ior = 2.6f;
            mat.iridescence_thickness = 36.f;
            matID = mPipeline->addMaterial(cardname + "_mat", mat);
        }

        auto trans =
        glm::translate(
            vec3(mCardBnd.getCenter().x,
                mCardBnd.getCenter().y,
                mCardBnd.getCenter().z))*
            glm::rotate((.5f - 2.f * float(i+1) / 8.f) * (float)M_PI,
                vec3(1.f, 0.f, 0.f))*
            glm::translate(vec3(-mCardBnd.getCenter().x,
                -1.1f * float(i+1) * mCardBnd.getSize().y - mCardBnd.getCenter().y,
                1.15f * mCardBnd.getSize().z));

        auto cardTrans = scene::StaticTransform::create(tolc(trans));

        auto cardMesh = scene::MeshShape::create(device, matID);
        cardMesh->load_from(*cardGeom);
        cardMesh->build(stream);
        auto cardID = mPipeline->addShape(std::move(cardMesh), cardTrans.get());

        _shapeNameToId[cardname] = cardID;
        _transforms.push_back(std::move(cardTrans));
    }
    
    // ground
    scene::StaticTransform gndTrans(tolc(glm::translate(vec3(0.f, -.3f, 0.f)) * glm::scale(vec3(50.f,1.f,50.f)))); 
    auto gndMatIdx = mPipeline->addMaterial("ground", render::make_diffuse());
    auto gndMesh = scene::MeshShape::create(device, gndMatIdx);
    gndMesh->load_from(ObjLoader(app::loadAsset("models/ground.obj")));
    gndMesh->build(stream);
    auto gndID = mPipeline->addShape(std::move(gndMesh), &gndTrans);
    _shapeNameToId["ground"] = gndID;
    
    // box
    {
        render::MaterialTextures textures;
        textures.albedo = render::TextureConverter::loadFile(
            app::getAssetPath("textures/paper_texture_11.png"), device, &matPool->stream(), true);
        textures.normal = render::TextureConverter::loadFile(
            app::getAssetPath("textures/papernormal.jpg"), device, &matPool->stream());
        render::TextureCompressionSettings compression;
        compression.enableCompression = true;  // Master toggle

        auto vatMat = render::make_diffuse();
        auto vatMatIdx = matPool->createMaterial("box_mat", vatMat, std::move(textures), compression);

        mVatMesh = scene::VATMesh::create(device);
        mVatMesh->load_folder(app::getAssetPath("models/unfold"), "unfold");
        mVatMesh->set_double_sided(true);
        mVatMesh->set_transform(luisa::rotation(luisa::make_float3(0.f, 1.f, 0.f), 
                                (float)M_PI) * luisa::scaling(1.25f));
        mVatMesh->build(*mPipeline, stream, vatMatIdx);
        mVatMesh->set_speed(1.f);
        mVatMesh->set_layer(*mPipeline, 1, mainMatIdx);
    }

    {
        // top light
        auto litMatIdx = mPipeline->addMaterial("lit",
            render::make_emissive(8.f * tolc(Color::hex(0xFFD5B8))));
        TriMesh whiteLightMesh = ObjLoader(app::loadAsset("models/demolit.obj"));
        auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

        auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(0.f, .0f, 0.f))));
        _shapeNameToId["lit"] = mPipeline->addLightShape(std::move(whiteLight), whiteLightTransform.get());
    }

    {
        // side light
        auto litMatIdx = mPipeline->addMaterial("sidelit",
            render::make_emissive(tolc(10.f * Color::hex(0xFFD5B8))));
        TriMesh whiteLightMesh = ObjLoader(app::loadAsset("models/demolit2.obj"));
        auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

        auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(0.f, .0f, 0.f))));
        _shapeNameToId["sidelit"] = mPipeline->addLightShape(std::move(whiteLight), whiteLightTransform.get());
    }

    {
        // wall light
        auto litMatIdx = mPipeline->addMaterial("walllit",
            render::make_emissive(tolc(5.f * Color::hex(0xFFD5B8))));
        TriMesh whiteLightMesh = ObjLoader(app::loadAsset("models/wall.obj"));
        auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

        auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(0.f, .0f, 0.f))));
        auto wallID = mPipeline->addLightShape(std::move(whiteLight), whiteLightTransform.get());
        _shapeNameToId["wall"] = wallID;
    }

    {
        // side top light
        auto litMatIdx = mPipeline->addMaterial("boxlit",
            render::make_emissive(tolc(10.f * Color::hex(0xFFD5B8))));
        TriMesh whiteLightMesh = ObjLoader(app::loadAsset("models/demolit3.obj"));
        auto whiteLight = scene::make_light(device, whiteLightMesh, litMatIdx);

        auto whiteLightTransform = scene::StaticTransform::create(tolc(glm::translate(vec3(0.f, .0f, 0.f))));
        _shapeNameToId["toplit"] = mPipeline->addLightShape(std::move(whiteLight), whiteLightTransform.get());
    }

    // instance mesh for flocking
    {
        auto flkMatIdx = mPipeline->addMaterial("flock", render::make_diffuse());
        auto sphere = scene::MeshShape::create(device, flkMatIdx);
        sphere->load_from(ObjLoader(app::loadAsset("models/flock.obj")));
        
        // 2. Register as prototype
        auto handle = mPipeline->addPrototype(std::move(sphere));

        // 3. Create instances with transforms
        luisa::vector<float4x4> transforms;
        const float side = .15f;
        for (uint i = 0; i < 8; ++i) {
            for (uint j = 0; j < 8; ++j) {
                vec3 p = vec3(
                    float(i) * side - 4.f * side,
                    0.0f,
                    float(j) * side - 4.f * side);
                transforms.push_back(translation(tolc(p)));

                FlockData flk;
                flk.p = p;
                flk.v = .01f * toci(util::randVec3());
                flk.leadership = util::randFloat(.2f, 1.f);
                flk.crowd = 1.f;
                _flockData.push_back(flk);
            }
        }
    
        mPipeline->addPrototypeInstances(handle, transforms, _flockIDs);
    }

    // lanterns as light
    {
        auto lanternMatIdx = mPipeline->addMaterial("lantern",
            render::make_emissive(tolc(3.f * Color::hex(0xFFD5B8))));
        TriMesh lanternMesh = ObjLoader(app::loadAsset("models/lantern.obj"));
        
        for (uint i = 0; i < 3; ++i) {
            auto p = luisa::make_float3(
                util::randFloat(-.5f, .5f),
                util::randFloat(.05f, .2f),
                util::randFloat(-.5f, .5f));

            FlockData flk;
            flk.p = toci(p);
            flk.v = .01f * toci(util::randVec3());
            flk.leadership = 0.f;
            flk.crowd = 1.f;
            _lanternData.push_back(flk);

            auto lantern = scene::make_light(device, lanternMesh, lanternMatIdx);
            auto lanternTransform = scene::AnimatedTransform::create(translation(p));
            _lanternIDs.push_back(mPipeline->addLightShape(std::move(lantern), lanternTransform.get()));
            _lanternTrans.push_back(std::move(lanternTransform));
        }
    }
#elif NT_ALEX
    auto glassMatIdx = mPipeline->addMaterial("gundam",
        render::make_conductor_metal(nt::render::MetalPreset::Gold)
    );
    scene::StaticTransform cubeTransform;
    auto gundamMesh = scene::MeshShape::create(device, glassMatIdx);
    gundamMesh->load_from(ObjLoader(app::loadAsset("models/gundamnt.obj")));
    gundamMesh->build(stream);
    mPipeline->addShape(std::move(gundamMesh), &cubeTransform);

#else

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
    // A test scene may already own the procedural geometry slot (TetCageScene
    // registers its type-3 comparison instance) — don't clobber it.
    if (mPipeline->proceduralGeom() == nullptr) {
    mProcMesh = scene::ProceduralGeometry::create(device);

    // --- Test: Sphere procedural primitives ---
    {
        // Two test spheres
        //mProcMesh->add_sphere(luisa::make_float3(-0.8f, 0.5f, 0.5f), 0.12f, 0);
        //mProcMesh->add_sphere(luisa::make_float3(0.8f, 0.0f, 0.5f), 0.08f, checkerMatIdx);
        //mProcMesh->add_cube(luisa::make_float3(0.5f, 0.3f, 0.5f), 0.08f, redMatIdx);
    }

    /* {
        auto mesh = TriMesh(ObjLoader(app::loadAsset("models/lantern.obj")) >> geom::Scale(5.f));
        auto deformId = mProcMesh->add_static_mesh(mesh);
        mProcMesh->add_deformable_instances(deformId, 1, checkerMatIdx);
        // Type-3 instances deform only with a shader registered in the
        // ShaderManager (see the contract on set_deform_shader_id) — without
        // one they render the rest-pose mesh.
        // sm.registerShader<2>("my_deform", ...); mProcMesh->set_deform_shader_id("my_deform", 256u);
    }*/

    // --- Test: VAT procedural primitive (ginkgo) ---
    {
        auto vatPath = app::getAssetPath("models/ginkgo/ginkgo0.vat");
        if (std::filesystem::exists(vatPath)) {
            auto results = mProcMesh->add_vat_from_file(vatPath);
            for (const auto& result : results)
                if (result.mesh_id != ~0u)
                    mProcMesh->add_instance(result.mesh_id, whiteMatIdx, 24.0f, 0.0f, .1f);
        }
        else {
            CI_LOG_W("ProceduralGeometry: ginkgo VAT not found at " << vatPath.string());
        }
    }

    mProcMesh->build(stream);
    mPipeline->setProceduralGeometry(std::move(mProcMesh));
    }
#endif
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
#if DEMO
    if (!mTimeline) return;
    for (auto& evt : mTimeline->events()) {
        auto shapeIt = _shapeNameToId.find(evt->targetName);
        auto shapeId = (shapeIt != _shapeNameToId.end())
            ? std::optional<scene::ShapeId>(shapeIt->second)
            : std::nullopt;
        if (!shapeId) {
            if (auto* curveEvt = dynamic_cast<timeline::CurveTrigger*>(evt.get())) {
                if (evt->name == "initShowCurve") {
                    evt->onUpdate = [this, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        float w = mCardBnd.getSize().x;
                        for (uint32_t i = 0; i < 8; i++) {
                            auto& cardTrans = _transforms[i + 1];
                            float v = float(i + 1);
                            auto trans = glm::translate(vec3(
                                -w * v * val,
                                -1.1f * v * mCardBnd.getSize().y,
                                w * v * val
                            ));
                            cardTrans->set_matrix(tolc(trans));

                            std::string cardname = "card_" + toString(i);
                            auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find(cardname)->second);
                            mPipeline->setShapeTransform(*shapeId, cardTrans->matrix(), scene::Change::Affine);
                        }

                    };
                } else if (evt->name == "initRotCurve") {
                    evt->onUpdate = [this, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        float w = mCardBnd.getSize().x;
                        for (uint32_t i = 0; i < 8; i++) {
                            auto& cardTrans = _transforms[i + 1];
                            quat r; vec3 p, s;
                            cardTrans->decompose(p, r, s);
                            
                            float v = float(i + 1);
                            auto trans =
                                glm::translate(
                                    vec3(-w * v * .05f + mCardBnd.getMin().x, 
                                         mCardBnd.getCenter().y - 1.1f * v * mCardBnd.getSize().y,
                                        w * v * .05f + mCardBnd.getCenter().z)) *
                                glm::rotate(-glm::min(2.f, 2.f * v * val) * (float)M_PI, vec3(0.f, 1.f, 0.f)) *
                                glm::translate(vec3(-mCardBnd.getMin().x, mCardBnd.getCenter().y, mCardBnd.getCenter().z));
                            cardTrans->set_matrix(trans);

                            std::string cardname = "card_" + toString(i);
                            auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find(cardname)->second);
                            mPipeline->setShapeTransform(*shapeId, tolc(trans), scene::Change::Affine);
                        }

                    };
                } else if (evt->name == "flipRotCurve") {
                    evt->onUpdate = [this, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        float w = mCardBnd.getSize().x;
                        for (uint32_t i = 0; i < 8; i++) {
                            auto& cardTrans = _transforms[i + 1];
                            quat r; vec3 p, s;
                            cardTrans->decompose(p, r, s);

                            float v = float(i + 1);
                            auto trans =
                                glm::translate(
                                    vec3(mCardBnd.getCenter().x,
                                        mCardBnd.getCenter().y,
                                        mCardBnd.getCenter().z)) *
                                glm::rotate((.5f - 2.f * glm::min(1.f, v * val)) * (float)M_PI, 
                                        vec3(1.f, 0.f, 0.f)) *
                                glm::translate(vec3(- mCardBnd.getCenter().x, 
                                    - 1.1f * v * mCardBnd.getSize().y - mCardBnd.getCenter().y, 
                                    1.15f * mCardBnd.getSize().z));
                            cardTrans->set_matrix(trans);

                            std::string cardname = "card_" + toString(i);
                            auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find(cardname)->second);
                            mPipeline->setShapeTransform(*shapeId, tolc(trans), scene::Change::Affine);
                        }
                    };
                } else if (evt->name == "mainCardPush") {
                    evt->onUpdate = [this, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        float h = mCardBnd.getSize().y;

                        auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("main")->second);
                        auto& mainTrans = _transforms[0];

                        auto trans = glm::translate(vec3(0.f, - h * val, 0.f));
                        mainTrans->set_matrix(trans);
                        mPipeline->setShapeTransform(*shapeId, tolc(trans), scene::Change::Affine);
                    };
                } else if (evt->name == "boxSize") {
                    evt->onUpdate = [this, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        mVatMesh->set_scale(1.25f * luisa::lerp(luisa::float3(.3f, 1.f, .225f), luisa::float3(1.f), val));
                    };
                } else if (evt->name == "cardBlend") {
                    evt->onUpdate = [this, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());

                        auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("main")->second);
                        uint matIdx = mPipeline->getShape(*shapeId)->material_layers() & 0xFFu;
                        auto matData = mPipeline->material()->getMaterial(matIdx).data;
                        matData.meta = glm::clamp(val, .0f, .99f);
                        mPipeline->material()->updateMaterialData(matIdx, matData);
                    };
                }
            } else if (auto* baseEvt = dynamic_cast<timeline::EventTrigger*>(evt.get())) {
                if (evt->name == "visible") {
                    evt->onUpdate = [this, baseEvt](float prog) {
                        mFlock = false;
                        mVatMesh->reset();
                        mVatMesh->set_playing(false);
                        mVatMesh->set_visible(*mPipeline, false);

                        // just for testing so we reset things 
                        mPipeline->loadConfig(app::getAssetPath("config.json"));
                        
                        auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("main")->second);
                        mPipeline->setShapeVisibility(*shapeId, true);
                        auto& mainTrans = _transforms[0];
                        
                        uint matIdx = mPipeline->getShape(*shapeId)->material_layers() & 0xFFu;
                        auto matData = mPipeline->material()->getMaterial(matIdx).data;
                        matData.meta = 0.f;
                        mPipeline->material()->updateMaterialData(matIdx, matData);

                        luisa::float4x4 idmat = make_float4x4(1.f);
                        mainTrans->set_matrix(idmat);
                        mPipeline->setShapeTransform(*shapeId, idmat);
                        
                        shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("toplit")->second);
                        mPipeline->setShapeVisibility(*shapeId, false);
                        shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("lit")->second);
                        mPipeline->setShapeVisibility(*shapeId, true);
                        shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("sidelit")->second);
                        mPipeline->setShapeVisibility(*shapeId, true);
                        shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("wall")->second);
                        mPipeline->setShapeVisibility(*shapeId, true);

                        const float side = .15f;
                        uint32_t flkID = 0;
                        for (uint i = 0; i < 8; ++i) {
                            for (uint j = 0; j < 8; ++j) {
                                vec3 p = vec3(
                                    float(i) * side - 4.f * side,
                                    0.0f,
                                    float(j) * side - 4.f * side);

                                auto& flk = _flockData[flkID++];
                                flk.p     = p;
                                flk.v     = .01f * toci(util::randVec3());
                                flk.crowd = 1.f;
                            }
                        }

                        for (uint i = 0; i < 3; ++i) {
                            auto p = luisa::make_float3(
                                util::randFloat(-.5f, .5f),
                                util::randFloat(.05f, .2f),
                                util::randFloat(-.5f, .5f));

                            auto& flk = _lanternData[i];
                            flk.p = toci(p);
                            flk.v = .01f * toci(util::randVec3());
                        }

                        for (const auto id : _flockIDs)
                            mPipeline->setShapeVisibility(id, false);
                        for (const auto id : _lanternIDs)
                            mPipeline->setShapeVisibility(id, false);
                        
                        for (uint32_t i = 0; i < 8; i++) {
                            std::string cardname = "card_" + toString(i);
                            shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find(cardname)->second);
                            mPipeline->setShapeVisibility(*shapeId, true);
                        }
                    };
                } else if (evt->name == "disable") {
                    evt->onUpdate = [this, baseEvt](float prog) {
                        for (uint32_t i = 0; i < 8; i++) {
                            std::string cardname = "card_" + toString(i);
                            auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find(cardname)->second);
                            mPipeline->setShapeVisibility(*shapeId, false);
                        }
                    };
                } else if (evt->name == "ShowVAT") {
                    evt->onUpdate = [this, baseEvt](float) {
                        mVatMesh->reset();
                        mVatMesh->set_speed(1.f);
                        mVatMesh->set_visible(*mPipeline, true);
                        mVatMesh->set_playing(true);

                        auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("toplit")->second);
                        mPipeline->setShapeVisibility(*shapeId, true);
                    };
                } else if (evt->name == "HideVAT") {
                    evt->onUpdate = [this, baseEvt](float) {
                        mVatMesh->set_playing(false);
                        mFlock = true;

                        auto shapeId = std::optional<scene::ShapeId>(_shapeNameToId.find("main")->second);
                        mPipeline->setShapeVisibility(*shapeId, false);

                        for (const auto id : _flockIDs)
                            mPipeline->setShapeVisibility(id, true);
                        for (const auto id : _lanternIDs)
                            mPipeline->setShapeVisibility(id, true);
                    };
                } else if (evt->name == "ResumeVAT") {
                    evt->onUpdate = [this, baseEvt](float) {
                        mVatMesh->set_speed(2.f/3.f);
                        mVatMesh->set_playing(true);
                    };
                } else if (evt->name == "StopFlock") {
                    evt->onUpdate = [this, baseEvt](float) {
                        mFlock = false;
                        for(const auto& id : _lanternIDs)
                            mPipeline->setShapeVisibility(id, false);
                    };
                }
            }
        } else {
            if (auto* pathEvt = dynamic_cast<timeline::PathTrigger*>(evt.get())) {
                evt->onUpdate = [this, id = *shapeId, pathEvt](float) {
                    auto pos = pathEvt->evaluatePosition(mTimeline->currentTime());
                    auto mat = luisa::make_float4x4(
                        luisa::translation(luisa::make_float3(pos.x, pos.y, pos.z)));
                    mPipeline->setShapeTransform(id, mat);
                };
            } else if (auto* colorEvt = dynamic_cast<timeline::ColorTrigger*>(evt.get())) {
                if (evt->name == "albedo") {
                    evt->onUpdate = [this, id = *shapeId, colorEvt](float) {
                        auto color = colorEvt->evaluateColor(mTimeline->currentTime());
                        uint matIdx = mPipeline->getShape(id)->material_layers() & 0xFFu;
                        auto matData = mPipeline->material()->getMaterial(matIdx).data;
                        matData.albedo = luisa::make_float4(color.r, color.g, color.b, color.a);
                        mPipeline->material()->updateMaterialData(matIdx, matData);
                    };
                } else {
                    evt->onUpdate = [this, id = *shapeId, colorEvt](float) {
                        auto color = colorEvt->evaluateColor(mTimeline->currentTime());
                        uint matIdx = mPipeline->getShape(id)->material_layers() & 0xFFu;
                        auto matData = mPipeline->material()->getMaterial(matIdx).data;

                        util::setHDRColorTint(matData.emission, luisa::make_float3(color.r,color.g,color.b));
                        mPipeline->material()->updateMaterialData(matIdx, matData);
                    };
                }
            } else if (auto* curveEvt = dynamic_cast<timeline::CurveTrigger*>(evt.get())) {
                if (evt->name == "lightIntensity") {
                    evt->onUpdate = [this, id = *shapeId, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        uint matIdx = mPipeline->getShape(id)->material_layers() & 0xFFu;
                        auto matData = mPipeline->material()->getMaterial(matIdx).data;
                        util::setHDRColorIntensity(matData.emission, val);
                        mPipeline->material()->updateMaterialData(matIdx, matData);
                    };
                } else if (evt->name == "cameraVisible") {
                    // Light keeps illuminating; only camera-path visibility
                    // (primary rays / mirror reflections / glass replays)
                    // follows the curve. Falls through to full hide otherwise.
                    evt->onUpdate = [this, id = *shapeId, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        mPipeline->setShapeCameraVisibility(id, val > 0.5f);
                    };
                } else {
                    evt->onUpdate = [this, id = *shapeId, curveEvt](float) {
                        float val = curveEvt->evaluateValue(mTimeline->currentTime());
                        mPipeline->setShapeVisibility(id, val > 0.5f);
                    };
                }
            }
        }
    }
#endif
}

void NewTypeEngine::saveScene() {
    nlohmann::json file;
    file["camera"] = mCamera->toJson();
    file["materials"] = mPipeline->captureMaterials();
#if DEMO
    file["dof"] = _dof->toJson();
    file["motionblur"] = _mb->toJson();
    if (_bloom) file["bloom"] = _bloom->toJson();
    if (_ca)    file["ca"]    = _ca->toJson();
#elif NT_ALLOW_RASTER_FEATURES
    if(_pc) file["pc"] = _pc->toJson();
#endif
    if (_fxaa)  file["fxaa"]  = _fxaa->toJson();

    try {
        auto path = app::getAssetPath("") / "scene.json";
        util::writeJsonWithBackup(path, file);
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
            if (config.contains("materials"))
                mPipeline->applyMaterials(config["materials"]);
#if DEMO
            _dof->load(config["dof"]);
            _mb->load(config["motionblur"]);
            if (_bloom && config.contains("bloom")) _bloom->load(config["bloom"]);
            if (_ca && config.contains("ca"))       _ca->load(config["ca"]);
#endif
            if (_fxaa && config.contains("fxaa"))   _fxaa->load(config["fxaa"]);
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

        // FXAA at AfterToneMap — AA stand-in while primary jitter/TAA is off.
        // Enabled by default; runs only in RGBA8 (tonemapped) output mode.
        auto fxaa = std::make_unique<feature::FxaaFeature>();
        _fxaa = fxaa.get();
        mPipeline->addFeature(std::move(fxaa));

#if NT_ENABLE_EDITOR
        // Gizmo overlay AFTER FXAA (FIFO within AfterToneMap): editor debug
        // drawing composited on top of the AA'd final image. Registering a
        // second AfterToneMap feature intentionally disables FXAA's tonemap
        // temp-route fast path (falls back to its copy path) — editor-only
        // cost; production compiles this out.
        auto gizmo = std::make_unique<feature::GizmoFeature>();
        _gizmo = gizmo.get();
        mPipeline->addFeature(std::move(gizmo));
#endif
    }
}

void NewTypeEngine::updatePipeline() {
    // Order the scene phase against the previous frame's render tail BEFORE
    // any scene dispatch: TetSolve transform rows and deformable vertex
    // uploads ride the compute stream, whose in-order wait must precede them
    // (Pipeline::beginUpdate; render-stream shaders still read those buffers).
    mPipeline->beginUpdate();

    // Timeline evaluates here — callbacks mutate transforms/materials before GPU upload
    if (mTimeline)
        mTimeline->evaluate(mDt);

    // Sample scene per-frame animation (animated transforms are polled by the pipeline)
    if (mScene) mScene->update(mTime, mDt, *mPipeline);

#if NT_ENABLE_MEDIA_PLAYER
    // Phase 6: advance video media clock + decode due frames. Decoder writes
    // to the shared texture under keyed mutex; bridge's Luisa Image<float>
    // sees the new bytes via NativeResourceExt wrap. No explicit sync against
    // Luisa's stream needed for v1 — read happens before beginFrame's stream
    // synchronize, so the wrap is up-to-date before kernels sample it.
    if (mVideoPlayer) {
        if (mVideoPlayer->update()) ++mVideoFrameCount;
    }
#endif
}

void NewTypeEngine::render() {
    auto& profiler = util::Profiler::instance();
    profiler.begin_profiling();

    // All deformable mesh updates on compute stream (same queue as BLAS rebuild)
    // to avoid D3D12 simultaneous-access errors across command queues.
#if DEMO
    if(mTick) mVatMesh->update(*mPipeline, mDt);
#endif


#if NT_ALLOW_RASTER_FEATURES
    if (_pc && mTick) {
        util::CpuScopedTimer _cpu_PC_update("PC/update");
        profiler.set_pass("PC/update");
        mPipeline->computeStream() << core::ShaderManager::instance().shader(
                _pcUpdateShader,
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

#if NT_ENABLE_TIMELINE_EDITOR
    // Push editor gizmos before the pipeline render — the GizmoFeature at
    // AfterToneMap drains the stack within the same frame's render.
    if (mTimeline && mTimeline->showGizmos())
        mTimeline->drawEditor();
#endif

    // Submit current frame's GPU work (no sync at end — deferred to next beginFrame)
    mPipeline->render(*renderer, mCamera->cam(), core::Pipeline::DebugTag::None, mDt);

    // Display previous frame (already synced in beginFrame)
    renderer->endFrame();
    profiler.end_profiling();

    if (mPerfMode) _perfCollectFrame();

    // --accum/--shot export harness: once the target frame count has been
    // accumulated, save the render-res frame to the requested path and quit.
    // (updateRecorder() consumes the pending capture immediately on the Dx12
    // path — the blocking download then quits before the next frame.)
    if (!mShotPath.empty() && mPipeline->progressiveAccumulation() &&
        mPipeline->progressiveFrameIndex() >= mAccumFrames) {
        CI_LOG_I("Offline export: " << mPipeline->progressiveFrameIndex()
                 << " frames accumulated -> " << mShotPath);
        renderer->recorder()->saveScreenshotTo(mShotPath);
        if (renderer->isDx12Present())
            renderer->updateRecorder();
        mShotPath.clear();
        quit();
    }
}

// Apply a camera projection: camera state + decoupled render resolution +
// jitter reference grid + history resets. Render-size defaults per type
// (equirect/cylindrical 2:1, fisheye square, room rig from its atlas); an
// explicit override wins.
void NewTypeEngine::_applyProjection(util::CameraProjection p,
                                     std::optional<luisa::uint2> resOverride) {
    std::optional<luisa::uint2> res;
    switch (p) {
        case util::CameraProjection::Equirect:
        case util::CameraProjection::Cylindrical:
            res = luisa::uint2{ 2048u, 1024u };
            break;
        case util::CameraProjection::Fisheye:
            res = luisa::uint2{ 1024u, 1024u };
            break;
        case util::CameraProjection::RoomRig: {
            const auto atlas = mCamera->roomRig().atlasSize();
            res = luisa::uint2{ atlas.x, atlas.y };
            break;
        }
        case util::CameraProjection::Perspective:
        default:
            break;
    }
    if (resOverride)
        res = resOverride;

    mCamera->setProjection(p);
    mPipeline->setRenderResolutionOverride(res);
    mCamera->setJitterResolution(res); // jitter at the render pixel grid
    mPipeline->requestProgressiveReset();
    CI_LOG_I("Projection: " << util::camera_projection_name(p)
        << (res ? std::string(" @ ") + std::to_string(res->x) + "x" + std::to_string(res->y)
                : std::string(" (window-derived)")));
}

void NewTypeEngine::_perfCollectFrame() {
    auto& profiler = util::Profiler::instance();
    // Warmup frames only advance the counter (shader caches, histories,
    // temporal accumulation settle); measurement starts after.
    if (mPerfFrame++ < mPerfWarmup) {
        if (mPerfFrame == mPerfWarmup)
            CI_LOG_I("Perf harness: warmup done, measuring " << mPerfFrames << " frames");
        return;
    }
    // Instantaneous fps (1/dt), NOT cinder's lifetime getAverageFps() —
    // startup/JIT frames would permanently bias a lifetime average and
    // confound A/B ordering (first run pays the shader disk-cache misses).
    // Median at finish: one-off mid-run stalls (lazy shader JIT on a
    // specialization flip, OS hiccups) must not drag the result.
    if (mDt > 0.0) mPerfFpsSamples.push_back(1.0 / mDt);
    // luisa::string (EASTL) → std::string for the accumulation maps.
    auto rowKey = [](const luisa::string& s) {
        return std::string(s.data(), s.size());
    };
    // GPU per-pass rows exist only in the validation build (StatsExt rides
    // the validation DLL — the accepted GPU-timing workflow); otherwise the
    // CSV still carries CPU pass timings + fps.
    for (auto& kv : profiler.last_gpu_stats()) {
        for (auto& item : kv.second.stream_scopes) {
            if (item->name == "Unknown") continue;
            float dt = (item->finished_time - item->start_time) / 10.f;
            mPerfGpuSum[rowKey(item->name)] += dt;
            mPerfGpuTotalSum += dt;
        }
    }
    for (auto& kv : profiler.cpu_timings()) {
        mPerfCpuSum[rowKey(kv.first)] += kv.second;
        mPerfCpuTotalSum += kv.second;
    }
    if (mPerfFrame - mPerfWarmup >= mPerfFrames) _perfFinish();
}

void NewTypeEngine::_perfFinish() {
    const uint32_t n = std::max(mPerfFrames, 1u);
    std::vector<std::pair<std::string, double>> rows(mPerfGpuSum.begin(), mPerfGpuSum.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    std::ofstream out(mPerfOut);
    out << "section,pass,avg_ms\n";
    for (auto& [name, sum] : rows) out << "gpu," << name << "," << (sum / n) << "\n";
    out << "summary,gpu_total_ms," << (mPerfGpuTotalSum / n) << "\n";
    std::vector<std::pair<std::string, double>> cpu(mPerfCpuSum.begin(), mPerfCpuSum.end());
    std::sort(cpu.begin(), cpu.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (auto& [name, sum] : cpu) out << "cpu," << name << "," << (sum / n) << "\n";
    out << "summary,cpu_total_ms," << (mPerfCpuTotalSum / n) << "\n";
    {
        auto s = mPerfFpsSamples;
        std::sort(s.begin(), s.end());
        double median = s.empty() ? 0.0 : s[s.size() / 2];
        double mean = 0.0;
        for (double v : s) mean += v;
        if (!s.empty()) mean /= static_cast<double>(s.size());
        out << "summary,fps_avg," << mean << "\n";
        out << "summary,fps_median," << median << "\n";
    }
    out << "summary,scene," << mSceneName << "\n";
    out << "summary,frames," << n << "\n";
    out << "summary,warmup," << mPerfWarmup << "\n";
    out << "summary,orbit," << (mPerfOrbit ? 1 : 0) << "\n";
    // Diagnostic rows (StatsExt state — why GPU rows may be empty).
    {
        auto& prof = util::Profiler::instance();
        size_t scopes = 0, unknown = 0;
        for (auto& kv : prof.last_gpu_stats()) {
            for (auto& item : kv.second.stream_scopes) {
                ++scopes;
                if (item->name == "Unknown") ++unknown;
            }
        }
        out << "summary,diag_streams," << prof.last_gpu_stats().size() << "\n";
        out << "summary,diag_scopes," << scopes << "\n";
        out << "summary,diag_unknown," << unknown << "\n";
    }
    out.close();
    CI_LOG_I("Perf harness: wrote " << mPerfOut
             << " - fps_median " << (mPerfFpsSamples.empty() ? 0.0 :
                 mPerfFpsSamples[mPerfFpsSamples.size() / 2])
             << ", gpu_total_ms " << (mPerfGpuTotalSum / n));
    // --perf-shot: one-shot capture at the end of the measured window (the
    // pending flag set here is consumed by recorder->update() later in this
    // same draw()) — same-frame-count captures for A/B image comparison.
    if (mPerfShot) {
        renderer->recorder()->saveScreenshot();
        // Dx12 present runs updateRecorder() BEFORE render(), so nothing would
        // consume the pending flag — quit() skips the next draw(). Consume it
        // here (one blocking download, then quit).
        if (renderer->isDx12Present())
            renderer->updateRecorder();
    }
    quit();
}

#if NT_DEBUG_VIZ
void NewTypeEngine::_ensureUiTestImage() {
    if (_uiTestImgInit) return;
    _uiTestImgInit = true;
    constexpr uint kSz = 256u;
    _uiTestImg = renderer->device().create_image<float>(PixelStorage::BYTE4, kSz, kSz);
    std::vector<luisa::uint> px(kSz * kSz);
    for (uint y = 0u; y < kSz; ++y)
        for (uint x = 0u; x < kSz; ++x) {
            uint r = x * 255u / (kSz - 1u);
            uint g = y * 255u / (kSz - 1u);
            uint b = ((x ^ y) & 8u) ? 255u : 0u; // stripe: reads as horizontal when V is correct
            px[y * kSz + x] = r | (g << 8u) | (b << 16u) | (0xffu << 24u);
        }
    core::Renderer::stream() << _uiTestImg.copy_from(px.data());
}
#endif

void NewTypeEngine::drawUi() {
    if (!mDrawUi) return;

    // Projection selector (docs/non_perspective_camera_report.md) — switching
    // re-applies the camera projection + decoupled render resolution and
    // resets accumulation histories.
    {
        ImGui::ScopedWindow scpWin("Projection", true);
        ImGui::SetWindowSize(ivec2(300, 220));
        ImGui::SetWindowPos(ivec2(660, 5));
        static constexpr const char* kProjNames[] =
            { "Perspective", "Equirectangular", "Cylindrical", "Fisheye", "Room Rig" };
        int proj = static_cast<int>(mCamera->projection());
        if (ImGui::Combo("Projection", &proj, kProjNames, 5)) {
            const auto p = static_cast<util::CameraProjection>(proj);
            if (p == util::CameraProjection::RoomRig) {
                auto rig = util::roomRigPreset(mRigPresetArg);
                rig.faceResX = mRigFaceRes;
                rig.faceResY = mRigFaceRes;
                mCamera->setRoomRig(rig);
            }
            _applyProjection(p, std::nullopt);
        }
        if (mCamera->projection() == util::CameraProjection::Fisheye) {
            float fov = mCamera->fisheyeFov();
            if (ImGui::SliderFloat("Fisheye FOV", &fov, 30.0f, 359.0f, "%.0f deg"))
                mCamera->setFisheyeFov(fov);
        }
        if (mCamera->projection() == util::CameraProjection::RoomRig) {
            static constexpr const char* kRigPresets[] = { "4wall", "4wall_floor", "cube" };
            static int rigPreset = 0;
            if (ImGui::Combo("Rig preset", &rigPreset, kRigPresets, 3)) {
                auto rig = util::roomRigPreset(kRigPresets[rigPreset]);
                rig.faceResX = mRigFaceRes;
                rig.faceResY = mRigFaceRes;
                mCamera->setRoomRig(rig);
                _applyProjection(util::CameraProjection::RoomRig, std::nullopt);
            }
            static int faceRes = static_cast<int>(mRigFaceRes);
            if (ImGui::SliderInt("Face res", &faceRes, 256, 2048, "%d")) {
                mRigFaceRes = static_cast<uint32_t>(faceRes);
                auto rig = mCamera->roomRig();
                rig.faceResX = mRigFaceRes;
                rig.faceResY = mRigFaceRes;
                mCamera->setRoomRig(rig);
                _applyProjection(util::CameraProjection::RoomRig, std::nullopt);
            }
        }
        const auto* rnd = renderer.get();
        if (rnd && rnd->renderSizeOverride())
            ImGui::TextDisabled("render %ux%u (window %ux%u, letterboxed)",
                                rnd->renderWidth(), rnd->renderHeight(),
                                rnd->width(), rnd->height());
        if (mCamera->projection() == util::CameraProjection::RoomRig) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "room rig renders raw (per-face denoising\ndeferred); use progressive "
                "accumulation\nfor clean stills");
        } else if (mCamera->projection() != util::CameraProjection::Perspective) {
            ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "upscaler + raster features guarded off;\nreal-time panoramic denoising "
                "active (Phase 2)");
        }
    }

    // ImGui — CPU-only, overlaps with prev frame GPU tail
    {
        ImGui::ScopedWindow scpWin("Engine", true);
        ImGui::SetWindowSize(ivec2(320, 500));
        ImGui::SetWindowPos(ivec2(5, 5));
        mPipeline->drawUi();
        if (_fxaa) _fxaa->drawUi();
#if NT_ENABLE_EDITOR
        if (_gizmo) _gizmo->drawUi();
#endif
#if NT_DEBUG_VIZ
        // ui_image smoke test — verifies SRV slot allocation, format mapping
        // and the V flip on the DX12 present path.
        if (ImGui::CollapsingHeader("UI Texture Test")) {
            _ensureUiTestImage();
            util::ui_image(_uiTestImg, luisa::float2(200.f), true);
        }
#endif
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
#if DEMO
        mVatMesh->drawUi();
#endif
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
    // File log target: captures CI_LOG_* (incl. the terminating CI_LOG_F with the
    // uncaught-exception what()) into a file — windowed apps have no console.
    ci::log::LogManager::instance()->addLogger(
        ci::log::LogManager::instance()->makeLogger<ci::log::LoggerFile>("build/app_log.txt"));
    // Create renderer — change Backend::CUDA to Backend::DirectX to test DX12.
    // --dx launch: hand Cinder's RendererD3d12 over so the Luisa device adopts
    // its ID3D12Device and presentation goes through its swap chain.
    constexpr core::Backend kBackend = core::Backend::DirectX;
    mDx12Renderer = std::dynamic_pointer_cast<ci::app::RendererD3d12>(getRenderer());
    renderer = core::Renderer::create(
        getWindowWidth(), getWindowHeight(), kBackend, false, mDx12Renderer.get());
    renderer->initialize(core::TextureType::Int8);

    if (mDx12Renderer)  mDx12Renderer->setVSyncEnabled(false);
    else                gl::enableVerticalSync(false);
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
        fs::path videoPath = "D:/Commercial Projects/HuarunVisualiser/assets/videos/cover_space.mp4";
        // app::getAssetPath("videos/2026_06_23_01_07_22.mp4");
        if (fs::exists(videoPath)) {
            mVideoPlayer = std::make_unique<media::VideoPlayer>();
            auto* luDevice = static_cast<ID3D12Device*>(core::Renderer::device().native_handle());
            if (mVideoPlayer->open(videoPath, core::Renderer::device(), luDevice)) {
                auto matPool = mPipeline->material();
                mVideoBindlessSlot = static_cast<int>(
                    matPool->register_external_image(mVideoPlayer->image()));
                // Publish prepared video frames at the post-synchronize point
                // in beginFrame — see VideoPlayer::publish_due_frame().
                mPipeline->setVideoPublishHook(
                    [this] { if (mVideoPlayer) mVideoPlayer->publish_due_frame(); });
                CI_LOG_D("[Phase6] video player ready, bindless slot=" << mVideoBindlessSlot);
                mVideoPlayer->set_looping(true);
            }
            else {
                mVideoPlayer.reset();
            }
        }
        else {
            CI_LOG_W("[Phase6] assets/video/test.mp4 not found - video player disabled");
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

    //==========================================================================
    // Perf harness args: --perf <frames> [--perf-out <path>]
    // [--perf-warmup <n>] [--perf-orbit] (perf review R2 item 1)
    //==========================================================================
    auto perfValue = [&](size_t i) -> std::string {
        return (i + 1 < cmdArgs.size() && !cmdArgs[i + 1].empty()
                && cmdArgs[i + 1][0] != '-') ? cmdArgs[i + 1] : std::string{};
    };
    for (size_t i = 0; i < cmdArgs.size(); ++i) {
        if (cmdArgs[i] == "--perf") {
            mPerfMode = true;
            // Perf runs measure the ANIMATED scene: mTick gates scene update
            // (VATMesh / TetCage / deform-state writes) and defaults off, so
            // without this every harness run would time a static scene.
            mTick = true;
            auto v = perfValue(i);
            if (!v.empty()) mPerfFrames = static_cast<uint32_t>(std::stoul(v));
        } else if (cmdArgs[i] == "--perf-out") {
            auto v = perfValue(i);
            if (!v.empty()) mPerfOut = v;
        } else if (cmdArgs[i] == "--perf-warmup") {
            auto v = perfValue(i);
            if (!v.empty()) mPerfWarmup = static_cast<uint32_t>(std::stoul(v));
        } else if (cmdArgs[i] == "--perf-orbit") {
            mPerfOrbit = true;
        } else if (cmdArgs[i] == "--perf-shot") {
            mPerfShot = true;
        } else if (cmdArgs[i] == "--projection") {
            mProjectionArg = perfValue(i);
        } else if (cmdArgs[i] == "--res") {
            auto v = perfValue(i);
            uint32_t w = 0u, h = 0u;
            if (sscanf(v.c_str(), "%ux%u", &w, &h) == 2 && w > 0u && h > 0u)
                mResOverrideArg = luisa::uint2{ w, h };
            else
                CI_LOG_W("Ignored --res '" << v << "' (expected WxH)");
        } else if (cmdArgs[i] == "--rig-preset") {
            mRigPresetArg = perfValue(i);
        } else if (cmdArgs[i] == "--rig-face-res") {
            auto v = perfValue(i);
            if (!v.empty()) mRigFaceRes = std::max(64u, static_cast<uint32_t>(std::stoul(v)));
        } else if (cmdArgs[i] == "--fisheye-fov") {
            auto v = perfValue(i);
            if (!v.empty()) mFisheyeFov = std::stof(v);
        } else if (cmdArgs[i] == "--accum") {
            auto v = perfValue(i);
            if (!v.empty()) mAccumFrames = static_cast<uint32_t>(std::stoul(v));
        } else if (cmdArgs[i] == "--shot") {
            mShotPath = perfValue(i);
        } else if (cmdArgs[i] == "--debug-viz") {
            auto v = perfValue(i);
            if (!v.empty()) mPipeline->setShadeDebugVizMode(std::stoi(v));
        } else if (cmdArgs[i] == "--resolver-param") {
            // Resolver params validation hook (docs/resolver_params_abi_plan.md):
            // --resolver-param <callable> <paramIndex> <value> — applied after
            // buildScene() (params register during the callable DLL load there).
            if (i + 3 < cmdArgs.size()) {
                mPendingResolverParams.emplace_back(
                    cmdArgs[i + 1],
                    static_cast<uint>(std::stoul(cmdArgs[i + 2])),
                    std::stof(cmdArgs[i + 3]));
                i += 3;
            }
        }
    }
    if (mPerfMode) {
        mDrawUi = false;
        CI_LOG_I("Perf harness: " << mPerfFrames << " frames (warmup "
                 << mPerfWarmup << "), orbit=" << mPerfOrbit
                 << ", shot=" << mPerfShot
                 << ", out=" << mPerfOut);
    }

    initGeometries();
#if !DEMO
    // The viz-epoch tag is the build-identity check: if the title doesn't end
    // with it, the running exe predates the current source (DSL kernels and
    // static linking both compile in at startup/build time respectively).
    getWindow()->setTitle("NewTypeEngine - realtime pathtracing with ReSTIR [" + mSceneName + "]"
        + newtype::render::kRelaxVizEpochTag);
#endif
    mPipeline->addEnvMap(Surface32f::create(
        loadImage(app::loadAsset("textures/colorful_studio_4k.hdr"))));
    // Build TLAS + LightSampler
    mPipeline->buildScene();
    // Deferred --resolver-param application (params registered during
    // buildScene's callable DLL load).
    for (const auto& [name, idx, value] : mPendingResolverParams) {
        if (!mPipeline->material()->setResolverParamValue(name, idx, value)) {
            CI_LOG_W("Resolver param apply failed: '" << name << "' idx " << idx);
        }
    }
    mPendingResolverParams.clear();
    initFeatures();
    loadScene();

    // Authored scene viewpoints (e.g. the FBX import camera) apply AFTER
    // loadScene() so they win over a camera saved for a previous scene in
    // assets/scene.json — otherwise the import opens viewed from inside the
    // model. Scenes without an authored camera (all test scenes except the
    // FBX one) are unaffected: loadScene's camera stands.
    mScene->applyCamera(*mCamera);

    //==========================================================================
    // Timeline
    //==========================================================================
    mTimeline = std::make_unique<timeline::Timeline>();
    auto tlPath = app::getAssetPath("timeline.json");
    if (fs::exists(tlPath)) {
        mTimeline->load(tlPath);
        wireTimelineCallbacks();
    }

    // Load saved engine config (parameters, feature toggles; materials now
    // load with the scene above)
    mPipeline->loadConfig(app::getAssetPath("config.json"));
    //mCamera->camUi().disable();

    // Non-perspective projection from --projection (after loadConfig so a
    // saved denoiser/renderScale config cannot fight the override/gates).
    if (!mProjectionArg.empty()) {
        const auto proj = util::camera_projection_from_string(mProjectionArg);
        if (proj) {
            if (*proj == util::CameraProjection::RoomRig) {
                auto rig = util::roomRigPreset(mRigPresetArg);
                rig.faceResX = mRigFaceRes;
                rig.faceResY = mRigFaceRes;
                mCamera->setRoomRig(rig);
            }
            mCamera->setFisheyeFov(mFisheyeFov);
            _applyProjection(*proj, mResOverrideArg);
        } else {
            CI_LOG_W("Unknown --projection '" << mProjectionArg << "' "
                     "(perspective|equirect|cylindrical|fisheye|roomrig)");
        }
    }
    // Offline export harness (--accum): independent of --projection so
    // perspective stills can be accumulated too.
    if (mAccumFrames > 0u) {
        mPipeline->setProgressiveAccumulation(true);
        CI_LOG_I("Offline export: " << mAccumFrames << " accumulated frames"
            << (mShotPath.empty() ? std::string() : " -> " + mShotPath));
    }
    
    auto& sm = newtype::core::ShaderManager::instance();

    // point cloud move shader
#if NT_ALLOW_RASTER_FEATURES
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
    _pcUpdateShader.assign("PointCloudUpdate");
#endif

    mTime = app::getElapsedSeconds();

    // Tear the engine down while the renderer is still alive on window close
    // (see shutdownEngine()). Connected after ImGui::Initialize() so the
    // ImGui D3D12 shutdown runs first.
    getWindow()->getSignalClose().connect([this] { shutdownEngine(); });
}

void NewTypeEngine::cleanup() {
    shutdownEngine();
}

void NewTypeEngine::shutdownEngine() {
    if (_engineDown)
        return;
    _engineDown = true;
    mScene.reset();
    mPipeline.release();
    renderer.reset();
}

void NewTypeEngine::update() {
    // Dx12 present mode: CinderImGui auto-render is disabled there, so the
    // ImGui frame must be opened before any UI code runs this app frame
    // (drawUi() at the end of update() builds inside it; DxPresent renders it
    // into the back buffer command list during draw()).
    if (renderer && renderer->isDx12Present()) {
        static double sLastUiTime = 0.0;
        double now = getElapsedSeconds();
        float uiDt = sLastUiTime > 0.0
            ? static_cast<float>(now - sLastUiTime) : (1.0f / 60.0f);
        sLastUiTime = now;
        renderer->beginUiFrame(uiDt, getWindowWidth(), getWindowHeight());
    }

    // Advance time every frame regardless of mTick so the denoiser (and any
    // other dt-dependent subsystem) always sees a valid delta. Previously
    // mDt was frozen at 0 (or a stale value) when mTick was off, which made
    // the denoiser's EMA alphas go to zero and produced misleading
    // "frozen vs flickering" comparisons when toggling mTick to test physics.
    float currTime = app::getElapsedSeconds();
    mDt = glm::clamp(currTime - mTime, 1.f / 90.f, 1.f / 30.f);

    if (mTick) {
#if DEMO
        if (mFlock) {
            float dt = 30.f * mDt;
            const float zoneRadius = .65f / 4.5f;// mix(120., 80., type);
            const float zoneRadiusSqrd = zoneRadius * zoneRadius;
            const float minThresh = .36f;//mix(.5, .55, type); //.6
            const float maxThresh = .81f;//mix(.9,  .8,  type);
            const float crowdMulti = .4f;//mix(.3, .4,  type);
            const float maxSpeed = .68f;
            const float minSpeed = .12f;

            const float myRadius = .05f; //6.
            const float scale = 1.25f;
            float shrink = glm::clamp((mTimeline->currentTime() - 30.f) / 2.f, .0f, 1.f);
            float h = .25f * (1.f - shrink);

            auto bounceWall = [myRadius, scale, shrink, h](vec3 tempNewPos, vec3& myVel) {
                bool hitWall = false;
                vec3 wallNormal = vec3(0.f);

                float yscale = glm::mix(1.f, glm::max(0.f, 1.f - tempNewPos.y / scale / h), shrink);
                if (tempNewPos.y - myRadius < -shrink * 8.25f * scale) {
                    hitWall = true;
                    wallNormal += vec3(0.f, 1.f, 0.f);
                } else if (tempNewPos.y + myRadius > scale * h) {
                    hitWall = true;
                    wallNormal += vec3(0.f, -1.f, 0.f);
                }

                if (tempNewPos.x - myRadius < yscale * scale * -.53033f) {
                    hitWall = true;
                    wallNormal += vec3(1.f, 0.f, 0.f);
                } else if (tempNewPos.x + myRadius > yscale * scale * .53033f) {
                    hitWall = true;
                    wallNormal += vec3(-1.f, 0.f, 0.f);
                }

                if (tempNewPos.z - myRadius < yscale * scale * -.53033f) {
                    hitWall = true;
                    wallNormal += vec3(0.f, 0.f, 1.f);
                } else if (tempNewPos.z + myRadius > yscale * scale * .53033f) {
                    hitWall = true;
                    wallNormal += vec3(0.f, 0.f, -1.f);
                }

                if (hitWall) {
                    float vdotN = glm::dot(-wallNormal, myVel);
                    myVel += .5f * (wallNormal * vdotN + wallNormal * glm::abs(vdotN));
                }
            };

            for (uint32_t l = 0; l < _lanternIDs.size(); l++) {
                auto& lantern = _lanternData[l]; 
                vec3 pull = lantern.p - vec3(.0f, .125f, .0f);
                vec3 v =
                    glm::normalize(Simplex::curlNoise(2.f * lantern.p + 40.f * float(l), currTime * .15f) * vec3(1.4f,1.f,1.4f)) - util::randFloat() * 2.25f * pull;
                vec3 tmppos = lantern.p + .25f * mDt * v;
                bounceWall(tmppos, v);
                lantern.v = v;
                lantern.p = .25f * mDt * v + lantern.p;

                vec3 dir = glm::normalize(lantern.v + vec3(.000001f, .0f, .0f));
                _lanternTrans[l]->set_trs(lantern.p, glm::quatLookAt(dir, vec3(0.f,1.f,0.f)), vec3(glm::max(.001f, 1.f - shrink)));
            }

            auto tmpCopy = _flockData;
            for (uint32_t i = 0; i < _flockData.size(); i++) {
                auto& my = _flockData[i];
                vec3 acc = vec3(0.f);
                float crowded = 1.f;
                float accMulti = my.leadership * dt * .5f;

                for (uint32_t j = 0; j < _flockData.size(); j++) {
                    if (i == j) continue;
                    auto& onode = tmpCopy[j];

                    vec3 dir = my.p - onode.p;

                    float distSqrd = glm::dot(dir, dir);
                    float dist = glm::sqrt(distSqrd);
                    vec3  dirNorm = dir / glm::max(dist, .000001f);
                    
                    if (distSqrd > 0.f && distSqrd < zoneRadiusSqrd) {
                        float percent = distSqrd / zoneRadiusSqrd + 0.0000001f;
                        crowded += (1.f - percent) * crowdMulti;

                        // IF FISH IS CLOSE, REPEL
                        if (percent < minThresh) {
                            float F = glm::clamp(minThresh / percent - 1.f, .0f, 1.f);
                            acc += dirNorm * F * accMulti;
                        // IF FISH IS IN THE SWEET SPOT, ALIGN	
                        } else if (percent < maxThresh) {
                            float threshDelta = maxThresh - minThresh;
                            float adjustedPercent = (percent - minThresh) / (threshDelta + 0.0000001f);
                            float F = (1.f - (glm::cos(adjustedPercent * 6.28318f) * -0.5f + 0.5f));
                            acc += glm::normalize(onode.v + vec3(.000001f)) * F * accMulti;
                        // IF FISH IS FAR, BUT WITHIN THE ACCEPTABLE ZONE, ATTRACT	
                        } else if (dist < zoneRadius) {
                            float threshDelta = 1.f - maxThresh;
                            float adjustedPercent = (percent - maxThresh) / (threshDelta + 0.0000001f);
                            float F = (1.0f - (glm::cos(adjustedPercent * 6.28318f) * -0.5f + 0.5f)) * accMulti;
                            acc -= dirNorm * F;
                        }
                    }
                }

                for (const auto& lantern : _lanternData) {
                    float minRad = .81f * zoneRadiusSqrd; 
                    float maxRad = 1.96f * zoneRadiusSqrd;

                    vec3 dirToLantern = my.p - lantern.p;
                    float distToLanternSqrd = glm::dot(dirToLantern, dirToLantern);

                    // IF WITHIN THE ZONE, REACT TO THE LANTERN
                    if (distToLanternSqrd > minRad && distToLanternSqrd < maxRad) {
                        float per = (maxRad - minRad) / (distToLanternSqrd - .5f * minRad);
                        acc -= glm::normalize(dirToLantern + vec3(.000001f)) * per * .1275f * dt;
                    // IF TOO CLOSE, MOVE AWAY MORE RAPIDLY
                    } else if (distToLanternSqrd < 16.f * myRadius * myRadius) {
                        acc += 3.25f * glm::normalize(dirToLantern + vec3(.000001f)); 
                    }
                }

                auto myVel = my.v + acc * dt;
                my.crowd -= (my.crowd - crowded) * (0.1f * dt);
                my.crowd = glm::max(0.f, my.crowd);

                float xPull = my.p.x / .53033f;
                float yPull = my.p.y / .25f;
                float zPull = my.p.z / .53033f;
                myVel -= shrink * vec3( -xPull * xPull * zPull,
                                        50.f,
                                        -zPull * zPull * zPull) * 1.7f;
                vec3 tempNewPos = my.p + myVel * mDt;
                bounceWall(tempNewPos, myVel);

                float newMaxSpeed = maxSpeed + my.crowd * 0.015f;
                float velLength = glm::length(myVel);
                if (velLength > newMaxSpeed) {
                    myVel = glm::normalize(myVel) * newMaxSpeed;
                } else if (velLength < minSpeed) {
                    myVel = glm::normalize(myVel + .00000001f) * minSpeed;
                }

                vec3 delta = dt * my.crowd * 0.005f * myVel;
                my.p += delta;
                my.v = myVel;

                vec3 dir   = glm::normalize(my.v + vec3(.000001f, .0f, .0f));
                vec3 right = normalize(cross(dir, vec3(0.f, 1.f, 0.f)));
                vec3 up    = normalize(cross(dir, right));
                auto trans =
                    glm::translate(my.p) * mat4(
                        vec4(right, 0.f),
                        vec4(dir,    0.f),
                        vec4(-up,   0.f),
                        vec4(0.f, 0.f, 0.f, 1.f)) * 
                    glm::scale(vec3(1.f- shrink));
                mPipeline->setShapeTransform(_flockIDs[i], tolc(trans), 
                    shrink > 0.f ? scene::Change::Scale : scene::Change::Affine);
            }
        }
#endif

        updatePipeline();
    }

    mTime = currTime;

    // Harness orbit (perf review R2 item 1): deterministic slow orbit around
    // the camera UI pivot — exercises temporal/spatial reuse, motion vectors
    // and ReLAX TA under motion. Runs before the cam() sync below.
    if (mPerfMode && mPerfOrbit) {
        auto& ciCam = mCamera->ciCam();
        if (mPerfOrbitAngle == 0.f) {
            mPerfOrbitEye0 = ciCam.getEyePoint();
            mPerfOrbitAngle = 1e-6f; // mark initialized (never exactly 0 again)
        } else {
            mPerfOrbitAngle += mDt * 0.3f; // ~0.3 rad/s
        }
        float c = std::cos(mPerfOrbitAngle), s = std::sin(mPerfOrbitAngle);
        glm::vec3 e0 = mPerfOrbitEye0;
        glm::vec3 eye(e0.x * c + e0.z * s, e0.y, -e0.x * s + e0.z * c);
        glm::vec3 target = ciCam.getPivotPoint();
        ciCam.setEyePoint(eye);
        ciCam.lookAt(eye, target);
    }

    const bool cameraMoved = mCamera->update();
    // Camera motion invalidates the offline running average.
    if (cameraMoved && mPipeline->progressiveAccumulation())
        mPipeline->requestProgressiveReset();
#if NT_ENABLE_AUDIO && NT_ENABLE_SPATIAL_AUDIO
    // Spatial audio listener: publish the camera pose (audio thread reads it
    // at block granularity) and advance the spatial system's frame logic
    // (test-tone orbit). Order matters: after Camera::update() so the pose is
    // current, before SoundController::update() so freshly spawned spatial
    // voices see the new listener.
    {
        auto& spatial = newtype::audio::SpatialAudioSystem::get();
        const auto& cd = mCamera->cam();
        spatial.setListener( ci::vec3( cd.position.x, cd.position.y, cd.position.z ),
                             ci::vec3( cd.right.x,   cd.right.y,   cd.right.z ),
                             ci::vec3( cd.up.x,      cd.up.y,      cd.up.z ),
                             ci::vec3( cd.front.x,   cd.front.y,   cd.front.z ) );
        spatial.update( mDt );
    }
#endif
    util::SoundController::get().update();
    drawUi();
}

void NewTypeEngine::draw() {
    if (renderer && renderer->isDx12Present()) {
        // Dx12 present: no GL context. endFrame() inside render() copies the
        // ready display target into the current back buffer and renders ImGui
        // on the same command list; Cinder presents right after draw().
        // Recorder first: its blocking download then overlaps the stream sync
        // Pipeline::beginFrame() performs anyway.
        renderer->updateRecorder();
        render();
        return;
    }

    gl::clear(Color(0.1f, 0.1f, 0.15f));
    //if(mTick)
    render();
    gl::draw(renderer->currentTexture(), getWindowBounds());

    // Capture frame for recording/screenshot
    //if (mTick && mTimeline->isPlaying() && renderer->recorder())
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
    else if (event.getCode() == KeyEvent::KEY_r)
        if (!renderer->recorder()->isRecording()) {
            renderer->recorder()->startRecording(60.f);
        } else if(renderer->recorder()->isRecording()) {
            renderer->recorder()->stopRecording();
        }
    else if (event.getCode() == KeyEvent::KEY_v) {
        // Diagnostic hotkey: cycle Shade Debug Viz 0 -> 9 (W/pdf/M/age)
        // -> 10 (weight_sum/rad/indirect) -> 0.
        int cur = mPipeline->shadeDebugVizMode();
        int m = cur == 0 ? 9 : (cur == 9 ? 10 : 0);
        mPipeline->setShadeDebugVizMode(m);
        CI_LOG_D("Shade Debug Viz mode: " << m);
    }
}

void NewTypeEngine::resize() {
    if (renderer) {
        renderer->resize(getWindowWidth(), getWindowHeight());
    }
    // Aspect follows the RENDER resolution: with a decoupled size override
    // (non-perspective harness) the window ratio would squeeze the frustum —
    // generate_ray's perspective branch consumes cam.aspect. The pano
    // projections don't read aspect, so this only matters for
    // --projection perspective --res WxH.
    if (const auto& override = renderer->renderSizeOverride())
        mCamera->ciCam().setAspectRatio(static_cast<float>(override->x) /
                                        static_cast<float>(override->y));
    else
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

namespace {
// Present-path selection (run before the app instance exists). Default is the
// D3D12 swap-chain path: the Luisa device adopts Cinder RendererD3d12's
// ID3D12Device and the frame handoff is an on-device copy into the back
// buffer (DxPresent) instead of the DX12->GL shared-heap interop. --gl falls
// back to the classic GL present path for fast GL prototypes.
bool sUseDx12Present(const std::vector<std::string>& args) {
    bool useDx12 = true;
    for (const auto& arg : args) {
        if (arg == "--gl") useDx12 = false;
        else if (arg == "--dx") useDx12 = true;
    }
    return useDx12;
}
} // namespace

CINDER_APP(NewTypeEngine, RendererGl, [](App::Settings* settings) {
    settings->setWindowSize (1920, 1080);
    settings->setResizable  (true);
    settings->setTitle      ("NewTypeEngine - realtime pathtracing with ReSTIR");
    settings->setConsoleWindowEnabled();

    if (sUseDx12Present(ci::app::Platform::get()->getCommandLineArgs())) {
        settings->setDefaultRenderer(ci::app::RendererD3d12::create(
            ci::app::RendererD3d12::Options().vsync(false)));
    }
})
