#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"

#include "newtype/util/Mesh.h"
#include "newtype/render/TextureConverter.h"
#include "newtype/render/MaterialPool.h"
#include "newtype/render/LightSampler.h"
#include "cinder/ObjLoader.h"
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
    core::RenderPtr mRenderer;

    // Camera
    util::CamPtr mCamera;
    // Pipeline (owns scene, G-Buffer, shaders, accumulation)
    core::PipelinePtr mPipeline;

#if NT_ENABLE_PROCEDURAL
    scene::ProcGeomPtr mProcMesh;
#endif

#if NT_ALLOW_RASTER_FEATURES
    feature::PointCloud* _pc = nullptr;
    feature::Trail* _trail = nullptr;
#endif

    float mTime = 0.0f, mDt = 0.f;
    bool  mTick = false, mDrawUi = true;

    std::unordered_map<std::string, scene::ShapeId> _shapeNameToId;
    
    // engine funcs
    void initGeometries();
    void initFeatures();
    void updatePipeline();
    void render();
    void drawUi();

    void saveScene();
    void loadScene();
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
    // Scene geometry
    //==========================================================================
    auto matIdx = matPool->createMaterial("diffuse", render::make_diffuse());

    auto trans = scene::StaticTransform::create();
    auto mesh = scene::MeshShape::create(device, matIdx);
    mesh->load_from(geom::Cube());
    mesh->build(stream);
    auto meshID = mPipeline->addShape(std::move(mesh), trans.get());
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
    
}

void NewTypeEngine::updatePipeline() {

}

void NewTypeEngine::render() {
    auto& profiler = util::Profiler::instance();
    profiler.begin_profiling();

    mPipeline->update(mTime, mDt);
    mPipeline->beginFrame(*mRenderer);
    mPipeline->render(*mRenderer, mCamera->cam(), core::Pipeline::DebugTag::None, mDt);
    mRenderer->endFrame();

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
        if (mRenderer->recorder())
            mRenderer->recorder()->drawUi();
    }
    {
        ImGui::ScopedWindow scpWin("Render", true);
        ImGui::SetWindowSize(ivec2(320, 500));
        ImGui::SetWindowPos(ivec2(330, 5));

        mPipeline->material()->drawUi();

#if NT_ALLOW_RASTER_FEATURES
        if (_pc) _pc->drawUi();
        if (_trail) _trail->drawUi();
#endif
    }
}

void NewTypeEngine::setup() {
    constexpr core::Backend kBackend = core::Backend::DirectX;
    mRenderer = core::Renderer::create(
        getWindowWidth(), getWindowHeight(), kBackend, false);
    mRenderer->initialize(core::TextureType::Int8);

    //gl::enableVerticalSync(false);
    //app::setFrameRate(300.f);

    ImGui::Initialize();
    mCamera = util::Camera::create(getWindow());

    //==========================================================================
    // Create Pipeline (compiles shaders, creates G-Buffer + accumulation images)
    //==========================================================================
    mPipeline = core::Pipeline::create(*mRenderer);

    initGeometries();
    //mPipeline->addEnvMap(Surface32f::create(
    //    loadImage(app::loadAsset("textures/colorful_studio_4k.hdr"))));
    
    // Build TLAS + LightSampler
    mPipeline->buildScene();
    initFeatures();
    loadScene();

    // Load saved config (parameters, materials, feature toggles)
    mPipeline->loadConfig(app::getAssetPath("config.json"));
    //mCamera->camUi().disable();

    mTime = app::getElapsedSeconds();
}

void NewTypeEngine::cleanup() {
    mPipeline.release();
    mRenderer.reset();
}

void NewTypeEngine::update() {
    if (mTick) {
        float currTime = app::getElapsedSeconds();
        mDt = glm::clamp(currTime - mTime, 1.f / 90.f, 1.f / 30.f);

        updatePipeline();
        mTime = currTime;
    }

    mCamera->update();
    drawUi();
}

void NewTypeEngine::draw() {
    gl::clear(Color(0.1f, 0.1f, 0.15f));
    render();
    
    gl::draw(mRenderer->currentTexture(), getWindowBounds());

    // Capture frame for recording/screenshot
    if (mTick && mRenderer->recorder())
        mRenderer->recorder()->update(mRenderer->captureFbo());
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
        mRenderer->recorder()->saveScreenshot();
    else if (event.getCode() == KeyEvent::KEY_u)
        mDrawUi = !mDrawUi;
    else if (event.getCode() == KeyEvent::KEY_c)
        mCamera->camUi().enable(!mCamera->camUi().isEnabled());
}

void NewTypeEngine::resize() {
    if (mRenderer)
        mRenderer->resize(getWindowWidth(), getWindowHeight());
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
    settings->setWindowSize (1440, 1440);
    settings->setResizable  (true);
    settings->setTitle      ("NewTypeEngine - realtime pathtracing with ReSTIR");
    settings->setConsoleWindowEnabled();
})
