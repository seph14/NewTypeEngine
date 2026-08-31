# Custom Material Callables — Example Usage

Custom materials use LuisaCompute's `Polymorphic<SurfaceResolver>` for runtime dispatch in GPU kernels. Each material type gets a tag (integer), and at shader compile time a `$switch` is generated over all registered implementations.

## Architecture

```
Pipeline constructor
  └─ Registers built-in IdentitySurfaceResolver × 14 (tags 0-13)
       │
User code ─── p->surfaceResolver().create<CustomResolver>()
       │     └─ Returns tag 14, 15, ...
       │
buildScene()
  └─ DLL loads (Debug_Runtime), registers more resolvers
  └─ All shaders compiled — $switch baked with all tags
```

The dispatch happens inside `resolve_surface()` **after** texture sampling and geometry reconstruction — so custom resolvers have access to `s.position`, `s.ns`, `s.albedo`, etc.

## Static Mode (Debug / Release)

Define a resolver class inheriting `SurfaceResolver`, register it between `Pipeline::create()` and `buildScene()`.

```cpp
// NewTypeEngine.cpp — in your app setup(), after Pipeline::create():

#include "newtype/render/Shading.h"

// ======================================================================
// 1) Define custom resolver classes
// ======================================================================

/// Checkerboard — procedural albedo tiling
class CheckerboardResolver : public render::SurfaceResolver {
public:
    void resolve(render::SurfaceData& s,
                 const Var<render::MaterialData>& material,
                 Float2 uv, Float2 screen_uv, Float3 wo, Float time,
                 const BindlessVar& tex_bindless,
                 UInt screen_w, UInt screen_h) const noexcept override {
        Float scale = 8.0f;
        Float cx = step(0.5f, fract(uv.x * scale));
        Float cy = step(0.5f, fract(uv.y * scale));
        Float checker = abs(cx - cy);  // XOR pattern
        s.albedo = lerp(make_float3(0.02f), s.albedo, checker);
    }
};

/// Pulsing emission — time-driven glow, uses surface position
class PulseResolver : public render::SurfaceResolver {
public:
    void resolve(render::SurfaceData& s,
                 const Var<render::MaterialData>& material,
                 Float2 uv, Float2 screen_uv, Float3 wo, Float time,
                 const BindlessVar& tex_bindless,
                 UInt screen_w, UInt screen_h) const noexcept override {
        Float pulse = sin(time * 0.05f) * 0.5f + 0.5f;
        s.emission = s.emission + make_float3(1.0f, 0.3f, 0.1f) * pulse * 3.0f;
    }
};

/// Proximity wear — uses s.position for geometry-aware effects
class WearResolver : public render::SurfaceResolver {
public:
    void resolve(render::SurfaceData& s,
                 const Var<render::MaterialData>& material,
                 Float2 uv, Float2 screen_uv, Float3 wo, Float time,
                 const BindlessVar& tex_bindless,
                 UInt screen_w, UInt screen_h) const noexcept override {
        Float dist = length(s.position - make_float3(0.0f));
        Float wear = saturate(1.0f - dist * 0.5f);
        s.roughness = lerp(s.roughness, 1.0f, wear);
        s.albedo = lerp(s.albedo, make_float3(0.3f), wear * 0.5f);
    }
};

void NewTypeApp::setup() {
    // ... renderer, camera setup ...
    mPipeline = core::Pipeline::create(*renderer);

    // ==================================================================
    // 2) Register custom resolvers (must happen BEFORE buildScene)
    //    Constructor already registered built-ins 0-13.
    //    create() returns the next sequential tag.
    // ==================================================================
    uint checkerTag = mPipeline->surfaceResolver().create<CheckerboardResolver>();
    // checkerTag == 14

    uint pulseTag = mPipeline->surfaceResolver().create<PulseResolver>();
    // pulseTag == 15

    uint wearTag = mPipeline->surfaceResolver().create<WearResolver>();
    // wearTag == 16

    // ==================================================================
    // 3) Create materials using the returned tags
    // ==================================================================
    render::MaterialData checkerMat{};
    checkerMat.type      = checkerTag;   // tag 14
    checkerMat.albedo    = luisa::make_float3(1.0f, 0.8f, 0.2f);
    checkerMat.roughness = 0.4f;
    uint checkerMatIdx   = mPipeline->addMaterial("checker", checkerMat);

    render::MaterialData pulseMat{};
    pulseMat.type      = pulseTag;       // tag 15
    pulseMat.albedo    = luisa::make_float3(0.5f);
    pulseMat.emission  = luisa::make_float3(0.0f);
    pulseMat.roughness = 0.3f;
    uint pulseMatIdx   = mPipeline->addMaterial("pulse", pulseMat);

    // ==================================================================
    // 4) Create geometry, build scene
    // ==================================================================
    auto& device = core::Renderer::device();
    auto& stream = core::Renderer::stream();

    auto checkerMesh = scene::MeshShape::create(device, checkerMatIdx);
    checkerMesh->load_from(ObjLoader(app::loadAsset("models/room_floor.obj")));
    checkerMesh->build(stream);
    scene::StaticTransform checkerXform;
    mPipeline->addShape(std::move(checkerMesh), &checkerXform);

    mPipeline->buildScene();  // compiles all shaders WITH custom resolvers
}
```

## Runtime Mode (Debug_Runtime)

The DLL auto-loads during `buildScene()`. Custom resolvers are registered via the DLL's `registerMaterialCallables` entry point. The host wraps each `SurfaceResolveFn` into a `CustomSurfaceResolver` automatically.

```cpp
// NewTypeEngine.cpp — same setup, but DLL handles registration
void NewTypeApp::setup() {
    mPipeline = core::Pipeline::create(*renderer);
    // Built-ins 0-13 registered in constructor.
    // DLL loaded during buildScene() below — registers customs at tags 14+.

    render::MaterialData checkerMat{};
    checkerMat.type    = 14u;  // first DLL-registered resolver
    checkerMat.albedo  = luisa::make_float3(1.0f, 0.5f, 0.2f);
    uint checkerMatIdx = mPipeline->addMaterial("checker", checkerMat);

    mPipeline->buildScene();  // loads DLL, compiles shaders
}
```

### Hot-Reload Flow (automatic)

1. Edit `runtime_shaders/CustomMaterials/CustomMaterialShader.cpp`
2. Save — `CallableDLLLoader` detects source change in `render()`
3. MSBuild rebuilds the DLL
4. Polymorphic reset, built-ins re-registered, new DLL resolvers registered
5. All shaders recompiled with updated resolvers
6. Temporal accumulation reset

**No restart needed.**

## Writing a New Resolver (DLL)

Edit `runtime_shaders/CustomMaterials/CustomMaterialShader.cpp`. The DLL uses the `SurfaceResolveFn` lambda interface — the host wraps it into a `CustomSurfaceResolver` automatically.

```cpp
// Add your callable as a static SurfaceResolveFn:
static SurfaceResolveFn my_hologram_resolver =
    [](SurfaceData s, Var<MaterialData> mat, Float2 uv,
       Float2 screen_uv, Float3 wo, Float time,
       const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {

    // Holographic scan lines
    Float scan = sin(uv.y * 100.0f + time * 0.1f) * 0.5f + 0.5f;
    scan = pow(scan, 8.0f);  // sharpen into thin lines

    // Tint blue-green
    s.albedo = lerp(s.albedo, make_float3(0.1f, 0.8f, 0.9f), scan * 0.6f);

    // Add emission along scan lines
    s.emission = s.emission + make_float3(0.0f, 0.5f, 0.6f) * scan * 2.0f;

    // Make it more glossy
    s.roughness = lerp(s.roughness, 0.05f, 0.5f);
    return s;
};

// Then register it in registerMaterialCallables():
extern "C" {
CUSTOM_MATERIAL_API void registerMaterialCallables(
    CallableRegisterFn registerFn, CallableClearFn clearFn) {
    // ... existing callables ...
    registerFn("hologram", my_hologram_resolver);
    printf("[CustomMaterialShader] Registered custom callables\n");
}
}
```

## SurfaceResolver API Reference

### Base Class

```cpp
class SurfaceResolver {
public:
    virtual ~SurfaceResolver() noexcept = default;
    virtual void resolve(
        SurfaceData& s,
        const Var<MaterialData>& material,
        Float2 uv, Float2 screen_uv, Float3 wo, Float time,
        const BindlessVar& tex_bindless,
        UInt screen_w, UInt screen_h) const noexcept;
};
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `s` | `SurfaceData&` | Resolved surface (in/out). Already has textures + geometry. Mutate directly. |
| `material` | `Var<MaterialData>` | Raw material data (type, constants, texture indices) |
| `uv` | `Float2` | Interpolated UV from vertices |
| `screen_uv` | `Float2` | Pixel position / resolution, [0,1] |
| `wo` | `Float3` | View direction (outgoing) |
| `time` | `Float` | `frame_count` as float — for animation |
| `tex_bindless` | `const BindlessVar&` | Texture bindless array — for custom texture sampling |
| `screen_w`, `screen_h` | `UInt` | Screen dimensions |

### SurfaceData Fields (mutable in resolve)

| Field | Type | What to modify |
|-------|------|---------------|
| `albedo` | `Float3` | Base color |
| `emission` | `Float3` | Emissive light output |
| `roughness` | `Float` | PBR roughness [0..1] |
| `metallic` | `Float` | Metalness [0..1] |
| `ao` | `Float` | Ambient occlusion [0..1] |
| `ior` | `Float` | Index of refraction |
| `alpha` | `Float` | Opacity |
| `position` | `Float3` | Vertex-interpolated position (read-only useful) |
| `ns` | `Float3` | Shading normal (possibly perturbed by normal map) |
| `geo_ns` | `Float3` | Geometric normal (never perturbed) |
| `tangent` | `Float3` | Tangent frame direction |
| `sheen`, `clearcoat`, etc. | `Float` | Additional PBR parameters |

### Built-in Types (0–13)

| ID | Name | Description |
|----|------|-------------|
| 0 | Null | Invisible |
| 1 | Diffuse | Lambertian |
| 2 | Conductor | Metallic (GGX) |
| 3 | Dielectric | Glass/transmissive |
| 4 | Plastic | Diffuse + clearcoat |
| 5 | Emissive | Light source |
| 6 | Subsurface | Subsurface scattering |
| 7 | Clearcoat | Clearcoat layer |
| 8 | Sheen | Fabric-like |
| 9 | Anisotropy | Anisotropic GGX |
| 10 | Iridescence | Thin-film interference |
| 11 | ThinDielectric | Thin glass |
| 12 | Unlit | No shading (uses `_pad0` for receiveGI flag) |
| 13 | Fabric | Fabric diffuse |

Built-in types use `IdentitySurfaceResolver` — no-op that passes `SurfaceData` through unchanged.

### Custom Types (14–32)

User-registered resolvers. Up to 19 custom types (14 through 32). Tags are assigned sequentially by `Polymorphic::create()`. The `$switch` dispatch in `resolve_surface()` generates one GPU `$case` per registered resolver at shader compile time.
