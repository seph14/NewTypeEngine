# Custom Material Callables — Example Usage

Custom materials use LuisaCompute's `Polymorphic<SurfaceResolver>` for runtime dispatch in GPU kernels. Each material type gets a tag (integer), and at shader compile time a `$switch` is generated over all registered implementations.

## Architecture

```
Pipeline constructor
  └─ Registers built-in IdentitySurfaceResolver × 14 (tags 0-13)
       │
User code ─── p->surfaceResolver().create<CustomResolver>()
       │     └─ Returns tag CustomType, CustomType+1, ...
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
    // checkerTag == render::Material::CustomType (14 with the current built-ins)

    uint pulseTag = mPipeline->surfaceResolver().create<PulseResolver>();
    // pulseTag == 15

    uint wearTag = mPipeline->surfaceResolver().create<WearResolver>();
    // wearTag == 16

    // ==================================================================
    // 3) Create materials using the returned tags
    // ==================================================================
    render::MaterialData checkerMat{};
    checkerMat.type      = checkerTag;   // == Material::CustomType
    checkerMat.albedo    = luisa::make_float3(1.0f, 0.8f, 0.2f);
    checkerMat.roughness = 0.4f;
    uint checkerMatIdx   = mPipeline->addMaterial("checker", checkerMat);

    render::MaterialData pulseMat{};
    pulseMat.type      = pulseTag;       // == Material::CustomType + 1
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

The DLL auto-loads during `buildScene()`. Custom resolvers are registered via the DLL's `registerMaterialCallables` entry point (or `registerMaterialCallables2` — ABI v2, which also declares runtime tuning params; the loader prefers it and falls back to v1 when absent). The host wraps each `SurfaceResolveFn` into a `CustomSurfaceResolver` automatically.

```cpp
// NewTypeEngine.cpp — same setup, but DLL handles registration
void NewTypeApp::setup() {
    mPipeline = core::Pipeline::create(*renderer);
    // Built-ins 0-13 registered in constructor.
    // DLL loaded during buildScene() below — registers customs at
    // tags >= render::Material::CustomType.

    render::MaterialData checkerMat{};
    checkerMat.type    = render::Material::CustomType;  // first DLL-registered resolver
    // (never hardcode 14u — the constant shifts when the engine adds a built-in type)
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

## Runtime Tuning Params (ABI v2 — `registerMaterialCallables2`)

With v1, every value tweak means editing the DLL source, a DLL rebuild, and a
full shader recompile. ABI v2 adds a runtime params channel: the DLL declares
named scalar sliders per callable, the engine hosts their current values in
a small device buffer, and the resolver reads them at dispatch time via
`resolver_params()`. Moving a slider is a 16-byte upload — **no DLL rebuild,
no shader recompile** (values never enter the AST, so kernel hashes stay
stable). Tune effects live from the `MaterialPool` "Resolver Params" UI, or
drive headless validation runs with
`--resolver-param <callable> <index> <value>`.

Take the hologram resolver above and make its line density and tint
tunable:

```cpp
// 1) Declare descriptors: name, min, max, default. Up to 32 scalars
//    per callable; scalars pack 4 per float4 in descriptor order
//    (descriptor j -> block [j/4], component j%4).
static const ResolverParamDesc kHologramParams[] = {
    {"line_density", 10.0f, 200.0f, 100.0f},
    {"scan_speed",    0.0f,   1.0f,   0.1f},
    {"tint_r",        0.0f,   1.0f,   0.1f},
    {"tint_g",        0.0f,   1.0f,   0.8f},
    {"tint_b",        0.0f,   1.0f,   0.9f},
};

// 2) Sentinel for "no params available" (v1 host, or params-buffer
//    overflow): keep hard-coded defaults equal to the def_v's so both
//    paths render the same picture.
static constexpr std::uint32_t kNoParamBase = ~0u;

// 3) The resolver becomes a factory capturing the params BASE by value:
static SurfaceResolveFn make_hologram_resolver(std::uint32_t base) {
    return [base](SurfaceData s, Var<MaterialData> mat, Float2 uv,
                  Float2 screen_uv, Float3 wo, Float time,
                  const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
        Float density = 100.0f, speed = 0.1f;      // fallbacks
        Float3 tint = make_float3(0.1f, 0.8f, 0.9f);
        if (base != kNoParamBase) {
            Float4 p0 = resolver_params(tex, base, 0u);  // block 0
            Float4 p1 = resolver_params(tex, base, 1u);  // block 1
            density = p0.x;  speed = p0.y;
            tint = make_float3(p0.z, p0.w, p1.x);        // rgb across blocks
        }

        Float scan = sin(uv.y * density + time * speed) * 0.5f + 0.5f;
        scan = pow(scan, 8.0f);
        s.albedo = lerp(s.albedo, tint, scan * 0.6f);
        s.emission = s.emission + tint * scan * 2.0f;
        s.roughness = lerp(s.roughness, 0.05f, 0.5f);
        return s;
    };
}

// 4) Export the v2 entry point. paramFn FIRST — it allocates the
//    callable's float4 base and returns it; then registerFn with the
//    base captured in the closure. paramFn may be null (host without
//    params support) — skip registration then, never crash.
extern "C" {
CUSTOM_MATERIAL_API void registerMaterialCallables2(
    CallableRegisterFn registerFn, ParamRegisterFn paramFn, CallableClearFn clearFn) {

    // ... existing v1-style callables unchanged ...

    std::uint32_t holoBase = kNoParamBase;
    if (paramFn) holoBase = paramFn("hologram", kHologramParams, 5u);
    registerFn("hologram", make_hologram_resolver(holoBase));
}
}
```

Lifecycle notes:

- Host capacity is 16 callables × 32 floats; `paramFn` returns `~0u` on
  overflow — treat it as the no-params case.
- Values survive DLL reloads keyed by callable **name**; keep descriptor
  order stable across reloads so indices (and shader hashes) don't shift.
- Editing a descriptor's `def_v` in the DLL source applies on the next
  reload only if that slider was never moved; user-tuned values win.
- A callable registered without `paramFn` keeps hard-coded behavior.

Working example in the tree: `make_glass_blend_resolver` in
`runtime_shaders/CustomMaterialShader/CustomMaterialShader.cpp` (the
cornell blend sphere's `band_center` / `band_width` / `diffuse_rgb` move
live). Full ABI details: docs/custom_material_callables.md and
docs/resolver_params_abi_plan.md.

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

### Custom Types (Material::CustomType .. kMaxTypes-1)

User-registered resolvers. Tags start at `render::Material::CustomType`
(`kFirstCustomMaterialType` in Material.h, currently 14 — derived from the
last built-in `MaterialType`, so it shifts automatically when the engine
adds a built-in) and run through `kMaxTypes-1` (32). Tags are assigned
sequentially by `Polymorphic::create()`. Always reference the constant
instead of the literal when setting `MaterialData::type`. The `$switch` dispatch in `resolve_surface()` generates one GPU `$case` per registered resolver at shader compile time.
