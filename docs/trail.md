# Trail Feature Usage Examples

## Quick Start: Static Trail

```cpp
#include "newtype/feature/Trail.h"

auto& device = core::Renderer::device();
auto& stream = core::Renderer::stream();

// Create trail feature: 1024 max segments, material index 3
auto trail = std::make_unique<feature::Trail>(
    device, 1024u, 3u,
    luisa::float3{-50.f, -50.f, -50.f},  // voxel bounds min
    luisa::float3{ 50.f,  50.f,  50.f},  // voxel bounds max
    256u                                  // voxel resolution
);

// Build point data on CPU — N points define N-1 segments (adjacent format)
luisa::vector<feature::TrailPoint> points;

for (int i = 0; i <= 100; i++) {  // 101 points → 100 segments
    float t = float(i) / 100.f;
    float x = cos(t * 6.f) * 2.f;
    float z = sin(t * 6.f) * 2.f;
    float y = t * 3.f;
    float w = 0.02f * (1.f - t);  // taper from thick to thin
    points.push_back({luisa::make_float4(x, y, z, w)});
}

// Upload and register (must happen BEFORE buildScene)
auto* trailPtr = trail.get();
trailPtr->uploadPoints(stream, points, points.size());
mPipeline->addFeature(std::move(trail));
```

## GPU Compute: Animated Trails

The recommended pattern is to build point data with a compute shader each frame, then upload via `BufferView`.

### 1. Register the compute shader

```cpp
// In setup(), after Pipeline creation:
auto& sm = newtype::core::ShaderManager::instance();

sm.registerShader<1>("TrailUpdate", [&](
    compute::BufferVar<feature::TrailPoint> points,
    compute::UInt point_count,
    compute::Float time
) noexcept {
    set_block_size(256u);
    UInt idx = dispatch_x();
    $if(idx >= point_count) { $return(); };

    // Example: helix that rotates over time
    Float t = cast<Float>(idx) / cast<Float>(point_count - 1u);
    Float angle = t * 6.f + time * 2.f;
    Float radius = 2.f * (1.f - t * 0.5f);
    Float height = t * 4.f;

    Float3 pos = make_float3(
        cos(angle) * radius,
        height,
        sin(angle) * radius
    );

    // Width: thicker at base, thinner at tip
    Float w = 0.03f * (1.f - t);

    points.write(idx, feature::TrailPoint{make_float4(pos, w)});
});
```

### 2. Dispatch each frame

```cpp
// In update():
auto& profiler = util::Profiler::instance();
profiler.set_pass("Trail/Update");
mPipeline->computeStream() << sm.shader<1,
    compute::Buffer<feature::TrailPoint>,
    luisa::uint,
    float>(
        "TrailUpdate",
        _trail->pointBuffer(),
        _trail->pointCount(),
        mTime
    ).dispatch(_trail->pointCount());
```

### 3. Multiple trails with different motion

For multiple independent trails (e.g., particles leaving trails), build all points into a single buffer and update with one dispatch:

```cpp
sm.registerShader<1>("MultiTrailUpdate", [&](
    compute::BufferVar<feature::TrailPoint> points,
    compute::BufferVar<luisa::float4> trail_heads,  // xyz=pos, w=width per trail
    compute::BufferVar<luisa::uint> trail_offsets,  // point offset per trail
    compute::BufferVar<luisa::uint> trail_lengths,  // point count per trail
    compute::UInt trail_count,
    compute::Float time
) noexcept {
    set_block_size(256u);
    UInt idx = dispatch_x();

    // Find which trail this point belongs to
    UInt trail_id = 0u;
    UInt acc = 0u;
    $for(t, trail_count) {
        $if(idx < acc + trail_lengths.read(t)) {
            trail_id = t;
            $break;
        };
        acc += trail_lengths.read(t);
    };
    $if(idx >= acc + trail_lengths.read(trail_id)) { $return(); };

    UInt local_idx = idx - acc;
    UInt pt_count = trail_lengths.read(trail_id);

    Float4 head = trail_heads.read(trail_id);
    Float3 pos = head.xyz();
    Float  base_w = head.w;

    // Each point traces back from head position
    Float frac = cast<Float>(local_idx) / cast<Float>(pt_count);
    Float3 pt_pos = pos - make_float3(0.f, frac * 2.f, 0.f);  // simple downward trail
    Float w = base_w * (1.f - frac);

    points.write(idx, feature::TrailPoint{make_float4(pt_pos, max(w, 0.f))});
});
```

## CPU Upload Pattern

When trails are computed on CPU (e.g., from physics simulation):

```cpp
void updateTrails(Stream& stream) {
    // Rebuild points from simulation state
    luisa::vector<feature::TrailPoint> points;

    for (auto& obj : trailObjects) {
        for (int i = 0; i < obj.positions.size(); i++) {
            float t = float(i) / (obj.positions.size() - 1);
            points.push_back({make_float4(
                obj.positions[i],
                obj.width * (1.f - t)
            )});
        }
    }

    // Stage to GPU
    auto staging = _device.create_buffer<feature::TrailPoint>(points.size());
    stream << staging.copy_from(luisa::span{points});
    _trail->uploadPoints(stream, staging.view(0u, points.size()), points.size());
}
```

## Shadow / DDA Integration

Trails automatically participate in DDA shadow rays — no extra code needed. The Trail feature:

1. **Voxelizes** points into the shared `VoxelGrid` during `onVoxelize()`
2. Pipeline runs **shared dilation** after all `AfterGBuffer` features
3. The shade shader falls back to `dda_march()` when `trace_closest()` misses the TLAS

The `shadowDilation` and `shadowRadiusScale` parameters control shadow thickness:

```cpp
// In drawUi or load():
trail->_shadowDilation = 3u;      // dilate voxels by 3 cells for thicker shadows
trail->_shadowRadiusScale = 5.0f; // inflate voxelized radius by 5x
```

## Vis Encoding

Trails reuse the same `is_point` bit (bit 30) in `gbuf_vis.y` as PointCloud. The deferred shade path treats both identically: envmap-based lighting + DDA shadow via the voxel grid. No shader changes needed when adding Trail alongside PointCloud.

## Custom Shader Hooks

Trail exposes two narrow customization hooks that let you override width computation and shading without subclassing. Set before `addFeature()` or swap at runtime with `recompile()`.

### Width Hook

`TrailWidthFn` replaces the default `lerp(pt0.w, pt1.w, t)` inside the vertex callable. Receives `(Float t, UInt seg_id, const BufferVar<TrailPoint>&)` → returns `Float` width.

```cpp
trail->setCustomWidthFn([](Float t, UInt seg_id,
                           const BufferVar<TrailPoint>& points) -> Float {
    Var<TrailPoint> pt0 = points.read(seg_id);
    Var<TrailPoint> pt1 = points.read(seg_id + 1u);
    Float base_width = lerp(pt0.pos_width.w, pt1.pos_width.w, t);
    // Smooth taper: full width at base, thin at tip
    Float taper = 1.0f - smoothstep(0.6f, 1.0f, t);
    return base_width * taper;
});
```

Width from segment velocity (faster = thinner):

```cpp
trail->setCustomWidthFn([](Float t, UInt seg_id,
                           const BufferVar<TrailPoint>& points) -> Float {
    Var<TrailPoint> pt0 = points.read(seg_id);
    Var<TrailPoint> pt1 = points.read(seg_id + 1u);
    Float base_width = lerp(pt0.pos_width.w, pt1.pos_width.w, t);
    Float seg_len = length(pt1.pos_width.xyz() - pt0.pos_width.xyz());
    Float velocity_scale = 1.0f / (1.0f + seg_len * 10.0f);
    return base_width * velocity_scale;
});
```

### Shade Hook

`TrailShadeFn` replaces the default `env_radiance * albedo` inside the transparent fragment shader. Receives `(Float3 normal, Var<MaterialData> mat, Float3 env_radiance)` → returns `Float3` lit color. The `env_radiance` is pre-computed along the surface normal — use it, ignore it, or combine with other terms.

```cpp
// Emissive glow + environment lighting
trail->setCustomShadeFn([](Float3 normal,
                           Var<render::MaterialData> mat,
                           Float3 env_radiance) -> Float3 {
    return env_radiance * mat.albedo.xyz() * 0.5f + mat.emission * 3.0f;
});
```

Fresnel edge glow:

```cpp
trail->setCustomShadeFn([](Float3 normal,
                           Var<render::MaterialData> mat,
                           Float3 env_radiance) -> Float3 {
    Float fresnel = pow(1.0f - max(dot(normal, make_float3(0.f, 1.f, 0.f)), 0.0f), 3.0f);
    return env_radiance * mat.albedo.xyz() + mat.emission * (1.0f + fresnel * 2.0f);
});
```

### Hot-Reload: Swap Hooks Between Frames

Call `setCustomWidthFn` / `setCustomShadeFn` then `recompile(device)`. The old shader is replaced atomically — next draw call uses the new one. Must be called between frames (not during rendering).

```cpp
// In your app class:
feature::Trail* _trail;
TrailShadeFn _glowShade, _defaultShade;
bool _useGlow = true;

// In setup():
_glowShade = [](Float3 normal, Var<render::MaterialData> mat,
                Float3 env_radiance) -> Float3 {
    Float fresnel = pow(1.0f - max(dot(normal, make_float3(0.f, 1.f, 0.f)), 0.0f), 3.0f);
    return env_radiance * mat.albedo.xyz() + mat.emission * (1.0f + fresnel * 2.0f);
};

_defaultShade = [](Float3 normal, Var<render::MaterialData> mat,
                   Float3 env_radiance) -> Float3 {
    return env_radiance * mat.albedo.xyz();
};

_trail->setCustomShadeFn(_glowShade);
_trail->recompile(device);

// In update() — toggle at runtime:
if (keyPressed('G')) {
    _useGlow = !_useGlow;
    _trail->setCustomShadeFn(_useGlow ? _glowShade : _defaultShade);
    _trail->recompile(device);
}
```

### DLL Hot-Reload

Callable definitions live in a DLL project. When the DLL is rebuilt, the app picks up the new functions and recompiles. Matches the `ShaderManager` pattern used by other runtime shaders.

```cpp
// === runtime_shaders/TrailShaders/TrailShaders.h ===

namespace trail_shaders {

TrailShadeFn getGlowShade() {
    return [](Float3 normal, Var<render::MaterialData> mat,
              Float3 env_radiance) -> Float3 {
        // Tweak these values and rebuild DLL — changes appear live
        float glow_intensity = 2.5f;
        float fresnel_power = 3.0f;
        Float fresnel = pow(1.0f - max(dot(normal, make_float3(0.f, 1.f, 0.f)), 0.0f),
                           fresnel_power);
        return env_radiance * mat.albedo.xyz() * 0.3f
             + mat.emission * glow_intensity * (1.0f + fresnel);
    };
}

TrailWidthFn getPulseWidth() {
    return [](Float t, UInt seg_id,
              const BufferVar<TrailPoint>& points) -> Float {
        Var<TrailPoint> pt0 = points.read(seg_id);
        Var<TrailPoint> pt1 = points.read(seg_id + 1u);
        Float base = lerp(pt0.pos_width.w, pt1.pos_width.w, t);
        float pulse_freq = 8.0f;
        float pulse_amp = 0.3f;
        return base * (1.0f + sin(t * pulse_freq * 6.2832f) * pulse_amp);
    };
}

} // namespace trail_shaders

extern "C" __declspec(dllexport)
void create_trail_hooks(TrailShadeFn& outShade, TrailWidthFn& outWidth) {
    outShade = trail_shaders::getGlowShade();
    outWidth = trail_shaders::getPulseWidth();
}
```

```cpp
// === In your app ===

// Setup: load DLL, get hooks
auto& sm = newtype::core::ShaderManager::instance();
sm.loadShader("TrailShaders", "runtime_shaders/TrailShaders/TrailShaders.cpp");

auto factory = sm.getDllFunction<decltype(&create_trail_hooks)>("create_trail_hooks");
TrailShadeFn shadeFn;
TrailWidthFn widthFn;
factory(shadeFn, widthFn);

_trail->setCustomShadeFn(shadeFn);
_trail->setCustomWidthFn(widthFn);
_trail->recompile(device);

// In update(): ShaderManager detects DLL rebuild
if (sm.update()) {
    // Re-fetch from freshly loaded DLL
    auto newFactory = sm.getDllFunction<decltype(&create_trail_hooks)>("create_trail_hooks");
    newFactory(shadeFn, widthFn);
    _trail->setCustomShadeFn(shadeFn);
    _trail->setCustomWidthFn(widthFn);
    _trail->recompile(device);
}
```

### Clear Hooks

Pass `{}` to revert to built-in defaults, then recompile:

```cpp
_trail->setCustomShadeFn({});   // revert to env_radiance * albedo
_trail->setCustomWidthFn({});   // revert to lerp(pt0.w, pt1.w, t)
_trail->recompile(device);
```

## Config

Enable in `Config.h`:

```cpp
#define NT_ALLOW_RASTER_FEATURES   1   // enables RasterBase + PointCloud + Trail
```

This flag gates the VoxelGrid, DDA shadow fallback in the shade shader, and all `RasterBase`-derived features.
