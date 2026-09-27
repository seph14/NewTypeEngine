# Hot-Reload Shader Examples

NewTypeEngine has two shader-registration paths through `newtype::core::ShaderManager`:

1. **Static registration** (`registerShader`) — compiles a kernel once at startup. Zero overhead, no hot-reload. Used in Debug/Release builds.
2. **DLL hot-reload** (`loadShader(generator, true)`) — shader lives in a separate `.vcxproj` that produces a DLL. A watch thread polls the source file; on change, the manager rebuilds the DLL on the main thread and swaps the compiled kernel in. Only available in the **Debug_Runtime** configuration.

The same call site (`sm.shader<dim, Args...>(name, ...)`) works for both — switching configurations does not require touching dispatch code.

---

## Mode Detection

`RT_RUNTIME` is the preprocessor switch. It is defined only in the `Debug_Runtime` configuration.

```cpp
if (newtype::core::ShaderManager::isRuntimeMode()) {
    // DLL hot-reload is active
} else {
    // Static compilation
}
```

You do not usually need to branch on this at call sites — the manager's templated API collapses both paths.

---

## Quick Start: Static Registration

The simplest path. Define a kernel inline and register it.

```cpp
#include "newtype/core/ShaderManager.h"

using namespace newtype::core;

auto& sm = ShaderManager::instance();

sm.registerShader<1>("PointCloudUpdate", [&](
    compute::BufferVar<luisa::float4> pos_buffer,
    compute::BufferVar<luisa::float4> vel_buffer,
    compute::UInt vertex_count,
    compute::Float dt
) noexcept {
    set_block_size(256u);
    UInt idx = dispatch_x();
    $if(idx >= vertex_count) { $return(); };

    Float4 pos = pos_buffer.read(idx);
    Float4 vel = vel_buffer.read(idx);
    pos_buffer.write(idx, pos + vel * dt);
});
```

Dispatch each frame:

```cpp
auto& sm = ShaderManager::instance();
mPipeline->computeStream() << sm.shader<1,
    compute::Buffer<luisa::float4>,
    compute::Buffer<luisa::float4>,
    uint, float>(
        "PointCloudUpdate",
        _posBuffer,
        _velBuffer,
        _pointCount,
        dt
    ).dispatch(_pointCount);
```

The `Args...` template parameters on `shader<dim, Args...>` must match the **host-side argument types** you pass at the dispatch call site (e.g. `compute::Buffer<T>` for buffers, `uint`/`float` for scalars). Inside the kernel definition, use the corresponding DSL `Var` types (`BufferVar<T>`, `UInt`, `Float`).

---

## Quick Start: DLL Hot-Reload

Use this for any shader you want to edit without restarting the app — typically the path tracer, deferred shade, or any heavy kernel under active iteration.

### 1. Implement `IShaderGenerator<dim>`

```cpp
// runtime_shaders/MyShader/MyShader.h
#include "newtype/core/IShaderGenerator.h"

class MyShader : public newtype::core::IShaderGenerator<2> {
public:
    using ShaderType = compute::Shader<2,
        compute::Image<float>, compute::Image<uint>,
        compute::Accel, util::CameraData, uint>;

    [[nodiscard]] std::string getName() const override { return "MyShader"; }
    [[nodiscard]] std::string getTypeName() const override { return "MyShader"; }

    [[nodiscard]] luisa::unique_ptr<luisa::compute::Resource>
    compile(luisa::compute::Device& device) override;
};
```

The DLL project must export `createMyShader(Device&)` and `destroyMyShader(Resource*)`. See `runtime_shaders/SimpleTestShader/SimpleTestShader.cpp` for a complete reference — the export macros (`SIMPLETEST_API`) and the `#ifndef RT_RUNTIME` guard around the `extern "C"` block are the load-bearing pieces.

### 2. Load with hot-reload enabled

```cpp
// In setup(), after the device is initialized:
auto& sm = newtype::core::ShaderManager::instance();
sm.launch();   // starts the file-watch thread (no-op in static mode)

MyShader generator;
sm.loadShader(generator, /*hotReload=*/true);
```

The manager expects:
- Project:  `runtime_shaders/<Name>Shader/<Name>Shader.vcxproj`
- Output:   `build/Runtime/x64/Debug_Runtime/<Name>Shader.dll`
- Source:   `runtime_shaders/<Name>Shader/<Name>Shader.cpp`

`<Name>` is what `getName()` returns (e.g. `"MyShader"`).

### 3. Dispatch

Same call site as the static path:

```cpp
stream << sm.shader<2,
    compute::Image<float>,
    compute::Image<uint>,
    compute::Accel,
    util::CameraData,
    uint>(
        "MyShader",
        outputImage,
        seedImage,
        accel,
        cameraData,
        frameIndex
    ).dispatch(width, height);
```

### 4. Pump the reload queue every frame

The watch thread only flags changed files. The actual destroy-rebuild-compile happens on the render thread — this guarantees no in-flight stream command holds a stale shader pointer.

```cpp
// In Pipeline::render(), before dispatching shaders:
#ifdef RT_RUNTIME
    ShaderManager::instance().processPendingReloads(_computeStream);
#endif
```

`PipelineRender.cpp` already wires this. If you are integrating ShaderManager into a different loop, you must add the call yourself.

### 5. (Optional) Reset accumulation on reload

```cpp
sm.setReloadCallback([](std::string_view name) {
    // E.g., reset path-tracer accumulation when the kernel changes
    requestAccumReset();
});
```

---

## Custom Material Callables — A Second Hot-Reload Path

Custom material resolvers (`SurfaceResolveFn`) are a separate DLL pathway used for procedural material edits — checkerboard, animated roughness, iridescent tint, etc. They do **not** go through `ShaderManager::loadShader`; they go through `Pipeline`'s callable registry.

```cpp
// runtime_shaders/CustomMaterialShader/CustomMaterialShader.cpp
static SurfaceResolveFn checkerboard_resolver =
    [](SurfaceData s, Var<MaterialData> mat,
       Float2 screen_uv, Float3 wo, Float time,
       const BindlessVar& tex, UInt w, UInt h) -> SurfaceData {
    Float2 uv = s.uv;
    Float scale = 10.0f;
    Float checker = step(0.5f, fract(uv.x * scale))
                  * step(0.5f, fract(uv.y * scale))
                  + step(0.5f, fract((uv.x + 0.5f / scale) * scale))
                  * step(0.5f, fract((uv.y + 0.5f / scale) * scale));
    checker = saturate(checker);
    Float3 dark = make_float3(.05f, 0.05f, 0.9f);
    s.albedo = lerp(dark, s.albedo, checker);
    return s;
};

extern "C" CUSTOM_MATERIAL_API void registerMaterialCallables(
    CallableRegisterFn registerFn, CallableClearFn clearFn) {
    registerFn("checkerboard", checkerboard_resolver);
}
```

The pipeline polls the DLL itself and recompiles any shader that consumed the changed callables:

```cpp
// Already wired in PipelineRender.cpp:
#ifdef RT_RUNTIME
    if (_callableDLL.isLoaded() && _callableDLL.checkAndReload()) {
        _recompileAllShaders();
        requestAccumReset();
    }
#endif
```

You consume a callable from inside a kernel by name:

```cpp
auto resolve = surfaceResolver.get("checkerboard");  // null if not registered
$if(resolve != 0u) {
    surface = resolve(surface, mat, screen_uv, wo, time, tex, w, h);
};
```

Use this path for **material-side** experimentation (albedo/roughness/emission tricks). Use the `IShaderGenerator` path for **kernel-side** experimentation (path tracer, deferred shade, post passes).

---

## Adding a New Runtime Shader DLL

1. Copy `runtime_shaders/TestShader/` as a template — the `.vcxproj` already has the right preprocessor defs (`RT_RUNTIME`, `PATH_TRACER_API` equivalent) and output path.
2. Rename files and the exported `createXxx`/`destroyXxx` functions.
3. Add the new `.vcxproj` to the solution if you want IDE builds, or build it manually:
   ```bash
   msbuild runtime_shaders/MyShader/MyShader.vcxproj \
       /p:Configuration=Debug_Runtime /p:Platform=x64
   ```
4. In app code: implement `IShaderGenerator`, call `sm.loadShader(gen, true)`.

The watch thread polls every 500 ms — saving the `.cpp` file in your editor triggers a rebuild on the next render frame. Compile errors are logged via `CI_LOG_E` and the previous shader stays live.

---

## Threading Notes

- **Watch thread**: only `GetFileTime` / `last_write_time` checks. Never compiles on this thread.
- **Render thread**: `processPendingReloads(stream)` runs the rebuild. It syncs the stream before destroying the old shader so pending GPU work finishes first.
- **DLL lifetime**: the old DLL is unloaded **after** its `destroyXxx` runs, which deletes the compiled `Shader` object → `device->destroy_shader(handle)` runs in normal app context. Destroying from `DLL_PROCESS_DETACH` is unsafe (vtable points into already-unmapped memory).

---

## API Reference

### Registration

| Method | Mode | Description |
|--------|------|-------------|
| `registerShader<dim>(name, def)` | Both | Compile inline; no hot-reload. `def` is a DSL lambda. |
| `loadShader(generator, hotReload=true)` | Runtime | Build DLL, watch source, swap on change. |
| `loadShader(generator, false)` | Both | Behaves like `registerShader` but goes through the generator. |
| `registerCallable<CallableT>(name, c)` | Both | Register a LuisaCompute `Callable` for use inside kernels. |

### Dispatch

| Method | Description |
|--------|-------------|
| `shader<dim, Args...>(name, args...)` | Look up by name, return `(*shader)(args...)`. Searches static then DLL registries. |
| `hasShader(name)` | True if loaded (static or DLL). |
| `shaderNames()` | All registered names. |

### Lifecycle

| Method | Mode | Description |
|--------|------|-------------|
| `launch()` | Runtime | Start the file-watch thread. No-op otherwise. |
| `processPendingReloads(stream)` | Runtime | Per-frame pump. Call from the render thread. |
| `setReloadCallback(cb)` | Runtime | Called after each successful reload. |
| `clear()` | Both | Drop all registered shaders. |

### Diagnostics

| Method | Description |
|--------|-------------|
| `isRuntimeMode()` | True iff `RT_RUNTIME` is defined. |

---

## Common Pitfalls

- **`Args...` mismatch**: the template params on `shader<2, Image<float>, Accel, ...>` must match dispatch call-site types **exactly**, in order. A mismatch compiles (the cast is inside the manager) but throws `std::runtime_error` at dispatch. If you change a kernel signature, update every call site.
- **Forgetting `processPendingReloads`**: the watch thread will log "will reload on main thread" forever but no swap ever happens. Always wire the pump into your render loop in runtime mode.
- **DLL built against the wrong config**: hot-reload requires the **Debug_Runtime** configuration for both the app and the shader DLL. A Debug-built DLL will link but crash on first dispatch because the `Resource` vtable layout differs.
- **Static-allocated shaders in DLLs**: avoid `static Shader s = device.compile(...)` inside the DLL. The destructor runs during `DLL_PROCESS_DETACH` and calls into the device vtable after the device-side teardown has begun. Heap-allocate and delete explicitly in `destroyXxx` — see `SimpleTestShader.cpp:241-256` (`destroySimpleTest`) for the pattern.
