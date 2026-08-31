# Shader Generator

Tools for generating hot-reloadable shader projects in NewTypeEngine.

## Installation

Requires Python 3.6+ and Jinja2:

```bash
pip install jinja2
```

---

## Compute Shader Generator

Generates hot-reloadable compute shader projects (Kernel2D) with DLL export interface and ShaderManager integration.

### Create a New Shader

```bash
python tools/shader_generator/shader_generator.py MyShader
```

This creates:
```
runtime_shaders/MyShader/
├── MyShader.vcxproj
├── MyShader.h
├── MyShaderAPI.h
└── MyShader.cpp
```

### Custom Kernel Parameters

```bash
python tools/shader_generator/shader_generator.py MyShader \
    --params "ImageFloat output, ImageUInt seed, Accel accel, CameraData camera, uint frame"
```

### List Registered Shaders

```bash
python tools/shader_generator/shader_generator.py --list
```

### Overwrite Existing Shader

```bash
python tools/shader_generator/shader_generator.py MyShader --force
```

### Use in Code

```cpp
#include "MyShader.h"

// Runtime mode (Debug_Runtime)
ShaderManager::instance().loadShader("MyShader",
    "runtime_shaders/MyShader/MyShader.cpp");
auto& shader = ShaderManager::instance().getShader<MyShaderType>("MyShader");

// Static mode (Debug/Release)
MyShader generator;
ShaderManager::instance().loadShader(generator);
auto& shader = ShaderManager::instance().getShader<MyShaderType>("MyShader");
```

---

## Material Callable Generator

Generates hot-reloadable material callable DLL projects. Each project contains `SurfaceResolveFn` callables that modify `SurfaceData` after texture sampling. Loaded by `CallableDLLLoader` (not ShaderManager).

Key differences from compute shaders:
- Exports `registerMaterialCallables`/`unregisterMaterialCallables` (not `create`/`destroy`)
- Uses `RT_RUNTIME` (not `RT_RUNTIME_DLL`)
- Registers callables via host-provided function pointers
- No `Shader2D<>` type — callables are lambdas that return `SurfaceData`

### Create with Named Callables

```bash
python tools/shader_generator/material_generator.py MyMaterials \
    --callables checkerboard,iridescent,wave
```

This creates:
```
runtime_shaders/MyMaterials/
├── MyMaterials.vcxproj
├── MyMaterials.h
├── MyMaterialsAPI.h
└── MyMaterials.cpp
```

### Create with Placeholder Stubs

```bash
# Generate 5 unnamed callable stubs (callable_1 through callable_5)
python tools/shader_generator/material_generator.py MyMaterials --count 5
```

### List Registered Material Projects

```bash
python tools/shader_generator/material_generator.py --list
```

### Overwrite Existing Project

```bash
python tools/shader_generator/material_generator.py MyMaterials --force
```

### Callable Signature

Each callable has this signature:

```cpp
SurfaceResolveFn = std::function<SurfaceData(
    SurfaceData s,          // resolved surface (albedo, roughness, metallic, etc.)
    Var<MaterialData> mat,  // raw material data
    Float2 uv,              // interpolated UV from vertices
    Float2 screen_uv,       // pixel position / resolution [0,1]
    Float3 wo,              // view direction (outgoing)
    Float time,             // frame_count as float
    const BindlessVar& tex, // texture bindless array
    UInt w, UInt h          // screen dimensions
)>
```

Modifiable `SurfaceData` fields: `albedo`, `emission`, `roughness`, `metallic`, `ior`, `albedo_alpha`, `ao`, `sheen`, `clearcoat`, `iridescence`, `ns` (shading normal), `tangent`, `position`.

### Integration

1. Add the generated `.vcxproj` to the Visual Studio solution
2. Build as `Debug_Runtime` for hot-reload, or `Debug`/`Release` for static linking
3. Callables are registered via `CallableDLLLoader` at runtime

---

## Template Files

Templates are stored in `tools/shader_generator/templates/`:

### Compute Shader Templates
- `shader.vcxproj.j2` - Visual Studio project
- `shader.h.j2` - Interface header
- `shaderAPI.h.j2` - DLL export interface (Shader2D type)
- `shader.cpp.j2` - Shader implementation

### Material Callable Templates
- `material.vcxproj.j2` - Visual Studio project (RT_RUNTIME, no RT_RUNTIME_DLL)
- `material.h.j2` - Minimal header
- `materialAPI.h.j2` - DLL export interface (registerMaterialCallables)
- `material.cpp.j2` - Callable implementations

## Registry Files

- `shaders.json` - Registered compute shader projects
- `materials.json` - Registered material callable projects
