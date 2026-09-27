# TetCageGeometry — Animated Tetrahedral-Cage Deformation

Massively instanced animated geometry (vegetation, cloth-like sheets) with
**zero per-instance CPU cost**. A `.tetcage` file (built offline by
the standalone `TetCage` tool from
[NewTypeEngine_Toolings](https://github.com/seph14/NewTypeEngine_Toolings)) dissects a rest-pose mesh into
per-tet "piece" micro-meshes plus the tetrahedral lattice ("cage") they were
cut from. Each piece becomes a static, immutable BLAS shared by every copy;
one TLAS instance is created per tet per copy.

Per frame a user deform shader animates the cage vertices on the GPU, and the
engine's TetSolve kernel derives each tet's rest→animated affine from its 4
cage corners **on the device**, writing world = copyWorld · [M|t] straight
into the engine instance-transform buffer. The TLAS copies those rows during
its build (LC fork `Accel::set_transform_buffer_on_update`), so a cage frame
is `[wind kernel → TetSolve → TLAS PREFER_UPDATE refit]`, entirely on the GPU
stream — no readback, no CPU solve, no per-instance `setShapeTransform`.
Target scale: 10⁵–10⁶ animated instances (the test scene runs ~157k at
1024 copies).

Neighboring tets share 3 cage verts, so their affine maps agree exactly on
the shared face — the animated surface stays watertight regardless of what
the deform shader writes.

## Quick Start

```cpp
#include "newtype/scene/TetCageGeometry.h"

auto cage = scene::TetCageGeometry::create(Renderer::device());
cage->load(app::getAssetPath("models/ginkgo/ginkgo0.tetcage"));

// Any number of copies; the world transform composes in front of every
// per-tet affine.
cage->add_copy(translation(0.f, 0.f, 0.f));
cage->add_copy(translation(5.f, 0.f, 0.f));

// Wind shader: a ShaderManager-registered 2-D compute shader
// (signature contract: TetCageGeometry.h, set_deform_shader_id).
cage->set_deform_shader_id("my_cage_deform", 256u);

cage->build(*pipeline, Renderer::stream(), materialId /*, double_sided=true*/);
```

## Per-Frame Update

```cpp
// BEFORE Pipeline::update():
// 1. Push deform params (opaque 64-byte per-copy state; the shader defines
//    the semantics). Uniform wind: one call, one upload command.
scene::TetCageDeformState st{};
st.params[0] = luisa::make_float4(time, strength, freq, 0.f);
cage->set_deform_state_all(st);       // or set_deform_state(copy, st) per copy

// 2. Dispatch wind + TetSolve on the compute stream.
cage->update(*pipeline);

// 3. When the animation stops (frozen/settled), converge prev <- curr once
//    so motion vectors return to zero:
cage->settle(*pipeline);
```

## Cage Deform Shader Contract

Register a 2-D compute shader with EXACTLY these 3 buffer parameters, then
hand its id to `set_deform_shader_id()`:

```cpp
core::ShaderManager::instance().registerShader<2>("my_cage_deform",
    [vertCount](compute::BufferVar<luisa::float3> anim_cage,      // rw: animated verts
                compute::BufferVar<luisa::float3> rest_cage,      // r:  rest verts
                compute::BufferVar<scene::TetCageDeformState> states) {
        UInt copy = dispatch_id().x;
        Var<scene::TetCageDeformState> st = states.read(copy);
        $for(v, dispatch_id().y, vertCount, block_size) {
            Float3 r = rest_cage.read(v);
            anim_cage.write(copy * vertCount + v, /* deformed r */);
        };
    });
```

Dispatch is `.dispatch(copyCount, blockSize)` — x = copy index, y is free for
the shader (grid-stride over the cage verts; each y lane must write disjoint
verts). `vertCount == cage->cage_vertex_count()` — capture it after `load()`,
before registration.

## Fallbacks & Validation

- Rows must form ONE contiguous TLAS run (v1 LC API constraint). Registration
  failure (e.g. another shape added between cage copies) logs and falls back
  to the original CPU path (wind readback + CPU solve + per-instance
  `setShapeTransform`).
- `NT_TETCAGE_CPU_PATH=1` forces the CPU path as an A/B pixel-diff baseline.
- `bake_initial_pose()` bakes the current deformed pose into the INITIAL
  instance matrices (build-time readback only) — useful to separate affine
  errors from upload-churn artifacts.
- Runtime `removeShape` interleaved with an animated cage can show one stale
  matrix frame on the swapped row before re-registration (documented v1
  limitation).

## Reference Scene

`src/tests/TetCageScene.cpp` (`--scene tetcage`) — copy row + analytic wind,
env knobs `NT_TETCAGE_COPIES` / `NT_TETCAGE_ASSET` / `NT_TETCAGE_FREEZE_AFTER`,
and a type-3 procedural A/B reference.
