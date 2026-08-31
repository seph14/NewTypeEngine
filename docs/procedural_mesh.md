# ProceduralGeometry — Deformable Mesh Mode

GPU-driven mesh deformation via LuisaCompute's `ProceduralPrimitive` API. Each deformable instance gets its own copy of vertex data in a shared buffer. A compute shader deforms vertices and recomputes AABBs every frame, then the BLAS refits for ray tracing.

Requires `NT_ENABLE_PROCEDURAL=1` in `Config.h`.

## Quick Start

```cpp
#include "newtype/core/Pipeline.h"
#include "newtype/scene/ProceduralGeometry.h"

using namespace newtype;
using namespace luisa;

// 1. Create
auto procGeom = scene::ProceduralGeometry::create(device);

// 2. Register a static mesh topology from a Cinder TriMesh (returns mesh_id)
ci::TriMesh triMesh(ci::geom::Plane().size(ci::vec2(5, 5)).subdivisions(ci::ivec2(32, 32)));
uint mesh_id = procGeom->add_static_mesh(triMesh);

// Or from raw buffers:
// uint mesh_id = procGeom->add_static_mesh(positions, normals, indices, vertex_count);

// 3. Instantiate N copies as deformable (returns first instance index)
uint firstInstance = procGeom->add_deformable_instances(
    mesh_id,       // from add_static_mesh
    20u,           // count
    materialLayers // 4x8-bit packed material indices
);

// 4. Build
procGeom->build(stream);

// 5. Hand off to Pipeline (before buildScene)
pipeline->setProceduralGeometry(std::move(procGeom));
```

`Pipeline::render()` calls `_procGeom->update(stream, time)` automatically each frame, which runs the deformation shader, recomputes AABBs, and refits the BLAS.

## Deformation State

Each deformable instance carries a `ProcDeformState` — 64 bytes of generic user data passed to the deformation shader.

```cpp
struct ProcDeformState {
    float4 params[4]; // 64 bytes — interpret freely
};
```

Set initial states at creation:

```cpp
luisa::vector<scene::ProcDeformState> states(20);
for (uint i = 0; i < 20; ++i) {
    float phase = float(i) * 0.3f;
    states[i].params[0] = make_float4(0.0f,   // time (overridden by update)
                                      0.5f,   // strength
                                      2.0f,   // frequency
                                      phase); // phase offset
}
uint firstInstance = procGeom->add_deformable_instances(
    mesh_id, 20u, materialLayers, states);
```

Override state per-frame (e.g., for wind gusts, interaction):

```cpp
// Per-frame, before Pipeline::render()
auto* procGeom = pipeline->proceduralGeom();
if (procGeom) {
    scene::ProcDeformState state;
    state.params[0] = make_float4(time, gustStrength, windFreq, 0.0f);
    procGeom->set_deform_state(instanceIdx, state);
}
```

`set_deform_state()` is cheap — it marks the instance dirty and uploads the 64-byte struct on the next `update()`.

## Generating Mesh Data

Pass a `ci::TriMesh` directly — positions, normals, and indices are converted automatically:

```cpp
#include "cinder/TriMesh.h"
#include "cinder/geom.h"

ci::TriMesh triMesh(ci::geom::Plane()
    .size(ci::vec2(5.0f, 5.0f))
    .subdivisions(ci::ivec2(32, 32)));

uint mesh_id = procGeom->add_static_mesh(triMesh);
```

Or from any `ci::geom::Source`:

```cpp
uint mesh_id = procGeom->add_static_mesh(
    *ci::TriMesh::create(ci::geom::Sphere().radius(1.0f).subdivisions(24)));
```

### Manual Conversion (Advanced)

If you already have raw `float3`/`Triangle` buffers (e.g., from a custom loader), use the span overload:

```cpp
uint mesh_id = procGeom->add_static_mesh(
    luisa::span<const float3>{positions.data(), positions.size()},
    luisa::span<const float3>{normals.data(), normals.size()},
    luisa::span<const compute::Triangle>{indices.data(), indices.size()},
    vertex_count);
```

## Custom Deformation Shader

The built-in deformation shader implements a simple wind effect. To use a custom deformation, modify `_deformShader` in `ProceduralGeometry::build()`, or subclass `ProceduralGeometry`.

The shader receives per-instance state and base mesh data, writes deformed positions + normals + AABB.

### Shader Signature

```cpp
auto deformShader = device.compile<1>(
    [deformStartIdx]( // capture _deformStartIdx for index mapping
        BufferVar<float3> positions,       // rw: per-instance positions
        BufferVar<float3> normals,         // rw: per-instance normals
        BufferVar<AABB> aabbs,             // w:  output AABBs
        BufferVar<ProcInstanceData> instances, // r: instance data
        BufferVar<ProcDeformState> states,     // r: deformation state
        BufferVar<float3> base_positions,  // r: undeformed base mesh
        BufferVar<uint> base_offsets,      // r: mesh_id → offset in base_positions
        BufferVar<compute::Triangle> indices // r: triangle indices
    ) noexcept {
        UInt deform_idx = dispatch_id().x;
        UInt idx = deformStartIdx + deform_idx;
        Var<ProcInstanceData> inst = instances.read(idx);
        Var<ProcDeformState> state = states.read(deform_idx);

        // Read state params
        Float time_val  = state.params[0].x;
        Float strength  = state.params[0].y;
        Float freq      = state.params[0].z;
        Float phase     = state.params[0].w;

        // Instance layout
        UInt base_offset = base_offsets.read(inst.mesh_id);
        UInt frame_idx   = cast<UInt>(inst.param);     // local index in batch
        UInt pos_base    = inst.packed_offsets & 0xFFFFu;
        UInt idx_base    = inst.packed_offsets >> 16u;

        // Deform vertices
        Float3 aabb_min = make_float3(1e10f);
        Float3 aabb_max = make_float3(-1e10f);

        $for(v, inst.vertex_count) {
            Float3 base_pos = base_positions.read(base_offset + v);

            // -- YOUR DEFORMATION HERE --
            Float bend = strength * sin(time_val * freq + phase + base_pos.y * 2.0f);
            Float3 deformed = make_float3(
                base_pos.x + bend,
                base_pos.y,
                base_pos.z + bend * 0.3f);

            positions.write(pos_base + v, deformed);
            aabb_min = min(aabb_min, deformed);
            aabb_max = max(aabb_max, deformed);
        };

        // Recompute normals (angle-weighted vertex normals)
        $for(v, inst.vertex_count) {
            normals.write(pos_base + v, make_float3(0.0f));
        };
        $for(t, inst.tri_count) {
            auto tri = indices.read(idx_base + t);
            Float3 p0 = positions.read(pos_base + tri.i0);
            Float3 p1 = positions.read(pos_base + tri.i1);
            Float3 p2 = positions.read(pos_base + tri.i2);
            Float3 fn = cross(p1 - p0, p2 - p0);
            Float area = length(fn);
            $if(area > 1e-8f) {
                Float3 n = fn / area;
                // accumulate angle-weighted normals per vertex...
                Float cos_a = dot(normalize(p1 - p0), normalize(p2 - p0));
                Float3 prev0 = normals.read(pos_base + tri.i0);
                normals.write(pos_base + tri.i0,
                    prev0 + n * acos(clamp(cos_a, -1.0f, 1.0f)));
                // ... same for tri.i1, tri.i2
            };
        };
        $for(v, inst.vertex_count) {
            Float3 n = normals.read(pos_base + v);
            Float len = length(n);
            $if(len > 1e-8f) { normals.write(pos_base + v, n / len); };
        };

        // Write AABB
        Var<AABB> aabb;
        aabb.packed_min = { aabb_min.x, aabb_min.y, aabb_min.z };
        aabb.packed_max = { aabb_max.x, aabb_max.y, aabb_max.z };
        aabbs.write(idx, aabb);
    });
```

### Param Layout Convention

| `params[N]` | `.x` | `.y` | `.z` | `.w` |
|-------------|------|------|------|------|
| `[0]` | time | strength | frequency | phase |
| `[1]` | (free) | (free) | (free) | (free) |
| `[2]` | (free) | (free) | (free) | (free) |
| `[3]` | (free) | (free) | (free) | (free) |

The convention above is used by the built-in wind shader. For custom shaders, interpret the 16 floats however you like — just be consistent between `ProcDeformState` writes and shader reads.

## Instance Data Layout

Each deformable instance stores `ProcInstanceData` (48 bytes):

| Field | Meaning |
|-------|---------|
| `type` | `3u` for deformable |
| `material_layers` | 4x8-bit packed material indices |
| `mesh_id` | Index into deform mesh registry |
| `packed_offsets` | hi16 = index buffer offset, lo16 = position buffer offset |
| `param` | Local index within the batch (set by `add_deformable_instances`) |
| `vertex_count` | Vertices per instance |
| `tri_count` | Triangles per instance |
| `frame_count` | Number of instances in this batch |
| `rotation` | Quaternion (unused for deformable, identity) |

## Mixed Primitive Scenes

Deformable meshes coexist with VAT meshes, spheres, and cubes in the same `ProceduralGeometry`:

```cpp
auto procGeom = scene::ProceduralGeometry::create(device);

// Analytical primitives
procGeom->add_sphere(make_float3(0, 1, 0), 0.5f, sphereMatIdx);
procGeom->add_cube(make_float3(3, 0.5f, 0), 0.3f, cubeMatIdx, rotation);

// VAT animated mesh
auto vat = procGeom->add_vat_from_file("models/character.vat");
procGeom->add_instance(vat.mesh_id, charMatIdx, 30.0f);

// Deformable mesh (vegetation, cloth, etc.)
uint grassMesh = procGeom->add_static_mesh(positions, normals, indices, vertCount);
procGeom->add_deformable_instances(grassMesh, 100u, grassMatIdx, initialStates);

// Single build — all share one BLAS
procGeom->build(stream);
pipeline->setProceduralGeometry(std::move(procGeom));
```

During `build()`, instances are stable-partitioned so all type=3 are contiguous. The deform shader dispatches only over `_deformInstanceCount` threads.

## Frame Lifecycle

```
                 setup()
                   │
     ┌─────────────▼──────────────┐
     │  add_static_mesh()         │
     │  add_deformable_instances()│
     │  set_deform_state()        │
     └─────────────┬──────────────┘
                   │
              build(stream)
                   │
     ┌─────────────▼──────────────────┐
     │ • Stable-partition type=3 last  │
     │ • Allocate combined GPU buffers │
     │ • Replicate base mesh × N      │
     │ • Compile deform shader        │
     │ • Run initial deformation      │
     │ • Build ProceduralPrimitive    │
     └─────────────┬──────────────────┘
                   │
            per-frame render()
                   │
     ┌─────────────▼──────────────────┐
     │ update(stream, time)           │
     │ • Upload dirty deform states   │
     │ • Dispatch deform shader       │
     │ • Refit BLAS (PREFER_UPDATE)   │
     └────────────────────────────────┘
```

## Performance Notes

- **One BLAS for all procedural types** — refit is `O(instance_count)`, not `O(triangle_count)`.
- **AABB computation is per-instance** — the deform shader writes one AABB per instance from the vertex loop.
- **Recommended < 1000 triangles per deformable mesh** — the deform shader loops over all vertices and triangles per instance. Higher poly counts work but are slower.
- **Dirty tracking**: `set_deform_state()` only uploads the 64 bytes for that specific instance. The deform shader always runs for all deformable instances (wind changes every frame by default).
- **No CPU readback** — all deformation is GPU-side. CPU only writes `ProcDeformState` params.

## API Reference

### Registration (before `build()`)

| Method | Returns | Description |
|--------|---------|-------------|
| `add_static_mesh(triMesh)` | `uint mesh_id` | Register from a `ci::TriMesh` (auto-converts positions/normals/indices) |
| `add_static_mesh(positions, normals, indices, vertex_count)` | `uint mesh_id` | Register from raw `float3`/`Triangle` spans |
| `add_deformable_instances(mesh_id, count, material_layers, initial_states)` | `uint first_instance_idx` | Create N deformable instances of a mesh |

### State Updates (anytime after `build()`)

| Method | Description |
|--------|-------------|
| `set_deform_state(instance_idx, state)` | Override 64-byte deformation params for one instance |

### Lifecycle

| Method | Description |
|--------|-------------|
| `build(stream)` | Allocate GPU buffers, compile shaders, initial deformation, build BLAS |
| `update(stream, time)` | Per-frame: advance animation, run deform shader, refit BLAS |

### Accessors

| Method | Returns | Description |
|--------|---------|-------------|
| `deform_state_buffer()` | `Buffer<ProcDeformState>&` | GPU deform state for custom shader reads |
| `vat_positions()` | `Buffer<float3>&` | Combined position buffer (VAT + deformable) |
| `vat_normals()` | `Buffer<float3>&` | Combined normal buffer |
| `vat_indices()` | `Buffer<Triangle>&` | Combined index buffer |
| `instance_count()` | `uint` | Total procedural instances (all types) |
| `blas()` | `ProceduralPrimitive&` | Underlying procedural BLAS |

## Ray Tracing Integration

Deformable meshes (type=3) are traced via `intersect_static()` in `ProceduralTrace.h`. This performs standard Möller-Trumbore ray-triangle intersection against the deformed positions — no frame interpolation (unlike VAT type=0). Shading reconstructs normals from the deformed vertex data using the angle-weighted normal that the deform shader computed.

Both `trace_closest()` and `trace_occluded()` handle type=3 automatically. No shader changes are needed — full ReSTIR DI/GI + ReLAX denoiser works unchanged for deformable instances.
