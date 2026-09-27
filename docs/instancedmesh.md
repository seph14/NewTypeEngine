# InstancedMesh — Shared BLAS Instancing

Efficiently render N copies of the same mesh with per-instance transforms. One BLAS, one vertex buffer, N TLAS entries. No shader changes needed — full ReSTIR DI/GI + denoiser works unchanged.

## Quick Start

```cpp
#include "newtype/core/Pipeline.h"
#include "newtype/scene/InstancedMesh.h"

using namespace newtype;
using namespace luisa;

// --- In setup (before buildScene) ---

// 1. Create a prototype mesh (any MeshShape — Cinder geom, OBJ, etc.)
auto sphere = scene::MeshShape::create(Renderer::device(), materialId);
sphere->load_from(ci::geom::Sphere().radius(1.0f).subdivisions(24));

// 2. Register as prototype
auto handle = pipeline.addPrototype(std::move(sphere));

// 3. Create instances with transforms
luisa::vector<float4x4> transforms;
for (uint i = 0; i < 20; ++i) {
    for (uint j = 0; j < 20; ++j) {
        transforms.push_back(
            translation(float(i) * 3.0f - 30.0f, 0.0f, float(j) * 3.0f - 30.0f));
    }
}

luisa::vector<scene::ShapeId> ids;
pipeline.addPrototypeInstances(handle, transforms, ids);

// 4. Build scene
pipeline.buildScene();
// All 400 spheres render with full path tracing
```

## Animated Instances (CPU-driven)

```cpp
// Create InstancedMesh for easy per-frame updates
auto handle = pipeline.addPrototype(std::move(sphere));
auto instMesh = pipeline.createInstancedMesh(handle, 400);

// Set initial transforms
luisa::vector<float4x4> transforms(400);
// ... fill transforms ...
instMesh->set_transforms(transforms);

// Register instances with Pipeline and wire their ids to the batch
luisa::vector<scene::ShapeId> ids;
pipeline.addPrototypeInstances(handle, transforms, ids);
for (uint i = 0; i < ids.size(); ++i) instMesh->set_instance_id(i, ids[i]);
pipeline.buildScene();

// --- Per-frame update ---
void update(float time) {
    luisa::vector<float4x4> newTransforms(400);
    for (uint i = 0; i < 400; ++i) {
        float angle = time * 0.5f + float(i) * 0.1f;
        newTransforms[i] = translation(
            cos(angle) * 10.0f,
            sin(time + i * 0.05f) * 2.0f,
            sin(angle) * 10.0f);
    }
    instMesh->set_transforms(newTransforms);
    // Pipeline::update() flushes the batch automatically (before
    // Geometry::update applies it to the TLAS via batch refit)
}
```

## GPU Compute Transforms — device-owned rows (preferred)

Since the GPU-side TLAS transform upload landed,
a compute shader can own the instance matrices outright: register the rows
with the pipeline, write them from any kernel, and the TLAS copies them from
the device during its build. No readback, no per-instance `setShapeTransform`,
CPU cost per frame is O(1).

```cpp
// After buildScene(): hand the rows to a device writer. Requires the
// instances' TLAS rows to form one contiguous run (they do when the batch is
// registered back-to-back — addShape between the instances breaks it).
luisa::vector<scene::ShapeId> ids;  // from addPrototypeInstances
pipeline.registerGpuTransformRows(luisa::span<const scene::ShapeId>{ids.data(), ids.size()});

// Compile a transform compute shader writing the ENGINE buffers:
uint baseRow = pipeline.geometryTlasRow(ids[0]);
auto transformShader = device.compile<1>([&](
    BufferVar<float4x4> transforms,     // rw: geometry instance transforms
    BufferVar<float4x4> transforms_prev,// rw: previous-frame transforms (motion blur)
    Float time, UInt base, UInt count
) {
    set_block_size(256u);
    UInt idx = dispatch_x();
    $if(idx < count) {
        Float angle = time + cast<float>(idx) * 0.01f;
        Float r = 10.0f + sin(cast<float>(idx) * 0.1f) * 5.0f;
        Float3 pos = make_float3(cos(angle) * r, 0.0f, sin(angle) * r);
        // prev <- old curr first (motion vectors)
        transforms_prev.write(base + idx, transforms.read(base + idx));
        transforms.write(base + idx, translation(pos));
    };
});

// --- Per-frame ---
auto* geom = pipeline.geometry();
stream << transformShader(geom->instance_transform_buffer(),
                          geom->instance_transform_prev_buffer(),
                          time, baseRow, count).dispatch(count);
pipeline.notifyGpuTransformsDirty();  // next update: TLAS copies rows on-device
```

Rules:
- Rows are addressed by TLAS row index (`geometryTlasRow(id)`), which is
  dense — a post-build add/remove reshuffles rows. Watch
  `pipeline.topologyGeneration()` and re-register when it changes (see
  TetCageGeometry::update for the pattern; on any change the registration is
  cleared and rows must be re-resolved).
- `setShapeTransform` on a registered row is rejected (the device writer owns
  the matrix).
- `TetCageGeometry` (docs/examples/tetcage.md) is the packaged version of
  this pattern: wind shader + TetSolve kernel + registered rows.

## GPU Compute Transforms — readback variant (legacy)

The older `InstancedMesh` route writes a private GPU buffer, then reads it
back so the CPU can issue per-instance `set_instance_transform`s. Still works,
but the device-owned path above avoids the 25KB+ GPU→CPU sync each frame:

```cpp
auto handle = pipeline.addPrototype(std::move(sphere));
auto instMesh = pipeline.createInstancedMesh(handle, 400);
pipeline.addPrototypeInstances(handle, initialTransforms, ids);
pipeline.buildScene();

// --- Per-frame ---
stream << transformShader(instMesh->transform_buffer(), time, 400u).dispatch(400)
       << instMesh->apply_gpu_transforms(stream);  // 25KB GPU→CPU readback
```

## Multiple Prototypes

```cpp
// Register several prototypes — each gets its own BLAS
auto sphereHandle = pipeline.addPrototype(
    makeMeshFromGeom(ci::geom::Sphere().radius(0.5f).subdivisions(16), matSphere));
auto cubeHandle = pipeline.addPrototype(
    makeMeshFromGeom(ci::geom::Cube().size(make_vec3(1.0f)), matCube));
auto torusHandle = pipeline.addPrototype(
    makeMeshFromGeom(ci::geom::Torus().radius(1.0f).thickness(0.3f), matTorus));

// Instance each prototype independently
luisa::vector<scene::ShapeId> sphereIds, cubeIds, torusIds;
pipeline.addPrototypeInstances(sphereHandle, sphereTransforms, sphereIds);
pipeline.addPrototypeInstances(cubeHandle,   cubeTransforms,   cubeIds);
pipeline.addPrototypeInstances(torusHandle,  torusTransforms,  torusIds);
pipeline.buildScene();
```

## Single Instance

```cpp
auto handle = pipeline.addPrototype(std::move(sphere));

// Add one instance at a specific position
auto id = pipeline.addPrototypeInstance(handle, translation(5.0f, 0.0f, 3.0f));

// Move it later
pipeline.setShapeTransform(id, translation(10.0f, 0.0f, 3.0f));

pipeline.buildScene();
```

## API Reference

### Pipeline Methods

| Method | Description |
|--------|-------------|
| `addPrototype(unique_ptr<MeshShape>)` | Register a prototype mesh. Returns `PrototypeHandle`. Not added to TLAS. |
| `addPrototypeInstance(handle, transform)` | Add one TLAS instance referencing prototype's BLAS. Returns `ShapeId`. |
| `addPrototypeInstances(handle, transforms, out_ids)` | Batch add N instances. |
| `getPrototype(handle)` | Get prototype `MeshShape*`. |
| `createInstancedMesh(handle, count)` | Create `InstancedMesh` with CPU+GPU transform buffers; flushed automatically by `update()`. |
| `releaseInstancedMesh(mesh)` | Stop flushing a created `InstancedMesh` (call before destroying one early). |
| `geometryTlasRow(id)` | Resolve a `ShapeId` to its dense TLAS row index (~0u if invalid). |
| `registerGpuTransformRows(ids)` | Hand a contiguous run of rows to a device writer (returns false if not one run). |
| `notifyGpuTransformsDirty()` | Mark GPU-owned rows dirty — next update, the TLAS copies them from the device. |
| `topologyGeneration()` | Bumps on post-build add/remove; a change invalidates GPU row registration. |

### InstancedMesh Methods

| Method | Description |
|--------|-------------|
| `set_transforms(span<float4x4>)` | Batch set all transforms from CPU array. Marks dirty. |
| `set_transform(index, float4x4)` | Set a single instance transform. Marks dirty. |
| `set_instance_id(index, id)` | Wire a `ShapeId` from `addPrototypeInstances` to its batch slot. |
| `transform_buffer()` | Get GPU `Buffer<float4x4>` for compute shader writes. |
| `apply_gpu_transforms(stream)` | Read GPU buffer → CPU mirror. Marks dirty. |
| `flush_transform_updates(geom)` | Apply dirty transforms to TLAS via `set_instance_transform()` (automatic for meshes created via `createInstancedMesh`). |
| `count()` | Number of instances. |
| `prototype()` | Prototype `MeshShape*`. |
| `instanceId(index)` | `ShapeId` for a specific instance. |

## Memory Comparison

For 400 instances of a 1500-triangle sphere (~200KB vertex data):

| Approach | BLAS | Vertex Buffers | Total GPU Memory |
|----------|------|---------------|------------------|
| `addShape()` × 400 | 400 | 400 × 200KB | ~80MB |
| Prototype instancing | 1 | 1 × 200KB | ~230KB |

## How It Works

1. `addPrototype()` stores the mesh, builds one BLAS, registers vertex/triangle buffers in the bindless array (2 slots).
2. `addInstance()` creates lightweight TLAS entries referencing the same BLAS handle. Each gets a unique TLAS instance index but shares the same `instance_buffer` bindless slots.
3. When a ray hits any instance, shaders read `instance_buffer[inst_id]` → same `(properties, material_layers, vert_slot, tri_slot)` → same bindless vertex reconstruction. The only difference is the TLAS transform applied by RT cores.
4. Transform updates: CPU array → `set_instance_transform()` × N → `_tlas.build(PREFER_UPDATE)` for fast TLAS refit.
