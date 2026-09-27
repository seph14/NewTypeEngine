# Material & Texture Examples

Two methods for creating materials with textures in NewTypeEngine, plus a
reference of the built-in material types and the 4-layer material system.

---

## Material Types (render/Material.h)

All helpers return a `render::MaterialData` for `pipeline.addMaterial(name, data)`
(or `MaterialPool::createMaterial`). Layered features (clearcoat, sheen,
iridescence, anisotropy, dispersion) can also be packed as modifier layers —
see below.

| Helper | Type | Key parameters |
|--------|------|----------------|
| `make_diffuse` | Diffuse | albedo, roughness |
| `make_conductor` | Conductor | albedo, roughness, (anisotropy) |
| `make_dielectric` | Dielectric | attenuation, ior, roughness, interior_priority, **dispersion** (KHR dispersion; >0 splits wavelengths) |
| `make_thin_dielectric` | ThinDielectric | attenuation, ior, roughness — cheap glass without refraction volume |
| `make_plastic` | Plastic | albedo, roughness, ior, clearcoat, clearcoat_gloss |
| `make_emissive` | Emissive | emission color |
| `make_subsurface` | Subsurface | albedo, roughness, flatness, attenuation, diffuse_trans, ior — thin-wall HK lobe |
| `make_paper` | Subsurface preset | diffuse_trans, attenuation_distance |
| `make_sheen` | Sheen | albedo, roughness, sheen, sheen_tint |
| `make_clearcoat` | Clearcoat modifier | clearcoat, clearcoat_gloss, ior |
| `make_iridescence` | Iridescence modifier | iridescence, ior, thickness, thickness_max (texture-driven when max > 0) |
| `make_anisotropy` | Anisotropy modifier | anisotropic, rotation |
| `make_unlit` | Unlit | albedo — raw emission-free display color |
| `make_fabric` | Fabric | albedo, roughness, fabric, sheen, sheen_tint |

```cpp
// Dispersion example — dispersive glass
uint glass = pipeline.addMaterial("glass",
    render::make_dielectric(luisa::make_float3(1.f), 1.52f, 0.f, 0.f, 0.4f));

// Iridescence example
uint soap = pipeline.addMaterial("soap",
    render::make_iridescence(1.f, 1.3f, 400.f, 800.f));
```

Custom material types (type >= 14) via DLL callables: see
docs/custom_material_callables.md and docs/examples/custom_material_callables.md.

## 4-Layer Material Packing

Every shape carries `_material_layers` — four 8-bit material indices packed
in one uint32: **layer 0 = base BSDF, layers 1–3 = modifiers** (e.g. clearcoat,
sheen, iridescence on top of a diffuse base). `0xFF` disables a layer.

```cpp
auto mesh = scene::MeshShape::create(device, /*layer0=*/diffuseIdx);
mesh->set_layer(1, clearcoatIdx);   // + clearcoat
mesh->set_layer(2, sheenIdx);       // + sheen
mesh->set_layer(3, 0xFFu);          // unused
// or set all four at once: mesh->set_material_layers(packed);
```

Details and lobe combination rules: docs/material_layer.md.

---

## Method 1: Manual Texture Loading

Load individual texture files and create a material with `MaterialTextures`.

```cpp
#include "newtype/render/MaterialPool.h"
#include "newtype/render/TextureConverter.h"

// ... inside setup(), after Pipeline is created ...

auto& device = core::Renderer::device();
auto  matPool = mPipeline->material();  // IMPORTANT: get pool first for stream access

// Create base material data
auto matData = render::make_diffuse(luisa::make_float3(1.f, 1.f, 1.f));

// Load textures — pass &matPool->stream() so upload and bindless update
// happen on the same stream (required by DX12 backend)
render::MaterialTextures textures;
textures.albedo = render::TextureConverter::loadFile(
    app::getAssetPath("textures/wood_albedo.jpg"), device, &matPool->stream());

// Optional: load more texture types
textures.normal = render::TextureConverter::loadFile(
    app::getAssetPath("textures/wood_normal.jpg"), device, &matPool->stream());
textures.rma = render::TextureConverter::loadFile(
    app::getAssetPath("textures/wood_orm.png"), device, &matPool->stream());

// Create material with textures
uint matIdx = matPool->createMaterial("wood", matData, std::move(textures));
```

### Notes

- `TextureConverter::loadFile` accepts `Stream*` as third argument. When provided, uploads on that stream without blocking. When `nullptr`, creates a local stream and synchronizes.
- The `MaterialTextures` struct supports: `albedo`, `normal`, `rma` (R=roughness, G=metallic, B=AO), `emissive`.
- Any texture left unset (default-constructed) is skipped — the material falls back to the flat constant from `MaterialData`.
- Use `matPool->stream()` to ensure image upload and bindless array update are on the same stream. Using a different stream causes a crash in the DX12 backend.

---

## Method 2: Auto-Load from Folder Convention

Use `createMaterialFromFolder` which auto-discovers textures by naming convention.

```cpp
auto& device = core::Renderer::device();
auto  matPool = mPipeline->material();

auto matData = render::make_diffuse(luisa::make_float3(1.f, 1.f, 1.f));

// Loads textures from: assets/textures/wood/wood_albedo.png, wood_normal.png, etc.
uint matIdx = matPool->createMaterialFromFolder(
    "wood",                                    // material name + filename prefix
    app::getAssetPath("textures/wood"),        // folder containing textures
    matData                                    // base MaterialData (texture indices set automatically)
);
```

### Naming Conventions Searched

For a material named `"wood"`, the loader tries these suffixes (in order, first match wins):

| Texture | Suffixes tried |
|---------|---------------|
| **Albedo** | `_albedo`, `_diffuse`, `_color`, `_basecolor`, `_base_color`, (empty) |
| **Normal** | `_normal`, `_nrm`, `_n` |
| **RMA** | `_rma`, `_orm`, `_rmo` (combined) |
| **Roughness** | `_roughness`, `_rough`, `_r` (packed into RMA if no combined) |
| **Metallic** | `_metallic`, `_metal`, `_m` (packed into RMA) |
| **AO** | `_ao`, `_ambient_occlusion`, `_occlusion` (packed into RMA) |
| **Emissive** | `_emissive`, `_emission`, `_emi`, `_e` |

File extensions tried: `.png`, `.jpg`, `.jpeg`, `.tga`, `.bmp`, `.tif`, `.tiff`, `.exr`, `.hdr`

If no combined RMA texture is found, separate roughness/metallic/AO textures are packed into one on the CPU.

---

## Using the Material

Assign the returned index to a mesh:

```cpp
auto mesh = scene::MeshShape::create(device, matIdx);
mesh->load_from(ObjLoader(app::loadAsset("models/wood_table.obj")));
mesh->build(stream);
mPipeline->addShape(std::move(mesh), &transform);
```

The engine handles the rest:
- **Shade shader**: `resolve_surface()` samples the bindless texture array using the material's texture indices and UVs from the vertex buffer
- **Denoiser PreFilter**: Also uses `resolve_surface()` to produce a textured albedo auxiliary image for correct demod/remod
- **Composition**: Remodulates denoised irradiance with the textured albedo
