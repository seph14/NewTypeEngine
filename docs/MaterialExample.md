# Material & Texture Examples

Two methods for creating materials with textures in NewTypeEngine.

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
