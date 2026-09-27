#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>
#include "cinder/Json.h"
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <memory>
#include <functional>

// Include TextureConverter for MaterialTextures and MaterialTextureLoader
#include "newtype/render/TextureConverter.h"
// Include new MaterialData and MaterialType from Material.h
#include "newtype/render/Material.h"

//==============================================================================
// Forward Declarations
//==============================================================================

namespace newtype::render {
class MaterialPool;

/// Parse MaterialData from a config-style JSON object (the schema written by
/// materialDataToJson and by the external FBXImporter tool). Missing fields
/// keep `d`'s current values — start from a default-constructed MaterialData
/// when creating new materials (see scene::ModelLoader).
void materialDataFromJson(const ci::Json& j, MaterialData& d);
}

//==============================================================================
// LUISA_STRUCT Registration for MaterialData (must be at global scope)
//==============================================================================

LUISA_STRUCT(newtype::render::MaterialData,
              type, albedoTexIdx, normalTexIdx, rmaTexIdx,
              emissiveTexIdx, transmissionTexIdx, anisoTexIdx, meta,
              albedo, emission,
              roughness, metallic, ior, alphacut,
              specular_tint, specular_trans,
              clearcoat, clearcoat_gloss,
              sheen, sheen_tint,
              anisotropic, anisotropic_rot,
              flatness, diffuse_trans,
              iridescence, iridescence_ior,
              attenuation, conductor_k,
              attenuation_distance,
              iridescence_thickness, bsdf_type_override,
              fabric,
              iridescenceTexIdx, iridescence_thickness_max,
              dispersion) {

    //==========================================================================
    // Shader Methods - Material Evaluation
    //==========================================================================

    [[nodiscard]] luisa::compute::Bool has_textures() noexcept {
        return (albedoTexIdx >= 0) | (normalTexIdx >= 0) |
               (rmaTexIdx >= 0) | (emissiveTexIdx >= 0);
    }

    [[nodiscard]] luisa::compute::Float3 get_albedo(
        ::luisa::compute::Float2 uv,
        ::luisa::compute::BindlessArray textures) noexcept {

        luisa::compute::Float3 result;
        $if(albedoTexIdx >= 0) {
            auto tex_sample = textures->tex2d(albedoTexIdx).sample(uv);
            result = albedo.xyz() * tex_sample.xyz();
        }
        $else {
            result = albedo.xyz();
        };
        return result;
    }

    [[nodiscard]] luisa::compute::Float get_albedo_alpha(
        ::luisa::compute::Float2 uv,
        ::luisa::compute::BindlessArray textures) noexcept {

        using luisa::compute::cast;
        luisa::compute::Float result;
        $if(albedoTexIdx >= 0) {
            result = albedo.w * textures->tex2d(cast<uint>(albedoTexIdx)).sample(uv).w;
        }
        $else {
            result = albedo.w;
        };
        return result;
    }

    [[nodiscard]] luisa::compute::Float get_roughness(
        ::luisa::compute::Float2 uv,
        ::luisa::compute::BindlessArray textures) noexcept {

        luisa::compute::Float result;
        $if(rmaTexIdx >= 0) {
            result = textures->tex2d(rmaTexIdx).sample(uv).x;
        }
        $else {
            result = roughness;
        };
        return result;
    }

    [[nodiscard]] luisa::compute::Float get_metallic(
        ::luisa::compute::Float2 uv,
        ::luisa::compute::BindlessArray textures) noexcept {

        luisa::compute::Float result;
        $if(rmaTexIdx >= 0) {
            result = textures->tex2d(rmaTexIdx).sample(uv).y;
        }
        $else {
            result = metallic;
        };
        return result;
    }

    [[nodiscard]] luisa::compute::Float get_ao(
        ::luisa::compute::Float2 uv,
        ::luisa::compute::BindlessArray textures) noexcept {

        luisa::compute::Float result;
        $if(rmaTexIdx >= 0) {
            result = textures->tex2d(rmaTexIdx).sample(uv).z;
        }
        $else {
            result = 1.0f;
        };
        return result;
    }

    [[nodiscard]] luisa::compute::Float3 get_emission(
        ::luisa::compute::Float2 uv,
        ::luisa::compute::BindlessArray textures) noexcept {

        luisa::compute::Float3 result;
        $if(emissiveTexIdx >= 0) {
            result = emission * textures->tex2d(emissiveTexIdx).sample(uv).xyz();
        }
        $else {
            result = emission;
        };
        return result;
    }

    [[nodiscard]] luisa::compute::Bool is_emissive() noexcept {
        using luisa::compute::cast;
        return (cast<int>(emissiveTexIdx) >= 0) |
               ((emission.x > 0.0f) | (emission.y > 0.0f) | (emission.z > 0.0f));
    }

    [[nodiscard]] luisa::compute::Bool is_delta() noexcept {
        // Conductor or Dielectric with zero roughness
        return (type == 2u) | (type == 3u) | (type == 11u);
    }

    [[nodiscard]] luisa::compute::Bool has_normal_map() noexcept {
        return normalTexIdx >= 0;
    }

    // Semantic alias: for Conductor materials, attenuation holds the real part
    // of the complex IOR. Same bits — different meaning per material type.
    [[nodiscard]] luisa::compute::Float3 get_conductor_eta() noexcept {
        return attenuation;
    }
};

//==============================================================================
// Continue namespace
//==============================================================================

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Material (CPU-side material with textures)
//==============================================================================

/**
 * @brief Material with both data and textures (owning)
 *
 * Combines MaterialData with owned textures for managing materials
 * on the CPU side. The pool stores these and provides indices.
 */
struct Material {
    /// First custom material-callback type id (alias of
    /// kFirstCustomMaterialType in Material.h, derived from the last built-in
    /// MaterialType). Reference this — never hardcode 14u — when assigning
    /// MaterialData::type for DLL/user-registered callables, so materials
    /// survive new built-in types (docs/custom_material_callables.md).
    static constexpr uint CustomType = kFirstCustomMaterialType;

    std::string       name;           // Material name for lookup
    MaterialData      data;           // POD data for GPU
    MaterialTextures  textures;       // Owned GPU textures
    uint              poolIndex;      // Index in material pool

    Material() : poolIndex(0) {}
};

//==============================================================================
// MaterialPool
//==============================================================================

/**
 * @brief Centralized material management
 *
 * Manages all materials in the scene, providing:
 * - Material creation and registration
 * - Material lookup by name or index
 * - GPU buffer management for material data
 * - Bindless texture array management
 *
 * Maximum 256 materials (indices fit in uint8_t for layer packing).
 */
class MaterialPool {
public:
    static constexpr uint kMaxMaterials = 256u;

    //==========================================================================
    // Resolver runtime params (docs/resolver_params_abi_plan.md)
    //
    // Custom material callables read per-callable tuning params from a small
    // float4 buffer bound at a RESERVED SLOT of the texture bindless array
    // (the resolver already receives it as `tex`). Values live host-side and
    // upload on change — value edits never touch shader hashes (no DXC).
    //==========================================================================

    /// Bindless slot reserved for the params buffer (textures start at 2 —
    /// slot 1 is the per-instance params buffer below).
    static constexpr uint kResolverParamsBindlessSlot = 0u;
    /// Max registered custom callables with params.
    static constexpr uint kMaxResolverParamCallables = 16u;
    /// Per-callable scalar budget (packed into kResolverParamsPerCallable/4 float4s).
    static constexpr uint kResolverParamsPerCallable = 32u;

    //==========================================================================
    // Per-instance custom data (track B2, docs/vertex-packing-instancing-plan.md)
    //
    // Geometry authors 4-float4 rows (64 B/instance, matching ProcDeformState)
    // keyed by TLAS instance row; custom material callables read them via
    // instance_params(tex, s.instance_index, i) — unbounded per-instance
    // variation without material-pool pressure. The pool hosts the buffer
    // (it owns the texture bindless array the resolver receives as `tex`) at
    // reserved slot 1; it materializes on the first upload, so scenes that
    // never author per-instance data pay only a small zero-filled placeholder
    /// Bindless slot reserved for the per-instance params buffer.
    static constexpr uint kInstanceParamsBindlessSlot = 1u;
    /// float4s per instance (64 B rows).
    static constexpr uint kInstanceParamsPerInstance = 4u;
    /// Placeholder rows bound before any upload (zero-filled; grown to the
    /// full instance count on the first authoring).
    static constexpr uint kInstanceParamsDefaultRows = 256u;

    /// One named scalar tuning param (min/max/default drive the UI slider).
    struct ResolverParamDesc {
        std::string name;
        float min_v;
        float max_v;
        float def_v;
    };

    //==========================================================================
    // Construction
    //==========================================================================

    explicit MaterialPool(
        luisa::compute::Device& device,
        uint maxMaterials = kMaxMaterials,
        uint maxTextures = 4096u);

    ~MaterialPool();

    // Non-copyable
    MaterialPool(const MaterialPool&) = delete;
    MaterialPool& operator=(const MaterialPool&) = delete;

    // Movable
    MaterialPool(MaterialPool&&) noexcept;
    MaterialPool& operator=(MaterialPool&&) noexcept;

    //==========================================================================
    // Material Creation
    //==========================================================================

    [[nodiscard]] uint createMaterial(
        const std::string& name,
        const MaterialData& data,
        MaterialTextures&& textures = {},
        const TextureCompressionSettings& compression = {});

    [[nodiscard]] uint createMaterial(
        const std::string& name,
        const luisa::float3& albedo,
        float roughness = 0.5f,
        float metallic = 0.0f,
        uint type = static_cast<uint>(MaterialType::Diffuse));

    [[nodiscard]] uint createMaterialFromFolder(
        const std::string& name,
        const std::filesystem::path& textureFolder,
        const MaterialData& baseData = {},
        const TextureCompressionSettings& compression = {});

    [[nodiscard]] uint cloneMaterial(
        const std::string& newName,
        uint sourceIndex,
        const MaterialData* modifications = nullptr);

    //==========================================================================
    // Material Access
    //==========================================================================

    [[nodiscard]] const Material& getMaterial(uint index) const;
    [[nodiscard]] Material& getMaterial(uint index);

    [[nodiscard]] const Material& getMaterial(const std::string& name) const;
    [[nodiscard]] Material& getMaterial(const std::string& name);

    [[nodiscard]] uint getIndex(const std::string& name) const;

    [[nodiscard]] bool hasMaterial(const std::string& name) const;
    [[nodiscard]] bool hasMaterial(uint index) const;

    //==========================================================================
    // Material Update
    //==========================================================================

    void updateMaterialData(uint index, const MaterialData& data);
    void updateMaterialTextures(uint index, MaterialTextures&& textures);

    //==========================================================================
    // UI
    //==========================================================================

    void drawUi();

    //==========================================================================
    // Resolver runtime params (see constants block above)
    //==========================================================================

    /// Register (or re-register after a DLL reload) a callable's param set.
    /// Name-keyed and stable across reloads: re-registration keeps current
    /// values, clamping them into the (possibly changed) [min,max] range.
    /// Returns the callable's float4 base in the params buffer — the shader
    /// side reads `resolver_params(tex, base, i)` — or ~0u when the budget
    /// (kMaxResolverParamCallables) is exhausted.
    [[nodiscard]] uint registerResolverParams(
        const std::string& callable,
        const luisa::vector<ResolverParamDesc>& descs);

    /// Set one param value by callable name + param index. Marks the buffer
    /// dirty; the next update() uploads (2KB). Returns false if unknown.
    [[nodiscard]] bool setResolverParamValue(
        const std::string& callable, uint paramIndex, float value);

    /// Re-apply the registered defaults (current descriptor def_v) for one
    /// callable — the UI "Reset" button; also the explicit way to pick up
    /// authored default changes without an engine restart. Returns false if
    /// unknown.
    [[nodiscard]] bool resetResolverParams(const std::string& callable);

    /// Registered callables in registration order (for UI iteration).
    [[nodiscard]] const luisa::vector<std::string>& resolverParamCallables() const noexcept {
        return mResolverParamNames;
    }
    /// Descriptors of one registered callable (nullptr when unknown).
    [[nodiscard]] const luisa::vector<ResolverParamDesc>* resolverParamDescs(
        const std::string& callable) const noexcept;
    /// Current value of one param (def_v when unknown).
    [[nodiscard]] float resolverParamValue(
        const std::string& callable, uint paramIndex) const noexcept;

    /// Commit hook for the params UI: invoked on slider-commit so the owner
    /// Pipeline can requestAccumReset() (MaterialPool cannot reach it).
    void setAccumResetCallback(std::function<void()> cb) noexcept {
        mAccumResetCb = std::move(cb);
    }

    //==========================================================================
    // Per-instance custom data (see constants block; track B2)
    //==========================================================================

    /// Upload full params rows in dense TLAS order (4 float4 per instance —
    /// rows.size() must be a multiple of kInstanceParamsPerInstance). Grows
    /// the capacity-bound buffer (rebind + bindless update on the given
    /// stream) when the instance count exceeds capacity; contents uploads
    /// are plain buffer writes. Called by Geometry alongside
    /// upload_instance_props via Pipeline::update.
    void uploadInstanceParams(
        luisa::compute::Stream& stream,
        luisa::span<const luisa::float4> rows);

    /// True once uploadInstanceParams has run at least once (rows resident
    /// for every TLAS instance). Geometry consults this to keep rows
    /// following swap-and-pop row moves.
    [[nodiscard]] bool instanceParamsResident() const noexcept {
        return mInstanceParamsResident;
    }

    //==========================================================================
    // Config serialization
    //==========================================================================

    ci::Json materialsToJson() const;
    void materialsFromJson(const ci::Json& j);

    //==========================================================================
    // GPU Synchronization
    //==========================================================================

    void update(luisa::compute::Stream& stream);
    void rebuild(luisa::compute::Stream& stream);

    //==========================================================================
    // Light rebuild tracking
    //==========================================================================

    [[nodiscard]] bool lightsNeedRebuild() const noexcept { return mLightsNeedRebuild; }
    void clearLightsNeedRebuild() noexcept { mLightsNeedRebuild = false; }

    //==========================================================================
    // Structural-change tracking (type / bsdf_type_override / alphacut crossing)
    // — drives Geometry's deferred _has_visible_glass recompute via Pipeline bridge
    //==========================================================================

    [[nodiscard]] bool structureDirty() const noexcept { return mStructureDirty; }
    void clearStructureDirty() noexcept { mStructureDirty = false; }

    //==========================================================================
    // External texture registration (Phase 6 — video player integration)
    // Registers a caller-owned Image<float> at the next available bindless slot
    // and returns the slot index. The caller retains ownership of the image.
    // Mirrors the per-texture assignment pattern at MaterialPool.cpp:376-395.
    //==========================================================================

    [[nodiscard]] uint register_external_image(
        luisa::compute::Image<float> const& image,
        luisa::compute::Sampler sampler = luisa::compute::Sampler::linear_linear_mirror());

    //==========================================================================
    // GPU Resource Access (for shader binding)
    //==========================================================================

    [[nodiscard]] luisa::compute::Buffer<MaterialData>& buffer() { return mMaterialBuffer; }
    [[nodiscard]] const luisa::compute::Buffer<MaterialData>& buffer() const { return mMaterialBuffer; }

    // Packed per-material similarity keys {roughness, luminance(F0), luminance(albedo)}
    // for the ReSTIR GI neighbor gates — 12 bytes instead of a 192-byte MaterialData
    // read per comparison. Kept in lockstep with mMaterialBuffer on every upload path.
    [[nodiscard]] luisa::compute::Buffer<luisa::float3>& simKeyBuffer() { return mSimKeyBuffer; }
    [[nodiscard]] const luisa::compute::Buffer<luisa::float3>& simKeyBuffer() const { return mSimKeyBuffer; }

    [[nodiscard]] luisa::compute::BindlessArray& textures() { return mTextureBindless; }
    [[nodiscard]] const luisa::compute::BindlessArray& textures() const { return mTextureBindless; }

    [[nodiscard]] luisa::compute::Stream& stream() { return mStream; }

    [[nodiscard]] uint count() const { return static_cast<uint>(mMaterials.size()); }

    //==========================================================================
    // Defaults
    //==========================================================================

    [[nodiscard]] uint defaultMaterialIndex() const { return mDefaultMaterialIndex; }
    [[nodiscard]] uint errorMaterialIndex() const { return mErrorMaterialIndex; }

private:
    void uploadMaterialTextures(Material& material, luisa::compute::Stream& stream);
    void markDirty(uint index);

    luisa::compute::Device& mDevice;
    luisa::compute::Stream  mStream;

    // Material storage (CPU side)
    luisa::vector<std::unique_ptr<Material>> mMaterials;
    luisa::unordered_map<std::string, uint>  mNameToIndex;

    // GPU resources
    luisa::compute::Buffer<MaterialData>     mMaterialBuffer;
    luisa::compute::Buffer<luisa::float3>    mSimKeyBuffer;
    luisa::compute::BindlessArray            mTextureBindless;

    // Texture management
    uint                mNextTextureSlot = 0;
    luisa::vector<uint> mDirtyMaterials;

    // Resolver runtime params (see public constants block)
    luisa::compute::Buffer<luisa::float4>            mResolverParams;      // 16*8 float4
    luisa::vector<luisa::float4>                     mResolverParamShadow; // host mirror
    bool                                             mResolverParamsDirty = false;
    luisa::vector<std::string>                       mResolverParamNames;  // index -> name
    luisa::unordered_map<std::string, uint>          mResolverParamIndex;  // name -> entry
    struct ResolverParamEntry {
        luisa::vector<ResolverParamDesc> descs;
        luisa::vector<float>             values;   // kResolverParamsPerCallable
    };
    luisa::vector<ResolverParamEntry>                mResolverParamEntries;
    std::function<void()>                            mAccumResetCb;

    // Pack one entry's scalar values into its float4 block of the shadow.
    void _packResolverParamShadow(uint idx, const ResolverParamEntry& entry);

    // Per-instance custom data (track B2): placeholder-bound at construction,
    // materialized by uploadInstanceParams; capacity grows to cover the full
    // instance count once any row is authored.
    luisa::compute::Buffer<luisa::float4> mInstanceParams;
    uint                                  mInstanceParamsCapacityRows = 0u;
    bool                                  mInstanceParamsResident = false;

    // Built-in materials
    uint mDefaultMaterialIndex = 0;
    uint mErrorMaterialIndex   = 0;

    bool mNeedsRebuild = false;
    bool mLightsNeedRebuild = false;
    bool mStructureDirty = false;  // type/bsdf_override/alphacut crossed — Geometry glass recompute needed
};

} // namespace newtype::render
