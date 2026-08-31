#include "newtype/render/MaterialPool.h"
#include "newtype/render/TextureConverter.h"
#include "newtype/render/MaterialSimilarity.h"
#include <algorithm>
#include "newtype/util/UiHelper.h"
#include "newtype/util/TypeConv.h"
#include "cinder/Log.h"
#include "cinder/CinderImGui.h"
#include "newtype/render/MetalData.h"

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// MaterialPool - Construction/Destruction
//==============================================================================

MaterialPool::MaterialPool(Device& device, uint maxMaterials, uint maxTextures)
    : mDevice(device)
    , mStream(device.create_stream(StreamTag::GRAPHICS)) {

    // Cap materials at 256 (indices must fit in uint8_t for layer packing)
    if (maxMaterials > kMaxMaterials) {
        CI_LOG_W("MaterialPool maxMaterials capped at " << kMaxMaterials
                  << " (requested " << maxMaterials << ")");
        maxMaterials = kMaxMaterials;
    }

    // Create GPU resources
    mMaterialBuffer = device.create_buffer<MaterialData>(maxMaterials);
    mSimKeyBuffer   = device.create_buffer<luisa::float3>(maxMaterials);
    mTextureBindless = device.create_bindless_array(maxTextures);

    // Create built-in materials using factory functions
    MaterialData defaultData = make_diffuse(
        luisa::make_float3(0.8f, 0.8f, 0.8f), 0.5f);
    mDefaultMaterialIndex = createMaterial("__default__", defaultData);

    MaterialData errorData = make_diffuse(
        luisa::make_float3(1.0f, 0.0f, 1.0f), 0.5f);  // Magenta
    mErrorMaterialIndex = createMaterial("__error__", errorData);

    // Initial upload
    rebuild(mStream);
}

MaterialPool::~MaterialPool() = default;

MaterialPool::MaterialPool(MaterialPool&& other) noexcept
    : mDevice(other.mDevice)
    , mStream(std::move(other.mStream))
    , mMaterials(std::move(other.mMaterials))
    , mNameToIndex(std::move(other.mNameToIndex))
    , mMaterialBuffer(std::move(other.mMaterialBuffer))
    , mSimKeyBuffer(std::move(other.mSimKeyBuffer))
    , mTextureBindless(std::move(other.mTextureBindless))
    , mNextTextureSlot(other.mNextTextureSlot)
    , mDirtyMaterials(std::move(other.mDirtyMaterials))
    , mDefaultMaterialIndex(other.mDefaultMaterialIndex)
    , mErrorMaterialIndex(other.mErrorMaterialIndex)
    , mNeedsRebuild(other.mNeedsRebuild) {
}

MaterialPool& MaterialPool::operator=(MaterialPool&& other) noexcept {
    if (this != &other) {
        mStream = std::move(other.mStream);
        mMaterials = std::move(other.mMaterials);
        mNameToIndex = std::move(other.mNameToIndex);
        mMaterialBuffer = std::move(other.mMaterialBuffer);
        mSimKeyBuffer = std::move(other.mSimKeyBuffer);
        mTextureBindless = std::move(other.mTextureBindless);
        mNextTextureSlot = other.mNextTextureSlot;
        mDirtyMaterials = std::move(other.mDirtyMaterials);
        mDefaultMaterialIndex = other.mDefaultMaterialIndex;
        mErrorMaterialIndex = other.mErrorMaterialIndex;
        mNeedsRebuild = other.mNeedsRebuild;
    }
    return *this;
}

//==============================================================================
// Material Creation
//==============================================================================

uint MaterialPool::createMaterial(
    const std::string& name,
    const MaterialData& data,
    MaterialTextures&& textures,
    const TextureCompressionSettings& compression) {

    // Check capacity
    if (mMaterials.size() >= kMaxMaterials) {
        CI_LOG_E("MaterialPool full (" << kMaxMaterials << "), cannot create '" << name << "'");
        return mErrorMaterialIndex;
    }

    // Check for duplicate name
    if (hasMaterial(name)) {
        CI_LOG_W("Material '" << name << "' already exists, returning existing index");
        return getIndex(name);
    }

    if (compression.enableCompression) {
        auto tryCompress = [&](Image<float>& tex, bool shouldCompress, PixelStorage format) {
            // compressTexture self-guards (>= 4x4, block-aligned, HALF4/FLOAT4);
            // pre-filtering 1x1 defaults here just avoids skip-log spam.
            if (shouldCompress && tex.valid() && tex.size().x >= 4u && tex.size().y >= 4u) {
                auto compressed = TextureConverter::compressTexture(tex, mDevice, mStream, format);
                if (compressed.valid()) {
                    tex = std::move(compressed);
                }
            }
        };

        tryCompress(textures.albedo,   compression.compressAlbedo,   PixelStorage::BC7);
        tryCompress(textures.normal,   compression.compressNormal,   PixelStorage::BC7);
        tryCompress(textures.rma,      compression.compressRMA,      PixelStorage::BC7);
        tryCompress(textures.emissive, compression.compressEmissive, PixelStorage::BC6);
    }

    // Create material
    auto material = std::make_unique<Material>();
    material->name = name;
    material->data = data;
    material->textures = std::move(textures);
    material->poolIndex = static_cast<uint>(mMaterials.size());

    // Upload textures to GPU
    uploadMaterialTextures(*material, mStream);

    // Add to pool
    uint index = material->poolIndex;
    mMaterials.push_back(std::move(material));
    mNameToIndex[name] = index;

    // Mark for upload
    markDirty(index);

    CI_LOG_V("Created material '" << name << "' with index " << index
             << " (type=" << static_cast<uint>(data.type) << ")");

    return index;
}

uint MaterialPool::createMaterial(
    const std::string& name,
    const luisa::float3& albedo,
    float roughness,
    float metallic,
    uint type) {

    MaterialData data;
    data.type = type;    data.albedo = luisa::float4{albedo.x, albedo.y, albedo.z, 1.0f};
    data.roughness = roughness;
    data.metallic = metallic;

    return createMaterial(name, data);
}

uint MaterialPool::createMaterialFromFolder(
    const std::string& name,
    const std::filesystem::path& textureFolder,
    const MaterialData& baseData,
    const TextureCompressionSettings& compression) {

    // Load textures using MaterialTextureLoader
    MaterialTextureLoader loader(mDevice);
    auto textures = loader.loadMaterial(textureFolder, name, compression);

    // Use base data, texture loader will set texture indices
    MaterialData data = baseData;
    data.albedoTexIdx = textures.albedoIdx;
    data.normalTexIdx = textures.normalIdx;
    data.rmaTexIdx = textures.rmaIdx;
    data.emissiveTexIdx = textures.emissiveIdx;

    CI_LOG_I("Loaded material '" << name << "' from " << textureFolder);

    return createMaterial(name, data, std::move(textures));
}

uint MaterialPool::cloneMaterial(
    const std::string& newName,
    uint sourceIndex,
    const MaterialData* modifications) {

    if (!hasMaterial(sourceIndex)) {
        CI_LOG_E("Cannot clone material: source index " << sourceIndex << " not found");
        return mErrorMaterialIndex;
    }

    const auto& source = getMaterial(sourceIndex);
    MaterialData data = source.data;

    // Apply modifications
    if (modifications) {
        data = *modifications;
        // Preserve texture indices from source if not specified
        if (modifications->albedoTexIdx < 0) data.albedoTexIdx = source.data.albedoTexIdx;
        if (modifications->normalTexIdx < 0) data.normalTexIdx = source.data.normalTexIdx;
        if (modifications->rmaTexIdx < 0) data.rmaTexIdx = source.data.rmaTexIdx;
        if (modifications->emissiveTexIdx < 0) data.emissiveTexIdx = source.data.emissiveTexIdx;
    }

    CI_LOG_I("Cloned material '" << source.name << "' to '" << newName << "'");

    return createMaterial(newName, data, MaterialTextures{});
}

//==============================================================================
// Material Access
//==============================================================================

const Material& MaterialPool::getMaterial(uint index) const {
    if (index >= mMaterials.size()) {
        CI_LOG_E("Material index " << index << " out of bounds, returning error material");
        return getMaterial(mErrorMaterialIndex);
    }
    return *mMaterials[index];
}

Material& MaterialPool::getMaterial(uint index) {
    if (index >= mMaterials.size()) {
        CI_LOG_E("Material index " << index << " out of bounds, returning error material");
        return getMaterial(mErrorMaterialIndex);
    }
    return *mMaterials[index];
}

const Material& MaterialPool::getMaterial(const std::string& name) const {
    auto it = mNameToIndex.find(name);
    if (it == mNameToIndex.end()) {
        CI_LOG_W("Material '" << name << "' not found, returning error material");
        return getMaterial(mErrorMaterialIndex);
    }
    return getMaterial(it->second);
}

Material& MaterialPool::getMaterial(const std::string& name) {
    auto it = mNameToIndex.find(name);
    if (it == mNameToIndex.end()) {
        CI_LOG_W("Material '" << name << "' not found, returning error material");
        return getMaterial(mErrorMaterialIndex);
    }
    return getMaterial(it->second);
}

uint MaterialPool::getIndex(const std::string& name) const {
    auto it = mNameToIndex.find(name);
    return (it != mNameToIndex.end()) ? it->second : mErrorMaterialIndex;
}

bool MaterialPool::hasMaterial(const std::string& name) const {
    return mNameToIndex.find(name) != mNameToIndex.end();
}

bool MaterialPool::hasMaterial(uint index) const {
    return index < mMaterials.size();
}

//==============================================================================
// Material Update
//==============================================================================

void MaterialPool::updateMaterialData(uint index, const MaterialData& data) {
    if (!hasMaterial(index)) {
        CI_LOG_E("Cannot update material: index " << index << " not found");
        return;
    }

    auto& mat = mMaterials[index];

    // Skip if no field changed — avoids spurious markDirty, GPU upload,
    // and (critically) mLightsNeedRebuild from the emission-change check below.
    if (material_data_equal(mat->data, data)) return;

    // check emissive
    if (!mLightsNeedRebuild && any(mat->data.emission > .0f)) {
        bool isEmissive = any(data.emission > 0.f);
        bool emissionChanged = any(mat->data.emission != data.emission);
        if (emissionChanged && isEmissive)
            mLightsNeedRebuild = true;
    }

    // Capture structural change before mutating mat->data — these fields drive
    // Geometry::_has_visible_glass / _has_transparent_shadow_casters, but the UI
    // path doesn't go through Geometry's setters, so flag for Pipeline to bridge.
    auto typeChanged    = (mat->data.type != data.type) ||
                          (mat->data.bsdf_type_override != data.bsdf_type_override);
    auto alphacutCross  = (mat->data.alphacut > 0.f) != (data.alphacut > 0.f);

    mat->data = data;
    markDirty(index);

    if (typeChanged || alphacutCross)
        mStructureDirty = true;

    CI_LOG_V("Updated material data for index " << index);
}

void MaterialPool::updateMaterialTextures(uint index, MaterialTextures&& textures) {
    if (!hasMaterial(index)) {
        CI_LOG_E("Cannot update material: index " << index << " not found");
        return;
    }

    auto& material    = *mMaterials[index];
    material.textures = std::move(textures);
    uploadMaterialTextures(material, mStream);
    markDirty(index);

    CI_LOG_V("Updated material textures for '" << material.name << "'");
}

//==============================================================================
// GPU Synchronization
//==============================================================================

void MaterialPool::update(Stream& stream) {
    if (mDirtyMaterials.empty() && !mNeedsRebuild)
        return;

    if (mNeedsRebuild) {
        rebuild(stream);
        return;
    }

    // Sort and merge consecutive dirty indices into batched ranges
    std::sort(mDirtyMaterials.begin(), mDirtyMaterials.end());

    size_t i = 0u;
    while (i < mDirtyMaterials.size()) {
        uint rangeStart = mDirtyMaterials[i];
        uint rangeEnd = rangeStart;

        // Extend while next dirty index is consecutive
        while (i + 1 < mDirtyMaterials.size() && mDirtyMaterials[i + 1] == rangeEnd + 1u) {
            rangeEnd = mDirtyMaterials[++i];
        }

        uint count = rangeEnd - rangeStart + 1u;

        // Stage the range once; upload both the full MaterialData and the
        // packed similarity key (same ranges, always in lockstep).
        luisa::vector<MaterialData> staging(count);
        luisa::vector<luisa::float3> keys(count);
        for (uint j = 0u; j < count; j++) {
            staging[j] = mMaterials[rangeStart + j]->data;
            keys[j] = material_similarity_key(staging[j]);
        }
        stream << mMaterialBuffer.view(rangeStart, count).copy_from(staging.data());
        stream << mSimKeyBuffer.view(rangeStart, count).copy_from(keys.data());

        i++;
    }

    stream << mTextureBindless.update();
    mDirtyMaterials.clear();
}

void MaterialPool::rebuild(Stream& stream) {
    CI_LOG_I("Rebuilding material pool GPU resources (" << mMaterials.size() << " materials)");

    vector<MaterialData> allData;
    vector<luisa::float3> allKeys;
    allData.reserve(mMaterials.size());
    allKeys.reserve(mMaterials.size());

    for (const auto& material : mMaterials) {
        allData.push_back(material->data);
        allKeys.push_back(material_similarity_key(material->data));
    }

    stream << mMaterialBuffer.copy_from(allData.data())
           << mSimKeyBuffer.copy_from(allKeys.data())
           << mTextureBindless.update()
           << synchronize();

    mDirtyMaterials.clear();
    mNeedsRebuild = false;
}

//==============================================================================
// Internal Helpers
//==============================================================================

void MaterialPool::uploadMaterialTextures(Material& material, Stream& stream) {
    auto& data      = material.data;
    auto& textures  = material.textures;

    auto assignTex = [&](const Image<float>& tex, int& idx, const char* texName) {
        if (tex.valid()) {
            idx = static_cast<int>(mNextTextureSlot++);
            mTextureBindless.emplace_on_update(idx, tex, Sampler::linear_linear_mirror());
            CI_LOG_V("  - " << texName << " texture assigned slot " << idx);
        }
    };

    assignTex(textures.albedo, data.albedoTexIdx, "albedo");
    assignTex(textures.normal, data.normalTexIdx, "normal");
    assignTex(textures.rma,      data.rmaTexIdx,      "rma");
    assignTex(textures.emissive, data.emissiveTexIdx, "emissive");

    stream << mTextureBindless.update()
           << synchronize();
}

uint MaterialPool::register_external_image(Image<float> const& image, Sampler sampler) {
    // Member function — mTextureBindless and mNextTextureSlot are private.
    // Mirrors the per-texture assignment pattern in uploadMaterialTextures.
    uint slot = mNextTextureSlot++;
    mTextureBindless.emplace_on_update(slot, image, sampler);
    CI_LOG_I("External image registered at bindless slot " << slot);
    return slot;
}

void MaterialPool::markDirty(uint index) {
    for (uint dirtyIdx : mDirtyMaterials)
        if (dirtyIdx == index) return;
    mDirtyMaterials.push_back(index);
}

//==============================================================================
// UI
//==============================================================================

namespace {
const char* getMaterialTypeName(uint type) {
    switch (static_cast<MaterialType>(type)) {
    case MaterialType::Null:           return "Null";
    case MaterialType::Diffuse:        return "Diffuse";
    case MaterialType::Conductor:      return "Conductor";
    case MaterialType::Dielectric:     return "Dielectric";
    case MaterialType::Plastic:        return "Plastic";
    case MaterialType::Emissive:       return "Emissive";
    case MaterialType::Subsurface:     return "Subsurface";
    case MaterialType::Clearcoat:      return "Clearcoat";
    case MaterialType::Sheen:          return "Sheen";
    case MaterialType::Anisotropy:     return "Anisotropy";
    case MaterialType::Iridescence:    return "Iridescence";
    case MaterialType::ThinDielectric: return "ThinDielectric";
    case MaterialType::Unlit:          return "Unlit";
    case MaterialType::Fabric:         return "Fabric";
    default:                           return "Custom";
    }
}

// Reset material data to type-appropriate defaults, preserving texture indices and meta.
static void resetMaterialForType(MaterialData& data, MaterialType newType) {
    auto tex_albedo     = data.albedoTexIdx;
    auto tex_normal     = data.normalTexIdx;
    auto tex_rma        = data.rmaTexIdx;
    auto tex_emissive   = data.emissiveTexIdx;
    auto tex_trans      = data.transmissionTexIdx;
    auto tex_aniso      = data.anisoTexIdx;
    auto meta           = data.meta;

    MaterialData fresh;
    fresh.type = static_cast<uint>(newType);
    switch (newType) {
    case MaterialType::Conductor: {
        const auto& alu = kMetalPresets[static_cast<uint>(MetalPreset::Aluminum)];
        fresh.metallic = 1.f; fresh.roughness = 0.3f;
        fresh.attenuation = alu.eta;  // complex IOR is the only conductor mode
        fresh.conductor_k = alu.k;
        break;
    }
    case MaterialType::Dielectric:     fresh.roughness = 0.f; fresh.specular_trans = 1.f; break;
    case MaterialType::Plastic:        fresh.roughness = 0.3f; break;
    case MaterialType::Emissive:       fresh.albedo = {0.f, 0.f, 0.f, 1.0f}; break;
    case MaterialType::Subsurface:     fresh.flatness = 1.f; fresh.ior = 1.4f; break;
    case MaterialType::Fabric:         fresh.fabric = 1.f; break;
    case MaterialType::ThinDielectric: fresh.roughness = 0.f; break;
    default: break;
    }

    fresh.albedoTexIdx       = tex_albedo;
    fresh.normalTexIdx       = tex_normal;
    fresh.rmaTexIdx          = tex_rma;
    fresh.emissiveTexIdx     = tex_emissive;
    fresh.transmissionTexIdx = tex_trans;
    fresh.anisoTexIdx        = tex_aniso;
    fresh.meta               = meta;

    data = fresh;
}

} // anonymous namespace

void MaterialPool::drawUi() {
    if (!ImGui::CollapsingHeader("Materials"))
        return;

    ImGui::Text("Count: %u / %u", count(), kMaxMaterials);

    for (uint i = 0; i < count(); i++) {
        Material& mat = getMaterial(i);
        // Copy, not reference: the widgets mutate `data` in place, then
        // updateMaterialData(i, data) compares it against mat.data. If `data`
        // were a reference, the early-exit `material_data_equal(mat->data, data)`
        // would compare the object with itself and always return — skipping
        // the GPU upload. Working on a copy lets the comparison see the diff.
        MaterialData data = mat.data;

        // Build header label with stable ID
        char label[128];
        snprintf(label, sizeof(label), "[%u] %s (%s)###mat_%u",
                 i, mat.name.c_str(), getMaterialTypeName(data.type), i);

        if (!ImGui::CollapsingHeader(label))
            continue;

        bool dirty = false;
        auto emissionBefore = data.emission;  // snapshot before UI modifies it
        ImGui::PushID(static_cast<int>(i));

        // --- Texture status (read-only) ---
        ImGui::Text("Textures:");
        ImGui::Indent();
        ImGui::Text("Albedo:    %s", data.albedoTexIdx >= 0 ? "assigned" : "none");
        ImGui::Text("Normal:    %s", data.normalTexIdx >= 0 ? "assigned" : "none");
        ImGui::Text("RMA:       %s", data.rmaTexIdx >= 0 ? "assigned" : "none");
        ImGui::Text("Emissive:  %s", data.emissiveTexIdx >= 0 ? "assigned" : "none");
        if (data.transmissionTexIdx >= 0 || data.anisoTexIdx >= 0) {
            ImGui::Text("Transmission: %s", data.transmissionTexIdx >= 0 ? "assigned" : "none");
            ImGui::Text("Anisotropy:   %s", data.anisoTexIdx >= 0 ? "assigned" : "none");
        }
        ImGui::Unindent();

        ImGui::Separator();

        float blend = glm::fract(data.meta);
        if (ImGui::DragFloat("Layer Blend", &blend, 0.005f, 0.f, .999f)) {
            data.meta = glm::floor(data.meta) + blend;
            dirty = true;
        }

        static const char* kBuiltinNames[] = {
            "Null",           // 0
            "Diffuse",        // 1
            "Conductor",      // 2
            "Dielectric",     // 3
            "Plastic",        // 4
            "Emissive",       // 5
            "Subsurface",     // 6
            "Clearcoat",      // 7
            "Sheen",          // 8
            "Anisotropy",     // 9
            "Iridescence",    // 10
            "ThinDielectric", // 11
            "Unlit",          // 12
            "Fabric"          // 13
        };

        bool isCustom = data.type > static_cast<uint>(MaterialType::Fabric);
        int mattype = isCustom ?
            static_cast<int>(data.bsdf_type_override) : static_cast<int>(data.type);
        if (ImGui::Combo("Type", &mattype, kBuiltinNames, IM_ARRAYSIZE(kBuiltinNames))) {
            if (isCustom) data.bsdf_type_override = static_cast<float>(mattype);
            else resetMaterialForType(data, static_cast<MaterialType>(mattype));
            dirty = true;
        }

        // --- Type-specific parameters ---
        switch (static_cast<MaterialType>(mattype)) {

        case MaterialType::Diffuse:
            dirty |= util::ui_color("Albedo", data.albedo);
            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Alpha Cutout", &data.alphacut, 0.005f, 0.f, 1.f);
            dirty |= util::ui_color_hdr("Emission", data.emission);
            break;

        case MaterialType::Conductor: {
            int p = 0;
            for (int i = 1; i <= 5; i++) {
                if (glm::abs(data.conductor_k.x - kMetalPresets[i].k.x) < .01f) {
                    p = i;
                    break;
                }
            }

            if (ImGui::Combo("Metal", &p, { "Custom","Gold","Silver","Copper","Aluminium","Brass" })) {
                if (p >= 1) {  // Custom keeps the hand-edited (eta, k) above
                    auto sample = kMetalPresets[p];
                    data.conductor_k = sample.k;
                    data.attenuation = sample.eta;
                    dirty = true;
                }
            }

            luisa::float3 ck = data.conductor_k.xyz();
            dirty |= util::ui_color_hdr("K", ck);
            data.conductor_k = luisa::float3{ ck.x, ck.y, ck.z };
            dirty |= util::ui_color("Eta", data.attenuation);

            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            break;
        }

        case MaterialType::Dielectric:
            dirty |= util::ui_color("Attenuation", data.attenuation);
            dirty |= ImGui::DragFloat("Attenuation Distance", &data.attenuation_distance, 0.05f, 0.001f, 100.f);
            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("IOR", &data.ior, 0.005f, 1.f, 8.f);
            dirty |= ImGui::DragFloat("Specular Transmittance", &data.specular_trans, 0.005f, 0.f, 1.f);
            // Stored in `metallic`, which is dead for Dielectric: nested-dielectric
            // interior priority (Schmidt-Budge). Lower = higher priority; 0 (default)
            // never cuts out other media. Overlapping volumes: give the outer
            // medium the smaller value (e.g. glass 0, liquid 1).
            if (ImGui::DragFloat("Interior Priority", &data.metallic, 1.f, 0.f, 15.f, "%.0f"))
                dirty = true;
            break;

        case MaterialType::Plastic:
            dirty |= util::ui_color("Albedo", data.albedo);
            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("IOR", &data.ior, 0.005f, 1.f, 8.f);
            dirty |= ImGui::DragFloat("Specular Tint", &data.specular_tint, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Clearcoat", &data.clearcoat, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Clearcoat Gloss", &data.clearcoat_gloss, 0.005f, 0.f, 1.f);
            break;

        case MaterialType::Emissive:
            dirty |= util::ui_color_hdr("Emission", data.emission);
            dirty |= util::ui_color("Albedo", data.albedo);
            break;

        case MaterialType::Subsurface:
            dirty |= util::ui_color("Albedo", data.albedo);
            dirty |= util::ui_color_hdr("Attenuation", data.attenuation);
            dirty |= ImGui::DragFloat("Attenuation Distance", &data.attenuation_distance, 0.001f, 0.001f, 0.15f);
            // Thin-wall subsurface (paper/leaves/lampshades): > 0 activates the
            // diffuse transmission lobe and suppresses the volumetric Burley
            // probe. Reflection and transmission share one energy budget.
            dirty |= ImGui::DragFloat("Thin Wall (Diffuse Transmission)", &data.diffuse_trans, 0.01f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Flatness", &data.flatness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("IOR", &data.ior, 0.005f, 1.f, 8.f);
            break;

        case MaterialType::Clearcoat:
            dirty |= ImGui::DragFloat("Clearcoat", &data.clearcoat, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Clearcoat Gloss", &data.clearcoat_gloss, 0.005f, 0.f, 1.f);
            break;

        case MaterialType::Sheen:
            dirty |= ImGui::DragFloat("Sheen", &data.sheen, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Sheen Tint", &data.sheen_tint, 0.005f, 0.f, 1.f);
            break;

        case MaterialType::Anisotropy:
            dirty |= ImGui::DragFloat("Anisotropic", &data.anisotropic, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Anisotropic Rotation", &data.anisotropic_rot, 0.005f, 0.f, 1.f);
            break;

        case MaterialType::Iridescence:
            dirty |= ImGui::DragFloat("Iridescence", &data.iridescence, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Iridescence IOR", &data.iridescence_ior, 0.005f, 1.f, 4.f);
            dirty |= ImGui::DragFloat("Iridescence Thickness", &data.iridescence_thickness, 1.f, 0.f, 2000.f, "%.0f nm");
            break;

        case MaterialType::ThinDielectric:
            dirty |= util::ui_color("Attenuation", data.attenuation);
            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("IOR", &data.ior, 0.005f, 1.f, 8.f);
            break;

        case MaterialType::Fabric:
            dirty |= util::ui_color("Albedo", data.albedo);
            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Fabric", &data.fabric, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Sheen", &data.sheen, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Sheen Tint", &data.sheen_tint, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Anisotropic", &data.anisotropic, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Anisotropic Rotation", &data.anisotropic_rot, 0.005f, 0.f, 1.f);
            break;

        case MaterialType::Unlit:
            dirty |= util::ui_color("Albedo", data.albedo);
            {
                bool receiveGI = glm::floor(data.meta) >= 0.5f;
                if (ImGui::Checkbox("Receive GI", &receiveGI)) {
                    data.meta = receiveGI ? 1.f : 0.f;
                    dirty = true;
                }
            }
            break;

        default:
            // Unknown/custom type: show type index and all fields
            ImGui::TextColored(ImVec4(1.f, 1.f, 0.f, 1.f), "Custom type #%u", data.type);
            dirty |= util::ui_color("Albedo", data.albedo);
            dirty |= util::ui_color_hdr("Emission", data.emission);
            dirty |= util::ui_color("Attenuation", data.attenuation);
            dirty |= ImGui::DragFloat("Roughness", &data.roughness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Metallic", &data.metallic, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("IOR", &data.ior, 0.005f, 1.f, 8.f);
            dirty |= ImGui::DragFloat("Alpha Cutout", &data.alphacut, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Specular Tint", &data.specular_tint, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Specular Transmittance", &data.specular_trans, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Clearcoat", &data.clearcoat, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Clearcoat Gloss", &data.clearcoat_gloss, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Sheen", &data.sheen, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Sheen Tint", &data.sheen_tint, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Anisotropic", &data.anisotropic, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Anisotropic Rotation", &data.anisotropic_rot, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Flatness", &data.flatness, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Diffuse Trans", &data.diffuse_trans, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Fabric", &data.fabric, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Iridescence", &data.iridescence, 0.005f, 0.f, 1.f);
            dirty |= ImGui::DragFloat("Iridescence IOR", &data.iridescence_ior, 0.005f, 1.f, 4.f);
            dirty |= ImGui::DragFloat("Attenuation Distance", &data.attenuation_distance, 0.05f, 0.001f, 100.f);
            dirty |= ImGui::DragFloat("Iridescence Thickness", &data.iridescence_thickness, 1.f, 0.f, 2000.f, "%.0f nm");
            break;
        }

        // --- Extra layers (additive lobes not already shown above) ---
        {
            auto mt = static_cast<MaterialType>(mattype);
            bool showClearcoat  = (mt == MaterialType::Diffuse || mt == MaterialType::Conductor ||
                                   mt == MaterialType::Subsurface);
            bool showSheen      = (mt == MaterialType::Diffuse || mt == MaterialType::Plastic ||
                                   mt == MaterialType::Subsurface);
            bool showIridescence = (mt == MaterialType::Diffuse || mt == MaterialType::Conductor ||
                                   mt == MaterialType::Plastic || mt == MaterialType::Subsurface ||
                                   mt == MaterialType::Fabric);
            bool showAnisotropy = (mt == MaterialType::Diffuse || mt == MaterialType::Conductor ||
                                   mt == MaterialType::Dielectric || mt == MaterialType::Plastic ||
                                   mt == MaterialType::Subsurface || mt == MaterialType::ThinDielectric);

            bool anyExtra = showClearcoat || showSheen || showIridescence || showAnisotropy;
            if (anyExtra && ImGui::CollapsingHeader("Layers")) {
                ImGui::Indent();
                // PushID per layer: CollapsingHeader doesn't open an ID scope,
                // so each header would collide with its same-named first drag.
                if (showClearcoat) {
                    ImGui::PushID("Clearcoat");
                    if (ImGui::CollapsingHeader("Clearcoat", ImGuiTreeNodeFlags_DefaultOpen)) {
                        dirty |= ImGui::DragFloat("Strength", &data.clearcoat, 0.005f, 0.f, 1.f);
                        dirty |= ImGui::DragFloat("Clearcoat Gloss", &data.clearcoat_gloss, 0.005f, 0.f, 1.f);
                    }
                    ImGui::PopID();
                }
                if (showSheen) {
                    ImGui::PushID("Sheen");
                    if (ImGui::CollapsingHeader("Sheen", ImGuiTreeNodeFlags_DefaultOpen)) {
                        dirty |= ImGui::DragFloat("Strength", &data.sheen, 0.005f, 0.f, 1.f);
                        dirty |= ImGui::DragFloat("Sheen Tint", &data.sheen_tint, 0.005f, 0.f, 1.f);
                    }
                    ImGui::PopID();
                }
                if (showIridescence) {
                    ImGui::PushID("Iridescence");
                    if (ImGui::CollapsingHeader("Iridescence", ImGuiTreeNodeFlags_DefaultOpen)) {
                        dirty |= ImGui::DragFloat("Strength", &data.iridescence, 0.005f, 0.f, 1.f);
                        dirty |= ImGui::DragFloat("Iridescence IOR", &data.iridescence_ior, 0.005f, 1.f, 4.f);
                        dirty |= ImGui::DragFloat("Iridescence Thickness", &data.iridescence_thickness, 1.f, 0.f, 2000.f, "%.0f nm");
                    }
                    ImGui::PopID();
                }
                if (showAnisotropy) {
                    ImGui::PushID("Anisotropy");
                    if (ImGui::CollapsingHeader("Anisotropy", ImGuiTreeNodeFlags_DefaultOpen)) {
                        dirty |= ImGui::DragFloat("Strength", &data.anisotropic, 0.005f, 0.f, 1.f);
                        dirty |= ImGui::DragFloat("Anisotropic Rotation", &data.anisotropic_rot, 0.005f, 0.f, 1.f);
                    }
                    ImGui::PopID();
                }
                ImGui::Unindent();
            }
        }

        ImGui::PopID();

        // Push changes to GPU if any widget was modified
        if (dirty) {
            if (!mLightsNeedRebuild) {
                // Flag light sampler rebuild if emission changed on an emissive material
                bool wasEmissive    = (emissionBefore.x > 0.f) || (emissionBefore.y > 0.f) || (emissionBefore.z > 0.f);
                bool isEmissive     = (data.emission.x > 0.f) || (data.emission.y > 0.f) || (data.emission.z > 0.f);
                bool emissionChanged= (emissionBefore.x != data.emission.x) ||
                    (emissionBefore.y != data.emission.y) ||
                    (emissionBefore.z != data.emission.z);
                if (emissionChanged && (wasEmissive || isEmissive))
                    mLightsNeedRebuild = true;
            }

            updateMaterialData(i, data);
        }
    }
}

//==============================================================================
// Config serialization
//==============================================================================

namespace {
ci::Json materialDataToJson(const MaterialData& d) {
    ci::Json j;
    j["type"] = d.type;
    j["albedo"] = toci(d.albedo);
    j["emission"] = toci(d.emission);
    j["roughness"] = d.roughness;
    j["metallic"] = d.metallic;
    j["ior"] = d.ior;
    j["alphacut"] = d.alphacut;
    j["specular_tint"] = d.specular_tint;
    j["specular_trans"] = d.specular_trans;
    j["clearcoat"] = d.clearcoat;
    j["clearcoat_gloss"] = d.clearcoat_gloss;
    j["sheen"] = d.sheen;
    j["sheen_tint"] = d.sheen_tint;
    j["anisotropic"] = d.anisotropic;
    j["anisotropic_rot"] = d.anisotropic_rot;
    j["flatness"] = d.flatness;
    j["diffuse_trans"] = d.diffuse_trans;
    j["fabric"] = d.fabric;
    j["iridescence"] = d.iridescence;
    j["iridescence_ior"] = d.iridescence_ior;
    j["attenuation"] = toci(d.attenuation);
    j["attenuation_distance"] = d.attenuation_distance;
    j["iridescence_thickness"] = d.iridescence_thickness;
    j["bsdf_type_override"] = d.bsdf_type_override;
    j["meta"] = d.meta;
    j["k"] = toci(d.conductor_k);
    return j;
}

void jsonToMaterialData(const ci::Json& j, MaterialData& d) {
    namespace nt = newtype;
    d.type = j.value("type", d.type);
    if (j.contains("albedo"))  d.albedo  = tolc(j.value("albedo",  toci(d.albedo)));
    if (j.contains("emission"))d.emission = tolc(j.value("emission", toci(d.emission)));
    d.roughness       = j.value("roughness",       d.roughness);
    d.metallic        = j.value("metallic",         d.metallic);
    d.ior             = j.value("ior",              d.ior);
    d.alphacut        = j.value("alphacut",          d.alphacut);
    d.specular_tint   = j.value("specular_tint",   d.specular_tint);
    d.specular_trans  = j.value("specular_trans",   d.specular_trans);
    d.clearcoat       = j.value("clearcoat",        d.clearcoat);
    d.clearcoat_gloss = j.value("clearcoat_gloss",  d.clearcoat_gloss);
    d.sheen           = j.value("sheen",             d.sheen);
    d.sheen_tint      = j.value("sheen_tint",        d.sheen_tint);
    d.anisotropic     = j.value("anisotropic",       d.anisotropic);
    d.anisotropic_rot = j.value("anisotropic_rot",   d.anisotropic_rot);
    d.flatness        = j.value("flatness",          d.flatness);
    d.diffuse_trans   = j.value("diffuse_trans",     d.diffuse_trans);
    d.fabric          = j.value("fabric",            d.fabric);
    d.iridescence     = j.value("iridescence",       d.iridescence);
    d.iridescence_ior = j.value("iridescence_ior",   d.iridescence_ior);
    if (j.contains("attenuation")) d.attenuation = tolc(j.value("attenuation", toci(d.attenuation)));
    d.attenuation_distance  = j.value("attenuation_distance",  d.attenuation_distance);
    d.iridescence_thickness = j.value("iridescence_thickness", d.iridescence_thickness);
    d.bsdf_type_override    = j.value("bsdf_type_override",    d.bsdf_type_override);
    d.meta                  = j.value("meta",                   d.meta);
    if (j.contains("k")) d.conductor_k = tolc(j.value("k", toci(d.conductor_k)));
    // Configs saved before complex-IOR-only conductors carry k = 0 (legacy
    // edge tint); migrate them to Aluminium so they still render as metal.
    if (d.type == static_cast<uint>(MaterialType::Conductor)
        && d.conductor_k.x == 0.f && d.conductor_k.y == 0.f && d.conductor_k.z == 0.f) {
        const auto& alu = kMetalPresets[static_cast<uint>(MetalPreset::Aluminum)];
        d.conductor_k = alu.k;
        d.attenuation = alu.eta;
    }
}
} // anonymous namespace

ci::Json MaterialPool::materialsToJson() const {
    ci::Json arr = ci::Json::array();
    for (uint i = 0; i < count(); i++) {
        const auto& mat = getMaterial(i);
        // Skip built-in materials
        if (mat.name == "__default__" || mat.name == "__error__") continue;
        ci::Json mj;
        mj["name"] = mat.name;
        mj["data"] = materialDataToJson(mat.data);
        arr.push_back(mj);
    }
    return arr;
}

void MaterialPool::materialsFromJson(const ci::Json& j) {
    if (!j.is_array()) return;
    for (const auto& mj : j) {
        if (!mj.contains("name")) continue;
        std::string name = mj["name"].get<std::string>();
        if (!hasMaterial(name)) continue;
        auto& mat = getMaterial(name);
        auto emissionBefore = mat.data.emission;
        MaterialData data = mat.data;
        if (mj.contains("data")) {
            jsonToMaterialData(mj["data"], data);
            // Flag light rebuild if emission changed on an emissive material
            bool wasEmissive = (emissionBefore.x > 0.f) || (emissionBefore.y > 0.f) || (emissionBefore.z > 0.f);
            bool isEmissive  = (data.emission.x > 0.f) || (data.emission.y > 0.f) || (data.emission.z > 0.f);
            bool emissionChanged = (emissionBefore.x != data.emission.x) ||
                                   (emissionBefore.y != data.emission.y) ||
                                   (emissionBefore.z != data.emission.z);
            if (emissionChanged && (wasEmissive || isEmissive))
                mLightsNeedRebuild = true;
            updateMaterialData(mat.poolIndex, data);
            CI_LOG_V("Restored material '" << name << "' from config");
        }
    }
}

} // namespace newtype::render
