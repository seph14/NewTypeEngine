#include "newtype/render/MaterialPool.h"
#include "newtype/render/TextureConverter.h"
#include "newtype/render/MaterialSimilarity.h"
#include <algorithm>
#include <bit>
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

    // Resolver params buffer at the reserved bindless slot (see header block).
    // Bound BEFORE any texture so texture slots can start after the reserved
    // block; contents upload via update() only when dirty. Mirrored in
    // mResolverParamShadow.
    mResolverParams = device.create_buffer<luisa::float4>(
        kMaxResolverParamCallables * (kResolverParamsPerCallable / 4u));
    mResolverParamShadow.assign(
        kMaxResolverParamCallables * (kResolverParamsPerCallable / 4u),
        luisa::float4(0.0f));
    mTextureBindless.emplace_on_update(kResolverParamsBindlessSlot, mResolverParams);

    // Per-instance params placeholder at reserved slot 1 (track B2):
    // zero-filled so a callable that reads it before any authoring sees
    // deterministic zeros; grown to the full instance count on the first
    // uploadInstanceParams. Textures start at slot 2.
    mInstanceParams = device.create_buffer<luisa::float4>(
        kInstanceParamsDefaultRows * kInstanceParamsPerInstance);
    mInstanceParamsCapacityRows = kInstanceParamsDefaultRows;
    luisa::vector<luisa::float4> zeros(
        kInstanceParamsDefaultRows * kInstanceParamsPerInstance, luisa::float4(0.0f));
    mStream << mInstanceParams.copy_from(zeros.data());
    mTextureBindless.emplace_on_update(kInstanceParamsBindlessSlot, mInstanceParams);

    mNextTextureSlot = kInstanceParamsBindlessSlot + 1u;

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
    , mResolverParams(std::move(other.mResolverParams))
    , mResolverParamShadow(std::move(other.mResolverParamShadow))
    , mResolverParamsDirty(other.mResolverParamsDirty)
    , mResolverParamNames(std::move(other.mResolverParamNames))
    , mResolverParamIndex(std::move(other.mResolverParamIndex))
    , mResolverParamEntries(std::move(other.mResolverParamEntries))
    , mInstanceParams(std::move(other.mInstanceParams))
    , mInstanceParamsCapacityRows(other.mInstanceParamsCapacityRows)
    , mInstanceParamsResident(other.mInstanceParamsResident)
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
        mResolverParams = std::move(other.mResolverParams);
        mResolverParamShadow = std::move(other.mResolverParamShadow);
        mResolverParamsDirty = other.mResolverParamsDirty;
        mResolverParamNames = std::move(other.mResolverParamNames);
        mResolverParamIndex = std::move(other.mResolverParamIndex);
        mResolverParamEntries = std::move(other.mResolverParamEntries);
        mInstanceParams = std::move(other.mInstanceParams);
        mInstanceParamsCapacityRows = other.mInstanceParamsCapacityRows;
        mInstanceParamsResident = other.mInstanceParamsResident;
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

    CI_LOG_D("Loaded material '" << name << "' from " << textureFolder);

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

    CI_LOG_D("Cloned material '" << source.name << "' to '" << newName << "'");

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
    if (mDirtyMaterials.empty() && !mNeedsRebuild && !mResolverParamsDirty)
        return;

    if (mNeedsRebuild) {
        rebuild(stream);
        // rebuild() doesn't touch the resolver params — fall through so a
        // pending params upload still flushes this frame.
        if (!mResolverParamsDirty) return;
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

    // Resolver params: contents-only upload when dirty (binding is fixed at
    // the reserved slot — no bindless update needed). 2KB, so the cost is
    // negligible even when a slider drags every frame.
    if (mResolverParamsDirty) {
        stream << mResolverParams.copy_from(mResolverParamShadow.data());
        mResolverParamsDirty = false;
    }
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
           << mTextureBindless.update();

    mDirtyMaterials.clear();
    mNeedsRebuild = false;
}

//==============================================================================
// Resolver Runtime Params (docs/resolver_params_abi_plan.md)
//==============================================================================

void MaterialPool::_packResolverParamShadow(uint idx, const ResolverParamEntry& entry) {
    uint base4 = idx * (kResolverParamsPerCallable / 4u);
    for (uint i = 0u; i < kResolverParamsPerCallable; i++) {
        float v = i < entry.descs.size() ? entry.values[i] : 0.0f;
        switch (i % 4u) {
            case 0u: mResolverParamShadow[base4 + i / 4u].x = v; break;
            case 1u: mResolverParamShadow[base4 + i / 4u].y = v; break;
            case 2u: mResolverParamShadow[base4 + i / 4u].z = v; break;
            default: mResolverParamShadow[base4 + i / 4u].w = v; break;
        }
    }
}

uint MaterialPool::registerResolverParams(
    const std::string& callable, const luisa::vector<ResolverParamDesc>& descs) {

    if (descs.empty() || descs.size() > kResolverParamsPerCallable) {
        CI_LOG_E("Resolver params for '" << callable << "': bad descriptor count "
                 << descs.size() << " (max " << kResolverParamsPerCallable << ")");
        return ~0u;
    }

    // Existing name (DLL reload): keep the entry and its CURRENT values —
    // the plan's value-survival contract — EXCEPT where the author changed
    // a default in the DLL source and the user never moved that slider:
    // then the authored default wins (editing kGlassBlendParams mid-session
    // visibly applies; a user-tuned value survives code edits). New
    // descriptors default; surviving ones clamp into the new range.
    if (auto it = mResolverParamIndex.find(callable); it != mResolverParamIndex.end()) {
        auto& entry = mResolverParamEntries[it->second];
        uint oldDescCount = static_cast<uint>(entry.descs.size());
        luisa::vector<float> oldDefs;
        oldDefs.reserve(oldDescCount);
        for (const auto& d : entry.descs) oldDefs.push_back(d.def_v);
        luisa::vector<float> old = std::move(entry.values);
        entry.descs = descs;
        entry.values.assign(kResolverParamsPerCallable, 0.0f);
        uint authoredApplied = 0u;
        for (uint i = 0u; i < descs.size(); i++) {
            const auto& d = entry.descs[i];
            float v = d.def_v;
            if (i < oldDescCount) {
                // Defaults and values are exact host copies — == is sound.
                bool authorChanged = oldDefs[i] != d.def_v;
                bool userTuned     = old[i] != oldDefs[i];
                v = (authorChanged && !userTuned) ? d.def_v : old[i];
                if (authorChanged && !userTuned) authoredApplied++;
            }
            entry.values[i] = std::clamp(v, d.min_v, d.max_v);
        }
        _packResolverParamShadow(it->second, entry);
        mResolverParamsDirty = true;
        CI_LOG_I("Resolver params: '" << callable << "' re-registered after DLL reload ("
                 << descs.size() << " params, values preserved"
                 << (authoredApplied ? ", authored-default changes applied: "
                                       + std::to_string(authoredApplied) : "")
                 << ")");
        return it->second * (kResolverParamsPerCallable / 4u);
    }

    if (mResolverParamNames.size() >= kMaxResolverParamCallables) {
        CI_LOG_W("Resolver params full (" << kMaxResolverParamCallables
                 << ") - '" << callable << "' params ignored (slot leaks only on"
                 << " rename; see plan risk 5)");
        return ~0u;
    }

    uint idx = static_cast<uint>(mResolverParamNames.size());
    mResolverParamNames.push_back(callable);
    mResolverParamIndex[callable] = idx;

    ResolverParamEntry entry;
    entry.descs = descs;
    entry.values.assign(kResolverParamsPerCallable, 0.0f);
    for (uint i = 0u; i < descs.size(); i++) {
        // Clamp the default into its own range — same rule as the reload
        // path (a def_v outside [min,max] is an authoring slip; both paths
        // must agree or fresh vs reloaded sessions diverge).
        entry.values[i] = std::clamp(descs[i].def_v, descs[i].min_v, descs[i].max_v);
    }
    mResolverParamEntries.push_back(std::move(entry));
    _packResolverParamShadow(idx, mResolverParamEntries[idx]);
    mResolverParamsDirty = true;
    CI_LOG_I("Resolver params: '" << callable << "' registered "
             << descs.size() << " params (float4 base " << idx * (kResolverParamsPerCallable / 4u) << ")");
    return idx * (kResolverParamsPerCallable / 4u);
}

bool MaterialPool::setResolverParamValue(
    const std::string& callable, uint paramIndex, float value) {

    auto it = mResolverParamIndex.find(callable);
    if (it == mResolverParamIndex.end() ||
        paramIndex >= mResolverParamEntries[it->second].descs.size()) {
        return false;
    }
    auto& entry = mResolverParamEntries[it->second];
    const auto& d = entry.descs[paramIndex];
    value = std::clamp(value, d.min_v, d.max_v);
    if (entry.values[paramIndex] == value) return false;
    entry.values[paramIndex] = value;
    CI_LOG_I("Resolver param '" << callable << "'." << d.name.c_str()
             << " = " << value << " (dirty upload next update)");

    // Pack scalar i into float4 [i/4].c[i%4] of the callable's block.
    uint f4 = it->second * (kResolverParamsPerCallable / 4u) + paramIndex / 4u;
    uint c  = paramIndex % 4u;
    switch (c) {
        case 0u: mResolverParamShadow[f4].x = value; break;
        case 1u: mResolverParamShadow[f4].y = value; break;
        case 2u: mResolverParamShadow[f4].z = value; break;
        default: mResolverParamShadow[f4].w = value; break;
    }
    mResolverParamsDirty = true;
    return true;
}

bool MaterialPool::resetResolverParams(const std::string& callable) {
    auto it = mResolverParamIndex.find(callable);
    if (it == mResolverParamIndex.end()) return false;
    auto& entry = mResolverParamEntries[it->second];
    for (uint i = 0u; i < entry.descs.size(); i++) {
        entry.values[i] = std::clamp(
            entry.descs[i].def_v, entry.descs[i].min_v, entry.descs[i].max_v);
    }
    _packResolverParamShadow(it->second, entry);
    mResolverParamsDirty = true;
    CI_LOG_I("Resolver params: '" << callable << "' reset to registered defaults");
    return true;
}

void MaterialPool::uploadInstanceParams(
    Stream& stream, luisa::span<const luisa::float4> rows) {
    if (rows.empty()) return;
    if (rows.size() % kInstanceParamsPerInstance != 0u) {
        CI_LOG_E("MaterialPool::uploadInstanceParams: row count " << rows.size()
            << " not a multiple of " << kInstanceParamsPerInstance
            << " — upload skipped");
        return;
    }
    uint neededRows = static_cast<uint>(rows.size()) / kInstanceParamsPerInstance;
    if (neededRows > mInstanceParamsCapacityRows) {
        // Grow (power-of-two, at least the default) and rebind at the reserved
        // slot on THIS stream — the rebind must be visible before this frame's
        // shader reads, so the bindless update rides the same command stream
        // as the contents upload (MaterialPool::update's texture-bindless
        // flush on mStream would be a queue too late and cross-stream).
        mInstanceParamsCapacityRows = std::max(kInstanceParamsDefaultRows,
            std::bit_ceil(neededRows));
        mInstanceParams = mDevice.create_buffer<luisa::float4>(
            mInstanceParamsCapacityRows * kInstanceParamsPerInstance);
        mTextureBindless.emplace_on_update(kInstanceParamsBindlessSlot, mInstanceParams);
        stream << mTextureBindless.update();
    }
    stream << mInstanceParams.view(0u, static_cast<uint>(rows.size()))
                 .copy_from(rows.data());
    if (!mInstanceParamsResident) {
        mInstanceParamsResident = true;
        CI_LOG_I("Per-instance params buffer materialized: " << neededRows
            << " rows (capacity " << mInstanceParamsCapacityRows << ") at bindless slot "
            << kInstanceParamsBindlessSlot);
    }
}

const luisa::vector<MaterialPool::ResolverParamDesc>* MaterialPool::resolverParamDescs(
    const std::string& callable) const noexcept {
    auto it = mResolverParamIndex.find(callable);
    return it == mResolverParamIndex.end()
         ? nullptr : &mResolverParamEntries[it->second].descs;
}

float MaterialPool::resolverParamValue(
    const std::string& callable, uint paramIndex) const noexcept {
    auto it = mResolverParamIndex.find(callable);
    if (it == mResolverParamIndex.end() ||
        paramIndex >= mResolverParamEntries[it->second].descs.size()) {
        return 0.0f;
    }
    return mResolverParamEntries[it->second].values[paramIndex];
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
    assignTex(textures.iridescence, data.iridescenceTexIdx, "iridescence");

    stream << mTextureBindless.update()
           << synchronize();
}

uint MaterialPool::register_external_image(Image<float> const& image, Sampler sampler) {
    // Member function — mTextureBindless and mNextTextureSlot are private.
    // Mirrors the per-texture assignment pattern in uploadMaterialTextures.
    uint slot = mNextTextureSlot++;
    mTextureBindless.emplace_on_update(slot, image, sampler);
    CI_LOG_D("External image registered at bindless slot " << slot);
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
    auto tex_irid       = data.iridescenceTexIdx;
    auto irid_max       = data.iridescence_thickness_max;
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
    fresh.iridescenceTexIdx  = tex_irid;
    fresh.iridescence_thickness_max = irid_max;
    fresh.meta               = meta;

    data = fresh;
}

} // anonymous namespace

void MaterialPool::drawUi() {
    if (!ImGui::CollapsingHeader("Materials"))
        return;

    ImGui::Text("Count: %u / %u", count(), kMaxMaterials);

    // Callable-driven glass blending authoring preset (docs/glass_blend_plan.md,
    // docs/custom_material_callables.md): a custom-type material classified as
    // Dielectric (bsdf_type_override=3) so the PSR glass branch runs; the
    // per-pixel blend fraction comes from the resolver's SurfaceData.glass_blend
    // (e.g. the DLL glass_blend_resolver example). Register the resolver BEFORE
    // buildScene() — the dispatch tag must match this material's type.
    if (ImGui::Button("Add Blendable Glass")) {
        MaterialData d = make_dielectric();
        d.albedo              = luisa::float4{0.85f, 0.45f, 0.15f, 1.0f}; // diffuse-side color
        d.roughness           = 0.f;
        d.type                = 18u;  // matches the DLL example's 5th callable
        d.bsdf_type_override  = 3.f;  // classify as Dielectric for the PSR gate
        char name[64];
        snprintf(name, sizeof(name), "blend_glass_%u", count());
        createMaterial(name, d);
    }

    // Resolver runtime params (docs/resolver_params_abi_plan.md): live tuning
    // sliders for callables that registered descriptors (ABI v2). Values are
    // host-side; update() uploads the 2KB buffer when dirty — no shader
    // recompile. Accum reset fires on slider COMMIT only (matches material-edit
    // semantics); the live drag preview intentionally keeps converging history.
    if (!mResolverParamNames.empty() && ImGui::CollapsingHeader("Resolver Params")) {
        for (const auto& callable : mResolverParamNames) {
            const auto* descs = resolverParamDescs(callable);
            if (descs == nullptr) continue;
            ImGui::PushID(callable.c_str());
            if (ImGui::TreeNode(callable.c_str())) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Reset")) {
                    resetResolverParams(callable);
                    if (mAccumResetCb) mAccumResetCb();
                }
                for (uint p = 0u; p < descs->size(); p++) {
                    const auto& d = (*descs)[p];
                    float v = resolverParamValue(callable, p);
                    if (ImGui::SliderFloat(d.name.c_str(), &v, d.min_v, d.max_v, "%.3f")) {
                        setResolverParamValue(callable, p, v);
                    }
                    if (ImGui::IsItemDeactivatedAfterEdit()) {
                        if (mAccumResetCb) mAccumResetCb();
                    }
                }
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
    }

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
        if (data.transmissionTexIdx >= 0 || data.anisoTexIdx >= 0 || data.iridescenceTexIdx >= 0) {
            ImGui::Text("Transmission: %s", data.transmissionTexIdx >= 0 ? "assigned" : "none");
            ImGui::Text("Anisotropy:   %s", data.anisoTexIdx >= 0 ? "assigned" : "none");
            ImGui::Text("Iridescence:  %s", data.iridescenceTexIdx >= 0 ? "assigned" : "none");
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
            // KHR_materials_dispersion: Abbe number, 0 = off. Lower = stronger
            // rainbow spread (typical glass 50-70, gem/flint 20-40). Values
            // below 5 are clamped to 5 by dispersed_ior() (BSDF.h art-direction
            // clamp) — the UI shows the floor so the widget doesn't imply
            // unlimited strength.
            dirty |= ImGui::DragFloat("Dispersion (Abbe V)", &data.dispersion, 0.25f, 0.f, 150.f, "%.0f (0=off, min 5)");
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
            dirty |= ImGui::DragFloat("Coat IOR", &data.ior, 0.005f, 1.f, 4.f);
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
            dirty |= ImGui::DragFloat("Thickness Max (map)", &data.iridescence_thickness_max, 1.f, -1.f, 2000.f, "%.0f nm");
            if (data.iridescenceTexIdx < 0 && data.iridescence_thickness_max >= 0.f)
                ImGui::TextDisabled("thickness map range needs an iridescence texture");
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
            if (static_cast<uint>(data.bsdf_type_override) == 3u ||
                static_cast<uint>(data.bsdf_type_override) == 11u) {
                ImGui::TextDisabled(
                    "glass-classified custom: per-pixel diffuse<->glass blend comes"
                    " from the resolver's SurfaceData.glass_blend (see docs/custom_material_callables.md)");
            }
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
            dirty |= ImGui::DragFloat("Thickness Max (map)", &data.iridescence_thickness_max, 1.f, -1.f, 2000.f, "%.0f nm");
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
                        dirty |= ImGui::DragFloat("Thickness Max (map)", &data.iridescence_thickness_max, 1.f, -1.f, 2000.f, "%.0f nm");
                        if (data.iridescenceTexIdx < 0 && data.iridescence_thickness_max >= 0.f)
                            ImGui::TextDisabled("thickness map range needs an iridescence texture");
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
    j["iridescence_thickness_max"] = d.iridescence_thickness_max;
    j["dispersion"] = d.dispersion;
    j["bsdf_type_override"] = d.bsdf_type_override;
    j["meta"] = d.meta;
    j["k"] = toci(d.conductor_k);
    return j;
}
} // anonymous namespace

void materialDataFromJson(const ci::Json& j, MaterialData& d) {
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
    d.iridescence_thickness_max = j.value("iridescence_thickness_max", d.iridescence_thickness_max);
    d.dispersion            = j.value("dispersion",            d.dispersion);
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
            materialDataFromJson(mj["data"], data);
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
