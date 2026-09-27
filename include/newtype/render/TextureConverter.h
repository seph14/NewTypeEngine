#pragma once

#include "cinder/Cinder.h"
#include "cinder/Surface.h"
#include "cinder/ImageIo.h"
#include <luisa/luisa-compute.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/backends/ext/tex_compress_ext.h>
#include <filesystem>
#include <string>
#include <vector>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// Texture Compression Settings
//==============================================================================

/**
 * @brief Controls which material textures get BCn block compression
 *
 * Block compression reduces VRAM from 128 bits/texel (FLOAT4) to 4-8 bits/texel.
 * Compressed textures are transparent to shaders — hardware decompresses on sample.
 * Only applied to textures >= 4x4 (block compression operates on 4x4 blocks).
 */
struct TextureCompressionSettings {
    bool enableCompression = false;   // Master toggle (off by default)
    bool compressAlbedo   = true;     // BC7 (high-quality LDR)
    bool compressNormal   = true;     // BC7 (BC5 ideal but no compressor yet)
    bool compressRMA      = true;     // BC7 (high-quality LDR)
    bool compressEmissive = true;     // BC6H (HDR-capable)
};

//==============================================================================
// MaterialTextures (owning texture resources)
//==============================================================================

/**
 * @brief Material textures container (owning)
 *
 * Owns the actual image resources. Stored separately from MaterialData
 * so MaterialData remains POD and can be copied to GPU buffers.
 *
 * RMA Texture: Combined Roughness-Metallic-AO texture for memory efficiency
 * - R channel: Roughness
 * - G channel: Metallic
 * - B channel: Ambient Occlusion
 */
struct MaterialTextures {
    luisa::compute::Image<float> albedo;
    luisa::compute::Image<float> normal;
    luisa::compute::Image<float> rma;        // Combined R:roughness, G:metallic, B:AO
    luisa::compute::Image<float> emissive;
    luisa::compute::Image<float> iridescence; // R:factor mask, G:thickness mix (min..max)

    // CPU-side pixel data for deferred upload (emptied after upload)
    luisa::vector<luisa::float4> albedoPixels;
    luisa::vector<luisa::float4> normalPixels;
    luisa::vector<luisa::float4> rmaPixels;
    luisa::vector<luisa::float4> emissivePixels;

    // Bindless array indices (set by addToBindless or MaterialPool)
    int albedoIdx{-1};
    int normalIdx{-1};
    int rmaIdx{-1};         // Combined RMA texture index
    int emissiveIdx{-1};

    // Check if texture is valid
    [[nodiscard]] bool hasAlbedo() const { return albedo.valid(); }
    [[nodiscard]] bool hasNormal() const { return normal.valid(); }
    [[nodiscard]] bool hasRMA() const { return rma.valid(); }
    [[nodiscard]] bool hasRoughness() const { return rma.valid(); }
    [[nodiscard]] bool hasMetallic() const { return rma.valid(); }
    [[nodiscard]] bool hasAO() const { return rma.valid(); }
    [[nodiscard]] bool hasEmissive() const { return emissive.valid(); }
    [[nodiscard]] bool hasIridescence() const { return iridescence.valid(); }
};

//==============================================================================
// MaterialTextureLoader
//==============================================================================

/**
 * @brief Load material textures from folder using naming conventions
 *
 * Automatically searches for textures using multiple naming patterns:
 * - {name}_albedo.png / {name}_diffuse.png / {name}_color.png / {name}_basecolor.png
 * - {name}_normal.png / {name}_nrm.png / {name}_n.png
 * - {name}_rma.png / {name}_orm.png / {name}_rmo.png (Combined RMA texture, tried first)
 * - {name}_roughness.png / {name}_metallic.png / {name}_ao.png (Packed into RMA if combined not found)
 * - {name}_emissive.png / {name}_emission.png / {name}_emi.png / {name}_e.png
 *
 * RMA Texture: R=roughness, G=metallic, B=AO
 */
class MaterialTextureLoader {
public:
    explicit MaterialTextureLoader(Device& device);

    /**
     * @brief Load material textures from folder
     *
     * Searches for textures using multiple naming conventions.
     * Missing textures stay invalid — shader-side -1 fallbacks supply
     * constant material values and the geometric normal.
     *
     * @param folderPath Path to texture folder
     * @param name Material name (without extension)
     * @return MaterialTextures structure with loaded textures and indices
     */
    [[nodiscard]] MaterialTextures loadMaterial(
        const std::filesystem::path& folderPath,
        const std::string& name,
        const TextureCompressionSettings& compression = {});

    /**
     * @brief Add material textures to bindless array
     *
     * Updates the MaterialTextures structure with bindless array indices.
     *
     * @param textures Material textures to add
     * @param bindless Bindless array to update
     * @param stream Stream for GPU operations
     * @return Base index in bindless array (or number of textures added)
     */
    [[nodiscard]] uint addToBindless(
        MaterialTextures& textures,
        BindlessArray& bindless,
        Stream& stream);

private:
    /**
     * @brief Try to load a texture with multiple possible name variations
     *
     * @param folder Base folder path
     * @param name Material name
     * @param suffixes List of suffixes to try (e.g., "_albedo", "_diffuse", "")
     * @param srgbToLinear Decode sRGB -> linear (color data only, default false)
     * @return Loaded texture, or an invalid image if not found — shader-side
     *         fallbacks (constant material values, geometric normal) take over
     */
    [[nodiscard]] Image<float> tryLoadTexture(
        const std::filesystem::path& folder,
        const std::string& name,
        const std::vector<std::string>& suffixes,
        bool srgbToLinear = false);

    /**
     * @brief Pack separate roughness, metallic, AO textures into combined RMA texture
     *
     * Creates a new texture where:
     * - R channel = roughness
     * - G channel = metallic
     * - B channel = AO
     *
     * @param roughness Roughness texture (can be invalid placeholder)
     * @param metallic Metallic texture (can be invalid placeholder)
     * @param ao AO texture (can be invalid placeholder)
     * @return Combined RMA texture
     */
    [[nodiscard]] Image<float> packRMA(
        const Image<float>& roughness,
        const Image<float>& metallic,
        const Image<float>& ao);

    /**
     * @brief Create a placeholder 1x1 white texture
     */
    [[nodiscard]] Image<float> createPlaceholderTexture(float value = 1.0f);

    /**
     * @brief Create a placeholder 1x1 RMA texture (R=0.5, M=0.0, AO=1.0)
     */
    [[nodiscard]] Image<float> createPlaceholderRMATexture();

    /**
     * @brief Create a placeholder 1x1 normal texture (flat normal)
     */
    [[nodiscard]] Image<float> createPlaceholderNormalTexture();

    Device& mDevice;
    Stream  mStream;
};

//==============================================================================
// TextureConverter
//==============================================================================

/**
 * @brief Convert Cinder Surface to LuisaCompute Image
 *
 * Provides easy interface for loading images through Cinder's
 * asset system and converting them for GPU path tracing.
 *
 * @code
 * // Load from asset
 * auto img = TextureConverter::loadAsset("textures/wood.png", device);
 *
 * // Convert from Surface
 * ci::Surface8u surface = ...;
 * auto img = TextureConverter::createTexture(surface, device);
 * @endcode
 */
class TextureConverter {
public:
    //==========================================================================
    // Cinder Surface Conversion
    //==========================================================================

    /**
     * @brief Create an image from Cinder Surface8u
     *
     * Uploads as HALF4. 8-bit image files are sRGB-encoded by convention;
     * pass srgbToLinear=true for color data (albedo, emissive) to decode via
     * the exact sRGB EOTF on the CPU. Leave it false for non-color data
     * (normal maps, roughness/metallic/AO) — those must not be converted.
     *
     * @param surface Cinder 8-bit surface (RGBA/RGB)
     * @param device LuisaCompute device
     * @param stream Optional stream for the upload (caller's stream)
     * @param srgbToLinear Decode sRGB -> linear (color data only, default false)
     * @return LuisaCompute image
     */
    [[nodiscard]] static Image<float> createTexture(
        const ci::Surface8u& surface,
        Device& device,
        Stream* stream = nullptr,
        bool srgbToLinear = false);

    /**
     * @brief Create an image from Cinder Surface32f (HDR)
     *
     * Float surfaces are assumed to be in linear color space.
     *
     * @param surface Cinder 32-bit float surface (RGBA/RGB)
     * @param device LuisaCompute device
     * @param generateMipmaps Whether to generate mipmaps (default: true)
     * @return LuisaCompute image
     */
    [[nodiscard]] static Image<float> createTexture(
        const ci::Surface32f& surface,
        Device& device,
        bool generateMipmaps = true);

    /**
     * @brief Create an image from Channel8u (grayscale)
     */
    [[nodiscard]] static Image<float> createTexture(
        const ci::Channel8u& channel,
        Device& device,
        bool generateMipmaps = true);

    /**
     * @brief Create an image from Channel32f (grayscale float)
     */
    [[nodiscard]] static Image<float> createTexture(
        const ci::Channel32f& channel,
        Device& device,
        bool generateMipmaps = true);

    //==========================================================================
    // Asset Loading
    //==========================================================================

    /**
     * @brief Load image from asset path (Cinder asset system)
     *
     * Uses Cinder's asset loading and handles common image formats:
     * PNG, JPG, EXR, HDR, TGA, BMP, etc. HDR formats are linear already;
     * srgbToLinear only affects 8-bit formats.
     *
     * @param assetPath Path relative to assets/ folder
     * @param device LuisaCompute device
     * @param generateMipmaps Whether to generate mipmaps (default: true)
     * @param srgbToLinear Decode sRGB -> linear for 8-bit formats (default false)
     * @return LuisaCompute image
     */
    [[nodiscard]] static Image<float> loadAsset(
        const std::filesystem::path& assetPath,
        Device& device,
        Stream* stream = nullptr,
        bool generateMipmaps = true,
        bool srgbToLinear = false);

    /**
     * @brief Load image from absolute file path
     *
     * @param filePath Absolute path to image file
     * @param device LuisaCompute device
     * @param stream Optional stream for the upload (caller's stream)
     * @param srgbToLinear Decode sRGB -> linear (color data only, default false)
     * @return LuisaCompute image
     */
    [[nodiscard]] static Image<float> loadFile(
        const std::filesystem::path& filePath,
        Device& device,
        Stream* stream = nullptr,
        bool srgbToLinear = false);

    /**
     * @brief Load image from an embedded exe resource (production twin of loadAsset)
     *
     * The Bundler rewrites literal `loadAsset("path", ...)` call sites into
     * `loadResource(RES_NAME, ...)`, where the CINDER_RESOURCE macro expands
     * into the leading (resourcePath, mswID, mswType) triple. The format is
     * detected from resourcePath's extension, same as loadAsset.
     *
     * @param resourcePath Path carried by the CINDER_RESOURCE macro
     * @param mswID Resource ID carried by the CINDER_RESOURCE macro
     * @param mswType Resource type string carried by the CINDER_RESOURCE macro
     * @param device LuisaCompute device
     * @param generateMipmaps Whether to generate mipmaps for HDR formats (default: true)
     * @param srgbToLinear Decode sRGB -> linear for 8-bit formats (default false)
     * @return LuisaCompute image
     */
    [[nodiscard]] static Image<float> loadResource(
        const std::filesystem::path& resourcePath,
        int mswID,
        const std::string& mswType,
        Device& device,
        Stream* stream = nullptr,
        bool generateMipmaps = true,
        bool srgbToLinear = false);

    //==========================================================================
    // Texture Compression
    //==========================================================================

    /**
     * @brief Compress a texture using GPU-accelerated BCn block compression
     *
     * Uses LuisaCompute's TexCompressExt device extension for fast GPU compression.
     * The compressed texture is transparent to shaders — hardware decompresses on sample.
     *
     * Generates a full CPU box-filtered mip chain and compresses every level,
     * so trilinear implicit-LOD sampling never reads uninitialized mips.
     * The chain stops before any level would be smaller than one BC block.
     *
     * @param source Source image (HALF4 or FLOAT4 storage)
     * @param device LuisaCompute device (must support TexCompressExt)
     * @param stream GPU stream for compression operations
     * @param format Target format (BC7 for LDR, BC6 for HDR)
     * @return Compressed image, or invalid image if compression is skipped
     *         (source < 4x4, non-block-aligned dimensions, unsupported storage)
     *         or fails — the caller keeps the uncompressed texture
     */
    [[nodiscard]] static Image<float> compressTexture(
        const Image<float>& source,
        Device& device,
        Stream& stream,
        PixelStorage format = PixelStorage::BC7);

    //==========================================================================
    // Bindless Array Creation
    //==========================================================================

    /**
     * @brief Create a bindless texture array from images
     *
     * @param images Vector of images to add
     * @param device LuisaCompute device
     * @return Bindless array with images
     */
    [[nodiscard]] static BindlessArray createBindlessArray(
        vector<Image<float>>& images,
        Device& device);

private:
    //==========================================================================
    // Pixel Format Conversion
    //==========================================================================

    [[nodiscard]] static PixelStorage getStorageFormat(const ci::Surface8u& surface);
    [[nodiscard]] static PixelStorage getStorageFormat(const ci::Surface32f& surface);
    [[nodiscard]] static PixelStorage getStorageFormat(const ci::Channel8u& channel);
    [[nodiscard]] static PixelStorage getStorageFormat(const ci::Channel32f& channel);

    //==========================================================================
    // Image Upload
    //==========================================================================

    /**
     * @brief Upload CPU data to GPU image
     */
    template<typename T>
    static void uploadImage(
        Image<float>& image,
        const T* data,
        size_t count,
        Stream& stream);
};

//==============================================================================
// Inline Implementations
//==============================================================================

template<typename T>
void TextureConverter::uploadImage(
    Image<float>& image,
    const T* data,
    size_t count,
    Stream& stream) {

    stream << image.copy_from(data)
           << synchronize();
}

} // namespace newtype::render
