#include "newtype/render/TextureConverter.h"
#include "newtype/util/PixelOps.h"
#include "newtype/util/ParallelFor.h"
#include "cinder/app/App.h"
#include "cinder/ImageIo.h"
#include "cinder/Log.h"
#include <luisa/core/clock.h>
#include <cmath>
#include <algorithm>

namespace newtype::render {

using namespace luisa;
using namespace luisa::compute;

//==============================================================================
// TextureConverter - Texture Compression
//==============================================================================

namespace {

// Max mip levels for a compressed texture. The chain stops well before 1x1
// so every level stays a whole number of BC blocks.
constexpr uint kMaxCompressionMipLevels = 8u;

// Band sizes for util::parallel_for — thread spin-up only pays off above these.
constexpr uint64_t kRowGrain = 16u;          // rows per band (band iterators)
constexpr uint64_t kElementGrain = 65536u;   // float/half elements per band

} // namespace

Image<float> TextureConverter::compressTexture(
    const Image<float>& source,
    Device& device,
    Stream& stream,
    PixelStorage format) {

    // BCn operates on 4x4 blocks and every mip level must be block-aligned,
    // so only accept sources >= 4x4 with both dimensions multiples of 4.
    if (!source.valid()) {
        return {};
    }
    auto size = source.size();
    if (size.x < 4u || size.y < 4u ||
        (size.x & 3u) != 0u || (size.y & 3u) != 0u) {
        CI_LOG_D("Skipping compression of " << size.x << "x" << size.y
                 << " texture (BCn requires dimensions >= 4 and multiples of 4)");
        return {};  // Caller keeps the uncompressed texture
    }
    // DX12 buffer->texture copies require a 256-byte-aligned row pitch, and
    // the DX backend aborts on violation. BC6/BC7 block rows are (w/4)*16
    // bytes and the BYTE4/HALF4 intermediates are w*4/w*8 bytes, so every
    // uploaded level — including level 0 — needs a width that is a multiple
    // of 64 to stay copyable.
    if ((size.x & 63u) != 0u) {
        CI_LOG_D("Skipping compression of " << size.x << "x" << size.y
                 << " texture (DX12 copy row-pitch alignment requires width % 64 == 0)");
        return {};  // Caller keeps the uncompressed texture
    }

    // The compressor input images below are HALF4/BYTE4; downloading level 0
    // in any other layout would reinterpret the bytes.
    auto storage = source.storage();
    if (storage != PixelStorage::HALF4 && storage != PixelStorage::FLOAT4) {
        CI_LOG_D("Skipping compression: unsupported source storage (need HALF4 or FLOAT4)");
        return {};
    }

    // Query compression extension
    auto tex_ext = device.extension<TexCompressExt>();
    if (!tex_ext) {
        CI_LOG_W("TexCompressExt not available, skipping compression");
        return {};
    }

    // Mip count: keep halving while the level stays a whole number of blocks
    // AND keeps a 256-byte-aligned row pitch (mip width % 64 == 0, see the
    // level-0 check above). Samplers are trilinear with implicit-LOD
    // sampling, so every level of the compressed image must be initialized —
    // leaving any level unwritten feeds garbage to minified pixels.
    uint levels = 1u;
    while (levels < kMaxCompressionMipLevels) {
        uint mw = size.x >> levels;
        uint mh = size.y >> levels;
        if (mw < 64u || (mw & 63u) != 0u || mh < 4u || (mh & 3u) != 0u) break;
        ++levels;
    }

    // Download level 0 in its native storage, then build the mip chain on the
    // CPU as float4 (box-filtered 2x2 — exact here since dimensions halve).
    std::vector<luisa::float4> levelPixels[kMaxCompressionMipLevels];
    size_t texelCount = static_cast<size_t>(size.x) * size.y;
    if (storage == PixelStorage::HALF4) {
        vector<half4> tmp(texelCount);
        stream << source.view(0).copy_to(luisa::span{tmp})
               << synchronize();
        levelPixels[0].resize(texelCount);
        auto* src = reinterpret_cast<const luisa::half*>(tmp.data());
        auto* dst = reinterpret_cast<float*>(levelPixels[0].data());
        util::parallel_for(texelCount * 4u, kElementGrain,
                           [&](uint64_t a, uint64_t b) {
                               util::f16ToF32(src + a, dst + a,
                                              static_cast<size_t>(b - a));
                           });
    } else {
        levelPixels[0].resize(texelCount);
        stream << source.view(0).copy_to(luisa::span{levelPixels[0]})
               << synchronize();
    }
    for (uint l = 1u; l < levels; ++l) {
        uint sw = size.x >> (l - 1u), sh = size.y >> (l - 1u);
        uint dw = size.x >> l,        dh = size.y >> l;
        const auto& src = levelPixels[l - 1u];
        auto& dst = levelPixels[l];
        dst.resize(static_cast<size_t>(dw) * dh);
        util::parallel_for(dh, kRowGrain, [&](uint64_t y0, uint64_t y1) {
            util::boxDownsample2x2(src.data() + 2u * y0 * sw, sw,
                                   dst.data() + y0 * dw, dw,
                                   static_cast<uint>(y1 - y0));
        });
    }

    // Compressor input per target format:
    // - BC7 is UNORM: quantize each level to 8-bit on the CPU (lossless for
    //   8-bit-origin data; the old GPU float->BYTE4 convert clamped HDR and
    //   ran only on level 0).
    // - BC6H is HDR half: keep the full range so emissive survives.
    const bool isHdr = (format == PixelStorage::BC6);
    Image<float> mipped = device.create_image<float>(
        isHdr ? PixelStorage::HALF4 : PixelStorage::BYTE4, size, levels);
    for (uint l = 0u; l < levels; ++l) {
        size_t n = static_cast<size_t>(size.x >> l) * (size.y >> l);
        if (isHdr) {
            vector<half4> pixels(n);
            auto* src = reinterpret_cast<const float*>(levelPixels[l].data());
            auto* dst = reinterpret_cast<luisa::half*>(pixels.data());
            util::parallel_for(n * 4u, kElementGrain, [&](uint64_t a, uint64_t b) {
                util::f32ToF16(src + a, dst + a, static_cast<size_t>(b - a));
            });
            stream << mipped.view(l).copy_from(pixels.data());
        } else {
            // BYTE4 = 4 bytes/texel in RGBA order
            std::vector<uint8_t> pixels(n * 4u);
            util::parallel_for(n, kElementGrain, [&](uint64_t a, uint64_t b) {
                util::quantizeUnorm8(levelPixels[l].data() + a, pixels.data() + a * 4u,
                                     static_cast<size_t>(b - a));
            });
            stream << mipped.view(l).copy_from(pixels.data());
        }
    }

    // Run GPU compression per level. BC6/BC7 blocks are 16 bytes; with all
    // levels block-aligned, level l is (w/4)*(h/4) blocks = w*h bytes.
    Clock clk;
    clk.tic();

    size_t totalUints = 0u;
    for (uint l = 0u; l < levels; ++l) {
        totalUints += static_cast<size_t>(size.x >> l) * (size.y >> l) / 4u;
    }
    Buffer<uint> staging = device.create_buffer<uint>(totalUints);

    TexCompressExt::Result result = TexCompressExt::Result::Success;
    size_t offset = 0u;
    for (uint l = 0u; l < levels && result == TexCompressExt::Result::Success; ++l) {
        size_t uints = static_cast<size_t>(size.x >> l) * (size.y >> l) / 4u;
        result = isHdr
            ? tex_ext->compress_bc6h(stream, mipped.view(l), staging.view(offset, uints))
            // BC7 — alpha_importance=0 for albedo/RMA, could be higher for alpha textures
            : tex_ext->compress_bc7(stream, mipped.view(l), staging.view(offset, uints), 0.0f);
        offset += uints;
    }
    stream << synchronize();

    if (result != TexCompressExt::Result::Success) {
        CI_LOG_W("Texture compression failed (format="
                 << (isHdr ? "BC6H" : "BC7")
                 << "), keeping uncompressed");
        return {};
    }

    // Copy compressed data from buffer to image, one level at a time
    auto compressed = device.create_image<float>(format, size, levels);
    offset = 0u;
    for (uint l = 0u; l < levels; ++l) {
        size_t uints = static_cast<size_t>(size.x >> l) * (size.y >> l) / 4u;
        stream << compressed.view(l).copy_from(staging.view(offset, uints));
        offset += uints;
    }
    stream << synchronize();

    auto elapsed = clk.toc();
    auto srcBytes = static_cast<uint64_t>(texelCount) *
                    (storage == PixelStorage::HALF4 ? 8u : 16u);
    auto dstBytes = static_cast<uint64_t>(totalUints) * 4u;
    CI_LOG_I("Compressed " << size.x << "x" << size.y << " texture to "
             << (isHdr ? "BC6H" : "BC7") << " (" << levels << " mips) in "
             << elapsed << " ms"
             << " (" << srcBytes << " -> " << dstBytes << " bytes, "
             << (srcBytes / std::max(dstBytes, uint64_t{1})) << "x reduction)");

    return compressed;
}

//==============================================================================
// MaterialTextureLoader
//==============================================================================

MaterialTextureLoader::MaterialTextureLoader(Device& device)
    : mDevice(device)
    , mStream(device.create_stream(StreamTag::GRAPHICS)) {
}

MaterialTextures MaterialTextureLoader::loadMaterial(
    const std::filesystem::path& folderPath,
    const std::string& name,
    const TextureCompressionSettings& compression) {

    CI_LOG_I("Loading material '" << name << "' from " << folderPath
             << (compression.enableCompression ? " (with BCn compression)" : ""));

    MaterialTextures textures;

    // Try to load each texture type with multiple suffix variations.
    // sRGB decode is applied to color data only (albedo/emissive); normals
    // and RMA are non-color data and must not be converted.
    textures.albedo = tryLoadTexture(folderPath, name, {
        "_albedo", "_diffuse", "_color", "_basecolor", "_base_color",
        "_Albedo", "_Diffuse", "_Color", "",  // Empty suffix tries {name}.png directly
    }, true);

    textures.normal = tryLoadTexture(folderPath, name, {
        "_normal", "_nrm", "_n", "_Normal", "_N"
    });

    textures.emissive = tryLoadTexture(folderPath, name, {
        "_emissive", "_emission", "_emi", "_e",
        "_Emissive", "_Emission", "_E"
    }, true);

    // Try to load combined RMA texture first (common naming: _rma, _orm, _rmo)
    Image<float> combinedRMA = tryLoadTexture(folderPath, name, {
        "_rma", "_orm", "_rmo", "_RMA", "_ORM", "_RMO"
    });

    if (combinedRMA.valid() && combinedRMA.size().x >= 4u) {
        // Found combined RMA texture
        CI_LOG_I("  Using combined RMA texture");
        textures.rma = std::move(combinedRMA);
    } else {
        // No combined RMA found, load separate textures and pack them
        CI_LOG_I("  No combined RMA found, packing separate textures");
        Image<float> roughness = tryLoadTexture(folderPath, name, {
            "_roughness", "_rough", "_r", "_Roughness", "_R"
        });
        Image<float> metallic = tryLoadTexture(folderPath, name, {
            "_metallic", "_metal", "_m", "_Metallic", "_M"
        });
        Image<float> ao = tryLoadTexture(folderPath, name, {
            "_ao", "_ambient_occlusion", "_occlusion", "_AO",
            "_AmbientOcclusion", "_Occlusion"
        });

        textures.rma = packRMA(roughness, metallic, ao);
    }

    // Apply BCn compression if enabled
    if (compression.enableCompression) {
        auto tryCompress = [&](Image<float>& tex, bool shouldCompress, PixelStorage format, const char* label) {
            if (shouldCompress && tex.valid() && tex.size().x >= 4u && tex.size().y >= 4u) {
                auto compressed = TextureConverter::compressTexture(tex, mDevice, mStream, format);
                if (compressed.valid()) {
                    tex = std::move(compressed);
                }
            }
        };

        tryCompress(textures.albedo,   compression.compressAlbedo,   PixelStorage::BC7, "albedo");
        tryCompress(textures.normal,   compression.compressNormal,   PixelStorage::BC7, "normal");
        tryCompress(textures.rma,      compression.compressRMA,      PixelStorage::BC7, "rma");
        tryCompress(textures.emissive, compression.compressEmissive, PixelStorage::BC6, "emissive");
    }

    // Set indices in MaterialTextures structure
    textures.albedoIdx   = textures.hasAlbedo() ? 0 : -1;
    textures.normalIdx   = textures.hasNormal() ? 0 : -1;
    textures.rmaIdx      = textures.hasRMA() ? 0 : -1;
    textures.emissiveIdx = textures.hasEmissive() ? 0 : -1;

    return textures;
}

uint MaterialTextureLoader::addToBindless(
    MaterialTextures& textures,
    BindlessArray& bindless,
    Stream& stream) {

    uint nextIdx = 0;

    auto assignSlot = [&](const Image<float>& img, int& idx) {
        if (img.valid() && idx < 0) {
            idx = static_cast<int>(nextIdx++);
            bindless.emplace_on_update(idx, img, Sampler::linear_linear_mirror());
        }
    };

    assignSlot(textures.albedo, textures.albedoIdx);
    assignSlot(textures.normal, textures.normalIdx);
    assignSlot(textures.rma, textures.rmaIdx);
    assignSlot(textures.emissive, textures.emissiveIdx);

    stream << bindless.update();

    return nextIdx;
}

Image<float> MaterialTextureLoader::tryLoadTexture(
    const std::filesystem::path& folder,
    const std::string& name,
    const std::vector<std::string>& suffixes,
    bool srgbToLinear) {

    // Try multiple file extensions
    const std::vector<std::string> extensions = {
        ".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".exr", ".hdr"
    };

    for (const auto& suffix : suffixes) {
        for (const auto& ext : extensions) {
            auto path = folder / (name + suffix + ext);

            // Try absolute path first
            try {
                if (std::filesystem::exists(path)) {
                    CI_LOG_V("  Loading texture: " << path);
                    return TextureConverter::loadFile(path, mDevice, nullptr, srgbToLinear);
                }
            } catch (...) {
                // Continue to next combination
            }

            // Try loading as Cinder asset (relative to assets folder)
            try {
                auto assetPath = path.string();
                // Remove leading "assets/" if present for Cinder's loadAsset
                if (assetPath.find("assets/") == 0) {
                    assetPath = assetPath.substr(7);
                }

                auto source = ci::app::loadAsset(assetPath);
                if (source) {
                    CI_LOG_V("  Loading asset texture: " << assetPath);
                    return TextureConverter::loadAsset(path, mDevice, true, srgbToLinear);
                }
            } catch (...) {
                // Continue
            }
        }
    }

    // Return an invalid image on miss — the existing shader-side -1 fallbacks
    // take over (geometric normal, constant roughness/metallic, emission).
    // A white placeholder would decode to a tilted tangent normal (1,1,0)
    // for normal maps and tint every missing channel.
    CI_LOG_V("  Texture not found: " << (folder / name));
    return {};
}

Image<float> MaterialTextureLoader::createPlaceholderTexture(float value) {
    // Create a 1x1 white/gray texture
    const float pixels[] = {value, value, value, 1.0f};

    auto image = mDevice.create_image<float>(PixelStorage::FLOAT4, 1, 1);
    mStream << image.copy_from(pixels)
           << synchronize();

    return image;
}

Image<float> MaterialTextureLoader::createPlaceholderRMATexture() {
    // Default RMA values: R=0.5 (medium roughness), G=0.0 (non-metallic), B=1.0 (full AO)
    const float pixels[] = {0.5f, 0.0f, 1.0f, 1.0f};

    auto image = mDevice.create_image<float>(PixelStorage::FLOAT4, 1, 1);
    mStream << image.copy_from(pixels)
           << synchronize();

    return image;
}

Image<float> MaterialTextureLoader::createPlaceholderNormalTexture() {
    // Flat normal pointing in +Z direction (0, 0, 1) in tangent space
    // Stored as (R, G, B) = (X, Y, Z) mapped to [0, 1]
    // So (0, 0, 1) becomes (0.5, 0.5, 1.0)
    const float pixels[] = {0.5f, 0.5f, 1.0f, 1.0f};

    auto image = mDevice.create_image<float>(PixelStorage::FLOAT4, 1, 1);
    mStream << image.copy_from(pixels)
           << synchronize();

    return image;
}

Image<float> MaterialTextureLoader::packRMA(
    const Image<float>& roughness,
    const Image<float>& metallic,
    const Image<float>& ao) {

    // If all textures are invalid (1x1 placeholders), return a default RMA
    bool allPlaceholders = true;
    uint width = 1, height = 1;

    auto checkAndGetSize = [&](const Image<float>& img, float placeholderValue) {
        if (img.valid()) {
            uint2 size = img.size();
            if (size.x > 1 || size.y > 1) {
                allPlaceholders = false;
                width = std::max(width, size.x);
                height = std::max(height, size.y);
            }
        }
    };

    checkAndGetSize(roughness, 0.5f);
    checkAndGetSize(metallic, 0.0f);
    checkAndGetSize(ao, 1.0f);

    if (allPlaceholders) {
        CI_LOG_V("  All RMA textures are placeholders, using default");
        return createPlaceholderRMATexture();
    }

    CI_LOG_V("  Packing RMA texture: " << width << "x" << height);

    // Create combined RMA texture
    Image<float> rma = mDevice.create_image<float>(PixelStorage::HALF4, width, height);

    // Helper to read a texture channel as floats resampled to the target
    // size. Downloads in the texture's native storage (a fixed float4 buffer
    // would either trip the debug size assertion or reinterpret half data as
    // float garbage), extracts the R channel, and converts in bulk.
    auto readChannel = [&](const Image<float>& tex, float defaultValue) -> vector<float> {
        vector<float> channel(static_cast<size_t>(width) * height, defaultValue);

        // Invalid or 1x1 inputs carry no information — keep the default
        if (!tex.valid()) return channel;
        uint2 texSize = tex.size();
        if (texSize.x <= 1u && texSize.y <= 1u) return channel;

        size_t texelCount = static_cast<size_t>(texSize.x) * texSize.y;
        vector<float> chan(texelCount);
        if (tex.storage() == PixelStorage::HALF4) {
            vector<half4> tmp(texelCount);
            mStream << tex.copy_to(luisa::span{tmp}) << synchronize();
            // Strided gather of the R channel into contiguous halfs, then one
            // bulk half->float pass (F16C 8-wide when available)
            vector<luisa::half> xs(texelCount);
            util::parallel_for(static_cast<uint64_t>(texelCount), kElementGrain,
                               [&](uint64_t a, uint64_t b) {
                for (uint64_t i = a; i < b; i++) xs[i] = tmp[i].x;
            });
            util::parallel_for(static_cast<uint64_t>(texelCount), kElementGrain,
                               [&](uint64_t a, uint64_t b) {
                util::f16ToF32(xs.data() + a, chan.data() + a,
                               static_cast<size_t>(b - a));
            });
        } else if (tex.storage() == PixelStorage::FLOAT4) {
            vector<float4> tmp(texelCount);
            mStream << tex.copy_to(luisa::span{tmp}) << synchronize();
            util::parallel_for(static_cast<uint64_t>(texelCount), kElementGrain,
                               [&](uint64_t a, uint64_t b) {
                for (uint64_t i = a; i < b; i++) chan[i] = tmp[i].x;
            });
        } else {
            CI_LOG_W("  packRMA: unsupported texture storage, using default channel");
            return channel;
        }

        // Scale or tile the texture to match target size
        util::parallel_for(static_cast<uint64_t>(height), kRowGrain,
                           [&](uint64_t y0, uint64_t y1) {
            for (uint64_t y = y0; y < y1; ++y) {
                for (uint x = 0; x < width; ++x) {
                    uint srcX = (x * texSize.x) / width;
                    uint srcY = (static_cast<uint>(y) * texSize.y) / height;
                    channel[y * width + x] = chan[srcY * texSize.x + srcX];  // R channel for grayscale
                }
            }
        });
        return channel;
    };

    auto roughnessChannel = readChannel(roughness, 0.5f);
    auto metallicChannel = readChannel(metallic, 0.0f);
    auto aoChannel = readChannel(ao, 1.0f);

    // Interleave channels as float4, then one bulk float->half pass into the
    // HALF4 upload buffer
    vector<half4> rmaPixels(static_cast<size_t>(width) * height);
    {
        std::vector<luisa::float4> combined(static_cast<size_t>(width) * height);
        util::parallel_for(static_cast<uint64_t>(width) * height, kElementGrain,
                           [&](uint64_t a, uint64_t b) {
            for (uint64_t i = a; i < b; i++) {
                combined[i] = make_float4(
                    roughnessChannel[i],   // R: Roughness
                    metallicChannel[i],    // G: Metallic
                    aoChannel[i],          // B: AO
                    1.0f);                 // A: Unused
            }
        });
        auto* src = reinterpret_cast<const float*>(combined.data());
        auto* dst = reinterpret_cast<luisa::half*>(rmaPixels.data());
        util::parallel_for(static_cast<uint64_t>(width) * height * 4u, kElementGrain,
                           [&](uint64_t a, uint64_t b) {
            util::f32ToF16(src + a, dst + a, static_cast<size_t>(b - a));
        });
    }

    // Upload combined texture
    mStream << rma.copy_from(rmaPixels.data())
           << synchronize();

    return rma;
}

//==============================================================================
// TextureConverter - Pixel Format Conversion
//==============================================================================

// Note: GPU alignment requires 4-component formats (BYTE4, FLOAT4)
// even for RGB images without alpha. Single-channel images use BYTE1/FLOAT1.

PixelStorage TextureConverter::getStorageFormat(const ci::Surface8u& surface) {
    // Always use BYTE4 for GPU alignment (RGB images padded with alpha=1.0f)
    return PixelStorage::BYTE4;
}

PixelStorage TextureConverter::getStorageFormat(const ci::Surface32f& surface) {
    // Always use FLOAT4 for GPU alignment (RGB images padded with alpha=1.0f)
    return PixelStorage::FLOAT4;
}

PixelStorage TextureConverter::getStorageFormat(const ci::Channel8u& channel) {
    return PixelStorage::BYTE1;
}

PixelStorage TextureConverter::getStorageFormat(const ci::Channel32f& channel) {
    return PixelStorage::FLOAT1;
}

//==============================================================================
// TextureConverter - Surface8u
//==============================================================================

Image<float> TextureConverter::createTexture(
    const ci::Surface8u& surface,
    Device& device,
    Stream* stream,
    bool srgbToLinear) {

    auto width = surface.getWidth();
    auto height = surface.getHeight();

    CI_LOG_V("Creating Image from Surface8u: " << width << "x" << height
             << " (HALF4, uint8->float"
             << (srgbToLinear ? " + sRGB->linear decode" : "") << ")");

    // Use HALF4 storage for compatibility with bindless arrays on all backends
    Image<float> image = device.create_image<float>(PixelStorage::HALF4, width, height);

    // Decode into a float4 staging buffer first (band-parallel, LUT-decoded
    // sRGB), then bulk-convert to half4 — per-texel half casts and a pow per
    // channel dominate Debug load time otherwise.
    size_t texelCount = static_cast<size_t>(width) * height;
    std::vector<luisa::float4> staging(texelCount);
    vector<half4> pixels(texelCount);

    const auto& lut = util::srgbDecodeLut();
    const bool hasAlpha = surface.hasAlpha();

    util::parallel_for(static_cast<uint64_t>(height), kRowGrain,
                       [&](uint64_t y0, uint64_t y1) {
        // Per-band iterator over just this row range — threads never share
        // iterator state and write disjoint output rows.
        auto iter = surface.getIter(ci::Area(0, static_cast<int>(y0),
                                             width, static_cast<int>(y1)));
        auto* out = staging.data() + y0 * width;
        if (srgbToLinear) {
            if (hasAlpha) {
                while (iter.line()) {
                    auto* p = out;
                    out += width;
                    while (iter.pixel()) {
                        *p++ = make_float4(
                            lut[iter.r()], lut[iter.g()], lut[iter.b()],
                            static_cast<float>(iter.a()) / 255.0f);
                    }
                }
            } else {
                while (iter.line()) {
                    auto* p = out;
                    out += width;
                    while (iter.pixel()) {
                        *p++ = make_float4(
                            lut[iter.r()], lut[iter.g()], lut[iter.b()], 1.f);
                    }
                }
            }
        } else {
            if (hasAlpha) {
                while (iter.line()) {
                    auto* p = out;
                    out += width;
                    while (iter.pixel()) {
                        *p++ = make_float4(
                            static_cast<float>(iter.r()) / 255.0f,
                            static_cast<float>(iter.g()) / 255.0f,
                            static_cast<float>(iter.b()) / 255.0f,
                            static_cast<float>(iter.a()) / 255.0f);
                    }
                }
            } else {
                while (iter.line()) {
                    auto* p = out;
                    out += width;
                    while (iter.pixel()) {
                        *p++ = make_float4(
                            static_cast<float>(iter.r()) / 255.0f,
                            static_cast<float>(iter.g()) / 255.0f,
                            static_cast<float>(iter.b()) / 255.0f, 1.f);
                    }
                }
            }
        }
    });

    {
        auto* src = reinterpret_cast<const float*>(staging.data());
        auto* dst = reinterpret_cast<luisa::half*>(pixels.data());
        util::parallel_for(texelCount * 4u, kElementGrain,
                           [&](uint64_t a, uint64_t b) {
                               util::f32ToF16(src + a, dst + a,
                                              static_cast<size_t>(b - a));
                           });
    }

    if (stream) {
        // Use caller's stream — upload + bindless update happen on same stream
        *stream << image.copy_from(pixels.data());
    } else {
        // No stream provided — upload on local stream and synchronize
        auto localStream = device.create_stream(StreamTag::GRAPHICS);
        localStream << image.copy_from(pixels.data())
                    << synchronize();
    }

    return image;
}

//==============================================================================
// TextureConverter - Surface32f (HDR)
//==============================================================================

Image<float> TextureConverter::createTexture(
    const ci::Surface32f& surface,
    Device& device,
    bool generateMipmaps) {

    auto width = surface.getWidth();
    auto height = surface.getHeight();
    auto storage = getStorageFormat(surface);

    CI_LOG_V("Creating Image from Surface32f (HDR): " << width << "x" << height
             << " (FLOAT4 - RGB padded with alpha=1.0f for GPU alignment)");

    auto image = device.create_image<float>(storage, width, height);

    // Float surfaces are already linear, just copy (band-parallel — the
    // sequential iterator walk over a 4K envmap is visible in Debug loads)
    vector<float4> pixels(static_cast<size_t>(width) * height);

    util::parallel_for(static_cast<uint64_t>(height), kRowGrain,
                       [&](uint64_t y0, uint64_t y1) {
        auto iter = surface.getIter(ci::Area(0, static_cast<int>(y0),
                                             width, static_cast<int>(y1)));
        auto* out = pixels.data() + y0 * width;
        while (iter.line()) {
            auto* p = out;
            out += width;
            while (iter.pixel()) {
                *p++ = make_float4(iter.r(), iter.g(), iter.b(), iter.a());
            }
        }
    });

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    uploadImage(image, pixels.data(), pixels.size(), stream);

    return image;
}

//==============================================================================
// TextureConverter - Channel8u (Grayscale)
//==============================================================================

Image<float> TextureConverter::createTexture(
    const ci::Channel8u& channel,
    Device& device,
    bool generateMipmaps) {

    auto width = channel.getWidth();
    auto height = channel.getHeight();

    CI_LOG_V("Creating Image from Channel8u: " << width << "x" << height);

    auto image = device.create_image<float>(PixelStorage::HALF1, width, height);

    // Convert to float [0, 1]
    vector<half> pixels;
    pixels.reserve(width * height);

    auto iter = channel.getIter();
    while (iter.line()) {
        while (iter.pixel()) {
            pixels.push_back(static_cast<half>(static_cast<float>(iter.v()) / 255.0f));
        }
    }

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    uploadImage(image, pixels.data(), pixels.size(), stream);

    return image;
}

//==============================================================================
// TextureConverter - Channel32f (Grayscale Float)
//==============================================================================

Image<float> TextureConverter::createTexture(
    const ci::Channel32f& channel,
    Device& device,
    bool generateMipmaps) {

    auto width = channel.getWidth();
    auto height = channel.getHeight();

    CI_LOG_V("Creating Image from Channel32f: " << width << "x" << height);

    auto image = device.create_image<float>(PixelStorage::FLOAT1, width, height);

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    uploadImage(image, channel.getData(), width * height, stream);

    return image;
}

//==============================================================================
// TextureConverter - Asset Loading
//==============================================================================

Image<float> TextureConverter::loadAsset(
    const std::filesystem::path& assetPath,
    Device& device,
    bool generateMipmaps,
    bool srgbToLinear) {

    CI_LOG_I("Loading texture asset: " << assetPath);

    // Use Cinder's asset system
    auto source = ci::app::loadAsset(assetPath.string());

    // Detect format and load
    auto ext = assetPath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // HDR formats — float data is already linear, the flag does not apply
    if (ext == ".exr" || ext == ".hdr" || ext == ".hdri") {
        auto surface = ci::Surface32f(ci::loadImage(source));
        return createTexture(surface, device, generateMipmaps);
    }

    // Standard formats (PNG, JPG, etc.)
    auto surface = ci::Surface8u(ci::loadImage(source));
    return createTexture(surface, device, nullptr, srgbToLinear);
}

Image<float> TextureConverter::loadFile(
    const std::filesystem::path& filePath,
    Device& device,
    Stream* stream,
    bool srgbToLinear) {

    CI_LOG_I("Loading texture file: " << filePath);

    auto ext = filePath.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // Standard formats
    auto surface = ci::Surface8u(ci::loadImage(filePath.string()));
    return createTexture(surface, device, stream, srgbToLinear);
}

//==============================================================================
// TextureConverter - Bindless Array
//==============================================================================

BindlessArray TextureConverter::createBindlessArray(
    vector<Image<float>>& images,
    Device& device) {

    auto bindless = device.create_bindless_array(images.size());

    for (size_t i = 0; i < images.size(); ++i) {
        if (images[i].valid()) {
            bindless.emplace_on_update(i, images[i], Sampler::linear_linear_mirror());
        }
    }

    auto stream = device.create_stream(StreamTag::GRAPHICS);
    stream << bindless.update()
           << synchronize();

    return bindless;
}

} // namespace newtype::render
