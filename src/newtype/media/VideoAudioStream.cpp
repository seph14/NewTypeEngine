#include "newtype/core/Config.h"

#if NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO

#include "newtype/media/VideoAudioStream.h"

#include <dxgi1_4.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

#include <algorithm>

using Microsoft::WRL::ComPtr;

namespace newtype::media {

int VideoAudioStream::s_instanceCount = 0;

VideoAudioStream::~VideoAudioStream() {
    close();
}

bool VideoAudioStream::open(std::filesystem::path const& path,
                            uint32_t outputSampleRate, uint32_t outputChannels) {
    if (_reader) {
        CI_LOG_W("VideoAudioStream::open called while already open; ignoring");
        return false;
    }

    // Own MFStartup refcount, same pattern as VideoDecoderD3D11. In practice the
    // decoder is opened first, but this stands alone so open order isn't load-bearing.
    if (s_instanceCount == 0) {
        HRESULT hr = MFStartup(MF_VERSION, MFSTARTUP_LITE);
        if (FAILED(hr)) {
            CI_LOG_E("VideoAudioStream: MFStartup failed: hr=0x" << std::hex << hr);
            return false;
        }
    }
    ++s_instanceCount;

    std::wstring wpath = path.wstring();
    ComPtr<IMFSourceReader> reader;
    HRESULT hr = MFCreateSourceReaderFromURL(wpath.c_str(), nullptr, &reader);
    if (FAILED(hr)) {
        CI_LOG_E("VideoAudioStream: MFCreateSourceReaderFromURL failed for '"
                 << path.string() << "': hr=0x" << std::hex << hr);
        close();
        return false;
    }

    // Audio stream only — deselect everything else so the demuxer doesn't
    // queue packets for streams we never read.
    hr = reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE);
    if (FAILED(hr)) {
        CI_LOG_E("VideoAudioStream: SetStreamSelection(ALL, FALSE) failed: hr=0x" << std::hex << hr);
        close();
        return false;
    }
    hr = reader->SetStreamSelection(MF_SOURCE_READER_FIRST_AUDIO_STREAM, TRUE);
    if (FAILED(hr)) {
        // Most common cause: the file has no audio stream at all.
        CI_LOG_I("VideoAudioStream: no audio stream in '" << path.string()
                 << "' - playing video-only (hr=0x" << std::hex << hr << ")");
        close();
        return false;
    }

    // Full output type: float32 at the graph's rate/channel count. MF splices in
    // whatever decoder + resampler/channel-mixer MFTs the source needs.
    ComPtr<IMFMediaType> outType;
    hr = MFCreateMediaType(&outType);
    if (SUCCEEDED(hr)) hr = outType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Audio);
    if (SUCCEEDED(hr)) hr = outType->SetGUID(MF_MT_SUBTYPE, MFAudioFormat_Float);
    if (SUCCEEDED(hr)) hr = outType->SetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, outputSampleRate);
    if (SUCCEEDED(hr)) hr = outType->SetUINT32(MF_MT_AUDIO_NUM_CHANNELS, outputChannels);
    if (SUCCEEDED(hr)) hr = reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM, nullptr, outType.Get());
    if (FAILED(hr)) {
        CI_LOG_E("VideoAudioStream: SetCurrentMediaType(Float32 " << outputSampleRate
                 << "Hz x" << outputChannels << ") failed: hr=0x" << std::hex << hr);
        close();
        return false;
    }

    // Read back what MF actually committed; a mismatch would corrupt the ring's
    // layout assumptions downstream, so fail loudly rather than play garbage.
    ComPtr<IMFMediaType> actual;
    hr = reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_AUDIO_STREAM, &actual);
    UINT32 actualRate = 0, actualChannels = 0;
    if (SUCCEEDED(hr)) actual->GetUINT32(MF_MT_AUDIO_SAMPLES_PER_SECOND, &actualRate);
    if (SUCCEEDED(hr)) actual->GetUINT32(MF_MT_AUDIO_NUM_CHANNELS, &actualChannels);
    if (FAILED(hr) || actualRate != outputSampleRate || actualChannels != outputChannels) {
        CI_LOG_E("VideoAudioStream: MF committed " << actualRate << "Hz x" << actualChannels
                 << ", requested " << outputSampleRate << "Hz x" << outputChannels);
        close();
        return false;
    }
    _sampleRate = actualRate;
    _channels   = actualChannels;

    // Duration from the presentation descriptor (0 if unavailable — non-fatal).
    PROPVARIANT dur; PropVariantInit(&dur);
    hr = reader->GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &dur);
    if (SUCCEEDED(hr) && dur.vt == VT_UI8) {
        _durationSec = static_cast<double>(dur.uhVal.QuadPart) * 1e-7;
    }
    PropVariantClear(&dur);

    _reader = reader.Detach();
    _eof.store(false, std::memory_order_relaxed);
    _anchorPending = true;    // anchor position on the first decoded sample's PTS
    _positionSec = 0.0;
    _anchorSec.store(0.0, std::memory_order_relaxed);
    _carryFrames = 0;
    _carryOffset = 0;

    CI_LOG_D("VideoAudioStream opened: " << _sampleRate << "Hz x" << _channels
             << " float32, duration=" << _durationSec << "s");
    return true;
}

bool VideoAudioStream::read_next(float* dst, size_t maxFrames, size_t& outFrames) {
    outFrames = 0;
    if (!_reader) return false;
    if (_eof.load(std::memory_order_relaxed) && _carryFrames == 0) return false;

    if (_carryOffset >= _carryFrames) {
        _carryFrames = 0;
        _carryOffset = 0;

        DWORD    flags = 0;
        LONGLONG pts   = 0;
        ComPtr<IMFSample> sample;

        HRESULT hr = _reader->ReadSample(
            MF_SOURCE_READER_FIRST_AUDIO_STREAM,
            0,           // synchronous (no callback configured)
            nullptr,     // actualStreamIndex out
            &flags,      // streamFlags out
            &pts,        // timestamp out
            &sample);    // sample out

        if (FAILED(hr)) {
            CI_LOG_E("VideoAudioStream: ReadSample failed: hr=0x" << std::hex << hr);
            _eof.store(true, std::memory_order_relaxed);
            return false;
        }
        if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
            _eof.store(true, std::memory_order_relaxed);
            return false;
        }
        if (!sample) {
            // No-data event (e.g. format change) — caller retries next pump.
            return true;
        }

        if (_anchorPending) {
            _positionSec = static_cast<double>(pts) * 1e-7;
            _anchorSec.store(_positionSec, std::memory_order_relaxed);
            _anchorPending = false;
        }

        ComPtr<IMFMediaBuffer> buffer;
        hr = sample->ConvertToContiguousBuffer(&buffer);
        if (FAILED(hr)) {
            CI_LOG_E("VideoAudioStream: ConvertToContiguousBuffer failed: hr=0x" << std::hex << hr);
            return false;
        }
        BYTE* data = nullptr;
        DWORD dataLen = 0;
        hr = buffer->Lock(&data, nullptr, &dataLen);
        if (FAILED(hr)) {
            CI_LOG_E("VideoAudioStream: buffer Lock failed: hr=0x" << std::hex << hr);
            return false;
        }

        // Take the WHOLE packet, whatever its size — truncating here would drop
        // audible chunks of every packet larger than the caller's maxFrames
        // (AAC = 1024 frames; resampler output can be larger still).
        size_t const sampleFrames = (dataLen / sizeof(float)) / _channels;
        if (_carry.size() < sampleFrames * _channels) _carry.resize(sampleFrames * _channels);
        std::memcpy(_carry.data(), data, sampleFrames * _channels * sizeof(float));
        buffer->Unlock();

        _carryFrames  = sampleFrames;
        _carryOffset  = 0;
    }

    size_t const copyFrames = std::min(maxFrames, _carryFrames - _carryOffset);
    std::memcpy(dst, _carry.data() + _carryOffset * _channels,
                copyFrames * _channels * sizeof(float));
    _carryOffset += copyFrames;

    outFrames = copyFrames;
    _positionSec += static_cast<double>(copyFrames) / static_cast<double>(_sampleRate);
    return true;
}

bool VideoAudioStream::seek(double sec) {
    if (!_reader) return false;
    if (sec < 0.0) sec = 0.0;

    // Same manual VT_I8 100-ns variant as VideoDecoderD3D11::seek — this SDK's
    // mfreadwrite.h takes the PROPVARIANT by const reference.
    PROPVARIANT pos;
    PropVariantInit(&pos);
    pos.vt             = VT_I8;
    pos.hVal.QuadPart  = static_cast<LONGLONG>(sec * 1e7);
    HRESULT hr = _reader->SetCurrentPosition(GUID_NULL, pos);
    PropVariantClear(&pos);
    if (FAILED(hr)) {
        CI_LOG_E("VideoAudioStream: SetCurrentPosition(" << sec << "s) failed: hr=0x" << std::hex << hr);
        return false;
    }

    _eof.store(false, std::memory_order_relaxed);
    _anchorPending = true;
    _carryFrames = 0;     // discard any post-seek stale packet remainder
    _carryOffset = 0;
    _positionSec = sec;   // provisional — first post-seek PTS is authoritative
    _anchorSec.store(sec, std::memory_order_relaxed);  // provisional, same rule
    return true;
}

void VideoAudioStream::close() {
    if (_reader) {
        _reader->Flush(MF_SOURCE_READER_ALL_STREAMS);
        _reader->Release();
        _reader = nullptr;
    }
    if (s_instanceCount > 0 && --s_instanceCount == 0) {
        MFShutdown();
    }
}

} // namespace newtype::media

#endif // NT_ENABLE_MEDIA_PLAYER && NT_ENABLE_AUDIO
