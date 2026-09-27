#pragma once

#include "newtype/core/Config.h"

#if NT_ENABLE_AUDIO

#include "cinder/audio/audio.h"
#include "cinder/DataSource.h"
#include "cinder/Filesystem.h"
#include "cinder/Json.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace newtype::util {

/// Real-time SFX + BGM controller. Singleton accessed via `SoundController::get()`.
///
/// Generalizes the older ref/SoundController's parallel water/fish effect code
/// paths into a single `unordered_map<string, EffectDef>`. Each EffectDef owns:
///   - a vector of buffers (one per variation — random pick on play)
///   - a fadeRate (per-frame level decrement when autoFade is on)
///   - a map of currently-playing instances keyed by user-supplied uid
///
/// BGM is kept as a dedicated single instance with its own API
/// (playBackgroundTrack / setBackgroundTrackLevel).
class SoundController {
public:
    /// One active audio graph (BufferPlayer -> Gain -> output) per playing instance.
    /// Spatial voices insert an ambisonics encoder after the gain
    /// (BufferPlayer -> Gain -> Encoder -> binaural bus -> output) and keep it
    /// in `spatialEncoder` for teardown on retire.
    struct SoundEffectNode {
        ci::audio::BufferPlayerNodeRef node;
        ci::audio::GainNodeRef         gain;
        ci::audio::NodeRef             spatialEncoder;
        float                          level    = 0.f;
        bool                           autoFade = false;
        bool                           isSpatial = false;
    };

private:
    /// Per-effect registration + active instances.
    struct EffectDef {
        std::vector<ci::audio::BufferRef>            buffers;
        float                                        fadeRate = 0.f;
        std::unordered_map<int, SoundEffectNode>     active;
    };

    float                                           _level = 1.f;
    bool                                            _audioAvailable = false;
    ci::audio::NodeRef                              _output;
    SoundEffectNode                                 _bgm;
    std::unordered_map<std::string, EffectDef>      _effects;

    int                                             _testUid = 0;

    SoundController();

    /// Shared one-shot builder; `worldPos` (optional) routes the voice
    /// through the spatial bus when spatial audio is available.
    void playEffectInternal(const std::string& id, int uid, float level,
                            const ci::vec3* worldPos);

public:
    ~SoundController();
    SoundController(const SoundController&)            = delete;
    SoundController& operator=(const SoundController&) = delete;

    /// Lazy singleton — first call constructs, lives until program exit.
    [[nodiscard]] static SoundController& get() {
        static SoundController instance;
        return instance;
    }

    // ---------- Master ----------
    void setSoundLevel(float level);
    /// Redirect the output node for *future* play calls. Existing nodes keep
    /// their original routing. Default: master audio context output.
    void setOutput(ci::audio::NodeRef node);

    /// Per-frame: advance auto-fade on all active SFX, retire finished nodes.
    /// Call from the main update loop.
    void update();

    // ---------- BGM (dedicated API) ----------
    /// Asset-relative path version (resolves via `ci::app::loadAsset`).
    void playBackgroundTrack    (const ci::fs::path& assetPath, float level);
    /// Data-source version: accepts `app::loadAsset("audio/x.wav")` or
    /// `app::loadResource(RES_NAME)` directly, so call sites written with the
    /// asset literal are Bundler-convertible.
    void playBackgroundTrack    (const ci::DataSourceRef& source, float level);
    void setBackgroundTrackLevel(float level);
    void stopBackgroundTrack    ();

    // ---------- SFX registration ----------
    /// Create an empty effect slot with the given fade rate.
    void registerEffect  (const std::string& id, float fadeRate);
    /// Append one variation buffer to an existing effect. Resolves `assetPath`
    /// via `ci::app::loadAsset`, so asset-relative paths work.
    void addEffectTrack  (const std::string& id, const ci::fs::path& assetPath);
    /// Data-source version of addEffectTrack (see playBackgroundTrack).
    void addEffectTrack  (const std::string& id, const ci::DataSourceRef& source);
    /// Scan a folder (resolved via `ci::app::getAssetPath`) for audio files
    /// and register each as a variation.
    void loadEffectFolder(const std::string& id, float fadeRate,
                          const ci::fs::path& folder);
    /// JSON-driven registration. See plan for schema.
    void loadConfig      (const ci::fs::path& jsonPath);

    // ---------- SFX playback ----------
    /// Start a one-shot for `(id, uid)` at `level`. If `(id, uid)` is already
    /// active, this is a no-op (matches ref dedup). Picks a random variation
    /// buffer and auto-fades immediately so the one-shot naturally decays.
    void playEffect   (const std::string& id, int uid, float level);
    /// Spatial variant: the one-shot is encoded as a 3rd-order ambisonic
    /// point source at `worldPos` (world space) and decoded binaurally.
    /// Distance attenuation (min(1, ref/d)) is applied on the voice gain.
    /// Multi-channel assets are L/R-averaged by the encoder — author spatial
    /// one-shots as mono for full control. Falls back to the non-spatial
    /// path when spatial audio is unavailable.
    void playEffect   (const std::string& id, int uid, float level,
                       const ci::vec3& worldPos);
    /// Mark an active instance for fade-out. No-op if not currently playing.
    void fadeOutEffect(const std::string& id, int uid);
    [[nodiscard]] bool isPlaying(const std::string& id, int uid) const;

    // ---------- UI ----------
    /// Opens its own ImGui window. Call from the main drawUi() inside an
    /// active ImGui frame.
    void drawUi();
};

} // namespace newtype::util

#else
// NT_ENABLE_AUDIO=0 stub — every method is a no-op, get() returns a static dummy.
namespace newtype::util {
class SoundController {
    SoundController() = default;
public:
    static SoundController& get() { static SoundController i; return i; }
    template<typename... A> void setSoundLevel          (A&&...) {}
    template<typename... A> void setOutput              (A&&...) {}
    template<typename... A> void update                 (A&&...) {}
    template<typename... A> void playBackgroundTrack    (A&&...) {}
    template<typename... A> void setBackgroundTrackLevel(A&&...) {}
    template<typename... A> void stopBackgroundTrack    (A&&...) {}
    template<typename... A> void registerEffect         (A&&...) {}
    template<typename... A> void addEffectTrack         (A&&...) {}
    template<typename... A> void loadEffectFolder       (A&&...) {}
    template<typename... A> void loadConfig             (A&&...) {}
    template<typename... A> void playEffect             (A&&...) {}
    template<typename... A> void fadeOutEffect          (A&&...) {}    template<typename... A> bool isPlaying              (A&&...) const { return false; }
    template<typename... A> void drawUi                 (A&&...) {}
};
} // namespace newtype::util
#endif
