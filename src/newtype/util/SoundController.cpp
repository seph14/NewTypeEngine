#include "newtype/util/SoundController.h"

#if NT_ENABLE_AUDIO

#include "cinder/CinderImGui.h"
#include "cinder/Log.h"
#include "cinder/Utilities.h"
#include "cinder/app/App.h"
#include "newtype/util/Rand.h"
#include <algorithm>
#include <cctype>

using namespace ci;
using namespace std;

namespace newtype::util {

namespace {
constexpr auto kSupportedAudioExts = { ".wav", ".mp3", ".ogg", ".aiff", ".aif", ".flac" };

bool isAudioFile(const fs::path& p) {
    if (!fs::is_regular_file(p)) return false;
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (auto&& e : kSupportedAudioExts) if (ext == e) return true;
    return false;
}
} // namespace

// ============================================================================
// Construction / destruction
// ============================================================================

SoundController::SoundController() {
    auto ctx    = audio::Context::master();
    _output     = ctx->getOutput();
    ctx->enable();
}

SoundController::~SoundController() {
    for (auto& [id, def] : _effects) {
        for (auto& [uid, node] : def.active) {
            if (node.node)   node.node->stop();
            if (node.gain)   node.gain->disconnectAll();
            if (node.node)   node.node->disconnectAll();
        }
        def.active.clear();
    }
    if (_bgm.node) { _bgm.node->stop(); _bgm.node->disconnectAll(); }
    if (_bgm.gain)   _bgm.gain->disconnectAll();
}

// ============================================================================
// Master
// ============================================================================

void SoundController::setSoundLevel(float level) { _level = level; }

void SoundController::setOutput(audio::NodeRef node) {
    if (node) _output = node;
}

void SoundController::update() {
    for (auto& [id, def] : _effects) {
        for (auto it = def.active.begin(); it != def.active.end(); ) {
            auto& node = it->second;
            if (node.autoFade) {
                node.level -= def.fadeRate;
                if (node.level < 0.f) {
                    node.node->stop();
                    node.gain->disconnectAll();
                    node.node->disconnectAll();
                    it = def.active.erase(it);
                    continue;
                }
                node.gain->setValue(_level * node.level);
            }
            ++it;
        }
    }
}

// ============================================================================
// BGM
// ============================================================================

void SoundController::playBackgroundTrack(const fs::path& assetPath, float level) {
    // Tear down any existing BGM graph first (ref silently leaked the old one).
    if (_bgm.node) { _bgm.node->stop(); _bgm.node->disconnectAll(); _bgm.node.reset(); }
    if (_bgm.gain) { _bgm.gain->disconnectAll();                    _bgm.gain.reset(); }

    auto ctx = audio::Context::master();
    auto src = audio::load(app::loadAsset(assetPath), ctx->getSampleRate());
    auto buf = src->loadBuffer();

    _bgm.node = ctx->makeNode(new audio::BufferPlayerNode(buf));
    _bgm.node->setLoopEnabled(true);
    _bgm.gain = ctx->makeNode(new audio::GainNode(_level * level));
    _bgm.node >> _bgm.gain >> _output;
    _bgm.level    = level;
    _bgm.autoFade = false;
    _bgm.node->start();
}

void SoundController::setBackgroundTrackLevel(float level) {
    if (!_bgm.gain) return;
    _bgm.gain->setValue(_level * level);
    _bgm.level = level;
}

void SoundController::stopBackgroundTrack() {
    if (!_bgm.node) return;
    _bgm.node->stop();
    _bgm.gain->disconnectAll();
    _bgm.node->disconnectAll();
    _bgm.node.reset();
    _bgm.gain.reset();
}

// ============================================================================
// SFX registration
// ============================================================================

void SoundController::registerEffect(const string& id, float fadeRate) {
    auto& def   = _effects[id];
    def.fadeRate = fadeRate;
}

void SoundController::addEffectTrack(const string& id, const fs::path& assetPath) {
    auto it = _effects.find(id);
    if (it == _effects.end()) {
        CI_LOG_W("SoundController::addEffectTrack — effect '" << id
                 << "' not registered. Call registerEffect first.");
        return;
    }
    auto ctx = audio::Context::master();
    auto src = audio::load(app::loadAsset(assetPath), ctx->getSampleRate());
    it->second.buffers.push_back(src->loadBuffer());
}

void SoundController::loadEffectFolder(const string& id, float fadeRate,
                                       const fs::path& folder) {
    registerEffect(id, fadeRate);
    auto root = app::getAssetPath(folder);
    if (!fs::exists(root) || !fs::is_directory(root)) {
        CI_LOG_W("SoundController::loadEffectFolder — '" << root
                 << "' is not a directory.");
        return;
    }
    vector<fs::path> files;
    for (auto& e : fs::directory_iterator(root)) {
        if (isAudioFile(e.path())) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    for (auto& f : files) addEffectTrack(id, f);
}

void SoundController::loadConfig(const fs::path& jsonPath) {
    Json j;
    try {
        j = Json::parse(loadString(app::loadAsset(jsonPath)));
    } catch (const std::exception& ex) {
        CI_LOG_EXCEPTION("SoundController::loadConfig — parse failed for "
                         + jsonPath.string(), ex);
        return;
    }

    if (j.contains("bgm")) {
        auto& b = j["bgm"];
        if (b.contains("file") && b.contains("level")) {
            playBackgroundTrack(b["file"].get<string>(),
                                b.value("level", 0.7f));
        }
    }

    if (j.contains("effects")) {
        for (auto& [id, cfg] : j["effects"].items()) {
            auto fadeRate = cfg.value("fadeRate", 1.f / 90.f);
            registerEffect(id, fadeRate);
            if (cfg.contains("files")) {
                for (auto& f : cfg["files"]) addEffectTrack(id, f.get<string>());
            }
            if (cfg.contains("folder")) {
                loadEffectFolder(id, fadeRate, cfg["folder"].get<string>());
            }
        }
    }
}

// ============================================================================
// SFX playback
// ============================================================================

void SoundController::playEffect(const string& id, int uid, float level) {
    auto it = _effects.find(id);
    if (it == _effects.end() || it->second.buffers.empty()) return;
    auto& def = it->second;
    if (def.active.find(uid) != def.active.end()) return;

    auto idx = Rand::randInt(static_cast<int32_t>(def.buffers.size()));
    auto ctx = audio::Context::master();

    SoundEffectNode node;
    node.node = ctx->makeNode(new audio::BufferPlayerNode(def.buffers[idx]));
    node.node->setName(id + ":" + to_string(uid));
    node.node->setLoopEnabled(false);
    node.gain = ctx->makeNode(new audio::GainNode(_level * level));
    node.node >> node.gain >> _output;
    node.level    = level;
    node.autoFade = true;
    node.node->start();
    def.active[uid] = std::move(node);
}

void SoundController::fadeOutEffect(const string& id, int uid) {
    auto it = _effects.find(id);
    if (it == _effects.end()) return;
    auto nodeIt = it->second.active.find(uid);
    if (nodeIt == it->second.active.end()) return;
    nodeIt->second.autoFade = true;
}

bool SoundController::isPlaying(const string& id, int uid) const {
    auto it = _effects.find(id);
    if (it == _effects.end()) return false;
    return it->second.active.find(uid) != it->second.active.end();
}

// ============================================================================
// UI
// ============================================================================

void SoundController::drawUi() {
    if (!ImGui::CollapsingHeader("Sound Controller")) return;
    ImGui::ScopedId scpId("snd");

    // ----- BGM transport -----
    if (ImGui::CollapsingHeader("BGM", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ScopedId bgmScope("bgm");
        if (!_bgm.node) {
            ImGui::TextDisabled("No BGM loaded");
        } else {
            ImGui::Text("playing (loop)");
            if (ImGui::Button("Stop")) stopBackgroundTrack();
            ImGui::SameLine();
            float lvl = _bgm.level;
            if (ImGui::SliderFloat("Level", &lvl, 0.f, 1.f, "%.2f")) {
                setBackgroundTrackLevel(lvl);
            }
        }
    }

    // ----- Effects -----
    for (auto& [id, def] : _effects) {
        ImGui::ScopedId idScope(id.c_str());
        if (!ImGui::CollapsingHeader(id.c_str())) continue;

        ImGui::Text("variations: %u   active: %u",
                    static_cast<unsigned>(def.buffers.size()),
                    static_cast<unsigned>(def.active.size()));
        ImGui::Text("fadeRate: %.4f", def.fadeRate);

        if (!def.buffers.empty()) {
            if (ImGui::Button("Play random")) playEffect(id, _testUid++, 1.0f);
        }

        // Snapshot the uid list — erase happens inside the loop buttons below.
        vector<int> uids;
        uids.reserve(def.active.size());
        for (auto& [uid, n] : def.active) uids.push_back(uid);

        for (auto uid : uids) {
            ImGui::ScopedId uidScope(uid);
            auto it = def.active.find(uid);
            if (it == def.active.end()) continue;
            auto& n = it->second;

            ImGui::Text("uid=%d  lvl=%.2f%s",
                        uid, n.level, n.autoFade ? "  (fading)" : "");
            ImGui::SameLine();
            if (ImGui::SmallButton("Stop")) fadeOutEffect(id, uid);
        }
    }
}

} // namespace newtype::util

#endif // NT_ENABLE_AUDIO
