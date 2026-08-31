#pragma once

#include <luisa/luisa-compute.h>
#include <chrono>
#include <mutex>
#include <unordered_map>
#include <string>
#include "cinder/Log.h"

namespace newtype::util {

    // Captures per-shader compile and cache-load durations. Methods are invoked
    // from LC's compile path (potentially concurrent across std::async threads),
    // so an internal mutex guards the start-time map.
    //
    // Hooked events:
    //   before/after_compile_shader_bytecode  → DXC compile (HLSL → DXIL)
    //   before/after_compile_shader_cache      → D3D12 PSO creation from cached DXIL
    //   before/after_load_shader_bytecode      → cache READ for DXIL blob
    //   before/after_load_shader_cache         → cache READ for PSO blob
    //
    // Enable by setting DeviceConfig::profiler at device creation (Renderer.cpp).
    class CompileProfiler : public luisa::compute::Profiler {
        using clock = std::chrono::steady_clock;
        using time_point = clock::time_point;

        struct Entry {
            time_point start;
            int        category;  // 0=bytecode, 1=cache, 2=load_bytecode, 3=load_cache
        };

        mutable std::mutex             _mtx;
        std::unordered_map<std::string, Entry> _pending;

        static const char* cat_name(int c) noexcept {
            switch (c) {
            case 0: return "DXC   ";
            case 1: return "PSO   ";
            case 2: return "CBCLD ";
            case 3: return "PSOLD ";
            default:return "????  ";
            }
        }

        void begin(std::string_view name, int cat) noexcept {
            std::lock_guard lk(_mtx);
            _pending[std::string{ name }] = Entry{ clock::now(), cat };
        }

        void end(std::string_view name) noexcept {
            auto t1 = clock::now();
            std::string key{ name };
            std::lock_guard lk(_mtx);
            if (auto it = _pending.find(key); it != _pending.end()) {
                auto dt_ms = std::chrono::duration<double, std::milli>(t1 - it->second.start).count();
                CI_LOG_I("[lcprof] " << cat_name(it->second.category)
                    << " " << dt_ms << " ms  " << key);
                _pending.erase(it);
            }
        }

    public:
        // --- Compile (cache-miss path) ---
        void before_compile_shader_bytecode(luisa::string_view name) noexcept override { begin(name, 0); }
        void after_compile_shader_bytecode(luisa::string_view name) noexcept override { end(name); }
        void before_compile_shader_cache(luisa::string_view name, luisa::string_view) noexcept override {
            begin(name, 1);
        }
        void after_compile_shader_cache(luisa::string_view name, luisa::string_view) noexcept override { end(name); }

        // --- Load (cache-hit path) ---
        void before_load_shader_bytecode(luisa::string_view name) noexcept override { begin(name, 2); }
        void after_load_shader_bytecode(luisa::string_view, bool) noexcept override { /* no name in after — skip
    per-event log */
        }
        void before_load_shader_cache(luisa::string_view name, luisa::string_view) noexcept override { begin(name, 3); }
        void after_load_shader_cache(luisa::string_view, luisa::string_view, bool) noexcept override {}

        // --- Resource lifecycle (not interesting for compile profiling) ---
        void allocate(uint64_t, uint64_t, size_t, luisa::string_view, luisa::vector<luisa::compute::TraceItem>&&) noexcept
            override {
        }
        void free(uint64_t) noexcept override {}
    };

}  // namespace newtype::util