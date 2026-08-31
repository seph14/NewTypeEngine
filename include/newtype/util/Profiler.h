#pragma once

/*

  Profiler â unified GPU + CPU pass timing.

  GPU: wraps StatsExt (set_next_dispatch_name) for per-dispatch GPU timestamps.
  CPU: wraps luisa::Clock for host-side wall-clock measurement.

  Usage:
    auto& profiler = Profiler::instance();
    profiler.begin_profiling();               // start GPU stats + reset CPU map

    // GPU pass naming (existing)
    profiler.set_pass("DI/G-Buffer");

    // CPU scoped timing (new)
    {
        CpuScopedTimer _("DI/G-Buffer");      // start_cpu_pass / end_cpu_pass
        // ... host work ...
    }

    profiler.end_profiling();                 // finalize GPU stats
    profiler.draw_ui();                       // shows both GPU + CPU tables
  */

#include <luisa/backends/ext/stats_ext.h>
#include <luisa/core/clock.h>
#include "newtype/core/Renderer.h"
#include "newtype/core/Config.h"

namespace newtype::util {
	class Profiler {
	public:
		// Non-copyable, non-movable
		Profiler(const Profiler&) = delete;
		Profiler& operator=(const Profiler&) = delete;
		Profiler(Profiler&&) = delete;
		Profiler& operator=(Profiler&&) = delete;

		static Profiler& instance();

		void begin_profiling();
		void set_pass		(luisa::string_view name);
		void end_profiling	();
		void draw_ui		();

		// CPU profiling
		void begin_cpu_frame	();
		void start_cpu_pass	(luisa::string_view name);
		void end_cpu_pass	();
		void end_cpu_frame	();

	protected:
		Profiler();
		~Profiler();

		luisa::compute::StatsExt* _stats;
		luisa::unordered_map<uint64_t, luisa::compute::StreamStats> _res;

		luisa::Clock _cpuClock;
		luisa::string _cpuCurrentPass;
		luisa::unordered_map<luisa::string, double> _cpuTimings;
	};

	// RAII scoped CPU timer â construct at top of a pass block
	class CpuScopedTimer {
	public:
		explicit CpuScopedTimer(luisa::string_view name) {
			Profiler::instance().start_cpu_pass(name);
		}
		~CpuScopedTimer() {
			Profiler::instance().end_cpu_pass();
		}
		CpuScopedTimer(const CpuScopedTimer&) = delete;
		CpuScopedTimer& operator=(const CpuScopedTimer&) = delete;
	};
}
