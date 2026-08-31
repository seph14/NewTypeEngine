#include "newtype/util/Profiler.h"
#include "cinder/CinderImGui.h"
#include "cinder/app/App.h"

namespace newtype::util {
	using namespace luisa;
	using namespace luisa::compute;
	using namespace newtype::core;

	Profiler& Profiler::instance() {
		static Profiler instance;
		return instance;
	}

	Profiler::Profiler()
		: _stats(nullptr) {
#if NT_PROFILING
		_stats = Renderer::device().extension<StatsExt>();
#endif
	}

	Profiler::~Profiler() {
#if NT_PROFILING
		if (_stats) {
			//delete _stats;
		}
#endif
	}

	void Profiler::begin_profiling() {
#if NT_PROFILING
		_stats->begin_stats();
		begin_cpu_frame();
#endif
	}

	void Profiler::set_pass(luisa::string_view name) {
#if NT_PROFILING
		_stats->set_next_dispatch_name(luisa::string(name));
#endif
	}

	void Profiler::end_profiling() {
#if NT_PROFILING
		_res = _stats->end_stats();
		end_cpu_frame();
#endif
	}

	//==========================================================================
	// CPU profiling
	//==========================================================================

	void Profiler::begin_cpu_frame() {
#if NT_PROFILING
		_cpuTimings.clear();
#endif
	}

	void Profiler::start_cpu_pass(luisa::string_view name) {
#if NT_PROFILING
		_cpuCurrentPass = luisa::string(name);
		_cpuClock.tic();
#endif
	}

	void Profiler::end_cpu_pass() {
#if NT_PROFILING
		double ms = _cpuClock.toc();
		auto it = _cpuTimings.find(_cpuCurrentPass);
		if (it != _cpuTimings.end()) {
			it->second += ms;
		} else {
			_cpuTimings.emplace(_cpuCurrentPass, ms);
		}
#endif
	}

	void Profiler::end_cpu_frame() {
		// no-op for now; future: compute rolling averages
	}

	//==========================================================================
	// UI
	//==========================================================================

	void Profiler::draw_ui() {
		if (ImGui::CollapsingHeader("Profiler")) {
			ImGui::ScopedId scpId("perf");

			float fps = ci::app::getWindow()->getApp()->getAverageFps();
			ImGui::Text("Fps: %.2f", fps);

#if NT_PROFILING
			// Group items by prefix (part before first '/')
			luisa::unordered_map<luisa::string, luisa::vector<std::pair<luisa::string, float>>> groups;
			float totalTime = 0.f;

			for (auto kv : _res) {
				auto& data = kv.second.stream_scopes;
				for (auto& item : data) {
					if (item->name == "Unknown") continue;
					float dt = (item->finished_time - item->start_time) / 10.f;
					totalTime += dt;

					std::string_view name(item->name);
					auto slashPos = name.find('/');
					if (slashPos != std::string_view::npos) {
						auto group = luisa::string(name.substr(0, slashPos));
						auto label = luisa::string(name.substr(slashPos + 1));
						groups[group].emplace_back(std::move(label), dt);
					} else {
						groups[""].emplace_back(luisa::string(name), dt);
					}
				}
			}

			// Render GPU groups
			if (ImGui::CollapsingHeader("GPU Timings")) {
				ImGui::ScopedId scpId("gpu_timings");
				for (auto& [group, items] : groups) {
					if (!group.empty()) {
						if (ImGui::CollapsingHeader(group.c_str())) {
							for (auto& [label, dt] : items) {
								ImGui::Text("%s : %.2f", label.c_str(), dt);
							}
						}
					} else {
						for (auto& [label, dt] : items) {
							ImGui::Text("%s : %.2f", label.c_str(), dt);
						}
					}
				}
				ImGui::Text("Total GPU : %.2f", totalTime);
			}

			// CPU timings
			if (ImGui::CollapsingHeader("CPU Timings")) {
				ImGui::ScopedId scpId("cpu_timings");
				luisa::unordered_map<luisa::string, luisa::vector<std::pair<luisa::string, double>>> cpuGroups;
				double cpuTotal = 0.0;

				for (auto& [name, ms] : _cpuTimings) {
					cpuTotal += ms;
					std::string_view sv(name);
					auto slashPos = sv.find('/');
					if (slashPos != std::string_view::npos) {
						auto group = luisa::string(sv.substr(0, slashPos));
						auto label = luisa::string(sv.substr(slashPos + 1));
						cpuGroups[group].emplace_back(std::move(label), ms);
					} else {
						cpuGroups[""].emplace_back(luisa::string(name), ms);
					}
				}

				for (auto& [group, items] : cpuGroups) {
					if (!group.empty()) {
						if (ImGui::CollapsingHeader(group.c_str())) {
							for (auto& [label, ms] : items) {
								ImGui::Text("%s : %.3f", label.c_str(), ms);
							}
						}
					} else {
						for (auto& [label, ms] : items) {
							ImGui::Text("%s : %.3f", label.c_str(), ms);
						}
					}
				}
				ImGui::Text("Total CPU : %.3f", cpuTotal);
			}
#endif
		}
	}
}
