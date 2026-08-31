#pragma once

// Minimal row/element-band parallel helper for pure-CPU pixel loops.
//
// fn(i0, i1) is invoked over disjoint half-open bands covering [0, count);
// the calling thread runs the first band and the rest spawn std::threads,
// all joined before returning (deterministic completion). Work smaller than
// 2*grain runs sequentially on the caller — small textures pay nothing.
//
// fn must touch only its own band (disjoint writes, shared reads are fine)
// and must not call into LuisaCompute — device/stream calls stay on the
// thread that owns them.

#include <thread>
#include <vector>

namespace newtype::util {

template<typename F>
void parallel_for(uint64_t count, uint64_t grain, F&& fn) {
    if (count == 0u) return;

    uint64_t bands = grain > 0u ? (count + grain - 1u) / grain : 1u;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0u) hw = 1u;
    if (bands > hw) bands = hw;
    if (bands < 2u) {
        fn(0u, count);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(bands - 1u));
    for (uint64_t b = 1u; b < bands; b++) {
        uint64_t i0 = b * count / bands;
        uint64_t i1 = (b + 1u) * count / bands;
        threads.emplace_back([&fn, i0, i1] { fn(i0, i1); });
    }
    fn(0u, count / bands);
    for (auto& t : threads) t.join();
}

} // namespace newtype::util
