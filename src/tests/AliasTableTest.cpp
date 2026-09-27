//==============================================================================
// AliasTableTest — LightSampler alias table distribution self-check
// (src/tests/AliasTableTest.cpp)
//
// Runs the pure-CPU Vose builder (LightSampler::build_alias_table_cpu) and
// verifies the property every light-sampling consumer depends on: the table's
// exact selection probability per entry equals weight / sum(weights).
// Cases: single entry, equal weights, extreme skew, hidden (zero-power)
// entries, the dim-lights-plus-bright-light sweep from the brightness-drop
// bug report, a deterministic random fuzz, and uniform-sampling mode.
// Results go to alias_selfcheck.log next to the exe (GUI subsystem has no
// console).
//
// Run with: NewTypeEngine.exe --scene alias
//==============================================================================

#include "cinder/app/App.h"
#include "cinder/Log.h"
#include "cinder/ObjLoader.h"
#include "cinder/GeomIo.h"
#include <fstream>
#include <random>
#include <cmath>

#include "TestScenes.h"
#include "newtype/core/Pipeline.h"
#include "newtype/core/Renderer.h"
#include "newtype/scene/MeshShape.h"
#include "newtype/scene/LightShape.h"
#include "newtype/render/Material.h"
#include "newtype/render/LightSampler.h"
#include "newtype/util/TypeConv.h"

using namespace luisa;
using namespace luisa::compute;
using namespace nt;
using namespace ci;
using namespace ci::app;

namespace {

//------------------------------------------------------------------------------
// Exact selection-probability enumeration of an alias table:
// bucket i is drawn with prob 1/n; inside it, the triangle is chosen with
// prob entry.pdf, else the alias target. Mirrors the GPU bucket selection
// in the DI presample / GI / SHARC kernels. When uniform_expected is set
// (uniform-sampling mode), the target distribution is 1/n per entry instead
// of weight/sum.
//------------------------------------------------------------------------------
bool check_distribution(const luisa::vector<render::AliasEntry> &table,
                        const luisa::vector<float> &weights,
                        bool uniform_expected,
                        double tol, double &max_err) {
    const size_t n = table.size();
    if (n == 0 || weights.size() != n) return false;

    std::vector<double> actual(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const auto &e = table[i];
        if (e.triangle_index >= n || e.alias_index >= n) return false;
        if (!(e.pdf >= 0.0f && e.pdf <= 1.0f)) return false;
        actual[e.triangle_index] += static_cast<double>(e.pdf) / static_cast<double>(n);
        if (e.pdf < 1.0f)
            actual[e.alias_index] += (1.0 - static_cast<double>(e.pdf)) / static_cast<double>(n);
    }

    max_err = 0.0;
    if (uniform_expected) {
        const double expected = 1.0 / static_cast<double>(n);
        for (size_t i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(actual[i] - expected));
    } else {
        double wsum = 0.0;
        for (float w : weights) wsum += std::max(w, 0.0f);
        if (wsum <= 0.0) return true;  // all-hidden fallback: structure checked above
        for (size_t i = 0; i < n; ++i) {
            const double expected = std::max(weights[i], 0.0f) / wsum;
            max_err = std::max(max_err, std::abs(actual[i] - expected));
        }
    }
    return max_err <= tol;
}

struct CheckResult {
    bool   ok = false;
    double max_err = 0.0;
};

CheckResult run_case(const char *name, luisa::vector<float> weights,
                     bool uniform_sampling = false) {
    (void)name;
    CheckResult r;
    auto table = render::LightSampler::build_alias_table_cpu(weights.data(), weights.size(), uniform_sampling);
    // Tolerance scales mildly with entry count (per-entry error is
    // O(n * float-eps) worst case in the probability enumeration).
    const double tol = 1e-5 * std::max(1.0, static_cast<double>(weights.size()) / 16.0);
    r.ok = check_distribution(table, weights, uniform_sampling, tol, r.max_err);
    return r;
}

//------------------------------------------------------------------------------
// The bug-report scenario: N dim lights + one new light whose emission is
// swept across the reported threshold. Verifies every sweep step realizes
// power/total exactly (the old builder skewed up to ~2x around it).
//------------------------------------------------------------------------------
bool check_brightness_sweep(double &worst_err) {
    constexpr uint kDimCount = 6;
    constexpr float kDimEmission[kDimCount] = {10.f, 12.f, 14.f, 16.f, 18.f, 20.f};
    constexpr float kTriArea = 0.25f;   // 2 triangles per light shape
    constexpr uint kDimTris = 2u, kNewTris = 2u;

    worst_err = 0.0;
    for (float emission : {20.f, 50.f, 80.f, 100.f, 120.f, 150.f, 200.f, 500.f, 1000.f}) {
        luisa::vector<float> weights;
        for (uint t = 0; t < kDimTris * kDimCount; ++t)
            weights.push_back(kDimEmission[t / kDimTris] * kTriArea);
        for (uint t = 0; t < kNewTris; ++t)
            weights.push_back(emission * kTriArea);

        auto table = render::LightSampler::build_alias_table_cpu(weights.data(), weights.size(), false);
        double err = 0.0;
        if (!check_distribution(table, weights, /*uniform_expected=*/false, 1e-5, err)) return false;
        worst_err = std::max(worst_err, err);
    }
    return true;
}

class AliasTableScene : public nt::test::TestScene {
public:
    const char* name() const override { return "alias"; }

    void build(core::Pipeline& pipeline) override {
        auto& device = core::Renderer::device();
        auto& stream = core::Renderer::stream();

        // Minimal renderable scene so the app runs normally after the check.
        auto grayMatIdx = pipeline.addMaterial("alias_gray",
            render::make_diffuse(luisa::make_float3(.6f)));
        auto lightMatIdx = pipeline.addMaterial("alias_light",
            render::make_emissive(luisa::make_float3(15.f)));
        auto lightMatIdx2 = pipeline.addMaterial("alias_light2",
            render::make_emissive(luisa::make_float3(25.f)));
        {
            scene::StaticTransform floorTrans(tolc(
                glm::translate(vec3(0.f, -1.9f, 0.f))));
            auto floor = scene::MeshShape::create(device, grayMatIdx);
            floor->load_from(geom::Cube().size(vec3(6.f, .1f, 6.f)));
            floor->build(stream);
            (void)pipeline.addShape(std::move(floor), &floorTrans);
            /*
            TriMesh lightMesh = ObjLoader(app::loadAsset("models/arealight.obj"));
            auto light = scene::make_light(device, lightMesh, lightMatIdx);
            auto lightTrans = scene::StaticTransform::create(tolc(
                glm::translate(vec3(0.f, 2.2f, 0.f)) * glm::scale(vec3(1.2f))));
            (void)pipeline.addLightShape(std::move(light), lightTrans.get());

            auto light2 = scene::make_light(device, lightMesh, lightMatIdx2);
            auto lightTrans2 = scene::StaticTransform::create(tolc(
                glm::translate(vec3(0.f, -1.2f, 0.f)) * glm::scale(vec3(1.2f))));
            (void)pipeline.addLightShape(std::move(light2), lightTrans2.get());
            */
        }

        run_selfcheck();
    }

private:
    static void run_selfcheck() {
        struct NamedCheck { const char *name; CheckResult result; };
        luisa::vector<NamedCheck> checks;

        // 1. Single entry
        checks.push_back({"single", run_case("single", {1.0f})});

        // 2. Equal weights (mild-skew regime where the old builder skewed most)
        checks.push_back({"equal_8",
            run_case("equal_8", {5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f, 5.f})});

        // 3. Extreme skew
        checks.push_back({"skew_1_10_100",
            run_case("skew_1_10_100", {1.f, 10.f, 100.f})});

        // 4. Hidden (zero-power) lights mixed with visible ones
        checks.push_back({"hidden_zeros",
            run_case("hidden_zeros", {0.f, 15.f, 0.f, 20.f, 0.f, 12.f})});

        // 5. count=0 edge (table must come back empty, no crash)
        {
            auto table = render::LightSampler::build_alias_table_cpu(nullptr, 0u, false);
            checks.push_back({"empty", {table.empty(), 0.0}});
        }

        // 6. Bug-report sweep: dims at 10-20, new light 20 -> 1000
        {
            double worst = 0.0;
            bool ok = check_brightness_sweep(worst);
            checks.push_back({"brightness_sweep", {ok, worst}});
        }

        // 7. Deterministic fuzz: sizes 2..64, log-uniform weights + zeros
        {
            std::mt19937 rng(1234u);
            std::uniform_int_distribution<size_t> size_dist(2, 64);
            std::uniform_real_distribution<float> mag_dist(-2.0f, 3.0f); // 10^-2 .. 10^3
            std::uniform_real_distribution<float> zero_dist(0.0f, 1.0f);

            bool all_ok = true;
            double worst = 0.0;
            for (int trial = 0; trial < 200; ++trial) {
                const size_t n = size_dist(rng);
                luisa::vector<float> weights(n);
                for (size_t i = 0; i < n; ++i)
                    weights[i] = (zero_dist(rng) < 0.1f) ? 0.0f
                                : std::pow(10.0f, mag_dist(rng));
                auto table = render::LightSampler::build_alias_table_cpu(weights.data(), n, false);
                double err = 0.0;
                const double tol = 1e-5 * static_cast<double>(n) / 16.0;
                if (!check_distribution(table, weights, /*uniform_expected=*/false, tol, err)) {
                    all_ok = false;
                    break;
                }
                worst = std::max(worst, err);
            }
            checks.push_back({"fuzz_200", {all_ok, worst}});
        }

        // 8. Uniform-sampling mode (weights ignored, equal probability)
        checks.push_back({"uniform_mode",
            run_case("uniform_mode", {1e-3f, 500.f, 42.f}, /*uniform_sampling=*/true)});

        //----------------------------------------------------------------------
        // Report
        //----------------------------------------------------------------------
        bool all_ok = true;
        for (auto &c : checks) {
            all_ok = all_ok && c.result.ok;
            CI_LOG_I("AliasTableTest [" << c.name << "] "
                << (c.result.ok ? "PASS" : "FAIL")
                << " (max_err=" << c.result.max_err << ")");
        }

        if (std::ofstream out{"alias_selfcheck.log", std::ios::app}) {
            for (auto &c : checks) {
                out << (c.result.ok ? "PASS" : "FAIL")
                    << " " << c.name
                    << " max_err=" << c.result.max_err << std::endl;
            }
            out << (all_ok ? "ALL PASS" : "FAILED") << std::endl;
        }

        if (all_ok) {
            CI_LOG_I("AliasTableTest: all alias-table distribution checks passed");
        } else {
            CI_LOG_E("AliasTableTest FAILED - see alias_selfcheck.log");
        }
    }
};

} // anonymous namespace

namespace newtype::test {
ScenePtr createAliasTableScene() { return std::make_unique<AliasTableScene>(); }
} // namespace newtype::test
