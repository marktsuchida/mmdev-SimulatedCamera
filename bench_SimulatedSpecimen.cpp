#include "Gaussian2DFilter.h"
#include "SimulatedSpecimen.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

void bm_simulated_specimen_draw(benchmark::State &state) {
    SimulatedSpecimen<std::uint16_t> specimen;
    const auto z_um = double(state.range(0));
    const std::size_t width = 512, height = 512;
    std::vector<std::uint16_t> buffer(width * height);
    auto *data = buffer.data();
    for ([[maybe_unused]] auto _ : state) {
        specimen.Draw(data, 0.0, 0.0, z_um, width, height, 0.2, 1000.0);
        benchmark::DoNotOptimize(data);
    }
}

BENCHMARK(bm_simulated_specimen_draw)
    ->Arg(0)
    ->Arg(50)
    ->Unit(benchmark::kMillisecond);

template <auto Func> void bm_gaussian_2d_filter(benchmark::State &state) {
    const std::size_t width = 512, height = 512;
    std::vector<float> image(width * height, 42.0f);
    auto *data = image.data();
    for ([[maybe_unused]] auto _ : state) {
        Func(data, width, height, 5.0);
        benchmark::DoNotOptimize(data);
    }
}

void bm_gaussian_2d_filter_scalar(benchmark::State &state) {
    bm_gaussian_2d_filter<gaussian_internal::FastGaussian2DScalar<float>>(
        state);
}

void bm_gaussian_2d_filter_simd(benchmark::State &state) {
    bm_gaussian_2d_filter<gaussian_internal::FastGaussian2DSIMD>(state);
}

BENCHMARK(bm_gaussian_2d_filter_scalar)->Unit(benchmark::kMillisecond);
BENCHMARK(bm_gaussian_2d_filter_simd)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();