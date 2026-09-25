// Copyright 2025-2026 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: BSD-2-Clause

#include "Detector.h"
#include "Gaussian2DFilter.h"
#include "Specimen.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

void bm_filaments_specimen_draw(benchmark::State &state) {
    FilamentsSpecimen specimen;
    const auto z_um = double(state.range(0));
    const std::size_t width = 512, height = 512;
    std::vector<float> signal(width * height);
    auto *data = signal.data();
    for ([[maybe_unused]] auto _ : state) {
        specimen.Draw(data, 0.0, 0.0, z_um, width, height, 0.2, 1.4f, 1000.0);
        benchmark::DoNotOptimize(data);
    }
}
BENCHMARK(bm_filaments_specimen_draw)
    ->Arg(0)
    ->Arg(50)
    ->Unit(benchmark::kMillisecond);

void bm_nuclei_specimen_draw(benchmark::State &state) {
    NucleiSpecimen specimen;
    const auto z_um = double(state.range(0));
    const std::size_t width = 512, height = 512;
    std::vector<float> signal(width * height);
    auto *data = signal.data();
    for ([[maybe_unused]] auto _ : state) {
        specimen.Draw(data, 0.0, 0.0, z_um, width, height, 0.2, 1.4f, 1000.0);
        benchmark::DoNotOptimize(data);
    }
}
BENCHMARK(bm_nuclei_specimen_draw)
    ->Arg(0)
    ->Arg(50)
    ->Unit(benchmark::kMillisecond);

void bm_readout(benchmark::State &state) {
    const auto level = float(state.range(0));
    const std::size_t width = 512, height = 512;
    std::vector<float> signal(width * height, level);
    const float *signalData = level > 0.0f ? signal.data() : nullptr;
    std::vector<std::uint16_t> out(width * height);
    auto *outData = out.data();
    rnd::mt19937 rng;
    for ([[maybe_unused]] auto _ : state) {
        ReadOut(signalData, outData, width * height, 50.0f, 100.0f, rng);
        benchmark::DoNotOptimize(outData);
    }
}
BENCHMARK(bm_readout)
    ->Arg(0)
    ->Arg(5)
    ->Arg(1000)
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
BENCHMARK(bm_gaussian_2d_filter_scalar)->Unit(benchmark::kMillisecond);

#ifdef USE_HIGHWAY_SIMD
void bm_gaussian_2d_filter_simd(benchmark::State &state) {
    bm_gaussian_2d_filter<gaussian_internal::FastGaussian2DSIMD>(state);
}
BENCHMARK(bm_gaussian_2d_filter_simd)->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_MAIN();
