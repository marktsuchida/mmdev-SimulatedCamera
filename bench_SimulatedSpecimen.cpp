#include "SimulatedSpecimen.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

void bm_simulated_specimen_draw(benchmark::State &state) {
    SimulatedSpecimen<std::uint16_t> specimen;
    const double z_um = state.range(0);
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

BENCHMARK_MAIN();