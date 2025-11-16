#include "SimulatedSpecimen.h"

#include <benchmark/benchmark.h>

#include <cstdint>
#include <vector>

void bm_simulated_specimen_draw(benchmark::State& state) {
    SimulatedSpecimen<std::uint16_t> specimen;
    std::vector<std::uint16_t> buffer(512 * 512);
    auto *data = buffer.data();
    for ([[maybe_unused]] auto _ : state) {
        specimen.Draw(data, 0.0, 0.0, 0.0, 512, 512, 0.2, 1000.0);
        benchmark::DoNotOptimize(data);
    }
}

BENCHMARK(bm_simulated_specimen_draw);

BENCHMARK_MAIN();