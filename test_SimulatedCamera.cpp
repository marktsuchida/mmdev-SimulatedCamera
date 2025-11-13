#include "SimulatedSpecimen.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <numeric>
#include <random>

TEST_CASE("GaussianKernel") {
    using Catch::Matchers::WithinAbs;
    const auto k = GaussianKernel(1, 1.0f);
    CHECK(k.size() == 3);
    CHECK_THAT(k[0] + k[1] + k[2], WithinAbs(1.0f, 1e-6));
    CHECK(k[0] < k[1]);
    CHECK(k[2] < k[1]);
}

TEST_CASE("GaussianKernel-zero-sigma") {
    const auto k1 = GaussianKernel(1, 0.0f);
    const std::vector expected1{0.0f, 1.0f, 0.0f};
    CHECK(k1 == expected1);

    const auto k0 = GaussianKernel(0, 0.0f);
    const std::vector expected0{1.0f};
    CHECK(k0 == expected0);
}

TEST_CASE("HConvolve") {
    std::vector data{1, 2, 3, 4, 5, 6};
    const std::vector kern{4, 5, 6};
    HConvolve(data.data(), 3, 2, kern.data(), 1);
    const std::vector expected{25, 32, 35, 70, 77, 80};
    CHECK(data == expected);
}

TEST_CASE("VConvolve") {
    std::vector data{1, 4, 2, 5, 3, 6};
    const std::vector kern{4, 5, 6};
    VConvolve(data.data(), 2, 3, kern.data(), 1);
    const std::vector expected{25, 70, 32, 77, 35, 80};
    CHECK(data == expected);
}

TEST_CASE("FastPoisson-mean-variance") {
    using Catch::Matchers::WithinAbs;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);

    // Test small and large lambda code paths.
    double const lambda = GENERATE(5.0, 100.0);
    constexpr std::size_t numSamples = 10000;

    std::vector<double> samples;
    std::generate_n(std::back_inserter(samples), numSamples,
                    [&] { return FastPoisson(lambda, rng, uniformDist); });
    CHECK(std::all_of(samples.begin(), samples.end(),
                      [](double s) { return s >= 0; }));

    double const tolerance = 0.05 * lambda;

    double const mean =
        std::accumulate(samples.begin(), samples.end(), 0.0) / numSamples;
    CHECK_THAT(mean, WithinAbs(lambda, tolerance));

    double const variance =
        std::transform_reduce(
            samples.begin(), samples.end(), 0.0, std::plus<>{},
            [&](double s) { return (s - mean) * (s - mean); }) /
        numSamples;
    CHECK_THAT(variance, WithinAbs(lambda, tolerance));
}

TEST_CASE("FastPoisson-tiny-lambda") {
    std::mt19937 rng(11111);
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);

    double sample = FastPoisson(0.01, rng, uniformDist);
    CHECK(sample >= 0.0);
}