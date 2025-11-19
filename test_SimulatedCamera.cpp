#include "Gaussian2DFilter.h"
#include "SimulatedSpecimen.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <numeric>

TEST_CASE("FastGaussian2D-q") {
    using Catch::Matchers::WithinAbs;

    SECTION("Eq 11a") {
        // Leave out the boundary case of q = 1.5 because it round-trips
        // poorly (ends up using the opposing branches of Eqs 11a and 11b,
        // which are not precisely continuous at q = 1.5).
        const float q = GENERATE(0.5f, 1.0f, 1.495f, 1.505f, 3.0f, 5.0f);

        // Use the reverse (q to sigma) formula given in the paper (Eq 11a):
        const float sigma = [](float q) {
            if (q >= 1.5f) {
                return 0.97588f + 1.01306f * q;
            } else {
                return 0.30560f + 1.71881f * q - 0.21639f * q * q;
            }
        }(q);

        CAPTURE(q, sigma);
        CHECK_THAT(gaussian_internal::q(sigma), WithinAbs(q, 1e-3));
    }

    SECTION("Boundary is (sort of) continuous") {
        // Actually it's not that close.
        CHECK_THAT(gaussian_internal::q(2.499999f),
                   WithinAbs(gaussian_internal::q(2.500001f), 0.1));
    }

    SECTION("Example given in the paper") {
        CHECK_THAT(gaussian_internal::q(6.04f), WithinAbs(5.0f, 1e-2));
    }
}

TEST_CASE("FastGaussian2D-bB") {
    using Catch::Matchers::WithinAbs;
    const auto i = GENERATE(0, 1, 2, 3);
    const float q = std::array{0.5f, 1.0f, 2.0f, 5.0f}[i];

    const float ex_b0 = std::array{3.210115625f, 5.872685000f, 15.556550000f,
                                   102.277025000f}[i];
    const float ex_b1 =
        std::array{2.09443875f, 6.56693000f, 26.44590000f, 241.95165000f}[i];
    const float ex_b2 = std::array{-.51535125f, -2.69471000f, -15.84528000f,
                                   -194.02875000f}[i];
    const float ex_b3 =
        std::array{.052775625f, .422205000f, 3.377640000f, 52.775625000f}[i];
    const float ex_B = 1.0f - (ex_b1 + ex_b2 + ex_b3) / ex_b0;

    const auto b = gaussian_internal::b(q);
    const auto bp = gaussian_internal::b_predivided(b);
    const auto B = gaussian_internal::B(bp);
    CHECK_THAT(b[0], WithinAbs(ex_b0, 1e-6));
    CHECK_THAT(b[1], WithinAbs(ex_b1, 1e-6));
    CHECK_THAT(b[2], WithinAbs(ex_b2, 1e-6));
    CHECK_THAT(b[3], WithinAbs(ex_b3, 1e-6));
    CHECK_THAT(B, WithinAbs(ex_B, 1e-6));
}

TEST_CASE("FastGaussian2D-ForwardFilter") {
    using Catch::Matchers::WithinAbs;
    const std::array<float, 3> bp = {0.125f, 0.25f, 0.5f};
    const float B = 0.125f; // Must add up to 1.0 with bp.

    SECTION("empty") {
        gaussian_internal::ForwardFilter<float, 1>(nullptr, 0, B, bp);
    }

    SECTION("size-1") {
        std::vector data = {1.0f};
        gaussian_internal::ForwardFilter<float, 1>(data.data(), 1, B, bp);
        CHECK_THAT(data[0], WithinAbs(1.0f, 1e-6));
    }

    SECTION("size-2") {
        std::vector data = {1.0f, 2.0f};
        gaussian_internal::ForwardFilter<float, 1>(data.data(), 2, B, bp);
        CHECK_THAT(data[0], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[1], WithinAbs(1.125f, 1e-6));
    }

    SECTION("size-3") {
        std::vector data = {1.0f, 2.0f, 3.0f};
        gaussian_internal::ForwardFilter<float, 1>(data.data(), 3, B, bp);
        CHECK_THAT(data[0], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[1], WithinAbs(1.125f, 1e-6));
        CHECK_THAT(data[2], WithinAbs(1.265625f, 1e-6));
    }

    SECTION("size-4") {
        std::vector data = {1.0f, 2.0f, 3.0f, 4.0f};
        gaussian_internal::ForwardFilter<float, 1>(data.data(), 4, B, bp);
        CHECK_THAT(data[0], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[1], WithinAbs(1.125f, 1e-6));
        CHECK_THAT(data[2], WithinAbs(1.265625f, 1e-6));
        CHECK_THAT(data[3], WithinAbs(1.439453125f, 1e-6));
    }

    SECTION("size-5-stride-3") {
        std::vector data = {1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f, 0.0f,
                            0.0f, 4.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f};
        gaussian_internal::ForwardFilter(data.data(), 5, B, bp, 3);
        CHECK_THAT(data[0], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[3], WithinAbs(1.125f, 1e-6));
        CHECK_THAT(data[6], WithinAbs(1.265625f, 1e-6));
        CHECK_THAT(data[9], WithinAbs(1.439453125f, 1e-6));
        CHECK_THAT(data[12], WithinAbs(1.683837890625f, 1e-6));
    }
}

TEST_CASE("FastGaussian2D-BackwardFilter") {
    using Catch::Matchers::WithinAbs;
    const std::array<float, 3> bp = {0.125f, 0.25f, 0.5f};
    const float B = 0.125f; // Must add up to 1.0 with bp.

    SECTION("empty") {
        gaussian_internal::BackwardFilter<float, 1>(nullptr, 0, B, bp);
    }

    SECTION("size-1") {
        std::vector data = {1.0f};
        gaussian_internal::BackwardFilter<float, 1>(data.data(), 1, B, bp);
        CHECK_THAT(data[0], WithinAbs(1.0f, 1e-6));
    }

    SECTION("size-2") {
        std::vector data = {2.0f, 1.0f};
        gaussian_internal::BackwardFilter<float, 1>(data.data(), 2, B, bp);
        CHECK_THAT(data[1], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[0], WithinAbs(1.125f, 1e-6));
    }

    SECTION("size-3") {
        std::vector data = {3.0f, 2.0f, 1.0f};
        gaussian_internal::BackwardFilter<float, 1>(data.data(), 3, B, bp);
        CHECK_THAT(data[2], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[1], WithinAbs(1.125f, 1e-6));
        CHECK_THAT(data[0], WithinAbs(1.265625f, 1e-6));
    }

    SECTION("size-4") {
        std::vector data = {4.0f, 3.0f, 2.0f, 1.0f};
        gaussian_internal::BackwardFilter<float, 1>(data.data(), 4, B, bp);
        CHECK_THAT(data[3], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[2], WithinAbs(1.125f, 1e-6));
        CHECK_THAT(data[1], WithinAbs(1.265625f, 1e-6));
        CHECK_THAT(data[0], WithinAbs(1.439453125f, 1e-6));
    }

    SECTION("size-5-stride-3") {
        std::vector data = {5.0f, 0.0f, 0.0f, 4.0f, 0.0f, 0.0f, 3.0f, 0.0f,
                            0.0f, 2.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        gaussian_internal::BackwardFilter(data.data(), 5, B, bp, 3);
        CHECK_THAT(data[12], WithinAbs(1.0f, 1e-6));
        CHECK_THAT(data[9], WithinAbs(1.125f, 1e-6));
        CHECK_THAT(data[6], WithinAbs(1.265625f, 1e-6));
        CHECK_THAT(data[3], WithinAbs(1.439453125f, 1e-6));
        CHECK_THAT(data[0], WithinAbs(1.683837890625f, 1e-6));
    }
}

TEST_CASE("FastGaussian2D") {
    using Catch::Matchers::WithinAbs;

    // Test that impulse response is as expected for Gaussian (with 'replicate'
    // boundaries).
    std::vector<float> data(7 * 7);
    data[7 * 3 + 3] = 1.0f;
    FastGaussian2D(data.data(), 7, 7, 1.0f);

    SECTION("symmetry") {
        SECTION("horizontal") {
            for (std::size_t j = 0; j < 7; ++j) {
                for (std::size_t i = 0; i < 4; ++i) {
                    CHECK_THAT(data[j * 7 + i],
                               WithinAbs(data[j * 7 + 7 - i - 1], 1e-2));
                }
            }
        }

        SECTION("vertical") {
            for (std::size_t j = 0; j < 4; ++j) {
                for (std::size_t i = 0; i < 7; ++i) {
                    CHECK_THAT(data[j * 7 + i],
                               WithinAbs(data[(7 - j - 1) * 7 + i], 1e-2));
                }
            }
        }
    }

    SECTION("isotropy") {
        for (std::size_t j = 0; j < 7; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                CHECK_THAT(data[j * 7 + i], WithinAbs(data[i * 7 + j], 1e-2));
            }
        }
    }

    SECTION("absolute") {
        // Compare an 1/8 wedge; rest is covered by symmetry and isotropy.
        // (0, 0),
        // (1, 0), (1, 1),
        // (2, 0), (2, 1), (2, 2),
        // (3, 0), (3, 1), (3, 2), (3, 3)

        // scipy.ndimage.gaussian_filter(data, sigma=1.0, mode="nearest")
        const std::vector<std::vector<float>> expected = {
            {
                1.96413974e-05f,
            },
            {
                2.39281205e-04f,
                2.91504184e-03f,
            },
            {
                1.07238396e-03f,
                1.30643112e-02f,
                5.85501805e-02f,
            },
            {
                1.76806225e-03f,
                2.15394077e-02f,
                9.65329280e-02f,
                1.59155892e-01f,
            },
        };

        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t i = 0; i <= j; ++i) {
                CAPTURE(j, i);
                CHECK_THAT(data[j * 7 + i], WithinAbs(expected[j][i], 0.05));
            }
        }
    }
}

TEST_CASE("FastPoisson-mean-variance") {
    using Catch::Matchers::WithinAbs;
    rnd::mt19937 rng(12345);
    rnd::uniform_real_distribution<double> uniformDist(0.0, 1.0);

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
    rnd::mt19937 rng(11111);
    rnd::uniform_real_distribution<double> uniformDist(0.0, 1.0);

    double sample = FastPoisson(0.01, rng, uniformDist);
    CHECK(sample >= 0.0);
}

#ifdef USE_HIGHWAY_SIMD
TEST_CASE("FastGaussian2D-SIMD-correctness") {
    using Catch::Matchers::WithinAbs;

    const std::size_t width = GENERATE(0, 8, 32, 64, 127, 128, 129);
    const std::size_t height = GENERATE(0, 8, 32, 64, 127, 128, 129);
    const float sigma = GENERATE(1.0f, 5.0f);

    CAPTURE(width, height, sigma);

    // Initialize with pattern
    std::vector<float> data_reference(width * height);
    for (std::size_t i = 0; i < data_reference.size(); ++i) {
        float val = static_cast<float>((i * 7919) % 1000) / 10.0f;
        data_reference[i] = val;
    }
    std::vector<float> data_test = data_reference;

    gaussian_internal::FastGaussian2DSIMD(data_test.data(), width, height,
                                          sigma);
    gaussian_internal::FastGaussian2DScalar(data_reference.data(), width,
                                            height, sigma);
    for (std::size_t i = 0; i < data_test.size(); ++i) {
        CAPTURE(i);
        CHECK_THAT(data_test[i], WithinAbs(data_reference[i], 1e-3f));
    }
}
#endif