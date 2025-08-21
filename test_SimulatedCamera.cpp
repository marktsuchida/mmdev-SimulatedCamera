#include "SimulatedSpecimen.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

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