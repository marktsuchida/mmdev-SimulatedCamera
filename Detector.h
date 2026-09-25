#pragma once

#include "Random.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Fast Poisson sampling (avoid std::poisson_distribution, which is slow to
// construct for each sample of different lambda).
// - For small lambda: Knuth DE, The Art of Computer Programming, Volume 2:
//   Seminumerical Algorithms, 3rd ed, section 3.4.1.F.3.
// - For large lambda: Gaussian approximation N(lambda, sqrt(lambda))
template <typename F, typename RNG>
F FastPoisson(F lambda, RNG &rng,
              rnd::uniform_real_distribution<F> &uniformDist) {
    if (lambda > F(10.0)) {
        rnd::normal_distribution<F> gaussianDist(lambda, std::sqrt(lambda));
        return std::max(F(0), gaussianDist(rng));
    } else {
        F const L = std::exp(-lambda);
        F p = 1.0;
        int k = 0;
        do {
            ++k;
            p *= uniformDist(rng);
        } while (p > L);
        return static_cast<F>(k - 1);
    }
}

// Simulates sensor readout. signal is the expected photoelectron count per
// pixel, or nullptr for no light. Applies shot noise (to signal), read noise,
// and offset, then rounds and clamps to the pixel range.
template <typename RNG>
void ReadOut(const float *signal, std::uint16_t *out, std::size_t nPixels,
             float readNoise, float offset, RNG &rng) {
    const bool hasReadNoise = readNoise > 0.0f;
    rnd::normal_distribution<float> readNoiseDistrib(
        0.0f, hasReadNoise ? readNoise : 1.0f);
    rnd::uniform_real_distribution<float> uniformDistForPoisson(0.0f, 1.0f);
    const float maxVal = float(std::numeric_limits<std::uint16_t>::max());
    for (std::size_t i = 0; i < nPixels; ++i) {
        float v = offset;
        if (hasReadNoise) {
            v += readNoiseDistrib(rng);
        }
        if (signal && signal[i] > 0.0f) {
            v += FastPoisson(signal[i], rng, uniformDistForPoisson);
        }
        out[i] = static_cast<std::uint16_t>(
            std::clamp(std::round(v), 0.0f, maxVal));
    }
}
