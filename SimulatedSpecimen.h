#pragma once

#include <blend2d.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

#ifdef USE_BOOST_RANDOM
#include <boost/random.hpp>
#define RAND_NS boost::random
#else
#include <random>
#define RAND_NS std
#endif

constexpr double PI = 3.1415926535897;

template <typename F>
inline F GaussianSigmaForDefocus(F defocus_um, F numericalAperture,
                                 F refractiveIndex) {
    const auto radius =
        numericalAperture * std::fabs(defocus_um) / refractiveIndex;
    // Apply an (arbitrary) multiplier and minimum.
    return F(0.5) * radius + F(1.0);
}

// Implementation of fast 2D Gaussian filtering according to:
// Young IT, van Vliet LJ, 1995, Signal Processing 44:139-151.
// Recursive implementation of the Gaussian filter.
// https://doi.org/10.1016/0165-1684(95)00020-E
namespace gaussian_internal {

// Eq 11b
template <typename F> inline float q(F sigma) {
    assert(sigma > F(0.0));

    // Note: This function, as given in the paper, is not continuous at sigma =
    // 2.5. The jump in q value is about 0.1. If we really care, we should
    // redo the fitting and approximations.

    if (sigma < F(2.5)) {
        return F(3.97156) -
               F(4.14554) * std::sqrt(F(1.0) - F(0.26891) * sigma);
    }
    return F(0.98711) * sigma - F(0.96330);
}

// Eq 8c
template <typename F> inline std::array<F, 4> b(F q) {
    const F qq = q * q;
    const F qqq = qq * q;
    return {
        F(1.57825) + F(2.44413) * q + F(1.4281) * qq + F(0.422205) * qqq,
        F(2.44413) * q + F(2.85619) * qq + F(1.26661) * qqq,
        -(F(1.4281) * qq + F(1.26661) * qqq),
        F(0.422205) * qqq,
    };
}

// Pre-divide b[1:3] by b[0] for Eqs 9a, 9b, 10
template <typename F>
inline std::array<F, 3> b_predivided(std::array<F, 4> b) {
    const F rb0 = F(1.0) / b[0];
    return {b[1] * rb0, b[2] * rb0, b[3] * rb0};
}

// Eq 10
template <typename F> inline F B(std::array<F, 3> bp) {
    return F(1.0) - (bp[0] + bp[1] + bp[2]);
}

// Eq 9a (Stride == 0 uses dynamic stride)
template <typename F, std::size_t Stride = 0>
inline void ForwardFilter(F *data, std::size_t size, F B, std::array<F, 3> bp,
                          std::size_t stride = Stride) {
    const auto s = Stride > 0 ? Stride : stride;
    F *const pend = data + size * s;
    F *p = data;
    // Replicate the 3 pixels outside the border.
    if (size > 0) {
        // First pixel unchanged.
    }
    p += s;
    if (size > 1) {
        *p = B * *p + (bp[0] + bp[1] + bp[2]) * *(p - s);
    }
    p += s;
    if (size > 2) {
        *p = B * *p + bp[0] * *(p - s) + (bp[1] + bp[2]) * *(p - 2 * s);
    }
    p += s;
    for (; p < pend; p += s) {
        *p = B * *p + bp[0] * *(p - s) + bp[1] * *(p - 2 * s) +
             bp[2] * *(p - 3 * s);
    }
}

// Eq 9b (Stride == 0 uses dynamic stride)
template <typename F, std::size_t Stride = 0>
inline void BackwardFilter(F *data, std::size_t size, F B, std::array<F, 3> bp,
                           std::size_t stride = Stride) {
    const auto s = Stride > 0 ? Stride : stride;
    F *const prend = data - 1;
    F *p = data + (size - 1) * s;
    // Replicate the 3 pixels outside of the border.
    if (size > 0) {
        // Rightmost pixel unchanged.
    }
    p -= s;
    if (size > 1) {
        *p = B * *p + (bp[0] + bp[1] + bp[2]) * *(p + s);
    }
    p -= s;
    if (size > 2) {
        *p = B * *p + bp[0] * *(p + s) + (bp[1] + bp[2]) * *(p + 2 * s);
    }
    p -= s;
    for (; p > prend; p -= s) {
        *p = B * *p + bp[0] * *(p + s) + bp[1] * *(p + 2 * s) +
             bp[2] * *(p + 3 * s);
    }
}

} // namespace gaussian_internal

template <typename F>
inline void FastGaussian2D(F *data, std::size_t width, std::size_t height,
                           F sigma) {
    const F q = gaussian_internal::q(sigma);
    const auto b = gaussian_internal::b(q);
    const auto bp = gaussian_internal::b_predivided(b);
    const auto B = gaussian_internal::B(bp);
    // Horizontal
    for (std::size_t j = 0; j < height; ++j) {
        auto *row = data + j * width;
        gaussian_internal::ForwardFilter<F, 1>(row, width, B, bp);
        gaussian_internal::BackwardFilter<F, 1>(row, width, B, bp);
    }
    // Vertical
    for (std::size_t i = 0; i < width; ++i) {
        auto *col = data + i;
        gaussian_internal::ForwardFilter<F>(col, height, B, bp, width);
        gaussian_internal::BackwardFilter<F>(col, height, B, bp, width);
    }
}

// Fast Poisson sampling (avoid std::poisson_distribution, which is slow to
// construct for each sample of different lambda).
// - For small lambda: Knuth DE, The Art of Computer Programming, Volume 2:
//   Seminumerical Algorithms, 3rd ed, section 3.4.1.F.3.
// - For large lambda: Gaussian approximation N(lambda, sqrt(lambda))
template <typename F, typename RNG>
F FastPoisson(F lambda, RNG &rng,
              RAND_NS::uniform_real_distribution<F> &uniformDist) {
    if (lambda > F(10.0)) {
        RAND_NS::normal_distribution<F> gaussianDist(lambda,
                                                     std::sqrt(lambda));
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

template <typename T> class SimulatedSpecimen {
    struct Filament {
        double x0, y0, x1, y1;
    };

    RAND_NS::mt19937 rng_;
    std::vector<Filament> filaments_;

  public:
    explicit SimulatedSpecimen() {
        using std::cos;
        using std::sin;
        RAND_NS::normal_distribution<> xy0Distrib(0.0, 1000.0);
        RAND_NS::uniform_real_distribution<> thetaDistrib(0.0, 2.0 * PI);
        RAND_NS::exponential_distribution<> lenDistrib(1e-3);
        for (int i = 0; i < 1000; ++i) {
            const double x0 = xy0Distrib(rng_);
            const double y0 = xy0Distrib(rng_);
            const double theta = thetaDistrib(rng_);
            const double len = lenDistrib(rng_);
            const double x1 = x0 + len * cos(theta);
            const double y1 = y0 + len * sin(theta);
            filaments_.push_back({x0, y0, x1, y1});
        }
    }

    void Draw(T *buffer, double x_um, double y_um, double z_um,
              std::size_t width, std::size_t height, double um_per_px,
              double intensity) {
        const auto &filaments = filaments_;

        BLImage img(static_cast<int>(width), static_cast<int>(height),
                    BL_FORMAT_XRGB32);
        BLContext ctx(img);
        ctx.clearAll();

        ctx.scale(1.0 / um_per_px);
        ctx.translate(x_um, y_um);

        BLPath origin_marker;
        origin_marker.moveTo(0.0, 0.0);
        origin_marker.lineTo(50.0, 50.0);
        origin_marker.addCircle(BLCircle(0.0, 0.0, 50.0));
        ctx.strokePath(origin_marker, BLRgba32(0xaaaaaaaa));

        for (const Filament &f : filaments) {
            BLPath path;
            path.moveTo(f.x0, f.y0);
            path.lineTo(f.x1, f.y1);
            ctx.strokePath(path, BLRgba32(0xffffffff));
        }

        ctx.end();

        BLImageData data;
        BLResult status = img.getData(&data);
        if (status != BL_SUCCESS) {
            std::memset(buffer, 0, sizeof(T) * width * height);
            return; // Give up (shouldn't happen).
        }

        const auto *pix = static_cast<const std::uint32_t *>(data.pixelData);

        // Stride is the stride of scanlines; negative stride means bottom-up.
        std::vector<float> fImage(width * height);
        const std::intptr_t stride = data.stride / sizeof(std::uint32_t);
        const std::intptr_t start = stride >= 0 ? 0 : (height - 1) * (-stride);
        for (std::intptr_t j = 0; j < std::intptr_t(height); ++j) {
            for (std::intptr_t i = 0; i < std::intptr_t(width); ++i) {
                const auto p = pix[start + i + j * stride];
                // Green sample
                fImage[i + j * width] = static_cast<float>((p >> 8) & 0xff);
            }
        }

        // Defocus
        const auto sigmaUm = GaussianSigmaForDefocus(float(z_um), 1.4f, 1.33f);
        const auto sigmaPixels = sigmaUm / float(um_per_px);
        FastGaussian2D(fImage.data(), width, height, sigmaPixels);

        // Scale by intensity
        std::transform(fImage.begin(), fImage.end(), fImage.begin(),
                       [i = float(intensity)](float p) { return p * i; });

        // Shot noise
        auto uniformDistForPoisson =
            RAND_NS::uniform_real_distribution<float>(0.0, 1.0);
        std::transform(
            fImage.begin(), fImage.end(), fImage.begin(), [&](float p) {
                return p > 0.0f ? FastPoisson(p, rng_, uniformDistForPoisson)
                                : p;
            });

        // Gaussian (~read) noise (TODO Adjustable? Scale?)
        // and dark offset (TODO adjustable?)
        auto noiseDistrib = RAND_NS::normal_distribution<float>(0.0, 50.0);
        const float darkOffset = 100.0f;
        std::transform(
            fImage.begin(), fImage.end(), fImage.begin(),
            [&](float p) { return p + noiseDistrib(rng_) + darkOffset; });

        // Clamp to pixel type range
        std::transform(fImage.begin(), fImage.end(), buffer, [](float v) {
            return static_cast<T>(std::clamp(
                std::round(v), 0.0f, float(std::numeric_limits<T>::max())));
        });
    }
};
