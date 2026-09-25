#pragma once

#include "Gaussian2DFilter.h"

#include <blend2d.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#ifdef USE_BOOST_RANDOM
#include <boost/random.hpp>
namespace rnd = boost::random;
#else
#include <random>
namespace rnd = std;
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

// Renders a specimen image: sets up the blend2d canvas (scaled/translated to
// specimen space and marked with an origin cross), invokes drawContent(ctx)
// to paint the specimen-specific structure, then applies the shared
// defocus blur, intensity scaling, and shot/read noise pipeline common to
// all simulated specimens. rng is used for the noise (and may also be used
// by drawContent).
template <typename T, typename ContentFn>
void RenderSpecimenImage(T *buffer, double x_um, double y_um, double z_um,
                         std::size_t width, std::size_t height,
                         double um_per_px, double na, double intensity,
                         rnd::mt19937 &rng, ContentFn &&drawContent) {
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

    drawContent(ctx);

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
    const auto sigmaUm =
        GaussianSigmaForDefocus(float(z_um), float(na), 1.33f);
    const auto sigmaPixels = sigmaUm / float(um_per_px);
    FastGaussian2D(fImage.data(), width, height, sigmaPixels);

    // Scale by intensity
    std::transform(fImage.begin(), fImage.end(), fImage.begin(),
                   [i = float(intensity)](float p) { return p * i; });

    // Shot noise
    auto uniformDistForPoisson =
        rnd::uniform_real_distribution<float>(0.0, 1.0);
    std::transform(fImage.begin(), fImage.end(), fImage.begin(), [&](float p) {
        return p > 0.0f ? FastPoisson(p, rng, uniformDistForPoisson) : p;
    });

    // Gaussian (~read) noise (TODO Adjustable? Scale?)
    // and dark offset (TODO adjustable?)
    auto noiseDistrib = rnd::normal_distribution<float>(0.0, 50.0);
    const float darkOffset = 100.0f;
    std::transform(fImage.begin(), fImage.end(), fImage.begin(), [&](float p) {
        return p + noiseDistrib(rng) + darkOffset;
    });

    // Clamp to pixel type range
    std::transform(fImage.begin(), fImage.end(), buffer, [](float v) {
        return static_cast<T>(std::clamp(
            std::round(v), 0.0f, float(std::numeric_limits<T>::max())));
    });
}

template <typename T> class FilamentsSpecimen {
    struct Filament {
        double x0, y0, x1, y1;
    };

    rnd::mt19937 rng_;
    std::vector<Filament> filaments_;

  public:
    explicit FilamentsSpecimen() {
        using std::cos;
        using std::sin;
        rnd::normal_distribution<> xy0Distrib(0.0, 1000.0);
        rnd::uniform_real_distribution<> thetaDistrib(0.0, 2.0 * PI);
        rnd::exponential_distribution<> lenDistrib(1e-3);
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
              double na, double intensity) {
        const auto &filaments = filaments_;
        RenderSpecimenImage(buffer, x_um, y_um, z_um, width, height, um_per_px,
                            na, intensity, rng_, [&](BLContext &ctx) {
                                for (const Filament &f : filaments) {
                                    BLPath path;
                                    path.moveTo(f.x0, f.y0);
                                    path.lineTo(f.x1, f.y1);
                                    ctx.strokePath(path, BLRgba32(0xffffffff));
                                }
                            });
    }
};

template <typename T> class NucleiSpecimen {
    struct Punctum {
        double x, y, radius, brightness;
    };
    struct Nucleus {
        double x, y, radius;
        std::vector<Punctum> puncta;
    };

    rnd::mt19937 rng_;
    std::vector<Nucleus> nuclei_;

  public:
    explicit NucleiSpecimen() {
        rnd::uniform_real_distribution<> xy0Distrib(-2000.0, 2000.0);
        rnd::uniform_real_distribution<> nucleusRadiusDistrib(15.0, 30.0);
        rnd::uniform_int_distribution<> punctumCountDistrib(1, 6);
        rnd::uniform_real_distribution<> punctumRadiusDistrib(0.5, 1.5);
        rnd::uniform_real_distribution<> punctumBrightnessDistrib(0.4, 1.0);

        for (int i = 0; i < 20; ++i) {
            Nucleus nucleus;
            nucleus.x = xy0Distrib(rng_);
            nucleus.y = xy0Distrib(rng_);
            nucleus.radius = nucleusRadiusDistrib(rng_);

            rnd::normal_distribution<> offsetDistrib(0.0,
                                                     nucleus.radius / 3.0);
            const int nPuncta = punctumCountDistrib(rng_);
            for (int p = 0; p < nPuncta; ++p) {
                nucleus.puncta.push_back({nucleus.x + offsetDistrib(rng_),
                                          nucleus.y + offsetDistrib(rng_),
                                          punctumRadiusDistrib(rng_),
                                          punctumBrightnessDistrib(rng_)});
            }
            nuclei_.push_back(std::move(nucleus));
        }
    }

    void Draw(T *buffer, double x_um, double y_um, double z_um,
              std::size_t width, std::size_t height, double um_per_px,
              double na, double intensity) {
        const auto &nuclei = nuclei_;
        RenderSpecimenImage(
            buffer, x_um, y_um, z_um, width, height, um_per_px, na, intensity,
            rng_, [&](BLContext &ctx) {
                for (const Nucleus &n : nuclei) {
                    ctx.fillCircle(BLCircle(n.x, n.y, n.radius),
                                   BLRgba32(40, 40, 40));
                    for (const Punctum &p : n.puncta) {
                        const auto v = static_cast<std::uint32_t>(
                            std::clamp(255.0 * p.brightness, 0.0, 255.0));
                        ctx.fillCircle(BLCircle(p.x, p.y, p.radius),
                                       BLRgba32(v, v, v));
                    }
                }
            });
    }
};
