#pragma once

#include "Gaussian2DFilter.h"
#include "Random.h"

#include <blend2d.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

constexpr double PI = 3.1415926535897;

template <typename F>
inline F GaussianSigmaForDefocus(F defocus_um, F numericalAperture,
                                 F refractiveIndex) {
    const auto radius =
        numericalAperture * std::fabs(defocus_um) / refractiveIndex;
    // Apply an (arbitrary) multiplier and minimum.
    return F(0.5) * radius + F(1.0);
}

// Renders the expected (noise-free) signal of a specimen into signal (width *
// height floats), overwriting it: sets up the blend2d canvas
// (scaled/translated to specimen space and marked with an origin cross),
// invokes drawContent(ctx) to paint the specimen-specific structure, then
// applies the defocus blur and intensity scaling common to all simulated
// specimens.
template <typename ContentFn>
void RenderSpecimenImage(float *signal, double x_um, double y_um, double z_um,
                         std::size_t width, std::size_t height,
                         double um_per_px, double na, double intensity,
                         ContentFn &&drawContent) {
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

    const std::size_t nPixels = width * height;

    BLImageData data;
    BLResult status = img.getData(&data);
    if (status != BL_SUCCESS) {
        std::fill(signal, signal + nPixels, 0.0f);
        return; // Give up (shouldn't happen).
    }

    const auto *pix = static_cast<const std::uint32_t *>(data.pixelData);

    // Stride is the stride of scanlines; negative stride means bottom-up.
    const std::intptr_t stride = data.stride / sizeof(std::uint32_t);
    const std::intptr_t start = stride >= 0 ? 0 : (height - 1) * (-stride);
    for (std::intptr_t j = 0; j < std::intptr_t(height); ++j) {
        for (std::intptr_t i = 0; i < std::intptr_t(width); ++i) {
            const auto p = pix[start + i + j * stride];
            // Green sample
            signal[i + j * width] = static_cast<float>((p >> 8) & 0xff);
        }
    }

    // Defocus
    const auto sigmaUm =
        GaussianSigmaForDefocus(float(z_um), float(na), 1.33f);
    const auto sigmaPixels = sigmaUm / float(um_per_px);
    FastGaussian2D(signal, width, height, sigmaPixels);

    // Scale by intensity
    std::transform(signal, signal + nPixels, signal,
                   [i = float(intensity)](float p) { return p * i; });
}

class FilamentsSpecimen {
    struct Filament {
        double x0, y0, x1, y1;
    };

    std::vector<Filament> filaments_;

  public:
    explicit FilamentsSpecimen() {
        using std::cos;
        using std::sin;
        rnd::mt19937 rng;
        rnd::normal_distribution<> xy0Distrib(0.0, 1000.0);
        rnd::uniform_real_distribution<> thetaDistrib(0.0, 2.0 * PI);
        rnd::exponential_distribution<> lenDistrib(1e-3);
        for (int i = 0; i < 1000; ++i) {
            const double x0 = xy0Distrib(rng);
            const double y0 = xy0Distrib(rng);
            const double theta = thetaDistrib(rng);
            const double len = lenDistrib(rng);
            const double x1 = x0 + len * cos(theta);
            const double y1 = y0 + len * sin(theta);
            filaments_.push_back({x0, y0, x1, y1});
        }
    }

    void Draw(float *signal, double x_um, double y_um, double z_um,
              std::size_t width, std::size_t height, double um_per_px,
              double na, double intensity) const {
        const auto &filaments = filaments_;
        RenderSpecimenImage(signal, x_um, y_um, z_um, width, height, um_per_px,
                            na, intensity, [&](BLContext &ctx) {
                                for (const Filament &f : filaments) {
                                    BLPath path;
                                    path.moveTo(f.x0, f.y0);
                                    path.lineTo(f.x1, f.y1);
                                    ctx.strokePath(path, BLRgba32(0xffffffff));
                                }
                            });
    }
};

class NucleiSpecimen {
    struct Punctum {
        double x, y, radius, brightness;
    };
    struct Nucleus {
        double x, y, radius;
        std::vector<Punctum> puncta;
    };

    std::vector<Nucleus> nuclei_;

  public:
    explicit NucleiSpecimen() {
        rnd::mt19937 rng;
        rnd::uniform_real_distribution<> xy0Distrib(-2000.0, 2000.0);
        rnd::uniform_real_distribution<> nucleusRadiusDistrib(15.0, 30.0);
        rnd::uniform_int_distribution<> punctumCountDistrib(1, 6);
        rnd::uniform_real_distribution<> punctumRadiusDistrib(0.5, 1.5);
        rnd::uniform_real_distribution<> punctumBrightnessDistrib(0.4, 1.0);

        for (int i = 0; i < 20; ++i) {
            Nucleus nucleus;
            nucleus.x = xy0Distrib(rng);
            nucleus.y = xy0Distrib(rng);
            nucleus.radius = nucleusRadiusDistrib(rng);

            rnd::normal_distribution<> offsetDistrib(0.0,
                                                     nucleus.radius / 3.0);
            const int nPuncta = punctumCountDistrib(rng);
            for (int p = 0; p < nPuncta; ++p) {
                nucleus.puncta.push_back({nucleus.x + offsetDistrib(rng),
                                          nucleus.y + offsetDistrib(rng),
                                          punctumRadiusDistrib(rng),
                                          punctumBrightnessDistrib(rng)});
            }
            nuclei_.push_back(std::move(nucleus));
        }
    }

    void Draw(float *signal, double x_um, double y_um, double z_um,
              std::size_t width, std::size_t height, double um_per_px,
              double na, double intensity) const {
        const auto &nuclei = nuclei_;
        RenderSpecimenImage(
            signal, x_um, y_um, z_um, width, height, um_per_px, na, intensity,
            [&](BLContext &ctx) {
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
