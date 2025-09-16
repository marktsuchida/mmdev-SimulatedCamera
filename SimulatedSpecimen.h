#pragma once

#include <blend2d.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

constexpr double PI = 3.1415926535897;

inline double GaussianSigmaForDefocus(double defocus_um,
                                      double numericalAperture,
                                      double refractiveIndex) {
    const auto radius =
        numericalAperture * std::fabs(defocus_um) / refractiveIndex;
    // Apply an (arbitrary) multiplier and minimum.
    return 0.5 * radius + 1.0;
}

inline float UnnormalizedGaussian(float x, float sigma) {
    return std::exp(-(x * x) / (2.0f * sigma * sigma));
}

inline std::vector<float> GaussianKernel(std::size_t kernelRadius,
                                         float sigma) {
    const auto kernelSize = 2 * kernelRadius + 1;
    std::vector<float> kernel(kernelSize);
    if (sigma < 1e-6) {
        kernel[kernelRadius] = 1.0f;
        return kernel;
    }

    std::vector<std::intptr_t> indices(kernelSize);
    std::iota(indices.begin(), indices.end(),
              -static_cast<intptr_t>(kernelRadius));
    std::transform(indices.begin(), indices.end(), kernel.begin(),
                   [sigma](std::intptr_t i) {
                       return UnnormalizedGaussian(float(i), sigma);
                   });
    const auto sum = std::accumulate(kernel.begin(), kernel.end(), 0.0f);
    const auto norm = 1.0f / sum;
    std::for_each(kernel.begin(), kernel.end(),
                  [norm](float &e) { e *= norm; });
    return kernel;
}

template <typename T>
void HConvolve(T *data, std::size_t width, std::size_t height, const T *kernel,
               std::size_t kernelRadius) {
    const std::size_t kernelSize = 2 * kernelRadius + 1;
    std::vector<T> paddedRow(kernelRadius + width + kernelRadius);
    for (std::size_t j = 0; j < height; ++j) {
        // Copy to paddedRow with reflecting boundary condition.
        const auto rowBegin = std::next(data, j * width);
        const auto rowEnd = std::next(data, (j + 1) * width);
        std::reverse_copy(std::next(rowBegin, 1),
                          std::next(rowBegin, 1 + kernelRadius),
                          paddedRow.begin());
        std::copy(rowBegin, rowEnd,
                  std::next(paddedRow.begin(), kernelRadius));
        std::reverse_copy(std::prev(rowEnd, 1 + kernelRadius),
                          std::prev(rowEnd, 1),
                          std::prev(paddedRow.end(), kernelRadius));

        for (std::size_t i = 0; i < width; ++i) {
            data[i + j * width] =
                std::inner_product(kernel, std::next(kernel, kernelSize),
                                   std::next(paddedRow.begin(), i), T(0));
        }
    }
}

template <typename T>
void VConvolve(T *data, std::size_t width, std::size_t height, const T *kernel,
               std::size_t kernelRadius) {
    const std::size_t kernelSize = 2 * kernelRadius + 1;
    std::vector<T> column(height); // Copy for simpler algorithm use.
    std::vector<T> paddedCol(kernelRadius + height + kernelRadius);
    for (std::size_t i = 0; i < width; ++i) {
        for (std::size_t j = 0; j < height; ++j) {
            column[j] = data[i + j * width];
        }

        // Copy to paddedCol with reflecting boundary condition.
        std::reverse_copy(std::next(column.begin(), 1),
                          std::next(column.begin(), 1 + kernelRadius),
                          paddedCol.begin());
        std::copy(column.begin(), column.end(),
                  std::next(paddedCol.begin(), kernelRadius));
        std::reverse_copy(std::prev(column.end(), 1 + kernelRadius),
                          std::prev(column.end(), 1),
                          std::prev(paddedCol.end(), kernelRadius));

        for (std::size_t j = 0; j < height; ++j) {
            data[i + j * width] =
                std::inner_product(kernel, std::next(kernel, kernelSize),
                                   std::next(paddedCol.begin(), j), T(0));
        }
    }
}

template <typename T> class SimulatedSpecimen {
    struct Filament {
        double x0, y0, x1, y1;
    };

    std::mt19937 rng_;
    std::vector<Filament> filaments_;

  public:
    explicit SimulatedSpecimen() {
        using std::cos;
        using std::sin;
        std::normal_distribution<> xy0Distrib(0.0, 1000.0);
        std::uniform_real_distribution<> thetaDistrib(0.0, 2.0 * PI);
        std::exponential_distribution<> lenDistrib(1e-3);
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
        ctx.translate(x_um, -y_um);

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

        const auto sigmaUm = GaussianSigmaForDefocus(z_um, 1.4, 1.33);
        const auto sigmaPixels = sigmaUm / um_per_px;

        // 3-sigma kernel radius, but max out at image size.
        const auto threeSigma =
            static_cast<std::intptr_t>(std::ceil(3.0 * sigmaPixels));
        const auto maxDim =
            static_cast<std::intptr_t>(std::max(width, height));
        const auto kernelRadius = std::min(threeSigma, maxDim);

        // Convolve in two 1D steps
        const auto kernel = GaussianKernel(kernelRadius, float(sigmaPixels));
        HConvolve(fImage.data(), width, height, kernel.data(), kernelRadius);
        VConvolve(fImage.data(), width, height, kernel.data(), kernelRadius);

        // Gaussian (~read) noise; TODO Adjustable? Scale?
        auto noiseDistrib = std::normal_distribution(0.0, 50.0);

        std::for_each(fImage.begin(), fImage.end(),
                      [this, intensity, &noiseDistrib](float &p) {
                          double pp = p;

                          pp *= intensity;

                          // Shot noise
                          if (pp > 0.0) {
                              auto shotDistrib =
                                  std::poisson_distribution<std::int64_t>(pp);
                              pp = static_cast<double>(shotDistrib(rng_));
                          }

                          // Gaussian noise
                          pp += noiseDistrib(rng_);

                          // Dark offset (TODO adjustable?)
                          pp += 100.0;

                          p = static_cast<float>(pp);
                      });

        std::transform(fImage.begin(), fImage.end(), buffer, [](float v) {
            return static_cast<T>(std::clamp(
                std::roundf(v), 0.0f, float(std::numeric_limits<T>::max())));
        });
    }
};
