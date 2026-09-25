#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <vector>

void FastGaussian2D(float *data, std::size_t width, std::size_t height,
                    float sigma);

namespace gaussian_internal {

// Eq 11b
template <typename F> inline F q(F sigma) {
    assert(sigma > F(0.0));

    // Note: This function, as given in the paper, is not continuous at sigma =
    // 2.5. The jump in q value is about 0.1. If we really care, we should
    // redo the fitting and approximations.

    // For a possible alternative (not used here), see Eq 11 in:
    // Yount IT, van Vliet LJ, van Ginkel M, 2002, IEEE Trans Sig Proc
    // 50:2798-2805. Recursive Gabor Filtering.
    // https://doi.org/10.1109/TSP.2002.804095

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
    if (size < 1)
        return;
    const auto s = Stride > 0 ? Stride : stride;
    F cur = data[0];
    // Replicate the 3 pixels outside the border.
    F prev1 = cur;
    F prev2 = cur;
    F prev3 = cur;
    for (std::size_t k = 1; k < size; ++k) {
        prev3 = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = B * data[k * s] + bp[0] * prev1 + bp[1] * prev2 + bp[2] * prev3;
        data[k * s] = cur;
    }
}

// Eq 9b (Stride == 0 uses dynamic stride)
template <typename F, std::size_t Stride = 0>
inline void BackwardFilter(F *data, std::size_t size, F B, std::array<F, 3> bp,
                           std::size_t stride = Stride) {
    if (size < 1)
        return;
    const auto s = Stride > 0 ? Stride : stride;
    F cur = data[(size - 1) * s];
    // Replicate the 3 pixels outside of the border.
    F next1 = cur;
    F next2 = cur;
    F next3 = cur;
    for (std::size_t k = size - 1; k-- > 0;) {
        next3 = next2;
        next2 = next1;
        next1 = cur;
        cur = B * data[k * s] + bp[0] * next1 + bp[1] * next2 + bp[2] * next3;
        data[k * s] = cur;
    }
}

template <typename F>
void FastGaussian2DScalar(F *data, std::size_t width, std::size_t height,
                          F sigma) {
    const auto bp = b_predivided(b(q(sigma)));
    const auto theB = B(bp);
    // Horizontal
    for (std::size_t j = 0; j < height; ++j) {
        auto *row = data + j * width;
        ForwardFilter<F, 1>(row, width, theB, bp);
        BackwardFilter<F, 1>(row, width, theB, bp);
    }
    // Vertical
    for (std::size_t i = 0; i < width; ++i) {
        auto *col = data + i;
        ForwardFilter<F>(col, height, theB, bp, width);
        BackwardFilter<F>(col, height, theB, bp, width);
    }
}

// Separable convolution with a sampled Gaussian kernel truncated at 3 sigma,
// with replicated boundaries.
template <typename F>
void DirectGaussian2D(F *data, std::size_t width, std::size_t height,
                      F sigma) {
    if (width == 0 || height == 0)
        return;
    const auto radius = std::max<std::ptrdiff_t>(
        1, static_cast<std::ptrdiff_t>(std::ceil(F(3) * sigma)));
    std::vector<F> kernel(static_cast<std::size_t>(2 * radius + 1));
    F sum = 0;
    for (std::ptrdiff_t k = -radius; k <= radius; ++k) {
        const F w = std::exp(-F(k * k) / (F(2) * sigma * sigma));
        kernel[static_cast<std::size_t>(k + radius)] = w;
        sum += w;
    }
    for (auto &w : kernel)
        w /= sum;

    std::vector<F> buf(std::max(width, height));
    const auto convolve = [&](std::size_t size, std::size_t stride, F *line) {
        for (std::size_t k = 0; k < size; ++k)
            buf[k] = line[k * stride];
        const auto last = static_cast<std::ptrdiff_t>(size) - 1;
        for (std::ptrdiff_t k = 0; k <= last; ++k) {
            F acc = 0;
            for (std::ptrdiff_t m = -radius; m <= radius; ++m) {
                const auto idx = std::clamp<std::ptrdiff_t>(k + m, 0, last);
                acc += kernel[static_cast<std::size_t>(m + radius)] *
                       buf[static_cast<std::size_t>(idx)];
            }
            line[static_cast<std::size_t>(k) * stride] = acc;
        }
    };
    for (std::size_t j = 0; j < height; ++j)
        convolve(width, 1, data + j * width);
    for (std::size_t i = 0; i < width; ++i)
        convolve(height, width, data + i);
}

// For unit test access
void FastGaussian2DSIMD(float *data, std::size_t width, std::size_t height,
                        float sigma);

} // namespace gaussian_internal