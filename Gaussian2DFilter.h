#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

void FastGaussian2D(float *data, std::size_t width, std::size_t height,
                    float sigma);

namespace gaussian_internal {

// Eq 11b
template <typename F> inline float q(F sigma) {
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
    const auto s = Stride > 0 ? Stride : stride;
    F *const pend = data + size * s;
    F *p = data;
    if (size < 1)
        return;
    F cur = *p;
    // Replicate the 3 pixels outside the border.
    F prev1 = cur;
    F prev2 = cur;
    F prev3 = cur;
    p += s;
    for (; p < pend; p += s) {
        prev3 = prev2;
        prev2 = prev1;
        prev1 = cur;
        cur = *p;
        cur = B * cur + bp[0] * prev1 + bp[1] * prev2 + bp[2] * prev3;
        *p = cur;
    }
}

// Eq 9b (Stride == 0 uses dynamic stride)
template <typename F, std::size_t Stride = 0>
inline void BackwardFilter(F *data, std::size_t size, F B, std::array<F, 3> bp,
                           std::size_t stride = Stride) {
    const auto s = Stride > 0 ? Stride : stride;
    F *const prend = data - 1;
    F *p = data + (size - 1) * s;
    if (size < 1)
        return;
    F cur = *p;
    // Replicate the 3 pixels outside of the border.
    F next1 = cur;
    F next2 = cur;
    F next3 = cur;
    p -= s;
    for (; p > prend; p -= s) {
        next3 = next2;
        next2 = next1;
        next1 = cur;
        cur = *p;
        cur = B * cur + bp[0] * next1 + bp[1] * next2 + bp[2] * next3;
        *p = cur;
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

// For unit test access
void FastGaussian2DSIMD(float *data, std::size_t width, std::size_t height,
                        float sigma);

} // namespace gaussian_internal