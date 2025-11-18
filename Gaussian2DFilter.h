#pragma once

#ifdef USE_HIGHWAY_SIMD
#include "hwy/highway.h"
#endif

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <type_traits>

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

#ifdef USE_HIGHWAY_SIMD

inline void ForwardFilterVerticalSIMD(float *data, std::size_t width,
                                      std::size_t height, float B,
                                      const std::array<float, 3> &bp) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    const std::size_t N = hn::Lanes(d);

    const auto vec_B = hn::Set(d, B);
    const auto vec_b1p = hn::Set(d, bp[0]);
    const auto vec_b2p = hn::Set(d, bp[1]);
    const auto vec_b3p = hn::Set(d, bp[2]);

    // Processes N adjacent columns simultaneously
    for (std::size_t i = 0; i + N <= width; i += N) {
        float *p0 = data + i;
        if (height < 1)
            return;
        auto row_curr = hn::LoadU(d, p0);
        // Replicate the 3 rows before the border.
        auto row_prev1 = row_curr;
        auto row_prev2 = row_curr;
        auto row_prev3 = row_curr;
        for (std::size_t j = 1; j < height; ++j) {
            float *const p = p0 + j * width;
            row_prev3 = row_prev2;
            row_prev2 = row_prev1;
            row_prev1 = row_curr;
            row_curr = hn::LoadU(d, p);
            const auto result = hn::MulAdd(
                vec_B, row_curr,
                hn::MulAdd(vec_b1p, row_prev1,
                           hn::MulAdd(vec_b2p, row_prev2,
                                      hn::Mul(vec_b3p, row_prev3))));
            hn::StoreU(result, d, p);
            row_curr = result;
        }
    }

    const std::size_t remainder_start = (width / N) * N;
    for (std::size_t i = remainder_start; i < width; ++i) {
        ForwardFilter<float>(data + i, height, B, bp, width);
    }
}

inline void BackwardFilterVerticalSIMD(float *data, std::size_t width,
                                       std::size_t height, float B,
                                       const std::array<float, 3> &bp) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    const std::size_t N = hn::Lanes(d);

    const auto vec_B = hn::Set(d, B);
    const auto vec_b1p = hn::Set(d, bp[0]);
    const auto vec_b2p = hn::Set(d, bp[1]);
    const auto vec_b3p = hn::Set(d, bp[2]);

    // Process N adjacent columns simultaneously
    for (std::size_t i = 0; i + N <= width; i += N) {
        float *p0 = data + i;
        if (height < 1)
            return;
        auto row_curr = hn::LoadU(d, p0 + (height - 1) * width);
        // Replicate the 3 rows after the border.
        auto row_next1 = row_curr;
        auto row_next2 = row_curr;
        auto row_next3 = row_curr;
        for (std::size_t j = height - 2;; --j) {
            float *const p = p0 + j * width;
            row_next3 = row_next2;
            row_next2 = row_next1;
            row_next1 = row_curr;
            row_curr = hn::LoadU(d, p);
            const auto result = hn::MulAdd(
                vec_B, row_curr,
                hn::MulAdd(vec_b1p, row_next1,
                           hn::MulAdd(vec_b2p, row_next2,
                                      hn::Mul(vec_b3p, row_next3))));
            hn::StoreU(result, d, p);
            row_curr = result;
            if (j == 0) {
                break; // Exit before underflow
            }
        }
    }

    const std::size_t remainder_start = (width / N) * N;
    for (std::size_t i = remainder_start; i < width; ++i) {
        BackwardFilter<float>(data + i, height, B, bp, width);
    }
}

inline void ForwardFilterHorizontalSIMD(float *data, std::size_t width,
                                        std::size_t height, float B,
                                        const std::array<float, 3> &bp) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    const hn::ScalableTag<std::int32_t> di;
    const std::size_t N = hn::Lanes(d);

    const auto vec_B = hn::Set(d, B);
    const auto vec_b1p = hn::Set(d, bp[0]);
    const auto vec_b2p = hn::Set(d, bp[1]);
    const auto vec_b3p = hn::Set(d, bp[2]);

    // Process N consecutive rows simultaneously
    for (std::size_t j = 0; j + N <= height; j += N) {
        const auto idx0 = hn::Iota(di, static_cast<std::int32_t>(j)) *
                          hn::Set(di, static_cast<std::int32_t>(width));
        if (width < 1)
            return;
        auto col_curr = hn::GatherIndex(d, data, idx0);
        // Replicate the 3 columns before the border.
        auto col_prev1 = col_curr;
        auto col_prev2 = col_curr;
        auto col_prev3 = col_curr;
        for (std::size_t i = 1; i < width; ++i) {
            const auto idx = idx0 + hn::Set(di, static_cast<std::int32_t>(i));
            col_prev3 = col_prev2;
            col_prev2 = col_prev1;
            col_prev1 = col_curr;
            col_curr = hn::GatherIndex(d, data, idx);
            const auto result = hn::MulAdd(
                vec_B, col_curr,
                hn::MulAdd(vec_b1p, col_prev1,
                           hn::MulAdd(vec_b2p, col_prev2,
                                      hn::Mul(vec_b3p, col_prev3))));
            hn::ScatterIndex(result, d, data, idx);
            col_curr = result;
        }
    }

    const std::size_t remainder_start = (height / N) * N;
    for (std::size_t j = remainder_start; j < height; ++j) {
        ForwardFilter<float, 1>(data + j * width, width, B, bp);
    }
}

inline void BackwardFilterHorizontalSIMD(float *data, std::size_t width,
                                         std::size_t height, float B,
                                         const std::array<float, 3> &bp) {
    namespace hn = hwy::HWY_NAMESPACE;
    const hn::ScalableTag<float> d;
    const hn::ScalableTag<std::int32_t> di;
    const std::size_t N = hn::Lanes(d);

    const auto vec_B = hn::Set(d, B);
    const auto vec_b1p = hn::Set(d, bp[0]);
    const auto vec_b2p = hn::Set(d, bp[1]);
    const auto vec_b3p = hn::Set(d, bp[2]);

    // Process N consecutive rows simultaneously
    for (std::size_t j = 0; j + N <= height; j += N) {
        const auto idx0 = hn::Iota(di, static_cast<std::int32_t>(j)) *
                          hn::Set(di, static_cast<std::int32_t>(width));
        if (width < 1)
            return;
        auto col_curr = hn::GatherIndex(
            d, data, idx0 + hn::Set(di, static_cast<std::int32_t>(width - 1)));
        // Replicate the 3 columns after the border.
        auto col_next1 = col_curr;
        auto col_next2 = col_curr;
        auto col_next3 = col_curr;
        for (std::size_t i = width - 2;; --i) {
            const auto idx = idx0 + hn::Set(di, static_cast<std::int32_t>(i));
            col_next3 = col_next2;
            col_next2 = col_next1;
            col_next1 = col_curr;
            col_curr = hn::GatherIndex(d, data, idx);
            const auto result = hn::MulAdd(
                vec_B, col_curr,
                hn::MulAdd(vec_b1p, col_next1,
                           hn::MulAdd(vec_b2p, col_next2,
                                      hn::Mul(vec_b3p, col_next3))));
            hn::ScatterIndex(result, d, data, idx);
            col_curr = result;
            if (i == 0)
                break; // Exit before underflow
        }
    }

    const std::size_t remainder_start = (height / N) * N;
    for (std::size_t j = remainder_start; j < height; ++j) {
        BackwardFilter<float, 1>(data + j * width, width, B, bp);
    }
}

#endif // USE_HIGHWAY_SIMD

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

#ifdef USE_HIGHWAY_SIMD

inline void FastGaussian2DSIMD(float *data, std::size_t width,
                               std::size_t height, float sigma) {
    const auto bp = b_predivided(b(q(sigma)));
    const auto theB = B(bp);
    ForwardFilterHorizontalSIMD(data, width, height, theB, bp);
    BackwardFilterHorizontalSIMD(data, width, height, theB, bp);
    ForwardFilterVerticalSIMD(data, width, height, theB, bp);
    BackwardFilterVerticalSIMD(data, width, height, theB, bp);
}

#endif // USE_HIGHWAY_SIMD

} // namespace gaussian_internal

template <typename F>
void FastGaussian2D(F *data, std::size_t width, std::size_t height, F sigma) {
#ifdef USE_HIGHWAY_SIMD
    if constexpr (std::is_same_v<F, float>) {
        return gaussian_internal::FastGaussian2DSIMD(data, width, height,
                                                     sigma);
    }
#endif
    gaussian_internal::FastGaussian2DScalar(data, width, height, sigma);
}
