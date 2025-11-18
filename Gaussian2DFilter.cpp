// Implementation of fast 2D Gaussian filtering according to:
// Young IT, van Vliet LJ, 1995, Signal Processing 44:139-151.
// Recursive implementation of the Gaussian filter.
// https://doi.org/10.1016/0165-1684(95)00020-E

#ifdef USE_HIGHWAY_DYNAMIC_DISPATCH
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "Gaussian2DFilter.cpp"
#include <hwy/foreach_target.h> // IWYU pragma: keep
#endif

#include "Gaussian2DFilter.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef USE_HIGHWAY_SIMD
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace gaussian_internal::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

inline void ForwardFilterVerticalSIMD(float *data, std::size_t width,
                                      std::size_t height, float B,
                                      const std::array<float, 3> &bp) {
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

inline void FastGaussian2DSIMDImpl(float *data, std::size_t width,
                                   std::size_t height, float sigma) {
    const auto bp = b_predivided(b(q(sigma)));
    const auto theB = B(bp);
    ForwardFilterHorizontalSIMD(data, width, height, theB, bp);
    BackwardFilterHorizontalSIMD(data, width, height, theB, bp);
    ForwardFilterVerticalSIMD(data, width, height, theB, bp);
    BackwardFilterVerticalSIMD(data, width, height, theB, bp);
}

} // namespace gaussian_internal::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace gaussian_internal {

#ifdef USE_HIGHWAY_DYNAMIC_DISPATCH
HWY_EXPORT(FastGaussian2DSIMDImpl);
#endif

void FastGaussian2DSIMD(float *data, std::size_t width, std::size_t height,
                        float sigma) {
#ifdef USE_HIGHWAY_DYNAMIC_DISPATCH
    HWY_DYNAMIC_DISPATCH(FastGaussian2DSIMDImpl)(data, width, height, sigma);
#else
    HWY_STATIC_DISPATCH(FastGaussian2DSIMDImpl)(data, width, height, sigma);
#endif
}

} // namespace gaussian_internal

#endif // HWY_ONCE

#endif // USE_HIGHWAY_SIMD

#if defined(USE_HIGHWAY_SIMD) == HWY_ONCE

// Entry point
void FastGaussian2D(float *data, std::size_t width, std::size_t height,
                    float sigma) {
#ifdef USE_HIGHWAY_SIMD
    return gaussian_internal::FastGaussian2DSIMD(data, width, height, sigma);
#endif
    gaussian_internal::FastGaussian2DScalar(data, width, height, sigma);
}

#endif
