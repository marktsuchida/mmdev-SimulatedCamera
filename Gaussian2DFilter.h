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
    const auto vec_bp0 = hn::Set(d, bp[0]);
    const auto vec_bp1 = hn::Set(d, bp[1]);
    const auto vec_bp2 = hn::Set(d, bp[2]);
    const auto vec_bp_sum = hn::Set(d, bp[0] + bp[1] + bp[2]);
    const auto vec_bp12_sum = hn::Set(d, bp[1] + bp[2]);

    // Processes N adjacent columns simultaneously
    for (std::size_t i = 0; i + N <= width; i += N) {
        if (height > 0) {
            // First row unchanged.
        }

        if (height > 1) {
            float *p = data + 1 * width + i;
            auto curr = hn::LoadU(d, p);
            auto prev0 = hn::LoadU(d, p - width);
            auto result = hn::MulAdd(vec_B, curr, hn::Mul(vec_bp_sum, prev0));
            hn::StoreU(result, d, p);
        }

        if (height > 2) {
            float *p = data + 2 * width + i;
            auto curr = hn::LoadU(d, p);
            auto prev1 = hn::LoadU(d, p - width);
            auto prev0 = hn::LoadU(d, p - 2 * width);
            auto result = hn::MulAdd(
                vec_B, curr,
                hn::MulAdd(vec_bp0, prev1, hn::Mul(vec_bp12_sum, prev0)));
            hn::StoreU(result, d, p);
        }

        for (std::size_t j = 3; j < height; ++j) {
            float *p = data + j * width + i;
            auto curr = hn::LoadU(d, p);
            auto prev1 = hn::LoadU(d, p - width);
            auto prev2 = hn::LoadU(d, p - 2 * width);
            auto prev3 = hn::LoadU(d, p - 3 * width);

            auto result =
                hn::MulAdd(vec_B, curr,
                           hn::MulAdd(vec_bp0, prev1,
                                      hn::MulAdd(vec_bp1, prev2,
                                                 hn::Mul(vec_bp2, prev3))));
            hn::StoreU(result, d, p);
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
    const auto vec_bp0 = hn::Set(d, bp[0]);
    const auto vec_bp1 = hn::Set(d, bp[1]);
    const auto vec_bp2 = hn::Set(d, bp[2]);
    const auto vec_bp_sum = hn::Set(d, bp[0] + bp[1] + bp[2]);
    const auto vec_bp12_sum = hn::Set(d, bp[1] + bp[2]);

    // Process N adjacent columns simultaneously
    for (std::size_t i = 0; i + N <= width; i += N) {
        if (height > 0) {
            // Last row unchanged.
        }

        if (height > 1) {
            float *p = data + (height - 2) * width + i;
            auto curr = hn::LoadU(d, p);
            auto next0 = hn::LoadU(d, p + width);
            auto result = hn::MulAdd(vec_B, curr, hn::Mul(vec_bp_sum, next0));
            hn::StoreU(result, d, p);
        }

        if (height > 2) {
            float *p = data + (height - 3) * width + i;
            auto curr = hn::LoadU(d, p);
            auto next1 = hn::LoadU(d, p + width);
            auto next0 = hn::LoadU(d, p + 2 * width);
            auto result = hn::MulAdd(
                vec_B, curr,
                hn::MulAdd(vec_bp0, next1, hn::Mul(vec_bp12_sum, next0)));
            hn::StoreU(result, d, p);
        }

        if (height > 3) {
            for (std::size_t j = height - 4;; --j) {
                float *p = data + j * width + i;
                auto curr = hn::LoadU(d, p);
                auto next1 = hn::LoadU(d, p + width);
                auto next2 = hn::LoadU(d, p + 2 * width);
                auto next3 = hn::LoadU(d, p + 3 * width);

                auto result = hn::MulAdd(
                    vec_B, curr,
                    hn::MulAdd(
                        vec_bp0, next1,
                        hn::MulAdd(vec_bp1, next2, hn::Mul(vec_bp2, next3))));
                hn::StoreU(result, d, p);

                if (j == 0) {
                    break; // Exit before underflow
                }
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
    const auto vec_bp0 = hn::Set(d, bp[0]);
    const auto vec_bp1 = hn::Set(d, bp[1]);
    const auto vec_bp2 = hn::Set(d, bp[2]);
    const auto vec_bp_sum = hn::Set(d, bp[0] + bp[1] + bp[2]);
    const auto vec_bp12_sum = hn::Set(d, bp[1] + bp[2]);

    // Process N consecutive rows simultaneously
    for (std::size_t j = 0; j + N <= height; j += N) {
        alignas(16) std::int32_t base_indices[4];
        for (std::size_t r = 0; r < N; ++r) {
            base_indices[r] = static_cast<std::int32_t>((j + r) * width);
        }
        auto base_idx = hn::Load(di, base_indices);

        if (width > 0) {
            // First column unchanged.
        }

        if (width > 1) {
            auto idx = hn::Add(base_idx, hn::Set(di, 1));
            auto curr = hn::GatherIndex(d, data, idx);
            auto idx_prev = base_idx;
            auto prev0 = hn::GatherIndex(d, data, idx_prev);
            auto result = hn::MulAdd(vec_B, curr, hn::Mul(vec_bp_sum, prev0));
            hn::ScatterIndex(result, d, data, idx);
        }

        if (width > 2) {
            auto idx = hn::Add(base_idx, hn::Set(di, 2));
            auto curr = hn::GatherIndex(d, data, idx);
            auto prev1 =
                hn::GatherIndex(d, data, hn::Add(base_idx, hn::Set(di, 1)));
            auto prev0 = hn::GatherIndex(d, data, base_idx);
            auto result = hn::MulAdd(
                vec_B, curr,
                hn::MulAdd(vec_bp0, prev1, hn::Mul(vec_bp12_sum, prev0)));
            hn::ScatterIndex(result, d, data, idx);
        }

        for (std::size_t i = 3; i < width; ++i) {
            auto offset = hn::Set(di, static_cast<std::int32_t>(i));
            auto idx = hn::Add(base_idx, offset);
            auto curr = hn::GatherIndex(d, data, idx);
            auto prev1 = hn::GatherIndex(
                d, data,
                hn::Add(base_idx,
                        hn::Set(di, static_cast<std::int32_t>(i - 1))));
            auto prev2 = hn::GatherIndex(
                d, data,
                hn::Add(base_idx,
                        hn::Set(di, static_cast<std::int32_t>(i - 2))));
            auto prev3 = hn::GatherIndex(
                d, data,
                hn::Add(base_idx,
                        hn::Set(di, static_cast<std::int32_t>(i - 3))));
            auto result =
                hn::MulAdd(vec_B, curr,
                           hn::MulAdd(vec_bp0, prev1,
                                      hn::MulAdd(vec_bp1, prev2,
                                                 hn::Mul(vec_bp2, prev3))));
            hn::ScatterIndex(result, d, data, idx);
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
    const auto vec_bp0 = hn::Set(d, bp[0]);
    const auto vec_bp1 = hn::Set(d, bp[1]);
    const auto vec_bp2 = hn::Set(d, bp[2]);
    const auto vec_bp_sum = hn::Set(d, bp[0] + bp[1] + bp[2]);
    const auto vec_bp12_sum = hn::Set(d, bp[1] + bp[2]);

    // Process N consecutive rows simultaneously
    for (std::size_t j = 0; j + N <= height; j += N) {
        alignas(16) std::int32_t base_indices[4];
        for (std::size_t r = 0; r < N; ++r) {
            base_indices[r] = static_cast<std::int32_t>((j + r) * width);
        }
        auto base_idx = hn::Load(di, base_indices);

        if (width > 0) {
            // Last column unchanged.
        }

        if (width > 1) {
            auto idx = hn::Add(
                base_idx, hn::Set(di, static_cast<std::int32_t>(width - 2)));
            auto curr = hn::GatherIndex(d, data, idx);
            auto next0 = hn::GatherIndex(
                d, data,
                hn::Add(base_idx,
                        hn::Set(di, static_cast<std::int32_t>(width - 1))));
            auto result = hn::MulAdd(vec_B, curr, hn::Mul(vec_bp_sum, next0));
            hn::ScatterIndex(result, d, data, idx);
        }

        if (width > 2) {
            auto idx = hn::Add(
                base_idx, hn::Set(di, static_cast<std::int32_t>(width - 3)));
            auto curr = hn::GatherIndex(d, data, idx);
            auto next1 = hn::GatherIndex(
                d, data,
                hn::Add(base_idx,
                        hn::Set(di, static_cast<std::int32_t>(width - 2))));
            auto next0 = hn::GatherIndex(
                d, data,
                hn::Add(base_idx,
                        hn::Set(di, static_cast<std::int32_t>(width - 1))));
            auto result = hn::MulAdd(
                vec_B, curr,
                hn::MulAdd(vec_bp0, next1, hn::Mul(vec_bp12_sum, next0)));
            hn::ScatterIndex(result, d, data, idx);
        }

        if (width > 3) {
            for (std::size_t i = width - 4;; --i) {
                auto idx = hn::Add(base_idx,
                                   hn::Set(di, static_cast<std::int32_t>(i)));
                auto curr = hn::GatherIndex(d, data, idx);
                auto next1 = hn::GatherIndex(
                    d, data,
                    hn::Add(base_idx,
                            hn::Set(di, static_cast<std::int32_t>(i + 1))));
                auto next2 = hn::GatherIndex(
                    d, data,
                    hn::Add(base_idx,
                            hn::Set(di, static_cast<std::int32_t>(i + 2))));
                auto next3 = hn::GatherIndex(
                    d, data,
                    hn::Add(base_idx,
                            hn::Set(di, static_cast<std::int32_t>(i + 3))));
                auto result = hn::MulAdd(
                    vec_B, curr,
                    hn::MulAdd(
                        vec_bp0, next1,
                        hn::MulAdd(vec_bp1, next2, hn::Mul(vec_bp2, next3))));
                hn::ScatterIndex(result, d, data, idx);

                if (i == 0)
                    break; // Exit before underflow
            }
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
