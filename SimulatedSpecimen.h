#include <blend2d.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

template <typename T> class SimulatedSpecimen {
    struct Filament {
        double x0, y0, x1, y1;
    };

    std::vector<Filament> filaments_;

    static bool PointInRect(double x, double y, double left, double top,
                            double right, double bottom) {
        return x >= left && x <= right && y >= top && y <= bottom;
    }

    static bool LineSegmentsIntersect(const Filament &s0, const Filament &s1) {
        const std::array<double, 2> d0{s0.x1 - s0.x0, s0.y1 - s0.y0};
        const std::array<double, 2> d1{s1.x1 - s1.x0, s1.y1 - s1.y0};
        const std::array<double, 2> d01{s1.x0 - s0.x0, s1.y0 - s0.y0};
        const double wedge01 = d0[0] * d1[1] - d0[1] * d1[0];
        if (std::fabs(wedge01) < 1e-12) { // Parallel
            return false;
        }
        // Solve for parameter values at intersecting point:
        const double t0 = (d1[0] * d01[1] - d1[1] * d01[0]) / wedge01;
        const double t1 = (d0[0] * d01[1] - d0[1] * d01[0]) / wedge01;
        // Intersection lies within segments:
        return t0 >= 0.0 && t0 <= 1.0 && t1 >= 0.0 && t1 <= 1.0;
    }

    std::vector<Filament> PrunedFilaments(double left, double top,
                                          double right, double bottom) {
        const Filament leftSeg{left, top, left, bottom};
        const Filament topSeg{left, top, right, top};
        const Filament rightSeg{right, top, right, bottom};
        const Filament bottomSeg{left, bottom, right, bottom};
        std::vector<Filament> ret;
        std::copy_if(
            filaments_.begin(), filaments_.end(), std::back_inserter(ret),
            [&](const Filament &f) {
                return PointInRect(f.x0, f.x1, left, top, right, bottom) ||
                       PointInRect(f.x1, f.y1, left, top, right, bottom) ||
                       LineSegmentsIntersect(f, leftSeg) ||
                       LineSegmentsIntersect(f, topSeg) ||
                       LineSegmentsIntersect(f, rightSeg) ||
                       LineSegmentsIntersect(f, bottomSeg);
            });
        return ret;
    }

  public:
    explicit SimulatedSpecimen() {
        using std::cos;
        using std::sin;
        std::mt19937 rng;
        std::normal_distribution<> xy0Distrib(0.0, 1000.0);
        std::uniform_real_distribution<> thetaDistrib(
            0.0, 2.0 * 3.14159265358979323846);
        std::exponential_distribution<> lenDistrib(1e-3);
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

    void Draw(T *buffer, double x_um, double y_um, double z_um,
              std::size_t width, std::size_t height, double um_per_px) {
        const double left = x_um;
        const double top = y_um;
        const double right = x_um + width * um_per_px;
        const double bottom = y_um + height * um_per_px;
        // TODO Do we really need to prune? Might be unnecessary.
        const auto filaments = PrunedFilaments(left, top, right, bottom);

        BLImage img(width, height, BL_FORMAT_XRGB32);
        BLContext ctx(img);
        ctx.clearAll();

        ctx.scale(1.0 / um_per_px);
        ctx.translate(x_um, -y_um);

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
        const auto stride = data.stride / sizeof(std::uint32_t);
        const auto start = stride >= 0 ? 0 : (height - 1) * (-stride);
        for (std::intptr_t j = 0; j < std::intptr_t(height); ++j) {
            for (std::intptr_t i = 0; i < std::intptr_t(width); ++i) {
                const auto p = pix[start + i + j * stride];
                buffer[i + j * width] = (p >> 8) & 0xff;
            }
        }

        // TODO Use float[] intermediate for:
        // TODO Defocus (gaussian approx)
        (void)z_um;
        // TODO Intensity from exposure (and dark offset)
        // TODO Add shot noise and read noise
        // TODO Convert to u8 or u16
    }
};
