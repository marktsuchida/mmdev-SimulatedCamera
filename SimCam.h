#pragma once

#include "SimHub.h"
#include "SimulatedSpecimen.h"

#include "DeviceBase.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

// In C++17 can make static constexpr member of SimCam
constexpr double EXPOSURE_MS_MIN = 0.001;
constexpr double EXPOSURE_MS_MAX = 10000.0;

class SimCam : public CCameraBase<SimCam> {
    std::string name_;

    static constexpr unsigned sensorWidth_ = 512;
    static constexpr unsigned sensorHeight_ = 512;

    SimulatedSpecimen<std::uint16_t> specimen_;

    // Camera state
    double exposure_ms_ = 100.0;
    unsigned roiX_ = 0;
    unsigned roiY_ = 0;
    unsigned roiWidth_ = sensorWidth_;
    unsigned roiHeight_ = sensorHeight_;

    // Last snap (always equal to ROI width/height)
    std::unique_ptr<std::uint16_t[]> snapBuffer_;

    // Sequence acquisition state
    std::thread seqThread_;
    std::mutex seqMutex_;
    std::condition_variable seqStopCV_;
    bool seqStopRequested_ = false;

  public:
    explicit SimCam(std::string name) : name_(std::move(name)) {}

    void GetName(char *buf) const final {
        CDeviceUtils::CopyLimitedString(buf, name_.c_str());
    }

    int Initialize() final { return DEVICE_OK; }
    int Shutdown() final { return DEVICE_OK; }
    bool Busy() final { return false; }

    unsigned GetImageWidth() const final { return roiWidth_; }
    unsigned GetImageHeight() const final { return roiHeight_; }
    unsigned GetImageBytesPerPixel() const final { return 2; /* for now */ }
    long GetImageBufferSize() const final {
        return GetImageBytesPerPixel() * GetImageWidth() * GetImageHeight();
    }

    unsigned GetBitDepth() const final { return 16; }

    int SnapImage() final {
        const auto startTime = std::chrono::steady_clock::now();

        auto *hub = static_cast<SimHub *>(GetParentHub());
        const auto z = hub->GetFocusUm();
        const auto xy = hub->GetXYUm();

        const std::size_t nPixels = roiWidth_ * roiHeight_;
        snapBuffer_ =
            std::unique_ptr<std::uint16_t[]>(new std::uint16_t[nPixels]);

        constexpr double umPerPx = 1.0; // TODO Objective/mag
        const double x = xy.first + umPerPx * double(roiX_);
        const double y = xy.second + umPerPx * double(roiY_);
        // TODO: Intensity could also change with objective mag and NA
        const double intensity = GetExposure() * GetBinning() * GetBinning();
        specimen_.Draw(snapBuffer_.get(), x, y, z, roiWidth_, roiHeight_,
                       umPerPx, intensity);

        const auto exposure_us = std::llround(1000.0 * GetExposure());
        std::this_thread::sleep_until(startTime +
                                      std::chrono::microseconds(exposure_us));

        return DEVICE_OK;
    }

    const unsigned char *GetImageBuffer() final {
        return reinterpret_cast<unsigned char *>(snapBuffer_.get());
    }

    int GetBinning() const final { return 1; }
    int SetBinning(int binSize) final {
        if (binSize != 1) {
            return DEVICE_INVALID_INPUT_PARAM;
        }
        return DEVICE_OK;
    }

    double GetExposure() const final { return exposure_ms_; }
    void SetExposure(double exposure_ms) final {
        exposure_ms_ =
            std::min(EXPOSURE_MS_MAX, std::max(EXPOSURE_MS_MIN, exposure_ms));
    }

    int GetROI(unsigned &x, unsigned &y, unsigned &width,
               unsigned &height) final {
        x = roiX_;
        y = roiY_;
        width = roiWidth_;
        height = roiHeight_;
        return DEVICE_OK;
    }

    int SetROI(unsigned x, unsigned y, unsigned width, unsigned height) final {
        if (x + width > sensorWidth_ || y + height > sensorHeight_) {
            return DEVICE_INVALID_INPUT_PARAM;
        }
        roiX_ = x;
        roiY_ = y;
        roiWidth_ = width;
        roiHeight_ = height;
        return DEVICE_OK;
    }

    int ClearROI() final { return SetROI(0, 0, sensorWidth_, sensorHeight_); }

    int StartSequenceAcquisition(long numImages, double interval_ms,
                                 bool stopOnOverflow) final {
        (void)interval_ms;
        (void)stopOnOverflow;

        if (IsCapturing()) {
            return DEVICE_CAMERA_BUSY_ACQUIRING;
        }

        if (numImages < 1) {
            return DEVICE_INVALID_INPUT_PARAM;
        }
        const std::size_t nImages = static_cast<std::size_t>(numImages);

        seqStopRequested_ = false;
        seqThread_ = std::thread([this, nImages] {
            const auto interval =
                std::chrono::microseconds(std::llround(exposure_ms_ * 1000.0));
            for (std::size_t i = 0; i < nImages; ++i) {
                {
                    std::unique_lock<std::mutex> lock(seqMutex_);
                    if (seqStopCV_.wait_for(lock, interval, [&] {
                            return seqStopRequested_;
                        })) {
                        break;
                    }
                }

                // This is _not_ how to implement real cameras, but for this
                // simulation we can implement in terms of snaps.
                SnapImage();
                GetCoreCallback()->InsertImage(
                    this, GetImageBuffer(), GetImageWidth(), GetImageHeight(),
                    GetImageBytesPerPixel());
            }
        });
        return DEVICE_OK;
    }

    int StopSequenceAcquisition() final {
        if (IsCapturing()) {
            {
                std::lock_guard<std::mutex> lock(seqMutex_);
                seqStopRequested_ = true;
            }
            seqStopCV_.notify_one();
            seqThread_.join();
        }
        return DEVICE_OK;
    }

    bool IsCapturing() final { return seqThread_.joinable(); }

    int IsExposureSequenceable(bool &yesno) const final {
        yesno = false;
        return DEVICE_OK;
    }
};
