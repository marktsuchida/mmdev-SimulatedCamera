// Copyright 2025-2026 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "Detector.h"
#include "SimHub.h"
#include "Specimen.h"

#include "DeviceBase.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// In C++17 can make static constexpr member of SimCam
constexpr double EXPOSURE_MS_MIN = 0.001;
constexpr double EXPOSURE_MS_MAX = 10000.0;

class SimCam : public CCameraBase<SimCam> {
    std::string name_;

    static constexpr unsigned sensorWidth_ = 512;
    static constexpr unsigned sensorHeight_ = 512;

    static constexpr const char *modeFilaments_ = "Filaments";
    static constexpr const char *modeNuclei_ = "Nuclei";

    SimHub *hub_ = nullptr;

    FilamentsSpecimen filamentsSpecimen_;
    NucleiSpecimen nucleiSpecimen_;

    // Settings, which may be read by the sequence thread
    mutable std::mutex settingsMutex_;
    std::string mode_ = modeFilaments_;
    double exposure_ms_ = 100.0;
    unsigned roiX_ = 0;
    unsigned roiY_ = 0;
    unsigned roiWidth_ = sensorWidth_;
    unsigned roiHeight_ = sensorHeight_;

    struct FrameSettings {
        double exposure_ms;
        unsigned roiX, roiY, roiWidth, roiHeight;
        std::string mode;
    };

    // Last snap (always equal to ROI width/height of last frame)
    std::unique_ptr<std::uint16_t[]> snapBuffer_;

    std::vector<float> signal_;
    rnd::mt19937 rng_;

    // Sequence acquisition state
    std::thread seqThread_;
    std::mutex seqMutex_;
    std::condition_variable seqStopCV_;
    bool seqStopRequested_ = false;
    std::atomic<bool> capturing_{false};

  public:
    explicit SimCam(std::string name) : name_(std::move(name)) {}

    ~SimCam() { StopSequenceAcquisition(); }

    void GetName(char *buf) const final {
        CDeviceUtils::CopyLimitedString(buf, name_.c_str());
    }

    int Initialize() final {
        hub_ = static_cast<SimHub *>(GetParentHub());
        if (!hub_) {
            return DEVICE_COMM_HUB_MISSING;
        }

        int ret = CreateProperty(
            MM::g_Keyword_Exposure, std::to_string(exposure_ms_).c_str(),
            MM::Float, false,
            new MM::ActionLambda(
                [this](MM::PropertyBase *pProp, MM::ActionType eAct) {
                    if (eAct == MM::BeforeGet) {
                        pProp->Set(GetExposure());
                    } else if (eAct == MM::AfterSet) {
                        double e{};
                        pProp->Get(e);
                        SetExposure(e);
                    }
                    return DEVICE_OK;
                }));
        assert(ret == DEVICE_OK);
        ret = SetPropertyLimits(MM::g_Keyword_Exposure, 0.001, 10'000.0);
        assert(ret == DEVICE_OK);

        ret = CreateProperty(MM::g_Keyword_Binning, "1", MM::Integer, false);
        assert(ret == DEVICE_OK);
        ret = AddAllowedValue(MM::g_Keyword_Binning, "1");
        assert(ret == DEVICE_OK);

        ret = CreateProperty(
            "Mode", mode_.c_str(), MM::String, false,
            new MM::ActionLambda(
                [this](MM::PropertyBase *pProp, MM::ActionType eAct) {
                    if (eAct == MM::BeforeGet) {
                        std::lock_guard<std::mutex> lock(settingsMutex_);
                        pProp->Set(mode_.c_str());
                    } else if (eAct == MM::AfterSet) {
                        std::string value;
                        pProp->Get(value);
                        std::lock_guard<std::mutex> lock(settingsMutex_);
                        mode_ = value;
                    }
                    return DEVICE_OK;
                }));
        assert(ret == DEVICE_OK);
        ret = AddAllowedValue("Mode", modeFilaments_);
        assert(ret == DEVICE_OK);
        ret = AddAllowedValue("Mode", modeNuclei_);
        assert(ret == DEVICE_OK);
        (void)ret;

        return DEVICE_OK;
    }
    int Shutdown() final { return StopSequenceAcquisition(); }
    bool Busy() final { return false; }

    unsigned GetImageWidth() const final {
        std::lock_guard<std::mutex> lock(settingsMutex_);
        return roiWidth_;
    }
    unsigned GetImageHeight() const final {
        std::lock_guard<std::mutex> lock(settingsMutex_);
        return roiHeight_;
    }
    unsigned GetImageBytesPerPixel() const final { return 2; /* for now */ }
    long GetImageBufferSize() const final {
        return GetImageBytesPerPixel() * GetImageWidth() * GetImageHeight();
    }

    unsigned GetBitDepth() const final { return 16; }

    int SnapImage() final {
        if (IsCapturing()) {
            return DEVICE_CAMERA_BUSY_ACQUIRING;
        }
        const auto startTime = std::chrono::steady_clock::now();
        const FrameSettings settings = SnapshotSettings();
        GenerateFrame(settings);
        std::this_thread::sleep_until(startTime +
                                      ExposureDuration(settings.exposure_ms));
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

    double GetExposure() const final {
        std::lock_guard<std::mutex> lock(settingsMutex_);
        return exposure_ms_;
    }
    void SetExposure(double exposure_ms) final {
        std::lock_guard<std::mutex> lock(settingsMutex_);
        exposure_ms_ =
            std::min(EXPOSURE_MS_MAX, std::max(EXPOSURE_MS_MIN, exposure_ms));
    }

    int GetROI(unsigned &x, unsigned &y, unsigned &width,
               unsigned &height) final {
        std::lock_guard<std::mutex> lock(settingsMutex_);
        x = roiX_;
        y = roiY_;
        width = roiWidth_;
        height = roiHeight_;
        return DEVICE_OK;
    }

    int SetROI(unsigned x, unsigned y, unsigned width, unsigned height) final {
        if (IsCapturing()) {
            return DEVICE_CAMERA_BUSY_ACQUIRING;
        }
        if (width == 0 || height == 0) {
            return DEVICE_INVALID_INPUT_PARAM;
        }
        if (x + width > sensorWidth_ || y + height > sensorHeight_) {
            return DEVICE_INVALID_INPUT_PARAM;
        }
        std::lock_guard<std::mutex> lock(settingsMutex_);
        roiX_ = x;
        roiY_ = y;
        roiWidth_ = width;
        roiHeight_ = height;
        return DEVICE_OK;
    }

    int ClearROI() final { return SetROI(0, 0, sensorWidth_, sensorHeight_); }

    int StartSequenceAcquisition(double interval) final {
        return StartSequenceAcquisition(std::numeric_limits<long>::max(),
                                        interval, false);
    }

    int StartSequenceAcquisition(long numImages, double interval_ms,
                                 bool stopOnOverflow) final {
        (void)interval_ms;
        (void)stopOnOverflow;

        if (capturing_) {
            return DEVICE_CAMERA_BUSY_ACQUIRING;
        }
        if (numImages < 1) {
            return DEVICE_INVALID_INPUT_PARAM;
        }
        if (seqThread_.joinable()) {
            seqThread_.join();
        }
        {
            std::lock_guard<std::mutex> lock(seqMutex_);
            seqStopRequested_ = false;
        }
        capturing_ = true;

        const std::size_t nImages = static_cast<std::size_t>(numImages);
        seqThread_ = std::thread([this, nImages] {
            const int ret = RunSequence(nImages);
            GetCoreCallback()->AcqFinished(this, ret);
            capturing_ = false;
        });
        return DEVICE_OK;
    }

    int StopSequenceAcquisition() final {
        {
            std::lock_guard<std::mutex> lock(seqMutex_);
            seqStopRequested_ = true;
        }
        seqStopCV_.notify_one();
        if (seqThread_.joinable()) {
            seqThread_.join();
        }
        return DEVICE_OK;
    }

    bool IsCapturing() final { return capturing_; }

    int IsExposureSequenceable(bool &yesno) const final {
        yesno = false;
        return DEVICE_OK;
    }

  private:
    static std::chrono::steady_clock::duration
    ExposureDuration(double exposure_ms) {
        return std::chrono::round<std::chrono::steady_clock::duration>(
            std::chrono::duration<double, std::milli>(exposure_ms));
    }

    FrameSettings SnapshotSettings() const {
        std::lock_guard<std::mutex> lock(settingsMutex_);
        return {exposure_ms_, roiX_, roiY_, roiWidth_, roiHeight_, mode_};
    }

    int RunSequence(std::size_t nImages) {
        int ret = GetCoreCallback()->PrepareForAcq(this);
        if (ret != DEVICE_OK) {
            return ret;
        }
        for (std::size_t i = 0; i < nImages; ++i) {
            const auto startTime = std::chrono::steady_clock::now();
            const FrameSettings settings = SnapshotSettings();
            GenerateFrame(settings);
            {
                std::unique_lock<std::mutex> lock(seqMutex_);
                if (seqStopCV_.wait_until(
                        lock,
                        startTime + ExposureDuration(settings.exposure_ms),
                        [&] { return seqStopRequested_; })) {
                    break;
                }
            }
            ret = GetCoreCallback()->InsertImage(
                this, reinterpret_cast<unsigned char *>(snapBuffer_.get()),
                settings.roiWidth, settings.roiHeight,
                GetImageBytesPerPixel());
            if (ret != DEVICE_OK) {
                break;
            }
        }
        return ret;
    }

    void GenerateFrame(const FrameSettings &s) {
        const auto z = hub_->GetFocusUm();
        const auto xy = hub_->GetXYUm();

        const std::size_t nPixels = std::size_t(s.roiWidth) * s.roiHeight;
        snapBuffer_ =
            std::unique_ptr<std::uint16_t[]>(new std::uint16_t[nPixels]);

        const double magnification = hub_->GetMagnification();
        const double na = hub_->GetNA();
        // 10um is a reasonable size for a CMOS pixel side length
        // and it makes pixel configuration simple.
        const double umPerPx = 10.0 / magnification;
        // FOV center is -stagePosition (needed for tiles to align).
        const double fovCenterX = xy.first;
        const double fovCenterY = -xy.second;
        const double x = fovCenterX - umPerPx * (double(s.roiX) -
                                                 double(sensorWidth_) / 2.0);
        const double y = fovCenterY - umPerPx * (double(s.roiY) -
                                                 double(sensorHeight_) / 2.0);
        // Derive intensity using epi-illumination formula
        const double intensity = 2800.0 * s.exposure_ms * GetBinning() *
                                 GetBinning() * (na * na * na * na) /
                                 (magnification * magnification);
        const float *signal = nullptr;
        if (hub_->IsShutterOpen()) {
            signal_.resize(nPixels);
            if (s.mode == modeNuclei_) {
                nucleiSpecimen_.Draw(signal_.data(), x, y, z, s.roiWidth,
                                     s.roiHeight, umPerPx, na, intensity);
            } else {
                filamentsSpecimen_.Draw(signal_.data(), x, y, z, s.roiWidth,
                                        s.roiHeight, umPerPx, na, intensity);
            }
            signal = signal_.data();
        }
        // TODO: Make read noise and offset adjustable?
        constexpr float readNoise = 50.0f;
        constexpr float darkOffset = 100.0f;
        ReadOut(signal, snapBuffer_.get(), nPixels, readNoise, darkOffset,
                rng_);
    }
};
