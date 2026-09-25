// Copyright 2025-2026 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include "DeviceBase.h"

#include <functional>
#include <mutex>
#include <string>
#include <utility>

class SimHub : public HubBase<SimHub> {
    std::string name_;

    // Held while invoking the functions, so that a peripheral's Shutdown()
    // (which replaces its function) cannot return while the camera is still
    // calling into it.
    mutable std::mutex mut_;
    std::function<double()> getFocusUmFunc_ = [] { return 0.0; };
    std::function<std::pair<double, double>()> getXYUmFunc_ = [] {
        return std::make_pair(0.0, 0.0);
    };
    std::function<double()> getMagnificationFunc_ = [] { return 1.0; };
    std::function<double()> getNAFunc_ = [] { return 1.0; };
    std::function<bool()> getShutterOpenFunc_ = [] { return true; };

  public:
    explicit SimHub(std::string name) : name_(std::move(name)) {}

    void GetName(char *buf) const final {
        CDeviceUtils::CopyLimitedString(buf, name_.c_str());
    }
    int Initialize() final { return DEVICE_OK; }
    int Shutdown() final { return DEVICE_OK; }
    bool Busy() final { return false; }
    int DetectInstalledDevices() final;

    template <typename F> void SetGetFocusUmFunction(F f) {
        std::lock_guard<std::mutex> lock(mut_);
        getFocusUmFunc_ = std::move(f);
    }

    template <typename F> void SetGetXYUmFunction(F f) {
        std::lock_guard<std::mutex> lock(mut_);
        getXYUmFunc_ = std::move(f);
    }

    template <typename F> void SetGetMagnificationFunction(F f) {
        std::lock_guard<std::mutex> lock(mut_);
        getMagnificationFunc_ = std::move(f);
    }

    template <typename F> void SetGetNAFunction(F f) {
        std::lock_guard<std::mutex> lock(mut_);
        getNAFunc_ = std::move(f);
    }

    template <typename F> void SetGetShutterOpenFunction(F f) {
        std::lock_guard<std::mutex> lock(mut_);
        getShutterOpenFunc_ = std::move(f);
    }

    double GetFocusUm() {
        std::lock_guard<std::mutex> lock(mut_);
        return getFocusUmFunc_();
    }
    std::pair<double, double> GetXYUm() {
        std::lock_guard<std::mutex> lock(mut_);
        return getXYUmFunc_();
    }
    double GetMagnification() {
        std::lock_guard<std::mutex> lock(mut_);
        return getMagnificationFunc_();
    }
    double GetNA() {
        std::lock_guard<std::mutex> lock(mut_);
        return getNAFunc_();
    }
    bool IsShutterOpen() {
        std::lock_guard<std::mutex> lock(mut_);
        return getShutterOpenFunc_();
    }
};
