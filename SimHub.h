#pragma once

#include "DeviceBase.h"

#include <functional>
#include <string>
#include <utility>

class SimHub : public HubBase<SimHub> {
    std::string name_;

    std::function<double()> getFocusUmFunc_ = [] { return 0.0; };
    std::function<std::pair<double, double>()> getXYUmFunc_ = [] {
        return std::make_pair(0.0, 0.0);
    };

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
        getFocusUmFunc_ = f;
    }

    template <typename F> void SetGetXYUmFunction(F f) { getXYUmFunc_ = f; }

    double GetFocusUm() { return getFocusUmFunc_(); }
    std::pair<double, double> GetXYUm() { return getXYUmFunc_(); }
};