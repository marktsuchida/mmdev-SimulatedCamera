#pragma once

#include "DeviceBase.h"

#include <functional>
#include <string>
#include <utility>

class SimHub : public HubBase<SimHub> {
    std::string name_;

    std::function<double()> getSpecimenFocusUmFunc_ = [] { return 0.0; };
    std::function<std::pair<double, double>()> getSpecimenXYUmFunc_ = [] {
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

    template <typename F> void SetGetSpecimenFocusUmFunction(F f) {
        getSpecimenFocusUmFunc_ = f;
    }

    template <typename F> void SetGetSpecimenXYUmFunction(F f) {
        getSpecimenXYUmFunc_ = f;
    }

    double GetSpecimenFocusUm() { return getSpecimenFocusUmFunc_(); }
    std::pair<double, double> GetSpecimenXYUm() {
        return getSpecimenXYUmFunc_();
    }
};