#pragma once

#include "SimHub.h"

#include "DeviceBase.h"

class SimShutter : public CShutterBase<SimShutter> {
    std::string name_;

    // Shutter state
    bool isOpen_ = false;

public:
    explicit SimShutter(std::string name) : name_(std::move(name)) {}

    bool Busy() final {
        // TODO: Could introduce a small delay here
        return false;
    }

    void GetName(char* name) const {
        CDeviceUtils::CopyLimitedString(name, name_.c_str());
    }
    
    int Fire(double /*deltaT*/) final {
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int Initialize() final {
        isOpen_ = false;
        auto *hub = static_cast<SimHub *>(GetParentHub());
        if (hub)
            hub->SetGetShutterOpenFunction([this] { return isOpen_; });
        return DEVICE_OK;
    }

    int Shutdown() final {
        isOpen_ = false;
        auto *hub = static_cast<SimHub *>(GetParentHub());
        if (hub)
            hub->SetGetShutterOpenFunction([] { return true; });
        return DEVICE_OK;
    }

    int SetOpen(bool open = true) {
        isOpen_ = open;
        return DEVICE_OK;
    }

    int GetOpen(bool& open) {
        open = isOpen_;
        return DEVICE_OK;
    }
};
