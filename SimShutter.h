#pragma once

#include "SimHub.h"

#include "DeviceBase.h"

#include <atomic>
#include <string>
#include <utility>

class SimShutter : public CShutterBase<SimShutter> {
    std::string name_;
    bool initialized_ = false;

    // Shutter state
    std::atomic<bool> isOpen_{false};

  public:
    explicit SimShutter(std::string name) : name_(std::move(name)) {}

    ~SimShutter() { Shutdown(); }

    bool Busy() final {
        // TODO: Could introduce a small delay here
        return false;
    }

    void GetName(char *name) const final {
        CDeviceUtils::CopyLimitedString(name, name_.c_str());
    }

    int Fire(double /*deltaT*/) final { return DEVICE_UNSUPPORTED_COMMAND; }

    int Initialize() final {
        isOpen_ = false;

        int ret = CreateIntegerProperty(
            MM::g_Keyword_State, 0, false,
            new MM::ActionLambda(
                [this](MM::PropertyBase *pProp, MM::ActionType eAct) {
                    if (eAct == MM::BeforeGet) {
                        pProp->Set(isOpen_ ? 1L : 0L);
                    } else if (eAct == MM::AfterSet) {
                        long v{};
                        pProp->Get(v);
                        return SetOpen(v != 0);
                    }
                    return DEVICE_OK;
                }));
        if (ret != DEVICE_OK)
            return ret;
        ret = AddAllowedValue(MM::g_Keyword_State, "0");
        if (ret != DEVICE_OK)
            return ret;
        ret = AddAllowedValue(MM::g_Keyword_State, "1");
        if (ret != DEVICE_OK)
            return ret;

        auto *hub = static_cast<SimHub *>(GetParentHub());
        hub->SetGetShutterOpenFunction([this] { return isOpen_.load(); });
        initialized_ = true;
        return DEVICE_OK;
    }

    int Shutdown() final {
        isOpen_ = false;
        if (initialized_) {
            auto *hub = static_cast<SimHub *>(GetParentHub());
            hub->SetGetShutterOpenFunction([] { return true; });
            initialized_ = false;
        }
        return DEVICE_OK;
    }

    int SetOpen(bool open = true) final {
        isOpen_ = open;
        return DEVICE_OK;
    }

    int GetOpen(bool &open) final {
        open = isOpen_;
        return DEVICE_OK;
    }
};
