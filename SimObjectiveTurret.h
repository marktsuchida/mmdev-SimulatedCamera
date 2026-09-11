#pragma once

#include "SimHub.h"
#include "DeviceBase.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>

class SimObjectiveTurret : public CStateDeviceBase<SimObjectiveTurret> {

    struct Objective {
        int magnification;
        double na;
        const char *medium = nullptr;
    };

    static constexpr std::array<Objective, 6> objectives_ = {{
        {10, 0.3},
        {20, 0.7},
        {40, 0.75},
        {40, 0.3, "Oil"},
        {60, 1.4, "Oil"},
        {100, 1.4, "Oil"},
    }};

public:
    SimObjectiveTurret(std::string name): name_(std::move(name)) {
        InitializeDefaultErrorMessages();
    }

    ~SimObjectiveTurret() {
        Shutdown();
    }

    int Initialize() final {
        int ret{};

        // create default positions and labels
        for (std::size_t i = 0; i < objectives_.size(); ++i) {
            const auto &obj = objectives_[i];
            char label[32];
            if (obj.medium) {
                std::snprintf(label, sizeof(label), "%dx %.2fNA %s",
                              obj.magnification, obj.na, obj.medium);
            } else {
                std::snprintf(label, sizeof(label), "%dx %.2fNA",
                              obj.magnification, obj.na);
            }
            SetPositionLabel(static_cast<long>(i), label);
        }

        // State
        // -----
        CPropertyAction* pAct = new CPropertyAction (this, &SimObjectiveTurret::OnState);
        ret = CreateIntegerProperty(MM::g_Keyword_State, 0, false, pAct);
        if (ret != DEVICE_OK)
            return ret;

        // Label
        // -----
        pAct = new CPropertyAction (this, &CStateBase::OnLabel);
        ret = CreateStringProperty(MM::g_Keyword_Label, "", false, pAct);
        if (ret != DEVICE_OK)
            return ret;

        auto *hub = static_cast<SimHub *>(this->GetParentHub());
        hub->SetGetMagnificationFunction([this] {
            return objectives_[static_cast<std::size_t>(state_)].magnification;
        });
        hub->SetGetNAFunction([this] {
            return objectives_[static_cast<std::size_t>(state_)].na;
        });

        initialized_ = true;
        return DEVICE_OK;
    }

    int Shutdown() final {
        if (initialized_) {
            auto *hub = static_cast<SimHub *>(this->GetParentHub());
            hub->SetGetMagnificationFunction([] { return 1.0; });
            hub->SetGetNAFunction([] { return 1.0; });
            initialized_ = false;
        }
        return DEVICE_OK;
    }

    void GetName(char* name) const final {
        CDeviceUtils::CopyLimitedString(name, name_.c_str());
    }

    bool Busy() {return false;};

    unsigned long GetNumberOfPositions() const {return static_cast<long>(objectives_.size());}

    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct) {
        if (eAct == MM::BeforeGet) {
            pProp->Set(state_);
        } else if (eAct == MM::AfterSet) {
            long newState;
            pProp->Get(newState);
            if (newState >= 0 && newState < static_cast<long>(objectives_.size())) {
                state_ = newState;
            } else {
                pProp->Set(state_);
                return DEVICE_INVALID_PROPERTY_VALUE;
            }
        }
        return DEVICE_OK;
    }

private:
    bool initialized_ = false;
    std::string name_;
    long state_ = 0;
};
