#pragma once

#include "SimHub.h"
#include "DeviceBase.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>

class SimObjectiveTurret : public CStateDeviceBase<SimObjectiveTurret> {

    // {magnification, NA} pairs.
    static constexpr std::array<std::pair<int, double>, 2> objectives_ = {{
        {4, 0.13},
        {60, 0.95},
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
            const auto &[magnification, na] = objectives_[i];
            char label[32];
            std::snprintf(label, sizeof(label), "%dx%gNA", magnification, na);
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
        hub->SetGetSpecimenMagnificationFunction([this] {
            return objectives_[static_cast<std::size_t>(state_)].first;
        });
        hub->SetGetSpecimenNAFunction([this] {
            return objectives_[static_cast<std::size_t>(state_)].second;
        });

        initialized_ = true;
        return DEVICE_OK;
    }

    int Shutdown() final {
        if (initialized_) {
            auto *hub = static_cast<SimHub *>(this->GetParentHub());
            hub->SetGetSpecimenMagnificationFunction([] { return 1.0; });
            hub->SetGetSpecimenNAFunction([] { return 1.0; });
            initialized_ = false;
        }
        return DEVICE_OK;
    }

    void GetName(char* name) const final {
        CDeviceUtils::CopyLimitedString(name, name_.c_str());
    }

    bool Busy() {return busy_;};

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
    bool busy_ = false;
    bool initialized_ = false;
    std::string name_;
    long state_ = 0;
};
