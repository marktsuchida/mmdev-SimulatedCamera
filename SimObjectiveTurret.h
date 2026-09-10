#pragma once

#include "SimHub.h"
#include "DeviceBase.h"

class SimObjectiveTurret : public CStateDeviceBase<SimObjectiveTurret> {

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
        SetPositionLabel(0, "1x0.3NA");
        SetPositionLabel(1, "4x1.0NA");

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
        hub->SetGetSpecimenUmPerPxFunction([this] {
            switch (state_) {
                default: return 10.0; break;
                case 1: return 2.5; break;
            }
        });
        hub->SetGetSpecimenNAFunction([this] {
            switch (state_) {
                default: return 0.3; break;
                case 1: return 1.0; break;
            }
        });

        initialized_ = true;
        return DEVICE_OK;
    }

    int Shutdown() final {
        if (initialized_) {
            auto *hub = static_cast<SimHub *>(this->GetParentHub());
            hub->SetGetSpecimenUmPerPxFunction([] { return 1.0; });
            hub->SetGetSpecimenNAFunction([] { return 1.0; });
            initialized_ = false;
        }
        return DEVICE_OK;
    }

    void GetName(char* name) const final {
        CDeviceUtils::CopyLimitedString(name, name_.c_str());
    }

    bool Busy() {return busy_;};

    unsigned long GetNumberOfPositions() const {return numPos_;}

    int OnState(MM::PropertyBase* pProp, MM::ActionType eAct) {
        if (eAct == MM::BeforeGet) {
            pProp->Set(state_);
        } else if (eAct == MM::AfterSet) {
            long newState;
            pProp->Get(newState);
            if (newState >= 0 && newState < numPos_) {
                state_ = newState;
            } else {
                pProp->Set(state_);
                return DEVICE_INVALID_PROPERTY_VALUE;
            }
        }
        return DEVICE_OK;
    }

private:
    long numPos_ = 2;
    bool busy_ = false;
    bool initialized_ = false;
    std::string name_;
    long state_ = 0;
};