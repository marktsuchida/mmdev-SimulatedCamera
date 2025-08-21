#pragma once

#include "DelayedNotifier.h"
#include "ProcessModel.h"
#include "SimHub.h"

#include "DeviceBase.h"

#include <array>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>

template <typename ProcModel>
class SimFocus : public CStageBase<SimFocus<ProcModel>> {
    // The process model operates in steps (only set to integers; round upon
    // readout).
    static constexpr double umPerStep_ = 0.1;

    std::string name_;
    ProcModel model_;
    DelayedNotifier delayer_;

    std::mutex notificationMut_;
    bool notificationsEnabled_ = false;

  public:
    explicit SimFocus(std::string name)
        : name_(std::move(name)), model_([this](std::array<double, 1> pv) {
              const long ipv = std::lround(pv[0]);
              this->LogMessage(("PV = " + std::to_string(ipv)).c_str(), true);
              {
                  std::lock_guard<std::mutex> lock(notificationMut_);
                  if (!notificationsEnabled_) {
                      return;
                  }
              }
              delayer_.Schedule([this, ipv] {
                  this->LogMessage(
                      ("Notifying: PV = " + std::to_string(ipv)).c_str(),
                      true);
                  this->OnStagePositionChanged(umPerStep_ * ipv);
              });
          }) {
        // Adjust default for stage-like velocity (100 um/s).
        model_.ReciprocalSlewRateSeconds(0.01 * umPerStep_);
    }

    int Initialize() final {
        this->CreateFloatProperty("UmPerStep", umPerStep_, true);

        this->CreateStringProperty(
            "NotificationsEnabled", notificationsEnabled_ ? "Yes" : "No",
            false,
            new MM::ActionLambda(
                [this](MM::PropertyBase *pProp, MM::ActionType eAct) {
                    if (eAct == MM::BeforeGet) {
                        pProp->Set(notificationsEnabled_ ? "Yes" : "No");
                    } else if (eAct == MM::AfterSet) {
                        std::string value;
                        pProp->Get(value);
                        notificationsEnabled_ = (value == "Yes");
                    }
                    return DEVICE_OK;
                }));
        this->AddAllowedValue("NotificationsEnabled", "No");
        this->AddAllowedValue("NotificationsEnabled", "Yes");

        this->CreateIntegerProperty(
            "ExternallySetSteps", static_cast<long>(model_.Setpoint()[0]),
            false,
            new MM::ActionLambda(
                [this](MM::PropertyBase *pProp, MM::ActionType eAct) {
                    if (eAct == MM::BeforeGet) {
                        // Keep last-set value
                    } else if (eAct == MM::AfterSet) {
                        long value{};
                        pProp->Get(value);
                        this->LogMessage(
                            ("sp = " + std::to_string(value) + " (external)")
                                .c_str(),
                            true);
                        model_.Setpoint({double(value)});
                    }
                    return DEVICE_OK;
                }));

        if (ProcModel::isAsync) {
            this->CreateFloatProperty(
                "SlewTimePerStep_s", model_.ReciprocalSlewRateSeconds(), false,
                new MM::ActionLambda(
                    [this](MM::PropertyBase *pProp, MM::ActionType eAct) {
                        if (eAct == MM::BeforeGet) {
                            pProp->Set(model_.ReciprocalSlewRateSeconds());
                        } else if (eAct == MM::AfterSet) {
                            double seconds{};
                            pProp->Get(seconds);
                            model_.ReciprocalSlewRateSeconds(seconds);
                        }
                        return DEVICE_OK;
                    }));
            this->SetPropertyLimits("SlewTimePerStep_s", 0.0001, 10.0);

            this->CreateFloatProperty(
                "UpdateInterval_s", model_.UpdateIntervalSeconds(), false,
                new MM::ActionLambda(
                    [this](MM::PropertyBase *pProp, MM::ActionType eAct) {
                        if (eAct == MM::BeforeGet) {
                            pProp->Set(model_.UpdateIntervalSeconds());
                        } else if (eAct == MM::AfterSet) {
                            double seconds{};
                            pProp->Get(seconds);
                            model_.UpdateIntervalSeconds(seconds);
                        }
                        return DEVICE_OK;
                    }));
            this->SetPropertyLimits("UpdateInterval_s", 0.0001, 10.0);

            using std::chrono::duration_cast;
            using std::chrono::microseconds;
            using FPSeconds = std::chrono::duration<double>;
            this->CreateFloatProperty(
                "NotificationDelay_s",
                duration_cast<FPSeconds>(delayer_.Delay()).count(), false,
                new MM::ActionLambda([this](MM::PropertyBase *pProp,
                                            MM::ActionType eAct) {
                    if (eAct == MM::BeforeGet) {
                        pProp->Set(duration_cast<FPSeconds>(delayer_.Delay())
                                       .count());
                    } else if (eAct == MM::AfterSet) {
                        double seconds{};
                        pProp->Get(seconds);
                        const auto delay = FPSeconds{seconds};
                        delayer_.Delay(duration_cast<microseconds>(delay));
                    }
                    return DEVICE_OK;
                }));
            this->SetPropertyLimits("NotificationDelay_s", 0.0, 1.0);
        }

        auto *hub = static_cast<SimHub *>(this->GetParentHub());
        hub->SetGetFocusUmFunction([this] {
            double z_um;
            (void)this->GetPositionUm(z_um);
            return z_um;
        });

        return DEVICE_OK;
    }

    int Shutdown() final {
        auto *hub = static_cast<SimHub *>(this->GetParentHub());
        hub->SetGetXYUmFunction([] { return std::make_pair(0.0, 0.0); });
        model_.Halt();
        delayer_.CancelAll();
        return DEVICE_OK;
    }

    bool Busy() final { return model_.IsSlewing(); }

    void GetName(char *name) const final {
        CDeviceUtils::CopyLimitedString(name, name_.c_str());
    }

    int SetPositionUm(double um) final {
        const long steps = std::lround(um / umPerStep_);
        return SetPositionSteps(steps);
    }

    int Stop() final {
        model_.Halt();
        return DEVICE_OK;
    }

    int GetPositionUm(double &um) final {
        long steps{};
        int ret = GetPositionSteps(steps);
        if (ret != DEVICE_OK) {
            return ret;
        }
        um = umPerStep_ * steps;
        return DEVICE_OK;
    }

    int SetPositionSteps(long steps) final {
        this->LogMessage(("sp = " + std::to_string(steps)).c_str(), true);
        model_.Setpoint({double(steps)});
        return DEVICE_OK;
    }

    int GetPositionSteps(long &steps) final {
        steps = std::lround(model_.ProcessVariable()[0]);
        return DEVICE_OK;
    }

    int SetOrigin() final { return DEVICE_UNSUPPORTED_COMMAND; }

    int GetLimits(double &lo, double &hi) final {
        // Conservatively keep steps within 32-bit range (we want exact integer
        // steps to be preserved by the double-based process model).
        lo = -2147483648 * umPerStep_;
        hi = +2147483647 * umPerStep_;
        return DEVICE_OK;
    }

    int IsStageSequenceable(bool &flag) const final {
        flag = false;
        return DEVICE_OK;
    }

    bool IsContinuousFocusDrive() const final { return false; }
};
