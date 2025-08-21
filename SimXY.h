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
#include <utility>

template <typename ProcModel>
class SimXY : public CXYStageBase<SimXY<ProcModel>> {
    // The process model operates in steps (only set to integers; round upon
    // readout). X and Y use the same step size.
    static constexpr double umPerStep_ = 0.1;
    std::string name_;
    ProcModel model_;
    DelayedNotifier delayer_;

    std::mutex notificationMut_;
    bool notificationsEnabled_ = false;

  public:
    explicit SimXY(std::string name)
        : name_(std::move(name)), model_([this](std::array<double, 2> pv) {
              const long ix = std::lround(pv[0]);
              const long iy = std::lround(pv[1]);
              this->LogMessage(
                  ("PV = " + std::to_string(ix) + ", " + std::to_string(iy))
                      .c_str(),
                  true);
              {
                  std::lock_guard<std::mutex> lock(notificationMut_);
                  if (!notificationsEnabled_) {
                      return;
                  }
              }
              delayer_.Schedule([this, ix, iy] {
                  this->LogMessage(("Notifying: PV = " + std::to_string(ix) +
                                    ", " + std::to_string(iy))
                                       .c_str(),
                                   true);
                  this->OnXYStagePositionChanged(umPerStep_ * ix,
                                                 umPerStep_ * iy);
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
        hub->SetGetXYUmFunction([this] {
            double x_um, y_um;
            (void)this->GetPositionUm(x_um, y_um);
            return std::make_pair(x_um, y_um);
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

    int GetLimitsUm(double &xMin, double &xMax, double &yMin,
                    double &yMax) final {
        long xLo, xHi, yLo, yHi;
        int ret = GetStepLimits(xLo, xHi, yLo, yHi);
        if (ret != DEVICE_OK) {
            return ret;
        }
        xMin = xLo * umPerStep_;
        xMax = xHi * umPerStep_;
        yMin = yLo * umPerStep_;
        yMax = yHi * umPerStep_;
        return DEVICE_OK;
    }

    int SetPositionSteps(long x, long y) final {
        this->LogMessage(
            ("sp = " + std::to_string(x) + ", " + std::to_string(y)).c_str(),
            true);
        model_.Setpoint({double(x), double(y)});
        return DEVICE_OK;
    }

    int GetPositionSteps(long &x, long &y) final {
        const auto pv = model_.ProcessVariable();
        x = std::lround(pv[0]);
        y = std::lround(pv[1]);
        return DEVICE_OK;
    }

    int Home() final {
        // We could simulate homing, but it is not currently clear what
        // notifications mean during a home (application code should explicitly
        // read the position after a home). For now, pretend that the stage
        // doesn't support homing.
        return DEVICE_UNSUPPORTED_COMMAND;
    }

    int Stop() final {
        model_.Halt();
        return DEVICE_OK;
    }

    int SetOrigin() final { return DEVICE_UNSUPPORTED_COMMAND; }

    int GetStepLimits(long &xMin, long &xMax, long &yMin, long &yMax) final {
        // Conservatively keep steps within 32-bit range (we want exact integer
        // steps to be preserved by the double-based process model).
        xMin = yMin = -2147483648;
        xMax = yMax = +2147483647;
        return DEVICE_OK;
    }

    double GetStepSizeXUm() final { return umPerStep_; }

    double GetStepSizeYUm() final { return umPerStep_; }

    int IsXYStageSequenceable(bool &flag) const final {
        flag = false;
        return DEVICE_OK;
    }
};
