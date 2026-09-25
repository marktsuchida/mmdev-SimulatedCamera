// Copyright 2025-2026 Board of Regents of the University of Wisconsin System
// SPDX-License-Identifier: BSD-2-Clause

#include "SimCam.h"
#include "SimFocus.h"
#include "SimHub.h"
#include "SimObjectiveTurret.h"
#include "SimShutter.h"
#include "SimXY.h"

#include "DeviceBase.h"
#include "ModuleInterface.h"

MODULE_API void InitializeModuleData() {
    RegisterDevice("SimHub", MM::HubDevice, "Hub for simulated camera");
}

MODULE_API MM::Device *CreateDevice(const char *name) {
    if (!name) {
        return nullptr;
    }
    const std::string n(name);
    if (n == "SimHub") {
        return new SimHub("SimHub");
    }
    if (n == "SimCam") {
        return new SimCam("SimCam");
    }
    if (n == "SimFocus") {
        return new SimFocus<AsyncProcessModel<1>>("SimFocus");
    }
    if (n == "SimXY") {
        return new SimXY<AsyncProcessModel<2>>("SimXY");
    }
    if (n == "SimObjectiveTurret") {
        return new SimObjectiveTurret("SimObjectiveTurret");
    }
    if (n == "SimShutter") {
        return new SimShutter("SimShutter");
    }
    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device *device) { delete device; }