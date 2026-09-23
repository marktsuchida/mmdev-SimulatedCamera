#include "SimCam.h"
#include "SimFocus.h"
#include "SimHub.h"
#include "SimShutter.h"
#include "SimXY.h"

#include "DeviceBase.h"
#include "ModuleInterface.h"

MODULE_API void InitializeModuleData() {
    RegisterDevice("SimHub", MM::HubDevice, "Hub for simulated camera");
    RegisterDevice("SimShutter", MM::ShutterDevice, "Simulated shutter");
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
    if (n == "SimShutter") {
        return new SimShutter("SimShutter");
    }
    return nullptr;
}

MODULE_API void DeleteDevice(MM::Device *device) { delete device; }