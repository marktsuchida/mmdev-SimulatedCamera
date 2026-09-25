#include "SimHub.h"

#include "SimCam.h"
#include "SimFocus.h"
#include "SimObjectiveTurret.h"
#include "SimShutter.h"
#include "SimXY.h"

int SimHub::DetectInstalledDevices() {
    ClearInstalledDevices();
    AddInstalledDevice(new SimCam("SimCam"));
    AddInstalledDevice(new SimFocus<AsyncProcessModel<1>>("SimFocus"));
    AddInstalledDevice(new SimObjectiveTurret("SimObjectiveTurret"));
    AddInstalledDevice(new SimXY<AsyncProcessModel<2>>("SimXY"));
    AddInstalledDevice(new SimShutter("SimShutter"));
    return DEVICE_OK;
}
