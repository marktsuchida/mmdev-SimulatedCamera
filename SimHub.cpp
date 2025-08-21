#include "SimHub.h"

#include "SimCam.h"
#include "SimFocus.h"
#include "SimXY.h"

int SimHub::DetectInstalledDevices() {
    ClearInstalledDevices();
    AddInstalledDevice(new SimCam("SimCam"));
    AddInstalledDevice(new SimFocus<AsyncProcessModel<1>>("SimFocus"));
    AddInstalledDevice(new SimXY<AsyncProcessModel<2>>("SimXY"));
    return DEVICE_OK;
}