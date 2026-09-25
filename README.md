<!--
Copyright 2025-2026 Board of Regents of the University of Wisconsin System
SPDX-License-Identifier: BSD-2-Clause
-->

# SimulatedMicroscope

A Micro-Manager device adapter that simulates a simple microscope. Like
DemoCamera, it needs no hardware, but its images are more realistic. The
specimen (filaments or nuclei) moves with the XY stage, blurs with defocus,
and responds to the objective, exposure, and shutter.

## Devices

Add **SimHub** in the Hardware Configuration Wizard, then choose any of its
peripherals:

- **SimCam**: 512 x 512, 16-bit camera. The `Mode` property selects the
  specimen (`Filaments` or `Nuclei`). Exposure scales the signal. Images
  include shot and read noise.
- **SimXY**: XY stage. Moving it moves the specimen in the image.
- **SimFocus**: Z stage. Moving away from focus blurs the image.
- **SimObjectiveTurret**: Objectives from 10x to 100x. Magnification sets the
  image scale, NA sets how quickly blur increases with defocus, and both
  affect brightness.
- **SimShutter**: When closed, the camera sees no light (noise only). If no
  shutter is loaded, light is always on.

The stages move at a finite speed and have properties to adjust their timing.

The image simulation is only roughly realistic and is not always physically
correct. Its details will likely change in future versions.

## Building

Requires a C++17 compiler, [Meson](https://mesonbuild.com/),
[Ninja](https://ninja-build.org/), and [just](https://just.systems/). Most
dependencies are downloaded automatically. Boost is optional. See the
[justfile](justfile) (or run `just`) for the available commands. Typically:

```sh
just build
just test
```

## Installing

The build produces `builddir/SimulatedMicroscope.mmdev`. Currently, you need to
copy this file into your Micro-Manager installation directory and rename it:

| Platform | File name                           |
| -------- | ----------------------------------- |
| Windows  | `mmgr_dal_SimulatedMicroscope.dll`  |
| Linux    | `mmgr_dal_SimulatedMicroscope.so.0` |
| macOS    | `mmgr_dal_SimulatedMicroscope`      |
