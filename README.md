# mmdev-SimulatedCamera

A simulated microscope for [Micro-Manager](https://micro-manager.org/): a
camera, a focus (Z) stage and an XY stage that together image a synthetic
specimen. Unlike the classic `DemoCamera` adapter, the stages here actually
affect the image: moving XY pans across the specimen, and moving Z away from
focus blurs it. Stage motion takes realistic time (the devices report busy
while moving) and can optionally emit position-change notifications, which
makes the adapter useful for testing acquisition software.

> **Note:** The Meson build is experimental. It is not (yet) part of the
> supported build system for Micro-Manager / mmCoreAndDevices.

## Devices

The adapter module registers a single hub device, `SimHub`. The camera and
stages are peripherals of that hub: add `SimHub` in the Hardware
Configuration Wizard and it will offer the three devices below. The hub
connects them, so the camera can see where the stages are.

### `SimCam` (camera)

- 512 × 512 sensor, 16-bit pixels, binning 1 only
- Exposure 0.001–10000 ms (default 100 ms); snapping takes as long as the
  exposure
- ROI supported
- Sequence acquisition supported; frames arrive at one frame per exposure
  time (the requested interval is ignored)

Each image is rendered from the current stage positions:

1. The specimen is 1000 randomly placed straight filaments, plus a gray
   circle (radius 50 µm) with a diagonal line marking the specimen origin.
   It is generated once per camera instance.
2. Image scale is fixed at 1 µm per pixel.
3. Defocus blur is a Gaussian whose width grows with the Z position (Z = 0
   is in focus).
4. Brightness scales with exposure time.
5. Shot (Poisson) noise, read (Gaussian) noise and a dark offset of 100 are
   added.

### `SimFocus` (Z stage) and `SimXY` (XY stage)

- 0.1 µm per step; default speed 100 µm/s
- Moves are asynchronous: the position slews toward the target over time,
  and `Busy()` is true until it arrives. `Stop()` halts it where it is.
- Home and SetOrigin are not supported.

Properties (both stages unless noted):

| Property               | Default | Meaning                                                   |
|------------------------|---------|-----------------------------------------------------------|
| `UmPerStep`            | 0.1     | Step size (read-only)                                     |
| `NotificationsEnabled` | `No`    | Emit stage position-changed callbacks while moving        |
| `SlewTimePerStep_s`    | 0.001   | Time to move one step; sets speed for the next move       |
| `UpdateInterval_s`     | 0.1     | How often the position (and notifications) update in motion |
| `NotificationDelay_s`  | 0       | Extra delay before each notification is sent              |
| `ExternallySetSteps`   | 0       | `SimFocus` only: move the stage as if someone turned the focus knob by hand |

The XY stage honors Micro-Manager's `TransposeMirrorX/Y` settings for
reported coordinates, but the image always follows the physical stage
position.

## Building

Requirements: a C++17 compiler, [Meson](https://mesonbuild.com/) ≥ 1.8.3,
Ninja and [just](https://just.systems/). The other dependencies (MMDevice,
Blend2D, Catch2, Google Benchmark, Highway) are downloaded automatically as
Meson subprojects into `subprojects/`.

[Boost](https://www.boost.org/) is optional and must be installed separately
(`brew install boost`, `apt install libboost-all-dev`, `scoop install
main/boost`, …; set `BOOST_ROOT` if needed). When present, it provides
faster random number generation.

| Command                     | What it does                                                 |
|-----------------------------|--------------------------------------------------------------|
| `just build`                | Configure (release) if needed, then compile into `builddir/` |
| `just test`                 | Build and run the unit tests                                 |
| `just benchmark [ARGS]`     | Build and run the benchmarks (args go to Google Benchmark)   |
| `just configure BUILDTYPE`  | Reconfigure as `debug`, `debugoptimized` or `release`        |
| `just configure-for-release`| Reconfigure with Boost, SIMD and SIMD dynamic dispatch required |
| `just clean`                | Remove build products                                        |

Meson options (pass to `just configure`, e.g. `just configure release
-Duse_simd=disabled`); all are `auto` by default, meaning on if available:

- `use_boost`: Boost.Random for random numbers
- `use_simd`: Highway SIMD version of the Gaussian blur
- `simd_dynamic_dispatch`: build SIMD code for several CPU targets and
  choose one at run time
- `benchmarks`: build the benchmark program

## Build output

The files of interest in `builddir/` are:

- **`SimulatedCamera.mmdev`**: the device adapter itself, a shared library
  (despite the extension) that Micro-Manager loads.
- **`test_SimulatedCamera`**: unit tests (Catch2), run by `just test`.
  Run it directly to pass Catch2 options, e.g. `builddir/test_SimulatedCamera
  --list-tests`.
- **`bench_SimulatedCamera`**: benchmarks (Google Benchmark) for specimen
  rendering and the Gaussian blur, run by `just benchmark`.

Everything else (`*.p/` object directories, `build.ninja`,
`compile_commands.json`, `meson-*`, `subprojects/`) is Meson/Ninja
bookkeeping.

## Using the adapter

MMCore finds device adapters by file name, and this build does not use
that naming yet. To load it, copy `SimulatedCamera.mmdev` into your
Micro-Manager (or pymmcore-plus) adapter directory under the name MMCore
expects:

| Platform | File name                          |
|----------|------------------------------------|
| macOS    | `libmmgr_dal_SimulatedCamera`      |
| Linux    | `libmmgr_dal_SimulatedCamera.so.0` |
| Windows  | `mmgr_dal_SimulatedCamera.dll`     |

The adapter is built against the latest MMDevice from GitHub, so its
device interface version must match your MMCore; otherwise loading fails
with "Incompatible device interface version". Use a recent nightly build.

Then load the `SimulatedCamera` module, add `SimHub`, and add `SimCam`,
`SimFocus` and `SimXY` as its peripherals. Snap an image, move the XY stage
to pan, and move Z away from 0 to defocus.

## Source layout

| File                                     | Contents                                              |
|------------------------------------------|-------------------------------------------------------|
| `DeviceAdapter.cpp`                      | Module entry points (device registration/creation)    |
| `SimHub.h`, `SimHub.cpp`                 | Hub; passes stage positions to the camera             |
| `SimCam.h`                               | Camera device                                         |
| `SimFocus.h`, `SimXY.h`                  | Stage devices                                         |
| `ProcessModel.h`                         | Models of timed motion toward a target                |
| `DelayedNotifier.h`                      | Sends callbacks after a set delay                     |
| `SimulatedSpecimen.h`                    | Specimen rendering (Blend2D), defocus, noise          |
| `Gaussian2DFilter.h`, `Gaussian2DFilter.cpp` | Fast recursive Gaussian blur (Young & van Vliet 1995), scalar and SIMD |
| `test_SimulatedCamera.cpp`               | Unit tests                                            |
| `bench_SimulatedSpecimen.cpp`            | Benchmarks                                            |
| `subprojects/`                           | Meson wrap files for dependencies                     |

## License

BSD, per the source file headers. Copyright Board of Regents of the
University of Wisconsin System.
