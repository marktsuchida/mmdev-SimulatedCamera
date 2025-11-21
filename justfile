# Show usage
help:
    @just --list
    @echo
    @echo 'Most dependencies are automatically downloaded, except for Boost.'
    @echo '(Boost is an optional dependency used for faster random numbers.)'
    @echo 'Use `apt install libboost-all-dev`, `brew install boost`, or'
    @echo '`scoop install main/boost`, etc. Set `BOOST_ROOT` if necessary.'

# Configure for debug/debugoptimized/release
configure BUILDTYPE *FLAGS:
    meson setup --reconfigure builddir --buildtype={{BUILDTYPE}} \
        --vsenv --wrap-mode=forcefallback --default-library=static \
        -Dlibhwy:contrib=disabled -Dgoogle-benchmark:tests=disabled \
        {{FLAGS}}

# Configure with optional features enabled
configure-for-release:
    @just configure release \
        -Duse_boost=enabled \
        -Duse_simd=enabled -Dsimd_dynamic_dispatch=enabled

_configure_if_not_configured:
    #!/usr/bin/env bash
    set -euxo pipefail
    if [ ! -d builddir ]; then
        just configure release
    fi

# Build
build: _configure_if_not_configured
    meson compile -C builddir

# Remove build products
clean:
    if [ -d builddir ]; then meson compile --clean -C builddir; fi

# Run unit tests
test: _configure_if_not_configured
    meson test -C builddir

# Run benchmarks
[positional-arguments]
benchmark *FLAGS: build
    ./builddir/bench_SimulatedCamera "$@"
