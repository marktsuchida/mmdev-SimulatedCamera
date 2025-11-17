# This Meson script is experimental and potentially incomplete. It is not part
# of the supported build system for Micro-Manager or mmCoreAndDevices (yet).

project(
    'mmdev-SimulatedCamera',
    'cpp',
    meson_version: '>= 1.8.3',
    default_options: [
        'cpp_std=c++17',
        'warning_level=3',
    ],
)

cxx = meson.get_compiler('cpp')

if cxx.get_id() in ['msvc', 'clang-cl']
    add_project_arguments('-DNOMINMAX', language: 'cpp')
endif

mmdevice_dep = dependency('mmdevice')

blend2d_dep = dependency('blend2d', include_type: 'system')

boost_dep = dependency(
    'boost',
    include_type: 'system',
    required: get_option('use_boost'),
)

highway_dep = dependency(
    'libhwy',
    include_type: 'system',
    required: get_option('use_simd'),
)

catch2_with_main_dep = dependency('catch2-with-main', include_type: 'system')

benchmark_dep = dependency('benchmark', include_type: 'system')

sources = [
    'DeviceAdapter.cpp',
    'SimHub.cpp',
]

cpp_args = []
if cxx.get_id() == 'clang'
    cpp_args += '-Wno-gnu-anonymous-struct'
    cpp_args += '-Wno-nested-anon-types'
endif

if boost_dep.found()
    cpp_args += '-DUSE_BOOST_RANDOM'
endif

if highway_dep.found()
    cpp_args += '-DUSE_HIGHWAY_SIMD'
endif

shared_module(
    'SimulatedCamera',
    sources,
    dependencies: [
        mmdevice_dep,
        blend2d_dep,
        boost_dep,
        highway_dep,
    ],
    cpp_args: cpp_args,
    gnu_symbol_visibility: 'inlineshidden',
    name_prefix: '',
    name_suffix: 'mmdev',
)

test_exe = executable(
    'test_SimulatedCamera',
    sources
    + [
        'test_SimulatedCamera.cpp',
    ],
    dependencies: [
        mmdevice_dep,
        blend2d_dep,
        boost_dep,
        catch2_with_main_dep,
        highway_dep,
    ],
    cpp_args: cpp_args,
)

test('SimulatedCamera tests', test_exe)

bench_exe = executable(
    'bench_SimulatedCamera',
    [
        'bench_SimulatedSpecimen.cpp',
    ],
    dependencies: [
        blend2d_dep,
        boost_dep,
        benchmark_dep,
        highway_dep,
    ],
    cpp_args: cpp_args,
)

benchmark('SimulatedCamera benchmarks', bench_exe)