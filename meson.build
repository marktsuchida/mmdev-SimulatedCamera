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

mmdevice_dep = dependency('mmdevice')

asmjit_dep = dependency('asmjit', include_type: 'system')
blend2d_dep = dependency('blend2d', include_type: 'system')

catch2_with_main_dep = dependency('catch2-with-main', include_type: 'system')

sources = [
    'DeviceAdapter.cpp',
    'SimHub.cpp',
]

cxx = meson.get_compiler('cpp')

cpp_args = []
if cxx.get_id() == 'clang'
    cpp_args += '-Wno-gnu-anonymous-struct'
    cpp_args += '-Wno-nested-anon-types'
endif

shared_module(
    'SimulatedCamera',
    sources,
    dependencies: [
        mmdevice_dep,
        blend2d_dep,
    ],
    cpp_args: cpp_args,
    gnu_symbol_visibility: 'inlineshidden',
    name_prefix: '',
    name_suffix: 'mmdev',
)

test_exe = executable(
    'test_SimulatedCamera',
    sources + [
        'test_SimulatedCamera.cpp',
    ],
    dependencies: [
        mmdevice_dep,
        blend2d_dep,
        catch2_with_main_dep,
    ],
    cpp_args: cpp_args,
)

test('SimulatedCamera tests', test_exe)