project(
    'asmjit',
    'cpp',
    meson_version: '>= 1.8.3',
    default_options: [
        'cpp_std=c++17',
        'warning_level=3',
    ],
)

asmjit_sources = [
    'src/asmjit/core/func.cpp',
    'src/asmjit/core/cpuinfo.cpp',
    'src/asmjit/core/rapass.cpp',
    'src/asmjit/core/errorhandler.cpp',
    'src/asmjit/core/zonestack.cpp',
    'src/asmjit/core/zone.cpp',
    'src/asmjit/core/rastack.cpp',
    'src/asmjit/core/inst.cpp',
    'src/asmjit/core/environment.cpp',
    'src/asmjit/core/codeholder.cpp',
    'src/asmjit/core/jitruntime.cpp',
    'src/asmjit/core/compiler.cpp',
    'src/asmjit/core/emitter.cpp',
    'src/asmjit/core/ralocal.cpp',
    'src/asmjit/core/codewriter.cpp',
    'src/asmjit/core/funcargscontext.cpp',
    'src/asmjit/core/zonevector.cpp',
    'src/asmjit/core/zonetree.cpp',
    'src/asmjit/core/string.cpp',
    'src/asmjit/core/archtraits.cpp',
    'src/asmjit/core/globals.cpp',
    'src/asmjit/core/assembler.cpp',
    'src/asmjit/core/builder.cpp',
    'src/asmjit/core/emithelper.cpp',
    'src/asmjit/core/virtmem.cpp',
    'src/asmjit/core/emitterutils.cpp',
    'src/asmjit/core/target.cpp',
    'src/asmjit/core/zonelist.cpp',
    'src/asmjit/core/formatter.cpp',
    'src/asmjit/core/zonehash.cpp',
    'src/asmjit/core/logger.cpp',
    'src/asmjit/core/support.cpp',
    'src/asmjit/core/operand.cpp',
    'src/asmjit/core/constpool.cpp',
    'src/asmjit/core/jitallocator.cpp',
    'src/asmjit/core/instdb.cpp',
    'src/asmjit/core/type.cpp',
    'src/asmjit/core/osutils.cpp',
    'src/asmjit/x86/x86emithelper.cpp',
    'src/asmjit/x86/x86instdb.cpp',
    'src/asmjit/x86/x86func.cpp',
    'src/asmjit/x86/x86operand.cpp',
    'src/asmjit/x86/x86formatter.cpp',
    'src/asmjit/x86/x86builder.cpp',
    'src/asmjit/x86/x86assembler.cpp',
    'src/asmjit/x86/x86instapi.cpp',
    'src/asmjit/x86/x86compiler.cpp',
    'src/asmjit/x86/x86rapass.cpp',
    'src/asmjit/arm/a64compiler.cpp',
    'src/asmjit/arm/a64instdb.cpp',
    'src/asmjit/arm/a64instapi.cpp',
    'src/asmjit/arm/a64formatter.cpp',
    'src/asmjit/arm/a64emithelper.cpp',
    'src/asmjit/arm/a64assembler.cpp',
    'src/asmjit/arm/a64operand.cpp',
    'src/asmjit/arm/a64builder.cpp',
    'src/asmjit/arm/armformatter.cpp',
    'src/asmjit/arm/a64func.cpp',
    'src/asmjit/arm/a64rapass.cpp',
]

asmjit_inc = include_directories('src')

cxx = meson.get_compiler('cpp')

asmjit_cpp_args = [
    '-DASMJIT_STATIC',
]
if cxx.get_id() == 'clang'
    asmjit_cpp_args += '-Wno-gnu-anonymous-struct'
    asmjit_cpp_args += '-Wno-nested-anon-types'
endif

asmjit_lib = static_library(
    'asmjit',
    asmjit_sources,
    cpp_args: asmjit_cpp_args,
    gnu_symbol_visibility: 'inlineshidden',
)

asmjit_dep = declare_dependency(
    include_directories: asmjit_inc,
    link_with: asmjit_lib,
)
meson.override_dependency('asmjit', asmjit_dep)
