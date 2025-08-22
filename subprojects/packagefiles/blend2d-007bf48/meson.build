project(
    'blend2d',
    'cpp',
    meson_version: '>= 1.8.3',
    default_options: [
        'cpp_std=c++17',
        'cpp_eh=none',
        'cpp_rtti=false',
        'warning_level=3',
    ],
)

asmjit_dep = dependency('asmjit')

blend2d_sources = [
    'src/blend2d/bitset_test.cpp',
    'src/blend2d/opentype/otcff_test.cpp',
    'src/blend2d/opentype/otface.cpp',
    'src/blend2d/opentype/otcff.cpp',
    'src/blend2d/opentype/otglyf_asimd.cpp',
    'src/blend2d/opentype/otlayout.cpp',
    'src/blend2d/opentype/otmetrics.cpp',
    'src/blend2d/opentype/otcore.cpp',
    'src/blend2d/opentype/otname.cpp',
    'src/blend2d/opentype/otglyfsimddata.cpp',
    'src/blend2d/opentype/otcmap.cpp',
    'src/blend2d/opentype/otglyf.cpp',
    'src/blend2d/opentype/otkern.cpp',
    'src/blend2d/fontvariationsettings_test.cpp',
    'src/blend2d/pathstroke.cpp',
    'src/blend2d/pipeline/pipedefs.cpp',
    'src/blend2d/pipeline/piperuntime.cpp',
    'src/blend2d/pipeline/jit/pipecompiler_test.cpp',
    'src/blend2d/pipeline/jit/fetchsolidpart.cpp',
    'src/blend2d/pipeline/jit/pipefunction.cpp',
    'src/blend2d/pipeline/jit/fetchutilscoverage.cpp',
    'src/blend2d/pipeline/jit/fetchpart.cpp',
    'src/blend2d/pipeline/jit/pipepart.cpp',
    'src/blend2d/pipeline/jit/compoppart.cpp',
    'src/blend2d/pipeline/jit/fetchutilspixelgather.cpp',
    'src/blend2d/pipeline/jit/pipecompiler_a64.cpp',
    'src/blend2d/pipeline/jit/fetchutilspixelaccess.cpp',
    'src/blend2d/pipeline/jit/fetchutilsinlineloops.cpp',
    'src/blend2d/pipeline/jit/pipegenruntime.cpp',
    'src/blend2d/pipeline/jit/pipecompiler_x86.cpp',
    'src/blend2d/pipeline/jit/fetchpatternpart.cpp',
    'src/blend2d/pipeline/jit/fetchgradientpart.cpp',
    'src/blend2d/pipeline/jit/pipeprimitives.cpp',
    'src/blend2d/pipeline/jit/fillpart.cpp',
    'src/blend2d/pipeline/jit/fetchpixelptrpart.cpp',
    'src/blend2d/pipeline/jit/pipecomposer.cpp',
    'src/blend2d/pipeline/reference/fixedpiperuntime.cpp',
    'src/blend2d/tables/tables_test.cpp',
    'src/blend2d/tables/tables.cpp',
    'src/blend2d/fonttagset.cpp',
    'src/blend2d/imagedecoder.cpp',
    'src/blend2d/geometry.cpp',
    'src/blend2d/path_test.cpp',
    'src/blend2d/image.cpp',
    'src/blend2d/fontface.cpp',
    'src/blend2d/format.cpp',
    'src/blend2d/pixelops/scalar_test.cpp',
    'src/blend2d/pixelops/interpolation.cpp',
    'src/blend2d/pixelops/funcs.cpp',
    'src/blend2d/unicode/unicode_test.cpp',
    'src/blend2d/unicode/unicode.cpp',
    'src/blend2d/imagescale.cpp',
    'src/blend2d/imagecodec.cpp',
    'src/blend2d/string_test.cpp',
    'src/blend2d/imageencoder.cpp',
    'src/blend2d/context.cpp',
    'src/blend2d/random.cpp',
    'src/blend2d/fontfeaturesettings_test.cpp',
    'src/blend2d/gradient_test.cpp',
    'src/blend2d/bitarray_test.cpp',
    'src/blend2d/codec/jpegops.cpp',
    'src/blend2d/codec/jpeghuffman.cpp',
    'src/blend2d/codec/pngopssimdimpl_test.cpp',
    'src/blend2d/codec/qoicodec.cpp',
    'src/blend2d/codec/pngcodec.cpp',
    'src/blend2d/codec/bmpcodec.cpp',
    'src/blend2d/codec/pngops.cpp',
    'src/blend2d/codec/codec_test.cpp',
    'src/blend2d/codec/jpegcodec.cpp',
    'src/blend2d/codec/pngops_asimd.cpp',
    'src/blend2d/object.cpp',
    'src/blend2d/rgba_test.cpp',
    'src/blend2d/trace.cpp',
    'src/blend2d/font.cpp',
    'src/blend2d/simd/simd_test.cpp',
    'src/blend2d/simd/simdarm_test_asimd.cpp',
    'src/blend2d/string.cpp',
    'src/blend2d/runtimescope.cpp',
    'src/blend2d/api-globals.cpp',
    'src/blend2d/fonttagdatainfo.cpp',
    'src/blend2d/glyphbuffer.cpp',
    'src/blend2d/matrix_test.cpp',
    'src/blend2d/context_test.cpp',
    'src/blend2d/array.cpp',
    'src/blend2d/var.cpp',
    'src/blend2d/compopinfo.cpp',
    'src/blend2d/filesystem.cpp',
    'src/blend2d/support/zeroallocator_test.cpp',
    'src/blend2d/support/bitops_test.cpp',
    'src/blend2d/support/zeroallocator.cpp',
    'src/blend2d/support/math.cpp',
    'src/blend2d/support/arenaallocator.cpp',
    'src/blend2d/support/scopedallocator.cpp',
    'src/blend2d/support/arenalist_test.cpp',
    'src/blend2d/support/algorithm_test.cpp',
    'src/blend2d/support/arenabitarray_test.cpp',
    'src/blend2d/support/arenahashmap.cpp',
    'src/blend2d/support/memops_test.cpp',
    'src/blend2d/support/arenatree_test.cpp',
    'src/blend2d/support/ptrops_test.cpp',
    'src/blend2d/support/intops_test.cpp',
    'src/blend2d/support/math_test.cpp',
    'src/blend2d/support/arenahashmap_test.cpp',
    'src/blend2d/pixelconverter.cpp',
    'src/blend2d/bitarray.cpp',
    'src/blend2d/api-nocxx.cpp',
    'src/blend2d/matrix.cpp',
    'src/blend2d/pattern.cpp',
    'src/blend2d/compression/deflatedefs.cpp',
    'src/blend2d/compression/checksum_asimd.cpp',
    'src/blend2d/compression/checksum.cpp',
    'src/blend2d/compression/deflatedecoderfast.cpp',
    'src/blend2d/compression/deflatedecoderutils.cpp',
    'src/blend2d/compression/checksum_test.cpp',
    'src/blend2d/compression/deflatedecoder.cpp',
    'src/blend2d/compression/checksum_asimd_crypto.cpp',
    'src/blend2d/compression/deflate_test.cpp',
    'src/blend2d/compression/deflateencoder.cpp',
    'src/blend2d/fontvariationsettings.cpp',
    'src/blend2d/path.cpp',
    'src/blend2d/pixelconverter_test.cpp',
    'src/blend2d/fonttagdataids_test.cpp',
    'src/blend2d/raster/workermanager.cpp',
    'src/blend2d/raster/workersynchronization.cpp',
    'src/blend2d/raster/rastercontext.cpp',
    'src/blend2d/raster/analyticrasterizer_test.cpp',
    'src/blend2d/raster/workerproc.cpp',
    'src/blend2d/raster/renderfetchdata.cpp',
    'src/blend2d/raster/workdata.cpp',
    'src/blend2d/raster/rastercontextops.cpp',
    'src/blend2d/raster/rendertargetinfo.cpp',
    'src/blend2d/gradient.cpp',
    'src/blend2d/fonttagdatainfo_test.cpp',
    'src/blend2d/threading/thread.cpp',
    'src/blend2d/threading/uniqueidgenerator.cpp',
    'src/blend2d/threading/futex.cpp',
    'src/blend2d/threading/threadpool_test.cpp',
    'src/blend2d/threading/threadpool.cpp',
    'src/blend2d/runtime.cpp',
    'src/blend2d/image_test.cpp',
    'src/blend2d/font_test.cpp',
    'src/blend2d/fontmanager.cpp',
    'src/blend2d/fontdata.cpp',
    'src/blend2d/fontfeaturesettings.cpp',
    'src/blend2d/var_test.cpp',
    'src/blend2d/array_test.cpp',
    'src/blend2d/random_test.cpp',
    'src/blend2d/fonttagdataids.cpp',
    'src/blend2d/bitset.cpp',
    'src/blend2d/imagecodec_test.cpp',
]

sse2_sources = [
    'src/blend2d/pipeline/jit/pipecompiler_test_sse2.cpp',
    'src/blend2d/pixelops/interpolation_sse2.cpp',
    'src/blend2d/matrix_sse2.cpp',
    'src/blend2d/codec/pngops_sse2.cpp',
    'src/blend2d/codec/jpegops_sse2.cpp',
    'src/blend2d/simd/simdx86_test_sse2.cpp',
    'src/blend2d/pixelconverter_sse2.cpp',
    'src/blend2d/compression/checksum_sse2.cpp',
]

ssse3_sources = [
    'src/blend2d/simd/simdx86_test_ssse3.cpp',
    'src/blend2d/pixelconverter_ssse3.cpp',
]

sse4_1_sources = [
    'src/blend2d/simd/simdx86_test_sse4_1.cpp',
]

sse4_2_sources = [
    'src/blend2d/opentype/otglyf_sse4_2.cpp',
    'src/blend2d/simd/simdx86_test_sse4_2.cpp',
    'src/blend2d/compression/checksum_sse4_2.cpp',
]

avx_sources = [
    'src/blend2d/matrix_avx.cpp',
    'src/blend2d/codec/pngops_avx.cpp',
    'src/blend2d/simd/simdx86_test_avx.cpp',
]

avx2_sources = [
    'src/blend2d/opentype/otglyf_avx2.cpp',
    'src/blend2d/pipeline/jit/pipecompiler_test_avx2fma.cpp',
    'src/blend2d/pixelops/interpolation_avx2.cpp',
    'src/blend2d/simd/simdx86_test_avx2.cpp',
    'src/blend2d/compression/deflatedecoderfast_avx2.cpp',
    'src/blend2d/pixelconverter_avx2.cpp',
]

avx512_sources = [
    'src/blend2d/simd/simdx86_test_avx512.cpp',
]

blend2d_inc = include_directories('src')

cxx = meson.get_compiler('cpp')

blend2d_cpp_args = [
    '-DBL_STATIC',
]

if cxx.get_id() in ['gcc', 'clang']
    blend2d_cpp_args += [
        '-fno-math-errno',
        '-fno-threadsafe-statics',
        '-fmerge-all-constants',
    ]
endif
if cxx.get_id() == 'gcc'
    blend2d_cpp_args += '-fno-semantic-interposition'
endif
if get_option('optimization') in ['2', '3', 's']
    blend2d_cpp_args += '-DBL_BUILD_RELEASE'
else
    blend2d_cpp_args += '-DBL_BUILD_DEBUG'
endif
if host_machine.cpu_family() == 'x86_64'
    if cxx.get_id() in ['gcc', 'clang', 'clang-cl', 'cl']
        blend2d_cpp_args += [
            '-DBL_BUILD_OPT_SSE2',
            '-DBL_BUILD_OPT_SSE3',
            '-DBL_BUILD_OPT_SSSE3',
            '-DBL_BUILD_OPT_SSE4_1',
            '-DBL_BUILD_OPT_SSE4_2',
        ]
        if cxx.get_id() in ['gcc', 'clang', 'cl']
            blend2d_cpp_args += [
                '-DBL_BUILD_OPT_AVX',
                '-DBL_BUILD_OPT_AVX2',
                '-DBL_BUILD_OPT_AVX512',
            ]
        endif
    endif
endif
if cxx.get_id() == 'clang'
    blend2d_cpp_args += '-Wno-gnu-anonymous-struct'
    blend2d_cpp_args += '-Wno-nested-anon-types'
endif

blend2d_lib = static_library(
    'blend2d',
    blend2d_sources,
    cpp_args: blend2d_cpp_args,
    dependencies: asmjit_dep,
    gnu_symbol_visibility: 'inlineshidden',
)

sse2_cpp_args = []
if cxx.get_id() in ['gcc', 'clang']
    sse2_cpp_args += '-msse2'
elif cxx.get_id() == 'msvc'
    sse2_cpp_args += '-arch:SSE2'
    sse2_cpp_args += '-D__SSE2__'
elif cxx.get_id() == 'clang-cl'
    sse2_cpp_args += '-arch:SSE2'
endif
if sse2_cpp_args.length() != 0 and host_machine.cpu_family() == 'x86_64'
    sse2_lib = static_library(
        'blend2d_sse2',
        sse2_sources,
        cpp_args: blend2d_cpp_args + sse2_cpp_args,
        gnu_symbol_visibility: 'inlineshidden',
    )
else
    sse2_lib = []
endif

ssse3_cpp_args = []
if cxx.get_id() in ['gcc', 'clang']
    ssse3_cpp_args += '-mssse3'
elif cxx.get_id() == 'msvc'
    ssse3_cpp_args += '-arch:SSE2'
    ssse3_cpp_args += '-D__SSSE3__'
elif cxx.get_id() == 'clang-cl'
    ssse3_cpp_args += '-arch:SSE2'
    ssse3_cpp_args += '-mssse3'
endif
if ssse3_cpp_args.length() != 0 and host_machine.cpu_family() == 'x86_64'
    ssse3_lib = static_library(
        'blend2d_ssse3',
        ssse3_sources,
        cpp_args: blend2d_cpp_args + ssse3_cpp_args,
        gnu_symbol_visibility: 'inlineshidden',
    )
else
    ssse3_lib = []
endif

sse4_1_cpp_args = []
if cxx.get_id() in ['gcc', 'clang']
    sse4_1_cpp_args += '-msse4.1'
elif cxx.get_id() == 'msvc'
    sse4_1_cpp_args += '-arch:SSE2'
    sse4_1_cpp_args += '-D__SSE4_1__'
elif cxx.get_id() == 'clang-cl'
    sse4_1_cpp_args += '-arch:SSE2'
    sse4_1_cpp_args += '-msse4.1'
endif
if sse4_1_cpp_args.length() != 0 and host_machine.cpu_family() == 'x86_64'
    sse4_1_lib = static_library(
        'blend2d_sse4_1',
        sse4_1_sources,
        cpp_args: blend2d_cpp_args + sse4_1_cpp_args,
        gnu_symbol_visibility: 'inlineshidden',
    )
else
    sse4_1_lib = []
endif

sse4_2_cpp_args = []
if cxx.get_id() in ['gcc', 'clang']
    sse4_2_cpp_args += '-msse4.2'
    sse4_2_cpp_args += '-mpclmul'
elif cxx.get_id() == 'msvc'
    sse4_2_cpp_args += '-arch:SSE2'
    sse4_2_cpp_args += '-D__SSE4_2__'
elif cxx.get_id() == 'clang-cl'
    sse4_2_cpp_args += '-arch:SSE2'
    sse4_2_cpp_args += '-msse4.2'
    sse4_2_cpp_args += '-mpclmul'
endif
if sse4_2_cpp_args.length() != 0 and host_machine.cpu_family() == 'x86_64'
    sse4_2_lib = static_library(
        'blend2d_sse4_2',
        sse4_2_sources,
        cpp_args: blend2d_cpp_args + sse4_2_cpp_args,
        gnu_symbol_visibility: 'inlineshidden',
    )
else
    sse4_2_lib = []
endif

avx_cpp_args = []
if cxx.get_id() in ['gcc', 'clang']
    avx_cpp_args += '-mavx'
elif cxx.get_id() == 'msvc'
    avx_cpp_args += '-arch:AVX'
endif
if avx_cpp_args.length() != 0 and host_machine.cpu_family() == 'x86_64'
    avx_lib = static_library(
        'blend2d_avx',
        avx_sources,
        cpp_args: blend2d_cpp_args + avx_cpp_args,
        gnu_symbol_visibility: 'inlineshidden',
    )
else
    avx_lib = []
endif

avx2_cpp_args = []
if cxx.get_id() in ['gcc', 'clang']
    avx2_cpp_args += '-mavx2'
elif cxx.get_id() == 'msvc'
    avx2_cpp_args += '-arch:AVX2'
endif
if avx2_cpp_args.length() != 0 and host_machine.cpu_family() == 'x86_64'
    avx2_lib = static_library(
        'blend2d_avx2',
        avx2_sources,
        cpp_args: blend2d_cpp_args + avx2_cpp_args,
        gnu_symbol_visibility: 'inlineshidden',
    )
else
    avx2_lib = []
endif

avx512_cpp_args = []
if cxx.get_id() in ['gcc', 'clang']
    avx512_cpp_args += '-mavx512f'
    avx512_cpp_args += '-mavx512bw'
    avx512_cpp_args += '-mavx512dq'
    avx512_cpp_args += '-mavx512cd'
    avx512_cpp_args += '-mavx512vl'
elif cxx.get_id() == 'msvc'
    avx512_cpp_args += '-arch:AVX512'
endif
if avx512_cpp_args.length() != 0 and host_machine.cpu_family() == 'x86_64'
    avx512_lib = static_library(
        'blend2d_avx512',
        avx512_sources,
        cpp_args: blend2d_cpp_args + avx512_cpp_args,
        gnu_symbol_visibility: 'inlineshidden',
    )
else
    avx512_lib = []
endif

blend2d_dep = declare_dependency(
    include_directories: blend2d_inc,
    link_with: [
        blend2d_lib,
        sse2_lib,
        ssse3_lib,
        sse4_1_lib,
        sse4_2_lib,
        avx_lib,
        avx2_lib,
        avx512_lib,
    ],
)
meson.override_dependency('blend2d', blend2d_dep)
