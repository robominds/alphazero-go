# The CUDA part of the build (-DAZ_USE_CUDA=ON), included by CMakeLists.txt
# after the azgo target exists.
#
# What it guarantees, so a run's bits depend only on the binary and the GPU:
#   - machine code (SASS) for the architectures in AZ_CUDA_ARCHITECTURES
#     only, every entry NN-real: no PTX that a newer driver would compile
#     differently (the context refuses any other GPU at run time);
#   - --fmad=false (no a * b + c contracted unless a kernel says fmaf), IEEE
#     division and square root, denormals kept, never --use_fast_math;
#   - the static CUDA runtime, so its version is fixed in the binary;
#   - a hash over the kernel sources, nvcc's version, the architectures and
#     these flags, for the engine fingerprint ("kernels 5d0c2e91"). Editing a
#     kernel re-runs configure, which recomputes it.
#
# Mark Castelluccio, with assistance from Claude (Anthropic).

# Part 1, before the azgo target: the checks, the architectures and the CUDA
# language. Part 2, azgo_cuda_configure_target(), once azgo exists.
if(CMAKE_VERSION VERSION_LESS 3.18)
  message(FATAL_ERROR "AZ_USE_CUDA=ON needs CMake 3.18 or later (CUDA architectures)")
endif()

set(AZ_CUDA_ARCHITECTURES "89-real" CACHE STRING
    "CUDA architectures to build machine code for, each NN-real (no PTX); e.g. 89-real or 86-real;89-real")
foreach(arch IN LISTS AZ_CUDA_ARCHITECTURES)
  if(NOT arch MATCHES "^[0-9]+-real$")
    message(FATAL_ERROR "AZ_CUDA_ARCHITECTURES entries must be NN-real (machine code only; "
                        "PTX would let a driver change results): got '${arch}'")
  endif()
endforeach()
if(CMAKE_CUDA_FLAGS MATCHES "use_fast_math")
  message(FATAL_ERROR "AZ_USE_CUDA=ON refuses --use_fast_math in CMAKE_CUDA_FLAGS: it changes results")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${AZ_CUDA_ARCHITECTURES})
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
find_package(CUDAToolkit REQUIRED)

set(AZGO_CUDA_FLAGS --fmad=false --prec-div=true --prec-sqrt=true --ftz=false -Xptxas=-O3)

function(azgo_cuda_configure_target)
  set_property(TARGET azgo PROPERTY CUDA_ARCHITECTURES ${AZ_CUDA_ARCHITECTURES})
  set_property(TARGET azgo PROPERTY CUDA_RUNTIME_LIBRARY Static)
  target_compile_options(azgo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${AZGO_CUDA_FLAGS} -Xcompiler=-Wall,-Wextra>)
  target_link_libraries(azgo PUBLIC CUDA::cudart_static)
  target_compile_definitions(azgo PUBLIC AZ_USE_CUDA)

  # The kernel hash.
  file(GLOB AZGO_CUDA_HASHED CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu
       ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cuh)
  list(SORT AZGO_CUDA_HASHED)
  set(AZGO_CUDA_HASH_INPUT "nvcc ${CMAKE_CUDA_COMPILER_VERSION}\narch ${AZ_CUDA_ARCHITECTURES}\n"
                           "flags ${AZGO_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_RELEASE}\n")
  foreach(f IN LISTS AZGO_CUDA_HASHED)
    file(SHA256 ${f} h)
    get_filename_component(name ${f} NAME)
    string(APPEND AZGO_CUDA_HASH_INPUT "${name} ${h}\n")
  endforeach()
  string(SHA256 AZGO_CUDA_HASH "${AZGO_CUDA_HASH_INPUT}")
  string(SUBSTRING "${AZGO_CUDA_HASH}" 0 8 AZGO_CUDA_HASH)
  string(REPLACE ";" "," AZGO_CUDA_ARCH_TEXT "${AZ_CUDA_ARCHITECTURES}")
  file(CONFIGURE OUTPUT ${CMAKE_BINARY_DIR}/generated/azgo_cuda_info.inc
       CONTENT "#define AZGO_CUDA_KERNEL_HASH \"${AZGO_CUDA_HASH}\"\n#define AZGO_CUDA_ARCHITECTURES \"${AZGO_CUDA_ARCH_TEXT}\"\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${AZGO_CUDA_HASHED})
  message(STATUS "CUDA: architectures ${AZ_CUDA_ARCHITECTURES}, nvcc ${CMAKE_CUDA_COMPILER_VERSION}, kernels ${AZGO_CUDA_HASH}")
endfunction()
