cmake_minimum_required(VERSION 3.16)
project(alphazero_go CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AZ_USE_METAL "GPU inference and training with Metal (macOS; implies AZ_USE_BLAS)" OFF)
option(AZ_USE_BLAS "Use BLAS for gemm (Accelerate on macOS, OpenBLAS elsewhere)" OFF)

# Metal runs the network on the Apple GPU. Everything that stays on the CPU
# with backend = metal (training in Phase A, the bots, grow) uses BLAS, which
# on macOS is Accelerate, so Metal turns BLAS on. It does so with a normal
# variable that shadows the cached option for this configuration only: the
# cache (and so `cmake -L` or ccmake) may still show AZ_USE_BLAS OFF, but the
# build uses BLAS, and switching Metal off again restores the cached choice.
if(AZ_USE_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "AZ_USE_METAL=ON needs macOS (Metal is Apple's GPU API)")
  endif()
  if(NOT AZ_USE_BLAS)
    message(STATUS "AZ_USE_METAL=ON turns AZ_USE_BLAS on (CPU work uses Accelerate)")
    set(AZ_USE_BLAS ON)
  endif()
  enable_language(OBJCXX)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  set(CMAKE_OBJCXX_EXTENSIONS OFF)
endif()

find_package(Threads REQUIRED)

# ---- Library ---------------------------------------------------------------
# Every src/*.cpp belongs to the library, so a task that adds a component adds
# no CMake lines. CONFIGURE_DEPENDS makes `cmake --build` re-glob when files
# appear. The BLAS backend is the one exception: it only compiles against a
# BLAS header, so it is dropped unless AZ_USE_BLAS is on. REMOVE_ITEM is a
# no-op while that file does not exist yet.
file(GLOB AZGO_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
if(NOT AZ_USE_BLAS)
  list(REMOVE_ITEM AZGO_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/backend_blas.cpp)
endif()
# src/metal/*.cpp is portable (what a build without Metal answers when asked
# about it) and always compiled; the Objective-C++ in src/metal/*.mm only
# with AZ_USE_METAL.
file(GLOB AZGO_METAL_PORTABLE CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/*.cpp)
list(APPEND AZGO_SOURCES ${AZGO_METAL_PORTABLE})
if(AZ_USE_METAL)
  file(GLOB AZGO_METAL_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/*.mm)
  list(APPEND AZGO_SOURCES ${AZGO_METAL_SOURCES})
endif()

add_library(azgo STATIC ${AZGO_SOURCES})
target_include_directories(azgo PUBLIC include)
target_compile_options(azgo PRIVATE -Wall -Wextra)
target_link_libraries(azgo PUBLIC Threads::Threads)

if(AZ_USE_METAL)
  # The kernels are compiled by the OS at run time (newLibraryWithSource), so
  # the build needs no `metal` shader compiler, which only Xcode ships: the
  # Command Line Tools suffice. CMake embeds the source as one C++ raw string,
  # with shader_types.h (the parameter structs shared with the C++ side)
  # pasted where kernels.metal includes it. Editing either file re-runs
  # configure on the next build.
  set(AZGO_METAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/metal)
  file(READ ${AZGO_METAL_DIR}/shader_types.h AZGO_SHADER_TYPES)
  file(READ ${AZGO_METAL_DIR}/kernels.metal AZGO_KERNELS)
  string(REPLACE "#include \"shader_types.h\"" "${AZGO_SHADER_TYPES}" AZGO_METAL_SOURCE "${AZGO_KERNELS}")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/azgo_metal_kernels.inc.in
                 ${CMAKE_BINARY_DIR}/generated/azgo_metal_kernels.inc @ONLY)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               ${AZGO_METAL_DIR}/shader_types.h ${AZGO_METAL_DIR}/kernels.metal)
  target_include_directories(azgo PRIVATE ${CMAKE_BINARY_DIR}/generated)
  target_compile_options(azgo PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
  target_link_libraries(azgo PUBLIC "-framework Metal" "-framework Foundation")
  target_compile_definitions(azgo PUBLIC AZ_USE_METAL)
endif()

if(AZ_USE_BLAS)
  if(APPLE)
    # Accelerate ships with macOS. ACCELERATE_NEW_LAPACK selects the current
    # cblas headers instead of the deprecated ones.
    target_link_libraries(azgo PUBLIC "-framework Accelerate")
    target_compile_definitions(azgo PUBLIC AZ_USE_BLAS ACCELERATE_NEW_LAPACK)
  else()
    find_package(BLAS REQUIRED)
    find_path(CBLAS_INCLUDE_DIR cblas.h PATH_SUFFIXES openblas)
    if(NOT CBLAS_INCLUDE_DIR)
      message(FATAL_ERROR "AZ_USE_BLAS=ON but cblas.h was not found (install OpenBLAS headers)")
    endif()
    target_include_directories(azgo PRIVATE ${CBLAS_INCLUDE_DIR})
    target_link_libraries(azgo PUBLIC ${BLAS_LIBRARIES})
    target_compile_definitions(azgo PUBLIC AZ_USE_BLAS)
  endif()
endif()

# ---- Tests -----------------------------------------------------------------
# One executable per tests/test_<component>.cpp, named after the file.
# -UNDEBUG: the default build type is Release, which defines NDEBUG and would
# compile every assert() in a test to nothing. Stripping it here keeps the
# tests meaningful without slowing the library down.
# AZGO_SOURCE_DIR lets tests find repository files such as configs/.
enable_testing()

file(GLOB AZGO_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp)
foreach(test_src ${AZGO_TESTS})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE azgo)
  target_compile_options(${test_name} PRIVATE -UNDEBUG)
  target_compile_definitions(${test_name} PRIVATE AZGO_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
# test_metal returns 77 where Metal is not built or not usable: CTest then
# reports it as Skipped, not failed.
set_tests_properties(test_metal PROPERTIES SKIP_RETURN_CODE 77)

# ---- Apps ------------------------------------------------------------------

# --- Apps (Tasks 21-22) -------------------------------------------------------
# apps/cli.hpp is header-only and lives next to the apps that use it rather
# than in include/azgo, so test_cli (created by the test glob above) needs
# apps/ on its include path.
target_include_directories(test_cli PRIVATE ${CMAKE_SOURCE_DIR}/apps)

add_executable(train apps/train.cpp)
target_link_libraries(train PRIVATE azgo)
target_include_directories(train PRIVATE ${CMAKE_SOURCE_DIR}/apps)

# End-to-end check of all the executables. ctest does not build targets:
# run `cmake --build build` first, as the usual workflow does.
add_test(NAME integration_smoke
         COMMAND bash ${CMAKE_SOURCE_DIR}/tests/integration_smoke.sh ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR})

foreach(app arena analyze bench grow gtp)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE azgo)
  target_include_directories(${app} PRIVATE ${CMAKE_SOURCE_DIR}/apps)
endforeach()
