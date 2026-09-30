cmake_minimum_required(VERSION 3.16)
project(alphazero_go CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(AZ_USE_BLAS "Use BLAS for gemm (Accelerate on macOS, OpenBLAS elsewhere)" OFF)

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

add_library(azgo STATIC ${AZGO_SOURCES})
target_include_directories(azgo PUBLIC include)
target_compile_options(azgo PRIVATE -Wall -Wextra)
target_link_libraries(azgo PUBLIC Threads::Threads)

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

# ---- Apps ------------------------------------------------------------------

# --- Apps (Tasks 21-22) -------------------------------------------------------
# apps/cli.hpp is header-only and lives next to the apps that use it rather
# than in include/azgo, so test_cli (created by the test glob above) needs
# apps/ on its include path.
target_include_directories(test_cli PRIVATE ${CMAKE_SOURCE_DIR}/apps)

add_executable(train apps/train.cpp)
target_link_libraries(train PRIVATE azgo)
target_include_directories(train PRIVATE ${CMAKE_SOURCE_DIR}/apps)
