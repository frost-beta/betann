cmake_minimum_required(VERSION 3.23)

project(betann)

option(BETANN_BUILD_TESTS "Build BetaNN's tests" ON)

# Use C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the BetaNN library.
add_library(betann STATIC)
target_sources(betann PRIVATE betann/device.cc
                              betann/kernels.cc
                      PUBLIC FILE_SET HEADERS
                             BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
                             FILES betann/betann.h
                                   betann/device.h
                                   betann/kernels.h)
target_include_directories(betann PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                                  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# Embed WGSL sources in binary.
set(BETANN_WGSL_SOURCES betann/wgsl/binary_contiguous.wgsl
                        betann/wgsl/binary_general.wgsl
                        betann/wgsl/binary_ops.wgsl
                        betann/wgsl/copy_contiguous.wgsl
                        betann/wgsl/copy_general.wgsl)
string(JOIN ":" BETANN_WGSL_SOURCES_ARG ${BETANN_WGSL_SOURCES})
add_custom_command(
  OUTPUT gen/wgsl_sources.h
  COMMAND ${CMAKE_COMMAND}
          -DBETANN_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
          -DBETANN_WGSL_SOURCES=${BETANN_WGSL_SOURCES_ARG}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/betann/wgsl/shader2h.cmake"
  DEPENDS betann/wgsl/shader2h.cmake
          ${BETANN_WGSL_SOURCES})
add_custom_target(wgsl_sources DEPENDS gen/wgsl_sources.h)
add_dependencies(betann wgsl_sources)

# Use fmt.
include(FetchContent)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 10.2.1
  EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(fmt)
target_link_libraries(betann PRIVATE $<BUILD_INTERFACE:fmt::fmt-header-only>)

# Link with dawn statically.
cmake_policy(SET CMP0097 NEW)  # enable GIT_SUBMODULES
FetchContent_Declare(
  dawn
  GIT_REPOSITORY https://dawn.googlesource.com/dawn
  GIT_TAG "chromium/6972"
  GIT_SHALLOW ON
  GIT_SUBMODULES ""  # no submodules
  UPDATE_DISCONNECTED ON  # no update when re-generating
  EXCLUDE_FROM_ALL)
set(DAWN_FETCH_DEPENDENCIES ON)
set(DAWN_USE_GLFW OFF)
set(DAWN_USE_X11 OFF)
set(DAWN_USE_WINDOWS_UI OFF)
set(DAWN_BUILD_SAMPLES OFF)
set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF)
set(TINT_BUILD_CMD_TOOLS OFF)
set(TINT_BUILD_TESTS OFF)
FetchContent_MakeAvailable(dawn)
target_link_libraries(betann PUBLIC $<BUILD_INTERFACE:dawncpp_headers>
                             PRIVATE $<LINK_ONLY:absl::strings>
                                     $<LINK_ONLY:dawn::dawn_common>
                                     $<LINK_ONLY:dawn::dawn_native>)

# Build the procs that call dawn_native directly.
find_package(Python3 REQUIRED)
DawnJSONGenerator(TARGET "webgpu_dawn_native_proc"
                  PRINT_NAME "Dawn native WebGPU procs"
                  OUTPUT_SOURCES WEBGPU_DAWN_NATIVE_PROC_GEN_SOURCES)
target_sources(betann PRIVATE ${WEBGPU_DAWN_NATIVE_PROC_GEN_SOURCES})

# Build tests.
if (BETANN_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(googletest)
  add_executable(betann_tests tests/binary_tests.cc
                              tests/copy_tests.cc)
  target_link_libraries(betann_tests PRIVATE betann
                                             GTest::gtest_main)
endif()

# Make the library installable.
install(TARGETS betann
        EXPORT BetaNNTargets
        FILE_SET HEADERS)
install(EXPORT BetaNNTargets
        FILE BetaNN.cmake
        DESTINATION share/cmake/betann)
