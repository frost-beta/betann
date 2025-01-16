cmake_minimum_required(VERSION 3.23)

project(betann)

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
                                   betann/binary.h
                                   betann/device.h
                                   betann/kernels.h)
target_include_directories(betann PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
                                  PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# Include WGSL sources in binary.
set(BETANN_WGSL_SOURCES betann/wgsl/binary.wgsl
                        betann/wgsl/utils.wgsl)
add_custom_command(
  OUTPUT gen/wgsl_sources.h
  COMMAND ${CMAKE_COMMAND}
          -DBETANN_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
          -DBETANN_WGSL_SOURCES=${BETANN_WGSL_SOURCES}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/betann/wgsl/shader2h.cmake"
  DEPENDS betann/wgsl/bin2h.cmake
          betann/wgsl/shader2h.cmake
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
set(DAWN_FETCH_DEPENDENCIES ON)
set(DAWN_USE_GLFW OFF)
set(DAWN_USE_X11 OFF)
set(DAWN_USE_WINDOWS_UI OFF)
set(DAWN_BUILD_SAMPLES OFF)
set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF)
set(TINT_BUILD_CMD_TOOLS OFF)
set(TINT_BUILD_TESTS OFF)
add_subdirectory(dawn)
target_link_libraries(betann PUBLIC dawncpp_headers
                             PRIVATE absl::strings
                                     dawn::dawn_common
                                     dawn::dawn_native)

# Build the procs that call dawn_native directly.
find_package(Python3 REQUIRED)
DawnJSONGenerator(TARGET "webgpu_dawn_native_proc"
                  PRINT_NAME "Dawn native WebGPU procs"
                  OUTPUT_SOURCES WEBGPU_DAWN_NATIVE_PROC_GEN_SOURCES)
target_sources(betann PRIVATE ${WEBGPU_DAWN_NATIVE_PROC_GEN_SOURCES})

# Build tests.
add_executable(hello_world hello_world.cc)
target_link_libraries(hello_world PRIVATE betann)
