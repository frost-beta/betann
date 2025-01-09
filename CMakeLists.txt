cmake_minimum_required(VERSION 3.23)

project(betann)

# Use C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the BetaNN library.
add_library(betann STATIC)
target_sources(betann PRIVATE src/device.cc
                      PUBLIC FILE_SET HEADERS
                             BASE_DIRS src
                             FILES src/betann.h)
target_include_directories(betann PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Link with dawn.
set(DAWN_FETCH_DEPENDENCIES ON)
set(DAWN_USE_GLFW OFF)
set(DAWN_USE_X11 OFF)
set(DAWN_USE_WINDOWS_UI OFF)
set(DAWN_BUILD_SAMPLES OFF)
set(TINT_BUILD_CMD_TOOLS OFF)
set(TINT_BUILD_TESTS OFF)
add_subdirectory(dawn)
target_link_libraries(betann PRIVATE dawn::dawn_common
                                     dawn::dawn_native
                                     dawn::dawn_wgpu_utils
                             PUBLIC dawncpp_headers
                                    dawncpp)

# Build the procs that call dawn_native directly.
find_package(Python3 REQUIRED)
DawnJSONGenerator(TARGET "webgpu_dawn_native_proc"
                  PRINT_NAME "Dawn native WebGPU procs"
                  OUTPUT_SOURCES WEBGPU_DAWN_NATIVE_PROC_GEN_SOURCES)
target_sources(betann PRIVATE ${WEBGPU_DAWN_NATIVE_PROC_GEN_SOURCES})

# Build tests.
add_executable(hello_world hello_world.cc)
target_link_libraries(hello_world PRIVATE betann)
