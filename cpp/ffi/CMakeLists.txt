cmake_minimum_required(VERSION 3.10)
project(openvino_vision_ffi CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- CXX bridge parameters (set by build.rs via cmake crate) ---
if(NOT DEFINED CXX_INCLUDE_DIR)
    message(FATAL_ERROR "CXX_INCLUDE_DIR not defined.")
endif()
if(NOT DEFINED CXX_GENERATED_SOURCE)
    message(FATAL_ERROR "CXX_GENERATED_SOURCE not defined.")
endif()
if(NOT DEFINED OPENVINO_VISION_DIR)
    message(FATAL_ERROR "OPENVINO_VISION_DIR not defined.")
endif()

# --- Import the openvino_vision library ---
add_subdirectory(${OPENVINO_VISION_DIR} openvino_vision_build)

# --- Define FFI Library ---
add_library(openvino_vision_ffi STATIC
    bridge.cpp
    ${CXX_GENERATED_SOURCE}
)

target_include_directories(openvino_vision_ffi PUBLIC
    ${CXX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..   # for bridge.cpp local includes
    ${OPENVINO_VISION_DIR}/include   # for openvino_vision/*.hpp and openvino_vision/ffi/bridge.hpp
)

# Link core library (PUBLIC transitively provides OpenCV, OpenVINO)
target_link_libraries(openvino_vision_ffi PUBLIC openvino_vision)

set_target_properties(openvino_vision_ffi PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

install(TARGETS openvino_vision_ffi DESTINATION lib)

message(STATUS "Configuring openvino_vision_ffi (CXX bridge for Rust)")
