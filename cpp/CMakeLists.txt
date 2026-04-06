cmake_minimum_required(VERSION 3.10)
project(ovi_vision CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- macOS Homebrew path hints ---
if(APPLE)
    set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv4" CACHE PATH "Path to OpenCV CMake config")
    set(OpenVINO_DIR "/opt/homebrew/opt/openvino/lib/cmake/openvino2025.0.0" CACHE PATH "Path to OpenVINO CMake config")
endif()

# --- Find Required Packages ---
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs)
find_package(OpenVINO REQUIRED)
# No gflags dependency — that belongs in applications, not the library

# --- Source Files ---
set(SOURCES
    src/action_detector.cpp
    src/align_transform.cpp
    src/cnn.cpp
    src/detector.cpp
    src/images_capture.cpp
    src/logger.cpp
    src/reid_gallery.cpp
    src/tracker.cpp
)

# --- Define Static Library ---
add_library(ovi_vision STATIC ${SOURCES})

# --- Public Include Directory ---
target_include_directories(ovi_vision PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --- Link Libraries (PUBLIC so consumers inherit them) ---
target_link_libraries(ovi_vision PUBLIC
    ${OpenCV_LIBS}
    openvino::runtime
)

message(STATUS "Configuring ovi_vision library")
message(STATUS "Sources: ${SOURCES}")
