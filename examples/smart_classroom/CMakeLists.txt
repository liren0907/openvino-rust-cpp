cmake_minimum_required(VERSION 3.10)
project(smart_classroom_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# --- macOS Homebrew path hints ---
if(APPLE)
    set(gflags_DIR "/opt/homebrew/lib/cmake/gflags" CACHE PATH "Path to gflags CMake config")
endif()

# Import the openvino_vision library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../cpp openvino_vision_build)

# Find gflags (only needed by the example, not the library)
find_package(gflags REQUIRED)

# Build the example executable
add_executable(smart_classroom_example main.cpp)
target_link_libraries(smart_classroom_example PRIVATE openvino_vision ${gflags_LIBRARIES})
