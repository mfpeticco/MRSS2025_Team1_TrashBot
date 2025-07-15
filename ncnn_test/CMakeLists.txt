cmake_minimum_required(VERSION 3.10)
project(yolo_ncnn_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# Find OpenCV via Homebrew
find_package(OpenCV REQUIRED)

# NCNN (Homebrew)
if(NCNN_ROOT)
  include_directories(${NCNN_ROOT}/include)
  link_directories(${NCNN_ROOT}/lib)
else()
  message(FATAL_ERROR "Please set -DNCNN_ROOT=$(brew --prefix ncnn)")
endif()

add_executable(demo main.cpp)

target_link_libraries(demo
  PRIVATE
    ${OpenCV_LIBS}
    ncnn
)