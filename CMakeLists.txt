cmake_minimum_required(VERSION 3.31)
project(Blue_OCR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenVINO_DIR "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/cmake")
set(OpenCV_DIR "C:/Program Files/opencv/build")
#set(OpenCV_DEBUG ON)
set(CLIPPER2_SRC_DIR "C:/Users/10633/CLionProjects/Blue_OCR/third_party/Clipper2Lib")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS}  INTERFACE )

add_library(clipper2 STATIC
        ${CLIPPER2_SRC_DIR}/src/clipper.engine.cpp
        ${CLIPPER2_SRC_DIR}/src/clipper.offset.cpp
        ${CLIPPER2_SRC_DIR}/src/clipper.rectclip.cpp
)

add_executable(Blue_OCR
        ./main.cpp
        include/infer.h
        source/infer.cpp
        include/interface.h
        source/interface.cpp
)

target_include_directories(clipper2 PUBLIC
        ${CLIPPER2_SRC_DIR}/include
)

target_link_libraries(Blue_OCR openvino::runtime ${OpenCV_LIBS} clipper2)