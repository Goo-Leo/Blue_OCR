cmake_minimum_required(VERSION 3.31)
project(Blue_OCR)

#if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#    add_compile_options(-O3 -march=native -ffast-math)
#elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")

#endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenVINO_DIR "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/cmake")
set(OpenCV_DIR "C:/Program Files/opencv/build")
set(CLIPPER2_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/Clipper2Lib")
set(INIH_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/inih")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS} INTERFACE ${PROJECT_SOURCE_DIR}/include thirdparty/inih)


add_library(clipper2 STATIC
        ${CLIPPER2_SRC_DIR}/src/clipper.engine.cpp
        ${CLIPPER2_SRC_DIR}/src/clipper.offset.cpp
        ${CLIPPER2_SRC_DIR}/src/clipper.rectclip.cpp
)

add_library(inih STATIC
        ${INIH_DIR}/INIReader.cpp
        ${INIH_DIR}/ini.c
)

add_executable(Blue_OCR WIN32
        source/main.cpp
        source/infer.cpp
        source/interface.cpp
)

#target_precompile_headers(Blue_OCR PRIVATE "pch.h")
target_include_directories(clipper2 PUBLIC ${CLIPPER2_SRC_DIR}/include)
target_link_libraries(Blue_OCR openvino::runtime ${OpenCV_LIBS} clipper2 inih)


if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(Blue_OCR PRIVATE /O2 /arch:AVX2 /fp:fast /MD /Zi)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(Blue_OCR PRIVATE /Od /RTC1 /MDd /Zi)
    endif()
endif()
