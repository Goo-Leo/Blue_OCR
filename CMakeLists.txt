cmake_minimum_required(VERSION 3.31)
project(Blue_OCR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenVINO_DIR "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/cmake")
set(OpenCV_DIR "C:/Program Files/opencv/build")
set(CLIPPER2_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/Clipper2Lib")
set(INIH_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/inih")
#set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")

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

target_include_directories(clipper2 PUBLIC ${CLIPPER2_SRC_DIR}/include)
target_link_libraries(Blue_OCR openvino::runtime ${OpenCV_LIBS} clipper2 inih)

#
## Install configuration
#set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")
#
## Install the main executable
#install(TARGETS Blue_OCR
#        RUNTIME DESTINATION bin
#        COMPONENT Runtime
#)
#
## Install OpenCV DLLs (Windows平台需要)
#if (WIN32)
#    # OpenCV DLLs
#    get_target_property(OPENCV_LOCATION opencv_world LOCATION)
#    if (OPENCV_LOCATION)
#        get_filename_component(OPENCV_DLL_DIR ${OPENCV_LOCATION} DIRECTORY)
#        install(DIRECTORY ${OPENCV_DLL_DIR}/
#                DESTINATION bin
#                COMPONENT Runtime
#                FILES_MATCHING PATTERN "opencv_world4110.dll"
#        )
#    endif ()
#
#    # OpenVINO DLLs
#    file(GLOB OPENVINO_DLLS "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/bin/intel64/Release/*.dll")
#    if (OPENVINO_DLLS)
#        install(FILES ${OPENVINO_DLLS}
#                DESTINATION bin
#                COMPONENT Runtime
#        )
#    endif ()
#
#    file(GLOB TBB_DLLS "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/3rdparty/tbb/bin/*.dll")
#    if (TBB_DLLS)
#        install(FILES ${TBB_DLLS}
#                DESTINATION bin
#                COMPONENT Runtime
#        )
#    endif ()
#
#endif ()