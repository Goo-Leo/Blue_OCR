cmake_minimum_required(VERSION 3.31)
project(Blue_OCR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenVINO_DIR "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/cmake")
set(OpenCV_DIR "C:/Program Files/opencv/build")
set(CLIPPER2_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/third_party/Clipper2Lib")

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "version: ${OpenCV_VERSION}")
message(STATUS "libraries: ${OpenCV_LIBS}")
message(STATUS "include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS}  INTERFACE ${PROJECT_SOURCE_DIR}/Include)


add_executable(Blue_OCR
        ./main.cpp
        source/infer.cpp
        source/interface.cpp
)

add_library(clipper2 STATIC
        ${CLIPPER2_SRC_DIR}/src/clipper.engine.cpp
        ${CLIPPER2_SRC_DIR}/src/clipper.offset.cpp
        ${CLIPPER2_SRC_DIR}/src/clipper.rectclip.cpp
)
target_include_directories(clipper2 PUBLIC
        ${CLIPPER2_SRC_DIR}/include
)

target_link_libraries(Blue_OCR openvino::runtime ${OpenCV_LIBS} clipper2)

# Install configuration
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory")

# Install the main executable
install(TARGETS Blue_OCR
        RUNTIME DESTINATION bin
        COMPONENT Runtime
)

# Install the static library (如果需要分发clipper2库)
install(TARGETS clipper2
        ARCHIVE DESTINATION lib
        COMPONENT Development
)

## Install clipper2 headers (如果需要分发头文件)
#install(DIRECTORY ${CLIPPER2_SRC_DIR}/include/
#        DESTINATION include/clipper2
#        COMPONENT Development
#        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
#)
#
## Install project headers (如果有公共头文件需要分发)
#if(EXISTS "${PROJECT_SOURCE_DIR}/Include")
#    install(DIRECTORY ${PROJECT_SOURCE_DIR}/Include/
#            DESTINATION include
#            COMPONENT Development
#            FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
#    )
#endif()

# Install OpenCV DLLs (Windows平台需要)
if(WIN32)
    # OpenCV DLLs
    get_target_property(OPENCV_LOCATION opencv_world LOCATION)
    if(OPENCV_LOCATION)
        get_filename_component(OPENCV_DLL_DIR ${OPENCV_LOCATION} DIRECTORY)
        install(DIRECTORY ${OPENCV_DLL_DIR}/
                DESTINATION bin
                COMPONENT Runtime
                FILES_MATCHING PATTERN "opencv_world*.dll"
        )
    else()
        # 如果无法自动找到，手动指定OpenCV DLL路径
        file(GLOB OPENCV_DLLS "C:/Program Files/opencv/build/x64/vc*/bin/*.dll")
        if(OPENCV_DLLS)
            install(FILES ${OPENCV_DLLS}
                    DESTINATION bin
                    COMPONENT Runtime
            )
        endif()
    endif()

    # OpenVINO DLLs
    file(GLOB OPENVINO_DLLS "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/bin/intel64/Release/*.dll")
    if(OPENVINO_DLLS)
        install(FILES ${OPENVINO_DLLS}
                DESTINATION bin
                COMPONENT Runtime
        )
    endif()

    # TBB DLLs (OpenVINO依赖)
    file(GLOB TBB_DLLS "C:/Program Files (x86)/Intel/openvino_2025.1.0/runtime/3rdparty/tbb/bin/*.dll")
    if(TBB_DLLS)
        install(FILES ${TBB_DLLS}
                DESTINATION bin
                COMPONENT Runtime
        )
    endif()
endif()

# 创建安装组件
set(CPACK_COMPONENTS_ALL Runtime Development)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "Blue OCR Runtime")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Blue OCR executable and required DLLs")
set(CPACK_COMPONENT_DEVELOPMENT_DISPLAY_NAME "Blue OCR Development")
set(CPACK_COMPONENT_DEVELOPMENT_DESCRIPTION "Headers and libraries for development")

include(CPack)

# 添加自定义install target用于VS
if(MSVC)
    set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT Blue_OCR)
endif()