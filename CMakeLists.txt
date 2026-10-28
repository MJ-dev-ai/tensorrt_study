cmake_minimum_required(VERSION 3.24)

project(tensorrt_study LANGUAGES CXX)

set(TENSORRT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/TensorRT-11.2.1.2-Windows-amd64-cuda-13.3/TensorRT-11.2.1.2"
    CACHE PATH "TensorRT SDK directory"
)

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS "${TENSORRT_ROOT}/include"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(TENSORRT_LIBRARY
    NAMES nvinfer_11 nvinfer
    PATHS "${TENSORRT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(TENSORRT_ONNX_LIBRARY
    NAMES nvonnxparser_11 nvonnxparser
    PATHS "${TENSORRT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

add_executable(inference_once src/inference_once.cpp src/cv/cv.cpp include/cv/cv.hpp)

target_compile_features(inference_once PRIVATE cxx_std_17)
target_compile_definitions(inference_once PRIVATE PROJECT_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(inference_once PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/externals/stb"
    "${TENSORRT_INCLUDE_DIR}"
)

target_link_libraries(inference_once PRIVATE
    "${TENSORRT_LIBRARY}"
    "${TENSORRT_ONNX_LIBRARY}"
    CUDA::cudart_static
)

if(MSVC)
    target_compile_options(inference_once PRIVATE /W4 /utf-8 /EHsc)
endif()

set_target_properties(inference_once PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<0:>"
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

file(GLOB TENSORRT_DLLS CONFIGURE_DEPENDS
    "${TENSORRT_ROOT}/bin/*.dll"
    "${TENSORRT_ROOT}/lib/*.dll"
)

if(NOT TENSORRT_DLLS)
    message(FATAL_ERROR "No TensorRT DLLs found in ${TENSORRT_ROOT}/bin or ${TENSORRT_ROOT}/lib")
endif()

add_custom_command(TARGET inference_once POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory
        "$<TARGET_FILE_DIR:inference_once>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        ${TENSORRT_DLLS}
        "$<TARGET_FILE_DIR:inference_once>"
    COMMAND_EXPAND_LISTS
    VERBATIM
)