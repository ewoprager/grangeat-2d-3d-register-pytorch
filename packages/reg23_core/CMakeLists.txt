cmake_minimum_required(VERSION 3.18)

# includes
include(CheckLanguage)

# Python3_EXECUTABLE must be set
if (NOT DEFINED Python3_EXECUTABLE OR Python3_EXECUTABLE STREQUAL "")
    message(FATAL_ERROR "Python3_EXECUTABLE is empty or not defined")
else ()
    message(STATUS "Python3_EXECUTABLE is '${Python3_EXECUTABLE}'")
endif ()

if (DEFINED CUDA_HOME)
    set(CMAKE_CUDA_COMPILER
        "${CUDA_HOME}/bin/nvcc"
        CACHE FILEPATH ""
    )
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(reg23-core LANGUAGES CXX)

find_package(Python COMPONENTS Development REQUIRED)

# sources
file(GLOB HEADERS
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS
     "src/backend/include/reg23_core/*.h"
)
file(GLOB SOURCES
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS
     "src/backend/cpu/*.cpp"
     "src/backend/*.cpp"
)

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found; building with CUDA support.")
    # CUDA-specific sources
    file(GLOB CUDA_SOURCES
         RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         CONFIGURE_DEPENDS
         "src/backend/cuda/*.cu"
    )
    # Get the PyTorch include directories for C++ and CUDA extensions
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths() + ['${CUDA_HOME}' + '/include'])"
                    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE RESULT
    )
else ()
    message(STATUS "CUDA not found. If you have CUDA installed, make sure to set the 'CUDA_HOME' environment variable.")
    # Get the PyTorch include directories for just C++ extensions
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths())"
                    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE RESULT
    )
endif ()

string(COMPARE EQUAL ${RESULT} "0" PROCESS_SUCCESS)
if (NOT PROCESS_SUCCESS)
    message(FATAL_ERROR "Failed to get torch.utils.cpp_extension.include_paths(): ${RESULT}")
endif ()
string(REPLACE "[" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "]" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "'" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "," ";" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})

if (CMAKE_CUDA_COMPILER)
    include_directories("${CMAKE_SOURCE_DIR}/src/backend/include"
                        ${TORCH_INCLUDE_DIRS}
                        ${Python_INCLUDE_DIRS}
                        "${CUDA_HOME}/include")

    add_library(${PROJECT_NAME} SHARED
                ${SOURCES}
                ${CUDA_SOURCES}
                ${HEADERS})

    set_target_properties(${PROJECT_NAME} PROPERTIES
                          CUDA_SEPARABLE_COMPILATION ON
    )

    # This will define __CUDACC__, so your IDE will check the code as though it will compile with NVCC. Comment this line
    # out to have it check your code as though it will compile with GCC / CLA.
    target_compile_definitions(${PROJECT_NAME} PRIVATE
                               __CUDACC__
    )
else ()
    include_directories("${CMAKE_SOURCE_DIR}/src/backend/include"
                        ${TORCH_INCLUDE_DIRS}
                        ${Python_INCLUDE_DIRS})

    add_library(${PROJECT_NAME} SHARED
                ${SOURCES}
                ${HEADERS})
endif ()

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
