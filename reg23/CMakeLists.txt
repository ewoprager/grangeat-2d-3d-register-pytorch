cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Try to find CUDA
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found! Building with CUDA support.")
endif ()

file(GLOB HEADERS
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS
     "src/backend/include/reg23/*.h"
)

file(GLOB SOURCES
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS
     "src/backend/cpu/*.cpp"
     "src/backend/*.cpp"
)

if (CUDAToolkit_FOUND)
    file(GLOB CUDA_SOURCES
         RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         CONFIGURE_DEPENDS
         "src/backend/cuda/*.cu"
    )

    message(STATUS "Python3_EXECUTABLE is '${Python3_EXECUTABLE}'")

    # Get the location of CUDA
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.CUDA_HOME)"
                    OUTPUT_VARIABLE CUDA_HOME
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE RESULT
    )

    message(STATUS "CUDA_HOME is '${CUDA_HOME}'")

    string(COMPARE EQUAL ${RESULT} "0" PROCESS_SUCCESS)
    if (NOT PROCESS_SUCCESS)
        message(FATAL_ERROR "Failed to get torch.utils.cpp_extension.CUDA_HOME: ${RESULT}")
    endif ()

    # Set the CUDA compiler path
    set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc")
endif ()

if (CUDAToolkit_FOUND)
    project(reg23
            LANGUAGES CXX CUDA)

    # Get the PyTorch include directories
    execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths() + [cpp_extension.CUDA_HOME + '/include'])"
                    OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE RESULT
    )

    set(CMAKE_CUDA_STANDARD 17)
else ()
    project(reg23 LANGUAGES CXX)

    # Get the PyTorch include directories
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


find_package(Python COMPONENTS Development REQUIRED)

if (CUDAToolkit_FOUND)
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
