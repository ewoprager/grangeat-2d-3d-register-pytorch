cmake_minimum_required(VERSION 3.18)

# Get the location of CUDA
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.CUDA_HOME)"
                OUTPUT_VARIABLE CUDA_HOME
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE RESULT
)
string(COMPARE EQUAL ${RESULT} "0" PROCESS_SUCCESS)
if (NOT PROCESS_SUCCESS)
    message(FATAL_ERROR "Failed to get torch.utils.cpp_extension.CUDA_HOME: ${RESULT}")
endif ()

# Set the CUDA compiler path
set(CMAKE_CUDA_COMPILER "${CUDA_HOME}/bin/nvcc")

project(Extension
        LANGUAGES CXX CUDA)

# Get the PyTorch include directories
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths() + [cpp_extension.CUDA_HOME + '/include'])"
                OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE RESULT
)
string(COMPARE EQUAL ${RESULT} "0" PROCESS_SUCCESS)
if (NOT PROCESS_SUCCESS)
    message(FATAL_ERROR "Failed to get torch.utils.cpp_extension.include_paths(): ${RESULT}")
endif ()
string(REPLACE "[" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "]" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "'" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "," ";" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})

include_directories(${TORCH_INCLUDE_DIRS}
                    "${CUDA_HOME}/include")

add_library(${PROJECT_NAME} SHARED mymuladd.cpp muladd.cu)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)