cmake_minimum_required(VERSION 3.15)

project(Extension)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; from torch.utils import cpp_extension; print(cpp_extension.include_paths())"
        OUTPUT_VARIABLE TORCH_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
#        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../.venv/bin
        RESULT_VARIABLE RESULT
)

string(COMPARE EQUAL ${RESULT} "0" PROCESS_SUCCESS)
if(NOT PROCESS_SUCCESS)
    message(FATAL_ERROR "Failed to get torch.utils.cpp_extension.include_paths(): ${RESULT}")
endif()

string(REPLACE "[" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "]" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "'" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "," ";" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})

include_directories(${TORCH_INCLUDE_DIRS})

add_library(${PROJECT_NAME} SHARED mymuladd.cpp)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)