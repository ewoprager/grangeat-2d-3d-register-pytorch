cmake_minimum_required(VERSION 3.18)

# Try to find CUDA
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA found! Building with CUDA support.")
endif ()


file(GLOB HEADERS
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS
     "include/**/*.h"
)

file(GLOB_RECURSE SOURCES
     RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
     CONFIGURE_DEPENDS
     "src/cpu/*.cpp"
     "src/*.cpp"
)

if (CUDAToolkit_FOUND)
    file(GLOB CUDA_SOURCES
         RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
         CONFIGURE_DEPENDS
         "src/cuda/*.cu"
    )

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

if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    enable_language(OBJCXX)

    find_library(METAL_LIB Metal)
    find_library(FOUNDATION_LIB Foundation)

    if (METAL_LIB AND FOUNDATION_LIB)
        message(STATUS "METAL_LIB and FOUNDATION_LIB both found! Building with MPS support.")
        file(GLOB MPS_SOURCES
             RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
             CONFIGURE_DEPENDS
             "src/mps/*.mm"
             "src/mps/shaders/*.metal"
        )
    endif ()
endif ()

string(COMPARE EQUAL ${RESULT} "0" PROCESS_SUCCESS)
if (NOT PROCESS_SUCCESS)
    message(FATAL_ERROR "Failed to get torch.utils.cpp_extension.include_paths(): ${RESULT}")
endif ()
string(REPLACE "[" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "]" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "'" "" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
string(REPLACE "," ";" TORCH_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})


if (CUDAToolkit_FOUND)
    include_directories("${CMAKE_SOURCE_DIR}/include"
                        ${TORCH_INCLUDE_DIRS}
                        "${CUDA_HOME}/include")


    if (METAL_LIB AND FOUNDATION_LIB)
        add_library(${PROJECT_NAME} SHARED
                    ${SOURCES}
                    ${CUDA_SOURCES}
                    ${MPS_SOURCES}
                    ${HEADERS})
    else ()
        add_library(${PROJECT_NAME} SHARED
                    ${SOURCES}
                    ${CUDA_SOURCES}
                    ${HEADERS})
    endif ()


    set_target_properties(${PROJECT_NAME} PROPERTIES
                          CUDA_SEPARABLE_COMPILATION ON
    )

    # This will define __CUDACC__, so your IDE will check the code as though it will compile with NVCC. Comment this line
    # out to have it check your code as though it will compile with GCC / CLA.
    target_compile_definitions(${PROJECT_NAME} PRIVATE
                               __CUDACC__
                               USE_CUDA
    )
else ()
    include_directories("${CMAKE_SOURCE_DIR}/include"
                        ${TORCH_INCLUDE_DIRS})

    if (METAL_LIB AND FOUNDATION_LIB)
        add_library(${PROJECT_NAME} SHARED
                    ${SOURCES}
                    ${MPS_SOURCES}
                    ${HEADERS})
    else ()
        add_library(${PROJECT_NAME} SHARED
                    ${SOURCES}
                    ${HEADERS})
    endif ()
endif ()

if (METAL_LIB AND FOUNDATION_LIB)
    target_link_libraries(${PROJECT_NAME}
                          PRIVATE
                          ${METAL_LIB}
                          ${FOUNDATION_LIB}
    )
    target_compile_options(${PROJECT_NAME} PRIVATE
                           -ObjC++       # ensure Objective-C++ mode
                           -fobjc-arc    # use ARC for Objective-C memory management
    )
    target_compile_definitions(${PROJECT_NAME} PRIVATE
                               USE_MPS
    )

endif ()

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)