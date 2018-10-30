cmake_minimum_required(VERSION 3.1)

include_directories(${PROJECT_SOURCE_DIR}/Console
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Algorithm
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Algorithm/Algorithm
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core/QuantumVirtualMachine
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core/QuantumVirtualMachine/CPUQuantumGates
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core/QuantumMachine
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core/Transform
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.GPUGates
                    ${PROJECT_SOURCE_DIR}/ThirdParty
                    ${PROJECT_SOURCE_DIR}
                    ${PYTHON_PATH_INCLUDE})

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)

if (MSVC)
    set(CompilerFlags
        CMAKE_CXX_FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_C_FLAGS
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        )
    foreach(CompilerFlag ${CompilerFlags})
        string(REPLACE "/MD" "/MT" ${CompilerFlag} "${${CompilerFlag}}")
    endforeach()
endif(MSVC)

option(USE_PYQPANDA "use pyQpanda" OFF)
if(USE_PYQPANDA)
    include_directories(${PROJECT_SOURCE_DIR}/pyQPanda)
    if (UNIX)
        set(PYTHON_PATH_INCLUDE "/usr/include/python3.6m")
        set(PYTHON_PATH_LIB "/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu")
        set(PYHON_LIB "libpython3.6m.so")
    elseif(WIN32)
        set(PYTHON_PATH_INCLUDE "C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python36_64\\include")
        set(PYTHON_PATH_LIB "C:\\Program Files (x86)\\Microsoft Visual Studio\\Shared\\Python36_64\\libs")
        set(PYHON_LIB "python36.lib")
    endif(UNIX)

    include_directories(${PYTHON_PATH_INCLUDE})
    link_directories(${PYTHON_PATH_LIB})
    link_libraries(${PYHON_LIB})
    add_subdirectory(pyQPanda)
endif(USE_PYQPANDA)

find_package(CUDA)
if(CUDA_FOUND)
    option (USE_CUDA "Use cuda" ON)
    include_directories(${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/QPanda-2.0.GPUGates)
    add_subdirectory(QPanda-2.0.GPUGates)
else(CUDA_FOUND)
    option (USE_CUDA "Use cuda" OFF)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        message("OPENMP FOUND")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif(OPENMP_FOUND)
endif(CUDA_FOUND)

configure_file(  
   "${PROJECT_SOURCE_DIR}/config.h.in"
   "${PROJECT_SOURCE_DIR}/config.h")

if(UNIX)
    add_definitions("-std=c++14")
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
    add_compile_options(-fPIC -fpermissive)
endif(UNIX)

add_subdirectory(Console)
add_subdirectory(QPanda-2.0.Algorithm)
add_subdirectory(QPanda-2.0.Core)
add_subdirectory(ThirdParty)


