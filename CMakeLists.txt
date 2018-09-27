cmake_minimum_required(VERSION 2.8)
if (UNIX)
    set(PYTHON_PATH_INCLUDE "/usr/local/include/python3.6/")
	set(PYTHON_PATH_LIB "/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu/")
elseif(WIN32)
    set(PYTHON_PATH_INCLUDE "F:\\Python\\Python35\\include")
	set(PYTHON_PATH_LIB "F:\\Python\\Python35\\libs")
endif(UNIX)
include_directories(${PROJECT_SOURCE_DIR}/Console
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Algorithm
                    ${PROJETC_SOURCE_DIR}/QPanda-2.0.Algorithm/Algorithm
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core/QuantumVirtualMachine
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core/QuantumMachin
                    ${PROJECT_SOURCE_DIR}/QPanda-2.0.Core/Transform
                    ${PROJECT_SOURCE_DIR}/ThirdParty
                    ${PROJECT_SOURCE_DIR}/pyQPanda
                    ${PROJECT_SOURCE_DIR}/pyQPanda/pybind11
                    ${PYTHON_PATH_INCLUDE})

if (UNIX)
    add_definitions("-std=c++14")
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
	add_compile_options(-fPIC -fpermissive)
endif(UNIX)

link_directories(${PYTHON_PATH_LIB})

if(UNIX)
    set(PYHON_LIB "libpython3.6m.so")
elseif(WIN32)
    set(PYHON_LIB "python35.lib")	
endif(UNIX)
link_libraries(${PYHON_LIB})	

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(Console)
add_subdirectory(QPanda-2.0.Algorithm)
add_subdirectory(QPanda-2.0.Core)
add_subdirectory(ThirdParty)
add_subdirectory(pyQPanda)

