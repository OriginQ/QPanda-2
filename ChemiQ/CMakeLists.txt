cmake_minimum_required(VERSION 3.1)
project(ChemiQ)

find_package(PythonInterp 3)
find_package(PythonLibs 3)
include_directories(${PYTHON_INCLUDE_DIRS})

file(GLOB_RECURSE ChemiQ_SRCS_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
add_library(${LIB_CHEMIQ} STATIC ${ChemiQ_SRCS_CPP} )
target_link_libraries(${LIB_CHEMIQ} ${LIB_QPANDA} ${LIB_QALG} ${LIB_VAR} ${PYTHON_LIBRARIES})

