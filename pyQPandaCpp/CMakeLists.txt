cmake_minimum_required(VERSION 3.1)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_subdirectory(pyQPanda.Core)
add_subdirectory(pyQPanda.Operator)

if (USE_CHEMIQ)
    add_subdirectory(pyQPanda.ChemiQ)
endif()

