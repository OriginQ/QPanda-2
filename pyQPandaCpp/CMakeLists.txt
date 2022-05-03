cmake_minimum_required(VERSION 3.1)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

set(PYQPANDA_POST_BUILD_TOOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pyQPanda/postbuildtool")
# try pip install package required by stubgen on cmake config runtime
if(PYQPANDA_STUBGEN)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install -r ${PYQPANDA_POST_BUILD_TOOL_DIR}/stubgen/requirements.txt)
endif(PYQPANDA_STUBGEN)

add_subdirectory(pyQPanda.Core)
add_subdirectory(pyQPanda.Operator)

if (USE_CHEMIQ)
    add_subdirectory(pyQPanda.ChemiQ)
endif()

