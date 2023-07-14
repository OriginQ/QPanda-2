cmake_minimum_required(VERSION 3.1)

set(PYQPANDA_POST_BUILD_TOOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../pyQPanda/postbuildtool")
set(PYQPANDA_STUBGEN ${PYQPANDA_POST_BUILD_TOOL_DIR}/stubgen/stubgen.py)
# try pip install package required by stubgen on cmake config runtime
if(PYQPANDA_STUBGEN)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install -r ${PYQPANDA_POST_BUILD_TOOL_DIR}/stubgen/requirements.txt)
endif(PYQPANDA_STUBGEN)

add_subdirectory(pyQPanda.Core)
add_subdirectory(pyQPanda.Operator)
   
       
