cmake_minimum_required(VERSION 3.1)
project(pyQPandaChemiQ)

file(GLOB_RECURSE PYQPANDA_CHEMIQ_SRC *.cpp)
add_library(${LIB_PYQPANDA_CHEMIQ} SHARED ${PYQPANDA_CHEMIQ_SRC})

if (UNIX)
    if (APPLE)
        set_target_properties(${LIB_PYQPANDA_CHEMIQ} PROPERTIES SUFFIX ".so")
    endif(APPLE)
    set_target_properties(${LIB_PYQPANDA_CHEMIQ} PROPERTIES PREFIX "")
elseif(WIN32)
    if (NOT MSVC)
        set_target_properties(${LIB_PYQPANDA_CHEMIQ} PROPERTIES PREFIX "")
    endif()
    set_target_properties(${LIB_PYQPANDA_CHEMIQ} PROPERTIES SUFFIX ".pyd")
endif(UNIX)

target_link_libraries(${LIB_PYQPANDA_CHEMIQ} ${LIB_QALG} ${PYTHON_LIBRARIES})

set(PYCHEMIQ_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../pyQPanda/pyqpanda/ChemiQ")
if (UNIX)
    set(PYCHEMIQ_OUT_LIB "${LIB_PYQPANDA_CHEMIQ}.so")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
elseif (WIN32)
    set(PYCHEMIQ_OUT_LIB "${LIB_PYQPANDA_CHEMIQ}.pyd")
endif(UNIX)

add_custom_command(OUTPUT ${PYCHEMIQ_OUT_LIB}
   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/${PYCHEMIQ_OUT_LIB} ${PYCHEMIQ_OUT_PATH}/${PYCHEMIQ_OUT_LIB}
 )
add_custom_target(CopyQPandaTaskChemiQ ALL DEPENDS ${PYCHEMIQ_OUT_LIB} ${LIB_PYQPANDA_CHEMIQ})

# generate stub file for IDE intelliSense after target CopyQPandaTaskChemiQ finished
if(PYQPANDA_STUBGEN)
    add_custom_command(TARGET CopyQPandaTaskChemiQ
        POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} ${PYQPANDA_POST_BUILD_TOOL_DIR}/stubgen/stubgen.py --search-path ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE} -m ${LIB_PYQPANDA_CHEMIQ} -o ${PYCHEMIQ_OUT_PATH}
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}
        VERBATIM
        USES_TERMINAL
    )


endif(PYQPANDA_STUBGEN)
