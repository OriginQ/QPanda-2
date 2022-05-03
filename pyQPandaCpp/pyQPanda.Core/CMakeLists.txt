cmake_minimum_required(VERSION 3.1)
project(pyQPanda)

set(LIB_PYQPANDA pyQPanda)
file(GLOB_RECURSE pyQPanda_SRC_CPP *.cpp)
add_library(${LIB_PYQPANDA} SHARED ${pyQPanda_SRC_CPP})
if (UNIX)
    if (APPLE)
        set_target_properties(${LIB_PYQPANDA} PROPERTIES SUFFIX ".so")
    endif(APPLE)
    set_target_properties(${LIB_PYQPANDA} PROPERTIES PREFIX "")
elseif(WIN32)
    if (NOT MSVC)
        set_target_properties(${LIB_PYQPANDA} PROPERTIES PREFIX "")
    endif()
    set_target_properties(${LIB_PYQPANDA} PROPERTIES SUFFIX ".pyd")
endif(UNIX)

target_link_libraries(${LIB_PYQPANDA} ${LIB_QPANDA} ${LIB_QALG} ${LIB_VAR} ${PYTHON_LIBRARIES})
set(PYQPANDA_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../pyQPanda/pyqpanda")
if (UNIX)
    set(PYQPANDA_OUT_LIB "${LIB_PYQPANDA}.so")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
elseif (WIN32)
    set(PYQPANDA_OUT_LIB "${LIB_PYQPANDA}.pyd")
endif(UNIX)

add_custom_command(OUTPUT ${PYQPANDA_OUT_LIB}
   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/${PYQPANDA_OUT_LIB} ${PYQPANDA_OUT_PATH}/${PYQPANDA_OUT_LIB}
 )
add_custom_target(CopyQPandaTask ALL DEPENDS ${PYQPANDA_OUT_LIB} ${LIB_PYQPANDA})

if (USE_CURL AND CURL_FOUND AND MSVC)
	add_custom_command(TARGET ${LIB_PYQPANDA} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CURL_DLL} ${PYQPANDA_OUT_PATH}/)
endif()

# generate stub file for IDE intelliSense after target CopyQPandaTask finished
if(PYQPANDA_STUBGEN)
    add_custom_command(TARGET CopyQPandaTask
        POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} ${PYQPANDA_POST_BUILD_TOOL_DIR}/stubgen/stubgen.py --search-path ${PYQPANDA_OUT_PATH} -m ${LIB_PYQPANDA} -o ${PYQPANDA_OUT_PATH}
        WORKING_DIRECTORY ${PYQPANDA_OUT_PATH}
        VERBATIM
        USES_TERMINAL
    )
endif(PYQPANDA_STUBGEN)