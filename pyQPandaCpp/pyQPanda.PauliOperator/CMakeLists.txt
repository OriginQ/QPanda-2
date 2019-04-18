cmake_minimum_required(VERSION 3.1)
project(pyQPandaPauliOperator)

set(LIB_PYQPANDA_PAUL pyQPandaPauliOperator)
file(GLOB_RECURSE pyQPanda_SRC_CPP *.cpp)
add_library(${LIB_PYQPANDA_PAUL} SHARED ${pyQPanda_SRC_CPP})

if (UNIX)
    if (APPLE)
        set_target_properties(${LIB_PYQPANDA_PAUL} PROPERTIES SUFFIX ".so")
    endif(APPLE)
    set_target_properties(${LIB_PYQPANDA_PAUL} PROPERTIES PREFIX "")
elseif(WIN32)
    if (NOT MSVC)
        set_target_properties(${LIB_PYQPANDA_PAUL} PROPERTIES PREFIX "")
    endif()
    set_target_properties(${LIB_PYQPANDA_PAUL} PROPERTIES SUFFIX ".pyd")
endif(UNIX)

target_link_libraries(${LIB_PYQPANDA_PAUL} ${LIB_QPANDA} ${LIB_QALG} ${LIB_VAR} ${PYTHON_LIBRARIES})

set(PYQPANDA_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../pyQPanda/pyqpanda/Hamiltonian/PauliOperator")
if (UNIX)
    set(PYQPANDA_OUT_LIB "${LIB_PYQPANDA_PAUL}.so")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${LIBRARY_OUTPUT_PATH})
elseif (WIN32)
    set(PYQPANDA_OUT_LIB "${LIB_PYQPANDA_PAUL}.pyd")
endif(UNIX)

add_custom_command(OUTPUT ${PYQPANDA_OUT_LIB}
   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/${PYQPANDA_OUT_LIB} ${PYQPANDA_OUT_PATH}/${PYQPANDA_OUT_LIB}
 )
add_custom_target(CopyQPandaTaskPauliOperator ALL DEPENDS ${PYQPANDA_OUT_LIB} ${LIB_PYQPANDA_PAUL})
