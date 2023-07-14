cmake_minimum_required(VERSION 3.1)
project(pyQPanda)

set(LIB_PYQPANDA pyQPanda)
file(GLOB_RECURSE pyQPanda_SRC_CPP *.cpp)

file(GLOB_RECURSE COMPONENTS_CPP ${CMAKE_SOURCE_DIR}/Components/HamiltonianSimulation/*.cpp
                                     ${CMAKE_SOURCE_DIR}/Components/MaxCutProblemGenerator/*.cpp
                                     ${CMAKE_SOURCE_DIR}/Components/NodeSortProblemGenerator/*.cpp
                                     ${CMAKE_SOURCE_DIR}/Components/Optimizer/*.cpp
                                     ${CMAKE_SOURCE_DIR}/Components/Operator/*.cpp
                                     ${CMAKE_SOURCE_DIR}/Components/Utils/*.cpp)


pybind11_add_module(${LIB_PYQPANDA} SHARED ${pyQPanda_SRC_CPP})
set_target_properties(${LIB_PYQPANDA}  PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
target_link_libraries(${LIB_PYQPANDA} PRIVATE ${LIB_QPANDA} ${LIB_QALG} ${LIB_VAR})

set(PYQPANDA_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../pyQPanda/pyqpanda")
set(PYQPANDA_OUT_LIB
        "${PYTHON_MODULE_PREFIX}${LIB_PYQPANDA}${PYTHON_MODULE_EXTENSION}")

if (USE_CURL AND CURL_FOUND AND MSVC)
    add_custom_command(TARGET ${LIB_PYQPANDA} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${CURL_DLL} ${PYQPANDA_OUT_PATH})                    
endif()

if (USE_OPENSSL AND OPENSSL_FOUND AND MSVC)
    add_custom_command(TARGET ${LIB_PYQPANDA} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy ${OPENSSL_DLL} ${PYQPANDA_OUT_PATH})                  
endif()

add_custom_command(OUTPUT COPY_RES
        COMMAND ${CMAKE_COMMAND} -E copy ${EXECUTABLE_OUTPUT_PATH}/${PYQPANDA_OUT_LIB} ${PYQPANDA_OUT_PATH})

add_custom_target(CopyTask ALL DEPENDS ${LIB_PYQPANDA} COPY_RES)

# generate stub file for IDE intelliSense after target CopyTask finished
if(PYQPANDA_STUBGEN)
    add_custom_command(TARGET CopyTask
        POST_BUILD
        COMMAND ${PYTHON_EXECUTABLE} ${PYQPANDA_STUBGEN} -m ${LIB_PYQPANDA} -o ${PYQPANDA_OUT_PATH}
        WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
        VERBATIM
        USES_TERMINAL
        )
endif(PYQPANDA_STUBGEN)
