cmake_minimum_required(VERSION 3.1)
project(Components)

file(GLOB_RECURSE COMPONENTS_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/HamiltonianSimulation/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/MaxCutProblemGenerator/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/NodeSortProblemGenerator/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/Operator/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.cpp)

if (USE_CHEMIQ)
    # use pybind11 find PythonInterp anda PythonLibs
    #find_package(PythonInterp 3 REQUIRED)
    #find_package(PythonLibs 3 REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_library(${LIB_COMPONENTS} STATIC ${COMPONENTS_SRC_CPP})
    target_link_libraries(${LIB_COMPONENTS} ${LIB_QPANDA} ${LIB_NLOPT})
else ()
    add_library(${LIB_COMPONENTS} STATIC ${COMPONENTS_SRC_CPP})
    target_link_libraries(${LIB_COMPONENTS} ${LIB_QPANDA} ${LIB_NLOPT})
endif()


INSTALL(TARGETS ${LIB_COMPONENTS}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${LIB_COMPONENTS} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda2>)
        
export(TARGETS ${LIB_COMPONENTS}
       APPEND FILE ${PROJECT_BINARY_DIR}/qalg-target.cmake
       NAMESPACE QPANDA::
       )




