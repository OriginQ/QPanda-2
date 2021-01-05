cmake_minimum_required(VERSION 3.1)
project(Components)

file(GLOB_RECURSE COMPONENTS_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/HamiltonianSimulation/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/MaxCutProblemGenerator/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/NodeSortProblemGenerator/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/Operator/*.cpp
                                     ${CMAKE_CURRENT_SOURCE_DIR}/Utils/*.cpp)

file(GLOB_RECURSE COMPONENTS_CHEMIQ_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/ChemiQ/*.cpp)

if (USE_CHEMIQ)
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs 3 REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    add_library(${LIB_COMPONENTS} STATIC ${COMPONENTS_SRC_CPP} ${COMPONENTS_CHEMIQ_SRC_CPP})
    target_link_libraries(${LIB_COMPONENTS} ${LIB_QPANDA} ${PYTHON_LIBRARIES})  
else ()
    add_library(${LIB_COMPONENTS} STATIC ${COMPONENTS_SRC_CPP})
    target_link_libraries(${LIB_COMPONENTS} ${LIB_QPANDA} ${PYTHON_LIBRARIES}) 
endif()



if (UNIX)
    set(LIB_SHARED_COMPONENTS DLL_COMPONENTS)
    add_library(${LIB_SHARED_COMPONENTS} SHARED ${COMPONENTS_SRC_CPP})
    target_link_libraries(${LIB_SHARED_COMPONENTS} ${LIB_QPANDA} ${PYTHON_LIBRARIES}) 
    set_target_properties(${LIB_SHARED_COMPONENTS} PROPERTIES
                          VERSION ${QPANDA_VERSION} SOVERSION ${QPANDA_MAJOR_VERSION})
    set_target_properties(${LIB_SHARED_COMPONENTS} PROPERTIES OUTPUT_NAME "${LIB_COMPONENTS}")
    set_target_properties(${DLL_COMPONENTS} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
    set_target_properties(${DLL_COMPONENTS} PROPERTIES CLEAN_DIRECT_OUTPUT 1)
endif()

INSTALL(TARGETS ${LIB_COMPONENTS}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)


target_include_directories(${LIB_COMPONENTS} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda-2>)
        
export(TARGETS ${LIB_COMPONENTS}
       APPEND FILE ${PROJECT_BINARY_DIR}/qalg-target.cmake
       NAMESPACE QPANDA::
       )




