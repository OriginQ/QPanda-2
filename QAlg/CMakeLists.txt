cmake_minimum_required(VERSION 3.1)
project(QAlg)

file(GLOB_RECURSE ALG_SRC_CPP ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE ALG_CHEMIQ_SRC_CPP ${CMAKE_SOURCE_DIR}/Components/ChemiQ/*.cpp)

if (USE_CHEMIQ)
    find_package(PythonInterp 3 REQUIRED)
    find_package(PythonLibs 3 REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})

    add_library(${LIB_QALG} STATIC ${ALG_SRC_CPP} ${ALG_CHEMIQ_SRC_CPP})
    target_link_libraries(${LIB_QALG} ${LIB_QPANDA} ${LIB_COMPONENTS})

    set(LIB_SHARED_QALG ChemiQ)
    add_library(${LIB_SHARED_QALG} SHARED ${ALG_SRC_CPP} ${ALG_CHEMIQ_SRC_CPP})
    target_link_libraries(${LIB_SHARED_QALG} ${LIB_QPANDA} ${LIB_COMPONENTS} pybind11::embed)
else ()
    add_library(${LIB_QALG} STATIC ${ALG_SRC_CPP})
    target_link_libraries(${LIB_QALG} ${LIB_QPANDA} ${LIB_COMPONENTS})
endif ()
set_target_properties(${LIB_SHARED_QALG} PROPERTIES OUTPUT_NAME "${LIB_SHARED_QALG}")

INSTALL(TARGETS ${LIB_QALG}
        EXPORT  ${TARGETS_EXPORT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

target_include_directories(${LIB_QALG} INTERFACE
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:include/qpanda2>)
        
export(TARGETS ${LIB_QALG}
       APPEND FILE ${PROJECT_BINARY_DIR}/qalg-target.cmake
       NAMESPACE QPANDA::
       )




