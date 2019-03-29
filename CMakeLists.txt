cmake_minimum_required(VERSION 3.1)
project(QPANDA)
include_directories(${PROJECT_SOURCE_DIR}/include/Applications
                    ${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/include/QAlg
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Algorithms
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Algorithms/VQE
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Algorithms/QAOA
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Components
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Components/Optimizer
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Components/Operator
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Components/MaxCutProblemGenerator
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Components/HamiltonianSimulation
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Utils
                    ${PROJECT_SOURCE_DIR}/include/QAlg/Utils/RJson
                    ${PROJECT_SOURCE_DIR}/include/Core
                    ${PROJECT_SOURCE_DIR}/include/Core/VirtualQuantumProcessor
                    ${PROJECT_SOURCE_DIR}/include/Core/VirtualQuantumProcessor/NoiseQPU
                    ${PROJECT_SOURCE_DIR}/include/Core/VirtualQuantumProcessor/GPUGates
                    ${PROJECT_SOURCE_DIR}/include/Core/VirtualQuantumProcessor/PartialAmplitude
                    ${PROJECT_SOURCE_DIR}/include/Core/VirtualQuantumProcessor/SingleAmplitude
                    ${PROJECT_SOURCE_DIR}/include/Core/QuantumMachine
                    ${PROJECT_SOURCE_DIR}/include/Core/QuantumCircuit
                    ${PROJECT_SOURCE_DIR}/include/Core/Utilities
                    ${PROJECT_SOURCE_DIR}/include/Core/Utilities/Transform
                    ${PROJECT_SOURCE_DIR}/include/Variational
                    ${PROJECT_SOURCE_DIR}/ThirdParty
                    ${PROJECT_SOURCE_DIR}/ThirdParty/gtest
                    ${PROJECT_SOURCE_DIR}/ThirdParty/gtest/include
                    ${PROJECT_SOURCE_DIR}/ThirdParty/bplus-tree
                    ${PROJECT_SOURCE_DIR}/ThirdParty/bplus-tree/include
                    ${PROJECT_SOURCE_DIR}
                    ${PYTHON_PATH_INCLUDE})

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
if (WIN32)
    if (MSVC)
        if (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
            foreach (flag
                CMAKE_C_FLAGS
                CMAKE_C_FLAGS_DEBUG
                CMAKE_C_FLAGS_RELEASE
                CMAKE_C_FLAGS_MINSIZEREL
                CMAKE_C_FLAGS_RELWITHDEBINFO
                CMAKE_CXX_FLAGS
                CMAKE_CXX_FLAGS_DEBUG
                CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_MINSIZEREL
                CMAKE_CXX_FLAGS_RELWITHDEBINFO)

                if (${flag} MATCHES "/MD")
                    string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
                endif()
                if (${flag} MATCHES "/MDd")
                    string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
                endif()
                if (${flag} MATCHES "/W3")
                    string(REGEX REPLACE "/W3" "/W0" ${flag} "${${flag}}")
                endif()
            endforeach()
        endif()
        Add_Definitions(-D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING)
        Add_Definitions(-D_CRT_SECURE_NO_WARNINGS)
     else (MSVC)
        add_definitions("-std=c++14 -w -DGTEST_USE_OWN_TR1_TUPLE=1")
        set(CMAKE_BUILD_TYPE "Release")
        set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
        set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
        add_compile_options(-fpermissive)
     endif(MSVC)
elseif(UNIX)
    add_definitions("-std=c++14 -w -DGTEST_USE_OWN_TR1_TUPLE=1")
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
    add_compile_options(-fPIC -fpermissive)
    link_libraries(dl pthread)
endif(WIN32)

option(FIND_CUDA "find cuda" ON) 
option(FIND_OPENMP "find openMP" ON)
if (FIND_OPENMP)
    find_package(OpenMP)
endif(FIND_OPENMP)
if(OPENMP_FOUND)
    option(USE_OPENMP "find OpenMP" ON)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else(OPENMP_FOUND)
    option(USE_OPENMP "not find OpenMP" OFF)
endif(OPENMP_FOUND)

find_package(CURL)
if (CURL_FOUND)
    option(USE_CURL "find Curl" ON)
    include_directories(${CURL_INCLUDE_DIR}) 
else (CURL_FOUND)
    option(USE_CURL "not find Curl" OFF)
endif (CURL_FOUND)

configure_file(  
   "${PROJECT_SOURCE_DIR}/config.h.in"
   "${PROJECT_SOURCE_DIR}/config.h")

add_subdirectory(QAlg)
add_subdirectory(Core)
add_subdirectory(ThirdParty)
add_subdirectory(Variational)
add_subdirectory(Applications)
add_subdirectory(test)
option(USE_PYQPANDA "use pyQpanda" ON)
if(USE_PYQPANDA)
    include_directories(${PROJECT_SOURCE_DIR}/pyQPandaCpp)
    add_subdirectory(pyQPandaCpp)
endif(USE_PYQPANDA)

INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include/qpanda-2)
INSTALL(EXPORT "${TARGETS_EXPORT_NAME}"
        NAMESPACE "${PROJECT_NAME}::"
        FILE ${PROJECT_NAME}Config.cmake
        DESTINATION lib/cmake/${PROJECT_NAME}
    )
install(FILES ${PROJECT_SOURCE_DIR}/FindQPANDA.cmake
         DESTINATION lib/cmake/)
