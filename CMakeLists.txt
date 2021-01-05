cmake_minimum_required(VERSION 3.1)
project(QPANDA)

set(QPANDA_MAJOR_VERSION 2)
set(QPANDA_MINOR_VERSION 1)
set(QPANDA_PATCH_VERSION 5)
set(QPANDA_VERSION "${QPANDA_MAJOR_VERSION}.${QPANDA_MINOR_VERSION}.${QPANDA_PATCH_VERSION}")
message(STATUS "QPANDA VERSION = " "[${QPANDA_VERSION}]")

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PROJECT_SOURCE_DIR}/include/Applications
                    ${PROJECT_SOURCE_DIR}/include/QAlg
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
                    ${PROJECT_SOURCE_DIR}/include/Core/Utilities/Transform/OriginIRCompiler
                    ${PROJECT_SOURCE_DIR}/include/Variational
                    ${PROJECT_SOURCE_DIR}/ThirdParty
                    ${PROJECT_SOURCE_DIR}/ThirdParty/gtest
                    ${PROJECT_SOURCE_DIR}/ThirdParty/gtest/include
                    ${PROJECT_SOURCE_DIR}/ThirdParty/bplus-tree
                    ${PROJECT_SOURCE_DIR}/ThirdParty/bplus-tree/include
                    ${PROJECT_SOURCE_DIR}/ThirdParty/antlr4/runtime/src
                    ${PROJECT_SOURCE_DIR})

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")

set(LIB_LIST)
set(EXTRA_LDFLAGS)

set(LIB_QPANDA QPanda2)
set(LIB_GTEST gtest)
set(LIB_BPLUSTREE BplusTree)
set(LIB_COMPONENTS Components)
set (LIB_ANTLR4 antlr4)
set(LIB_QALG QAlg)

set(LIB_PYQPANDA pyQPanda)
set(LIB_PYQPANDA_PAUL pyQPandaPauliOperator)
set(LIB_PYQPANDA_CHEMIQ pyQPandaChemiQ)

if(NOT CMAKE_BUILD_TYPE)
   message(STATUS ":  No build type selected, default is Release")
   set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    list(APPEND EXTRA_LDFLAGS
        -D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING
        -D_CRT_SECURE_NO_WARNINGS
        /bigobj)

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

            if (${flag} MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
            endif()

            if (${flag} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
            
            if (${flag} MATCHES "/W3")
                string(REGEX REPLACE "/W3" "/W0" ${flag} "${${flag}}")
            endif()
        endforeach()
    endif()
elseif (MINGW)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fpermissive)

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
elseif (UNIX)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fPIC
        -fpermissive)

    if (NOT APPLE)
        list(APPEND LIB_LIST
            dl
            pthread)
    endif ()

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
endif()
add_definitions(${EXTRA_LDFLAGS})


if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(UUID REQUIRED uuid)
endif()
if(APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif()

option(FIND_CUDA "find cuda" OFF)
if (FIND_CUDA)
    find_package(CUDA)
endif(FIND_CUDA)
if(CUDA_FOUND)
    option (USE_CUDA "Use cuda" ON)
    include_directories(${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
    option (USE_CUDA "Use cuda" OFF)
endif(CUDA_FOUND)

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
    list(APPEND LIB_LIST
        ${CURL_LIBRARY})
else (CURL_FOUND)
    option(USE_CURL "not find Curl" OFF)
endif (CURL_FOUND)

option(FIND_MPI "find MPI" OFF)
if (FIND_MPI)
    find_package(MPI)
endif(FIND_MPI)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    option(USE_MPI "find MPI" ON)
    list(APPEND LIB_LIST
        ${MPI_LIBRARIES})
elseif(MPI_FOUND)
    option(USE_MPI "not find MPI" OFF)
endif (MPI_FOUND)

option(USE_CHEMIQ "use ChemiQ" OFF)

add_subdirectory(ThirdParty)
add_subdirectory(Core)
add_subdirectory(Components)
add_subdirectory(QAlg)
add_subdirectory(Applications)
add_subdirectory(test)

option(USE_PYQPANDA "use pyQpanda" OFF)
if(USE_PYQPANDA)
    include_directories(${PROJECT_SOURCE_DIR}/pyQPandaCpp)
    add_subdirectory(pyQPandaCpp)
endif(USE_PYQPANDA)

configure_file("${PROJECT_SOURCE_DIR}/config.h.in"
                "${PROJECT_SOURCE_DIR}/include/QPandaConfig.h")
configure_file("${PROJECT_SOURCE_DIR}/configVersion.h.in"
                "${PROJECT_SOURCE_DIR}/include/QPandaVersion.h"
                @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}ConfigVersion.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
               @ONLY)


#add extra search paths for libraries and includes
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" CACHE PATH "The directory the headers are installed in")
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" CACHE STRING "Directory where lib will install")

if(UNIX OR CYGWIN)
    set(_CMAKE_INSTALL_DIR "${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}")
elseif(WIN32)
    SET(_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/cmake")
endif()
set(CMAKE_INSTALL_DIR "${_CMAKE_INSTALL_DIR}" CACHE PATH "The directory cmake files are installed in")


set(QPANDA_CONFIG "QPandaConfig.json")
add_custom_command(OUTPUT ${QPANDA_CONFIG}
   COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/${QPANDA_CONFIG} ${EXECUTABLE_OUTPUT_PATH}/${QPANDA_CONFIG}
 )
add_custom_target(CopyQPandaConfig ALL DEPENDS ${QPANDA_CONFIG})

install(DIRECTORY include/
    DESTINATION "${INCLUDE_INSTALL_DIR}/qpanda2")

install(EXPORT "${TARGETS_EXPORT_NAME}"
        NAMESPACE "${PROJECT_NAME}::"
        FILE ${PROJECT_NAME}Config.cmake
        DESTINATION ${CMAKE_INSTALL_DIR})
install(FILES ${PROJECT_SOURCE_DIR}/FindQPANDA.cmake
         DESTINATION "${LIB_INSTALL_DIR}/cmake/")

install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
        DESTINATION "${CMAKE_INSTALL_DIR}")

if(NOT TARGET uninstall)
  configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
      "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
      IMMEDIATE @ONLY)

  add_custom_target(uninstall
      COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
