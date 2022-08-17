cmake_minimum_required(VERSION 3.1)
project(QPANDA)

set(QPANDA_MAJOR_VERSION 2)
set(QPANDA_MINOR_VERSION 1)
set(QPANDA_PATCH_VERSION 11)
set(QPANDA_VERSION "${QPANDA_MAJOR_VERSION}.${QPANDA_MINOR_VERSION}.${QPANDA_PATCH_VERSION}")
message(STATUS "QPANDA VERSION = " "[${QPANDA_VERSION}]")
# disable compiler verbose output, default is off, enable it when want to check compiler/linker option
set(CMAKE_VERBOSE_MAKEFILE OFF)

include_directories(${PROJECT_SOURCE_DIR}/Extensions
                    ${PROJECT_SOURCE_DIR}/include
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
                    ${PROJECT_SOURCE_DIR}/include/ThirdParty/nlopt
                    ${PROJECT_SOURCE_DIR}/ThirdParty
                    ${PROJECT_SOURCE_DIR}/ThirdParty/gtest
                    ${PROJECT_SOURCE_DIR}/ThirdParty/gtest/include
                    ${PROJECT_SOURCE_DIR}/ThirdParty/bplus-tree
                    ${PROJECT_SOURCE_DIR}/ThirdParty/bplus-tree/include
                    ${PROJECT_SOURCE_DIR}/ThirdParty/antlr4/runtime/src
                    ${PROJECT_SOURCE_DIR}/ThirdParty/rabbit
                    ${PROJECT_SOURCE_DIR}/ThirdParty/mpfit/include
                    ${PROJECT_SOURCE_DIR})

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/lib)
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")

set(LIB_LIST)
set(LIB_DIR_LIST)
set(EXTRA_LDFLAGS)

set(LIB_QPANDA QPanda2)
set(LIB_GTEST gtest)
set(LIB_BPLUSTREE BplusTree)
set(LIB_COMPONENTS Components)
set(LIB_NLOPT nlopt)
set(LIB_MPFIT mpfit)
set(LIB_ANTLR4 antlr4)
set(LIB_QALG QAlg)

set(LIB_PYQPANDA pyQPanda)
set(LIB_PYQPANDA_PAUL pyQPandaPauliOperator)
set(LIB_PYQPANDA_CHEMIQ pyQPandaChemiQ)

option(FIND_MPI "find MPI" OFF)
option(FIND_CUDA "find cuda" OFF)
option(USE_CHEMIQ "use ChemiQ" OFF)
option(FIND_OPENMP "find openMP" ON)
option(USE_PYQPANDA "Enable pyQPanda" OFF)
option(USE_EXTENSION "EXTENSION FUNTIONS" OFF)
option(USE_CURL "Enable the QCloudMachine" ON)
option(USE_MSVC_RUNTIME_LIBRARY_DLL "MD/MT" OFF)
option(PYQPANDA_STUBGEN "generate stub file for pyQPanda" ON)
option(USE_SIMD "use simd  " OFF)

if(NOT CMAKE_BUILD_TYPE)
   message(STATUS ":  No build type selected, default is Release")
   set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_SIMD)
    include(${CMAKE_SOURCE_DIR}/cmake/FindAVX2.cmake)
    CHECK_SUPPORT_AVX2()
    if(AVX2_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${AVX2_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${AVX2_FLAGS}")
        option (USE_SIMD "use simd" ON)
        message("found AVX2")
        message("use flags ${AVX2_FLAGS}")
    else()
        option (USE_SIMD "use simd" OFF)
        message("not found AVX2")
    endif()
endif()

if (MSVC)
    list(APPEND EXTRA_LDFLAGS
        -D_SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING
        -D_CRT_SECURE_NO_WARNINGS
        -DNOMINMAX 
        /bigobj) 



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

        if (NOT USE_MSVC_RUNTIME_LIBRARY_DLL)
            if (${flag} MATCHES "/MDd")
                string(REGEX REPLACE "/MDd" "/MTd" ${flag} "${${flag}}")
            endif()

            if (${flag} MATCHES "/MD")
                string(REGEX REPLACE "/MD" "/MT" ${flag} "${${flag}}")
            endif()
        endif()

        if (${flag} MATCHES "/W3")
            string(REGEX REPLACE "/W3" "/W0" ${flag} "${${flag}}")
        endif()
    endforeach()
elseif (MINGW)
    list(APPEND EXTRA_LDFLAGS
        -w
        -DGTEST_USE_OWN_TR1_TUPLE=1
        -fpermissive)
    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -Og -ggdb -m64 -Wa, -mbig-obj")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3  -DNDEBUG")
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

    set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g  -ggdb")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DNDEBUG")
endif()

add_definitions(${EXTRA_LDFLAGS} ${SIMD_FLAGS})

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(UUID REQUIRED uuid)
endif()
if(APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
endif()

if (FIND_CUDA)
    find_package(CUDA)
endif(FIND_CUDA)
if(CUDA_FOUND)
    option (USE_CUDA "Use cuda" ON)
    include_directories(${CUDA_INCLUDE_DIRS})
    foreach(CUDA_LIB_DIR ${CUDA_LIBRARIES})
		get_filename_component(CUDA_LIB_DIR ${CUDA_LIB_DIR} DIRECTORY)
		list(APPEND LIB_DIR_LIST ${CUDA_LIB_DIR})
	endforeach()
	link_directories(${LIB_DIR_LIST})
else(CUDA_FOUND)
    option (USE_CUDA "Use cuda" OFF)
endif(CUDA_FOUND)

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

list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".dll" ".a")
if (USE_CURL)
    find_package(CURL)
    if (CURL_FOUND)
        option(USE_CURL "find Curl" ON)
        include_directories(${CURL_INCLUDE_DIR})
        list(APPEND LIB_LIST
            ${CURL_LIBRARY})

		if (MSVC)
			find_library(CURL_DLL "libcurl.dll")
			set(CURL_DLL_OUTPUT "libcurl.dll")
			add_custom_command(OUTPUT ${CURL_DLL_OUTPUT}
			   COMMAND ${CMAKE_COMMAND} -E copy ${CURL_DLL} ${EXECUTABLE_OUTPUT_PATH}/${CURL_DLL_OUTPUT})
			add_custom_target(CopyCurlTask ALL DEPENDS ${CURL_DLL_OUTPUT})
		endif()
    else (CURL_FOUND)
        # as use OPTION command, CMakeCahe have saved USE_CURL = ON, cmake then will read CMakeCache
        # use FORCE to turnoff  USE_CURL in CMakeCache
        set(USE_CURL OFF CACHE BOOL "not find Curl" FORCE)
    endif (CURL_FOUND)
endif(USE_CURL)

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


add_subdirectory(ThirdParty)
add_subdirectory(Core)
add_subdirectory(Components)
add_subdirectory(QAlg)
add_subdirectory(Applications)
add_subdirectory(test)

if(USE_PYQPANDA)
    include_directories(${PROJECT_SOURCE_DIR}/pyQPandaCpp)
    add_subdirectory(pyQPandaCpp)
endif(USE_PYQPANDA)


if(NOT (WINDOWS OR CYGWIN))
	set(prefix ${CMAKE_INSTALL_PREFIX})
	set(exec_prefix "\${prefix}")
	set(libdir "\${exec_prefix}/lib${LIB_SUFFIX}")
	set(bindir "\${exec_prefix}/bin")
	set(includedir "\${prefix}/include")
	set(QPANDA_LIBS "-lComponents -lQAlg -lQPanda2 -lantlr4")
	set(QPANDA_CFLAGS "")
	configure_file("${PROJECT_SOURCE_DIR}/qpanda-config.in"
			        "${PROJECT_BINARY_DIR}/qpanda-config")
	configure_file("${PROJECT_SOURCE_DIR}/qpanda-config.in"
			        "${PROJECT_BINARY_DIR}/qpanda-config" @ONLY)
	install(PROGRAMS ${PROJECT_BINARY_DIR}/qpanda-config DESTINATION bin)
endif()


configure_file("${PROJECT_SOURCE_DIR}/config.h.in"
                "${PROJECT_SOURCE_DIR}/include/QPandaConfig.h")
configure_file("${PROJECT_SOURCE_DIR}/configVersion.h.in"
                "${PROJECT_SOURCE_DIR}/include/QPandaVersion.h"
                @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}ConfigVersion.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
               @ONLY)


#add extra search paths for libraries and includes
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
if(UNIX OR CYGWIN)
    set(_CMAKE_INSTALL_DIR "${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME}")
elseif(WIN32)
    SET(_CMAKE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/cmake")
endif()
set(CMAKE_INSTALL_DIR "${_CMAKE_INSTALL_DIR}")


set(QPANDA_CONFIG "QPandaConfig.json")
add_custom_command(OUTPUT ${QPANDA_CONFIG}
   COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/${QPANDA_CONFIG} ${EXECUTABLE_OUTPUT_PATH}/${QPANDA_CONFIG}
 )
add_custom_target(CopyQPandaConfig ALL DEPENDS ${QPANDA_CONFIG})

install(DIRECTORY include/
    DESTINATION "${INCLUDE_INSTALL_DIR}/qpanda2")
if (USE_EXTENSION)
    install(DIRECTORY Extensions/
            DESTINATION "${INCLUDE_INSTALL_DIR}/qpanda2/Extensions"
            FILES_MATCHING PATTERN "*.h")
else(USE_EXTENSION)
    install(FILES Extensions/Extensions.h
            DESTINATION "${INCLUDE_INSTALL_DIR}/qpanda2/Extensions/")
endif(USE_EXTENSION)


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
