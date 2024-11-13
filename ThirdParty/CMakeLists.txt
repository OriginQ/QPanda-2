cmake_minimum_required(VERSION 3.1)
add_subdirectory(googletest)
add_subdirectory(antlr4)
add_subdirectory(bplus-tree)
add_subdirectory(nlopt)
add_subdirectory(mpfit)
add_subdirectory(bz2)

if(USE_PYQPANDA)
    add_subdirectory(pybind11)
endif()

# Installation method for third-party libraries with only header files
install(DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/rapidjson  DESTINATION 
	${CMAKE_INSTALL_PREFIX}/include/qpanda2/ThirdParty)

install(DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/rabbit  DESTINATION 
	${CMAKE_INSTALL_PREFIX}/include/qpanda2/ThirdParty)

install(DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/Eigen DESTINATION 
		${CMAKE_INSTALL_PREFIX}/include/qpanda2/ThirdParty)

install(DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/EigenUnsupported DESTINATION 
		${CMAKE_INSTALL_PREFIX}/include/qpanda2/ThirdParty)

install(DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/uintwide DESTINATION 
		${CMAKE_INSTALL_PREFIX}/include/qpanda2/ThirdParty)

install(DIRECTORY  ${CMAKE_CURRENT_SOURCE_DIR}/bz2 DESTINATION 
		${CMAKE_INSTALL_PREFIX}/include/qpanda2/ThirdParty)
