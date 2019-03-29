#.rst:
# FindQPANDA
# --------
#
# Find qpanda
#
# Find the native qpanda headers and libraries.
#
# ::
#
#   QPANDA_INCLUDE_DIR    - QPanda-2 include  etc.
#   THIRD_INCLUDE_DIR     - ThirdParty include
#   QPANDA_LIBRARIES      - List of libraries when using QPanda.
#   QALG_LIBRARY          - QAlg library
#   VAR_LIBRARY           - Variational library
#   TINY_LIBRARY          - TinyXML library
#   QPANDA_LIBRARY        - QPanda-2 library
#   QPANDA_FOUND          - True if curl found.
#   


# Look for the header file.
find_path(QPANDA_INCLUDE_DIR NAMES qpanda-2/Core/QPanda.h PATHS ${CMAKE_INSTALL_PREFIX}/include)
set(QPANDA_INCLUDE_DIR "${QPANDA_INCLUDE_DIR}/qpanda-2/")
find_path(THIRD_INCLUDE_DIR NAMES qpanda-2/ThirdParty/TinyXML/tinyxml.h PATHS ${CMAKE_INSTALL_PREFIX}/include)
set(THIRD_INCLUDE_DIR "${THIRD_INCLUDE_DIR}/qpanda-2/ThirdParty")
mark_as_advanced(QPANDA_INCLUDE_DIR)
# Look for the library (sorted from most current/relevant entry to least).
find_library(QALG_LIBRARY NAMES QAlg PATHS ${CMAKE_INSTALL_PREFIX}/lib)
find_library(QPANDA_LIBRARY NAMES QPanda2.0 PATHS ${CMAKE_INSTALL_PREFIX}/lib)
find_library(VAR_LIBRARY NAMES Variational PATHS ${CMAKE_INSTALL_PREFIX}/lib)
find_library(TINY_LIBRARY NAMES TinyXML PATHS ${CMAKE_INSTALL_PREFIX}/lib)
find_library(GPU_LIBRARY NAMES QPanda-2.0.GPUQGates PATHS ${CMAKE_INSTALL_PREFIX}/lib)
mark_as_advanced(QPANDA_LIBRARY)
mark_as_advanced(QALG_LIBRARY)
mark_as_advanced(VAR_LIBRARY)
if(QPANDA_INCLUDE_DIR AND QPANDA_LIBRARY)
    set(QPANDA_FOUND 1)
else(QPANDA_INCLUDE_DIR AND QPANDA_LIBRARY)
    message("QPANDA Not Find")
endif(QPANDA_INCLUDE_DIR AND QPANDA_LIBRARY)
if(QPANDA_FOUND)
    if(GPU_LIBRARY)
        set(QPANDA_LIBRARIES ${QPANDA_LIBRARY} ${QALG_LIBRARY} ${VAR_LIBRARY} ${TINY_LIBRARY} ${GPU_LIBRARY})
    else(GPU_LIBRARY)
        set(QPANDA_LIBRARIES ${QPANDA_LIBRARY} ${QALG_LIBRARY} ${VAR_LIBRARY} ${TINY_LIBRARY})
    endif(GPU_LIBRARY)
    mark_as_advanced(QPANDA_LIBRARIES)
endif()