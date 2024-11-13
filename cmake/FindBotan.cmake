# You need to put this file into the cmake_module_path for cmake to use it.
# This file should go into a folder called cmake in your project root.
# Guidance from https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Find-Libraries#writing-find-modules

# Sets Botan_FOUND, Botan_LIBRARIES, Botan_INCLUDE_DIRS, and creates an imported static library target named Botan::Botan

if (NOT WIN32)
    find_package(PkgConfig)
    pkg_check_modules(PC_BOTAN QUIET botan)
    set(PC_BOTAN_DEFINITIONS ${PC_BOTAN_CFLAGS_OTHER})
endif (NOT WIN32)

find_path(Botan_INCLUDE_DIR botan/build.h
        HINTS ${PC_BOTAN_INCLUDEDIR} ${PC_BOTAN_INCLUDE_DIRS} "C:/"
        PATH_SUFFIXES botan/include/botan-3
        )

find_library(Botan_LIBRARY NAMES botan-3
        HINTS ${PC_BOTAN_LIBDIR} ${PC_BOTAN_LIBRARY_DIRS} "C:/"
        PATH_SUFFIXES botan/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Botan DEFAULT_MSG
        Botan_LIBRARY Botan_INCLUDE_DIR)
mark_as_advanced(Botan_FOUND Botan_INCLUDE_DIR Botan_LIBRARY)

if (Botan_FOUND)
    get_filename_component(Botan_INCLUDE_DIRS ${Botan_INCLUDE_DIR} DIRECTORY)
endif ()

if (Botan_FOUND AND NOT TARGET Botan::Botan)
    add_library(Botan::Botan STATIC IMPORTED)
    if (MSVC)
        target_compile_options(Botan::Botan INTERFACE /wd4250)
    endif (MSVC)
    target_include_directories(Botan::Botan
            INTERFACE
            ${Botan_INCLUDE_DIR})
    set_target_properties(Botan::Botan PROPERTIES
            IMPORTED_LOCATION ${Botan_LIBRARY}
            )
endif ()