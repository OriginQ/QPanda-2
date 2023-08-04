cmake_minimum_required(VERSION 3.1)

add_subdirectory(Core)
if(USE_EXTENSION)
   # add_subdirectory(Core)
    add_subdirectory(QAlg)
endif(USE_EXTENSION)

add_subdirectory(Components)
add_subdirectory(Overall_scan)

