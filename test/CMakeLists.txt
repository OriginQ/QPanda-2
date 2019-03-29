cmake_minimum_required(VERSION 3.1)
add_subdirectory(QAlg.test)
add_subdirectory(QPanda.Core.test)
add_subdirectory(Variational.test)
add_subdirectory(Autodiff.test)
add_subdirectory(QProgTransform.test)
add_subdirectory(QProgTransformQuil.test)
add_subdirectory(QProgTransformBinary.test)
add_subdirectory(QVM.test)


if (USE_CURL)
    #add_subdirectory(CloudHttp.test)
endif (USE_CURL)