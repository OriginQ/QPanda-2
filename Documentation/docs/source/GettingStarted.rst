如何使用
===========
为了兼容 \ **高效**\与\ **便捷**\，我们为您提供了C++ 和 Python（pyQPanda）两个版本，pyQPanda封装了C++对外提供的接口。

Python 
---------

pyQPanda只需要通过pip就可安装使用。

    .. code-block:: c

        pip install pyqpanda

我们接下来通过一个示例介绍pyQPanda的使用，此例子构造了一个量子叠加态。在量子程序中依次添加H门和CNOT门，最后对所有的量子比特进行测量操作。此时，将有50%的概率得到00或者11的测量结果。
    
    .. code-block:: c
    
        from pyqpanda import *

        init(QMachineType.CPU)
        prog = QProg()
        q = qAlloc_many(2)
        c = cAlloc_many(2)
        prog.insert(H(q[0]))
        prog.insert(CNOT(q[0],q[1]))
        prog.insert(measure_all(q,c))
        result = run_with_configuration(prog, cbit_list = c, shots = 1000)
        print(result)
        finalize()

运行结果如下:

    .. code-block:: c

        {'00': 493, '11': 507}

C++
---------

使用QPanda 2相对于pyQPanda会复杂一些，不过学会编译和使用QPanda 2，您会有更多的体验，话不多说，我们先从介绍Linux下的编译环境开始。

编译环境
>>>>>>>>>>

在下载编译之前，我们需要：

.. list-table::

    * - software
      - version
    * - CMake
      - >= 5.0
    * - GCC
      - >= 3.1 
    * - Python
      - >= 3.6.0  


下载和编译
>>>>>>>>>>>>

我们需要在Linux终端下输入以下命令：

    .. code-block:: c

        $ git clone https://github.com/OriginQ/QPanda-2.git
        $ cd qpanda-2
        $ mkdir build
        $ cd build
        $ cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. 
        $ make
    
安装
>>>>>>>>

编译完成后，安装就简单的多，只需要输入以下命令：

    .. code-block:: c

        $ make install

开始量子编程
>>>>>>>>>>>>>>

现在我们来到最后一关，创建和编译自己的量子应用。

我相信对于关心如何使用QPanda 2的朋友来说，如何创建C++项目，不需要我多说。不过，我还是需要提供CMakelist的示例，方便大家参考。

    .. code-block:: c

        cmake_minimum_required(VERSION 3.1)
        project(testQPanda)
        SET(CMAKE_INSTALL_PREFIX "/usr/local")
        SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} "${CMAKE_INSTALL_PREFIX} lib/cmake")

        add_definitions("-std=c++14 -w -DGTEST_USE_OWN_TR1_TUPLE=1")
        set(CMAKE_BUILD_TYPE "Release")
        set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
        set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
        add_compile_options(-fPIC -fpermissive)
        find_package(QPANDA REQUIRED)
        if (QPANDA_FOUND)

            include_directories(${QPANDA_INCLUDE_DIR}
                            ${THIRD_INCLUDE_DIR})
            add_executable(${PROJECT_NAME} test.cpp)
            target_link_libraries(${PROJECT_NAME} ${QPANDA_LIBRARIES})
        endif (QPANDA_FOUND)


下面的示例和Python版本提供的示例是一样的，在这里我就不多说了。

    .. code-block:: c

        #include "QPanda.h"
        #include <stdio.h>
        using namespace QPanda;
        int main()
        {
            init(QMachineType::CPU);
            QProg prog;
            auto q = qAllocMany(2);
            auto c = cAllocMany(2);
            prog << H(q[0])
                << CNOT(q[0],q[1])
                << MeasureAll(q, c);
            auto results = runWithConfiguration(prog, c, 1000);
            for (auto result : results){
                printf("%s : %d\n", result.first.c_str(), result.second);
            }
            finalize();
        }

最后，编译，齐活。

    .. code-block:: c

        $ mkdir build
        $ cd build
        $ cmake .. 
        $ make

运行结果如下:

    .. code-block:: c

        00 : 512
        11 : 488 

