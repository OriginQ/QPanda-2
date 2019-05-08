## QPanda 2

[![Build Status](https://travis-ci.org/OriginQ/QPanda-2.svg?branch=master)](https://travis-ci.org/OriginQ/QPanda-2)
[![Documentation Status](https://readthedocs.org/projects/qpanda-2/badge/?version=latest)](https://qpanda-2.readthedocs.io/zh_CN/latest/?badge=latest)


![图片: ](./Documentation/img/1.png)

QPanda 2(**Q**uantum **P**rogramming **A**rchitecture for **N**ISQ **D**evice **A**pplications)是一个**高效**的量子计算开发工具库，可用于实现各种**量子算法**，QPanda 2基于C++实现，并可扩展到Python。 


## 优点

* **QPanda 2快**，它利用OpenMP、MPI加速量子逻辑门模拟算法，计算性能逼近CPU/GPU的理论FLOPS；
* **QPanda 2功能全面**，它提供本地的单振幅、部分振幅、全振幅、含噪声量子虚拟机，并可直接连接量子云服务器，运行量子程序；
* **QPanda 2工具多**，它可根据真实量子计算机的数据参数，提供量子线路优化/转换工具，方便用户探索NISQ装置上有实用价值的量子算法；
* **QPanda 2使用方便**，它根据不同的需求，向用户提供面向过程和面向对象两种风格的API，方便不熟悉编程的用户使用。

## 文档

* QPanda 2的使用文档，位于 https://qpanda-2.readthedocs.io/zh_CN/latest/ 
* pyQPanda的使用文档，位于 https://qpanda-2.readthedocs.io/zh_CN/doucmentation-python/


## 兼容性

QPanda 2是跨平台的。我们曾在以下平台/编译器组合下测试：

* Visual Studio 2017 在 Windows (64-bit)
* GCC 5.4.0 在 Ubuntu (64-bit)
* AppleClang 10.0.0 在 Mac OS X (64-bit)

## 如何使用

为了兼容**高效**与**便捷**，我们为您提供了C++ 和 Python（pyQPanda）两个版本，pyQPanda封装了C++对外提供的接口。

### Python 

pyQPanda只需要通过pip就可安装使用。

    pip install pyqpanda

我们接下来通过一个示例介绍pyQPanda的使用，此例子构造了一个量子叠加态。在量子程序中依次添加H门和CNOT门，最后对所有的量子比特进行测量操作。此时，将有50%的概率得到00或者11的测量结果。

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
    
    {'00': 493, '11': 507}

### C++ 
使用QPanda 2相对于pyQPanda会复杂一些，不过学会编译和使用QPanda 2，您会有更多的体验，更多详情可以阅读[使用文档](https://qpanda-2.readthedocs.io/zh_CN/latest/)。话不多说，我们先从介绍Linux下的编译环境开始。

#### 编译环境

在下载编译之前，我们需要：

| software                | version         |
|-------------------------|-----------------|
| GCC                     | >= 5.0          |
| CMake                   | >= 3.1          |
| Python                  | >= 3.6.0        |

#### 下载和编译

我们需要在Linux终端下输入以下命令：

    $ git clone https://github.com/OriginQ/QPanda-2.git
    $ cd qpanda-2
    $ mkdir build
    $ cd build
    $ cmake -DCMAKE_INSTALL_PREFIX=/usr/local .. 
    $ make
    
#### 安装
编译完成后，安装就简单的多，只需要输入以下命令：

      $ make install

#### 开始量子编程

现在我们来到最后一关，创建和编译自己的量子应用。

我相信对于关心如何使用QPanda 2的朋友来说，如何创建C++项目，不需要我多说。不过，我还是需要提供CMakelist的示例，方便大家参考。

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
    
    $ mkdir build
    $ cd build
    $ cmake .. 
    $ make

运行结果如下:

    00 : 512
    11 : 488 

一个示例不够？请移步到[Applications](./Applications)，这里有更多示例。

## 团队介绍

很多朋友咨询过这样一个问题，我使用QPanda 2编写的量子应用，以后能直接应用到量子计算机上么？我们的回答是肯定的，因为我们的开发团队来自**合肥本源量子**，中国**第一家**做量子计算的公司。我们本源是集**量子计算软件**和**量子计算机硬件**于一身的公司。QPanda 2的设计之初充分考虑了**量子计算机体系架构**，从根本上保证了软硬件对接的问题。

 ## License
 Apache License 2.0

 Copyright (c) 2017-2019 By Origin Quantum Computing. All Right Reserved.
