.. QPanda 2 documentation master file, created by
   sphinx-quickstart on Tue Jan 22 14:31:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

QPanda 2
====================================
|build-status|

.. |build-status| image:: https://travis-ci.org/OriginQ/QPanda-2.svg?branch=master
    :alt: build status
    :scale: 100%
    :target: https://travis-ci.org/OriginQ/QPanda-2

一种功能齐全，运行高效的量子软件开发工具包
---------------------------------------------

QPanda-2（Quantum Panda 2 Software Development Kit）是本源量子开发的开源量子程序开发工具包。其支持主流的量子逻辑门操作，并且可对不同平台下的量子程序进行针对性优化，可适配多种量子芯片。QPanda-2使用C++语言作为经典宿主语言，并支持转换为QRunes、QASM、QUil书写的量子语言。

QPanda-2支持：

* 主流的量子逻辑门，并封装了QAOA、VQE等主流的量子算法。

* 本地模拟，最高可支持到32位量子比特（具体模拟的量子比特数与宿主机硬件配置相关）。

* 在量子程序中，添加经典辅助数据类型，可在量子程序运行时，实现经典辅助变量运算操作和条件判断操作。

* 添加控制流功能（Qif、QWhile），使得量子程序可进行逻辑判断和循环操作。

* 提供量子程序、量子线路、量子逻辑门、量子机器的注册接口，方便用户在不改变QPanda-2整体框架的前提下，满足自身需求。

* 用户可通过配置文件设置目标量子计算机的元数据信息（如量子比特数、量子比特的结构拓扑图，支持的量子逻辑门等信息），实现量子程序的优化。

目前量子计算机还处于研发阶段，研究人员在不同的体系上面构建量子计算机芯片，比如超导约瑟夫森结，半导体量子点，离子阱，NMR等，不同的体系下的量子计算机芯片有不同的性质，QPanda-2的一个设计原则就是适用于不同架构的量子计算机芯片。通过长时间的调研分析，结合本源量子本身的量子计算机硬件研发经验，QPanda-2的设计者抽象出一套标准数据协议。结合标准数据协议，QPanda-2可通过三个步骤来保证其通用性：

* 初始化：初始化两个部分：一、初始化量子机器，获取其元数据信息；二、构建量子程序。

* 优化：根据元数据信息，把目标量子程序中的量子指令转化为目标量子计算机支持的量子指令集序列，并优化转换后的量子指令集序列。

* 运行：运行量子程序并收集计算结果。

QPanda-2的设计思想
----------------------

1. 全系列兼容：
********************

QPanda-2 的目标是兼容所有量子计算机。底层量子计算机现在由于正处快速发展期，所以芯片、测控等实现细节都不确定。QPanda简化并规避了诸多量子计算机的物理细节而为用户提供了标准化的接口。通过QPanda-2构建的量子计算机，本身是通过经典的程序语言对其进行交互，所以它可以被用于任意的云量子计算机，本地量子计算机，或者是实验中的量子原型机。通过QPanda-2构建的量子应用则不会受到硬件变动的影响。

2. 标准架构：
*********************

QPanda-2提供了标准化的量子程序（Quantum Program）架构。架构者认为，在量子机器（Quantum Machine）中执行的程序和在经典计算机中执行的程序应该彻底区分开来，特别是涉及到经典控制的部分。物理上，芯片的退相干（Decoherence）时间极为短暂，这使得量子程序中的控制流并非在狭义的CPU中完成，而更有可能会采用极低延时的FPGA或其它嵌入式器件作为其测控系统实现。我们认为，量子机器包含了量子芯片与其测控系统，一个量子程序被视作是对一个原子的操作，直到执行完毕才返回结果给经典计算机。 量子程序的架构包含：量子逻辑门、量子线路、量子分支线路和量子循环线路。在QPanda-2里这几种元素均以接口的形式被提供，我们提供了一组这些接口的实现类作为基础的数据接口。用户可以重写这些接口并将实现类进行注册，系统会选择用户的类对默认实现类进行覆盖，并且保持其它结构的不变。

3. 标准化量子机器模型：
***********************

我们提供了标准化的量子机器模型。通常，量子程序是静态加载到量子机器里，并且量子程序本身也是被静态地构建的。这意味我们可以在量子程序被执行前，对量子程序进行静态检查和分析，获取其中的信息（而非执行它）。能检查的要素例如：量子比特是否越界，经典寄存器是否超过硬件允许的范围等等。而能进行的预处理则包含：任意的量子程序被替换到对应真实芯片的拓扑结构和基本逻辑门集合上（硬件兼容），量子程序的运行时长判断，量子程序的优化等等。 量子机器模型还定义了量子程序的标准构建过程。例如从量子比特池中申请空闲比特，从内存中申请空间，将程序加载到量子机器中，或者在已有的量子程序中附加一段新的量子程序。和量子程序的部分类似，量子机器本身的任何架构也是接口化的，用户也可以对接口进行覆写以应对不同硬件的需求。


.. toctree::
    :maxdepth: 1

    ChangeLog
   
目录：

.. toctree::
    :maxdepth: 2

    Tutorial

.. toctree::
    :caption: 深入学习
    :maxdepth: 2

    QGate
    QCircuit
    QWhile
    QIf
    QProg
    QuantumMachine
    Measure
    PMeasure

.. toctree::
    :caption: 工具组件
    :maxdepth: 2

    QGateValidity
    QGateCounter
    QProgClockCycle
    QProgStored
    QProgDataParse
    QRunesToQProg

.. toctree::
    :caption: 量子程序转换
    :maxdepth: 2
    
    QProgToQASM
    QProgToQRunes
    QProgToQuil

.. toctree::
    :caption: 量子算法
    :maxdepth: 2

    QAOA
    VQE

.. toctree::
    :caption: 算法组件
    :maxdepth: 2

    PauliOperator
    FermionOperator
    Optimizer

.. toctree::
    :caption: VQNet
    :maxdepth: 2
    
    Variable
    VarOperator
    Expression
    可变量子逻辑门
    可变量子线路
    GradientOptimizer
    VQNetExample
