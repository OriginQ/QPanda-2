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

**一种功能齐全，运行高效的量子软件开发工具包**

QPanda 2是一个基于量子编程的开发环境，可以被应用于各类量子算法的编程实现。QPanda 2利用C++编写，并可以扩展到Python上。

QPanda 2拥有多层级结构，可部署于各类量子计算机或量子虚拟机进行计算，可以应用于量子计算领域的研究和产品开发。
QPanda 2由本源量子软件团队开发和维护。自2018年起，QPanda 2依据Apache 2.0 License发布于GitHub。

.. toctree::
    :maxdepth: 1

    ChangeLog
   
.. toctree::
    :maxdepth: 2

    GettingStarted
    Tutorial

深入学习
-----------

.. toctree::
    :maxdepth: 2

    QGate
    QCircuit
    QWhile
    QIf
    QProg
    QuantumMachine
    Measure
    PMeasure

工具组件
-----------

.. toctree::
    :maxdepth: 2

    QGateValidity
    QGateCounter
    QProgClockCycle
    QProgStored
    QProgDataParse
    QRunesToQProg

量子程序转换
-----------
.. toctree::
    :maxdepth: 2
    
    QProgToQASM
    QProgToQRunes
    QProgToQuil

算法组件
-----------

.. toctree::
    :maxdepth: 2

    PauliOperator
    FermionOperator
    Optimizer

.. toctree::
    :caption: VQNet
    :maxdepth: 2
    
    Var
    VarOperator
    VQG
    VQC
    GradientOptimizer
    VQNetExample
