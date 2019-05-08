量子程序
==============
----

量子程序设计用于量子程序的编写与构造，一般地， 可以理解为一个操作序列。由于量子算法中也会包含经典计算，因而业界设想，最近将来出现的量子计算机是混合结构的，它包含两大部分一部分是经典计算机，负责执行经典计算与控制；另一部分是量子设备，负责执行量子计算。QPanda-2将量子程序的编程过程视作经典程序运行的一部分，在整个外围的宿主机程序中，一定包含创建量子程序的部分。

.. _api_introduction:

接口介绍
>>>>>>>>>>>>>>>>
----

在QPanda2中，QProg是量子编程的一个容器类，是一个量子程序的最高单位。它也是QNode中的一种，初始化一个QCircuit对象有以下两种

C++风格

    .. code-block:: c

        QProg prog = QProg();

C语言风格

    .. code-block:: c

        QProg prog = CreateEmptyQProg();

你可以通过如下方式向QProg尾部填充节点

    .. code-block:: c

        QProg << QNode;

或者
    
    .. code-block:: c

        QProg.pushBackNode(QNode);

QNode的类型有QGate，QPorg，QIf，Measure等等，QProg支持插入所有类型的QNode

通常一个QProg类型内部结构复杂，需要对其进行拆分遍历等过程，QPanda2提供了相关接口

获取QProg内部第一个节点与最后一个节点

    .. code-block:: c

        QProg prog = QProg();
        NodeIter first_node = prog.getFirstNodeIter();
        NodeIter last_node  = prog.getLastNodeIter();

在QProg内部插入与删除节点操作

    .. code-block:: c

        QProg prog = QProg();
        NodeIter insert_node_iter = prog.insertQNode(NodeIter&, QNode*);
        NodeIter delete_node_iter = prog.deleteQNode(NodeIter&);

    .. note:: 
        - NodeIter是一个关于QNode的代理类，类似于STL容器的迭代器类型，支持自增与自减等操作
        - QProg节点插入删除操作会返回指向原位置的NodeIter

    __ ./QProg.html#api-introduction

    __ ./QCircuit.html#api-introduction

    __ ./QGate.html#api-introduction

    __ ./Measure.html#api-introduction

    __ ./QIf.html#api-introduction

    __ ./QWhile.html#api-introduction

实例
>>>>>>>>>>
----

    .. code-block:: c

        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            init();
            auto qvec = qAllocMany(4);
            auto cvec = cAllocMany(4);

            QProg prog;
            prog << H(qvec[0]) << X(qvec[1])
                << iSWAP(qvec[0], qvec[1])
                << CNOT(qvec[1], qvec[2])
                << H(qvec[3]) << MeasureAll(qvec ,cvec);

            auto result = runWithConfiguration(prog, cvec, 1000);
            for (auto &val : result)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            finalize();
            return 0;
        }


运行结果：

    .. code-block:: c

        1000, 242
        1001, 277
        1110, 254
        1111, 227