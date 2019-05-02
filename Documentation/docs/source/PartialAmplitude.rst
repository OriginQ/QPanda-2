.. _部分振幅量子虚拟机:

部分振幅量子虚拟机
===============
----

目前用经典计算机模拟量子虚拟机的主流解决方案有全振幅与单振幅两种。除此之外，还有部分振幅量子虚拟机，该方案能在更低的硬件条件下，实现更高的模拟效率。
然而该方案并不适用于所有的量子线路，使用情况如下所述。

算法适用条件
>>>>>>>>>>>>>
----

 - ``量子比特数目要求`` 。部分振幅算法要求量子线路的总量子比特数目必须为偶数，否则无法对量子线路进行有效拆分。
 - ``双比特量子逻辑门要求`` 。若量子线路中出现跨节点的双量子逻辑门，即以线路中量子比特数为总量子比特数的一半为界线，双量子逻辑门的控制位量子比特与目标位量子比特分别处于界线的两侧，称为跨节点。部分振幅模拟由于其算法特殊性，对于某些跨节点双量子逻辑门，如 ``CNOT`` 、 ``CZ`` 等，可以将其分解为 ``P0`` 、 ``P1`` 和基础单量子逻辑门的组合，而不支持如 ``CR`` 、``iSWAP`` 、 ``SqiSWAP`` 等。

使用介绍
>>>>>>>>>>>>>>>>
----

QPanda2中设计了 ``PartialAmplitudeQVM`` 类用于运行部分振幅模拟量子计算，同时提供了相关接口，它的使用很简单。

首先构建一个部分振幅量子虚拟机

    .. code-block:: c

        auto machine = new PartialAmplitudeQVM();

然后必须使用 ``PartialAmplitudeQVM::init()`` 初始化量子虚拟机环境

    .. code-block:: c

        machine->init();

接着进行量子程序的构建、装载工作

    .. code-block:: c

        auto prog = QProg();
        auto qlist = machine->allocateQubits(10);
        auto clist = machine->allocateCBits(10);

        for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
        prog << CZ(qlist[1], qlist[5]) << CZ(qlist[3], qlist[5]) << CZ(qlist[2], qlist[4]);
        ...
        machine->run(prog);

最后调用计算接口，我们设计多种返回值的接口用于满足不同的计算需求，具体见示例所述：

实例
>>>>>>>>>>
----

.. _部分振幅示例程序:
以下示例展示了单振幅量子虚拟机接口的使用方式

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            auto machine = new SingleAmplitudeQVM();
            machine->init();

            auto prog = QProg();
            auto qlist = machine->allocateQubits(10);
            auto clist = machine->allocateCBits(10);

            auto prog = QProg();
            for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
            prog << CZ(qlist[1], qlist[5])
                 << CZ(qlist[3], qlist[7])
                 << CZ(qlist[0], qlist[4])
                 << RZ(qlist[7], PI / 4)
                 << RX(qlist[5], PI / 4)
                 << RX(qlist[4], PI / 4)
                 << RY(qlist[3], PI / 4)
                 << CZ(qlist[2], qlist[6])
                 << RZ(qlist[3], PI / 4)
                 << RZ(qlist[8], PI / 4)
                 << CZ(qlist[9], qlist[5])
                 << RY(qlist[2], PI / 4)
                 << RZ(qlist[9], PI / 4)
                 << CZ(qlist[2], qlist[3]);
                
            machine->run(prog);
            auto res = machine->getQStat();
            for (auto val : res)
            {
                std::cout << val<< std::endl;
            }

上述程序的计算结果如下

    .. code-block:: c

        (-0.00647209,-0.00647209)
        (9.46438e-18,-0.00915291)
        (-0.00647209,-0.00647209)
        ...

若使用其它接口对上述量子程序进行操作：

    - ``PMeasure(int)`` ,使用示例

        .. code-block:: c

            auto res = machine->PMeasure(6);
            for (auto val :res)
            {
                std::cout << val.first << " : " << val.second << std::endl;
            }

        结果输出如下：

        .. code-block:: c

            0 : 8.37758e-05
            1 : 8.37758e-05
            2 : 8.37758e-05
            3 : 8.37758e-05
            4 : 0.000488281
            5 : 0.000488281

    - ``PMeasure(QVec,int)`` ,使用示例

        .. code-block:: c

            QVec qvec;
            for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { qvec.emplace_back(val); });

            auto res = machine->PMeasure(qvec,6);
            for (auto val :res)
            {
                std::cout << val.first << " : " << val.second << std::endl;
            }

        结果输出如下：

        .. code-block:: c

            8.37758e-05
            8.37758e-05
            8.37758e-05
            8.37758e-05
            0.000488281
            0.000488281

    - ``getProbDict(qvec,int)`` ,使用示例

        .. code-block:: c

            QVec qvec;
            for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { qvec.emplace_back(val); });

            auto res = machine->getProbDict(qvec,6);
            for (auto val :res)
            {
                std::cout << val.first << " : " << val.second << endl;
            }

        结果输出如下：

        .. code-block:: c

            0000000000 : 8.37758e-05
            0000000001 : 8.37758e-05
            0000000010 : 8.37758e-05
            0000000011 : 8.37758e-05
            0000000100 : 0.000488281
            0000000101 : 0.000488281

    - ``getProbTupleList(qvec,int)`` ,使用示例

        .. code-block:: c

            QVec qvec;
            for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { qvec.emplace_back(val); });

            auto res = machine->getProbTupleList(qvec,6);
            for (auto val :res)
            {
                std::cout << val.first << " : " << val.second << endl;
            }

        结果输出如下：

        .. code-block:: c

            0 : 8.37758e-05
            1 : 8.37758e-05
            2 : 8.37758e-05
            3 : 8.37758e-05
            4 : 0.000488281
            5 : 0.000488281

