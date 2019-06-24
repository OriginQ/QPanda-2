.. _部分振幅量子虚拟机:

部分振幅量子虚拟机
===============
----

目前用经典计算机模拟量子虚拟机的主流解决方案有全振幅与单振幅两种。除此之外，还有部分振幅量子虚拟机，该方案能在更低的硬件条件下，实现更高的模拟效率。
部分振幅算法的基本思想是将大比特的量子计算线路图拆分成若干个小比特线路图，具体数量视线路扶持情况而定。

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

构建还可以采用另一种方式，即读取QRunes文件形式，例如

    .. code-block:: c

        machine->run("D:\\QRunes");

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
                 << CR(qlist[2], qlist[7], PI / 2);
                
            machine->run(prog);
            auto res = machine->getQStat();
            cout << res["0000000000"] << endl;
            cout << res["0000000001"] << endl;

上述程序的计算结果如下

    .. code-block:: c

        (-0.00647209,-0.00647209)
        (8.5444e-18,-0.00915291)
        ...

若使用其他接口：
    - ``PMeasure(std::string)`` ,使用示例

        .. code-block:: c

            auto res = machine->PMeasure("6");
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

    - ``PMeasure(QVec,std::string)`` ,使用示例

        .. code-block:: c

            QVec qv = { qlist[1],qlist[2],qlist[3] ,qlist[4] ,qlist[5] ,qlist[6] ,qlist[7] ,qlist[8],qlist[9] };
            auto res2 = machine->PMeasure(qv, "6");

            for (auto val :res)
            {
                std::cout << val.first << " : " << val.second << std::endl;
            }

        结果输出如下：

        .. code-block:: c

            0 : 0.000167552
            1 : 0.000167552
            2 : 0.000976562
            3 : 0.000976562
            4 : 0.000976562
            5 : 0.000976562

    - ``getProbDict(qvec,std::string)`` ,使用示例

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

    - ``PMeasure_bin_index(std::string)`` ,使用示例

        .. code-block:: c

            auto res = PMeasure_bin_index("0000000001");
            std::cout << res << std::endl;

        通过二进制形式下标测量指定振幅，结果输出如下：

        .. code-block:: c

            8.37758e-05

    - ``PMeasure_dec_index(std::string)`` ,使用示例

        .. code-block:: c

            auto res = PMeasure_bin_index("1");
            std::cout << res << std::endl;

        通过十进制形式下标测量指定振幅，结果输出如下：

        .. code-block:: c

            8.37758e-05

