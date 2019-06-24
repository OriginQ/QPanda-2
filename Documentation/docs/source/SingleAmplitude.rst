.. _单振幅量子虚拟机:

单振幅量子虚拟机
===============
----

目前我们可以通过量子计算的相关理论，用经典计算机实现模拟量子虚拟机。
量子虚拟机的模拟主要有全振幅与单振幅两种解决方案，其主要区别在于，全振幅一次模拟计算就能算出量子态的所有振幅，单振幅一次模拟计算只能计算出 :math:`2^{N}` 个振幅中的一个。

然而全振幅模拟量子计算时间较长，计算量随量子比特数指数增长，
在现有硬件下，无法模拟超过49量子比特。通过单振幅量子虚拟机技术可以模拟超过49比特，同时模拟速度有较大提升，且算法的计算量不随量子比特数指数提升。

使用介绍
>>>>>>>>>>>>>>>>
----

QPanda2中设计了 ``SingleAmplitudeQVM`` 类用于运行单振幅模拟量子计算，同时提供了相关接口，它的使用也很简单。

首先构建一个单振幅量子虚拟机

    .. code-block:: c

        auto machine = new SingleAmplitudeQVM();
然后必须使用 ``SingleAmplitudeQVM::init()`` 初始化量子虚拟机环境

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

.. _单振幅示例程序:
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

            for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
            prog << CZ(qlist[1], qlist[5])
                 << CZ(qlist[3], qlist[5])
                 << CZ(qlist[2], qlist[4])
                 << CZ(qlist[3], qlist[7])
                 << CZ(qlist[0], qlist[4])
                 << RY(qlist[7], PI / 2)
                 << RX(qlist[8], PI / 2)
                 << RX(qlist[9], PI / 2)
                 << CR(qlist[0], qlist[1], PI)
                 << CR(qlist[2], qlist[3], PI)
                 << RY(qlist[4], PI / 2)
                 << RZ(qlist[5], PI / 4)
                 << RX(qlist[6], PI / 2)
                 << RZ(qlist[7], PI / 4)
                 << CR(qlist[8], qlist[9], PI)
                 << CR(qlist[1], qlist[2], PI)
                 << RY(qlist[3], PI / 2)
                 << RX(qlist[4], PI / 2)
                 << RX(qlist[5], PI / 2)
                 << CR(qlist[9], qlist[1], PI)
                 << RY(qlist[1], PI / 2)
                 << RY(qlist[2], PI / 2)
                 << RZ(qlist[3], PI / 4)
                 << CR(qlist[7], qlist[8], PI);
                
            machine->run(prog);
            auto res = machine->getQStat();
            cout << res["0000000000"] << endl;
            cout << res["0000000001"] << endl;

    getQStat()接口表示输出计算后的量子态复振幅，输出结果用map容器保存，key为量子态对应的字符串，value为对应的振幅，上述程序的计算结果如下

    .. code-block:: c

        (0.040830060839653015,-9.313225746154785e-10j)
        (0.040830060839653015,-9.313225746154785e-10j)
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

            0 : 0.00166709
            1 : 0.00166709
            2 : 0.000286028
            3 : 0.000286028
            4 : 0.000286028
            5 : 0.000286028

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

            0 : 0.00333419
            1 : 0.000572056
            2 : 0.000572056
            3 : 0.00333419
            4 : 0.00333419
            5 : 0.000572056

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

            0000000000 : 0.00166709
            0000000001 : 0.00166709
            0000000010 : 0.000286028
            0000000011 : 0.000286028
            0000000100 : 0.000286028
            0000000101 : 0.000286028

    - ``PMeasure_bin_index(std::string)`` ,使用示例

        .. code-block:: c

            auto res = PMeasure_bin_index("0000000001");
            std::cout << res << std::endl;

        通过二进制形式下标测量指定振幅，结果输出如下：

        .. code-block:: c

            0.00166709

    - ``PMeasure_dec_index(std::string)`` ,使用示例

        .. code-block:: c

            auto res = PMeasure_bin_index("1");
            std::cout << res << std::endl;

        通过十进制形式下标测量指定振幅，结果输出如下：

        .. code-block:: c

            0.00166709

