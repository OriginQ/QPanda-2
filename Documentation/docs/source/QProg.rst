量子程序
==============
----

量子程序设计用于量子程序的编写与构造，一般地， 可以理解为一个操作序列。由于量子算法中也会包含经典计算，因而业界设想，最近将来出现的量子计算机是混合结构的，它包含两大部分一部分是经典计算机，负责执行经典计算与控制；另一部分是量子设备，负责执行量子计算。QPanda-2将量子程序的编程过程视作经典程序运行的一部分，在整个外围的宿主机程序中，一定包含创建量子程序的部分。

.. _api_introduction:

接口介绍
>>>>>>>>>>>>>>>>
----

.. cpp:class:: QProg

    量子编程的一个容器类，是一个量子程序的最高单位。

    .. cpp:function:: getNodeType()

       **功能**
            获取节点类型。
       **参数**
            无
       **返回值**
            节点类型。

    .. cpp:function:: QProg & operator <<(T)

       **功能**
            像量子程序中添加节点。
       **参数**
            - T QProg__ 、QCircuit__ 、QGate__ 、QMeasure__ 、QIfProg__ 和 QWhileProg__ 类型
       **返回值**
            量子程序。

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
            auto circuit = CreateEmptyCircuit();
            circuit << CNOT(qvec[0], qvec[1]) << CNOT(qvec[1], qvec[2])
                    << CNOT(qvec[2], qvec[3]);
            QProg prog;
            prog << H(qvec[0]) << circuit << Measure(qvec[0], cvec[0]);
            load(prog);

            std::vector<int> measure0 = {0, 0};
            for (int i = 0; i < 10000; i++)
            {
                run();
                auto result = getResultMap();

                for (auto &val : result)
                {
                    if (val.second)
                    {
                        measure0[1]++;
                    }
                    else
                    {
                        measure0[0]++;
                    }
                }
            }

            for (auto &val : measure0)
            {
                std::cout << val << std::endl;;
            }
            finalize();
            return 0;
        }

