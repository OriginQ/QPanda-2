量子程序
===========
----

量子程序设计用于书写量子计算机程序。由于量子算法中也会包含经典计算，因而业界设想，最近将来出现的量子计算机是混合结构的，它包含两大部分一部分是经典计算机，负责执行经典计算与控制；另一部分是量子设备，负责执行量子计算。

量子程序类
>>>>>>>>>>
----

.. cpp:class:: QProg

    该类用于表述一个量子程序节点的各项信息，同时包含多种可调用的接口。

    .. cpp:function:: getNodeType()

       **功能**
        - 获取节点类型
       **参数**
        - 无
       **返回值**
        - 节点类型

    .. cpp:function:: insertQNode(NodeIter & iter, QNode * pNode)

       **功能**
        - 插入一个节点

       **参数**
        - NodeIter
        - QNode*

       **返回值**
        - NodeIter

    .. cpp:function:: deleteQNode(NodeIter & iter)

       **功能**
        - 删除一个节点

       **参数**
        - NodeIter

       **返回值**
        - NodeIter

    .. cpp:function:: QProg & operator <<(T)

       **功能**
        - 像量子程序中添加节点

       **参数**
        - 量子逻辑门、量子线路、量子程序、量子测量、QIf、QWhile节点类型

       **返回值**
        - 量子程序

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

