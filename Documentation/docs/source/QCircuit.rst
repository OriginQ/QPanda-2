量子线路
====================
----

量子线路，也称量子逻辑电路是最常用的通用量子计算模型，表示在抽象概念下，对于量子比特进行操作的线路。组成包括了量子比特、线路（时间线），以及各种逻辑门。最后常需要量子测量将结果读取出来。

不同于传统电路是用金属线所连接以传递电压讯号或电流讯号，在量子线路中，线路是由时间所连接，亦即量子比特的状态随着时间自然演化，过程中是按照汉密顿运算符的指示，一直到遇上逻辑门而被操作。

由于组成量子线路的每一个量子逻辑门都是一个 ``酉算子`` ，所以整个量子线路整体也是一个大的酉算子。


量子算法线路图
>>>>>>>>>>>>>>>>>>>>>
----

在目前的量子计算理论研究中，各种量子算法常用量子线路表示，比如下方列出的量子算法中的 ``HHL算法`` 量子线路图。


.. image:: imags/hhl.bmp
   :align: center   
   
量子线路类及常用接口介绍
>>>>>>>>>>>>>>>>>>>>>>
----

.. cpp:class:: QCircuit

    该类用于表述一个线路节点的各项信息，同时包含多种可调用的接口。

    .. cpp:function:: QCircuit::getNodeType()

       **功能**
        - 获取节点类型

       **参数**
        - 无

       **返回值**
        - 节点类型

    .. cpp:function:: QCircuit::setDagger(bool)

       **功能**
        - 设置量子线路转置共轭形式

       **参数**
        - bool

       **返回值**
        - 无

    .. cpp:function:: QCircuit::setControl(std::vector<Qubit*>&)

       **功能**
        - 设置量子线路受控状态

       **参数**
        - std::vector<Qubit *>

       **返回值**
        - 无

    .. cpp:function:: QCircuit::isDagger()

       **功能**
        - 判断是否处于转置共轭形式

       **参数**
        - 无

       **返回值**
        - QuantumGate*

    .. cpp:function:: QCircuit::dagger()

       **功能**
        - 返回一个当前量子线路节点转置共轭形式的新节点

       **参数**
        - 无

       **返回值**
        - QCircuit

    .. cpp:function:: QCircuit::control(std::vector<Qubit*>&)

       **功能**
        - 返回一个当前量子线路节点施加控制操作的新节点

       **参数**
        - std::vector<Qubit *>

       **返回值**
        - QCircuit

实例
>>>>>>>>>>>
----

    .. code-block:: c
    
        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            init();
            auto qvec = qAllocMany(4);
            auto cbits = cAllocMany(4);
            QCircuit circuit;
            circuit << H(qvec[0]) << CNOT(qvec[0], qvec[1])
                    << CNOT(qvec[1], qvec[2]) << CNOT(qvec[2], qvec[3]);

            circuit.setDagger(true);
            std::vector<Qubit *> qubits = {qvec[0], qvec[3]};
            circuit.setControl(qubits);
            auto prog = CreateEmptyQProg();
            prog << H(qvec[3]) << circuit << Measure(qvec[3], cbits[3]);
            load(prog);
            run();
            auto result = getResultMap();
            for (auto &val : result)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            finalize();
            return 0;
        }

