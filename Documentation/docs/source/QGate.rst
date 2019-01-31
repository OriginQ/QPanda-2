量子逻辑门
====================
----

经典计算中，最基本的单元是比特，最基本的控制模式是逻辑门。类似地，在量子计算中，特别是量子线路计算模型中，处理量子比特的基本方式就是量子逻辑门。
它们是量子线路的构建模块，就像传统数字电路的经典逻辑门一样，与许多经典逻辑门不同，量子逻辑门是可逆的。

量子逻辑门由酉矩阵表示。最常见的量子门在一个或两个量子位的空间上工作，就像常见的经典逻辑门在一个或两个位上操作一样。

常见量子逻辑门矩阵形式
>>>>>>>>>>>>>>>>>>>>>
----

.. image:: imags/H.svg
   :align: left
   :width: 65

``Hadamard`` ：

.. math:: \begin{bmatrix} 1/\sqrt {2} & 1/\sqrt {2} \\ 1/\sqrt {2} & -1/\sqrt {2} \end{bmatrix}\quad

.. image:: imags/X.svg
   :align: left
   :width: 65

``Pauli-X`` ：

.. math:: \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\quad

.. image:: imags/Y.svg
   :align: left
   :width: 65

``Pauli-Y`` ：

.. math:: \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}\quad

.. image:: imags/Z.svg
   :align: left
   :width: 65

``Pauli-Z`` ：

.. math:: \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}\quad

.. image:: imags/not.svg
   :align: left
   :width: 65

``NOT`` ：

.. math:: \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\quad

.. image:: imags/Xθ.svg
   :align: left
   :width: 65

``RX`` ：

.. math:: \begin{bmatrix} \cos(θ/2) & -1i×\sin(θ/2) \\ -1i×\sin(θ/2) & \cos(θ/2) \end{bmatrix}\quad

.. image:: imags/Yθ.svg
   :align: left
   :width: 65

``RY`` ：

.. math:: \begin{bmatrix} \cos(θ/2) & \sin(θ/2) \\ \sin(θ/2) & \cos(θ/2) \end{bmatrix}\quad

.. image:: imags/Zθ.svg
   :align: left
   :width: 65

``RZ`` ：

.. math:: \begin{bmatrix} \exp(-iθ/2) & 0 \\ 0 & \exp(iθ/2) \end{bmatrix}\quad

.. image:: imags/+.svg
   :align: left
   :width: 65

``CNOT`` ：

.. math:: \begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}\quad

.. image:: imags/CR.svg
   :align: left
   :width: 65

``CR`` ：

.. math:: \begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & \exp(iθ) \end{bmatrix}\quad

.. image:: imags/切换.svg
   :align: left
   :width: 65

``iSWAP`` ：

.. math:: \begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 0 & -i & 0 \\ 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}\quad

.. image:: imags/Toff.svg
   :align: left
   :width: 65

``Toffoli`` ：

.. math:: \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0  \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 
                          0 & 0 & 1 & 0 & 0 & 0 & 0 & 0  \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 
                          0 & 0 & 0 & 0 & 1 & 0 & 0 & 0  \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 
                          0 & 0 & 0 & 0 & 0 & 0 & 0 & 1  \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ \end{bmatrix}\quad


量子逻辑门类及常用接口介绍
>>>>>>>>>>>>>>>>>>>>>>>>
----

.. cpp:class:: QGate

    该类用于表述一个量子逻辑门节点的各项信息，同时包含多种可调用的接口。

    .. cpp:function:: QGate::getNodeType()

       **功能**
        - 获取量子逻辑门节点类型

       **参数**
        - 无

       **返回值**
        - 节点类型

    .. cpp:function:: QGate::setDagger(bool)

       **功能**
        - 设置量子逻辑门转置共轭形式

       **参数**
        - bool

       **返回值**
        - bool

    .. cpp:function:: QGate::setControl(std::vector<Qubit*>&)

       **功能**
        - 设置量子逻辑门受控状态

       **参数**
        - std::vector<Qubit *>

       **返回值**
        - bool

    .. cpp:function:: QGate::getQGate()

       **功能**
        - 获取量子逻辑门参数

       **参数**
        - 无

       **返回值**
        - QuantumGate*

    .. cpp:function:: QGate::dagger()

       **功能**
        - 返回一个当前节点量子逻辑门转置共轭形式的新节点

       **参数**
        - 无

       **返回值**
        - QGate

    .. cpp:function:: QGate::control(std::vector<Qubit*>&)

       **功能**
        - 返回一个当前节点量子逻辑门施加控制操作的新节点

       **参数**
        - 无

       **返回值**
        - QGate

实例
>>>>>>>>>>
----

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            init(QuantumMachine_type::CPU);
            auto c = cAllocMany(2);
            auto q = qAllocMany(4);

            auto gate0 = H(q[0]);
            gate0.setDagger(true); // 设置量子逻辑门转置共轭
            auto gate1 = CNOT(q[0], q[1]);
            auto gate2 = CNOT(q[1], q[2]);
            std::vector<Qubit *> qubits = {q[0], q[3]};
            gate2.setControl(qubits); // 设置逻辑门的受控量子比特
            auto gate3 = CNOT(q[2], q[3]);

            auto prog = CreateEmptyQProg();
            prog << gate0 << gate1 << gate2 << gate3;
            auto result = probRunTupleList(prog, q);
            for(auto & aiter : result)
            {
                std::cout << aiter.first << " : " << aiter.second << std::endl;
            }

            finalize();
            return 0;
        }