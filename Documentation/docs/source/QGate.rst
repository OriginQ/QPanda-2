量子逻辑门
====================
----

经典计算中，最基本的单元是比特，最基本的控制模式是逻辑门。类似地，在量子计算中，特别是量子线路计算模型中，处理量子比特的基本方式就是量子逻辑门。
它们是量子线路的构建模块，就像传统数字电路的经典逻辑门一样，与许多经典逻辑门不同，量子逻辑门是可逆的。

量子逻辑门由酉矩阵表示。最常见的量子门在一个或两个量子位的空间上工作，就像常见的经典逻辑门在一个或两个位上操作一样。

常见量子逻辑门矩阵形式
>>>>>>>>>>>>>>>>>>>>>>>>
----

.. |H| image:: images/H.svg
   :width: 70px
   :height: 70px

.. |X| image:: images/X.svg
   :width: 70px
   :height: 70px

.. |Y| image:: images/Y.svg
   :width: 70px
   :height: 70px
   
.. |Z| image:: images/Z.svg
   :width: 70px
   :height: 70px

.. |NOT| image:: images/not.svg
   :width: 70px
   :height: 70px

.. |RX| image:: images/Xθ.svg
   :width: 70px
   :height: 70px

.. |RY| image:: images/Yθ.svg
   :width: 70px
   :height: 70px

.. |RZ| image:: images/Zθ.svg
   :width: 70px
   :height: 70px

.. |CNOT| image:: images/+.svg
   :width: 70px
   :height: 70px

.. |CR| image:: images/CR.svg
   :width: 70px
   :height: 70px

.. |iSWAP| image:: images/切换.svg
   :width: 70px
   :height: 70px

.. |Toffoli| image:: images/Toff.svg
   :width: 70px
   :height: 70px

================================================================================================ =======================         ============================================================================================================================
|H|                                                                                                 ``Hadamard``                        .. math:: \begin{bmatrix} 1/\sqrt {2} & 1/\sqrt {2} \\ 1/\sqrt {2} & -1/\sqrt {2} \end{bmatrix}\quad
|X|                                                                                                 ``Pauli-X``                         .. math:: \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\quad
|Y|                                                                                                 ``Pauli-Y``                         .. math:: \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}\quad
|Z|                                                                                                 ``Pauli-Z``                         .. math:: \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}\quad
|NOT|                                                                                               ``NOT``                             .. math:: \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\quad
|RX|                                                                                                ``RX``                              .. math:: \begin{bmatrix} \cos(θ/2) & -1i×\sin(θ/2) \\ -1i×\sin(θ/2) & \cos(θ/2) \end{bmatrix}\quad
|RY|                                                                                                ``RY``                              .. math:: \begin{bmatrix} \cos(θ/2) & \sin(θ/2) \\ \sin(θ/2) & \cos(θ/2) \end{bmatrix}\quad
|RZ|                                                                                                ``RZ``                              .. math:: \begin{bmatrix} \exp(-iθ/2) & 0 \\ 0 & \exp(iθ/2) \end{bmatrix}\quad
|CNOT|                                                                                              ``CNOT``                            .. math:: \begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}\quad
|CR|                                                                                                ``CR``                              .. math:: \begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & \exp(iθ) \end{bmatrix}\quad
|iSWAP|                                                                                             ``iSWAP``                           .. math:: \begin{bmatrix} 1 & 0 & 0 & 0  \\ 0 & 0 & -i & 0 \\ 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}\quad
|Toffoli|                                                                                           ``Toffoli``                         .. math:: \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0  \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1  \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ \end{bmatrix}\quad
================================================================================================ =======================         ============================================================================================================================

.. _api_introduction:

接口介绍
>>>>>>>>>>>>>>>>>>>>>>>>>>>>
----

.. cpp:class:: QGate

    该类用于表述一个量子逻辑门节点的各项信息，同时包含多种可调用的接口。

    .. cpp:function:: NodeType getNodeType()

       **功能**
            获取量子逻辑门节点类型
       **参数**
            无
       **返回值**
            节点类型

    .. cpp:function:: void setDagger(bool)

       **功能**
            设置量子逻辑门转置共轭形式
       **参数**
            - bool 是否dagger
       **返回值**
            无

    .. cpp:function:: void setControl(std::vector<Qubit*>&)

       **功能**
            设置量子逻辑门受控状态
       **参数**
            - std::vector<Qubit *> 设置作为控制位的一组量子比特
       **返回值**
            无

    .. cpp:function:: QuantumGate *getQGate()

       **功能**
            获取量子逻辑门参数
       **参数**
            无
       **返回值**
            量子逻辑门参数

    .. cpp:function:: QGate dagger()

       **功能**
            返回一个当前节点量子逻辑门转置共轭形式的副本
       **参数**
            无
       **返回值**
            量子逻辑门

    .. cpp:function:: QGate control(std::vector<Qubit*>&)

       **功能**
            返回一个当前节点量子逻辑门施加控制操作的副本
       **参数**
            - std::vector<Qubit*>& 设置作为控制位的一组量子比特
       **返回值**
            量子逻辑门

.. note:: QGate构建时必须接受参数，否则是没有意义的，参数一般是Qubit类型和浮点数类型，浮点数类型一般象征着可变的角度。


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

            auto prog = CreateEmpxyQProg();
            prog << gate0 << gate1 << gate2 << gate3;
            auto result = probRunTupleList(prog, q);
            for(auto & aiter : result)
            {
                std::cout << aiter.first << " : " << aiter.second << std::endl;
            }

            finalize();
            return 0;
        }