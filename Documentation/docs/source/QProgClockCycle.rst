.. _QProgClockCycle:

统计量子程序时钟周期
=======================

简介
--------------

已知每个量子逻辑门在运行时所需时间的条件下，估算一个量子程序运行所需要的时间。每个量子逻辑门的时间设置在项目的元数据配置文件 ``QPandaConfig.xml`` 中，
如果未设置则会给定一个默认值，单量子门的默认时间为2，双量子门的时间为5。

配置文件可仿照下面设置
***********************

.. code-block:: xml

    <QGate>
        <SingleGate>
            <Gate time = "2">rx</Gate>
            <Gate time = "2">Ry</Gate>
            <Gate time = "2">RZ</Gate>
            <Gate time = "2">S</Gate>
            <Gate time = "2">H</Gate>
            <Gate time = "2">X1</Gate>
        </SingleGate>
        <DoubleGate>
            <Gate time = "5">CNOT</Gate>
            <Gate time = "5">CZ</Gate>
            <Gate time = "5">ISWAP</Gate>
        </DoubleGate>
    </QGate>

接口介绍
--------------

``QProgClockCycle`` 类是QPanda 2提供的一个将量子程序转换为Quil指令集的工具类，我们先用QPanda 2构建一个量子程序：

    .. code-block:: c
          
        auto qubits = qvm->allocateQubits(4);
        auto prog = CreateEmptyQProg();
        prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
                << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI/4);

然后调用 ``QProgClockCycle`` 类得到量子程序的时钟周期

    .. code-block:: c
          
        QProgClockCycle t(qvm);
        t.traversal(prog);
        auto time = t.count();

我们还可以使用QPanda2封装的一个接口：

    .. code-block:: c
          
        auto time = getQProgClockCycle(qvm, prog);   

实例
--------------

    .. code-block:: c
    
        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            auto qvm = initQuantumMachine();
            auto qubits = qvm->allocateQubits(4);
            auto prog = CreateEmptyQProg();
            prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
                    << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI/4);

            auto time = getQProgClockCycle(qvm, prog);
            std::cout << "clockCycle : " << time << std::endl;

            qvm->finalize();
            delete qvm;

            return 0;
        }

运行结果：

    .. code-block:: c

        clockCycle : 14
    
