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

``QProgClockCycle`` 类是pyQPanda提供的一个将量子程序转换为Quil指令集的工具类，我们先用pyQPanda构建一个量子程序：

    .. code-block:: python
          
        prog = QProg()
        prog.insert(H(qubits[0])).insert(CNOT(qubits[0], qubits[1]))\
            .insert(iSWAP(qubits[1], qubits[2])).insert(RX(qubits[3], PI / 4))

然后调用 ``get_clock_cycle`` 接口得到量子程序的时钟周期

    .. code-block:: python
          
        clock_cycle = get_clock_cycle(qvm, prog)

实例
--------------

    .. code-block:: python
    
        from pyqpanda import *

        PI = 3.1415926

        if __name__ == "__main__":
            qvm = init_quantum_machine(QMachineType.CPU)
            qubits = qvm.qAlloc_many(4)
            cbits = qvm.cAlloc_many(4)

            prog = QProg()
            prog.insert(H(qubits[0])).insert(CNOT(qubits[0], qubits[1]))\
                .insert(iSWAP(qubits[1], qubits[2])).insert(RX(qubits[3], PI / 4))

            clock_cycle = get_clock_cycle(qvm, prog)
            print("clock_cycle: " + str(clock_cycle))
            qvm.finalize()


运行结果：

    .. code-block:: c

        clock_cycle: 14

    
