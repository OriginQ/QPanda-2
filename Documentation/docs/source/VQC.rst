可变量子线路
===================

在 ``VQNet`` 中量子操作 ``qop`` 和 ``qop_pmeasure`` 都需要使用可变量子线路作为参数。 
可变量子线路(``VariationalQuantumCircuit``，别名 ``VQC``)是用来存储含有可变参数的量子线路信息，
``VQC`` 主要由可变量子逻辑门（``VQG``）组成。使用时可以向 ``VQC`` 插入普通量子逻辑门，或者普通量子线路、以及 ``VQG`` 或另外一个 ``VQC``，
在插入普通量子逻辑门或普通量子线路时，其在内部将普通量子逻辑门转换成一组含有固定参数的 ``VQG``。
变量可以和 ``VQC`` 中的不同 ``VQG`` 相关，``VQC`` 对象会保存变量和 ``VQG`` 之间的映射。

接口介绍
-------------

量子程序 ``QProg`` 无法直接加载可变量子线路，但是我们可以通过调用可变量子线路的 ``feed`` 接口来生成一个普通量子线路。

.. code-block:: python

    x = var(1)
    y = var(2)

    vqc = VariationalQuantumCircuit()
    vqc.insert(VariationalQuantumGate_H(q[0]))
    vqc.insert(VariationalQuantumGate_RX(q[0], x))
    vqc.insert(VariationalQuantumGate_RY(q[1], y))

    circuit = vqc.feed()
    prog = QProg()
    prog.insert(circuit)

实例
-------------

.. code-block:: python

    from pyqpanda import *
    
    if __name__=="__main__":
        machine = init_quantum_machine(QMachineType.CPU_SINGLE_THREAD)
        q = machine.qAlloc_many(2)

        x = var(1)
        y = var(2)


        vqc = VariationalQuantumCircuit()
        vqc.insert(VariationalQuantumGate_H(q[0]))
        vqc.insert(VariationalQuantumGate_RX(q[0], x))
        vqc.insert(VariationalQuantumGate_RY(q[1], y))

        vqc.insert(RZ(q[0], 3))
        
        circ = QCircuit()
        circ.insert(RX(q[0],3))
        circ.insert(RY(q[1],4))

        vqc.insert(circ)

        circuit1 = vqc.feed()

        prog = QProg()
        prog.insert(circuit1)

        print(to_QRunes(prog, machine))

        x.set_value([[3.]])
        y.set_value([[4.]])

        circuit2 = vqc.feed()
        prog2 = QProg()
        prog2.insert(circuit2)
        print(to_QRunes(prog2, machine))

.. image:: images/VQC_Example.png
