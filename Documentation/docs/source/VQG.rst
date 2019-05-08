可变量子逻辑门
======================
要在VQNet中使用量子操作 ``qop`` 或 ``qop_pmeasure`` ，就必须要包含可变量子线路(``VariationalQuantumCircuit``，别名 ``VQC``)，而可变量子逻辑门则是构成 ``VariationalQuantumCircuit`` 的基本单位。 可变量子逻辑门(``VariationalQuantumGate``，别名 ``VQG``)，内部维护着一组变量参数以及一组常量参数。
在构造 ``VQG`` 的时候只能对其中一组参数进行赋值。若含有一组常量参数，则可以通过 ``VQG`` 生成含确定参数的普通量子逻辑门, 若含有变量参数，则可以动态修改参数值，并生成对应的参数的普通量子逻辑门。

目前在 ``pyQPanda`` 中定义了如下可变量子逻辑门，它们都继承自 ``VariationalQuantumGate`` 。

===========================  ========== 
 VQG                           别名
===========================  ==========  
VariationalQuantumGate_H      VQG_H
VariationalQuantumGate_RX     VQG_RX
VariationalQuantumGate_RY     VQG_RY
VariationalQuantumGate_RZ     VQG_RZ
VariationalQuantumGate_CZ     VQG_CZ
VariationalQuantumGate_CNOT   VQG_CNOT
===========================  ========== 

接口介绍
-------------

我们可以通过向可变量子线路中插入可变量子逻辑门，来使用可变量子逻辑门。我们可以向需要传入参数的可变量子逻辑门中传入变量参数，
例如我们对可变量子逻辑门RX和RY传入变量参数x和y。也可以对可变量子逻辑门传入常量参数，例如RZ我们传入了一个常量参数0.12。
我们可以通过修改变量的参数，从而来改变可变量子逻辑门中的参数。

.. code-block:: python

    x = var(1)
    y = var(2)
    
    vqc = VariationalQuantumCircuit()
    vqc.insert(VariationalQuantumGate_H(q[0]))
    vqc.insert(VariationalQuantumGate_RX(q[0], x))
    vqc.insert(VariationalQuantumGate_RY(q[1], y))
    vqc.insert(VariationalQuantumGate_RZ(q[0], 0.12))
    vqc.insert(VariationalQuantumGate_CZ(q[0], q[1]))
    vqc.insert(VariationalQuantumGate_CNOT(q[0], q[1]))

    circuit1 = vqc.feed()

    x.set_value(3)
    y.set_value(4)

    circuit2 = vqc.feed()

实例
----------

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
        vqc.insert(VariationalQuantumGate_RZ(q[0], 0.12))
        vqc.insert(VariationalQuantumGate_CZ(q[0], q[1]))
        vqc.insert(VariationalQuantumGate_CNOT(q[0], q[1]))

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

.. image:: images/VQG_Example.png