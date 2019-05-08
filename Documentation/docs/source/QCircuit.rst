量子线路
====================
----

量子线路，也称量子逻辑电路是最常用的通用量子计算模型，表示在抽象概念下，对于量子比特进行操作的线路。组成包括了量子比特、线路（时间线），以及各种逻辑门。最后常需要量子测量将结果读取出来。

不同于传统电路是用金属线所连接以传递电压讯号或电流讯号，在量子线路中，线路是由时间所连接，亦即量子比特的状态随着时间自然演化，过程中是按照哈密顿运算符的指示，一直到遇上逻辑门而被操作。

由于组成量子线路的每一个量子逻辑门都是一个 ``酉算子`` ，所以整个量子线路整体也是一个大的酉算子。


量子算法线路图
>>>>>>>>>>>>>>>>>>>>>
----

在目前的量子计算理论研究中，各种量子算法常用量子线路表示，比如下方列出的量子算法中的 ``HHL算法`` 量子线路图。

.. image:: images/hhl.bmp
   :align: center   

.. _api_introduction:

接口介绍
>>>>>>>>>>>>>>>>>>>>>>>>>>>>
----

在QPanda2中，QCircuit类是一个仅装载量子逻辑门的容器类型，它也是QNode中的一种，初始化一个QCircuit对象有以下两种

    .. code-block:: python

        cir = QCircuit()

或

    .. code-block:: python

        cir = CreateEmptyCircuit()

你可以通过如下方式向QCircuit尾部填充节点

    .. code-block:: python

        cir.insert(node)

node的类型可以为QGate或QCircuit。所以QCircuit类是一个QGate对象和另一些QCircuit对象的集合。

我们还可以获得QCircuit的转置共轭之后的量子线路，使用方式为：

        .. code-block:: python
        
            cir_dagger = cir.dagger()

如果想复制当前的量子线路，并给复制的量子线路添加控制比特，可以使用下面的方式：

        .. code-block:: python
            
                qvec = [qubits[0], qubits[1]]
                cir_control = cir.control(qvec)

    .. note:: 
        - 向QCircuit中插入QPorg，QIf，Measure中不会报错，但是运行过程中可能会产生预料之外的错误
        - 一个构建好的QCircuit不能直接参与量子计算与模拟，需要进一步构建成QProg类型

为了简化量子线路的设计，pyqpanda中额外封装了一些构建量子逻辑线路的接口，如下：

    .. code-block:: python

        cir = single_gate_apply_to_all(gate, qubit_list) # 作用是对输入qubits合集中的每一个元素进行目标单比特门操作，返回这些量子逻辑门组成的量子线路
        cir2 = Hadamard_Circuit(qubit_list) # 作用是对输入qubits合集中的每一个元素进行Hadamard门操作，返回Hadamard门组成的量子线路


实例
>>>>>>>>>>>
----

    .. code-block:: python
    
        from pyqpanda import *

        if __name__ == "__main__":

            init(QMachineType.CPU)
            qubits = qAlloc_many(4)
            cbits = cAlloc_many(4)
            prog = QProg()
            circuit = CreateEmptyCircuit()

            circuit.insert(H(qubits[0])) \
                .insert(CNOT(qubits[0], qubits[1])) \
                .insert(CNOT(qubits[1], qubits[2])) \
                .insert(CNOT(qubits[2], qubits[3]))

            prog.insert(circuit).insert(Measure(qubits[0], cbits[0]))
            result = run_with_configuration(prog, cbits, 1000)
            print(result)

            finalize()


运行结果：

    .. code-block:: python

        {'0000': 508, '1000': 492}

