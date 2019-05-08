.. _pyQPanda-Core:

核心组件
========
----

我们对QPanda2的核心工具组件接口进行了封装，它们的使用方式与C++版本的接口类似，但是稍有不同，具体见下方详细介绍。

量子逻辑门
>>>>>>>>>>
----

pyQPanda构建量子逻辑门的方式与QPanda2完全相同，只不过简化了QPanda2 :ref:`量子逻辑门` 相关API，仅仅保留了dagger和control方法

    .. code-block:: python
    
        from pyqpanda import *

        init(QMachineType.CPU)
        q = qAlloc_many(5)
        c = cAlloc_many(5)

        control_list = [q[1],q[2]]
        H_dagger  = H(q[0]).dagger()
        H_control = H(q[0]).control(control_list)
        finalize()

量子逻辑门的dagger和control方法均是复制一份当前的量子逻辑门，并更新复制的量子逻辑门的dagger与control标记

量子线路与量子程序
>>>>>>>>>>>>>>>>>
----

pyQPanda初始化量子线路与量子程序的方式与QPanda2完全相同，均有两种风格接口，即

    .. code-block:: python

        cir = QCircuit()
        prog = QProg()

或者

    .. code-block:: python

        cir  = CreateEmptyQCircuit()
        prog = CreateEmptyQProg()

与QPanda2不同的是，pyQPanda向量子线路与量子程序插入节点使用 ``.insert`` 方法

    .. code-block:: python

        from pyqpanda import *

        init(QMachineType.CPU)

        prog = QProg()
        cir = QCircuit()

        q = qAlloc_many(6)
        c = cAlloc_many(6)

        cir.insert(H(q[1]))
        prog.insert(H(q[0]))\
            .insert(CNOT(q[0],q[1]))\
            .insert(cir)

        finalize()

同时在pyQPanda中额外封装了一些构建量子逻辑线路或量子逻辑门的接口，简化了程序设计过程，比如

    - ``Toffoli(control1,control2,target)`` 。三比特量子逻辑门Toffoli实际上是CCNOT门，前两个参数是控制比特，最后一个参数是目标比特。
    
    - ``single_gate_apply_to_all(gate,qubit_list)`` 。该接口的作用是对输入qubits合集中的每一个元素进行目标单比特门操作，返回这些量子逻辑门组成的量子线路
    
    - ``Hadamard_Circuit(qubit_list)`` 。该接口是对上一个接口的进一步简化，作用是对输入qubits合集中的每一个元素进行Hadamard门操作，返回Hadamard门组成的量子线路。
    
    - ``meas_all(qubit_list,cbit_list)`` 。该接口用于对所有qubit进行测量的操作，返回一个QProg。

量子程序控制流
>>>>>>>>>>
----

pyQPanda中创建QIf、QWhile节点的方法与QPanda2中的完全相同，在此不做赘述。