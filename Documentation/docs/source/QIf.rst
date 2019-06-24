QIf
==========
----

QIf表示量子程序条件判断操作，输入参数为条件判断表达式，功能是执行条件判断。

.. _api_introduction:

接口介绍
>>>>>>>>>>>
----

在QPanda2中，QIfProg类用于表示执行量子程序条件判断操作，它也是QNode中的一种，初始化一个QIfProg对象有以下两种：

    .. code-block:: python

        qif = QIfProg(ClassicalCondition, QNode)
        qif = QIfProg(ClassicalCondition, QNode, QNode)

或

    .. code-block:: python

        qif = CreateIfProg(ClassicalCondition, QNode)
        qif = CreateIfProg(ClassicalCondition, QNode, QNode)

上述函数需要提供两种类型参数，即ClassicalCondition量子表达式与QNode节点，
当传入1个QNode参数时，QNode表示正确分支节点，当传入2个QNode参数时，第一个表示正确分支节点，第二个表示错误分支节点。
可以传入的QNode类型有： QProg、QCircuit、QGate、QWhileProg、QIfProg、QMeasure。

实例
>>>>>>>>>
----

    .. code-block:: python

        from pyqpanda import *

        if __name__ == "__main__":

            init(QMachineType.CPU)
            qubits = qAlloc_many(3)
            cbits = cAlloc_many(3)
            cbits[0].setValue(0)
            cbits[1].setValue(3)

            prog = QProg()
            branch_true = QProg()
            branch_false = QProg()
            branch_true.insert(H(qubits[0])).insert(H(qubits[1])).insert(H(qubits[2]))
            branch_false.insert(H(qubits[0])).insert(CNOT(qubits[0], qubits[1])).insert(CNOT(qubits[1], qubits[2]))

            qif = CreateIfProg(cbits[0] > cbits[1], branch_true, branch_false)
            prog.insert(qif)
            result = prob_run_tuple_list(prog, qubits, -1)
            print(result)

            finalize()


运行结果：

    .. code-block:: python

        [(0, 0.4999999999999999), (7, 0.4999999999999999), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0), (5, 0.0), (6, 0.0)]

