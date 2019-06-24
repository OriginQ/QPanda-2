QWhile
==============
----

量子程序循环控制操作，输入参数为条件判断表达式，功能是执行while循环操作。

.. _api_introduction:

接口介绍
>>>>>>>>>>>>>
----

在QPanda2中，QWhileProg类用于表示执行量子程序while循环操作，它也是QNode中的一种，初始化一个QWhileProg对象有以下两种

    .. code-block:: python

        qwile = QWhileProg(ClassicalCondition, QNode)

或

    .. code-block:: python

        qwile = CreateWhileProg(ClassicalCondition, QNode)

上述函数需要提供两个参数，即ClassicalCondition量子表达式与QNode节点
可以传入的QNode类型有： QProg、QCircuit、QGate、QWhileProg、QIfProg、QMeasure。

实例
>>>>>>>>>>
----

    .. code-block:: python

        from pyqpanda import *

        if __name__ == "__main__":

            init(QMachineType.CPU)
            qubits = qAlloc_many(3)
            cbits = cAlloc_many(3)
            cbits[0].setValue(0)
            cbits[1].setValue(1)

            prog = QProg()
            prog_while = QProg()
            prog_while.insert(H(qubits[0])).insert(H(qubits[1])).insert(H(qubits[2]))\
                    .insert(assign(cbits[0], cbits[0] + 1)).insert(Measure(qubits[1], cbits[1]))
            qwhile = CreateWhileProg(cbits[1], prog_while)
            prog.insert(qwhile)

            result = directly_run(prog)
            print(cbits[0].eval())
            print(result)
            finalize()


运行结果：

    .. code-block:: python

        2
        {'c1': False}