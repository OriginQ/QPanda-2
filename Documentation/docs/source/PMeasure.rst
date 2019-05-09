.. _PMeasure:

概率测量
==================

概率测量是指获得目标量子比特的振幅，目标量子比特可以是一个量子比特也可以是多个量子比特的集合。 在QPanda2中概率测量又称为 ``PMeasure`` 。
概率测量和 :ref:`Measure` 是完全不同的过程，Measure是执行了一次测量， 并返回一个确定的0/1结果，并且改变了量子态，
PMeasure是获得我们所关注的量子比特的振幅，并不会改变量子态，PMeasure的输入参数是 ``QVec``， 它指定了我们关注的量子比特。
例如，一共有10个Qubit的系统，我们指定了前三个Qubit作为PMeasure的目标，就会输出一个长度为8的list。

接口介绍
----------------

QPanda2提供了三种获得PMeasure结果的方式，其中有 ``prob_run_list`` 、 ``prob_run_tuple_list``  、 ``prob_run_dict``。

- ``prob_run_list`` ： 获得目标量子比特的概率测量结果列表。
- ``prob_run_tuple_list``： 获得目标量子比特的概率测量结果， 其对应的下标为十进制。
- ``prob_run_dict`` ： 获得目标量子比特的概率测量结果， 其对应的下标为二进制。

这三个函数的使用方式是一样的，下面就以 ``prob_run_dict`` 为例介绍，使用方式如下：

    .. code-block:: python

        prog = QProg()
        prog.insert(H(qubits[0]))\
            .insert(CNOT(qubits[0], qubits[1]))\
            .insert(CNOT(qubits[1], qubits[2]))\
            .insert(CNOT(qubits[2], qubits[3]))

        result = prob_run_dict(prog, qubits, 3)

第一个参数是量子程序， 第二个参数是 ``QVec`` 它指定了我们关注的量子比特。
第三个参的值为-1时，是指我们以第二个参数中所有的量子比特作为目标，当其值不为-1时，则表示我们关注 ``QVec`` 中的前几个。
如上述例子，一共有4个Qubit的系统， 第三个参数设置为3，得到结果将会是8个元素。

实例
-----------

    .. code-block:: python

        from pyqpanda import *

        if __name__ == "__main__":
            qvm = init_quantum_machine(QMachineType.CPU)
            qubits = qvm.qAlloc_many(2)
            cbits = qvm.cAlloc_many(2)

            prog = QProg()
            prog.insert(H(qubits[0]))\
                .insert(CNOT(qubits[0], qubits[1]))

            print("prob_run_dict: ")
            result1 = prob_run_dict(prog, qubits, -1)
            print(result1)

            print("prob_run_tuple_list: ")
            result2 = prob_run_tuple_list(prog, qubits, -1)
            print(result2)

            print("prob_run_list: ")
            result3 = prob_run_list(prog, qubits, -1)
            print(result3)

            qvm.finalize()


运行结果：

    .. code-block:: python

        prob_run_dict: 
        {'00': 0.4999999999999999, '01': 0.0, '10': 0.0, '11': 0.4999999999999999}
        prob_run_tuple_list: 
        [(0, 0.4999999999999999), (3, 0.4999999999999999), (1, 0.0), (2, 0.0)]
        prob_run_list: 
        [0.4999999999999999, 0.0, 0.0, 0.4999999999999999]


