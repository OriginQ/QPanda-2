.. _PMeasure:

概率测量
==================

概率测量是指获得目标量子比特的振幅，目标量子比特可以是一个量子比特也可以是多个量子比特的集合。 在QPanda2中概率测量又称为 ``PMeasure`` 。
概率测量和 :ref:`Measure` 是完全不同的过程，Measure是执行了一次测量， 并返回一个确定的0/1结果，并且改变了量子态，
PMeasure是获得我们所关注的量子比特的振幅，并不会改变量子态，PMeasure的输入参数是 ``QVec``， 它指定了我们关注的量子比特。
例如，一共有10个Qubit的系统，我们指定了前三个Qubit作为PMeasure的目标，就会输出一个长度为8的vector。

接口介绍
----------------

QPanda2提供了三种获得PMeasure结果的方式，其中有 ``probRunList`` 、 ``probRunTupleList``  、 ``probRunDict``。

- ``probRunList`` ： 获得目标量子比特的概率测量结果， 并没有其对应的下标。
- ``probRunTupleList``： 获得目标量子比特的概率测量结果， 其对应的下标为十进制。
- ``probRunDict`` ： 获得目标量子比特的概率测量结果， 其对应的下标为二进制。

这三个函数的使用方式是一样的，下面就以 ``probRunDict`` 为例介绍，使用方式如下：

    .. code-block:: c

        auto qubits = qvm->allocateQubits(4);
        auto cbits = qvm->allocateCBits(4);

        QProg prog;
        prog   << H(qubits[0])
                << CNOT(qubits[0], qubits[1])
                << CNOT(qubits[1], qubits[2])
                << CNOT(qubits[2], qubits[3]);
        auto result = probRunDict(prog, qubits， 3);

第一个参数是量子程序， 第二个参数是 ``QVec`` 它指定了我们关注的量子比特。
第三个参的值为-1时，是指我们以第二个参数中所有的量子比特作为目标，当其值不为-1时，则表示我们关注 ``QVec`` 中的前几个。
如上述例子，一共有4个Qubit的系统， 第三个参数设置为3，得到结果将会是8个元素。

除了上述的方式外，我们还可以先使用 ``directlyRun``， 再调用 ``getProbList`` 、 ``getProbTupleList`` 、 ``getProbDict`` 得到和上述三种方法一样的结果。

实例
-----------

    .. code-block:: c

        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            auto qvm = initQuantumMachine();
            auto qubits = qvm->allocateQubits(2);
            auto cbits = qvm->allocateCBits(2);

            QProg prog;
            prog << H(qubits[0])
                << CNOT(qubits[0], qubits[1]);

            std::cout << "probRunDict: " << std::endl;
            auto result1 = probRunDict(prog, qubits);
            for (auto &val: result1)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            std::cout << "probRunTupleList: " << std::endl;
            auto result2 = probRunTupleList(prog, qubits);
            for (auto &val: result2)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            std::cout << "probRunList: " << std::endl;
            auto result3 = probRunList(prog, qubits);
            for (auto &val: result3)
            {
                std::cout << val<< std::endl;
            }

            qvm->finalize();
            delete qvm;
            return 0;
        }

运行结果：

    .. code-block:: c

        probRunDict: 
        00, 0.5
        01, 0
        10, 0
        11, 0.5
        probRunTupleList: 
        0, 0.5
        3, 0.5
        1, 0
        2, 0
        probRunList: 
        0.5
        0
        0
        0.5

