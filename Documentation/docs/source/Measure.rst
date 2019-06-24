.. _Measure:

量子测量
================

量子测量是指通过外界对量子系统进行干扰来获取需要的信息，测量门使用的是蒙特卡洛方法的测量。在量子线路中用如下图标表示：

.. image:: images/measure-01.png
    :width: 65

.. _api_introduction:

接口介绍
----------------

本章主要介绍获得量子测量对象、根据配置运行含有量子测量的量子程序、快速测量。

在量子程序中我们需要对某个量子比特做测量操作，并把测量结果存储到经典寄存器上，可以通过下面的方式获得一个测量对象：

    .. code-block:: c

        auto measure = Measure(qubit, cbit); 

可以看到Measure接两个参数， 第一个是测量比特，第二个是经典寄存器。

如果想测量所有的量子比特并将其存储到对应的经典寄存器上， 可以如下操作：

    .. code-block:: c

        auto measure_all = MeasureAll(qubits， cbits);

其中qubits的类型是 ``QVec`` ， cbits的类型是 ``vector<ClassicalCondition>``。

.. note:: ``MeasureAll`` 的返回值类型是 ``QProg``。

在得到含有量子测量的程序后，我们可以调用 ``directlyRun`` 或 ``runWithConfiguration`` 来得到量子程序的测量结果。

``directlyRun`` 的功能是运行量子程序并返回运行的结果， 使用方法如下：

    .. code-block:: c

        QProg prog;
        prog << H(qubits[0])
            << CNOT(qubits[0], qubits[1])
            << CNOT(qubits[1], qubits[2])
            << CNOT(qubits[2], qubits[3])
            << Measure(qubits[0], cbits[0]);

        auto result = directlyRun(prog);

``runWithConfiguration`` 的功能是末态在目标量子比特序列在量子程序多次运行结果中出现的次数， 使用方法如下：

    .. code-block:: c

        QProg prog;
        prog   << H(qubits[0])
                << H(qubits[1])
                << H(qubits[2])
                << H(qubits[3])
                << MeasureAll(qubits, cbits); // 测量所有的量子比特

        auto result = runWithConfiguration(prog, cbits, 1000);

其中第一个参数是量子程序，第二个参数是经典寄存器， 第三个参数是运行的次数。

实例
----------

    .. code-block:: c

        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            auto qvm = initQuantumMachine();
            auto qubits = qvm->allocateQubits(4);
            auto cbits = qvm->allocateCBits(4);
            QProg prog;
            
            prog   << H(qubits[0])
                    << H(qubits[1])
                    << H(qubits[2])
                    << H(qubits[3])
                    << MeasureAll(qubits, cbits);

            auto result = quickMeasure(prog, 1000);
            for (auto &val: result)
            {
                std::cout << val.first << ", " << val.second << std::endl;
            }

            qvm->finalize();
            delete qvm;

            return 0;
        }

运行结果：

    .. code-block:: c

        0000, 47
        0001, 59
        0010, 74
        0011, 66
        0100, 48
        0101, 62
        0110, 71
        0111, 61
        1000, 70
        1001, 57
        1010, 68
        1011, 63
        1100, 65
        1101, 73
        1110, 55
        1111, 61


