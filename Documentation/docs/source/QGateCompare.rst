.. _QGateCompare:

不支持的量子逻辑门统计
======================

简介
--------------

统计量子程序、量子线路、量子循环控制、量子条件控制中不支持的量子逻辑门个数，也可以判断一个量子逻辑门是否支持。

接口介绍
--------------

``QGateCompare`` 类是统计一个量子程序(量子线路、量子循环控制、量子条件控制)中量子逻辑门个数的工具类，我们先用QPanda2构建一个量子程序：

    .. code-block:: c
          
        auto qubits = qAllocMany(4);
        auto cbits = cAllocMany(4);

        QProg prog;
        prog << X(qubits[0])
                << Y(qubits[1])
                << H(qubits[0])
                << RX(qubits[0], 3.14)
                << iSWAP(qubits[2], qubits[3]);
                << Measure(qubits[1], cbits[0]);

然后调用 ``QGateCompare`` 类统计不支持量子逻辑门的个数

    .. code-block:: c
          

        std::vector<std::string> single_gates = {"H"}; // 支持的单量子逻辑门类型
        std::vector<std::string> double_gates = {"CNOT"}; // 支持的双量子逻辑门类型
        std::vector<std::vector<std::string>> gates = {single_gates, double_gates};
        QGateCompare t(gates);
        t.traversal(prog);
        size_t num = t.count();

我们还可以使用QPanda2封装的一个接口：

    .. code-block:: c
          
        size_t num = size_t num = getUnSupportQGateNumber(prog, gates);

.. note:: 统计 ``QCircuit`` 、 ``QWhileProg`` 、``QIfProg`` 、 ``QGate`` 中不支持的量子逻辑门的个数和 ``QProg`` 类似。

实例
-------------

    .. code-block:: c
    
        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            init();
            auto qubits = qAllocMany(4);
            auto cbits = cAllocMany(4);

            QProg prog;
            prog << X(qubits[0])
                    << Y(qubits[1])
                    << H(qubits[0])
                    << RX(qubits[0], 3.14)
                    << iSWAP(qubits[2], qubits[3])
                    << Measure(qubits[1], cbits[0]);

            std::vector<std::string> single_gates = {"H"}; // 支持的单量子逻辑门类型
            std::vector<std::string> double_gates = {"CNOT"}; // 支持的双量子逻辑门类型
            std::vector<std::vector<std::string>> gates = {single_gates, double_gates};

            size_t num = getUnSupportQGateNumber(prog, gates);
            std::cout << "unsupport QGate num: " << num << std::endl;
            finalize();

            return 0;
        }

运行结果：

    .. code-block:: c

        unsupport QGate num: 4

    
