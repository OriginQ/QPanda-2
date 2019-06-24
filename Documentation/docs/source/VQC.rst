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

.. code-block:: cpp

    MatrixXd m1(1, 1);
    MatrixXd m2(1, 1);
    m1(0, 0) = 1;
    m2(0, 0) = 2;

    var x(m1);
    var y(m2);

    VQC vqc;
    vqc.insert(VQG_H(q[0]));
    vqc.insert(VQG_RX(q[0], x));
    vqc.insert(VQG_RY(q[1], y));

    QCircuit circuit = vqc.feed();
    QProg prog;
    prog << circuit;

我们可以调用 ``get_var_in_which_gate`` 接口来获取到指定变量在可变量子线路中对应的可变量子逻辑门。
我们可以通过向feed接口传入变量对应的可变量子逻辑门，变量在可变量子逻辑门中的索引位置，以及偏移值，来改变指定可变量子逻辑门中变量参数的偏移值。

.. code-block:: cpp

    auto gates = vqc.get_var_in_which_gate(x);

    int pos = shared_ptr<VariationalQuantumGate>(gates[0])->var_pos(x);

    vector<tuple<weak_ptr<VariationalQuantumGate>, size_t, double>> plus;
    plus.push_back(make_tuple(gates[0], pos, 3));

    QCircuit circuit2 = vqc.feed(plus);

实例
-------------

.. code-block:: cpp

    #include "QPanda.h"
    #include "Variational/var.h"

    int main()
    {
        using namespace std;
        using namespace QPanda;
        using namespace QPanda::Variational;

        constexpr int qnum = 2;

        QuantumMachine *machine = initQuantumMachine(CPU_SINGLE_THREAD);
        auto q = machine->allocateQubits(qnum);

        MatrixXd m1(1, 1);
        MatrixXd m2(1, 1);
        m1(0, 0) = 1;
        m2(0, 0) = 2;

        var x(m1);
        var y(m2);

        VQC vqc;
        vqc.insert(VQG_H(q[0]));
        vqc.insert(VQG_RX(q[0], x));
        vqc.insert(VQG_RY(q[1], y));

        QCircuit circuit = vqc.feed();
        QProg prog;
        prog << circuit;

        std::cout << transformQProgToQRunes(prog, machine) << std::endl << std::endl;

        auto gates = vqc.get_var_in_which_gate(x);

        int pos = shared_ptr<VariationalQuantumGate>(gates[0])->var_pos(x);

        vector<tuple<weak_ptr<VariationalQuantumGate>, size_t, double>> plus;
        plus.push_back(make_tuple(gates[0], pos, 3));

        QCircuit circuit2 = vqc.feed(plus);
        QProg prog2;
        prog2 << circuit2;

        std::cout << transformQProgToQRunes(prog2,machine) << std::endl;

        return 0;
    }

.. image:: images/VQC_Example.png
