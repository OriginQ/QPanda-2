可变量子逻辑门(VQG)
======================
要在VQNet中使用量子操作 ``qop`` 或 ``qop_pmeasure`` ,就必须要包含可变量子线路(``VQC``)，而可变量子逻辑门则是构成 ``VQC`` 的基本单位。 可变量子逻辑门(``VariationalQuantumGate``，别名: ``VQG``)，内部维护着一组变量参数以及一组常量参数。
在构造 ``VQG`` 的时候只能对其中一组参数进行赋值。若含有一组常量参数，则可以通过 ``VQG`` 生成含确定参数的普通量子逻辑门, 若含有变量参数，则可以动态修改参数值，并生成对应的参数的普通量子逻辑门。

目前在QPanda::Variational中定义了如下可变量子逻辑门，它们都继承自 ``VQG`` 。

.. list-table::

    * - VQG                                        
      - 别名                              
    * - VariationalQuantumGate_H
      - VQG_H
    * - VariationalQuantumGate_RX
      - VQG_RX 
    * - VariationalQuantumGate_RY
      - VQG_RY  
    * - VariationalQuantumGate_RZ
      - VQG_RZ  
	* - VariationalQuantumGate_CNOT
      - VQG_CNOT  
	* - VariationalQuantumGate_CZ
      - VQG_CZ  

接口介绍
-----------------

.. cpp:class:: VariationalQuantumGate

   .. cpp:function:: VariationalQuantumGate()

        **功能**
            构造函数。
        **参数**
            无

   .. cpp:function:: size_t n_var()
      
        **功能**
            该可变量子逻辑门内部变量个数。
        **参数**
            无
        **返回值**
            变量个数。

   .. cpp:function:: std::vector<var>& get_vars()

        **功能**      
            获取该可变量子逻辑门内部变量。
        **参数**
            无
        **返回值**
            该可变量子逻辑门内部变量。

   .. cpp:function:: std::vector<double>& get_constants()
      
        **功能**
            获取该可变量子逻辑门内部常量。
        **参数**
            无
        **返回值**
            该可变量子逻辑门内部常量。

   .. cpp:function:: int var_pos(var _var)

        **功能**      
            获取变量在该可变量子逻辑门内部索引。
        **参数**
            - _var 变量
        **返回值**
            内部索引，如果不存在返回-1。

   .. cpp:function:: virtual QGate feed() const = 0
      
        **功能**
            实例化 ``QGate`` 。
        **参数**
            无
        **返回值**
            普通量子逻辑门。

   .. cpp:function:: virtual QGate feed(std::map<size_t, double> offset) const

        **功能**      
            通过指定偏移来实例化 ``QGate`` 。
        **参数**
            - offset 变量对应的偏移映射
        **返回值**
            普通量子逻辑门。

   .. virtual std::shared_ptr<VariationalQuantumGate> copy() = 0
      
        **功能**
            获取当前可变逻辑门的一份拷贝。
        **参数**
            无
        **返回值**
            当前可变逻辑门的一份拷贝。

下面将简要介绍各个可变量子逻辑门的构造方式

.. cpp:class:: VariationalQuantumGate_H

   .. cpp:function:: VariationalQuantumGate_H(Qubit* q)

        **功能**
            H门构造函数。
        **参数**
            - q 目标比特 

.. cpp:class:: VariationalQuantumGate_RX

   .. cpp:function:: VariationalQuantumGate_RX(Qubit* q, var _var)

        **功能**
            RX门构造函数。
        **参数**
            - q 目标比特 
            - _var 参数变量

   .. cpp:function:: VariationalQuantumGate_RX(Qubit* q, double angle)

        **功能**
            RX门构造函数。
        **参数**
            - q 目标比特 
            - angle 参数

.. cpp:class:: VariationalQuantumGate_RY

   .. cpp:function:: VariationalQuantumGate_RY(Qubit* q, var _var)

        **功能**
            RY门构造函数。
        **参数**
            - q 目标比特 
            - _var 参数变量

   .. cpp:function:: VariationalQuantumGate_RY(Qubit* q, double angle)

        **功能**
            RY门构造函数。
        **参数**
            - q 目标比特 
            - angle 参数

.. cpp:class:: VariationalQuantumGate_RZ

   .. cpp:function:: VariationalQuantumGate_RZ(Qubit* q, var _var)

        **功能**
            RZ门构造函数。
        **参数**
            - q 目标比特 
            - _var 参数变量

   .. cpp:function:: VariationalQuantumGate_RZ(Qubit* q, double angle)

        **功能**
            RZ门构造函数。
        **参数**
            - q 目标比特 
            - angle 参数

.. cpp:class:: VariationalQuantumGate_CZ

   .. cpp:function:: VariationalQuantumGate_CZ(Qubit* q1, Qubit* q2)

        **功能**
            CZ门构造函数。
        **参数**
            - q1 控制比特 
            - q2 目标比特

.. cpp:class:: VariationalQuantumGate_CNOT

   .. cpp:function:: VariationalQuantumGate_CNOT(Qubit* q1, Qubit* q2)

        **功能**
            CNOT门构造函数。
        **参数**
            - q1 控制比特 
            - q2 目标比特

实例
----------

.. code-block:: cpp

    #include "QPanda.h"
    #include "Variational/var.h"

    int main()
    {
        using namespace QPanda;
        using namespace QPanda::Variational;

        constexpr int qnum = 2;

        QuantumMachine *machine = initQuantumMachine(QuantumMachine_type::CPU_SINGLE_THREAD);
        std::vector<Qubit*> q;
        for (int i = 0; i < qnum; ++i)
        {
            q.push_back(machine->Allocate_Qubit());
        }

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
        vqc.insert(VQG_RZ(q[0], 0.123));
        vqc.insert(VQG_CZ(q[0], q[1]));
        vqc.insert(VQG_CNOT(q[0], q[1]));

        QCircuit circuit = vqc.feed();
        QProg prog;
        prog << circuit;

        std::cout << qProgToQRunes(prog) << std::endl << std::endl;

        m1(0, 0) = 3;
        m2(0, 0) = 4;

        x.setValue(m1);
        y.setValue(m2);

        QCircuit circuit2 = vqc.feed();
        QProg prog2;
        prog2 << circuit2;

        std::cout << qProgToQRunes(prog2) << std::endl;

        return 0;
    }

.. image:: images/VQG_Example.png