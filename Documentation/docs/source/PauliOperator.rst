泡利运算符(PauliOperator)
============================

量子运算符可以表示为Pauli运算符I，X，Y，Z的组合。``PauliOperator`` 类除了提供Pauli运算符之间加、减和乘的基础的运算操作，
还提供了诸如求共轭、打印和判空等常用操作。

接口介绍
-------------

在QPanda中我们实现了该组件，使用该组件必须包含QPanda命名空间 ``QPanda::PauliOperator``，详见 ``PauliOperator.h`` 。 

.. cpp:class:: PauliOperator

   .. cpp:function:: PauliOperator()

        **功能**
            构造函数。创建一个空的Pauli运算符。
        **参数**
            无

   .. cpp:function:: PauliOperator(const complex_d &value)

        **功能**
            构造函数。通过传入 ``complex_d`` 类型数据来构造Fermion运算符I的系数。
        **参数**
            - value 系数值

   .. cpp:function:: PauliOperator(double value)
      
        **功能**
            构造函数。通过传入double类型数据来构造pauli运算符I的系数。
        **参数**
            - value 系数值

   .. cpp:function:: PauliOperator(const std::string &key, const complex_d &value)
      
        **功能**
            构造函数。构造指定Pauli算符及其系数。
        **参数**
            - key 字符串类型的Pauli算符
            - value 系数值

   .. cpp:function:: PauliOperator(const PauliMap &map)
      
        **功能**
            构造函数。通过传入 ``PauliMap`` 类型数据来构造pauli运算符。
        **参数**
            - map 一组Pauli算符及其系数

   .. cpp:function:: PauliOperator dagger()
      
        **功能**
            求该Pauli运算符的共轭Pauli运算符。
        **参数**
            无
        **返回值**
            Pauli运算符

   .. cpp:function:: PauliOperator remapQubitIndex(std::map<size_t, size_t> &index_map)
      
        **功能**
            对该Pauli运算符中索引从0开始分配映射。例如该运算符只含有"Z1 Z20"，重新映射得到"Z0 Z1"。 
        **参数**
            - index_map 用于存放新的映射关系
        **返回值**
            Pauli运算符

   .. cpp:function:: size_t getMaxIndex()
      
        **功能**
            获得该Pauli运算符使用的索引值数。如果为空则返回0，否则返回最大下标索引值+1的结果。例如"Z0 Z5",返回的数值是6。
        **参数**
            无
        **返回值**
            Pauli运算符使用的索引值。

   .. cpp:function:: bool isEmpty()
      
        **功能**
            测试该Pauli运算符是否为空。
        **参数**
            无
        **返回值**
            如果为空返回true，否则返回false。

   .. cpp:function:: bool isAllPauliZorI()
      
        **功能**
            测试该Pauli运算符是否全为"Z"或"I"。
        **参数**
            无
        **返回值**
            如果是返回true，否则返回false。

   .. cpp:function:: void setErrorThreshold(double threshold)
      
        **功能**
            设置误差阈值，如果小于该误差阈值则认为其系数为0，不参与打印输出及计算。
        **参数**
            - threshold 阈值
        **返回值**
            无

   .. cpp:function:: std::string toString()
      
        **功能**
            返回Pauli运算符的string表达式。
        **参数**
            无
        **返回值**
            Pauli运算符的string表达式。

   .. cpp:function:: QHamiltonian toHamiltonian(bool *ok = nullptr)
      
        **功能**
            返回Pauli运算符的哈密顿量表达式。
        **参数**
            - ok 如果该Pauli运算符是哈密顿量这设置为true，否则为false
        **返回值**
            哈密顿量表达式。

   .. cpp:function:: PauliData data()
      
        **功能**
            返回Pauli运算符的数据。
        **参数**
            无
        **返回值**
            Pauli运算符的数据。

实例
-------------

.. code-block:: cpp
    
    #include "Operator/PauliOperator.h"

    int main()
    {
        QPanda::PauliOperator a("Z0 Z1", 2);
        QPanda::PauliOperator b("X5 Y6", 3);

        auto plus = a + b;
        auto minus = a - b;
        auto muliply = a * b;

        std::cout << "a + b = " << plus << std::endl << std::endl;
        std::cout << "a - b = " << minus << std::endl << std::endl;
        std::cout << "a * b = " << muliply << std::endl << std::endl;

        std::cout << "Index : " << muliply.getMaxIndex() << std::endl << std::endl;

        std::map<size_t, size_t> index_map;
        auto remap_pauli = muliply.remapQubitIndex(index_map);

        std::cout << "remap_pauli : " << remap_pauli << std::endl << std::endl;
        std::cout << "Index : " << remap_pauli.getMaxIndex() << std::endl;

        return 0;
    }

.. image:: images/PauliOperatorTest.png