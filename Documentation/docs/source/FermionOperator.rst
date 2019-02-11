费米子运算符(FermionOperator)
==================================

跟 ``PauliOperator`` 类似，``FermionOperator`` 类也提供了Fermion运算符之间加、减和乘的基础的运算操作。通过整理功能可以得到一份有序排列的运算符。

接口介绍
------------

.. cpp:class:: FermionOperator

   .. cpp:function:: FermionOperator()

        **功能**
            构造函数。创建一个空的Pauli运算符。
        **参数**
            无

   .. cpp:function:: PauliOperator(const complex_d &value)

        **功能**
            构造函数。通过传入 ``complex_d`` 类型数据来构造Fermion运算符I的系数。
        **参数**
            - value 系数值

   .. cpp:function:: FermionOperator(double value)

        **功能**  
            构造函数。通过传入double类型数据来构造Fermion运算符I的系数。
        **参数**
            - value 系数值

   .. cpp:function:: FermionOperator(const std::string &key, const complex_d &value)
      
        **功能**  
            构造函数。构造指定Fermion算符及其系数。
        **参数**
            - key 字符串类型的Fermion算符
            - value 系数值

   .. cpp:function:: FermionOperator(const FermionMap &map)
      
        **功能** 
            构造函数。通过传入 ``FermionMap`` 类型数据来构造Fermion运算符。
        **参数**
            - map Fermion算符及其系数的映射

   .. cpp:function:: FermionOperator normal_ordered()
      
        **功能**
            对该Fermion运算符进行整理。在这个转换中我们规定张量因子从高到低进行排序，并且创建（ :math:`a_x` ）出现在湮没（ :math:`a_x^\dagger` ）之前。
        **参数**
            无
        **返回值**
            Fermion运算符

   .. cpp:function:: void setAction(char create, char annihilation)

        **功能**  
            设置标识符，默认创建标识符为"+"，湮没标识符为""。
        **参数**
            - create 创建标识符
            - annihilation 湮没标识符
        **返回值**
            Pauli运算符

   .. cpp:function:: bool isEmpty()
      
        **功能**  
            测试该Pauli运算符是否为空。
        **参数**
            无
        **返回值**
            如果为空返回true，否则返回false。

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

   .. cpp:function:: FermionData data()

        **功能**  
            返回Fermion运算符的数据。
        **参数**
            无
        **返回值**
            Fermion运算符的数据。

.. note::
    湮没: x 表示 :math:`a_x` ;
    创建: x+ 表示 :math:`a_x^\dagger` 。
    例如 "1+ 3 5+ 1"则代表 :math:`a_1^\dagger \ a_3 \ a_5^\dagger \ a_1`

实例
--------------

.. code-block:: cpp

    #include "Operator/FermionOperator.h"

    int main()
    {
        QPanda::FermionOperator a("0 1+", 2);
        QPanda::FermionOperator b("2+ 3", 3);

        auto plus = a + b;
        auto minus = a - b;
        auto muliply = a * b;

        std::cout << "a + b = " << plus << std::endl << std::endl;
        std::cout << "a - b = " << minus << std::endl << std::endl;
        std::cout << "a * b = " << muliply << std::endl << std::endl;

        std::cout << "normal_ordered(a + b) = " << plus.normal_ordered() << std::endl << std::endl;
        std::cout << "normal_ordered(a - b) = " << minus.normal_ordered() << std::endl << std::endl;
        std::cout << "normal_ordered(a * b) = " << muliply.normal_ordered() << std::endl << std::endl;

        return 0;
    }

.. image:: images/FermionOperatorTest.png   