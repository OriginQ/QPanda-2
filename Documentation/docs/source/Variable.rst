变量
=========

变量是VQNet中的一种数据类型，用于存储特定混合量子经典网络的变量。 通常任务是优化变量以最小化成本函数。 变量可以是标量，矢量或矩阵。

变量具有树形结构，它可以包含子节点或父节点。 如果变量没有子节点，那么我们称之为叶子节点。 我们可以将变量设置为特定值，另一方面，如果变量的所有叶子节点都已设置了值，我们也可以获得该变量的值。


接口介绍
--------------

.. cpp:class:: var

   .. cpp:function:: var(std::shared_ptr<impl> data)

        **功能**
            构造函数。通过数据的指针进行构造。
        **参数**
            - data 数据指针

   .. cpp:function:: var(double data)

        **功能**
            构造函数。通过浮点型的数据构造一个标量变量。
        **参数**
            - data 标量数据

   .. cpp:function:: var(const MatrixXd& data)

        **功能**
            构造函数。通过 ``Eigen`` 库中的 ``MatrixXd`` 类型数据的构造一个矢量或矩阵变量。
        **参数**
            - data 矢量或矩阵数据

   .. cpp:function:: var(double data, bool isDifferentiable)

        **功能**
            构造函数。构造一个是否可微分的变量，如果isDifferentiable为false，则该变量作用类似placeholder。
        **参数**
            - data 标量数据
            - isDifferentiable 是否可以微分

   .. cpp:function:: var(const MatrixXd& data, bool isDifferentiable)

        **功能**
            构造函数。构造一个是否可微分的变量，如果isDifferentiable为false，则该变量作用类似placeholder。
        **参数**
            - data 矢量或矩阵数据
            - isDifferentiable 是否可以微分

   .. cpp:function:: var(op_type op, const std::vector<var>& children)

        **功能**
            构造函数。通过操作及其孩子变量进行构造。例如：c = a + b，c是a和b的父变量，a和b是c的孩子变量。
        **参数**
            - op 操作符类型
            - childen 操作符的孩子变量

   .. cpp:function:: var clone()
      
        **功能**
            克隆。
        **参数**
            无
        **返回值**
            当前变量的副本。

   .. cpp:function:: virtual size_t getNumOpArgs()
      
        **功能**
            获取当前操作的参数个数。
        **参数**
            无
        **返回值**
            操作的参数个数。

   .. cpp:function:: MatrixXd getValue() const
      
        **功能**
            获取变量的值。
        **参数**
            无
        **返回值**
            变量的值。

   .. cpp:function:: void setValue(const MatrixXd& data)
      
        **功能**
            设置变量的值。
        **参数**
            - data 矩阵类型的数据
        **返回值**
            无

   .. cpp:function:: op_type getOp() const
      
        **功能**
            获取变量对应的操作类型。
        **参数**
            无
        **返回值**
            操作类型。

   .. cpp:function:: void setOp(op_type op)

        **功能**     
            设置变量的操作类型。
        **参数**
            - op 操作类型
        **返回值**
            无

   .. cpp:function:: std::vector<var>& getChildren() const
      
        **功能**
            获取当前变量的孩子变量。
        **参数**
            无
        **返回值**
            当前变量的孩子变量。

   .. cpp:function:: std::vector<var> getParents() const

        **功能**      
            获取当前变量的父亲变量。
        **参数**
            无
        **返回值**
            当前变量的父亲变量。

   .. cpp:function:: long getUseCount() const
      
        **功能**
            获取变量被引用的次数。
        **参数**
            无
        **返回值**
            引用次数。

   .. cpp:function:: bool getValueType() const

        **功能**      
            获取变量被引用的次数。
        **参数**
            无
        **返回值**
            引用次数。
       
   .. cpp:function:: MatrixXd _eval()
      
        **功能**
            根据孩子变量的数值以及当前的操作计算当前变量的值。
        **参数**
            无
        **返回值**
            ``MatrixXd`` 类型的值，如果是标量，返回的是1x1的矩阵。

   .. cpp:function:: MatrixXd _back_single(const MatrixXd& dx, size_t op_idx)

        **功能**  
            求当前变量对索引值为op_idx孩子节点的偏导值。
        **参数**
            - dx 链式法则中上一层函数（外层函数）的偏导值
            - op_idx 孩子结点的索引
        **返回值**
            当前变量对索引值为op_idx孩子节点的偏导值。

   .. cpp:function:: std::vector<MatrixXd> _back(const MatrixXd& dx, const std::unordered_set<var>& nonconsts)
      
        **功能**
            求当前变量对非常量孩子节点的偏导值。
        **参数**
            - dx 链式法则中上一层函数（外层函数）的偏导值
            - nonconsts 非常量孩子节点
        **返回值**
            当前变量非常量孩子节点的偏导值。

   .. cpp:function:: std::vector<MatrixXd> _back(const MatrixXd& dx)
      
        **功能**
            求当前变量对所有孩子节点的偏导值。
        **参数**
            - dx 链式法则中上一层函数（外层函数）的偏导值
        **返回值**
            当前变量对所有孩子节点的偏导值。

   .. cpp:function:: const var operator[](int subscript)
      
        **功能**
            创建一个下标操作的变量。
        **参数**
            - subscript 下标
        **返回值**
            新的变量。

实例
---------------

.. code-block:: cpp

    #include "Variational/var.h"

    int main()
    {
        using namespace QPanda::Variational;

        var const_var(1);

        MatrixXd m1(2, 2);
        m1 << 1, 2, 3, 4;

        MatrixXd m2(2, 2);
        m2 << 5, 6, 7, 8;

        var var1(m1);
        var var2(m2);

        var sum = var1 + var2;
        var minus(op_type::minus, {var2, var1});
        var multiply = var1 * var2;

        MatrixXd dx = MatrixXd::Ones(2, 2);

        std::cout << "const_var: " << std::endl << const_var.getValue() << std::endl;
        std::cout << "var1: " << std::endl << var1.getValue() << std::endl;
        std::cout << "var2: " << std::endl << var2.getValue() << std::endl;
        std::cout << "sum: "  << std::endl << sum._eval() << std::endl;
        std::cout << "    op_type: " << int(sum.getOp()) << std::endl;
        std::cout << "    NumOpArgs: " << int(sum.getNumOpArgs()) << std::endl;
        std::cout << "minus: "  << std::endl << minus._eval() << std::endl;
        std::cout << "    op_type: " << int(minus.getOp()) << std::endl;
        std::cout << "    NumOpArgs: " << int(minus.getNumOpArgs()) << std::endl;
        std::cout << "multiply: "  << std::endl << multiply._eval() << std::endl;
        std::cout << "    op_type: " << int(multiply.getOp()) << std::endl;
        std::cout << "    NumOpArgs: " << int(multiply.getNumOpArgs()) << std::endl;
        std::cout << "Derivative multipy to var1:" <<std::endl<< multiply._back_single(dx, 0)<<std::endl;
        std::cout << "Derivative multipy to var2:" <<std::endl<< multiply._back_single(dx, 1)<<std::endl;


        MatrixXd m3(2, 2);
        m3 << 4, 3, 2, 1;
        var1.setValue(m3);

        std::cout << "sum: "  << std::endl << sum._eval() << std::endl;
        std::cout << "minus: "  << std::endl << minus._eval() << std::endl;
        std::cout << "multiply: "  << std::endl << multiply._eval() << std::endl;
        std::cout << "matrix_var1 UseCount: " << var1.getUseCount() << std::endl;

        return 0;
    }

.. image:: images/VarExample.png