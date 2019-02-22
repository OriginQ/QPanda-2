优化算法(梯度下降法)
========================

本章节将讲解VQNet中优化算法的使用，包括经典梯度下降算法和改进后的梯度下降算法，它们都是在求解机器学习算法的模型参数，即无约束优化问题时，最常采用的方法之一。我们在 ``QPanda::Variational`` 中实现了这些算法，``VanillaGradientDescentOptimizer`` 、 ``MomentumOptimizer`` 、 ``AdaGradOptimizer`` 、 ``RMSPropOptimizer`` 和 ``AdamOptimizer``，它们都继承自 ``Optimizer`` 。

接口介绍
-------------

.. cpp:class:: Optimizer

   .. cpp:function:: Optimizer(var lost_function, double learning_rate = 0.01)

        **功能**
            构造函数。
        **参数**
            - lost_function 损失函数表达式
            - learning_rate 学习率

   .. cpp:function:: virtual std::unordered_set<var> get_variables() = 0
      
        **功能**  
            获取损失函数内部变量。
        **参数**
            无
        **返回值**
            损失函数内部变量。

   .. cpp:function:: std::unordered_map<var, MatrixXd> compute_gradients(std::unordered_set<var> &var_set) = 0
      
        **功能** 
            计算指定变量的梯度值。
        **参数**
            - var_set 变量组
        **返回值**
            变量的对应的梯度值。

   .. cpp:function:: double get_loss() = 0

        **功能**  
            计算损失函数值。
        **参数**
            无
        **返回值**
            损失函数值。

   .. cpp:function:: bool run(std::unordered_set<var> &leaves, size_t t = 0) = 0
      
        **功能**
            执行一次优化。
        **参数**
            - leaves 待优化的参数节点
            - t 当前优化的次数
        **返回值**
            是否运行成功

.. cpp:class:: VanillaGradientDescentOptimizer

   .. cpp:function:: static std::shared_ptr<Optimizer> minimize(var lost_function ,double learning_rate, double stop_condition)
      
        **功能**
            通过传入指定参数构造优化器。
        **参数**
            - lost_function 损失函数表达式
            - learning_rate 学习率
            - stop_condition 结束条件[暂未使用]
        **返回值**
            优化器。

.. cpp:class:: MomentumOptimizer

   .. cpp:function:: static std::shared_ptr<Optimizer> minimize(var &lost, double learning_rate = 0.01, double momentum = 0.9)
      
        **功能**  
            通过传入指定参数构造优化器。
        **参数**
            - lost 损失函数表达式
            - learning_rate 学习率
            - momentum 动量系数
        **返回值**
            优化器。

.. cpp:class:: AdaGradOptimizer

   .. cpp:function:: static std::shared_ptr<Optimizer> minimize(var &lost, double learning_rate = 0.01, double initial_accumulator_value = 0.0, double epsilon = 0.0000000001)
      
        **功能**
            通过传入指定参数构造优化器。
        **参数**
            - lost 损失函数表达式
            - learning_rate 学习率
            - initial_accumulator_value 累加量的起始值
            - epsilon 很小的数值以避免零分母      
        **返回值**
            优化器。

.. cpp:class:: RMSPropOptimizer

   .. cpp:function:: static std::shared_ptr<Optimizer> minimize(var &lost, double learning_rate = 0.001, double decay = 0.9, double epsilon = 0.0000000001)
      
        **功能**  
            通过传入指定参数构造优化器。
        **参数**
            - lost 损失函数表达式
            - learning_rate 学习率
            - decay 历史或即将到来的梯度的贴现因子。
            - epsilon 很小的数值以避免零分母       
        **返回值**
            优化器。

.. cpp:class:: RMSPropOptimizer

   .. cpp:function:: static std::shared_ptr<Optimizer> minimize(var &lost, double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 0.0000000001)
      
        **功能** 
            通过传入指定参数构造优化器。
        **参数**
            - lost 损失函数表达式
            - learning_rate 学习率
            - beta1 衰减率。
            - beta2 衰减率。
            - epsilon 很小的数值以避免零分母      
        **返回值**
            优化器。 

实例
-------------

给定一些散列点，我们来拟合一条直线，使得散列点到直线的距离和最小。定义直线的函数的表达式为 ``y = w*x + b`` ，接下来我们将通过使用优化算法得到w和b的优化值。

.. code-block:: cpp

    #include "Variational/Optimizer.h"

    int main()
    {
        using namespace QPanda::Variational;

        MatrixXd train_x(17, 1);
        MatrixXd train_y(17, 1);

        train_x(0, 0) = 3.3;
        train_x(1, 0) = 4.4;
        train_x(2, 0) = 5.5;
        train_x(3, 0) = 6.71;
        train_x(4, 0) = 6.93;
        train_x(5, 0) = 4.168;
        train_x(6, 0) = 9.779;
        train_x(7, 0) = 6.182;
        train_x(8, 0) = 7.59;
        train_x(9, 0) = 2.167;
        train_x(10, 0) = 7.042;
        train_x(11, 0) = 10.791;
        train_x(12, 0) = 5.313;
        train_x(13, 0) = 7.997;
        train_x(14, 0) = 5.654;
        train_x(15, 0) = 9.27;
        train_x(16, 0) = 3.1;
        train_y(0, 0) = 1.7;
        train_y(1, 0) = 2.76;
        train_y(2, 0) = 2.09;
        train_y(3, 0) = 3.19;
        train_y(4, 0) = 1.694;
        train_y(5, 0) = 1.573;
        train_y(6, 0) = 3.366;
        train_y(7, 0) = 2.596;
        train_y(8, 0) = 2.53;
        train_y(9, 0) = 1.221;
        train_y(10, 0) = 2.827;
        train_y(11, 0) = 3.465;
        train_y(12, 0) = 1.65;
        train_y(13, 0) = 2.904;
        train_y(14, 0) = 2.42;
        train_y(15, 0) = 2.94;
        train_y(16, 0) = 1.3;

        var X(train_x);
        var Y(train_y);

        var W(1.0, true);
        var b(1.0, true);

        var Y_ = W * X + b;
        auto loss = sum(poly(Y - Y_, 2) / train_x.rows());
        auto optimizer = VanillaGradientDescentOptimizer::minimize(loss, 0.01, 1.e-6);
        //auto optimizer = MomentumOptimizer::minimize(loss, 0.01, 1.e-6);
        //auto optimizer = AdaGradOptimizer::minimize(loss, 0.01, 1.e-6);
        //auto optimizer = RMSPropOptimizer::minimize(loss, 0.01, 1.e-6);
        //auto optimizer = AdamOptimizer::minimize(loss, 0.01, 1.e-6);

        auto leaves = optimizer->get_variables();
        for (size_t i = 0u; i < 1000; i++)
        {
            optimizer->run(leaves);
            std::cout << "i: " << i << "\t" << optimizer->get_loss()
                << "\t W:" << QPanda::Variational::eval(W, true)
                << "\t b:" << QPanda::Variational::eval(b, true)
                << std::endl;
        }

        return 0;
    }

.. image:: images/GradientExample.png

我们将散列点和拟合的直线进行绘图

.. image:: images/GradientExamplePlot.png