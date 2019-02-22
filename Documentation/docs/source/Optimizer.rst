优化算法(直接搜索法)
=======================

本章节将讲解优化算法的使用，包括 ``Nelder-Mead`` 算法跟 ``Powell`` 算法，它们都是一种直接搜索算法。我们在 ``QPanda`` 中实现了这两个算法，``OriginNelderMead`` 和 ``OriginPowell`` ，
这两个类都继承自 ``AbstractOptimizer`` 。

接口介绍
--------------

.. cpp:class:: AbstractOptimizer

   .. cpp:function:: AbstractOptimizer()

        **功能**
            构造函数。
        **参数**
            无

   .. cpp:function:: void registerFunc(const QFunc &func, const vector_d &optimized_para)
      
        **功能**
            注册优化函数（针对特定问题求解期望的函数）。
        **参数**
            - func 优化函数
            - optimized_para 初始优化参数
        **返回值**
            无

   .. cpp:function:: void setDisp(bool disp)
      
        **功能**
            是否展示当前迭代最优结果。
        **参数**
            - disp 如果配置为true则展示，否则不展示
        **返回值**
            无

   .. cpp:function:: void setAdaptive(bool adaptive)
      
        **功能**
            设置Nelder-Mead算法适应参数，使算法参数适应维度。
        **参数**
            - adaptive 如果配置为true则启用，否则不启用
        **返回值**
            无

   .. cpp:function:: void setXatol(double xatol)
      
        **功能**
            设置迭代之间优化参数(xopt)的绝对误差阈值，主要用于判断是否收敛。
        **参数**
            - xatol 误差阈值
        **返回值**
            无

   .. cpp:function:: void setFatol(double fatol)
      
        **功能**
            设置迭代之间func(xopt)的绝对误差阈值，主要用于判断是否收敛。
        **参数**
            - xatol 误差阈值
        **返回值**
            无

   .. cpp:function:: void setMaxFCalls(size_t max_fcalls)
      
        **功能**
                设置函数func(xopt)最大调用次数，当超过该阈值将停止迭代。
        **参数**
            - max_fcalls 最大调用次数
        **返回值**
            无

   .. cpp:function:: void setMaxIter(size_t max_iter)
      
        **功能**
            设置最大迭代次数，当超过该阈值将停止迭代。
        **参数**
            - max_iter 最大迭代次数
        **返回值**
            无

   .. cpp:function:: virtual void exec() = 0
      
        **功能**
            执行优化算法。
        **参数**
            无
        **返回值**
            无

   .. cpp:function:: virtual QOptimizationResult getResult()

        **功能**      
            获取优化结果。
        **参数**
            无
        **返回值**
            优化结果。

我们可以通过 ``OptimizerFactory`` 来生成指定的优化器。

.. cpp:class:: OptimizerFactory

   .. cpp:function:: static std::unique_ptr<AbstractOptimizer> makeOptimizer(const OptimizerType &optimizer)
      
        **功能**
            通过指定类型来生成优化器。
        **参数**
            - optimizer 优化器类型
        **返回值**
            优化器。

   .. cpp:function:: static std::unique_ptr<AbstractOptimizer> makeOptimizer(const std::string &optimizer)
      
        **功能**
            通过指定类型来生成优化器。
        **参数**
            - optimizer 优化器类型
        **返回值**
            优化器。

实例
--------------

给定一些散列点，我们来拟合一条直线，使得散列点到直线的距离和最小。定义直线的函数的表达式为 ``y = w*x + b`` ，接下来我们将通过使用优化算法得到w和b的优化值。 首先定义求期望的函数

.. code-block:: cpp

    QPanda::QResultPair myFunc(QPanda::vector_d para)
    {
        std::vector<double> x = {3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                    2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1};

        std::vector<double> y = {1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53,
                    1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3};

        std::vector<double> y_;

        for (auto i = 0u; i < x.size(); i++)
        {
            y_.push_back(para[0]*x[i] + para[1]);
        }

        float loss = 0;
        for (auto i = 0u; i < y.size(); i++)
        {
            loss += std::pow(y[i] - y_[i], 2)/y.size();
        }

        return std::make_pair("", loss);
    }

我们使用 ``Nelder-Mead`` 算法进行优化

.. code-block:: cpp

    #include "Optimizer/AbstractOptimizer.h"
    #include "Optimizer/OptimizerFactory.h"
    #include <iostream>

    int main()
    {
        auto optimizer = QPanda::OptimizerFactory::makeOptimizer(QPanda::OptimizerType::NELDER_MEAD);

        QPanda::vector_d init_para{0, 0};
        optimizer->registerFunc(myFunc, init_para);
        optimizer->setXatol(1e-6);
        optimizer->setFatol(1e-6);
        optimizer->setMaxFCalls(200);
        optimizer->setMaxIter(200);
        optimizer->exec();

        auto result = optimizer->getResult();

        std::cout << result.message << std::endl;
        std::cout << "         Current function value: "
            << result.fun_val << std::endl;
        std::cout << "         Iterations: "
            << result.iters << std::endl;
        std::cout << "         Function evaluations: "
            << result.fcalls << std::endl;

        std::cout << "         Optimized para: " << std::endl;
        for (auto i = 0u; i < result.para.size(); i++)
        {
            std::cout << "             " << result.para[i] << std::endl;
        }
    }

.. image:: images/OptimizerTest.png

我们将散列点和拟合的直线进行绘图

.. image:: images/OptimizerPlot.png
