优化算法(梯度下降法)
========================

本章节将讲解VQNet中优化算法的使用，包括经典梯度下降算法和改进后的梯度下降算法，它们都是在求解机器学习算法的模型参数，即无约束优化问题时，最常采用的方法之一。
我们在 ``pyQPanda`` 中实现了这些算法，``VanillaGradientDescentOptimizer`` 、 ``MomentumOptimizer`` 、 ``AdaGradOptimizer`` 、 ``RMSPropOptimizer`` 和 ``AdamOptimizer``，它们都继承自 ``Optimizer`` 。

接口介绍
----------

我们通过调用梯度下降优化器的 ``minimize`` 接口来生成一个优化器。常见的梯度下降优化器构造方式如下所示，

.. code-block:: python

    VanillaGradientDescentOptimizer.minimize(
        loss,  # 损失函数 
        0.01,  # 学习率
        1.e-6) # 结束条件

    MomentumOptimizer.minimize(
        loss,  # 损失函数 
        0.01,  # 学习率
        0.9)   # 动量系数

    AdaGradOptimizer.minimize(
        loss,  # 损失函数 
        0.01,  # 学习率
        0.0,   # 累加量起始值
        1.e-10)# 很小的数值以避免零分母

    RMSOptimizer.minimize(
        loss,  # 损失函数 
        0.01,  # 学习率
        0.9,   # 历史或即将到来的梯度的贴现因子
        1.e-10)# 很小的数值以避免零分母

    AdamOptimizer.minimize(
        loss,  # 损失函数 
        0.01,  # 学习率
        0.9,   # 一阶动量衰减系数
        0.999, # 二阶动量衰减系数
        1.e-10)# 很小的数值以避免零分母

实例
-------------

示例代码主要演示对离散点用直线进行拟合，我们定义训练数据X和Y，这两个变量表示离散点的坐标。定义两个可微分的变量w和b，其中w表示斜率b表示y轴截距。定义变量Y下划线表示斜率w乘上变量x加上截距。

接着我们定义损失函数loss。计算变量Y和变量Y下划线之间的均方值。

我们调用梯度下降优化器的 ``minimize`` 接口以损失函数，学习率和结束条件作为参数构造生成一个经典梯度下降优化器。

我们通过优化器的 ``get_variables`` 接口可以获得所有可微分的节点。

我们定义迭代次数为1000。然后调用优化器的 ``run`` 接口执行一次优化操作，其第二个参数表示当前的优化次数，目前只有 ``AdamOptimizer`` 这个优化器使用到了这个参数，其它优化器我们直接给0值即可。

我们可以通过优化器 ``get_loss`` 接口获得当前优化后的损失值。我们通过eval接口可以求得可微分变量的当前值。

.. code-block:: python

    from pyqpanda import *
    import numpy as np

    x = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59,
                 2.167, 7.042, 10.791, 5.313, 7.997, 5.654, 9.27,3.1])
    y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 
                 1.221, 2.827, 3.465, 1.65, 2.904, 2.42, 2.94,1.3])


    X = var(x.reshape(len(x), 1))
    Y = var(y.reshape(len(y), 1))

    W = var(0, True)
    B = var(0, True)

    Y_ = W*X + B

    loss = sum(poly(Y_ - Y, var(2))/len(x))

    optimizer = VanillaGradientDescentOptimizer.minimize(loss, 0.01, 1.e-6)
    leaves = optimizer.get_variables()

    for i in range(1000):
        optimizer.run(leaves, 0)
        loss_value = optimizer.get_loss()
        print("i: ", i, " loss: ", loss_value, " W: ", eval(W,True), " b: ", eval(B, True))
    

.. image:: images/GradientExample.png

我们将散列点和拟合的直线进行绘图

.. code-block:: python

    import matplotlib.pyplot as plt
    
    w2 = W.get_value()[0, 0]
    b2 = B.get_value()[0, 0]

    plt.plot(x, y, 'o', label = 'Training data')
    plt.plot(x, w2*x + b2, 'r', label = 'Fitted line')
    plt.legend()
    plt.show()

.. image:: images/GradientExamplePlot.png