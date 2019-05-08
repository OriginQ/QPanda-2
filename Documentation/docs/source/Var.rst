变量
=========

变量类是实现符号计算的用户类，用于存储特定混合量子经典网络的变量。 通常任务是优化变量以最小化成本函数。 变量可以是标量，矢量或矩阵。

变量具有树形结构，它可以包含子节点或父节点。 如果变量没有子节点，那么我们称之为叶子节点。 我们可以将变量设置为特定值，另一方面，如果变量的所有叶子节点都已设置了值，我们也可以获得该变量的值。


接口介绍
--------------

我们可以通过传入一个浮点型的数据来构造一个标量变量，也可以通过传入numpy库生成的多维数组来构造一个矢量或矩阵变量。

.. code-block:: python

    v1 = var(1)

    a = np.array([[1.],[2.],[3.],[4.]])
    v2 = var(a)
    b = np.array([[1.,2.],[3.,4.]])
    v3 = var(b)

.. note:: 

    在定义变量的时候，可以定义变量的类型是否可以微分，默认情况下我们定义的变量的类型都是不可微分的，不可微分的变量相当于 ``placeholder``。
    定义可微分的变量时，需要指定构造函数的第二个参数为True, 例如：v1 = var(1, True)。

我们可以先定义计算对应的表达式，表达式由变量之间进行加减乘除操作或其它操作组成，表达式也是一个变量。

.. code-block:: python
   
    v1 = var(10)
    v2 = var(5)
  
    add = v1 + v2
    minus = v1 - v2
    multiply = v1 * v2
    divide = v1 / v2

我们可以在不改变表达式结构的情况下，通过 ``set_value`` 接口改变某个变量的值，即可得到不同的计算结果。我们可以调用 ``eval`` 接口，来计算该变量当前的值。

.. code-block:: python
   
    v1 = var(1)
    v2 = var(2)
    
    add = v1 + v2
    print(eval(add)) # 输出为[[3.]]

    v1.set_value([[3.]])
    print(eval(add)) # 输出为[[5.]]

.. note:: 

    变量的 ``get_value`` 接口返回的是变量当前的值，不会根据变量的叶子节点来计算当前变量的值。如果变量是个表达式需要根据叶子节点来计算其值，需要调用 ``eval`` 接口进行前向求值。

实例
---------------

下面我们将以更多的示例来展示变量类相关接口的使用。

.. code-block:: python

    from pyqpanda import *
    import numpy as np

    if __name__=="__main__":

        m1 = np.array([[1., 2.],[3., 4.]])
        v1 = var(m1)

        m2 = np.array([[5., 6.],[7., 8.]])
        v2 = var(m2)

        sum = v1 + v2
        minus = v1 - v2
        multiply = v1 * v2

        print("v1: ", v1.get_value())
        print("v2: ", v2.get_value())
        print("sum: " , eval(sum))
        print("minus: " , eval(minus))
        print("multiply: " , eval(multiply))

        m3 = np.array([[4., 3.],[2., 1.]])
        v1.set_value(m3)

        print("sum: " , eval(sum))
        print("minus: " , eval(minus))
        print("multiply: " , eval(multiply))

.. image:: images/VarExample.png