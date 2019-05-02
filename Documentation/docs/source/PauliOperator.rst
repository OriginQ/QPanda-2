泡利算符类
============================

泡利算符是一组三个2×2的幺正厄米复矩阵，又称酉矩阵。我们一般都以希腊字母 \\(\\sigma\\)（西格玛）来表示，记作 \\(\\sigma_{x}\\)，\\(\\sigma_{y}\\)，\\(\\sigma_{z}\\)。
在 ``QPanda`` 中我们称它们为 \\(X\\) 门，\\(Y\\) 门，\\(Z\\) 门。
它们对应的矩阵形式如下表所示。

.. |X| image:: images/X.svg
   :width: 70px
   :height: 70px

.. |Y| image:: images/Y.svg
   :width: 70px
   :height: 70px
   
.. |Z| image:: images/Z.svg
   :width: 70px
   :height: 70px

====================== =======================         =====================================================================
|X|                     $$\\sigma_{x}$$                   .. math:: \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\quad
|Y|                     $$\\sigma_{y}$$                   .. math:: \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}\quad
|Z|                     $$\\sigma_{z}$$                   .. math:: \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}\quad
====================== =======================         =====================================================================

泡利算符的运算规则如下：

**1.** 泡利算符与自身相乘得到是单位矩阵

$$\\sigma_{x}\\sigma_{x} = I$$ 
$$\\sigma_{y}\\sigma_{y} = I$$ 
$$\\sigma_{z}\\sigma_{z} = I$$ 

**2.** 泡利算符与单位矩阵相乘，无论是左乘还是右乘，其值不变

$$\\sigma_{x}I = I\\sigma_{x} = \\sigma_{x}$$ 
$$\\sigma_{y}I = I\\sigma_{y} = \\sigma_{y}$$ 
$$\\sigma_{z}I = I\\sigma_{z} = \\sigma_{z}$$ 

**3.** 顺序相乘的两个泡利算符跟未参与计算的泡利算符是 \\(i\\) 倍的关系

$$\\sigma_{x}\\sigma_{y} = i\\sigma_{z}$$ 
$$\\sigma_{y}\\sigma_{z} = i\\sigma_{x}$$ 
$$\\sigma_{z}\\sigma_{x} = i\\sigma_{y}$$ 

**4.** 逆序相乘的两个泡利算符跟未参与计算的泡利算符是 \\(-i\\) 倍的关系

$$\\sigma_{y}\\sigma_{x} = -i\\sigma_{z}$$ 
$$\\sigma_{z}\\sigma_{y} = -i\\sigma_{x}$$ 
$$\\sigma_{x}\\sigma_{z} = -i\\sigma_{y}$$ 


接口介绍
-------------

根据泡利算符的上述性质，我们在 ``QPanda`` 中实现了泡利算符类 ``PauliOperator``。我们可以很容易的构造泡利算符类，例如

.. code-block:: cpp

    using namespace QPanda;

    // 构造一个空的泡利算符类
    PauliOperator p1;  
    
    // 2倍的"泡利Z0"张乘"泡利Z1"
    PauliOperator p2("Z0 Z1", 2);

    // 2倍的"泡利Z0"张乘"泡利Z1" + 3倍的"泡利X1"张乘"泡利Y2"
    PauliOperator p3({{"Z0 Z1", 2},{"X1 Y2", 3}});    

    // 构造一个单位矩阵，其系数为2，等价于PauliOperator p4("", 2); 
    PauliOperator p4(2); 


其中PauliOperator p2("Z0 Z1", 2)表示的是 \\(2\\sigma_{0}^{z}\\otimes\\sigma_{1}^{z}\\)。

.. note:: 
    
    构造泡利算符类的时候，字符串里面包含的字符只能是空格、 \\(X\\)、 \\(Y\\) 和 \\(Z\\)中的一个或多个，包含其它字符将会抛出异常。
    另外，同一个字符串里面同一泡利算符的比特索引不能相同，例如：PauliOperator("Z0 Z0", 2)将会抛出异常。

泡利算符类之间可以做加、减、乘等操作，计算返回结果还是一个泡利算符类。

.. code-block:: cpp

    using namespace QPanda;

    PauliOperator a("Z0 Z1", 2);
    PauliOperator b("X5 Y6", 3);

    PauliOperator plus = a + b;
    PauliOperator minus = a - b;
    PauliOperator muliply = a * b;

泡利算符类支持打印功能，我们可以将泡利算符类打印输出到屏幕上，方便查看其值。

.. code-block:: cpp

    using namespace QPanda;

    PauliOperator a("Z0 Z1", 2);
    
    std::cout << a << std::endl

我们在实际使用的时候，常常需要知道该泡利算符类操作了多少个量子比特，这时候我们通过调用泡利算符类getMaxIndex接口即可得到。
如果是空的泡利算符类调用getMaxIndex接口则返回0，否则返回其最大下标索引值加1的结果。

.. code-block:: cpp

    using namespace QPanda;

    PauliOperator a("Z0 Z1", 2);
    PauliOperator b("X5 Y6", 3);
    
    // 输出的值为2
    std::cout << a.getMaxIndex() << std::endl;
    // 输出的值为7
    std::cout << b.getMaxIndex() << std::endl;

如果我们构造的的泡利算符类，其中泡利算符的下标索引不是从0开始分配的，例如PauliOperator b("X5 Y6", 3)调用getMaxIndex接口返回的使用的比特数是7，其实其
只使用了2个比特。我们如何才能返回其真实用到的比特数呢。我们可以调用泡利算符类里面remapQubitIndex接口，它的功能是对泡利算符类中的索引从0比特开始分配映射，
并返回新的泡利算符类，该接口需要传入一个map来保存前后下标的映射关系。

.. code-block:: cpp

    using namespace QPanda;

    PauliOperator b("X5 Y6", 3);

    std::map<size_t, size_t> index_map;
    auto c = b.remapQubitIndex(index_map);
    
    // 输出的值为7
    std::cout << b.getMaxIndex() << std::endl;
    // 输出的值为2
    std::cout << c.getMaxIndex() << std::endl;


实例
-------------

以下实例主要是展示 ``PauliOperator`` 接口的使用方式.

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