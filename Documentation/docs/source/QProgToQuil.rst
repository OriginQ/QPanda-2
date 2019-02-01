量子程序转化为Quil
======================

简介
--------------

| Quil可以从一个很低级的层次直接描述量子程序、量子算法，它的地位类似于经典计算机中的硬件描述语言或者汇编语言。Quil基本采用“指令+参数列表”的设计方法。一个简单的量子程序例子如下：

.. code-block:: c

    X 0
    Y 1
    CNOT 0 1
    H 0
    RX(-3.141593) 0
    MEASURE 1 [0]

- ``X`` 的作用是对目标量子比特进行 ``Pauli-X`` 门操作。与之类似的关键词有 ``Y`` 、``Z``  、 ``H`` 等等。
- ``Y`` 的作用是对目标量子比特进行 ``Pauli-Y`` 门操作。
- ``CNOT`` 的作用是对两个量子比特执行 ``CNOT`` 操作。输入参数为控制量子比特序号和目标量子比特序号。
- ``H`` 的作用是对目标量子比特进行 ``Hadamard`` 门操作。
- ``MEASURE`` 的作用对目标量子比特进行测量并将测量结果保存在对应的经典寄存器里面，输入参数为目标量子比特序号和保存测量结果的经典寄存器序号。

.. _pyQuil: https://pyquil.readthedocs.io/en/stable/compiler.html

上述仅为Quil指令集语法的一小部分， 详细介绍请参考 pyQuil_

接口介绍
-----------------

.. cpp:function:: std::string qProgToQuil(QProg &prog)
    
    **功能**
        量子程序转化为Quil指令集
    **参数**
        - prog 量子程序
    **返回值**
        Quil指令集

实例
---------------

    .. code-block:: c

        #include <QPanda.h>
        USING_QPANDA


        int main(void)
        {
            init();
            QProg prog;
            auto qvec = qAllocMany(4);
            auto cvec = cAllocMany(4);

            prog << X(qvec[0])
                << Y(qvec[1])
                << H(qvec[0])
                << RX(qvec[0], 3.14)
                << Measure(qvec[1], cvec[0])
                ;
            load(prog);

            auto quil = qProgToQuil(prog);
            std::cout << quil << std::endl;
            finalize();
            return 0;
        }


