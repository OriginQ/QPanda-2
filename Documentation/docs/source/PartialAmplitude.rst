.. _部分振幅量子虚拟机:

部分振幅量子虚拟机
=========================
----

目前用经典计算机模拟量子虚拟机的主流解决方案有全振幅与单振幅两种。除此之外，还有部分振幅量子虚拟机，该方案能在更低的硬件条件下，实现更高的模拟效率。
然而该方案并不适用于所有的量子线路，使用情况如下所述。

算法适用条件
>>>>>>>>>>>>>
----

 - ``量子比特数目要求`` 。部分振幅算法要求量子线路的总量子比特数目必须为偶数，否则无法对量子线路进行有效拆分。
 - ``双比特量子逻辑门要求`` 。若量子线路中出现跨节点的双量子逻辑门，即以线路中量子比特数为总量子比特数的一半为界线，双量子逻辑门的控制位量子比特与目标位量子比特分别处于界线的两侧，称为跨节点。部分振幅模拟由于其算法特殊性，对于某些跨节点双量子逻辑门，如 ``CNOT`` 、 ``CZ`` 等，可以将其分解为 ``P0`` 、 ``P1`` 和基础单量子逻辑门的组合，而不支持如 ``CR`` 、``iSWAP`` 、 ``SqiSWAP`` 等。

使用介绍
>>>>>>>>>>>>>>>>
----

其使用方式与前面介绍的量子虚拟机模块非常类似，首先通过 ``PartialAmpQVM`` 初始化一个部分振幅量子虚拟机对象用于管理后续一系列行为

    .. code-block:: python

        machine = PartialAmpQVM()

然后是量子程序的初始化、构建与装载过程，以QPanda2的 :ref:`部分振幅示例程序`来演示：

    .. code-block:: python

        machine.initQVM()

        q = machine.qAlloc_many(10)
        c = machine.cAlloc_many(10)

        prog = QProg()

        prog.insert(Hadamard_Circuit(q))\
            .insert(CZ(q[1], q[5]))\
            .insert(CZ(q[3], q[7]))\
            .insert(CZ(q[0], q[4]))\
            .insert(RZ(q[7], PI / 4))\
            .insert(RX(q[5], PI / 4))\
            .insert(RX(q[4], PI / 4))\
            .insert(RY(q[3], PI / 4))\
            .insert(CZ(q[2], q[6]))\
            .insert(RZ(q[3], PI / 4))\
            .insert(RZ(q[8], PI / 4))\
            .insert(CZ(q[9], q[5]))\
            .insert(RY(q[2], PI / 4))\
            .insert(RZ(q[9], PI / 4))\
            .insert(CZ(q[2], q[3]))

        machine.run(prog)

部分接口使用如下：

    - ``get_qstate()``

        .. code-block:: python

            result = machine.get_qstate()
            print(result)

        运行结果如下:

        .. code-block:: python

            (-0.0064720869120793835,-0.0064720869120793185j)
            (-3.5497357850862835e-17,-0.009152913087920036j)
            (-0.0064720869120793835,-0.0064720869120793185j)
            ...

    - ``pmeasure(size_t)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure(6)
            print(result)

        运行结果如下:

        .. code-block:: python

            [(0, 8.377581799501766e-05),
             (1, 8.377581799501789e-05), 
             (2, 8.377581799501766e-05), 
             (3, 8.377581799501789e-05), 
             (4, 0.00048828124999996357), 
             (5, 0.0004882812499999648)]

    - ``pmeasure(QVec,size_t)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure(q,6)
            print(result)

        运行结果如下:

        .. code-block:: python

            [8.377581799501766e-05, 
             8.377581799501789e-05, 
             8.377581799501766e-05, 
             8.377581799501789e-05, 
             0.0004882812499999635, 
             0.0004882812499999648] 

    - ``get_prob_dict(qvec,size_t)`` ,使用示例

        .. code-block:: python

            result = machine.get_prob_dict(q,6)
            print(result)

        运行结果如下:

        .. code-block:: python

            {'0000000000': 8.377581799501766e-05, 
             '0000000001': 8.377581799501789e-05, 
             '0000000010': 8.377581799501766e-05, 
             '0000000011': 8.377581799501789e-05, 
             '0000000100': 0.00048828124999996357, 
             '0000000101': 0.0004882812499999648}

    - ``get_prob_tuple_list(qvec,size_t)`` ,使用示例

        .. code-block:: python

            result = machine.get_prob_tuple_list(q,6)
            print(result)

        运行结果如下:

        .. code-block:: python

            [(0, 8.377581799501766e-05), 
             (1, 8.377581799501789e-05), 
             (2, 8.377581799501766e-05),
             (3, 8.377581799501789e-05),
             (4, 0.00048828124999996357), 
             (5, 0.0004882812499999648)]  

