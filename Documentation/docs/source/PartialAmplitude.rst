.. _部分振幅量子虚拟机:

部分振幅量子虚拟机
=========================
----

目前用经典计算机模拟量子虚拟机的主流解决方案有全振幅与单振幅两种。除此之外，还有部分振幅量子虚拟机，该方案能在更低的硬件条件下，实现更高的模拟效率。

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
            print(result["0000000000"])
            print(result["0000000001"])

        运行结果如下:

        .. code-block:: python

            (-0.0064720869120793835-0.0064720869120793185j)
            (-3.5497357850862835e-17-0.009152913087920036j)

    - ``pmeasure(string)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure("6")
            print(result)

        运行结果如下:

        .. code-block:: python

            {'0': 8.377581799501766e-05, 
             '1': 8.377581799501789e-05, 
             '2': 8.37758179950177e-05, 
             '3': 8.377581799501786e-05, 
             '4': 0.00048828124999996357, 
             '5': 0.0004882812499999648}

    - ``pmeasure(QVec,string)`` ,使用示例

        .. code-block:: python

            qlist = [q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]]
            result = machine.pmeasure(qlist, "3")
            print(result)

        运行结果如下:

        .. code-block:: python

            {'0': 0.00016755163599003553, 
             '1': 0.00016755163599003556, 
             '2': 0.0009765624999999284}

    - ``get_prob_dict(qvec,string)`` ,使用示例

        .. code-block:: python

            qlist = [q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]]
            result = machine.get_prob_dict(qlist, "3")
            print(result)

        运行结果如下:

        .. code-block:: python

            {'000000000': 0.00016755163599003553, 
             '000000001': 0.00016755163599003556, 
             '000000010': 0.0009765624999999284}

    - ``pmeasure_bin_index(string)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure_bin_index("0000000000")
            print(result)

        结果输出如下：

        .. code-block:: python

            8.377581799501766e-05

    - ``pmeasure_dec_index(string)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure_bin_index("1")
            print(result)

        结果输出如下：

        .. code-block:: python

            8.377581799501766e-05

