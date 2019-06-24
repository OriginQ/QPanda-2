.. _单振幅量子虚拟机:

单振幅量子虚拟机
======================
----

目前我们可以通过量子计算的相关理论，用经典计算机实现模拟量子虚拟机。
量子虚拟机的模拟主要有全振幅与单振幅两种解决方案，其主要区别在于，全振幅一次模拟计算就能算出量子态的所有振幅，单振幅一次模拟计算只能计算出 :math:`2^{N}` 个振幅中的一个。

然而全振幅模拟量子计算时间较长，计算量随量子比特数指数增长，
在现有硬件下，无法模拟超过49量子比特。通过单振幅量子虚拟机技术可以模拟超过49量子比特，同时模拟速度有较大提升，且算法的计算量不随量子比特数指数提升。

使用介绍
>>>>>>>>>>>>>>>>
----

其使用方式与前面介绍的量子虚拟机模块非常类似，首先通过 ``SingleAmpQVM`` 初始化一个单振幅量子虚拟机对象用于管理后续一系列行为

    .. code-block:: python

        machine = SingleAmpQVM()

然后是量子程序的初始化、构建与装载过程：

    .. code-block:: python

        machine.initQVM()

        q = machine.qAlloc_many(10)
        c = machine.cAlloc_many(10)

        prog = QProg()

        prog.insert(Hadamard_Circuit(q))\
            .insert(CZ(q[1], q[5]))\
            .insert(CZ(q[3], q[5]))\
            .insert(CZ(q[2], q[4]))\
            .insert(CZ(q[3], q[7]))\
            .insert(CZ(q[0], q[4]))\
            .insert(RY(q[7], PI / 2))\
            .insert(RX(q[8], PI / 2))\
            .insert(RX(q[9], PI / 2))\
            .insert(CR(q[0], q[1], PI))\
            .insert(CR(q[2], q[3], PI))\
            .insert(RY(q[4], PI / 2))\
            .insert(RZ(q[5], PI / 4))\
            .insert(RX(q[6], PI / 2))\
            .insert(RZ(q[7], PI / 4))\
            .insert(CR(q[8], q[9], PI))\
            .insert(CR(q[1], q[2], PI))\
            .insert(RY(q[3], PI / 2))\
            .insert(RX(q[4], PI / 2))\
            .insert(RX(q[5], PI / 2))\
            .insert(CR(q[9], q[1], PI))\
            .insert(RY(q[1], PI / 2))\
            .insert(RY(q[2], PI / 2))\
            .insert(RZ(q[3], PI / 4))\
            .insert(CR(q[7], q[8], PI))

        machine.run(prog)

部分接口使用如下：

    - ``get_qstate()``

        .. code-block:: python

            result = machine.get_qstate()
            print(result["0000000000"])
            print(result["0000000001"])

        运行结果如下:

        .. code-block:: python

            (0.040830060839653015,-9.313225746154785e-10j)
            (0.040830060839653015,-9.313225746154785e-10j)

    - ``pmeasure(string)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure("6")
            print(result)

        运行结果如下:

        .. code-block:: python

            (0, 0.0016670938348397613)
            (1, 0.0016670938348397613)
            (2, 0.0002860281092580408)
            (3, 0.0002860281092580408)
            (4, 0.0002860281092580408)
            (5, 0.0002860281092580408)

    - ``pmeasure(QVec,string)`` ,使用示例

        .. code-block:: python

            qlist = [q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9]]
            result = machine.pmeasure(qlist, "3")
            print(result)

        运行结果如下:

        .. code-block:: python

            {'0': 0.0033341876696795225, 
             '1': 0.0005720562185160816, 
             '2': 0.0005720562185160816}

    - ``get_prob_dict(qvec,string)`` ,使用示例

        .. code-block:: python

            result = machine.get_prob_dict(q,"6")
            print(result)

        运行结果如下:

        .. code-block:: python

            {'0000000000': 0.0016670938348397613, 
             '0000000001': 0.0016670938348397613, 
             '0000000010': 0.0002860281092580408, 
             '0000000011': 0.0002860281092580408,
             '0000000100': 0.0002860281092580408, 
             '0000000101': 0.0002860281092580408}

    - ``pmeasure_bin_index(string)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure_bin_index("0000000000")
            print(result)

        结果输出如下：

        .. code-block:: python

            0.0016670938348397613

    - ``pmeasure_dec_index(string)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure_bin_index("1")
            print(result)

        结果输出如下：

        .. code-block:: python

            0.0016670938348397613
