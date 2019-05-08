.. _pyQPanda-OtherQuantumMachine:

其他量子虚拟机
=============
----

除了最基础的全振幅量子虚拟机之外，pyQPanda封装了其他不同功能的量子虚拟机。

根据量子计算模拟方法的不同，可分为全振幅、单振幅以及部分振幅量子虚拟机。

根据机器硬件的不同，可分为CPU量子虚拟机与GPU量子虚拟机等等

同时根据任务处理方式的不同，又可分为单计算节点量子虚拟机，分布式计算集群量子虚拟机以及量子云虚拟机等等

单振幅量子虚拟机
>>>>>>>>>>>>>>>
----

关于单振幅虚拟机的介绍可以参考QPanda2的 :ref:`单振幅量子虚拟机`，下面重点介绍在pyQPanda中如何使用。

其使用方式与前面介绍的量子虚拟机模块非常类似，首先通过 ``SingleAmpQVM`` 初始化一个单振幅量子虚拟机对象用于管理后续一系列行为

    .. code-block:: python

        machine = SingleAmpQVM()

然后是量子程序的初始化、构建与装载过程，以QPanda2的 :ref:`单振幅示例程序`来演示：

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
            print(result)

        运行结果如下:

        .. code-block:: python

            (0.040830060839653015,-9.313225746154785e-10j)
            (0.040830060839653015,-9.313225746154785e-10j)
            (-0.016912365332245827,0j)
            ...

    - ``pmeasure(size_t)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure(6)
            print(result)

        运行结果如下:

        .. code-block:: python

            (0, 0.0016670938348397613)
            (1, 0.0016670938348397613)
            (2, 0.0002860281092580408)
            (3, 0.0002860281092580408)
            (4, 0.0002860281092580408)
            (5, 0.0002860281092580408)

    - ``pmeasure(QVec,size_t)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure(q,6)
            print(result)

        运行结果如下:

        .. code-block:: python

            0.0016670938348397613
            0.0016670938348397613
            0.0002860281092580408
            0.0002860281092580408
            0.0002860281092580408
            0.0002860281092580408

    - ``get_prob_dict(qvec,size_t)`` ,使用示例

        .. code-block:: python

            result = machine.get_prob_dict(q,6)
            print(result)

        运行结果如下:

        .. code-block:: python

            {'0000000000': 0.0016670938348397613, 
             '0000000001': 0.0016670938348397613, 
             '0000000010': 0.0002860281092580408, 
             '0000000011': 0.0002860281092580408,
             '0000000100': 0.0002860281092580408, 
             '0000000101': 0.0002860281092580408}

    - ``get_prob_tuple_list(qvec,size_t)`` ,使用示例

        .. code-block:: python

            result = machine.get_prob_tuple_list(q,6)
            print(result)

        运行结果如下:

        .. code-block:: python

            [(0, 0.0016670938348397613), 
             (1, 0.0016670938348397613), 
             (2, 0.0002860281092580408), 
             (3, 0.0002860281092580408), 
             (4, 0.0002860281092580408), 
             (5, 0.0002860281092580408)]    

    - ``PMeasure_index(size_t)`` ,使用示例

        .. code-block:: python

            result = machine.pmeasure_index(1)
            print(result)

        结果输出如下：

        .. code-block:: python

            0.0016670938348397613
    


部分振幅量子虚拟机
>>>>>>>>>>>>>>>>>
----

关于部分振幅虚拟机的介绍可以参考QPanda2的 :ref:`部分振幅量子虚拟机`，下面重点介绍在pyQPanda中如何使用。

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

量子云虚拟机
>>>>>>>>>>>>>>>
----

量子计算的模拟方法受限于机器的硬件水平，因此对于复杂的量子线路模拟有必要借助于高性能计算机集群，用云计算的方式替代本地计算，
从而一定程度上减轻了用户的计算成本，并帮助用户获得更好的计算体验。

pyQPanda封装了量子云虚拟机，可以向本源量子的计算服务器集群发送计算指令，同时根据生成的唯一任务标识进行计算结果的查询等操作。

首先通过 ``QCloud()`` 构建量子云虚拟机对象，然后用 ``initQVM()`` 初始化系统资源

        .. code-block:: python

            QCM = QCloud()
            QCM.initQVM()

接着构建量子程序

        .. code-block:: python

            qlist = QCM.qAlloc_many(10)
            clist = QCM.qAlloc_many(10)
            prog = QProg()
            for i in qlist:
            
            prog.insert(Hadamard_Circuit(q))\
                .insert(CZ(qlist[1], qlist[5]))\
                .insert(CZ(qlist[3], qlist[7]))\
                .insert(CZ(qlist[0], qlist[4]))\
                .insert(RZ(qlist[7], PI / 4))\
                .insert(RX(qlist[5], PI / 4))\
                .insert(RX(qlist[4], PI / 4))\
                .insert(RY(qlist[3], PI / 4))\
                .insert(CZ(qlist[2], qlist[6]))\
                .insert(RZ(qlist[3], PI / 4))\
                .insert(RZ(qlist[8], PI / 4))\
                .insert(CZ(qlist[9], qlist[5]))\
                .insert(RY(qlist[2], PI / 4))\
                .insert(RZ(qlist[9], PI / 4))\
                .insert(CZ(qlist[2], qlist[3]))

量子云虚拟机有两种计算任务提交接口。即 ``run_with_configuration(测量操作)`` 和 ``prob_run_dict(概率测量)`` ,di

    - ``run_with_configuration(QProg，dict)`` ：

        测量操作前需要先配置操作参数：

        .. code-block:: python

            param = {"RepeatNum": 1000, "token": "3CD107AEF1364924B9325305BF046FF3", "BackendType": QMachineType.NOISE}

        参数说明：

            - ``RepeatNum`` ：测量操作重复的次数
            - ``token`` ：量子云平台用户验证标识
            - ``BackendType`` ：量子虚拟机类型

        然后提交计算任务

        .. code-block:: python

            task = QCM.run_with_configuration(prog, param)
            print(task)
        
        根据输出结果可以看到当前任务标识(TaskId)和任务状态(TaskState)
        
        .. code-block:: python

            {"TaskId":"1904301115021600","TaskState":"1"}

        利用 ``get_result`` 接口,通过TaskId就可以对计算结果进行查询
        
        .. code-block:: python

                result = QCM.get_result("1904301115021600")
                print(result)

        结果输出如下：
        
        .. code-block:: python

            0000000000 , 1.0

    - ``prob_run_dict(QProg，dict)`` ：

        概率操作前也需要先配置操作参数，与测量不同，仅需要配置 ``token`` (量子云平台用户验证标识)与 ``BackendType`` (量子虚拟机类型)即可。

        .. code-block:: python

            param2 = {"token": "3CD107AEF1364924B9325305BF046FF3","BackendType": QMachineType.CPU}

        然后提交计算任务

        .. code-block:: python

            task = QCM.prob_run_dict(prog, param)
            print(task)
        
        根据输出结果可以看到当前任务标识(TaskId)和任务状态(TaskState)
        
        .. code-block:: python

            {"TaskId":"1904301115021601","TaskState":"1"}

        利用 ``get_result`` 接口,通过TaskId就可以对计算结果进行查询
        
        .. code-block:: python

                result = QCM.get_result("1904301115021601")
                print(result)

        结果输出如下：
        
        .. code-block:: python

            '0011000010': 0.0028459116820049733, 
            '0011100011': 0.0028459116820049733, 
            '0011110011': 0.0028459116820049733, 
            ...





























