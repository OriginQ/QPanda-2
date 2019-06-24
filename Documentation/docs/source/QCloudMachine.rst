量子云虚拟机
=====================
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

量子云虚拟机有两种计算任务提交接口。即 ``run_with_configuration(测量操作)`` 和 ``prob_run_dict(概率测量操作)`` ,

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

    .. note:: 
        - 量子云平台用户验证标识token需要用户从本源量子云平台个人信息下获取。
        - 量子云虚拟机除了经典的全振幅算法以外，现已支持单振幅、部分振幅等量子虚拟机模拟。