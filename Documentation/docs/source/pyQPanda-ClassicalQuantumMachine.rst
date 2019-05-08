.. _pyQPanda-ClassicalQuantumMachine:

经典量子虚拟机
=============
----

关于量子虚拟机可以参考QPanda2中的量子虚拟机介绍，pyQPanda中的量子虚拟机使用流程与C++版本的类似,分为多种方式

全局量子虚拟机接口
>>>>>>>>>>>>>>>>>
----

pyQPanda提供了一种最简单的量子虚拟机使用方式，即通过全局量子虚拟机接口，其内部使用的是全振幅量子计算模拟方法

    .. code-block:: python
    
        init(QMachineType.CPU)

首先必须通过init(QMachineType.CPU)全局函数进行量子虚拟机与系统资源初始化，接着用qAlloc_many()和cAlloc_many()全局函数进行量子程序构建 

    .. code-block:: python
    
        prog = QProg()
        q = qAlloc_many(2)
        c = cAlloc_many(2)
        prog.insert(H(q[0]))
        prog.insert(CNOT(q[0],q[1]))
        prog.insert(measure_all(q,c))

最后用 ``run_with_configuration(QProg,cbit_list,shots)`` 运行量子程序

    .. code-block:: python
    
        result = run_with_configuration(prog, cbit_list = c, shots = 1000)
        print(result)
        finalize()

该接口表示执行多次测量操作，得到的运行结果如下:

    .. code-block:: python

        {'00': 493, '11': 507}

    .. note:: 
        - 需要传入的参数依次为包含测量节点的量子程序，保存测量结果的cbit lists以及重复执行量子程序中的测量操作次数
        - 运行的量子程序中必须有测量节点，计算的结果才有意义。

若使用其它接口对上述量子程序进行操作：

    - ``directly_run(QProg)`` ,使用示例

        .. code-block:: python

            result = directly_run(prog)
            print(result)

        该接口表示执行单次测量操作，运行结果如下：
        
        .. code-block:: python

            {'c0': True, 'c1': True}
    
        .. note:: 
            - 该接口还有另一种使用场景，即进行量子态概率操作之前，比如 ``PMeasure`` 、 ``PMeasure_no_index`` 必须先调用该接口。

    - ``prob_run_tuple_list(QProg,qubit_list,select_max)`` ,使用示例

        .. code-block:: python

                result = prob_run_tuple_list(prog,q,4)
                print(result)

        运行结果如下：

        .. code-block:: python

            [(0, 1.0), (1, 0.0), (2, 0.0), (3, 0.0)]

        .. note:: 
            - 该接口参数除了QProg与qubit_list之外，select_max表示返回结果集的前多少项

            - PMeasure概率测量操作接口最后一个参数select max为非必要参数，其值表示返回结果集的前多少项，默认为-1，即返回所有结果

    - ``prob_run_dict(QProg,qubit_list,select_max)`` ,使用示例

        .. code-block:: python

                result = prob_run_dict(prog,q,4)
                print(result)

        运行结果如下：

        .. code-block:: python

            {'00': 1.0, '01': 0.0, '10': 0.0, '11': 0.0}

    - ``prob_run_list(QProg,qubit_list,select_max)`` ,使用示例

        .. code-block:: python

                result = prob_run_list(prog,q,4)
                print(result)

        运行结果如下：

        .. code-block:: python

            [0.0, 0.0, 0.0, 1]

非全局量子虚拟机接口
>>>>>>>>>>>>>>>>>
----

上述接口实际上是通过全局隐藏的量子虚拟机对象调用实现的，功能并不完善，因此pyQPanda提供另一种量子虚拟机使用方法，本质上来说大同小异，比如:

        .. code-block:: python

            machine = init_quantum_machine(QMachineType.CPU)
            machine.initQVM()

            q = machine.qAlloc_many(6)
            c = machine.cAlloc_many(6)

            prog = QProg()
            prog.insert(Hadamard_Circuit(q))\
                .insert(T(q[0]))\
                .insert(Y(q[1]))\
                .insert(RX(q[3], PI / 3))\
                .insert(RY(q[2], PI / 3))\
                .insert(CNOT(q[1], q[5]))
                .insert(measure_all(q,c))

            measure_result = machine.directly_run(prog)
            pmeasure_result = machine.pmeasure(q)

            prob_dict_result = machine.prob_run_dict(prog,q)
            prob_list_result = machine.prob_run_list(prog,q)

可以看到，上述例子是通过初始化一个实例来管理一系列量子虚拟机行为的，内部实现机制与使用方法与全局量量子虚拟机接口完全相同

.. note:: 
    - 虽然全局或非全局量子虚拟机接口内部实现是完全一样的，但是不能混淆使用，因为他们分别属于不同的量子虚拟机对象
    - 在pyQPanda中更建议使用非全局量子虚拟机接口
