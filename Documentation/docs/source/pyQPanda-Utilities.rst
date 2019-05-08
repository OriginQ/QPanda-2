.. _pyQPanda-Utilities:

常用工具接口
===========
----

pyQPanda提供的常用工具囊括了几乎所有的QPanda2工具接口，使用方式基本相同，都是以量子程序为操作对象，对其进行转换、解析、统计以及存储等。

因此使用这些工具之前我们需要进行基本的量子程序构建

    .. code-block:: python

        from pyqpanda import *

        machine = init_quantum_machine(QMachineType.CPU)

        prog = QProg()
        cir = QCircuit()
        q = machine.qAlloc_many(6)
        c = machine.cAlloc_many(6)

        prog.insert(H(q[0]))\
            .insert(Y(q[5]))\
            .insert(S(q[2]))\
            .insert(RX(q[0], PI / 3))\
            .insert(RY(q[2], PI / 3))\
            .insert(RZ(q[4], PI / 3))\
            .insert(CZ(q[0], q[1]))\
            .insert(measure_all(q, c))

量子程序转化QRunes
>>>>>>>>>>>>>>>>>>
----

关于QRunes指令格式可以参考QPanda2中的 :ref:`QRunes介绍` 部分。pyQPanda使用 ``to_QRunes`` 接口实现转换，以上述代码为例，调用接口实例如下：

    .. code-block:: python

        print(to_QRunes(prog,machine))

    运行结果：

    .. code-block:: python

        QINIT 6
        CREG 6
        H 0
        Y 5
        S 2
        RX 0,"0.785398"
        RY 2,"1.047198"
        RZ 4,"1.047198"
        CZ 0,1
        MEASURE 0,$0
        MEASURE 1,$1
        MEASURE 2,$2
        MEASURE 3,$3
        MEASURE 4,$4
        MEASURE 5,$5

量子程序转化QASM
>>>>>>>>>>>>>>>>
----

关于QASM指令格式可以参考QPanda2中的 :ref:`QASM介绍` 部分。pyQPanda使用 ``to_QASM`` 接口实现转换，以上述代码为例，调用接口实例如下：

    .. code-block:: python

        print(to_QASM(prog,machine))

    运行结果：

    .. code-block:: python

        openqasm 2.0;
        qreg q[6];
        creg c[6];
        h q[0];
        y q[5];
        s q[2];
        rx(0.785398) q[0];
        ry(1.047198) q[2];
        rz(1.047198) q[4];
        cz q[0],q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        measure q[3] -> c[3];
        measure q[4] -> c[4];
        measure q[5] -> c[5];

量子程序转化Quil
>>>>>>>>>>>>>>>>>>
----

关于Quil指令格式可以参考QPanda2中的 :ref:`Quil介绍` 部分。pyQPanda使用 ``to_Quil`` 接口实现转换，以上述代码为例，调用接口实例如下：

    .. code-block:: python

        print(to_Quil(prog,machine))

    运行结果：

    .. code-block:: python

        H 0
        Y 5
        S 2
        RX(0.785398) 0
        RY(1.047198) 2
        RZ(1.047198) 4
        CZ 0 1
        MEASURE 0 [0]
        MEASURE 1 [1]
        MEASURE 2 [2]
        MEASURE 3 [3]
        MEASURE 4 [4]
        MEASURE 5 [5]

量子逻辑门数量统计
>>>>>>>>>>>>>>>>>>
----

逻辑门的统计是指统计一个量子线路或量子程序中所有的量子逻辑门个数方法。pyQPanda使用 ``count_gate`` 接口实现该功能，以上述代码为例，调用接口实例如下：

    .. code-block:: python

        print(count_gate(prog))
        print(count_gate(cir))

    运行结果：

    .. code-block:: python

        13
        0

统计量子程序时钟周期
>>>>>>>>>>>>>>>>>>>>>>
----

统计量子程序时钟周期用于估算一个量子程序运行所需要的时间。用户可自由配置每个量子逻辑门的时间。

如果未设置则会给定一个默认值，单量子门的默认时间为2，双量子门的时间为5。

pyQPanda使用 ``get_clock_cycle`` 接口实现该功能，以上述代码为例，调用接口实例如下：

    .. code-block:: python

        print(get_clock_cycle(prog,machine))

    运行结果：

    .. code-block:: python

        17

量子程序转化二进制数据
>>>>>>>>>>>>>>>>>>>>>
----

量子程序转化二进制数据的功能是根据给定的转换格式协议，将量子程序以二进制的方式存储，从而降低了数据的存储开销

pyQPanda使用 ``get_bin_data`` 接口实现该功能，以上述代码为例，调用接口实例如下：

    .. code-block:: python

        bin_data = get_bin_data(prog,machine))

同时pyQPanda提供了另一个接口，先将量子程序转化成二进制数据，然后通过Base64编码生成字符串

Base64是网络上最常见的用于传输8Bit字节码的编码方式之一，它是一种基于64个可打印字符来表示二进制数据的方法，具有不可读性，需要解码后才能阅读。

pyQPanda使用 ``get_bin_str`` 接口实现该功能，调用接口实例如下：

    .. code-block:: python

        bin_str = get_bin_str(prog,machine))










