量子程序转化QRunes
=======================
----

通过该功能模块，你可以解析通过QPanda2构建的量子程序，将其中包含的量子比特信息以及量子逻辑门操作信息提取出来，得到按固定格式存储的QRunes指令集。

.. _本源量子计算云平台官网: https://qcode.qubitonline.cn/QCode/index.html

.. _QRunes介绍:

QRunes介绍
>>>>>>>>>>>>>>>>>
----

QRunes是本源量子推出的量子指令集,是一种运行在量子计算机上的一组基本指令，它可以直接控制量子计算机的运行。
QRunes可以从一个很低级的层次直接描述量子程序、量子算法，它的地位类似于经典计算机中的硬件描述语言或者汇编语言。

特别要提到的是，QRunes的设计目的是为了直接操纵量子计算机的动作，在一个QRunes程序中，只包含了量子计算机一次执行所需要进行的动作。
也就是说，这一组指令集不包含任何逻辑判断。语言所不具备的变量系统，都将以更高层次的量子语言去封装。

QRunes的语法十分直接，基本采用了“指令+参数列表”的设计方法，一个简单的量子程序的例子如下所示:

    ::

        QINIT 6
        CREG 2
        H 0
        CNOT 0,1
        CONTROL 1
        X 1
        Y 2
        ENDCONTROL 1
        DAGGER
        X 2
        CZ 0,1
        ENDDAGGER
        MEASURE 0,$0
        MEASURE 1,$1

QRunes语句中部分关键词作用如下：

 -  ``%`` 的作用是从%开始，到该行的结尾，是程序的行注释，就类似于C语言的"//",注释的语句会被完全忽略。
 -  ``QUINT`` 的作用是在量子程序中第一行（除注释之外）显式定义量子比特数,这一行定义将被自动附带到程序的开头。
 -  ``CREG`` 的作用是在一个量子程序中第二行（除注释之外）显式定义经典寄存器数。在量子计算机运行时，所有的测量值都会被保存到经典计算机上并且导出。这一行定义将被自动附带到程序的第二行。
 -  ``H`` 的作用是对目标量子比特进行Hadamard门操作,与之类似的关键词有X、Y、NOT等等。
 -  ``CNOT`` 的作用是对两个量子比特执行CNOT操作。输入参数为控制量子比特序号和目标量子比特序号,与之类似的关键词有CZ,ISWAP等。
 -  ``MEASURE`` 的作用对目标量子比特进行测量并将测量结果保存在对应的经典寄存器里面，输入参数为目标量子比特序号和保存测量结果的经典寄存器序号。
 -  ``CONTROL & ENDCONTROL`` 的作用是根据经典寄存器的值对CONTROL与ENDCONTROL语句之间的操作进行受控操作
 -  ``DAGGER & ENDDAGGER`` 的作用是对DAGGER与ENDDAGGER语句之间的操作进行转置共轭操作

上述语句只是QRunes语法中的一小部分,QRunes支持更多的逻辑门种类同时还包含每个量子线路和每个量子逻辑门中是否包含受控量子比特信息以及是否Dagger。

关于QRunes更多详细信息的介绍、使用与体验请参考 `本源量子计算云平台官网`_

QPanda2提供了QRunes转换工具接口 ``std::string transformQProgToQRunes(QProg &, QuantumMachine*)`` 该接口使用非常简单，具体可参考下方示例程序。

实例
>>>>>>>>>>>>>>
----

下面的例程通过简单的接口调用演示了量子程序转化QRunes指令集的过程

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            auto qvm = initQuantumMachine();

            auto prog = CreateEmptyQProg();
            auto cir = CreateEmptyCircuit();

            auto q = qvm->allocateQubits(6);
            auto c = qvm->allocateCBits(6);


            cir << Y(q[2]) << H(q[2]);
            cir.setDagger(true);

            auto h1 = H(q[1]);
            h1.setDagger(true);
            
            prog << H(q[1]) 
                << X(q[2]) 
                << h1 
                << RX(q[1], 2 / PI) 
                << cir 
                << CR(q[1], q[2], PI / 2)
                <<MeasureAll(q,c);

            cout << transformQProgToQRunes(prog,qvm);

            qvm->finalize();
            delete qvm;
            return 0;
        }



具体步骤如下:

 - 首先在主程序中用 ``initQuantumMachine()`` 初始化一个量子虚拟机对象，用于管理后续一系列行为

 - 接着用 ``allocateQubits()`` 和 ``allocateCBits()`` 初始化量子比特与经典寄存器数目

 - 然后调用 ``CreateEmptyQProg()`` 构建量子程序

 - 最后调用接口 ``transformQProgToQRunes`` 输出QRunes指令集并用 ``finalize()`` 释放系统资源

运行结果如下：

    .. code-block:: c

        QINIT 6
        CREG 6
        H 1
        X 2
        DAGGER
        H 1
        ENDAGGER
        RX 1,"0.636620"
        DAGGER
        Y 2
        H 2
        ENDAGGER
        CR 1,2,"1.570796"
        MEASURE 0,$0
        MEASURE 1,$1
        MEASURE 2,$2
        MEASURE 3,$3
        MEASURE 4,$4
        MEASURE 5,$5

   .. note:: 对于暂不支持的操作类型，QRunes会显示UnSupported XXXNode，其中XXX为具体的节点类型。
