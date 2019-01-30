量子程序转化QASM
====================
----

通过该功能模块，你可以解析通过QPanda2构建的量子程序，将其中包含的量子比特信息以及量子逻辑门操作信息提取出来，得到按固定格式存储的QASM指令集。

.. _QASM介绍:
.. _IBM Q Experience量子云平台: https://quantumexperience.ng.bluemix.net/qx/editor

QASM介绍
>>>>>>>>>>>>
----

QASM(Quantum Assembly Language)是IBM公司提出的量子汇编语言，与 :ref:`QRunes介绍` 中的语法规则类似，一段QASM代码如下所示：

    :: 

        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[10];
        creg c[10];

        x q[0];
        h q[1];
        tdg q[2];
        sdg q[2];
        cx q[0],q[2];
        cx q[1],q[4];
        u1(pi) q[0];
        u2(pi,pi) q[1];
        u3(pi,pi,pi) q[2];
        cz q[2],q[5];
        ccx q[3],q[4],q[6];
        cu3(pi,pi,pi) q[0],q[1];
        measure q[2] -> c[2];
        measure q[0] -> c[0];


需要注意的是，QASM的语法格式与QRunes形相似而神不同，主要区别有以下几点:

 - QRunes对于需要进行转置共轭操作的量子逻辑门与量子线路，需要将目标置于DAGGER与ENDAGGER语句之间，而QASM会直接进行转化。
 - QRunes支持对量子逻辑门与量子线路施加控制操作，而QASM不支持，在对量子程序转化QASM指令集之前，会对其中包含的控制操作进行分解。


关于QASM更多详细信息的介绍、使用与体验请参考 `IBM Q Experience量子云平台`_


功能函数接口
>>>>>>>>>>>>
----

你可以通过调用 ``qProgToQASM(QProg &)`` 接口来调用该功能,该接口说明如下：

    .. cpp:function:: qProgToQASM(QProg &)

       **功能**
        - 将量子程序转化为QASM

       **参数**
        - 待转化的量子程序

       **返回值**
        - QASM指令集

使用例程
>>>>>>>>
----

下面的例程通过简单的接口调用演示了量子程序转化QASM指令集的过程

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            init(QuantumMachine_type::CPU);

            auto qubit = qAllocMany(6);
            auto cbit  = cAllocMany(2);     
            auto prog = CreateEmptyQProg();

            prog << CZ(qubit[0], qubit[2]) << H(qubit[1]) << CNOT(qubit[1], qubit[2]) 
                 << RX(qubit[0],pi/2) << Measure(qubit[1],cbit[1]);

            std::cout << qProgToQASM(prog) << std::endl;

            finalize();
            return 0;
        }


具体步骤如下:

 - 首先在主程序中用 ``init()`` 进行全局初始化

 - 接着用 ``qAllocMany()`` 和 ``cAllocMany()`` 初始化量子比特与经典寄存器数目

 - 然后调用 ``CreateEmptyQProg()`` 构建量子程序

 - 最后调用接口 ``qProgToQASM(QProg &)`` 输出QASM指令集并用 ``finalize()`` 释放系统资源
