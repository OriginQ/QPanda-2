量子程序转化QASM
=====================
----

通过该功能模块，你可以解析通过QPanda2构建的量子程序，将其中包含的量子比特信息以及量子逻辑门操作信息提取出来，得到按固定格式存储的QASM指令集。

.. _QASM介绍:
.. _IBM Q Experience量子云平台: https://quantumexperience.ng.bluemix.net/qx/editor

QASM介绍
>>>>>>>>>>>>>>>
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

QPanda2提供了QASM转换工具接口 ``std::string transformQProgToQASM(QProg &, QuantumMachine*)`` 该接口使用非常简单，具体可参考下方示例程序。

实例
>>>>>>>>>>>>>>
----

下面的例程通过简单的接口调用演示了量子程序转化QASM指令集的过程

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

            cout << transformQProgToQASM(prog,qvm);

            qvm->finalize();
            delete qvm;
            return 0;
        }

具体步骤如下:

 - 首先在主程序中用 ``initQuantumMachine()`` 初始化一个量子虚拟机对象，用于管理后续一系列行为

 - 接着用 ``allocateQubits()`` 和 ``allocateCBits()`` 初始化量子比特与经典寄存器数目

 - 然后调用 ``CreateEmptyQProg()`` 构建量子程序

 - 最后调用接口 ``transformQProgToQASM`` 输出QASM指令集并用 ``finalize()`` 释放系统资源


运行结果如下：

    .. code-block:: c

        openqasm 2.0;
        qreg q[6];
        creg c[6];
        h q[1];
        x q[2];
        hdg q[1];
        rx(0.636620) q[1];
        hdg q[2];
        ydg q[2];
        cr(1.570796) q[1],q[2];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        measure q[3] -> c[3];
        measure q[4] -> c[4];
        measure q[5] -> c[5];