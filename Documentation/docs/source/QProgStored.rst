.. _QProgStored:

量子程序存储为二进制文件
==========================

简介
--------------
由于将两子程序以字符的形式存储会占用很多空间，所以定义一种协议将量子程序以二进制的方式存储

接口介绍
--------------

.. cpp:function:: void qProgBinaryStored(QProg &prog, const std::string &filename = DEF_QPROG_FILENAME)
    
    **功能**
        量子程序存储为二进制文件
    **参数**  
        - prog 量子程序
        - filename 文件名 
    **返回值**
        无  

实例
--------------

    .. code-block:: c
    
        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            init();
            auto qubits = qAllocMany(4);
            auto cbits = cAllocMany(1);
            auto prog = CreateEmptyQProg();
            prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
                << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI/4)
                << Measure(qubits[0], cbits[0]);
            qProgBinaryStored(prog);

            finalize();
            return 0;
        }
    
