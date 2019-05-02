.. _QProgStored:

量子程序序列化
==========================

简介
--------------
定义一种协议将量子程序序列化为二进制数据，方便量子程序的存储与传输。

接口介绍
--------------

``QProgStored`` 类是QPanda2提供的一个将量子程序序列化的工具类，我们先用QPanda2构建一个量子程序：

    .. code-block:: c
          
        auto qubits = qvm_store->allocateQubits(4);
        auto cbits = qvm_store->allocateCBits(4);

        QProg prog;
        prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
                << CNOT(qubits[1], qubits[2])
                << CNOT(qubits[2], qubits[3])
                ;
然后调用 ``QProgStored`` 类实现序列化

    .. code-block:: c
          
        QProgStored t(qvm);
        t.transform(prog);
        auto instructions = t.getInsturctions();

QPanda2还提供了封装好的接口来实现量子程序序列化，上述的转化代码也可以修改为：

    .. code-block:: c
          
        auto instructions = transformQProgToBinary(prog, qvm);

此外，QPanda2还提供了将序列化后的量子程序存储到文件中的方法， 现在讲上述量子程序以二进制的方式存储到 ``QProg.dat`` 文件中， 可以调用 ``QProgStored``
类中的方法

    .. code-block:: c
          
        QProgStored storeProg(qm);
        storeProg.transform(prog);
        storeProg.store("QProg.dat");

我们还可以使用QPanda2封装的一个接口：

    .. code-block:: c
          
        storeQProgInBinary(prog, qvm, "QProg.dat");

实例
--------------

    .. code-block:: c
    
        #include <QPanda.h>
        #include <Core/Utilities/base64.hpp>
        USING_QPANDA

        int main(void)
        {
            auto qvm_store = initQuantumMachine();
            auto qubits = qvm_store->allocateQubits(4);
            auto cbits = qvm_store->allocateCBits(4);
            cbits[0].setValue(0);

            QProg prog;
            prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
                    << CNOT(qubits[1], qubits[2])
                    << CNOT(qubits[2], qubits[3])
                    ;
            auto data = transformQProgToBinary(prog, qvm_store);
            auto base64_data = Base64::encode(data.data(), data.size()); // 将得到的二进制数据以base64的方式编码
            std::string data_str(base64_data.begin(), base64_data.end());
            std::cout << data_str << std::endl;

            qvm_store->finalize();
            delete qvm_store;
            return 0;
        }
        
运行结果：

    .. code-block:: c

        AAAAAAQAAAAEAAAABAAAAA4AAQAAAAAAJAACAAAAAQAkAAMAAQACACQABAACAAMA    
    
.. note:: 二进制数据不能直接输出，以base64的编码格式编码，得到相应的字符串
