量子虚拟机
=================

在真正的量子计算机没有成型之前，需要使用量子虚拟机承担量子算法，量子应用的验证的问题，QPanda-2的量子虚拟机是全振幅量子虚拟机，其模拟的量子比特数跟用户机器配置息息相关，所以用户可根据自己的机器配置申请量子比特数。用户可以C和C++两种方式使用量子虚拟机。QPanda-2计算方式支持CPU和GPU，可以通过初始化来配置。

接口介绍
--------------

QuantumMachine_type
````````````````````````
.. code-block:: c

    enum QuantumMachine_type {
        CPU,
        GPU,
        CPU_SINGLE_THREAD
    };

C 模式的接口类型
```````````````````

.. cpp:function:: bool init(QuantumMachine_type type = CPU)

    **功能**
        初始化虚拟机
    **参数**
        - type 选择量子虚拟机的类型 QuantumMachine_type_
    **返回值**
        是否初始化成功

.. _qAlloc:
.. cpp:function:: Qubit* qAlloc()

    **功能**
        申请一个量子比特
    **参数**
        - 无
    **返回值**
        量子比特


.. cpp:function:: Qubit* qAlloc(size_t stQubitAddr)

    **功能**
        在指定位置申请一个量子比特
    **参数**
        无
    **返回值**
        量子比特

.. _qAllocMany:
.. cpp:function:: QVec qAllocMany(size_t stQubitNumber)

    **功能**
        申请多个量子比特
    **参数**
        - qubit_count 量子比特个数
    **返回值**
        量子比特容器

.. _cAlloc:
.. cpp:function:: ClassicalCondition cAlloc()

    **功能**
        申请一个量子表达式
    **参数**
        无
    **返回值**
        量子表达式

.. cpp:function:: ClassicalCondition cAlloc(size_t stCBitaddr)

    **功能**
        在指定位置申请一个量子表达式
    **参数**
        无
    **返回值**
        量子表达式

.. _cAllocMany:
.. cpp:function:: std::vector<ClassicalCondition> cAllocMany(size_t stCBitNumber)

    **功能**
        申请多个量子表达式
    **参数**
        - cbit_count 量子表达式个数
    **返回值**
        量子表达式容器

.. cpp:function:: void load(QProg& q)

    **功能**
        加载量子程序
    **参数**
        - prog 量子程序
    **返回值**
        无


.. cpp:function:: void append(QProg& q)

    **功能**
        追加量子程序
    **参数**
        - prog 量子程序
    **返回值**
        无

.. cpp:function:: void run()

    **功能**
        运行量子程序
    **参数**
        无
    **返回值**
        无

.. cpp:function:: void finalize()

    **功能**
        释放资源，与 init_ 配对使用
    **参数**
        无
    **返回值**
        无

.. _getResultMap:
.. cpp:function:: std::map<std::string, bool> getResultMap()

    **功能**
        获得量子程序运行结果
    **参数**
        无
    **返回值**
        经典寄存器地址及其存储的测量量子比特的结果

实例
>>>>>>>>>>>>>>>

    .. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            init(QuantumMachine_type::CPU);  // 初始化量子虚拟机
            auto c = cAllocMany(2);          // 申请经典寄存器
            auto q = qAllocMany(2);          // 申请量子比特

            QProg prog;
            prog << H(q[0])
                << H(q[1])
                << Measure(q[0],c[0])
                << Measure(q[1],c[1]);

            auto result = runWithConfiguration(prog,c,100);
            for(auto & aiter : result)
            {
                std::cout << aiter.first << " : " << aiter.second << std::endl;
            }

            finalize();                     // 释放量子虚拟机
            return 0;
        }

C++ 模式的接口类型
`````````````````````

.. cpp:class:: OriginQVM

    该类的功能是量子虚拟机的构建和使用。

    .. cpp:function:: bool init(QuantumMachine_type type = CPU)

        初始化量子虚拟机， 参照 init_

    .. cpp:function:: Qubit* Allocate_Qubit()

        申请一个量子比特， 参照 qAlloc_

    .. cpp:function:: Qubit* Allocate_Qubit(size_t qubit_num)

        在指定位置申请一个量子比特， 参照 qAlloc_

    .. cpp:function:: QVec Allocate_Qubits(size_t qubit_count)

        申请多个量子比特， 参照 qAllocMany_

    .. cpp:function:: ClassicalCondition Allocate_CBit()

        申请一个量子表达式， 参照 cAlloc_

    .. cpp:function:: ClassicalCondition Allocate_CBit(size_t stCbitNum)

        在指定位置申请一个量子表达式， 参照 cAlloc_

    .. cpp:function:: std::vector<ClassicalCondition> Allocate_CBits(size_t cbit_count)

        申请多个量子表达式， 参照 cAllocMany_

    .. cpp:function:: void load(QProg &prog)

        加载量子程序， 参照 load_

    .. cpp:function:: void append(QProg& prog)

        追加量子程序， 参照 append_

    .. cpp:function:: void run()

        运行量子程序， 参照 run_

    .. cpp:function:: void finalize()

        释放资源， 参照 finalize_

    .. cpp:function:: std::map<std::string, bool> getResultMap()

        获得量子程序运行结果， 参照 getResultMap_

实例
>>>>>>>>>>>>>>>>>>>>

.. code-block:: c

        #include "QPanda.h"
        USING_QPANDA

        int main(void)
        {
            auto qvm = initQuantumMachine(QuantumMachine_type::CPU);  // 初始化量子虚拟机
            auto cbits = qvm->Allocate_CBits(2); // 申请经典寄存器
            auto qvec = qvm->Allocate_Qubits(2); // 申请量子比特

            QProg prog;
            prog << H(qvec[0]) << H(qvec[1])
                    << Measure(qvec[0],cbits[0])
                    << Measure(qvec[1],cbits[1]);

            auto result =qvm-> runWithConfiguration(prog, cbits, 100);
            for(auto & aiter : result)
            {
                std::cout << aiter.first << " : " << aiter.second << std::endl;
            }

            qvm->finalize();
            return 0;
        }