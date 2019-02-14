统计量子程序时钟周期
=======================

简介
--------------

已知每个量子逻辑门在运行时所需时间的条件下，估算一个量子程序运行所需要的时间。每个量子逻辑门的时间设置在项目的元数据配置文件 ``MetadataConfig.xml`` 中，如果未设置则会给定一个默认值，单量子门的默认时间为2，双量子门的时间为5。

配置文件可仿照下面设置
***********************

.. code-block:: xml

    <SingleGate>
        <Gate time = "2">rx</Gate>
        <Gate time = "2">Ry</Gate>
        <Gate time = "2">RZ</Gate>
        <Gate time = "2">S</Gate>
        <Gate time = "2">H</Gate>
        <Gate time = "2">X1</Gate>
    </SingleGate>
    <DoubleGate>
        <Gate time = "5">CNOT</Gate>
        <Gate time = "5">CZ</Gate>
        <Gate time = "5">ISWAP</Gate>
    </DoubleGate>

接口介绍
--------------

.. cpp:function:: size_t getQProgClockCycle(QProg &prog)
    
    **功能**
        统计量子程序的时钟周期。
    **参数**  
        - prog 量子程序
    **返回值** 
        量子程序的时钟周期。    

实例
--------------

    .. code-block:: c++
    
        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            init();
            auto qubits = qAllocMany(4);
            auto prog = CreateEmptyQProg();
            prog << H(qubits[0]) << CNOT(qubits[0], qubits[1]) 
                 << iSWAP(qubits[1], qubits[2]) << RX(qubits[3], PI/4);
            auto time = getQProgClockCycle(prog);
            std::cout << "clockCycle : " << time << std::endl;

            finalize();
            return 0;
        }
    
