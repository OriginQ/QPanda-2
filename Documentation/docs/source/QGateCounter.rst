逻辑门统计
===============

简介
--------------

逻辑门的统计是指统计一个量子线路或量子程序中所有的量子逻辑门个数方法

接口介绍
--------------

.. cpp:function:: size_t countQGateUnderQCircuit(AbstractQuantumCircuit * pQCircuit)
    
    **功能**
        统计量子线路中的量子逻辑门个数
    **参数**  
        - pQCircuit 量子线路指针
    **返回值** 
        量子线路中的量子逻辑门个数    

.. cpp:function:: size_t countQGateUnderQProg(AbstractQuantumProgram * pQProg)

    **功能**
        统计量子程序中的量子逻辑门个数 
    **参数**
        - pQProg 量子程序指针      
    **返回值** 
        量子程序中的量子逻辑门个数 

实例
-------------

    .. code-block:: c
    
        #include <QPanda.h>
        USING_QPANDA

        int main(void)
        {
            init();
            auto qubits = qAllocMany(4);
            auto cbits = cAllocMany(4);

            auto circuit = CreateEmptyCircuit(); 
            circuit << H(qubits[0]) << X(qubits[1]) << S(qubits[2])
                    << CNOT(qubits[0], qubits[1]) << iSWAP(qubits[1], qubits[2])
                    << RX(qubits[3], PI/4);
            auto count = countQGateUnderQCircuit(&circuit);
            std::cout << "QCircuit count: " << count << std::endl;

            auto prog = CreateEmptyQProg();
            prog << Y(qubits[0]) << CZ(qubits[2], qubits[3]) << circuit;
            count = countQGateUnderQProg(&prog); 
            std::cout << "QProg count: " << count << std::endl;

            finalize();
            return 0;
        }
    
