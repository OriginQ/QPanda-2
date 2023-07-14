/// ProbMeasure class inface test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

TEST(ProbMeasureCase,test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto qubits = qvm.qAllocMany(2);

    QProg prog;
    prog << H(qubits[0])
        << CNOT(qubits[0], qubits[1]);

    //std::cout << "probRunDict: " << std::endl;
    auto result1 = qvm.probRunDict(prog, qubits);
    for (auto& val : result1)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }

    //std::cout << "probRunTupleList: " << std::endl;
    auto result2 = qvm.probRunTupleList(prog, qubits);
    for (auto& val : result2)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }

    //std::cout << "probRunList: " << std::endl;
    auto result3 = qvm.probRunList(prog, qubits);
    for (auto& val : result3)
    {
       // std::cout << val << std::endl;
    }


}