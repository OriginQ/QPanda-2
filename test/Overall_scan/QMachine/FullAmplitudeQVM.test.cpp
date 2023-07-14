/// AmplitudeQVM Inface test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA


TEST(AmplitudeQVMCase,test) {
    CPUQVM qvm;
    qvm.init();
    auto qubits = qvm.qAllocMany(4);
    auto cbits = qvm.cAllocMany(4);

  
    QProg prog;
    prog << H(qubits[0])
        << CNOT(qubits[0], qubits[1])
        << CNOT(qubits[1], qubits[2])
        << CNOT(qubits[2], qubits[3])
        << Measure(qubits[0], cbits[0]);

    auto result = qvm.runWithConfiguration(prog, cbits, 1000);

    for (auto& val : result)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }

    auto result1 = directlyRun(prog);
    for (auto& val : result1)
    {
        //std::cout << val.first << ", " << val.second << std::endl;
    }
    qvm.finalize();
}