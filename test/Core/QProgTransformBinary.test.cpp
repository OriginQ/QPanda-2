#include <iostream>
#include "QPanda.h"
#include "gtest/gtest.h"
#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>  

USING_QPANDA

TEST(QProgTransformBinaryStore, QBinaryStore)
{
    auto qvm = initQuantumMachine();
    auto qubits = qvm->allocateQubits(4);
    auto cbits = qvm->allocateCBits(4);

    cbits[0].set_val(0);

    QProg prog;
    prog << H(qubits[3]);

    //std::cout << transformQProgToOriginIR(prog,qvm) << std::endl;;
    std::string _excepted = transformQProgToOriginIR(prog, qvm);
    directlyRun(prog);

    auto result = getProbTupleList(qubits);

    //ASSERT_EQ(result.end()->second,0);
    storeQProgInBinary(prog, qvm, "QProg.dat");
    destroyQuantumMachine(qvm);
    const std::string IR = R"(QINIT 4
CREG 4
H q[3])";
    ASSERT_EQ(_excepted, IR);
    return;
}

TEST(QProgTransformBinaryParse, QBinaryParse)
{
    auto qvm = initQuantumMachine();
    QProg parseProg;
    QVec qubits;
    std::vector<ClassicalCondition> cbits;

    binaryQProgFileParse(qvm, "QProg.dat", qubits, cbits, parseProg);
    //std::cout << "Parse" << std::endl;
    //std::cout << transformQProgToOriginIR(parseProg, qvm) << std::endl;
    directlyRun(parseProg);

    auto result = getProbTupleList(qubits);
    /*for (auto &val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }*/
    
    const std::string IR = R"(QINIT 4
CREG 4
H q[3])";
    ASSERT_EQ(transformQProgToOriginIR(parseProg, qvm), IR);
    destroyQuantumMachine(qvm);
    return;
}




