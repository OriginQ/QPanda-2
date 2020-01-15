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

    cbits[0].setValue(0);

    QProg prog;
    prog << H(qubits[3]);

    std::cout << transformQProgToOriginIR(prog,qvm) << std::endl;;

    directlyRun(prog);

    auto result = getProbTupleList(qubits);
    for (auto &val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }
    storeQProgInBinary(prog, qvm, "QProg.dat");
    destroyQuantumMachine(qvm);
    return;
}

TEST(QProgTransformBinaryParse, QBinaryParse)
{
    auto qvm = initQuantumMachine();
    QProg parseProg;
    QVec qubits;
    std::vector<ClassicalCondition> cbits;

    binaryQProgFileParse(qvm, "QProg.dat", qubits, cbits, parseProg);
    std::cout << "Parse" << std::endl;
    std::cout << transformQProgToOriginIR(parseProg, qvm) << std::endl;
    directlyRun(parseProg);

    auto result = getProbTupleList(qubits);
    for (auto &val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }

    destroyQuantumMachine(qvm);

    return;
}


TEST(QProgTransformBinaryData, QBinaryData)
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

    auto result = probRunTupleList(prog, qubits);
    for (auto &val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }
    destroyQuantumMachine(qvm_store);

    auto qvm = initQuantumMachine();
    QProg parseProg;
    QVec qubits_parse;
    std::vector<ClassicalCondition> cbits_parse;

    binaryQProgDataParse(qvm, data, qubits_parse, cbits_parse, parseProg);
    std::cout << "binary data Parse" << std::endl;
    //std::cout << transformToQRunes(parseProg) << std::endl;

    auto result_parse = probRunTupleList(parseProg, qubits_parse);
    for (auto &val : result_parse)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }

    destroyQuantumMachine(qvm);

    return;
}



