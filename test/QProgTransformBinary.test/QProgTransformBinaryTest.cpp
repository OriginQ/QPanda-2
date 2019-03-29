#include <iostream>
#include "QPanda.h"
#include "gtest/gtest.h"
#include "QPanda.h"
#include <map>
#include <cstdlib>
#include <sstream>
#include <string>
#include <algorithm>  

USING_QPANDA

TEST(QProgTransformBinaryStore, QBinaryStore)
{
    init();
    auto qubits = qAllocMany(4);
    auto cbits = cAllocMany(4);
    cbits[0].setValue(0);

    QProg prog;
    prog << H(qubits[3]);

    std::cout << qProgToQRunes(prog) << std::endl;;

    directlyRun(prog);

    auto result = getProbTupleList(qubits);
    for (auto &val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }
    storeQProgInBinary(qubits.size(), cbits.size(), prog);
    finalize();
    return;
}

TEST(QProgTransformBinaryParse, QBinaryParse)
{
    init();
    QProg parseProg;
    QVec qubits;
    std::vector<ClassicalCondition> cbits;

    binaryQProgFileParse(qubits, cbits, parseProg);
    std::cout << "Parse" << std::endl;
    std::cout << qProgToQRunes(parseProg) << std::endl;

    directlyRun(parseProg);

    auto result = getProbTupleList(qubits);
    for (auto &val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }

    finalize();

    return;
}


TEST(QProgTransformBinaryData, QBinaryData)
{
    init();
    auto qubits = qAllocMany(4);
    auto cbits = cAllocMany(4);
    cbits[0].setValue(0);

    QProg prog;
    prog << H(qubits[0]) << CNOT(qubits[0], qubits[1])
              << CNOT(qubits[1], qubits[2])
              << CNOT(qubits[2], qubits[3])
              ;
    auto data = getQProgBinaryData(4, 4, prog);

    auto result = probRunTupleList(prog, qubits);
    for (auto &val : result)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }
    finalize();

    init();
    QProg parseProg;
    QVec qubits_parse;
    std::vector<ClassicalCondition> cbits_parse;

    binaryQProgDataParse(qubits_parse, cbits_parse, parseProg, data);
    std::cout << "binary data Parse" << std::endl;
    //std::cout << qProgToQRunes(parseProg) << std::endl;

    auto result_parse = probRunTupleList(parseProg, qubits_parse);
    for (auto &val : result_parse)
    {
        std::cout << val.first << ", " << val.second << std::endl;
    }

    finalize();

    return;
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
