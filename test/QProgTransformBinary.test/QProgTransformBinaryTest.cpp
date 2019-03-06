#include <iostream>
#include "QPanda.h"
#include <iostream>
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
    qProgBinaryStored(qubits.size(), cbits.size(), prog);
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
}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}