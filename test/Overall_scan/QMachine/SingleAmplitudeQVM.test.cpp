/// SingleAmplitudeQVM Inface test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

TEST(SingleAmplitudeQVMCase,test) {
	SingleAmplitudeQVM saqvm;
	saqvm.init();
	auto cbits = saqvm.allocateCBits(10);
	auto qv = saqvm.allocateQubits(10);

    auto prog = QProg();
    for_each(qv.begin(), qv.end(), [&](Qubit* val) { prog << H(val); });
    prog << CZ(qv[1], qv[5])
        << CZ(qv[3], qv[5])
        << CZ(qv[2], qv[4])
        << CZ(qv[3], qv[7])
        << CZ(qv[0], qv[4])
        << RY(qv[7], PI / 2)
        << RX(qv[8], PI / 2)
        << RX(qv[9], PI / 2)
        << CR(qv[0], qv[1], PI)
        << CR(qv[2], qv[3], PI)
        << RY(qv[4], PI / 2)
        << RZ(qv[5], PI / 4)
        << RX(qv[6], PI / 2)
        << RZ(qv[7], PI / 4)
        << CR(qv[8], qv[9], PI)
        << CR(qv[1], qv[2], PI)
        << RY(qv[3], PI / 2)
        << RX(qv[4], PI / 2)
        << RX(qv[5], PI / 2)
        << CR(qv[9], qv[1], PI)
        << RY(qv[1], PI / 2)
        << RY(qv[2], PI / 2)
        << RZ(qv[3], PI / 4)
        << CR(qv[7], qv[8], PI);

    
    saqvm.run(prog, qv);
   // std::cout << saqvm.pMeasureDecindex("1") << std::endl;

    saqvm.run(prog, qv);
    //std::cout << saqvm.pMeasureBinindex("0000000010") << std::endl;

    saqvm.run(prog, qv);
    auto res_1 = saqvm.getProbDict(qv);

    auto res2 = saqvm.probRunDict(prog, qv);
    for (auto& val : res_1)
    {
       // std::cout << val.first << " : " << val.second << std::endl;
    }

    // no have directlyRun
    // auto res = saqvm.directlyRun(prog);
    
}