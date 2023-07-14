#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA

TEST(drawAndlayerPro,test) {
    auto qvm = CPUQVM();
    qvm.init();
    auto q = qvm.qAllocMany(3);
    auto c = qvm.cAllocMany(3);
    QProg prog;
    QCircuit cir1, cir2;

    auto gate = S(q[1]);
   // gate.setDagger(true);
    cir1 << H(q[0]) << S(q[1]) << CNOT(q[0], q[1]) << CZ(q[1], q[2]);
    //cir1.setDagger(true);
    cir2 << cir1 << CU(1, 2, 3, 4, q[0], q[2]) << S(q[2]) << CR(q[2], q[1], PI / 2);
    //cir2.setDagger(true);
    prog << cir2;// << MeasureAll(q, c);

 

    std::cout << prog << std::endl;

    auto layerinfo = circuit_layer(prog);

    std::cout << " circuit layer " << layerinfo.first << std::endl;
    

    for (auto& ve : layerinfo.second) {
        for (int i = 0; i < ve.size(); ++i) {
            std::cout << ve[i].m_gate_type << " ";
        }
        std::cout << std::endl;
    }
   
}