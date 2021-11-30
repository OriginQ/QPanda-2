#include "QPanda.h"
#include "gtest/gtest.h"


static void test1()
{
    NoiseQVM qm;
    qm.init();
    auto q = qm.qAllocMany(3);
    auto c = qm.cAllocMany(3);
    qm.set_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::HADAMARD_GATE, 0.02);
    qm.set_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.05);

    QProg prog;
    prog << H(q[0])
        << CNOT(q[0], q[1])
        << CNOT(q[1], q[2])
        //<< CNOT(q[2], q[3])
        //<< MeasureAll(q, c)
        ;

    auto density = state_tomography_density(prog, q, &qm, 1024);
    for (size_t i = 0; i < density.size(); i++)
    {
        for (auto &value : density[i])
        {
            cout << value << "\t";
        }
        cout << endl;
    }


    return;
}

static void test2()
{


    return;
}
TEST(QuantumStateTomography, test)

{
    test1();
}