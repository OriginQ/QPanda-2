/// NoiseQVM test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA


TEST(NosieQVMCase,test) {
    NoiseQVM qvm;
    qvm.init();
    auto q = qvm.qAllocMany(4);
    auto c = qvm.cAllocMany(4);

    qvm.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.1);
    QVec qv0 = { q[0], q[1] };
    qvm.set_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::HADAMARD_GATE, 0.1, qv0);
    std::vector<QVec> qves = { {q[0], q[1]}, {q[1], q[2]} };
    qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1, qves);

    double f0 = 0.9;
    double f1 = 0.85;
    qvm.set_readout_error({ {f0, 1 - f0}, {1 - f1, f1} });
    qvm.set_rotation_error(0.05);

    QProg prog;
    prog << X(q[0]) << H(q[0])
        << CNOT(q[0], q[1])
        << CNOT(q[1], q[2])
        << CNOT(q[2], q[3])
        << MeasureAll(q, c);

    auto result = qvm.runWithConfiguration(prog, c, 1000);
    for (auto& item : result)
    {
       // cout << item.first << " : " << item.second << endl;
    }

}