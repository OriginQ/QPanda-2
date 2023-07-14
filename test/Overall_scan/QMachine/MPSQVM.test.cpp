/// MPSQVM test

#include "QPanda.h"
#include "gtest/gtest.h"

USING_QPANDA


TEST(MPSQVMCase,test) {
    MPSQVM qvm;

    qvm.init();
    auto qlist = qvm.qAllocMany(10);
    auto clist = qvm.cAllocMany(10);

    QProg prog;
    prog << HadamardQCircuit(qlist)
        << CZ(qlist[1], qlist[5])
        << CZ(qlist[3], qlist[5])
        << CZ(qlist[2], qlist[4])
        << CZ(qlist[3], qlist[7])
        << CZ(qlist[0], qlist[4])
        << RY(qlist[7], PI / 2)
        << RX(qlist[8], PI / 2)
        << RX(qlist[9], PI / 2)
        << CR(qlist[0], qlist[1], PI)
        << CR(qlist[2], qlist[3], PI)
        << RY(qlist[4], PI / 2)
        << RZ(qlist[5], PI / 4)
        << RX(qlist[6], PI / 2)
        << RZ(qlist[7], PI / 4)
        << CR(qlist[8], qlist[9], PI)
        << CR(qlist[1], qlist[2], PI)
        << RY(qlist[3], PI / 2)
        << RX(qlist[4], PI / 2)
        << RX(qlist[5], PI / 2)
        << CR(qlist[9], qlist[1], PI)
        << RY(qlist[1], PI / 2)
        << RY(qlist[2], PI / 2)
        << RZ(qlist[3], PI / 4)
        << CR(qlist[7], qlist[8], PI)
        << MeasureAll(qlist, clist);

    auto measure_result = qvm.runWithConfiguration(prog, clist, 1000);
    for (auto val : measure_result)
    {
       // cout << val.first << " : " << val.second << endl;
    }

    auto pmeasure_result = qvm.probRunDict(prog, qlist, -1);
    for (auto val : pmeasure_result)
    {
        //cout << val.first << " : " << val.second << endl;
    }

    qvm.finalize();
}

TEST(MPS, test)                     
{
    MPSQVM qvm;
    qvm.init();

    auto q = qvm.qAllocMany(2);
    auto c = qvm.cAllocMany(2);

    qvm.add_single_noise_model(NOISE_MODEL::PHASE_DAMPING_OPRATOR, GateType::PAULI_X_GATE, 0.8);
    qvm.add_single_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.8);
    //qvm.add_single_noise_model(NOISE_MODEL::DEPOLARIZING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.99);
    //qvm.add_single_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.49);
    //qvm.add_single_noise_model(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 5, 5, 0.9);
    //qvm.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1);
    //QVec qv0 = { q[0], q[1] };
    //qvm.set_noise_model(NOISE_MODEL::DEPHASING_KRAUS_OPERATOR, GateType::HADAMARD_GATE, 0.1, qv0);
    //std::vector<QVec> qves = { {q[0], q[1]}, {q[1], q[2]} };
    //qvm.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, GateType::CNOT_GATE, 0.1, qves);

    QProg prog;
    //prog << X(q[0]) << MeasureAll(q, c);
    prog << H(q[0]) << X(q[1]) << H(q[1]) << CNOT(q[0], q[1]) << H(q[0]) << Measure(q[0], c[0]);

    //auto noise_ptr = (NoiseQVM*)(&qvm);
    //auto noise_ptr = dynamic_cast<NoiseQVM*>(&qvm);
    //single_qubit_rb(&qvm, q[0], {6}, 1, 100);

    auto result = qvm.runWithConfiguration(prog, c, 10000);
    for (auto& item : result)
    {
        //std::cout << item.first << " : " << item.second << std::endl;
    }

    return;
}