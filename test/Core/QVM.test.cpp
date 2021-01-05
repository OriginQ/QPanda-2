#include <time.h>
#include <iostream>
#include <numeric>
#include "QPanda.h"
#include <functional>
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/OriginCollection.h"
USING_QPANDA
using namespace std;

TEST(CPUQVMTest, testInit)
{
    return;
    CPUQVM qvm;
    ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
    ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);

    qvm.init();
    ASSERT_NO_THROW(auto qvec = qvm.allocateQubits(2));
    ASSERT_NO_THROW(auto cvec = qvm.allocateCBits(2));

    ASSERT_THROW(auto qvec = qvm.allocateQubits(26), qalloc_fail);
    ASSERT_THROW(auto cvec = qvm.allocateCBits(257), calloc_fail); 

    qvm.finalize();
    ASSERT_THROW(auto qvec = qvm.allocateQubits(2), qvm_attributes_error);
    ASSERT_THROW(auto cvec = qvm.allocateCBits(2), qvm_attributes_error);
    ASSERT_THROW(auto qvec = qvm.getAllocateQubit(), qvm_attributes_error);
    ASSERT_THROW(auto qvec = qvm.getAllocateCMem(), qvm_attributes_error);
    ASSERT_THROW(auto qvec = qvm.getResultMap(), qvm_attributes_error);
}

TEST(NoiseMachineTest, test)
{
    //return;
    rapidjson::Document doc;
    doc.Parse("{}");
    Value value(rapidjson::kObjectType);
    Value value_h(rapidjson::kArrayType);
    value_h.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
    value_h.PushBack(0.5, doc.GetAllocator());
    value.AddMember("H", value_h, doc.GetAllocator());

    Value value_rz(rapidjson::kArrayType);
    value_rz.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
    value_rz.PushBack(0.5, doc.GetAllocator());
    value.AddMember("RZ", value_rz, doc.GetAllocator());

    Value value_cnot(rapidjson::kArrayType);
    value_cnot.PushBack(DAMPING_KRAUS_OPERATOR, doc.GetAllocator());
    value_cnot.PushBack(0.5, doc.GetAllocator());
    value.AddMember("CPHASE", value_cnot, doc.GetAllocator());
    doc.AddMember("noisemodel", value, doc.GetAllocator());

    NoiseQVM qvm;
    qvm.init(doc);
    auto qvec = qvm.allocateQubits(16);
    auto cvec = qvm.allocateCBits(16);
    auto prog = QProg();

    QCircuit  qft = CreateEmptyCircuit();
    for (auto i = 0; i < qvec.size(); i++)
    {
        qft << H(qvec[qvec.size() - 1 - i]);
        for (auto j = i + 1; j < qvec.size(); j++)
        {
            qft << CR(qvec[qvec.size() - 1 - j], qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
        }
    }

    prog << qft << qft.dagger()
        << MeasureAll(qvec,cvec);

    rapidjson::Document doc1;
    doc1.Parse("{}");
    auto &alloc = doc1.GetAllocator();
    doc1.AddMember("shots",10 , alloc);

    clock_t start = clock();
    auto result = qvm.runWithConfiguration(prog, cvec, doc1);
    clock_t end = clock();
    std::cout << end - start << endl;

    for (auto &aiter : result)
    {
        std::cout << aiter.first << " : " << aiter.second << endl;
    }
    //auto state = qvm.getQState();
    //for (auto &aiter : state)
    //{
    //    std::cout << aiter << endl;
    //}
    qvm.finalize();

    getchar();
}

double getStateProb(complex<double> val)
{
    return val.real()*val.real() + val.imag()*val.imag();
}

TEST(QVM, PartialAmplitudeQVM)
{
    auto machine = new PartialAmplitudeQVM();
    machine->init();
    auto qlist = machine->allocateQubits(40);
    auto clist = machine->allocateCBits(40);
    auto Toffoli = X(qlist[20]);
    Toffoli.setControl({ qlist[18], qlist[19] });

    auto prog = QProg();
    prog << H(qlist[18])
         << X(qlist[19])
         << Toffoli;

	machine->run(prog);

    std::vector<string> subSet = { "0000000000000000000001000000000000000000" ,
                                   "0000000000000000000010000000000000000000" ,
                                   "0000000000000000000011000000000000000000" ,
                                   "0000000000000000000100000000000000000000" ,
                                   "0000000000000000000101000000000000000000" ,
                                   "0000000000000000000110000000000000000000" ,
                                   "0000000000000000000111000000000000000000" ,
                                   "1000000000000000000000000000000000000000" };
    auto result = machine->PMeasure_subset(subSet);

    for (auto val : result)
    {
        std::cout << val.first << " : " << val.second << std::endl;
    }
    getchar();
}

TEST(QVM, SingleAmplitudeQVM)
{
    //throw exception();
    auto machine = new SingleAmplitudeQVM();
    machine->init();
    auto qlist = machine->allocateQubits(10);
    auto clist = machine->allocateCBits(10);

    auto prog = QProg();
    for_each(qlist.begin(), qlist.end(), [&](Qubit *val) { prog << H(val); });
    prog << CZ(qlist[1], qlist[5])
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
        << CR(qlist[7], qlist[8], PI);

    //machine->run(prog);

    cout << machine->PMeasure_bin_index(prog, "0000000000") << endl;
    cout << machine->PMeasure_dec_index(prog, "1") << endl;

    machine->finalize();
    getchar();
}


void GHZ(int a)
{
    CPUQVM qvm;
    qvm.setConfigure({ 64,64 });
    qvm.init();

    auto q = qvm.qAllocMany(a);
    auto c = qvm.cAllocMany(a);

    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < a - 1; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }
    prog << MeasureAll(q, c);


    const string ss = "GHZ_"+to_string(a);
    write_to_originir_file(prog, &qvm, ss);
}

QStat run_origin_circuit()
{
    CPUQVM cpu;
    cpu.setConfigure({ 64,64 });
    cpu.init();

    auto q = cpu.qAllocMany(3);
    auto c = cpu.cAllocMany(3);

    QProg prog;
    prog << H(q[0])
        << H(q[1])
        << H(q[2])

        << RX(q[0], PI / 6)
        << RY(q[1], PI / 3)
        << RX(q[2], PI / 6)

        << CNOT(q[0], q[1])
        << H(q[2])

        << RY(q[0], PI / 4)
        << RZ(q[1], PI / 4)

        << RX(q[0], PI / 6)
        << CR(q[2], q[1], PI / 6)

        << RY(q[1], PI / 6)
        << RY(q[2], PI / 3);

    QProg prog1;
    prog1 << H(q[0])
        << CNOT(q[0], q[1])
        << CNOT(q[0], q[2])
        << CNOT(q[1], q[2])
        << CZ(q[1], q[2]);

    cpu.directlyRun(prog);
    return cpu.getQState();
}

const double _SQ2 = 1 / 1.4142135623731;
const double _PI = 3.14159265358979;

const QStat mi{ 1., 0., 0., 1. };
const QStat mz{ 1., 0., 0., -1. };
const QStat mx{ 0., 1., 1., 0. };
const QStat my{ 0., qcomplex_t(0., -1.), qcomplex_t(0., 1.), 0. };
const QStat mh{ _SQ2, _SQ2, _SQ2, -_SQ2 };
const QStat mt{ 1, 0, 0, qcomplex_t(_SQ2, _SQ2) };

QStat m10 = mi * mz;
QStat m11 = mi;

//QStat m20 = mi * (-1) * mz;
QStat m20 = mi * mz;
QStat m21 = mx;

QStat m30 = mx;
QStat m31 = mh * mh * mx;

QStat m40 = my;
QStat m41 = mh * mt * mh * mt * mx;


QStat run_subset_circuit10()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << H(q[0])
        << H(q[1])

        << RX(q[0], PI / 6)
        << RY(q[1], PI / 3)

        << CNOT(q[0], q[1])

        << RY(q[0], PI / 4)
        << U4(m10, q[1])

        << RX(q[0], PI / 6);

    cpu.directlyRun(prog);
    return cpu.getQState();
}

QStat run_subset_circuit20()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << H(q[0])
        << H(q[1])

        << RX(q[0], PI / 6)
        << RY(q[1], PI / 3)

        << CNOT(q[0], q[1])

        << RY(q[0], PI / 4)
        << U4(m20, q[1])

        << RX(q[0], PI / 6);


    cpu.directlyRun(prog);
    return cpu.getQState();
}

QStat run_subset_circuit30()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << H(q[0])
        << H(q[1])

        << RX(q[0], PI / 6)
        << RY(q[1], PI / 3)

        << CNOT(q[0], q[1])

        << RY(q[0], PI / 4)
        << U4(m30, q[1])

        << RX(q[0], PI / 6);


    cpu.directlyRun(prog);
    return cpu.getQState();
}

QStat run_subset_circuit40()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << H(q[0])
        << H(q[1])

        << RX(q[0], PI / 6)
        << RY(q[1], PI / 3)

        << CNOT(q[0], q[1])

        << RY(q[0], PI / 4)
        << U4(m40, q[1])

        << RX(q[0], PI / 6);


    cpu.directlyRun(prog);
    return cpu.getQState();
}

QStat run_subset_circuit11()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << U4(m11, q[0])
        << H(q[1])

        << RX(q[1], PI / 6)

        << RZ(q[0], PI / 4)
        << H(q[1])

        << CR(q[1], q[0], PI / 6)

        << RY(q[0], PI / 6)
        << RY(q[1], PI / 3);

    cpu.directlyRun(prog);
    return cpu.getQState();
}

QStat run_subset_circuit21()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << U4(m21, q[0])
        << H(q[1])

        << RX(q[1], PI / 6)

        << RZ(q[0], PI / 4)
        << H(q[1])

        << CR(q[1], q[0], PI / 6)

        << RY(q[0], PI / 6)
        << RY(q[1], PI / 3);

    cpu.directlyRun(prog);
    return cpu.getQState();
}

QStat run_subset_circuit31()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << U4(m31, q[0])
        << H(q[1])

        << RX(q[1], PI / 6)

        << RZ(q[0], PI / 4)
        << H(q[1])

        << CR(q[1], q[0], PI / 6)

        << RY(q[0], PI / 6)
        << RY(q[1], PI / 3);

    cpu.directlyRun(prog);
    return cpu.getQState();
}

QStat run_subset_circuit41()
{
    CPUQVM cpu;
    cpu.init();

    auto q = cpu.qAllocMany(2);
    auto c = cpu.cAllocMany(2);

    QProg prog;
    prog << U4(m41, q[0])
        << H(q[1])

        << RX(q[1], PI / 6)

        << RZ(q[0], PI / 4)
        << H(q[1])

        << CR(q[1], q[0], PI / 6)

        << RY(q[0], PI / 6)
        << RY(q[1], PI / 3);

    cpu.directlyRun(prog);
    return cpu.getQState();
} 

TEST(QVM, MPSQVM)
{
    //auto result = run_origin_circuit();

    //auto result10 = run_subset_circuit10();
    //auto result11 = run_subset_circuit11();

    //auto result20 = run_subset_circuit20();
    //auto result21 = run_subset_circuit21();

    //auto result30 = run_subset_circuit30();
    //auto result31 = run_subset_circuit31();

    //auto result40 = run_subset_circuit40();
    //auto result41 = run_subset_circuit41();

    ////000
    //auto c1 = result10[0] * result11[0] +
    //    result20[0] * result21[0] +
    //    result30[0] * result31[0] +
    //    result40[0] * result41[0];

    ////001
    //auto c2 = result10[0] * result11[1] +
    //    result20[0] * result21[1] +
    //    result30[0] * result31[1] +
    //    result40[0] * result41[1];

    ////010
    //auto c3 = result10[1] * result11[2] +
    //    result20[1] * result21[2] +
    //    result30[1] * result31[2] +
    //    result40[1] * result41[2];

    ////011
    //auto c4 = result10[1] * result11[3] +
    //    result20[1] * result21[3] +
    //    result30[1] * result31[3] +
    //    result40[1] * result41[3];

    ////100
    //auto c5 = result10[2] * result11[0] +
    //    result20[2] * result21[0] +
    //    result30[2] * result31[0] +
    //    result40[2] * result41[0];

    ////101
    //auto c6 = result10[2] * result11[1] +
    //    result20[2] * result21[1] +
    //    result30[2] * result31[1] +
    //    result40[2] * result41[1];

    ////110
    //auto c7 = result10[3] * result11[2] +
    //    result20[3] * result21[2] +
    //    result30[3] * result31[2] +
    //    result40[3] * result41[2];

    ////111
    //auto c8 = result10[3] * result11[3] +
    //    result20[3] * result21[3] +
    //    result30[3] * result31[3] +
    //    result40[3] * result41[3];

    //QStat sub_result = { c1,c2,c3,c4,c5,c6,c7,c8 };

    //prob_vec sub_probs;
    //prob_vec ori_probs;
    //for (auto val : sub_result)
    //{
    //    sub_probs.emplace_back(std::norm(val));
    //}

    //for (auto val : result)
    //{
    //    ori_probs.emplace_back(std::norm(val));
    //}

    //auto sum_sub_val = std::accumulate(sub_probs.begin(), sub_probs.end(), 0.);
    //auto sum_ori_val = std::accumulate(ori_probs.begin(), ori_probs.end(), 0.);
    //
    //std::cout << "origin result | sub_result : " << endl;
    //for (auto i = 0; i < result.size(); ++i)
    //{
    //    std::cout << result[i] << " | " << sub_result[i] / 2. << endl;
    //}

    //return;
    MPSQVM mps;
    mps.setConfigure({ 64,64 });
    mps.init();

    auto q = mps.qAllocMany(6);
    auto c = mps.cAllocMany(6);

    //QProg prog;
    //prog << H(q[0])
    //     << CNOT(q[0], q[1])
    //     << CNOT(q[0], q[2])
    //     << CNOT(q[0], q[2]);
    //<< CNOT(q[0], q[2]);
    //<< CNOT(q[0], q [2]);
        //<< CR(q[0], q[1], PI / 3)
        //<< CR(q[1], q[2], PI / 4)
        //<< CR(q[0], q[2], PI / 6);

    QProg prog;
    prog << X(q[0])
         << X(q[1])
         << CNOT(q[0], q[1])
         << MeasureAll(q,c);

    auto q0 = { q[0] };
    auto q1 = { q[1] };
    std::vector<QVec> qs = {{ q[0],q[1] }};

    //mps.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, CNOT_GATE, 0.5, qs);
    //mps.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, PAULI_X_GATE, 0.9999, q0);

    mps.set_measure_error(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, 0.9999, q1);
    mps.set_measure_error(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, 0.0001, q0);
    QStat id = { 1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                0,0,0,1 };

    QStat _CNOT = { 1,0,0,0,
            0,1,0,0,
            0,0,0,1,
            0,0,1,0 };

    //mps.set_mixed_unitary_error(GateType::CNOT_GATE, { _CNOT,id }, { 0.5, 0.5 });

    auto a = mps.runWithConfiguration(prog, c, 1000);

    for (auto val : a)
    {
        cout << val.first << " : " << val.second << endl;
    }

    mps.finalize();
    getchar();
}
