#include <atomic>
#include <string>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/RandomCircuit.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrixSimulator.h"

USING_QPANDA

#define CHECK_RUN_TIME_BEGIN                  auto start_time = chrono::system_clock::now()\

#define CHECK_RUN_TIME_END_AND_COUT_SECONDS(argv)   auto final_time = chrono::system_clock::now();\
                                                    auto duration = chrono::duration_cast<chrono::microseconds>(final_time - start_time);\
                                                    std::cout << argv << " -> run_time counts :" << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << std::endl\

using namespace std;
using namespace rapidjson;

static QHamiltonian test_hamiltonian_data()
{
    /*  test data:

        "" : -0.097066,
        "X0 X1 Y2 Y3" : -0.045303,
        "X0 Y1 Y2 X3" : 0.045303,
        "Y0 X1 X2 Y3" : 0.045303,
        "Y0 Y1 X2 X3" : -0.045303,
        "Z0" : 0.171413,
        "Z0 Z1" : 0.168689,
        "Z0 Z2" : 0.120625,
        "Z0 Z3" : 0.165928,
        "Z1" : 0.171413,
        "Z1 Z2" : 0.165928,
        "Z1 Z3" : 0.120625,
        "Z2" : -0.223432,
        "Z2 Z3" : 0.174413,
        "Z3" : -0.223432
    */

    QHamiltonian hamiltonian;

    //"" : -0.097066
    QTerm q0;
    hamiltonian.emplace_back(make_pair(q0, -0.097066));

    // "X0 X1 Y2 Y3" : -0.045303
    QTerm q1;
    q1[0] = 'X';
    q1[1] = 'X';
    q1[2] = 'Y';
    q1[3] = 'Y';
    hamiltonian.emplace_back(make_pair(q1, -0.045303));

    // "X0 Y1 Y2 X3" : 0.045303
    QTerm q2;
    q2[0] = 'X';
    q2[1] = 'Y';
    q2[2] = 'Y';
    q2[3] = 'X';
    hamiltonian.emplace_back(make_pair(q2, 0.045303));

    // "Y0 X1 X2 Y3" : 0.045303
    QTerm q3;
    q3[0] = 'Y';
    q3[1] = 'X';
    q3[2] = 'X';
    q3[3] = 'Y';
    hamiltonian.emplace_back(make_pair(q3, 0.045303));

    // "Y0 Y1 X2 X3" : -0.045303
    QTerm q4;
    q4[0] = 'Y';
    q4[1] = 'Y';
    q4[2] = 'X';
    q4[3] = 'X';
    hamiltonian.emplace_back(make_pair(q4, -0.045303));

    //"Z0" : 0.171413
    QTerm q5;
    q5[0] = 'Z';
    hamiltonian.emplace_back(make_pair(q5, 0.171413));

    //"Z0 Z1" : 0.168689
    QTerm q6;
    q6[0] = 'Z';
    q6[1] = 'Z';
    hamiltonian.emplace_back(make_pair(q6, 0.168689));

    //"Z0 Z2" : 0.120625
    QTerm q7;
    q7[0] = 'Z';
    q7[2] = 'Z';
    hamiltonian.emplace_back(make_pair(q7, 0.120625));

    //"Z0 Z3" : 0.165928
    QTerm q8;
    q8[0] = 'Z';
    q8[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q8, 0.165928));

    //"Z1" : 0.171413
    QTerm q9;
    q9[1] = 'Z';
    hamiltonian.emplace_back(make_pair(q9, 0.171413));

    //"Z1 Z2" : 0.165928
    QTerm q10;
    q10[1] = 'Z';
    q10[2] = 'Z';
    hamiltonian.emplace_back(make_pair(q10, 0.165928));

    //"Z1 Z3" : 0.120625
    QTerm q11;
    q11[1] = 'Z';
    q11[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q11, 0.120625));

    //"Z2" : -0.223432
    QTerm q12;
    q12[2] = 'Z';
    hamiltonian.emplace_back(make_pair(q12, -0.223432));

    //"Z2 Z3" : 0.174413
    QTerm q13;
    q13[2] = 'Z';
    q13[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q13, 0.174413));

    //"Z3" : -0.223432
    QTerm q14;
    q14[3] = 'Z';
    hamiltonian.emplace_back(make_pair(q14, -0.223432));

    return hamiltonian;
}

static double kl_divergence(const vector<double> &input_data, const vector<double> &output_data) 
{
    int size = input_data.size();
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        if (input_data[i] - 0.0 > 1e-6) 
        {
            result += input_data[i] * std::log(input_data[i] / output_data[i]);
        }
    }

    return abs(result);
}

static QCircuit qft_prog(const QVec& qvec)
{
    QCircuit qft = CreateEmptyCircuit();
    for (auto i = 0; i < qvec.size(); i++)
    {
        qft << H(qvec[qvec.size() - 1 - i]);
        for (auto j = i + 1; j < qvec.size(); j++)
        {
            qft << CR(qvec[qvec.size() - 1 - j],
                qvec[qvec.size() - 1 - i], 2 * PI / (1 << (j - i + 1)));
        }
    }
    return qft;
}

static QProg ghz_prog(const QVec& q)
{
    auto prog = QProg();
    prog << H(q[0]);

    for (auto i = 0; i < q.size() - 1; ++i)
    {
        prog << CNOT(q[i], q[i + 1]);
    }

    return prog;
}

void density_matrix_expval()
{
    DensityMatrixSimulator simulator;
    simulator.init();

    QVec qv;
    std::vector<ClassicalCondition> cv;

    std::string originir = R"(QINIT 4
                            CREG 0
                            H q[0]
                            CNOT q[0],q[1]
                            CNOT q[1],q[3]
                            CNOT q[2],q[3]
                            RZ q[3],(-0.0060418116)
                            )";

    auto prog = convert_originir_string_to_qprog(originir, &simulator, qv, cv);

    auto qnum_expval = simulator.get_expectation(prog, test_hamiltonian_data(), { 0,1,2,3 });

    auto expval = simulator.get_expectation(prog, test_hamiltonian_data(), qv);
    EXPECT_NEAR(expval, 0.134744, 1e-6);
    return;
}

void density_matrix_result()
{
    DensityMatrixSimulator simulator;
    simulator.init();

    auto q = simulator.qAllocMany(6);
    auto c = simulator.cAllocMany(6);

    auto prog = QProg();
    prog << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) << H(q[4]) << H(q[5]);
    prog << T(q[0]);
    prog << T(q[0]).dagger();
    prog << S(q[1]);
    prog << S(q[1]).dagger();
    prog << X(q[1]);
    prog << Y(q[0]);
    prog << Z(q[1]);
    prog << U1(q[1], 1).dagger();
    prog << RX(q[1], -1);
    prog << RY(q[1], -1);
    prog << RZ(q[1], -1);
    prog << CZ(q[4], q[5]);
    prog << CR(q[4], q[5], PI / 2);
    prog << CR(q[4], q[5], PI / 2).dagger();
    prog << CNOT(q[0], q[1]);
    prog << SWAP(q[0], q[5]);
    prog << iSWAP(q[1], q[4]);
    prog << iSWAP(q[1], q[4]).dagger();
    prog << Toffoli(q[0], q[1], q[2]);
    prog << RXX(q[1], q[3], 10);
    prog << RYY(q[1], q[3], 10);
    prog << RZZ(q[1], q[3], 10);
    prog << RZX(q[1], q[3], 10);

    auto density_matrix_result = simulator.get_density_matrix(prog);

    auto reduced_density_matrix_result1 = simulator.get_reduced_density_matrix(prog, q);
    auto reduced_density_matrix_result3 = simulator.get_reduced_density_matrix(prog, { 0, 1 });

    auto dec_probability = simulator.get_probability(prog, 0);
    auto bin_probability = simulator.get_probability(prog, "000000");

    std::vector<std::string> bin_indices = { "000000","000001" };
    auto probabilities1 = simulator.get_probabilities(prog);
    auto probabilities2 = simulator.get_probabilities(prog, q);
    auto probabilities3 = simulator.get_probabilities(prog, { 0, 1 });
    auto probabilities4 = simulator.get_probabilities(prog, bin_indices);


    return;
}

void density_matrix_assert()
{
    DensityMatrixSimulator simulator;
    simulator.init(false);

    auto q = simulator.qAllocMany(6);
    auto c = simulator.cAllocMany(6);

    try
    {
        auto prog = QProg();
        prog << Z(q[1]).control({ q[1] });

        auto density_matrix_result = simulator.get_density_matrix(prog);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << endl;
    }

    try
    {
        auto prog = QProg();
        prog << Z(q[1]).control({ q[0], q[0] });

        auto density_matrix_result = simulator.get_density_matrix(prog);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << endl;
    }

    try
    {
        auto prog = QProg();
        prog << Reset(q[0]);

        auto density_matrix_result = simulator.get_density_matrix(prog);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << endl;
    }



    return;
}

void density_matrix_bitflip()
{
    DensityMatrixSimulator simulator;
    simulator.init();

    auto q = simulator.qAllocMany(6);
    auto c = simulator.cAllocMany(6);

    auto prog = QProg();
    prog << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) << H(q[4]) << H(q[5]);
    prog << T(q[0]);
    prog << S(q[1]);
    prog << X(q[1]);
    prog << Y(q[0]);
    prog << Z(q[1]);
    prog << U1(q[1], 1);
    prog << RX(q[1], -1);
    prog << RY(q[1], -1);
    prog << RZ(q[1], -1);
    prog << CZ(q[4], q[5]);
    prog << CNOT(q[0], q[1]);
    prog << SWAP(q[0], q[5]);
    prog << iSWAP(q[1], q[4]);
    prog << Toffoli(q[0], q[1], q[2]);
    prog << RXX(q[1], q[3], 10);
    prog << RYY(q[1], q[3], 10);
    prog << RZZ(q[1], q[3], 10);
    prog << RZX(q[1], q[3], 10);

    auto density_matrix = simulator.get_density_matrix(prog);

    const QVec noise_qubits = { q[0] };
    simulator.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, { GateType::PAULI_X_GATE, GateType::HADAMARD_GATE }, 0.5);
    simulator.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, { GateType::T_GATE }, 0.5, noise_qubits);
    simulator.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, { GateType::CNOT_GATE }, 0.5);

    std::vector<QVec> qnum_qubits = { {q[0], q[1], q[2]} };
    simulator.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, GateType::PAULI_X_GATE, 0.5, qnum_qubits);
    prob_vec probs = simulator.get_probabilities(prog);

    return;
}

void density_matrix_damping()
{
    DensityMatrixSimulator simulator;
    simulator.init();

    auto q = simulator.qAllocMany(6);
    auto c = simulator.cAllocMany(6);

    auto prog = QProg();
    prog << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) << H(q[4]) << H(q[5]);
    prog << T(q[0]);
    prog << S(q[1]);
    prog << X(q[1]);
    prog << Y(q[0]);
    prog << Z(q[1]);
    prog << U1(q[1], 1);
    prog << RX(q[1], -1);
    prog << RY(q[1], -1);
    prog << RZ(q[1], -1);
    prog << CZ(q[4], q[5]);
    prog << CNOT(q[0], q[1]);
    prog << SWAP(q[0], q[5]);
    prog << iSWAP(q[1], q[4]);
    prog << Toffoli(q[0], q[1], q[2]);
    prog << RXX(q[1], q[3], 10);
    prog << RYY(q[1], q[3], 10);
    prog << RZZ(q[1], q[3], 10);
    prog << RZX(q[1], q[3], 10);

    auto density_matrix = simulator.get_density_matrix(prog);

    simulator.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, { GateType::PAULI_X_GATE }, 0.5);
    simulator.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, { GateType::PAULI_Y_GATE }, 0.5);
    simulator.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, { GateType::HADAMARD_GATE }, 0.5);
    simulator.set_noise_model(NOISE_MODEL::DAMPING_KRAUS_OPERATOR, { GateType::CNOT_GATE }, 0.5);

    prob_vec probs = simulator.get_probabilities(prog);

    return;
}

void density_matrix_decoherence()
{
    DensityMatrixSimulator simulator;
    simulator.init();

    auto q = simulator.qAllocMany(6);
    auto c = simulator.cAllocMany(6);

    auto prog = QProg();
    prog << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]) << H(q[4]) << H(q[5]);
    prog << T(q[0]);
    prog << S(q[1]);
    prog << X(q[1]);
    prog << Y(q[0]);
    prog << Z(q[1]);
    prog << U1(q[1], 1);
    prog << U1(q[0], 1);
    prog << U2(q[1], 1, 2);
    prog << U3(q[2], 1, 2, -30);
    prog << U4(8, 12, 10, 4, q[0]);
    prog << CZ(q[4], q[5]);
    prog << CZ(q[0], q[1]);
    prog << CNOT(q[0], q[1]);
    prog << SWAP(q[0], q[5]);
    prog << iSWAP(q[1], q[4]);
    prog << RXX(q[1], q[3], 10);
    prog << RYY(q[1], q[3], 10);
    prog << RZZ(q[1], q[3], 10);
    prog << RZX(q[1], q[3], 10);

    simulator.set_noise_model(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, { GateType::U1_GATE }, 5, 2, 0.5);
    simulator.set_noise_model(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, { GateType::U2_GATE }, 5, 2, 0.5);
    simulator.set_noise_model(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, { GateType::PAULI_Y_GATE }, 5, 2, 0.5);
    simulator.set_noise_model(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, { GateType::CNOT_GATE }, 5, 2, 0.5);

    const QVec noise_single_qubits = { q[0] };
    simulator.set_noise_model(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, { GateType::U3_GATE, GateType::U4_GATE }, 5, 2, 0.5, noise_single_qubits);
    
    const std::vector<QVec> noise_double_qubits = { {q[0], q[1]} };
    simulator.set_noise_model(NOISE_MODEL::DECOHERENCE_KRAUS_OPERATOR, { GateType::CZ_GATE }, 5, 2, 0.5, noise_double_qubits);

    prob_vec probs = simulator.get_probabilities(prog);
    return;
}

void density_matrix_measure()
{
    DensityMatrixSimulator simulator;
    simulator.init();

    auto q = simulator.qAllocMany(6);
    auto c = simulator.cAllocMany(6);

    auto prog = QProg();
    prog << X(q[0]).control({ q[1], q[2] }) 
         << H(q[0]).control({ q[1] }) 
         << CNOT(q[0], q[1])
         << Toffoli(q[3], q[4], q[5])
         << MeasureAll(q, c);

    simulator.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, { GateType::CNOT_GATE }, 0.36);
    simulator.set_noise_model(NOISE_MODEL::BITFLIP_KRAUS_OPERATOR, { GateType::HADAMARD_GATE }, 0.36);

    auto density_matrix = simulator.get_density_matrix(prog);
    return;
}

void density_matrix_multi_control()
{
    DensityMatrixSimulator simulator;
    simulator.init();

    auto q = simulator.qAllocMany(5);
    auto c = simulator.cAllocMany(5);

    auto prog = QProg();
    prog << H(q[0]) << H(q[1]) << H(q[2]) << H(q[3]);
    prog << X(q[0]).control({ q[2], q[3], q[4]});
    prog << Y(q[0]).control({ q[2] });
    prog << Y(q[0]).control({ q[2], q[3] });
    prog << Y(q[0]).control({ q[2], q[3], q[4] });
    prog << Z(q[1]).control({ q[2], q[3], q[4]});
    prog << U1(q[1], 1).control({ q[2], q[3], q[4]});
    prog << RX(q[1], -1).control({ q[2] });
    prog << RX(q[1], -1).control({ q[2], q[3] });
    prog << RX(q[1], -1).control({ q[2], q[3], q[4] });
    prog << SWAP(q[0], q[1]).control({ q[2]});
    prog << SWAP(q[0], q[1]).control({ q[2], q[3]});
    prog << SWAP(q[0], q[1]).control({ q[2], q[3], q[4]});
    prog << CR(q[1], q[0], PI / 4).control({ q[2] });
    prog << CR(q[1], q[0], PI / 4).control({ q[2], q[3]});
    prog << CR(q[1], q[0], PI / 4).control({ q[2], q[3], q[4]});
    prog << iSWAP(q[1], q[0]).control({ q[2], q[3], q[4]});
    prog << Toffoli(q[0], q[1], q[2]);
    prog << RXX(q[1], q[3], 10);
    prog << RYY(q[1], q[3], 10);
    prog << RZZ(q[1], q[3], 10);
    prog << RZX(q[1], q[3], 10);

    auto density_matrix = simulator.get_density_matrix(prog);

    Qnum reduced_qubits = {};
    auto reduced_density_matrix = simulator.get_reduced_density_matrix(prog, reduced_qubits);
    return;
}

TEST(DensityMatrix, test)
{
    density_matrix_expval();
    density_matrix_result();
    density_matrix_assert();
    density_matrix_measure();
    density_matrix_bitflip();
    density_matrix_damping();
    density_matrix_decoherence();
    density_matrix_multi_control();

    cout << "density matrix test..." << endl;
    return;
}

