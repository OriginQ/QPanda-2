#include <atomic>
#include <string>
#include "Core/Core.h"
#include "gtest/gtest.h"
#include "Core/Utilities/Tools/RandomCircuit.h"
#include "Core/VirtualQuantumProcessor/Stabilizer/Stabilizer.h"

USING_QPANDA

#define CHECK_RUN_TIME_BEGIN                  auto start_time = chrono::system_clock::now()\

#define CHECK_RUN_TIME_END_AND_COUT_SECONDS         auto final_time = chrono::system_clock::now();\
                                                    auto duration = chrono::duration_cast<chrono::microseconds>(final_time - start_time);\
                                                    std::cout << "test run time counts :" << double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den << std::endl\

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

void stabilizer_measure()
{
    Stabilizer simulator;
    simulator.init();

    auto q = simulator.qAllocMany(15);
    auto c = simulator.cAllocMany(15);

    auto prog = QProg();
    prog << H(q[0])
        << X(q[0])
        << Y(q[0]).control({q[2]})
        << Z(q[0])
        << SWAP(q[0], q[1])
        << I(q[0])
        << Reset(q[0])
        << CZ(q[0], q[1])
        << CZ(q[0], q[9])
        << CNOT(q[9], q[1])
        << Measure(q[0], c[0])
        << Measure(q[1], c[1])
        << Measure(q[9], c[9]);

    CHECK_RUN_TIME_BEGIN;
    auto result = simulator.runWithConfiguration(prog, 1000);
    //auto result = simulator.probRunDict(prog, q);
    CHECK_RUN_TIME_END_AND_COUT_SECONDS;

    for (auto val : result)
    {
        std::cout << val.first << " : " << val.second << std::endl;
    }

    return;
}

void stabilizer_pmeasure()
{
    Stabilizer simulator;
    simulator.init();

    auto q = simulator.qAllocMany(25);
    auto c = simulator.cAllocMany(25);

    auto prog = QProg();
    prog << H(q[0])
         << CZ(q[0], q[1])
         << CZ(q[0], q[9])
         << CNOT(q[0], q[1]);

    std::vector<std::string> clifford_gates;
    clifford_gates.emplace_back("H");
    clifford_gates.emplace_back("S");
    clifford_gates.emplace_back("X");
    clifford_gates.emplace_back("Y");
    clifford_gates.emplace_back("Z");
    clifford_gates.emplace_back("CZ");
    clifford_gates.emplace_back("CNOT");

    QProg rng_circuit;
    rng_circuit << random_qcircuit(q, 10, clifford_gates);
    rng_circuit << MeasureAll(q, c);

    //std::cout << random_prog << std::endl;

    CHECK_RUN_TIME_BEGIN;
    //auto result = simulator.runWithConfiguration(random_prog, 100);
    auto result = simulator.probRunDict(prog, { q[0],q[1],q[2] });

    for (auto val : result)
    {
        std::cout << val.first << " : " << val.second << std::endl;
    }

    CHECK_RUN_TIME_END_AND_COUT_SECONDS;
    

    return;
}

TEST(Stabilizer, test)
{
    stabilizer_measure();
    stabilizer_pmeasure();

    cout << "stabilizer.test test..." << endl;
    return;
}

