#include "utils.h"
#include <vector>
#include <fstream>
#include <memory>
#include <iostream>
#include <chrono>
#include <limits>
#include "gtest/gtest.h"
#include "QPanda.h"
#include "Operator/PauliOperator.h"
#include "Variational/VarPauliOperator.h"
#include "Core/VirtualQuantumProcessor/NoiseQPU/NoiseModel.h"
USING_QPANDA
using namespace std;
using namespace QPanda::Variational;

TEST(AutoDiffTest, test_no_quantum)
{
    auto machine = initQuantumMachine();
    auto qlist = machine->allocateQubits(4);
    auto prog = CreateEmptyQProg();
    auto vqc = VariationalQuantumCircuit();

    auto v1 = var(0, true);
    VarPauliOperator p1("X0", complex_var(v1, var(0, false)));
    auto p2 = p1.data();
    auto tmp = p2[0].second.first;
    vqc.insert(VariationalQuantumGate_RX(qlist[0], tmp));

    PauliOperator op("Z0", 1);
    auto loss = qop(vqc, op, machine, qlist);

    MatrixXd x(1, 1);
    x(0, 0) = Q_PI_2;

    v1.setValue(x);

    std::cout << eval(loss, true) << std::endl;

}

VQC parity_check_circuit(std::vector<Qubit*> qubit_vec)
{
    VQC circuit;
    for (auto i = 0; i < qubit_vec.size() - 1; i++)
    {
        circuit.insert(VQG_CNOT(
            qubit_vec[i],
            qubit_vec[qubit_vec.size() - 1]));
    }

    return circuit;
}

VQC simulateZTerm(
    const std::vector<Qubit*> &qubit_vec,
    var coef,
    var t)
{
    VQC circuit;
    if (0 == qubit_vec.size())
    {
        return circuit;
    }
    else if (1 == qubit_vec.size())
    {
        circuit.insert(VQG_RZ(qubit_vec[0], coef * t*-1));
    }
    else
    {
        circuit.insert(parity_check_circuit(qubit_vec));
        circuit.insert(VQG_RZ(qubit_vec[qubit_vec.size() - 1], coef * t*-1));
        circuit.insert(parity_check_circuit(qubit_vec));
    }

    return circuit;
}

VQC simulatePauliZHamiltonian(
    const std::vector<Qubit*>& qubit_vec,
    const QPanda::QHamiltonian & hamiltonian,
    var t)
{
    VQC circuit;

    for (auto j = 0; j < hamiltonian.size(); j++)
    {
        std::vector<Qubit*> tmp_vec;
        auto item = hamiltonian[j];
        auto map = item.first;

        for (auto iter = map.begin(); iter != map.end(); iter++)
        {
            if ('Z' != iter->second)
            {
                QCERR("Bad pauliZ Hamiltonian");
                throw std::string("Bad pauliZ Hamiltonian.");
            }

            tmp_vec.push_back(qubit_vec[iter->first]);
        }

        if (!tmp_vec.empty())
        {
            circuit.insert(simulateZTerm(tmp_vec, item.second, t));
        }
    }

    return circuit;
}

TEST(AutoDiffTest, qaoa)
{
    QPanda::QPauliMap pauli_map;
    pauli_map.insert(std::make_pair("Z0 Z4", 0.73));
    pauli_map.insert(std::make_pair("Z0 Z5", 0.33));
    pauli_map.insert(std::make_pair("Z0 Z6", 0.5));
    pauli_map.insert(std::make_pair("Z1 Z4", 0.69));

    pauli_map.insert(std::make_pair("Z1 Z5", 0.36));
    pauli_map.insert(std::make_pair("Z2 Z5", 0.88));
    pauli_map.insert(std::make_pair("Z2 Z6", 0.58));
    pauli_map.insert(std::make_pair("Z3 Z5", 0.67));

    pauli_map.insert(std::make_pair("Z3 Z6", 0.43));


    PauliOperator op(pauli_map);

    QuantumMachine *machine = initQuantumMachine();
    vector<Qubit*> q;
    for (int i = 0; i < op.getMaxIndex(); ++i)
        q.push_back(machine->allocateQubit());

    VQC vqc;
    for_each(q.begin(), q.end(), [&vqc](Qubit* qbit)
    {
        vqc.insert(VQG_H(qbit));
    });

    int qaoa_step = 1;

    var x(MatrixXd::Random(2 * qaoa_step, 1));

    for (int i = 0; i < 2 * qaoa_step; i += 2)
    {
        vqc.insert(simulatePauliZHamiltonian(q, op.toHamiltonian(), x[i + 1]));
        for (auto _q : q) {
            vqc.insert(VQG_RX(_q, x[i]));
        }
    }
    std::unordered_map<var, MatrixXd> grad = { { x, zeros_like(x) }, };
    var loss = qop(vqc, op, machine, q);
    std::vector<var> leaves = { x };
    expression exp(loss);
    std::unordered_set<var> leaf_set = exp.findNonConsts(leaves);
    int iterations = 10;
    double learning_rate = 0.1;
    for (int i = 0; i < iterations; i++) {
        std::cout << "LOSS : " << eval(loss, true) << std::endl;
        back(exp, grad, leaf_set);
        x.setValue(x.getValue().array() - learning_rate * grad[x].array());
    }
    getchar();
    EXPECT_EQ(true, true);
}
