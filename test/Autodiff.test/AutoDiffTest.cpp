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
USING_QPANDA
using namespace std;
using namespace QPanda::Variational;
static int test_main();

TEST(AutoDiffTest, test_no_quantum)
{
    auto machine=initQuantumMachine();
    auto qlist=machine->allocateQubits(4);
    auto prog=CreateEmptyQProg();
    auto vqc=VariationalQuantumCircuit();

    auto v1 = var(0, true);
    VarPauliOperator p1("X0", complex_var(v1, var(0, false)));
    auto p2=p1.data();
    auto tmp = p2[0].second.first;
    vqc.insert(VariationalQuantumGate_RX(qlist[0],tmp));

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
        circuit.insert( VQG_CNOT(
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
    pauli_map.insert(std::make_pair("Z0 Z6", 0.49));
    pauli_map.insert(std::make_pair("Z6 Z1", 0.59));
    pauli_map.insert(std::make_pair("Z1 Z7", 0.44));
    pauli_map.insert(std::make_pair("Z7 Z2", 0.36));

    pauli_map.insert(std::make_pair("Z2 Z8", 0.63));
    pauli_map.insert(std::make_pair("Z8 Z13", 0.36));
    pauli_map.insert(std::make_pair("Z13 Z19", 0.81));
    pauli_map.insert(std::make_pair("Z19 Z14", 0.29));

    pauli_map.insert(std::make_pair("Z14 Z9", 0.52));
    pauli_map.insert(std::make_pair("Z9 Z4", 0.43));
    pauli_map.insert(std::make_pair("Z13 Z18", 0.72));
    pauli_map.insert(std::make_pair("Z18 Z12", 0.40));

    pauli_map.insert(std::make_pair("Z12 Z7", 0.60));
    pauli_map.insert(std::make_pair("Z12 Z17", 0.71));
    pauli_map.insert(std::make_pair("Z17 Z11", 0.50));
    pauli_map.insert(std::make_pair("Z11 Z6", 0.64));

    pauli_map.insert(std::make_pair("Z11 Z16", 0.57));
    pauli_map.insert(std::make_pair("Z16 Z10", 0.41));
    pauli_map.insert(std::make_pair("Z10 Z5", 0.23));
    pauli_map.insert(std::make_pair("Z10 Z15", 0.40));

    pauli_map.insert(std::make_pair("Z5 Z0", 0.18));

    PauliOperator op(pauli_map);

    QuantumMachine *machine = initQuantumMachine();
    vector<Qubit*> q;
    for (int i = 0; i < op.getMaxIndex(); ++i) 
        q.push_back(machine->allocateQubit());

    VQC vqc;
    for_each(q.begin(), q.end(), [&vqc](Qubit* qubit)
    {
        vqc.insert(VQG_H(qubit));
    });

    int qaoa_step = 1;

    var x(MatrixXd::Random(2 * qaoa_step, 1));

    for (int i = 0; i < 2*qaoa_step; i+=2)
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
    EXPECT_EQ(true, true);
}

using namespace QPanda::Variational;

typedef unsigned uint;

template<typename M>
M load_csv (const std::string & path, bool verbose = false) {
    using namespace Eigen;
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            if(verbose)
                std::cout << cell << std::endl;
            
            double val = std::stod(cell);
            values.push_back(val);
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

// Returns a one-hot encoding representation
// Assumes the matrix is in the shape (Nx1)
MatrixXd one_hot(const MatrixXd& m){
    int max_num = 0;
    for(int i = 0; i < m.rows(); i++){
       max_num = std::max(max_num, int(m(i,0))); 
    }
    MatrixXd onehot = MatrixXd::Zero(m.rows(), max_num+1);
    for(int i = 0; i < m.rows(); i++){
        onehot(i, int(m(i,0))) = 1;
    }

    return onehot;
}

static int test_main(){
    
    auto start = std::chrono::steady_clock::now();

    // TIMING START
    const double learning_rate = 0.001;
    const double scale_init = 0.01;
    const size_t iterations = 100;
    MatrixXd A = load_csv<MatrixXd>("iris.csv");

    var X(A.leftCols(4)), 
        y(one_hot(A.rightCols(1))), 
        w1(MatrixXd::Random(4,10)*scale_init),
        w2(MatrixXd::Random(10,3)*scale_init);
    
    /*
	std::cout << "X : " << X.getValue() << std::endl;
    std::cout << "y : " << y.getValue() << std::endl;
    std::cout << "w1 : " << w1.getValue() << std::endl;
    std::cout << "w2 : " << w2.getValue() << std::endl;
	*/
    
    

    std::unordered_map<var, MatrixXd> m = {
        { w1, zeros_like(w1) },
        { w2, zeros_like(w2) },
    };

    // Setting up the equation
    var sigm1 = 1 / (1 + exp(-1 * dot(X, w1)));
    var sigm2 = 1 / (1 + exp(-1 * dot(sigm1, w2)));
    var loss = sum(-1 * (y * log(sigm2) + (1-y) * log(1-sigm2)));


    std::vector<var> leaves = {w1, w2};
    expression exp(loss);
    std::unordered_set<var> leaf_set = exp.findNonConsts(leaves);
    // 找到所有 non-const, 当设定了leaves之后，可以一直向上找
    // 直到找到所有的待求导节点

    for(int i = 0; i < iterations; i++){
        std::cout << "LOSS : " << eval(loss, true) << std::endl;
        // et::back(exp, m , leaf_set);
        back(exp, m, leaf_set);
        w1.setValue( w1.getValue().array() - learning_rate * m[w1].array() );
        w2.setValue( w2.getValue().array() - learning_rate * m[w2].array() );
    }
    // TIMING END
    
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    
    MatrixXd ans(y.getValue().rows(), 6);
    ans.leftCols(3) << y.getValue();
    ans.rightCols(3) << sigm2.getValue();
    std::cout << "PREDICTION : " << ans << std::endl;
    std::cout << "Time elapsed : " << 
        std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    return 0;
}
