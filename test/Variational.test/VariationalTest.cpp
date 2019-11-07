#include <iostream>
#include <limits>
#include "gtest/gtest.h"
#include "QPanda.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include <math.h>

#include "Components/Operator/PauliOperator.h"
#include "Variational/var.h"
#include "Variational/expression.h"
#include "Variational/utils.h"
#include "Variational/Optimizer.h"
#include "Variational/VarFermionOperator.h"
#include "Variational/VarPauliOperator.h"

using namespace QPanda::Variational;
using namespace QPanda;
using namespace std;

TEST(VariationalTest, VarOperatorTest)
{
    using namespace QPanda;
    using namespace Variational;
    var a(2, true);
    var b(3, true);

    VarFermionOperator fermion_op("1+ 0", a);
    VarPauliOperator pauli_op("Z1 Z0", b);
    std::cout << fermion_op << std::endl;
    std::cout << pauli_op << std::endl;
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
    QVec &qubit_vec,
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
    QVec& qubit_vec,
    const QPanda::QHamiltonian & hamiltonian,
    var t)
{
    VQC circuit;

    for (auto j = 0; j < hamiltonian.size(); j++)
    {
        QVec tmp_vec;
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

TEST(VariationalTest, QAOATest)
{
    PauliOperator::PauliMap pauli_map{
        {"Z0 Z4", 0.73},{"Z2 Z5", 0.88},
        {"Z0 Z5", 0.33},{"Z2 Z6", 0.58},
        {"Z0 Z6", 0.50},{"Z3 Z5", 0.67},
        {"Z1 Z4", 0.69},{"Z3 Z6", 0.43},
        {"Z1 Z5", 0.36}
    };

    PauliOperator op(pauli_map);

    QuantumMachine *machine = initQuantumMachine();
    QVec qlist;
    for (int i = 0; i < op.getMaxIndex(); ++i)
        qlist.push_back(machine->allocateQubit());

    VQC vqc;
    for_each(qlist.begin(), qlist.end(), [&vqc](Qubit* qbit)
    {
        vqc.insert(VQG_H(qbit));
    });

    int qaoa_step = 2;

    var x(MatrixXd::Random(2 * qaoa_step, 1), true);

    for (auto i = 0u; i < 2*qaoa_step; i+=2)
    {
        vqc.insert(simulatePauliZHamiltonian(qlist, op.toHamiltonian(), x[i + 1]));
        for (auto _q : qlist) {
            vqc.insert(VQG_RX(_q, x[i]));
        }
    }

    var loss = qop(vqc, op, machine, qlist);
    auto optimizer = VanillaGradientDescentOptimizer::minimize(loss, 0.1, 1.e-6);

    auto leaves = optimizer->get_variables();
    constexpr size_t iterations = 50;
    for (auto i = 0u; i < iterations; i++)
    {
        optimizer->run(leaves);
        std::cout << "iter: " << i << " loss : " << optimizer->get_loss() << std::endl;
    }

    QProg prog;
    QCircuit circuit = vqc.feed();
    prog << circuit;

    directlyRun(prog);
    auto result = quickMeasure(qlist, 100);

    for (auto i:result)
    {
        std::cout << i.first << " : " << i.second << std::endl;
    }

	destroyQuantumMachine(machine);
}

TEST(VariationalTest, VQPTest)
{
#ifdef legacy_1214
    VQP problem(
        4,
        CPU_SINGLE_THREAD
    );
    auto q = problem.get_qubits();
    VQC circ;
    auto var1 = Variable();
    auto gate = VQG_H(q[0]);
    circ.insert(gate)
        .insert(VQG_RX(q[1], var1));

    problem.set_circuit(circ);

    map<string, complex_d> paulimap = { {"Z0",1 }, {"Z1",1} };

    PauliOperator hamiltonian(paulimap);

    problem.set_hamiltonian(hamiltonian);

    auto expectation = problem.feed();

    EXPECT_EQ(0 + 1, expectation);
#endif   

	
}


TEST(VariationalTest, OptimizerTest)
{
#ifdef legacy_1214
    VQP problem(
        4,
        CPU_SINGLE_THREAD
    );
    auto q = problem.get_qubits();
    VQC circ;
    auto var1 = Variable();
    circ.insert(VQG_H(q[0]))
        .insert(VQG_RX(q[1], var1));

    problem.set_circuit(circ);

    map<string, complex_d> paulimap = { {"Z1",1} };

    PauliOperator hamiltonian(paulimap);

    problem.set_hamiltonian(hamiltonian);

    VanillaGradientDescentOptimizer vgdo = VanillaGradientDescentOptimizer();
    vgdo.set_problem(problem, OptimizerMode::MINIMIZE);
    size_t max_times = 100;
    for (auto i = 0; i < max_times; i++)
    {
        vgdo.run();
            << vgdo.get_loss() << endl;
    }

    auto expectation = problem.feed();

    EXPECT_EQ(0 + 1, expectation);
#endif
}

TEST(VariationalTest, sigmoidTest)
{
#ifdef legacy_1214
	var x(MatrixXd::Zero(1, 3));
	var z = sigmoid(x);

	var z1 = 1 / (1 + exp(-1*x));
	var dz1 = z1*(1 - z1); // dz1 is the derivative of z1
	
	std::unordered_map<var, MatrixXd> grad = { { x, zeros_like(x) }, };
	std::vector<var> leaves = { x };

	//x.setValue(scalar(0));

	//MatrixXd tmp(2, 2);
	//tmp << 1, 2, 3, 4;
	MatrixXd tmp(1, 3);
	tmp << 0, 1, 2;

	x.setValue(tmp);

	expression exp(z);
	std::unordered_set<var> leaf_set = exp.findNonConsts(leaves); //?
	eval(z, true);
	eval(z1, true);
	eval(dz1, true);
	back(exp, grad, leaf_set);
	
	/*
	std::cout << "x=" << "\n" << x.getValue() << std::endl;
	std::cout << "z=" << "\n" << z.getValue() << std::endl;
	std::cout << "z1=" << "\n" << z1.getValue() << std::endl;
	std::cout << "z1=" << "\n" << z1.getValue() << std::endl;
	std::cout << "dz/dx=" << "\n" << grad[x].array() << std::endl;
	std::cout << "dz1/dx=" << "\n" << dz1.getValue() << std::endl;
	*/


	MatrixXd expected_z = z1.getValue();
	MatrixXd actual_z = z.getValue();
	EXPECT_EQ(expected_z, actual_z);

	MatrixXd expected_dz = dz1.getValue();
	MatrixXd actual_dz = grad[x].array();
	EXPECT_EQ(expected_dz, actual_dz);

#endif
}


TEST(VariationalTest, softmaxTest)
{
#ifdef legacy_1214
	var x(MatrixXd::Zero(1, 3));
	var z = softmax(x);
	var z1 = exp(x) / sum(exp(x));

	std::unordered_map<var, MatrixXd> grad = { { x, zeros_like(x) }, };
	std::vector<var> leaves = { x };

	MatrixXd tmp(1, 3);
	//tmp << 1, 1, 2; 
	tmp << 0.1, 0.1, 0.2;
	x.setValue(tmp);

	expression exp(z);
	std::unordered_set<var> leaf_set = exp.findNonConsts(leaves);
	eval(z, true);
	eval(z1, true);

	MatrixXd zV = z1.getValue();
	MatrixXd t = MatrixXd::Zero(zV.size(), zV.size());
	for (auto i = 0; i < zV.size(); i++) {
		//std::cout << "zV=" << "\n" << zV(i) << std::endl;
		t(i, i) = zV(i);
	}
	MatrixXd zz1 = t - zV.transpose()*zV;
	std::cout << "zz1=" << "\n" << zz1 << std::endl;
	std::cout << "================"<< std::endl;

	MatrixXd dx = MatrixXd::Ones(1, zV.size());
	MatrixXd dz1 = dx * zz1;

	back(exp, grad, leaf_set);

	std::cout << "x=" << "\n" << x.getValue() << std::endl;
	std::cout << "z=" << "\n" << z.getValue() << std::endl;
	std::cout << "z1=" << "\n" << z1.getValue() << std::endl;
	std::cout << "dz/dx=" << "\n" << grad[x].array() << std::endl;
	std::cout << "dz1/dx=" << "\n" << dz1 << std::endl;

	MatrixXd expected_z = z1.getValue();
	MatrixXd actual_z = z.getValue(); 
	EXPECT_EQ(expected_z, actual_z);

	MatrixXd expected_dz = dz1;
	MatrixXd actual_dz = grad[x].array();   
	EXPECT_EQ(expected_dz, actual_dz);
#endif
}


TEST(VariationalTest, crossEntropyTest) {
#ifdef legacy_1214
	var x(MatrixXd::Zero(1, 3));
	var y(MatrixXd::Zero(1, 3));


	var z = crossEntropy(y, x);
	var z1 = -1*sum(y * log(x));

	//std::unordered_map<var, MatrixXd> grad = { { x, zeros_like(x) }, };
	//std::vector<var> leaves = { x };

	std::unordered_map<var, MatrixXd> grad = { { y, zeros_like(y) }, };
	std::vector<var> leaves = { y };


	MatrixXd tmp1(1, 3);
	tmp1 << 0.4, 0.3, 0.3;

	MatrixXd tmp2(1, 3);
	tmp2 << 1, 0, 0;

	x.setValue(tmp1);
	y.setValue(tmp2);

	expression exp(z);
	std::unordered_set<var> leaf_set = exp.findNonConsts(leaves);
	eval(z, true);
	eval(z1, true);
	back(exp, grad, leaf_set);

	/*
	MatrixXd yy = y.getValue();
	MatrixXd xx = x.getValue();
	MatrixXd dz1_dx = MatrixXd::Zero(1,xx.size()); // dz1_dx Ϊz1����x��ƫ����
	for (auto i = 0; i < xx.size(); i++) {
		dz1_dx(i) = -yy(i) / xx(i);
	}
	*/

	MatrixXd xx = x.getValue();
	MatrixXd dz1_dy = MatrixXd::Zero(1, xx.size());
	for (auto i = 0; i < xx.size(); i++) {
		dz1_dy(i) = - log( xx(i) );
	}

	std::cout << "x=" << "\n" << x.getValue() << std::endl;
	std::cout << "y=" << "\n" << y.getValue() << std::endl;
	std::cout << "z=" << "\n" << z.getValue() << std::endl;
	std::cout << "z1=" << "\n" << z1.getValue() << std::endl;
	//std::cout << "dz/dx=" << "\n" <<grad[x].array() << std::endl;
	//std::cout << "dz1/dx=" << "\n" << dz1_dx  << std::endl;
	std::cout << "dz/dy=" << "\n" << grad[y].array() << std::endl;
	std::cout << "dz1/dy=" << "\n" << dz1_dy << std::endl;
	MatrixXd expected_z = z1.getValue();
	MatrixXd actual_z = z.getValue();
	EXPECT_EQ(expected_z, actual_z);

	//MatrixXd expected_dz_dx = dz1_dx;
	//MatrixXd actual_dz_dx = grad[x].array(); 
	//EXPECT_EQ(expected_dz_dx, actual_dz_dx);

	MatrixXd expected_dz_dy = dz1_dy;
	MatrixXd actual_dz_dy = grad[y].array();
	EXPECT_EQ(expected_dz_dy, actual_dz_dy);

#endif
}

TEST(VariationalTest, dropoutTest) {
#ifdef legacy_1214
	var x(MatrixXd::Zero(1, 3));

	var p(MatrixXd::Zero(1, 3)); 
	var z = dropout(x, p);

	std::unordered_map<var, MatrixXd> grad = { { x, zeros_like(x) }, };
	std::vector<var> leaves = { x };

	//p.setValue(scalar(0));
	MatrixXd tmp1(1, 3);
	tmp1 << 1, 1, 2;
	MatrixXd tmp2(1, 3);
	tmp2 << 0.8, 0.8, 0.8; // p is probability

	x.setValue(tmp1);
	p.setValue(tmp2);

	expression exp(z);
	std::unordered_set<var> leaf_set = exp.findNonConsts(leaves);
	eval(z, true);
	back(exp, grad, leaf_set);

	
	std::cout << "x=" << "\n" << x.getValue() << std::endl;
	std::cout << "z=" << "\n" << z.getValue() << std::endl;
	std::cout << "dz/dx=" << "\n" << grad[x].array() << std::endl;
	EXPECT_EQ(true, true);
#endif
}


TEST(VariationalTest, VanillaGradientDescentTest) {
//#ifdef legacy_1214
		MatrixXd train_x(17, 1);
		MatrixXd train_y(17, 1);

		train_x(0, 0) = 3.3;
		train_x(1, 0) = 4.4;
		train_x(2, 0) = 5.5;
		train_x(3, 0) = 6.71;
		train_x(4, 0) = 6.93;
		train_x(5, 0) = 4.168;
		train_x(6, 0) = 9.779;
		train_x(7, 0) = 6.182;
		train_x(8, 0) = 7.59;
		train_x(9, 0) = 2.167;
		train_x(10, 0) = 7.042;
		train_x(11, 0) = 10.791;
		train_x(12, 0) = 5.313;
		train_x(13, 0) = 7.997;
		train_x(14, 0) = 5.654;
		train_x(15, 0) = 9.27;
		train_x(16, 0) = 3.1;
		train_y(0, 0) = 1.7;
		train_y(1, 0) = 2.76;
		train_y(2, 0) = 2.09;
		train_y(3, 0) = 3.19;
		train_y(4, 0) = 1.694;
		train_y(5, 0) = 1.573;
		train_y(6, 0) = 3.366;
		train_y(7, 0) = 2.596;
		train_y(8, 0) = 2.53;
		train_y(9, 0) = 1.221;
		train_y(10, 0) = 2.827;
		train_y(11, 0) = 3.465;
		train_y(12, 0) = 1.65;
		train_y(13, 0) = 2.904;
		train_y(14, 0) = 2.42;
		train_y(15, 0) = 2.94;
		train_y(16, 0) = 1.3;

		var X(train_x);
		var Y(train_y);

		var W(1.0, true);
		var b(1.0, true);

		//auto Y_ = W * X + b;
		var Y_ = W * X + b;
		auto loss = sum(poly(Y - Y_, 2) / 17);
		auto optimizer = VanillaGradientDescentOptimizer::minimize(loss, 0.01, 1.e-6);
		//auto optimizer = MomentumOptimizer::minimize(loss, 0.01, 1.e-6);
		//auto optimizer = AdaGradOptimizer::minimize(loss, 0.01, 1.e-6);
		//auto optimizer = RMSPropOptimizer::minimize(loss, 0.01, 1.e-6);
		//auto optimizer = AdamOptimizer::minimize(loss, 0.01, 1.e-6);

		auto leaves = optimizer->get_variables();

		std::cout << "loss:" << "\t" << optimizer->get_loss() << std::endl;
		size_t iter = 1000;
		for (size_t i = 0; i < iter; i++)
		{
			optimizer->run(leaves);
			std::cout << "i: " << i << "\t" << optimizer->get_loss()
				<< "\t W:" << QPanda::Variational::eval(W, true) << "\t b:" << QPanda::Variational::eval(b, true)
				<< std::endl;
		}
		EXPECT_EQ(true, true);
//#endif
}


TEST(VariationalTest, pauseTest) {
	system("pause");
}







