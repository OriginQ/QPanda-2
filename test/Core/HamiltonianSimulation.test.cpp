#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>

using namespace std;
USING_QPANDA

#define PRINT_INFORMATION 1

bool test_pauliXSimulation()
{
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto q = qAllocMany(4);
	auto cbits = cAllocMany(4);

	auto prog = createEmptyQProg();

	Matrix2cd mat_operator;
	mat_operator.real() << 0, 1, 1, 0;
	mat_operator.imag() << 0, 0, 0, 0;

	QCircuit circuit;
	circuit << H(q[0])
			<< X(q[0])
			<< RZ(q[0], -PI)
			<< X(q[0])
			<< RZ(q[0], PI)
			<< H(q[0]);
	QOperator sss(circuit);
	QStat unitary = sss.get_matrix();
	std::complex<double> conf(0, -1);
	qmatrix_t U = expMat(conf, mat_operator, PI);

	double www = average_gate_fidelity(U, unitary);
#ifdef PRINT_INFORMATION
	std::cout << "*************************************" << endl;
	std::cout << "the operator matrix is:" << endl;
	std::cout << mat_operator << endl;
	std::cout << "the hamiltonian-form matrix is:" << endl;
	std::cout << U << endl;
	std::cout << "the circuit-form matrix is:";
	std::cout << unitary << endl;
	std::cout << "The state_fidelity is " << www << endl;
#endif // PRINT_INFORMATION
	return true;
}

bool test_pauliYSimulation()
{
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto q = qAllocMany(4);
	auto cbits = cAllocMany(4);

	auto prog = createEmptyQProg();
	QCircuit circuit;

	Matrix2cd mat_operator;
	mat_operator.real() << 0, 0, 0, 0;
	mat_operator.imag() << 0, -1, 1, 0;

	circuit << RX(q[0], PI / 2)
			<< X(q[0])
			<< RZ(q[0], -PI)
			<< X(q[0])
			<< RZ(q[0], PI)
			<< RX(q[0], -PI / 2);
	QOperator sss(circuit);
	QStat unitary = sss.get_matrix();

	qcomplex_t conf(0, 1);
	qmatrix_t U = expMat(conf, mat_operator, PI);

	double www = average_gate_fidelity(U, unitary);
#ifdef PRINT_INFORMATION
	std::cout << "*************************************" << endl;
	std::cout << "the operator matrix is:" << endl;
	std::cout << mat_operator << endl;
	std::cout << "the hamiltonian-form matrix is:" << endl;
	std::cout << U << endl;
	std::cout << "the circuit-form matrix is:";
	std::cout << unitary << endl;
	std::cout << "The state_fidelity is " << www << endl;
#endif // PRINT_INFORMATION
	return true;
}

bool test_pauliZSimulation()
{
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto q = qAllocMany(4);
	auto cbits = cAllocMany(4);

	auto prog = createEmptyQProg();
	QCircuit circuit;

	Matrix2cd mat_operator;
	mat_operator.real() << 1, 0, 0, -1;
	mat_operator.imag() << 0, 0, 0, 0;

	circuit << X(q[0])
			<< RZ(q[0], -PI)
			<< X(q[0])
			<< RZ(q[0], PI);
	QOperator sss(circuit);
	QStat unitary = sss.get_matrix();

	std::complex<double> conf(0, -1);
	qmatrix_t U = expMat(conf, mat_operator, PI);

	double www = average_gate_fidelity(U, unitary);
#ifdef PRINT_INFORMATION
	std::cout << "*************************************" << endl;
	std::cout << "the operator matrix is:" << endl;
	std::cout << mat_operator << endl;
	std::cout << "the hamiltonian-form matrix is:" << endl;
	std::cout << U << endl;
	std::cout << "the circuit-form matrix is:";
	std::cout << unitary << endl;
	std::cout << "The state_fidelity is " << www << endl;
#endif // PRINT_INFORMATION
	return true;
}

bool test_RandomSimulation()
{
	QuantumMachine* machine = initQuantumMachine(CPU);
	auto q = qAllocMany(4);
	auto cbits = cAllocMany(4);

	auto prog = createEmptyQProg();
	QCircuit circuit;

	Matrix2cd mat_operator;
	mat_operator.real() << 0.5, 0.1667, 0.1667, 0.5;
	mat_operator.imag() << 0, 0, 0, 0;

	circuit << X(q[0])
			<< RZ(q[0], -PI / 2)
			<< X(q[0])
			<< RZ(q[0], -PI / 2)
			<< U3(q[0], -0.1667 * 2 * PI, PI / 2, -PI / 2);
	QOperator sss(circuit);
	QStat unitary = sss.get_matrix();

	std::complex<double> conf(0, -1);
	qmatrix_t U = expMat(conf, mat_operator, PI);

	double www = average_gate_fidelity(U, unitary);
#ifdef PRINT_INFORMATION
	std::cout << "*************************************" << endl;
	std::cout << "the operator matrix is:" << endl;
	std::cout << mat_operator << endl;
	std::cout << "the hamiltonian-form matrix is:" << endl;
	std::cout << U << endl;
	std::cout << "the circuit-form matrix is:";
	std::cout << unitary << endl;
	std::cout << "The state_fidelity is " << www << endl;
#endif // PRINT_INFORMATION
	return true;
}


TEST(HamiltonianSimulation, test_hs)
{
	bool test_val = false;
	try
	{
		test_val = test_pauliXSimulation();
		test_val = test_val && test_pauliYSimulation();
		test_val = test_val && test_pauliZSimulation();
		test_val = test_val && test_RandomSimulation();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
		test_val = false;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
		test_val = false;
	}

	ASSERT_TRUE(test_val);
}