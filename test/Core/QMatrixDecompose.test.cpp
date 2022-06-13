#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>
#include "Extensions/Extensions.h"

USING_QPANDA
using namespace std;
bool test_matrix_decompose_1()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(8);
	auto c = qvm->allocateCBits(8);

	const QStat target_matrix = { qcomplex_t(0.6477054522122977, 0.1195417767870219), qcomplex_t(-0.16162176706189357, -0.4020495632468249), qcomplex_t(-0.19991615329121998, -0.3764618308248643), qcomplex_t(-0.2599957197928922, -0.35935248873007863),
		qcomplex_t(-0.16162176706189363, -0.40204956324682495), qcomplex_t(0.7303014482204584, -0.4215172444390785), qcomplex_t(-0.15199187936216693, 0.09733585496768032), qcomplex_t(-0.22248203136345918, -0.1383600597660744),
		qcomplex_t(-0.19991615329122003, -0.3764618308248644), qcomplex_t(-0.15199187936216688, 0.09733585496768032), qcomplex_t(0.6826630277354306, -0.37517063774206166), qcomplex_t(-0.3078966462928956, -0.2900897445133085),
		qcomplex_t(-0.2599957197928923, -0.3593524887300787), qcomplex_t(-0.22248203136345912, -0.1383600597660744), qcomplex_t(-0.30789664629289554, -0.2900897445133085), qcomplex_t(0.6640994547408099, -0.338593803336005) };

	auto cir = matrix_decompose_qr({ q[0], q[1] }, target_matrix);
	//auto cir = matrix_decompose_householder({ q[0], q[1] }, target_matrix, false);
	//cout << "decomposed circuit:" << cir << endl;
	const auto mat_2 = getCircuitMatrix(cir);
	cout << "mat_2:\n" << mat_2 << endl;

	auto prog = QProg();
	prog << cir;
	qvm->directlyRun(prog);
	auto stat = qvm->getQState();

	destroyQuantumMachine(qvm);

	if ( 0 == mat_compare(target_matrix, mat_2, MAX_COMPARE_PRECISION)){
		return true;
	}

	return false;
}

bool test_matrix_decompose_2()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(8);
	auto c = qvm->allocateCBits(8);

	QCircuit test_cir;
	test_cir /*<< H(q[0]) << H(q[1])*/ << CNOT(q[0], q[1]) /*<< T(q[1])*/;
	QStat stat_0;
	/*{
		auto prog = QProg();
		prog << test_cir;
		qvm->directlyRun(prog);
		stat_0 = qvm->getQState();
		std::cout << "stat_0:\n" << stat << endl;
	}*/
	const QStat target_matrix = getCircuitMatrix(test_cir);
	std::cout << "target_matrix:\n" << target_matrix << endl;
	
	auto cir = matrix_decompose_qr({ q[0], q[1] }, target_matrix);
	//auto cir = matrix_decompose_householder({ q[0], q[1] }, target_matrix);
	std::cout << "decomposed circuit:" << cir << endl;
	const auto mat_2 = getCircuitMatrix(cir);
	std::cout << "mat_2:\n" << mat_2 << endl;

	auto prog = QProg();
	prog << cir;
	qvm->directlyRun(prog);
	auto stat = qvm->getQState();

	destroyQuantumMachine(qvm);

	if ( 0 == mat_compare(target_matrix, mat_2, MAX_COMPARE_PRECISION)){
		return true;
	}

	return false;
}

bool test_matrix_decompose2pualis()
{
	QuantumMachine* machine = initQuantumMachine(CPU);
	Matrix4d mat;
	mat << 15, 9, 5, -3,
		9, 15, 3, 2,
		5, 3, 4, 96,
		-3, 96, -9, 6;
	mat *= (1 / 4.0);
	cout << "The matrix is:" << endl;
	cout << mat << endl;
	EigenMatrixX eigMat = mat;
	PualiOperatorLinearCombination res;
	matrix_decompose_paulis(machine, eigMat, res);
	cout << endl << "**************************************************" << endl;
	cout << "The linear combination of unitaries(coefficient) is:" << endl;
	for (auto& val : res)
	{
		cout << val.first << ", ";
	}
	cout << endl << "**************************************************" << endl;
	std::cout << "The linear combination of unitaries(circuit) is:" << endl;
	for (auto& val : res)
	{
		cout << val.second << endl;
	}
	return true;
}

TEST(QMatrixDecompose, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_matrix_decompose_1();
		test_val = (test_val && test_matrix_decompose_2());
		test_val = (test_val && test_matrix_decompose2pualis());
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
