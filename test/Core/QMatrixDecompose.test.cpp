#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>
#include "Extensions/Extensions.h"
#include "Core/Utilities/UnitaryDecomposer/QSDecomposition.h"
#include "Core/Utilities/UnitaryDecomposer/MatrixUtil.h"

USING_QPANDA
using namespace std;
using namespace Eigen;

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
	QMatrixXd eigMat = mat;
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

bool test_unitary_decomposer_nq(int qnum, DecompositionMode type, bool is_positive_seq)
{
	std::cout << "qnum : " << qnum << std::endl;
	std::cout << "decomposition mode : " << (int)type << std::endl;

	int ret = true;
	auto qvm = new CPUQVM();
	qvm->init();

	auto qv = qvm->qAllocMany(qnum);
	int dim = pow(2, qnum);

	Eigen::MatrixXcd in_mat = random_unitary(dim);
	auto start = std::chrono::steady_clock::now();
	
	auto cir_dec_nq = unitary_decomposer_nq(in_mat, qv,type , is_positive_seq);

	auto end = std::chrono::steady_clock::now();
	auto used_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "unitary_decomposer_nq use milliseconds : " << used_time << std::endl;

	int cnot_num = count_qgate_num(cir_dec_nq, GateType::CNOT_GATE);
	std::cout << "cnot num :" << cnot_num << std::endl;

	auto mat_out = getCircuitMatrix(cir_dec_nq, is_positive_seq);
	auto mat_src = eigen2qstat(in_mat);
	if (mat_compare(mat_src, mat_out, 1e-5) !=0){
		std::cout << "unitary_decomposer_nq fail\n";
		ret = false;
	}

	qvm->finalize();
	delete qvm;

	return ret;
}


TEST(QMatrixDecompose, test2)
{
	try
	{
		/*	int qnum = 3;
			auto type = DecompositionMode::QSD;
			bool is_positive_seq = false;
			ret = test_unitary_decomposer_nq(qnum, type, is_positive_seq);*/
		for (int i = 3; i < 7; i++)
		{
			int qnum = i;
			for (int j = 0; j < (int)DecompositionMode::CSD; j++)
			{
				DecompositionMode type = static_cast<DecompositionMode>(j);
				if (type == DecompositionMode::HOUSEHOLDER_QR || type == DecompositionMode::QR)
				{
					std::cout << "pass HOUSEHOLDER_QR or QR \n" << std::endl;
					continue;
				}
				bool ret=test_unitary_decomposer_nq(qnum, type, true);
				ASSERT_TRUE(ret);
			}
		}
		
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
		ASSERT_TRUE(false);
	}

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
