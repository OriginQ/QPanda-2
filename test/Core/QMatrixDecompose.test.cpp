#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>

USING_QPANDA

bool test_matrix_decompose_1()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(8);
	auto c = qvm->allocateCBits(8);

	QStat target_matrix = { qcomplex_t(0.6477054522122977, 0.1195417767870219), qcomplex_t(-0.16162176706189357, -0.4020495632468249), qcomplex_t(-0.19991615329121998, -0.3764618308248643), qcomplex_t(-0.2599957197928922, -0.35935248873007863),
		qcomplex_t(-0.16162176706189363, -0.40204956324682495), qcomplex_t(0.7303014482204584, -0.4215172444390785), qcomplex_t(-0.15199187936216693, 0.09733585496768032), qcomplex_t(-0.22248203136345918, -0.1383600597660744),
		qcomplex_t(-0.19991615329122003, -0.3764618308248644), qcomplex_t(-0.15199187936216688, 0.09733585496768032), qcomplex_t(0.6826630277354306, -0.37517063774206166), qcomplex_t(-0.3078966462928956, -0.2900897445133085),
		qcomplex_t(-0.2599957197928923, -0.3593524887300787), qcomplex_t(-0.22248203136345912, -0.1383600597660744), qcomplex_t(-0.30789664629289554, -0.2900897445133085), qcomplex_t(0.6640994547408099, -0.338593803336005) };
	
	auto cir = matrix_decompose({q[0], q[1]}, target_matrix);
	cout << "decomposed circuit:" << cir << endl;

	destroyQuantumMachine(qvm);
	return true;
}

TEST(QMatrixDecompose, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_matrix_decompose_1();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "QMatrixDecompose test over, press Enter to continue." << endl;
	getchar();

	ASSERT_TRUE(test_val);
}