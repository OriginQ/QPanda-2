#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>

#define PRINT_TRACE 0

USING_QPANDA
using namespace std;
bool test_decompose_multiple_control_qgate_1()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	/*CPUQVM qvm1;
	qvm1.init();
	auto qvm = &qvm1;*/
	auto q = qvm->allocateQubits(6);
	auto c = qvm->allocateCBits(6);
	/*QVec q;
	vector<ClassicalCondition> c;*/

	/*CPUQVM tmp_cpu;
	tmp_cpu.init();
	auto q2 = tmp_cpu.allocateQubits(16);
	auto c2 = tmp_cpu.allocateCBits(16);*/

	QProg prog1;
	QProg prog2;

	//QProg prog3;
	//prog3 << RY(q[2], 2.5) << X(q[2]).control({ q[0], q[1]/*, q[3], q[4]*/ }) << RY(q[2], -2.5) << X(q[2]).control({ q[0], q[1]/*, q[3], q[4]*/ }) 
	//	/*<< Measure(q[0], c[0]) << Measure(q[1], c[1]) << Measure(q[2], c[2]) << Measure(q[3], c[3]) << Measure(q[4], c[4])*/;

	//QProg prog4 = remap(prog3, { q2[0], q2[1], q2[2], q2[3], q2[4] }, { /*c2[0], c2[1], c2[2], c2[3], c2[4]*/ });
	//QProg prog5 = remap(prog3, { q2[6], q2[7], q2[8], q2[9], q2[10] }, {/* c2[5], c2[6], c2[7], c2[8], c2[9]*/ });
	//prog1 << prog5 << prog4 << prog3;
	//cout << "prog1:" << prog1 << endl;

	//std::vector<ClassicalCondition> cc = { c2[0], c2[1], c2[2], c2[3], c2[4] , c2[5], c2[6], c2[7], c2[8], c2[9] };
	//auto result = tmp_cpu.runWithConfiguration(prog1, cc, 100);

	//prog1 << RZ(q[2], -4.2).control({q[1] });
	//prog2 << RZ(q[2], 2.5) << X(q[2]).control(q[1]) << RZ(q[2], -2.5) << X(q[2]).control(q[1]);
	//prog2 << RZ(q[2], -2.1) << CNOT(q[1], q[2]) << RZ(q[2], 2.1) << CNOT(q[1], q[2]);

	//prog1 << RX(q[2], 18).control({ q[1] });
	//prog1 << RX(q[2], 4).control({ q[1] });
	//prog2 << RX(q[2], 2.5) << Z(q[2]).control(q[1]) << RX(q[2], -2.5) << Z(q[2]).control(q[1]);
	//prog2 << RX(q[2], 9) << CZ(q[1], q[2]) << RX(q[2], -9) << CZ(q[1], q[2]);

	//prog1 << RY(q[2], 18).control({ q[1] });
	//prog2 << RY(q[2], 2.5) << X(q[2]).control(q[1]) << RY(q[2], -2.5) << X(q[2]).control(q[1]);
	//prog2 << RY(q[2], 9) << CZ(q[1], q[2]) << RY(q[2], -9) << CZ(q[1], q[2]);

	prog1 << RX(q[2], 5).control({ q[0], q[1], q[3] });
	prog2 << RX(q[2], 2.5) << Z(q[2]).control({q[0], q[1], q[3]}) << RX(q[2], -2.5) << Z(q[2]).control({ q[0], q[1], q[3] });
	//prog2 << RX(q[2], 2.5) << Y(q[2]).control({ q[0], q[1], q[3] }) << RX(q[2], -2.5) << Y(q[2]).control({ q[0], q[1], q[3] });

	/*prog1 << RY(q[2], 5).control({ q[0], q[1], q[3], q[4] });
	prog2 << RY(q[2], 2.5) << Z(q[2]).control({ q[0], q[1], q[3], q[4] }) << RY(q[2], -2.5) << Z(q[2]).control({ q[0], q[1], q[3], q[4] });*/

	/*prog1 << RZ(q[2], 7).control({ q[0], q[1], q[3], q[4] });
	prog2 << RZ(q[2], 3.5) << X(q[2]).control({ q[0], q[1], q[3], q[4] }) << RZ(q[2], -3.5) << X(q[2]).control({ q[0], q[1], q[3], q[4] });*/
	//prog2 << RZ(q[2], 3.5) << Y(q[2]).control({ q[0], q[1], q[3], q[4] }) << RZ(q[2], -3.5) << Y(q[2]).control({ q[0], q[1], q[3], q[4] });

	//prog1 << X(q[2]).control({q[0], q[1]});
	///*prog2 << H(q[2]) << CNOT(q[1], q[2]) << T(q[2]).dagger() << CNOT(q[0], q[2]) << T(q[2]) << CNOT(q[1], q[2])
	//	<< T(q[1]).dagger() << T(q[2]).dagger() << CNOT(q[0], q[2]) << CNOT(q[0], q[1]) << T(q[2]) << T(q[1]).dagger() << H(q[2])
	//	<< CNOT(q[0], q[1]) << T(q[0]) << S(q[1]);*/
	//prog2 << H(q[2]) << CNOT(q[1], q[2]) << T(q[2]).dagger() << CNOT(q[0], q[2]) << T(q[2]) << CNOT(q[1], q[2])
	//	<< T(q[2]).dagger() << CNOT(q[0], q[2]) << CNOT(q[0], q[1]) << T(q[2]) << T(q[1]).dagger() << H(q[2])
	//	<< CNOT(q[0], q[1]) << T(q[0]) << T(q[1]);


	/*QCircuit toffoli_cir;
	toffoli_cir << H(q[2]) << CNOT(q[1], q[2]) << T(q[2]).dagger() << CNOT(q[0], q[2]) << T(q[2]) << CNOT(q[1], q[2])
		<< T(q[2]).dagger() << CNOT(q[0], q[2]) << CNOT(q[0], q[1]) << T(q[2]) << T(q[1]).dagger() << H(q[2])
		<< CNOT(q[0], q[1]) << T(q[0]) << T(q[1]);*/

	/*prog1 << RX(q[2], 5).control({ q[0], q[1] }); //多控z门如何处理？
	prog2 << RX(q[2], 2.5) << Z(q[2]).control({ q[0], q[1] }) << RX(q[2], -2.5) << Z(q[2]).control({ q[0], q[1] });*/

	//prog1 << RY(q[2], 3).control({ q[0], q[1] });
	////prog2 << RY(q[2], 1.5) << toffoli_cir << RY(q[2], -1.5) << toffoli_cir;
	//prog2 << RY(q[2], 1.5) << X(q[2]).control({ q[0], q[1] }) << RY(q[2], -1.5) << X(q[2]).control({ q[0], q[1] });

	decompose_multiple_control_qgate(prog1, qvm);
	decompose_multiple_control_qgate(prog2, qvm);  

	//QProg test_prog = convert_originir_to_qprog("E:\\test_prog3.ir", qvm, q, c);
	/*QCircuit cir1;
	cir1 << RZ(q[2], 5).dagger() << RY(q[2], 5.5) << RX(q[2], 2.5) << U4(2,3,4,5, q[2]).dagger();
	cir1.setDagger(true); 
	cir1.setControl(q[1]);
	prog1 << cir1 << RX(q[2], 5).dagger();
	prog2 << cir1 << RX(q[2], 5).dagger();*/
	/*prog1 << test_prog;
	prog2 << test_prog;*/

	/*prog1 << RX(q[0], 5);
	prog2 << RX(q[0], 5);*/
	const QStat result_mat1 = getCircuitMatrix(prog1/*, true*/);
	//sub_cir_optimizer(prog2, {}, QCircuitOPtimizerMode::Merge_U3);
	//sub_cir_optimizer(prog1, {}, QCircuitOPtimizerMode::Merge_U3);

	//const QStat result_mat1 = getCircuitMatrix(prog1/*, true*/);
	const QStat result_mat2 = getCircuitMatrix(prog2/*, true*/);

	cout << "prog1:" << prog1 << endl;
	cout << "result_mat1" << result_mat1 << endl;

	cout << "prog2:" << prog2 << endl;
	cout << "result_mat2" << result_mat2 << endl;

	auto prog1_gate_size = getQGateNum(prog1);
	auto prog2_gate_size = getQGateNum(prog2);
	cout << "prog1_gate_size = " << prog1_gate_size << ", prog2_gate_size = " << prog2_gate_size << endl;
	if (result_mat1 == result_mat2)
	{
		cout << "==============" << endl;
	}
	else
	{
		cout << "!!!!!!!!!!!!" << endl;
	}

	return true;
}

static bool test_decompose_swap_gate()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(6);
	// auto c = qvm->allocateCBits(6);

	QCircuit cir1;
	QCircuit cir2;
	cir1 << SWAP(q[0], q[1]);
	cir2 << H(q[1]) <<CZ(q[1], q[0]) << H(q[1])
		<< H(q[0]) << CZ(q[0], q[1]) << H(q[0])
		<< H(q[1]) << CZ(q[1], q[0]) << H(q[1]);

	QCircuit cir3;
	cir3 << U3(q[1], PI/2.0, 0, PI) << CZ(q[1], q[0]) << U3(q[1], PI / 2.0, 0, PI)
		<< U3(q[0], PI / 2.0, 0, PI) << CZ(q[0], q[1]) << U3(q[0], PI / 2.0, 0, PI)
		<< U3(q[1], PI / 2.0, 0, PI) << CZ(q[1], q[0]) << U3(q[1], PI / 2.0, 0, PI);
	auto mat1 = getCircuitMatrix(cir1);
	auto mat2 = getCircuitMatrix(cir3);
	if (0 == mat_compare(mat1, mat2, 1e-10))
	{
		cout << "oKKKKKKKKKKKKK" << endl;
	}

	decompose_multiple_control_qgate(cir2, qvm);
	write_to_originir_file(cir2, qvm, "D://swap.ir");

	cout << cir2 << endl;

	return true;
}

TEST(MultipleControlGateDecompose, test1)
{
	bool test_val = false;
	try
	{
		//test_val = test_decompose_multiple_control_qgate_1();
		test_val = test_decompose_swap_gate();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "MultipleControlGateDecompose test over, press Enter to continue." << endl;
	getchar();

	ASSERT_TRUE(test_val);
}