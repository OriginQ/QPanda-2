#include "QPanda.h"
#include "gtest/gtest.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"

using namespace std;
USING_QPANDA

static size_t g_shot = 1e5;

bool test_cir_optimize_fun1() 
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(4);
	auto c = qvm->allocateCBits(4);

	QCircuit cir;
	QCircuit cir2;

	QProg prog;
	cir /*<< CU(1, 2, 3, 4, q[1], q[0])*/ /*<< (H(q[1])) << X(q[2]) *//*<< RZ(q[1], PI / 2) << Y(q[2])*/
		<< (CR(q[0], q[3], PI / 2)) /*<< (S(q[2])) << S(q[1]) << RZ(q[1], PI / 2) */
		/*<< RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2)*/ << Y(q[0]) /*<< SWAP(q[3], q[1])*/
		/*<< CU(1, 2, 3, 4, q[1], q[0])*/ << (H(q[1])) << X(q[2]) /*<< RX(q[1], PI / 2) << RX(q[1], PI / 2)*/ << Y(q[2])
		<< CR(q[2], q[3], PI / 2) /*<< CU(1, 2, 3, 4, q[1], q[0])*/ << (H(q[1])) /*<< X(q[2]) << RZ(q[1], PI / 2) << Y(q[2])*/;
	cir2 << H(q[1]) << X(q[2]) << X(q[2])/* << H(q[1]) << X(q[3]) << X(q[3])*/;

	QCircuit cir3;
	QCircuit cir4;
	cir3 << H(q[1]) << H(q[2]) << CNOT(q[2], q[1]) << H(q[1]) << H(q[2]);
	cir4 << CNOT(q[1], q[2]);

	QCircuit cir5;
	QCircuit cir6;
	double theta_1 = PI / 3.0;
	cir5 << RZ(q[3], PI / 2.0) << CZ(q[3], q[0]) << RX(q[3], PI / 2.0) << RZ(q[3], theta_1) << RX(q[3], -PI / 2.0) << CZ(q[3], q[0]) << RZ(q[3], -PI / 2.0);
	cir6 << CZ(q[3], q[0]) << H(q[3]) << RZ(q[3], theta_1) << H(q[3]) << CZ(q[3], q[0]);

	std::vector<std::pair<QCircuit, QCircuit>> optimitzer_cir;
	QCircuitOptimizerConfig config_reader;
	config_reader.get_replace_cir(optimitzer_cir);
	/*for (auto cir_item : optimitzer_cir)
	{
		cout << "target cir:" << endl << cir_item.first << endl;
		cout << "replaceed cir:" << endl << cir_item.second << endl;
	}*/

	prog << cir << cir2 << Reset(q[1]) << cir3 << cir5 << MeasureAll(q, c);
	cout << "src QProg:" << endl;
	cout << prog << endl;
	{
		printf("Measure result for src quantum program:\n");
		auto result = runWithConfiguration(prog, c, g_shot);
		for (const auto& _r : result) {
			printf("%s:%5f\n", _r.first.c_str(), (double)_r.second / (double)g_shot);
		}
	}

	cir_optimizer(prog, optimitzer_cir, QCircuitOPtimizerMode::Merge_H_X);

	//prog << cir << cir2 << Reset(q[1]) << cir4 << cir6 << MeasureAll(q, c);

	cout << "The optimizered QProg:" << endl;
	cout << prog << endl;
	{
		printf("Measure result for src optimizered-quantum-program:\n");
		auto result = runWithConfiguration(prog, c, g_shot);
		for (const auto& _r : result) {
			printf("%s:%5f\n", _r.first.c_str(), (double)_r.second / (double)g_shot);
		}
	}

	destroyQuantumMachine(qvm);
	return true;
}

bool test_cir_optimize_fun11()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(4);
	auto c = qvm->allocateCBits(4);

	QCircuit cir7;
	QCircuit cir8;
	cir7 << H(q[0]) << CNOT(q[1], q[0]) << H(q[0]);
	cir8 << CZ(q[1], q[0]);

	std::vector<std::pair<QCircuit, QCircuit>> optimitzer_cir;
	optimitzer_cir.push_back(make_pair(cir7, cir8));
	for (auto cir_item : optimitzer_cir)
	{
		cout << "target cir:" << endl << cir_item.first << endl;
		cout << "replaceed cir:" << endl << cir_item.second << endl;
	}

	QProg prog;
	prog << H(q[0])<< H(q[2])<< H(q[3])<< CNOT(q[1], q[0])<< H(q[0])
		<< CNOT(q[1], q[2])<< H(q[2])<< CNOT(q[2], q[3])<< H(q[3]);
	cout << "befort optimizered QProg:" << endl;
	cout << prog << endl;

	sub_cir_replace(prog, optimitzer_cir);

	cout << "The optimizered QProg:" << endl;
	cout << prog << endl;

	destroyQuantumMachine(qvm);
	return true;
}

bool test_cir_optimize_fun2()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(4);
	auto c = qvm->allocateCBits(4);

	QCircuit cir;
	QCircuit cir2;

	QProg prog;
	cir /*<< CU(1, 2, 3, 4, q[1], q[0])*/ /*<< (H(q[1])) << X(q[2]) << RZ(q[1], PI / 2) << Y(q[2])*/
		/*<< (CR(q[0], q[3], PI / 2)) << (S(q[2])) << S(q[1]) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2) << RZ(q[1], PI / 2)
		<< RZ(q[1], PI / 2) << Y(q[0]) << SWAP(q[3], q[1])
		<< CU(1, 2, 3, 4, q[1], q[0]) << (H(q[1])) << X(q[2]) << RX(q[1], PI / 2) << RX(q[1], PI / 2) << Y(q[2])*/
		/*<< CR(q[2], q[3], PI / 2) << CU(1, 2, 3, 4, q[1], q[0])*/ << (H(q[1])) << X(q[2]) << RZ(q[1], PI / 2) /*<< Y(q[2])*/;
	cir2 << H(q[1]) << X(q[2]) << X(q[2]) << H(q[1]) << X(q[3]) << X(q[3]);

	QCircuit cir3;
	cir3 << H(q[1]) << H(q[2]) << CNOT(q[2], q[1]) << H(q[1]) << H(q[2]);

	QCircuit cir5;
	double theta_1 = PI / 3.0;
	cir5 << RZ(q[3], PI / 2.0).dagger() << CZ(q[0], q[3]).dagger() << RX(q[3], PI / 2.0).dagger() 
		<< RZ(q[3], theta_1) << RX(q[3], -PI / 2.0) << CZ(q[3], q[0]) << RZ(q[3], -PI / 2.0);
	cir5.setDagger(true);
	//cout << "cir5" << cir5 << endl;
	prog << cir << cir2 /*<< Reset(q[1])*/ << cir3 << cir5 << cir.dagger() << cir.dagger()/*<< MeasureAll(q, c)*/;
	cout << "prog" << prog << endl;
	//draw_qprog(prog, 0, true);
	const auto src_mat = getCircuitMatrix(prog);
	cout << "src_mat:" << endl << src_mat << endl;

	single_gate_optimizer(prog, QCircuitOPtimizerMode::Merge_U3);

	const auto result_mat = getCircuitMatrix(prog);
	cout << "result_mat:" << endl << result_mat << endl;

	cout << "The optimizered QProg:" << endl;
	cout << prog << endl;

	destroyQuantumMachine(qvm);

	if (src_mat == result_mat)
	{
		cout << "//////////////////////// right,\\\\\\\\\\\\\\\\\\\\\\'" << endl;
	}
	else
	{
		cout << "----------------- wrong-------------" << endl;
		return false;
	}

	return true;
}

static bool test_cir_optimize_3() {
	bool ret = true;
	auto qvm = initQuantumMachine(QMachineType::CPU);
	auto q = qvm->allocateQubits(2);
	auto c = qvm->allocateCBits(2);
	QProg prog;
	QCircuit cir;
	QCircuit cir2;
	QCircuit cir3;

	prog << U3(q[0], 1.570796, 4.712389, 1.570796) /*<< U2(q[0], 1.570796, -3.141593)*/
		/*<< BARRIER(q[0])*/ /*<< RY(q[0], PI/2.0)*/ << RPhi(q[0], -PI / 2.0, PI / 2.0);

	const auto mat1 = getCircuitMatrix(prog);
	cout << "mat1:" << mat1 << endl;

	cout << "The source QProg:" << endl;
	cout << prog << endl;

	cir_optimizer(prog, std::vector<std::pair<QCircuit, QCircuit>>(), QCircuitOPtimizerMode::Merge_U3);
	cout << " transfer_to_u3_gate " << prog << endl;

	auto mat2 = getCircuitMatrix(prog);
	cout << "mat2:" << mat2 << endl;
	if (mat1 == mat2)
	{
		cout << "oKKKKKKKKKKKKKKKKKKKK" << endl;
	}
	else
	{
		cout << "EEEErrorrrrrrrrrrrr" << endl;
	}

	auto result = runWithConfiguration(prog, c, 100);
	//auto result = probRunDict(prog, q);
	for (auto &val : result)
	{
		std::cout << val.first << ", " << val.second << std::endl;
	}

	destroyQuantumMachine(qvm);
	return ret;
}

static bool test_cir_optimize_4()
{
	auto qvm = initQuantumMachine(QMachineType::CPU);
	qvm->setConfigure({ 128,128 });
	QVec q;
	vector<ClassicalCondition> c;

	QProg prog = convert_originir_to_qprog("E://HHL_prog-0602.ir", qvm, q, c);
	//QProg prog = convert_originir_to_qprog("E://QPE_prog0601-errrr.ir", qvm, q, c);
	//cout << "src prog:" << prog << endl;

	/*const auto mat1 = getCircuitMatrix(prog);
	cout << "mat1:" << mat1 << endl;*/

	{
		directlyRun(prog);
		//auto stat = qvm->getQState();
		auto result2 = getProbDict(q);

		cout << "src qpe stat:\n";
		for (auto &val : result2)
		{
			cout << val.second << "\n";
		}
		cout << endl;
	}

	decompose_multiple_control_qgate(prog, qvm);
	/*cout << "after u3 prog:" << prog << endl;
	const auto mat2 = getCircuitMatrix(prog);
	cout << "mat2:" << mat2 << endl;*/

	//single_gate_optimizer(prog, QCircuitOPtimizerMode::Merge_U3);
	//write_to_originir_file(prog, machine, "E://HHL_prog.ir");
	/*if (0 != mat_compare(mat1, mat2, 1e-10))
	{
		cout << "0KKKKKKKKKKK" << endl;
	}
	else
	{
		cout << "FFFFFFFFFFFFF" << endl;
	}*/

	PTrace("quantum circuit is running ...");
	//auto start = chrono::system_clock::now();
	directlyRun(prog);
	//auto stat = qvm->getQState();
	auto result2 = getProbDict(q);
	/*auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	PTrace("run HHL used: "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< " s");*/

	cout << "u3-qpe stat:\n";
	for (auto &val : result2)
	{
		cout << val.second << "\n";
	}
	cout << endl;

	qvm->finalize();
	return true;
}

TEST(QCircuitOptimizer, test1)
{
	bool test_val = false;
	try
	{
		//test_val = test_cir_optimize_fun1();
		//test_val = test_cir_optimize_fun11();
		//test_val = test_cir_optimize_fun2();
		//test_val = test_cir_optimize_3();
		test_val = test_cir_optimize_4();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "QCircuitOptimizer test over, press Enter to continue." << endl;
	getchar();

	ASSERT_TRUE(test_val);
}