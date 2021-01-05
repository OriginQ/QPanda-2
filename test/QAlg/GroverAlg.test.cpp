#include <iostream>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "QPanda.h"
#include <time.h>
#include "gtest/gtest.h"

USING_QPANDA
using namespace std;

bool gorver_test_fun1();
bool gorver_test_fun2();
bool gorver_test_fun3();
bool gorver_test_fun333();
bool gorver_test_fun22();

TEST(GroverAlg, test1)
{
	bool test_val = false;
	try
	{
		//test_val = gorver_test_fun1();
		test_val = gorver_test_fun2();
		//test_val = gorver_test_fun3();
		//test_val = gorver_test_fun22();
		//test_val = gorver_test_fun333();
	}
	catch (const std::exception& e)
	{
		cout << "Error:Catch an exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Error: Catch a unknow exception." << endl;
	}

	ASSERT_TRUE(test_val);

	cout << "\n Grover test over, press Enter to continue..." << endl;
	getchar();
}

bool gorver_test_fun1()
{
	auto machine = initQuantumMachine(CPU);
	auto x = machine->allocateCBit();
	std::vector<SearchDataByUInt> search_sapce;
	search_sapce.push_back(8);
	search_sapce.push_back(7);
	search_sapce.push_back(6);
	search_sapce.push_back(0);
	search_sapce.push_back(6);
	search_sapce.push_back(3);
	search_sapce.push_back(6);
	search_sapce.push_back(4);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(7);
	search_sapce.push_back(14);
	search_sapce.push_back(9);
	search_sapce.push_back(12);
	search_sapce.push_back(4);
	search_sapce.push_back(9);
	search_sapce.push_back(9);
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(3);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(11);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);

	/*for (size_t i = 0; i < 5; i++)
	{
		search_sapce.insert(search_sapce.end(), search_sapce.begin() + 20, search_sapce.end());
	}*/
	cout << "Grover will search through " << search_sapce.size()<< " data." << endl;

	std::vector<size_t> result_index_vec;

	//test
	size_t indexx = 0;
	for (const auto &item : search_sapce)
	{
		if (item == SearchDataByUInt(6))
		{
			result_index_vec.push_back(indexx);
		}
		++indexx;
	}

	cout << "The target result's index:" << endl;
	for (const auto &result_item : result_index_vec)
	{
		cout << result_item << " ";
	}
	cout << endl;
	result_index_vec.clear();

	cout << "Start grover search algorithm:" << endl;
	QProg grover_Qprog = grover_alg_search_from_vector(search_sapce, x == 6, result_index_vec, machine, 2);

	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");

	cout << "The result's index:" << endl;
	for (const auto &result_item : result_index_vec)
	{
		cout << result_item << " ";
	}

	destroyQuantumMachine(machine);
	return true;
}

bool gorver_test_fun2()
{
	auto machine = initQuantumMachine(CPU);
	auto q = machine->allocateQubits(2);
	auto x = machine->allocateCBit();
	std::vector<SearchDataByUInt> search_sapce;
	search_sapce.push_back(8);
	search_sapce.push_back(7);
	search_sapce.push_back(6);
	search_sapce.push_back(0);
	search_sapce.push_back(6);
	search_sapce.push_back(3);
	search_sapce.push_back(6);
	search_sapce.push_back(4);
	/*search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(7);
	search_sapce.push_back(14);
	search_sapce.push_back(9);
	search_sapce.push_back(12);
	search_sapce.push_back(4);
	search_sapce.push_back(9);
	search_sapce.push_back(9);
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(3);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7)*/;
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(11);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);

	for (size_t i = 0; i < 0; i++)
	{
		search_sapce.insert(search_sapce.end(), search_sapce.begin() + 20, search_sapce.end());
	}
	cout << "Grover will search through " << search_sapce.size() << " data." << endl;

	/*for (auto aiter : search_sapce)
	{
		printf("%d, ", aiter.get_val());
	}*/

	//test
	/*size_t indexx = 0;
	for (const auto &item : search_sapce)
	{
		if (item == SearchDataByUInt(6))
		{
			result_index_vec.push_back(indexx);
		}
		++indexx;
	}

	cout << "The target result's index:" << endl;
	for (const auto &result_item : result_index_vec)
	{
		cout << result_item << " ";
	}
	cout << endl;
	result_index_vec.clear();*/

	cout << "Start grover search algorithm:" << endl;
	QVec measure_qubits;
	QProg grover_Qprog = build_grover_alg_prog(search_sapce, x == 6, machine, measure_qubits, 1);

	/*QProg test_prog;
	test_prog << H(q[0]) << H(q[1]) << X(q[0]) << X(q[0]) << X(q[1]) << H(q[1]) << H(q[0]) << X(q[0]) << X(q[0]) << H(q[0]);*/
	//std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec;
	sub_cir_optimizer(grover_Qprog/*, optimizer_cir_vec*/);

	cout << "grover_Qprog" << grover_Qprog << endl;

	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");

	/*QVec qv;
	get_all_used_qubits(grover_Qprog, qv);*/
	//quantum_chip_adapter(grover_Qprog, machine, qv);
	//cout << "grover_Qprog:" << endl;
	//cout << grover_Qprog << endl;

	//measure
	printf("Strat pmeasure.\n");
	auto result = probRunDict(grover_Qprog, measure_qubits);

	//get result
	auto result_index_vec = search_target_from_measure_result(result, search_sapce.size());
	cout << "The result's index:" << endl;
	for (const auto &result_item : result_index_vec)
	{
		cout << result_item << " ";
	}

	destroyQuantumMachine(machine);
	return true;
}

bool gorver_test_fun22()
{
	auto machine = initQuantumMachine(CPU);
	auto q = machine->allocateQubits(2);
	auto x = machine->allocateCBit();
	//std::vector<int> search_sapce = {3, 6, 6, 9, 10, 15, 11, 5};
	std::vector<int> search_sapce = { 7, 14, 14, 20, 26 };

	cout << "Grover will search through " << search_sapce.size() << " data." << endl;

	

	cout << "Start grover search algorithm:" << endl;
	QVec measure_qubits;
	QProg grover_Qprog = build_grover_prog(search_sapce, x == 14, machine, measure_qubits, 1);

	/*QProg test_prog;
	test_prog << H(q[0]) << H(q[1]) << X(q[0]) << X(q[0]) << X(q[1]) << H(q[1]) << H(q[0]) << X(q[0]) << X(q[0]) << H(q[0]);*/
	//std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec;
	//sub_cir_optimizer(grover_Qprog/*, optimizer_cir_vec*/);

	cout << "grover_Qprog" << grover_Qprog << endl;

	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");


	//measure
	printf("Strat pmeasure.\n");
	auto result = probRunDict(grover_Qprog, measure_qubits);

	//get result
	auto result_index_vec = search_target_from_measure_result(result, search_sapce.size());

	cout << "The result's index:" << endl;
	for (const auto &result_item : result_index_vec)
	{
		cout << result_item << " ";
	}

	destroyQuantumMachine(machine);
	return true;
}

static QCircuit index_to_circuit(QVec index_qubits, size_t index) {
	QCircuit ret_cir;
	for (size_t i = 0; i < index_qubits.size(); ++i)
	{
		if (0 == index % 2)
		{
			ret_cir << X(index_qubits[i]);
		}

		index /= 2;
	}

	return ret_cir;
}

static QCircuit data_to_cir(QVec &oracle_qubits, int data)
{
	QCircuit ret_cir;
	for (size_t i = 0; i < oracle_qubits.size(); ++i)
	{
		if (0 != data % 2)
		{
			ret_cir << X(oracle_qubits[i]);
		}

		data /= 2;
	}

	return ret_cir;
}

static QCircuit build_condition_circuit(QVec &oracle_qubits, Qubit* ancilla_qubit, int search_data)
{
	QCircuit ret_cir;
	auto ancilla_gate = X(ancilla_qubit);
	ancilla_gate.setControl(oracle_qubits);

	QCircuit search_cir;
	for (size_t i = 0; i < oracle_qubits.size(); ++i)
	{
		if (0 == search_data % 2)
		{
			search_cir << X(oracle_qubits[i]);
		}

		search_data /= 2;
	}

	ret_cir << search_cir << ancilla_gate << search_cir;
	//PTraceQCircuit("ret_cir", ret_cir);
	return ret_cir;
}

static QCircuit build_diffusion_circuit(const QVec &qvec) 
{
	vector<Qubit*> controller(qvec.begin(), --(qvec.end()));
	QCircuit c;
	c << apply_QGate(qvec, H);
	c << apply_QGate(qvec, X);
	c << Z(qvec.back()).control(controller);
	c << apply_QGate(qvec, X);
	c << apply_QGate(qvec, H);

	return c;
}

bool gorver_test_fun3()
{
	auto machine = initQuantumMachine(CPU);
	auto q_a = machine->allocateQubits(4);
	auto q_b = machine->allocateQubits(4);
	auto q_k = machine->allocateQubits(5);
	auto q_d = machine->allocateQubits(8);
	auto q_anc = machine->allocateQubit();
	QProg prog;
	
	prog << applyQGate(q_a, H) << applyQGate(q_b, H)/* << X(q_anc) << H(q_anc)*/;
	//prog << bind_data(3, q_a) << bind_data(5, q_b);

	QCircuit multip_cir;
	multip_cir << QMultiplier(q_a, q_b, q_k, q_d);
	cout << "multip_cir:" << endl;
	cout << multip_cir << endl;

	QVec q_dd;
	for (auto q_d_itr = q_d.rbegin(); q_d_itr != q_d.rend(); ++q_d_itr)
	{
		q_dd.push_back(*q_d_itr);
	}

	QVec& q_multip_result_qubit = q_d;
	QCircuit search_data_cir = build_condition_circuit(q_multip_result_qubit, q_anc, 15);
	cout << "search_data_cir:" << endl;
	cout << search_data_cir << endl;
	cout << "press enter to continue:" << endl;
	getchar();

	auto measure_qubits = q_a + q_b;
	QCircuit diffu_cir = build_diffusion_circuit(measure_qubits);
	cout << "diffu_cir:" << endl;
	cout << diffu_cir << endl;
	cout << "press enter to continue:" << endl;
	getchar();

	QCircuit rsa_cir;
	rsa_cir << multip_cir << search_data_cir /*<< search_data_cir.dagger() << multip_cir.dagger() << diffu_cir*/;

	prog << rsa_cir /*<< rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir*/;
	cout << "prog:" << endl;
	cout << prog << endl;

	//measure
	
	printf("Strat pmeasure.\n");
	//auto result = probRunDict(prog, measure_qubits);
	auto result = probRunDict(prog, { q_anc });
	//auto result = probRunDict(prog, q_multip_result_qubit);

	//get result
	double total_val = 0.0;
	for (auto& var : result) { total_val += var.second; }
	const double average_probability = total_val / result.size();
	size_t search_result_index = 0;

	std::vector<int> result_index_vec;
	printf("pmeasure result:\n");
	for (auto aiter : result)
	{
		printf("%s:%5f\n", aiter.first.c_str(), aiter.second);
		if (average_probability < aiter.second)
		{
			result_index_vec.push_back(search_result_index);
		}
		++search_result_index;
	}

	printf("result index:\n");
	for (auto i : result_index_vec)
	{
		printf("%d, ", i);
	}

	destroyQuantumMachine(machine);
	return true;
}

bool gorver_test_fun333()
{
	auto machine = initQuantumMachine(CPU);
	auto q_index = machine->allocateQubits(4);
	auto q_data = machine->allocateQubits(3);
	auto q_anc = machine->allocateQubit();
	QProg prog;

	prog << applyQGate(q_index, H);

	QCircuit cir_data_1 = data_to_cir(q_data, 1);
	QCircuit cir_data_3 = data_to_cir(q_data, 3);
	QCircuit cir_data_5 = data_to_cir(q_data, 5);

	cir_data_1.setControl(q_index);
	cir_data_3.setControl(q_index);
	cir_data_5.setControl(q_index);
	QCircuit index_0 = index_to_circuit(q_index, 0);
	QCircuit index_1 = index_to_circuit(q_index, 1);
	QCircuit index_2 = index_to_circuit(q_index, 2);
	QCircuit data_cir;
	data_cir << index_0 << cir_data_1 << index_0
		<< index_1 << cir_data_3 << index_1
		<< index_2 << cir_data_5 << index_2;

	QVec q_dd;
	for (auto q_d_itr = q_data.rbegin(); q_d_itr != q_data.rend(); ++q_d_itr)
	{
		q_dd.push_back(*q_d_itr);
	}

	QVec& q_multip_result_qubit = q_data;
	QCircuit search_data_cir = build_condition_circuit(q_multip_result_qubit, q_anc, 5);
	cout << "search_data_cir:" << endl;
	cout << search_data_cir << endl;
	cout << "press enter to continue:" << endl;
	getchar();


	QCircuit rsa_cir;
	rsa_cir << data_cir << search_data_cir /*<< search_data_cir.dagger() << multip_cir.dagger() << diffu_cir*/;

	prog << rsa_cir /*<< rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir*/;
	cout << "prog:" << endl;
	cout << prog << endl;

	//measure

	printf("Strat pmeasure.\n");
	//auto result = probRunDict(prog, measure_qubits);
	auto result = probRunDict(prog, { q_anc });
	//auto result = probRunDict(prog, q_multip_result_qubit);

	//get result
	double total_val = 0.0;
	for (auto& var : result) { total_val += var.second; }
	const double average_probability = total_val / result.size();
	size_t search_result_index = 0;

	std::vector<int> result_index_vec;
	printf("pmeasure result:\n");
	for (auto aiter : result)
	{
		printf("%s:%5f\n", aiter.first.c_str(), aiter.second);
		if (average_probability < aiter.second)
		{
			result_index_vec.push_back(search_result_index);
		}
		++search_result_index;
	}

	printf("result index:\n");
	for (auto i : result_index_vec)
	{
		printf("%d, ", i);
	}

	destroyQuantumMachine(machine);
	return true;
}

bool gorver_test_fun4()
{
	auto machine = initQuantumMachine(CPU);
	auto q_a_index = machine->allocateQubits(3);
	auto q_b_index = machine->allocateQubits(3);
	auto q_a = machine->allocateQubits(3);
	auto q_b = machine->allocateQubits(3);
	auto q_k = machine->allocateQubits(4);
	auto q_d = machine->allocateQubits(5);
	auto q_anc = machine->allocateQubit();
	QProg prog;

	prog << applyQGate(q_a, H) << applyQGate(q_b, H) << X(q_anc) << H(q_anc);

	QCircuit multip_cir;
	multip_cir << QMultiplier(q_a, q_b, q_k, q_d);
	cout << "multip_cir:" << endl;
	cout << multip_cir << endl;

	QVec q_dd;
	for (auto q_d_itr = q_d.rbegin(); q_d_itr != q_d.rend(); ++q_d_itr)
	{
		q_dd.push_back(*q_d_itr);
	}

	QCircuit search_data_cir = build_condition_circuit(q_d, q_anc, 15);
	cout << "search_data_cir:" << endl;
	cout << search_data_cir << endl;
	cout << "press enter to continue:" << endl;
	getchar();

	QCircuit rsa_cir;
	rsa_cir << multip_cir << search_data_cir << search_data_cir.dagger() << multip_cir.dagger();

	prog << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir << rsa_cir;
	cout << "prog:" << endl;
	cout << prog << endl;

	//measure
	auto measure_qubits = q_a + q_b;
	printf("Strat pmeasure.\n");
	auto result = probRunDict(prog, measure_qubits);

	//get result
	double total_val = 0.0;
	for (auto& var : result) { total_val += var.second; }
	const double average_probability = total_val / result.size();
	size_t search_result_index = 0;

	std::vector<int> result_index_vec;
	printf("pmeasure result:\n");
	for (auto aiter : result)
	{
		printf("%s:%5f\n", aiter.first.c_str(), aiter.second);
		if (average_probability < aiter.second)
		{
			result_index_vec.push_back(search_result_index);
		}
		++search_result_index;
	}

	printf("result index:\n");
	for (auto i : result_index_vec)
	{
		printf("%d, ", i);
	}

	destroyQuantumMachine(machine);
	return true;
}