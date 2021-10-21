#include <iostream>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "QPanda.h"
#include <time.h>
#include "gtest/gtest.h"

USING_QPANDA
using namespace std;

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

	//cout << "Grover will search through " << search_sapce.size()<< " data." << endl;

	size_t _i = 0;
	std::vector<size_t> result_index_vec_actual;
	for (const auto &item : search_sapce)
	{
		if (item == SearchDataByUInt(6)){
			result_index_vec_actual.push_back(_i);
		}
		++_i;
	}

	/*cout << "The target result's index:\n";
	for (const auto &result_item : result_index_vec_actual)
	{
		cout << result_item << " ";
	}
	cout << endl;*/

	std::vector<size_t> result_index_vec;
	//cout << "Start grover search algorithm:" << endl;
	QProg grover_Qprog = grover_alg_search_from_vector(search_sapce, x == 6, result_index_vec, machine, 2);
	
	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");
	
	//cout << "The result's index:" << endl;
	for (int i = 0;i < result_index_vec.size(); ++i)
	{
		//cout << result_index_vec[i] << " ";
		if (result_index_vec[i] != result_index_vec_actual[i]) {
			return false;
		}
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
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(11);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);

	for (size_t i = 0; i < 0; i++)
	{
		search_sapce.insert(search_sapce.end(), search_sapce.begin(), search_sapce.end());
	}
	//cout << "Grover will search through " << search_sapce.size() << " data." << endl;

	//test
	std::vector<size_t> result_index_vec_actual;
	size_t _i = 0;
	for (const auto &item : search_sapce)
	{
		if (item == SearchDataByUInt(6)){
			result_index_vec_actual.push_back(_i);
		}
		++_i;
	}

	/*cout << "The target result's index:\n";
	for (const auto &result_item : result_index_vec_actual){
		cout << result_item << " ";
	}
	cout << endl;*/

	//cout << "Start grover search algorithm:" << endl;
	QVec measure_qubits;
	QProg grover_Qprog = build_grover_alg_prog(search_sapce, x == 6, machine, measure_qubits, 1);

	//cout << "grover_Qprog" << grover_Qprog << endl;

	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");

	//measure
	//printf("Strat pmeasure.\n");
	auto result = probRunDict(grover_Qprog, measure_qubits);

	//get result
	auto result_index_vec = search_target_from_measure_result(result, measure_qubits.size());
	
	//cout << "The result's index:\n";
	for (int i = 0; i < result_index_vec.size(); i++)
	{
		//cout << result_index_vec[i] << " ";
		if (result_index_vec[i] != result_index_vec_actual[i]) {
			return false;
		}
	}
	//cout << endl;
	destroyQuantumMachine(machine);
	return true;
}

bool gorver_test_fun3()
{
	auto machine = initQuantumMachine(CPU);
	auto x = machine->allocateCBit();
	std::vector<uint32_t> search_sapce = { 2, 5, 5, 4, 3 };

	//cout << "Grover will search through " << search_sapce.size() << " data." << endl;
	//cout << "Start grover search algorithm:" << endl;
	QVec measure_qubits;
	QProg grover_Qprog = build_grover_prog(search_sapce, x == 5, machine, measure_qubits, 1);
	decompose_multiple_control_qgate(grover_Qprog, machine);

	//test
	std::vector<size_t> result_index_vec_actual;
	size_t _i = 0;
	for (const auto &item : search_sapce)
	{
		if (item == 5) {
			result_index_vec_actual.push_back(_i);
		}
		++_i;
	}

	//measure
	//printf("Measuring...\n");
	auto result = probRunDict(grover_Qprog, measure_qubits);

	//get result
	auto result_index_vec = search_target_from_measure_result(result, measure_qubits.size());

	//cout << "The result's index:\n";
	for (int i = 0; i < result_index_vec.size(); i++)
	{
		//cout << result_index_vec[i] << " ";
		if (result_index_vec[i] != result_index_vec_actual[i]){
			return false;
		}
	}
	//cout << endl;
	destroyQuantumMachine(machine);
	return true;
}

TEST(GroverAlg, test1)
{
	bool test_val = false;
	try
	{
		test_val = gorver_test_fun1();
		test_val = test_val && gorver_test_fun2();
		test_val = test_val && gorver_test_fun3();
	}
	catch (const std::exception& e)
	{
		cout << "Error:Catch an exception: " << e.what() << endl;
		test_val = false;
	}
	catch (...)
	{
		cout << "Error: Catch a unknow exception." << endl;
		test_val = false;
	}

	ASSERT_TRUE(test_val);

	//cout << "\n Grover test over." << endl;
}