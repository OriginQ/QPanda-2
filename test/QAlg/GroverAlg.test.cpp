#include <iostream>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "QPanda.h"
#include <time.h>
#include "gtest/gtest.h"

USING_QPANDA
using namespace std;

#pragma comment(lib, "lib\\QAlg.lib")
#pragma comment(lib, "lib\\Components.lib")

bool gorver_test_fun1();
bool gorver_test_fun2();

TEST(GroverAlg, test1)
{
	bool test_val = false;
	try
	{
		//test_val = gorver_test_fun1();
		test_val = gorver_test_fun2();
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
	auto x = machine->allocateCBit();
	std::vector<SearchDataByUInt> search_sapce;
	search_sapce.push_back(8);
	search_sapce.push_back(7);
	search_sapce.push_back(6);
	search_sapce.push_back(0);
	/*search_sapce.push_back(6);
	search_sapce.push_back(3);
	search_sapce.push_back(6);
	search_sapce.push_back(4);*/
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
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(11);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);*/

	for (size_t i = 0; i < 0; i++)
	{
		search_sapce.insert(search_sapce.end(), search_sapce.begin() + 20, search_sapce.end());
	}
	cout << "Grover will search through " << search_sapce.size() << " data." << endl;

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

	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");

	QVec qv;
	get_all_used_qubits(grover_Qprog, qv);
	//quantum_chip_adapter(grover_Qprog, machine, qv);
	//cout << "grover_Qprog:" << endl;
	//cout << grover_Qprog << endl;

	//measure
	PTrace("Strat pmeasure.\n");
	auto result = probRunDict(grover_Qprog, qv);

	//get result
	/*double total_val = 0.0;
	for (auto& var : result) { total_val += var.second; }
	const double average_probability = total_val / result.size();
	size_t search_result_index = 0;*/

	PTrace("pmeasure result:\n");
	for (auto aiter : result)
	{
		PTrace("%s:%5f\n", aiter.first.c_str(), aiter.second);
	}

	destroyQuantumMachine(machine);
	return true;
}