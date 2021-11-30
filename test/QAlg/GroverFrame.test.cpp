#include "QAlg/Grover/GroverFrame.h"
#include "Core/Utilities/Tools/OriginCollection.h"
#include "Core/Utilities/Tools/Uinteger.h"
#include "QAlg/Oracle/SearchSpace.h"
#include "QPanda.h"
#include <time.h>
#include "gtest/gtest.h"
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<cmath>



USING_QPANDA
using namespace std;


bool gorver_test1()
{
	auto qvm = initQuantumMachine();
	auto qubits = qvm->qAllocMany(3);
	auto cbits = qvm->cAllocMany(3);

	QCircuit circuit_prepare;

	// test0 (No input)
	// test1 (All Hadamard)
	circuit_prepare = H(qubits);

	std::vector<std::string> mark_data;
	mark_data = { "010","011" };  // Too many mark_data will fail

	QCircuit state_prepare;
	state_prepare.is_empty();
	QProg prog = grover_search(circuit_prepare, state_prepare, mark_data, qubits, qubits, qvm);
	//std::cout << prog << std::endl;

	// Probability measurement(No need to add measure door)
	auto result = probRunDict(prog, qubits);
	auto result_index = search_target_from_measure_result(result, qubits.size());

	//std::cout << "Probability" << std::endl;
	//for (auto& val : result_index)
	//{
	//	//std::cout << val.first << " : " << val.second << std::endl;
	//	std::cout << val << std::endl;
	//}

	std::vector<size_t> result_index_actual = { 2 ,3 };
	for (int i = 0; i < result_index.size(); i++)
	{
		//cout << result_index_actual[i] << " ";
		//cout << index_actual[i] << " ";
		if (result_index[i] != result_index_actual[i])
		{
			return false;
		}
	}

	destroyQuantumMachine(qvm);
	return true;
}


bool gorver_test2()
{
	auto machine = initQuantumMachine(CPU);
	auto x = machine->allocateCBit();

	// TEST2(state of quantum)
	std::vector<SearchDataByUInt> search_sapce;
	search_sapce.push_back(2);
	search_sapce.push_back(6);
	search_sapce.push_back(2);
	search_sapce.push_back(6);
	search_sapce.push_back(5);
	search_sapce.push_back(1);  //5
	search_sapce.push_back(4);
	search_sapce.push_back(3);
	search_sapce.push_back(3);
	//search_sapce.push_back(4);
	//search_sapce.push_back(5);  // 10
	//search_sapce.push_back(1);
	//search_sapce.push_back(4);
	//search_sapce.push_back(4);
	//search_sapce.push_back(15);
	//search_sapce.push_back(20);  // 15
	//search_sapce.push_back(4);
	//search_sapce.push_back(4);
	//search_sapce.push_back(15);
	//search_sapce.push_back(20);
	//search_sapce.push_back(1);   // 20
	//search_sapce.push_back(1);
	//search_sapce.push_back(10);
	//search_sapce.push_back(30);

	SearchSpace<SearchDataByUInt> grover_search_input(machine, x);
	QCircuit grover_Qprog = grover_search_input.build_to_circuit(search_sapce);
	auto data_qubits = grover_search_input.get_data_qubits();
	auto inedx_qubits = grover_search_input.get_index_qubits();
	QCircuit grover_in = H(inedx_qubits);

	int data_size = data_qubits.size();

	unsigned int ori_data1 = 1;
	string mark_data1 = integerToBinary(ori_data1, data_size);
	unsigned int ori_data2 = 2;
	string mark_data2 = integerToBinary(ori_data2, data_size);
	unsigned int ori_data3 = 7;
	string mark_data3 = integerToBinary(ori_data3, data_size);


	std::vector<std::string> mark_data;
	mark_data = { mark_data1 ,mark_data2, mark_data3 };  // Too many mark_data will fail

	//grover_in << grover_Qprog;
	QCircuit state_quantum;
	state_quantum = grover_Qprog;
	//state_quantum.is_empty();

	QProg prog = grover_search(grover_in, state_quantum, mark_data, data_qubits, inedx_qubits, machine);

	std::cout << prog << std::endl;

	// Probability measurement
	auto result = probRunDict(prog, inedx_qubits);
	//std::cout << "Probability" << std::endl;

	/*for (auto& val : result)
	{
		std::cout << val.f
		irst << " : " << val.second << std::endl;
	}*/

	// get index number
	auto result_index = search_target_from_measure_result(result, inedx_qubits.size());
	/*for (int i = 0; i < result_index.size(); i++)
	{
		cout << result_index[i] << " ";
	}*/

	std::vector<size_t> result_index_actual;
	size_t _i = 0;
	for (const auto& item : search_sapce)
	{
		if (item == SearchDataByUInt(1) || item == SearchDataByUInt(2) || item == SearchDataByUInt(7))
		{
			result_index_actual.push_back(_i);
		}

		++_i;
	}

	for (int i = 0; i < result_index_actual.size(); i++)
	{
		//cout << result_index_actual[i] << " ";
		//cout << index_actual[i] << " ";
		if (result_index[i] != result_index_actual[i])
		{
			return false;
		}
	}

	destroyQuantumMachine(machine);
	return true;
}

TEST(GroverFrameAlg, test)
{
	bool test_val = false;
	try
	{
		test_val = gorver_test1();
		test_val = test_val && gorver_test2();
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

}