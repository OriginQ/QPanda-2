#include <iostream>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "QPanda.h"
#include <time.h>
#include "gtest/gtest.h"

USING_QPANDA
using namespace std;

TEST(QuantumWalkGrover, test1)
{
	auto machine = initQuantumMachine(CPU);
	auto x = machine->allocateCBit();

	//std::vector<SearchDataByUInt> search_vec = { 7, 5, 6, 6, 7, 2, 3, 6/*, 9, 6*//*, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6*/
	///*, 7, 2, 13, 12, 14, 7, 8, 9, 10, 11, 15, 14, 5, 7, 2, 13, 12, 14, 7, 8, 6, 10, 11, 15, 14, 5, 9, 12, 13, 11*/
	///*, 7, 2, 13, 12, 14, 7, 8, 9, 10, 11, 15, 14, 5, 7, 2, 13, 12, 14, 7, 8, 6, 10, 11, 15, 14, 5, 9, 12, 13, 11*/ 
	///*, 7, 2, 13, 12, 14, 7, 8, 9, 10, 11, 15, 14, 5, 7, 2, 13, 12, 14, 7, 8, 6, 10, 11, 15, 14, 5, 9, 12, 13, 11*/ };
	std::vector<SearchDataByUInt> search_vec = { 6, 6, 6, 15, 16, 6, 6, 4, 6, 6, 6, 6, 14, 6, 0 };

	std::vector<size_t> result_index_vec;
	const int search_data = 0;

	//test
	size_t indexx = 0;
	for (const auto &item : search_vec)
	{
		if (item == SearchDataByUInt(search_data))
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

	try
	{
		cout << "Start quantum walk search algorithm:" << endl;
		QProg quantum_walk_prog = quantum_walk_alg_search_from_vector(search_vec, x == search_data, machine, result_index_vec, 2);
		//cout << "quantum_walk_prog:" << quantum_walk_prog << endl;
	}
	catch (const std::exception& e)
	{
		cout << "Error:Catch an exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Error: Catch a unknow exception." << endl;
	}

	cout << "The result's index:" << endl;
	for (const auto &result_item : result_index_vec)
	{
		cout << result_item << " ";
	}

	destroyQuantumMachine(machine);

	cout << "\n Quantum walk test over, press Enter to continue..." << endl;
	getchar();
}