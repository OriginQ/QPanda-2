#include <iostream>
#include "Core/Utilities/Tools/OriginCollection.h"
#include "QPanda.h"
#include <time.h>
#include "gtest/gtest.h"

USING_QPANDA
using namespace std;

static bool qw_gorver_test_1()
{
	auto machine = initQuantumMachine(CPU);
	auto x = machine->allocateCBit();
	std::vector<SearchDataByUInt> search_vec = { 6, 6, 6, 15, 16, 6, 6, 4, 6, 6, 6, 6, 14, 6, 0 };
	std::vector<size_t> result_index_vec;
	const int search_data = 6;

	size_t indexx = 0;
	for (const auto &item : search_vec)
	{
		if (item == SearchDataByUInt(search_data)) {
			result_index_vec.push_back(indexx);
		}
		++indexx;
	}

	std::vector<size_t> result_index_vec_qw;
	QProg quantum_walk_prog =
		quantum_walk_alg_search_from_vector(search_vec, x == search_data, machine, result_index_vec_qw, 2);

	
	if (result_index_vec_qw.size() != result_index_vec.size()){
		return false;
	}

	uint32_t i = 0;
	for (const auto &result_item : result_index_vec)
	{
		if (result_item != result_index_vec_qw[i++]){
			return false;
		}
	}

	destroyQuantumMachine(machine);
	return true;
}

TEST(QuantumWalkGrover, test1)
{
	bool test_val = false;

	try{
		test_val = qw_gorver_test_1();
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