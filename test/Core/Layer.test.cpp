#include "gtest/gtest.h"
#include "QPanda.h"
#include <iostream>
#include <vector>
#include <stdio.h>
#include <memory>

#define PRINT_TRACE 0

USING_QPANDA
using namespace std;

static bool test_layer_1()
{
	CPUQVM qvm;
	qvm.init();
	auto qv = qvm.qAllocMany(8);
	auto c = qvm.cAllocMany(8);
	
	QProg prog_dj;
	prog_dj << CZ(qv[2], qv[3]) << CZ(qv[4], qv[6]) << H(qv[0]) << X(qv[1]) << H(qv[1]) << CNOT(qv[0], qv[1]) << H(qv[0])/*<<Measure(qv[0],c[0])*/;

	QProg prog_grover;
	prog_grover << SWAP(qv[5], qv[0]) << H(qv[2]) << X(qv[2]) << SWAP(qv[1], qv[3]) << H(qv[3]) << X(qv[3]) << CZ(qv[2], qv[3])
		<< X(qv[2]) << SWAP(qv[2], qv[0]) << CZ(qv[3], qv[7]) << H(qv[2]) << Z(qv[2]) << H(qv[7])
		<< X(qv[3]) << CZ(qv[5], qv[7]) << H(qv[3]) << Z(qv[3]) << CZ(qv[2], qv[3]) << X(qv[5]) << SWAP(qv[2], qv[4])
		<< H(qv[2]) << H(qv[3]) << H(qv[6]) << Measure(qv[2], c[2]) << Measure(qv[3], c[3]);

	/*cout << "The src prog_grover:" << prog_grover << endl;
	auto_add_barrier_before_mul_qubit_gate(prog_grover);
	cout << "after add_barrier prog_grover:" << prog_grover << endl;*/

	QProg prog;
	prog << X(qv[0]) << CZ(qv[1], qv[2]) << X(qv[3]) << X(qv[3])
		<< Measure(qv[0], c[0]) << Measure(qv[1], c[1]) << Measure(qv[2], c[2]);
	cout << "The src prog:" << prog << endl;

	auto layer_info = prog_layer(prog);

	{
		/* output layer info */
		auto _layer_text = draw_qprog(prog, layer_info);

#if defined(WIN32) || defined(_WIN32)
		_layer_text = fit_to_gbk(_layer_text);
		_layer_text = Utf8ToGbkOnWin32(_layer_text.c_str());
#endif

		std::cout << _layer_text << std::endl;
	}

	return true;
}

TEST(Layer, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_layer_1();
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	ASSERT_TRUE(test_val);
}