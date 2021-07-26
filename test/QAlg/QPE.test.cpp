#include "gtest/gtest.h"
#include <iostream>
#include "QPanda.h"
#include <time.h>

USING_QPANDA
using namespace std;

#ifndef PRECISION
#define PRECISION 0.000001
#endif // !PRECISION

static QCircuit build_U_fun(QVec qubits) 
{
	QCircuit cir_u;
	//cir_u << T(qubits[0]);
	cir_u << U1(qubits[0], 2.0*PI / 3.0);
	return cir_u;
}

static bool test_qpe1()
{
	auto machine = initQuantumMachine(CPU);
	size_t first_register_qubits_cnt = 3;
	QVec first_register = machine->allocateQubits(first_register_qubits_cnt);

	size_t second_register_qubits_cnt = 1;
	QVec second_register = machine->allocateQubits(second_register_qubits_cnt);

	QCircuit cir_qpe = build_QPE_circuit(first_register, second_register, build_U_fun);
	//cout << cir_qpe << endl;
	QProg qpe_prog;
	qpe_prog << X(second_register[0]) << cir_qpe;

	auto result1 = probRunDict(qpe_prog, first_register);

	for (auto &val : result1)
	{
		val.second = abs(val.second) < PRECISION ? 0.0 : val.second;
	}

	//std::cout << "QPE result:" << endl;
	for (auto &val : result1)
	{
		if (result1.size() != 8 && result1.begin()->second != 0.015625)
			return false;
		else
			return true;
		//std::cout << val.first << ", " << val.second << std::endl;
	}

	//ASSERT_EQ(result1[0]->second, 0.015625);

	return true;
}

TEST(QPE, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_qpe1();
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