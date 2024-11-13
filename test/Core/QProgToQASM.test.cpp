#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <time.h>
#include <vector>

#include "include/QPanda.h"

using namespace QPanda;
using namespace std;

#ifndef PRECISION
#define PRECISION 0.000001
#endif // !PRECISION



bool test() {
	std::cout << "test_qprog_to_qasm" << std::endl;
	auto tmp_qvm = CPUQVM();
	tmp_qvm.init();
	auto q= tmp_qvm.allocateQubits(9);
	auto c = tmp_qvm.allocateCBits(2);
	QProg prog;
	QCircuit cir;
	cir << X(q[0]) << H(q[8]);
	prog << cir;
	std::string qasm_str = convert_qprog_to_qasm(prog,&tmp_qvm);
	std::cout << "qasm_str:\n" << qasm_str << std::endl;
	std::string expected_res = "OPENQASM 2.0;\n"
		"include \"qelib1.inc\";\n"
		"qreg q[9];\n"
		"creg c[0];\n"
		"u3(3.1415926535897931,-1.5707963267948966,1.5707963267948966) q[0];\n"
		"u3(1.5707963267948966,2.2204460492503131e-16,3.1415926535897931) q[8];";
	std::cout << "expected_res:\n" << expected_res << std::endl;

	return qasm_str == expected_res;
}


TEST(QProgToQASM, test)
{
	bool res = true;
	res = res && test();

	ASSERT_TRUE(res);
}


