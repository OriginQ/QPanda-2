#include "gtest/gtest.h"
#include <iostream>
#include <string>
#include <time.h>
#include <vector>
#include <cmath>

#include "include/QPanda.h"
//#include "Core/QuantumCircuit/QuantumGate.h"
//#include "Core/Utilities/Compiler/PyquilToOriginIR.h"
//#include "Core/Utilities/Tools/MultiControlGateDecomposition.h"

using namespace QPanda;
using namespace std;


namespace Test_Pyquil {
	std::string gate_pyquil1 = R"(DECLARE ro BIT[3]
DECLARE theta REAL[1]
DECLARE alpha REAL[1]
I 0
X 0
Y 0
Z 0
H 0
PHASE(theta[0]) 0
S 0
T 0
CZ 1 0
XY(theta[0]) 0 1
CPHASE00(alpha[0]) 1 1
CPHASE(alpha[0]) 1 0
CPHASE01(alpha[0]) 1 0
CPHASE10(alpha[0]) 1 0
CNOT 1 0
SWAP 0 1
CSWAP 2 0 1
MEASURE 0 ro[0]
MEASURE 1 ro[1]
)";

	std::string gate_pyquil2 = R"(DECLARE ro BIT[3]
I 3
X 3
Y 3
Z 3
H 3
PHASE(3.18) 3
S 3
T 3
CZ 4 3
XY(3.18) 3 4
CPHASE00(3.141592) 4 4
CPHASE01(3.141) 4 3
CNOT 4 3
SWAP 3 4
CSWAP 2 3 4
MEASURE 3 ro[0]
MEASURE 4 ro[1]
)";

	std::string gate_pyquil3 = R"(DECLARE ro BIT[3]
DECLARE theta REAL[1]
DECLARE alpha REAL[1]
DAGGER RX(1.0471975511965976) 0
DAGGER DAGGER RX(1.0471975511965976) 0
CONTROLLED RX(1.0471975511965976) 1 0
CONTROLLED DAGGER RX(1.0471975511965976) 1 0
DAGGER CONTROLLED RX(1.0471975511965976) 1 0
CONTROLLED CONTROLLED RX(1.0471975511965976) 2 1 0
CONTROLLED CONTROLLED RX(1.0471975511965976) 4 3 0
FORKED RX(3.14, 4.13) 0 1
)";
	std::string gate_pyquil4 = R"(DECLARE ro BIT[2]
DECLARE shot_count INTEGER[1]
MOVE shot_count[0] 1000
LABEL @start-loop
H 0
CNOT 0 1
MEASURE 0 ro[0]
MEASURE 1 ro[1]
SUB shot_count[0] 1
JUMP-UNLESS @end-loop shot_count[0]
JUMP @start-loop
LABEL @end-loop
)";

	std::string gate_pyquil5 = R"(DECLARE ro BIT[2]
DECLARE shot_count INTEGER[1]
MOVE shot_count[0] 1000
LABEL @start-loop
H 0
CNOT 0 1
MEASURE 0 ro[0]
MEASURE 1 ro[1]
SUB shot_count[0] 1
JUMP-UNLESS @end-loop shot_count[0]
JUMP @start-loop
LABEL @end-loop
)";
	std::string gate_pyquil6 = R"(DECLARE ro BIT[3]
DECLARE theta REAL[1]
DECLARE alpha REAL[1]
DECLARE shot_count INTEGER[1]
MOVE shot_count[0] 1000
LABEL @start-loop
H 0
CNOT 0 1
MEASURE 0 ro[0]
MEASURE 1 ro[1]
SUB shot_count[0] 1
JUMP-UNLESS @end-loop shot_count[0]
JUMP @start-loop
LABEL @end-loop
)";

	std::string XY_pyquil1 = R"(DECLARE ro BIT[3]
XY(3.14) 1 0
XY(6.28) 0 11
)";
	std::string XY_pyquil2 = R"(DECLARE ro BIT[3]
DECLARE theta REAL[2]
MOVE theta[0] 3.14
XY(theta[0]) 0 1
)";
	std::string XY_pyquil3 = R"(DECLARE ro BIT[3]
DECLARE theta REAL[2]
MOVE theta[0] 3.14
XY(theta[0]) 2 4
XY(theta[0]) 2 1
)";
	std::string PSWAP_ISWAP_pyquil = R"(DECLARE ro BIT[3]
DECLARE theta REAL[2]
PSWAP(3.14) 0 1
ISWAP 0 1
)";
	
	bool test_convert_pyquil_string_to_originir(const std::string& pyquil_str) {
		auto qvm = CPUQVM();
		qvm.init();
		std::cout << "### pyquil_str:\n" << pyquil_str << std::endl;
		std::string originir_str = convert_pyquil_string_to_originir(pyquil_str);
		std::cout << "### originir_str:\n" << originir_str << std::endl;
		QProg prog = convert_originir_string_to_qprog(originir_str, &qvm);
		std::cout << "### prog:\n"<<prog << std::endl;
		return true;
	}

	bool test_convert_pyquil_file_to_originir(const std::string& pyquil_str) {
		std::cout << "### pyquil_str:\n" << pyquil_str << std::endl;
		//prepare pyquil_file
		std::ofstream ofs("pyquil_file_for_test.txt", std::ios::out);
		if (ofs.is_open()) {
			ofs << pyquil_str;
			ofs.close();
			std::cout << "### pyquil_str has saved to file:" << "pyquil_file_for_test.txt" << std::endl;
		}
		else {
			std::cerr << "Error [test_convert_pyquil_file_to_originir] can't open file:" << "pyquil_file_for_test.txt" << std::endl;
			return false;
		}
		//test convert
		auto qvm = CPUQVM();
		qvm.init();
		std::string originir_str = convert_pyquil_file_to_originir("pyquil_file_for_test.txt");
		std::cout << "### originir_str:\n" << originir_str << std::endl;
		QProg prog = convert_originir_string_to_qprog(originir_str, &qvm);
		std::cout << "### prog:\n" << prog << std::endl;
		return true;
	}

	bool test_convert_pyquil_string_to_qprog(const std::string& pyquil_str) {
		auto qvm = CPUQVM();
		qvm.init();
		std::cout << "### pyquil_str:\n" << pyquil_str << std::endl;
		QProg prog = convert_pyquil_string_to_qprog(pyquil_str, &qvm);
		std::cout << "### prog:\n" << prog << std::endl;
		return true;
	}

	bool test_convert_pyquil_file_to_qprog(const std::string& pyquil_str) {
		std::cout << "### pyquil_str:\n" << pyquil_str << std::endl;
		//prepare pyquil_file
		std::ofstream ofs("pyquil_file_for_test.txt",std::ios::out);
		if (ofs.is_open()) {
			ofs << pyquil_str;
			ofs.close();
			std::cout << "### pyquil_str has saved to file:" << "pyquil_file_for_test.txt" << std::endl;
		}
		else {
			std::cerr << "Error [test_convert_pyquil_file_to_originir] can't open file:" << "pyquil_file_for_test.txt" << std::endl;
			return false;
		}
		//test convert
		auto qvm = CPUQVM();
		qvm.init();
		QProg prog = convert_pyquil_file_to_qprog("pyquil_file_for_test.txt",&qvm);
		std::cout << "### prog:\n" << prog << std::endl;
		return true;
	}
};


TEST(PyquilToOriginIR, convert_pyquil_string_to_originir)
{
	std::cout << " ######convert_pyquil_string_to_originir" << std::endl;
	bool test_actual = true;
	/*test_actual &= Test_Pyquil::test_convert_pyquil_string_to_originir(Test_Pyquil::XY_pyquil1);
	test_actual &= Test_Pyquil::test_convert_pyquil_string_to_originir(Test_Pyquil::XY_pyquil2);
	test_actual &= Test_Pyquil::test_convert_pyquil_string_to_originir(Test_Pyquil::XY_pyquil3);*/
	test_actual &= Test_Pyquil::test_convert_pyquil_string_to_originir(Test_Pyquil::PSWAP_ISWAP_pyquil);

	ASSERT_TRUE(test_actual);
}

TEST(PyquilToOriginIR, convert_pyquil_string_to_qprog)
{
	std::cout << " ######convert_pyquil_string_to_qprog" << std::endl;
	bool test_actual = true;
	/*test_actual &= Test_Pyquil::test_convert_pyquil_string_to_qprog(Test_Pyquil::XY_pyquil1);
	test_actual &= Test_Pyquil::test_convert_pyquil_string_to_qprog(Test_Pyquil::XY_pyquil2);
	test_actual &= Test_Pyquil::test_convert_pyquil_string_to_qprog(Test_Pyquil::XY_pyquil3);*/
	test_actual &= Test_Pyquil::test_convert_pyquil_string_to_qprog(Test_Pyquil::PSWAP_ISWAP_pyquil);

	ASSERT_TRUE(test_actual);
}

TEST(PyquilToOriginIR, convert_pyquil_file_to_originir)
{
	std::cout << " ######convert_pyquil_file_to_originir" << std::endl;
	bool test_actual = true;
	/*test_actual &= Test_Pyquil::test_convert_pyquil_file_to_originir(Test_Pyquil::XY_pyquil1);
	test_actual &= Test_Pyquil::test_convert_pyquil_file_to_originir(Test_Pyquil::XY_pyquil2);
	test_actual &= Test_Pyquil::test_convert_pyquil_file_to_originir(Test_Pyquil::XY_pyquil3);*/
	test_actual &= Test_Pyquil::test_convert_pyquil_file_to_originir(Test_Pyquil::PSWAP_ISWAP_pyquil);

	ASSERT_TRUE(test_actual);
}

TEST(PyquilToOriginIR, convert_pyquil_file_to_qprog)
{
	std::cout << " ######convert_pyquil_file_to_qprog" << std::endl;
	bool test_actual = true;
	/*test_actual &= Test_Pyquil::test_convert_pyquil_file_to_qprog(Test_Pyquil::XY_pyquil1);
	test_actual &= Test_Pyquil::test_convert_pyquil_file_to_qprog(Test_Pyquil::XY_pyquil2);
	test_actual &= Test_Pyquil::test_convert_pyquil_file_to_qprog(Test_Pyquil::XY_pyquil3);*/
	test_actual &= Test_Pyquil::test_convert_pyquil_file_to_qprog(Test_Pyquil::PSWAP_ISWAP_pyquil);

	ASSERT_TRUE(test_actual);
}