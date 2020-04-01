#include "gtest/gtest.h"
#include "QPanda.h"

using namespace std;
USING_QPANDA

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	std::cout << "Test: QAdder" << std::endl;

	auto qvm = initQuantumMachine();

	auto a = qvm->allocateQubits(3);
	auto b = qvm->allocateQubits(3);
	Qubit *c = qvm->allocateQubit();
	Qubit *is_carry = qvm->allocateQubit();
	QProg prog;

	prog << X(a[0]) << X(a[1])              // Preparation of addend a = |011> 
		<< X(b[0]) << X(b[1]) << X(b[2])    // Preparation of addend b = |111>
	   << isCarry(a, b, c, is_carry);		// Return carry item of a + b
	  //<< QAdderIgnoreCarry(a, b, c);		// Return a + b (ignore carry item of a + b)
	 //<< QAdder(a, b, c, is_carry);		    // Return a + b 

	qvm->directlyRun(prog);
	auto temp = dynamic_cast<IdealMachineInterface *>(qvm);


	auto result1 = temp->quickMeasure({ is_carry }, 1000); 
	auto result2 = temp->quickMeasure(a, 1000);
	auto result3 = temp->quickMeasure(b, 1000);
	destroyQuantumMachine(qvm);

	std::cout <<  " The carry item of a + b : "  << std::endl;
	for (auto &val : result1)
	{
		std::cout << val.first << ", " << val.second << std::endl;
	}

	std::cout << " The result of a + b minus the carry term : " << std::endl;
	for (auto &val : result2)
	{
		std::cout << val.first << ", " << val.second << std::endl;
	}

	std::cout << " The invariant addend b : " << std::endl;
	for (auto &val : result3)
	{
		std::cout << val.first << ", " << val.second << std::endl;
	}
	system("pause");
	return RUN_ALL_TESTS();
}