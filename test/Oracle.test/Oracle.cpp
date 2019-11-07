#include <map>
#include "QPanda.h"
#include "Core/Utilities/Tools/XMLConfigParam.h"
#include <algorithm>  
#include "gtest/gtest.h"
using namespace std;
USING_QPANDA

TEST(Oracle, createoracle)
{
    std::cout << "======================================" << std::endl;
	auto qm = initQuantumMachine();
	auto qs = qAllocMany(5);
	QProg prog;
	prog << oracle(qs, "test1");	
	std::string s =	transformQProgToOriginIR(prog, qm);
	cout << s<<endl;
	std::cout << "======================================" << std::endl;

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}