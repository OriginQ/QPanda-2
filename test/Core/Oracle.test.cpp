#include <map>
#include "QPanda.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include <algorithm>  
#include "gtest/gtest.h"
using namespace std;
USING_QPANDA

TEST(Oracle, createoracle)
{
    //std::cout << "======================================" << std::endl;
	auto qm = initQuantumMachine();
	auto qs = qAllocMany(5);
	QProg prog;
	prog << oracle(qs, "test1");	
	std::string s =	transformQProgToOriginIR(prog, qm);
	//cout << s<<endl;
	//std::cout << "======================================" << std::endl;

	std::string excepted_val = R"(QINIT 5
CREG 0
test1 q[0],q[1],q[2],q[3],q[4])";
	ASSERT_EQ(s, excepted_val);
	//std::cout << "Oracle  tests over" << std::endl;
}
