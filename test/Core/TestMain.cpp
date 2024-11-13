#include "gtest/gtest.h"
using namespace std;

int main(int argc, char** argv) 
{
#if 1
	//Run a single test of Alg Part
    ::testing::GTEST_FLAG(filter) = "Stabilizer.test";
    //::testing::GTEST_FLAG(filter) = "QPilotOSMachine.test";
    //::testing::GTEST_FLAG(filter) = "QubitMapping.test1";
	//::testing::FLAGS_gtest_filter("QASMToQProg2.test");
	::testing::FLAGS_gtest_filter = "DrawLatex.test3";
	
#else
	//Run All Core Part
#endif

	::testing::InitGoogleTest(&argc, argv);
	const auto ret = RUN_ALL_TESTS();
	cout << "Core Part GTest over, press Enter to continue." << endl;
	getchar();
	return ret;
}