#include "gtest/gtest.h"
using namespace std;

int main(int argc, char** argv) 
{
#if 1
	//Run a single test of Alg Part
    //::testing::GTEST_FLAG(filter) = "RxxRyyRzzRzxGate.test1";
    ::testing::GTEST_FLAG(filter) = "QPilotOSMachine.test";
#else
	//Run All Core Part
#endif

	::testing::InitGoogleTest(&argc, argv);
	const auto ret = RUN_ALL_TESTS();
	cout << "Core Part GTest over, press Enter to continue." << endl;
	getchar();
	return ret;
}