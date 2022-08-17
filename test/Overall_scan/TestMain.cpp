#include "gtest/gtest.h"
using namespace std;

int main(int argc, char** argv) 
{
#if 0
	//Run a single test of Alg Part
	::testing::GTEST_FLAG(filter) = "RxxRyyRzzRzxGate.test1";
#else
	//Run All Overrall_scan Part
#endif

	::testing::InitGoogleTest(&argc, argv);
	const auto ret = RUN_ALL_TESTS();
	cout << "Overrall_scan Part GTest over, press Enter to continue." << endl;
	getchar();
	return ret;
}