#include "gtest/gtest.h"
#include "QPanda.h"

using namespace std;
USING_QPANDA

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);

	return RUN_ALL_TESTS();
}

