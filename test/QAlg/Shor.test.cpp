#include "gtest/gtest.h"
#include "QPanda.h"

bool test_Shor()
{
	int N = 15, factor_1, factor_2;
	bool p = Shor_factorization(N).first;
	return p;
}

TEST(Shor, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_Shor();
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