#include "gtest/gtest.h"
#include "QPanda.h"
using namespace std;
static bool test_func_1()
{
	std::vector<std::string>  T0 = { "bread", "cheese", "milk" };
	std::vector<std::string>  T1 = { "bread", "butter" };
	std::vector<std::string>  T2 = { "cheese", "milk", "bread" };
	std::vector<std::string>  T3 = { "bread", "cheese", "milk" };
	std::vector<std::string>  T4 = { "cheese", "butter", "milk" };
	std::vector<std::vector<std::string > > data;
	data.push_back(T0);
	data.push_back(T1);
	data.push_back(T2);
	data.push_back(T3);
	data.push_back(T4);
	QARM qarm(data);
	std::map<std::string, double> ret = qarm.run();
	for (auto r : ret)
	{
		std::cout << r.first << ": " << r.second << std::endl;
	}

	return true;
}


TEST(QARM, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_func_1();
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