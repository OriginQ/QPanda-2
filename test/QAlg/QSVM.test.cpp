#include "gtest/gtest.h"
#include "QPanda.h"

static bool test_func_1()
{
	std::vector<double> row1 = { 7.55151, 0.58003, 1 };
	std::vector<double> row2 = { 3.542485, 1.977398, -1 };
	std::vector<double> row3 = { 4.658191, 2.507396, -1 };
	std::vector<std::vector<double > > data = { row1, row2, row3 };

	std::vector<double> query_x = { 0, 1 };

	bool ret = qsvm_algorithm(data, query_x);

	std::stringstream str_result;
	str_result << " { " << query_x[0] << "," << query_x[1] << " } :";
	if (ret)
		str_result << "belongs to class 1";
	else
		str_result << "belongs to class 2";

	std::cout << str_result.str() << std::endl;

	return true;
}


TEST(QSVM, test1)
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