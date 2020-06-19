#include "gtest/gtest.h"
#include <iostream>
#include "QPanda.h"
#include <time.h>

USING_QPANDA
using namespace std;

inline bool test_fun1()
{
	auto machine = initQuantumMachine(CPU);


	auto prog = QProg();

	QStat A = {
	qcomplex_t(15.0 / 4.0, 0), qcomplex_t(9.0 / 4.0, 0), qcomplex_t(5.0 / 4.0, 0), qcomplex_t(-3.0 / 4.0, 0),
	qcomplex_t(9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0),
	qcomplex_t(5.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0),
	qcomplex_t(-3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0)
	};

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	double coffe = sqrt(340);
	for (auto &val : result)
	{
		val *= coffe;
		std::cout << val << " ";
		if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

inline bool test_fun2()
{
	auto machine = initQuantumMachine(CPU);


	auto prog = QProg();

	QStat A = {
	qcomplex_t(3.0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(7.0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(4.0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(2.0, 0)
	};

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	double coffe = sqrt(1);
	for (auto &val : result)
	{
		val *= coffe;
		std::cout << val << " ";
		if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

inline bool test_fun3()
{
	auto machine = initQuantumMachine(CPU);


	auto prog = QProg();

	QStat A = {
	qcomplex_t(0.8, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(2.5, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(1.25, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0.4, 0)
	};

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	double coffe = sqrt(1);
	for (auto &val : result)
	{
		val *= coffe;
		std::cout << val << " ";
		if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

inline bool test_fun4()
{
	QStat A = {
		qcomplex_t(2, 0), qcomplex_t(-2, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(1, 0),
		qcomplex_t(-2, 0), qcomplex_t(4, 0), qcomplex_t(-2, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(-2, 0), qcomplex_t(5, 0), qcomplex_t(-2, 0), qcomplex_t(0, 0), qcomplex_t(-1, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-2, 0), qcomplex_t(3, 0), qcomplex_t(-1, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-1, 0), qcomplex_t(4, 0), qcomplex_t(-3, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-1, 0), qcomplex_t(0, 0), qcomplex_t(-3, 0), qcomplex_t(5, 0), qcomplex_t(-1, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-1, 0), qcomplex_t(1, 0), qcomplex_t(0, 0),
		qcomplex_t(1, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0)
	};

	std::vector<double> b = { -2, 0, -1, 1, 4, -1, -1, 0 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	for (auto &val : result)
	{
		std::cout << (val.real()*val.real() + val.imag()*val.imag()) << " ";
		if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

inline bool test_fun5()
{
	QStat A = {
		qcomplex_t(1, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0.70710678, 0.70710678)
	};

	/*QStat A = {
		qcomplex_t(1, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(-0.5, 0.866025)
	};*/

	std::vector<double> b = { 0, 1 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	for (auto &val : result)
	{
		std::cout << (val.real()/**val.real()*/ + val.imag()/**val.imag()*/) << " ";
		if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

inline bool test_fun6()
{
	QStat A = {
		qcomplex_t(1, 0), qcomplex_t(-1, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(1, 0),
		qcomplex_t(-1, 0), qcomplex_t(3, 0), qcomplex_t(-2, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(-2, 0), qcomplex_t(5, 0), qcomplex_t(-3, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-3, 0), qcomplex_t(5, 0), qcomplex_t(-2, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-2, 0), qcomplex_t(5, 0), qcomplex_t(-3, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-3, 0), qcomplex_t(4, 0), qcomplex_t(-1, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-1, 0), qcomplex_t(1, 0), qcomplex_t(0, 0),
		qcomplex_t(1, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0)
	};

	std::vector<double> b = { -1.0, -1.0, -1.0, 1.0, 5.0, -2.0, -1.0, 0 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	for (auto &val : result)
	{
		std::cout << val.real()/**val.real()*/ << ", " << val.imag()/**val.imag()*/ << "i ";
		if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

inline bool test_fun7()
{
	/*QStat A = {
		qcomplex_t(0.002, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0.323, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0.00024, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-1.6, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-0.5, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(1.1005, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-0.8, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0.0015, 0)
	};*/

	QStat A = {
		qcomplex_t(2, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(3, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(4, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-6, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-5, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(7, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(-8, 0), qcomplex_t(0, 0),
		qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(2, 0)
	};

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	for (auto &val : result)
	{
		std::cout << val << " ";
		if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

TEST(HHL, test1)
{
	bool test_val = false;
	try
	{
		test_val = test_fun1();
		//test_val = test_fun2();
		//test_val = test_fun3();
		//test_val = test_fun4();
		//test_val = test_fun5();
		//test_val = test_fun6();
		//test_val = test_fun7();
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