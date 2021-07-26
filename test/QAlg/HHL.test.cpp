#include "gtest/gtest.h"
#include <iostream>
#include "QPanda.h"
#include <time.h>
#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"


#include "Extensions/Extensions.h"

#ifdef USE_EXTENSION

USING_QPANDA
using namespace std;

#define MAX_PRECISION 1e-10

/**
* This is a example from <<Quantum Circuit Design for Solving Linear Systems of Equations>>(2012; by Yudong Cao，Anmer Daskin...)
*/
static bool test_fun1()
{
	auto machine = initQuantumMachine(CPU);
	auto prog = QProg();

	QStat A = {
	qcomplex_t(15.0 / 4.0, 0), qcomplex_t(9.0 / 4.0, 0), qcomplex_t(5.0 / 4.0, 0), qcomplex_t(-3.0 / 4.0, 0),
	qcomplex_t(9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0),
	qcomplex_t(5.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0),
	qcomplex_t(-3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0)
	};

	//std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };
	std::vector<double> b = { 1, 1, 1, 1 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	//double coffe = sqrt(340);

	QStat _actual = { qcomplex_t(-0.0625,0) ,qcomplex_t(0.4375,0),
					  qcomplex_t(0.6875, 0) ,qcomplex_t(0.8125, 0) };
	//ouput solution
	const auto _p = 1e-2;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) ||
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2) {
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;

}

/**
* @The HHL result:
   (0.5,0) (0.25,0)
   (0.125,0) (0.0625,0)]
*/
static bool test_fun2()
{
	auto machine = initQuantumMachine(CPU);


	auto prog = QProg();

	QStat A = {
	qcomplex_t(1.0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(2.0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(4.0, 0), qcomplex_t(0, 0),
	qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(0, 0), qcomplex_t(8.0, 0)
	};

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	//double coffe = sqrt(340);

	QStat _actual = { qcomplex_t(0.5,0) ,qcomplex_t(0.25,0),
					  qcomplex_t(0.125,0) ,qcomplex_t(0.0625,0) };
	//ouput solution
	const auto _p = 1e-1;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) ||
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2) {
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

//(0.626818, 0) (0.2, 0)
//(0.4, 0) (1.24507, 0)
static bool test_fun3()
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

	QStat result = HHL_solve_linear_equations(A, b, 1);
	int w = 0;
	//double coffe = sqrt(340);

	QStat _actual = { qcomplex_t(0.626818, 0) ,qcomplex_t(0.2, 0),
					  qcomplex_t(0.4, 0) ,qcomplex_t(1.24507, 0) };
	//ouput solution
	const auto _p = 1e-1;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) ||
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2) {
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
	return true;
}
//(0.0232445, 0) (1.04619, 0)
//(2.07265, 0) (3.08467, 0)
//(4.09632, 0) (3.09583, 0)
//(2.1017, 0) (0.030159, 0)
static bool test_fun4()
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
	//double coffe = sqrt(340);

	QStat _actual = { qcomplex_t(0.0232445, 0) ,qcomplex_t(1.04619, 0),
					  qcomplex_t(2.07265, 0) ,qcomplex_t(3.08467, 0),
	                  qcomplex_t(4.09632, 0) ,qcomplex_t(3.09583, 0),
	                  qcomplex_t(2.1017, 0) ,qcomplex_t(0.030159, 0)
	                 };
	//ouput solution
	const auto _p = 1e-4;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) ||
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2) {
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}



//(-0.0185844, 0) (0.933218, 0)
//(1.91188, 0) (2.90063, 0)
//(3.88659, 0) (2.87717, 0)
//(1.85705, 0) (-0.0505955, 0)
static bool test_fun5()
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
	//double coffe = sqrt(340);

	QStat _actual = { qcomplex_t(-0.0185844, 0) ,qcomplex_t(0.933218, 0),
					  qcomplex_t(1.91188, 0) ,qcomplex_t(2.90063, 0),
					  qcomplex_t(3.88659, 0) ,qcomplex_t(2.87717, 0),
					  qcomplex_t(1.85705, 0) ,qcomplex_t(-0.0505955, 0)
	                 };
	//ouput solution
	const auto _p = 1e-4;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) ||
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2) {
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

/* 设置求解精度为2，得到HHL解为：(1.12452,0) (0.374246,0)
*/
static bool test_fun6()
{
	auto machine = initQuantumMachine(CPU);

	QStat A = {
        qcomplex_t(1.0 / 1.0, 0), qcomplex_t(-1.0 / 3.0, 0),
        qcomplex_t(-1.0 / 3.0, 0), qcomplex_t(1.0 / 1.0, 0)
	};

	std::vector<double> b = { 1, 0 };

	std::vector<double> tmp_b = b;
	double norm_coffe = 0.0;
	for (const auto& item : tmp_b)
	{
		norm_coffe += (item*item);
	}

	norm_coffe = sqrt(norm_coffe);
	for (auto& item : tmp_b)
	{
		item = item / norm_coffe;
	}

	//build HHL quantum program
	QProg prog;
	HHLAlg hhl_alg(machine);
	QCircuit hhl_cir = hhl_alg.get_hhl_circuit(A, tmp_b, 2);
	prog << hhl_cir;

	//PTrace("HHL quantum circuit is running ...\n");
	directlyRun(prog);
	auto stat = machine->getQState();
	machine->finalize();
	stat.erase(stat.begin(), stat.begin() + (stat.size() / 2));

	// normalise
	QStat stat_normed;
	for (auto &val : stat)
	{
		stat_normed.push_back(val * norm_coffe * hhl_alg.get_amplification_factor());
	}

	for (auto &val : stat_normed)
	{
		qcomplex_t tmp_val((abs(val.real()) < MAX_PRECISION ? 0.0 : val.real()), (abs(val.imag()) < MAX_PRECISION ? 0.0 : val.imag()));
		val = tmp_val;
	}

	//get solution
	QStat result;
	for (size_t i = 0; i < b.size(); ++i)
	{
		result.push_back(stat_normed.at(i));
	}

	QStat _actual = { qcomplex_t(1.12452,0), qcomplex_t(0.374246,0)};
	//ouput solution
	int w = 0;
	const auto _p = 1e-5;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) || 
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2){
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

/* 设置求解精度为2，得到HHL解为：
(2185.3,0) (2205.46,0)
(2231.63,0) (2265.44,0)
(2169.26,0) (2192.92,0)
(2224.78,0) (2258.67,0)

*/
static bool test_fun7()
{
	auto machine = initQuantumMachine(CPU);
	auto prog = QProg();

	QStat A = {
	  10,  -4,   1,   0,  -4,   0,   0,   0,
      -4,  11,  -4,   1,   0,  -4,   0,   0,
       1,  -4,  11,  -4,   0,   0,  -4,   0,
       0,   1,  -4,  10,   0,   0,   0,  -4,
      -4,   0,   0,   0,  10,  -4,   1,   0,
       0,  -4,   0,   0,  -4,  11,  -4,   1,
       0,   0,  -4,   0,   1,  -4,  11,  -4,
       0,   0,   0,  -4,   0,   1,  -4,  14
	};

	std::vector<double> b = { 6587.2570531667, 89.6725401793, -47.1589126840, 6898.3749015014,
	6406.4504474986, -13.1713639276, -88.1442430419, 15853.8611183183};

	QStat result = HHL_solve_linear_equations(A, b, 2);
	int w = 0;
	QStat _actual = { qcomplex_t(2185.3,0) ,qcomplex_t(2205.46,0),
					  qcomplex_t(2231.63,0) ,qcomplex_t(2265.44,0),
					  qcomplex_t(2169.26,0), qcomplex_t(2192.92,0),
                      qcomplex_t(2224.78,0),  qcomplex_t(2258.67,0) };
	//ouput solution
    //int w = 0;
	const auto _p = 1e-1;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) ||
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2) {
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

/* HHL解为：(0,0) (1,0)
*/
static bool test_fun8()
{
	auto machine = initQuantumMachine(CPU);
	auto prog = QProg();

	QStat A = {
	  0,  1,  
	  1,  0
	};

	std::vector<double> b = { 1,0 };

	QStat result = HHL_solve_linear_equations(A, b, 1);
	int w = 0;
	QStat _actual = { qcomplex_t(0,0) ,qcomplex_t(1,0)};
	//ouput solution
	//int w = 0;
	const auto _p = 1e-1;
	for (size_t i = 0; i < result.size(); i++)
	{
		const auto& val = result[i];
		std::cout << val << " ";
		if ((abs(abs(val.real()) - abs(_actual[i].real())) > _p) ||
			(abs(abs(val.imag()) - abs(_actual[i].imag())) > _p))
		{
			return false;
		}

		if (++w == 2) {
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
		test_val = test_fun2();
		test_val = test_fun3();
		test_val = test_fun4();
		test_val = test_fun5();
		test_val = test_fun6();
		test_val = test_fun7();
		test_val = test_fun8();
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

#endif
