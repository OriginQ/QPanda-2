#include "gtest/gtest.h"
#include <iostream>
#include "QPanda.h"
#include <time.h>
#include "Components/NodeSortProblemGenerator/NodeSortProblemGenerator.h"

USING_QPANDA
using namespace std;

#define MAX_PRECISION 1e-10

/**
* This is a example from <<Quantum Circuit Design for Solving Linear Systems of Equations>>(2012; by Yudong Cao£¬Anmer Daskin...)
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

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };

	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	//double coffe = sqrt(340);
	for (auto &val : result)
	{
		//val *= coffe;
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

static bool test_fun2()
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

static bool test_fun5()
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

static bool test_fun6()
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

static bool test_fun7()
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

static bool test_fun8()
{
	//std::vector<std::vector<double>> graph{
	//	//   O, A, B, C, D, E, F, G
	//		{0, 1, 0, 0, 0, 0, 0, 0,},
	//		{1, 0, 1, 0, 0, 0, 0, 0,},
	//		{0, 1, 0, 1, 1, 1, 0, 0,},
	//		{0, 0, 1, 0, 0, 1, 1, 0,},
	//		{0, 0, 1, 0, 0, 1, 0, 1,},
	//		{0, 0, 1, 1, 1, 0, 1, 1,},
	//		{0, 0, 0, 1, 0, 1, 0, 0,},
	//		{0, 0, 0, 0, 1, 1, 0, 0,}
	//};

	std::vector<std::vector<double>> graph{
		// 1  2  3  4  5  6  7  8  9 10 11 12
		  {0, 1 ,0 ,0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //1
		  {1, 0 ,1 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //2
		  {0, 1 ,0 ,1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //3
		  {0, 0 ,1 ,0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0}, //4
		  {0, 0 ,0 ,1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}, //5
		  {1, 0 ,0 ,0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0}, //6
		  {0, 0 ,0 ,0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}, //7
		  {0, 0 ,0 ,0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0}, //8
		  {0, 0 ,0 ,0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0}, //9
		  {0, 0 ,0 ,1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0}, //10
		  {0, 0 ,0 ,0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0}, //11
		  {0, 0 ,0 ,1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0}, //12
		  {0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //13
		  {0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //14
		  {0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, //15
		  {0, 0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}  //16
	};

	NodeSortProblemGenerator problem;
	problem.setProblemGraph(graph);
	problem.exec();

	std::cout << "Classical Liner result: " << std::endl;
	std::cout << problem.getLinearSolverResult();
	std::cout << std::endl;

	auto oA = problem.getMatrixA();
	auto ob = problem.getVectorB();

	//QStat A;
	//for (auto i = 0; i < oA.rows(); i++)
	//{
	//	for (auto j = 0; j < oA.cols(); j++)
	//	{
	//		A.push_back(oA.row(i)[j]);
	//	}
	//}
	auto A = Eigen_to_QStat(oA);
	//Eigen::MatrixXcd

	std::vector<double> b;
	for (auto i = 0; i < ob.size(); i++)
	{
		b.push_back(ob[i]);
	}

	std::cout << "HHL:" << std::endl;
	QStat result = HHL_solve_linear_equations(A, b);
	int w = 0;
	for (auto& val : result)
	{
		std::cout << val << " ";
		if (++w == 1)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;

	return true;
}

static bool test_fun9()
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
	auto prog = QProg();
	QCircuit hhl_cir = build_HHL_circuit(A, tmp_b, machine);
	prog << hhl_cir;
	
	QVec qv;
	get_all_used_qubits(prog, qv);
	cout << "befort mapp: " << qv.size() << endl;
	quantum_chip_adapter(prog, machine, qv);
	cout << "after mapp: " << qv.size() << endl;

	size_t gate_num = getQGateNum(prog);
	cout << "gate_num: " << gate_num << endl;
	getchar();
	auto seq = prog_layer(prog);
	size_t layer_cnt = seq.size();
	cout << "layer_cnt: " << layer_cnt << endl;

	PTrace("HHL quantum circuit is running ...\n");
	directlyRun(prog);
	auto stat = machine->getQState();
	machine->finalize();
	stat.erase(stat.begin(), stat.begin() + (stat.size() / 2));

	// normalise
	double norm = 0.0;
	for (auto &val : stat)
	{
		norm += ((val.real() * val.real()) + (val.imag() * val.imag()));
	}
	norm = sqrt(norm);

	QStat stat_normed;
	for (auto &val : stat)
	{
		stat_normed.push_back(val / qcomplex_t(norm, 0));
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
		//test_val = test_fun8();
		//test_val = test_fun9();
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