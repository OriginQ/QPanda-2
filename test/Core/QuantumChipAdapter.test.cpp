#include "gtest/gtest.h"
#include <iostream>
#include "QPanda.h"
#include <time.h>

USING_QPANDA
using namespace std;

#define PTrace printf
#define MAX_PRECISION 1e-10

bool QuantumChipAdapter_test_1()
{
	auto machine = initQuantumMachine(CPU);

	QStat A = {
	qcomplex_t(15.0 / 4.0, 0), qcomplex_t(9.0 / 4.0, 0), qcomplex_t(5.0 / 4.0, 0), qcomplex_t(-3.0 / 4.0, 0),
	qcomplex_t(9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0),
	qcomplex_t(5.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0),
	qcomplex_t(-3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0)
	};

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };

	QCircuit hhl_cir = build_HHL_circuit(A, b, machine);
	QProg hhl_prog(hhl_cir);
	//cout << "hhl_prog" << endl << hhl_prog << endl;
	//getchar();
	/*const auto mat_src = getCircuitMatrix(hhl_prog);
	cout << "mat_src:" << endl << mat_src << endl;*/

	QVec qv;
	get_all_used_qubits(hhl_prog, qv);
	quantum_chip_adapter(hhl_prog, machine, qv);

	/*const auto mat_after_adapter = getCircuitMatrix(hhl_prog);
	cout << "mat_after_adapter:" << endl << mat_after_adapter << endl;*/

	/*if (mat_src == mat_after_adapter)
	{
		cout << "matrix okKKKKKKKKKKKKKKKKKKKKKK" << endl;
	}
	else
	{
		cout << "matrix fffffffffff" << endl;
	}*/

	if (0)
	{
		auto result1 = probRunDict(hhl_prog, qv);

		for (auto &val : result1)
		{
			val.second = abs(val.second) < 0.000001 ? 0.0 : val.second;
		}

		std::cout << "hhl global result:" << endl;
		for (auto &val : result1)
		{
			std::cout << val.first << ", " << val.second << std::endl;
		}

		std::cout << "MMMMMMMMMMMMMMMMMMMMMMMMMM" << endl;
		getchar();
	}


	PTrace("start pmeasure.\n");
	directlyRun(hhl_prog);
	PTrace("start getQState.\n");
	auto stat = machine->getQState();
	PTrace("start machine->finalize.\n");
	machine->finalize();
	PTrace("get result.\n");
	PTrace("source result:.\n");
	for (size_t i = 0; i < stat.size(); ++i)
	{
		//result.push_back(stat_normed.at(i));
		cout << stat.at(i) << endl;
	}
	PTrace("source result endl===================:.\n");
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
		//result.push_back(stat_normed.at(i));
		cout << stat_normed.at(i) << endl;
	}

	destroyQuantumMachine(machine);

	return true;
}

static QCircuit build_U_fun(QVec qubits)
{
	QCircuit cir_u;
	cir_u << T(qubits[0]);
	//cir_u << U1(qubits[0], 2.0*PI / 3.0);
	return cir_u;
}

bool QuantumChipAdapter_test_2()
{
	auto machine = initQuantumMachine(CPU);
	/*auto machine = new CPUQVM();
	machine->init();*/
	size_t first_register_qubits_cnt = 5;
	QVec first_register = machine->allocateQubits(first_register_qubits_cnt);

	size_t second_register_qubits_cnt = 1;
	QVec second_register = machine->allocateQubits(second_register_qubits_cnt);

	QCircuit cir_qpe = build_QPE_circuit(first_register, second_register, build_U_fun);
	cout << cir_qpe << endl;
	QProg qpe_prog;
	qpe_prog << X(second_register[0]) << cir_qpe;

	//cout << "qpe_prog" << endl << qpe_prog << endl;
	QVec qv;
	get_all_used_qubits(qpe_prog, qv);
	quantum_chip_adapter(qpe_prog, machine, qv);
	

	PTrace("start pmeasure.\n");
	auto result1 = probRunDict(qpe_prog, qv);
	/*machine->directlyRun(qpe_prog);
	auto result1 = machine->PMeasure_no_index(qv);*/


	//auto result1 = probRunDict(qpe_prog, first_register);

	/*for (auto &val : result1)
	{
		val.second = abs(val.second) < PRECISION ? 0.0 : val.second;
	}*/

	std::cout << "QPE result:" << endl;
	for (auto &val : result1)
	{
		std::cout << val.first << ", " << val.second << std::endl;
		//std::cout << val << std::endl;
	}


	destroyQuantumMachine(machine);
	/*machine->finalize();
	delete machine;*/

	return true;
}

bool QuantumChipAdapter_test_3()
{
	auto machine = initQuantumMachine(CPU);
	auto x = machine->allocateCBit();
	std::vector<SearchDataByUInt> search_sapce;
	search_sapce.push_back(8);
	search_sapce.push_back(7);
	search_sapce.push_back(6);
	search_sapce.push_back(0);
	search_sapce.push_back(6);
	search_sapce.push_back(3);
	search_sapce.push_back(6);
	search_sapce.push_back(4);
	/*search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(6);
	search_sapce.push_back(7);
	search_sapce.push_back(14);
	search_sapce.push_back(9);
	search_sapce.push_back(12);
	search_sapce.push_back(4);
	search_sapce.push_back(9);
	search_sapce.push_back(9);
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(3);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);
	search_sapce.push_back(21);
	search_sapce.push_back(15);
	search_sapce.push_back(11);
	search_sapce.push_back(11);
	search_sapce.push_back(3);
	search_sapce.push_back(9);
	search_sapce.push_back(7);*/

	for (size_t i = 0; i < 0; i++)
	{
		search_sapce.insert(search_sapce.end(), search_sapce.begin() + 20, search_sapce.end());
	}
	cout << "Grover will search through " << search_sapce.size() << " data." << endl;

	

	//test
	/*size_t indexx = 0;
	for (const auto &item : search_sapce)
	{
		if (item == SearchDataByUInt(6))
		{
			result_index_vec.push_back(indexx);
		}
		++indexx;
	}*/

	/*cout << "The target result's index:" << endl;
	for (const auto &result_item : result_index_vec)
	{
		cout << result_item << " ";
	}
	cout << endl;
	result_index_vec.clear();*/

	cout << "Start grover search algorithm:" << endl;
	QVec measure_qubits;
	QProg grover_Qprog = build_grover_alg_prog(search_sapce, x == 6, machine, measure_qubits,  2);

	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");

	QVec qv;
	get_all_used_qubits(grover_Qprog, qv);
	quantum_chip_adapter(grover_Qprog, machine, qv);
	//cout << "grover_Qprog:" << endl;
	//cout << grover_Qprog << endl;

	//measure
	PTrace("Strat pmeasure.\n");
	auto result = probRunDict(grover_Qprog, qv);

	//get result
	/*double total_val = 0.0;
	for (auto& var : result) { total_val += var.second; }
	const double average_probability = total_val / result.size();
	size_t search_result_index = 0;*/

	PTrace("pmeasure result:\n");
	for (auto aiter : result)
	{
		PTrace("%s:%5f\n", aiter.first.c_str(), aiter.second);
		/*if (average_probability < aiter.second)
		{
			result_index_vec.push_back(search_result_index);
		}
		++search_result_index;*/
	}

	destroyQuantumMachine(machine);
	return true;
}

TEST(QuantumChipAdapter, test1)
{
	bool test_val = false;
	try
	{
		//test_val = QuantumChipAdapter_test_1();
		//test_val = QuantumChipAdapter_test_2();
		test_val = QuantumChipAdapter_test_3();
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