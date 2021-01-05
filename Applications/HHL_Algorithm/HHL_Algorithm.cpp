/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "Core/Core.h"
#include "QAlg/QAlg.h"

using namespace std;
using namespace QPanda;

/**
* This is a example from <<Quantum Circuit Design for Solving Linear Systems of Equations>>(2012; by Yudong Cao, Anmer Daskin...)
*/
static bool HHL_test_fun()
{
	auto machine = initQuantumMachine(CPU);

	QStat A = {
	qcomplex_t(15.0 / 4.0, 0), qcomplex_t(9.0 / 4.0, 0), qcomplex_t(5.0 / 4.0, 0), qcomplex_t(-3.0 / 4.0, 0),
	qcomplex_t(9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0),
	qcomplex_t(5.0 / 4.0, 0), qcomplex_t(3.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0),
	qcomplex_t(-3.0 / 4.0, 0), qcomplex_t(-5.0 / 4.0, 0), qcomplex_t(-9.0 / 4.0, 0), qcomplex_t(15.0 / 4.0, 0)
	};

	std::vector<double> b = { 0.5, 0.5, 0.5, 0.5 };

	auto prog = QProg();
	QCircuit hhl_cir = build_HHL_circuit(A, b, machine);
	prog << hhl_cir;

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

	const double max_precision = 1e-10;
	for (auto &val : stat_normed)
	{
		qcomplex_t tmp_val((abs(val.real()) < max_precision ? 0.0 : val.real()), (abs(val.imag()) < max_precision ? 0.0 : val.imag()));
		val = tmp_val;
	}

	//get solution
	QStat result;
	for (size_t i = 0; i < b.size(); ++i)
	{
		result.push_back(stat_normed.at(i));
	}

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

int main()
{
	try
	{
	   HHL_test_fun();
	}
	catch (const std::exception& e)
	{
		cerr << "HHL running error: got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cerr << "HHL running error: unknow exception." << endl;
	}

	cout << "\n HHL test over, press Enter to continue..." << endl;
	getchar();

	return 0;
}
