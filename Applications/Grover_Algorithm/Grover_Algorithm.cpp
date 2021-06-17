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
#include "Core/Utilities/Tools/Utils.h"
#include "QAlg/Grover/GroverAlgorithm.h"
#include "QAlg/Grover/QuantumWalkGroverAlg.h"

USING_QPANDA
using namespace std;

static size_t g_shot = 100000;

static uint32_t classcial_search(const std::vector<uint32_t>& search_space, const std::vector<uint32_t>& search_data,
	std::vector<size_t>& search_result)
{
	search_result.clear();
	uint32_t search_times = 0;
	for (size_t i = 0; i < search_space.size(); ++i)
	{
		if (search_space[i] == search_data[0]){
			search_result.emplace_back(i);
		}
		++search_times;
	}

	return search_times;
}

static uint32_t quantum_grover_search(const std::vector<uint32_t>& search_space, const std::vector<uint32_t>& search_data,
	std::vector<size_t>& search_result)
{
	search_result.clear();
	uint32_t search_times = 0;
	uint32_t repeat_times = 0;
	auto machine = initQuantumMachine(CPU);
	machine->setConfigure({ 64,64 });
	auto x = machine->allocateCBit();

	std::vector<size_t> search_result_for_check;
	for (size_t i = 0; i < search_space.size(); ++i)
	{
		if (search_space[i] == search_data[0]) {
			search_result_for_check.emplace_back(i);
		}
	}

	cout << "Grover will search through " << search_space.size() << " data." << endl;
	cout << "Start grover search algorithm:" << endl;
	
	QProg grover_Qprog;
	QVec measure_qubits;
	uint32_t qubit_size = 0;
	vector<ClassicalCondition> c;
	const double max_repeat = 3.1415926 * sqrt(((double)search_space.size()) / ((double)search_result_for_check.size())) / 4.0;
	while (true)
	{
		measure_qubits.clear();
		grover_Qprog = build_grover_prog(search_space, x == search_data[0], machine, measure_qubits, ++repeat_times);
		search_times += repeat_times;

		if (0 == qubit_size){
			QVec _qv;
			qubit_size = grover_Qprog.get_used_qubits(_qv);
			printf("Number of used-qubits: %u.\n", qubit_size);
		}

		if (0 == c.size()){
			c = machine->allocateCBits(measure_qubits.size());
		}

		//measure
		printf("Strat measure.\n");
		//auto result = probRunDict(grover_Qprog, measure_qubits);
		grover_Qprog << MeasureAll(measure_qubits, c);
		auto result = runWithConfiguration(grover_Qprog, c, g_shot);

		//get result
		std::map<string, double> normal_result;
		for (const auto& _r : result){
			normal_result.insert(std::make_pair(_r.first, (double)_r.second/(double)g_shot));
		}
		search_result = search_target_from_measure_result(normal_result);
		if ((search_result.size() > 0)
			|| ((search_result.size() == 0) && (max_repeat < repeat_times))){
			break;
		}
	}

	/*if (search_result_for_check.size() != search_result.size())
	{
		search_result.clear();
	}
	else
	{
		for (size_t i = 0; i < search_result_for_check.size(); ++i)
		{
			if (search_result[i] != search_result_for_check[i]) {
				search_result.clear();
				break;
			}
		}
	}*/
	

	//for test
	//write_to_originir_file(grover_Qprog, machine, "grover_prog_0.txt");
	
	/*grover_Qprog << MeasureAll(measure_qubits, c);*/
	cout << "Draw grover_Qprog:" << grover_Qprog << endl;
	
	destroyQuantumMachine(machine);
	return search_times;
}

static uint32_t quantum_walk_search(const std::vector<uint32_t>& search_space, const std::vector<uint32_t>& search_data,
	std::vector<size_t>& search_result)
{
	search_result.clear();
	uint32_t search_times = 0;
	uint32_t repeat_times = 1;
	auto machine = initQuantumMachine(CPU);
	machine->setConfigure({ 64,64 });
	auto x = machine->allocateCBit();

	std::vector<size_t> search_result_for_check;
	for (size_t i = 0; i < search_space.size(); ++i)
	{
		if (search_space[i] == search_data[0]) {
			search_result_for_check.emplace_back(i);
		}
	}

	cout << "quantum-walk-alg will search through " << search_space.size() << " data." << endl;
	cout << "Start quantum-walk search algorithm:" << endl;

	QProg quantum_walk_prog;
	QVec measure_qubits;
	uint32_t qubit_size = 0;
	vector<ClassicalCondition> c;
	const double max_repeat = 3.1415926 * sqrt(((double)search_space.size()) / ((double)search_result_for_check.size())) / 4.0;
	while (true)
	{
		measure_qubits.clear();
		quantum_walk_prog = build_quantum_walk_search_prog(search_space,
			x == search_data[0], machine, measure_qubits, ++repeat_times);

		search_times += repeat_times;

		if (0 == qubit_size) {
			QVec _qv;
			qubit_size = quantum_walk_prog.get_used_qubits(_qv);
			printf("Number of used-qubits: %u.\n", qubit_size);
		}

		if (0 == c.size()) {
			c = machine->allocateCBits(measure_qubits.size());
		}

		//measure
		printf("Strat measure.\n");
		//auto result = probRunDict(quantum_walk_prog, measure_qubits);
		quantum_walk_prog << MeasureAll(measure_qubits, c);
		auto result = runWithConfiguration(quantum_walk_prog, c, g_shot);

		//get result
		std::map<string, double> normal_result;
		for (const auto& _r : result) {
			normal_result.insert(std::make_pair(_r.first, (double)_r.second / (double)g_shot));
		}
		search_result = search_target_from_measure_result(normal_result);
		if ((search_result.size() > 0)
			|| ((search_result.size() == 0) && (4 < repeat_times))) {
			break;
		}
	}

	/*if (search_result_for_check.size() != search_result.size())
	{
		search_result.clear();
	}
	else
	{
		for (size_t i = 0; i < search_result_for_check.size(); ++i)
		{
			if (search_result[i] != search_result_for_check[i]) {
				search_result.clear();
				break;
			}
		}
	}*/

	//for test
	//write_to_originir_file(quantum_walk_prog, machine, "grover_prog_0.txt");
	//auto c = machine->allocateCBits(measure_qubits.size());
	/*quantum_walk_prog << MeasureAll(measure_qubits, c);*/
	//cout << "Draw quantum_walk_prog:" << quantum_walk_prog << endl;

	destroyQuantumMachine(machine);
	return search_times;
}

static uint32_t read_search_space_file(const std::string& data_file, std::vector<uint32_t>& search_space)
{
	ifstream data_file_reader(data_file);
	if (!data_file_reader.is_open()){
		QCERR_AND_THROW(run_fail, "can not open this file: " << data_file);
	}

	auto trim = [](string& str) {
		size_t _pos = 0;
		const char* _s = " ";
		_pos = str.find(_s, _pos);
		while (string::npos != _pos) {
			str.replace(_pos, 1, "");
			_pos = str.find(_s, _pos);
		}
	};

	auto get_line_data_fun = [&trim](const string& line_str, std::vector<uint32_t>& data_vec) {
		uint32_t offset = 0;
		size_t _pos_1 = 0;
		while (true)
		{
			_pos_1 = line_str.find_first_of(',', offset);
			if (string::npos == _pos_1) {
				break;
			}

			auto _data_str = line_str.substr(offset, _pos_1 - offset);
			//trim(complex_data_str);
			offset = _pos_1 + 1;

			data_vec.push_back(atoi(_data_str.c_str()));
		}
	};

	string line_data;
	uint32_t line_index = 0;
	while (getline(data_file_reader, line_data)){
		get_line_data_fun(line_data, search_space);
	}

	return search_space.size();
}

int main(int argc, char* argv[])
{
	const std::string parameter_descr_str = R"(
    The legal parameter form is as follows:
    QGrover.exe [Option] [search-space-file] [search-data]
    Option:
        -g: run Grover-search-algorithm
        -c: run classcial-search-algorithm
        -r: run quantum-walk-search-algorithm
    example:
    To search for 100 in data.txt use Grover-search-algorithm, execute the following command:
        Search.exe -g data.txt 100
    )";
#if 1
	/* read param
	*/
	int search_type = -1; /**< 0: classcial-search-algorithm; 1:Grover-search-algorithm; 2:quantum-walk-search-algorithm */
	std::string data_file = "";
	std::vector<uint32_t> search_data;
	try
	{
		if (argc >= 4)
		{
			if (0 == strcmp(argv[1], "-c")){
				search_type = 0;
			}
			else if (0 == strcmp(argv[1], "-g")) {
				search_type = 1;
			}
			else if (0 == strcmp(argv[1], "-r")) {
				search_type = 2;
			}
			else{
				QCERR_AND_THROW(init_fail, "Error: parameter error.");
			}

			data_file = argv[2];
			cout << "search-space-file: " << data_file << endl;

			for (size_t i = 3; i < argc; ++i){
				search_data.emplace_back(atol(argv[i]));
			}
			cout << "search-data: ";
			for (const auto& _data : search_data){
				cout << _data << " ";
			}
			cout << endl;
		}
		else{
			QCERR_AND_THROW(init_fail, "Error: parameter error.");
		}
	}
	catch (...)
	{
		cout << "Parameter error." << parameter_descr_str << endl;
		return -1;
	}
#else
	int search_type = 2;
	std::string data_file = "data1.txt";
	std::vector<uint32_t> search_data = {21};
#endif
	/* run search algorithm
	*/
	std::vector<size_t> search_result;
	uint32_t search_cnt = 0;
	try
	{
		std::vector<uint32_t> search_space;
		uint32_t search_space_size = read_search_space_file(data_file, search_space);
		cout << "search_space_size: " << search_space_size << endl;

		switch (search_type)
		{
		case 0:
			search_cnt = classcial_search(search_space, search_data, search_result);
			break;

		case 1:
			search_cnt = quantum_grover_search(search_space, search_data, search_result);
			break;

		case 2:
			search_cnt = quantum_walk_search(search_space, search_data, search_result);
			break;

		default:
			QCERR_AND_THROW(run_fail, "Error: search type error.");
			break;
		}
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
		return -1;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
		return -1;
	}

	cout << "Search result:\n";
	for (const auto &result_item : search_result)
	{
		cout << result_item << " ";
	}

	string search_str;
	switch (search_type)
	{
	case 0:
		search_str = "classic search";
		break;

	case 1:
		search_str = "quantum-Grover search";
		break;

	case 2:
		search_str = "quantum-walk search";
		break;
	}

	cout << "\nThe number of " << search_str << " queries: " << search_cnt;
	cout << "\nSearch over." << endl;
	/*cout << "\nSearch over, press Enter to continue." << endl;
	getchar();*/

	return 0;
}


