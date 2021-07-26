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

#include "Extensions/Extensions.h"

using namespace std;
using namespace QPanda;

static uint32_t g_precision = 0;
static string g_hhl_data_file;
static const string g_result_file_prefix = "HHL_result_";

static uint32_t load_data_file(const std::string& data_file, std::vector<double>& A, std::vector<double>& b)
{
	ifstream data_file_reader(data_file);
	if (!data_file_reader.is_open()) {
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

	auto get_line_data_fun = [&trim](const string& line_str, std::vector<double>& data_vec) {
		uint32_t offset = 0;
		size_t _pos_1 = 0;
		while (true)
		{
			_pos_1 = line_str.find_first_of(' ', offset);
			if (string::npos == _pos_1) {
				if (line_str.length() > offset)
				{
					_pos_1 = line_str.length();
				}
				else
				{
					break;
				}
				
			}

			auto _data_str = line_str.substr(offset, _pos_1 - offset);
			//trim(complex_data_str);
			offset = _pos_1 + 1;
			if (_data_str.length() == 0){
				continue;
			}

            const auto _c = _data_str.at(0);
            if((_c < '0' || _c > '9') && (_c != '-'))
            {
                break;
            }

			data_vec.push_back(atof(_data_str.c_str()));
		}
	};

	string line_data;
	uint32_t line_index = 0;
	uint32_t A_dimension = 0;
	bool b_read_A_end = false;
	while (getline(data_file_reader, line_data)) {
		if (b_read_A_end){
			get_line_data_fun(line_data, b);
			break;
		}
		else{
			get_line_data_fun(line_data, A);
		}
		
		if (++line_index == 1)
		{
			A_dimension = A.size();
		}
		else if (line_index == A_dimension)
		{
			b_read_A_end = true;
		}
	}

	if ((b.size() != A_dimension) || (A.size() != (A_dimension * A_dimension)))
	{
		QCERR_AND_THROW(init_fail, "Error: data-file error, A_dimension = " << A_dimension);
	}

	return A_dimension;
}

static std::string _tostring(const double val, const int precision = 15)
{
	std::ostringstream out;
	out.precision(precision);
	out << val;
	return out.str();
}

static void HHL_run(const std::string& data_file)
{
	std::vector<double> A;
	std::vector<double> b;
	uint32_t A_dimension = load_data_file(data_file, A, b);

	QStat Q_A;
	for (const auto& i : A){
		Q_A.push_back(i);
	}

	/*cout << "A:\n" << Q_A << "\n";
	cout << "b:\n";
	for (const auto& i : b) {
		cout << _tostring(i) << " ";
	}
	cout << "\n";*/

	QStat result = HHL_solve_linear_equations(Q_A, b, g_precision);
	int w = 0;

	auto _file_name_pos = g_hhl_data_file.find_last_of('/');
	if (_file_name_pos == (std::numeric_limits<size_t>::max)())
	{
		_file_name_pos = g_hhl_data_file.find_last_of('\\');
		if (_file_name_pos == (std::numeric_limits<size_t>::max)()){
			_file_name_pos = 0;
		}
	}

	string output_file;
	if (0 == _file_name_pos){
		output_file = g_result_file_prefix + g_hhl_data_file;
	}
	else{
		output_file = g_result_file_prefix + g_hhl_data_file.substr(_file_name_pos + 1);
	}
	
	ofstream outfile(ofstream(output_file, ios::out | ios::binary));
	if (!outfile.is_open())
	{
		QCERR("Can NOT open the output file: " << output_file);
	}

	cout << "HHL_result of " << g_hhl_data_file << ": " << A_dimension << "-dimensional matrix:\n";
	outfile << "HHL_result of " << g_hhl_data_file << ": " << A_dimension << "-dimensional matrix:\n";
	for (auto &val : result)
	{
		std::cout << val << " ";
		outfile << _tostring(val.real()).c_str() << ", " << _tostring(val.imag());
		//if (++w == 2)
		{
			//w = 0;
			std::cout << "\n";
			outfile << "\n";
		}
	}
	std::cout << std::endl;
	outfile << std::endl;
	if (outfile.is_open()) { outfile.close(); }
	
	return;
}

int main(int argc, char* argv[])
{
	const std::string parameter_descr_str = R"(
    Version: 2.2
    The legal parameter form is as follows:
    HHL_Algorithm [data-file] [precision]
    data-file: configure A and b for linear-system-equation: Ax=b.
    precision: The count of digits after the decimal point,
			 default is 0, indicates that there are only integer solutions.
    example:
    To run data.txt through HHL algorithm, and the precision is 0.01, execute the following command:
        HHL_Algorithm data.txt 2

    data_file example:
      10   -4    1    0   -4    0    0    0
      -4   11   -4    1    0   -4    0    0
       1   -4   11   -4    0    0   -4    0
       0    1   -4   10    0    0    0   -4
      -4    0    0    0   14   -4    1    0
       0   -4    0    0   -4   11   -4    1
       0    0   -4    0    1   -4   15   -4
       0    0    0   -4    0    1   -4   10
       6760.4076656465 97.7868332786 9.2014490849 6812.0635727021 15690.3500503387 36.7530619312 9452.9188994963 6752.0327860161
    )";

	//std::string data_file = "data.txt";
#if 0
	try
	{
		if (argc == 3){
			g_hhl_data_file = argv[1];
			g_precision = atoi(argv[2]);
			cout << "got param, data file: " << g_hhl_data_file << 
				", precision: " << 1.0/(double)pow(10, g_precision) << endl;
		}
		else{
			QCERR_AND_THROW(init_fail, "Error: parameter error: Incomplete parameters.");
		}
	}
	catch (...)
	{
		cout << parameter_descr_str << endl;
		return -1;
	}
#else
	//g_hhl_data_file = "11HHL_test_data.txt";
	g_hhl_data_file = "E://tmp//HHL_test//QuntumLY32.txt";
	g_precision = 1;
#endif

	HHL_run(g_hhl_data_file);

	cout << "HHL_Algorithm run over, press Enter to continue." << endl;
	getchar();

	return 0;
}
