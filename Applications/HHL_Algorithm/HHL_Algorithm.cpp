/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

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
#include "preconditionByOrigin.h"
#include "Extensions/Extensions.h"

using namespace std;
using namespace QPanda;

extern std::string noise_json;
static string g_hhl_data_file;
static uint32_t g_precision = 0;
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
			if (_data_str.length() == 0) {
				continue;
			}

			const auto _c = _data_str.at(0);
			if ((_c < '0' || _c > '9') && (_c != '-'))
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
		if (b_read_A_end) {
			get_line_data_fun(line_data, b);
			break;
		}
		else {
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

static QStat HHL_run(const std::vector<double>& A, const std::vector<double>& b)
{
    QStat Q_A;
    for (const auto& i : A) {
        Q_A.push_back(i);
    }

    return HHL_solve_linear_equations(Q_A, b, g_precision);
}

static QStat HHL_run_with_preprocess(const std::vector<double>& A, const std::vector<double>& b)
{
    MatrixXd A_bad(b.size(), b.size());
    VectorXd b_bad(b.size());
    for (int i = 0; i < b.size(); ++i)
    {
        for (int j = 0; j < b.size(); ++j) {
            A_bad(i, j) = A[i * b.size() + j];
        }
        b_bad(i) = b[i];
    }

    //A'
    MatrixXd M;
    auto result = DynamicSparseApproximateInverse(A_bad, b_bad, 0.2, b.size(), M);
    MatrixXd A_prime(2 * b.size(), 2 * b.size());
    A_prime.topLeftCorner(b.size(), b.size()) = A_prime.bottomRightCorner(b.size(), b.size()) = MatrixXd::Zero(4, 4);
    A_prime.topRightCorner(b.size(), b.size()) = result.first;
    A_prime.bottomLeftCorner(b.size(), b.size()) = result.first.transpose();

    //b'
    auto b_good = result.second;
    std::vector<double> b2;
    for (int i = 0; i < b_good.size(); ++i){
        b2.push_back(b_good(i));
    }

    std::vector<double> b_prime;
    for (int i = 0; i < b.size(); ++i){
        b_prime.push_back(b2[i]);
    }

    for (int i = 0; i < b.size(); ++i){
        b_prime.push_back(0);
    }

    auto A_prime_mat = Eigen_to_QStat(A_prime);

    QStat Q_A;
    for (const auto& i : A) {
        Q_A.push_back(i);
    }

    /*cout << "A:\n" << Q_A << "\n";
    cout << "b:\n";
    for (const auto& i : b) {
        cout << _tostring(i) << " ";
    }
    cout << "\n";*/

    //QStat result = HHL_solve_linear_equations(Q_A, b, g_precision);
    QStat result_prime = HHL_solve_linear_equations(A_prime_mat, b_prime, g_precision);
    result_prime.erase(result_prime.begin(), result_prime.begin() + (result_prime.size() / 2));

    return result_prime;
}

static void run(const std::string& data_file)
{
	std::vector<double> A;
	std::vector<double> b;
	uint32_t A_dimension = load_data_file(data_file, A, b);
#if 0
	/*MatrixXd A_bad(b.size(), b.size());
	VectorXd b_bad(b.size());
	for (int i = 0; i < b.size(); ++i) 
    {
		for (int j = 0; j < b.size(); ++j) {
			A_bad(i, j) = A[i * b.size() + j];
		}
		b_bad(i) = b[i];
	}*/

	//A'
	/*MatrixXd M;
	auto result = DynamicSparseApproximateInverse(A_bad, b_bad, 0.2, b.size(), M);
	MatrixXd A_prime(2 * b.size(), 2 * b.size());
	A_prime.topLeftCorner(b.size(), b.size()) = A_prime.bottomRightCorner(b.size(), b.size()) = MatrixXd::Zero(4, 4);
	A_prime.topRightCorner(b.size(), b.size()) = result.first;
	A_prime.bottomLeftCorner(b.size(), b.size()) = result.first.transpose();*/

	//b'
	/*auto b_good = result.second;
	std::vector<double> b2;
	for (int i = 0; i < b_good.size(); ++i)
	{
		b2.push_back(b_good(i));
	}

	std::vector<double> b_prime;
	for (int i = 0; i < b.size(); ++i)
		b_prime.push_back(b2[i]);
	for (int i = 0; i < b.size(); ++i)
		b_prime.push_back(0);

	auto A_prime_mat = Eigen_to_QStat(A_prime);*/

	/*QStat Q_A;
	for (const auto& i : A) {
		Q_A.push_back(i);
	}*/

	/*cout << "A:\n" << Q_A << "\n";
	cout << "b:\n";
	for (const auto& i : b) {
		cout << _tostring(i) << " ";
	}
	cout << "\n";*/

	//QStat result = HHL_solve_linear_equations(Q_A, b, g_precision);
	//QStat result_prime = HHL_solve_linear_equations(A_prime_mat, b_prime, g_precision);
#endif
    QStat result = HHL_run(A, b);
    //QStat result = HHL_run_with_preprocess(A, b);

	auto _file_name_pos = g_hhl_data_file.find_last_of('/');
	if (_file_name_pos == (std::numeric_limits<size_t>::max)())
	{
		_file_name_pos = g_hhl_data_file.find_last_of('\\');
		if (_file_name_pos == (std::numeric_limits<size_t>::max)()) {
			_file_name_pos = 0;
		}
	}

	string output_file;
	if (0 == _file_name_pos) {
		output_file = g_result_file_prefix + g_hhl_data_file;
	}
	else {
		output_file = g_result_file_prefix + g_hhl_data_file.substr(_file_name_pos + 1);
	}

	ofstream outfile(ofstream(output_file, ios::out | ios::binary));
	if (!outfile.is_open()){
		QCERR("Can NOT open the output file: " << output_file);
	}

	cout << "HHL_result of " << g_hhl_data_file << ": " << A_dimension << "-dimensional matrix:\n";
	outfile << "HHL_result of " << g_hhl_data_file << ": " << A_dimension << "-dimensional matrix:\n";
	for (int i = 0, w = 0; i < result.size(); ++i)
	{
		std::cout << result[i] << " ";
		outfile << _tostring(result[i].real()).c_str() << ", " << _tostring(result[i].imag());
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

#define MAX_PRECISION 1e-10
static int test_hhl_luojg(void)
{
    auto qvm = CPUQVM();
    qvm.init();
    QVec qvec = qvm.qAllocMany(4);
    auto circuit = QCircuit();
    auto prog = QProg();
    circuit << H(qvec[1])
        << H(qvec[2])
        << RX(qvec[0], -PI / 2).control(qvec[1])
        << U1(qvec[1], 3 * PI / 4)
        << CNOT(qvec[2], qvec[0])
        << SWAP(qvec[1], qvec[2])
        << H(qvec[1])
        << U1(qvec[1], -PI / 2).control(qvec[2])
        << H(qvec[2]);

    prog << RY(qvec[0], PI)
        << circuit
        << SWAP(qvec[1], qvec[2])
        << RY(qvec[3], PI / 1024).control(qvec[2])
        << RY(qvec[3], PI / 2048).control(qvec[1])
        /*<< RY(qvec[3], PI / 16).control(qvec[2])
        << RY(qvec[3], PI / 32).control(qvec[1])*/
        << SWAP(qvec[2], qvec[1])
        << circuit.dagger();

    qvm.directlyRun(prog);
    double C_coeff = 1 / (sin(PI / 2048));
    //double C_coeff = 1 / (sin(PI / 16));
    cout << C_coeff << endl;
    auto stat = qvm.getQState();
    qvm.finalize();
    stat.erase(stat.begin(), stat.begin() + (stat.size() / 2));
    QStat stat_normed;
    for (auto& val : stat)
    {
        stat_normed.push_back(val * C_coeff);
    }
    for (auto& val : stat_normed)
    {
        qcomplex_t tmp_val((abs(val.real()) < MAX_PRECISION ? 0.0 : val.real()), (abs(val.imag()) < MAX_PRECISION ? 0.0 : val.imag()));
        val = tmp_val;
    }
    // get solution
    QStat result;
    for (size_t i = 0; i < 2; ++i)
    {
        result.push_back(stat_normed.at(i));

    }
    for (auto& val : result)
    {
        cout << val << endl;
    }
    return 0;
}

int main(int argc, char* argv[])
{
    {
        /* just for test */
        //test_hhl_luojg();
        //return 0;
    }

	std::cout << "Version: 2.3.230403" << std::endl;
	const std::string parameter_descr_str = R"(    
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
		if (argc == 3) {
			g_hhl_data_file = argv[1];
			g_precision = atoi(argv[2]);
			cout << "got param, data file: " << g_hhl_data_file <<
				", precision: " << 1.0 / (double)pow(10, g_precision) << endl;
		}
		else if (argc == 4)
		{
			noise_json = argv[3];
			g_hhl_data_file = argv[1];
			g_precision = atoi(argv[2]);
			cout << "got param, data file: " << g_hhl_data_file <<
				", precision: " << 1.0 / (double)pow(10, g_precision) << endl;

			std::cout << "noise file:" << noise_json << std::endl;
		}
		else {
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
	//g_hhl_data_file = "E://tmp//HHL_test//QuntumLY32.txt";
    g_hhl_data_file = "E:\\tmp\\HHL_test_DiLiSuo\\HHL_Ab_liuyi0402-test3u.txt";
	g_precision = 2;
#endif

	run(g_hhl_data_file);

	cout << "HHL_Algorithm run over." << endl;
    getchar();
	return 0;
}
