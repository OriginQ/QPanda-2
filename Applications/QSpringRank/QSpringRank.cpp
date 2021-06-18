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
#include "QAlg/HHL/HHL.h"
#include <time.h>
#include <fstream>
#include <iostream>

using namespace std;
using namespace QPanda;

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceCircuit(cir) (std::cout << cir << endl)
#define PTraceCircuitMat(cir) { auto m = getCircuitMatrix(cir); std::cout << m << endl; }
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceCircuit(cir)
#define PTraceCircuitMat(cir)
#define PTraceMat(mat)
#endif

using AdjacentDataT = double;
using AdjacentMatrix = std::vector<std::vector<AdjacentDataT>>;

size_t get_rank_element_cnt(AdjacentMatrix &mat)
{
	const size_t rank_element_cnt = mat[0].size();
	if (rank_element_cnt != mat.size())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: the size of the input AdjacentMatrix for SpringRank must be N*N.");
	}

	return rank_element_cnt;
}

static vector<unsigned int> get_rank(QStat& hhl_result, const size_t element_cnt)
{
	using sort_item = std::pair<double, unsigned int>;
	vector<sort_item> sort_vec;
	unsigned int index = 0;
	for (const auto& i : hhl_result)
	{
		sort_vec.push_back(sort_item(i.real(), index));
		if (element_cnt == (++index))
		{
			break;
		}
	}
	sort(sort_vec.begin(), sort_vec.end(), [](sort_item& a, sort_item& b) {return a.first > b.first; });

	vector<unsigned int> rank_s;
	for (auto& i : sort_vec)
	{
		rank_s.push_back(i.second);
	}

	return rank_s;
}

static int adjacent_matrix_to_hermitian(const AdjacentMatrix& adjacent_mat, QStat &A, std::vector<double>& b)
{
	int ret = 0;
	size_t rows = adjacent_mat.size();
	std::vector<double> k_in(rows, 0);
	std::vector<double> k_out;

	for (size_t i = 0; i < rows; ++i)
	{
		double t = 0;
		double l = 0;
		for (size_t j = 0; j < rows; ++j)
		{
			t += adjacent_mat[i][j];
			k_in.at(j) += adjacent_mat[i][j];
		}
		k_out.push_back(t);
	}

	/*cout << "k_in:" << endl;
	for (auto i : k_in)
	{
		cout << i << " ";
	}
	cout << endl;

	cout << "k_out:" << endl;
	for (auto i : k_out)
	{
		cout << i << " ";
	}
	cout << endl;*/

	for (size_t i = 0; i < rows; ++i)
	{
		b.push_back(k_out[i] - k_in[i]);
	}

	/*cout << "b:" << endl;
	for (auto i : b)
	{
		cout << i << " ";
	}
	cout << endl;*/


	A.clear();
	A.resize((rows)*(rows));
	for (size_t i = 0; i < rows; ++i)
	{
		A[(rows)*i + i] = (k_out[i] + k_in[i]);
	}

	/*cout << "tmp A:" << endl;
	cout << A << endl;*/

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < rows; ++j)
		{
			A[(rows)*i + j] -= (adjacent_mat[i][j] + adjacent_mat[j][i]);
		}
	}

	/*cout << "result A:" << endl;
	cout << A << endl;*/

	return ret;
}

static double get_random(double min = 0, double max = 200)
{
	double s;
	//static long last_rand = clock();
	//static long last_rand = 9;
	//srand(last_rand);
	//auto last_rand = rand();
	s = ((double)(rand()% 1000) / 1000.0)*(max - min) + min;
	return s;
}

static AdjacentMatrix build_random_matrix(size_t dimension_cnt)
{
	AdjacentMatrix ret_matrix(dimension_cnt);
    //srand(9);
    srand(time(NULL));
	const float edges_num = 3;
	float threshold_val = 4096.0 * edges_num / (float)dimension_cnt;
	for (size_t col = 0; col < dimension_cnt; ++col)
	{
		for (size_t i = 0; i < dimension_cnt; ++i)
		{
			if (col == i)
			{
				ret_matrix[col].push_back(0);
				continue;
			}

			const auto r = get_random(0, 4096);
			//ret_matrix[col].push_back(r > 5 ? 0 : r);
			ret_matrix[col].push_back(r > threshold_val ? 0 : get_random(1, 10));
			//PTrace("Got a rand-num: %d.\n", ret_matrix[col].back());
		}
	}

	return ret_matrix;
}

static AdjacentMatrix build_random_diagonal_matrix(size_t dimension_cnt)
{
	AdjacentMatrix ret_matrix(dimension_cnt);

	for (size_t col = 0; col < dimension_cnt; ++col)
	{
		for (size_t i = 0; i < dimension_cnt; ++i)
		{
			if (col == i)
			{
				auto r = get_random(0, 1.0);
				ret_matrix[col].push_back(r > 7 ? 0 : r);
				PTrace("Got a rand-num: %f.\n", ret_matrix[col].back());
				continue;
			}
			ret_matrix[col].push_back(0);
		}
	}

	return ret_matrix;
}

static vector<unsigned int> quantum_spring_rank(AdjacentMatrix &mat)
{
	const size_t rank_element_cnt = get_rank_element_cnt(mat);
	QStat A222;
	for (size_t i = 0; i < mat.size(); i++)
	{
		for (auto& j : mat.at(i))
		{
			A222.push_back(j);
		}
	}
	std::cout << "The A222:" << endl << A222 << endl;
	QStat A;
	std::vector<double> b;
	adjacent_matrix_to_hermitian(mat, A, b);

	HHLAlg::expand_linear_equations(A, b);

#if PRINT_TRACE
	std::cout << "The b:" << endl;
	for (auto& i : b)
	{
		//i = 1;
		std::cout << i << ", ";
	}

	std::cout << endl;
	std::cout << "The A:" << endl << A << endl;

	/*cout << "Press Enter to coninute." << endl;
	getchar();*/
#endif

	QStat result = HHL_solve_linear_equations(A, b, 2);

#if 1
	std::cout << "HHL result:" << endl;
	int w = 0;
	for (auto &val : result)
	{
		std::cout << val << " ";
		//if (++w == 2)
		{
			w = 0;
			std::cout << std::endl;
		}
	}
	std::cout << std::endl;
#endif

	return get_rank(result, rank_element_cnt);
}
#if 1
/** The target rank: 
19, 22, 17, 18, 25, 23, 21, 29, 20, 14, 16, 1, 12, 27, 0, 15, 3, 4, 10, 28, 13, 30, 24, 5, 2, 9, 26, 7, 6, 8, 11,
*/
bool SpringRank_test1()
{
	cout << "On SpringRank_test1." << endl;
	QStat A;
	std::vector<double> b;

	const size_t rank_element_cnt = 31;
	AdjacentMatrix adjacent_mat(rank_element_cnt, std::vector<AdjacentDataT>(rank_element_cnt, 0));
	adjacent_mat[19][25] = 2;
	adjacent_mat[25][18] = 1;
	adjacent_mat[17][25] = 3;
	adjacent_mat[18][29] = 2;
	adjacent_mat[17][18] = 1;
	adjacent_mat[17][16] = 2;
	adjacent_mat[29][16] = 4;
	adjacent_mat[16][0] = 3;
	adjacent_mat[16][27] = 3;
	adjacent_mat[27][30] = 1;
	adjacent_mat[27][15] = 2;
	adjacent_mat[15][26] = 4;
	adjacent_mat[15][24] = 5;
	adjacent_mat[24][8] = 2;
	adjacent_mat[24][9] = 1;
	adjacent_mat[2][8] = 3;
	adjacent_mat[2][6] = 2;
	adjacent_mat[6][11] = 3;
	adjacent_mat[28][9] = 3;
	adjacent_mat[28][7] = 1;
	adjacent_mat[9][7] = 2;
	adjacent_mat[7][11] = 2;
	adjacent_mat[13][7] = 3;
	adjacent_mat[10][11] = 1;
	adjacent_mat[5][11] = 2;
	adjacent_mat[5][2] = 1;
	adjacent_mat[3][2] = 3;
	adjacent_mat[0][2] = 2;
	adjacent_mat[1][0] = 2;
	adjacent_mat[1][3] = 1;
	adjacent_mat[3][4] = 1;
	adjacent_mat[4][3] = 2;
	adjacent_mat[3][5] = 3;
	adjacent_mat[10][4] = 1;
	adjacent_mat[12][13] = 2;
	adjacent_mat[14][12] = 1;
	adjacent_mat[20][12] = 6;
	adjacent_mat[23][14] = 1;
	adjacent_mat[22][23] = 2;
	adjacent_mat[22][21] = 1;
	adjacent_mat[19][21] = 1;
	adjacent_mat[19][20] = 2;
	adjacent_mat[22][20] = 3;

	auto rank = quantum_spring_rank(adjacent_mat);

	cout << "Got ranks:" << endl;
	for (auto &val : rank)
	{
		std::cout << val << ", ";
	}
	std::cout << std::endl;

	return true;
}

/* the target sort should be: 1,4,3,0,5,2,6
*/
bool SpringRank_test3()
{
	QStat A;
	std::vector<double> b;

	const size_t rank_element_cnt = 7;
	AdjacentMatrix adjacent_mat(rank_element_cnt, std::vector<AdjacentDataT>(rank_element_cnt, 0));
	adjacent_mat[1][3] = 1;
	adjacent_mat[1][0] = 2;
	adjacent_mat[3][4] = 1;
	adjacent_mat[4][3] = 2;
	adjacent_mat[3][5] = 3;
	adjacent_mat[3][2] = 3;
	adjacent_mat[0][2] = 2;
	adjacent_mat[5][2] = 1;
	adjacent_mat[2][6] = 2;

	auto rank = quantum_spring_rank(adjacent_mat);
	cout << "Got ranks:" << endl;
	for (auto &val : rank)
	{
		std::cout << val << ", ";
	}
	std::cout << std::endl;

	return true;
}

bool SpringRank_test4()
{
	QStat A;
	std::vector<double> b;

	QCircuit cir;
	QCircuit cir2;

	const size_t rank_element_cnt = 3;
	AdjacentMatrix adjacent_mat(rank_element_cnt, std::vector<AdjacentDataT>(rank_element_cnt, 0));
	adjacent_mat[0][1] = 1;
	adjacent_mat[1][0] = 1;

	auto rank = quantum_spring_rank(adjacent_mat);
	cout << "Got ranks:" << endl;
	for (auto &val : rank)
	{
		std::cout << val << ", ";
	}
	std::cout << std::endl;

	return true;
}
#endif
bool SpringRank_test5(const size_t cnt = 31)
{
	const size_t rank_element_cnt = cnt;
	AdjacentMatrix test_random_matrix = build_random_matrix(rank_element_cnt);
	//AdjacentMatrix test_random_matrix = build_random_diagonal_matrix(rank_element_cnt);

	auto rank = quantum_spring_rank(test_random_matrix);
	cout << "Got ranks:" << endl;
	for (auto &val : rank)
	{
		std::cout << val << ", ";
	}
	std::cout << std::endl;

	return true;
}

AdjacentMatrix get_matrix_from_file(const std::string data_file)
{
	ifstream data_file_reader(data_file);
	if (!data_file_reader.is_open())
	{
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

	auto get_line_data_fun = [&trim](const string& line_str, std::vector<AdjacentDataT>& data_vec) {
		uint32_t offset = 0;
		size_t _pos_1 = 0;
		size_t _pos_2 = 0;
		size_t interval_pos = 0;/**< 复数实部虚部的分割位置 */
		data_vec.clear();
		while (true)
		{
			_pos_1 = line_str.find_first_of('(', offset);
			if (string::npos == _pos_1) {
				break;
			}

			_pos_2 = line_str.find_first_of(')', _pos_1);
			if (string::npos == _pos_2) {
				QCERR_AND_THROW(run_fail, "Error, filed to parse line_data: " << line_str << " on start pos:" << _pos_1);
			}

			auto complex_data_str = line_str.substr(_pos_1 + 1, _pos_2 - _pos_1 - 1);
			//trim(complex_data_str);
			offset = _pos_2;

			interval_pos = complex_data_str.find_first_of(',');
			if (string::npos == interval_pos) {
				QCERR_AND_THROW(run_fail, "Error, filed to parse complex_data_str: " << complex_data_str);
			}

			double _real = atof(complex_data_str.substr(0, interval_pos).c_str());
			double _imag = atof(complex_data_str.substr(interval_pos + 1).c_str());
			data_vec.push_back(_real);
		}
	};

	size_t rank_element_cnt = 0;
	AdjacentMatrix matrix;
	string line_data;
	uint32_t line_index = 0;
	while (getline(data_file_reader, line_data))
	{
		if (0 == line_index)
		{
			std::vector<AdjacentDataT> data_vec;
			get_line_data_fun(line_data, data_vec);
			rank_element_cnt = data_vec.size();
			matrix.resize(rank_element_cnt);
			matrix[0].swap(data_vec);
		}
		else
		{
			get_line_data_fun(line_data, matrix[line_index]);
			if (matrix[line_index].size() != rank_element_cnt){
				QCERR_AND_THROW(run_fail, "Error: read matrix-date file error on line:" << line_index);
			}
		}

		++line_index;
	}

	return matrix;
}

bool SpringRank_test6(const std::string data_file)
{
	auto matrix = get_matrix_from_file(data_file);
	auto rank = quantum_spring_rank(matrix);
	cout << "Got ranks:" << endl;
	for (auto &val : rank)
	{
		std::cout << val << ", ";
	}
	std::cout << std::endl;

	return true;
}

int main(int argc, char* argv[])
{
	bool test_val = false;
	uint32_t d = 0;
	std::string data_file = "data.txt";
	if (argc > 1)
	{
		if (nullptr != strstr(argv[1], ".txt"))
		{
			data_file = argv[1];
			cout << "got param, data file: " << data_file << endl;
		}
		else
		{
			d = atoi(argv[1]);
			cout << "got param: " << d << endl;
		}
		
	}
	else
	{
		cout << "default param: " << data_file << endl;
	}
	

	try
	{
		//data_file = ;
		////test_val = SpringRank_test6(data_file);
		//test_val = SpringRank_test1();

		if (0 < d)
		{
			//test_val = SpringRank_test1();
			//test_val = SpringRank_test3();
			//test_val = SpringRank_test4();
			test_val = SpringRank_test5(d);
		}
		else
		{
			test_val = SpringRank_test6(data_file);
		}
	}
	catch (const std::exception& e)
	{
		cout << "Got a exception: " << e.what() << endl;
	}
	catch (...)
	{
		cout << "Got an unknow exception: " << endl;
	}

	cout << "QSpringRank run over, press Enter to continue." << endl;
	getchar();

	return 0;
}
