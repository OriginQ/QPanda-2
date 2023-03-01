#include"include/QAlg/Error_mitigation/Sample.h"
#include <bitset>
USING_QPANDA

inline std::string ull2binary(const size_t& value, const size_t& length) {
	std::bitset<64> temp(value);
	std::string str = temp.to_string();
	std::string str_back(str.begin() + (64 - length), str.end());
	return str_back;
}
inline std::string bit_balance(const char &c,const int &repeat) {
	std::string str = "";
	for (int i = 0; i < repeat; ++i) {
		str += c;
	}
	return str;
}
inline std::string generate_random_str(int randomlength){
	int m = rand() % ((1<<randomlength) + 1);
	std::bitset<64> temp(m);
	std::string str = temp.to_string();
	std::string random_str(str.begin() + (64- randomlength), str.end());
	return random_str;
}
Sample::Sample(const QVec &q) {
	total_qubit = q;
	size = q.size();
}
std::vector<QCircuit> Sample::balance_sample() {
	for (int i = 1; i < size+1; ++i) {
		std::string str1,str2;
		str1 = "";
		str2 = "";
		int size_j = ceil((double)size / (double)i);
		for (int j = 0; j < size_j; ++j) {
			str1 += bit_balance((j&1)+'0',i);
			str2 += bit_balance(((j+1) & 1) + '0', i);
		}
		std::string str_tmp1(str1.begin(), str1.begin() + size);
		std::string str_tmp2(str2.begin(), str2.begin() + size);
		cir_str.push_back(str_tmp1);
		cir_str.push_back(str_tmp2);
	}
	std::vector<QCircuit> circ;
	for (int i = 0; i < cir_str.size(); ++i) {
		int cnt = 0;
		QCircuit circuit = QCircuit();
		for (int j = size - 1; j >= 0; --j) {
			if (cir_str[i][j] == '1') {
				circuit << X(total_qubit[cnt]);
			}
			cnt++;
		}
		circ.push_back(circuit);
	}
	return circ;
}
std::vector<QCircuit> Sample::bit_flip_average_sample() 
{

	size_t n_measure = 4 * (1<<size);
	for (int i = 0; i < n_measure; ++i) {
		cir_str.push_back(generate_random_str(size));
	}
	std::vector<QCircuit> circ;
	for (int i = 0; i < cir_str.size(); ++i) {
		int cnt = 0;
		QCircuit circuit = QCircuit();
		for (int j = size - 1; j >= 0; --j) {
			if (cir_str[i][j] == '1') {
				circuit << X(total_qubit[cnt]);
			}
		}
		circ.push_back(circuit);
	}
	return circ;
}

std::vector<QCircuit> Sample::independent_sample() 
{
	std::vector<QCircuit> circ;
	for (int i = 0; i < size; ++i) {
		QCircuit circuit0,circuit1;
		circuit1 << X(total_qubit[i]);
		circ.push_back(circuit0);
		circ.push_back(circuit1);
		cir_str.push_back(ull2binary(0, size));
		cir_str.push_back(ull2binary((1<<i), size));
	}
	return circ;
}
std::vector<QCircuit> Sample::full_sample()
{
	std::vector<QCircuit> circ;
	for (int i = 0; i < (1 << size); ++i) {
		std::string str = ull2binary(i, size);
		cir_str.push_back(str);
		int cnt = 0;
		QCircuit cir = QCircuit();
		for (int j = 0; j < size; ++j) {
			if (str[size - j - 1] == '1') {
				cir << X(total_qubit[cnt]);
			}
			cnt++;
		}
		circ.push_back(cir);
	}
	return circ;
}
std::vector<std::string> Sample::get_cir_str() 
{
	return cir_str;
}