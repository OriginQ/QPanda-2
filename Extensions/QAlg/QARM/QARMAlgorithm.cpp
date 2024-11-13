#include "QAlg/QARM/QARMAlgorithm.h"
#include "Core/Utilities/Tools/Utils.h"
#include <set>
#include <cmath>
#include "QPanda.h"
using namespace std;
USING_QPANDA

QARM::QARM(std::vector <std::vector<std::string> > data)
{
	transaction_data = data;
	transaction_number = transaction_data.size();

	std::set<std::string>items_set;
	for (int i = 0; i < transaction_data.size(); i++)
	{
		for (int j = 0; j < transaction_data[i].size(); j++)
		{
			items_set.insert(transaction_data[i][j]);
		}
	}
	int id = 1;
	for (auto item : items_set)
	{
		items.push_back({ id });
		items_dict.insert({ id, item });
		id++;
	}

	for (int i = 0; i < transaction_data.size(); i++)
	{
		std::vector<int> temp(items.size(), 0);

		for (int j = 0; j < transaction_data[i].size(); j++)
		{
			for (auto it : items_dict)
			{
				if (transaction_data[i][j] == it.second)
					temp[it.first - 1] = it.first;
			}
		}
		transaction_matrix.push_back(temp);
	}

	items_length = items.size();
	items_qubit_number = get_qubit_number(items_length - 1);
	transaction_qubit_number = get_qubit_number(transaction_number - 1);
	index_qubit_number = items_qubit_number + transaction_qubit_number;
	digit_qubit_number = get_qubit_number(*max_element(items.begin(), items.end()));

	m_qvm = new CPUQVM();
	m_qvm->init();
}

QARM::~QARM()
{
	m_qvm->finalize();
	delete m_qvm;

}

// gets the number of qubits based on the number of items or transactions
int QARM::get_qubit_number(int number)
{
	return (int)floor(log2(number) + 1);
}

std::vector<int> QARM::create_c1(std::vector <std::vector <int > > data)
{
	std::vector<int> c1;
	for (auto transaction : data)
	{
		for (auto item : transaction)
		{
			auto iter = find(c1.begin(), c1.end(), item);
			if (item != 0 && iter == c1.end())
			{
				c1.push_back(item);
			}
		}
	}
	return c1;
}


// Draws a circuit based on the current number
QCircuit QARM::get_number_circuit(QVec qlist, int position, int number, int qubit_number)
{
	auto cir = QCircuit();

	std::string str_bin = dec2bin(number, qubit_number);

	reverse(str_bin.begin(), str_bin.end());
	for (int i = 0; i < str_bin.size(); i++)
	{
		if (str_bin[i] == '1')
		{
			cir << X(qlist[position + i]);
		}
	}

	return cir;
}

// circuit U code in Oracle
QCircuit QARM::encode_circuit(QVec qlist, int position, int index_qubit_number, int items_length, int transaction_number)
{
	auto cir = QCircuit();

	// the quantum bits of the control subline corresponding to the number of bits in the index space
	QVec control_qubit;

	for (int i = 0; i < index_qubit_number; i++)
	{
		control_qubit.push_back(qlist[position + i]);
	}

	int information_position = position + index_qubit_number;

	for (int n = 0; n < transaction_number; n++)
	{
		// X gate coding for the transaction line index quantum circuit
		auto t_x_cir = get_number_circuit(qlist, position + items_qubit_number, (int)pow(2, transaction_qubit_number) - 1 - n, transaction_qubit_number);

		for (int m = 0; m < items_length; m++)
		{
			auto  item_x_cir = get_number_circuit(qlist, position, (int)pow(2, items_qubit_number) - 1 - m, items_qubit_number);

			cir << item_x_cir;
			cir << t_x_cir;
			auto sub_cir = get_number_circuit(qlist, information_position, transaction_matrix[n][m], digit_qubit_number);
			cir << sub_cir.control(control_qubit);
			cir << t_x_cir;
			cir << item_x_cir;
		}
	}
	return cir;
}

// define the lookup circuit in Oracle
QCircuit QARM::query_circuit(QVec qlist, int position, int target_number)
{
	auto cir = QCircuit();

	auto sub_cir = get_number_circuit(qlist, position, (int)pow(2, digit_qubit_number) - target_number - 1, digit_qubit_number);
	cir << sub_cir;

	QVec control_qubit;
	for (int i = 0; i < digit_qubit_number; i++)
	{
		control_qubit.push_back(qlist[position + i]);
	}
	cir << X(qlist[position + digit_qubit_number]).control(control_qubit);
	cir << sub_cir;
	return cir;
}


// define the phase circuit in Oracle
QCircuit QARM::transfer_to_phase(QVec qlist, int position)
{
	auto cir = QCircuit();
	cir << U1(qlist[position + 1], PI).control({ qlist[position] });
	return cir;
}


// defining  Oracle circuit
QCircuit QARM::oracle_cir(QVec qlist, int position, int locating_number)
{
	// U circuit
	auto u_cir = encode_circuit(qlist, position, index_qubit_number, items_length, transaction_number);

	// S circuit
	auto  s_cir = query_circuit(qlist, position + index_qubit_number, locating_number);

	// phase circuit
	auto transfer_cir = transfer_to_phase(qlist, position + index_qubit_number + digit_qubit_number);

	auto cir = QCircuit();
	cir << u_cir;
	cir << s_cir;
	cir << transfer_cir;
	cir << s_cir.dagger();
	cir << u_cir.dagger();

	return cir;
}


//define  coin circuit
QCircuit QARM::coin_cir(QVec qlist, int position)
{
	int u1_position = position + 1 + 2 * index_qubit_number + digit_qubit_number;
	int swap_interval = index_qubit_number;
	int coin_position = position;
	auto cir = QCircuit();
	auto control_cir = QCircuit();
	QVec control_qubit;
	for (int i = 0; i < index_qubit_number; i++)
	{
		cir << H(qlist[coin_position + i]);
		cir << X(qlist[coin_position + i]);
		control_qubit.push_back(qlist[coin_position + i]);
	}

	control_cir << U1(qlist[u1_position], PI);
	cir << control_cir.control(control_qubit);

	for (int i = 0; i < index_qubit_number; i++)
	{
		cir << X(qlist[coin_position + i]);
		cir << H(qlist[coin_position + i]);
		cir << SWAP(qlist[coin_position + i], qlist[coin_position + i + swap_interval]);
	}
	return cir;
}

// define G（k）circuit
QCircuit QARM::gk_cir(QVec qlist, int position, int locating_number)
{
	auto cir = QCircuit();
	auto oracle = oracle_cir(qlist, position + index_qubit_number, locating_number);
	auto coin = coin_cir(qlist, position);
	cir << oracle;
	cir << coin;
	return cir;
}


//define loop iteration circuit
prob_dict QARM::iter_cir(QVec qlist, std::vector<ClassicalCondition> clist, int position, int locating_number, int iter_number)
{
	auto prog = QProg();
	for (int i = 0; i < index_qubit_number; i++)
	{
		prog << H(qlist[position + index_qubit_number + i]);
	}
	prog << X(qlist[position + 2 * index_qubit_number + digit_qubit_number]);
	prog << H(qlist[position + 2 * index_qubit_number + digit_qubit_number]);
	prog << X(qlist[position + 2 * index_qubit_number + digit_qubit_number + 1]);

	auto cir = gk_cir(qlist, position, locating_number);
	for (int i = 0; i < iter_number; i++)
	{
		prog << cir;
	}
	QVec result_qubit;
	for (int i = 0; i < index_qubit_number; i++)
	{
		result_qubit.push_back(qlist[i]);
	}

	prob_dict result = m_qvm->probRunDict(prog, result_qubit, -1);
	return result;
}

// iteration count
int QARM::iter_number()
{
	int count;
	int estimate_count = (int)floor(PI * sqrt(pow(2, index_qubit_number)) / 2);
	if (estimate_count % 2)
		count = estimate_count;
	else
		count = estimate_count + 1;

	if (count >= 9)
		count -= 4;

	return count;
}

// result reduction
std::vector<std::vector<int>> QARM::get_result(QVec qlist, std::vector<ClassicalCondition> clist, int position, int locating_number, int iter_number)
{
	prob_dict ret = iter_cir(qlist, clist, position, locating_number, iter_number);
	std::vector<double> val_list;
	for (auto val : ret)
	{
		val_list.push_back(round(val.second * 10000) / 10000);
	}

	double max_val = *max_element(val_list.begin(), val_list.end());

	std::vector<int> index;
	for (int i = 0; i < val_list.size(); i++)
	{
		if (fabs(max_val - val_list[i]) < 1e-6)
		{
			index.push_back(i);
		}
	}
	std::vector<std::vector<int>>  result = get_index(index);
	return result;
}


static int bin2dec(std::string str_bin)
{
	int i = 0;
	const char *pch = str_bin.c_str();
	while (*pch == '0' || *pch == '1')
	{
		i <<= 1;
		i |= *pch++ - '0';
	}
	return i;
}

// process the total query result index
std::vector<std::vector<int>> QARM::get_index(std::vector<int> index)
{
	std::vector<std::vector<int>> result;
	for (int i = 0; i < index.size(); i++)
	{
		int idx = index[i];
		std::string str_bin = dec2bin(idx, index_qubit_number);
		int transaction_index = bin2dec(str_bin.substr(0, transaction_qubit_number));
		int item_index = bin2dec(str_bin.substr(transaction_qubit_number, str_bin.length()));
		result.push_back({ transaction_index , item_index });
	}

	return result;
}

// find f1 [(1,), (2,), (3,), (4,)]
void QARM::find_f1(QVec qlist, std::vector<ClassicalCondition> clist, int position, std::vector<int> c1, double min_support,
	std::vector<std::vector<int > > &f1, std::map<std::vector<int>, std::pair<std::vector<int>, double> > &f1_dict)
{
	int iter_num = iter_number();
	std::map<int, std::vector<int> >ck_dict;

	for (auto locating_number : c1)
	{
		std::vector <int > temp;
		std::vector<std::vector<int>> result = get_result(qlist, clist, position, locating_number, iter_num);
		for (int i = 0; i < result.size(); i++)
		{
			temp.push_back(result[i][0]);
		}
		ck_dict.insert({ locating_number , temp });
	}


	for (auto iter : ck_dict)
	{
		int key = iter.first;
		std::vector<int> val = iter.second;
		double support = (double)val.size() / (double)transaction_number;

		if (support > min_support || fabs(support - min_support) < 1e-6)
		{
			f1_dict[{key}] = { val, support };
			f1.push_back({ key });
		}
	}
}

// find fk
void QARM::find_fk(int k, std::vector<std::vector<int > > &fk, std::map<std::vector<int>, std::pair<std::vector<int>, double> > &fk_dict, double min_support)
{
	std::vector<std::vector<int > > fn;
	std::map<std::vector<int>, std::pair<std::vector<int>, double> > fn_dict;
	int len_fk = fk.size();
	for (int i = 0; i < len_fk; i++)
	{
		for (int j = i + 1; j < len_fk; j++)
		{
			auto  first1 = fk[i].begin();
			auto last1 = fk[i].begin() + (k - 2);
			std::vector<int> L1(first1, last1);
			sort(L1.begin(), L1.end());

			auto  first2 = fk[j].begin();
			auto last2 = fk[j].begin() + (k - 2);
			std::vector<int> L2(first2, last2);
			sort(L2.begin(), L2.end());
			if (L1 == L2)
			{
				std::vector<int > c;
				set_union(fk[i].begin(), fk[i].end(), fk[j].begin(), fk[j].end(), back_inserter(c));//并集

				std::vector<int> index1 = fk_dict[fk[i]].first;
				std::vector<int> index2 = fk_dict[fk[j]].first;
				std::vector<int> index_list;

				set_intersection(index1.begin(), index1.end(), index2.begin(), index2.end(), back_inserter(index_list));//交集

				double support = (double)index_list.size() / (double)transaction_number;

				if (support > min_support || fabs(support - min_support) < 1e-6)
				{
					fn_dict[c] = { index_list, support };
					fn.push_back(c);
				}
			}
		}
	}
	fk = fn;
	fk_dict = fn_dict;
}

// frequent item set statistics
void QARM::fk_result(QVec qlist, std::vector<ClassicalCondition> clist, int position, double min_support,
	std::vector<std::vector<std::vector<int > > > &fn, std::map<std::vector<int>, std::pair<std::vector<int>, double> > &fn_dict)
{
	auto c1 = create_c1(transaction_matrix);
	std::vector<std::vector<int > >  f1, temp;
	std::map<std::vector<int >, std::pair<std::vector<int>, double> >f1_dict, temp_dict;
	find_f1(qlist, clist, position, c1, min_support, f1, f1_dict);

	int k = 2;
	auto fk = f1;
	auto fk_dict = f1_dict;
	while (fk.size())
	{
		fn.push_back(fk);

		for (auto iter : fk_dict)
			fn_dict.insert(iter);

		find_fk(k, fk, fk_dict, min_support);

		k += 1;
	}
}

// confidence calculation
double QARM::conf_x_y(double supp_xy, double supp_x)
{
	return supp_xy / supp_x;
}

static bool issubset(std::vector<int> v1, std::vector<int> v2)
{
	int i = 0, j = 0;
	int m = v1.size();
	int n = v2.size();
	if (m < n)
		return false;

	sort(v1.begin(), v1.end());
	sort(v2.begin(), v2.end());
	while (i < n&&j < m)
	{
		if (v1[j] < v2[i])
		{
			j++;
		}
		else if (v1[j] == v2[i])
		{
			j++;
			i++;
		}
		else if (v1[j] > v2[i])
		{
			return false;
		}
	}

	if (i < n)
		return false;
	else
		return true;
}

// statistical confidence
std::map<std::string, double> QARM::get_all_conf(QVec qlist, std::vector<ClassicalCondition> clist, int position, double min_conf)
{
	std::map<std::string, double> conf_dict;
	std::vector<std::vector<std::vector<int > > > fn;
	std::map<std::vector<int>, std::pair<std::vector<int>, double> > fn_dict;
	double min_support = 0.4;
	fk_result(qlist, clist, position, min_support, fn, fn_dict);

	if (fn.size() < 2)
		return conf_dict;

	for (int i = 1; i < fn.size(); i++)
	{
		for (int j = 0; j < fn[i].size(); j++)
		{
			auto backward = fn[i][j];

			for (int k = 0; k < fn[i - 1].size(); k++)
			{
				auto forward = fn[i - 1][k];
				if (issubset(backward, forward))
				{
					double supp_xy = fn_dict[backward].second;
					double supp_x = fn_dict[forward].second;
					double conf = conf_x_y(supp_xy, supp_x);
					if (conf >= min_conf)
					{
						auto cause = forward;
						std::vector<int> effect;
						set_difference(backward.begin(), backward.end(), forward.begin(),forward.end(), std::inserter(effect, effect.begin()));
						std::string key = get_conf_key(cause, effect);
						conf_dict[key] = conf;
					}
				}
			}
		}
	}
	return conf_dict;
}


// converts to a string based on a number
std::string QARM::get_conf_key(std::vector<int> cause, std::vector<int> effect)
{
	std::string cause_str, effect_str;
	for (int i = 0; i < cause.size(); i++)
	{
		cause_str += items_dict[cause[i]];
		cause_str += ",";
	}
	cause_str = cause_str.substr(0, cause_str.length() - 1);

	for (int i = 0; i < effect.size(); i++)
	{
		effect_str += items_dict[effect[i]];
		effect_str += ",";
	}
	effect_str = effect_str.substr(0, effect_str.length() - 1);
	std::string  key = cause_str + "->" + effect_str;
	return key;
}

std::map<std::string, double> QARM::run()
{
	QVec qlist = m_qvm->qAllocMany(20);
	std::vector<ClassicalCondition> clist = m_qvm->cAllocMany(20);

	double min_support = 0.4;
	std::vector<std::vector<std::vector<int > > > fn;
	std::map<std::vector<int>, std::pair<std::vector<int>, double> > fn_dict;
	int position = 0;
	fk_result(qlist, clist, position, min_support, fn, fn_dict);
	double min_conf = 0.6;
	std::map<std::string, double> conf_result = get_all_conf(qlist, clist, position, min_conf);
	return conf_result;
}

std::map<std::string, double> QPanda::qarm_algorithm(std::vector<std::vector<std::string>> data)
{
	QARM qram(data);
	return qram.run();
}

