/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

encode 

Author: Zoutianrui
*/

#ifndef  AMPLITUDE_ENCODE_H
#define  AMPLITUDE_ENCODE_H

#include "Core/Core.h"
#include <complex>
QPANDA_BEGIN




class StateNode{
public:
	int index;
	int level;
	double amplitude;
	StateNode* left;
	StateNode* right;
	StateNode(int out_index, int out_level, double out_amplitude, StateNode* out_left, StateNode* out_right);
};
class NodeAngleTree {

public:
	int index;
	int level;
	int qubit_index;
	double angle;
	NodeAngleTree* left;
	NodeAngleTree* right;
	NodeAngleTree(int out_index, int out_level, int out_qubit_index,double out_angle, NodeAngleTree* out_left, NodeAngleTree* out_right);
};

/**
* @brief Encode Class
* @ingroup quantum Algorithm
*/
class Encode {
public:
	Encode();
/**
* @brief  amplitude encode
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::vector<double>& the target datas which will be encoded to the quantum state.
* @note The coding data must meet normalization conditions.
*/
	void amplitude_encode(const QVec &q, const std::vector<double>& data);

	void amplitude_encode(const QVec &q, const std::vector<complex<double>>& data);

	void amplitude_encode_recursive(const QVec &q, const std::vector<double>& data);

	void amplitude_encode_recursive(const QVec &qubits, const QStat& full_cur_vec);

/**
* @brief  angle encode
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::vector<double>& the target datas which will be encoded to the quantum state angle.
* @param[in] GateType gate_type gate types used in circuit.
* @note The coding data must meet [0,PI].
* @note This concrete implementation is from https://arxiv.org/pdf/2003.01695.pdf.
*/
	void angle_encode(const QVec &q, const std::vector<double>& data, const GateType& gate_type=GateType::RY_GATE);

/**
* @brief  dense angle encode
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::vector<double>& the target datas which will be encoded to the quantum state angle and phase.
* @note The coding data must meet [0,PI].
* @note The algorithm is implemented using U3 gates, and each gate is loaded with two data, theta, phi respectively.
* @note This concrete implementation is from https://arxiv.org/pdf/2003.01695.pdf.
*/
	void dense_angle_encode(const QVec &q, const std::vector<double>& data);

/**
* @brief divide conquer amplitude encode
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::vector<double>& the target datas which will be encoded to the quantum state.
* @note The coding data must meet normalization conditions.
* @note The algorithm is implemented using CSWAP gates and data.size-1 qubits, effectively reducing the depth of quantum circuits.
* @note This concrete implementation is from https://arxiv.org/pdf/2008.01511.pdf.
*/
	void dc_amplitude_encode(const QVec &q, const std::vector<double>& data);

/**
* @brief  basic encode
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::string& the target datas which will be encoded to the quantum state.
* @note The coding data must meet binary string.
* @note It converts a binary string x of length n into a quantum state with n qubits.
*/
	void basic_encode(const QVec &q,const std::string& data);

/**
* @brief  bidirectional amplitude encode
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::vector<double>& the target datas which will be encoded to the quantum state.
* @param[in] int spilit which is the level of the angle tree.
* @note The coding data must meet normalization conditions.
* @note The algorithm is implemented using top_down and bottom_up method to generate circuit.
* @note The depth and width of the quantum circuit can be adjusted through the spilit parameter.
* @note This concrete implementation is from https://arxiv.org/pdf/2108.10182.pdf.
*/
	void bid_amplitude_encode(const QVec &q, const std::vector<double>& data, const int split=0);

/**
* @brief  instantaneous quantum polynomial style encoding
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::vector<double>& the target datas which will be encoded to the quantum state.
* @param[in] std::vector<std::pair<int, int>>control_vector which is entanglement of qubits.
* @param[in] bool inverse, whether the added is the inverse circuit of the encoding circuit, the default is ``False``, that is, adding a normal encoding circuit.
* @param[in] int repeats, number of coding layers.
* @param[in] GateType gate_type gate types used in circuit.
* @note Encode the input classical data using IQP encoding.
* @note This concrete implementation is from https://arxiv.org/pdf/1804.11326.pdf.
*/
	void iqp_encode(const QVec &q, const std::vector<double>& data, const std::vector<std::pair<int, int>>& control_vector = {},
		            const bool& inverse=false, const int& repeats = 1);

/**
* @brief  double sparse quantum state preparation
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::map<std::string, double>& or std::map<std::string, std::complex<double>> Sparse array or map with real values.
* @note The data must meet normalization conditions.
* @note The algorithm can prepare the corresponding quantum state according to the input data.
* @note This concrete implementation is from https://arxiv.org/pdf/2108.13527.pdf.
*/

	void ds_quantum_state_preparation(const QVec &q, const std::map<std::string, double>&);

	void ds_quantum_state_preparation(const QVec &q, const std::map<std::string, std::complex<double>>&);

	void ds_quantum_state_preparation(const QVec &q, const std::vector<double>&);

	void ds_quantum_state_preparation(const QVec &q, const std::vector<std::complex<double>>&);

/**
* @brief sparse isometries quantum state preparation
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::map<std::string, double>& or std::map<std::string, std::complex<double>> Sparse array or map with real values.
* @note The data must meet normalization conditions.
* @note The algorithm can prepare the corresponding quantum state according to the input data without auxiliary  quantum bits.
* @note This concrete implementation is from https://arxiv.org/pdf/2006.00016.pdf.
*/
	void sparse_isometry(const QVec &q, const std::map<std::string, double>&);

	void sparse_isometry(const QVec &q, const std::map<std::string, complex<double>>&);

	void sparse_isometry(const QVec &q, const std::vector<double>&);

	void sparse_isometry(const QVec &q, const std::vector<complex<double>>&);

/**
* @brief schmidt decomposition encode
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::vector<double>& or std::vector<complex<double>>&the target datas which will be encoded to the quantum state.
* @note The coding data must meet normalization conditions.
* @note This concrete implementation is from https://arxiv.org/pdf/2107.09155.pdf.
*/
	void schmidt_encode(const QVec &q, const std::vector<double>& data);

/**
* @brief an efficient quantum state preparation of sparse vector
* @ingroup quantum Algorithm.
* @param[in] QVec& Available qubits.
* @param[in] std::map<std::string, double>& or std::map<std::string, std::complex<double>> Sparse array or map with real values.
* @note The coding data must meet normalization conditions.
* @note This concrete implementation is from https://ieeexplore.ieee.org/document/9586240.
*/
	void efficient_sparse(const QVec &q, const std::vector<double>& data);

	void efficient_sparse(const QVec &q, const std::map<string, double>&data);

	void efficient_sparse(const QVec &q, const std::vector<qcomplex_t>& data);

	void efficient_sparse(const QVec &q, const std::map<string, qcomplex_t>&data);

/**
* @brief  get the corresponding quantum circuit
*/
	QCircuit get_circuit();

/**
* @brief  get qubits loaded with classical data
*/

	QVec get_out_qubits();

/**
* @brief  get the corresponding normalization parameters
*/

	double get_normalization_constant();

protected:
	void _generate_circuit(std::vector<std::vector<double>> &betas, const QVec &quantum_input);

	void _recursive_compute_beta(const std::vector<double>&input_vector, std::vector<std::vector<double>>&betas,const int count);

	QCircuit _recursive_compute_beta(const QVec& q, const std::vector<double>&data);

	void _index(const int value, const QVec &control_qubits, const int numberof_controls);

	void _dc_generate_circuit(std::vector<std::vector<double>> &betas, const QVec &quantum_input, const int cnt);

	StateNode* _state_decomposition(int nqubits, std::vector<double>data);

	NodeAngleTree* _create_angles_tree(StateNode* state_tree);

	void _add_register(NodeAngleTree *angle_tree, int start_level);

	void _add_registers(NodeAngleTree* angle_tree, std::queue<int>&q, int start_level);

	void _top_down_tree_walk(NodeAngleTree* angle_tree, const QVec &q, int start_level, std::vector<NodeAngleTree*>control_nodes = {}, 
		                     std::vector<NodeAngleTree*>target_nodes = {});

	void _bottom_up_tree_walk(NodeAngleTree* angle_tree, const QVec &q, int start_level);

	void _apply_cswaps(NodeAngleTree* angle_tree, const QVec &q);

	void _output(NodeAngleTree* angletree, const QVec &q);

	std::vector<NodeAngleTree*> _children(std::vector<NodeAngleTree*> nodes);

	std::vector<int> _select_controls(std::string binary_string);

	void _flip_flop(const QVec& u, const QVec& m, std::vector<int>control, int numqbits);

	template<typename T>
	void _load_superposition(const QVec& u, const QVec& a, const QVec& m,std::vector<int>control, int numqubits, T feature, double& norm);

	void _mcuvchain(const QVec &u, const QVec a, const QVec m, std::vector<int>control, std::vector<double> angle, int numqbits);

	std::vector<double> _compute_matrix_angles(double feature, double norm);

	std::vector<double> _compute_matrix_angles(std::complex<double> feature, double norm);

	template<typename T>
	std::string _get_index_nz(std::map<std::string, T>, int);

	template<typename T>
	std::string _get_index_zero(std::map<std::string, T>, int,int);

	template<typename T>
	std::map<std::string, T> _next_state(char ctrl_state, int index_differ, std::string index_zero, 
		     std::vector<int>remain, std::map<std::string, T> state, std::vector<int>target_cx);

	template<typename T>
	std::map<std::string, T> _pivoting(QCircuit& qcir,const QVec &qubits, std::string index_zero, 
		            std::string index_nonzero, int target_size, std::map<std::string, T> state);

	void _unitary(const QVec &q, EigenMatrixXc gate);


	std::map<std::string, qcomplex_t> _build_state_dict(const std::vector<qcomplex_t> &state);

	std::map<std::string, double> _build_state_dict(const std::vector<double> &state);

	int _maximizing_difference_bit_search(vector<string> &b_strings, std::vector<std::string> &t0, std::vector<std::string> &t1, std::vector<int> &dif_qubits);

	std::vector<string> _build_bit_string_set(const std::vector<string> &b_strings, const std::string bitstr1, std::vector<int> &dif_qubits, std::vector<int> &dif_values);

	std::vector<std::string> _bit_string_search(std::vector<string> b_strings, std::vector<int> &dif_qubits, std::vector<int> &dif_values);

	template<typename T>
	void _search_bit_strings_for_merging(std::string &bitstr1, std::string &bitstr2, int &dif_qubit,std::vector<int> &dif_qubits, const std::map<std::string, T> &state);
	
	std::string _apply_x_operation_to_bit_string(const std::string &b_string, const int &qubit_indexes);
	
	std::string _apply_cx_operation_to_bit_string(const std::string &b_string, const std::vector<int>qubit_indexes);

	template<typename T>
	std::map<std::string, T> _update_state_dict_according_to_operation(std::map<std::string, T>state_dict, const std::string &operation,
		const std::vector<int> &qubit_indexes, const vector<std::string>& merge_strings = {});

	template<typename T>
	std::map<std::string, T> _update_state_dict_according_to_operation(std::map<std::string, T>state_dict, const std::string &operation,
		const int &qubit_index, const vector<std::string>& merge_strings = {});

	template<typename T>
	std::map<std::string, T> _equalize_bit_string_states(std::string &bitstr1, std::string & bitstr2, int &dif, 
		                         std::map<std::string, T> &state_dict, QVec &q);
	template<typename T>
	std::map<std::string, T> _apply_not_gates_to_qubit_index_list(std::string &bitstr1, std::string & bitstr2, const std::vector<int> dif_qubits, std::map<std::string, T> &state_dict, QVec &q);
	
	template<typename T>
	std::map<std::string, T> _preprocess_states_for_merging(std::string &bitstr1, std::string & bitstr2, int &dif, const std::vector<int> dif_qubits, std::map<std::string, T> &state_dict, QVec &q);
	
	std::vector<double>_compute_angles(qcomplex_t amplitude_1, qcomplex_t amplitude_2);
	
	std::vector<double> _compute_angles(double amplitude_1, double amplitude_2);

	template<typename T>
	std::map<std::string, T> _merging_procedure(std::map<std::string, T> &state_dict, QVec &q);

	double compute_norm(const vector<qcomplex_t> &data);
	
	double compute_norm(const vector<double> &data);

	void normalized(std::vector<double>&data);

	void _schmidt(const QVec &q, const std::vector<double>& data);

private:

	QCircuit m_qcircuit;

	QVec m_out_qubits;

	double m_data_std;

};

inline QCircuit amplitude_encode(QVec q, std::vector<double> data, const bool b_need_check_normalization = true)
{
	if (b_need_check_normalization)
	{
		const double max_precision = 1e-13;

		//check parameter b
		double tmp_sum = 0.0;
		//cout << "on amplitude_encode" << endl;
		for (const auto i : data)
		{
			//cout << i << ", ";
			tmp_sum += (i*i);
		}
		//cout << "tmp_sum = " << tmp_sum << endl;
		if (std::abs(1.0 - tmp_sum) > max_precision)
		{
			if (std::abs(tmp_sum) < max_precision)
			{
				QCERR("Error: The input vector b is zero.");
				return QCircuit();
			}

			//cout << "sum = " << std::abs(1.0 - tmp_sum) << endl;
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
		}
	}

	if (data.size() > (1 << q.size()))
	{
		throw run_fail("Amplitude_encode parameter error.");
	}
	while (data.size() < (1 << q.size()))
	{
		data.push_back(0);
	}
	QCircuit qcir;
	double sum_0 = 0;
	double sum_1 = 0;
	size_t high_bit = (size_t)log2(data.size()) - 1;
	if (high_bit == 0)
	{
		if (data[0] > 1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], 2 * acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] > 1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], -2 * acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] < -1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], 2 * acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] < -1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], 2 * (2 * PI - acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1]))));
		}
		else if (std::abs(data[0]) < 1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], PI);
		}
		else if (std::abs(data[0]) < 1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], -PI);
		}
		else if (std::abs(data[1]) < 1e-20 && data[0] < -1e-20)
		{
			qcir << RY(q[0], 2 * PI);
		}
		else if (std::abs(data[0]) < 1e-20 && std::abs(data[1]) < 1e-20)
		{
			throw run_fail("Amplitude_encode error.");
		}
	}
	else
	{
		for (auto i = 0; i < (data.size() >> 1); i++)
		{
			sum_0 += data[i] * data[i];
			sum_1 += data[i + (data.size() >> 1)] * data[i + (data.size() >> 1)];
		}
		if (sum_0 + sum_1 > 1e-20)
			qcir << RY(q[high_bit], 2 * acos(sqrt(sum_0 / (sum_0 + sum_1))));
		else
		{
			throw run_fail("Amplitude_encode error.");
		}

		if (sum_0 > 1e-20)
		{
			QVec temp({ q[high_bit] });
			std::vector<double> vtemp(data.begin(), data.begin() + (data.size() >> 1));
			qcir << X(q[high_bit]) << amplitude_encode(q - temp, vtemp, false).control({ q[high_bit] }) << X(q[high_bit]);
		}
		if (sum_1 > 1e-20)
		{
			QVec temp({ q[high_bit] });
			std::vector<double> vtemp(data.begin() + (data.size() >> 1), data.end());
			qcir << amplitude_encode(q - temp, vtemp, false).control({ q[high_bit] });
		}
	}

	return qcir;
}


inline QCircuit amplitude_encode(QVec qubits, const QStat& full_cur_vec)
{
	const auto _dimension = full_cur_vec.size();
	if (full_cur_vec.size() != _dimension)
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: current vector size error on HQRDecompose.");
	}

	QCircuit cir_swap_qubits;
	for (size_t i = 0; (i * 2) < (qubits.size() - 1); ++i)
	{
		cir_swap_qubits << SWAP(qubits[i], qubits[qubits.size() - 1 - i]);
	}

	std::vector<double> ui_mod(_dimension);
	std::vector<double> ui_angle(_dimension);
	double tatal = 0.0;
	for (size_t i = 0; i < _dimension; ++i)
	{
		auto tmp_m = full_cur_vec[i].real() * full_cur_vec[i].real() + full_cur_vec[i].imag() * full_cur_vec[i].imag();
		tatal += tmp_m;
		ui_mod[i] = sqrt(tmp_m);
		ui_angle[i] = arg(full_cur_vec[i]);
	}

	QStat mat_d(_dimension * _dimension, qcomplex_t(0, 0));
	for (size_t i = 0; i < _dimension; ++i)
	{
		mat_d[i + i * _dimension] = exp(qcomplex_t(0, ui_angle[i]));
	}

	QCircuit cir_d = diagonal_matrix_decompose(qubits, mat_d);

	QCircuit cir_y = amplitude_encode(qubits, ui_mod);
	QCircuit cir_P;
	cir_P << cir_y << cir_swap_qubits << cir_d << cir_swap_qubits;

	return cir_P;
}

/**
* @brief  init superposition state
* @ingroup quantum Algorithm
* @param[in] QVec Available qubits
* @param[in] size_t the target data
* @return the circuit
* @note
*/


#if 0
inline QCircuit init_superposition_state(QVec q, size_t d){
	QCircuit qcir;

	size_t highest_bit = (size_t)log2(d - 1);

	if (d == (1 << (int)log2(d)))
	{
		for (auto i = 0; i < (int)log2(d); i++)
		{
			qcir << H(q[i]);
		}
	}
	else if (d == 3)
	{
		qcir << RY(q[1], 2 * acos(sqrt(2.0 / 3))) << X(q[1]);
		qcir << H(q[0]).control({ q[1] }) << X(q[1]);
	}
	else
	{
		qcir << RY(q[highest_bit], 2 * acos(sqrt((1 << highest_bit)*1.0 / d)));
		QCircuit qcir1;
		for (auto i = 0; i < highest_bit; i++)
		{
			qcir1 << H(q[i]);
		}
		qcir << X(q[highest_bit]) << qcir1.control({ q[highest_bit] }) << X(q[highest_bit]);

		size_t d1 = d - (1 << highest_bit);
		if (d > 1)
		{
			QCircuit qcir2 = init_superposition_state(q, d1);
			qcir2.setControl({ q[highest_bit] });
			qcir << qcir2;
		}
	}
	return qcir;
}
#endif
inline QCircuit init_superposition_state(QVec q, size_t d)
{
	QCircuit qcir;

	size_t highest_bit = (size_t)log2(d - 1);

	if (d == (1 << (int)log2(d)))
	{
		for (auto i = 0; i < (int)log2(d); i++)
		{
			qcir << H(q[i]);
		}
	}
	else if (d == 3)
	{
		qcir << RY(q[1], 2 * acos(sqrt(2.0 / 3))) << X(q[1]);
		qcir << H(q[0]).control({ q[1] }) << X(q[1]);
	}
	else
	{
		qcir << RY(q[highest_bit], 2 * acos(sqrt((1 << highest_bit)*1.0 / d)));
		QCircuit qcir1;
		for (auto i = 0; i < highest_bit; i++)
		{
			qcir1 << H(q[i]);
		}
		qcir << X(q[highest_bit]) << qcir1.control({ q[highest_bit] }) << X(q[highest_bit]);

		size_t d1 = d - (1 << highest_bit);
		if (d1 > 1)
		{
			QCircuit qcir2 = init_superposition_state(q, d1);
			qcir2.setControl({ q[highest_bit] });
			qcir << qcir2;
		}
	}
	return qcir;
}

QPANDA_END

#endif