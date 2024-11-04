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

#ifndef  ENCODE_H
#define  ENCODE_H



#include <numeric>
#include <queue>
#include <algorithm>
#include <EigenUnsupported/Eigen/CXX11/Tensor>
#include "Core/Utilities/QProgInfo/KAK.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
using std::vector;

QPANDA_BEGIN

class StateNode {
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
	NodeAngleTree(int out_index, int out_level, int out_qubit_index, double out_angle, NodeAngleTree* out_left, NodeAngleTree* out_right);
};
inline std::vector<int> complement(std::vector<int> subsys, int n) {

	std::vector<int> all(n);
	std::vector<int> subsys_bar(n - subsys.size());

	std::iota(std::begin(all), std::end(all), 0);
	std::sort(std::begin(subsys), std::end(subsys));
	std::set_difference(std::begin(all), std::end(all), std::begin(subsys),
		std::end(subsys), std::begin(subsys_bar));

	return subsys_bar;
}
inline void n2multiidx(int n, int numdims, const int* const dims, int* result) noexcept {
	for (int i = 0; i < numdims; ++i) {
		result[numdims - i - 1] = n % (dims[numdims - i - 1]);
		n /= (dims[numdims - i - 1]);
	}
}
inline int multiidx2n(const int* const midx, int numdims,
	const int* const dims) noexcept {

	int part_prod[2 * 64];

	int result = 0;
	part_prod[numdims - 1] = 1;
	for (int i = 1; i < numdims; ++i) {
		part_prod[numdims - i - 1] = part_prod[numdims - i] * dims[numdims - i];
		result += midx[numdims - i - 1] * part_prod[numdims - i - 1];
	}

	return result + midx[numdims - 1];
}
/**
* @brief Encode Class
* @ingroup quantum Algorithm
*/
class Encode {
	template <typename Scalar>
	using dyn_col_vect = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

	template <typename Scalar>
	using dyn_mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

	//template <typename Scalar>
	//using MatrixX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

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

	void amplitude_encode(const QVec &q, const std::vector<qcomplex_t>& data);

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
	void angle_encode(const QVec &q, const std::vector<double>& data, const GateType& gate_type = GateType::RY_GATE);

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
	void basic_encode(const QVec &q, const std::string& data);

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
	void bid_amplitude_encode(const QVec &q, const std::vector<double>& data, const int split = 0);

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
		const bool& inverse = false, const int& repeats = 1);

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

	void sparse_isometry(const QVec &q, const std::map<std::string, qcomplex_t>&);

	void sparse_isometry(const QVec &q, const std::vector<double>&);

	void sparse_isometry(const QVec &q, const std::vector<qcomplex_t>&);

	/**
	* @brief schmidt decomposition encode
	* @ingroup quantum Algorithm.
	* @param[in] QVec& Available qubits.
	* @param[in] std::vector<double>& or std::vector<complex<double>>&the target datas which will be encoded to the quantum state.
	* @note The coding data must meet normalization conditions.
	* @note This concrete implementation is from https://arxiv.org/pdf/2107.09155.pdf.
	*/
	void schmidt_encode(const QVec &q, const std::vector<double>& data, const double cutoff = 0.0);

	/**
	* @brief an efficient quantum state preparation of sparse vector
	* @ingroup quantum Algorithm.
	* @param[in] QVec& Available qubits.
	* @param[in] std::map<std::string, double>& or std::map<std::string, std::complex<double>> Sparse array or map with real values.
	* @note The coding data must meet normalization conditions.
	* @note This concrete implementation is from https://ieeexplore.ieee.org/document/9586240.
	*/
	void efficient_sparse(const QVec &q, const std::vector<double>& data);

	void efficient_sparse(const QVec &q, const std::map<std::string, double>&data);

	void efficient_sparse(const QVec &q, const std::vector<qcomplex_t>& data);

	void efficient_sparse(const QVec &q, const std::map<std::string, qcomplex_t>&data);
	template<typename Derived>
	void approx_mps_encode(const QVec &q, const std::vector<Derived>& data, const int &layers = 3, const int &step = 100)
	{
		vector<Derived>data_temp(data);
		//_normalized(data_temp);
		if (!_check_normalized(data_temp)) {
			throw run_fail("Data is not normalized");
		}
		int size = data_temp.size();
		int log_ceil_size = std::ceil(std::log2(size));
		if (data_temp.size() > (1 << log_ceil_size))
		{
			throw run_fail("Qubits size error.");
		}
		while (data_temp.size() < (1 << log_ceil_size))
		{
			data_temp.push_back(0);
		}
		if (data_temp.size() <= 2)
		{
			throw run_fail("Two elements data is not suitable for MPS");
		}
		size_t N = data_temp.size();

		N = ceil(log2(N));
		int count = 0;
		vector<Derived> data_(data_temp);
		vector<QCircuit> circuit_block;


		vector<vector<dyn_mat<Derived>>>unitary_block_v_d;
		while (count < layers) {

			auto MPS = _to_approx_MPS(data_, N, 2);


			_orthogonalize_mps(MPS, 0, N - 1, 2);

			std::pair<QCircuit, std::vector<dyn_mat<Derived>>>circuit = _embedded_circuit(MPS, q, 2);

			unitary_block_v_d.push_back(circuit.second);

			circuit_block.push_back(circuit.first);
			QProg prog1 = QProg();
			prog1 << circuit.first;

			auto stat = get_partial_unitary(prog1);


			dyn_mat<Derived> _U(1 << N, 1 << N);
			_qstat2eigen(stat, N, _U);

			dyn_col_vect<Derived> v2(data.size());
			int cnt = 0;
			for (Derived i : data_) {
				v2(cnt++) = i;
			}

			auto _data = _U.transpose().conjugate()*(v2);

			for (int i = 0; i < _data.rows(); ++i) {
				data_[i] = _data(i, 0);
			}
			std::reverse(circuit_block.begin(), circuit_block.end());
			QProg prog = QProg();
			for (auto i : circuit_block) {
				prog << i;
			}

			std::reverse(circuit_block.begin(), circuit_block.end());
			count++;
		}
		std::reverse(unitary_block_v_d.begin(), unitary_block_v_d.end());
		vector<dyn_mat<Derived>>unitary_block;
		for (auto i : unitary_block_v_d) {
			for (auto j : i) {
				unitary_block.push_back(j);
			}
		}

		QProg prog = QProg();

		m_qcircuit = _iter_optimize(q, data_temp, unitary_block, step);

		m_out_qubits = q;
		return;
	};

	/**
	* @brief  get the corresponding quantum circuit
	*/
	QCircuit get_circuit();

	/**
	* @brief  get qubits loaded with classical data
	*/

	QVec get_out_qubits();



	double get_fidelity(const std::vector<qcomplex_t> &data);

	double get_fidelity(const std::vector<double> &data);

	double get_fidelity(const std::vector<float> &data);
protected:
	void _generate_circuit(std::vector<std::vector<double>> &betas, const QVec &quantum_input);

	void _recursive_compute_beta(const std::vector<double>&input_vector, std::vector<std::vector<double>>&betas, const int count);

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
	void _load_superposition(const QVec& u, const QVec& a, const QVec& m, std::vector<int>control, int numqubits, T feature, double& norm);

	void _mcuvchain(const QVec &u, const QVec a, const QVec m, std::vector<int>control, std::vector<double> angle, int numqbits);

	std::vector<double> _compute_matrix_angles(double feature, double norm);

	std::vector<double> _compute_matrix_angles(std::complex<double> feature, double norm);

	template<typename T>
	std::string _get_index_nz(std::map<std::string, T>, int);

	template<typename T>
	std::string _get_index_zero(std::map<std::string, T>, int, int);

	template<typename T>
	std::map<std::string, T> _next_state(char ctrl_state, int index_differ, std::string index_zero,
		std::vector<int>remain, std::map<std::string, T> state, std::vector<int>target_cx);

	template<typename T>
	std::map<std::string, T> _pivoting(QCircuit& qcir, const QVec &qubits, std::string &index_zero,
		std::string &index_nonzero, int target_size, std::map<std::string, T> state);

	void _unitary(const QVec &q, QMatrixXd gate, const double cutoff);


	std::map<std::string, qcomplex_t> _build_state_dict(const std::vector<qcomplex_t> &state);

	std::map<std::string, double> _build_state_dict(const std::vector<double> &state);

	int _maximizing_difference_bit_search(std::vector<std::string> &b_strings, std::vector<std::string> &t0, std::vector<std::string> &t1, std::vector<int> &dif_qubits);

	std::vector<std::string> _build_bit_string_set(const std::vector<std::string> &b_strings, const std::string bitstr1, std::vector<int> &dif_qubits, std::vector<int> &dif_values);

	std::vector<std::string> _bit_string_search(std::vector<std::string> b_strings, std::vector<int> &dif_qubits, std::vector<int> &dif_values);

	template<typename T>
	void _search_bit_strings_for_merging(std::string &bitstr1, std::string &bitstr2, int &dif_qubit, std::vector<int> &dif_qubits, const std::map<std::string, T> &state);

	std::string _apply_x_operation_to_bit_string(const std::string &b_string, const int &qubit_indexes);

	std::string _apply_cx_operation_to_bit_string(const std::string &b_string, const std::vector<int>qubit_indexes);

	template<typename T>
	std::map<std::string, T> _update_state_dict_according_to_operation(std::map<std::string, T>state_dict, const std::string &operation,
		const std::vector<int> &qubit_indexes, const std::vector<std::string>& merge_strings = {});

	template<typename T>
	std::map<std::string, T> _update_state_dict_according_to_operation(std::map<std::string, T>state_dict, const std::string &operation,
		const int &qubit_index, const std::vector<std::string>& merge_strings = {});

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

	double compute_norm(const std::vector<qcomplex_t> &data);

	double compute_norm(const std::vector<double> &data);
	template <typename T>
	bool _check_normalized(std::vector<T>&data) {
		double tmp = 0;
		for (int i = 0; i < data.size(); ++i) {
			if (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value)
			{
				tmp += std::abs(data[i]);
			}else
			{
				tmp += std::abs(data[i])* std::abs(data[i]);
			}
			
		}

		if (std::abs(tmp - 1) > 1e-8) {
			return false;
		}
		return true;
	}



	void _schmidt(const QVec &q, const std::vector<double>& data, const double cutoff);

	template<typename Derived>
	std::vector<Eigen::Tensor<Derived, 3, Eigen::RowMajor>> _to_approx_MPS(const std::vector<Derived> state, int N, const int chi = 2)
	{
		int d = 2;
		int last_svd_dim = 1;
		std::vector<Eigen::Tensor<Derived, 3, Eigen::RowMajor>>MPS;
		Eigen::Tensor<Derived, 1, Eigen::RowMajor> state_tensor(state.size());
		for (int i = 0; i < (1 << N); ++i) {
			state_tensor(i) = state[i];
		}
		std::vector< Eigen::Tensor<Derived, 2, Eigen::RowMajor>>state_tensor_v(N);
		Eigen::array<Eigen::DenseIndex, 2> dim = { last_svd_dim * d, 1 << (N - 1) };
		Eigen::Tensor<Derived, 2, Eigen::RowMajor> m = state_tensor.reshape(dim);
		state_tensor_v[0] = m;
		for (int i = 0; i < N - 1; ++i) {
			Eigen::Tensor<Derived, 2, Eigen::RowMajor> mat;
			int size_r = 1 << (N - i - 1);
			if (i > 0) {
				Eigen::array<Eigen::DenseIndex, 2> dim = { last_svd_dim * d, size_r };
				mat = state_tensor_v[i].reshape(dim);
			}
			else {
				mat = state_tensor_v[0];
			}

			dyn_mat<Derived> eigen_matrix(last_svd_dim * d, size_r);
			for (int i = 0; i < last_svd_dim * d; ++i) {
				for (int j = 0; j < size_r; ++j) {
					eigen_matrix(i, j) = mat(i, j);
				}
			}

			Eigen::JacobiSVD<dyn_mat<Derived>> svd(eigen_matrix, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);
			//std::cout << eigen_matrix << std::endl;
			dyn_mat<Derived> V = svd.matrixV().transpose().conjugate(), U = svd.matrixU();
			//std::cout << U<<std::endl;

			auto A = svd.singularValues();
			//std::cout << A << std::endl;
			dyn_mat<Derived> PartU;
			dyn_mat<Derived> PartV;
			if (U.cols() < chi) {
				PartU = U;
				PartV = V;
			}
			else {
				PartU = U.leftCols(chi);
				PartV = V.topRows(chi);
			}

			dyn_mat<Derived> S = A.array().matrix().asDiagonal();
			dyn_mat<Derived> part_s;
			if (S.rows() < chi) {
				part_s = S;
			}
			else {
				part_s = S.topRows(chi);
			}
			dyn_mat<Derived> state_m = part_s * V;

			size_t rows = state_m.rows();
			size_t cols = state_m.cols();

			Eigen::Tensor<Derived, 2, Eigen::RowMajor> mat_(rows, cols);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					mat_(i, j) = state_m(i, j);
				}
			}
			state_tensor_v[i + 1] = mat_;

			rows = PartU.rows();
			cols = PartU.cols();
			Eigen::Tensor<Derived, 3, Eigen::RowMajor> U_;

			Eigen::Tensor<Derived, 2, Eigen::RowMajor> partu(rows, cols);
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					partu(i, j) = PartU(i, j);
				}
			}
            if (i > 0) {
                int l = (std::min)(last_svd_dim, chi);
				int r = (int)rows * cols / (l *d);
				Eigen::array<Eigen::DenseIndex, 3> dimension = { l, d, r };
				U_ = partu.reshape(dimension);
				//std::cout << U_ << std::endl;
				MPS.push_back(U_);
			}
			else {
				Eigen::array<Eigen::DenseIndex, 3> dimension = { 1, d, d };
				U_ = partu.reshape(dimension);
				MPS.push_back(U_);
				//std::cout << U_ << std::endl;
			}

			if (A.size() < chi) {
				last_svd_dim = A.size();
			}
			else {
				last_svd_dim = chi;
			}
		}
		Eigen::array<Eigen::DenseIndex, 3> dim_ = { d,d,1 };
		Eigen::Tensor<Derived, 3, Eigen::RowMajor> _mat = state_tensor_v[N - 1].reshape(dim_);
		MPS.push_back(_mat);

		return MPS;
	};

	template<typename Derived>
	void _orthogonalize_mps(std::vector<Eigen::Tensor<Derived, 3, Eigen::RowMajor>> &mps, int l0, int l1, const int &chi)
	{
		int N = l1 + 1;
		vector<int>virtual_dim;
		for (int i = 0; i < N; ++i) {
			virtual_dim.push_back(2);
		}
		virtual_dim[0] = 1;
		for (int n = l0; n < l1; ++n) {
			dyn_mat<Derived> mat = _left2right_decompose_tensor(mps[n], virtual_dim, n, chi);

			_matrix2tensor(mps[n + 1], mat);
		}
		return;
	};
	template<typename Derived>
	Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> _left2right_decompose_tensor(Eigen::Tensor<Derived, 3, Eigen::RowMajor>& tensor, std::vector<int> &virtual_dim, int cnt, const int &chi)
	{
        const Eigen::Tensor<int, 3>::Dimensions & s = tensor.dimensions();
        int dim = (std::min)(s[0] * s[1], s[2]);
		Eigen::Tensor<Derived, 2, Eigen::RowMajor>tensor_tmp{ s[0] * s[1], s[2] };

		Eigen::array<Eigen::DenseIndex, 2> dimension = { s[0] * s[1], s[2] };
		tensor_tmp = tensor.reshape(dimension);
		dyn_mat<Derived> matrix(s[0] * s[1], s[2]);
		for (int i = 0; i < s[0] * s[1]; ++i) {
			for (int j = 0; j < s[2]; ++j) {
				matrix(i, j) = tensor_tmp(i, j);
			}
		}
		dyn_mat<Derived> v;
		if (chi > 1) {
			Eigen::JacobiSVD<dyn_mat<Derived>> svd(matrix, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);
			dyn_mat<Derived> V = svd.matrixV().transpose().conjugate(), U = svd.matrixU();
			auto A = svd.singularValues();
			auto PartV = V.topRows(dim);
			dyn_mat<Derived> S = A.array().matrix().asDiagonal();
			dyn_mat<Derived> part_s = S.topRows(dim);
			dyn_mat<Derived> state_m = part_s * V;
			dyn_mat<Derived> PartQ = U.leftCols(dim);
			int row = PartQ.rows();
			int col = PartQ.cols();
			Eigen::Tensor<Derived, 2, Eigen::RowMajor>tensor_tmp_(row, col);
			for (int i = 0; i < row; ++i) {
				for (int j = 0; j < col; ++j) {
					tensor_tmp_(i, j) = PartQ(i, j);
				}
            }
            Eigen::array<Eigen::DenseIndex, 3> dim_ = { s[0],s[1],(std::min)(s[0] * s[1], s[2]) };
			Eigen::Tensor<Derived, 3, Eigen::RowMajor> tensor_ = tensor_tmp_.reshape(dim_);

			for (int i = 0; i < s[0]; ++i) {
				for (int j = 0; j < s[1]; ++j) {
					for (int k = 0; k < dim; ++k) {
						tensor(i, j, k) = tensor_(i, j, k);
					}
				}
			}
			v = state_m.transpose();
		}
		else {
			Eigen::HouseholderQR<dyn_mat<Derived>> qr;
			qr.compute(matrix);
			dyn_mat<Derived> V = qr.matrixQR().template triangularView<Eigen::Upper>();
			dyn_mat<Derived> PartV = V.topRows(dim);
			dyn_mat<Derived> Q = qr.householderQ();
			dyn_mat<Derived> PartQ = Q.leftCols(dim);

			int row = PartQ.rows();
			int col = PartQ.cols();
			Eigen::Tensor<Derived, 2, Eigen::RowMajor>tensor_tmp_(row, col);
			for (int i = 0; i < row; ++i) {
				for (int j = 0; j < col; ++j) {
					tensor_tmp_(i, j) = PartQ(i, j);
				}
            }
            Eigen::array<Eigen::DenseIndex, 3> dim_ = { s[0],s[1],(std::min)(s[0] * s[1], s[2]) };
			Eigen::Tensor<Derived, 3, Eigen::RowMajor> tensor_ = tensor_tmp_.reshape(dim_);

			for (int i = 0; i < s[0]; ++i) {
				for (int j = 0; j < s[1]; ++j) {
					for (int k = 0; k < dim; ++k) {
						tensor(i, j, k) = tensor_(i, j, k);
					}
				}
			}
			v = PartV.transpose();
		}
		virtual_dim[cnt] = dim;
		return v;
	};
	template<typename Derived>
	void _matrix2tensor(Eigen::Tensor<Derived, 3, Eigen::RowMajor>& tensor, Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> mat)
	{
		const Eigen::Tensor<int, 3>::Dimensions & s = tensor.dimensions();
		int size = s.size();
		int r = 1;
		for (int i = 1; i < size; ++i) {
			r *= s[i];
		}
		dyn_mat<Derived> matrix = mat.transpose();
		Eigen::array<Eigen::DenseIndex, 2> dim = { s[0],r };
		Eigen::Tensor<Derived, 2, Eigen::RowMajor> m = tensor.reshape(dim);

		dyn_mat<Derived> mat1(s[0], r);
		for (int i = 0; i < s[0]; ++i) {
			for (int j = 0; j < r; ++j) {
				mat1(i, j) = m(i, j);
			}
		}

		dyn_mat<Derived> matrix1 = matrix * mat1;

		int row = matrix1.rows();
		int col = matrix1.cols();
		Eigen::Tensor<Derived, 2, Eigen::RowMajor> m_tensor(row, col);

		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j) {
				m_tensor(i, j) = matrix1(i, j);
			}
		}

		int l = mat.cols();
		Eigen::array<Eigen::DenseIndex, 3> dim1 = { l,s[1],s[2] };
		Eigen::Tensor<Derived, 3, Eigen::RowMajor> tensor1 = m_tensor.reshape(dim1);

		for (int i = 0; i < l; ++i) {
			for (int j = 0; j < s[1]; ++j) {
				for (int k = 0; k < s[2]; ++k) {
					tensor(i, j, k) = tensor1(i, j, k);
				}
			}
		}
		return;
	};
	template<typename Derived>
	Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic> _extend_to_unitary(Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>& iso, int lines, int cols)
	{
		{
			auto _kernel_cod = [](dyn_mat<Derived> &mat) {
				Eigen::CompleteOrthogonalDecomposition<dyn_mat<Derived>>cod;
				cod.compute(mat);
				unsigned rk = cod.rank();
				dyn_mat<Derived> P = cod.colsPermutation();
				dyn_mat<Derived> V = cod.matrixZ().transpose();
				dyn_mat<Derived> kernel = P * V.block(0, rk, V.rows(), V.cols() - rk);
				return kernel;
			};

			dyn_mat<Derived> unitary;
			if (lines == cols)
			{
				unitary = iso;
			}
			else
			{
				dyn_mat<Derived> iso_t = iso.transpose();
				dyn_mat<Derived> null_space = _kernel_cod(iso_t);

				unitary = dyn_mat<Derived>::Zero(lines, lines);
				unitary.block(0, 0, lines, cols) = iso;
				unitary.block(0, cols, lines, lines - cols) = null_space;
			}

			return unitary;
		}
	};
	double _kl_divergence(const std::vector<double> &input_data, const std::vector<double> &output_data);
	template<typename Derived>
	std::pair<QCircuit, std::vector<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>>> _embedded_circuit(std::vector<Eigen::Tensor<Derived, 3, Eigen::RowMajor>> &mps, const QVec &q, int chi)
	{
		QCircuit circuit = QCircuit();
		QCircuit cir_qsd = QCircuit();
		int N = mps.size();
		int	d = 2;
		int count = 0;
		Eigen::Tensor<Derived, 3, Eigen::RowMajor> A_tmp = mps[N - 1];
		Eigen::array<Eigen::DenseIndex, 2> dimension = { d, d };
		dyn_col_vect<Derived> A_v(d*d);

		Eigen::Tensor<Derived, 2, Eigen::RowMajor> A = A_tmp.reshape(dimension);
		std::vector<dyn_mat<Derived>> unitary_block(N - 1);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				A_v(count) = A(i, j);
				count++;
			}
		}
		A_v.normalize();
		double tmp = 0;

		count = 0;
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				A(i, j) = A_v(count);
				count++;
			}
		}
		dyn_mat<Derived> matrix(1, d*d);
		count = 0;
		for (int i = 0; i < 1; ++i) {
			for (int j = 0; j < d*d; ++j) {
				matrix(i, j) = A_v(count);
				count++;
			}
		}

		Eigen::JacobiSVD<dyn_mat<Derived>> svd(matrix, Eigen::DecompositionOptions::ComputeFullU | Eigen::DecompositionOptions::ComputeFullV);
		dyn_mat<Derived> V = (svd.matrixU()(0, 0)*svd.singularValues()(0))*svd.matrixV().conjugate();


		dyn_mat<Derived>unitary(d*d, d*d);
		unitary.col(0) = A_v;
		for (int i = 1; i < d*d; ++i) {

			unitary.col(i) = svd.matrixU()(0, 0)*svd.singularValues()(0)*V.col(i);
		}

		unitary_block[0] = unitary;
		std::vector<Eigen::Tensor<Derived, 3, Eigen::RowMajor>> mps_v(N);
		for (int i = 0; i < N; ++i) {
			mps_v[i] = mps[N - i - 1];
		}


		count = 1;
		for (int i = 1; i < N - 1; ++i) {

			const Eigen::Tensor<int, 3>::Dimensions & s = mps_v[i].dimensions();
			int chi_l = s[0];
			int chi_r = s[2];
			int D = log2(chi_l*d);
			QVec qubits;
			for (int j = 0; j < D; ++j) {
				qubits.push_back(q[count + j]);
			}
			dyn_mat<Derived> unitary_tmp(chi_l*d, chi_r);
			Eigen::array<Eigen::DenseIndex, 2> dimension = { chi_l*d, chi_r };
			Eigen::Tensor<Derived, 2, Eigen::RowMajor>tensor_tmp = mps_v[i].reshape(dimension);
			for (int i = 0; i < chi_l*d; ++i) {
				for (int j = 0; j < chi_r; ++j) {
					unitary_tmp(i, j) = tensor_tmp(i, j);
				}
			}

			dyn_mat<Derived> unitary1 = _extend_to_unitary(unitary_tmp, chi_l*d, chi_r);

			unitary_block[i] = unitary1;

			count++;
		}
		Eigen::Tensor<Derived, 3, Eigen::RowMajor> tmp_ = mps_v[N - 1];
		Eigen::array<Eigen::DenseIndex, 2> dimension1 = { d, d };
		dyn_mat<Derived> _t(d, d);
		Eigen::Tensor<Derived, 2, Eigen::RowMajor> _tmp = mps_v[N - 1].reshape(dimension1);
		for (int i = 0; i < d; ++i) {
			for (int j = 0; j < d; ++j) {
				_t(i, j) = _tmp(i, j);
			}
		}


		dyn_mat<Derived> unitary_tmp = kroneckerProduct(_t, dyn_mat<Derived>::Identity(2, 2));

		unitary_tmp = unitary_tmp * unitary_block[N - 2];


		unitary_block[N - 2] = unitary_tmp;

		_gen_circuit(circuit, q, N, unitary_block);
		return std::make_pair(circuit, unitary_block);
	};
	Eigen::MatrixXcd _partial_trace(const int axis1, const int axis2, const int size, const Eigen::MatrixXcd & M);
	void _qstat2eigen(const QStat &stat, int N, Eigen::MatrixXcd &_U);
	void _qstat2eigen(const QStat &stat, int N, Eigen::MatrixXd &_U);
	void _qstat2eigen(const QStat &stat, int N, Eigen::MatrixXf &_U);
	void _gen_circuit(QCircuit &circuit, const QVec& q, const int N, const std::vector<Eigen::MatrixXf>& unitary_block);
	void _gen_circuit(QCircuit &circuit, const QVec& q, const int N, const std::vector<Eigen::MatrixXd>& unitary_block);
	void _gen_circuit(QCircuit &circuit, const QVec& q, const int N, const std::vector<Eigen::MatrixXcd>& unitary_block);
	template<typename Derived>
	void _orthonormalize(dyn_mat<Derived>& ColVecs)
	{
		//const typename dyn_mat<Derived>::EvalReturnType& rA = ColVecs.derived();
		ColVecs.col(0).normalize();
		Derived temp;
		for (std::size_t k = 0; k != ColVecs.cols() - 1; ++k)
		{
			for (std::size_t j = 0; j != k + 1; ++j)
			{
                ColVecs.col(j).adjointInPlace();
                //ColVecs.col(j) * ColVecs.col(k + 1);
                //temp = ColVecs.col(j).adjointInPlace() * ColVecs.col(k + 1);
                ColVecs.col(k + 1) -= ColVecs.col(j) * ColVecs.col(j) * ColVecs.col(k + 1);

			}
			ColVecs.col(k + 1).normalize();
		}
	};
	template<typename Derived>
	dyn_mat<Derived> _ptrace(const dyn_mat<Derived>& rA, const std::vector<int>& target,
		const std::vector<int>& dims, const bool &complex_flag) {
		int D = static_cast<int>(rA.rows());
		int n = dims.size();
		int n_subsys = target.size();
		int n_subsys_bar = n - n_subsys;
		int Dsubsys = 1;
		for (int i = 0; i < n_subsys; ++i)
			Dsubsys *= dims[target[i]];
		int Dsubsys_bar = D / Dsubsys;

		int Cdims[64];
		int Csubsys[64];
		int Cdimssubsys[64];
		int Csubsys_bar[64];
		int Cdimssubsys_bar[64];

		int Cmidxcolsubsys_bar[64];

		std::vector<int> subsys_bar = complement(target, n);
		std::copy(std::begin(subsys_bar), std::end(subsys_bar),
			std::begin(Csubsys_bar));

		for (int i = 0; i < n; ++i) {
			Cdims[i] = dims[i];
		}
		for (int i = 0; i < n_subsys; ++i) {
			Csubsys[i] = target[i];
			Cdimssubsys[i] = dims[target[i]];
		}
		for (int i = 0; i < n_subsys_bar; ++i) {
			Cdimssubsys_bar[i] = dims[subsys_bar[i]];
		}

		dyn_mat<Derived> result = dyn_mat<Derived>(Dsubsys_bar, Dsubsys_bar);
		if (target.size() == dims.size()) {
			result(0, 0) = rA.trace();
			return result;
		}

		if (target.empty())
			return rA;

		auto worker = [&](int i) noexcept {
			// use static allocation for speed!
			int Cmidxrow[64];
			int Cmidxcol[64];
			int Cmidxrowsubsys_bar[64];
			int Cmidxsubsys[64];

			/* get the row/col multi-indexes of the complement */
			n2multiidx(i, n_subsys_bar, Cdimssubsys_bar,
				Cmidxrowsubsys_bar);
			/* write them in the global row/col multi-indexes */
			for (int k = 0; k < n_subsys_bar; ++k) {
				Cmidxrow[Csubsys_bar[k]] = Cmidxrowsubsys_bar[k];
				Cmidxcol[Csubsys_bar[k]] = Cmidxcolsubsys_bar[k];
			}
			Derived sm = 0;


			for (int a = 0; a < Dsubsys; ++a) {
				// get the multi-index over which we do the summation
				n2multiidx(a, n_subsys, Cdimssubsys, Cmidxsubsys);
				// write it into the global row/col multi-indexes
#pragma omp parallel for num_threads(omp_get_max_threads())
				for (int k = 0; k < n_subsys; ++k)
					Cmidxrow[Csubsys[k]] = Cmidxcol[Csubsys[k]] =
					Cmidxsubsys[k];

				// now do the sum
				sm += rA(multiidx2n(Cmidxrow, n, Cdims),
					multiidx2n(Cmidxcol, n, Cdims));
			}

			return sm;
		}; /* end worker */

		for (int j = 0; j < Dsubsys_bar; ++j) // column major order for speed
		{
			// compute the column multi-indexes of the complement
			n2multiidx(j, n_subsys_bar, Cdimssubsys_bar,
				Cmidxcolsubsys_bar);
#pragma omp parallel for num_threads(omp_get_max_threads())
			for (int i = 0; i < Dsubsys_bar; ++i) {
				result(i, j) = worker(i);
			}
		}
		return result;
	};
	template<typename Derived>
	QCircuit _iter_optimize(const QVec &q, const std::vector<Derived> &data,
		std::vector<Eigen::Matrix<Derived, Eigen::Dynamic, Eigen::Dynamic>> &unitary_block,
		const int T)
	{
		int N = (int)(log2(data.size()));
		std::vector<Derived> _data((1 << N));
		_data[0] = 1;
		int size = unitary_block.size();
		QCircuit cir_qsd = QCircuit();
		QCircuit circuit = QCircuit();

		dyn_mat<Derived> data_mat = dyn_mat<Derived>::Zero((1 << N), 1);
		for (int i = 0; i < (1 << N); ++i) {
			data_mat(i, 0) = data[i];
		}
		double fid = 0;
		double fid_last = -1;
		int iter = 0;
		bool complex_flag = true;
		std::vector<int>dims(N, 2);
		QProg prog2 = QProg();
		QCircuit _circuit = QCircuit();
		_gen_circuit(_circuit, q, N, unitary_block);

		prog2 << _circuit.dagger();
		auto state1 = get_partial_unitary(prog2);
		dyn_mat<Derived> left_mat(1 << N, 1 << N);
		if (std::abs(state1[0].imag() - 0) < 1e-8) {
			complex_flag = false;
			for (int i = 0; i < state1.size(); ++i) {
				left_mat(i / (1 << N), i % (1 << N)) = state1[i].real();
			}
		}
		left_mat = kroneckerProduct(dyn_mat<Derived>::Identity(1 << (N - 2), 1 << (N - 2)), unitary_block[0])*left_mat;
		dyn_mat<Derived> right_mat = dyn_mat<Derived>::Identity(1 << N, 1 << N);
		while (iter < T) {

			for (int i = 0; i < size; ++i) {
				QCircuit cir_temp = QCircuit();
				if (i != 0) {
					//auto start = chrono::system_clock::now();
					dyn_mat<Derived> mat_temp_top1 = dyn_mat<Derived>::Identity(1 << i % (N - 1), 1 << i % (N - 1));
					dyn_mat<Derived> mat_temp_top2 = dyn_mat<Derived>::Identity(1 << (i - 1) % (N - 1), 1 << (i - 1) % (N - 1));
					dyn_mat<Derived> mat_temp_bottom1 = dyn_mat<Derived>::Identity(1 << N - (i % (N - 1)) - 2, 1 << N - (i % (N - 1)) - 2);
					dyn_mat<Derived> mat_temp_bottom2 = dyn_mat<Derived>::Identity(1 << N - ((i - 1) % (N - 1)) - 2, 1 << N - ((i - 1) % (N - 1)) - 2);
					dyn_mat<Derived> tmp = kroneckerProduct(mat_temp_bottom1, unitary_block[i]);
					dyn_mat<Derived> tmp_ = kroneckerProduct(tmp, mat_temp_top1);
					dyn_mat<Derived> tmp_conj = kroneckerProduct(mat_temp_bottom2, unitary_block[i - 1].transpose().conjugate());
					dyn_mat<Derived> tmp_conj_ = kroneckerProduct(tmp_conj, mat_temp_top2);
					left_mat = tmp_ * left_mat;
					right_mat = right_mat * tmp_conj_;
				}
				dyn_mat<Derived> _m0 = dyn_mat<Derived>::Zero(1, (1 << N));
				_m0(0, 0) = 1;
				dyn_mat<Derived> M_ = _m0 * right_mat;
				auto state = left_mat * data_mat;
				dyn_mat<Derived> M = state * M_;
				std::vector<int> target(N - 2);
				int cnt = 0;
				for (int k = 0; k < N; ++k) {
					if (k != N - i % (N - 1) - 1 && k != N - i % (N - 1) - 2) {
						target[cnt++] = k;
					}
				}
				dyn_mat<Derived> matrix1 = _ptrace(M, target, dims, complex_flag);
				dyn_mat<Derived> U2;
				Eigen::JacobiSVD<dyn_mat<Derived>> svd1(matrix1, Eigen::DecompositionOptions::ComputeThinU | Eigen::DecompositionOptions::ComputeThinV);
				dyn_mat<Derived> U1 = svd1.matrixU();
				dyn_mat<Derived> V1 = svd1.matrixV().transpose().conjugate();

				unitary_block[i] = U1 * V1;
				if (!unitary_block[i].isUnitary(1e-6)) {
					_orthonormalize(unitary_block[i]);
				}
			}
			dyn_mat<Derived> mat_temp_top = dyn_mat<Derived>::Identity(1 << (N - 2), 1 << (N - 2));

			dyn_mat<Derived> tmp_ = kroneckerProduct(unitary_block[size - 1].transpose().conjugate(), mat_temp_top);

			dyn_mat<Derived> _tmp = kroneckerProduct(mat_temp_top, unitary_block[0]);
			left_mat = _tmp * right_mat * tmp_;
			right_mat = dyn_mat<Derived>::Identity(1 << N, 1 << N);

			iter++;

		}

		for (int i = 0; i < size; ++i) {

			double _tol = 1e-7;
			if (std::is_same<Derived, float>::value
				|| std::is_same<Derived, std::complex<float>>::value)
			{
				_tol = 1e-4;
			}

			Eigen::Matrix4cd _in_mat;
			for (int _r = 0; _r < unitary_block[i].rows(); _r++)
			{
				for (int _c = 0; _c < unitary_block[i].cols(); _c++) {
					_in_mat(_r, _c) = unitary_block[i](_r, _c);
				}
			}
			cir_qsd << unitary_decomposer_2q(_in_mat, { q[i % (N - 1)],q[i % (N - 1) + 1] }, true, _tol);
		}


		return cir_qsd;
	};
	QCircuit _decompose2q(Eigen::MatrixXcd matrix, const QVec&q);
private:

	QCircuit m_qcircuit;

	QVec m_out_qubits;



};
QPANDA_END

#endif
