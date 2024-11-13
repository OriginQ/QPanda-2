#include "Core/Core.h"
#include <complex>
#include <algorithm>
#include <EigenUnsupported/Eigen/CXX11/Tensor>
#include <bitset>
#include <iomanip>
#include <numeric> 

#include "Core/Utilities/Encode/Encode.h"

#include "Core/Utilities/UnitaryDecomposer/QSDecomposition.h"
#include "Core/Utilities/UnitaryDecomposer/IsometryDecomposition.h"
#include "Core/Utilities/UnitaryDecomposer/UniformlyControlledGates.h"
#include "QPanda.h"

USING_QPANDA

StateNode::StateNode(int out_index, int out_level, double out_amplitude, StateNode* out_left, StateNode* out_right) :index(out_index), level(out_level), amplitude(out_amplitude)
{
	if (nullptr == out_left)
	{
		left = nullptr;
	}
	else
	{
		left = out_left;
	}
	if (nullptr == out_right)
	{
		right = nullptr;
	}
	else
	{
		right = out_right;
	}
}
NodeAngleTree::NodeAngleTree(int out_index, int out_level, int out_qubit_index, double out_angle, NodeAngleTree* out_left, NodeAngleTree* out_right) :index(out_index), level(out_level), qubit_index(out_qubit_index), angle(out_angle)
{
	if (nullptr == out_left)
	{
		left = nullptr;
	}
	else
	{
		left = out_left;
	}
	if (nullptr == out_right)
	{
		right = nullptr;
	}
	else
	{
		right = out_right;
	}
}



Encode::Encode()
{
	m_qcircuit = QCircuit();
	m_out_qubits = QVec();

};

void Encode::basic_encode(const QVec &q, const std::string& data) {

	for (auto i : data) {
		if (i != '0' && i != '1') {
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b is error.");
		}
	}
	if (q.size() < data.size()) {
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input qvec q is error.");
	}

	int k = 0;
	string data_temp = data;
	std::reverse(data_temp.begin(), data_temp.end());

	for (auto i : data_temp)
	{
		if (i == '1')
			m_qcircuit << X(q[k]);
		++k;
	}

	for (int i = 0; i < k; ++i) {
		m_out_qubits.push_back(q[i]);
	}

}

void Encode::amplitude_encode_recursive(const QVec &q, const std::vector<double>& data)
{
	vector<double>data_temp(data);

	if (!_check_normalized(data_temp)) {
		throw run_fail("Data is not normalized");
	}

	if (data.size() > (1 << q.size()))
	{
		throw run_fail("Amplitude_encode parameter error.");
	}

	while (data_temp.size() < (1 << q.size()))
	{
		data_temp.push_back(0);
	}
	m_qcircuit = _recursive_compute_beta(q, data_temp);

	m_out_qubits = q;

	return;
}

void Encode::amplitude_encode_recursive(const QVec &qubits, const QStat& full_cur_vec)
{
	const auto _dimension = full_cur_vec.size();
	const double max_precision = 1e-13;

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

	double tmp_sum = 0.0;
	for (size_t i = 0; i < _dimension; ++i)
	{
		auto tmp_m = full_cur_vec[i].real() * full_cur_vec[i].real() + full_cur_vec[i].imag() * full_cur_vec[i].imag();
		tmp_sum += tmp_m;
		tatal += tmp_m;
		ui_mod[i] = sqrt(tmp_m);
		ui_angle[i] = arg(full_cur_vec[i]);
	}

	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}
	QStat mat_d(_dimension * _dimension, qcomplex_t(0, 0));
	for (size_t i = 0; i < _dimension; ++i)
	{
		mat_d[i + i * _dimension] = std::exp(qcomplex_t(0, ui_angle[i]));
	}

	QCircuit cir_d = diagonal_matrix_decompose(qubits, mat_d);

	amplitude_encode_recursive(qubits, ui_mod);
	m_qcircuit << cir_swap_qubits << cir_d << cir_swap_qubits;

	return;
}

QCircuit Encode::_recursive_compute_beta(const QVec& q, const std::vector<double>&data)
{

	QCircuit qcir;
	double sum_0 = 0;
	double sum_1 = 0;
	size_t high_bit = (size_t)std::log2(data.size()) - 1;

	if (high_bit == 0)
	{
		if (data[0] > 1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], 2 * ::acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] > 1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], -2 * ::acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] < -1e-20 && data[1] > 1e-20)
		{
			qcir << RY(q[0], 2 * ::acos(data[0] / sqrt(data[0] * data[0] + data[1] * data[1])));
		}
		else if (data[0] < -1e-20 && data[1] < -1e-20)
		{
			qcir << RY(q[0], 2 * (2 * PI - ::acos(data[0] / std::sqrt(data[0] * data[0] + data[1] * data[1]))));
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
			qcir << RY(q[high_bit], 2 * ::acos(std::sqrt(sum_0 / (sum_0 + sum_1))));
		else
		{
			throw run_fail("Amplitude_encode error.");
		}

		if (sum_0 > 1e-20)
		{
			QVec temp({ q[high_bit] });
			std::vector<double> vtemp(data.begin(), data.begin() + (data.size() >> 1));
			qcir << X(q[high_bit]) << _recursive_compute_beta(q - temp, vtemp).control({ q[high_bit] }) << X(q[high_bit]);
		}
		if (sum_1 > 1e-20)
		{
			QVec temp({ q[high_bit] });
			std::vector<double> vtemp(data.begin() + (data.size() >> 1), data.end());
			qcir << _recursive_compute_beta(q - temp, vtemp).control({ q[high_bit] });
		}
	}

	return qcir;
}

void Encode::amplitude_encode(const QVec &q, const std::vector<double>& data)
{
	vector<double>data_temp(data);

	if (!_check_normalized(data_temp)) {
		throw run_fail("Data is not normalized");
	}

	if (data_temp.size() > (1 << q.size()))
	{
		throw run_fail("Amplitude_encode parameter error.");
	}

	QVec qubits;
	int k = 0;

	for (auto i : q) {
		if (k >= std::ceil(std::log2(data.size()))) break;
		qubits.push_back(i);
		++k;
	}

	while (data_temp.size() < (1 << qubits.size()))
	{
		data_temp.push_back(0);
	}

	std::vector<std::vector<double>>betas(qubits.size());
	std::vector<double>input(data_temp);
	_recursive_compute_beta(data_temp, betas, (int)(qubits.size() - 1));
	_generate_circuit(betas, qubits);

	for (int i = 0; i < std::ceil(std::log2(data.size())); ++i)
	{
		m_out_qubits.push_back(q[i]);
	}

	return;
}
void Encode::amplitude_encode(const QVec &q, const std::vector<complex<double>>& data)
{
	vector<complex<double>> data_temp(data);
	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.real()*i.real()) + (i.imag()*i.imag());
	}

	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}

	double norm = 1.0;
	if (data_temp.size() > (1 << q.size()))
	{
		throw run_fail("Amplitude_encode parameter error.");
	}
	QVec qubits;
	int k = 0;

	for (auto i : q) {
		if (k >= std::ceil(std::log2(data.size()))) break;
		qubits.push_back(i);
		++k;
	}

	while (data_temp.size() < (1 << qubits.size()))
	{
		data_temp.push_back(0);
	}

	const auto _dimension = data_temp.size();
	if (data_temp.size() != _dimension)
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
		auto tmp_m = data_temp[i].real() * data_temp[i].real() + data_temp[i].imag() * data_temp[i].imag();
		tatal += tmp_m;
		ui_mod[i] = std::sqrt(tmp_m);
		ui_angle[i] = arg(data_temp[i]);
	}

	QStat mat_d(_dimension * _dimension, qcomplex_t(0, 0));
	for (size_t i = 0; i < _dimension; ++i)
	{
		mat_d[i + i * _dimension] = exp(qcomplex_t(0, ui_angle[i]));
	}

	amplitude_encode(qubits, ui_mod);
	QCircuit cir_d = diagonal_matrix_decompose(qubits, mat_d);
	m_qcircuit << cir_swap_qubits << cir_d << cir_swap_qubits;

	for (int i = 0; i < ceil(log2(data.size())); ++i)
	{
		m_out_qubits.push_back(q[i]);
	}
	return;
}
void Encode::_generate_circuit(std::vector<std::vector<double>> &betas, const QVec &quantum_input) {
	int numberof_controls = 0;
	int size = quantum_input.size();
	QVec control_bits;
	int num = 0;
	for (auto angles : betas) {
		if (numberof_controls == 0) {
			m_qcircuit << RY(quantum_input[size - 1], angles[0]);
			numberof_controls += 1;
			control_bits.push_back(quantum_input[size - 1]);
		}
		else
		{
			for (int k = angles.size() - 1; k >= 0; --k)
			{

				if (k == angles.size() - 1) {

					num = angles.size() - 1 - k;

				}
				else
				{
					num = (angles.size() - 1 - k) ^ (angles.size() - 2 - k);
				}
				_index(num, control_bits, numberof_controls);
				m_qcircuit << RY(quantum_input[size - 1 - numberof_controls], angles[k]).control(control_bits);
				//_index(num, control_bits, numberof_controls);
			}
			_index(angles.size() - 1, control_bits, numberof_controls);
			control_bits.push_back(quantum_input[size - 1 - numberof_controls]);
			numberof_controls += 1;
		}
	}
	return;
}

void Encode::angle_encode(const QVec &q, const std::vector<double>& data, const GateType& gate_type)
{

	if (data.size() > q.size())
	{
		throw run_fail("Qubit_encode parameter error.");
	}
	switch (gate_type)
	{
	case RX_GATE:
		for (auto i = 0; i < data.size(); ++i)
		{
			m_qcircuit << RX(q[i], data[i]);
		}
		break;

	case RY_GATE:
		for (auto i = 0; i < data.size(); ++i)
		{
			m_qcircuit << RY(q[i], data[i]);
		}
		break;

	case RZ_GATE:
		for (auto i = 0; i < data.size(); ++i)
		{
			m_qcircuit << RZ(q[i], data[i]);
		}
		break;

	default:
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input gate type error.");
		break;
	}

	for (int i = 0; i < data.size(); ++i) {
		m_out_qubits.push_back(q[i]);
	}
	return;
}

void Encode::_recursive_compute_beta(const std::vector<double>&input_vector, std::vector<std::vector<double>>&betas, int count) {
	if (input_vector.size() > 1)
	{
		size_t size = input_vector.size() / 2, cnt = 0;
		std::vector<double>new_x(size);
		std::vector<double>beta(size);

		for (auto k = 0; k < input_vector.size(); k += 2) {
			double norm = std::sqrt(input_vector[k] * input_vector[k] + input_vector[k + 1] * input_vector[k + 1]);
			new_x[cnt] = norm;
			if (norm == 0) {
				beta[cnt] = 0;
			}
			else if (input_vector[k] < 0) {
				beta[cnt] = 2 * PI - 2 * ::asin(input_vector[k + 1] / norm);
			}
			else
			{
				beta[cnt] = 2 * ::asin(input_vector[k + 1] / norm);
			}
			++cnt;
		}

		_recursive_compute_beta(new_x, betas, count - 1);
		betas[count] = beta;
	}
	return;
}

void Encode::_dc_generate_circuit(std::vector<std::vector<double>>& betas, const QVec &quantum_input, const int cnt) {
	int k = 0;

	for (auto angles : betas) {
		for (auto angle : angles) {
			m_qcircuit << RY(quantum_input[k++], angle);
		}
	}

	int last = quantum_input.size() - 1;
	int	actual = (int)(last - 0.5) / 2;
	int	level = (int)(last - 0.5) / 2;

	while (actual >= 0) {
		int left_index = (int)(2 * actual + 1);
		int right_index = (int)(2 * actual + 2);
		while (right_index <= last) {
			m_qcircuit << SWAP(quantum_input[left_index], quantum_input[right_index]).control(quantum_input[actual]);
			left_index = (int)(2 * left_index + 1);
			right_index = (int)(2 * right_index + 1);
		}
		actual--;
		if (level != (int)(actual - 0.5) / 2) {
			level--;
		}
	}

	int next_index = cnt;

	while (next_index - 1 > 0) {
		m_out_qubits.push_back(quantum_input[(next_index >> 1) - 1]);
		next_index = (int)((next_index >> 1));
	}

	return;
}

void Encode::dc_amplitude_encode(const QVec &q, const std::vector<double>& data)
{
	vector<double>data_temp(data);

	if (!_check_normalized(data_temp)) {
		throw run_fail("Data is not normalized");
	}

	int size = data_temp.size();
	int log_ceil_size = std::ceil(std::log2(size));
	if (1 << log_ceil_size > q.size() + 1) {
		throw run_fail("Dc_Amplitude_encode parameter error.");
	}
	while (data_temp.size() < (1 << log_ceil_size))
	{
		data_temp.push_back(0);
	}

	std::vector<std::vector<double>>betas(std::log2(data_temp.size()));
	std::vector<double>input(data_temp);
	_recursive_compute_beta(data_temp, betas, (int)(std::log2(data_temp.size()) - 1));
	_dc_generate_circuit(betas, q, (int)data_temp.size());

	return;
}

void Encode::_index(const int value, const QVec &control_qubits, const int numberof_controls) {
	bitset<32> temp(value);
	std::string str = temp.to_string();

	for (int i = 32 - numberof_controls, k = 0; i < 32; ++i, ++k)
	{
		if (str[i] == '1')
		{
			m_qcircuit << X(control_qubits[k]);
		}
	}
}

void Encode::dense_angle_encode(const QVec &q, const std::vector<double>& data)
{

	if (data.size() > q.size() * 2)
	{
		throw run_fail("Dense_angle_encode parameter error.");
	}

	std::vector<double>data_temp(data);
	if ((int)(data.size()) & 1)
	{
		data_temp.push_back(0);
	}

	int k = (int)(data_temp.size());
	k /= 2;

	for (auto i = 0; i < data_temp.size() / 2; ++i)
	{
		m_qcircuit << U3(q[i], data_temp[i], data_temp[k], 0.0);
		++k;
	}

	for (int i = 0; i < data_temp.size() / 2; ++i) {
		m_out_qubits.push_back(q[i]);
	}

	return;
}
void Encode::bid_amplitude_encode(const QVec &q, const std::vector<double>& data, const int split) {

	vector<double>data_temp(data);

	if (!_check_normalized(data_temp)) {
		throw run_fail("Data is not normalized");
	}

	int n_qubits = std::ceil(std::log2(data_temp.size()));
	int split_temp = split;

	if (split == 0) {
		if (n_qubits & 1) {
			split_temp = n_qubits / 2 + 1;
		}
		else {
			split_temp = n_qubits / 2;
		}
	}

	int size = data_temp.size();
	int size_next = 1 << n_qubits;

	if ((split_temp + 1)*(size_next / (1 << split_temp)) - 1 > q.size()) {
		throw run_fail("Bid_Amplitude_encode parameter error.");
	}

	if (split_temp > std::ceil(std::log2(data_temp.size()))) {
		throw run_fail("Bid_Amplitude_encode parameter error.");
	}

	while (data_temp.size() < (1 << n_qubits))
	{
		data_temp.push_back(0);
	}

	StateNode*	state_tree = _state_decomposition(n_qubits, data_temp);

	NodeAngleTree*	angle_tree = _create_angles_tree(state_tree);

	_add_register(angle_tree, n_qubits - split_temp);

	_top_down_tree_walk(angle_tree, q, n_qubits - split_temp);

	_bottom_up_tree_walk(angle_tree, q, n_qubits - split_temp);

	_output(angle_tree, q);

	delete state_tree;
	delete angle_tree;

	return;
}
void Encode::_output(NodeAngleTree* angle_tree, const QVec &q) {
	if (angle_tree) {
		if (angle_tree->left) {
			_output(angle_tree->left, q);
		}
		else {
			_output(angle_tree->right, q);
		}
		m_out_qubits.push_back(q[angle_tree->qubit_index]);
	}
}
StateNode* Encode::_state_decomposition(int nqubits, vector<double>data)
{
	int size = data.size();
	vector<StateNode*>new_nodes;
	vector<StateNode*>nodes;

	for (int i = 0; i < size; ++i)
	{
		new_nodes.push_back(new StateNode(i, nqubits, data[i], nullptr, nullptr));
	}

	while (nqubits > 0)
	{
		nodes.swap(new_nodes);
		new_nodes.clear();
		nqubits--;
		int k = 0;
		int n_nodes = nodes.size();
		while (k < n_nodes) {
			double mag = std::sqrt(nodes[k]->amplitude* nodes[k]->amplitude + nodes[k + 1]->amplitude*nodes[k + 1]->amplitude);
			new_nodes.push_back(new StateNode((int)(nodes[k]->index / 2), nqubits, mag, nodes[k], nodes[k + 1]));
			k = k + 2;
		}
	}

	return new_nodes[0];
}
NodeAngleTree* Encode::_create_angles_tree(StateNode* state_tree)
{
	double angle = 0.0;
	if (state_tree->right)
	{
		double Amp = 0.0;
		if (state_tree->amplitude - 0.0 > 1e-6)
		{
			Amp = state_tree->right->amplitude / state_tree->amplitude;
		}
		angle = 2 * ::asin(Amp);
	}

	NodeAngleTree *node = new NodeAngleTree(state_tree->index, state_tree->level, 0, angle, nullptr, nullptr);

	if (state_tree->right->left&&state_tree->right->right) {
		node->right = _create_angles_tree(state_tree->right);
		node->left = _create_angles_tree(state_tree->left);
	}

	return node;
}
void Encode::_add_registers(NodeAngleTree* angle_tree, std::queue<int>&q, int start_level)
{
	if (angle_tree) {
		angle_tree->qubit_index = q.front();
		q.pop();
		if (angle_tree->level < start_level)
		{
			_add_registers(angle_tree->left, q, start_level);
			_add_registers(angle_tree->right, q, start_level);
		}
		else {
			if (angle_tree->left)
			{
				_add_registers(angle_tree->left, q, start_level);
			}
			else
			{
				_add_registers(angle_tree->right, q, start_level);
			}
		}

	}
}
void Encode::_top_down_tree_walk(NodeAngleTree* angle_tree, const QVec &q, int start_level, std::vector<NodeAngleTree*>control_nodes, std::vector<NodeAngleTree*>target_nodes) {
	if (angle_tree) {
		if (angle_tree->level < start_level) {
			_top_down_tree_walk(angle_tree->left, q, start_level);
			_top_down_tree_walk(angle_tree->right, q, start_level);
		}
		else
		{
			if (target_nodes.empty()) {
				target_nodes.push_back(angle_tree);
			}
			else
			{
				target_nodes = _children(target_nodes);
			}
			std::vector<double>angle;
			for (auto node : target_nodes) {
				angle.push_back(node->angle);
			}
			int target_qubits_index = target_nodes[0]->qubit_index;
			std::vector<int>control_qubits_index;
			for (auto node : control_nodes) {
				control_qubits_index.push_back(node->qubit_index);
			}
			std::reverse(angle.begin(), angle.end());
			int numberof_controls = control_qubits_index.size();
			QVec control_qubits;
			for (int i : control_qubits_index) {
				control_qubits.push_back(q[i]);
			}
			for (int k = 0; k < angle.size(); ++k) {
				_index(k, control_qubits, numberof_controls);
				if (control_qubits.empty()) {
					m_qcircuit << RY(q[target_qubits_index], angle[k]);
				}
				else {
					m_qcircuit << RY(q[target_qubits_index], angle[k]).control(control_qubits);
				}
				_index(k, control_qubits, numberof_controls);
			}
			control_nodes.push_back(angle_tree);
			_top_down_tree_walk(angle_tree->left, q, start_level, control_nodes, target_nodes);
		}
	}
}
void Encode::_apply_cswaps(NodeAngleTree* angle_tree, const QVec &q) {

	if (angle_tree->angle != 0.0) {

		auto left = angle_tree->left;
		auto right = angle_tree->right;

		while (left&&right) {
			m_qcircuit << SWAP(q[left->qubit_index], q[right->qubit_index]).control(q[angle_tree->qubit_index]);
			left = left->left;
			if (right->left) {
				right = right->left;
			}
			else
			{
				right = right->right;
			}
		}
	}
}
void Encode::_bottom_up_tree_walk(NodeAngleTree* angle_tree, const QVec &q, int start_level) {
	if (angle_tree && angle_tree->level < start_level) {

		m_qcircuit << RY(q[angle_tree->qubit_index], angle_tree->angle);

		_bottom_up_tree_walk(angle_tree->left, q, start_level);

		_bottom_up_tree_walk(angle_tree->right, q, start_level);

		_apply_cswaps(angle_tree, q);
	}
}
vector<NodeAngleTree*> Encode::_children(vector<NodeAngleTree*> nodes)
{
	vector<NodeAngleTree*>tree;

	for (auto node : nodes)
	{
		if (node->left) tree.push_back(node->left);
		if (node->right)tree.push_back(node->right);
	}
	return tree;
}
void Encode::_add_register(NodeAngleTree *angle_tree, int start_level)
{
	int level = 0;
	std::vector<int>level_nodes;
	std::vector<NodeAngleTree*>nodes;
	nodes.push_back(angle_tree);

	while (nodes.size() > 0) {
		level_nodes.push_back(nodes.size());
		nodes = _children(nodes);
		level += 1;
	}

	int noutput = level;
	int nqubits = 0;

	for (int i = 0; i < start_level; ++i) {
		nqubits += level_nodes[i];
	}

	nqubits += level_nodes[start_level] * (noutput - start_level);
	int nancilla = nqubits - noutput;
	std::queue<int>q;

	for (int i = 0; i < nqubits; ++i)
	{
		q.push(i);
	}

	_add_registers(angle_tree, q, start_level);
}

void Encode::iqp_encode(const QVec &q, const vector<double>& data, const std::vector<pair<int, int>>& control_vector, const bool& inverse, const int& repeats)
{
	int size = data.size();

	if (size > q.size()) {
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: data size or qubits size error.");
	}

	std::vector<pair<int, int>> control_vector_temp = control_vector;
	int repeats_temp = repeats;

	if (control_vector.empty())
	{
		for (int i = 0; i < size - 1; ++i) {
			control_vector_temp.push_back({ i,i + 1 });
		}
	}

	std::vector<double>data_temp(data);
	QCircuit cir_tmp = QCircuit();

	while (repeats_temp > 0)
	{
		repeats_temp--;
		for (int i = 0; i < size; ++i) {
			cir_tmp << H(q[i]);
		}

		for (int i = 0; i < size; ++i) {
			cir_tmp << RZ(q[i], data[i]);
		}

		for (auto i : control_vector_temp) {
			cir_tmp << CNOT(q[i.first], q[i.second]) << RZ(q[i.second], data[i.first] * data[i.second]) << CNOT(q[i.first], q[i.second]);
		}

		if (inverse) {
			m_qcircuit << cir_tmp.dagger();
		}
		else {
			m_qcircuit << cir_tmp;
		}
	}



	for (int i = 0; i < data.size(); ++i)
	{
		m_out_qubits.push_back(q[i]);
	}

	return;
}

void Encode::ds_quantum_state_preparation(const QVec &q, const std::map<std::string, double>& data)
{

	if (data.empty()) {

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data must not null.");
	}

	int size = (*data.begin()).first.size();

	if (size * 2 > q.size()) {

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input qubits size error.");
	}

	for (auto i : data)
	{
		if (i.first.size() != size)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must have same dimension.");
		}
		for (char c : i.first)
		{
			if (c != '0'&&c != '1')
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must be binary string.");
			}
		}

	}
	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.second*i.second);
	}
	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}

	double norm = 1.0;
	m_qcircuit << X(q[0]);
	int k = 0;
	auto iter = data.begin();
	int numqubits = (*iter).first.size();
	QVec u = q[0];
	QVec a;
	QVec m;
	for (int i = 1; i < numqubits; ++i) {
		a.push_back(q[i]);
	}
	for (int i = numqubits; i < 2 * numqubits; ++i) {
		m.push_back(q[i]);
	}
	for (auto i : data)
	{
		string binary_string = i.first;
		double feature = i.second;
		vector<int>control = _select_controls(binary_string);
		_flip_flop(u, m, control, numqubits);
		_load_superposition(u, a, m, control, numqubits, feature, norm);
		if (k < data.size() - 1) {
			_flip_flop(u, m, control, numqubits);
		}
		else {
			break;
		}
		k++;
	}
	m_out_qubits = m;
}
void Encode::ds_quantum_state_preparation(const QVec &q, const std::vector<double>&data) {

	map<string, double>state = _build_state_dict(data);
	return ds_quantum_state_preparation(q, state);
}

void Encode::ds_quantum_state_preparation(const QVec &q, const std::vector<std::complex<double>>&data)
{

	map<string, complex<double>>state = _build_state_dict(data);
	return ds_quantum_state_preparation(q, state);
}
void Encode::ds_quantum_state_preparation(const QVec &q, const std::map<std::string, std::complex<double>>& data) {

	if (data.empty())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data must not null.");
	}

	int size = (*data.begin()).first.size();

	if (size * 2 > q.size()) {

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input qubits size error.");
	}

	for (auto i : data)
	{
		if (i.first.size() != size)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must have same dimension.");
		}
		for (char c : i.first)
		{
			if (c != '0'&&c != '1')
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must be binary string.");
			}
		}

	}

	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.second.real()*i.second.real()) + (i.second.imag()*i.second.imag());
	}
	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}

	double norm = 1.0;
	m_qcircuit << X(q[0]);
	int k = 0;
	auto iter = data.begin();
	int numqubits = (*iter).first.size();
	QVec u = q[0];
	QVec a;
	QVec m;
	for (int i = 1; i < numqubits; ++i) {
		a.push_back(q[i]);
	}
	for (int i = numqubits; i < 2 * numqubits; ++i) {
		m.push_back(q[i]);
	}
	for (auto i : data)
	{
		string binary_string = i.first;
		complex<double> feature = i.second;
		vector<int>control = _select_controls(binary_string);
		_flip_flop(u, m, control, numqubits);
		_load_superposition(u, a, m, control, numqubits, feature, norm);
		if (k < data.size() - 1) {
			_flip_flop(u, m, control, numqubits);
		}
		else {
			break;
		}
		k++;
	}
	m_out_qubits = m;

}
void Encode::_flip_flop(const QVec& u, const QVec& m, std::vector<int>control, int numqbits)
{
	for (int i : control)
	{
		m_qcircuit << CNOT(u[0], m[i]);
	}
}

template<typename T>
void Encode::_load_superposition(const QVec& u, const QVec& a, const QVec& m, std::vector<int>control, int numqbits, T feature, double& norm)
{
	vector<double>angle = _compute_matrix_angles(feature, norm);

	if (control.size() == 0)
	{
		m_qcircuit << U3(u[0], angle[0], angle[1], angle[2]);
	}
	else if (control.size() == 1)
	{
		m_qcircuit << U3(u[0], angle[0], angle[1], angle[2]).control(m[control[0]]);
	}
	else
	{
		_mcuvchain(u, a, m, control, angle, numqbits);
	}

	norm = norm - std::abs(feature*feature);
	return;
}
std::vector<int> Encode::_select_controls(string binary_string)
{
	vector<int>control_qubits;

	for (int i = binary_string.size() - 1; i >= 0; --i)
	{
		if (binary_string[i] == '1')
		{
			control_qubits.push_back(binary_string.size() - i - 1);
		}
	}

	return control_qubits;
}
void Encode::_mcuvchain(const QVec &u, const QVec a, const QVec m, std::vector<int>control, std::vector<double> angle, int numqbits)
{
	vector<int>reverse_control(control);
	reverse(reverse_control.begin(), reverse_control.end());
	m_qcircuit << X(a[numqbits - 2]).control({ m[reverse_control[0]], m[reverse_control[1]] });
	std::vector<std::vector<int>>tof;
	tof.resize(numqbits);
	int k = numqbits - 1;

	for (int i = 2; i < reverse_control.size(); ++i)
	{
		m_qcircuit << X(a[k - 2]).control({ m[i], a[k - 1] });
		tof[reverse_control[i]].push_back(k - 1);
		tof[reverse_control[i]].push_back(k - 2);
		k -= 1;
	}

	m_qcircuit << U3(u[0], angle[0], angle[1], angle[2]).control(a[k - 1]);

	for (int i = control.size() - 3; i >= 0; i -= 2)
	{
		m_qcircuit << X(a[tof[control[i]][1]]).control({ m[control[i]], a[tof[control[i]][0]] });
	}

	m_qcircuit << X(a[numqbits - 2]).control({ m[control[control.size() - 1]], m[control[control.size() - 2]] });

	return;
}
std::vector<double> Encode::_compute_matrix_angles(std::complex<double> feature, double norm)
{
	double alpha = 0.0, beta = 0.0, phi = 0.0, cos_value = 0.0;
	double phase = std::abs(feature*feature);
	if (norm - phase < 1e-6) {
		cos_value = 0.0;
	}
	else {
		cos_value = std::sqrt((norm - phase) / norm);
	}
	double value = std::min(cos_value, 1.0);

	if (value < -1)
	{
		value = -1;
	}

	cos_value = value;
	alpha = 2 * (::acos(cos_value));
	beta = ::acos(-feature.real() / std::sqrt(std::abs(feature*feature)));

	if (feature.imag() < 0) {
		beta = 2 * PI - beta;
	}

	phi = -beta;

	return { alpha,beta,phi };
}
std::vector<double> Encode::_compute_matrix_angles(double feature, double norm)
{
	double alpha = 0.0, beta = 0.0, phi = 0.0;
	double sin_value = -feature / std::sqrt(norm);
	double value = std::min(sin_value, 1.0);

	if (value < -1)
	{
		value = -1;
	}

	sin_value = value;
	alpha = 2 * (::asin(sin_value));

	return { alpha,beta,phi };
}
void Encode::sparse_isometry(const QVec &q, const std::map<std::string, double>& data)
{

	if (data.empty())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data must not null.");
	}

	int size = (*data.begin()).first.size();

	for (auto i : data)
	{
		if (i.first.size() != size)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must have same dimension.");
		}
		for (char c : i.first)
		{
			if (c != '0'&&c != '1')
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must be binary string.");
			}
		}

	}

	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{

		tmp_sum += i.second*i.second;
	}

	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}

	if (data.size() == 1)
	{
		basic_encode(q, (*data.begin()).first);
		return;
	}

	std::string key;
	QVec qubits;
	key = (*data.begin()).first;
	int	n_qubits = key.size();

	if (n_qubits > q.size()) {
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input qubits size error.");
	}

	int	non_zero = data.size();
	int	target_size = ceil(log2(non_zero));
	std::map<std::string, double> next_state = data;
	int k = 0;
	for (auto i : q) {
		qubits.push_back(i);
		++k;
		if (k >= n_qubits) break;
	}

	auto iter = qubits.end();
	--iter;
	QVec qubits_reverse;

	while (iter != qubits.begin())
	{
		qubits_reverse.push_back(*iter);
		--iter;
	}

	qubits_reverse.push_back(*qubits.begin());
	string index_nonzero = _get_index_nz(next_state, n_qubits - target_size);
	QCircuit qcir;
	while (!index_nonzero.empty()) {
		std::string index_zero = _get_index_zero(next_state, n_qubits, non_zero);
		std::map<std::string, double> next = _pivoting(qcir, qubits_reverse, index_zero, index_nonzero, target_size, next_state);
		next_state = next;
		index_nonzero = _get_index_nz(next_state, n_qubits - target_size);
	}
	std::vector<double>dense_state(1 << target_size);

	for (auto i : next_state) {
		int cnt = stoi(i.first, 0, 2);
		dense_state[cnt] = i.second;
	}

	QCircuit cir;
	cir = qcir.dagger();

	if (non_zero <= 2) {
		m_qcircuit << RY(qubits[0], 2 * ::acos(dense_state[0]));
	}
	else
	{
		int size = ceil(log2(dense_state.size()));

		QVec q(qubits.begin(), qubits.begin() + size);
		amplitude_encode(q, dense_state);
	}

	m_qcircuit << cir;
	k = 0;
	m_out_qubits = qubits;

	return;
}
template<typename T>
std::string Encode::_get_index_nz(std::map<std::string, T>data, int target_size)
{
	std::string index_nonzero;
	int k = 0;

	for (auto i : data)
	{
		std::string temp(i.first.begin(), i.first.begin() + target_size);
		std::string target;
		for (int j = 0; j < target_size; ++j)
		{
			target.push_back('0');
		}
		if (temp != target) {
			index_nonzero = i.first;
			break;
		}
	}

	return index_nonzero;
}
void Encode::sparse_isometry(const QVec &q, const std::vector<double>& data) {
	map<string, double>state = _build_state_dict(data);
	return sparse_isometry(q, state);
}

void Encode::sparse_isometry(const QVec &q, const std::vector<complex<double>> &data) {
	map<string, complex<double>>state = _build_state_dict(data);
	return sparse_isometry(q, state);
}
void Encode::sparse_isometry(const QVec &q, const std::map<std::string, complex<double>>& data)
{

	if (data.empty())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data must not null.");
	}

	int size = (*data.begin()).first.size();

	for (auto i : data)
	{
		if (i.first.size() != size)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must have same dimension.");
		}
		for (char c : i.first)
		{
			if (c != '0'&&c != '1')
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must be binary string.");
			}
		}

	}

	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.second.real()*i.second.real()) + (i.second.imag()*i.second.imag());
	}

	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}

	if (data.size() == 1)
	{
		basic_encode(q, (*data.begin()).first);
		return;
	}

	std::string key;
	QVec qubits;
	key = (*data.begin()).first;
	int	n_qubits = key.size();

	if (n_qubits > q.size()) {
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input qubits size error.");
	}
	int	non_zero = data.size();
	int	target_size = ceil(log2(non_zero));
	std::map<std::string, complex<double>> next_state = data;
	int k = 0;

	for (auto i : q) {
		qubits.push_back(i);
		++k;
		if (k >= n_qubits) break;
	}

	auto iter = qubits.end();
	--iter;
	QVec qubits_reverse;

	while (iter != qubits.begin())
	{
		qubits_reverse.push_back(*iter);
		--iter;
	}

	qubits_reverse.push_back(*qubits.begin());
	string index_nonzero = _get_index_nz(next_state, n_qubits - target_size);
	QCircuit qcir;

	while (!index_nonzero.empty()) {
		std::string index_zero = _get_index_zero(next_state, n_qubits, non_zero);
		std::map<std::string, complex<double>> next = _pivoting(qcir, qubits_reverse, index_zero, index_nonzero, target_size, next_state);
		next_state = next;
		index_nonzero = _get_index_nz(next_state, n_qubits - target_size);
	}

	std::vector<complex<double>>dense_state(1 << target_size);

	for (auto i : next_state) {
		int cnt = stoi(i.first, 0, 2);
		dense_state[cnt] = i.second;
	}

	QCircuit cir;
	cir = qcir.dagger();
	size = ceil(log2(dense_state.size()));
	QVec q_temp(qubits.begin(), qubits.begin() + size);
	amplitude_encode(q_temp, dense_state);
	m_qcircuit << cir;
	k = 0;
	m_out_qubits = qubits;

	return;
}
template<typename T>
std::string Encode::_get_index_zero(std::map<std::string, T>data, int n_qubits, int non_zero)
{
	std::string index_zero;
	size_t count = 1 << n_qubits;

	for (int i = 0; i < count; ++i)
	{
		bitset<32> temp(i);
		std::string str = temp.to_string();
		std::string txt(str.begin() + 32 - n_qubits, str.end());
		bool flag = false;
		for (auto i : data)
		{
			if (i.first == txt) {
				flag = true;
				break;
			}
		}
		if (!flag)
		{
			index_zero = txt;
			break;
		}
	}

	return index_zero;
}

template<typename T>
std::map<std::string, T> Encode::_next_state(char ctrl_state, int index_differ, std::string index_zero, std::vector<int>remain, std::map<std::string, T> state, std::vector<int>target_cx)
{
	std::map<char, char>tab;
	tab['0'] = '1';
	tab['1'] = '0';
	std::map<std::string, T>new_state;

	for (auto i : state)
	{
		std::string n_index, str;
		if (i.first[index_differ] == ctrl_state)
		{

			for (int k = 0; k < i.first.size(); ++k)
			{
				if (find(target_cx.begin(), target_cx.end(), k) != target_cx.end())
				{
					n_index = n_index + tab[i.first[k]];
				}
				else
				{
					n_index = n_index + i.first[k];
				}
			}
		}
		else
		{
			n_index = i.first;
		}

		bool flag = true;
		int cnt = remain[0];

		while (cnt < n_index.size() && cnt < index_zero.size())
		{
			if (n_index[cnt] != index_zero[cnt])
			{
				flag = false;
				break;
			}
			++cnt;
		}

		if (flag)
		{

			for (int i = 0; i < index_differ; ++i)
			{
				str.push_back(n_index[i]);
			}
			str.push_back(tab[i.first[index_differ]]);
			for (int i = index_differ + 1; i < n_index.size(); ++i)
			{
				str.push_back(n_index[i]);
			}
		}

		new_state[str.empty() ? str = n_index : str] = state[i.first];

	}

	return new_state;
}

template<typename T>
std::map<std::string, T>  Encode::_pivoting(QCircuit& qcir, const QVec &qubits, std::string &index_zero, std::string &index_nonzero, int target_size, std::map<std::string, T> state)
{
	int n_qubits = index_zero.size();
	std::vector<int>remain, target;

	for (int i = 0; i < n_qubits - target_size; ++i)
	{
		target.push_back(i);
	}

	for (int i = n_qubits - target_size; i < n_qubits; ++i)
	{
		remain.push_back(i);
	}

	int index_differ = 0;
	char ctrl_state;

	for (int k = 0; k < target.size(); ++k)
	{
		if (index_nonzero[target[k]] != index_zero[target[k]]) {
			index_differ = target[k];
			ctrl_state = index_nonzero[target[k]];
			break;
		}
	}

	vector<int>target_cx;
	for (int k = 0; k < target.size(); ++k)
	{
		if (index_differ != target[k] && index_nonzero[target[k]] != index_zero[target[k]]) {
			qcir << X(qubits[target[k]]).control(qubits[index_differ]);
			target_cx.push_back(target[k]);
		}
	}

	for (int k = 0; k < remain.size(); ++k) {
		if (index_nonzero[remain[k]] != index_zero[remain[k]]) {
			qcir << X(qubits[remain[k]]).control(qubits[index_differ]);
			target_cx.push_back(remain[k]);
		}
	}

	for (int k = 0; k < remain.size(); ++k) {
		if (index_zero[remain[k]] == '0')
		{
			qcir << X(qubits[remain[k]]);
		}
	}

	QVec control_qubits;

	for (int k = 0; k < remain.size(); ++k) {
		control_qubits.push_back(qubits[remain[k]]);
	}

	qcir << X(qubits[index_differ]).control(control_qubits);

	for (int k = 0; k < remain.size(); ++k) {
		if (index_zero[remain[k]] == '0')
		{
			qcir << X(qubits[remain[k]]);
		}
	}

	std::map<std::string, T> new_state = _next_state(ctrl_state, index_differ, index_zero, remain, state, target_cx);

	return new_state;
}

void Encode::schmidt_encode(const QVec &q, const std::vector<double>& data, const double cutoff)
{
	vector<double>data_temp(data);

	if (!_check_normalized(data_temp)) {
		throw run_fail("Data is not normalized");
	}

	if (data_temp.size() > (1 << q.size()))
	{
		throw run_fail("Schmidt_encode parameter error.");
	}

	_schmidt(q, data_temp, cutoff);

	int count = log2(data_temp.size());
	int cnt = 0;
	for (auto i : q)
	{
		if (cnt < count) {
			m_out_qubits.push_back(i);
		}
		cnt++;
	}

	return;
}
void Encode::_schmidt(const QVec &q, const std::vector<double>& data, const double cutoff) {

	vector<double>data_temp = data;

	QVec qubits;
	int k = 0;

	for (auto i : q) {
		if (k >= ceil(log2(data.size()))) break;
		qubits.push_back(i);
		++k;
	}

	if (qubits.size() == 1) {

		if (data_temp[0] < 0) {

			m_qcircuit << RY(qubits[0], 2 * PI - 2 * ::acos(fminl(fmaxl(data_temp[0], -1.0), 1.0)));

		}
		else {

			m_qcircuit << RY(qubits[0], 2 * ::acos(fminl(fmaxl(data_temp[0], -1.0), 1.0)));

		}

		return;
	}

	while (data_temp.size() < (1 << qubits.size()))
	{
		data_temp.push_back(0);
	}

	int size = data_temp.size();
	int n_qubits = log2(size);
	int	r = n_qubits % 2;
	int row = 1 << (n_qubits >> 1);
	int col = 1 << ((n_qubits >> 1) + r);
	QMatrixXd eigen_matrix = QMatrixXd::Zero(row, col);
	k = 0;

	for (auto rdx = 0; rdx < row; ++rdx)
	{
		for (auto cdx = 0; cdx < col; ++cdx)
		{
			eigen_matrix(rdx, cdx) = data_temp[k++];
		}
	}

	Eigen::JacobiSVD<QMatrixXd> svd(eigen_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
	QMatrixXd V = svd.matrixV().transpose(), U = svd.matrixU();
	auto A = svd.singularValues();
	size_t rows = U.rows();
	size_t cols = U.cols();
	int length = 1;
	while (length < A.size() && A[length] >= A[0] * cutoff) {
		length += 1;
	}
	//length = 1 << (size_t)ceil(log2(length));
	VectorXd A_cut(length);
	for (size_t i = 0; i < length; ++i)
	{
		A_cut[i] = A[i];
	}
	auto PartU = U.leftCols(length);
	auto PartV = V.topRows(length);
	QVec A_qubits, B_qubits;

	k = 0;
	for (auto i : qubits)
	{
		if (k < floor(n_qubits / 2 + r)) {
			A_qubits.push_back(i);
		}
		++k;
	}

	k = A_qubits.size();

	for (; k < qubits.size(); ++k)
	{
		B_qubits.push_back(qubits[k]);
	}
	size_t bit = int(log2(length));

	if (int(log2(length)) > 0) {
		QVec reg_tmp;
		for (int i = 0; i < bit; ++i) {
			reg_tmp.push_back(B_qubits[i]);
		}
		A_cut.normalize();
		_unitary(reg_tmp, A_cut, cutoff);
	}
	//if (A_vec.size() > 2)
	//{
	//	_schmidt(B_qubits, A_vec, cutoff);
	//}
	//else
	//{
	//	if (A_vec[0] < 0) {

	//		m_qcircuit << RY(B_qubits, 2 * PI - 2 * ::acos(fminl(fmaxl(A_vec[0], -1.0), 1.0)));


	//	}
	//	else {

	//		m_qcircuit << RY(B_qubits, 2 * ::acos(fminl(fmaxl(A_vec[0], -1.0), 1.0)));

	//	}
	//}
	for (int i = 0; i < bit; ++i)
	{
		m_qcircuit << CNOT(B_qubits[i], A_qubits[i]);
	}

	_unitary(B_qubits, PartU, cutoff);

	_unitary(A_qubits, PartV.transpose(), cutoff);

	return;
}

void Encode::_unitary(const QVec &q, QMatrixXd gate, const double cutoff) {
	int n_qubits = q.size();
	IsoScheme scheme = IsoScheme::KNILL;
	QCircuit circuit = QCircuit();
	if (gate.cols() == 1 & (n_qubits & 1 == 0 || n_qubits < 4)) {
		vector<double>data_tmp;
		for (int i = 0; i < gate.rows(); ++i) {
			data_tmp.push_back(gate(i, 0));
		}
		_schmidt(q, data_tmp, cutoff);
	}
	else if (gate.rows() > gate.cols()) {
		circuit = isometry_decomposition(gate, q, scheme);
	}
	else {
		circuit = unitary_decomposer_nq(gate, q, DecompositionMode::QSD, true);
	}
	m_qcircuit << circuit;

	return;
}

void Encode::efficient_sparse(const QVec &q, const std::vector<double>& data)
{

	map<string, double>state = _build_state_dict(data);

	efficient_sparse(q, state);

}

void Encode::efficient_sparse(const QVec &q, const std::map<string, double>&data)
{
	if (data.empty())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data must not null.");
	}

	int size = (*data.begin()).first.size();


	for (auto i : data)
	{
		if (i.first.size() != size)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must have same dimension.");
		}
		for (char c : i.first)
		{
			if (c != '0'&&c != '1')
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must be binary string.");
			}
		}

	}

	if (1 << (*data.begin()).first.size() > 1 << (int)(q.size())) {

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input qubits size error.");
	}

	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.second*i.second);
	}

	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}

	QVec reverse_q;
	for (int i = q.size() - 1; i >= 0; --i) {
		reverse_q.push_back(q[i]);
	}

	map<string, double>state(data);
	int n_qubits = (*data.begin()).first.size();
	while (state.size() > 1) {
		state = _merging_procedure(state, reverse_q);
	}

	string b_string = (*state.begin()).first;

	for (int i = 0; i < b_string.size(); ++i)
	{
		if (b_string[i] == '1') {
			m_qcircuit << X(reverse_q[i]);
		}
	}

	m_qcircuit = m_qcircuit.dagger();
	for (int i = n_qubits - 1; i >= 0; --i) {
		m_out_qubits.push_back(reverse_q[i]);
	}

	return;
}

void Encode::efficient_sparse(const QVec &q, const std::vector<qcomplex_t>& data)
{
	map<string, qcomplex_t>state = _build_state_dict(data);

	efficient_sparse(q, state);

}


void Encode::efficient_sparse(const QVec &q, const std::map<string, qcomplex_t>&data)
{
	if (data.empty())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data must not null.");
	}

	int size = (*data.begin()).first.size();


	for (auto i : data)
	{
		if (i.first.size() != size)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must have same dimension.");
		}
		for (char c : i.first)
		{
			if (c != '0'&&c != '1')
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data.key must be binary string.");
			}
		}

	}

	if (1 << (*data.begin()).first.size() > 1 << (int)(q.size())) {

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input qubits size error.");
	}

	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.second.real()*i.second.real()) + (i.second.imag()*i.second.imag());
	}

	if (std::abs(1.0 - tmp_sum) > max_precision)
	{
		if (std::abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}

	QVec reverse_q;
	for (int i = q.size() - 1; i >= 0; --i) {
		reverse_q.push_back(q[i]);
	}

	map<string, qcomplex_t>state(data);
	int n_qubits = (*data.begin()).first.size();
	while (state.size() > 1) {
		state = _merging_procedure(state, reverse_q);
	}

	string b_string = (*state.begin()).first;


	for (int i = 0; i < b_string.size(); ++i)
	{
		if (b_string[i] == '1') {
			m_qcircuit << X(reverse_q[i]);
		}
	}

	m_qcircuit = m_qcircuit.dagger();
	for (int i = n_qubits - 1; i >= 0; --i) {
		m_out_qubits.push_back(reverse_q[i]);
	}

	return;
}
std::map<std::string, double> Encode::_build_state_dict(const std::vector<double> &state)
{

	int n_qubits = (int)ceil(log2(state.size()));
	std::map<std::string, double> state_dict;
	int cnt = 0;
	for (auto i : state) {
		if (i != 0) {
			bitset<32> temp(cnt);
			std::string str = temp.to_string();
			string binary_string(str.begin() + 32 - n_qubits, str.end());
			state_dict[binary_string] = i;
		}
		++cnt;
	}
	return state_dict;
}
std::map<std::string, qcomplex_t> Encode::_build_state_dict(const std::vector<qcomplex_t> &state)
{
	int n_qubits = (int)ceil(log2(state.size()));
	std::map<std::string, qcomplex_t> state_dict;
	int cnt = 0;
	for (auto i : state) {
		if (i.real()*i.real() + i.imag()*i.imag() != 0) {
			bitset<32> temp(cnt);
			std::string str = temp.to_string();
			string binary_string(str.begin() + 32 - n_qubits, str.end());
			state_dict[binary_string] = i;
		}
		++cnt;
	}
	return state_dict;
}

int Encode::_maximizing_difference_bit_search(vector<string> &b_strings, std::vector<std::string> &t0, std::vector<std::string> &t1, std::vector<int> &dif_qubits)
{

	int bit_index = 0;
	int set_difference = 0;
	vector<int> bit_search_space;
	int cnt = 0;
	for (int i = 0; i < b_strings[0].size(); ++i) {
		bool flag = true;
		for (int j : dif_qubits) {
			if (i - '0' == j) {
				flag = false;
				break;
			}
		}
		if (flag) bit_search_space.push_back(cnt);
		++cnt;
	}
	for (int bit : bit_search_space) {
		vector<string> temp_t0;
		vector<string> temp_t1;
		for (string bit_string : b_strings) {
			if (bit_string[bit] == '0') {
				temp_t0.push_back(bit_string);
			}
			else {
				temp_t1.push_back(bit_string);
			}
		}
		if (!temp_t0.empty() && !temp_t1.empty()) {
			int temp_difference = abs((int)(temp_t0.size() - temp_t1.size()));
			if (temp_difference == 0 && t0.empty() && t1.empty()) {
				t0 = temp_t0;
				t1 = temp_t1;
				bit_index = bit;
			}
			else if (temp_difference > set_difference) {
				t0 = temp_t0;
				t1 = temp_t1;
				bit_index = bit;
				set_difference = temp_difference;
			}
		}
	}
	return bit_index;
}

std::vector<string> Encode::_build_bit_string_set(const std::vector<string> &b_strings, const std::string bitstr1, std::vector<int> &dif_qubits, std::vector<int> &dif_values)
{
	vector<string> bit_string_set;

	for (auto b_string : b_strings) {
		bool include_string = true;
		int cnt = 0;
		for (auto i : dif_qubits) {
			if (b_string[i] != dif_values[cnt] + '0') {
				include_string = false;
				break;
			}
			cnt++;
		}
		if (include_string&&b_string != bitstr1) bit_string_set.push_back(b_string);
	}

	return bit_string_set;
}

vector<string> Encode::_bit_string_search(std::vector<string> b_strings, std::vector<int> &dif_qubits, std::vector<int> &dif_values)
{
	vector<string> temp_strings = b_strings;
	while (temp_strings.size() > 1) {
		vector<string> t0;
		vector<string> t1;
		int bit = _maximizing_difference_bit_search(temp_strings, t0, t1, dif_qubits);
		if (find(dif_qubits.begin(), dif_qubits.end(), bit) == dif_qubits.end()) {
			dif_qubits.push_back(bit);
		}
		if (t0.size() < t1.size()) {
			dif_values.push_back(0);
			temp_strings = t0;
		}
		else {
			dif_values.push_back(1);
			temp_strings = t1;
		}
	}
	return temp_strings;

}

template<typename T>
void Encode::_search_bit_strings_for_merging(std::string &bitstr1, std::string &bitstr2, int &dif_qubit, std::vector<int> &dif_qubits, const std::map<std::string, T> &state)
{
	vector<string>t0;
	vector<string>t1;
	vector<string>b_strings1, b_strings2;
	vector<int>dif_values;
	for (auto it : state) {
		b_strings1.push_back(it.first);
		b_strings2.push_back(it.first);
	}
	if (b_strings1.size() == 2) {
		int bit = _maximizing_difference_bit_search(b_strings1, t0, t1, dif_qubits);
		dif_qubit = bit;
		bitstr1 = t1[0];
		bitstr2 = t0[0];
	}
	else {
		b_strings1 = _bit_string_search(b_strings1, dif_qubits, dif_values);
		dif_qubit = dif_qubits.back();
		dif_qubits.pop_back();
		dif_values.pop_back();
		bitstr1 = b_strings1[0];

		//b_strings2.erase(bitstr1);
		b_strings2 = _build_bit_string_set(b_strings2, bitstr1, dif_qubits, dif_values);
		vector<string>::iterator itr = b_strings2.begin();
		while (itr != b_strings2.end())
		{
			if (*itr == bitstr1)
			{
				b_strings2.erase(itr);
				break;
			}
			itr++;

		}
		b_strings1 = _bit_string_search(b_strings2, dif_qubits, dif_values);
		bitstr2 = b_strings1[0];
	}
	return;

}
std::string Encode::_apply_x_operation_to_bit_string(const std::string &b_string, const int &qubit_indexes)
{
	string temp = b_string;
	if (temp[qubit_indexes] == '0') {
		temp[qubit_indexes] = '1';
	}
	else {
		temp[qubit_indexes] = '0';
	}

	return temp;
}

std::string Encode::_apply_cx_operation_to_bit_string(const std::string &b_string, const std::vector<int>qubit_indexes)
{
	string temp = b_string;
	if (b_string[qubit_indexes[0]] == '1')
	{
		if (temp[qubit_indexes[1]] == '0') {
			temp[qubit_indexes[1]] = '1';
		}
		else {
			temp[qubit_indexes[1]] = '0';
		}
	}
	return temp;
}

template<typename T>
std::map<std::string, T> Encode::_update_state_dict_according_to_operation(std::map<std::string, T>state_dict, const std::string &operation,
	const int &qubit_index, const vector<std::string>& merge_strings)
{
	map<std::string, T>new_state_dict;
	if (operation == "merge") {
		if (!merge_strings.empty()) {
			new_state_dict = state_dict;
			vector<T>merge_data(2);
			merge_data[0] = state_dict[merge_strings[0]];
			merge_data[1] = state_dict[merge_strings[1]];
			double norm = compute_norm(merge_data);
			new_state_dict.erase(merge_strings[1]);
			new_state_dict[merge_strings[0]] = norm;
		}
	}
	else {
		for (auto i : state_dict) {

			string temp_string;
			temp_string = _apply_x_operation_to_bit_string(i.first, qubit_index);
			new_state_dict[temp_string] = i.second;

		}

	}
	return new_state_dict;

}
double Encode::compute_norm(const vector<qcomplex_t> &data) {
	double norm = 0.0;
	for (int i = 0; i < data.size(); ++i) {
		norm += data[i].real()*data[i].real() + data[i].imag()*data[i].imag();
	}
	return sqrt(norm);
}

double Encode::compute_norm(const vector<double> &data) {
	double norm = 0.0;
	for (int i = 0; i < data.size(); ++i) {
		norm += data[i] * data[i];
	}
	return sqrt(norm);
}
template<typename T>
std::map<std::string, T> Encode::_update_state_dict_according_to_operation(std::map<std::string, T>state_dict, const std::string &operation,
	const std::vector<int> &qubit_indexes, const vector<std::string>& merge_strings)
{

	map<std::string, T>new_state_dict;
	if (operation == "merge") {
		if (!merge_strings.empty()) {
			new_state_dict = state_dict;
			vector<T>merge_data(2);
			merge_data[0] = state_dict[merge_strings[0]];
			merge_data[1] = state_dict[merge_strings[1]];
			double norm = compute_norm(merge_data);
			new_state_dict.erase(merge_strings[1]);
			new_state_dict[merge_strings[0]] = norm;
		}
	}
	else {
		for (auto i : state_dict) {

			string temp_string;
			temp_string = _apply_cx_operation_to_bit_string(i.first, qubit_indexes);
			new_state_dict[temp_string] = i.second;

		}

	}
	return new_state_dict;
}

template<typename T>
std::map<std::string, T> Encode::_equalize_bit_string_states(std::string &bitstr1, std::string & bitstr2, int &dif,
	std::map<std::string, T> &state_dict, QVec &q)
{
	vector<int>b_index_list;
	for (int i = 0; i < bitstr1.size(); ++i) {
		if (i != dif)b_index_list.push_back(i);
	}
	for (int b_index : b_index_list) {
		if (bitstr1[b_index] != bitstr2[b_index]) {
			m_qcircuit << CNOT(q[dif], q[b_index]);
			bitstr1 = _apply_cx_operation_to_bit_string(bitstr1, { dif,b_index });
			bitstr2 = _apply_cx_operation_to_bit_string(bitstr2, { dif,b_index });
			state_dict = _update_state_dict_according_to_operation(state_dict, "cx", { dif, b_index });
		}
	}
	return state_dict;
}

template<typename T>
std::map<std::string, T> Encode::_apply_not_gates_to_qubit_index_list(std::string &bitstr1, std::string & bitstr2, const std::vector<int> dif_qubits, std::map<std::string, T> &state_dict, QVec &q)
{
	for (int b_index : dif_qubits) {
		if (bitstr2[b_index] != '1') {
			m_qcircuit << X(q[b_index]);
			bitstr1 = _apply_x_operation_to_bit_string(bitstr1, b_index);
			bitstr2 = _apply_x_operation_to_bit_string(bitstr2, b_index);
			state_dict = _update_state_dict_according_to_operation(state_dict, "x", b_index);
		}
	}
	return state_dict;
}

template<typename T>
std::map<std::string, T> Encode::_preprocess_states_for_merging(std::string &bitstr1, std::string & bitstr2, int &dif, const std::vector<int> dif_qubits, std::map<std::string, T> &state_dict, QVec &q)
{
	if (bitstr1[dif] != '1') {
		m_qcircuit << X(q[dif]);
		bitstr1 = _apply_x_operation_to_bit_string(bitstr1, dif);
		bitstr2 = _apply_x_operation_to_bit_string(bitstr2, dif);
		state_dict = _update_state_dict_according_to_operation(state_dict, "x", dif);
	}
	state_dict = _equalize_bit_string_states(bitstr1, bitstr2, dif, state_dict, q);
	state_dict = _apply_not_gates_to_qubit_index_list(bitstr1, bitstr2, dif_qubits, state_dict, q);
	return state_dict;
}

std::vector<double> Encode::_compute_angles(qcomplex_t amplitude_1, qcomplex_t amplitude_2)
{
	vector<double> angles(3);
	double norm = sqrt(amplitude_1.real()*amplitude_1.real() + amplitude_1.imag()*amplitude_1.imag() + amplitude_2.real()*amplitude_2.real() + amplitude_2.imag()*amplitude_2.imag());
	angles[0] = 2 * ::asin(abs(amplitude_2 / norm));
	angles[1] = -log(amplitude_2 / norm).imag();
	angles[2] = -log(amplitude_1 / norm).imag() - angles[1];
	return angles;
}

vector<double> Encode::_compute_angles(double amplitude_1, double amplitude_2)
{
	double norm = sqrt(amplitude_1*amplitude_1 + amplitude_2 * amplitude_2);
	if (amplitude_1 < 0) {
		return { 2 * PI - 2 * ::asin(amplitude_2 / norm),0.0,0.0 };
	}
	return { 2 * ::asin(amplitude_2 / norm),0.0,0.0 };
}

template<typename T>
std::map<std::string, T> Encode::_merging_procedure(std::map<std::string, T> &state_dict, QVec &q)
{

	string bitstr1, bitstr2;
	int dif = 0;
	vector<int> dif_qubits;
	vector<double>angles(3);
	_search_bit_strings_for_merging(bitstr1, bitstr2, dif, dif_qubits, state_dict);
	state_dict = _preprocess_states_for_merging(bitstr1, bitstr2, dif, dif_qubits, state_dict, q);
	angles = _compute_angles(state_dict[bitstr1], state_dict[bitstr2]);
	QVec control_qubits;
	for (int i : dif_qubits) {
		control_qubits.push_back(q[i]);
	}
	m_qcircuit << U3(q[dif], angles[0], angles[1], angles[2]).control(control_qubits);
	vector<int> qubit_indexes;
	state_dict = _update_state_dict_according_to_operation(state_dict, "merge", qubit_indexes, { bitstr1, bitstr2 });
	return state_dict;
}
double Encode::_kl_divergence(const vector<double> &input_data, const vector<double> &output_data) {
	int size = input_data.size();
	double result = 0.0;
	for (int i = 0; i < size; ++i) {
		if (input_data[i] - 0.0 > 1e-6) {
			result += input_data[i] * std::log(input_data[i] / output_data[i]);
		}
	}
	return abs(result);

}

void Encode::_gen_circuit(QCircuit &circuit, const QVec& q, const int N, const std::vector<Eigen::MatrixXf>& unitary_block) {

	int size = unitary_block.size();
	for (int j = 0; j < size; ++j) {

		Eigen::MatrixXd unitary = unitary_block[j].cast<double>();
		circuit << QPanda::QOracle({ q[j % (N - 1)],q[j % (N - 1) + 1] }, unitary, 1e-6);

	}
	return;
}
void Encode::_gen_circuit(QCircuit &circuit, const QVec& q, const int N, const vector<MatrixXd>& unitary_block) {
	int size = unitary_block.size();
	for (int j = 0; j < size; ++j) {

		circuit << QPanda::QOracle({ q[j % (N - 1)],q[j % (N - 1) + 1] }, unitary_block[j], 1e-8);

	}
	return;
}
void Encode::_gen_circuit(QCircuit &circuit, const QVec& q, const int N, const std::vector<Eigen::MatrixXcd>& unitary_block) {
	int size = unitary_block.size();

	for (int j = 0; j < size; ++j) {

		circuit << QOracle({ q[j % (N - 1)],q[j % (N - 1) + 1] }, unitary_block[j], 1e-8);

	}
	return;
}
void Encode::_qstat2eigen(const QStat &stat, int N, MatrixXd &_U) {
	for (int i = 0; i < stat.size(); ++i) {
		_U(i / (1 << N), i % (1 << N)) = stat[i].real();
	}
}
void Encode::_qstat2eigen(const QStat &stat, int N, Eigen::MatrixXf &_U) {
	for (int i = 0; i < stat.size(); ++i) {
		_U(i / (1 << N), i % (1 << N)) = stat[i].real();
	}
}
void Encode::_qstat2eigen(const QStat &stat, int N, Eigen::MatrixXcd &_U) {
	for (int i = 0; i < stat.size(); ++i) {
		_U(i / (1 << N), i % (1 << N)) = stat[i];
	}
}
QCircuit Encode::_decompose2q(Eigen::MatrixXcd matrix, const QVec&q) {
	Eigen::MatrixXcd q1 = matrix.block<2, 2>(0, 0);
	Eigen::MatrixXcd q2 = matrix.block<2, 2>(2, 2);
	Eigen::MatrixXcd q3 = matrix.block<2, 2>(0, 2);
	Eigen::MatrixXcd q4 = matrix.block<2, 2>(2, 0);
	Eigen::JacobiSVD<Eigen::MatrixXcd> svd1(q1, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXcd U1 = svd1.matrixU();
	std::cout << U1 << std::endl;
	Eigen::MatrixXcd V1 = svd1.matrixV().transpose();
	auto C = svd1.singularValues();
	Eigen::JacobiSVD<Eigen::MatrixXcd> svd2(q2, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXcd U2 = svd2.matrixU();

	std::cout << U2 << std::endl;
	Eigen::MatrixXcd V2 = svd2.matrixV().transpose();
	Eigen::JacobiSVD<Eigen::MatrixXcd> svd3(q3, Eigen::ComputeFullU | Eigen::ComputeFullV);
	auto S = svd3.singularValues();
	std::cout << svd3.matrixU() << std::endl;
	Eigen::JacobiSVD<Eigen::MatrixXcd> svd4(q4, Eigen::ComputeFullU | Eigen::ComputeFullV);
	std::cout << svd4.singularValues() << std::endl;
	vector<double>theta_u(2);
	vector<double>theta_v(2);
	theta_u[0] = ::acos(U1(0, 0).real()) * 2;
	theta_u[1] = ::acos(U2(0, 0).real()) * 2;
	theta_v[0] = ::acos(V1(0, 0).real()) * 2;
	theta_v[1] = ::acos(V2(0, 0).real()) * 2;
	for (int i = 0; i < 2; ++i) {
		std::cout << C(i) << S(i) << std::endl;
	}
	std::cout << C(0)*C(0) + S(1)*S(1) << std::endl;
	std::cout << C(1)*C(1) + S(0)*S(0) << std::endl;
	double theta1, theta2;
	theta1 = ::acos(C(0)) * 2;
	theta2 = ::acos(C(1)) * 2;
	QCircuit circuit = QCircuit();
	circuit << ucry_decomposition(q[1], q[0], theta_v) << ucry_decomposition(q[0], q[1], { theta1,theta2 }) << ucry_decomposition(q[1], q[0], theta_u);
	std::cout << circuit << std::endl;
	return circuit;
}
Eigen::MatrixXcd Encode::_partial_trace(const int axis1, const int axis2, const int size, const Eigen::MatrixXcd & M) {

	int rows1 = 1 << (size - axis2 - 1);
	int rows2 = 1 << (axis1);
	Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(4, 4);
	Eigen::MatrixXcd partial = Eigen::MatrixXcd::Zero(4, 4);
	Eigen::MatrixXcd mat1, mat2, part1, part2;

	for (int i = 0; i < rows1; ++i) {
		mat1 = Eigen::MatrixXcd::Zero(1, rows1);
		mat1(0, i) = 1;
		part1 = kroneckerProduct(mat1, mat);
		for (int j = 0; j < rows2; ++j) {
			mat2 = Eigen::MatrixXcd::Zero(1, rows2);
			mat2(0, j) = 1;
			part2 = kroneckerProduct(part1, mat2);
			partial += part2 * M*part2.transpose().conjugate();
		}
	}

	return partial;
}

QCircuit Encode::get_circuit() {

	return m_qcircuit;
}
QVec Encode::get_out_qubits() {

	return m_out_qubits;
}

double Encode::get_fidelity(const vector<float> &data) {

	vector<float>data_temp(data);

	//_normalized(data_temp);

	vector<qcomplex_t>data_complex(data_temp.size());

	for (int i = 0; i < data_temp.size(); ++i) {
		data_complex[i] = qcomplex_t(data_temp[i], 0);
	}
	return get_fidelity(data_complex);
}
double Encode::get_fidelity(const vector<double> &data) {

	vector<double>data_temp(data);

	//_normalized(data_temp);

	vector<qcomplex_t>data_complex(data_temp.size());

	for (int i = 0; i < data_temp.size(); ++i) {
		data_complex[i] = qcomplex_t(data_temp[i], 0);
	}
	return get_fidelity(data_complex);
}
double Encode::get_fidelity(const vector<qcomplex_t> &data) {
	auto machine = CPUQVM();
	machine.init();
	QProg prog = QProg();
	vector<qcomplex_t>data_temp(data);
	prog << m_qcircuit;
	machine.directlyRun(prog);
	auto encode_stat = machine.getQStat();
	Eigen::VectorXcd v2 = Eigen::Map<Eigen::VectorXcd>(data_temp.data(), data_temp.size());
	Eigen::VectorXcd v1 = Eigen::Map<Eigen::VectorXcd>(encode_stat.data(), encode_stat.size());
	auto fidelity = v1.transpose().conjugate()*v2;
	auto res = fidelity.real()*fidelity.real() + fidelity.imag()*fidelity.imag();
	machine.finalize();
	return res(0, 0);
}
