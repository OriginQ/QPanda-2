#include "QAlg/Base_QCircuit/AmplitudeEncode.h"
#include "QPanda.h"
#include <bitset>
#include<algorithm>
QPANDA_BEGIN
using namespace std;

StateNode::StateNode(int out_index, int out_level, double out_amplitude, StateNode* out_left, StateNode* out_right):index(out_index),level(out_level),amplitude(out_amplitude)
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
NodeAngleTree::NodeAngleTree(int out_index, int out_level, int out_qubit_index, double out_angle, NodeAngleTree* out_left, NodeAngleTree* out_right):index(out_index), level(out_level), qubit_index(out_qubit_index),angle(out_angle)
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
	m_data_std = 1.0;

};

void Encode::normalized(std::vector<double>& data) 
{
	double tmp_sum = 0.0;



	for (const auto i : data)
	{
	
		tmp_sum += (i*i);
	}

	const double max_precision = 1e-13;

	if (abs(1.0 - tmp_sum) > max_precision) {

		m_data_std = tmp_sum;

		m_data_std = sqrt(m_data_std);

		Map<VectorXd> v1(data.data(), data.size());
		VectorXd v2 = Map<VectorXd>(data.data(), data.size());
		v2.normalize();
		vector<double> vec(v2.data(), v2.data() + v2.size());
		data = vec;
		vec.clear();
	}

	return;
}

void Encode::basic_encode(QVec &q, const std::string& data) {

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

void Encode::amplitude_encode_recursive(QVec &q, const std::vector<double>& data)
{
	vector<double>data_temp(data);

	normalized(data_temp);

	if (data.size() > (1 << q.size()))
	{
		throw run_fail("Amplitude_encode parameter error.");
	}

	while (data_temp.size() < (1 << q.size()))
	{
		data_temp.push_back(0);
	}
	m_qcircuit=_recursive_compute_beta(q, data_temp);

	m_out_qubits = q;

	return;
}

void Encode::amplitude_encode_recursive(QVec qubits, const QStat& full_cur_vec)
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
		tmp_sum += tmp_m * tmp_m;
		tatal += tmp_m;
		ui_mod[i] = sqrt(tmp_m);
		ui_angle[i] = arg(full_cur_vec[i]);
	}

	if (abs(1.0 - tmp_sum) > max_precision)
	{
		if (abs(tmp_sum) < max_precision)
		{
			QCERR("Error: The input vector b is zero.");
			return;
		}

		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the normalization condition.");
	}
	QStat mat_d(_dimension * _dimension, qcomplex_t(0, 0));
	for (size_t i = 0; i < _dimension; ++i)
	{
		mat_d[i + i * _dimension] = exp(qcomplex_t(0, ui_angle[i]));
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

void Encode::amplitude_encode(QVec &q, const std::vector<double>& data)
{

	vector<double>data_temp(data);

	normalized(data_temp);

	if (data_temp.size() > (1 << q.size()))
	{
		throw run_fail("Amplitude_encode parameter error.");
	}

	QVec qubits;
	int k = 0;

	for (auto i : q) {
		if (k >= ceil(log2(data.size()))) break;
		qubits.push_back(i);
		++k;
	}

	while (data_temp.size() < (1 << qubits.size()))
	{
		data_temp.push_back(0);
	}

	std::vector<std::vector<double>>betas(qubits.size());
	std::vector<double>input(data_temp);
	_recursive_compute_beta(data_temp, betas,(int)(qubits.size()-1));
	_generate_circuit(betas, qubits);

	for (int i = 0; i < ceil(log2(data.size())); ++i)
	{
		m_out_qubits.push_back(q[i]);
	}

	return;
}
void Encode::amplitude_encode(QVec &q, const std::vector<complex<double>>& data)
{
	vector<complex<double>> data_temp(data);
	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.real()*i.real())+(i.imag()*i.imag());
	}

	if (abs(1.0 - tmp_sum) > max_precision)
	{
		if (abs(tmp_sum) < max_precision)
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
		if (k >= ceil(log2(data.size()))) break;
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
		ui_mod[i] = sqrt(tmp_m);
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
void Encode::_generate_circuit(std::vector<std::vector<double>> &betas, QVec &quantum_input) {
	int numberof_controls = 0;
	int size = quantum_input.size();
	QVec control_bits;

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
				_index(angles.size() - 1-k,control_bits, numberof_controls);
				m_qcircuit << RY(quantum_input[size - 1 - numberof_controls], angles[k]).control(control_bits);
				_index(angles.size() - 1-k, control_bits, numberof_controls);
			}
			control_bits.push_back(quantum_input[size - 1 - numberof_controls]);
			numberof_controls += 1;
		}
	}
	return;
}

void Encode::angle_encode(QVec &q,const std::vector<double>& data, const GateType& gate_type)
{
	for (const auto i : data)
	{
			if (i < 0 || i > PI)
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the condition of [0,PI].");
			}
	}

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

void Encode::_recursive_compute_beta(const std::vector<double>&input_vector, std::vector<std::vector<double>>&betas,int count) {
	if (input_vector.size() > 1)
	{
		size_t size = input_vector.size() / 2,cnt=0;
		std::vector<double>new_x(size);
		std::vector<double>beta(size);

		for (auto k = 0; k < input_vector.size(); k += 2) {
			double norm = sqrt(input_vector[k] * input_vector[k] + input_vector[k + 1] * input_vector[k + 1]);
			new_x[cnt]=norm;
			if (norm == 0) {
				beta[cnt]=0;
			}
			else if (input_vector[k] < 0) {
				beta[cnt]=2 * PI - 2 * asin(input_vector[k + 1] / norm);
			}
			else
			{
				beta[cnt]=2 * asin(input_vector[k + 1] / norm);
			}
			++cnt;
		}

		_recursive_compute_beta(new_x, betas,count-1);
		betas[count]=beta;
	}
	return;
}

void Encode::_dc_generate_circuit(std::vector<std::vector<double>>& betas, QVec& quantum_input, const int cnt) {
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

	while (next_index -1 > 0) {
		m_out_qubits.push_back(quantum_input[(next_index >> 1) - 1]);
		next_index = (int)((next_index >> 1));
	}

	return;
}

void Encode::dc_amplitude_encode(QVec &q, const std::vector<double>& data)
{
	vector<double>data_temp(data);

	normalized(data_temp);

	int size = data_temp.size();
	int log_ceil_size = ceil(log2(size));
	if (1<<log_ceil_size > q.size()+1) {
		throw run_fail("Dc_Amplitude_encode parameter error.");
	}
	while (data_temp.size() < (1<<log_ceil_size))
	{
		data_temp.push_back(0);
	}

	std::vector<std::vector<double>>betas(log2(data_temp.size()));
	std::vector<double>input(data_temp);
	_recursive_compute_beta(data_temp, betas, (int)(log2(data_temp.size())-1));
	_dc_generate_circuit(betas, q, (int)data_temp.size());

	return;
}

void Encode::_index(const int value, QVec control_qubits, const int numberof_controls) {
	bitset<32> temp(value);
	std::string str = temp.to_string();

	for (int i = 32-numberof_controls,k=0; i < 32; ++i,++k) 
	{
		if (str[i] == '1') 
		{
			m_qcircuit << X(control_qubits[k]);
		}
	}
}

void Encode::dense_angle_encode(QVec &q, const std::vector<double>& data) 
{
	for (const auto i : data)
	{
			if (i < 0 || i > PI)
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input vector b must satisfy the condition of [0,PI].");
			}
	}

	if (data.size() > q.size()*2)
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

	for (auto i = 0; i < data_temp.size()/2; ++i)
	{
		m_qcircuit << U3(q[i], data_temp[i], data_temp[k],0.0);
		++k;
	}

	for (int i = 0; i < data_temp.size() / 2; ++i) {
		m_out_qubits.push_back(q[i]);
	}

	return;
}
void Encode::bid_amplitude_encode(QVec &q,const std::vector<double>& data, const int split) {

	vector<double>data_temp(data);

	normalized(data_temp);

	int n_qubits = ceil(log2(data_temp.size()));
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
	int size_next = 1<<n_qubits;

	if ((split_temp + 1)*(size_next / (1<<split_temp)) - 1 > q.size()) {
		throw run_fail("Bid_Amplitude_encode parameter error.");
	}

	if (split_temp > ceil(log2(data_temp.size()))) {
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
void Encode::_output(NodeAngleTree* angle_tree, QVec q) {
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
			double mag = sqrt(nodes[k]->amplitude* nodes[k]->amplitude + nodes[k+1]->amplitude*nodes[k+1]->amplitude);
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
		angle = 2 * asin(Amp);
	}

	NodeAngleTree *node = new NodeAngleTree(state_tree->index, state_tree->level,0,angle, nullptr, nullptr);

	if (state_tree->right->left&&state_tree->right->right) {
		node->right = _create_angles_tree(state_tree->right);
		node->left = _create_angles_tree(state_tree->left);
	}

	return node;
}
void Encode::_add_registers(NodeAngleTree* angle_tree,std::queue<int>&q, int start_level) 
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
void Encode::_top_down_tree_walk(NodeAngleTree* angle_tree, QVec q,int start_level, std::vector<NodeAngleTree*>control_nodes, std::vector<NodeAngleTree*>target_nodes) {
	if (angle_tree) {
		if (angle_tree->level < start_level) {
			_top_down_tree_walk(angle_tree->left, q,start_level);
			_top_down_tree_walk(angle_tree->right, q,start_level);
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
void Encode::_apply_cswaps(NodeAngleTree* angle_tree, QVec q) {

	if (angle_tree->angle != 0.0) {

		auto left = angle_tree->left;
		auto right = angle_tree->right;

		while (left&&right) {
			m_qcircuit << SWAP(q[left->qubit_index], q[right->qubit_index]).control(q[angle_tree->qubit_index]);
			left = left->left;
			if (right->left) {
				right=right->left;
			}
			else 
			{
				right=right->right;
			}
		}
	}
}
void Encode::_bottom_up_tree_walk(NodeAngleTree* angle_tree,QVec q, int start_level) {
	if (angle_tree && angle_tree->level < start_level) {

		m_qcircuit<<RY(q[angle_tree->qubit_index], angle_tree->angle);

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
	std:; vector<NodeAngleTree*>nodes;
	nodes.push_back(angle_tree);

	while (nodes.size() > 0) {
		level_nodes.push_back(nodes.size());
		nodes=_children(nodes);
		level += 1;
	}

	int noutput = level;
	int nqubits = 0;

	for (int i = 0; i < start_level;++i) {
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

void Encode::iqp_encode(QVec &q, const vector<double>& data, const std::vector<pair<int, int>>& control_vector, const bool& inverse, const int& repeats) 
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
	QCircuit cir_tmp=QCircuit();

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

void Encode::ds_quantum_state_preparation(QVec &q, const std::map<std::string, double>& data) {
	const double max_precision = 1e-13;
	double tmp_sum = 0.0;

	for (const auto i : data)
	{
		tmp_sum += (i.second*i.second);
	}
	if (abs(1.0 - tmp_sum) > max_precision)
	{
		if (abs(tmp_sum) < max_precision)
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

	for (auto i : data) 
	{
		string binary_string = i.first;
		double feature = i.second;
		vector<int>control = _select_controls(binary_string);
		_flip_flop(q,control,numqubits);
		_load_superposition(q,control,numqubits,feature,norm);
		if (k < data.size() - 1) {
			_flip_flop(q,control, numqubits);
		}
		else {
			break;
		}
		k++;
	}

	for (int i = numqubits; i < 2 * numqubits; ++i) {
		m_out_qubits.push_back(q[i]);
	}
}
void Encode::ds_quantum_state_preparation(QVec &q, const std::map<std::string, std::complex<double>>& data) {
	if (data.empty()) 
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: The input map data must not null.");
	}

	int size = (*data.begin()).first.size();

	for (auto i : data) 
	{
		if(i.first.size()!=size)
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
		tmp_sum += (i.second.real()*i.second.real())+(i.second.imag()*i.second.imag());
	}

	if (abs(1.0 - tmp_sum) > max_precision)
	{
		if (abs(tmp_sum) < max_precision)
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

	for (auto i : data)
	{
		string binary_string = i.first;
		complex<double> feature = i.second;
		vector<int>control = _select_controls(binary_string);
		_flip_flop(q, control, numqubits);
		_load_superposition(q, control, numqubits, feature, norm);
		if (k < data.size() - 1) {
			_flip_flop(q, control, numqubits);
		}
		else {
			break;
		}
		k++;
	}

	for (int i = numqubits; i < 2 * numqubits; ++i) {
		m_out_qubits.push_back(q[i]);
	}
}
void Encode::_flip_flop(QVec& q,std::vector<int>control,int numqbits) 
{
	for (int i : control) 
	{
		m_qcircuit << CNOT(q[0], q[i+numqbits]);
	}
}

template<typename T>
void Encode::_load_superposition(QVec& q, std::vector<int>control, int numqbits, T feature, double& norm)
{
	vector<double>angle = _compute_matrix_angles(feature, norm);

	if (control.size() == 0)
	{
		m_qcircuit << U3(q[0], angle[0],angle[1],angle[2]);
	}
	else if (control.size() == 1)
	{
		m_qcircuit << U3(q[0], angle[0], angle[1], angle[2]).control(q[control[0] + numqbits]);
	}
	else
	{
		_mcuvchain(q, control, angle, numqbits);
	}

	norm = norm - abs(feature*feature);
	return;
}
std::vector<int> Encode::_select_controls(string binary_string) 
{
	vector<int>control_qubits;

	for (int i = binary_string.size()-1; i>=0; --i)
	{
		if (binary_string[i] == '1') 
		{
			control_qubits.push_back(binary_string.size()-i-1);
		}
	}

	return control_qubits;
}
void Encode::_mcuvchain(QVec &q,std::vector<int>control, std::vector<double> angle,int numqbits)
{
	vector<int>reverse_control(control);
	reverse(reverse_control.begin(), reverse_control.end());
	m_qcircuit << X(q[numqbits - 1]).control({q[reverse_control[0] + numqbits], q[reverse_control[1] + numqbits]});
	std::vector<std::vector<int>>tof;
	tof.resize(numqbits);
	int k = numqbits;

	for (int i = 2; i < reverse_control.size(); ++i) 
	{
		m_qcircuit << X(q[k - 2]).control({q[reverse_control[i] + numqbits], q[k-1]});
		tof[reverse_control[i]].push_back(k-1);
		tof[reverse_control[i]].push_back(k - 2);
		k -= 1;
	}

	m_qcircuit << U3(q[0], angle[0], angle[1], angle[2]).control(q[k - 1]);

	for (int i = control.size()-3; i >=0; i-=2)
	{
		m_qcircuit << X(q[tof[control[i]][1]]).control({ q[control[i] + numqbits], q[tof[control[i]][0]] });
	}

	m_qcircuit << X(q[numqbits-1]).control({ q[control[control.size() - 1]+numqbits], q[control[control.size() - 2]+numqbits] });

	return;
}
std::vector<double> Encode::_compute_matrix_angles(std::complex<double> feature, double norm)
{
	double alpha = 0.0, beta = 0.0, phi = 0.0;
	double phase = abs(feature*feature);
	double cos_value = sqrt((norm - phase) / norm);
	double value = min(cos_value, 1.0);

	if (value < -1)
	{
		value = -1;
	}

	cos_value = value;
	alpha = 2 * (acos(cos_value));
	beta = acos(-feature.real() / sqrt(abs(feature*feature)));

	if (feature.imag() < 0) {
		beta = 2 * PI - beta;
	}

	phi = -beta;
	
	return { alpha,beta,phi };
}
std::vector<double> Encode::_compute_matrix_angles(double feature, double norm)
{
	double alpha = 0.0, beta = 0.0, phi = 0.0;
	double sin_value = -feature / sqrt(norm);
	double value = min(sin_value, 1.0);

	if (value < -1)
	{
		value = -1;
	}

	sin_value = value;
	alpha = 2 * (asin(sin_value));

	return { alpha,beta,phi };
}
void Encode::sparse_isometry(QVec &q, const std::map<std::string, double>& data) 
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

	if (abs(1.0 - tmp_sum) > max_precision)
	{
		if (abs(tmp_sum) < max_precision)
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
		std::string index_zero = _get_index_zero(next_state,n_qubits, non_zero);
		std::map<std::string, double> next = _pivoting(qcir,qubits_reverse,index_zero, index_nonzero, target_size, next_state);
		next_state = next;
		index_nonzero = _get_index_nz(next_state, n_qubits - target_size);
	}

	std::vector<double>dense_state;

	for (auto i : next_state) {
		dense_state.push_back(i.second);
	}

	QCircuit cir;
	cir =qcir.dagger();

	if (non_zero <= 2) {
		m_qcircuit << RY(qubits[0], 2 * acos(dense_state[0]));
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
		for (int i = 0; i < target_size; ++i) 
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

void Encode::sparse_isometry(QVec &q, const std::map<std::string, complex<double>>& data)
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

	if (abs(1.0 - tmp_sum) > max_precision)
	{
		if (abs(tmp_sum) < max_precision)
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

	std::vector<complex<double>>dense_state;
	for (auto i : next_state) {
		dense_state.push_back(i.second);
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
	size_t count = 1<<non_zero;

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
		if(!flag)
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
		std::string n_index,str;
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
				flag= false;
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
			for (int i = index_differ+1; i < n_index.size(); ++i) 
			{
				str.push_back(n_index[i]);
			}
		}

		new_state[str.empty() ? str = n_index : str] = state[i.first];

	}

	return new_state;
}

template<typename T>
std::map<std::string, T>  Encode::_pivoting(QCircuit& qcir, QVec &qubits, std::string index_zero, std::string index_nonzero, int target_size, std::map<std::string, T> state)
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

void Encode::schmidt_encode(QVec &q, const std::vector<double>& data)
{
	vector<double>data_temp(data);

	normalized(data_temp);

	if (data_temp.size() > (1 << q.size()))
	{
		throw run_fail("Schmidt_encode parameter error.");
	}

	QVec qubits;
	int k = 0;

	for (auto i : q) {
		if (k >= ceil(log2(data.size()))) break;
		qubits.push_back(i);
		++k;
	}

	while (data_temp.size() < (1 << qubits.size()))
	{
		data_temp.push_back(0);
	}

	int size = data_temp.size();
	int n_qubits = log2(size);
	int	r = n_qubits % 2;
	int row = 1<<(n_qubits>>1);
	int col = 1 << ((n_qubits >> 1)+r);
	EigenMatrixXc eigen_matrix = EigenMatrixXc::Zero(row, col);
	k = 0;

	for (auto rdx = 0; rdx < row; ++rdx)
	{
		for (auto cdx = 0; cdx < col; ++cdx)
		{
			eigen_matrix(rdx, cdx) = data[k++];
		}
	}

	JacobiSVD<EigenMatrixXc> svd(eigen_matrix, ComputeFullU | ComputeFullV);
	EigenMatrixXc V = svd.matrixV(), U = svd.matrixU();
	auto A = svd.singularValues();
	std::vector<std::vector<double>>U_vec, V_vec;
	std::vector<double>A_vec;
	size_t rows = U.rows();
	size_t cols = U.cols();

	for (size_t i = 0; i < rows; ++i)
	{
		A_vec.push_back((double)(A(i)));
	}

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

	if (A_vec.size() > 2)
	{
		schmidt_encode(B_qubits, A_vec);
	}
	else
	{
		if (A_vec[0] < 0) {
			m_qcircuit << RY(B_qubits, 2 * PI - 2 * acos(A_vec[0]));
		}
		else {
			m_qcircuit << RY(B_qubits, 2 * acos(A_vec[0]));
		}
	}

	for (int i = 0; i < floor(n_qubits / 2); ++i)
	{
		m_qcircuit << CNOT(B_qubits[i], A_qubits[i]);
	}

	_unitary(B_qubits, U);

	_unitary(A_qubits, V);

	for (auto i : A_qubits) 
	{
		m_out_qubits.push_back(i);
	}

	for (auto i : B_qubits)
	{
		m_out_qubits.push_back(i);
	}

	return;
}
void Encode::_unitary(QVec &q, EigenMatrixXc gate) {

	QCircuit circuit = matrix_decompose_qr(q, gate,true);

	m_qcircuit << circuit;

	return;
}
QCircuit Encode::get_circuit() {

	return m_qcircuit;
}
QVec Encode::get_out_qubits() {

	return m_out_qubits;
}
double Encode::get_normalization_constant() {

	return m_data_std;
}
QPANDA_END