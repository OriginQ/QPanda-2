#include"include/QAlg/Error_mitigation/Correction.h"
#include <chrono>
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include <Eigen/KroneckerProduct>
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
USING_QPANDA


Mitigation::Mitigation(const QVec& q) {
	m_sample = Sample(q);
	m_qubits = q;
	m_shots = 8192;
	m_qvm = new CPUQVM();
	m_qvm->init();

	m_miti_prob.resize(1 << q.size());
	m_noise = NoiseModel();
}
Mitigation::Mitigation(const QVec& q, QuantumMachine* qvm, NoiseModel& noise,size_t shots)
{
	m_sample = Sample(q);
	m_qubits = q;
	m_shots = shots;

	m_miti_prob.resize(1 << q.size());
	m_qvm = qvm;
	m_qvm->init();
	m_noise = noise;
}
Eigen::MatrixXd Mitigation::prob2density(const std::map<std::string, size_t>& res, const size_t& size)
{
	Eigen::VectorXd prob = Eigen::VectorXd::Zero(size);
	Eigen::MatrixXd density_mat = Eigen::MatrixXd::Zero(size,size);
	for (auto i : res) {
		size_t location = binary2ull(i.first);
		prob(location) = (double)(i.second) / (double)(m_shots);
	}
	for(int i=0;i<size;++i)
	{
		if (std::abs(prob(i) - 0) > 1e-8) {
			Eigen::VectorXd temp= Eigen::VectorXd::Zero(size);
			temp(i) = 1;
			density_mat += prob(i)*(temp.transpose() * temp);
		}
	}
	return density_mat;
}
double Mitigation::get_expection(Eigen::MatrixXd& A, Eigen::MatrixXd& Ham) 
{
	return (A * Ham).trace();
}
std::vector<Eigen::Matrix2d> Mitigation::calc_circ_balance(const int& shots)
{
	std::vector<Eigen::Matrix2d> calc_mats(m_qubits.size());
	std::vector<size_t> good_rep(m_qubits.size());
	auto c = m_qvm->cAllocMany(m_qubits.size());
	std::vector<QCircuit> circ = m_sample.full_sample();
	std::vector<std::string> cir_str = m_sample.get_cir_str();
	for (int i = 0; i < circ.size(); ++i) {
		QProg prog = QProg();
		prog << circ[i] << MeasureAll(m_qubits, c);
		auto res = m_qvm->runWithConfiguration(prog, c, shots, m_noise);
		std::string target = cir_str[i];
		double denom = shots * m_qubits.size();
		for(auto j:res){
			std::string s = j.first;
			size_t val = j.second;
			for (int k = 0; k < m_qubits.size(); ++k) {
				if (s[k] == target[k]) {
					good_rep[m_qubits.size() - k-1] += val;
				}
			}
		}
		for (int n = 0; n < m_qubits.size(); ++n) {
			if (target[m_qubits.size() - n - 1] == '0') {
				calc_mats[n](0, 0) += (double)good_rep[n] / denom;
			}
			else {
				calc_mats[n](1, 1) += (double)good_rep[n] / denom;
			}
		}
		for (int num = 0; num < m_qubits.size(); ++num) {
			calc_mats[num](0, 1) = 1.0 - calc_mats[num](0, 0);
			calc_mats[num](1, 0) = 1.0 - calc_mats[num](1, 1);
		}
	}
	return calc_mats;
}
std::vector<Eigen::Matrix2d> Mitigation::calc_circ_independ(const int& shots) 
{
	std::vector<Eigen::Matrix2d> calc_mats(m_qubits.size());
	std::vector<size_t> good_rep(m_qubits.size());
	auto c = m_qvm->cAllocMany(m_qubits.size());
	std::vector<QCircuit> circ = m_sample.independent_sample();
	std::vector<std::string> cir_str = m_sample.get_cir_str();
	for (int i = 0; i < circ.size(); i+=2) {
		QProg prog0 = QProg();
		QProg prog1 = QProg();
		prog0 << circ[i] << Measure(m_qubits[(int)(i/2)], c[0]);
		prog1 << circ[i+1] << Measure(m_qubits[(int)(i / 2)], c[0]);
		auto res0 = m_qvm->runWithConfiguration(prog0, c, shots, m_noise);
		auto res1 = m_qvm->runWithConfiguration(prog1, c, shots, m_noise);
		calc_mats[(int)(i / 2)](0, 0) = (double)res0["000"] / shots;
		calc_mats[(int)(i / 2)](0, 1) = 1- calc_mats[(int)(i / 2)](0, 0);
		calc_mats[(int)(i / 2)](1, 1) = (double)res1["001"] / shots;
		calc_mats[(int)(i / 2)](1, 0) = 1 - calc_mats[(int)(i / 2)](1, 1);
	}
	return calc_mats;
}

std::vector<Eigen::Matrix2d> Mitigation::calc_circ_bfa(const int& shots)
{
	std::vector<Eigen::Matrix2d> calc_mats(m_qubits.size());
	std::vector<size_t> good_rep(m_qubits.size());
	auto c = m_qvm->cAllocMany(m_qubits.size());
	std::vector<QCircuit> circ = m_sample.bit_flip_average_sample();
	std::vector<std::string> cir_str = m_sample.get_cir_str();
	std::vector<double> prob_vec(1 << m_qubits.size());
	for (int i = 0; i < circ.size(); ++i) {
		QProg prog = QProg();
		prog << circ[i] << MeasureAll(m_qubits, c);
		auto res = m_qvm->runWithConfiguration(prog, c, shots, m_noise);
		std::string target = cir_str[i];
		std::reverse(target.begin(), target.end());
		double denom = cir_str.size();
		for (auto j : res) {
			std::string s = j.first;
			std::reverse(s.begin(), s.end());
			size_t val = j.second;
			std::string key = "";
			int cnt = 0;
			for (char c : target) {
				if (c == '1') {
					if (s[cnt] == '0') {
						key += '1';
					}
					else {
						key += '0';
					}
				}
				else {
					key += s[cnt];
				}
				cnt++;
			}
			prob_vec[binary2ull(key)] += 1;
		}
		for (int n = 0; n < (1 << m_qubits.size()); ++n) {
			std::string str = ull2binary(n, m_qubits.size());
			int cnt = 0;
			for (char c : str) {
				if (c == '0') {
					good_rep[cnt] += prob_vec[n];
				}
				cnt++;
			}
		}
		for (int n = 0; n < m_qubits.size(); ++n) {
			if (target[m_qubits.size() - n - 1] == '0') {
				calc_mats[n](0, 0) += (double)good_rep[n] / denom;
			}
			else {
				calc_mats[n](1, 1) += (double)good_rep[n] / denom;
			}
		}
		for (int num = 0; num < m_qubits.size(); ++num) {
			calc_mats[num](0, 1) = 1.0 - calc_mats[num](0, 0);
			calc_mats[num](1, 0) = 1.0 - calc_mats[num](1, 1);
		}
	}
	return calc_mats;
}

void Mitigation::readout_error_mitigation(const sample_method& method, const std::vector<double>& m_unmiti_prob)
{
	Eigen::SparseMatrix<double> response_mat((1<<m_qubits.size()),(1 << m_qubits.size()));
	
	auto c=m_qvm->cAllocMany(m_qubits.size());
	switch (method)
	{
	case Full:
	{
		std::vector<QCircuit> circ = m_sample.full_sample();
		std::vector<std::string> cir_str = m_sample.get_cir_str();
		for (int i = 0; i < circ.size(); ++i) {
			QProg prog = QProg();
			prog << circ[i] << MeasureAll(m_qubits, c);
			auto res = m_qvm->runWithConfiguration(prog, c, m_shots, m_noise);
			for (int j = 0; j < response_mat.cols(); ++j) {
				double value = ((double)res[ull2binary(j, m_qubits.size())] / (double)m_shots);
				if (value - 0 > 1e-5) {
					size_t location = binary2ull(cir_str[i]);
					response_mat.insert(j, binary2ull(cir_str[i])) = value;
				}
				//response_mat(j, binary2ull(cir_str[i])) = (double)(res[ull2binary(j, response_mat.cols())]/m_shots);
			}
		}
		m_miti_prob = gmres_correct(response_mat, m_unmiti_prob);
		break;
	}
	case Independ:
	{
		std::vector<QCircuit> circ = m_sample.independent_sample();
		std::vector<Eigen::Matrix2d> mat=calc_circ_independ(m_shots);
		Eigen::MatrixXd m = Eigen::MatrixXd::Identity(1, 1);
		for (int i = mat.size()-1; i >=0 ; --i) {
			m = Eigen::kroneckerProduct(m, mat[i]).eval();
		}

		m_miti_prob = square_correct(m, m_unmiti_prob);
		break;

	}
	case Balance:
	{
		std::vector<QCircuit> circ = m_sample.balance_sample();
		std::vector<Eigen::Matrix2d> mat = calc_circ_balance(m_shots);
		Eigen::MatrixXd m = Eigen::MatrixXd::Identity(1, 1);
		for (int i = mat.size() - 1; i >= 0; --i) {
			m = Eigen::kroneckerProduct(m, mat[i]).eval();
		}

		m_miti_prob = square_correct(m, m_unmiti_prob);
		break;
	}
	case BFA:
	{
		std::vector<QCircuit> circ = m_sample.bit_flip_average_sample();
		std::vector<std::string> cir_str = m_sample.get_cir_str();
		for (int i = 0; i < circ.size(); ++i) {
			QProg prog = QProg();
			prog << circ[i] << MeasureAll(m_qubits, c);
			auto res = m_qvm->runWithConfiguration(prog, c, 1, m_noise);
			for (int j = 0; j < response_mat.cols(); ++j) {
				double value = ((double)res[ull2binary(j, m_qubits.size())] / (double)m_shots);
				if (value - 0 > 1e-5) {
					size_t location = binary2ull(cir_str[i]);
					response_mat.insert(j, binary2ull(cir_str[i])) = value;
				}
				//response_mat(j, binary2ull(cir_str[i])) = (double)(res[ull2binary(j, response_mat.cols())]/m_shots);
			}
		}
		m_miti_prob = gmres_correct(response_mat, m_unmiti_prob);
		break;
	}
	default:
		break;
	}
	return;
}

void Mitigation::zne_error_mitigation(QCircuit& circuit, const std::vector<size_t>& order) 
{
	std::vector<double> beta(order.size());
	std::vector<std::vector<double>> error_result(order.size());
	QProg prog_ = QProg();
	prog_ << circuit;
	auto proglayer=prog_layer(prog_);
	size_t layer = proglayer.size();
	std::cout << layer << std::endl;
	size_t helf_layer = std::ceil((double)layer / 2.0);
	for (int i = 1; i <= order.size(); ++i) {
		double num = 1;
		for (int j = 1; j <= order.size(); ++j) {
			if (j != i) {
				num *= (double)(order[j]) / (double)(order[j]-order[i]);
			}
		}
		beta[i - 1] = num;
	}
	for (int i = 1; i <= order.size(); ++i) {
		QProg prog = QProg();
		prog << circuit;
		//QCircuit cir = random_qcircuit(m_qubits, layer, { "X","Y","Z","CNOT","H","CZ","S" });
		//QCircuit helf_cir = random_qcircuit(m_qubits, helf_layer, { "X","Y","Z","CNOT","H","CZ","S" });
		if (i & 1) {

			for (int j = 1; j < i; j+=2) {
				prog << circuit << circuit.dagger();
			}
		}
		else {
			for (int j = 1; j < i-1; j += 2) {
				prog << circuit << circuit.dagger();
			}
			//prog << helf_cir << helf_cir.dagger();
			//prog << get_prog_from_layer(proglayer, helf_layer, layer - 1).dagger()<< get_prog_from_layer(proglayer, helf_layer, layer - 1);
			//if (layer & 1) {
			//	prog << I(m_qubits);
			//}
		}
		auto c = m_qvm->cAllocMany(m_qubits.size());

		prog << MeasureAll(m_qubits, c);
		//std::cout << prog << std::endl;
		auto res = m_qvm->runWithConfiguration(prog, m_shots,m_noise);


		error_result[i - 1].resize(1 << m_qubits.size());
		for (auto j : res) {
			size_t value = binary2ull(j.first);
			error_result[i - 1][value] = (double)j.second / (double)(m_shots);
		}
		//Eigen::MatrixXd A = prob2density(res, (1 << m_qubits.size()));
		//Eigen::MatrixXd Ham = hamiltonian;
		//error_result[i - 1] = get_expection(A, Ham);
	}

	//for (auto k : error_result) {
	//	for(auto j:k)
	//	std::cout << j << std::endl;
	//	std::cout << "end" << std::endl;
	//}
	
	for (int i = 0; i < (1 << m_qubits.size()); ++i) {
		for (int k = 0; k < order.size(); ++k) {
			m_miti_prob[i]+= beta[k] * error_result[k][i];
		}
	}
	Eigen::VectorXd prob_norm = Eigen::Map<Eigen::VectorXd>(m_miti_prob.data(),m_miti_prob.size());
	prob_norm.normalize();
	std::vector<double> prob_tmp(prob_norm.data(), prob_norm.data()+prob_norm.size());
	m_miti_prob = prob_tmp;
	for (int i = 0; i < m_miti_prob.size(); ++i) {
		m_miti_prob[i] = m_miti_prob[i] * m_miti_prob[i];
	}
	return;
}

std::vector<int> Mitigation::sample_circuit(const std::tuple<double, std::vector<std::pair<double,GateType>>>& representation, const int& num_samples,std::vector<QCircuit> &circuits,const QVec &q)
{

	double norm;
	std::vector<int> signs(num_samples,1);

	std::vector<GateType>type(num_samples*2);
	for (int i = 0; i < num_samples; ++i) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::default_random_engine generator(seed);

		std::normal_distribution<double> distribution(0.0, 1.0);

		double pre = distribution(generator);
		//double pre = random_generator19937();
		//std::cout <<"random number:" << pre << std::endl;
		//double prob = 0.0;
	
			//prob += j ==0 ? std::abs(std::get<1>(representation)[0].first): std::abs(std::get<1>(representation)[1].first);
			if (pre < std::abs(std::get<1>(representation)[0].first)) {
				/*type[i] = std::get<1>(representation)[j% std::get<1>(representation).size()].second;*/
				break;
			}
			else
			{
				std::random_device rd;  //Will be used to obtain a seed for the random number engine
				std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
				std::uniform_int_distribution<> distrib(1, 15);
				int j = distrib(gen);
				//std::cout << j << std::endl;
				switch (j)
				{
				case 0:
				{
					break;
				}
				case 1:
				{
					circuits[i] <<X(q[1]);
					signs[i] = -1;
					break;
				}
				case 2:
				{
					circuits[i]  << Y(q[1]);
					signs[i] = -1;
					break;
				}
				case 3:
				{
					circuits[i] << Z(q[1]);
					signs[i] = -1;
					break;
				}
				case 4:
				{
					circuits[i] << X(q[0]);
					signs[i] = -1;
					break;
				}
				case 5:
				{
					circuits[i] << X(q[0]) << X(q[1]);

					break;
				}
				case 6:
				{
					circuits[i] << X(q[0]) << Y(q[1]);

					break;
				}
				case 7:
				{
					circuits[i] << X(q[0]) << Z(q[1]);

					break;
				}
				case 8:
				{
					circuits[i] << Y(q[0]);
					signs[i] = -1;
					break;
				}
				case 9:
				{
					circuits[i] << Y(q[0]) << X(q[1]);

					break;
				}
				case 10:
				{
					circuits[i] << Y(q[0]) << Y(q[1]);

					break;
				}
				case 11:
				{
					circuits[i] << Y(q[0]) << Z(q[1]);

					break;
				}
				case 12:
				{
					circuits[i] << Z(q[0]);
					signs[i] = -1;
					break;
				}
				case 13:
				{
					circuits[i] << Z(q[0]) << X(q[1]);

					break;
				}
				case 14:
				{
					circuits[i] << Z(q[0]) << Y(q[1]);

					break;
				}
				case 15:
				{
					circuits[i] << Z(q[0]) << Z(q[1]);

					break;

				}

				default:
					break;
				}	
				break;
			}
		
	}
	return signs;
}
void Mitigation::quasi_probability(QCircuit& circuit, const std::tuple<double, std::vector<std::pair<double, GateType>>>&representation,const int &num_samples) {

	flatten(circuit);
	std::vector<QCircuit> circuits(num_samples);
	double norm = 1.0;
	
	std::vector<int> signs(num_samples,1);

	for (auto gate_itr = circuit.getFirstNodeIter(); gate_itr != circuit.getEndNodeIter(); ++gate_itr)
	{
		auto gate_tmp = std::dynamic_pointer_cast<QNode>(*gate_itr);
		if ((*gate_tmp).getNodeType() != NodeType::GATE_NODE) {
			continue;
		}
		
		auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(gate_tmp);
		QVec used_qv;
		gate_node->getQuBitVector(used_qv);
		auto gate_type = gate_node->getQGate()->getGateType();
		for (int i = 0; i < num_samples; ++i) {
			circuits[i].insertQNode(circuits[i].getLastNodeIter(), gate_tmp);

		}
		if (gate_type != GateType::CNOT_GATE) {
			continue;
		}
		else {

			std::vector<int> result_seq=sample_circuit(representation, num_samples,circuits,used_qv);
			//std::tuple<std::vector<GateType>, std::vector<int>, double> result_seq2 = sample_circuit(representation, num_samples);
			norm *= std::get<0>(representation);

			for (int i = 0; i < num_samples; ++i) {
				
				signs[i] *= result_seq[i];
			}
		}
		
	}
	//std::cout << "signs:" << std::endl;
	//for (auto i : signs)
	//	std::cout << i << std::endl;
	auto c = m_qvm->cAllocMany(m_qubits.size());

	Eigen::VectorXd miti_prob = Eigen::VectorXd::Zero(1 << (m_qubits.size()));
	for (int i = 0; i < num_samples; ++i) {
		QProg prog;
		prog << circuits[i]<<MeasureAll(m_qubits,c);
		//std::cout << circuits[i];
		auto res = m_qvm->runWithConfiguration(prog, m_shots, m_noise);
		std::vector<double> unmiti_prob(1 << (m_qubits.size()));
		for (int i = 0; i < res.size(); ++i) {
			unmiti_prob[i] = res[ull2binary(i, m_qubits.size())]/(double)(m_shots);
		}
		//std::cout << signs[i] << std::endl;
		Eigen::VectorXd prob_norm = Eigen::Map<Eigen::VectorXd>(unmiti_prob.data(), unmiti_prob.size());
		prob_norm =prob_norm * signs[i]*norm;
		miti_prob += prob_norm;
	}
	//std:: cout << norm << std::endl;
	miti_prob /= num_samples;
	//miti_prob *= norm;
	//for (int i = 0; i < miti_prob.size(); ++i) {
	//	miti_prob(i) = std::abs(miti_prob(i));
	//	
	//}
	miti_prob.normalize();

	for (int i = 0; i < miti_prob.size(); ++i) {
		miti_prob(i) = miti_prob(i) * miti_prob(i);

	}

	std::vector<double> prob_tmp(miti_prob.data(), miti_prob.data() + miti_prob.size());
	m_miti_prob = prob_tmp;
	return;
}