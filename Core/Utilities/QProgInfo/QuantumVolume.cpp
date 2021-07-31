#include "Core/Utilities/QProgInfo/QuantumVolume.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgInfo/KAK.h"
#include "Core/Utilities/Compiler/QASMToQProg.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"
#include "ThirdParty/Eigen/Eigen"
#include <numeric>
#include <algorithm>





USING_QPANDA

using namespace std;
Eigen::Matrix4cd randomUnitary()
{
	// Using QR decomposition to generate a random unitary
	Eigen::Matrix4cd mat = Eigen::Matrix4cd::Random();
	auto QR = mat.householderQr();
	Eigen::Matrix4cd qMat = QR.householderQ() * Eigen::Matrix4cd::Identity();
	return qMat;
}

static void llong_to_bin_stream(std::stringstream& out_str, long long int v)
{
	long long int a = v % 2;
	v = v >> 1;
	if (v == 0)
	{
		out_str << a;
		return;
	}
	else
	{
		llong_to_bin_stream(out_str, v);
	}
	out_str << a;
}

QuantumVolume::QuantumVolume(MeasureQVMType type, QuantumMachine* qvm)
{
	m_qvm = new CPUQVM();
	m_qvm->init();
	m_qvm->setConfigure({ 300, 300 });
	m_qvm_type = type;
	if (m_qvm_type == MeasureQVMType::WU_YUAN)
		m_qcm = dynamic_cast<QCloudMachine*>(qvm);
	else
		m_noise_qvm = dynamic_cast<NoiseQVM*>(qvm);
}



QuantumVolume::~QuantumVolume()
{
	m_qvm->finalize();
	delete m_qvm;
}

//Generate uniformly random permutation Pj of [0...n-1]
std::vector<int> QuantumVolume::randomPerm(int depth)
{
	std::vector<int> perm;
	for (int j = 0; j < depth; j++)
	{
		perm.push_back(j);
	}
	//std::random_shuffle(perm.begin(), perm.end());  /** Cpp17 cant support */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::shuffle(perm.begin(), perm.end(), std::default_random_engine(seed));
	return perm;
}

void QuantumVolume::createQvCircuits(std::vector<std::vector<int> > qubit_lists, int ntrials,
	std::vector<std::vector <QvCircuit> >&circuits, std::vector<std::vector <QvCircuit> >&circuits_nomeas)
{
	m_qubit_lists = qubit_lists;
	for (auto qubits : qubit_lists)
		m_depth_list.push_back(qubits.size());

	m_ntrials = ntrials;
	circuits.resize(ntrials);
	circuits_nomeas.resize(ntrials);
	KAK kak;
	for (int trial = 0; trial < ntrials; trial++)
	{
		circuits[trial].resize(m_depth_list.size());
		circuits_nomeas[trial].resize(m_depth_list.size());
		for (int depthidx = 0; depthidx < m_depth_list.size(); depthidx++)
		{
			int depth = m_depth_list[depthidx];
			int q_max = *std::max_element(qubit_lists[depthidx].begin(), qubit_lists[depthidx].end());

			QVec qr, qr2;
			std::vector<ClassicalCondition> cr;
			for (int i = 0; i < q_max + 1; i++)
			{
				if (m_qvm_type == MeasureQVMType::WU_YUAN)
					qr.push_back(m_qcm->allocateQubitThroughPhyAddress(i));
				else
					qr.push_back(m_noise_qvm->allocateQubitThroughPhyAddress(i));

			}
			for (int i = 0; i < depth; i++)
			{
				qr2.push_back(m_qvm->allocateQubitThroughPhyAddress(i));

				if (m_qvm_type == MeasureQVMType::WU_YUAN)
					cr.push_back(m_qcm->cAlloc(i));
				else
					cr.push_back(m_noise_qvm->cAlloc(i));
			}

			auto qc = QCircuit();
			auto qc2 = QCircuit();
			QVec temp_qv;
			for (int j = 0; j < depth; j++)
			{
				std::vector<int> perm = randomPerm(depth);

				for (int k = 0; k < (int)floor(depth / 2); k++)
				{
					auto mat = randomUnitary();
					auto qubit_1 = qr[qubit_lists[depthidx][perm[2 * k]]];
					auto qubit_2 = qr[qubit_lists[depthidx][perm[2 * k + 1]]];

					auto kak_description = kak.decompose(mat);
					qc << kak_description.to_qcircuit(qubit_1, qubit_2);

					qubit_1 = qr2[perm[2 * k]];
					qubit_2 = qr2[perm[2 * k + 1]];
					qc2 << kak_description.to_qcircuit(qubit_1, qubit_2);
				}
			}

			circuits_nomeas[trial][depthidx].cir << qc2;
			circuits_nomeas[trial][depthidx].depth = depth;
			circuits_nomeas[trial][depthidx].trial = trial;
			circuits_nomeas[trial][depthidx].qv = qr2;

			QVec mea_qv;
			for (int i = 0; i < qubit_lists[depthidx].size(); i++)
				mea_qv.push_back(qr[qubit_lists[depthidx][i]]);
			circuits[trial][depthidx].cir << qc;
			circuits[trial][depthidx].depth = depth;
			circuits[trial][depthidx].trial = trial;
			circuits[trial][depthidx].cv = cr;
			circuits[trial][depthidx].qv = mea_qv;
		}
	}
}

void QuantumVolume::calcHeavyOutput(int trial, int depth, prob_vec probs)
{
	if ((probs.size() % 2) != 0)
	{
		QCERR("probs  size error !");
		throw invalid_argument("probs  size error !");
	}
	auto temp_probs = probs;
	std::sort(temp_probs.begin(), temp_probs.end());
	double median = (temp_probs[temp_probs.size() / 2] + temp_probs[temp_probs.size() / 2 - 1]) / 2;

	double heavy_output = 0;
	std::vector<std::string> heavy_string;
	for (int i = 0; i < probs.size(); i++)
	{
		if (probs[i] > median)
		{
			heavy_output += probs[i];

			std::stringstream temp_str;
			llong_to_bin_stream(temp_str, i);
			std::string stat_str = temp_str.str();
			while (depth > stat_str.length())
				stat_str = "0" + stat_str;

			heavy_string.push_back(stat_str);
		}
	}
	std::pair<int, int > key(depth, trial);

	m_heavy_outputs.insert(std::pair< std::pair<int, int >, std::vector<std::string > >(key, heavy_string));
	m_heavy_output_prob_ideal.insert(std::pair <std::pair<int, int >, double >(key, heavy_output));
}

void QuantumVolume::calcIdealResult(std::vector<std::vector <QvCircuit> >&circuits_nomeas)
{
	for (int trial = 0; trial < m_ntrials; trial++)
	{
		for (int depthidx = 0; depthidx < circuits_nomeas[trial].size(); depthidx++)
		{
			QProg prog;
			prog = circuits_nomeas[trial][depthidx].cir;
			QVec  qv = circuits_nomeas[trial][depthidx].qv;
			m_qvm->directlyRun(prog);
			prob_vec result = m_qvm->PMeasure_no_index(qv);
			circuits_nomeas[trial][depthidx].result = result;
			calcHeavyOutput(trial, m_depth_list[depthidx], result);
		}
	}
}

void QuantumVolume::calcINoiseResult(std::vector<std::vector <QvCircuit> >&circuits, int shots)
{
	m_shots = shots;
	for (int trial = 0; trial < m_ntrials; trial++)
	{
		for (int depthidx = 0; depthidx < circuits[trial].size(); depthidx++)
		{
			QProg prog;
			prog << circuits[trial][depthidx].cir;
			auto qv = circuits[trial][depthidx].qv;
			auto  cv = circuits[trial][depthidx].cv;
			QPANDA_ASSERT(qv.size() != cv.size(), "error!");
			for (size_t i = 0; i < qv.size(); i++)
				prog << Measure(qv[i], cv[i]);
			
			std::map<std::string, size_t> result;
			if (m_qvm_type == MeasureQVMType::WU_YUAN)
			{
				std::map<std::string, double> tmp_res = m_qcm->real_chip_measure(prog, m_shots);
				for (auto iter : tmp_res)
					result[iter.first] = (size_t)(m_shots * iter.second);
			}
			else
				 result = m_noise_qvm->runWithConfiguration(prog, cv, m_shots);

			/*	for (auto val : result)
					cout << val.first << " : " << val.second << endl;*/

			std::pair<int, int > key(m_depth_list[depthidx], trial);
			auto iter = m_heavy_outputs.find(key);
			if (iter == m_heavy_outputs.end())
			{
				QCERR("find m_heavy_outputs error !");
				throw invalid_argument("find m_heavy_outputs error !");
			}

			std::vector<std::string> stat_vec = iter->second;

			size_t counts = 0;
			for (int i = 0; i < stat_vec.size(); i++)
			{
				auto iter_res = result.find(stat_vec[i]);
				if (iter_res != result.end())
					counts += iter_res->second;
			}
			m_heavy_output_counts.insert(std::pair <std::pair<int, int >, size_t >(key, counts));
		}
	}
}

size_t QuantumVolume::volumeResult()
{
	size_t qv_size = 1;
	
	//for (int i =0; i< m_ydata.size(); i++)
	//{
	//	auto data = m_ydata[i];
	//	std::cout << "m_ydata[" << i << "] : ";
	//	for (auto val : data)
	//		std::cout << val << " ";
	//	
	//	std::cout << std::endl;
	//}

	for (int i = 0; i < m_success_list.size(); i++)
	{
		int depth = m_depth_list[i];
		std::pair<bool, float> cfd_result = m_success_list[i];
		//std::cout << "depth : " << depth << " cfd : "
		//	<< cfd_result.second << " result : " << cfd_result.first << std::endl;

		m_heavy_output_counts;
		m_heavy_output_prob_ideal;
		m_heavy_outputs;
		if (cfd_result.first == true)
			qv_size = pow(2, depth);
	}
	return qv_size;
}

void QuantumVolume::calcStatistics()
{
	std::vector<float> exp_vals;
	std::vector<float> ideal_vals;
	m_ydata.resize(4);
	for (int i = 0; i < 4; i++)
	{
		m_ydata[i].resize(m_depth_list.size());
	}
	exp_vals.resize(m_ntrials, 0);
	ideal_vals.resize(m_ntrials, 0);
	m_success_list.resize(m_depth_list.size());

	for (int depthidx = 0; depthidx < m_depth_list.size(); depthidx++)
	{
		int exp_shots = 0;
		for (int trialidx = 0; trialidx < m_ntrials; trialidx++)
		{
			std::pair<int, int > key(m_depth_list[depthidx], trialidx);
			auto iter_1 = m_heavy_output_counts.find(key);
			if (iter_1 == m_heavy_output_counts.end())
			{
				QCERR("find m_heavy_output_counts error !");
				throw invalid_argument("find m_heavy_output_counts error !");
			}
			exp_vals[trialidx] = iter_1->second;
			exp_shots += m_shots;

			auto iter_2 = m_heavy_output_prob_ideal.find(key);
			if (iter_2 == m_heavy_output_prob_ideal.end())
			{
				QCERR("find m_heavy_output_prob_ideal error !");
				throw invalid_argument("find m_heavy_output_prob_ideal error !");
			}

			ideal_vals[trialidx] = iter_2->second;
		}
		m_ydata[0][depthidx] = std::accumulate(exp_vals.begin(), exp_vals.end(), 0.0) / exp_shots;

		m_ydata[1][depthidx] = std::pow((m_ydata[0][depthidx] * (1.0 - m_ydata[0][depthidx]) / m_ntrials), 0.5);
		m_ydata[2][depthidx] = std::accumulate(ideal_vals.begin(), ideal_vals.end(), 0.0) / ideal_vals.size();
		m_ydata[3][depthidx] = std::pow((m_ydata[2][depthidx] * (1 - m_ydata[2][depthidx]) / m_ntrials), 0.5);

		float hmean = m_ydata[0][depthidx];
		bool cfd_succ = false;
		float cfd = 0;
		if (hmean > (2.0 / 3.0))
		{
			cfd = 0.5 * (1 + std::erf((hmean - 2.0 / 3.0) / (1e-10 + m_ydata[1][depthidx]) / std::pow(2.0, 0.5)));
			if (cfd > 0.975)
				cfd_succ = true;
		}
		std::pair<bool, float> cfd_result(cfd_succ, cfd);
		m_success_list[depthidx] = cfd_result;
	}
}

size_t QuantumVolume::calcQuantumVolume(const std::vector<std::vector<int>> &qubit_lists,
	int ntrials, int shots)
{
	vector<vector <QvCircuit >> cirs;
	vector<vector <QvCircuit >>cirs_nomea;
	createQvCircuits(qubit_lists, ntrials, cirs, cirs_nomea);
	calcIdealResult(cirs_nomea);
	calcINoiseResult(cirs, shots);
	calcStatistics();
	return volumeResult();
}

size_t QPanda::calculate_quantum_volume(NoiseQVM* qvm, std::vector <std::vector<int> >qubit_lists, int ntrials, int shots)
{
	QuantumVolume qv(MeasureQVMType::NOISE, qvm);
	return qv.calcQuantumVolume(qubit_lists, ntrials, shots);
}

size_t QPanda::calculate_quantum_volume(QCloudMachine* qvm, std::vector <std::vector<int> >qubit_lists, int ntrials, int shots)
{
	QuantumVolume qv(MeasureQVMType::WU_YUAN, qvm);
	return qv.calcQuantumVolume(qubit_lists, ntrials, shots);
}
