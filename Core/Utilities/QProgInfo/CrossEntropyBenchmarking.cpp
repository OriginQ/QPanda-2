#include "Core/Utilities/QProgInfo/CrossEntropyBenchmarking.h"
#include <set>
#include <vector>
#include <map>
#include<random>
#include <bitset>
USING_QPANDA
using namespace std;
#define BIT_SIZE 32


const static std::vector<QStat>& _single_gates_params()
{
	static std::vector<QStat> gate_params;
	static bool init = false;
	if (!init)
	{
		QStat Xpow_ = { qcomplex_t(0.5, 0.5) , qcomplex_t(0.5,-0.5), qcomplex_t(0.5,-0.5),  qcomplex_t(0.5, 0.5) };
		QStat Ypow_ = { qcomplex_t(0.5, 0.5) , qcomplex_t(-0.5,-0.5), qcomplex_t(0.5,0.5),  qcomplex_t(0.5, 0.5) };
		QStat phasedXpow_ = { qcomplex_t(0.5, 0.5) , qcomplex_t(0, -1 / sqrt(2)), qcomplex_t(1 / sqrt(2),0),  qcomplex_t(0.5, 0.5) };
		gate_params = { Xpow_ ,Ypow_, phasedXpow_ };
		init = true;
	}
	return gate_params;
}



static std::vector<std::vector<int>> random_choice(int max, int layer, int lenght)
{
	std::mt19937 gen(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_int_distribution<> u(0, max);
	std::vector<std::vector<int>> result(layer, std::vector<int>(lenght));
	for(int i = 0 ;i < layer; i++)
	{
		for (int j = 0; j < lenght; j++)
			result[i][j] =  u(gen);
	}
	return result;
}

void CrossEntropyBenchmarking::_build_entangling_layers(GateType gt)
{
	typedef QGate(*gate_func)(Qubit*, Qubit*);
	switch (gt)
	{
	case CNOT_GATE:
		m_double_gate_func = (gate_func)CNOT;
		break;
	case CZ_GATE:
		m_double_gate_func = (gate_func)CZ;
		break;
	case ISWAP_GATE:
		m_double_gate_func = (gate_func)iSWAP;
		break;
	case SQISWAP_GATE:
		m_double_gate_func = (gate_func)SqiSWAP;
		break;
	case SWAP_GATE:
		m_double_gate_func = (gate_func)SWAP;
		break;
	/*
	case CU_GATE:
		break;
	case CPHASE_GATE:
		break;
	case ISWAP_THETA_GATE:
		break;
	case TWO_QUBIT_GATE:
		break;
	*/
	default:
		QCERR("Unsupported gate type ! ");
		throw invalid_argument("Unsupported gate type ! ");
		break;
	}
}

void CrossEntropyBenchmarking::_random_half_rotations(int num_layers)
{
	m_mea_single_rots.clear();
	m_exp_single_rots.clear();
	m_mea_single_rots.resize(num_layers);
	m_exp_single_rots.resize(num_layers);

	auto rot_ops = _single_gates_params();
	auto rd_vect = random_choice(2, m_mea_qubits.size(), num_layers);
	for (int i = 0; i < num_layers; i++)
	{
		for (int j = 0; j < m_mea_qubits.size(); j++)
		{
			m_mea_single_rots[i]<< U4(rot_ops[rd_vect[j][i]], m_mea_qubits[j]);
			m_exp_single_rots[i] << U4(rot_ops[rd_vect[j][i]], m_exp_qubits[j]);
		}
	}
}

void CrossEntropyBenchmarking::_build_xeb_circuits(std::vector<QProg>& exp_prog, std::vector<QProg>& mea_prog)
{	
	int max_cycles = *max_element(m_cycle_range.begin(), m_cycle_range.end());

	 _random_half_rotations(max_cycles);

	for (auto num_cycles : m_cycle_range)
	{
		QProg exp_circuit;
		QProg mea_circuit;

		for (int i = 0; i < num_cycles; i++)
		{
			exp_circuit << m_exp_single_rots[i];
			mea_circuit << m_mea_single_rots[i];
			exp_circuit << m_double_gate_func(m_exp_qubits[0], m_exp_qubits[1]);
			mea_circuit << m_double_gate_func(m_mea_qubits[0], m_mea_qubits[1]);
		}
		for (int i = 0; i < m_mea_qubits.size(); i++)
		{
			mea_circuit << Measure(m_mea_qubits[i], m_mea_cc[i]);
		}

		exp_prog.push_back(exp_circuit);
		mea_prog.push_back(mea_circuit);
	}
}

CrossEntropyBenchmarking::CrossEntropyBenchmarking(MeasureQVMType type, QuantumMachine* qvm)
{
	m_mea_qvm_type = type;
	if (m_mea_qvm_type == MeasureQVMType::WU_YUAN)
		m_cloud_qvm = dynamic_cast<QCloudMachine *>(qvm);
	else 
		m_mea_qvm = dynamic_cast<NoiseQVM*>(qvm);

	m_exp_qvm = new CPUQVM();
	m_exp_qvm->init();
}

CrossEntropyBenchmarking::~CrossEntropyBenchmarking()
{
	m_exp_qvm->finalize();
	delete m_exp_qvm;
}
std::map<int, double> CrossEntropyBenchmarking::calculate_xeb_fidelity(GateType gt, Qubit * qbit0, Qubit  *qbit1, const std::vector<int>& cycle_range, int num_circuits, int shots)
{
	m_cycle_range = cycle_range;
	m_num_circuits = num_circuits;
	m_exp_qubits = m_exp_qvm->qAllocMany(2);
	if (m_mea_qvm_type == MeasureQVMType::WU_YUAN)
	{
		m_mea_qubits = { qbit0 , qbit1 };
		m_mea_cc = m_cloud_qvm->cAllocMany(m_mea_qubits.size());
	}
	else
	{
		m_mea_qubits = { qbit0 , qbit1 };
		m_mea_cc = m_mea_qvm->cAllocMany(m_mea_qubits.size());
	}

	ProbsDict probs_exp(m_cycle_range.size(), vector<vector<double>>(m_num_circuits));
	ProbsDict probs_meas(m_cycle_range.size(), vector<vector<double>>(m_num_circuits));
	
	_build_entangling_layers(gt);
	for(int k= 0; k < m_num_circuits; k ++)
	{
		std::vector<QProg> exp_all_progs, mea_all_progs;
		_build_xeb_circuits(exp_all_progs, mea_all_progs);
		for (int i = 0; i < m_cycle_range.size(); i++)
		{
			m_exp_qvm->directlyRun(exp_all_progs[i]);
			auto  exp_probs = m_exp_qvm->PMeasure_no_index(m_exp_qubits);
			probs_exp[i][k] = exp_probs;
		}

		for (int i = 0; i < m_cycle_range.size(); i++)
		{
			vector<double> mea_probs(pow(m_mea_qubits.size(), 2), 0.0);
			if (m_mea_qvm_type ==MeasureQVMType::WU_YUAN)
			{
				auto mea_result = m_cloud_qvm->real_chip_measure(mea_all_progs[i], shots);
				for (auto val : mea_result)
				{
					bitset<BIT_SIZE> temp_bit(val.first);
					int idx = temp_bit.to_ulong();
					mea_probs[idx] = val.second;
				}
			}
			else 
			{
				auto  mea_result = m_mea_qvm->runWithConfiguration(mea_all_progs[i], m_mea_cc, shots);
				 for (auto val : mea_result)
				 {
					 bitset<BIT_SIZE> temp_bit(val.first);
					 int idx = temp_bit.to_ulong();
					 mea_probs[idx] = (double)val.second / (double)shots;
				 }
			}
			probs_meas[i][k] = mea_probs ;
		}
	}

	return  _xeb_fidelities(probs_exp, probs_meas);
}

std::map<int, double> CrossEntropyBenchmarking::_xeb_fidelities(const ProbsDict& ideal_probs,
	const ProbsDict& actual_probs)
{
	int num_states = pow(m_mea_qubits.size(), 2);

	//std::vector<float> xeb_result;
	std::map<int, double> xeb_result;

	for (int i = 0; i < m_cycle_range.size(); i++)
	{
		float pp_cross = 0.0, pp_exp = 0.0, f_meas = 0.0, f_exp = 0.0;
		for (int j = 0; j < m_num_circuits; j++)
		{
			for (int k = 0; k < num_states; k++)
			{
				pp_cross += ideal_probs[i][j][k] * actual_probs[i][j][k];
				pp_exp += ideal_probs[i][j][k] * ideal_probs[i][j][k];
			}
			f_meas+= pp_cross * num_states - 1.0;
			f_exp += pp_exp * num_states - 1.0;
		}
		float xeb = (f_meas / (float)m_num_circuits) / (f_exp / (float)m_num_circuits);
		xeb_result[m_cycle_range[i]] = xeb;
	}
	return xeb_result;
}

std::map<int, double> QPanda::double_gate_xeb(NoiseQVM* qvm, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& range, int num_circuits, int shots, GateType gt)
{
	CrossEntropyBenchmarking cb(MeasureQVMType::NOISE, qvm);
	return cb.calculate_xeb_fidelity(gt, qbit0, qbit1, range, num_circuits, shots);
}

std::map<int, double> QPanda::double_gate_xeb(QCloudMachine* qvm, Qubit* qbit0, Qubit* qbit1, const std::vector<int>& range, int num_circuits, int shots, GateType gt )
{
	CrossEntropyBenchmarking cb(MeasureQVMType::WU_YUAN, qvm);
	return cb.calculate_xeb_fidelity(gt, qbit0, qbit1, range, num_circuits, shots);
}
