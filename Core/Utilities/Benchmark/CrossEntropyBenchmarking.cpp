#include <map>
#include <vector>
#include <random>
#include <bitset>
#include <memory>
#include "Core/Utilities/Benchmark/CrossEntropyBenchmarking.h"
#include "Core/VirtualQuantumProcessor/DensityMatrix/DensityMatrixSimulator.h"

USING_QPANDA

using namespace std;

#define BIT_SIZE 32

#if defined(USE_CURL)

const static std::vector<QStat>& rot_ops_params()
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

std::function<QGate(Qubit*,Qubit*)> CrossEntropyBenchmarking::get_benchmarking_gate(GateType gate_name)
{
	typedef QGate(*gate_function)(Qubit*, Qubit*);

	switch (gate_name)
	{
	case CNOT_GATE: return (gate_function)CNOT;
	case CZ_GATE: return (gate_function)CZ;
	case ISWAP_GATE: return (gate_function)iSWAP;
	case SQISWAP_GATE: return (gate_function)SqiSWAP;
	case SWAP_GATE: return (gate_function)SWAP;
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
		QCERR("Unsupported Benchmarking Gate");
		throw invalid_argument("Unsupported Benchmarking Gate");
		break;
	}
}

void CrossEntropyBenchmarking::random_half_rotations(int num_layers)
{
	m_mea_single_rots.clear();
	m_exp_single_rots.clear();

	m_mea_single_rots.resize(num_layers);
	m_exp_single_rots.resize(num_layers);

	auto rot_ops = rot_ops_params();
	auto rd_vect = random_choice(2, m_mea_qubits.size(), num_layers);

	for (int i = 0; i < num_layers; i++)
	{
		for (int j = 0; j < m_mea_qubits.size(); j++)
		{
			m_mea_single_rots[i] << U4(rot_ops[rd_vect[j][i]], m_mea_qubits[j]);
			m_exp_single_rots[i] << U4(rot_ops[rd_vect[j][i]], m_exp_qubits[j]);
		}
	}
}

void CrossEntropyBenchmarking::build_xeb_circuits(std::vector<QProg>& exp_prog, std::vector<QProg>& mea_prog, GateType gate_name)
{	
	int max_cycles = *max_element(m_cycle_range.begin(), m_cycle_range.end());

	random_half_rotations(max_cycles);

    auto double_gate = get_benchmarking_gate(gate_name);
	for (auto num_cycles : m_cycle_range)
	{
		QProg exp_circuit;
		QProg mea_circuit;

		for (int i = 0; i < num_cycles; i++)
		{
			exp_circuit << m_exp_single_rots[i];
			mea_circuit << m_mea_single_rots[i];

			exp_circuit << double_gate(m_exp_qubits[0], m_exp_qubits[1]);
			mea_circuit << double_gate(m_mea_qubits[0], m_mea_qubits[1]);
		}

		for (int i = 0; i < m_mea_qubits.size(); i++)
		{
			mea_circuit << Measure(m_mea_qubits[i], m_mea_cc[i]);
		}

		exp_prog.push_back(exp_circuit);
		mea_prog.push_back(mea_circuit);
	}
}

CrossEntropyBenchmarking::CrossEntropyBenchmarking(QuantumMachine* machine)
{
    auto qvm_ptr = dynamic_cast<QVM*>(machine);

    if (qvm_ptr == nullptr)
        QCERR_AND_THROW(run_fail, "QuantumMachine dynamic_cast error");

    m_machine_type = qvm_ptr->get_machine_type();
    m_machine_ptr = machine;
}

CrossEntropyBenchmarking::CrossEntropyBenchmarking(QCloudTaskConfig config)
{
    m_qcloud.setConfigure({ 72,72 });
    m_qcloud.init(config.cloud_token);

    m_cloud_config = config;

    m_machine_type = QMachineType::QCloud;
}


CrossEntropyBenchmarking::~CrossEntropyBenchmarking()
{}


std::map<int, double> CrossEntropyBenchmarking::_xeb_fidelities(
    const multi_probs& ideal_probs,
    const multi_probs& actual_probs)
{
    auto num_states = 1ull << m_mea_qubits.size();

    //std::vector<double> xeb_result;
    std::map<int, double> xeb_result;

    for (int i = 0; i < m_cycle_range.size(); i++)
    {
        double pp_cross = 0.0, pp_exp = 0.0, f_meas = 0.0, f_exp = 0.0;
        for (int j = 0; j < m_num_circuits; j++)
        {
            for (int k = 0; k < num_states; k++)
            {
                pp_cross += ideal_probs[i][j][k] * actual_probs[i][j][k];
                pp_exp += ideal_probs[i][j][k] * ideal_probs[i][j][k];
            }

            f_meas += pp_cross * num_states - 1.0;
            f_exp += pp_exp * num_states - 1.0;
        }

        double xeb = (double)((f_meas / num_states) / (f_exp / num_states));
        xeb_result[m_cycle_range[i]] = xeb;
    }
    return xeb_result;
}

std::map<int, double> CrossEntropyBenchmarking::calculate_xeb_fidelity(
    GateType gate_name, 
    Qubit * qubit_0, 
    Qubit  *qubit_1, 
    const std::vector<int>& cycle_range,
    int num_circuits, 
    int shots,
    RealChipType chip_type)
{
	m_cycle_range = cycle_range;
	m_num_circuits = num_circuits;

    //ideal probs
    DensityMatrixSimulator simulator;
    simulator.init();
    m_exp_qubits = simulator.qAllocMany(2);

    m_mea_qubits = { qubit_0 , qubit_1 };
    m_mea_cc = m_machine_ptr->cAllocMany(m_mea_qubits.size());

	multi_probs probs_exp(m_cycle_range.size(), vector<vector<double>>(m_num_circuits));
	multi_probs probs_meas(m_cycle_range.size(), vector<vector<double>>(m_num_circuits));

	for(int k= 0; k < m_num_circuits; k ++)
	{
		std::vector<QProg> exp_all_progs, mea_all_progs;

		build_xeb_circuits(exp_all_progs, mea_all_progs, gate_name);

		for (int i = 0; i < m_cycle_range.size(); i++)
		{
			auto exp_probs = simulator.get_probabilities(exp_all_progs[i],m_exp_qubits);
			probs_exp[i][k] = exp_probs;
		}

		for (int i = 0; i < m_cycle_range.size(); i++)
		{
			std::vector<double> mea_probs(1ull << m_mea_qubits.size(), 0.0);

            switch (m_machine_type)
            {
            case QMachineType::QCloud:
                {
                    auto qcloud_ptr = dynamic_cast<QCloudMachine*>(m_machine_ptr);
                    auto cloud_result = qcloud_ptr->real_chip_measure(mea_all_progs[i], shots, 
                        chip_type, 
                        false, 
                        false, 
                        false);
                    
                    for (auto val : cloud_result)
                    {
                        bitset<BIT_SIZE> temp_bit(val.first);
                        int idx = temp_bit.to_ulong();
                        mea_probs[idx] = val.second;
                    }

                    break;
                }

            case QMachineType::NOISE:
                {
                    auto noise_ptr = dynamic_cast<NoiseQVM*>(m_machine_ptr);
                    auto  mea_result = noise_ptr->runWithConfiguration(mea_all_progs[i], m_mea_cc, shots);
                    for (auto val : mea_result)
                    {
                        bitset<BIT_SIZE> temp_bit(val.first);
                        int idx = temp_bit.to_ulong();
                        mea_probs[idx] = (double)val.second / (double)shots;
                    }

                    break;
                }

            case QMachineType::DENSITY_MATRIX:
                {
                    auto density_matrix_ptr = dynamic_cast<DensityMatrixSimulator*>(m_machine_ptr);
                    auto  mea_result = density_matrix_ptr->get_probabilities(mea_all_progs[i]);
                    
                    for (auto i = 0; i < mea_result.size(); ++i)
                        mea_probs[i] = mea_result[i];

                    break;
                }

            default: QCERR_AND_THROW(std::runtime_error, "QMachineType error");
            }

			probs_meas[i][k] = mea_probs ;
		}
	}

	return  _xeb_fidelities(probs_exp, probs_meas);
}

std::map<int, double> CrossEntropyBenchmarking::calculate_xeb_fidelity(
    GateType gate_name,
    int qubit_0,
    int qubit_1,
    const std::vector<int>& cycle_range,
    int num_circuits)
{
    m_cycle_range = cycle_range;
    m_num_circuits = num_circuits;

    //ideal probs
    DensityMatrixSimulator simulator;
    simulator.init();
    m_exp_qubits = simulator.qAllocMany(2);

    m_mea_qubits = { m_qcloud.allocateQubitThroughPhyAddress(qubit_0),
                     m_qcloud.allocateQubitThroughPhyAddress(qubit_1)};
    m_mea_cc = m_qcloud.cAllocMany(m_mea_qubits.size());

    multi_probs probs_exp(m_cycle_range.size(), vector<vector<double>>(m_num_circuits));
    multi_probs probs_meas(m_cycle_range.size(), vector<vector<double>>(m_num_circuits));

    for (int k = 0; k < m_num_circuits; k++)
    {
        std::vector<QProg> exp_all_progs, mea_all_progs;

        build_xeb_circuits(exp_all_progs, mea_all_progs, gate_name);

        for (int i = 0; i < m_cycle_range.size(); i++)
        {
            auto exp_probs = simulator.get_probabilities(exp_all_progs[i], m_exp_qubits);
            probs_exp[i][k] = exp_probs;
        }

        for (int i = 0; i < m_cycle_range.size(); i++)
        {
            std::vector<double> mea_probs(1ull << m_mea_qubits.size(), 0.0);

            auto cloud_result = m_qcloud.real_chip_measure(mea_all_progs[i],
                m_cloud_config.shots,
                m_cloud_config.chip_id,
                m_cloud_config.open_amend,
                m_cloud_config.open_mapping,
                m_cloud_config.open_optimization);

                for (auto val : cloud_result)
                {
                    bitset<BIT_SIZE> temp_bit(val.first);
                    int idx = temp_bit.to_ulong();
                    mea_probs[idx] = val.second;
                }


            probs_meas[i][k] = mea_probs;
        }
    }

    return  _xeb_fidelities(probs_exp, probs_meas);
}

std::map<int, double> QPanda::double_gate_xeb(QuantumMachine* machine,
    Qubit* qubit_0,
    Qubit* qubit_1,
    const std::vector<int>& range,
    int num_circuits,
    int shots,
    RealChipType type,
    GateType gate_name)
{
	CrossEntropyBenchmarking benchmarking(machine);
	return benchmarking.calculate_xeb_fidelity(gate_name, qubit_0, qubit_1, range, num_circuits, shots, type);
}

std::map<int, double> QPanda::double_gate_xeb(QCloudTaskConfig config,
    int qubit_0,
    int qubit_1,
    const std::vector<int>& range,
    int num_circuits,
    GateType gate_name)
{
    CrossEntropyBenchmarking benchmarking(config);
    return benchmarking.calculate_xeb_fidelity(gate_name, qubit_0, qubit_1, range, num_circuits);
}

#endif