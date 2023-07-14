#include "Core/QuantumMachine/PartialAmplitudeQVM.h"
#include <set>

USING_QPANDA
using angleParameter = QGATE_SPACE::AbstractSingleAngleParameter;
using namespace std;

static void get_dec_index(std::vector<string> &bin_index, std::vector<uint128_t> &dec_index)
{
	for (auto val : bin_index)
	{
		uint128_t dec_value = 0;
		size_t len = val.size();

		for (size_t i = 0; i < len; ++i)
		{
			bool bin = (val[len - i - 1] != '0');
			uint128_t temp = static_cast<uint128_t>(bin) << i;
			dec_value |= temp;
		}

		dec_index.emplace_back(dec_value);
	}

    return;
}


static void get_couple_state_index(uint128_t num, uint64_t& under_index, uint64_t& upper_index, uint32_t qubit_num)
{
    uint32_t half_qubit = qubit_num / 2;
    long long lower_mask = (1ull << half_qubit) - 1;
    under_index = (uint64_t)(num & lower_mask);
    upper_index = (uint64_t)(num - under_index) >> half_qubit;
    return;
}

void PartialAmplitudeQVM::init(BackendType type)
{
    if (BackendType::CPU == type)
        m_simulator = std::make_unique<CPUImplQPU<double>>();
    else if (BackendType::MPS == type)
        m_simulator = std::make_unique<MPSImplQPU>();
#ifdef USE_CUDA
    else if (BackendType::GPU == type)
        m_simulator = std::make_unique<GPUImplQPU>();
#endif // USE_CUDA
    else
        QCERR_AND_THROW(run_fail, "PartialAmplitudeQVM::init");

	_Config.maxQubit = 80;
	_Config.maxCMem = 80;
	_start();
}

void PartialAmplitudeQVM::computing_graph(int qubit_num,const cir_type& circuit, QStat& state)
{
	state.resize(1ull << qubit_num);

	try
	{
		m_simulator->initState(0, 1, qubit_num);
		m_graph_backend.computing_graph(circuit, m_simulator);

		auto graph_state = m_simulator->getQState();

		state.assign(graph_state.begin(), graph_state.end());
	}
	catch (const std::exception& e)
	{
        QCERR_AND_THROW(run_fail, e.what());
	}

    return;
}

void PartialAmplitudeQVM::caculate_qstate(QStat &state)
{
    auto qubit_num = m_graph_backend.m_qubit_num;
    auto graph_num = m_graph_backend.m_sub_graph.size();
    state.resize(1ull << qubit_num, 0);

    for (auto graph_index = 0; graph_index < graph_num; ++graph_index)
    {
        QStat under_graph_state;
        computing_graph(qubit_num / 2, m_graph_backend.m_sub_graph[graph_index][0], under_graph_state);

        QStat upper_graph_state;
        computing_graph(qubit_num - (qubit_num / 2), m_graph_backend.m_sub_graph[graph_index][1], upper_graph_state);

        for (size_t state_idx = 0;
             state_idx < (1ull << qubit_num);
             ++state_idx)
        {
            uint64_t under_index, upper_index;
            get_couple_state_index(state_idx, under_index, upper_index, m_graph_backend.m_qubit_num);
            state[state_idx] += under_graph_state[under_index] * upper_graph_state[upper_index];
        }
    }

    return ;
}


qcomplex_t PartialAmplitudeQVM::pmeasure_bin_index(std::string amplitude)
{
	uint128_t index = 0;
	size_t qubit_num = amplitude.size();
	for (size_t i = 0; i < qubit_num; ++i)
	{
		index += (amplitude[qubit_num - i - 1] != '0') << i ;
	}

	return pmeasure_dec_index(integerToString(index));
}

qcomplex_t PartialAmplitudeQVM::pmeasure_dec_index(std::string amplitude)
{
	uint128_t dec_amplitude(amplitude.c_str());

	auto qubit_num = m_graph_backend.m_qubit_num;
	auto graph_num = m_graph_backend.m_sub_graph.size();

	qcomplex_t result;
	for (auto graph_index = 0; graph_index < graph_num; ++graph_index)
	{
		QStat under_graph_state;
		computing_graph(qubit_num / 2, m_graph_backend.m_sub_graph[graph_index][0], under_graph_state);

		QStat upper_graph_state;
		computing_graph(qubit_num - (qubit_num / 2), m_graph_backend.m_sub_graph[graph_index][1], upper_graph_state);

		uint64_t under_index, upper_index;
		get_couple_state_index(dec_amplitude, under_index, upper_index, m_graph_backend.m_qubit_num);

        if (1 == qubit_num)
            result += upper_graph_state[amplitude != "0"];
        else
            result += under_graph_state[under_index] * upper_graph_state[upper_index];
	}

	return result;
}


stat_map PartialAmplitudeQVM::pmeasure_subset(const std::vector<std::string>& amplitude)
{
    uint128_t max_index = (uint128_t)1 << getAllocateQubitNum();

    for (auto val : amplitude)
    {
        auto temp_amplitude = uint128_t(val.c_str());

        if (max_index <= temp_amplitude)
            QCERR_AND_THROW(run_fail, "current pmeasure amplitude > max_amplitude");
    }

	std::vector<uint128_t> dec_state;
	for (auto state : amplitude)
	{
		uint128_t val(state.c_str());
		dec_state.emplace_back(val);
	}

	auto qubit_num = m_graph_backend.m_qubit_num;
	auto graph_num = m_graph_backend.m_sub_graph.size();

	QStat result(dec_state.size());
	for (auto graph_index = 0; graph_index < graph_num; ++graph_index)
	{
		QStat under_graph_state;
		computing_graph(qubit_num / 2, m_graph_backend.m_sub_graph[graph_index][0], under_graph_state);

		QStat upper_graph_state;
		computing_graph(qubit_num - (qubit_num / 2), m_graph_backend.m_sub_graph[graph_index][1], upper_graph_state);

        for (size_t idx = 0; idx < dec_state.size(); ++idx)
		{
			uint64_t under_index, upper_index;
			get_couple_state_index(dec_state[idx], under_index, upper_index, m_graph_backend.m_qubit_num);

            if (1 == qubit_num)
                result[idx] += upper_graph_state[amplitude[idx] != "0"];
            else
                result[idx] += under_graph_state[under_index] * upper_graph_state[upper_index];
		}
	}

	stat_map state_result;
	for (auto idx = 0; idx < amplitude.size(); ++idx)
	{
		auto pair = std::make_pair(amplitude[idx], result[idx]);

		state_result.insert(pair);
	}
    return state_result;
}

prob_dict PartialAmplitudeQVM::getProbDict(const QVec &qlist)
{
    QStat state;
    caculate_qstate(state);

    std::vector<size_t> qubits_addr;
    for_each(qlist.begin(), qlist.end(), [&](Qubit *q){
        qubits_addr.push_back(q->get_phy_addr());
    });

    stable_sort(qubits_addr.begin(), qubits_addr.end());
    qubits_addr.erase(unique(qubits_addr.begin(),
                             qubits_addr.end(), [](size_t a, size_t b){return a == b;}),
                             qubits_addr.end());
    size_t measure_qubit_num = qubits_addr.size();

    prob_dict res;
    for (size_t i = 0; i < state.size(); i++)
    {
        size_t idx = 0;
        for (size_t j = 0; j < measure_qubit_num; j++)
        {
            idx += (((i >> (qubits_addr[j])) % 2) << j);
        }

        string bin_idx = integerToBinary(idx, measure_qubit_num);
        auto iter = res.find(bin_idx);
        if (res.end() == iter)
        {
            res.insert({bin_idx, std::norm(state[i])});
        }
        else
        {
            iter->second += std::norm(state[i]);
        }
    }

    return res;
}

prob_dict PartialAmplitudeQVM::probRunDict(QProg &prog, const QVec &qlist)
{
    run(prog);
    return getProbDict(qlist);
}

prob_vec PartialAmplitudeQVM::getProbList(const QVec &qlist)
{
    QStat state;
    caculate_qstate(state);

    std::vector<size_t> qubits_addr;
    for_each(qlist.begin(), qlist.end(), [&](Qubit *q){
        qubits_addr.push_back(q->get_phy_addr());
    });

    stable_sort(qubits_addr.begin(), qubits_addr.end());
    qubits_addr.erase(unique(qubits_addr.begin(),
                             qubits_addr.end(), [](size_t a, size_t b){return a == b;}),
                             qubits_addr.end());
    size_t measure_qubit_num = qubits_addr.size();

    prob_vec res(1ull << measure_qubit_num, 0);
    for (size_t i = 0; i < state.size(); i++)
    {
        size_t idx = 0;
        for (size_t j = 0; j < measure_qubit_num; j++)
        {
            idx += (((i >> (qubits_addr[j])) % 2) << j);
        }

        res[idx] += std::norm(state[i]);
    }

    return res;
}

prob_vec PartialAmplitudeQVM::probRunList(QProg &prog, const QVec &qlist)
{
    run(prog);
    return getProbList(qlist);
}


void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore measure");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore controlflow");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQNoiseNode> cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR_AND_THROW(std::invalid_argument, "PartialAmplitudeQVM not support execute Virtual Noise Node");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQDebugNode> cur_node, std::shared_ptr<QNode> parent_node){
	QCERR_AND_THROW(std::invalid_argument, "PartialAmplitudeQVM not support Debug");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
{
	Traversal::traversal(cur_node, true, *this);
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
{
	Traversal::traversal(cur_node, *this);
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore classical prog");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQuantumReset>  cur_node, std::shared_ptr<QNode> parent_node)
{
	QCERR("ignore reset");
}

void PartialAmplitudeQVM::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
	if (nullptr == cur_node || nullptr == cur_node->getQGate())
	{
		QCERR("pQGate is null");
		throw invalid_argument("pQGate is null");
	}

	QVec qubits_vector;
	cur_node->getQuBitVector(qubits_vector);

    auto gate_type = (unsigned short)cur_node->getQGate()->getGateType();
    switch (gate_type)
    {
        case GateType::P0_GATE:
        case GateType::P1_GATE:
        case GateType::PAULI_Y_GATE:
        case GateType::PAULI_Z_GATE:
        case GateType::X_HALF_PI:
        case GateType::Y_HALF_PI:
        case GateType::Z_HALF_PI:
        case GateType::HADAMARD_GATE:
        case GateType::T_GATE:
        case GateType::S_GATE:
        {
            auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit }, {});
        }
        break;

        case GateType::PAULI_X_GATE:
        {
            auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();

            QVec control_qvec;
            cur_node->getControlVector(control_qvec);

            if (control_qvec.empty())
            {
                construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit }, {});
            }
            else
            {
                auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
                auto ctr_qubit = control_qvec[0]->getPhysicalQubitPtr()->getQubitAddr();
                auto tof_qubit = control_qvec[1]->getPhysicalQubitPtr()->getQubitAddr();

                construct_qnode(TOFFOLI_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit, tof_qubit }, {});

                m_graph_backend.m_spilt_num += (m_graph_backend.is_corss_node(ctr_qubit, tar_qubit)) ||
                    (m_graph_backend.is_corss_node(ctr_qubit, tof_qubit)) ||
                    (m_graph_backend.is_corss_node(tar_qubit, tof_qubit));
            }
        }
        break;

        case GateType::U1_GATE:
        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::RZ_GATE:
        {
            auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto param_ptr = dynamic_cast<angleParameter*>(cur_node->getQGate());
            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit }, { param_ptr->getParameter() });
        }
        break;

        case GateType::U2_GATE:
        {
            auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto u2_gate = dynamic_cast<QGATE_SPACE::U2*>(cur_node->getQGate());

            prob_vec params;
            params.emplace_back(u2_gate->get_phi());
            params.emplace_back(u2_gate->get_lambda());

            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit }, params);
        }
        break;

        case GateType::U3_GATE:
        {
            auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto u3_gate = dynamic_cast<QGATE_SPACE::U3*>(cur_node->getQGate());

            prob_vec params;
            params.emplace_back(u3_gate->get_theta());
            params.emplace_back(u3_gate->get_phi());
            params.emplace_back(u3_gate->get_lambda());

            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit }, params);
        }
        break;

        case GateType::U4_GATE:
        {
            auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto angle_param = dynamic_cast<QGATE_SPACE::AbstractAngleParameter *>(cur_node->getQGate());

            prob_vec params;
            params.emplace_back(angle_param->getAlpha());
            params.emplace_back(angle_param->getBeta());
            params.emplace_back(angle_param->getGamma());
            params.emplace_back(angle_param->getDelta());

            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit }, params);
        }
        break;

        case GateType::SWAP_GATE:
        {
            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            if (m_graph_backend.is_corss_node(ctr_qubit, tar_qubit))
            {
                // SWAP(0, 1) => CNOT(0, 1) + CNOT(1, 0) + CNOT(0, 1)

                construct_qnode(CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
                construct_qnode(CNOT_GATE, cur_node->isDagger(), { ctr_qubit,tar_qubit }, {});
                construct_qnode(CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});

                m_graph_backend.m_spilt_num += 3;
            }
            else
            {
                construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            }
        }
        break;

        case GateType::ISWAP_GATE:
        {
            // iSWAP(0, 1) => 
            // CU(0, 1)(1.570796, 3.141593, 0.000000, 0.000000).dag + 
            // CU(0, 1)(-1.570796, 6.283185, 3.141593, 0.000000).dag +
            // CU(1, 0)(-1.570796, 3.141593, 3.141593, 0.000000).dag +
            // CU(0, 1)(1.570796, 6.283185, 3.141593, 0.000000).dag
           
            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            if (m_graph_backend.is_corss_node(ctr_qubit, tar_qubit))
            {
                prob_vec params0 = { PI / 2, PI, 0, 0 };
                prob_vec params1 = { -PI / 2, 2 * PI, PI, 0 };
                prob_vec params2 = { -PI / 2, PI, PI, 0 };
                prob_vec params3 = { PI / 2, 2 * PI, PI, 0 };

                construct_qnode(CU_GATE, !cur_node->isDagger(), { tar_qubit,ctr_qubit }, params0);
                construct_qnode(CU_GATE, !cur_node->isDagger(), { tar_qubit,ctr_qubit }, params1);
                construct_qnode(CU_GATE, !cur_node->isDagger(), { ctr_qubit,tar_qubit }, params2);
                construct_qnode(CU_GATE, !cur_node->isDagger(), { tar_qubit,ctr_qubit }, params3);

                m_graph_backend.m_spilt_num += 4;
            }
            else
            {
                construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            }
        }
        break;

        case GateType::SQISWAP_GATE:
        {
            // SqiSWAP(0, 1) => 
            // CU(0, 1)(1.570796, 3.141593, 0.000000, 0.000000).dag + 
            // CU(0, 1)(1.570796, 6.283185, 3.141593, 0.000000).dag +
            // CU(1, 0)(1.570796, 0.000000, 1.570796, 3.141593).dag +
            // CU(0, 1)(-1.570796, 6.283185, 3.141593, 0.000000).dag
           
            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            if (m_graph_backend.is_corss_node(ctr_qubit, tar_qubit))
            {
                prob_vec params0 = { PI / 2, PI, 0, 0 };
                prob_vec params1 = { PI / 2, 2 * PI, PI, 0 };
                prob_vec params2 = { PI / 2, 0, PI / 2, PI };
                prob_vec params3 = { -PI / 2, 2 * PI, PI, 0 };

                construct_qnode(CU_GATE, !cur_node->isDagger(), { tar_qubit,ctr_qubit }, params0);
                construct_qnode(CU_GATE, !cur_node->isDagger(), { tar_qubit,ctr_qubit }, params1);
                construct_qnode(CU_GATE, !cur_node->isDagger(), { ctr_qubit,tar_qubit }, params2);
                construct_qnode(CU_GATE, !cur_node->isDagger(), { tar_qubit,ctr_qubit }, params3);

                m_graph_backend.m_spilt_num += 4;
            }
            else
            {
                construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            }
        }
        break;

        case GateType::CNOT_GATE:
        case GateType::CZ_GATE:
        {
            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            m_graph_backend.m_spilt_num += m_graph_backend.is_corss_node(ctr_qubit, tar_qubit);
        }
        break;

        case GateType::CPHASE_GATE:
        {
            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            auto param_ptr = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter *>(cur_node->getQGate());

            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit,ctr_qubit }, { param_ptr->getParameter() });
            m_graph_backend.m_spilt_num += m_graph_backend.is_corss_node(ctr_qubit, tar_qubit);
        }
        break;

        case GateType::CU_GATE:
        {
            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            auto angle_param = dynamic_cast<QGATE_SPACE::AbstractAngleParameter *>(cur_node->getQGate());

            prob_vec params;
            params.emplace_back(angle_param->getAlpha());
            params.emplace_back(angle_param->getBeta());
            params.emplace_back(angle_param->getGamma());
            params.emplace_back(angle_param->getDelta());

            construct_qnode(gate_type, cur_node->isDagger(), { tar_qubit,ctr_qubit }, params);
            m_graph_backend.m_spilt_num += m_graph_backend.is_corss_node(ctr_qubit, tar_qubit);
        }
        break;

        case GateType::RXX_GATE:
        {
            //RXX(0, 1, theta) => 
            //H(0) + H(1) + 
            //CNOT(0, 1) + 
            //RZ(1, theta) +
            //CNOT(0, 1) +
            //H(1) + H(0)

            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            auto angle_param = dynamic_cast<angleParameter*>(cur_node->getQGate())->getParameter();

            construct_qnode(GateType::HADAMARD_GATE, cur_node->isDagger(), { ctr_qubit }, {});
            construct_qnode(GateType::HADAMARD_GATE, cur_node->isDagger(), { tar_qubit }, {});
            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            construct_qnode(GateType::RZ_GATE, cur_node->isDagger(), { tar_qubit }, { angle_param });
            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            construct_qnode(GateType::HADAMARD_GATE, cur_node->isDagger(), { tar_qubit }, {});
            construct_qnode(GateType::HADAMARD_GATE, cur_node->isDagger(), { ctr_qubit }, {});

            m_graph_backend.m_spilt_num += 2;
        }
        break;

        case GateType::RYY_GATE:
        {
            //RYY(0, 1, theta) => 
            //RX(0, PI / 2) + RX(1, PI / 2) + 
            //CNOT(0, 1) + 
            //RZ(1, theta) +
            //CNOT(0, 1) + 
            //RX(0, -PI / 2) + RX(1, -PI / 2) + 

            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            auto angle_param = dynamic_cast<angleParameter*>(cur_node->getQGate())->getParameter();

            construct_qnode(GateType::RX_GATE, cur_node->isDagger(), { ctr_qubit }, { PI / 2 });
            construct_qnode(GateType::RX_GATE, cur_node->isDagger(), { tar_qubit }, { PI / 2 });
            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            construct_qnode(GateType::RZ_GATE, cur_node->isDagger(), { tar_qubit }, { angle_param });
            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            construct_qnode(GateType::RX_GATE, cur_node->isDagger(), { ctr_qubit }, { -PI / 2 });
            construct_qnode(GateType::RX_GATE, cur_node->isDagger(), { tar_qubit }, { -PI / 2 });

            m_graph_backend.m_spilt_num += 2;
        }
        break;

        case GateType::RZZ_GATE:
        {
            //RZZ(0, 1, theta) => 
            //CNOT(0, 1) + 
            //RZ(1, theta) +
            //CNOT(0, 1) + 

            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            auto angle_param = dynamic_cast<angleParameter*>(cur_node->getQGate())->getParameter();

            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            construct_qnode(GateType::RZ_GATE, cur_node->isDagger(), { tar_qubit }, { angle_param });
            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});

            m_graph_backend.m_spilt_num += 2;
        }
        break;

        case GateType::RZX_GATE:
        {
            //RXX(0, 1, theta) => 
            //H(1) + 
            //CNOT(0, 1) + 
            //RZ(1, theta) +
            //CNOT(0, 1) +
            //H(1)

            auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
            auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();

            auto angle_param = dynamic_cast<angleParameter*>(cur_node->getQGate())->getParameter();

            construct_qnode(GateType::HADAMARD_GATE, cur_node->isDagger(), { tar_qubit }, {});
            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            construct_qnode(GateType::RZ_GATE, cur_node->isDagger(), { tar_qubit }, { angle_param });
            construct_qnode(GateType::CNOT_GATE, cur_node->isDagger(), { tar_qubit,ctr_qubit }, {});
            construct_qnode(GateType::HADAMARD_GATE, cur_node->isDagger(), { tar_qubit }, {});

            m_graph_backend.m_spilt_num += 2;
        }
        break;

        case GateType::BARRIER_GATE:break;
        default:
        {
            string erroe_msg = "UnSupported QGate Node, Gate Type : " + to_string(gate_type);
            QCERR_AND_THROW(undefine_error, erroe_msg);
        }
        break;
    }
}

void PartialAmplitudeQVM::construct_graph()
{
	auto qubit_num = getAllocateQubit();
	if (!m_graph_backend.m_spilt_num)
	{
		m_graph_backend.split_circuit(m_graph_backend.m_circuit);
	}
	else
	{
		m_graph_backend.traversal(m_graph_backend.m_circuit);
	}
}


void PartialAmplitudeQVM::construct_qnode(int gate_type, bool is_dagger, const std::vector<size_t>& qubits, const std::vector<double>& params)
{
    QGateNode node;
    node.gate_type = gate_type;
    node.is_dagger = is_dagger;
    node.params = params;
    node.qubits = qubits;

    m_graph_backend.m_circuit.emplace_back(node);

    return;
}