#include "Core/Utilities/QProgInfo/QProgClockCycle.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include "Core/Utilities/QProgTransform/QProgToQCircuit.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"


using namespace std;
USING_QPANDA

using sequence_gate_t = SeqNode<DAGSeqNode>;

QPanda::QProgClockCycle::QProgClockCycle(QPanda::QuantumMachine *qm)
{
    m_gate_time = qm->getGateTimeMap();
}

QPanda::QProgClockCycle::QProgClockCycle()
{
    QuantumMetadata metadata;
    metadata.getGateTime(m_gate_time);
}

QProgClockCycle::~QProgClockCycle()
{
}

size_t QProgClockCycle::count(QProg &prog, bool optimize /* = false */)
{
    if (optimize)
    {
        std::ifstream reader(CONFIG_PATH);
        QPANDA_ASSERT(!reader.is_open(), "No config file.");
        decompose_multiple_control_qgate(prog, m_machine);
        cir_optimizer_by_config(prog, CONFIG_PATH, QCircuitOPtimizerMode::Merge_U3);
    }

    /*QProgTopologSeq<GateNodeInfo, SequenceNode> m_prog_topo_seq;
    m_prog_topo_seq.prog_to_topolog_seq(prog, SequenceNode::construct_sequence_node);
    TopologSequence<SequenceNode>& graph_seq = m_prog_topo_seq.get_seq();*/
    std::shared_ptr<QProgDAG> dag = qprog_to_DAG(prog);
    TopologSequence<DAGSeqNode> graph_seq = dag->build_topo_sequence();
    size_t clock_cycle = 0;

    for (auto &layer : graph_seq)
    {
        auto iter = std::max_element(layer.begin(), layer.end(),
            [=](const sequence_gate_t &a, const sequence_gate_t &b)
        {
            GateType gate_type_a = static_cast<GateType>(a.first.m_node_type);
            GateType gate_type_b = static_cast<GateType>(b.first.m_node_type);
            auto time_a = getQGateTime(gate_type_a);
            auto time_b = getQGateTime(gate_type_b);
            return time_a < time_b;
        });

        auto node_type = iter->first.m_node_type;
        auto gate_type = static_cast<GateType>(iter->first.m_node_type);
        clock_cycle += getQGateTime(gate_type);
    }

    return clock_cycle;
}

void QProgClockCycle::get_time_map()
{
    QuantumMetadata metadata;
    metadata.getGateTime(m_gate_time);
}

size_t QProgClockCycle::getDefalutQGateTime(GateType gate_type)
{
    const size_t kSingleGateDefaultTime = 1;
    const size_t kDoubleGateDefaultTime = 2;

    switch (gate_type)
    {
    case PAULI_X_GATE:
    case PAULI_Y_GATE:
    case PAULI_Z_GATE:
    case X_HALF_PI:
    case Y_HALF_PI:
    case Z_HALF_PI:
    case HADAMARD_GATE:
    case T_GATE:
    case S_GATE:
    case I_GATE:
    case RX_GATE:
    case RY_GATE:
    case RZ_GATE:
    case U1_GATE:
    case U2_GATE:
    case U3_GATE:
    case U4_GATE:
    case BARRIER_GATE:
        return kSingleGateDefaultTime;
    case CU_GATE:
    case CNOT_GATE:
    case CZ_GATE:
    case CPHASE_GATE:
    case ISWAP_THETA_GATE:
    case ISWAP_GATE:
    case SWAP_GATE:
    case SQISWAP_GATE:
    case TWO_QUBIT_GATE:
        return kDoubleGateDefaultTime;
    case DAGNodeType::RESET:
    case DAGNodeType::MEASURE:
        return 0;
    default:
        std::string error_msg = "Bad nodeType -> " + to_string(gate_type);
        QCERR_AND_THROW(run_fail, error_msg);
    }

    return 0;
}

size_t QProgClockCycle::getQGateTime(GateType gate_type)
{
    // -1 means measure, -2 means reset
    QPANDA_RETURN(-1 == gate_type || -2 == gate_type, 0);

    auto iter = m_gate_time.find(gate_type);
    size_t gate_time_value = 0;

    if (m_gate_time.end() == iter)
    {
        gate_time_value = getDefalutQGateTime(gate_type);
        m_gate_time.insert({ gate_type, gate_time_value });
    }
    else
    {
        gate_time_value = iter->second;
    }

    return gate_time_value;
}

size_t QPanda::getQProgClockCycle(QProg &prog, QuantumMachine *qm, bool optimize /* = false */)
{
    QProgClockCycle counter(qm);
    return counter.count(prog, optimize);
}

size_t QPanda::get_qprog_clock_cycle(QProg & prog, QuantumMachine * qm, bool optimize /* = false */)
{
    QProgClockCycle counter(qm);
    return counter.count(prog, optimize);
}

#include <algorithm>
QProgInfoCount QProgClockCycle::count_layer_info(QProg &prog, std::vector<GateType> selected_types)
{
    /*QProgTopologSeq<GateNodeInfo, SequenceNode> m_prog_topo_seq;
    m_prog_topo_seq.prog_to_topolog_seq(prog, SequenceNode::construct_sequence_node);
    TopologSequence<SequenceNode>& graph_seq = m_prog_topo_seq.get_seq();*/
    std::shared_ptr<QProgDAG> dag = qprog_to_DAG(prog);
    TopologSequence<DAGSeqNode> graph_seq = dag->build_topo_sequence();

    for (auto val : selected_types)
        m_prog_info.selected_gate_nums[val] = 0;

    for (auto &layer : graph_seq)
    {
        m_prog_info.layer_num++;

        bool is_full_single_gate = true;
        bool is_full_double_gate = true;

        for (auto layer_iter = layer.begin(); layer_iter != layer.end(); layer_iter++)
        {
            m_prog_info.node_num++;

            GateType gate_node_type = static_cast<GateType>(layer_iter->first.m_node_type);
            auto control_qubits = layer_iter->first.m_node_ptr->m_control_vec;

            auto selected_iter = std::find(selected_types.begin(), selected_types.end(), gate_node_type);

            if (selected_iter != selected_types.end()) 
            {
                auto iter = m_prog_info.selected_gate_nums.find(gate_node_type);

                if (iter != m_prog_info.selected_gate_nums.end())
                    iter->second++;
                else
                    m_prog_info.selected_gate_nums[gate_node_type] = 1;
            }

            if ((DAGNodeType::MEASURE > gate_node_type) && (DAGNodeType::NUKNOW_SEQ_NODE_TYPE < gate_node_type))
            {
                m_prog_info.gate_num++;

                //single gate with 0 control -> single gate 
                if (is_single_gate(gate_node_type) && control_qubits.empty())
                {
                    is_full_double_gate = false;
                    m_prog_info.single_gate_num++;
                }
                //single gate with 1 control -> double gate 
                else if (is_single_gate(gate_node_type) && 1 == control_qubits.size())
                {
                    is_full_single_gate = false;
                    m_prog_info.double_gate_num++;
                }
                //double gate with 0 control -> double gate 
                else if (is_double_gate(gate_node_type) && control_qubits.empty())
                {
                    is_full_single_gate = false;
                    m_prog_info.double_gate_num++;
                }
                else
                {
                    is_full_single_gate = false;
                    is_full_double_gate = false;
                    m_prog_info.multi_control_gate_num++;
                }
            }
            else
            {
                is_full_single_gate = false;
                is_full_double_gate = false;
            }
        }

        if (is_full_single_gate && !is_full_double_gate)
            m_prog_info.single_gate_layer_num++;

        if (is_full_double_gate && !is_full_single_gate)
            m_prog_info.double_gate_layer_num++;
    }

    return m_prog_info;
}



size_t QPanda::get_qprog_clock_cycle_chip(LayeredTopoSeq &layer_info, std::map<GateType, size_t> gate_time_map)
{
    size_t clock_cycle = 0;
    for (auto &layer : layer_info)
    {
        size_t clock_cycle_layer = 0;
        for (auto &gate : layer)
        {
            auto node = gate.first;
            if (node->m_type < 0 || node->m_type > DAGNodeType::MAX_GATE_TYPE){
                if ((0 == clock_cycle) && (DAGNodeType::MEASURE == node->m_type)) {
                    clock_cycle = 1;
                }
                continue;
            }

            GateType gate_type = static_cast<GateType>(gate.first->m_gate_type);
            auto time_gate = gate_time_map.find(gate_type);
            if (time_gate == gate_time_map.end())
            {
                std::string error_msg = "Bad nodeType -> " + to_string(gate_type) + ", for chip.";
                QCERR_AND_THROW(run_fail, error_msg);
            }

            clock_cycle_layer = clock_cycle_layer > time_gate->second ? clock_cycle_layer : time_gate->second;
        }

        clock_cycle += clock_cycle_layer;
    }

    return clock_cycle;
}
