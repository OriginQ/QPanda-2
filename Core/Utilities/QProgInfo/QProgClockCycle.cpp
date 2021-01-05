#include "Core/Utilities/QProgInfo/QProgClockCycle.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include "Core/Utilities/QProgTransform/QProgToQCircuit.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Tools/QCircuitOptimize.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"


using namespace std;
USING_QPANDA

using sequence_gate_t = SeqNode<SequenceNode>;

QPanda::QProgClockCycle::QProgClockCycle(QPanda::QuantumMachine *qm)
{
    m_gate_time = qm->getGateTimeMap();
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

	QProgTopologSeq<GateNodeInfo, SequenceNode> m_prog_topo_seq;
	m_prog_topo_seq.prog_to_topolog_seq(prog, SequenceNode::construct_sequence_node);
	TopologSequence<SequenceNode>& graph_seq = m_prog_topo_seq.get_seq();
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
        return kSingleGateDefaultTime;
    case CU_GATE:
    case CNOT_GATE:
    case CZ_GATE:
    case CPHASE_GATE:
    case ISWAP_THETA_GATE:
    case ISWAP_GATE:
    case SQISWAP_GATE:
    case TWO_QUBIT_GATE:
        return kDoubleGateDefaultTime;
    default:
        QCERR("Bad nodeType");
        throw std::runtime_error("Bad nodeType");
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
