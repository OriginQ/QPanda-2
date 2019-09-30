#include "Core/Utilities/QProgToDAG/GraphMatch.h"
#include "Core/QuantumCircuit/QuantumGate.h"
#include "Core/QuantumCircuit/QGate.h"
#include <memory>
#include <cmath>

USING_QPANDA
using namespace std;
using namespace QGATE_SPACE;

static const double precision = 1e-6;

static int nodeType(int type)
{
    switch (type)
    {
        case -1:
        case GateType::T_GATE:
        case GateType::S_GATE:
        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::RZ_GATE:
        case GateType::U1_GATE:
        case GateType::U2_GATE:
        case GateType::U3_GATE:
        case GateType::U4_GATE:
        case GateType::X_HALF_PI:
        case GateType::Y_HALF_PI:
        case GateType::Z_HALF_PI:
        case GateType::PAULI_X_GATE:
        case GateType::PAULI_Y_GATE:
        case GateType::PAULI_Z_GATE:
        case GateType::HADAMARD_GATE:
            return QNodeGateType::SINGLE_OR_MEASURE_GATE;

        case GateType::CU_GATE:
        case GateType::CNOT_GATE:
        case GateType::CZ_GATE:
        case GateType::CPHASE_GATE:
        case GateType::ISWAP_THETA_GATE:
        case GateType::ISWAP_GATE:
        case GateType::SQISWAP_GATE:
        case GateType::SWAP_GATE:
        case GateType::TWO_QUBIT_GATE:
            return QNodeGateType::DOUBLE_GATE;

        default:
            QCERR("get gate type error");
            throw std::runtime_error("get gate type error");
            break;
    }
}

static bool nodeIdentifier(SequenceNode &node, MatchVector &match_result)
{
    for (const auto &result : match_result)
    {
        for (const auto &layer : result)
        {
            for (const auto &graph_node : layer)
            {
                if (graph_node.m_vertex_num == node.m_vertex_num)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool nodeIdentifier(SequenceNode &node, ResultVector &match_result)
{
    for (const auto &layer : match_result)
    {
        for (const auto &graph_node : layer)
        {
            if (graph_node.m_vertex_num == node.m_vertex_num)
            {
                return false;
            }
        }
    }
    return true;
}

static void deleteSequenceNode(size_t vertice_num, TopologincalSequence &seq)
{
    for (auto &layer : seq)
    {
        SequenceLayer::iterator iter;
        for (iter = layer.begin(); iter != layer.end();)
        {
            if (iter->first.m_vertex_num == vertice_num)
            {
                iter = layer.erase(iter);
            }
            else
            {
                ++iter;
            }
        }
    }
}


static size_t getLayerNumber(const TopologincalSequence &graph_seq,size_t vertice_num)
{
    for (auto i = 0; i != graph_seq.size(); ++i)
    {
        for (auto j = 0; j != graph_seq[i].size(); ++j)
        {
            if (vertice_num == graph_seq[i][j].first.m_vertex_num)
            {
                return i;
            }
        }
    }
}

void GraphMatch::updateQubits(LayerVector &layer, std::vector<size_t> &qvec)
{
    if (qvec.empty())
    {
        for (auto node : layer)
        {
            qvec.emplace_back(getTarQubit(node, m_graph_dag));
            if (QNodeGateType::DOUBLE_GATE == nodeType(node.m_node_type))
            {
                qvec.emplace_back(getCtrQubit(node, m_graph_dag));
            }
        }
    }
    else
    {
        for (auto node : layer)
        {
            if (QNodeGateType::DOUBLE_GATE == nodeType(node.m_node_type))
            {
                auto tar_iter = find(qvec.begin(), qvec.end(), getTarQubit(node, m_graph_dag));
                auto ctr_iter = find(qvec.begin(), qvec.end(), getCtrQubit(node, m_graph_dag));

                if ((qvec.end() == tar_iter) && (qvec.end() != ctr_iter))
                {
                    qvec.emplace_back(getTarQubit(node, m_graph_dag));
                }
                else if ((qvec.end() == ctr_iter) && (qvec.end() != tar_iter))
                {
                    qvec.emplace_back(getCtrQubit(node, m_graph_dag));
                }
                else if ((qvec.end() == ctr_iter) && (qvec.end() == tar_iter))
                {
                    QCERR("updateQubits");
                }
                else{}
            }
            else
            {
                auto tar_iter = find(qvec.begin(), qvec.end(), getTarQubit(node, m_graph_dag));
                if (qvec.end() == tar_iter)
                {
                    QCERR("updateQubits");
                }
            }
        }
    }
}

bool GraphMatch::qubitContain(SequenceNode &node, std::vector<size_t> &qvec)
{
    if (qvec.empty())
    {
        return true;
    }
    else
    {
        if (QNodeGateType::DOUBLE_GATE == nodeType(node.m_node_type))
        {
            auto tar_iter = find(qvec.begin(), qvec.end(), getTarQubit(node, m_graph_dag));
            auto ctr_iter = find(qvec.begin(), qvec.end(), getCtrQubit(node, m_graph_dag));
            
            return tar_iter != ctr_iter;
        }
        else
        {
            return qvec.end() != find(qvec.begin(), qvec.end(), getTarQubit(node, m_graph_dag));
        }
    }
}

size_t GraphMatch::getTarQubit(SequenceNode &node, QProgDAG &dag)
{
    auto _node = dag.getVertex(node.m_vertex_num);
    switch (node.m_node_type)
    {
        case -1:
        {
            auto measure_ptr = std::dynamic_pointer_cast<QMeasure>(_node);
            return measure_ptr->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
        }
        break;

        default:
        {
            auto gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(_node);

            QVec qvec;
            gate_ptr->getQuBitVector(qvec);

            auto tar_qubit = QNodeGateType::DOUBLE_GATE == nodeType(node.m_node_type)
                ? qvec[1] : qvec[0];

            return tar_qubit->getPhysicalQubitPtr()->getQubitAddr();

        }
        break;
    }
}

size_t GraphMatch::getCtrQubit(SequenceNode &node, QProgDAG &dag)
{
    auto _node = dag.getVertex(node.m_vertex_num);
    if (QNodeGateType::DOUBLE_GATE == nodeType(node.m_node_type))
    {
        auto gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(_node);

        QVec qvec;
        gate_ptr->getQuBitVector(qvec);
        return qvec[0]->getPhysicalQubitPtr()->getQubitAddr();
    }
    else
    {
        QCERR("get control qubit error");
        throw std::runtime_error("get control qubit");
    }
}

bool GraphMatch::qnodeContain(LayerVector &graph_node_vec, LayerVector &query_node_vec)
{
    vector<int> query, graph;
    for_each(graph_node_vec.begin(), graph_node_vec.end(),
        [&](SequenceNode &val) {graph.emplace_back(val.m_node_type); });
    for_each(query_node_vec.begin(), query_node_vec.end(),
        [&](SequenceNode &val) {query.emplace_back(val.m_node_type); });

    for (const auto &val : query)
    {
        auto iter = find(graph.begin(), graph.end(), val);
        if (iter != graph.end())
        {
            graph.erase(iter);
        }
        else
        {
            return false;
        }
    }
    return true;
}

bool GraphMatch::graphQuery(TopologincalSequence &graph_seq,
    TopologincalSequence &query_seq,
    MatchVector &match_result)
{
    bool graph_match{ true };
    while (graph_match)
    {
        Qnum qvec;
        ResultVector result;
        size_t graph_layer_index{ 0 };
        size_t graph_query_index{ 0 };
        for (auto i = 0; i < query_seq.size();)
        {
            bool layer_match{ false };
            for (int j = graph_layer_index; j < graph_seq.size();)
            {
                if (compareCurLayer(graph_seq[j], query_seq[i], match_result, result, qvec))
                {
                    graph_layer_index = ++j;
                    layer_match = true;
                    break;
                }
                else
                {
                    ++j;
                    if (i == 0)
                    {
                        graph_query_index = j;
                    }
                }
            }

            if (layer_match)
            {
                ++i;
            }
            else
            {
                i = 0;
                qvec.clear();
                result.clear();
                graph_layer_index = ++graph_query_index;
                if (graph_layer_index >= graph_seq.size())
                {
                    break;
                }
            }
        }

        graph_match = result.empty() ? false : true;
        if (!result.empty())
        {
            match_result.emplace_back(result);
        }
    }

    return !match_result.empty();
}

bool GraphMatch::compareCurLayer(SequenceLayer &graph_layer,
                                 SequenceLayer &query_layer,
                                 MatchVector &match_result, 
                                 ResultVector &temp_result,
                                 std::vector<size_t> &qvec)
{
    std::vector<SequenceNode> cur_layer;
    for (auto query_node : query_layer)
    {
        bool node_match{ false };
        for (auto graph_node : graph_layer)
        {
            if (qubitContain(graph_node.first, qvec) &&
                nodeIdentifier(graph_node.first, temp_result) &&
                nodeIdentifier(graph_node.first, match_result) &&
                compareCurQNode(graph_node, query_node))
            {
                cur_layer.emplace_back(graph_node.first);
                node_match = true;
                break;
            }
        }

        if (!node_match)
        {
            temp_result.clear();
            return false;
        }
    }

    updateQubits(cur_layer, qvec);
    temp_result.emplace_back(cur_layer);
    return true;
}

bool GraphMatch::compareCurQNode(std::pair<SequenceNode, std::vector<SequenceNode>> &graph_node, 
                                 std::pair<SequenceNode, std::vector<SequenceNode>> &query_node)
{
    auto query_type = query_node.first.m_node_type;
    auto graph_type = graph_node.first.m_node_type;

    if ((query_type != graph_type) || 
        !qnodeContain(graph_node.second,query_node.second)||
        !compareGateParm(graph_node.first,m_graph_dag,query_node.first,m_query_dag))
    {
        return false;
    }
    else
    {  
        if (QNodeGateType::SINGLE_OR_MEASURE_GATE == nodeType(query_type))
        {
            if (!query_node.second.empty())
            {
                if (QNodeGateType::DOUBLE_GATE == nodeType(query_node.second[0].m_node_type))
                {
                    auto query_tar_qubit = getTarQubit(query_node.first, m_query_dag);
                    auto query_edge_tar_qubit = getTarQubit(query_node.second[0], m_query_dag);
                    auto query_edge_ctr_qubit = getCtrQubit(query_node.second[0], m_query_dag);

                    if (graph_node.second.empty())
                    {
                        return false;
                    }
                    else
                    {
                        auto graph_tar_qubit = getTarQubit(graph_node.first, m_graph_dag);
                        auto graph_edge_tar_qubit = getTarQubit(graph_node.second[0], m_graph_dag);
                        auto graph_edge_ctr_qubit = getCtrQubit(graph_node.second[0], m_graph_dag);

                        return (((query_tar_qubit == query_edge_tar_qubit) && 
                                 (graph_tar_qubit == graph_edge_tar_qubit)) ||
                                ((query_tar_qubit == query_edge_ctr_qubit) && 
                                 (graph_tar_qubit == graph_edge_ctr_qubit)));
                    }
                }
            }
        }
        else //QNodeGateType::DOUBLE_GATE
        {
            if (!query_node.second.empty())
            {
                auto query_tar_qubit = getTarQubit(query_node.first, m_query_dag);
                auto query_ctr_qubit = getCtrQubit(query_node.first, m_query_dag);

                auto graph_tar_qubit = getTarQubit(graph_node.first, m_graph_dag);
                auto graph_ctr_qubit = getCtrQubit(graph_node.first, m_graph_dag);

                for (auto val : query_node.second)
                {
                    if (QNodeGateType::SINGLE_OR_MEASURE_GATE == nodeType(val.m_node_type))
                    {
                        auto query_edge_tar_qubit = getTarQubit(val, m_query_dag);

                        bool is_compare{ false };
                        for (auto graph_val : graph_node.second)
                        {
                            if (graph_val.m_node_type == val.m_node_type)
                            {
                                auto graph_edge_tar_qubit = getTarQubit(graph_val, m_graph_dag);
                                if (((query_tar_qubit == query_edge_tar_qubit) && 
                                     (graph_tar_qubit == graph_edge_tar_qubit)) ||
                                    ((query_ctr_qubit == query_edge_tar_qubit)  && 
                                     (graph_ctr_qubit == graph_edge_tar_qubit)))
                                {
                                    is_compare = true;
                                    break;
                                }
                            }
                        }

                        if (!is_compare)
                        {
                            return false;
                        }
                    }
                    else //QNodeGateType::DOUBLE_GATE
                    {
                        auto query_edge_tar_qubit = getTarQubit(val, m_query_dag);
                        auto query_edge_ctr_qubit = getCtrQubit(val, m_query_dag);

                        bool is_compare{ false };
                        for (auto graph_val : graph_node.second)
                        {
                            if (graph_val.m_node_type == val.m_node_type)
                            {
                                auto graph_edge_tar_qubit = getTarQubit(graph_val, m_graph_dag);
                                auto graph_edge_ctr_qubit = getCtrQubit(graph_val, m_graph_dag);

                                if (((query_tar_qubit == query_edge_tar_qubit) && 
                                     (graph_tar_qubit == graph_edge_tar_qubit)) ||
                                    ((query_ctr_qubit == query_edge_tar_qubit) && 
                                     (graph_ctr_qubit == graph_edge_tar_qubit)) ||
                                    ((query_tar_qubit == query_edge_ctr_qubit) && 
                                     (graph_tar_qubit == graph_edge_ctr_qubit)) ||
                                    ((query_ctr_qubit == query_edge_ctr_qubit) && 
                                     (graph_ctr_qubit == graph_edge_ctr_qubit)))

                                {
                                    is_compare = true;
                                    break;
                                }
                            }
                        }

                        if (!is_compare)
                        {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }
}

void GraphMatch::SequenceToQProg(TopologincalSequence &seq, QProg &prog, QProgDAG &dag)
{
    for (auto layer: seq)
    {
        SequenceToQProg(layer, prog, dag);
    }
}

void GraphMatch::SequenceToQProg(SequenceLayer &layer, QProg &prog, QProgDAG &dag)
{
    for (auto node : layer)
    {
        prog.pushBackNode(dag.getVertex(node.first.m_vertex_num));
    }
}

void QubitsCompare::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
    if (nullptr == cur_node || nullptr == cur_node->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    QVec qubits_vector;
    cur_node->getQuBitVector(qubits_vector);

    switch (cur_node->getQGate()->getGateType())
    {
    case GateType::PAULI_X_GATE:
    case GateType::PAULI_Y_GATE:
    case GateType::PAULI_Z_GATE:
    case GateType::HADAMARD_GATE:
    case GateType::X_HALF_PI:
    case GateType::Y_HALF_PI:
    case GateType::Z_HALF_PI:
    case GateType::T_GATE:
    case GateType::S_GATE:
    case GateType::RX_GATE:
    case GateType::RY_GATE:
    case GateType::RZ_GATE:
    case GateType::U1_GATE:
    {
        auto tar_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();
        auto tar_iter = find(m_qubit_vec.begin(), m_qubit_vec.end(), tar_qubit);
        if (m_qubit_vec.end() == tar_iter)
        {
            m_qubit_vec.emplace_back(tar_qubit);
        }
    }
    break;

    case GateType::CNOT_GATE:
    case GateType::CZ_GATE:
    case GateType::ISWAP_GATE:
    case GateType::SQISWAP_GATE:
    case GateType::CPHASE_GATE:
    {
        auto tar_qubit = qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr();
        auto ctr_qubit = qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr();

        if (m_qubit_vec.end() == find(m_qubit_vec.begin(), m_qubit_vec.end(), tar_qubit))
        {
            m_qubit_vec.emplace_back(tar_qubit);
        }

        if (m_qubit_vec.end() == find(m_qubit_vec.begin(), m_qubit_vec.end(), ctr_qubit))
        {
            m_qubit_vec.emplace_back(ctr_qubit);
        }
    }
    break;

    default:
        QCERR("do not support this gate type");
        throw invalid_argument("do not support this gate type");
        break;
    }
}

void GraphMatch::insertQNodes(MatchVector &result, TopologincalSequence &replace_seq,
    TopologincalSequence &graph_seq, QuantumMachine* qvm)
{
    QubitsCompare construct;
    for (const auto & replace : result)
    {
        std::vector<size_t> compare_vec;
        std::map<size_t, size_t> compare_map;

        for (const auto &layer : replace)
        {
            for (auto node : layer)
            {
                auto tar_qubit = getTarQubit(node, m_graph_dag);
                auto tar_iter = find(compare_vec.begin(), compare_vec.end(), tar_qubit);
                if (compare_vec.end() == tar_iter)
                {
                    compare_vec.emplace_back(tar_qubit);
                }

                if (QNodeGateType::DOUBLE_GATE==nodeType(node.m_node_type))
                {
                    auto ctr_qubit = getCtrQubit(node, m_graph_dag);
                    auto ctr_iter = find(compare_vec.begin(), compare_vec.end(), ctr_qubit);
                    if (compare_vec.end() == ctr_iter)
                    {
                        compare_vec.emplace_back(ctr_qubit);
                    }
                }
            }
        }

        if (compare_vec.size() != m_compare_vec.size())
        {
            QCERR("Qubits compare error!");
            return;
        }

        //sort(compare_vec.begin(), compare_vec.end());
        for (auto i = 0; i < m_compare_vec.size(); ++i)
        {
            compare_map.insert(make_pair(m_compare_vec[i], compare_vec[i]));
        }

        SequenceLayer prog_layer;
        TopologincalSequence prog_seq;

        auto layer_size = result.front().size();
        if (layer_size <= replace_seq.size()) //H 1 -> X 1 , Y 1 , Z 1
        {
            LayerVector layer_vector;
            TopologincalSequence seq;

            for (auto i = 0; i < layer_size - 1; ++i)
            {
                SequenceLayer node_layer;
                for (auto &node : replace_seq[i])
                {
                    auto seq_node = contsructQNode(compare_map, node.first, qvm, construct);
                    node_layer.emplace_back(make_pair(seq_node, layer_vector));
                }

                seq.emplace_back(node_layer);
            }

            SequenceLayer node_layer;
            for (auto i = layer_size - 1; i < replace_seq.size(); ++i)
            {
                for (auto &node : replace_seq[i])
                {
                    auto seq_node = contsructQNode(compare_map, node.first, qvm, construct);
                    node_layer.emplace_back(make_pair(seq_node, layer_vector));
                }
            }
            seq.emplace_back(node_layer);

            for (auto i = 0; i < replace.size(); ++i)
            {
                auto layer_num = getLayerNumber(graph_seq, replace[i].front().m_vertex_num);

                for (auto node : seq[i])
                {
                    graph_seq[layer_num].emplace_back(make_pair(node.first, layer_vector));
                }
            }
        }
        else  //X1 , Y 1 , Z 1 -> H 1 , T 1
        {
            LayerVector layer_vector;
            TopologincalSequence seq;

            for (auto i = 0; i < replace_seq.size(); ++i)
            {
                SequenceLayer node_layer;
                for (auto &node : replace_seq[i])
                {
                    auto seq_node = contsructQNode(compare_map, node.first, qvm, construct);
                    node_layer.emplace_back(make_pair(seq_node, layer_vector));
                }

                seq.emplace_back(node_layer);
            }


            for (auto i = 0; i < replace_seq.size(); ++i)
            {
                auto layer_num = getLayerNumber(graph_seq, replace[i].front().m_vertex_num);

                for (auto &node : seq[i])
                {
                    graph_seq[layer_num].emplace_back(make_pair(node.first, layer_vector));
                }
            }
        }
    }

    for (const auto &replace : result)
    {
        for (const auto &layer : replace)
        {
            for (const auto &node : layer)
            {
                deleteSequenceNode(node.m_vertex_num, graph_seq);
            }
        }
    }
}

SequenceNode GraphMatch::contsructQNode(const std::map<size_t, size_t> &compare_map,
    SequenceNode &node, QuantumMachine* qvm, QubitsCompare &construct)
{
    size_t vertice_num{ 0 };
    auto node_type = node.m_node_type;
    switch (node_type)
    {
        case GateType::PAULI_X_GATE:
        case GateType::PAULI_Y_GATE:
        case GateType::PAULI_Z_GATE:
        case GateType::HADAMARD_GATE:
        case GateType::T_GATE:
        case GateType::S_GATE:
        case GateType::X_HALF_PI:
        case GateType::Y_HALF_PI:
        case GateType::Z_HALF_PI:
        {
            auto tar_qubit = getTarQubit(node, m_replace_dag);
            auto rel_qubit = qvm->allocateQubitThroughPhyAddress(compare_map.find(tar_qubit)->second);
            //auto gate = construct.m_singleGateFunc.find(node.first.m_node_type)->second(rel_qubit);
                
            auto gate = construct.m_singleGateFunc.find(node_type)->second(rel_qubit);
            vertice_num = m_graph_dag.addVertex(&gate);
        }
        break;

        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::RZ_GATE:
        case GateType::U1_GATE:
        {
            auto tar_qubit = getTarQubit(node, m_replace_dag);

            auto _node = m_replace_dag.getVertex(node.m_vertex_num);
            auto gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(_node);
            angleParameter * parm = dynamic_cast<angleParameter*>(gate_ptr->getQGate());
            
            auto rel_qubit = qvm->allocateQubitThroughPhyAddress(compare_map.find(tar_qubit)->second);
            //auto gate = construct.m_angleGateFunc.find(node_type)->second(rel_qubit, parm->getParameter());
            auto gate = construct.m_angleGateFunc.find(node_type)->second(rel_qubit, parm->getParameter());
            vertice_num = m_graph_dag.addVertex(&gate);
        }
        break;

        case GateType::CNOT_GATE:
        case GateType::CZ_GATE:
        case GateType::ISWAP_GATE:
        case GateType::SQISWAP_GATE:
        {
            auto tar_qubit = getTarQubit(node, m_replace_dag);
            auto ctr_qubit = getCtrQubit(node, m_replace_dag);

            auto rel_tar_qubit = qvm->allocateQubitThroughPhyAddress(compare_map.find(tar_qubit)->second);
            auto rel_ctr_qubit = qvm->allocateQubitThroughPhyAddress(compare_map.find(ctr_qubit)->second);

            //auto gate = construct.m_doubleGateFunc.find(node_type)->second(rel_ctr_qubit, rel_tar_qubit);
            auto gate = construct.m_doubleGateFunc.find(node_type)->second(rel_ctr_qubit, rel_tar_qubit);
            vertice_num = m_graph_dag.addVertex(&gate);
        }
        break;

        case GateType::CPHASE_GATE:
        {
            auto tar_qubit = getTarQubit(node, m_replace_dag);
            auto ctr_qubit = getCtrQubit(node, m_replace_dag);

            auto rel_tar_qubit = qvm->allocateQubitThroughPhyAddress(compare_map.find(tar_qubit)->second);
            auto rel_ctr_qubit = qvm->allocateQubitThroughPhyAddress(compare_map.find(ctr_qubit)->second);

            auto _node = m_replace_dag.getVertex(node.m_vertex_num);
            auto gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(_node);
            angleParameter * parm = dynamic_cast<angleParameter*>(gate_ptr->getQGate());

            //auto gate = construct.m_doubleAngleGateFunc.find(node_type)->second(rel_ctr_qubit, rel_tar_qubit, parm->getParameter());
            auto gate = construct.m_doubleAngleGateFunc.find(node_type)->second(rel_ctr_qubit, rel_tar_qubit, parm->getParameter());
            vertice_num = m_graph_dag.addVertex(&gate);
        }
        break;

        default:
        {
            QCERR("do not support this gate type");
            throw invalid_argument("do not support this gate type");
            break;
        }
    }

    SequenceNode gate_node{ node_type ,vertice_num };
    return gate_node;
}

void QubitsCompare::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node, std::shared_ptr<QNode> parent_node)
{
    QCERR("can not query & replace measure node is null");
    throw invalid_argument("query & replace measure node error");
}

void QubitsCompare::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
{
    QCERR("Does not support ControlFlowNode ");
    throw std::runtime_error("Does not support ControlFlowNode");
}

void QubitsCompare::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
{
    Traversal::traversal(cur_node, false, *this);
}

void QubitsCompare::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
{
    Traversal::traversal(cur_node, *this);
}

void QubitsCompare::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
{
    QCERR("Does not support ClassicalProg ");
    throw std::runtime_error("Does not support ClassicalProg");
}

bool GraphMatch::qubitsCompare(std::vector<size_t> query_qvec,
                               std::vector<size_t> replace_qvec)
{
    m_compare_vec = replace_qvec;

    std::sort(query_qvec.begin(), query_qvec.end());
    std::sort(replace_qvec.begin(), replace_qvec.end());
    if (query_qvec != replace_qvec)
    {
        QCERR("Qubits does not correspond");
        return false;
    }

    return true;
}


QubitsCompare::QubitsCompare()
{
    m_singleGateFunc.insert(make_pair(GateType::HADAMARD_GATE, H));
    m_singleGateFunc.insert(make_pair(GateType::T_GATE, T));
    m_singleGateFunc.insert(make_pair(GateType::S_GATE, S));

    m_singleGateFunc.insert(make_pair(GateType::PAULI_X_GATE, X));
    m_singleGateFunc.insert(make_pair(GateType::PAULI_Y_GATE, Y));
    m_singleGateFunc.insert(make_pair(GateType::PAULI_Z_GATE, Z));

    m_singleGateFunc.insert(make_pair(GateType::X_HALF_PI, X1));
    m_singleGateFunc.insert(make_pair(GateType::Y_HALF_PI, Y1));
    m_singleGateFunc.insert(make_pair(GateType::Z_HALF_PI, Z1));

    m_doubleGateFunc.insert(make_pair(GateType::CNOT_GATE, CNOT));
    m_doubleGateFunc.insert(make_pair(GateType::CZ_GATE, CZ));
    //m_doubleGateFunc.insert(make_pair(GateType::ISWAP_GATE, iSWAP));
    m_doubleGateFunc.insert(make_pair(GateType::SQISWAP_GATE, SqiSWAP));

    m_angleGateFunc.insert(make_pair(GateType::RX_GATE, RX));
    m_angleGateFunc.insert(make_pair(GateType::RY_GATE, RY));
    m_angleGateFunc.insert(make_pair(GateType::RZ_GATE, RZ));
    m_angleGateFunc.insert(make_pair(GateType::U1_GATE, U1));

    m_doubleAngleGateFunc.insert(make_pair(GateType::CPHASE_GATE, CR));
}


bool GraphMatch::compareGateParm(SequenceNode &graph_node, QProgDAG &graph_dag,
    SequenceNode &query_node, QProgDAG &query_dag)
{
    auto g_node = graph_dag.getVertex(graph_node.m_vertex_num);
    auto q_node = query_dag.getVertex(query_node.m_vertex_num);
    switch (graph_node.m_node_type)
    {
        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::RZ_GATE:
        case GateType::U1_GATE:
        case GateType::CPHASE_GATE:
        {
            auto g_gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(g_node);
            auto q_gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(q_node);

            auto g_parm_ptr = dynamic_cast<angleParameter*>(g_gate_ptr->getQGate());
            auto q_parm_ptr = dynamic_cast<angleParameter*>(q_gate_ptr->getQGate());
            
            double g_parm = g_parm_ptr->getParameter();
            double q_parm = q_parm_ptr->getParameter();

            return fabs(g_parm - q_parm) <= precision;
        }
        break;

        default: return true;
    }
}