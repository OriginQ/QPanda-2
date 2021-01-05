#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
#include "Core/QuantumCircuit/QGate.h"
#include <memory>
#include <cmath>

USING_QPANDA
using namespace std;

void GraphMatch::_replace_node(const ResultVector &match_vector, TopologSequence<SequenceNode> &seq)
{
    for (const auto &result : match_vector)
    {
        for (const auto &node : result)
        {
            for (auto &layer : seq)
            {
                for (auto iter = layer.begin(); iter != layer.end();)
                {
                    iter = (iter->first.m_vertex_num == node.m_vertex_num) ?
                        layer.erase(iter) : ++iter;
                }
            }
        }
    }
}

static TopoNode get_layer_node(const SequenceNode &node, const TopologSequence<SequenceNode> &seq)
{
    for (const auto &layer : seq)
    {
        for (const auto &layer_node : layer)
        {
            if (layer_node.first.m_vertex_num == node.m_vertex_num)
            {
                return layer_node;
            }
        }
    }
}

static size_t get_layer_num(const TopologSequence<SequenceNode> &graph_seq,size_t vertice_num)
{
	size_t i = 0;
    for (auto layer_itr = graph_seq.begin(); layer_itr != graph_seq.end(); ++layer_itr, ++i)
    {
        for (auto node_iter = layer_itr->begin(); node_iter != layer_itr->end(); ++node_iter)
        {
            if (vertice_num == node_iter->first.m_vertex_num)
            {
                return i;
            }
        }
    }

	QCERR_AND_THROW(run_fail, "Error: failed to get_layer_num.");
}

Qnum GraphMatch::_get_qubit_vector(const SequenceNode &node, QProgDAG<GateNodeInfo> &dag)
{
    auto vertex_node = dag.get_vertex_node(node.m_vertex_num);
	std::shared_ptr<QNode> p_QNode = *(vertex_node.m_itr);
    switch (node.m_node_type)
    {
	case SequenceNodeType::MEASURE:
	{
		auto measure_ptr = std::dynamic_pointer_cast<QMeasure>(p_QNode);
		return { measure_ptr->getQuBit()->getPhysicalQubitPtr()->getQubitAddr() };
	}
	case SequenceNodeType::RESET:
	{
		auto reset_ptr = std::dynamic_pointer_cast<QReset>(p_QNode);
		return { reset_ptr->getQuBit()->getPhysicalQubitPtr()->getQubitAddr() };
	}
    default:
	{
		auto gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(p_QNode);
		if (gate_ptr == nullptr)
		{
			QCERR_AND_THROW(init_fail, "Error: failed to transfer to QGate node.");
		}

		QVec qvec;
		gate_ptr->getQuBitVector(qvec);

		Qnum qubit_addr_vec;
		for_each(qvec.begin(), qvec.end(), [&](Qubit* qubit)
		{
			qubit_addr_vec.emplace_back(qubit->getPhysicalQubitPtr()->getQubitAddr());
		});
		return qubit_addr_vec;
	}
    }
}

bool GraphMatch::_compare_edge(const LayerVector &graph_node_vec, const LayerVector &query_node_vec)
{
    std::vector<int> query, graph;
    for_each(graph_node_vec.begin(), graph_node_vec.end(),
        [&](const SequenceNode &val) {graph.emplace_back(val.m_node_type); });
    for_each(query_node_vec.begin(), query_node_vec.end(),
        [&](const SequenceNode &val) {query.emplace_back(val.m_node_type); });

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

bool GraphMatch::query(TopologSequence<SequenceNode> &graph_seq, TopologSequence<SequenceNode> &query_seq,
                       ResultVector &match_result)
{
    if (m_query_dag.is_connected_graph())
    { 
        std::cout << "query graph is non-connect graph" << std::endl;
        return false;
    }

	TopoLayer  node_match_vector;
    for (auto &query_layer : query_seq)
    {
        for (auto &query_node : query_layer)
        {
            std::vector<SequenceNode> node_vector;
            for (auto &graph_layer : graph_seq)
            {
				for (auto &graph_node : graph_layer)
                {
                    if (_compare_node(graph_node, query_node))
                    {
                        node_vector.emplace_back(graph_node.first);
                    }
                }
            }

            if (node_vector.empty())
            {
                return false;
            }
            else
            {
                node_match_vector.emplace_back(make_pair(query_node.first, node_vector));
            }
        }
    }

    auto _find = [&](SequenceNode& node)
    {
        for (const auto& _node : node_match_vector)
        {
            for (const auto& match_node : _node.second)
            {
                if (match_node.m_vertex_num == node.m_vertex_num)
                {
                    return true;
                }
            }
        }
        return false;
    };

    ResultVector result_vector;
    auto match_num = node_match_vector.size();
	auto iter = ++(query_seq.begin());
    for (; iter != query_seq.end(); ++iter)
    {
        for (auto &node : (*iter))
        {
            LayerVector pre_query_node_vector;
            _get_pre_node(node.first.m_vertex_num, query_seq, pre_query_node_vector);

            LayerVector cur_node_vector;
            for (const auto &match_node : node_match_vector)
            {
                if (match_node.first.m_vertex_num == node.first.m_vertex_num)
                {
                    cur_node_vector = match_node.second;
                    break;
                }
            }

            for (auto &match_node : cur_node_vector)
            {
                LayerVector pre_match_node_vector;
                _get_pre_node(match_node.m_vertex_num, graph_seq, pre_match_node_vector);

                if (1 == pre_query_node_vector.size())
                {
                    if (1 == pre_match_node_vector.size())
                    {
                        auto q_node_first = get_layer_node(pre_query_node_vector[0], query_seq);
                        auto m_node_first = get_layer_node(pre_match_node_vector[0], graph_seq);

                        if (_compare_node(m_node_first, q_node_first) && _find(pre_match_node_vector[0]))
                        {
                            LayerVector temp = { match_node ,pre_match_node_vector[0] };
                            result_vector.emplace_back(temp);
                        }
                    }
                    else if (2 == pre_match_node_vector.size())
                    {
                        auto q_node_first = get_layer_node(pre_query_node_vector[0], query_seq);
                        auto m_node_first = get_layer_node(pre_match_node_vector[0], graph_seq);
                        auto m_node_second = get_layer_node(pre_match_node_vector[1], graph_seq);

                        bool compare_first = _compare_node(m_node_first, q_node_first);
                        bool compare_second = _compare_node(m_node_second, q_node_first);

                        if (compare_first && _find(pre_match_node_vector[0]))
                        {
                            LayerVector temp = { match_node ,pre_match_node_vector[0] };
                            result_vector.emplace_back(temp);
                        }
                        else if (compare_second && _find(pre_match_node_vector[1]))
                        {
                            LayerVector temp = { match_node ,pre_match_node_vector[1] };
                            result_vector.emplace_back(temp);
                        }
                        else
                        {}
                    }
                    else
                    {}
                }
                else
                {
                    if (2 == pre_match_node_vector.size())
                    {
                        auto q_node_first = get_layer_node(pre_query_node_vector[0], query_seq);
                        auto q_node_second = get_layer_node(pre_query_node_vector[1], query_seq);

                        auto m_node_first = get_layer_node(pre_match_node_vector[0], graph_seq);
                        auto m_node_second = get_layer_node(pre_match_node_vector[1], graph_seq);

                        bool cur_node_match = (_compare_node(m_node_first, q_node_first) && _compare_node(m_node_second, q_node_second)) ||
                                              (_compare_node(m_node_first, q_node_second) && _compare_node(m_node_second, q_node_first));

                        if (cur_node_match && _find(pre_match_node_vector[0]) && _find(pre_match_node_vector[1]))
                        {
                            LayerVector temp = { match_node ,pre_match_node_vector[0] ,pre_match_node_vector[1] };
                            result_vector.emplace_back(temp);
                        }
                    }
                    else
                    {}
                }
            }
        }
    }

    if (result_vector.empty())
    {
        return false;
    }
    else
    {
        for_each(result_vector.begin(), result_vector.end(), [&](LayerVector& layer)
        {
            sort(layer.begin(), layer.end(), [&](SequenceNode a, SequenceNode b)
            {return a.m_vertex_num < b.m_vertex_num; });
        });

        match_result = { result_vector.front() };
        for (auto &val : result_vector)
        {
            bool intersection{ false };
            LayerVector match_union;

            for (auto iter = match_result.begin(); iter != match_result.end();)
            {
                LayerVector match_intersection;
                std::set_intersection(val.begin(), val.end(), iter->begin(), iter->end(), std::back_inserter(match_intersection));
                if (!match_intersection.empty())
                {
                    intersection = true;
                    if (match_union.empty())
                    {
                        std::set_union(val.begin(), val.end(), iter->begin(), iter->end(), std::back_inserter(match_union));
                    }
                    else
                    {
                        LayerVector temp_union;
                        std::set_union(match_union.begin(), match_union.end(), iter->begin(), iter->end(), std::back_inserter(temp_union));
                        match_union = temp_union;
                    }
                    iter = match_result.erase(iter);
                }
                else
                {
                    ++iter;
                }
            }

            match_result.emplace_back(intersection ? match_union : val);
        }

        for (auto iter = match_result.begin(); iter != match_result.end();)
        {
            iter = iter->size() != match_num ? match_result.erase(iter) : ++iter;
        }
    }

	return true;
}

bool GraphMatch::_compare_node(TopoNode &graph_node, TopoNode &query_node)
{
    if ((query_node.first.m_node_type != graph_node.first.m_node_type) ||
        !_compare_edge(graph_node.second, query_node.second) ||
        !_compare_parm(graph_node.first, query_node.first))
    {
        return false;
    }
    else
    {
        auto graph_qvec = _get_qubit_vector(graph_node.first, m_graph_dag);
        auto query_qvec = _get_qubit_vector(query_node.first, m_query_dag);

        if (query_node.second.empty())
        {
            return true;
        }
        else
        {
            if (1 == query_qvec.size())
            {
                auto query_connected_qvec = _get_qubit_vector(query_node.second.front(), m_query_dag);

                if (2 == query_connected_qvec.size())
                {
                    if (graph_node.second.empty())
                    {
                        return false;
                    }
                    else
                    {
                        auto graph_connected_qvec = _get_qubit_vector(graph_node.second.front(), m_graph_dag);
                        return (((query_qvec[0] == query_connected_qvec[1]) &&
                                 (graph_qvec[0] == graph_connected_qvec[1])) ||
                                ((query_qvec[0] == query_connected_qvec[0]) &&
                                 (graph_qvec[0] == graph_connected_qvec[0])));
                    }
                }
            }
            else //QNodeGateType::DOUBLE_GATE
            {
                for (auto val : query_node.second)
                {
                    bool is_compare{ false };
                    auto query_connected_qvec = _get_qubit_vector(val, m_query_dag);

                    if (1 == query_connected_qvec.size())
                    {
                        for (auto graph_val : graph_node.second)
                        {
                            auto graph_connected_qvec = _get_qubit_vector(graph_val, m_graph_dag);
                            if (graph_val.m_node_type == val.m_node_type)
                            {
                                if (((query_qvec[1] == query_connected_qvec[0]) &&
                                     (graph_qvec[1] == graph_connected_qvec[0])) ||

                                    ((query_qvec[0] == query_connected_qvec[0]) &&
                                     (graph_qvec[0] == graph_connected_qvec[0])))
                                {
                                    is_compare = true;
                                    break;
                                }
                            }
                        }
                    }
                    else //QNodeGateType::DOUBLE_GATE
                    {
                        for (auto graph_val : graph_node.second)
                        {
                            auto graph_connected_qvec = _get_qubit_vector(graph_val, m_graph_dag);
                            if (graph_val.m_node_type == val.m_node_type)
                            {
                                if (((query_qvec[1] == graph_connected_qvec[1]) &&
                                     (graph_qvec[1] == graph_connected_qvec[1])) ||

                                    ((query_qvec[0] == query_connected_qvec[1]) &&
                                     (graph_qvec[0] == graph_connected_qvec[1])) ||

                                    ((query_qvec[1] == query_connected_qvec[0]) &&
                                     (graph_qvec[1] == graph_connected_qvec[0])) ||

                                    ((query_qvec[0] == query_connected_qvec[0]) &&
                                     (graph_qvec[0] == graph_connected_qvec[0])))

                                {
                                    is_compare = true;
                                    break;
                                }
                            }
                        }
                    }

                    if (!is_compare)
                    {
                        return false;
                    }
                }
            }

            return true;
        }
    }
}

void GraphMatch::_convert_prog(TopologSequence<SequenceNode> &seq, QProg &prog)
{
    for (auto &layer: seq)
    {
        for (auto &node : layer)
        {
			const auto& vertex_node = m_graph_dag.get_vertex_node(node.first.m_vertex_num);
            prog.pushBackNode(*(vertex_node.m_itr));
        }
    }
}

void GraphMatch::_convert_node(ResultVector &match_vector, TopologSequence<SequenceNode> &replace_seq,
	TopologSequence<SequenceNode> &graph_seq, QuantumMachine* qvm)
{
    if (m_replace_dag.is_connected_graph())
    {
        std::cout << "replace graph is non-connected graph" << std::endl;
        return;
    }

    for (const auto & result : match_vector)
    {
        std::vector<size_t> compare_vec;
        std::map<size_t, size_t> compare_map;

        for (const auto &node : result)
        {
            auto qubit_vec = _get_qubit_vector(node, m_graph_dag);

            for (const auto &qubit_addr : qubit_vec)
            {
                if (compare_vec.end() == find(compare_vec.begin(), compare_vec.end(), qubit_addr))
                {
                    compare_vec.emplace_back(qubit_addr);
                }
            }
        }

        //sort(compare_vec.begin(), compare_vec.end());
        for (auto i = 0; i < m_compare_vec.size(); ++i)
        {
            compare_map.insert(make_pair(m_compare_vec[i], compare_vec[i]));
        }

        size_t insert_layer_num{0};
        std::map<size_t, size_t> qubit_layer_map;
        for (const auto &node : result)
        {
            auto node_layer_num = get_layer_num(graph_seq, node.m_vertex_num);
            auto qubit_vec = _get_qubit_vector(node, m_graph_dag);
            for (const auto &qubit_addr : qubit_vec)
            {
                auto iter = qubit_layer_map.find(qubit_addr);
                if (qubit_layer_map.cend() == iter)
                {
                    qubit_layer_map.insert(make_pair(qubit_addr, node_layer_num));
                    insert_layer_num = node_layer_num > insert_layer_num ?
                        node_layer_num : insert_layer_num;
                }
            }
        }

		std::vector<SequenceNode> connect_vector;      //always empty
        for (auto &layer : replace_seq)
        {
            for (auto &node : layer)
            {
                SequenceNode gate_node;
                _convert_gate(node.first, qvm, compare_map, gate_node);
				graph_seq.at(insert_layer_num).emplace_back(make_pair(gate_node, connect_vector));
            }
        }
    }

    _replace_node(match_vector, graph_seq);
}

bool GraphMatch::_compare_qnum(Qnum query_qvec, Qnum replace_qvec)
{
    m_compare_vec = replace_qvec;

    std::sort(query_qvec.begin(), query_qvec.end());
    std::sort(replace_qvec.begin(), replace_qvec.end());

    return query_qvec == replace_qvec ? true : false;
}

bool GraphMatch::_compare_parm(SequenceNode &graph_node, SequenceNode &query_node)
{
    auto g_node = *(m_graph_dag.get_vertex_node(graph_node.m_vertex_num).m_itr);
    auto q_node = *(m_query_dag.get_vertex_node(graph_node.m_vertex_num).m_itr);
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

            auto g_parm_ptr = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(g_gate_ptr->getQGate());
            auto q_parm_ptr = dynamic_cast<QGATE_SPACE::AbstractSingleAngleParameter*>(q_gate_ptr->getQGate());
            
            double g_parm = g_parm_ptr->getParameter();
            double q_parm = q_parm_ptr->getParameter();

            return fabs(g_parm - q_parm) <= 1e-6;
        }
        break;

        default:  return true;
    }
}

void GraphMatch::_get_pre_node(size_t node_num, TopologSequence<SequenceNode> &query_seq, LayerVector &result)
{
    for (const auto &query_layer : query_seq)
    {
        for (const auto &query_node : query_layer)
        {
            for (const auto &connected_node : query_node.second)
            {
                if (connected_node.m_vertex_num == node_num)
                {
                    result.emplace_back(query_node.first);
                }
            }
        }
    }
}

void GraphMatch::_convert_gate(SequenceNode& old_node, QuantumMachine* qvm, std::map<size_t, size_t> &compare_map, SequenceNode& new_node)
{
    QVec qvec;
    auto qubit_vec = _get_qubit_vector(old_node, m_replace_dag);
    try
    {
        for_each(qubit_vec.begin(), qubit_vec.end(), [&](size_t qubit_addr)
        {
            qvec.emplace_back(qvm->allocateQubitThroughPhyAddress(compare_map.find(qubit_addr)->second));
        });

        auto _node = *(m_replace_dag.get_vertex_node(old_node.m_vertex_num).m_itr);
        if (NodeType::GATE_NODE == _node->getNodeType())
        {
            auto gate_ptr = std::dynamic_pointer_cast<AbstractQGateNode>(_node);
            auto temp_gate =  copy_qgate(gate_ptr->getQGate(), qvec);
            new_node.m_node_type = old_node.m_node_type;
			NodeIter tmp_iter = m_graph_dag.add_gate(dynamic_pointer_cast<QNode>(temp_gate.getImplementationPtr()));
			GateNodeInfo tmp_node(tmp_iter);
            new_node.m_vertex_num = m_graph_dag.add_vertex(tmp_node);
        }
        else
        {
            QCERR("node type error");
            throw invalid_argument("node type error");
        }
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw run_fail(e.what());
    }
}
