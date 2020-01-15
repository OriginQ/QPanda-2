#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include <algorithm>
USING_QPANDA
using namespace std;

size_t QProgDAG::add_vertex(std::shared_ptr<QNode> node)
{
    m_dag_prog.pushBackNode(node);
	QProgDAG::NodeInfo node_info(m_dag_prog.getLastNodeIter());

	if (GATE_NODE == node->getNodeType())
	{
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(node);
		node_info.m_dagger = p_gate->isDagger();
		p_gate->getQuBitVector(node_info.m_qubits_vec);
		p_gate->getControlVector(node_info.m_control_vec);
	}
	
    return add_vertex(node_info);
}

void QProgDAG::add_qubit_map(size_t tar_qubit, size_t vertice_num)
{
    auto tar_iter = find(m_qubit_vec.begin(), m_qubit_vec.end(), tar_qubit);
    if (m_qubit_vec.end() == tar_iter)
    {
        m_qubit_vec.emplace_back(tar_qubit);
    }

    auto iter = qubit_vertices_map.find(tar_qubit);
    if (iter != qubit_vertices_map.end())
    {
        size_t in_vertex_num = iter->second.back();
        add_edge(in_vertex_num, vertice_num);
        qubit_vertices_map[iter->first].emplace_back(vertice_num);
    }
    else
    {
        std::vector<size_t> vertice_vec = { vertice_num };
        qubit_vertices_map.insert(make_pair(tar_qubit, vertice_vec));
    }
}


size_t QProgDAG::add_vertex(const NodeInfo& node_info)
{
    auto vertice_num = m_vertices_map.size();
    m_vertices_map.insert(make_pair(vertice_num, node_info));
    return vertice_num;
}

void QProgDAG::add_edge(size_t in_num, size_t out_num)
{
    for (auto val : m_edges_vector)
    {
        if (val.first == in_num && val.second == out_num)
        {
            return;
        }
    }
    m_edges_vector.emplace_back(make_pair(in_num, out_num));
}
   
void QProgDAG::get_adjacency_matrix(const vertices_map &dag_map, AdjacencyMatrix & matrix)
{
    matrix = AdjacencyMatrix::Zero(dag_map.size(), dag_map.size());

    for (const auto &vertice : dag_map)
    {
        for (const auto &edge : m_edges_vector)
        {
            if (edge.first == vertice.first)
            {
                matrix(edge.first, edge.second) = 1;
            }
        }
    }
}


std::shared_ptr<QNode> QProgDAG::get_vertex(size_t vertice_num)  const
{
	return *(get_vertex_nodeIter(vertice_num));
}

NodeIter QProgDAG::get_vertex_nodeIter(size_t vertice_num) const
{
	if (m_vertices_map.size() <= vertice_num)
	{
		QCERR("vertice_num error");
		throw std::runtime_error("vertice_num error");
	}
	return m_vertices_map.find(vertice_num)->second.m_itr;
}

void QProgDAG::getTopologincalSequence(TopologicalSequence &seq)
{
    AdjacencyMatrix matrix;
    get_adjacency_matrix(m_vertices_map, matrix);

    auto col_mat = matrix.colwise().sum();
    AdjacencyMatrix flag_mat = AdjacencyMatrix::Zero(2, m_vertices_map.size());
    for (auto i = 0; i < m_vertices_map.size(); ++i)
    {
        flag_mat(0, i) = col_mat(0, i);
    }

    while (!flag_mat.row(1).minCoeff())
    {
        SequenceLayer seq_layer;
        _get_cur_layer_vertices(flag_mat, seq_layer);
        seq.emplace_back(seq_layer);
    }
}


void QProgDAG::_get_cur_layer_vertices(AdjacencyMatrix &matrix, SequenceLayer &seq_layer)
{
    auto count = m_vertices_map.size();
    for (auto i = 0; i < count; ++i)
    {
        if ((matrix(1, i) == 0) && (matrix(0, i) == 0))
        {
            SequenceNode node = construct_sequence_node(i);
            std::vector<SequenceNode> connected_vec;
            for (const auto &edge : m_edges_vector)
            {
                if (edge.first == i)
                {
                    connected_vec.emplace_back(construct_sequence_node(edge.second));
                }
            }
            seq_layer.emplace_back(make_pair(node, connected_vec));
            matrix(1, i) = -1;
        }
    }
    for (auto i = 0; i < count; ++i)
    {
        if ((matrix(1, i) == -1) && (matrix(0, i) == 0))
        {
            for (const auto &edge: m_edges_vector)
            {
                if (edge.first == i)
                {
                    --matrix(0, edge.second);
                }
            }
            matrix(1, i) = 1;
        }
    }
}

SequenceNode QProgDAG::construct_sequence_node(size_t vertice)
{
    SequenceNode node;
    QNode * node_ptr = (*(m_vertices_map.find(vertice)->second.m_itr)).get();
    if (NodeType::GATE_NODE == node_ptr->getNodeType())
    {
        auto pQGate = dynamic_cast<AbstractQGateNode*>(node_ptr);
        node.m_node_type = pQGate->getQGate()->getGateType();
        node.m_vertex_num = vertice;
    }
    else if (NodeType::MEASURE_GATE == node_ptr->getNodeType())
    {
        auto pMeasure = dynamic_cast<AbstractQuantumMeasure*>(node_ptr);
        node.m_node_type = SequenceNodeType::MEASURE;
        node.m_vertex_num = vertice;
    }
	else if (NodeType::RESET_NODE == node_ptr->getNodeType())
	{
		auto pMeasure = dynamic_cast<AbstractQuantumMeasure*>(node_ptr);
		node.m_node_type = SequenceNodeType::RESET;
		node.m_vertex_num = vertice;
	}
    else
    {
        QCERR("node type error");
        throw std::runtime_error("node type error");
    }
    return node;
}

bool QProgDAG::is_connected_graph()
{
    AdjacencyMatrix matrix;
    get_adjacency_matrix(m_vertices_map, matrix);

    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.rows(); j++)
        {
            if (matrix(i, j))
            {
                for (int k = 0; k < matrix.rows(); k++)
                {
                    if (matrix(k, i))
                    {
                        matrix(k, j) = 1;
                    }
                }
            }
        }
    }
    for (int i = 0; i < matrix.rows(); i++)
    {
        for (int j = 0; j < matrix.rows(); j++)
        {
            if (!matrix(i, j))
            {
                return false;
            }
        }
    }

    return true;
}