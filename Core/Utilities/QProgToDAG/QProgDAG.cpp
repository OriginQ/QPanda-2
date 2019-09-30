#include "Core/Utilities/QProgToDAG/QProgDAG.h"
#include <algorithm>
USING_QPANDA
using namespace std;

size_t QProgDAG::addVertex(QNode *node)
{
    m_dag_prog.pushBackNode(node);
    return addVertex(m_dag_prog.getLastNodeIter());
}


size_t QProgDAG::addVertex(const NodeIter& iter)
{
    auto vertice_num = m_vertices_map.size();
    m_vertices_map.insert(make_pair(vertice_num, iter));
    return vertice_num;
}

void QProgDAG::addEgde(size_t in_num, size_t out_num)
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
   
void QProgDAG::constructAdjacencyMatrix(const vertices_map &dag_map, AdjacencyMatrix & matrix)
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


std::shared_ptr<QNode> QProgDAG::getVertex(size_t vertice_num)  const
{
	return *(getVertexNodeIter(vertice_num));
    /*if (m_vertices_map.size() <= vertice_num)
    {
        QCERR("vertice_num error");
        throw std::runtime_error("vertice_num error");
    }
    return *(m_vertices_map.find(vertice_num)->second);*/
}

NodeIter QProgDAG::getVertexNodeIter(size_t vertice_num) const
{
	if (m_vertices_map.size() <= vertice_num)
	{
		QCERR("vertice_num error");
		throw std::runtime_error("vertice_num error");
	}
	return m_vertices_map.find(vertice_num)->second;
}

void QProgDAG::getTopologincalSequence(TopologincalSequence &seq)
{
    AdjacencyMatrix matrix;
    constructAdjacencyMatrix(m_vertices_map, matrix);

    auto col_mat = matrix.colwise().sum();
    AdjacencyMatrix flag_mat = AdjacencyMatrix::Zero(2, m_vertices_map.size());
    for (auto i = 0; i < m_vertices_map.size(); ++i)
    {
        flag_mat(0, i) = col_mat(0, i);
    }

    //cout << flag_mat << endl << endl;
    while (!flag_mat.row(1).minCoeff())
    {
        SequenceLayer seq_layer;
        getCurLayerVertices(flag_mat, seq_layer);
        seq.emplace_back(seq_layer);
        //cout << flag_mat << endl << endl;
    }
}


void QProgDAG::getCurLayerVertices(AdjacencyMatrix &matrix, SequenceLayer &seq_layer)
{
    auto count = m_vertices_map.size();
    for (auto i = 0; i < count; ++i)
    {
        if ((matrix(1, i) == 0) && (matrix(0, i) == 0))
        {
            SequenceNode node = constructSequenceNode(i);
            std::vector<SequenceNode> connected_vec;
            for (const auto &edge : m_edges_vector)
            {
                if (edge.first == i)
                {
                    connected_vec.emplace_back(constructSequenceNode(edge.second));
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


SequenceNode QProgDAG::constructSequenceNode(size_t vertice)
{
    SequenceNode node;
    QNode * node_ptr = (*(m_vertices_map.find(vertice)->second)).get();
    if (NodeType::GATE_NODE == node_ptr->getNodeType())
    {
        auto pQGate = dynamic_cast<AbstractQGateNode*>(node_ptr);
        node.m_node_type = pQGate->getQGate()->getGateType();
        node.m_vertex_num = vertice;
    }
    else if (NodeType::MEASURE_GATE == node_ptr->getNodeType())
    {
        auto pMeasure = dynamic_cast<AbstractQuantumMeasure*>(node_ptr);
        node.m_node_type = -1;
        node.m_vertex_num = vertice;
    }
    else
    {
        QCERR("node type error");
        throw std::runtime_error("node type error");
    }
    return node;
}
