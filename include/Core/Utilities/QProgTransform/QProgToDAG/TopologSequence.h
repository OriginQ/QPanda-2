#ifndef TOPOLOG_SEQUENCE_H
#define TOPOLOG_SEQUENCE_H

#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <vector>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include <functional>

QPANDA_BEGIN

/*
* store the sequence node and the next sequence node, 
* for double qubits gate, control qubit first, and then target qubit
*/
//template <class T>
//struct SeqNode
//{
//	T m_cur_node;
//	std::vector<T> m_successor_nodes;
//};

template <class T>
using SeqNode = std::pair<T, std::vector<T>>;

template <class T>
using SeqLayer = std::vector<SeqNode<T>>;

template <class T>
class TopologSequence : public std::vector<SeqLayer<T>>
{
public:
	TopologSequence()
		:m_cur_layer(0)
	{}
	virtual ~TopologSequence() {}

private:
	size_t m_cur_layer;
};

template <class seq_node_T>
class DAGToTopologSequence
{
public:
	template <class dag_data_T>
	using tranf_fun = std::function<seq_node_T(dag_data_T&, void* user_data)>;

	template <class dag_data_T>
	DAGToTopologSequence(TopologSequence<seq_node_T>& seq, QProgDAG<dag_data_T>& dag, tranf_fun<dag_data_T> dag_data_to_seq_node_tranf_fun)
		:m_sequence(seq)
	{
		build_topo_sequence(dag, dag_data_to_seq_node_tranf_fun);
	}

protected:
	template <class dag_data_T>
	void build_topo_sequence(QProgDAG<dag_data_T>& dag, tranf_fun<dag_data_T> tranf_fun) {
		if (dag.m_qubit_vec.size() == 0)
		{
			QCERR_AND_THROW(run_fail, "Error: Failed to get QProg_DAG, the prog is empty.");
		}

		AdjacencyMatrix matrix;
		dag.get_adjacency_matrix(matrix);

		auto col_mat = matrix.colwise().sum();
		AdjacencyMatrix flag_mat = AdjacencyMatrix::Zero(2, dag.m_vertices_map.size());
		for (auto i = 0; i < dag.m_vertices_map.size(); ++i)
		{
			flag_mat(0, i) = col_mat(0, i);
		}

		while (!flag_mat.row(1).minCoeff())
		{
			SeqLayer<seq_node_T> seq_layer;
			get_cur_layer_vertices(dag, tranf_fun, flag_mat, seq_layer);
			m_sequence.emplace_back(seq_layer);
		}
	}

	template <class dag_data_T>
	void get_cur_layer_vertices(QProgDAG<dag_data_T> dag, tranf_fun<dag_data_T> tranf_fun, AdjacencyMatrix &matrix, SeqLayer<seq_node_T> &seq_layer) {
		auto count = dag.m_vertices_map.size();
		for (auto i = 0; i < count; ++i)
		{
			if ((matrix(1, i) == 0) && (matrix(0, i) == 0))
			{
				seq_node_T node = construct_sequence_node(dag, tranf_fun, i);
				std::vector<seq_node_T> connected_vec;
				for (const auto &edge : dag.m_edges_vector)
				{
					if (edge.first == i)
					{
						connected_vec.emplace_back(construct_sequence_node(dag, tranf_fun, edge.second));
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
				for (const auto &edge : dag.m_edges_vector)
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

	/**
    * @brief  construct sequence node
    * @param[in]  size_t vertex num
    * @return     QPanda::SequenceNode
    */
	template <class dag_data_T>
	seq_node_T construct_sequence_node(QProgDAG<dag_data_T> dag, tranf_fun<dag_data_T> tranf_fun, size_t vertice) {
		return tranf_fun(dag.m_vertices_map.find(vertice)->second, (void*)vertice);
	}

public:
	TopologSequence<seq_node_T>& m_sequence;

private:
};

template <typename dag_node_T, typename seq_node_T>
class QProgTopologSeq
{
public:
	QProgTopologSeq() {}
	~QProgTopologSeq() {}

	template <typename node_T, class Function>
	void prog_to_topolog_seq(node_T &node, Function && f) {
		m_prog_to_dag.traversal(node, m_dag);
		DAGToTopologSequence<SequenceNode>::tranf_fun<GateNodeInfo> fun = f;
		DAGToTopologSequence<seq_node_T>(m_seq, m_dag, fun);
	}

	QProgDAG<dag_node_T>& get_dag() { return m_dag; }
	TopologSequence<seq_node_T>& get_seq() { return m_seq; }

private:
	QProgToDAG m_prog_to_dag;
	QProgDAG<dag_node_T> m_dag;
	TopologSequence<seq_node_T> m_seq;
};

using TopoNode = SeqNode<SequenceNode>;
using TopoLayer = SeqLayer<SequenceNode>;

QPANDA_END

#endif // TOPOLOG_SEQUENCE_H