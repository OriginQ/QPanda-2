#ifndef TOPOLOG_SEQUENCE_H
#define TOPOLOG_SEQUENCE_H

#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <vector>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
//#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
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

//template <class seq_node_T>
//class DAGToTopologSequence
//{
//public:
//	template <class dag_data_T>
//	using tranf_fun = std::function<seq_node_T(dag_data_T&, void* user_data)>;
//
//	template <class dag_data_T>
//	DAGToTopologSequence(TopologSequence<seq_node_T>& seq, QProgDAG& dag, tranf_fun<dag_data_T> dag_data_to_seq_node_tranf_fun)
//		:m_sequence(seq)
//	{
//		build_topo_sequence(dag, dag_data_to_seq_node_tranf_fun);
//	}
//
//protected:
//	
//
//public:
//	
//
//private:
//};

//template <typename dag_node_T, typename seq_node_T>
//class QProgTopologSeq
//{
//public:
//	QProgTopologSeq() {}
//	~QProgTopologSeq() {}
//
//	template <typename node_T, class Function>
//	void prog_to_topolog_seq(node_T &node, Function && f) {
//		m_prog_to_dag.traversal(node, m_dag);
//		DAGToTopologSequence<SequenceNode>::tranf_fun<GateNodeInfo> fun = f;
//		DAGToTopologSequence<seq_node_T>(m_seq, m_dag, fun);
//	}
//
//	QProgDAG& get_dag() { return m_dag; }
//	TopologSequence<seq_node_T>& get_seq() { return m_seq; }
//
//private:
//	QProgToDAG m_prog_to_dag;
//	QProgDAG m_dag;
//	TopologSequence<seq_node_T> m_seq;
//};

QPANDA_END

#endif // TOPOLOG_SEQUENCE_H