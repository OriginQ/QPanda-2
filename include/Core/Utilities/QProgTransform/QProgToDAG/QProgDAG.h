/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QProgDAG.h
Author: doumenghan
Updated in 2019/08/06 

Classes for QProgDAG.

*/
/*! \file QProgDAG.h */
#ifndef  QPROGDAG_H
#define  QPROGDAG_H

#include <vector>
#include <memory>
#include <set>
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "ThirdParty/Eigen/Sparse"
#include "ThirdParty/Eigen/Dense"
#include "Core/Utilities/QProgTransform/QProgToDAG/TopologSequence.h"

QPANDA_BEGIN


enum DAGNodeType
{
	NUKNOW_SEQ_NODE_TYPE = -1,

	/************************** 
	* GateType: 0~0XF0
	* @see GateType
	*/

	MAX_GATE_TYPE = 0XF0,
	/**************************
	* non-GateType
	*/
	MEASURE = 0XF1,
	QUBIT = 0XF2,
	RESET
};

struct QProgDAGNode
{
	NodeIter m_itr;
	bool m_dagger;
	QVec m_qubits_vec;
	QVec m_control_vec;
	std::vector<double> m_angles; 

	QProgDAGNode()
		:m_itr(NodeIter()), m_dagger(false)
	{}
	void copy(std::shared_ptr<QProgDAGNode> node) {
		m_itr = node->m_itr;
		m_dagger = node->m_dagger;
		m_qubits_vec = node->m_qubits_vec;
		m_control_vec = node->m_control_vec;
	}
};

struct QProgDAGEdge
{
	uint32_t m_from;
	uint32_t m_to;
	uint32_t m_qubit;

	QProgDAGEdge(uint32_t from_val, uint32_t to_val, uint32_t qubit_val)
		:m_from(from_val), m_to(to_val), m_qubit(qubit_val)
	{}

	bool operator <  (const QProgDAGEdge& e) const { 
		if (m_from == e.m_from){
			if (m_to == e.m_to)
			{
				return m_qubit < e.m_qubit;
			}
			
			return m_to < e.m_to;
		}
		
		return m_from < e.m_from;
	}

	bool operator ==  (const QProgDAGEdge& e) const {
		return (m_from == e.m_from) && (m_to == e.m_to) && (m_qubit == e.m_qubit);
	}
};

class QProgDAGVertex
{
public:
	std::shared_ptr<QProgDAGNode> m_node;
	uint32_t m_id;
	DAGNodeType m_type;
	uint32_t m_layer;
	bool m_invalid;
	std::vector<uint32_t> m_pre_node;
	std::vector<uint32_t> m_succ_node;
	std::vector<QProgDAGEdge> m_pre_edges;
	std::vector<QProgDAGEdge> m_succ_edges;
	QProgDAGVertex() 
		:m_id(0), m_type(NUKNOW_SEQ_NODE_TYPE), m_layer(0), m_invalid(false)
	{}

	bool is_pre_adjoin(const uint32_t& n) {
		for (const auto& _pre_n : m_pre_node) {
			if (_pre_n == n) { return true; }
		}

		return false;
	}

	bool is_succ_adjoin(const uint32_t& n) {
		for (const auto& _succ_n : m_succ_node) {
			if (_succ_n == n) { return true; }
		}

		return false;
	}

	void remove_pre_edge(const QProgDAGEdge& e) {
		remove_edge(m_pre_edges, e);
	}

	void remove_succ_edge(const QProgDAGEdge& e) {
		remove_edge(m_succ_edges, e);
	}

private:
	void remove_edge(std::vector<QProgDAGEdge>& edges, const QProgDAGEdge& target_e) {
		for (auto itr = edges.begin(); itr != edges.end();)
		{
			if (target_e == *itr) {
				itr = edges.erase(itr);
			}
			else
			{
				++itr;
			}
		}
	}

};

struct DAGSeqNode
{
	int m_node_type; // SequenceNodeType(on case of m_node_type < 0) and GateType
	size_t m_vertex_num;

	/**
	* @brief  construct sequence node
	* @param[in]  size_t vertex num
	* @return     QPanda::SequenceNode
	*/
	DAGSeqNode() 
		:m_node_type(NUKNOW_SEQ_NODE_TYPE), m_vertex_num(0)
	{}

	DAGSeqNode(const QProgDAGVertex& dag_vertex){
		m_vertex_num = dag_vertex.m_id;
		const auto& gate_node = *(dag_vertex.m_node);
		auto node_ptr = (*(gate_node.m_itr));
		if (NodeType::GATE_NODE == node_ptr->getNodeType())
		{
			auto pQGate = std::dynamic_pointer_cast<AbstractQGateNode>(node_ptr);
			m_node_type = pQGate->getQGate()->getGateType();
		}
		else if (NodeType::MEASURE_GATE == node_ptr->getNodeType())
		{
			m_node_type = DAGNodeType::MEASURE;
		}
		else if (NodeType::RESET_NODE == node_ptr->getNodeType())
		{
			m_node_type = DAGNodeType::RESET;
		}
		else
		{
			QCERR("node type error");
			m_node_type = DAGNodeType::NUKNOW_SEQ_NODE_TYPE;
		}
	}

	bool operator == (const DAGSeqNode &node) const { return (this->m_vertex_num == node.m_vertex_num); }
	bool operator <  (const DAGSeqNode &node) const { return (this->m_vertex_num < node.m_vertex_num); }
	bool operator >  (const DAGSeqNode &node) const { return (this->m_vertex_num > node.m_vertex_num); }
};

using AdjacencyMatrix = Eigen::MatrixXi;
using DAGTopoNode = SeqNode<DAGSeqNode>;
using DAGTopoLayer = SeqLayer<DAGSeqNode>;

/**
* @class QProgDAG
* @ingroup Utilities
* @brief transform QProg to DAG(directed acyclic graph)
* @note
*/
class QProgDAG
{
	template <class seq_node_T>
	friend class DAGToTopologSequence;

public:
	QProgDAG(): m_subgraph(false){}

    /**
    * @brief  add vertex
    * @param[in]  node_info 
    * @return     size_t vertex num
    */
	void add_vertex(std::shared_ptr<QProgDAGNode> n, DAGNodeType type);
    
    /**
    * @brief  get adjacency_matrix
    * @param[in]   vertices_map&
    * @param[out]  AdjacencyMatrix&
    * @return     void
    */
    void get_adjacency_matrix(AdjacencyMatrix & matrix) {
		matrix = AdjacencyMatrix::Zero(m_vertex_vec.size(), m_vertex_vec.size());

		for (const auto &vertice : m_vertex_vec)
		{
			for (const auto &_n : vertice.m_succ_node)
			{
				matrix(vertice.m_id, _n) = 1;
			}
		}
	}

    /**
    * @brief  get vertex by vertex num
    * @param[in]  size_t vertex num
    * @return     std::shared_ptr<QPanda::QNode> qnode
    */
	const QProgDAGVertex& get_vertex(const size_t vertice_num) const {
		for (auto& _vertex : m_vertex_vec) {
			if (_vertex.m_id == vertice_num) return _vertex;
		}
		QCERR_AND_THROW(run_fail, "Error: vertice_num error.");
		return m_vertex_vec[vertice_num];
	}

	bool is_connected_graph();

	TopologSequence<DAGSeqNode> build_topo_sequence();

	std::set<QProgDAGEdge> get_edges() const;

	void remove_edge(const QProgDAGEdge& e);

	const std::vector<QProgDAGVertex>& get_vertex_c() const { return m_vertex_vec;}
	std::vector<QProgDAGVertex>& get_vertex() { return m_vertex_vec; }
	const auto& get_qubit_vertices_map() const { return qubit_vertices_map; }
	// chenmingyu add
	std::shared_ptr<QProg> dag_to_qprog();
	std::vector<std::shared_ptr<QProgDAG>> partition(std::vector<std::set<uint32_t>> par_list);
	// end
protected:
	/**
	* @brief  add edge
	* @param[in]  size_t vertex num
	* @param[in]  size_t vertex num
	* @return     void
	*/
	void add_edge(uint32_t in_num, uint32_t out_num, uint32_t qubit);

	/**
	* @brief  add qubit map
	* @param[in]  size_t qubit
	* @param[in]  size_t vertex num
	* @return     void
	*/
	void add_qubit_map(Qubit* tar_qubit, size_t vertice_num);

public:
	std::map<uint32_t, Qubit*> m_qubits;
	// chenmingyu add
	std::map<size_t, std::vector<size_t>> m_layer_set;
	bool m_subgraph;
private:
	std::vector<QProgDAGVertex> m_vertex_vec;
    std::map<size_t, std::vector<size_t>> qubit_vertices_map;
};

QPANDA_END
#endif
