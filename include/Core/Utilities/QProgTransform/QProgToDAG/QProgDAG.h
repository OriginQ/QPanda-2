/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
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

	QProgDAGEdge(uint32_t from, uint32_t to, uint32_t qubit)
		:m_from(from), m_to(to), m_qubit(qubit)
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
		:m_id(0), m_type(NUKNOW_SEQ_NODE_TYPE), m_layer(0)
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
	QProgDAG(): m_subgraph(false)
		{}

    /**
    * @brief  add vertex
    * @param[in]  node_info 
    * @return     size_t vertex num
    */
    void add_vertex(std::shared_ptr<QProgDAGNode> n, DAGNodeType type) {
		// add to vertex_vec
		QProgDAGVertex v;
		v.m_id = m_vertex_vec.size();
		v.m_node = n;
		v.m_type = type;
		m_vertex_vec.emplace_back(v);

		//update edge
		uint32_t cur_layer = 0;
		auto tmp_qv = n->m_qubits_vec + n->m_control_vec;
		for_each(tmp_qv.begin(), tmp_qv.end(), [&](Qubit* qubit){
			if (qubit_vertices_map.find(qubit->get_phy_addr()) != qubit_vertices_map.end()){
				const auto _last_node = qubit_vertices_map.at(qubit->get_phy_addr()).back();
				const auto _layer = get_vertex(_last_node).m_layer + 1;
				if ((_layer) > cur_layer) {
					cur_layer = _layer;
				}
			}

			add_qubit_map(qubit, v.m_id);
		});

		//update node-layer
		m_vertex_vec[v.m_id].m_layer = cur_layer;
	}
    
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

    bool is_connected_graph() {
		AdjacencyMatrix matrix;
		get_adjacency_matrix(matrix);

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

	TopologSequence<DAGSeqNode> build_topo_sequence() {
		TopologSequence<DAGSeqNode> seq;
		if (m_qubits.size() == 0){
			QCERR_AND_THROW(run_fail, "Error: Failed to get QProg_DAG, the prog is empty.");
		}

		for (const auto &_vertex : m_vertex_vec)
		{
			if (_vertex.m_layer + 1 > seq.size())
			{
				seq.resize(_vertex.m_layer + 1);
			}

			DAGSeqNode node(_vertex);
			std::vector<DAGSeqNode> connected_vec;

			const auto& succ_node = _vertex.m_succ_node;
			for (const auto &_n : succ_node){
				connected_vec.emplace_back(DAGSeqNode(get_vertex(_n)));
			}

			seq[_vertex.m_layer].emplace_back(make_pair(node, connected_vec));
		}

		return seq;
	}

	std::set<QProgDAGEdge> get_edges() const { 
		std::set<QProgDAGEdge> all_edges;
		for (const auto& _vertex : m_vertex_vec){
			all_edges.insert(_vertex.m_succ_edges.begin(), _vertex.m_succ_edges.end());
		}

		return all_edges;
	}

	void remove_edge(const QProgDAGEdge& e) {
		const auto from_node = e.m_from;
		const auto to_node = e.m_to;

		for (auto _itr = m_vertex_vec[from_node].m_succ_node.begin();
			_itr != m_vertex_vec[from_node].m_succ_node.end(); ++_itr){
			if (to_node == *_itr)
			{
				m_vertex_vec[from_node].m_succ_node.erase(_itr);
				break;
			}
		}
		
		for (auto _itr = m_vertex_vec[to_node].m_pre_node.begin();
			_itr != m_vertex_vec[to_node].m_pre_node.end(); ++_itr){
			if (from_node == *_itr)
			{
				m_vertex_vec[to_node].m_pre_node.erase(_itr);
				break;
			}
		}

		auto tmp_e = e;
		m_vertex_vec[from_node].remove_succ_edge(tmp_e);
		m_vertex_vec[to_node].remove_pre_edge(tmp_e);
	}

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
	void add_edge(uint32_t in_num, uint32_t out_num, uint32_t qubit) {
		m_vertex_vec[in_num].m_succ_node.push_back(out_num);
		m_vertex_vec[out_num].m_pre_node.push_back(in_num);

		const QProgDAGEdge _e(in_num, out_num, qubit);
		m_vertex_vec[in_num].m_succ_edges.emplace_back(_e);
		m_vertex_vec[out_num].m_pre_edges.emplace_back(_e);
	}

	/**
	* @brief  add qubit map
	* @param[in]  size_t qubit
	* @param[in]  size_t vertex num
	* @return     void
	*/
	void add_qubit_map(Qubit* tar_qubit, size_t vertice_num) {
		const uint32_t tar_q_i = tar_qubit->get_phy_addr();
		m_qubits.insert(std::make_pair(tar_q_i, tar_qubit));

		auto iter = qubit_vertices_map.find(tar_q_i);
		if (iter != qubit_vertices_map.end())
		{
			size_t in_vertex_num = iter->second.back();
			add_edge(in_vertex_num, vertice_num, tar_q_i);
			qubit_vertices_map[iter->first].emplace_back(vertice_num);
		}
		else
		{
			qubit_vertices_map.insert(std::make_pair(tar_q_i, std::vector<size_t>({ vertice_num })));
		}
	}

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
