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
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "ThirdParty/Eigen/Sparse"
#include "ThirdParty/Eigen/Dense"

QPANDA_BEGIN


enum SequenceNodeType
{
	MEASURE = -1,
	RESET = -2
};

struct GateNodeInfo
{
	GateNodeInfo(const NodeIter itr)
		:m_itr(itr), m_dagger(false)
	{}

	NodeIter m_itr;
	bool m_dagger;
	QVec m_qubits_vec;
	QVec m_control_vec;
};

struct SequenceNode
{
    int m_node_type; // SequenceNodeType(on case of m_node_type < 0) and GateType
    size_t m_vertex_num;

    bool operator == (const SequenceNode &node) const { return (this->m_vertex_num == node.m_vertex_num); }
    bool operator <  (const SequenceNode &node) const { return (this->m_vertex_num  < node.m_vertex_num); }
    bool operator >  (const SequenceNode &node) const { return (this->m_vertex_num  > node.m_vertex_num); }

	/**
   * @brief  construct sequence node
   * @param[in]  size_t vertex num
   * @return     QPanda::SequenceNode
   */
	static SequenceNode construct_sequence_node(GateNodeInfo& gate_node, void* user_data) {
		SequenceNode node;
		size_t vertice_num = (size_t)user_data;
		auto node_ptr = (*(gate_node.m_itr));
		if (NodeType::GATE_NODE == node_ptr->getNodeType())
		{
			auto pQGate = std::dynamic_pointer_cast<AbstractQGateNode>(node_ptr);
			node.m_node_type = pQGate->getQGate()->getGateType();
			node.m_vertex_num = vertice_num;
		}
		else if (NodeType::MEASURE_GATE == node_ptr->getNodeType())
		{
			node.m_node_type = SequenceNodeType::MEASURE;
			node.m_vertex_num = vertice_num;
		}
		else if (NodeType::RESET_NODE == node_ptr->getNodeType())
		{
			node.m_node_type = SequenceNodeType::RESET;
			node.m_vertex_num = vertice_num;
		}
		else
		{
			QCERR("node type error");
			throw std::runtime_error("node type error");
		}
		return node;
	}
};

using edges_vec = std::vector<std::pair<size_t, size_t>>; 
using AdjacencyMatrix = Eigen::MatrixXi;

/**
* @class QProgDAG
* @ingroup Utilities
* @brief transform QProg to DAG(directed acyclic graph)
* @note
*/
template <class T>
class QProgDAG
{
	template <class seq_node_T>
	friend class DAGToTopologSequence;

public:
	using vertices_map = std::map<size_t, T>;

public:
	QProgDAG() {}

    /**
    * @brief  add vertex
    * @param[in]  node_info 
    * @return     size_t vertex num
    */
    size_t add_vertex(const T& node) {
		auto vertice_num = m_vertices_map.size();
		m_vertices_map.insert(std::make_pair(vertice_num, node));
		return vertice_num;
	}

    /**
    * @brief  add edge
    * @param[in]  size_t vertex num
    * @param[in]  size_t vertex num
    * @return     void
    */
    void add_edge(size_t in_num, size_t out_num) {
		for (auto val : m_edges_vector)
		{
			if (val.first == in_num && val.second == out_num)
			{
				return;
			}
		}
		m_edges_vector.emplace_back(std::make_pair(in_num, out_num));
	}
    
    /**
    * @brief  get adjacency_matrix
    * @param[in]   vertices_map&
    * @param[out]  AdjacencyMatrix&
    * @return     void
    */
    void get_adjacency_matrix(AdjacencyMatrix & matrix) {
		matrix = AdjacencyMatrix::Zero(m_vertices_map.size(), m_vertices_map.size());

		for (const auto &vertice : m_vertices_map)
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

    /**
    * @brief  get vertex by vertex num
    * @param[in]  size_t vertex num
    * @return     std::shared_ptr<QPanda::QNode> qnode
    */
	const T& get_vertex_node(const size_t vertice_num) const {
		if (m_vertices_map.size() <= vertice_num)
		{
			QCERR_AND_THROW(run_fail, "Error: vertice_num error.");
		}
		return m_vertices_map.find(vertice_num)->second;
	}

    /**
    * @brief  add qubit map
    * @param[in]  size_t qubit
    * @param[in]  size_t vertex num
    * @return     void
    */
    void add_qubit_map(size_t tar_qubit, size_t vertice_num) {
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
			qubit_vertices_map.insert(std::make_pair(tar_qubit, vertice_vec));
		}
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

	NodeIter add_gate(std::shared_ptr<QNode> node) {
		m_dag_prog.pushBackNode(node);
		return m_dag_prog.getLastNodeIter();
	}

public:
	std::vector<size_t> m_qubit_vec;

private:
    QProg m_dag_prog;
    edges_vec m_edges_vector;
    vertices_map m_vertices_map;
    std::map<size_t, std::vector<size_t>> qubit_vertices_map;
};

QPANDA_END
#endif
