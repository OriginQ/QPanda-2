#include <memory>
#include <algorithm>
#include <queue>
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"

USING_QPANDA

void QProgDAG::add_vertex(std::shared_ptr<QProgDAGNode> n, DAGNodeType type) {
	// add to vertex_vec
	QProgDAGVertex v;
	v.m_id = m_vertex_vec.size();
	v.m_node = n;
	v.m_type = type;
	m_vertex_vec.emplace_back(v);

	//update edge
	uint32_t cur_layer = 0;
	auto tmp_qv = n->m_qubits_vec + n->m_control_vec;
	for_each(tmp_qv.begin(), tmp_qv.end(), [&](Qubit* qubit) {
		if (qubit_vertices_map.find(qubit->get_phy_addr()) != qubit_vertices_map.end()) {
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

bool QProgDAG::is_connected_graph()
{
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

TopologSequence<DAGSeqNode> QProgDAG::build_topo_sequence() {
	TopologSequence<DAGSeqNode> seq;
	if (m_qubits.size() == 0) {
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
		for (const auto &_n : succ_node) {
			connected_vec.emplace_back(DAGSeqNode(get_vertex(_n)));
		}

		seq[_vertex.m_layer].emplace_back(make_pair(node, connected_vec));
	}

	return seq;
}

std::set<QProgDAGEdge> QProgDAG::get_edges() const {
	std::set<QProgDAGEdge> all_edges;
	for (const auto& _vertex : m_vertex_vec) {
		all_edges.insert(_vertex.m_succ_edges.begin(), _vertex.m_succ_edges.end());
	}

	return all_edges;
}

void QProgDAG::remove_edge(const QProgDAGEdge& e) {
	const auto from_node = e.m_from;
	const auto to_node = e.m_to;

	for (auto _itr = m_vertex_vec[from_node].m_succ_node.begin();
		_itr != m_vertex_vec[from_node].m_succ_node.end(); ++_itr) {
		if (to_node == *_itr)
		{
			m_vertex_vec[from_node].m_succ_node.erase(_itr);
			break;
		}
	}

	for (auto _itr = m_vertex_vec[to_node].m_pre_node.begin();
		_itr != m_vertex_vec[to_node].m_pre_node.end(); ++_itr) {
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

void QProgDAG::add_edge(uint32_t in_num, uint32_t out_num, uint32_t qubit) {
	m_vertex_vec[in_num].m_succ_node.push_back(out_num);
	m_vertex_vec[out_num].m_pre_node.push_back(in_num);

	const QProgDAGEdge _e(in_num, out_num, qubit);
	m_vertex_vec[in_num].m_succ_edges.emplace_back(_e);
	m_vertex_vec[out_num].m_pre_edges.emplace_back(_e);
}

void QProgDAG::add_qubit_map(Qubit* tar_qubit, size_t vertice_num)
{
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

std::shared_ptr<QProg> QProgDAG::dag_to_qprog() {
	std::map<int, int> in_degree;
	std::queue<int> topo_queue;
	auto prog = CreateEmptyQProg();
	for (auto& _vertex : m_vertex_vec) {
		if (_vertex.m_invalid) continue;
		in_degree[_vertex.m_id] = _vertex.m_pre_edges.size();
	}
	for (auto& _map_itr : in_degree) {
		if (_map_itr.second == 0) {
			topo_queue.push(_map_itr.first);
		}
	}
	while (topo_queue.size() > 0) {
		auto k = topo_queue.front();
		auto& vertex = m_vertex_vec[k];
		bool measure_find = false;
		if (vertex.m_type != GateType::GATE_UNDEFINED) {
			auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*(vertex.m_node->m_itr));
			QVec qubit_vec = vertex.m_node->m_qubits_vec;
			QVec control_vec;
			p_gate->getControlVector(control_vec);

			auto _g = QGate(p_gate);
			auto _new_gate = deepCopy(_g);
			_new_gate.remap(qubit_vec);
			_new_gate.setControl(control_vec);
			prog.insertQNode(prog.getLastNodeIter(), std::dynamic_pointer_cast<QNode>(_new_gate.getImplementationPtr()));
			for (auto& _succ_edge : vertex.m_succ_edges) {
				auto target = _succ_edge.m_to;
				in_degree[target] --;
				if (in_degree[target] == 0) {
					topo_queue.push(target);
				}
			}
		}
		else {
			auto new_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(vertex.m_node->m_itr));
			auto CBit = new_node->getCBit();
			auto qidx = new_node->getQuBit()->get_phy_addr();
			auto QBit = m_qubits[qidx];
			prog << Measure(QBit, CBit);
		}
		topo_queue.pop();
	}
	return std::make_shared<QProg>(prog);
}

std::vector<std::shared_ptr<QProgDAG>> QProgDAG::partition(std::vector<std::set<uint32_t>> par_list) {
	std::vector<std::vector<QProgDAGVertex>> subgraph;
	std::vector<QProgDAGVertex> _vertex_vec;
	for (int i = 0; i < par_list.size(); ++i) {
		subgraph.push_back(_vertex_vec);
	}
	for (int i = 0; i < par_list.size(); ++i) {
		auto vertices = m_vertex_vec;
		auto _vec = par_list[i];
		for (auto _vertex_id : _vec) {
			auto & _vertex = vertices[_vertex_id];
			for (auto itr = _vertex.m_pre_edges.begin(); itr != _vertex.m_pre_edges.end();) {
				if (_vec.find(itr->m_from) == _vec.end()) {
					itr = _vertex.m_pre_edges.erase(itr);
				}
				else {
					++itr;
				}
			}
			for (auto itr = _vertex.m_succ_edges.begin(); itr != _vertex.m_succ_edges.end();) {
				if (_vec.find(itr->m_to) == _vec.end()) {
					itr = _vertex.m_succ_edges.erase(itr);
				}
				else {
					++itr;
				}
			}
		}
		for (auto _vertex_id : _vec) {
			subgraph[i].push_back(vertices[_vertex_id]);
		}
	}
	std::vector<std::shared_ptr<QProgDAG>> subgraph_list;
	for (int i = 0; i < par_list.size(); ++i) {
		auto dag = std::make_shared<QProgDAG>();
		dag->m_subgraph = true;
		dag->m_vertex_vec = subgraph[i];
		dag->m_qubits = m_qubits;
		subgraph_list.push_back(dag);
	}
	return subgraph_list;
}