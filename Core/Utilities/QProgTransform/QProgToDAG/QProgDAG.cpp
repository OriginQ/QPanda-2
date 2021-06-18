#include <memory>
#include <algorithm>
#include <queue>
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"

USING_QPANDA
/*
void QProgDAG::add_vertex(std::shared_ptr<QProgDAGNode> n, DAGNodeType type) {
	// add to vertex_vec
	QProgDAGVertex v;
	v.m_id = m_vertex_vec.size();
	v.m_node = n;
	v.m_type = type;
	v.m_invalid = false;
	m_vertex_vec.emplace_back(v);

	//update edge
	uint32_t cur_layer = 0;
	auto tmp_qv = n->m_qubits_vec + n->m_control_vec;
	for_each(tmp_qv.begin(), tmp_qv.end(), [&](Qubit* qubit) {
		if (qubit_vertices_map.find(qubit->get_phy_addr()) != qubit_vertices_map.end())
		{
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
*/
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