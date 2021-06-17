#include <algorithm>
#include <functional>
#include <sstream>
#include <string>

#include "Core/Utilities/QProgTransform/QCircuitRewrite.h"
USING_QPANDA
using namespace std;

/*******************************************************************
*                      class QCircuitRewrite.cpp
********************************************************************/
static int times = 1;
std::atomic<size_t> QCircuitRewrite::m_job_cnt{ 0 };

std::shared_ptr<QProgDAG> QCircuitRewrite::generator_to_dag(QCircuitGenerator& cir_gen) {
	QProgDAG dag;

	auto& cir_node_vec = cir_gen.get_cir_node_vec();
	for (auto& cir_node : cir_node_vec) {
		auto node = std::make_shared<QProgDAGNode>();
		auto node_ptr = std::make_shared<QProgDAGVertex>();
		QVec control_vec;
		QVec qubit_vec;
		for (const auto& i : cir_node->m_target_q) {
			qubit_vec.emplace_back(m_qv[i]);
		}
		for (const auto& i : cir_node->m_control_q) {
			control_vec.emplace_back(m_qv[i]);
		}
		node->m_qubits_vec = qubit_vec;
		node->m_control_vec = control_vec;
		node->m_dagger = cir_node->m_is_dagger;
		for (auto& _angle : cir_node->m_angle) {
			node->m_angles.push_back(angle_str_to_double(_angle));
		}
		dag.add_vertex(node, (DAGNodeType)(TransformQGateType::getInstance()[cir_node->m_op]));
	}
	return std::make_shared<QProgDAG>(dag);
}

QProg QCircuitRewrite::replace_subgraph(std::shared_ptr<QProgDAG> g, QCircuitGenerator::Ref cir_gen) {
	auto& vertex_vec = g->get_vertex();
	std::queue<int> topo_queue;
	auto prog = CreateEmptyQProg();
	std::map<int, int> indegree;
	for (auto& _map : m_match_list) {
		for (auto& _pair : _map.match_vertices) {
			vertex_vec[_pair.first].m_invalid = true;
		}
	}
	for (auto& _vertex : vertex_vec) {
		if (!_vertex.m_invalid) {
			indegree[_vertex.m_id] = _vertex.m_pre_node.size();
		}
	}
	for (auto& _map_itr : indegree) {
		if (_map_itr.second == 0) {
			topo_queue.push(_map_itr.first);
		}
	}
	for (auto itr = m_match_list.begin(); itr != m_match_list.end(); ++itr) {
		QVec q;
		std::map<int, int> qubit_map = itr->match_qubits;
		std::map<int, int> vertex_map = itr->match_vertices;
		std::map<int, double> angle_map = itr->match_angles;
		std::map<int, int> core_3;
		std::vector<double> angle;
		std::for_each(angle_map.begin(), angle_map.end(),
			[&](std::pair<int, double> p) {angle.push_back(p.second); });
		std::for_each(qubit_map.begin(), qubit_map.end(),
			[&](std::pair<int, int> p) {core_3[p.second] = p.first; });
		for (int i = 0; i < cir_gen->get_circuit_width(); ++i) {
			q.emplace_back(g->m_qubits[core_3[i]]);
		}
		cir_gen->set_param(q, angle);
		while (topo_queue.size() > 0) {
			auto k = topo_queue.front();
			auto& vertex = vertex_vec[k];
			if (!vertex.m_invalid) {
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
						if (indegree.find(target) == indegree.end()) continue;
						indegree[target] --;
						if (indegree[target] == 0) {
							topo_queue.push(target);
						}
					}
				}
				else {
					auto new_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(vertex.m_node->m_itr));
					auto CBit = new_node->getCBit();
					auto qidx = new_node->getQuBit()->get_phy_addr();
					auto QBit = g->m_qubits[qidx];
					prog << Measure(QBit, CBit);
				}
			}
			topo_queue.pop();
		}
		prog << cir_gen->get_cir();
		std::for_each(vertex_map.begin(), vertex_map.end(),
			[&](std::pair<int, int> p) {
				auto vertex = vertex_vec[p.first];
				for (auto& _succ_edge : vertex.m_succ_edges) {
					auto target = _succ_edge.m_to;
					if (indegree.find(target) == indegree.end()) continue;
					indegree[target] --;
					if (indegree[target] == 0) {
						topo_queue.push(target);
					}
				}
			});
	}
	while (topo_queue.size() > 0) {
		auto k = topo_queue.front();
		auto& vertex = vertex_vec[k];
		bool measure_find = false;
		if (!vertex.m_invalid) {
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
					if (indegree.find(target) == indegree.end()) continue;
					indegree[target] --;
					if (indegree[target] == 0) {
						topo_queue.push(target);
					}
				}
			}
			else {
				auto new_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(vertex.m_node->m_itr));
				auto CBit = new_node->getCBit();
				auto qidx = new_node->getQuBit()->get_phy_addr();
				auto QBit = g->m_qubits[qidx];
				prog << Measure(QBit, CBit);
			}
		}
		topo_queue.pop();
	}
	return prog;
}

void QCircuitRewrite::recursiveMatch(std::shared_ptr<QProgDAG> pattern, std::shared_ptr<QProgDAG> graph) {
	auto vertices = graph->get_vertex();
	if (m_struct.core_1.size() == pattern->get_vertex().size()) {
		if (m_sem.g_qubit_tobematched.size() != 0) {
			for (auto g_begin = m_sem.g_qubit_tobematched.begin(), p_begin = m_sem.p_qubit_tobematched.begin();
				g_begin != m_sem.g_qubit_tobematched.end(); ++g_begin, ++p_begin) {
				m_sem.core_3[*p_begin] = *g_begin;
				m_sem.core_4[*g_begin] = *p_begin;
			}
		}
		m_match_list.insert({ m_struct.core_2, m_sem.core_4, m_sem.angle_map });
		for (auto& _node : m_struct.core_2) {
			matched.emplace(_node.first);
		}
		return;
	}
	//int gsize = graph->get_vertex().size();
	//for (int j = 0; j < gsize; ++j) {
	//	if (m_struct.core_2.count(j) != 0) continue;
	//	auto restore_sem = m_sem;
	//	StructMatch match;
	//	if (feasibilityRules(pattern, graph, i, j, match)) {
	//		auto temp_match = m_struct;
	//		m_struct = match;
	//		recursiveMatch(pattern, graph, i + 1);
	//		m_struct = temp_match;
	//	}
	//	m_sem = restore_sem;
	//}
	int gsize = graph->get_vertex().size();
	std::vector<std::pair<int, int>> candidate_set;
	if (!m_struct.T1_in.empty() && !m_struct.T2_in.empty()) {
		for (const auto& _itr1 : m_struct.T1_in) {
			for (const auto& _itr2 : m_struct.T2_in) {
				candidate_set.push_back(std::make_pair(_itr1, _itr2));
			}
		}
		for (auto& _q_itr : candidate_set) {
			if (m_struct.core_2.count(_q_itr.second)) continue;
			auto restore_sem = m_sem;
			StructMatch match;
			if (feasibilityRules(pattern, graph, _q_itr.first, _q_itr.second, match)) {
				auto temp_match = m_struct;
				m_struct = match;
				recursiveMatch(pattern, graph);
				m_struct = temp_match;
			}
			m_sem = restore_sem;
		}
	}
	else if (!m_struct.T1_out.empty() && !m_struct.T2_out.empty()) {
		for (const auto& _itr1 : m_struct.T1_out) {
			for (const auto& _itr2 : m_struct.T2_out) {
				candidate_set.push_back(std::make_pair(_itr1, _itr2));
			}
		}
		for (auto& _q_itr : candidate_set) {
			if (m_struct.core_2.count(_q_itr.second)) continue;
			auto restore_sem = m_sem;
			StructMatch match;
			if (feasibilityRules(pattern, graph, _q_itr.first, _q_itr.second, match)) {
				auto temp_match = m_struct;
				m_struct = match;
				recursiveMatch(pattern, graph);
				m_struct = temp_match;
			}
			m_sem = restore_sem;
		}
	}
	else {
		for (int j = 0; j < graph->get_vertex_c().size(); ++j) {
			candidate_set.push_back(std::make_pair(0, vertices[j].m_id));
		}
		for (auto& _q_itr : candidate_set) {
			if (m_struct.core_2.count(_q_itr.second)) continue;
			auto restore_sem = m_sem;
			StructMatch match;
			if (feasibilityRules(pattern, graph, _q_itr.first, _q_itr.second, match)) {
				auto temp_match = m_struct;
				m_struct = match;
				recursiveMatch(pattern, graph);
				m_struct = temp_match;
			}
			m_sem = restore_sem;
		}
	}
}

void QCircuitRewrite::PatternMatch(std::shared_ptr<QProgDAG> pattern, std::shared_ptr<QProgDAG> graph) {
	recursiveMatch(pattern, graph);
	m_job_cnt = m_job_cnt + 1;
}

bool QCircuitRewrite::feasibilityRules(std::shared_ptr<QProgDAG> pattern, std::shared_ptr<QProgDAG> graph, int n, int m, StructMatch& match) {
	auto p_n = pattern->get_vertex(n);
	auto g_m = graph->get_vertex(m);

	// every node can only be matched once 
	if (matched.find(m) != matched.end()) {
		return false;
	}
	// semantic feasibility on node
	// check the node type
	if (p_n.m_type != g_m.m_type || p_n.m_node->m_dagger != g_m.m_node->m_dagger) {
		return false;
	}
	// check the angle of gates
	{
		auto p_angles = p_n.m_node->m_angles;
		auto g_angles = g_m.m_node->m_angles;
		if (p_angles.size() != g_angles.size()) return false;
		for (int i = 0; i < p_angles.size(); ++i) {
			int j = (int)(p_angles[i] + 0.5) / ANGLE_VAR_BASE;
			// if the angle of pattern is variable
			if (0 == (int)(p_angles[i] + 0.5) % ANGLE_VAR_BASE && j > 0) {
				if (m_sem.angle_map.find(j) == m_sem.angle_map.end()) {
					m_sem.angle_map[j] = g_angles[i];
				}
				else {
					if (m_sem.angle_map[j] != g_angles[i]) return false;
				}
			}
			// if the angle of pattern is constant
			else {
				if (p_angles[i] != g_angles[i]) return false;
			}
		}
	}
	// check the mapping between the control qubit -- the mapping for control qubits have many possibilities 
	if (p_n.m_type == g_m.m_type) {
		// if the numbers of control qubit don't match, return false
		if (p_n.m_node->m_control_vec.size() != g_m.m_node->m_control_vec.size()) return false;
		if (p_n.m_node->m_control_vec.size() != 0) {
			// graph_qubit is the set of qubit operated by g_m
			// pattern_qubit is the set of qubit operated by p_n 
			std::set<int> graph_qubit;
			std::set<int> pattern_qubit;
			for (int j = 0; j < p_n.m_node->m_control_vec.size(); ++j) {
				graph_qubit.insert(g_m.m_node->m_control_vec[j]->get_phy_addr());
				pattern_qubit.insert(p_n.m_node->m_control_vec[j]->get_phy_addr());
			}
			// traverse every qubit q operated by p_n, check whether there is corresponding qubit q' in g_m
			// If so, check whether q' is the qubit operated by g_m. 
			//		If not, return false since it disobey the built rules 
			//		Otherwise, delete both of them from graph_qubit and pattern_qubit
			for (int j = 0; j < p_n.m_node->m_control_vec.size(); ++j) {
				if (m_sem.core_3.find(p_n.m_node->m_control_vec[j]->get_phy_addr()) != m_sem.core_3.end()) {
					int p = p_n.m_node->m_control_vec[j]->get_phy_addr();
					int g = m_sem.core_3[p];
					if (graph_qubit.find(g) == graph_qubit.end()) return false;
					pattern_qubit.erase(pattern_qubit.find(p));
					graph_qubit.erase(graph_qubit.find(g));
				}
			}
			// After delete, if the number of rest qubit is equal and only one qubit rest, we can build up a mapping relation between them
			// else the relation is unable to built up
			if (graph_qubit.size() == pattern_qubit.size() && graph_qubit.size() == 1) {

			}
			else {
				std::for_each(graph_qubit.begin(), graph_qubit.end(), [&](int _q) {m_sem.g_qubit_tobematched.insert(_q); });
				std::for_each(pattern_qubit.begin(), pattern_qubit.end(), [&](int _q) {m_sem.p_qubit_tobematched.insert(_q); });
			}
		}
	}
	// check the mapping of qubits
	for (int i = 0; i < p_n.m_node->m_qubits_vec.size(); ++i) {
		int from = g_m.m_node->m_qubits_vec[i]->get_phy_addr();
		int to = p_n.m_node->m_qubits_vec[i]->get_phy_addr();
		if (m_sem.core_4.find(from) == m_sem.core_4.end() && m_sem.core_3.find(to) == m_sem.core_3.end()) {
			m_sem.core_4[from] = to;
			m_sem.core_3[to] = from;
		}
		else if (m_sem.core_4.find(from) == m_sem.core_4.end() || m_sem.core_4[from] != to) {
			return false;
		}
	}
	// Whether the outgoing edges of p_n have corresponding edge in the outgoing edges of g_m
	// Traverse the outgoing edges of p_n
	for (auto p_begin = p_n.m_succ_edges.cbegin(), p_end = p_n.m_succ_edges.cend(); p_begin != p_end; ++p_begin) {
		// p_dest is the destination of p_begin
		int p_dest = p_begin->m_to;
		if (m_struct.core_1.find(p_dest) != m_struct.core_1.cend()) {
			auto g_dest = m_struct.core_1.at(p_dest);
			bool find = false;
			for (auto g_begin = g_m.m_succ_edges.begin(), g_end = g_m.m_succ_edges.end(); g_begin != g_end; ++g_begin) {
				// if the destination of edge g_begin is core_1[p_dest] and the m_qubit of g_begin is not matched or mapped to m_qubit of p_begin
				// we find the edge map to verify the feasibility rules
				// Indeed, here is a problem that maybe there exist match left out
				if ((g_dest == g_begin->m_to) && (m_sem.core_4.find(g_begin->m_qubit) == m_sem.core_4.end()
					|| m_sem.core_4.at(g_begin->m_qubit) == p_begin->m_qubit)) {
					if (m_sem.core_4.find(g_begin->m_qubit) == m_sem.core_4.end()) {
						m_sem.core_4[g_begin->m_qubit] = p_begin->m_qubit;
						m_sem.core_3[p_begin->m_qubit] = g_begin->m_qubit;
					}
					find = true;
					break;
				}
			}
			if (!find) return false;
		}
	}
	// The same for the ingoing edge
	for (auto p_begin = p_n.m_pre_edges.cbegin(), p_end = p_n.m_pre_edges.cend(); p_begin != p_end; ++p_begin) {
		int p_dest = p_begin->m_from;
		if (m_struct.core_1.find(p_dest) != m_struct.core_1.cend()) {
			auto g_dest = m_struct.core_1.at(p_dest);
			bool find = false;
			for (auto g_begin = g_m.m_pre_edges.begin(), g_end = g_m.m_pre_edges.end(); g_begin != g_end; ++g_begin) {
				if (g_dest == g_begin->m_from && (m_sem.core_4.find(g_begin->m_qubit) == m_sem.core_4.end()
					|| m_sem.core_4.at(g_begin->m_qubit) == p_begin->m_qubit)) {
					if (m_sem.core_4.find(g_begin->m_qubit) == m_sem.core_4.end()) {
						m_sem.core_4[g_begin->m_qubit] = p_begin->m_qubit;
						m_sem.core_3[p_begin->m_qubit] = g_begin->m_qubit;
					}
					find = true;
					break;
				}
			}
			if (!find) return false;
		}
	}
	// Rule R_pred
	for (auto p_begin = p_n.m_pre_edges.cbegin(), p_end = p_n.m_pre_edges.cend(); p_begin != p_end; ++p_begin) {
		auto v = p_begin->m_from;
		if (m_struct.core_1.find(v) == m_struct.core_1.cend()) continue;
		auto w = m_struct.core_1.at(v);
		if (!g_m.is_pre_adjoin(w)) return false;
	}
	for (auto g_begin = g_m.m_pre_edges.cbegin(), g_end = g_m.m_pre_edges.cend(); g_begin != g_end; ++g_begin) {
		auto v = g_begin->m_from;
		if (m_struct.core_2.find(v) == m_struct.core_2.cend()) continue;
		auto w = m_struct.core_2.at(v);
		if (!p_n.is_pre_adjoin(w)) return false;
	}
	// Rule R_succ
	for (auto p_begin = p_n.m_succ_edges.cbegin(), p_end = p_n.m_succ_edges.cend(); p_begin != p_end; ++p_begin) {
		auto v = p_begin->m_to;
		if (m_struct.core_1.find(v) == m_struct.core_1.cend()) continue;
		auto w = m_struct.core_1.at(v);
		if (g_m.is_succ_adjoin(w)) return false;
	}
	for (auto g_begin = g_m.m_succ_edges.cbegin(), g_end = g_m.m_succ_edges.cend(); g_begin != g_end; ++g_begin) {
		auto v = g_begin->m_to;
		if (m_struct.core_2.find(v) == m_struct.core_2.cend()) continue;
		auto w = m_struct.core_2.at(v);
		if (!p_n.is_succ_adjoin(w)) return false;
	}
	// Rule R_in & R_out
	{
		std::set<int> succ_n;
		std::set<int> succ_m;
		std::set<int> pred_n;
		std::set<int> pred_m;
		std::set<int> inter_1;
		std::set<int> inter_2;
		std::for_each(p_n.m_succ_node.begin(), p_n.m_succ_node.end(), [&](uint32_t node) {succ_n.insert(node); });
		std::for_each(g_m.m_succ_node.begin(), g_m.m_succ_node.end(), [&](uint32_t node) {succ_m.insert(node); });
		std::for_each(p_n.m_pre_node.begin(), p_n.m_pre_node.end(), [&](uint32_t node) {pred_n.insert(node); });
		std::for_each(g_m.m_pre_node.begin(), g_m.m_pre_node.end(), [&](uint32_t node) {pred_m.insert(node); });
		std::set_intersection(succ_n.begin(), succ_n.end(), m_struct.T1_in.begin(), m_struct.T1_in.end(),
			std::inserter(inter_1, inter_1.begin()));
		std::set_intersection(succ_m.begin(), succ_m.end(), m_struct.T2_in.begin(), m_struct.T2_in.end(),
			std::inserter(inter_2, inter_2.begin()));
		if (inter_1.size() > inter_2.size()) return false;
		inter_1.clear();
		inter_2.clear();
		std::set_intersection(succ_n.begin(), succ_n.end(), m_struct.T1_out.begin(), m_struct.T1_out.end(),
			std::inserter(inter_1, inter_1.begin()));
		std::set_intersection(succ_m.begin(), succ_m.end(), m_struct.T2_out.begin(), m_struct.T2_out.end(),
			std::inserter(inter_2, inter_2.begin()));
		if (inter_1.size() > inter_2.size()) return false;
		inter_1.clear();
		inter_2.clear();
		std::set_intersection(pred_n.begin(), pred_n.end(), m_struct.T1_in.begin(), m_struct.T1_in.end(),
			std::inserter(inter_1, inter_1.begin()));
		std::set_intersection(pred_m.begin(), pred_m.end(), m_struct.T2_in.begin(), m_struct.T2_in.end(),
			std::inserter(inter_2, inter_2.begin()));
		if (inter_1.size() > inter_2.size()) return false;
		inter_1.clear();
		inter_2.clear();
		std::set_intersection(pred_n.begin(), pred_n.end(), m_struct.T1_out.begin(), m_struct.T1_out.end(),
			std::inserter(inter_1, inter_1.begin()));
		std::set_intersection(pred_m.begin(), pred_m.end(), m_struct.T2_out.begin(), m_struct.T2_out.end(),
			std::inserter(inter_2, inter_2.begin()));
		if (inter_1.size() > inter_2.size()) return false;
	}
	match = m_struct;
	match.core_1[n] = m;
	match.core_2[m] = n;
	if (match.T1_in.find(n) != match.T1_in.end()) match.T1_in.erase(n);
	if (match.T1_out.find(n) != match.T1_out.end()) match.T1_out.erase(n);
	if (match.T2_in.find(m) != match.T2_in.end()) match.T2_in.erase(m);
	if (match.T2_out.find(m) != match.T2_out.end()) match.T2_out.erase(m);
	std::for_each(p_n.m_pre_edges.begin(), p_n.m_pre_edges.end(), [&](QProgDAGEdge _e) {if (m_struct.core_1.find(_e.m_from) == m_struct.core_1.end()) match.T1_in.insert(_e.m_from); });
	std::for_each(g_m.m_pre_edges.begin(), g_m.m_pre_edges.end(), [&](QProgDAGEdge _e) {if (m_struct.core_2.find(_e.m_from) == m_struct.core_2.end()) match.T2_in.insert(_e.m_from); });
	std::for_each(p_n.m_succ_edges.begin(), p_n.m_succ_edges.end(), [&](QProgDAGEdge _e) {if (m_struct.core_1.find(_e.m_to) == m_struct.core_1.end()) match.T1_out.insert(_e.m_to); });
	std::for_each(g_m.m_succ_edges.begin(), g_m.m_succ_edges.end(), [&](QProgDAGEdge _e) {if (m_struct.core_2.find(_e.m_to) == m_struct.core_2.end()) match.T2_out.insert(_e.m_to); });
	times++;
	return true;
}

std::shared_ptr<QProg> QCircuitRewrite::circuitRewrite(QProg prog, uint32_t num_of_thread) {
	/*const std::string key_name = "pattern";
	const std::string pattern_name = "nopara";*/
	//std::string config_data = "D:\\QPanda_cmy\\pattern.json";

	/*if (!m_config_file.load_config(config_data)) {
		QCERR_AND_THROW(run_fail, "Error: failed to load the config file.");
		return 0;
	}*/

	/*auto& doc = m_config_file.get_root_element();

	if (!(doc.HasMember(key_name.c_str()))) {
		QCERR_AND_THROW(run_fail, "Error: special_ctrl_single_gate config error, no key-string.");
		return 0;
	}
	auto& optimizer_config = doc[key_name.c_str()];
	if (!(optimizer_config.HasMember(pattern_name.c_str())) || (!(optimizer_config[pattern_name.c_str()].IsArray()))) {
		QCERR_AND_THROW(run_fail, "Error: special_ctrl_single_gate config error.");
		return 0;
	}
	auto& pattern_config = optimizer_config[pattern_name.c_str()];*/
	/*std::vector<std::pair<QCircuitGenerator::Ref, QCircuitGenerator::Ref>> optimizer_cir_vec;
	for (size_t i = 0; i < pattern_config.Size(); ++i) {
		auto& optimizer_item = pattern_config[i];
		auto src_cir = std::make_shared<QCircuitGenerator>();
		QCircuitConfigReader(optimizer_item["src"]["circuit"], *src_cir);
		src_cir->set_circuit_width(optimizer_item["qubits"].GetInt());
		auto dst_cir = std::make_shared<QCircuitGenerator>();
		QCircuitConfigReader(optimizer_item["dst"]["circuit"], *dst_cir);
		dst_cir->set_circuit_width(optimizer_item["qubits"].GetInt());
		optimizer_cir_vec.push_back(std::make_pair(src_cir, dst_cir));
	}*/
	int k = 1;
	for (auto i = 0; i < m_optimizer_cir_vec.size(); ++i) {
		auto& cir_pair = m_optimizer_cir_vec[i];
		auto src_dag = generator_to_dag(*(cir_pair.first));
		auto graph_dag = qprog_to_DAG(prog);
		if (num_of_thread == 1 || (num_of_thread > 1 && graph_dag->m_layer_set.size() < 100)) {
			recursiveMatch(src_dag, graph_dag);
			if (m_match_list.size() > 0) {
				prog = replace_subgraph(graph_dag, cir_pair.second);
			}
			matched.clear();
			m_match_list.clear();
		}
		else {
			auto par_list = DAGPartition(graph_dag, num_of_thread, 0);
			auto subgraph_list = graph_dag->partition(par_list);
			std::vector<QCircuitRewrite> rewriter_list(num_of_thread);
			m_thread_pool.init_thread_pool(num_of_thread);
			m_job_cnt = 0;
			for (int i = 0; i < num_of_thread; ++i) {
				m_thread_pool.append(std::bind(&QCircuitRewrite::PatternMatch, &rewriter_list[i], src_dag, subgraph_list[0]));
			}
			while (QCircuitRewrite::m_job_cnt != num_of_thread) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
			for (int i = 0; i < num_of_thread; ++i) {
				for (auto itr = rewriter_list[i].m_match_list.begin(); itr != rewriter_list[i].m_match_list.end(); ++itr) {
					auto _vertices = itr->match_vertices;
					bool contradict = false;
					for (auto _ver : _vertices) {
						if (matched.find(_ver.first) != matched.end()) {
							contradict = true;
							break;
						}
					}
					if (contradict) continue;
					for (auto _ver : _vertices) {
						matched.insert(_ver.first);
					}
					m_match_list.insert(*itr);
				}
			}
			if (m_match_list.size() > 0) {
				prog = replace_subgraph(graph_dag, cir_pair.second);
			}
			matched.clear();
			m_match_list.clear();
		}
	}
	std::cout << times << std::endl;
	return std::make_shared<QProg>(prog);
}

std::vector<std::set<uint32_t>> QCircuitRewrite::DAGPartition(std::shared_ptr<QProgDAG> graph, uint32_t par_num, uint32_t overlap) {
	uint32_t step = graph->m_layer_set.size() / par_num;
	std::vector < std::pair<uint32_t, uint32_t> > loc;
	for (auto i = 0; i < par_num; ++i) {
		if (i == par_num - 1) {
			loc.push_back(std::make_pair(step * i, graph->m_layer_set.size()));
			break;
		}
		loc.push_back(std::make_pair(step * i, (i+1)*step+overlap));
	}
	std::vector<std::set<uint32_t>> par_list;
	for (auto i = 0; i < par_num; ++i) {
		std::set<uint32_t> _buffer;
		for (auto j = loc[i].first, k = loc[i].second; j < k; ++j) {
			auto _vertices = graph->m_layer_set[j];
			for (auto& _index : _vertices) {
				_buffer.insert(_index);
			}
		}
		par_list.push_back(_buffer);
	}
	return par_list;
}

int QCircuitRewrite::load_pattern_conf(const std::string& config_data)
{
	const std::string key_name = "pattern";
	const std::string pattern_name = "nopara";

	if (!m_config_file.load_config(config_data)) {
		QCERR_AND_THROW(run_fail, "Error: failed to load the config file.");
		return -1;
	}

	auto src_circuit = CreateEmptyCircuit();
	auto dst_circuit = CreateEmptyCircuit();

	auto& doc = m_config_file.get_root_element();

	if (!(doc.HasMember(key_name.c_str()))) {
		QCERR_AND_THROW(run_fail, "Error: special_ctrl_single_gate config error, no key-string.");
		return -1;
	}
	auto& optimizer_config = doc[key_name.c_str()];
	if (!(optimizer_config.HasMember(pattern_name.c_str())) || (!(optimizer_config[pattern_name.c_str()].IsArray()))) {
		QCERR_AND_THROW(run_fail, "Error: special_ctrl_single_gate config error.");
		return -1;
	}
	auto& pattern_config = optimizer_config[pattern_name.c_str()];
	for (size_t i = 0; i < pattern_config.Size(); ++i) {
		auto& optimizer_item = pattern_config[i];
		auto src_cir = std::make_shared<QCircuitGenerator>();
		QCircuitConfigReader(optimizer_item["src"]["circuit"], *src_cir);
		src_cir->set_circuit_width(optimizer_item["qubits"].GetInt());
		auto dst_cir = std::make_shared<QCircuitGenerator>();
		QCircuitConfigReader(optimizer_item["dst"]["circuit"], *dst_cir);
		dst_cir->set_circuit_width(optimizer_item["qubits"].GetInt());
		m_optimizer_cir_vec.push_back(std::make_pair(src_cir, dst_cir));
	}

	return 0;
}

/*******************************************************************
*                      public interface
********************************************************************/
void QPanda::sub_cir_replace(QProg& src_prog, const std::string& config_data, const uint32_t& thread_cnt /*= 1*/)
{
	QCircuitRewrite rewriter;
	rewriter.load_pattern_conf(config_data);
	auto aa = rewriter.circuitRewrite(src_prog, 1);
	src_prog = *aa;
	return;
}

void QPanda::sub_cir_replace(QCircuit& src_cir, const std::string& config_data, const uint32_t& thread_cnt /*= 1*/)
{
	QProg tmp_prog(src_cir);
	sub_cir_replace(tmp_prog, config_data, thread_cnt);

	src_cir = QProgFlattening::prog_flatten_to_cir(tmp_prog);

	return;
}

