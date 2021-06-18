#include "Core/Utilities/QProgTransform/SU4TopologyMatch.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"

USING_QPANDA
using namespace std;
using namespace QGATE_SPACE;

SU4TopologyMatch::SU4TopologyMatch(QuantumMachine * machine, QVec &qv, ArchType arch_type)
	:m_qvm(machine), m_qv(qv)
{
	m_gate_costs = {
	{GateType::U1_GATE, 1},
	{GateType::U2_GATE, 1},
	{GateType::U3_GATE, 1},
	{GateType::CNOT_GATE, 10},
	};

	build_coupling_map(arch_type);
}

void  SU4TopologyMatch::build_coupling_map(ArchType type)
{
	switch (type)
	{
	case ArchType::IBM_QX5_ARCH:
	{
		m_coupling_map.clear();
		m_nqubits = 16;
		m_coupling_map.insert({ 1, 0 });
		m_coupling_map.insert({ 1, 2 });
		m_coupling_map.insert({ 2, 3 });
		m_coupling_map.insert({ 3, 14 });
		m_coupling_map.insert({ 3, 4 });
		m_coupling_map.insert({ 5, 4 });
		m_coupling_map.insert({ 6, 5 });
		m_coupling_map.insert({ 6, 11 });
		m_coupling_map.insert({ 6, 7 });
		m_coupling_map.insert({ 7, 10 });
		m_coupling_map.insert({ 8, 7 });
		m_coupling_map.insert({ 9, 8 });
		m_coupling_map.insert({ 9, 10 });
		m_coupling_map.insert({ 11, 10 });
		m_coupling_map.insert({ 12, 5 });
		m_coupling_map.insert({ 12, 11 });
		m_coupling_map.insert({ 12, 13 });
		m_coupling_map.insert({ 13, 4 });
		m_coupling_map.insert({ 13, 14 });
		m_coupling_map.insert({ 15, 0 });
		m_coupling_map.insert({ 15,14 });
		m_coupling_map.insert({ 15,2 });
	}
	break;
	case ArchType::ORIGIN_VIRTUAL_ARCH:
	{
		m_coupling_map.clear();
		std::vector<std::vector<double>> qubit_matrix;
		int qubit_num = 0;
		JsonConfigParam config;
		config.load_config(CONFIG_PATH);
		config.getMetadataConfig(qubit_num, qubit_matrix);
		m_nqubits = qubit_num;
		for (int i = 0; i < m_nqubits; i++)
		{
			for (int j = 0; j < m_nqubits; j++)
			{
				if (qubit_matrix[i][j] > 1e-6)
				{
					m_coupling_map.insert({ i, j });
				}
			}
		}
	}
	break;
	default:
		break;
	}
}

void SU4TopologyMatch::transform_qprog(QProg prog, std::vector<gate> &circuit)
{
	std::shared_ptr<QProgDAG> dag = qprog_to_DAG(prog);
	TopologSequence<DAGSeqNode> tp_seq = dag->build_topo_sequence();

	std::set<int > used_qubits;
	std::vector<int> tp_sort;
	for (auto layer_seq : tp_seq)
	{
		for (auto seq : layer_seq)
		{
			tp_sort.push_back(seq.first.m_vertex_num);
		}
	}
	for (auto vertex : tp_sort)
	{
		std::shared_ptr<QNode> qnode = *(dag->get_vertex(vertex).m_node->m_itr);
		if (qnode->getNodeType() != NodeType::GATE_NODE)
		{
			QCERR("node type not support!");
			throw invalid_argument("node type not support!");
		}
		gate g;
		auto qgate_node = std::dynamic_pointer_cast<AbstractQGateNode>(qnode);
		int gate_type = qgate_node->getQGate()->getGateType();
		QVec qv;

		qgate_node->getQuBitVector(qv);
		g.type = gate_type;
		g.is_dagger = qgate_node->isDagger();
		g.is_flip = false;
		if (gate_type == GateType::CNOT_GATE)
		{
			g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
			g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();

		}
		else if (gate_type == GateType::U1_GATE)
		{
			g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
			g.control = -1;
			auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter*>(qgate_node->getQGate());
			double angle = gate_parameter->getParameter();
			g.param.push_back(angle);
		}
		else if (gate_type == GateType::U2_GATE)
		{
			g.control = -1;
			g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
			QGATE_SPACE::U2 *u2_gate = dynamic_cast<QGATE_SPACE::U2*>(qgate_node->getQGate());
			double phi = u2_gate->get_phi();
			double lam = u2_gate->get_lambda();

			g.param.push_back(phi);
			g.param.push_back(lam);
		}
		else if (gate_type == GateType::U3_GATE)
		{
			g.control = -1;
			g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
			QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>(qgate_node->getQGate());
			double theta = u3_gate->get_theta();
			double phi = u3_gate->get_phi();
			double lam = u3_gate->get_lambda();
			g.param.push_back(theta);
			g.param.push_back(phi);
			g.param.push_back(lam);
		}
		else
		{
			QCERR("gate type not support!");
			throw invalid_argument("gate type not support!");
		}

		if (g.control != -1)
			used_qubits.insert(g.control);

		used_qubits.insert(g.target);

		circuit.push_back(g);
	}
	m_used_qubits = used_qubits.size();
}

void SU4TopologyMatch::pre_processing(std::vector<gate> circuit, gates_digraph &grouped_gates)
{
	std::vector<int> last_index(m_nqubits, -1);

	std::vector<std::vector<gate>>single_qubit_gates(m_nqubits);
	int nnodes = 0;
	for (auto g : circuit)
	{
		if (g.type == GateType::U3_GATE
			|| g.type == GateType::U2_GATE
			|| g.type == GateType::U1_GATE)
		{
			if (last_index[g.target] == -1)
				single_qubit_gates[g.target].push_back(g);
			else
			{
				grouped_gates.vertexs[last_index[g.target]].first.push_back(g);
			}
		}
		else if (g.type == GateType::CNOT_GATE)
		{
			int q1 = g.control;
			int q2 = g.target;
			std::vector<gate> gates;
			if (last_index[q1] == -1 && last_index[q2] == -1)
			{
				gates.insert(gates.end(), single_qubit_gates[g.target].begin(), single_qubit_gates[g.target].end());
				gates.insert(gates.end(), single_qubit_gates[g.control].begin(), single_qubit_gates[g.control].end());
				gates.push_back(g);
				grouped_gates.add_vertex({ gates,  {q1, q2} });
				single_qubit_gates[q1].clear();
				single_qubit_gates[q2].clear();
				last_index[q1] = nnodes;
				last_index[q2] = nnodes;
				nnodes += 1;
			}
			else if (last_index[q1] == -1)
			{
				gates.insert(gates.end(), single_qubit_gates[q1].begin(), single_qubit_gates[q1].end());
				gates.push_back(g);
				grouped_gates.add_vertex({ gates , {q1 , q2} });
				grouped_gates.add_edge(last_index[q2], nnodes);
				single_qubit_gates[q1].clear();
				last_index[q1] = nnodes;
				last_index[q2] = nnodes;
				nnodes += 1;
			}
			else if (last_index[q2] == -1)
			{
				gates.insert(gates.end(), single_qubit_gates[q2].begin(), single_qubit_gates[q2].end());
				gates.push_back(g);
				grouped_gates.add_vertex({ gates , {q1 , q2} });

				grouped_gates.add_edge(last_index[q2], nnodes);
				single_qubit_gates[q2].clear();
				last_index[q1] = nnodes;
				last_index[q2] = nnodes;
				nnodes += 1;
			}
			else
			{
				if (last_index[q2] == last_index[q1] && m_nqubits != grouped_gates.vertexs[last_index[q1]].second.size())
				{
					grouped_gates.vertexs[last_index[q1]].first.push_back(g);
				}
				else
				{
					gates.push_back(g);
					grouped_gates.add_vertex({ gates, {q1, q2} });
					grouped_gates.add_edge(last_index[q1], nnodes);
					grouped_gates.add_edge(last_index[q2], nnodes);
					last_index[q1] = nnodes;
					last_index[q2] = nnodes;
					nnodes += 1;
				}
			}
		}
	}

	for (int i = 0; i < m_nqubits; i++)
	{
		if (!single_qubit_gates[i].empty())
		{
			std::vector<gate> gates;
			gates.insert(gates.end(), single_qubit_gates[i].begin(), single_qubit_gates[i].end());
			grouped_gates.add_vertex({ gates, {i} });
			last_index[i] = nnodes;
			single_qubit_gates[i].clear();
			nnodes += 1;
		}
	}
}

void SU4TopologyMatch::bfs(int start, std::vector< std::vector<int> > &dist)
{
	std::queue<int> q;
	std::set<int> visited;
	int v;
	std::pair<int, int> edge;

	visited.insert(start);
	q.push(start);
	dist[start][start] = 0;
	while (!q.empty())
	{
		v = q.front();
		q.pop();
		for (auto edge : m_coupling_map)
		{
			if (edge.first == v && visited.find(edge.second) == visited.end())
			{
				visited.insert(edge.second);
				q.push(edge.second);
				dist[start][edge.second] = dist[start][v] + 1;
			}
			else if (edge.second == v && visited.find(edge.first) == visited.end())
			{
				visited.insert(edge.first);
				q.push(edge.first);
				dist[start][edge.first] = dist[start][v] + 1;
			}
		}
	}
}


void SU4TopologyMatch::a_star_search(std::set<std::pair<int, int>>& applicable_gates, std::vector<int> & map, std::vector<int> & loc, std::set<std::pair<int, int>>& free_swaps, node & result)
{
	std::priority_queue<node, std::vector<node>, node_cmp> q;

	int tmp_qubit1, tmp_qubit2;
	std::set<int> used_qubits;
	std::set<int> interacted_qubits;

	for (auto g : applicable_gates)
	{
		used_qubits.insert(g.first);
		used_qubits.insert(g.second);
	}

	node current;
	current.cost_fixed = 0;
	current.cost_heur = 0;
	current.qubits = map;
	current.locations = loc;
	current.is_goal = false;
	current.swaps = std::vector<std::pair<int, int>>();

	q.push(current);

	// perform A* search
	while (!q.top().is_goal)
	{
		current = q.top();
		q.pop();

		// determine all successor nodes (one for( each SWAP gate that can be applied)
		for (auto edge : m_coupling_map)
		{
			// apply only SWAP operations including at least one qubit in used_qubits 
			if (used_qubits.find(current.qubits[edge.first]) == used_qubits.end()
				&& used_qubits.find(current.qubits[edge.second]) == used_qubits.end())
			{
				continue;
			}
			std::pair<int, int> g;

			// do not apply the same SWAP gate twice in a row
			if (current.swaps.size() > 0)
				g = current.swaps[current.swaps.size() - 1];
			if (g.first == edge.first && g.second == edge.second)
				continue;


			node new_node;
			new_node.swaps = current.swaps;
			new_node.swaps.push_back(edge);

			// initialize the new node with the mapping of the current node 
			new_node.qubits = current.qubits;
			new_node.locations = current.locations;

			// update mapping of the qubits resulting from adding a SWAP gate
			tmp_qubit1 = new_node.qubits[edge.first];
			tmp_qubit2 = new_node.qubits[edge.second];
			new_node.qubits[edge.first] = tmp_qubit2;
			new_node.qubits[edge.second] = tmp_qubit1;

			if (tmp_qubit1 != -1)
				new_node.locations[tmp_qubit1] = edge.second;
			if (tmp_qubit2 != -1)
				new_node.locations[tmp_qubit2] = edge.first;

			// determine fixed cost of new node;
			interacted_qubits.clear();
			new_node.cost_fixed = 0;
			for (auto edge : new_node.swaps)
			{
				// only add the cost of a swap gate if it is not"free"
				if (interacted_qubits.find(edge.first) != interacted_qubits.end()
					|| interacted_qubits.find(edge.first) != interacted_qubits.end()
					|| free_swaps.find(edge) == free_swaps.end())
				{
					new_node.cost_fixed += 1;
				}
				interacted_qubits.insert(edge.first);
				interacted_qubits.insert(edge.second);
			}

			new_node.is_goal = false;
			new_node.cost_heur = 0;

			// check wheter a goal state is reached (i.e. whether any gate can be applied) and determine heuristic cost
			for (auto g : applicable_gates)
			{
				if (m_dist[new_node.locations[g.first]][new_node.locations[g.second]] == 1)
				{
					new_node.is_goal = true;
				}
				// estimate remaining cost (the heuristic is not necessarily admissible and, hence, may yield to sub-optimal local solutions)
				new_node.cost_heur += m_dist[new_node.locations[g.first]][new_node.locations[g.second]] - 1;
			}

			q.push(new_node);
		}
	}

	result = q.top();
}

void SU4TopologyMatch::add_rewritten_gates(std::vector<gate> gates_original, std::vector<int> locations, std::vector<gate> &compiled_circuit)
{
	gate new_gate;
	for (auto g : gates_original)
	{
		new_gate = g;
		if (g.control != -1)
		{
			new_gate.control = locations[g.control];
		}
		new_gate.target = locations[g.target];
		compiled_circuit.push_back(new_gate);
	}
}

void  SU4TopologyMatch::find_initial_permutation(gates_digraph grouped_gates, node & result)
{
	int q0, q1;
	node init_perm;
	std::set<pair<int, int>> initial_gates;
	std::vector <int> first_interaction(m_nqubits, -1);

	// search for "best initial mapping" (regarding a certain heuristic) using an A* search algorithm.This is only feasable for a small number of qubits
	if (m_used_qubits <= 8)  
	{
		for (auto vertex : grouped_gates.vertexs)
		{
			int vertex_id = vertex.first;
			std::vector<int> vertex_qubits = vertex.second.second;
			std::vector<gate>  vertex_gates = vertex.second.first;

			int degree = grouped_gates.in_degree(vertex_id);
			if (degree == 0)
			{
				if (vertex_qubits.size() != 2)
					continue;

				q0 = vertex_qubits[0];
				q1 = vertex_qubits[1];
				// the mapping for all gates in the first layer shall be satified
				initial_gates.insert(std::pair<int, int>(q0, q1));

				// determine the qubit with which q0 and q1 interact next
				for (auto edge : grouped_gates.edges)
				{
					if (edge.first == vertex_id)
					{
						size_t successors_id = edge.second;
						std::vector<int> successors_qubits = grouped_gates.vertexs[successors_id].second;
						if (q0 == successors_qubits[0])
							first_interaction[successors_qubits[1]] = q0;
						else if (q0 == successors_qubits[1])
							first_interaction[successors_qubits[0]] = q0;
						else if (q1 == successors_qubits[0])
							first_interaction[successors_qubits[1]] = q1;
						else if (q1 == successors_qubits[1])
							first_interaction[successors_qubits[0]] = q1;
					}
				}
			}
		}
	}

	// call an A* algorithm to determe the best initial mapping
	std::priority_queue<node, std::vector<node>, node_cmp> q;
	int min_dist;
	std::pair<int, int> gate, edge;

	// create a new node representing the initial mapping (none of the qubits is mapped yet);
	node current;
	current.cost_fixed = 0;
	current.cost_heur = 0;
	current.locations = std::vector<int >(m_nqubits, -1);
	current.qubits = std::vector<int >(m_nqubits, -1);

	for (auto gate : initial_gates)
		current.remaining_gates.push_back(gate);

	q.push(current);

	// perform A* search
	while (q.top().remaining_gates.size() != 0)
	{
		current = q.top();
		q.pop();
		gate = current.remaining_gates.back();
		current.remaining_gates.pop_back();

		// determine all successor nodes (a gate group acting on a pair of qubits can be applied to any edge in the coupling map)
		// we enforce mapping these groups to an edge in the coupling map in order to avoid SWAPs before appliying the first gate
		for (auto edge : m_coupling_map)
		{
			if (current.qubits[edge.first] != -1 || current.qubits[edge.second] != -1)
				continue;

			// create a new node and initialize the new node with the mapping of the current node 
			node new_node;
			new_node.locations = current.locations;
			new_node.qubits = current.qubits;
			new_node.remaining_gates = current.remaining_gates;
			new_node.qubits[edge.first] = gate.first;
			new_node.locations[gate.first] = edge.first;
			new_node.qubits[edge.second] = gate.second;
			new_node.locations[gate.second] = edge.second;
			new_node.cost_fixed = current.cost_fixed;
			new_node.cost_heur = 0;

			if (first_interaction[gate.first] != -1
				&& new_node.locations[first_interaction[gate.first]] != -1)
			{
				new_node.cost_fixed += m_dist[new_node.locations[gate.first]][new_node.locations[first_interaction[gate.first]]];
			}
			else
			{
				min_dist = m_nqubits;
				for (int i = 0; i < m_nqubits; i++)
				{
					if (new_node.qubits[i] == -1)
						min_dist = std::min(min_dist, m_dist[new_node.locations[gate.first]][i]);
				}
				new_node.cost_heur += min_dist;
			}
			if (first_interaction[gate.second] != -1 && new_node.locations[first_interaction[gate.second]] != -1)
			{
				new_node.cost_fixed += m_dist[new_node.locations[gate.second]][new_node.locations[first_interaction[gate.second]]];
			}
			else
			{
				min_dist = m_nqubits;
				for (int i = 0; i < m_nqubits; i++)
				{
					if (new_node.qubits[i] == -1)
						min_dist = std::min(min_dist, m_dist[new_node.locations[gate.second]][i]);
					new_node.cost_heur += min_dist;
				}
			}

			q.push(new_node);

			// create a second new node (since there are two qubits involved) ;
			new_node.locations = current.locations;
			new_node.qubits = current.qubits;
			new_node.remaining_gates = current.remaining_gates;
			new_node.qubits[edge.second] = gate.first;
			new_node.locations[gate.first] = edge.second;
			new_node.qubits[edge.first] = gate.second;
			new_node.locations[gate.second] = edge.first;
			new_node.cost_fixed = current.cost_fixed;
			new_node.cost_heur = 0;

			if (first_interaction[gate.first] != -1 && new_node.locations[first_interaction[gate.first]] != -1)
			{
				new_node.cost_fixed += m_dist[new_node.locations[gate.first]][new_node.locations[first_interaction[gate.first]]];
			}
			else
			{
				min_dist = m_nqubits;
				for (int k = 0; k < m_nqubits; k++)
				{
					if (new_node.qubits[k] == -1)
						min_dist = std::min(min_dist, m_dist[new_node.locations[gate.first]][k]);
				}
				new_node.cost_heur += min_dist;
			}

			if (first_interaction[gate.second] != -1 && new_node.locations[first_interaction[gate.second]] != -1)
			{
				new_node.cost_fixed += m_dist[new_node.locations[gate.second]][new_node.locations[first_interaction[gate.second]]];
			}
			else
			{
				min_dist = m_nqubits;
				for (int k = 0; k < m_nqubits; k++)
				{
					if (new_node.qubits[k] == -1)
						min_dist = std::min(min_dist, m_dist[new_node.locations[gate.second]][k]);
				}
				new_node.cost_heur += min_dist;
			}

			q.push(new_node);
		}
	}

	result = q.top();
}

void SU4TopologyMatch::a_star_mapper(gates_digraph grouped_gates, std::vector<gate> &compiled_circuit)
{
	std::vector<std::pair<int, int> > applied_gates;

	// determine the minimal distances between two qubits
	m_dist.resize(m_nqubits);
	for (int i = 0; i < m_nqubits; i++)
	{
		m_dist[i].resize(m_nqubits);
	}
	for (int i = 0; i < m_nqubits; i++)
	{
		bfs(i, m_dist);
	}

	// locations[q1] for a logical qubit q1 gives the physical qubit that q1 is mapped to
	// qubits[Q1] for a physical qubit Q1 gives the logaic qubit that is mapped to Q1
	std::vector<int>  locations(m_nqubits, -1);
	std::vector<int>  qubits(m_nqubits, -1);

	int q0, q1;

	std::set<std::pair<int, int> > applicable_gates;

	std::set<int> used_qubits;
	std::set<std::pair<int, int>> free_swaps;
	int min_dist = -1;
	int min_q1 = -1;
	int min_q2 = -1;

	node initial_perm_result;
	find_initial_permutation(grouped_gates, initial_perm_result);
	locations = initial_perm_result.locations;
	qubits = initial_perm_result.qubits;

	// conduct the mapping of the circuit
	while (grouped_gates.vertexs.size() > 0)
	{
		// add all gates that can be directly applied to the circuit
		while (true)
		{
			std::set<int> remove_vertex_id;
			applicable_gates.clear();
			for (auto vertex : grouped_gates.vertexs)
			{
				int vertex_id = vertex.first;
				std::vector<int> vertex_qubits = vertex.second.second;
				std::vector<gate>  vertex_gates = vertex.second.first;

				int degree = grouped_gates.in_degree(vertex_id);
				if (degree == 0)
				{
					if (vertex_qubits.size() != 2)
					{
						// add single qubit gates to the compiled circuit
						for (auto q : vertex_qubits)
						{
							for (int qq = 0; qq < m_nqubits; qq++)
							{
								if (locations[qq] == -1)
								{
									locations[qq] = q;
									qubits[q] = qq;
									break;
								}
							}
						}
						add_rewritten_gates(vertex_gates, locations, compiled_circuit);
						remove_vertex_id.insert(vertex_id);
					}
					else
					{
						// map all yet unmapped qubits that occur in gates that can be applied
						q0 = vertex_qubits[0];
						q1 = vertex_qubits[1];

						if (locations[q0] == -1 && locations[q1] == -1)
						{
							// case: both qubits are not yet mapped
							min_dist = m_nqubits;
							min_q1 = -1;
							min_q2 = -1;
							// find best initial mapping
							for (int i = 0; i < m_nqubits; i++)
							{
								for (int j = i + 1; j < m_nqubits; j++)
								{
									if (qubits[i] == -1 && qubits[j] == -1 && m_dist[i][j] < min_dist)
									{
										min_dist = m_dist[i][j];
										min_q1 = i;
										min_q2 = j;
									}
								}
							}
							locations[q0] = min_q1;
							locations[q1] = min_q2;
							qubits[min_q1] = q0;
							qubits[min_q2] = q1;
						}
						else if (locations[q0] == -1)
						{
							// case: only q0 is not yet mapped
							min_dist = m_nqubits;
							min_q1 = -1;
							// find best initial mapping
							for (int i = 0; i < m_nqubits; i++)
							{
								if (qubits[i] == -1 && m_dist[i][locations[q1]] < min_dist)
								{
									min_dist = m_dist[i][locations[q1]];
									min_q1 = i;
								}
							}
							locations[q0] = min_q1;
							qubits[min_q1] = q0;
						}
						else if (locations[q1] == -1)
						{
							// case: only q1 is not yet mapped
							min_dist = m_nqubits;
							min_q1 = -1;
							// find best initial mapping
							for (int i = 0; i < m_nqubits; i++)
							{
								if (qubits[i] == -1 && m_dist[i][locations[q0]] < min_dist)
								{
									min_dist = m_dist[i][locations[q0]];
									min_q1 = i;
								}
							}
							locations[q1] = min_q1;
							qubits[min_q1] = q1;
						}

						// gates with a distance of 1 can be directly applied
						if (m_dist[locations[q0]][locations[q1]] == 1)
						{
							add_rewritten_gates(vertex_gates, locations, compiled_circuit);
							if (m_coupling_map.find(std::pair<int, int>(locations[q0], locations[q1])) != m_coupling_map.end())
								applied_gates.push_back(std::pair<int, int>(locations[q0], locations[q1]));
							else
								applied_gates.push_back(std::pair<int, int>(locations[q1], locations[q0]));

							// remove nodes representing the added gates
							remove_vertex_id.insert(vertex_id);
						}
						else
						{
							// gates with a distance greater than 1 can potentially be applied(after fixing the mapping)
							applicable_gates.insert({ q0, q1 });
						}
					}
				}
			}

			if (remove_vertex_id.size() == 0)
			{
				break;
			}
			else
			{
				for (auto id : remove_vertex_id)
				{
					grouped_gates.remove_vertex(id);
				}
			}
		}

		// check whether all gates have been successfully applied
		if (applicable_gates.size() == 0)
			break;

		// determine which SWAPs can be applied for "free".
		//A SWAP on qubits q0 and q1 does not cost anything if the group of gates between q0 and q1 have been directly applied before it.
		used_qubits.clear();
		free_swaps.clear();
		for (int i = applied_gates.size() - 1; i > -1; i--)
		{
			if (used_qubits.find(applied_gates[i].first) == used_qubits.end()
				&& used_qubits.find(applied_gates[i].second) == used_qubits.end())
			{
				free_swaps.insert(applied_gates[i]);
				used_qubits.insert(applied_gates[i].first);
				used_qubits.insert(applied_gates[i].second);
			}
		}

		node result;

		// Apply A* to find a permutation such that further gates can be applied
		a_star_search(applicable_gates, qubits, locations, free_swaps, result);

		locations = result.locations;
		qubits = result.qubits;
		m_qubits = qubits;
		m_locations = locations;

		// add SWAPs to the compiled circuit to modify the current the mapping
		for (auto swap : result.swaps)
		{
			gate cnot1, cnot2;
			cnot1.type = cnot2.type = GateType::CNOT_GATE;
			cnot1.is_flip = cnot2.is_flip = false;
			cnot1.is_dagger = cnot2.is_dagger = false;

			cnot1.control = swap.first;
			cnot1.target = swap.second;
			cnot2.control = swap.second;
			cnot2.target = swap.first;

			compiled_circuit.push_back(cnot1);
			compiled_circuit.push_back(cnot2);
			compiled_circuit.push_back(cnot1);
			applied_gates.push_back(swap);
		}
	}
}

void SU4TopologyMatch::build_qprog(std::vector<gate>  compiled_circuit, QProg & mapped_prog)
{
	auto loc = m_locations;
	std::map<int, int> mapping_result;  // ph => QVec id
	for (int i = 0; i < loc.size(); i++)
	{
		if (loc[i] != -1)
		{
			int index = mapping_result.size();
			mapping_result.insert(std::pair<int, int>(loc[i], index));
		}
	}
	QVec qv;
	for (int i = 0; i < loc.size(); i++)
	{
		if (loc[i] != -1)
			qv.push_back(m_qvm->allocateQubitThroughPhyAddress(loc[i]));
	}

	m_qv = qv;

	for (auto g : compiled_circuit)
	{
		if (g.control != -1)
		{
			if (m_coupling_map.find(std::pair<int, int>(g.control, g.target)) == m_coupling_map.end()
				&& m_coupling_map.find(std::pair<int, int>(g.target, g.control)) != m_coupling_map.end())
			{
				g.is_flip = true;
				int temp = g.control;
				g.control = g.target;
				g.target = temp;
			}

			auto iter_c = mapping_result.find(g.control);
			if (iter_c == mapping_result.end())
			{
				QCERR("find mapping_result error!");
				throw invalid_argument("find mapping_result error!");
			}
			g.control = iter_c->second;
		}

		auto iter_t = mapping_result.find(g.target);
		if (iter_t == mapping_result.end())
		{
			QCERR("find mapping_result error!");
			throw invalid_argument("find mapping_result error!");
		}
		g.target = iter_t->second;


		switch (g.type)
		{
		case GateType::U1_GATE:
		{
			double angle = g.param[0];
			QGate single_angle_gate = U1(qv[g.target], angle);
			mapped_prog << single_angle_gate;
		}
		break;

		case GateType::CNOT_GATE:
		{
			QGate cnot_gate = CNOT(qv[g.control], qv[g.target]);
			if (g.is_flip)
			{
				mapped_prog << U2(qv[g.target], 0 , PI)
					<< U2(qv[g.control], 0, PI)
					<< cnot_gate
					<< U2(qv[g.control], 0, PI)
					<< U2(qv[g.target], 0, PI);
			}
			else
			{
				mapped_prog << cnot_gate;
			}
		}
		break;
		case  GateType::U2_GATE:
		{
			QGate u2_gate = U2(qv[g.target], g.param[0], g.param[1]);
			mapped_prog << u2_gate;
		}
		break;
		case  GateType::U3_GATE:
		{
			QGate u3_gate = U3(qv[g.target], g.param[0], g.param[1], g.param[2]);
			mapped_prog << u3_gate;
		}
		break;
		default:
		{
			QCERR("error! unsupported QGate");
			throw invalid_argument("error! unsupported QGate");
		}
		break;
		}
	}
}

void SU4TopologyMatch::mapping_qprog(QProg prog, QProg &mapped_prog)
{
	gates_digraph grouped_gates ;
	std::vector<gate> original_circuit, compiled_circuit;
	transform_qprog(prog, original_circuit);

	pre_processing(original_circuit, grouped_gates);

	a_star_mapper(grouped_gates, compiled_circuit);
	
	gates_digraph compiled_grouped_gates;

	pre_processing(compiled_circuit, compiled_grouped_gates);

	build_qprog(compiled_circuit, mapped_prog);
}

QProg  QPanda::su4_circiut_topology_match(QProg prog, QVec &qv, QuantumMachine *machine, ArchType arch_type)
{
	if (nullptr == machine)
	{
		QCERR("Quantum machine is nullptr");
		throw std::invalid_argument("Quantum machine is nullptr");
	}

	QProg outprog;
	SU4TopologyMatch match = SU4TopologyMatch(machine, qv, arch_type);
	match.mapping_qprog(prog, outprog);
	return outprog;
}
