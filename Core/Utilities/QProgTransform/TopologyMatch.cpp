#include "Core/Utilities/QProgTransform/TopologyMatch.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"

USING_QPANDA
using namespace std;

#define DUMMY_SWAP_GATE  -2

#define PRINTF_MAPPING_RESULT 0 

TopologyMatch::TopologyMatch(QuantumMachine* machine, QProg prog,
	const std::string& conf)
	:m_prog(prog)
{
	m_nqubits = machine->getAllocateQubit();
	m_qvm = machine;
	buildGraph(m_graph, m_positions, conf);

	if (m_nqubits > m_positions)
	{
		QCERR("ERROR before mapping: more logical qubits than physical ones!");
		throw runtime_error("ERROR before mapping: more logical qubits than physical ones!");
	}

	m_dist = buildDistTable(m_positions, m_graph, m_swap_cost, m_flip_cost);
}

TopologyMatch::~TopologyMatch()
{
}


bool TopologyMatch::isContains(vector<int> v, int e)
{
	for (auto it = v.begin(); it != v.end(); it++)
	{
		if (*it == e)
			return true;
	}
	return false;
}

void  TopologyMatch::buildGraph(set<edge>& graph, size_t& positions, const std::string& conf)
{
	graph.clear();
	vector<vector<double>> qubit_matrix;
	int qubit_num = 0;
	JsonConfigParam config;
	config.load_config(conf);
	config.getMetadataConfig(qubit_num, qubit_matrix);
	positions = qubit_num;
	for (int i = 0; i < positions; i++)
	{
		for (int j = 0; j < positions; j++)
		{
			if (qubit_matrix[i][j] > 1e-6)
			{
				edge e;
				e.v1 = i;
				e.v2 = j;
				graph.insert(e);
			}
		}
	}
}

//Breadth first search algorithm to determine the shortest paths between two physical qubits
int TopologyMatch::breadthFirstSearch(int start, int goal, const set<edge>& graph, size_t swap_cost, size_t flip_cost)
{
	queue<vector<int> > queue;
	vector<int> v;
	v.push_back(start);
	queue.push(v);
	vector<vector<int> > solutions;

	int length = 0;
	set<int> successors;
	while (!queue.empty())
	{
		v = queue.front();
		queue.pop();
		int current = v[v.size() - 1];
		if (current == goal)
		{
			length = v.size();
			solutions.push_back(v);
			break;
		}
		else
		{
			successors.clear();
			for (auto e : graph)
			{
				if (e.v1 == current && !isContains(v, e.v2))
				{
					successors.insert(e.v2);
				}
				if (e.v2 == current && !isContains(v, e.v1))
				{
					successors.insert(e.v1);
				}
			}
			for (set<int>::iterator it = successors.begin(); it != successors.end(); it++)
			{
				vector<int> v2 = v;
				v2.push_back(*it);
				queue.push(v2);
			}
		}
	}
	while (!queue.empty() && queue.front().size() == length)
	{
		if (queue.front()[queue.front().size() - 1] == goal)
		{
			solutions.push_back(queue.front());
		}
		queue.pop();
	}

	for (int i = 0; i < solutions.size(); i++)
	{
		vector<int> v = solutions[i];
		for (int j = 0; j < v.size() - 1; j++)
		{
			edge e;
			e.v1 = v[j];
			e.v2 = v[j + 1];
			if (graph.find(e) != graph.end())
			{
				return (length - 2) * swap_cost;
			}
		}
	}

	return (length - 2) * swap_cost + flip_cost;
}

vector<vector<int> >
TopologyMatch::buildDistTable(int positions,
	const set<edge>& graph, size_t swap_cost, size_t flip_cost)
{
	vector<vector<int> > dist;

	dist.resize(positions);

	for (int i = 0; i < positions; ++i)
		dist[i].resize(positions);

	for (int i = 0; i < dist.size(); i++)
	{
		for (int j = 0; j < dist[i].size(); j++)
		{
			if (i != j)
			{
				dist[i][j] = breadthFirstSearch(i, j, graph, swap_cost, flip_cost);
			}
			else
			{
				dist[i][i] = 0;
			}
		}
	}
	return dist;
}


void TopologyMatch::createNodeFromBase(node base_node,
	vector<edge>& swaps, int nswaps, node& new_node)
{
	new_node.qubits = base_node.qubits;
	new_node.locations = base_node.locations;

	new_node.swaps = vector<vector<edge> >();
	new_node.nswaps = base_node.nswaps + nswaps;
	for (vector<vector<edge> >::iterator it2 = base_node.swaps.begin();
		it2 != base_node.swaps.end(); it2++)
	{
		vector<edge> new_v(*it2);
		new_node.swaps.push_back(new_v);
	}

	new_node.depth = base_node.depth + 5;
	new_node.cost_fixed = base_node.cost_fixed + m_swap_cost * nswaps;
	new_node.cost_heur = 0;

	vector<edge> new_swaps;
	for (int i = 0; i < nswaps; i++)
	{
		new_swaps.push_back(swaps[i]);
		int tmp_qubit1 = new_node.qubits[swaps[i].v1];
		int tmp_qubit2 = new_node.qubits[swaps[i].v2];

		new_node.qubits[swaps[i].v1] = tmp_qubit2;
		new_node.qubits[swaps[i].v2] = tmp_qubit1;

		if (tmp_qubit1 != -1)
		{
			new_node.locations[tmp_qubit1] = swaps[i].v2;
		}
		if (tmp_qubit2 != -1)
		{
			new_node.locations[tmp_qubit2] = swaps[i].v1;
		}
	}
	new_node.swaps.push_back(new_swaps);
	new_node.done = 1;
}

void TopologyMatch::calculateHeurCostForNextLayer(int next_layer, node& new_node)
{
	new_node.cost_heur2 = 0;
	if (next_layer != -1)
	{
		for (auto g : m_layers[next_layer])
		{
			if (g.control != -1)
			{
				vector<vector<int> > dist = m_dist;
				if (new_node.locations[g.control] == -1
					&& new_node.locations[g.target] == -1)
				{
				}
				else if (new_node.locations[g.control] == -1)
				{
					int min = 1000;
					for (int i = 0; i < new_node.qubits.size(); i++)
					{
						if (new_node.qubits[i] == -1
							&& dist[i][new_node.locations[g.target]] < min)
						{
							min = dist[i][new_node.locations[g.target]];
						}
					}
					new_node.cost_heur2 = new_node.cost_heur2 + min;
				}
				else if (new_node.locations[g.target] == -1)
				{
					int min = 1000;
					for (int i = 0; i < new_node.qubits.size(); i++)
					{
						if (new_node.qubits[i] == -1
							&& dist[new_node.locations[g.control]][i] < min)
						{
							min = dist[new_node.locations[g.control]][i];
						}
					}
					new_node.cost_heur2 = new_node.cost_heur2 + min;
				}
				else
				{
					new_node.cost_heur2 = new_node.cost_heur2 + \
						dist[new_node.locations[g.control]][new_node.locations[g.target]];
				}
			}
		}
	}
}

void TopologyMatch::expandNode(const vector<int>& qubits, int qubit,
	vector<edge>& swaps, int nswaps,
	vector<int>& used, node base_node,
	const vector<gate>& layer_gates, int next_layer)
{
	if (qubit == qubits.size())
	{
		//base case: insert node into queue
		if (nswaps == 0)
		{
			return;
		}
		node new_node;
		createNodeFromBase(base_node, swaps, nswaps, new_node);

		for (auto g : layer_gates)
		{
			if (g.control != -1)
			{
				vector<vector<int> > dist = m_dist;

				auto n_c = new_node.locations[g.control];
				auto n_t = new_node.locations[g.target];
				new_node.cost_heur = new_node.cost_heur + dist[n_c][n_t];

				if (dist[n_c][n_t] > m_flip_cost)
				{
					new_node.done = 0;
				}
			}
		}

		//Calculate heuristics for the cost of the following layer
		calculateHeurCostForNextLayer(next_layer, new_node);

		m_nodes.push(new_node);
	}
	else
	{
		expandNode(qubits, qubit + 1, swaps, nswaps, \
			used, base_node, layer_gates, next_layer);

		for (auto e : m_graph)
		{
			if (e.v1 == base_node.locations[qubits[qubit]]
				|| e.v2 == base_node.locations[qubits[qubit]])
			{
				if (!used[e.v1] && !used[e.v2])
				{
					used[e.v1] = 1;
					used[e.v2] = 1;
					swaps[nswaps].v1 = e.v1;
					swaps[nswaps].v2 = e.v2;
					expandNode(qubits, qubit + 1, swaps, nswaps + 1, used,
						base_node, layer_gates, next_layer);
					used[e.v1] = 0;
					used[e.v2] = 0;
				}
			}
		}
	}
}

int TopologyMatch::getNextLayer(int layer)
{
	int next_layer = layer + 1;
	while (next_layer < m_layers.size())
	{
		for (auto g : m_layers[next_layer])
		{
			if (g.control != -1)
			{
				return next_layer;
			}
		}
		next_layer++;
	}
	return -1;
}

TopologyMatch::node
TopologyMatch::fixLayerByAStar(int layer,
	vector<int>& map, vector<int>& loc)
{
	int next_layer = getNextLayer(layer);

	node n;
	n.cost_fixed = 0;
	n.cost_heur = n.cost_heur2 = 0;

	n.swaps = vector<vector<edge> >();
	n.done = 1;

	vector<gate> layer_gates = m_layers[layer];
	vector<int> considered_qubits;

	// Find a mapping for all logical qubits 
	// in the CNOTs of the layer that are not yet mapped
	for (auto g : layer_gates)
	{
		if (g.control != -1)
		{
			vector<vector<int> > dist = m_dist;

			considered_qubits.push_back(g.control);
			considered_qubits.push_back(g.target);
			if (loc[g.control] == -1 && loc[g.target] == -1)
			{
				set<edge> possible_edges;
				for (auto it = m_graph.begin(); it != m_graph.end(); it++)
				{
					if (map[it->v1] == -1 && map[it->v2] == -1)
					{
						possible_edges.insert(*it);
					}
				}
				if (!possible_edges.empty())
				{
					edge e = *possible_edges.begin();
					loc[g.control] = e.v1;
					map[e.v1] = g.control;
					loc[g.target] = e.v2;
					map[e.v2] = g.target;
				}
				else
				{
					QCERR("no edge available!");
					throw runtime_error("no edge available!");
				}
			}
			else if (loc[g.control] == -1)
			{
				int min = 1000;
				int min_pos = -1;
				for (int i = 0; i < map.size(); i++)
				{
					if (map[i] == -1 && dist[i][loc[g.target]] < min)
					{
						min = dist[i][loc[g.target]];
						min_pos = i;
					}
				}
				map[min_pos] = g.control;
				loc[g.control] = min_pos;
			}
			else if (loc[g.target] == -1)
			{
				int min = 1000;
				int min_pos = -1;
				for (int i = 0; i < map.size(); i++)
				{
					if (map[i] == -1 && dist[loc[g.control]][i] < min)
					{
						min = dist[loc[g.control]][i];
						min_pos = i;
					}
				}
				map[min_pos] = g.target;
				loc[g.target] = min_pos;
			}
			n.cost_heur = max(n.cost_heur, dist[loc[g.control]][loc[g.target]]);
		}
		else
		{
		}
	}

	if (n.cost_heur > m_flip_cost)
	{
		n.done = 0;
	}
	n.qubits = map;
	n.locations = loc;

	m_nodes.push(n);

	vector<int> used(map.size(), 0);
	vector <edge> edges(considered_qubits.size());

	//Perform an A* search to find the cheapest permuation
	while (!m_nodes.top().done)
	{
		node first_node = m_nodes.top();
		m_nodes.pop();
		expandNode(considered_qubits, 0, edges, 0, \
			used, first_node, layer_gates, next_layer);
	}

	node result = m_nodes.top();

	while (!m_nodes.empty())
	{
		m_nodes.pop();
	}
	return result;
}

void TopologyMatch::mappingQProg(QVec& qv, QProg& mapped_prog)
{
	traversalQProgToLayers();

	vector<int> qubits(m_positions, -1);
	vector<int> locations(m_nqubits, -1);

	vector<gate> all_gates;
	int total_swaps = 0;

	//Fix the mapping of each layer
	for (int i = 0; i < m_layers.size(); i++)
	{
		node result = fixLayerByAStar(i, qubits, locations);

		locations.clear();
		qubits.clear();
		locations = result.locations;
		qubits = result.qubits;

		//The first layer does not require a permutation of the qubits
		if (i != 0)
		{
			//Add the required SWAPs to the circuits
			for (auto swaps : result.swaps)
			{
				for (auto e : swaps)
				{
					// Insert a dummy SWAP gate to allow for
					// tracking the m_positions of the logical qubits
					gate g;
					g.control = e.v1;
					g.target = e.v2;
					g.type = DUMMY_SWAP_GATE;

					all_gates.push_back(g);
					total_swaps++;
				}
			}
		}

		//Add all gates of the layer to the circuit
		vector<gate> layer = m_layers[i];
		for (auto g : layer)
		{
			if (g.control == -1)
			{
				if (locations[g.target] == -1)
				{
					// handle the case that the qubit is not yet mapped. 
					// this happens if the qubit has not yet occurred in a CNOT gate
					gate g2 = g;
					g2.target = -g.target - 1;
					all_gates.push_back(g2);
				}
				else
				{
					g.target = locations[g.target];
					all_gates.push_back(g);
				}
			}
			else
			{
				g.target = locations[g.target];
				g.control = locations[g.control];

				edge e;
				e.v1 = g.control;
				e.v2 = g.target;
				all_gates.push_back(g);
			}
		}
	}

	//Fix the position of the single qubit gates
	for (vector<gate>::reverse_iterator it = all_gates.rbegin(); it != all_gates.rend(); it++)
	{
		if (DUMMY_SWAP_GATE == it->type)
		{
			int tmp_qubit1 = qubits[it->control];
			int tmp_qubit2 = qubits[it->target];
			qubits[it->control] = tmp_qubit2;
			qubits[it->target] = tmp_qubit1;

			if (tmp_qubit1 != -1)
			{
				locations[tmp_qubit1] = it->target;
			}
			if (tmp_qubit2 != -1)
			{
				locations[tmp_qubit2] = it->control;
			}

		}
		if (it->target < 0)
		{
			int target = -(it->target + 1);
			if (locations[target] == -1)
			{
				//This qubit occurs only in single qubit gates -> it can be mapped to an arbirary physical qubit
				int loc = 0;
				while (qubits[loc] != -1)
				{
					loc++;
				}
				locations[target] = loc;
				qubits[loc] = target;
			}
			it->target = locations[target];
		}
	}

#if PRINTF_MAPPING_RESULT
	cout << "The mapping results : " << endl;
	for (int i = 0; i < locations.size(); i++)
	{
		if (locations[i] != -1)
		{
			cout << i << " -> " << locations[i] << endl;
		}
	}

	cout << "Swap qubits : " << endl;
	for (auto g : all_gates)
	{
		if (DUMMY_SWAP_GATE == g.type)
		{
			cout << g.target << " <-> " << g.control << endl;
		}
	}

#endif

	buildResultingQProg(all_gates, locations, qv, mapped_prog);
}

vector<int> TopologyMatch::getGateQaddrs(const QProgDAGVertex& _v)
{
	vector<int> qaddrs;
	for (auto& _q : _v.m_node->m_control_vec)
		qaddrs.push_back(_q->get_phy_addr());

	for (auto& _q : _v.m_node->m_qubits_vec)
		qaddrs.push_back(_q->get_phy_addr());

	return qaddrs;
		}

void TopologyMatch::buildResultingQProg(const vector<gate>& resulting_gates, vector<int> loc, QVec& qv, QProg& prog)
{
	map<int, int> mapping_result;  // ph => QVec id
	vector<pair<int, int>>swap_vec;

	for (auto g : resulting_gates)
	{
		if (DUMMY_SWAP_GATE == g.type)
			swap_vec.push_back(make_pair(g.target, g.control));
	}

	for (int i = 0; i < loc.size(); i++)
	{
		if (loc[i] != -1)
		{
			int idx = mapping_result.size();
			mapping_result[loc[i]] = idx;
		}
	}

	for (auto swap : swap_vec)
	{
		if (mapping_result.find(swap.first) == mapping_result.end())
		{
			int idx = mapping_result.size();
			mapping_result[swap.first] = idx;
		}
		if (mapping_result.find(swap.second) == mapping_result.end())
		{
			int idx = mapping_result.size();
			mapping_result[swap.second] = idx;
		}
	}

	for (auto it : swap_vec)
	{
		swap(mapping_result[it.first], mapping_result[it.second]);
	}

	loc.resize(mapping_result.size());
	for (auto map : mapping_result)
	{
		loc[map.second] = map.first;
	}

	qv.clear();
	for (int i = 0; i < loc.size(); i++)
	{
		qv.push_back(m_qvm->allocateQubitThroughPhyAddress(loc[i]));
	}

	map<int, QVec> barrier_qv_map;
	for (auto g : resulting_gates)
	{
		Qubit* q_ctrl;
		Qubit* q_tar;
		if (g.control != -1)
		{
			q_ctrl = qv[mapping_result[g.control]];
		}

		q_tar = qv[mapping_result[g.target]];

		if (g.type == DUMMY_SWAP_GATE)
		{
#if 0
			prog << U3(q_ctrl, PI / 2.0, 0, PI) << CZ(q_ctrl, q_tar) << U3(q_ctrl, PI / 2.0, 0, PI)
				<< U3(q_tar, PI / 2.0, 0, PI) << CZ(q_tar, q_ctrl) << U3(q_tar, PI / 2.0, 0, PI)
				<< U3(q_ctrl, PI / 2.0, 0, PI) << CZ(q_ctrl, q_tar) << U3(q_ctrl, PI / 2.0, 0, PI);
#else
			prog << SWAP(q_ctrl, q_tar);
#endif

			continue;
		}

		auto _v_id = g.vertex_id;
		auto _v = m_dag->get_vertex(_v_id);
		vector<int> qaddrs = getGateQaddrs(_v);
		shared_ptr<QNode> qnode = *(_v.m_node->m_itr);

		if (g.control == -1)
		{
			if (_v.m_type == BARRIER_GATE)
			{
				barrier_qv_map[g.barrier_id].push_back(q_tar);
				if (barrier_qv_map[g.barrier_id].size() == g.barrier_size)
				{
					prog << BARRIER(barrier_qv_map[g.barrier_id]);
				}
			}
			else
			{
				auto pgate = dynamic_pointer_cast<AbstractQGateNode>(*(_v.m_node->m_itr));
 				auto g = QGate(pgate);
				QGate new_gate = deepCopy(g);
				new_gate.remap({ q_tar });
				prog << new_gate;
			}
		}
		else
		{
			auto pgate = dynamic_pointer_cast<AbstractQGateNode>(*(_v.m_node->m_itr));
			auto g = QGate(pgate);
			QGate new_gate = deepCopy(g);
			if (_v.m_node->m_control_vec.size() == 1)
			{
				new_gate.clear_control();
				new_gate.remap({ q_tar });
				prog << new_gate.control(q_ctrl);
			}
			else
			{
				new_gate.remap({ q_ctrl, q_tar });
				prog << new_gate;
			}
		}
	}

    // rebuild measure
    auto mea_info = m_prog.get_measure_qubits_cbits();
    for (const auto& iter : mea_info)
    {
        auto q_tar = qv[iter.first->get_phy_addr()]; 
        prog << Measure(q_tar, iter.second);
    }
}

void TopologyMatch::traversalQProgToLayers()
{
	if (!m_prog.is_measure_last_pos())
	{
		QCERR("measure is not at the end of the circuit!");
		throw invalid_argument("measure  is not at the end of the circuit!");
	}

	m_dag = qprog_to_DAG(m_prog);
	auto top_seq = m_dag->build_topo_sequence();
	vector<int> top_order;
	int barrier_id = 0;
	for (auto& layer_seq : top_seq)
	{
		vector<gate> gates;
		for (auto& seq : layer_seq)
		{
			auto _v_id = seq.first.m_vertex_num;
			auto tmp_qv = getGateQaddrs(m_dag->get_vertex(_v_id));
            int type = m_dag->get_vertex(_v_id).m_type;
            if (type == (int)DAGNodeType::MEASURE)
            {
                // the measurement operation is in the last layer
                continue;
            }
            else if (type == (int)GateType::BARRIER_GATE)
            {
				for (auto& qaddr : tmp_qv)
				{
					gate g;
					g.vertex_id = _v_id;
					g.control = -1;
					g.target = qaddr;
					g.barrier_id = barrier_id;
					g.barrier_size = tmp_qv.size();
					gates.push_back(g);
				}
				barrier_id++;
			}
			else
			{
				gate g;
				g.vertex_id = _v_id;
				if (tmp_qv.size() == 2)
				{
					g.control = tmp_qv[0];
					g.target = tmp_qv[1];
				}
				else if (tmp_qv.size() == 1)
				{
					g.control = -1;
					g.target = tmp_qv[0];
				}
				else
				{
					QCERR("qubits size error!");
					throw invalid_argument("qubits size error!");
				}
				gates.push_back(g);
			}
		}
		m_layers.push_back(gates);
	}
}

QProg QPanda::topology_match(QProg prog, QVec& qv, QuantumMachine* machine,
	const string& conf /*= CONFIG_PATH*/)
{
	if (nullptr == machine)
	{
		QCERR("Quantum machine is nullptr");
		throw invalid_argument("Quantum machine is nullptr");
	}

	QProg mapped_prog;
	TopologyMatch match = TopologyMatch(machine, prog, conf);
	match.mappingQProg(qv, mapped_prog);

	return mapped_prog;
}