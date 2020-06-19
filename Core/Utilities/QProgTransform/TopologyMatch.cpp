#include "Core/Utilities/QProgTransform/TopologyMatch.h"
#include "Core/Utilities/Tools/XMLConfigParam.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"

USING_QPANDA
using namespace std;
using namespace QGATE_SPACE;

#define DUMMY_SWAP_GATE  -2

#define PRINTF_MAPPING_RESULT 0 


static QGate iSWAPGateNotheta(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
	return iSWAP(targitBit_fisrt, targitBit_second);
}


bool TopologyMatch::isContains(std::vector<int> v, int e)
{
	for (std::vector<int>::iterator it = v.begin(); it != v.end(); it++)
	{
		if (*it == e)
		{
			return true;
		}
	}
	return false;
}

bool TopologyMatch::isReversed(std::set<edge> graph, edge det_edge)
{
	if (graph.find(det_edge) != graph.end())
	{
		return false;
	}
	else
	{
		int tmp = det_edge.v1;
		det_edge.v1 = det_edge.v2;
		det_edge.v2 = tmp;
		if (graph.find(det_edge) == graph.end())
		{
			QCERR("detect edga invalid");
			throw runtime_error("detect edga invalid");
		}
		return true;
	}
}

TopologyMatch::TopologyMatch(QuantumMachine * machine,  SwapQubitsMethod method, ArchType arch_type)
{
	m_nqubits = machine->getAllocateQubit();
	m_qvm = machine;
	buildGraph(arch_type, m_graph, m_positions);

	if (m_nqubits > m_positions)
	{
		QCERR("ERROR before mapping: more logical qubits than physical ones!");
		throw runtime_error("ERROR before mapping: more logical qubits than physical ones!");
	}

	QuantumMetadata metaData("QPandaConfig.xml");
	std::vector<std::string> single_gates_vec, double_gates_vec;
	metaData.getQGate(single_gates_vec, double_gates_vec);
	bool b_suppert_swap = false;
	for (auto gate_name : double_gates_vec)
	{
		if (method == ISWAP_GATE_METHOD && gate_name == "ISWAP")
		{
			b_suppert_swap = true;
			break;
		}
		else if (method == CZ_GATE_METHOD && gate_name == "CZ")
		{
			b_suppert_swap = true;
			break;
		}
		else if (method == CNOT_GATE_METHOD && gate_name == "CNOT")
		{
			b_suppert_swap = true;
			break;
		}
		else if (method == SWAP_GATE_METHOD && gate_name == "SWAP")
		{
			b_suppert_swap = true;
			break;
		}
	}
	if (!b_suppert_swap)
	{
		QCERR("ERROR swap qubits method is not supported!");
		throw runtime_error("ERROR swap qubits method is not supported!");
	}

	m_pTransformSwap = TransformSwapAlgFactory::GetFactoryInstance().CreateByType(method);
	m_swap_cost = m_pTransformSwap->getSwapCost();
	m_flip_cost = m_pTransformSwap->getFlipCost();

	std::vector<std::vector<int> > dist;
	dist = buildDistTable(m_positions, m_graph, m_swap_cost, 4);
	m_gate_dist_map.insert(make_pair(GateType::CNOT_GATE, dist));
	dist = buildDistTable(m_positions, m_graph, m_swap_cost, 0);
	m_gate_dist_map.insert(make_pair(GateType::SWAP_GATE, dist));
	m_gate_dist_map.insert(make_pair(GateType::CZ_GATE, dist));
	m_gate_dist_map.insert(make_pair(GateType::ISWAP_GATE, dist));
	dist = buildDistTable(m_positions, m_graph, m_swap_cost, 2 * m_swap_cost);
	m_gate_dist_map.insert(make_pair(GateType::CPHASE_GATE, dist));
	m_gate_dist_map.insert(make_pair(GateType::CU_GATE, dist));

	m_singleGateFunc.insert(make_pair(GateType::PAULI_X_GATE, X));
	m_singleGateFunc.insert(make_pair(GateType::PAULI_Y_GATE, Y));
	m_singleGateFunc.insert(make_pair(GateType::PAULI_Z_GATE, Z));
	m_singleGateFunc.insert(make_pair(GateType::X_HALF_PI, X1));
	m_singleGateFunc.insert(make_pair(GateType::Y_HALF_PI, Y1));
	m_singleGateFunc.insert(make_pair(GateType::Z_HALF_PI, Z1));
	m_singleGateFunc.insert(make_pair(GateType::HADAMARD_GATE, H));
	m_singleGateFunc.insert(make_pair(GateType::T_GATE, T));
	m_singleGateFunc.insert(make_pair(GateType::S_GATE, S));

	m_singleAngleGateFunc.insert(make_pair(GateType::RX_GATE, RX));
	m_singleAngleGateFunc.insert(make_pair(GateType::RY_GATE, RY));
	m_singleAngleGateFunc.insert(make_pair(GateType::RZ_GATE, RZ));
	m_singleAngleGateFunc.insert(make_pair(GateType::U1_GATE, U1));

	m_doubleGateFunc.insert(make_pair(GateType::CNOT_GATE, CNOT));
	m_doubleGateFunc.insert(make_pair(GateType::CZ_GATE, CZ));
	m_doubleGateFunc.insert(make_pair(GateType::ISWAP_GATE, iSWAPGateNotheta));
	m_doubleGateFunc.insert(make_pair(GateType::SQISWAP_GATE, SqiSWAP));

	m_doubleAngleGateFunc.insert(make_pair(GateType::CPHASE_GATE, CR));
}

TopologyMatch::~TopologyMatch()
{
	if (nullptr != m_pTransformSwap)
	{
		delete m_pTransformSwap;
		m_pTransformSwap = nullptr;
	}
}

void  TopologyMatch::buildGraph(ArchType type, std::set<edge> &graph, size_t &positions)
{
	switch (type)
	{
	case ArchType::IBM_QX5_ARCH:
	{
		graph.clear();
		positions = 16;
		edge e;
		e.v1 = 1;
		e.v2 = 0;
		graph.insert(e);
		e.v1 = 1;
		e.v2 = 2;
		graph.insert(e);
		e.v1 = 2;
		e.v2 = 3;
		graph.insert(e);
		e.v1 = 3;
		e.v2 = 14;
		graph.insert(e);
		e.v1 = 3;
		e.v2 = 4;
		graph.insert(e);
		e.v1 = 5;
		e.v2 = 4;
		graph.insert(e);
		e.v1 = 6;
		e.v2 = 5;
		graph.insert(e);
		e.v1 = 6;
		e.v2 = 11;
		graph.insert(e);
		e.v1 = 6;
		e.v2 = 7;
		graph.insert(e);
		e.v1 = 7;
		e.v2 = 10;
		graph.insert(e);
		e.v1 = 8;
		e.v2 = 7;
		graph.insert(e);
		e.v1 = 9;
		e.v2 = 8;
		graph.insert(e);
		e.v1 = 9;
		e.v2 = 10;
		graph.insert(e);
		e.v1 = 11;
		e.v2 = 10;
		graph.insert(e);
		e.v1 = 12;
		e.v2 = 5;
		graph.insert(e);
		e.v1 = 12;
		e.v2 = 11;
		graph.insert(e);
		e.v1 = 12;
		e.v2 = 13;
		graph.insert(e);
		e.v1 = 13;
		e.v2 = 4;
		graph.insert(e);
		e.v1 = 13;
		e.v2 = 14;
		graph.insert(e);
		e.v1 = 15;
		e.v2 = 0;
		graph.insert(e);
		e.v1 = 15;
		e.v2 = 14;
		graph.insert(e);
		e.v1 = 15;
		e.v2 = 2;
		graph.insert(e);
	}
	break;
	case ArchType::ORIGIN_VIRTUAL_ARCH:
	{
		graph.clear();
		std::vector<std::vector<int>> qubit_matrix;
		int qubit_num = 0;
		XmlConfigParam xml_config;
		xml_config.loadFile("QPandaConfig.xml");
		xml_config.getMetadataConfig(qubit_num, qubit_matrix);
		positions = qubit_num;
		edge e;
		for (int i = 0; i < positions; i++)
		{
			for (int j = 0; j < positions; j++)
			{
				if (qubit_matrix[i][j])
				{
					e.v1 = i;
					e.v2 = j;
					graph.insert(e);
				}
			}
		}

	}
	break;
	default:
		break;
	}
}

//Breadth first search algorithm to determine the shortest paths between two physical qubits
int TopologyMatch::breadthFirstSearch(int start, int goal, const std::set<edge>& graph, size_t swap_cost, size_t flip_cost)
{
	queue<vector<int> > queue;
	vector<int> v;
	v.push_back(start);
	queue.push(v);
	vector<vector<int> > solutions;

	int length;
	std::set<int> successors;
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

std::vector<std::vector<int> > TopologyMatch::buildDistTable(int positions, const std::set<edge> &graph, size_t swap_cost, size_t flip_cost)
{
	std::vector<std::vector<int> > dist;

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

std::vector<std::vector<int> > TopologyMatch::getGateDistTable(int gate_type)
{
	std::vector<std::vector<int> > dist;
	auto iter = m_gate_dist_map.find(gate_type);
	if (iter == m_gate_dist_map.end())
	{
		QCERR("no find!");
		throw runtime_error("no find!");
	}
	dist = iter->second;
	return dist;
}

void TopologyMatch::createNodeFromBase(node base_node, vector<edge> &swaps, int nswaps, node &new_node)
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

void TopologyMatch::calculateHeurCostForNextLayer(int next_layer, node &new_node)
{
	new_node.cost_heur2 = 0;
	if (next_layer != -1)
	{
		for (auto g : m_layers[next_layer])
		{
			if (g.control != -1)
			{
				std::vector<std::vector<int> > dist = getGateDistTable(g.type);
				if (new_node.locations[g.control] == -1 && new_node.locations[g.target] == -1)
				{
				}
				else if (new_node.locations[g.control] == -1)
				{
					int min = 1000;
					for (int i = 0; i < new_node.qubits.size(); i++)
					{
						if (new_node.qubits[i] == -1 && dist[i][new_node.locations[g.target]] < min)
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
						if (new_node.qubits[i] == -1 && dist[new_node.locations[g.control]][i] < min)
						{
							min = dist[new_node.locations[g.control]][i];
						}
					}
					new_node.cost_heur2 = new_node.cost_heur2 + min;
				}
				else
				{
					new_node.cost_heur2 = new_node.cost_heur2 + dist[new_node.locations[g.control]][new_node.locations[g.target]];
				}
			}
		}
	}
}

void TopologyMatch::expandNode(const vector<int>& qubits, int qubit, vector<edge> &swaps, int nswaps,
	vector<int> &used, node base_node, const vector<gate>& layer_gates, int next_layer)
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
				std::vector<std::vector<int> > dist = getGateDistTable(g.type);

				new_node.cost_heur = new_node.cost_heur + dist[new_node.locations[g.control]][new_node.locations[g.target]];

				if (dist[new_node.locations[g.control]][new_node.locations[g.target]] > m_flip_cost)
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
		expandNode(qubits, qubit + 1, swaps, nswaps, used, base_node, layer_gates, next_layer);

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

TopologyMatch::node TopologyMatch::fixLayerByAStar(int layer, std::vector<int> &map, std::vector<int> &loc)
{
	int next_layer = getNextLayer(layer);

	node n;
	n.cost_fixed = 0;
	n.cost_heur = n.cost_heur2 = 0;

	n.swaps = vector<vector<edge> >();
	n.done = 1;

	vector<gate> layer_gates = vector<gate>(m_layers[layer]);
	vector<int> considered_qubits;

	//Find a mapping for all logical qubits in the CNOTs of the layer that are not yet mapped
	for (auto g : layer_gates)
	{
		if (g.control != -1)
		{
			std::vector<std::vector<int> > dist = getGateDistTable(g.type);

			considered_qubits.push_back(g.control);
			considered_qubits.push_back(g.target);
			if (loc[g.control] == -1 && loc[g.target] == -1)
			{
				set<edge> possible_edges;
				for (set<edge>::iterator it = m_graph.begin(); it != m_graph.end(); it++)
				{
					if (map[it->v1] == -1 && map[it->v2] == -1) {
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
		expandNode(considered_qubits, 0, edges, 0, used, first_node, layer_gates, next_layer);
	}

	node result = m_nodes.top();

	while (!m_nodes.empty())
	{
		m_nodes.pop();
	}
	return result;
}

void TopologyMatch::mappingQProg(QProg prog, QVec &qv, QProg &mapped_prog)
{
	traversalQProgToLayers(&prog);
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
					//Insert a dummy SWAP gate to allow for tracking the m_positions of the logical qubits
					gate gg;
					if (isReversed(m_graph, e))
					{
						gg.control = e.v2;
						gg.target = e.v1;
					}
					else
					{
						gg.control = e.v1;
						gg.target = e.v2;
					}
					gg.type = DUMMY_SWAP_GATE;
					gg.is_dagger = false;
					gg.is_flip = false;

					all_gates.push_back(gg);
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
					//handle the case that the qubit is not yet mapped. This happens if the qubit has not yet occurred in a CNOT gate
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

				if (isReversed(m_graph, e))	//flip the direction of the CNOT by inserting H gates
				{
					g.is_flip = true;
					int tmp = g.target;
					g.target = g.control;
					g.control = tmp;
				}

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
			it->target = locations[target];
			if (locations[target] == -1)
			{
				//This qubit occurs only in single qubit gates -> it can be mapped to an arbirary physical qubit
				int loc = 0;
				while (qubits[loc] != -1)
				{
					loc++;
				}
				locations[target] = loc;

				it->target = target;
			}
		}
	}

#if PRINTF_MAPPING_RESULT
	std::cout << "The mapping results : " << std::endl;
	for (int i = 0; i < locations.size(); i++)
	{
		if (locations[i] != -1)
		{
			std::cout << i << " -> " << locations[i] << std::endl;
		}
	}

	std::cout << "Swap qubits : " << std::endl;
	for (auto g : all_gates)
	{
		if (DUMMY_SWAP_GATE == g.type)
		{
			std::cout << g.target << " <-> " << g.control << std::endl;
		}
	}

#endif

	buildResultingQProg(all_gates, locations, qv, mapped_prog);
}


void TopologyMatch::buildResultingQProg(const std::vector<gate> &resulting_gates, std::vector<int> loc, QVec &qv, QProg &prog)
{
	vector<int> last_layer(m_positions, -1);
	vector<vector<gate> > mapped_qporg;
	std::map<int, int> mapping_result;  // ph => QVec id
	std::vector<std::pair<int, int>>swap_vec;

	for (auto g : resulting_gates)
	{
		if (DUMMY_SWAP_GATE == g.type)
		{
			swap_vec.push_back(std::pair<int, int>(g.target, g.control));
		}
	}

	for (int i = 0; i < loc.size(); i++)
	{
		if (loc[i] != -1)
		{
			int index = mapping_result.size();
			mapping_result.insert(std::pair<int, int>(loc[i], index));
		}
	}

	for (auto swap : swap_vec)
	{
		auto iter_1 = mapping_result.find(swap.first);
		if (iter_1 == mapping_result.end())
		{
			int index = mapping_result.size();
			mapping_result.insert(std::pair<int, int>(swap.first, index));
		}
		auto iter_2 = mapping_result.find(swap.second);
		if (iter_2 == mapping_result.end())
		{
			int index = mapping_result.size();
			mapping_result.insert(std::pair<int, int>(swap.second, index));
		}
	}

	for (auto swap : swap_vec)
	{
		std::swap(mapping_result[swap.first], mapping_result[swap.second]);
	}

	loc.resize(mapping_result.size());
	for (auto map : mapping_result)
	{
		loc[map.second] = map.first;
	}

	QVec q;
	for (int i = 0; i < loc.size(); i++)
	{
		q.push_back(m_qvm->allocateQubitThroughPhyAddress(loc[i]));
	}
	qv = q;

	for (auto g : resulting_gates)
	{
		if (g.control == -1)
		{
			int layer = last_layer[g.target] + 1;

			if (mapped_qporg.size() <= layer)
			{
				mapped_qporg.push_back(vector<gate>());
			}
			mapped_qporg[layer].push_back(g);
			last_layer[g.target] = layer;
		}
		else
		{
			int layer = max(last_layer[g.control], last_layer[g.target]) + 1;
			if (mapped_qporg.size() <= layer)
			{
				mapped_qporg.push_back(vector<gate>());
			}
			mapped_qporg[layer].push_back(g);

			last_layer[g.target] = layer;
			last_layer[g.control] = layer;
		}
	}

	for (auto layer_gates : mapped_qporg)
	{
		for (auto g : layer_gates)
		{
			if (g.control != -1)
			{
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
			case DUMMY_SWAP_GATE:
				m_pTransformSwap->transform(q[g.control], q[g.target], prog);
				break;
			case GateType::PAULI_X_GATE:
			case GateType::PAULI_Y_GATE:
			case GateType::PAULI_Z_GATE:
			case GateType::X_HALF_PI:
			case GateType::Y_HALF_PI:
			case GateType::Z_HALF_PI:
			case GateType::HADAMARD_GATE:
			case GateType::T_GATE:
			case GateType::S_GATE:
			{
				auto iter = m_singleGateFunc.find(g.type);
				if (m_singleGateFunc.end() == iter)
				{
					QCERR("unsupported QGate");
					throw invalid_argument("unsupported QGate");
				}

				QGate single_gate = iter->second(q[g.target]);
				single_gate.setDagger(g.is_dagger);
				prog << single_gate;

			}
			break;

			case GateType::RX_GATE:
			case GateType::RY_GATE:
			case GateType::RZ_GATE:
			case GateType::U1_GATE:
			{
				auto iter = m_singleAngleGateFunc.find(g.type);
				if (m_singleAngleGateFunc.end() == iter)
				{
					QCERR("unsupported QGate");
					throw invalid_argument("unsupported QGate");
				}
				double angle = g.param[0];
				QGate single_angle_gate = iter->second(q[g.target], angle);
				single_angle_gate.setDagger(g.is_dagger);
				prog << single_angle_gate;
			}
			break;

			case GateType::CNOT_GATE:
			{
				QGate cnot_gate = CNOT(q[g.control], q[g.target]);
				cnot_gate.setDagger(g.is_dagger);
				if (g.is_flip)
				{
					prog << H(q[g.target])
						<< H(q[g.control])
						<< cnot_gate
						<< H(q[g.control])
						<< H(q[g.target]);
				}
				else
				{
					prog << cnot_gate;
				}
			}
			break;

			case GateType::SWAP_GATE:
				m_pTransformSwap->transform(q[g.control], q[g.target], prog);
				break;
			case GateType::CZ_GATE:
			case GateType::ISWAP_GATE:
			case GateType::SQISWAP_GATE:
			{
				auto iter = m_doubleGateFunc.find(g.type);
				if (m_doubleGateFunc.end() == iter)
				{
					QCERR("unsupported QGate");
					throw invalid_argument("unsupported QGate");
				}
				QGate double_gate = iter->second(q[g.control], q[g.target]);
				double_gate.setDagger(g.is_dagger);
				prog << double_gate;
			}
			break;
			case GateType::CPHASE_GATE:
			{
				auto iter = m_doubleAngleGateFunc.find(g.type);
				if (m_doubleAngleGateFunc.end() == iter)
				{
					QCERR("unsupported QGate");
					throw invalid_argument("unsupported QGate");
				}
				double angle = g.param[0];
				QGate cr_gate = iter->second(q[g.control], q[g.target], angle);
				cr_gate.setDagger(g.is_dagger);

				if (g.is_flip)
				{
					m_pTransformSwap->transform(q[g.control], q[g.target], prog);
					prog << cr_gate;
					m_pTransformSwap->transform(q[g.control], q[g.target], prog);
				}
				else
				{
					prog << cr_gate;
				}
			}
			break;
			case  GateType::CU_GATE:
			{
				QGate cu_gate = CU(g.param[0], g.param[1], g.param[2], g.param[3], q[g.control], q[g.target]);
				cu_gate.setDagger(g.is_dagger);
				if (g.is_flip)
				{
					m_pTransformSwap->transform(q[g.control], q[g.target], prog);
					prog << cu_gate;
					m_pTransformSwap->transform(q[g.control], q[g.target], prog);
				}
				else
				{
					prog << cu_gate;
				}
			}
			break;
			case  GateType::U2_GATE:
			{
				QGate u2_gate = U2(q[g.target], g.param[0], g.param[1]);
				u2_gate.setDagger(g.is_dagger);
				prog << u2_gate;
			}
			break;
			case  GateType::U3_GATE:
			{
				QGate u3_gate = U3(q[g.target], g.param[0], g.param[1], g.param[2]);
				u3_gate.setDagger(g.is_dagger);
				prog << u3_gate;
			}
			break;
			case  GateType::U4_GATE:
			{
				QGate u4_gate = U4(g.param[0], g.param[1], g.param[2], g.param[3], q[g.target]);
				u4_gate.setDagger(g.is_dagger);
				prog << u4_gate;
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
}

void TopologyMatch::traversalQProgToLayers(QProg *prog)
{
	if (nullptr == prog)
	{
		QCERR("p_prog is null");
		throw runtime_error("p_prog is null");
		return;
	}

	m_last_layer.resize(m_nqubits, -1);

	bool isDagger = false;
	execute(prog->getImplementationPtr(), nullptr, isDagger);
}

void TopologyMatch::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QVec qgate_ctrl_qubits;
	cur_node->getControlVector(qgate_ctrl_qubits);
	if (!qgate_ctrl_qubits.empty())
	{
		QCERR("control qubits in qgate are not supported!");
		throw invalid_argument("control qubits in qgate are not supported!");
	}

	gate g;
	QVec qv;
	int layer;
	cur_node->getQuBitVector(qv);
	auto type = cur_node->getQGate()->getGateType();

	g.type = type;
	g.is_dagger = cur_node->isDagger() ^ is_dagger;
	g.is_flip = false;
	switch (type)
	{
	case GateType::PAULI_X_GATE:
	case GateType::PAULI_Y_GATE:
	case GateType::PAULI_Z_GATE:
	case GateType::X_HALF_PI:
	case GateType::Y_HALF_PI:
	case GateType::Z_HALF_PI:
	case GateType::HADAMARD_GATE:
	case GateType::T_GATE:
	case GateType::S_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();

		layer = m_last_layer[g.target] + 1;
		m_last_layer[g.target] = layer;
	}
	break;

	case GateType::RX_GATE:
	case GateType::RY_GATE:
	case GateType::RZ_GATE:
	case GateType::U1_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();

		auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter*>(cur_node->getQGate());
		double angle = gate_parameter->getParameter();
		g.param.push_back(angle);

		layer = m_last_layer[g.target] + 1;
		m_last_layer[g.target] = layer;
	}
	break;

	case GateType::CNOT_GATE:
	case GateType::CZ_GATE:
	case GateType::ISWAP_GATE:
	case GateType::SWAP_GATE:
	{
		g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();

		layer = max(m_last_layer[g.target], m_last_layer[g.control]) + 1;
		m_last_layer[g.target] = m_last_layer[g.control] = layer;
	}
	break;
	case GateType::CPHASE_GATE:
	{
		g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();

		auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter*>(cur_node->getQGate());
		double angle = gate_parameter->getParameter();
		g.param.push_back(angle);

		layer = max(m_last_layer[g.target], m_last_layer[g.control]) + 1;
		m_last_layer[g.target] = m_last_layer[g.control] = layer;
	}
	break;
	case GateType::U2_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		QGATE_SPACE::U2 *u2_gate = dynamic_cast<QGATE_SPACE::U2*>(cur_node->getQGate());
		double phi = u2_gate->get_phi();
		double lam = u2_gate->get_lambda();

		g.param.push_back(phi);
		g.param.push_back(lam);

		layer = m_last_layer[g.target] + 1;
		m_last_layer[g.target] = layer;
	}
	break;
	case GateType::U3_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>(cur_node->getQGate());
		double theta = u3_gate->get_theta();
		double phi = u3_gate->get_phi();
		double lam = u3_gate->get_lambda();
		g.param.push_back(theta);
		g.param.push_back(phi);
		g.param.push_back(lam);

		layer = m_last_layer[g.target] + 1;
		m_last_layer[g.target] = layer;
	}
	break;
	case GateType::U4_GATE:
	{
		g.control = -1;
		g.target = qv[0]->getPhysicalQubitPtr()->getQubitAddr();

		auto angle = dynamic_cast<AbstractAngleParameter *>(cur_node->getQGate());

		double alpha = angle->getAlpha();
		double beta = angle->getBeta();
		double gamma = angle->getGamma();
		double delta = angle->getDelta();

		g.param.push_back(alpha);
		g.param.push_back(beta);
		g.param.push_back(gamma);
		g.param.push_back(delta);

		layer = m_last_layer[g.target] + 1;
		m_last_layer[g.target] = layer;
	}
	break;
	case GateType::CU_GATE:
	{
		g.control = qv[0]->getPhysicalQubitPtr()->getQubitAddr();
		g.target = qv[1]->getPhysicalQubitPtr()->getQubitAddr();

		auto angle = dynamic_cast<AbstractAngleParameter *>(cur_node->getQGate());

		double alpha = angle->getAlpha();
		double beta = angle->getBeta();
		double gamma = angle->getGamma();
		double delta = angle->getDelta();

		g.param.push_back(alpha);
		g.param.push_back(beta);
		g.param.push_back(gamma);
		g.param.push_back(delta);

		layer = max(m_last_layer[g.target], m_last_layer[g.control]) + 1;
		m_last_layer[g.target] = m_last_layer[g.control] = layer;
	}
	break;
	default:
		break;
	}

	if (m_layers.size() <= layer)
	{
		m_layers.push_back(vector<gate>());
	}
	m_layers[layer].push_back(g);

}

void TopologyMatch::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	Traversal::traversal(cur_node, *this, is_dagger);
}

void TopologyMatch::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QVec circuit_ctrl_qubits;
	cur_node->getControlVector(circuit_ctrl_qubits);
	if (!circuit_ctrl_qubits.empty())
	{
		QCERR("control qubits in circuit are not supported!");
		throw invalid_argument("control qubits in circuit are not supported!");
	}

	bool bDagger = cur_node->isDagger() ^ is_dagger;
	Traversal::traversal(cur_node, true, *this, bDagger);
}

void TopologyMatch::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be quantum measure node here.");
	throw invalid_argument("transform error, there shouldn't be quantum measure node here.");
}

void TopologyMatch::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be quantum reset node here.");
	throw invalid_argument("transform error, there shouldn't be quantum reset node here.");
}

void TopologyMatch::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be control flow node here.");
	throw invalid_argument("transform error, there shouldn't be control flow node here.");
}

void TopologyMatch::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	QCERR("transform error, there shouldn't be classicalProg here.");
	throw invalid_argument("transform error, there shouldn't be classicalProg here.");
}


QProg QPanda::topology_match(QProg prog, QVec &qv, QuantumMachine * machine, SwapQubitsMethod method, ArchType arch_type)
{
	if (nullptr == machine)
	{
		QCERR("Quantum machine is nullptr");
		throw std::invalid_argument("Quantum machine is nullptr");
	}

	QProg outprog;
	TopologyMatch match = TopologyMatch(machine, method, arch_type);
	match.mappingQProg(prog, qv, outprog);
	return outprog;
}