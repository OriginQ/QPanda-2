#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/GetQubitTopology.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include <numeric>
#include <unordered_map>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <random>
#include <variant>
#include "Core/Utilities/Tools/RandomEngine/RandomEngine.h"
#include "QMapping/SabreQMapping.h"
#include "QMapping/OBMTQMapping.h"

#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceMat(mat)
#endif

using namespace std;
using namespace QPanda;

using WeightedSwap = std::pair<double, Swap>;

static ArchGraph::sRef build_sub_graph(const Mapping& init_mapping, ArchGraph::sRef src_graph);

static uint64_t get_current_time()
{
	const std::chrono::system_clock::duration duration_since_epoch
		= std::chrono::system_clock::now().time_since_epoch(); /* 从1970-01-01 00:00:00到当前时间点的时长 */
	return std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_epoch).count();
}

/*******************************************************************
*                      class MappingFinder
* @brief Get the mapping.
********************************************************************/
class MappingFinder{
public:
	typedef MappingFinder* Ref;
	typedef std::unique_ptr<MappingFinder> uRef;
	typedef std::pair<Mapping, uint32_t> MappingAndNSwaps;
	ShortestDistanceByBFS m_shortest_distance;
	size_t m_hops = 1;

	void set_hops(size_t hops)
	{
		if (hops > 0) {
			m_hops = hops;
		}
	}

	Mapping find(ArchGraph::Ref graph) {
		uint32_t qbits = graph->size();
		Mapping mapping(qbits);

		for (uint32_t i = 0; i < qbits; ++i) {
			mapping[i] = i;
		}

		uint32_t _tmp = 0;
		for (; qbits > 1;)
		{
			const uint32_t _r = ceil(random_generator19937(0, qbits - 1));
			_tmp = mapping[qbits - 1];
			mapping[qbits - 1] = mapping[_r];
			mapping[_r] = _tmp;
			--qbits;
		}	

		return mapping;
	}

	/*
	 * @brief Gets the connected subgraph of the specified number of nodes in the graph
	 * @param[in, out] sub_archgraph : Holds the sum of the weights of the connected subgraph and the set of connected subgraphs
	 * @param[in, out] temp : Holds the set of connected subgraph
	 * @param[in, out] visited : Holds the flag for whether a node in the graph is accessed
	 * @param[in] g : archgraph
	 * @param[in] fidelity : The weight(fidelity) matrix of the graph
	 * @param[in] sum : The weight sum of the current connected subgraph
	 * @param[in] u : The node of archgraph
	 * @param[in] quantum_num : The number of nodes of the current connected subgraph
	 * @param[in] target_num : The number of nodes of the target connected subgraph
	 */
	void search_dfs(std::map<double, std::vector<std::vector<uint32_t>>, greater<double>>& sub_archgraph, std::vector<uint32_t>& temp,
		std::vector<bool>& visited, ArchGraph::sRef g, std::vector<std::vector<double>>& fidelity, double sum, uint32_t u,
        int& quantum_num, int target_num)
	{
		temp.emplace_back(u);
		quantum_num++;
		visited[u] = true;
		if (quantum_num == target_num)
		{
            /* 基于子图进行最短路径搜索 */
            ShortestDistanceByBFS sub_graph_shortest_distance;
            ArchGraph::sRef tmp_graph = build_sub_graph(temp, g);
            sub_graph_shortest_distance.init(tmp_graph.get());

			/* Combine quantum circuit fidelity and quantum dispersion as weights */
			//const double res = /*sum * 0.2 +*/ m_shortest_distance.get_overall_dispersion(temp) * 10/*0.8*/;
            const double res = sub_graph_shortest_distance.get_overall_dispersion(temp);
            //std::cout << "get sub graph fidelity-res: " << res << std::endl;
			sub_archgraph[res].emplace_back(temp);
			return;
		}

		for (uint32_t v : g->adj(u))
		{
			if (!visited[v]) 
			{
				search_dfs(sub_archgraph, temp, visited, g, fidelity, sum + fidelity[u][v] * 10, v, quantum_num, target_num);
				/*temp.pop_back();
				quantum_num--;
				visited[v] = false;*/
			}
		}
	}

    /*
     * @brief 广度优先算法: Gets the connected subgraph of the specified number of nodes in the graph
     * @param[in, out] matched_subgraph : Holds the sum of the weights of the connected subgraph and the set of connected subgraphs
     * @param[in, out] temp : Holds the set of connected subgraph
     * @param[in, out] visited : Holds the flag for whether a node in the graph is accessed
     * @param[in] g : archgraph
     * @param[in] fidelity : The weight(fidelity) matrix of the graph
     * @param[in] sum : The weight sum of the current connected subgraph
     * @param[in] u : The node of archgraph
     * @param[in] quantum_num : The number of nodes of the current connected subgraph
     * @param[in] target_num : The number of nodes of the target connected subgraph
     */
    void search_bfs(std::map<double, std::vector<std::vector<uint32_t>>, greater<double>>& matched_subgraph,
		std::vector<uint32_t>& temp, std::vector<bool>& visited, const ArchGraph::sRef g,
		std::vector<std::vector<double>>& fidelity,
		double weight, uint32_t u, int& quantum_num, int target_num)
    {
        auto handle_new_node_func = [&](const uint32_t new_node)->bool {
            temp.emplace_back(new_node);
            ++quantum_num;
            visited[new_node] = true;
            if (quantum_num == target_num)
            {
                /* 基于子图进行最短路径搜索 */
                ShortestDistanceByBFS sub_graph_shortest_distance;
                ArchGraph::sRef tmp_graph = build_sub_graph(temp, g);
                //ArchGraph::sRef tmp_graph = mArchGraph;
                //std::cout << "Got temp_graph: \n" << tmp_graph->dotify() << "\n------ graph_text end ------" << std::endl;
                sub_graph_shortest_distance.init(tmp_graph.get());

                /* Combine quantum circuit fidelity and quantum dispersion as weights */
                //const double res = /*sum * 0.2 +*/ m_shortest_distance.get_overall_dispersion(temp) * 10/*0.8*/;
                const double res = sub_graph_shortest_distance.get_overall_dispersion(temp);
                //std::cout << "Got sub graph_" << matched_subgraph.size() << " fidelity-res: " << res << std::endl;
                matched_subgraph[res].emplace_back(temp);
                return true; /* 找到合适子图，返回true */
            }

            return false; /* 未找到合适子图 */
        };

        bool b_find_sub_graph = false;
        std::vector<uint32_t> new_node_vec;
        for (uint32_t v : g->adj(u))
        {
            if (!visited[v])
            {
                /* 将每个相邻的点加入子图 */
                new_node_vec.emplace_back(v);

                /* 如果找到子图，就不再搜索，其他的子图组合会在对应的节点的子图搜索过程中，被发现 */
                if (handle_new_node_func(v))
                {
                    b_find_sub_graph = true;

					for (const uint32_t& new_v : new_node_vec)
                    {
                        visited[temp.back()] = false;
                        temp.pop_back();
                        --quantum_num;
                    }

                    new_node_vec.clear();
                }
            }
        }

        /* 如果本次搜索已经找到合适子图，就不再递归搜索 */
        if (b_find_sub_graph) 
        {
            return;
        }
        else
        {
            for (uint32_t v : new_node_vec)
            {
                search_bfs(matched_subgraph, temp, visited, g, fidelity, weight + fidelity[u][v] * 10, v, quantum_num, target_num);
            }
        }
    }

	/*
	 * @brief Recursively generates permutation combinations 
	 * @param[in, out] mappings : Permutation of the combined result set
	 * @param[in, out] sub_graph : The elements that are arranged and combined
	 */
	void arrange(std::set<std::vector<uint32_t>>& mappings, std::vector<uint32_t>& sub_graph, int first, int len)
	{
		/* All numbers are filled in */
		if (first == len) {
			mappings.insert(sub_graph);
			return;
		}
		for (int i = first; i < len; ++i) {
			/* Dynamically maintain arrays */ 
			swap(sub_graph[i], sub_graph[first]);
			/* Continue to recursively fill in the next number */ 
			arrange(mappings, sub_graph, first + 1, len);
			/* Undo the operation */ 
			swap(sub_graph[i], sub_graph[first]);
		}
	}
	
	/**
	 * @brief get matched mapping, descend order by fidelity!
	 * @parma[in] ArchGraph::sRef connected graph
	 * @parma[in] QProg prog Quantum programs
	 * @param[out] std::map<double, std::vector<Mapping>> all matched mappings in descending order by fidelity
	 */
    std::map<double, std::vector<std::vector<uint32_t>>, std::greater<double>>
        get_matched_qubits_block(const ArchGraph::sRef& graph, const QProg& prog, const QPanda::ShortestDistanceByBFS& shortest_distance)
    {
		m_shortest_distance = shortest_distance;

		std::map<double, std::vector<std::vector<uint32_t>>, greater<double>> connected_graphs;	/**< Connected subgraph collections */
		std::vector<uint32_t> temp;
		std::vector<bool> visited(graph->size(), false);
		QVec vec;
		const size_t qubits_size = get_all_used_qubits(prog, vec);

		double weight = 0;
		int quantum_num = 0;
		std::vector<std::vector<double>> fidelity = graph->get_adj_weight_matrix();

		for (uint32_t i = 0; i < graph->get_vertex_count(); ++i) {
			temp.clear();
			temp.push_back(i);
			quantum_num = 1;
			visited[i] = true;
			search_bfs(connected_graphs, temp, visited, graph, fidelity, weight, i, quantum_num, qubits_size);

			std::fill(visited.begin(), visited.end(), false);
			if (connected_graphs.size() > 0) {
				i += m_hops;
			}
		}

		/* deduplication in one-dimensiona */
		for (auto& [fidelity, qubits_blocks] : connected_graphs) {
			std::sort(qubits_blocks.begin(), qubits_blocks.end());
			qubits_blocks.erase(std::unique(qubits_blocks.begin(), qubits_blocks.end()), qubits_blocks.end());
		}
		return connected_graphs;
	}

	/**
	 * @brief get initial mapping
	 * @parma[in] ArchGraph::sRef g  A pointer to a power graph
	 * @parma[in] QProg prog Quantum programs
	 * @parma[in] QPanda::ShortestDistanceByBFS& m_shortest_distance The shortest path related object of the right graph, providing the shortest path interface and the least weight path interface
	 * @param[out] std::vector<Mapping> best_mappings
	 */
    std::vector<Mapping> find(ArchGraph::sRef& g, QProg qmod, const QPanda::ShortestDistanceByBFS& shortest_distance, uint32_t mRandomMappings)
    {
		auto sub_archgraph = get_matched_qubits_block(g, qmod, shortest_distance);
        if (sub_archgraph.size() == 0){
            return std::vector<Mapping>();
        }

		return find(g, qmod, mRandomMappings, sub_archgraph.begin()->second.front());
	}

	std::vector<Mapping> find(ArchGraph::sRef& g, QProg qmod,  uint32_t mRandomMappings,
		const std::vector<uint32_t>& sub_archgraph)
	{
		/* Select subgraph mappings with better overall fidelity and dispersion
		 */
		std::vector<std::set<uint32_t>> sub_fidelity_archgraph; /**< The largest set of continuous subgraphs for fidelity and dispersion synthesis */
		/*size_t i = 0;
		for (auto iter = sub_archgraph.begin(); (iter != sub_archgraph.end()) && (i < 3); ++i, ++iter)
		{
			for (auto m : iter->second)
			{
				sub_fidelity_archgraph.push_back({m.begin(), m.end()});
			}
		}*/
        sub_fidelity_archgraph.push_back({ sub_archgraph.begin(), sub_archgraph.end() });

        /* 删除重复元素 */
        /* 240712 zhaody 只有一个元素，不需要去重 */
        /*auto last = std::unique(sub_fidelity_archgraph.begin(), sub_fidelity_archgraph.end(),
                                [](const std::set<uint32_t>& a, const std::set<uint32_t>& b) {
            return a == b;
        });
        sub_fidelity_archgraph.erase(last, sub_fidelity_archgraph.end());*/
		
		std::set<std::vector<uint32_t>> mappings;/**< Mapping set */
		for (auto sub_graph : sub_fidelity_archgraph)
		{
			/* Pick a partial mapping randomly
			 */
			uint32_t change_num = mRandomMappings / sub_fidelity_archgraph.size();
			std::vector<uint32_t> temp{sub_graph.begin(), sub_graph.end()};
			do {
				mappings.insert(temp);
            }
            while ((change_num-- > 0)
				&& (mappings.size() < mRandomMappings)
				&& std::next_permutation(temp.begin(), temp.end()));

			if (mappings.size() >= mRandomMappings) {
				break;
			}
		}

		/* Remove the BARRIE doors that are added by default */
		DynamicQCircuitGraph cir_graph(qmod);
		auto& front_layer0 = cir_graph.get_front_layer();		/**< Get the execution gate dependency */
		for (uint32_t _i = 0; _i < front_layer0.size(); _i++)
		{
			auto& front_layer_nodes = front_layer0.get_front_layer_nodes();
			if ((front_layer_nodes.size() == 1) && (BARRIER_GATE == front_layer_nodes.at(0)->m_cur_node->m_gate_type))
			{
				_i = front_layer0.remove_node(_i);
			}
		}

		/* Filter mapping by execution gate dependencies */
		std::vector<Mapping> temp_mappings;
		bool flag = false;					/**< Gets the dependency success*/
		do
		{
			flag = false;
			auto& front_layer = cir_graph.get_front_layer();
			for (uint32_t _i = 0; _i < front_layer.size(); )
			{
				flag = true;
				temp_mappings.clear();
				/* Gets the qubit of the execution gate */
				const auto& cur_gate = *((front_layer[_i])->m_cur_node->m_iter);
				auto deps = build_deps(cur_gate);
				/*
				* Determine whether the deps is a BARRIE GATE, and if it is , delete.
				* The compilation part will split the line and add barrie gates to some complex gates
				*/
				if (deps.size() == 0){
					_i = front_layer.remove_node(_i);
					continue;
				}
				for (auto it = mappings.begin(); it != mappings.end(); )
				{
					if (!g->hasEdge((*it)[deps[0].mFrom], (*it)[deps[0].mTo]))
					{
						/* Holds mappings that do not satisfy the current execution gate dependency */
						temp_mappings.push_back(*it);
						it = mappings.erase(it);
					}
					else
						++it;
				}
	
				/* Delete the execution gate */
				_i = front_layer.remove_node(_i);
	
				if (mappings.empty())
				{
					flag = false;
					break;
				}
			}
        }
        while (flag);
	
		/* If no mappings satisfy the dependency, the mappings that do not satisfy the dependency are retained */
		if (mappings.empty())
		{
			for (auto m : temp_mappings)
			{
				mappings.insert(m);
			}
		}

		/* 
		 * Map the remaining unmatched qubits to facilitate the later reverse reverse mapping process  
		 */
		std::vector<bool> mapped(g->size(), false);		/**< Qubit mapping flags */
		std::vector<Mapping> best_mappings;				/**< Optimal set of mappings */
		for (const auto& b_mapping : mappings){
			best_mappings.emplace_back(b_mapping);
		}
	
		return best_mappings;
	}

	static uRef Create() {
		return uRef(new MappingFinder());
	}
};

SabreQAllocator::SabreQAllocator(ArchGraph::sRef ag, uint32_t mLookAhead, uint32_t mIterations, uint32_t mRandomMappings)
    : AbstractQubitMapping(ag), m_look_ahead(mLookAhead), m_max_iterations(mIterations), m_max_random_mappings(mRandomMappings), m_swap_cnt(0){}

std::optional<SabreQAllocator::MappingAndNSwaps>
SabreQAllocator::allocateWithInitialMapping(const Mapping& initialMapping, DynamicQCircuitGraph cir_graph, 
	QPanda::QuantumMachine *qvm, ArchGraph::sRef arch_graph, bool issueInstructions)
{
	auto mapping = initialMapping;
    std::map<QNodeRef, uint32_t> reached;
    std::set<QNodeRef> pastLookAhead;
    uint32_t swapNum = 0;
    uint32_t step_index = 0;
    uint32_t cut_step_swap_size = 0;
	//std::set<std::pair<uint32_t, uint32_t>> unConnectedNode;	/* 记录当前层不可执行门的节点对 */
    while (true)
	{
		/*
		 * Handle logic gates that can be executed directly: execute_gate
		 */
        bool changed = false;		/**< Whether there is an executable gate to join the execute_gate_list */
        do 
		{
            changed = false;
			//unConnectedNode.clear();
            std::vector<pPressedCirNode> issueNodes;				/**< execute_gate_list */
			auto& front_layer = cir_graph.get_front_layer();	/**< Front-layer qubit gate set (i.e., both previous gates of both qubits are executed) */
			for (uint32_t _i = 0; _i < front_layer.size();)
			{
				const auto& cur_gate = *((front_layer[_i])->m_cur_node->m_iter);
				/* Get the logic gate dependency */
				auto deps = build_deps(cur_gate);
				if (!deps.empty())
				{
					auto dep = deps[0];
					uint32_t u = mapping[dep.mFrom], v = mapping[dep.mTo];

					/*
					 * Determine whether the physical qubits of two logical qubit mappings are directly connected to each other
					 * If so, it means that it can be executed directly to add the current logic gate to the execute_gate_list
					 * If not, it means that it cannot be directly executed, reserved, and subsequently inserted into the swap gate
					 */
					/* 若量子比特不相连，则加入 */
					if (!arch_graph->hasEdge(u, v) && !arch_graph->hasEdge(v, u)) {
						++_i;
						/*unConnectedNode.insert(std::make_pair(u, v));
						unConnectedNode.insert(std::make_pair(v, u));*/
						continue;
					}
				}
				
				issueNodes.emplace_back(front_layer[_i]);
				_i = front_layer.remove_node(_i);
				changed = true;
                ++step_index;
                cut_step_swap_size = 0;
			}

			/*
			 * Remapping
			 */
			if (issueInstructions)
			{
				for (auto cNode : issueNodes)
				{
					for (const auto &gate_info : cNode->m_relation_pre_nodes)
					{
						remap_node_to_new_prog(*gate_info->m_iter, mapping, qvm);
					}

					remap_node_to_new_prog(*(cNode->m_cur_node->m_iter), mapping, qvm);

					for (const auto &gate_info : cNode->m_relation_successor_nodes)
					{
						remap_node_to_new_prog(*gate_info->m_iter, mapping, qvm);
					}
				}
			}
        } while (changed);

		/* Get the logic gate that cannot be executed directly from the front layer */ 
		const auto& cur_layer = cir_graph.get_front_layer_c();

        // If there is no node in the current layer, it means that
        // we have reached the end of the algorithm. i.e. we processed
        // all nodes already.
        if (cur_layer.size() == 0)
			break;

        std::unordered_map<QNodeRef, Dep> currentLayer;
        std::vector<Dep> nextLayer;

		for (uint32_t _i = 0; _i < cur_layer.size(); ++_i)
		//for (const auto& _node : cur_layer.m_front_layer_nodes)
		{
			const auto& _node = cur_layer[_i];
			auto _gate = *(_node->m_cur_node->m_iter);
			auto dep = build_deps(_gate)[0];
			pastLookAhead.insert(_gate);
			currentLayer[_gate] = dep;
		}

		/*
		 * Gets the post logic gate of the currently processed gate
		 */
		const auto& topo_seq = cir_graph.get_layer_topo_seq();
		for (const auto& layer : topo_seq)	/* topo_seq: vector<vector<pair<T, vector<T>>>> */
		{
			for (const auto& _n : layer)	/* layer: vector<pair<T, vector<T>>> */
			{
				const auto& _n_ref = *(_n.first->m_cur_node->m_iter);
				if (pastLookAhead.find(_n_ref) == pastLookAhead.end()){
					auto deps = build_deps(_n_ref);
					if (!deps.empty()) nextLayer.push_back(deps[0]);
				}
			}

			if (nextLayer.size() > m_look_ahead)
			{
				break;
			}
		}

		/* Generates an assignment mapping (maps the architecture's qubits to the logical ones) of size archQ. */
        auto invM = InvertMapping(mPQubits, mapping);
        auto best = WeightedSwap(UNDEF_UINT32, Swap { 0, 0 });

		/* 计算交换两个量子比特后，当前层和下一层的保真度权重大小，并依此选择swap方案 */
		auto weight_calc_func = [&](uint32_t u, uint32_t v) {
			auto& cpy = mapping;
			std::swap(cpy[invM[u]], cpy[invM[v]]); /* 临时交换，用于计算cost */

			double currentLCost = 0;
			double nextLCost = 0;

			for (auto& [node_ref, dep] : currentLayer) {
				/* Consider the cost from qubit distance */
				currentLCost += m_shortest_distance.get(cpy[dep.mFrom], cpy[dep.mTo]);

				/* Consider cost in terms of qubit fidelity */
				//currentLCost += (1 - m_shortest_distance.get_fidelity(cpy[dep.mFrom], cpy[dep.mTo])) * 10 ;
			}

			for (const auto& dep : nextLayer) {
				/* Consider the cost from qubit distance */
				nextLCost += m_shortest_distance.get(cpy[dep.mFrom], cpy[dep.mTo]);

				/* Consider cost in terms of qubit fidelity */
				//nextLCost += (1 - m_shortest_distance.get_fidelity(cpy[dep.mFrom], cpy[dep.mTo])) * 10;
			}

			//currentLCost = currentLCost / currentLayer.size();
			//if (!nextLayer.empty()) nextLCost = nextLCost / nextLayer.size();
			double cost = currentLCost + 0.2 * nextLCost;

			if (cost < best.first) {
				best.first = cost;
				best.second = Swap{u, v};
			}

			std::swap(cpy[invM[u]], cpy[invM[v]]); /* 取消临时交换，还原原有Mapping */
		};

		for (auto& [node_ref, dep] : currentLayer)
		{
			uint32_t from = mapping[dep.mFrom], to = mapping[dep.mTo];
			for (auto v : arch_graph->adj(from)){
                weight_calc_func(from, v);
			}

			for (auto v : arch_graph->adj(to)){
                weight_calc_func(v, to);
			}
		}

        auto swap = best.second;
        std::swap(mapping[invM[swap.u]], mapping[invM[swap.v]]);

        if (issueInstructions) {
			m_mapped_prog << _swap(qvm->allocateQubitThroughPhyAddress(swap.u), qvm->allocateQubitThroughPhyAddress(swap.v));
        }

        ++swapNum;

        if (++cut_step_swap_size > 64) { /* max swap count: 64 */
            //QCERR_AND_THROW(run_fail, 
            //    "Error: Exceeded the maximum number of swap gates for one step on sabre-mapping. current step:"
			//	<< step_index << ", swap-cnt:" << cut_step_swap_size);
			return std::nullopt;
        }
		//std::cout << "step_index = " << step_index << ", cut_step_swap_size = " << cut_step_swap_size << std::endl;
    }

    return MappingAndNSwaps(mapping, swapNum);
}

void SabreQAllocator::remap_node_to_new_prog(QNodeRef node, const Mapping& mapping, QPanda::QuantumMachine *qvm)
{
    QVec qv;
    QNodeDeepCopy reproduction;
    switch (node->getNodeType())
    {
    case GATE_NODE:
    {
        auto new_node = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQGateNode>(node));
        if (new_node.getQuBitVector(qv) > 1)
        {
            const auto qbit_0 = qvm->allocateQubitThroughPhyAddress(mapping[qv[0]->get_phy_addr()]);
            const auto qbit_1 = qvm->allocateQubitThroughPhyAddress(mapping[qv[1]->get_phy_addr()]);
            new_node.remap({ qbit_0, qbit_1 });
        }
        else
        {
            const auto qbit = qvm->allocateQubitThroughPhyAddress(mapping[qv[0]->get_phy_addr()]);

            QVec c_qv;
            new_node.getControlVector(c_qv);
            new_node.clear_qubits();
            new_node.remap({ qbit });

            for (auto &_q : c_qv)
            {
                _q = qvm->allocateQubitThroughPhyAddress(mapping[_q->get_phy_addr()]);
            }
            new_node.setControl(c_qv);
        }
        m_mapped_prog << new_node;
	}
	break;

	case MEASURE_GATE:
	{
		auto new_node = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQuantumMeasure>(node));
		auto cbit = new_node.getCBit();
		auto qidx = mapping[new_node.getQuBit()->get_phy_addr()];
		auto qbit = qvm->allocateQubitThroughPhyAddress(qidx);
		m_mapped_prog << Measure(qbit, cbit);
	}
	break;

	case  RESET_NODE:
	{
		auto new_node = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQuantumReset>(node));
		auto qidx = mapping[new_node.getQuBit()->get_phy_addr()];
		auto qbit = qvm->allocateQubitThroughPhyAddress(qidx);
		m_mapped_prog << Reset(qbit);
	}
	break;

	default:
		QCERR_AND_THROW(run_fail, "Error: circuit node type error.");
		break;
	}
}

/* @brief 根据初始映射构建临时子图 
 * @param[in] const Mapping& 初始映射
 * @param[in] ArchGraph::sRef 原始图
 * @return 临时子图
 */
static ArchGraph::sRef build_sub_graph(const Mapping& init_mapping, const ArchGraph::sRef src_graph)
{
    const std::vector<uint32_t>& phy_partition = init_mapping; /* 物理连通块 */
    std::map<uint32_t, uint32_t> p2v_map; /* 物理bit-》逻辑bit映射关系 */
    std::map<uint32_t, uint32_t> v2p_map; /* 逻辑bit-》物理bit映射关系 */
    uint32_t v_idx = 0;
    for (const auto& p_val : phy_partition)
    {
        p2v_map[p_val] = v_idx;
        v2p_map[v_idx] = p_val;
        ++v_idx;
    }
	uint32_t max_v = p2v_map.rbegin()->first;

    /* 根据连通比特块构建图 */
    //const auto phy_partition_qubits_size = phy_partition.size();
    const auto phy_partition_qubits_size = max_v + 1;
    std::shared_ptr<QPanda::ArchGraph> arch_graph = QPanda::ArchGraph::Create(phy_partition_qubits_size);
    arch_graph->putReg(std::to_string(phy_partition_qubits_size), std::to_string(phy_partition_qubits_size));
    for (auto iter_1 = phy_partition.begin(); iter_1 != phy_partition.end(); ++iter_1)
    {
        for (auto iter_2 = phy_partition.begin(); iter_2 != phy_partition.end(); ++iter_2)
        {
            if ((*iter_1) == (*iter_2)) {
                continue;
            }

            //if ((*iter_2) > matrix_connect.size())
            if ((*iter_2) > src_graph->size())
            {
                //PILOT_OS_ERROR(ErrorCode::UNDEFINED_ERROR, "");
                QCERR_AND_THROW(run_fail,
                    "Error, partition size error, qubit_" << (*iter_2)
                    << ", src_graph size:" << src_graph->size());
            }

            //if (matrix_connect[*iter_1][*iter_2] > 0)
            if (src_graph->hasEdge((*iter_1), (*iter_2)) && (src_graph->getW((*iter_1), (*iter_2)) > 0)) {
                //arch_graph->putEdge(v2p_map[*iter_1], v2p_map[*iter_2], src_graph->getW((*iter_1), (*iter_2)));
                arch_graph->putEdge((*iter_1), (*iter_2), src_graph->getW((*iter_1), (*iter_2)));
            }
        }
    }

    return arch_graph;
}

/* @brief Get best mapping qubits blocks by fidelity
 * @param[in] ArchGraph::sRef& graph
 * @param[in] QProg& Quantum programs
 * @return The best mapping qubits blocks by fidelity
 */
std::map<double, std::vector<Mapping>, std::greater<double>>
SabreQAllocator::select_best_qubits_blocks(ArchGraph::sRef& arch_graph, QProg& qmod)
{
	m_shortest_distance.init(arch_graph.get());
	MappingFinder mappingFinder;
	mappingFinder.set_hops(m_hops);
	//auto temp_result = mappingFinder.get_matched_qubits_block(arch_graph, qmod, m_shortest_distance);
	//return (!temp_result.empty()) ? temp_result.begin()->second : std::vector<Mapping>{};
	return mappingFinder.get_matched_qubits_block(arch_graph, qmod, m_shortest_distance);
}

Mapping SabreQAllocator::allocate(QProg qmod, QuantumMachine *qvm)
{
	DynamicQCircuitGraph cir_graph(qmod);

	QProg prog_reverse;
	for (auto gate_itr = qmod.getFirstNodeIter(); gate_itr != qmod.getEndNodeIter(); ++gate_itr) 
	{
		prog_reverse.insertQNode(prog_reverse.getHeadNodeIter(), *gate_itr);
	}
	DynamicQCircuitGraph cir_graph_reverse(prog_reverse);

    Mapping init_mapping;
	MappingFinder mappingFinder;
	mappingFinder.set_hops(m_hops);
    MappingAndNSwaps best(init_mapping, (std::numeric_limits<uint32_t>::max)());
    ArchGraph::sRef best_sub_graph;

	/* Optimized initial mapping and bidirectional iterative mapping
	 */
	std::vector<Mapping> initial_mappings;
	if (m_specified_blocks.size() > 0) {	/* Preferentially use the specified qubit blocks */
		initial_mappings = mappingFinder.find(mArchGraph, qmod, m_max_random_mappings, m_specified_blocks);

        m_target_sub_graph = build_sub_graph(initial_mappings[0], mArchGraph);
        //std::cout << "Gottttttt temp_graph: \n" << m_target_sub_graph->dotify() << "\n------ graph_text end ------" << std::endl;
        m_shortest_distance.init(m_target_sub_graph.get()); /* m_shortest_distance 需要目标子图存在 */
	}
	else {
        m_shortest_distance.init(mArchGraph.get());
		initial_mappings = mappingFinder.find(mArchGraph, qmod, m_shortest_distance, m_max_random_mappings);
	}

    if (initial_mappings.size() == 0)
    {
        QVec qv;
        const uint32_t used_qubit_size = qmod.get_used_qubits(qv);
        initial_mappings.emplace_back(Mapping());
        for (size_t i = 0; i < used_qubit_size; ++i){
            initial_mappings.front().emplace_back(i);
        }
    }

    auto mapping_func = [&](const Mapping& init_mapping) -> bool {
        ArchGraph::sRef tmp_graph = build_sub_graph(init_mapping, mArchGraph);
        //ArchGraph::sRef tmp_graph = mArchGraph;
        //std::cout << "Got temp_graph: \n" << tmp_graph->dotify() << "\n------ graph_text end ------" << std::endl;

        /* Forward mapping */
        auto resultFinal = allocateWithInitialMapping(init_mapping, cir_graph, qvm, tmp_graph, false);
		if (!resultFinal) {
			return false;
		}
        /* Reverse mapping */
        auto resultInit = allocateWithInitialMapping(resultFinal->first, cir_graph_reverse, qvm, tmp_graph, false);
		if (!resultInit) {
			return false;
		}
        /* Forward mapping */
        resultFinal = allocateWithInitialMapping(resultInit->first, cir_graph, qvm, tmp_graph, false);
		if (!resultFinal) {
			return false;
		}

        if (resultFinal->second < best.second) {
            best = MappingAndNSwaps(resultInit->first, resultFinal->second);
            best_sub_graph = tmp_graph;
        }
		return true;
	};

	bool found_mapping_way {false};
	if (m_max_iterations == 0 && initial_mappings.size() >= 30)
	{
		/* Traverse the initial mapping set and bidirectional iterative mapping optimization */
		for (uint32_t i = 0; i < 30; ++i)
		{
			/* the initial mapping */
			init_mapping = initial_mappings[ceil(random_generator19937(-1, initial_mappings.size() - 1))];
			/* Bidirectional Iterative mapping and preserving optimal values */
			found_mapping_way |= mapping_func(init_mapping);
		}
	}
	else if ((m_max_iterations == 0 && initial_mappings.size() < 30) || (m_max_iterations != 0 && initial_mappings.size() <= m_max_iterations))
	{
		/* Initial mapping is randomly selected and bidirectional iterative mapping optimization */
		for (uint32_t i = 0; i < initial_mappings.size(); ++i)
		{
			init_mapping = initial_mappings[i];
			found_mapping_way |= mapping_func(init_mapping);
		}
	}
	else
	{
		for (uint32_t i = 0; i < m_max_iterations; ++i)
		{
			init_mapping = initial_mappings[ceil(random_generator19937(-1, initial_mappings.size() - 1))];
			found_mapping_way |= mapping_func(init_mapping);
		}
	}

	if (!found_mapping_way) {
		QCERR_AND_THROW(run_fail, "Warning: Exceeded the maximum number of swap gates for one step on sabre-mapping.");
	}
	/* Remapping using a high-quality initial map with two forward mappings and one reverse mapping */
	m_init_mapping = best.first;
    if (auto r = allocateWithInitialMapping(best.first, cir_graph, qvm, best_sub_graph, true); r) {
    	m_swap_cnt = r->second;
    	return r->first;
	}
	return {};
}

SabreQAllocator::uRef SabreQAllocator::Create(ArchGraph::sRef ag, uint32_t mLookAhead /*= 20*/,
	uint32_t mIterations /*= 0*/, uint32_t mRandomMappings/* = 10000*/) {
    return uRef(new SabreQAllocator(ag, mLookAhead, mIterations, mRandomMappings));
}

/*******************************************************************
*                      public interface
********************************************************************/
QProg QPanda::SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv, uint32_t mLookAhead /*= 20*/,
	uint32_t mIterations /*= 0*/, const QMappingConfig& config_data, uint32_t mRandomMappings/* = 10000*/, uint32_t hops/* =1 */)
{
	std::vector<uint32_t> init_map;
	std::map<double, std::vector<std::vector<uint32_t>>, std::greater<double>> specified_blocks;
	auto mapped_prog = SABRE_mapping(prog, quantum_machine, qv, init_map,
		specified_blocks, mLookAhead, mIterations, config_data, mRandomMappings, hops);

	return mapped_prog;
}

QProg QPanda::SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv,
	std::map<double, std::vector<std::vector<uint32_t>>, std::greater<double>>& specified_blocks,
	uint32_t mLookAhead /*= 20*/, uint32_t mIterations /*= 0*/,
	const QMappingConfig& config_data, uint32_t mRandomMappings/* = 10000*/, uint32_t hops/* =1 */)
{
	std::vector<uint32_t> init_map;
	auto mapped_prog = SABRE_mapping(prog, quantum_machine, qv, init_map,
		specified_blocks, mLookAhead, mIterations, config_data, mRandomMappings, hops);

	return mapped_prog;
}

QProg QPanda::SABRE_mapping(QProg prog, QuantumMachine *quantum_machine,
	QVec &qv, std::vector<uint32_t>& init_map,
	std::map<double, std::vector<std::vector<uint32_t>>, std::greater<double>>& specified_blocks,
	uint32_t mLookAhead /*= 20*/, uint32_t mIterations /*= 0*/,
	const QMappingConfig& config_data, uint32_t mRandomMappings/* = 10000*/, uint32_t hops/* =1 */)
{
	if (prog.is_empty())
	{
		return prog;
	}

    flatten(prog);
	auto prog_copy = /*deepCopy*/(prog);

	//if (specified_blocks.size() == 0) {
	std::map<size_t, size_t> pre_map = map_to_continues_qubits(prog_copy, quantum_machine);

	QVec used_qv;
	/* Get all the used  quantum bits in the input prog */
	get_all_used_qubits(prog, used_qv);	
	prog_copy.insertQNode(prog_copy.getHeadNodeIter(), std::dynamic_pointer_cast<QNode>(BARRIER(used_qv).getImplementationPtr()));
	//}

	RemoveMeasureNode measure_cutter;
	measure_cutter.remove_measure(prog_copy);
	auto measure_info = measure_cutter.get_measure_info();
	auto measure_node = measure_cutter.get_measure_node();

    //ArchGraph::sRef graph = OptBMTQAllocator::build_arch_graph(config_data);
    auto graph = OptBMTQAllocator::build_arch_graph(config_data);

	auto allocator = SabreQAllocator::Create(graph, mLookAhead, mIterations, mRandomMappings);

    if (specified_blocks.size() > 0){ /* 只保留一个qubit块，这里为了兼容现有map接口，做临时适配修改 */
        allocator->set_specified_block(specified_blocks.begin()->second.front());
    }
	
	allocator->set_hops(hops);
    allocator->run(prog_copy, quantum_machine);

    QVec mapping_qv;
	auto mapping = allocator->get_final_mapping();
	for (auto val : mapping) {
        mapping_qv.push_back(quantum_machine->allocateQubitThroughPhyAddress(val));
	}
	
	auto mapped_prog = allocator->get_mapped_prog();
	for (auto _i = 0; _i < measure_node.size(); ++_i)
	{
		const auto& _mea = measure_node[_i];
		const auto measure_qubit_index = mapping_qv[_mea->getQuBit()->get_phy_addr()];
		const auto& _mea_info = measure_info[_i];
		mapped_prog << Measure(measure_qubit_index, _mea_info.second);
	}

	init_map = allocator->get_init_mapping();
	auto target_itr = mapped_prog.getFirstNodeIter();
	auto _first_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*target_itr);
	const auto first_gate_type = _first_gate->getQGate()->getGateType();
	if (first_gate_type != BARRIER_GATE){
		QCERR_AND_THROW(run_fail, "Error: unknow error on sabre-mapping.");
	}
	mapped_prog.deleteQNode(target_itr);
    qv.clear();
    mapped_prog.get_used_qubits(qv);
    auto qv_size = qv.size();
    qv.clear();
    for (int i = 0; i < qv_size; i++)
    {
        qv.emplace_back(mapping_qv[i]);
    }

	return mapped_prog;
}
