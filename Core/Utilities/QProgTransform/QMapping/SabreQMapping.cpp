#include "Core/Utilities/QProgTransform/QMapping//SabreQMapping.h"
#include "Core/Utilities/QProgTransform/QMapping/OBMTQMapping.h"
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

/*******************************************************************
*                      class RandomMappingFinder
* @brief Randomizes the mapping.
********************************************************************/
uint32_t Seed = std::chrono::system_clock::now().time_since_epoch().count();  /**< Seed to be used in random algorithms. */
uint32_t SeedStat; /**< Seed used in the random allocator. */

int rnd(int i) {
	static std::default_random_engine generator(Seed);
	static std::uniform_int_distribution<int> distribution(0, i - 1);
	return distribution(generator);
}

class RandomMappingFinder{
public:
	typedef RandomMappingFinder* Ref;
	typedef std::unique_ptr<RandomMappingFinder> uRef;

	Mapping find(ArchGraph::Ref g) {
		uint32_t qbits = g->size();
		Mapping mapping(qbits);

		for (uint32_t i = 0; i < qbits; ++i) {
			mapping[i] = i;
		}

		// "Generating" the initial mapping.
		SeedStat = Seed;
		std::random_shuffle(mapping.begin(), mapping.end(), rnd);

		return mapping;
	}

	/// \brief Creates an instance of this class.
	static uRef Create() {
		return uRef(new RandomMappingFinder());
	}
};


SabreQAllocator::SabreQAllocator(ArchGraph::sRef ag, uint32_t max_look_ahead, uint32_t max_iterations)
    : AbstractQubitMapping(ag), mLookAhead(max_look_ahead), mIterations(max_iterations), m_swap_cnt(0){}

SabreQAllocator::MappingAndNSwaps
SabreQAllocator::allocateWithInitialMapping(const Mapping& initialMapping, DynamicQCircuitGraph cir_graph, 
	QPanda::QuantumMachine *qvm, bool issueInstructions) {
	auto mapping = initialMapping;
    std::map<QNodeRef, uint32_t> reached;
    std::set<QNodeRef> pastLookAhead;
    uint32_t swapNum = 0;
    while (true) {
        bool changed = false;
        do {
            changed = false;
            std::set<pPressedCirNode> issueNodes; // execute_gate_list
			auto& front_layer = cir_graph.get_front_layer();
			for (uint32_t _i = 0; _i < front_layer.size();)
			{
				const auto& cur_gate = *((front_layer[_i])->m_cur_node->m_iter);
				auto deps = build_deps(cur_gate);
				if (!deps.empty()) {
					auto dep = deps[0];
					uint32_t u = mapping[dep.mFrom], v = mapping[dep.mTo];

					if (!mArchGraph->hasEdge(u, v) &&
						!mArchGraph->hasEdge(v, u)) {
						++_i;
						continue;
					}
				}

				issueNodes.insert(front_layer[_i]);
				_i = front_layer.remove_node(_i);
				changed = true;
			}

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

        std::unordered_map<QNodeRef, Dep> currentLayer;
        std::vector<Dep> nextLayer;

		// Get the logic gate that cannot be executed directly from the front layer
		const auto& cur_layer = cir_graph.get_front_layer_c();
		for (uint32_t _i = 0; _i < cur_layer.size(); ++_i)
		//for (const auto& _node : cur_layer.m_front_layer_nodes)
		{
			const auto& _node = cur_layer[_i];
			auto _gate = *(_node->m_cur_node->m_iter);
			auto dep = build_deps(_gate)[0];
			pastLookAhead.insert(_gate);
			currentLayer[_gate] = dep;
		}

        // If there is no node in the current layer, it means that
        // we have reached the end of the algorithm. i.e. we processed
        // all nodes already.
        if (currentLayer.empty()) break;

		const auto& topo_seq = cir_graph.get_layer_topo_seq();
		for (const auto& layer : topo_seq)
		{
			for (const auto& _n : layer)
			{
				const auto& _n_ref = *(_n.first->m_cur_node->m_iter);
				if (pastLookAhead.find(_n_ref) == pastLookAhead.end()){
					auto deps = build_deps(_n_ref);
					if (!deps.empty()) nextLayer.push_back(deps[0]);
				}
			}

			if (nextLayer.size() > mLookAhead)
			{
				break;
			}
		}

        std::set<uint32_t> usedQubits;
        for (auto pair : currentLayer) {
            usedQubits.insert(mapping[pair.second.mFrom]);
            usedQubits.insert(mapping[pair.second.mTo]);
        }

        auto invM = InvertMapping(mPQubits, mapping);
        auto best = WeightedSwap(UNDEF_UINT32, Swap { 0, 0 });

        for (auto u : usedQubits) {
            for (auto v : mArchGraph->adj(u)) {
                auto cpy = mapping;
                std::swap(cpy[invM[u]], cpy[invM[v]]);

                double currentLCost = 0;
                double nextLCost = 0;

                for (auto pair : currentLayer) {
                    currentLCost += m_shortest_distance.get(cpy[pair.second.mFrom], cpy[pair.second.mTo]);
                }

                for (const auto& dep : nextLayer) {
                    nextLCost += m_shortest_distance.get(cpy[dep.mFrom], cpy[dep.mTo]);
                }

                currentLCost = currentLCost / currentLayer.size();
                if (!nextLayer.empty()) nextLCost = nextLCost / nextLayer.size();
                double cost = currentLCost + 0.5 * nextLCost;

                if (cost < best.first) {
                    best.first = cost;
                    best.second = Swap { u, v };
                }
            }
        }

        auto swap = best.second;
        std::swap(mapping[invM[swap.u]], mapping[invM[swap.v]]);
        if (issueInstructions) {
			m_mapped_prog << _swap(qvm->allocateQubitThroughPhyAddress(swap.u), qvm->allocateQubitThroughPhyAddress(swap.v));
        }

        ++swapNum;
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
			new_node.remap({ qbit });
			if (new_node.getQGate()->getGateType() == BARRIER_GATE)
			{
				QVec c_qv;
				new_node.getControlVector(c_qv);
				for (auto &_q : c_qv)
				{
					_q = qvm->allocateQubitThroughPhyAddress(mapping[_q->get_phy_addr()]);
				}
				new_node.clear_control();
				new_node.setControl(c_qv);
			}
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

Mapping SabreQAllocator::allocate(QProg qmod, QuantumMachine *qvm) {
	DynamicQCircuitGraph cir_graph(qmod);

	QProg prog_reverse;
	for (auto gate_itr = qmod.getFirstNodeIter(); gate_itr != qmod.getEndNodeIter(); ++gate_itr) 
	{
		prog_reverse.insertQNode(prog_reverse.getHeadNodeIter(), *gate_itr);
	}
	DynamicQCircuitGraph cir_graph_reverse(prog_reverse);
	m_shortest_distance.init(mArchGraph.get());

    Mapping initialM, finalM;

    RandomMappingFinder mappingFinder;
    MappingAndNSwaps best(initialM, (std::numeric_limits<uint32_t>::max)());

    for (uint32_t i = 0; i < mIterations; ++i) {
        initialM = mappingFinder.find(mArchGraph.get());
        auto resultFinal = allocateWithInitialMapping(initialM, cir_graph, qvm, false);
        auto resultInit = allocateWithInitialMapping(resultFinal.first, cir_graph_reverse, qvm, false);
        resultFinal = allocateWithInitialMapping(resultInit.first, cir_graph, qvm, false);
        if (resultFinal.second < best.second) {
            best = MappingAndNSwaps(resultInit.first, resultFinal.second);
        }
    }

	m_init_mapping = best.first;
	//std::cout << "Initial Mapping: " << MappingToString(m_init_mapping) << std::endl;
    auto r = allocateWithInitialMapping(best.first, cir_graph, qvm, true);
	m_swap_cnt = r.second;

    return r.first;
}

SabreQAllocator::uRef SabreQAllocator::Create(ArchGraph::sRef ag, uint32_t max_look_ahead /*= 20*/,
	uint32_t max_iterations /*= 10*/) {
    return uRef(new SabreQAllocator(ag, max_look_ahead, max_iterations));
}

/*******************************************************************
*                      public interface
********************************************************************/
QProg QPanda::SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv, uint32_t max_look_ahead /*= 20*/,
	uint32_t max_iterations /*= 10*/, const std::string& config_data /*= CONFIG_PATH*/)
{
	std::vector<uint32_t> init_map;
	auto mapped_prog = SABRE_mapping(prog, quantum_machine, qv, init_map,
		max_look_ahead, max_iterations, config_data);

	return mapped_prog;
}

QProg QPanda::SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv, std::vector<uint32_t>& init_map,
	uint32_t max_look_ahead /*= 20*/, uint32_t max_iterations /*= 10*/, const std::string& config_data /*= CONFIG_PATH*/)
{
	if (prog.is_empty())
	{
		return prog;
	}

	auto prog_copy = /*deepCopy*/(prog);
	std::map<size_t, size_t> pre_map = map_to_continues_qubits(prog_copy, quantum_machine);

	ArchGraph::sRef g = OptBMTQAllocator::build_arch_graph(config_data);
	auto allocator = SabreQAllocator::Create(g, max_look_ahead, max_iterations);
	allocator->run(prog_copy, quantum_machine);

	auto mapping = allocator->get_final_mapping();
	qv.clear();
	for (auto val : mapping) {
		qv.push_back(quantum_machine->allocateQubitThroughPhyAddress(val));
	}

	init_map = allocator->get_init_mapping();
	return allocator->get_mapped_prog();
}
