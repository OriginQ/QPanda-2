#include "Core/Utilities/QProgTransform/BMT//OptBMTQAllocator.h"
#include "Core/Utilities/QProgTransform/BMT//TokenSwapFinder.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include <algorithm>
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/GetQubitTopology.h"

using namespace BMT;
using namespace opt_bmt;
using namespace std;
using namespace QPanda;

#define  TEST_RELIABILITY 0

uint32_t g_max_children = (std::numeric_limits<uint32_t>::max)();
uint32_t g_MaxPartialSolutions = /*5*/(std::numeric_limits<uint32_t>::max)(); /**< Limits the max number of partial solutions per step. */
uint32_t Partitions = 0;

/// \brief Selects the candidates randomly, based on their cost.
///
/// It first calculates the weight $w_i$ of each element $i$:
/// \f[
///     w_i = \sum_{j}{c_j^2} - c_i^2
/// \f]
///
/// After, it calculates the probability $p_i$ of each element, based
/// on their weigth:
/// \f[
///     p_i = w_i / \sum_{j}{w_j}
/// \f]
///
/// Then, from a random number $r$ in [0, 1], we choose the first element $e$
/// such that $r < \sum_{i = 0}^{e}{p_i}$.
class WeightedRouletteCandidateSelector : public CandidateSelector {
public:
	typedef WeightedRouletteCandidateSelector* Ref;
	typedef std::unique_ptr<WeightedRouletteCandidateSelector> uRef;

private:
	std::mt19937 mGen;
	std::uniform_real_distribution<double> mDist;

	WeightedRouletteCandidateSelector() : mGen((std::random_device())()), mDist(0.0) {}

public:
	std::vector<MappingCandidate> select(uint32_t maxCandidates, const std::vector<MappingCandidate>& candidates) override {
		uint32_t selectionNumber = (std::min)(maxCandidates, (uint32_t)candidates.size());
		uint32_t deletedW = 0;
		uint32_t sqSum = 0;
		uint32_t wSum = 0;

		std::vector<MappingCandidate> selected;
		std::vector<uint32_t> weight;
		std::vector<bool> wasSelected(candidates.size(), false);

		if (selectionNumber == (uint32_t)candidates.size())
			return candidates;

		std::cout << "Filtering " << candidates.size() << " candidates" << std::endl;
		for (const auto& cand : candidates) {
			sqSum += (cand.cost * cand.cost);
		}

		if (sqSum == 0) {
			weight.assign(candidates.size(), 1);
		}
		else {
			for (const auto& cand : candidates) {
				weight.push_back(sqSum - (cand.cost * cand.cost));
			}
		}

		for (auto w : weight) {
			wSum += w;
		}

		for (uint32_t i = 0; i < selectionNumber; ++i) {
			double r = mDist(mGen);
			double cummulativeProbability = 0;
			uint32_t j = 0;

			while (cummulativeProbability < r && j < weight.size()) {
				if (!wasSelected[j]) {
					cummulativeProbability += (double)weight[j] / ((double)wSum - deletedW);
				}

				++j;
			}

			--j;
			wSum -= deletedW;
			deletedW = weight[j];

			wasSelected[j] = true;
			selected.push_back(candidates[j]);
		}

		return selected;
	}

	static uRef Create() { return uRef(new WeightedRouletteCandidateSelector()); }
};


// --------------------- OptBMTQAllocator ------------------------
OptBMTQAllocator::OptBMTQAllocator(ArchGraph::sRef ag)
    : QbitAllocator(ag)
	, mMaxPartial((std::numeric_limits<uint32_t>::max)())
	, mMaxChildren((std::numeric_limits<uint32_t>::max)())
    , mGen((std::random_device())())
	, mDistribution(0.0)
{}

std::vector<MappingCandidate>
OptBMTQAllocator::extendCandidates(const Dep& dep,
                                   const std::vector<bool>& mapped,
                                   const std::vector<MappingCandidate>& candidates) {
    typedef std::pair<uint32_t, uint32_t> Pair;

    std::vector<MappingCandidate> newCandidates;
    uint32_t a = dep.mFrom, b = dep.mTo;

    for (auto cand : candidates) 
	{
		std::vector<MappingCandidate> localCandidates;
        std::vector<Pair> pairV;
        auto inv = InvertMapping(mPQubits, cand.m, false);

        if (mapped[a] && mapped[b]) 
		{
            uint32_t u = cand.m[a], v = cand.m[b];

            if (mArchGraph->hasEdge(u, v) || mArchGraph->hasEdge(v, u)) 
			{
                pairV.push_back(Pair(u, v));
            }

        } 
		else if (!mapped[a] && !mapped[b]) 
		{
            for (uint32_t u = 0; u < mPQubits; ++u)
			{
                if (inv[u] != _undef) continue;
                for (uint32_t v : mArchGraph->adj(u)) 
				{
                    if (inv[v] != _undef) continue;
                    pairV.push_back(Pair(u, v));
                }
            }

        } 
		else 
		{
            uint32_t mappedV;

            if (!mapped[a]) mappedV = b;
            else mappedV = a;

            uint32_t u = cand.m[mappedV];

            for (uint32_t v : mArchGraph->adj(u))
			{
                if (inv[v] == _undef)
				{
                    if (mappedV == a) pairV.push_back(Pair(u, v));
                    else pairV.push_back(Pair(v, u));
                }
            }
        }

        for (auto& pair : pairV)
		{
            auto cpy = cand;
            cpy.m[a] = pair.first;
            cpy.m[b] = pair.second;
            cpy.cost += get_CZ_cost(pair.first, pair.second);
            cpy.reliability *= mCnotReliability[pair.first][pair.second];

            localCandidates.push_back(cpy);
        }

		std::vector<MappingCandidate> selected = mChildrenCSelector->select(mMaxChildren, localCandidates);
		newCandidates.insert(newCandidates.end(), selected.begin(), selected.end());
    }

    return newCandidates;
}

void OptBMTQAllocator::setCandidatesWeight(std::vector<MappingCandidate>& candidates,
                                           Graph& lastPartitionGraph) {
    for (auto& candidate : candidates) {
        candidate.weight = 0;
    }

    for (uint32_t a = 0; a < mVQubits; ++a) {
        if (candidates[0].m[a] == _undef) continue;

        for (uint32_t b : lastPartitionGraph.succ(a)) {
            if (candidates[0].m[b] == _undef) continue;

            for (auto& candidate : candidates) {
                candidate.weight += mDistance[candidate.m[a]][candidate.m[b]];
            }
        }
    }
}

std::vector<MappingCandidate>
OptBMTQAllocator::filterCandidates(const std::vector<MappingCandidate>& candidates) {
    uint32_t selectionNumber = (std::min)(mMaxPartial, (uint32_t) candidates.size());

    if (selectionNumber >= (uint32_t) candidates.size())
        return candidates;

    std::cout << "Filtering " << candidates.size() << " Partial." << std::endl;

    std::vector<MappingCandidate> selected;

    std::priority_queue<MappingCandidate,
                        std::vector<MappingCandidate>,
                        std::greater<MappingCandidate>> queue;
    for (auto candidate : candidates) {
        queue.push(candidate);
    }

    for (uint32_t i = 0; i < selectionNumber; ++i) {
        selected.push_back(queue.top());
        queue.pop();
    }

    return selected;
}

std::vector<std::vector<MappingCandidate>> OptBMTQAllocator::phase1(QPanda::QProg prog, LayeredTopoSeq& layer_info)
{
    // First Phase:
    //     in this phase, we divide the program in layers, such that each layer is satisfied
    //     by any of the mappings inside 'candidates'.
    //
    mPP.push_back(std::vector<QNodeRef>());

    std::vector<std::vector<MappingCandidate>> collection;
    std::vector<MappingCandidate> candidates { { Mapping(mVQubits, _undef), 0 } };
    std::vector<bool> mapped(mVQubits, false);

    Graph lastPartitionGraph(mVQubits);
    Graph partitionGraph(mVQubits);

	std::cout << "PHASE 1 >>>> Solving SIP Instances" << std::endl;

	auto mXbitSize = mVQubits;
	//auto layer_info = prog_layer(prog);
	std::priority_queue<CNodeCandidate,
		std::vector<CNodeCandidate>,
		std::greater<CNodeCandidate>> nodeQueue;
	QVec last_layer_qubits;
	/*for (size_t layer_index = 0; (layer_index < layer_info.size()) || (nodeQueue.size() > 0); )*/
	for (auto layer_iter = layer_info.begin(); (layer_iter != layer_info.end()) || (nodeQueue.size() > 0); )
	{
		bool b_stay_cur_layer = false;
		bool b_no_double_gate = true;
		std::list<QNodeRef> circuitNodeCandidatesVector;
		if (layer_iter != layer_info.end())
		{
			auto& cur_layer = *layer_iter;
			for (auto gate_iter = cur_layer.begin(); gate_iter != cur_layer.end(); )
			{
				const auto tmp_node = gate_iter->first;
				auto q = tmp_node->m_target_qubits - last_layer_qubits;
				bool b_qubit_multiplex = (q.size() != tmp_node->m_target_qubits.size());
				b_stay_cur_layer = (b_stay_cur_layer || b_qubit_multiplex);

				if ((tmp_node->m_target_qubits.size() < 2))
				{
					if (!b_qubit_multiplex)
					{
						mPP.back().push_back(*(tmp_node->m_iter));
						gate_iter = cur_layer.erase(gate_iter);
						continue;
					}
				}
				else
				{
					b_no_double_gate = false;
					if (!b_qubit_multiplex)
					{
						circuitNodeCandidatesVector.push_back(*(tmp_node->m_iter));
						last_layer_qubits += tmp_node->m_target_qubits;
						gate_iter = cur_layer.erase(gate_iter);
						continue;
					}
				}

				++gate_iter;
			}
		}

		if (b_no_double_gate && nodeQueue.empty())
		{
			++layer_iter;
			continue;
		}

		for (const auto& cnode : circuitNodeCandidatesVector)
		{
            CNodeCandidate cNCand;
            cNCand.cNode = cnode;
            cNCand.dep = (build_deps(cnode))[0];
           
			calc_node_weight(cNCand, partitionGraph, mapped);

            nodeQueue.push(cNCand);
        }

        CNodeCandidate cNCand;
        std::vector<MappingCandidate> newCandidates;

        // In the order stablished above, the first dependency we get to satisfy,
        // we take it in, until there is no more nodes in the queue or until the
        // `newCandidates` generated is not empty.
		std::list<CNodeCandidate> remain_candidate_list;
        while (!nodeQueue.empty())
		{
            cNCand = nodeQueue.top();
            nodeQueue.pop();

            newCandidates = extendCandidates(cNCand.dep, mapped, candidates);

            if (!newCandidates.empty())
			{
                setCandidatesWeight(newCandidates, lastPartitionGraph);
                newCandidates = filterCandidates(newCandidates);

				QVec q;
				std::dynamic_pointer_cast<AbstractQGateNode>(cNCand.cNode)->getQuBitVector(q);
				last_layer_qubits -= q;

                break;
            }
			remain_candidate_list.push_back(cNCand);
        }

        if (newCandidates.empty())
		{
            collection.push_back(candidates);

            // Reseting all data from the last partition.
            candidates = { { Mapping(mVQubits, _undef), 0 } };
            mapped.assign(mVQubits, false);
            mPP.push_back(std::vector<QNodeRef>());

            lastPartitionGraph = partitionGraph;
        } 
		else
		{
            auto dep = cNCand.dep;
            uint32_t a = dep.mFrom, b = dep.mTo;

            if (!mapped[b]) 
			{
                partitionGraph.succ(b).clear();
                partitionGraph.pred(b).clear();
                mapped[b] = true;
            }

            if (!mapped[a])
			{
                partitionGraph.succ(a).clear();
                partitionGraph.pred(a).clear();
                mapped[a] = true;
            }

            mapped[a] = true;
            mapped[b] = true;
            partitionGraph.putEdge(a, b);

            candidates = newCandidates;
            mPP.back().push_back(cNCand.cNode);

			if ((!b_stay_cur_layer))
			{
				++layer_iter;
			}
        }

		for (auto& candidate_node : remain_candidate_list)
		{
			if (newCandidates.empty())
			{
				calc_node_weight(candidate_node, partitionGraph, mapped);
			}
			nodeQueue.push(candidate_node);
		}
    }

    collection.push_back(candidates);

    return collection;
}

std::vector<std::vector<opt_bmt::MappingCandidate>> OptBMTQAllocator::phase1(QPanda::PressedTopoSeq& pressed_layer_info)
{
	// First Phase:
	//     in this phase, we divide the program in layers, such that each layer is satisfied
	//     by any of the mappings inside 'candidates'.
	//
	mPP.push_back(std::vector<QNodeRef>());

	std::vector<std::vector<MappingCandidate>> collection;
	std::vector<MappingCandidate> candidates{ { Mapping(mVQubits, _undef), 0 } };
	std::vector<bool> mapped(mVQubits, false);

	Graph lastPartitionGraph(mVQubits);
	Graph partitionGraph(mVQubits);

	std::cout << "PHASE 1 >>>> Solving SIP Instances" << std::endl;

	auto mXbitSize = mVQubits;
	std::priority_queue<CNodeCandidate,
		std::vector<CNodeCandidate>,
		std::greater<CNodeCandidate>> nodeQueue;
	QVec last_layer_qubits;
	for (auto layer_iter = pressed_layer_info.begin(); (layer_iter != pressed_layer_info.end()) || (nodeQueue.size() > 0); )
	{
		bool b_stay_cur_layer = false;
		bool b_no_double_gate = true;
		std::list<PressedCirNode> circuitNodeCandidatesVector;
		if (layer_iter != pressed_layer_info.end())
		{
			auto& cur_layer = *layer_iter;
			for (auto gate_iter = cur_layer.begin(); gate_iter != cur_layer.end(); )
			{
				const auto tmp_node = gate_iter->first;
				auto q = tmp_node.m_cur_node->m_target_qubits - last_layer_qubits;
				bool b_qubit_multiplex = (q.size() != tmp_node.m_cur_node->m_target_qubits.size());
				b_stay_cur_layer = (b_stay_cur_layer || b_qubit_multiplex);

				if ((tmp_node.m_cur_node->m_target_qubits.size() < 2))
				{
					QCERR_AND_THROW(run_fail, "Error: illegal single-gate in pressed_layer_info.");
					/*if (!b_qubit_multiplex)
					{
						mPP.back().push_back(*(tmp_node.m_cur_node->m_iter));
						gate_iter = cur_layer.erase(gate_iter);
						continue;
					}*/
				}
				else
				{
					b_no_double_gate = false;
					if (!b_qubit_multiplex)
					{
						circuitNodeCandidatesVector.push_back(tmp_node);
						last_layer_qubits += tmp_node.m_cur_node->m_target_qubits;
						gate_iter = cur_layer.erase(gate_iter);
						continue;
					}
				}

				++gate_iter;
			}
		}

		if (b_no_double_gate && nodeQueue.empty())
		{
			++layer_iter;
			continue;
		}

		for (const auto& cnode : circuitNodeCandidatesVector)
		{
			CNodeCandidate cNCand;
			cNCand.cNode = *(cnode.m_cur_node->m_iter);
			cNCand.dep = (build_deps(*(cnode.m_cur_node->m_iter)))[0];
			cNCand.m_relation_pre_nodes.assign(cnode.m_relation_pre_nodes.begin(), cnode.m_relation_pre_nodes.end()) ;
			cNCand.m_relation_successor_nodes.assign(cnode.m_relation_successor_nodes.begin(), cnode.m_relation_successor_nodes.end());

			calc_node_weight(cNCand, partitionGraph, mapped);

			nodeQueue.push(cNCand);
		}

		CNodeCandidate cNCand;
		std::vector<MappingCandidate> newCandidates;

		// In the order stablished above, the first dependency we get to satisfy,
		// we take it in, until there is no more nodes in the queue or until the
		// `newCandidates` generated is not empty.
		std::list<CNodeCandidate> remain_candidate_list;
		while (!nodeQueue.empty())
		{
			cNCand = nodeQueue.top();
			nodeQueue.pop();

			newCandidates = extendCandidates(cNCand.dep, mapped, candidates);

			if (!newCandidates.empty())
			{
				setCandidatesWeight(newCandidates, lastPartitionGraph);
				newCandidates = filterCandidates(newCandidates);

				QVec q;
				std::dynamic_pointer_cast<AbstractQGateNode>(cNCand.cNode)->getQuBitVector(q);
				last_layer_qubits -= q;

				break;
			}
			remain_candidate_list.push_back(cNCand);
		}

		if (newCandidates.empty())
		{
			collection.push_back(candidates);

			// Reseting all data from the last partition.
			candidates = { { Mapping(mVQubits, _undef), 0 } };
			mapped.assign(mVQubits, false);
			mPP.push_back(std::vector<QNodeRef>());

			lastPartitionGraph = partitionGraph;
		}
		else
		{
			auto dep = cNCand.dep;
			uint32_t a = dep.mFrom, b = dep.mTo;

			if (!mapped[b])
			{
				partitionGraph.succ(b).clear();
				partitionGraph.pred(b).clear();
				mapped[b] = true;
			}

			if (!mapped[a])
			{
				partitionGraph.succ(a).clear();
				partitionGraph.pred(a).clear();
				mapped[a] = true;
			}

			mapped[a] = true;
			mapped[b] = true;
			partitionGraph.putEdge(a, b);

			candidates = newCandidates;

			for (auto& pre_node : cNCand.m_relation_pre_nodes)
			{
				mPP.back().emplace_back(*(pre_node->m_iter));
			}
			mPP.back().push_back(cNCand.cNode);
			for (auto& successor_node : cNCand.m_relation_successor_nodes)
			{
				mPP.back().emplace_back(*(successor_node->m_iter));
			}

			if ((!b_stay_cur_layer))
			{
				++layer_iter;
			}
		}

		for (auto& candidate_node : remain_candidate_list)
		{
			if (newCandidates.empty())
			{
				calc_node_weight(candidate_node, partitionGraph, mapped);
			}
			nodeQueue.push(candidate_node);
		}
	}

	collection.push_back(candidates);

	return collection;
}

void OptBMTQAllocator::calc_node_weight(CNodeCandidate& cNCand, const Graph& partitionGraph, 
	const std::vector<bool>& mapped)
{
	uint32_t a = cNCand.dep.mFrom, b = cNCand.dep.mTo;
	// If we have the edge (a, b), so either:
			//     - `a` and `b` are close to each other in this partition; or
			//     - `a` and `b` were close to each other in the previous partition,
			//     and now up to one of them is mapped.
			//
			// Even though `partitionGraph` is an undirected graph, when we clear the
			// successors and predecessors later, we can't really be sure that
			// (a, b) => (b, a).
	if (partitionGraph.hasEdge(a, b) || partitionGraph.hasEdge(b, a))
	{
		// We don't need to do nothing if we find this case, since we know
		// that both are mapped and close to each other.
		if (mapped[a] && mapped[b]) { cNCand.weight = 1; }
		else if (mapped[a] || mapped[b]) { cNCand.weight = 2; }
		else { cNCand.weight = 3; }
	}
	else
	{
		// The order here is a bit different, since we want to delay the creation
		// of a new partition as much as we can.
		if (mapped[a] && mapped[b]) { cNCand.weight = 6; }
		else if (mapped[a] || mapped[b]) { cNCand.weight = 4; }
		else { cNCand.weight = 5; }
	}
}

uint32_t OptBMTQAllocator::getNearest(uint32_t u, const InverseMap& inv) {
    uint32_t minV = 0;



#if TEST_RELIABILITY
    double minDist = 0.0;
	for (uint32_t v = 0; v < mPQubits; ++v) {
		if (inv[v] == _undef && mSwapDist[u][v] > minDist) {
			minDist = mSwapDist[u][v];
			minV = v;
		}
	}
#else
    uint32_t minDist = _undef;
	for (uint32_t v = 0; v < mPQubits; ++v) {
		if (inv[v] == _undef && mDistance[u][v] < minDist) {
			minDist = mDistance[u][v];
			minV = v;
		}
	}
#endif
    return minV;
}

void OptBMTQAllocator::propagateLiveQubits(const Mapping& fromM, Mapping& toM) {
    auto toInv = InvertMapping(mPQubits, toM, false);

    for (uint32_t i = 0; i < mVQubits; ++i) {
        if (toM[i] == _undef && fromM[i] != _undef) {
            if (toInv[fromM[i]] == _undef) {
                toM[i] = fromM[i];
            } else {
                toM[i] = getNearest(fromM[i], toInv);
            }

            toInv[toM[i]] = i;
        }
    }
}

double OptBMTQAllocator::estimateSwapReliability(const Mapping& fromM, const Mapping& toM)
{
	double totalReliability = 1.0;

	for (uint32_t i = 0, e = fromM.size(); i < e; ++i)
	{
		if (fromM[i] != _undef && toM[i] != _undef && fromM[i] != toM[i])
		{
			totalReliability *= mSwapDist[fromM[i]][toM[i]];
		}
	}

    return totalReliability;
}

uint32_t OptBMTQAllocator::estimateSwapCost(const Mapping& fromM, const Mapping& toM) {
    uint32_t totalDistance = 0;

    for (uint32_t i = 0, e = fromM.size(); i < e; ++i) {
        if (fromM[i] != _undef && toM[i] != _undef) {
            totalDistance += mDistance[fromM[i]][toM[i]];
        }
    }

    return totalDistance * 30;
}

std::vector<Mapping>
OptBMTQAllocator::tracebackPath(const std::vector<std::vector<TracebackInfo>>& mem,
                                uint32_t idx) {
    std::vector<Mapping> mappings;
    uint32_t layers = mem.size();

    for (int32_t i = layers - 1; i >= 0; --i) {
        auto info = mem[i][idx];
        mappings.push_back(info.m);
        idx = info.parent;
    }
    
    std::reverse(mappings.begin(), mappings.end());
    return mappings;
}

SwapSeq OptBMTQAllocator::getTransformingSwapsFor(const Mapping& fromM,
                                                  Mapping toM) {

    for (uint32_t i = 0; i < mVQubits; ++i) {
        /*EfdAbortIf(fromM[i] != _undef && toM[i] == _undef,
                   "Assumption that previous mappings have same mapped qubits "
                   << "than current mapping broken.");*/

        if (fromM[i] == _undef && toM[i] != _undef) {
            toM[i] = _undef;
        }
    }

    auto fromInv = InvertMapping(mPQubits, fromM, false);
    auto toInv = InvertMapping(mPQubits, toM, false);

    return mTSFinder->find(fromInv, toInv);
}

void OptBMTQAllocator::normalize(MappingSwapSequence& mss)
{
    // Fill the last mapping, so that all qubits are mapped in the end.
    auto &lastMapping = mss.mappings.back();
    Fill(mPQubits, lastMapping);

    auto inv = InvertMapping(mPQubits, lastMapping);

    for (uint32_t i = mss.mappings.size() - 1; i > 0; --i) {
        mss.mappings[i - 1] = mss.mappings[i];
        Mapping& mapping = mss.mappings[i - 1];

        auto swaps = mss.swapSeqs[i - 1];

        for (auto it = swaps.rbegin(), end = swaps.rend(); it != end; ++it) {
            uint32_t u = it->u, v = it->v;
            uint32_t a = inv[u], b = inv[v];
            if (a < mVQubits) mapping[a] = v;
            if (b < mVQubits) mapping[b] = u;
            std::swap(inv[u], inv[v]);
        }
    }
}

MappingSwapSequence
OptBMTQAllocator::phase2(const std::vector<std::vector<MappingCandidate>>& collection) {
    // Second Phase:
    //     here, the idea is to use, perhaps, dynamic programming to test all possibilities
    //     for 'glueing' the sequence of collection together.
    uint32_t layers = collection.size();
    uint32_t layerMaxSize = 0;

    for (uint32_t i = 0; i < layers; ++i)
	{
        layerMaxSize = (std::max)(layerMaxSize, (uint32_t) collection[i].size());
    }
    
	std::cout << "PHASE 2 >>>> Dynamic Programming" << std::endl;
	std::cout << "Layers: " << layers << std::endl;
	std::cout << "MaxSize: " << layerMaxSize << std::endl;

    std::vector<std::vector<TracebackInfo>> mem(layers, std::vector<TracebackInfo>());

    for (uint32_t i = 0, e = collection[0].size(); i < e; ++i) 
	{
        mem[0].push_back({ collection[0][i].m, _undef, collection[0][i].cost, 0, collection[0][i].reliability, 1.0 });
    }

    for (uint32_t i = 1; i < layers; ++i) 
	{
        // INF << "Beginning: " << i << " of " << layers << " layers." << std::endl;

        uint32_t jLayerSize = collection[i].size();
        for (uint32_t j = 0; j < jLayerSize; ++j)
		{
            // Timer jt;
            // jt.start();

            TracebackInfo best = { {}, _undef, _undef, 0,0.0,0.0 };
            uint32_t kLayerSize = collection[i - 1].size();

            for (uint32_t k = 0; k < kLayerSize; ++k) 
			{
                auto mapping = collection[i][j].m;

                propagateLiveQubits(mem[i - 1][k].m, mapping);

                uint32_t mappingCost = mem[i - 1][k].mappingCost + collection[i][j].cost;

                double mappingReliability = mem[i - 1][k].mappingReliability * collection[i][j].reliability;
                double swapEstimatedReliability = estimateSwapReliability(mem[i - 1][k].m, mapping) *
					mem[i - 1][k].swapEstimatedReliability;
                double estimatedReliability = mappingReliability * swapEstimatedReliability;

                uint32_t swapEstimatedCost = estimateSwapCost(mem[i - 1][k].m, mapping) +
                                             mem[i - 1][k].swapEstimatedCost;
                uint32_t estimatedCost = mappingCost + swapEstimatedCost;

#if TEST_RELIABILITY
				if (estimatedReliability > best.mappingReliability * best.swapEstimatedReliability)
				{
					best = { mapping, k, mappingCost, swapEstimatedCost, mappingReliability,swapEstimatedReliability };
				}
#else
				if (estimatedCost < best.mappingCost + best.swapEstimatedCost)
				{
					best = { mapping, k, mappingCost, swapEstimatedCost, mappingReliability,swapEstimatedReliability };
				}
#endif
            }

            mem[i].push_back(best);

            // jt.stop();
            // INF << "(i:" << i << ", j:" << j << "): "
            //     << ((double) jt.getMilliseconds()) / 1000.0 << std::endl;
        }

		//std::cout << "End: " << i + 1 << " of " << layers << " layers." << std::endl;
    }

	MappingSwapSequence best = { {}, {}, _undef, 0.0 };

    for (uint32_t idx = 0, end = mem.back().size(); idx < end; ++idx) 
	{
        std::vector<SwapSeq> swapSeqs;
        std::vector<Mapping> mappings = tracebackPath(mem, idx);

        uint32_t swapCost = 0;
        uint32_t mappingCost = mem.back()[idx].mappingCost;

        double swapReliability = 1.0;
        double mappingReliability = mem.back()[idx].mappingReliability;

        for (uint32_t i = 1; i < layers; ++i) 
		{
            auto swaps = getTransformingSwapsFor(mappings[i - 1], mappings[i]);
            swapSeqs.push_back(swaps);

            for (const auto& s : swaps) 
			{
                swapCost += getSwapCost(s.u, s.v);
            }

			for (const auto& s : swaps)
			{
                swapReliability *= mSwapDist[s.u][s.v];
			}
        }


#if TEST_RELIABILITY

		double measureReli = 1.0;
	
		for (auto val : mappings.back())
				measureReli *= mMeaReliability[val];

		if (swapReliability * mappingReliability * measureReli > best.reliability)
		{
			best.mappings = mappings;
			best.swapSeqs = swapSeqs;
			best.cost = swapCost + mappingCost;
			best.reliability = swapReliability * mappingReliability * measureReli;
		}
#else

		if (swapCost + mappingCost < best.cost)
		{
            best.mappings = mappings;
            best.swapSeqs = swapSeqs;
			best.cost = swapCost + mappingCost;
			best.reliability = swapReliability * mappingReliability;
		}
#endif
    }

    normalize(best);
    return best;
}


Mapping OptBMTQAllocator::phase3(const MappingSwapSequence& mss, QPanda::QuantumMachine *qvm) 
{
    // Third Phase:
    //     build the operations vector by tracebacking from the solution we have
    //     found. For this, we have to go through every dependency again.
    uint32_t idx = 0;
    auto initial = mss.mappings[idx];
    auto mapping = initial;

    /*QubitRemapVisitor visitor(mapping, mXtoN);
    std::vector<Node::uRef> issuedInstructions;*/
    for (auto& partition : mPP) 
	{
        if (idx > 0) 
		{
            auto swaps = mss.swapSeqs[idx - 1];

            for (auto swp : swaps) 
			{
                uint32_t u = swp.u, v = swp.v;

                if (!mArchGraph->hasEdge(u, v)) 
				{
                    std::swap(u, v);
                }

                /*issuedInstructions.push_back(
                        CreateISwap(mArchGraph->getNode(u)->clone(),
                                    mArchGraph->getNode(v)->clone()));*/
				m_mapped_prog << SWAP(qvm->allocateQubitThroughPhyAddress(u), qvm->allocateQubitThroughPhyAddress(v));
                mCirReliability *= mSwapDist[u][v];
            }
        }

        mapping = mss.mappings[idx++];

        for (auto& node : partition) 
		{
            // We are sure that there are no instruction dependency that has more than
            // one dependency.
            auto iDependencies = build_deps(node);

            if (iDependencies.size() < 1) 
			{
                /*auto cloned = node->clone();
                cloned->apply(&visitor);
                issuedInstructions.push_back(std::move(cloned));*/
               
				QVec qv;
				QNodeDeepCopy reproduction;
				switch (node->getNodeType())
				{
				case GATE_NODE:
                {
                    auto new_node = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQGateNode>(node));
                    new_node.getQuBitVector(qv);
                    auto qidx = mapping[qv[0]->get_phy_addr()];
                    auto qbit = qvm->allocateQubitThroughPhyAddress(qidx);
                    new_node.remap({ qbit });
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
                continue;
            }

            auto dep = iDependencies[0];

            uint32_t a = dep.mFrom, b = dep.mTo;
            uint32_t u = mapping[a], v = mapping[b];

            /*EfdAbortIf((u == _undef || v == _undef) ||
                       (!mArchGraph->hasEdge(u, v) && !mArchGraph->hasEdge(v, u)),
                       "Can't satisfy dependency (" << u << ", " << v << ") "
                       << "with " << idx << "-th mapping: " << MappingToString(mapping));*/

            /*Node::uRef newNode;*/
            if (mArchGraph->hasEdge(u, v)) 
			{
               /* newNode = node->clone();
                newNode->apply(&visitor);*/
				QNodeDeepCopy reproduction;
				auto q_u = qvm->allocateQubitThroughPhyAddress(u);
				auto q_v = qvm->allocateQubitThroughPhyAddress(v);
				auto new_gate = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQGateNode>(node));
				new_gate.remap({ q_u, q_v });
				m_mapped_prog << new_gate;
                mCirReliability *= mCnotReliability[u][v];
            } 
			else if (mArchGraph->hasEdge(v, u))
			{
                /*newNode = CreateIRevCX(mArchGraph->getNode(u)->clone(),
                                       mArchGraph->getNode(v)->clone());*/
				auto q_u = qvm->allocateQubitThroughPhyAddress(u);
				auto q_v = qvm->allocateQubitThroughPhyAddress(v);
				m_mapped_prog << H(q_u) << H(q_v) << CNOT(q_v, q_u) << H(q_u) << H(q_v);
                mCirReliability *= mCnotReliability[v][u];
            } 
			else
			{
                /*EfdAbortIf(true,
                           "Mapping " << MappingToString(mapping)
                           << " not able to satisfy dependency "
                           << "(" << a << "{" << u << "}, " << b << "{" << v << "})");*/
            }

        }
    }

    std::cout << "Circuit Reliability : " << mCirReliability << std::endl;
    return mapping;
}

void OptBMTQAllocator::createSwapReliabilityDist()
{
	int qnum = 0;
	JsonConfigParam config;
	const std::string config_data;
	config.load_config(/*config_data*/);
	config.getMetadataConfig(qnum, mCnotReliability);
   
    mMeaReliability.resize(qnum);
    for (int i = 0; i < qnum; i++)
    {
        mMeaReliability[i] = 1.0;
    }

	auto graph = mCnotReliability;
	for (int i = 0; i < qnum; i++)
	{
		for (int j = 0; j < qnum; j++)
		{
			if (i == j)
				graph[i][j] == 0.0;
			else if (graph[i][j] > 1e-6)
				graph[i][j] = 1.0 - graph[i][j];
			else
				graph[i][j] = DBL_MAX;
		}
	}
    std::vector<std::vector<int>> path(qnum, std::vector<int>(qnum));
    std::vector<std::vector<double>> dist(qnum, std::vector<double>(qnum));

	for (int i = 0; i < qnum; i++)
	{
		for (int j = 0; j < qnum; j++)
		{
			dist[i][j] = graph[i][j];
			path[i][j] = j;
		}
	}

	for (int k = 0; k < qnum; k++)
	{
		for (int i = 0; i < qnum; i++)
		{
			for (int j = 0; j < qnum; j++)
			{
				if ((dist[i][k] + dist[k][j] < dist[i][j])
                    && (dist[i][k] != DBL_MAX)
                    && (dist[k][j] != DBL_MAX) 
                    && (i != j))
				{
					dist[i][j] = dist[i][k] + dist[k][j];
					path[i][j] = path[i][k];
				}
			}
		}
	}

    mSwapDist.resize(qnum);
	for (int i = 0; i < qnum; i++)
	{
		mSwapDist[i].resize(qnum);
		for (int j = 0; j < qnum; j++)
		{
			int prev = i;
			double reliability = 1.0;
			int cur = path[i][j];
			while (cur != j)
			{
				reliability *= std::pow(mCnotReliability[prev][cur], 3);
				prev = cur;
				cur = path[cur][j];
			}
            reliability *= std::pow(mCnotReliability[prev][j], 3);
	
            mSwapDist[i][j] = reliability;
		}
	}
}

void OptBMTQAllocator::init(QPanda::QProg prog) 
{
    mMaxPartial = g_MaxPartialSolutions;
	mMaxChildren = g_max_children;
    mTSFinder = SimplifiedApproxTSFinder::Create();
    mTSFinder->setGraph(mArchGraph.get());
	setChildrenSelector(WeightedRouletteCandidateSelector::Create());
    mBFSDistance.init(mArchGraph.get());
    mDistance.assign(mPQubits, std::vector<uint32_t>(mPQubits, 0));

    for (uint32_t i = 0; i < mPQubits; ++i) {
        for (uint32_t j = i + 1; j < mPQubits; ++j) {
            auto dist = mBFSDistance.get(i, j);
            mDistance[i][j] = dist;
            mDistance[j][i] = dist;
        }
    }

    createSwapReliabilityDist();
}

Mapping OptBMTQAllocator::allocate(QPanda::QProg qmod, QPanda::QuantumMachine *qvm) 
{
    init(qmod);
	uint32_t nofDeps = 1/*get_dependencies_cnt(qmod)*/;
    auto initialMapping = IdentityMapping(mPQubits);

    if (nofDeps > 0) 
	{
		auto start = chrono::system_clock::now();
		auto layer_info = prog_layer(qmod);
		auto end = chrono::system_clock::now();
		auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
		cout << "The layer takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds. layers_cnt:" << layer_info.size() << endl;
		{
			size_t gate_cnt = 0;
			for (auto& layer : layer_info)
			{
				gate_cnt += layer.size();
			}
			cout << "total gate_cnt=" << gate_cnt << endl;
			cout << "Press enter to continue." << endl;
			getchar();
		}
		start = chrono::system_clock::now();
        auto phase1Output = phase1(qmod, layer_info);
		end = chrono::system_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(end - start);
		cout << "The opt_bmt phase1 takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds" << endl;

		start = chrono::system_clock::now();
        auto phase2Output = phase2(phase1Output);
		end = chrono::system_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(end - start);
		cout << "The opt_bmt phase2 takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds" << endl;

		start = chrono::system_clock::now();
        initialMapping = phase3(phase2Output, qvm);
		end = chrono::system_clock::now();
		duration = chrono::duration_cast<chrono::microseconds>(end - start);
		cout << "The opt_bmt phase3 takes "
			<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
			<< "seconds" << endl;

        Partitions = mPP.size();
    }

    return initialMapping;
}

OptBMTQAllocator::uRef OptBMTQAllocator::Create(ArchGraph::sRef ag) {
    return uRef(new OptBMTQAllocator(ag));
}

uint32_t BMT::get_dependencies_cnt(QPanda::QProg prog)
{
	TopologyData toto_data = get_double_gate_block_topology(prog);
	const auto topo_size = toto_data.size();
	uint32_t ret = 0;
	for (size_t i = 0; i < topo_size; ++i)
	{
		for (size_t j = i + 1; j < topo_size; ++j)
		{
			ret += toto_data[i][j];
		}
	}

	return ret;
}

Dependencies BMT::build_deps(QNodeRef pnode)
{
	Dependencies dep;
	dep.mCallPoint = pnode;

	const auto node_type = pnode->getNodeType();
	if (node_type != GATE_NODE)
	{
		return dep;
	}

	switch ((std::dynamic_pointer_cast<AbstractQGateNode>(pnode))->getQGate()->getGateType())
	{
	case CU_GATE:
	case CNOT_GATE:
	case CZ_GATE:
	case CPHASE_GATE:
	case ISWAP_THETA_GATE:
	case ISWAP_GATE:
	case SQISWAP_GATE:
	case TWO_QUBIT_GATE:
	{
		QVec qv;
		(std::dynamic_pointer_cast<AbstractQGateNode>(pnode))->getQuBitVector(qv);
		dep.mDeps.push_back({ (uint32_t)(qv.front()->get_phy_addr()), (uint32_t)(qv.back()->get_phy_addr()) });
	}
	break;

	default:
		break;
	}

	return dep;
}