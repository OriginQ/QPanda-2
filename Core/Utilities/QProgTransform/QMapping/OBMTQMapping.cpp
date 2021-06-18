#include "Core/Utilities/QProgTransform/QMapping//OBMTQMapping.h"
#include "Core/Utilities/QProgTransform/QMapping//TokenSwapFinder.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include <algorithm>
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/Utilities/Tools/GetQubitTopology.h"
#include "Core/Utilities/Tools/QProgFlattening.h"

using namespace std;
using namespace QPanda;


#define PRINT_TRACE 0
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceMat(mat)
#endif

class CheckMultipleQGate : protected TraverseByNodeIter
{
public:
	CheckMultipleQGate()
		:m_double_gate_cnt(0)
	{}

	uint32_t get_double_gate_cnt(QProg prog) {
		traverse_qprog(prog);

		return m_double_gate_cnt;
	}

protected:
	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		if (cur_node->getQGate()->getGateType() == BARRIER_GATE)
		{
			return;
		}

		QVec qubits;
		cur_node->getQuBitVector(qubits);
		cur_node->getControlVector(qubits);
		qubits += cir_param.m_control_qubits;

		if (qubits.size() > 1){
			++m_double_gate_cnt;
		}
	}

private:
	uint32_t m_double_gate_cnt;
};

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

		PTrace("Filtering %llu candidates.", candidates.size());
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

/*******************************************************************
*                      class OptBMTQAllocator
********************************************************************/
OptBMTQAllocator::OptBMTQAllocator(ArchGraph::sRef ag, bool  optimization, uint32_t max_partial /*= (std::numeric_limits<uint32_t>::max)()*/,
	uint32_t max_children /*= (std::numeric_limits<uint32_t>::max)()*/)
	: AbstractQubitMapping(ag)
	, m_optimization(optimization)
	, m_max_partial(max_partial)
	, m_max_children(max_children)
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
                if (inv[u] != UNDEF_UINT32) continue;
                for (uint32_t v : mArchGraph->adj(u)) 
				{
                    if (inv[v] != UNDEF_UINT32) continue;
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
                if (inv[v] == UNDEF_UINT32)
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

		std::vector<MappingCandidate> selected = mChildrenCSelector->select(m_max_children, localCandidates);
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
        if (candidates[0].m[a] == UNDEF_UINT32) continue;

        for (uint32_t b : lastPartitionGraph.succ(a)) {
            if (candidates[0].m[b] == UNDEF_UINT32) continue;

            for (auto& candidate : candidates) {
                candidate.weight += mDistance[candidate.m[a]][candidate.m[b]];
            }
        }
    }
}

std::vector<MappingCandidate>
OptBMTQAllocator::filterCandidates(const std::vector<MappingCandidate>& candidates) {
    uint32_t selectionNumber = (std::min)(m_max_partial, (uint32_t) candidates.size());

    if (selectionNumber >= (uint32_t) candidates.size())
        return candidates;

    //std::cout << "Filtering " << candidates.size() << " Partial." << std::endl;

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

std::vector<std::vector<MappingCandidate>> OptBMTQAllocator::phase1(LayeredTopoSeq& layer_info)
{
    // First Phase:
    //     in this phase, we divide the program in layers, such that each layer is satisfied
    //     by any of the mappings inside 'candidates'.
    //
    mPP.push_back(std::vector<QNodeRef>());
    std::vector<std::vector<MappingCandidate>> collection;
    std::vector<MappingCandidate> candidates { { Mapping(mVQubits, UNDEF_UINT32), 0 } };
    std::vector<bool> mapped(mVQubits, false);
    Graph lastPartitionGraph(mVQubits);
    Graph partitionGraph(mVQubits);

	PTrace("OPT-BMT PHASE 1 : Solving SIP Instances.");
	auto mXbitSize = mVQubits;
	using CandidateCirNode = CNodeCandidate<QNodeRef>;
	std::priority_queue<CandidateCirNode,
		std::vector<CandidateCirNode>,
		std::greater<CandidateCirNode>> nodeQueue;
	QVec last_layer_qubits;
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
				auto used_qv = tmp_node->m_target_qubits + tmp_node->m_control_qubits;
				auto q = used_qv - last_layer_qubits;
				bool b_qubit_multiplex = (q.size() != used_qv.size());
				b_stay_cur_layer = (b_stay_cur_layer || b_qubit_multiplex);

				if (tmp_node->m_gate_type == BARRIER_GATE)
				{
					if (!b_qubit_multiplex)
					{
						mPP.back().push_back(*(tmp_node->m_iter));
						gate_iter = cur_layer.erase(gate_iter);
						continue;
					}
				}

				if ((used_qv.size() < 2))
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
						last_layer_qubits += used_qv;
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
			CandidateCirNode cNCand;
            cNCand.cNode = cnode;
            cNCand.dep = (build_deps(cnode))[0];
           
			calc_node_weight(cNCand, partitionGraph, mapped);

            nodeQueue.push(cNCand);
        }

		CandidateCirNode cNCand;
        std::vector<MappingCandidate> newCandidates;

        // In the order stablished above, the first dependency we get to satisfy,
        // we take it in, until there is no more nodes in the queue or until the
        // `newCandidates` generated is not empty.
		std::list<CandidateCirNode> remain_candidate_list;
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
            candidates = { { Mapping(mVQubits, UNDEF_UINT32), 0 } };
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

			if ((!b_stay_cur_layer) && (layer_iter == layer_info.end()))
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

std::vector<std::vector<MappingCandidate>> OptBMTQAllocator::phase1(DynamicQCircuitGraph& cir_graph)
{
	// First Phase:
	//     in this phase, we divide the program in layers, such that each layer is satisfied
	//     by any of the mappings inside 'candidates'.
	//
	mPP.push_back(std::vector<QNodeRef>());
	std::vector<std::vector<MappingCandidate>> collection;
	std::vector<MappingCandidate> candidates{ { Mapping(mVQubits, UNDEF_UINT32), 0 } };
	std::vector<bool> mapped(mVQubits, false);
	Graph lastPartitionGraph(mVQubits);
	Graph partitionGraph(mVQubits);

	PTrace("OPT-BMT PHASE 1 : Solving SIP Instances.");
	auto mXbitSize = mVQubits;
	using CandidatePressedCirNode = CNodeCandidate<pPressedCirNode>;
	//QVec last_layer_qubits;
	//for (auto layer_iter = layer_info.begin(); (layer_iter != layer_info.end()) || (nodeQueue.size() > 0); )
	while (true)
	{
		auto& front_layer = cir_graph.get_front_layer();
		if ((front_layer.size() == 0)/* && (nodeQueue.size() == 0)*/)
		{
			break;
		}

		bool b_stay_cur_layer = false;
		bool b_no_double_gate = true;
		auto& circuitNodeCandidatesVector = front_layer;

		/** 处理单门和barrier门
		*/
		for (uint32_t _i = 0; _i < circuitNodeCandidatesVector.size(); )
		{
			const auto tmp_node = (circuitNodeCandidatesVector[_i])->m_cur_node;
			if (tmp_node->m_gate_type == BARRIER_GATE ||
				(tmp_node->m_target_qubits.size() == 1))
			{
				append_pressed_node_to_mapped_collection(mPP, circuitNodeCandidatesVector[_i]);
				_i = front_layer.remove_node(_i);
			}
			else
			{
				++_i;
			}
		}

		if ((circuitNodeCandidatesVector.size() == 0) /*&& nodeQueue.empty()*/){
			continue;
		}

		std::priority_queue<CandidatePressedCirNode,
			std::vector<CandidatePressedCirNode>,
			std::greater<CandidatePressedCirNode>> nodeQueue;
		for ( uint32_t _i = 0 ; _i < circuitNodeCandidatesVector.size(); ++_i)
		{
			const auto& _item = circuitNodeCandidatesVector[_i];
			CandidatePressedCirNode cNCand;
			cNCand.cNode = _item;

			const QNodeRef& p_node = *(_item->m_cur_node->m_iter);
			cNCand.dep = (build_deps(p_node))[0];

			calc_node_weight(cNCand, partitionGraph, mapped);
			nodeQueue.push(cNCand);
		}

		CandidatePressedCirNode cNCand;
		std::vector<MappingCandidate> newCandidates;

		// In the order stablished above, the first dependency we get to satisfy,
		// we take it in, until there is no more nodes in the queue or until the
		// `newCandidates` generated is not empty.
		//std::list<CandidatePressedCirNode> remain_candidate_list;
		while (!nodeQueue.empty())
		{
			cNCand = nodeQueue.top();
			nodeQueue.pop();

			newCandidates = extendCandidates(cNCand.dep, mapped, candidates);

			if (!newCandidates.empty())
			{
				setCandidatesWeight(newCandidates, lastPartitionGraph);
				newCandidates = filterCandidates(newCandidates);

				circuitNodeCandidatesVector.remove_node(cNCand.cNode);

				break;
			}
		}

		if (newCandidates.empty())
		{
			collection.push_back(candidates);

			// Reseting all data from the last partition.
			candidates = { { Mapping(mVQubits, UNDEF_UINT32), 0 } };
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
			append_pressed_node_to_mapped_collection(mPP, cNCand.cNode);
		}
	}

	collection.push_back(candidates);

	return collection;
}

void OptBMTQAllocator::append_pressed_node_to_mapped_collection(std::vector<std::vector<QNodeRef>>& mapped_nodes,
	const pPressedCirNode& p_pressed_cir_node)
{
	for (auto& pre_node : p_pressed_cir_node->m_relation_pre_nodes)
	{
		mapped_nodes.back().emplace_back(*(pre_node->m_iter));
	}

	mapped_nodes.back().push_back(*(p_pressed_cir_node->m_cur_node->m_iter));

	for (auto& successor_node : p_pressed_cir_node->m_relation_successor_nodes)
	{
		mapped_nodes.back().emplace_back(*(successor_node->m_iter));
	}
}

template <typename T>
void OptBMTQAllocator::calc_node_weight(CNodeCandidate<T>& cNCand, const Graph& partitionGraph, 
	const std::vector<bool>& mapped)
{
#define NO_EDGE_MAPPED_WEIGHT  6
#define NO_EDGE_NO_MAPPED_WEIGHT  5
#define NO_EDGE_SINGLE_MAPPED_WEIGHT  4
#define EDGE_NO_MAPPED_WEIGHT  3
#define EDGE_SINGLE_MAPPED_WEIGHT  2
#define EDGE_MAPPED_WEIGHT  1

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
		if (mapped[a] && mapped[b]) { cNCand.weight = EDGE_MAPPED_WEIGHT; }
		else if (mapped[a] || mapped[b]) { cNCand.weight = EDGE_SINGLE_MAPPED_WEIGHT; }
		else { cNCand.weight = EDGE_NO_MAPPED_WEIGHT; }
	}
	else
	{
		// The order here is a bit different, since we want to delay the creation
		// of a new partition as much as we can.
		if (mapped[a] && mapped[b]) { cNCand.weight = NO_EDGE_MAPPED_WEIGHT; }
		else if (mapped[a] || mapped[b]) { cNCand.weight = NO_EDGE_SINGLE_MAPPED_WEIGHT; }
		else { cNCand.weight = NO_EDGE_NO_MAPPED_WEIGHT; }
	}
}

uint32_t OptBMTQAllocator::getNearest(uint32_t u, const InverseMap& inv) {
    uint32_t minV = 0;
	if (m_optimization == true)
	{
		double maxFidelity = 0.0;
		uint32_t minDist = UNDEF_UINT32;
		for (uint32_t v = 0; v < mPQubits; ++v)
		{
			auto diff = mSwapDist[u][v] - maxFidelity;
			if (inv[v] == UNDEF_UINT32 &&
				(diff > 0 || (abs(diff) < 1e-6 && mDistance[u][v] < minDist)))
			{
				maxFidelity = mSwapDist[u][v];
				minDist = mDistance[u][v];
				minV = v;
			}
		}
	}
	else
	{
		uint32_t minDist = UNDEF_UINT32;
		for (uint32_t v = 0; v < mPQubits; ++v) {
			if (inv[v] == UNDEF_UINT32 && mDistance[u][v] < minDist) {
				minDist = mDistance[u][v];
				minV = v;
			}
		}
	}
    return minV;
}

void OptBMTQAllocator::propagateLiveQubits(const Mapping& fromM, Mapping& toM) {
    auto toInv = InvertMapping(mPQubits, toM, false);

    for (uint32_t i = 0; i < mVQubits; ++i) {
        if (toM[i] == UNDEF_UINT32 && fromM[i] != UNDEF_UINT32) {
            if (toInv[fromM[i]] == UNDEF_UINT32) {
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
		if (fromM[i] != UNDEF_UINT32 && toM[i] != UNDEF_UINT32 && fromM[i] != toM[i])
		{
			totalReliability *= mSwapDist[fromM[i]][toM[i]];
		}
	}

    return totalReliability;
}

uint32_t OptBMTQAllocator::estimateSwapCost(const Mapping& fromM, const Mapping& toM) {
    uint32_t totalDistance = 0;

    for (uint32_t i = 0, e = fromM.size(); i < e; ++i) {
        if (fromM[i] != UNDEF_UINT32 && toM[i] != UNDEF_UINT32) {
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

SwapSeq OptBMTQAllocator::getTransformingSwapsFor(const Mapping& fromM, Mapping toM) {

    for (uint32_t i = 0; i < mVQubits; ++i) {
		if ((fromM[i] != UNDEF_UINT32) && (toM[i] == UNDEF_UINT32)){
			QCERR_AND_THROW(run_fail, 
				"Assumption that previous mappings have same mapped qubits than current mapping broken.");
		}

        if (fromM[i] == UNDEF_UINT32 && toM[i] != UNDEF_UINT32) {
            toM[i] = UNDEF_UINT32;
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
    
	PTrace("OPT-BMT PHASE 2 : Dynamic Programming.\nLayers: %llu, MaxSize: %llu\n", layers, layerMaxSize);
    std::vector<std::vector<TracebackInfo>> mem(layers, std::vector<TracebackInfo>());
    for (uint32_t i = 0, e = collection[0].size(); i < e; ++i) 
	{
        mem[0].push_back({ collection[0][i].m, UNDEF_UINT32, collection[0][i].cost, 0, collection[0][i].reliability, 1.0 });
    }

    for (uint32_t i = 1; i < layers; ++i) 
	{
        uint32_t jLayerSize = collection[i].size();
        for (uint32_t j = 0; j < jLayerSize; ++j)
		{
            TracebackInfo best = { {}, UNDEF_UINT32, UNDEF_UINT32, 0,0.0,0.0 };
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
				if (m_optimization == true)
				{
					auto bestReliability = best.mappingReliability * best.swapEstimatedReliability;
					auto bestCost = best.mappingCost + best.swapEstimatedCost;
					auto diff = estimatedReliability - bestReliability;
					if (diff > 0 || (abs(diff) < 1e-6 && estimatedCost < bestCost))
					{
						best = { mapping, k, mappingCost, swapEstimatedCost, mappingReliability,swapEstimatedReliability };
					}
				}
				else
				{
					if (estimatedCost < best.mappingCost + best.swapEstimatedCost)
					{
						best = { mapping, k, mappingCost, swapEstimatedCost, mappingReliability,swapEstimatedReliability };
					}
				}
			}

            mem[i].push_back(best);
        }
    }

	MappingSwapSequence best = { {}, {}, UNDEF_UINT32, 0.0 };
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

		if (m_optimization == true)
		{
			double measureReli = 1.0;
			for (auto val : mappings.back())
				measureReli *= mMeaReliability[0];

			auto diffReli = swapReliability * mappingReliability * measureReli - best.reliability;
			auto isBetterCost = (swapCost + mappingCost) < best.cost;
			if (diffReli > 0 || (abs(diffReli) < 1e-6 && isBetterCost))
			{
				best.mappings = mappings;
				best.swapSeqs = swapSeqs;
				best.cost = swapCost + mappingCost;
				best.reliability = swapReliability * mappingReliability * measureReli;
			}
		}
		else
		{
			if (swapCost + mappingCost < best.cost)
			{
				best.mappings = mappings;
				best.swapSeqs = swapSeqs;
				best.cost = swapCost + mappingCost;
				best.reliability = swapReliability * mappingReliability;
			}
		}
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
    const auto& initial_mapping = mss.mappings[idx];
	m_init_mapping = initial_mapping;
	//std::cout << "Initial Mapping: " << MappingToString(m_init_mapping) << std::endl;
    auto final_mapping = initial_mapping;
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

				m_mapped_prog << _swap(qvm->allocateQubitThroughPhyAddress(u), qvm->allocateQubitThroughPhyAddress(v));
                mCirReliability *= mSwapDist[u][v];
            }
        }

		final_mapping = mss.mappings[idx++];

        for (auto& node : partition) 
		{
            // We are sure that there are no instruction dependency that has more than
            // one dependency.
            auto iDependencies = build_deps(node);

            if (iDependencies.size() < 1) 
			{
				QVec qv;
				QNodeDeepCopy reproduction;
				switch (node->getNodeType())
				{
				case GATE_NODE:
                {
                    auto new_node = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQGateNode>(node));
                    new_node.getQuBitVector(qv);
                    auto qidx = final_mapping[qv[0]->get_phy_addr()];
                    auto qbit = qvm->allocateQubitThroughPhyAddress(qidx);
                    new_node.remap({ qbit });
					if (new_node.getQGate()->getGateType() == BARRIER_GATE)
					{
						QVec c_qv;
						new_node.getControlVector(c_qv);
						for (auto &_q : c_qv)
						{
							_q = qvm->allocateQubitThroughPhyAddress(final_mapping[_q->get_phy_addr()]);
						}
						new_node.clear_control();
						new_node.setControl(c_qv);
					}
                    m_mapped_prog << new_node;
                }
					break;

				case MEASURE_GATE:
				{
					auto new_node = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQuantumMeasure>(node));
                    auto cbit = new_node.getCBit();
                    auto qidx = final_mapping[new_node.getQuBit()->get_phy_addr()];
                    auto qbit = qvm->allocateQubitThroughPhyAddress(qidx);
					m_mapped_prog << Measure(qbit, cbit);
				}
					break;

				case  RESET_NODE:
                {
					auto new_node = reproduction.copy_node(std::dynamic_pointer_cast<AbstractQuantumReset>(node));
                    auto qidx = final_mapping[new_node.getQuBit()->get_phy_addr()];
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
            uint32_t u = final_mapping[a], v = final_mapping[b];

			if ((u == UNDEF_UINT32 || v == UNDEF_UINT32) ||
				(!mArchGraph->hasEdge(u, v) && !mArchGraph->hasEdge(v, u)))
			{
				QCERR_AND_THROW(run_fail,"Can't satisfy dependency (" << u << ", " << v << ") "
					<< "with " << idx << "-th mapping: " << MappingToString(final_mapping));
			}

            if (mArchGraph->hasEdge(u, v)) 
			{
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
				auto q_u = qvm->allocateQubitThroughPhyAddress(u);
				auto q_v = qvm->allocateQubitThroughPhyAddress(v);
				m_mapped_prog << H(q_u) << H(q_v) << CNOT(q_v, q_u) << H(q_u) << H(q_v);
                mCirReliability *= mCnotReliability[v][u];
            } 
			else
			{
				QCERR_AND_THROW(run_fail, "Mapping " << MappingToString(final_mapping)
					<< " not able to satisfy dependency "
					<< "(" << a << "{" << u << "}, " << b << "{" << v << "})");
            }
        }
    }

    return final_mapping;
}

void OptBMTQAllocator::createSwapReliabilityDist()
{
	const uint32_t qnum = mArchGraph->get_vertex_count();
	mCnotReliability = mArchGraph->get_adj_weight_matrix();
   
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
    mTSFinder = SimplifiedApproxTSFinder::Create();
    mTSFinder->set_graph(mArchGraph.get());
	setChildrenSelector(WeightedRouletteCandidateSelector::Create());
	m_shortest_distance.init(mArchGraph.get());
    mDistance.assign(mPQubits, std::vector<uint32_t>(mPQubits, 0));

    for (uint32_t i = 0; i < mPQubits; ++i) {
        for (uint32_t j = i + 1; j < mPQubits; ++j) {
            auto dist = m_shortest_distance.get(i, j);
            mDistance[i][j] = dist;
            mDistance[j][i] = dist;
        }
    }

    createSwapReliabilityDist();
}

Mapping OptBMTQAllocator::allocate(QPanda::QProg prog, QPanda::QuantumMachine *qvm) 
{
    init(prog);
	uint32_t nofDeps = get_dependencies_cnt(prog);
    auto mapping = IdentityMapping(mPQubits);

    if (nofDeps > 0) 
	{ 
		//auto layer_info = prog_layer(prog);
		//auto phase1Output = phase1(layer_info);

		DynamicQCircuitGraph cir_graph(prog);
		auto phase1Output = phase1(cir_graph);

        auto phase2Output = phase2(phase1Output);

		mapping = phase3(phase2Output, qvm);
    }
	else
	{
		m_init_mapping = mapping;
		m_mapped_prog = prog;
	}

    return mapping;
}

OptBMTQAllocator::uRef OptBMTQAllocator::Create(ArchGraph::sRef ag, bool optimization /*= false*/, uint32_t max_partial /*= (std::numeric_limits<uint32_t>::max)()*/,
	uint32_t max_children /*= (std::numeric_limits<uint32_t>::max)()*/) {
    return uRef(new OptBMTQAllocator(ag, optimization,  max_partial, max_children));
}

ArchGraph::sRef OptBMTQAllocator::build_arch_graph(const std::string& config_data /*= CONFIG_PATH*/) {
	if (config_data.length() < 6){
		return nullptr;
	}

	string suffix = config_data.substr(config_data.length() - 5);
	transform(suffix.begin(), suffix.end(), suffix.begin(), ::tolower);
	if (0 == suffix.compare(".json")){
		return JsonParser<ArchGraph>::ParseFile(config_data);
	}

	return JsonParser<ArchGraph>::ParseString(config_data);
}

/*******************************************************************
*                      public interface
********************************************************************/
uint32_t QPanda::get_dependencies_cnt(QPanda::QProg prog)
{
	CheckMultipleQGate dependencies_cnt;
	return dependencies_cnt.get_double_gate_cnt(prog);
}

Dependencies QPanda::build_deps(QNodeRef pnode)
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

QPanda::QProg QPanda::OBMT_mapping(QPanda::QProg prog, QPanda::QuantumMachine *quantum_machine,
	QPanda::QVec &qv, std::vector<uint32_t>& init_map, 
	bool optimization/* =false*/,
	uint32_t max_partial /*= (std::numeric_limits<uint32_t>::max)()*/,
	uint32_t max_children /*= (std::numeric_limits<uint32_t>::max)()*/, const std::string& config_data /*= CONFIG_PATH*/)
{
	if (prog.is_empty())
	{
		return prog;
	}

	auto prog_copy = /*deepCopy*/(prog);
	std::map<size_t, size_t> pre_map = map_to_continues_qubits(prog_copy, quantum_machine);

	ArchGraph::sRef g = OptBMTQAllocator::build_arch_graph(config_data);
#if PRINT_TRACE
	{
		//for test
		auto graph_text = g->dotify();
		cout << "graph_text: \n" << graph_text << endl;
		cout << "graph_text end ---------" << endl;
	}
#endif

	auto allocator = OptBMTQAllocator::Create(g, optimization, max_partial, max_children);
	allocator->run(prog_copy, quantum_machine);
	auto mapping = allocator->get_final_mapping();

	QVec new_qv;
	for (auto val : mapping){
		new_qv.push_back(quantum_machine->allocateQubitThroughPhyAddress(val));
	}

	qv.assign(new_qv.begin(), new_qv.begin() + pre_map.size());
	const auto& _init_mapping = allocator->get_init_mapping();
	init_map.assign(_init_mapping.begin(), _init_mapping.begin() + pre_map.size());

	return allocator->get_mapped_prog();
}

QProg QPanda::OBMT_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv,
	bool optimization /*= false*/,
	uint32_t max_partial /*= (std::numeric_limits<uint32_t>::max)()*/,
	uint32_t max_children /*= (std::numeric_limits<uint32_t>::max)()*/,
	const std::string& config_data /*= CONFIG_PATH*/)
{
	std::vector<uint32_t> init_map;
	auto mapped_prog = OBMT_mapping(prog, quantum_machine, qv, init_map, optimization, max_partial,
		max_children, config_data);

	return mapped_prog;
}