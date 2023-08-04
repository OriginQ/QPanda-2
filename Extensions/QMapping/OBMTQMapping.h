#ifndef OPT_BMT_Q_MAPPING_H
#define OPT_BMT_Q_MAPPING_H

#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include <random>
#include <queue>


#include "QMapping/QubitMapping.h"
#include "QMapping/ShortestDistanceByBFS.h"
#include "QMapping/TokenSwapFinder.h"


QPANDA_BEGIN

/**
* @brief Structure for abstracting dependencies.
*/
struct Dep {
	uint32_t mFrom;
	uint32_t mTo;
};

/**
* @brief Represents a sequence of dependencies (should be treated as parallel dependencies) for each node.
*/
struct Dependencies {
	typedef std::vector<Dep>::iterator Iterator;
	typedef std::vector<Dep>::const_iterator ConstIterator;

	std::vector<Dep> mDeps;
	QNodeRef mCallPoint;

	/**
    * @brief Forwards to the \em mDeps attribute.
    */
	const Dep& operator[](uint32_t i) const { return mDeps[i]; }
	/**
	* @brief Forwards to the \em mDeps attribute.
	*/
	Dep& operator[](uint32_t i) { return mDeps[i]; }

	/**
	* @brief Forwards to the \em mDeps attribute.
	*/
	bool empty() const { return mDeps.empty(); }
 
	/**
	* @brief Forwards to the \em mDeps attribute.
	*/
	uint32_t size() const { return mDeps.size(); }

	/**
	* @brief Forwards to the \em mDeps attribute.
	*/
	Iterator begin() { return mDeps.begin(); }

	/**
	* @brief Forwards to the \em mDeps attribute.
	*/
	ConstIterator begin() const { return mDeps.begin(); }

	/**
	* @brief Forwards to the \em mDeps attribute.
	*/
	Iterator end() { return mDeps.end(); }

	/**
	* @brief Forwards to the \em mDeps attribute.
	*/
	ConstIterator end() const { return mDeps.end(); }
};

/**
* @brief Composition of each candidate in phase 1.
*/
template <typename T>
struct CNodeCandidate {
	Dep dep;
	T cNode;
	uint32_t weight;

	bool operator>(const CNodeCandidate& rhs) const {
		return weight > rhs.weight;
	}
};

typedef std::vector<uint32_t> Vector;
typedef std::vector<Vector> Matrix;

typedef std::vector<Mapping> MappingVector;
typedef std::vector<std::vector<Mapping>> MappingVectorCollection;

/**
* @brief Keep track of the sequence of `Mapping`s and its cost.
*/
struct MappingSeq {
	MappingVector mappingV;
	uint32_t mappingCost;
};

typedef std::vector<SwapSeq> SwapSeqVector;

typedef std::vector<QNodeRef> PPartition;
typedef std::vector<PPartition> PPartitionCollection;

/**
* @brief Composition of each candidate in phase 1.
*/
struct MappingCandidate {
	Mapping m;
	uint32_t cost;
	double reliability = 1.0;
	uint32_t weight;

	bool operator>(const MappingCandidate& rhs) const {
		return weight > rhs.weight;
	}
};

/**
* @brief Holds the sequence of `Mapping`s and `Swaps`to be executed.
*/
struct MappingSwapSequence {
	std::vector<Mapping> mappings;
	std::vector<SwapSeq> swapSeqs;
	uint32_t cost;
	double reliability;
};

/**
* @brief Interface for selecting candidates (if they are greater than a max) in phase 1.
*/
struct CandidateSelector
{
	typedef CandidateSelector* Ref;
	typedef std::unique_ptr<CandidateSelector> uRef;
	virtual ~CandidateSelector() = default;
	/// \brief Selects \em maxCandidates from \em candidates.
	virtual std::vector<MappingCandidate> select(uint32_t maxCandidates, const std::vector<MappingCandidate>& candidates) = 0;
};

/**
* @brief Necessary information for getting the combinations in phase 2.
*/
struct TracebackInfo {
	Mapping m;
	uint32_t parent;
	uint32_t mappingCost;
	uint32_t swapEstimatedCost;

	double mappingReliability;
	double swapEstimatedReliability;
};

/**
* @brief Subgraph Isomorphism based Qubit Allocator.
   This QAllocator is split into 3 phases:
   1.Partitions the program into a number of smaller programs,
	 and find all* subgraph isomorphisms from the graph of that
	 program to the coupling graph (architecture);
   2.Dynamic programming that tests all combinations of subgraph
	 isomorphisms, while estimating the cost of glueing themtogether;
   3.Reconstructs the selected sequence of subgraph isomorphismsinto a program.
*/
class OptBMTQAllocator : public AbstractQubitMapping {
public:
	typedef OptBMTQAllocator* Ref;
	typedef std::unique_ptr<OptBMTQAllocator> uRef;

	std::vector<std::vector<double>> mCnotReliability;
	std::vector<std::vector<double>> mSwapDist;
	std::vector<double> mMeaReliability;
	double mCirReliability = 1.0;

public:
	static uRef Create(QPanda::ArchGraph::sRef ag,  bool b_enable_fidelity = false,uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(),
		uint32_t max_children = (std::numeric_limits<uint32_t>::max)());
    static ArchGraph::sRef build_arch_graph(const QMappingConfig& config_data = QMappingConfig());
    static ArchGraph::sRef build_arch_graph(const std::vector<std::vector<double>>& matrix_connect, const std::vector<uint32_t>& phy_partition);

	void setChildrenSelector(CandidateSelector::uRef sel) {
		mChildrenCSelector = std::move(sel);
	}
    bool b_not_need_mapping() const { return m_b_no_need_mapping; }
	void check_candidate_mapping(std::vector<std::vector<MappingCandidate>>& candidate_mapping);

	std::map<double, std::vector<Mapping>, std::greater<double>>
	select_best_qubits_blocks(ArchGraph::sRef& arch_graph, QProg& qmod);

private:
	OptBMTQAllocator(QPanda::ArchGraph::sRef ag, bool b_enable_fidelity, uint32_t max_partial, uint32_t max_children);
	Mapping allocate(QPanda::QProg prog, QPanda::QuantumMachine *qvm) override; 
	void init(QPanda::QProg);

	std::vector<std::vector<MappingCandidate>> phase1(QPanda::LayeredTopoSeq& layer_info);
	std::vector<std::vector<MappingCandidate>> phase1(DynamicQCircuitGraph& cir_graph);
	MappingSwapSequence phase2(const std::vector<std::vector<MappingCandidate>>& collection);
	Mapping phase3(const MappingSwapSequence& mss, QPanda::QuantumMachine *qvm);

	std::vector<MappingCandidate> extendCandidates(const Dep& dep, const std::vector<bool>& mapped,
		const std::vector<MappingCandidate>& candidates);
	void setCandidatesWeight(std::vector<MappingCandidate>& candidates,
		QPanda::Graph& lastPartitionGraph);

	std::vector<MappingCandidate> filterCandidates(const std::vector<MappingCandidate>& candidates);
	uint32_t getNearest(uint32_t u, const InverseMap& inv);
	void propagateLiveQubits(const Mapping& fromM, Mapping& toM);

	double estimateSwapReliability(const Mapping& fromM, const Mapping& toM);
	uint32_t estimateSwapCost(const Mapping& fromM, const Mapping& toM);
	std::vector<Mapping> tracebackPath(const std::vector<std::vector<TracebackInfo>>& mem, uint32_t idx);
	SwapSeq getTransformingSwapsFor(const Mapping& fromM, Mapping toM);
	void normalize(MappingSwapSequence& mss);
	template <typename T>
	void calc_node_weight(CNodeCandidate<T>& cNCand, const QPanda::Graph& partitionGraph,
		const std::vector<bool>& mapped);
	void createSwapReliabilityDist();
	void append_pressed_node_to_mapped_collection(std::vector<std::vector<QNodeRef>>& mapped_nodes,
		const pPressedCirNode& p_pressed_cir_node);

protected:
	uint32_t m_max_partial; /**< Limits the max number of partial solutions per step. */
	uint32_t m_max_children;
	QPanda::ShortestDistanceByBFS m_shortest_distance;

	std::vector<std::vector<QNodeRef>> mPP;
	std::vector<std::vector<uint32_t>> mDistance;

	TokenSwapFinder::uRef mTSFinder;

	std::mt19937 mGen;
	std::uniform_real_distribution<double> mDistribution;
	CandidateSelector::uRef mChildrenCSelector;
	const bool m_b_enable_fidelity;

private:
	std::set<std::pair<QPanda::Qubit*, QPanda::Qubit*>> m_bridge_qubit_pairs;
    bool m_b_no_need_mapping;
};

uint32_t get_dependencies_cnt(QPanda::QProg prog);
Dependencies build_deps(QNodeRef pnode);

/**
* @brief OPT-BMT mapping
* @ingroup Utilities
* @param[in] prog  the target prog
* @param[in] QuantumMachine *  quantum machine
* @param[out] std::map<uint32_t, Qubit*> & The final-mapping bit sequence
* @param[in] bool  If enable fidelity calculation
* @param[in] uint32_t  Limits the max number of partial solutions per step, There is no limit by default
* @param[in] uint32_t  Limits the max number of candidate-solutions per double gate, There is no limit by default
* @param[in] const std::string config data, @See JsonConfigParam::load_config()
* @return QProg   mapped  quantum program
*/
QPanda::QProg OBMT_mapping(QPanda::QProg prog, QPanda::QuantumMachine *quantum_machine,
	std::map<uint32_t, Qubit*> &final_map, std::vector<uint32_t>& init_map,
	bool b_enable_fidelity = false,
	uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(),
	uint32_t max_children = (std::numeric_limits<uint32_t>::max)(), 
    const QMappingConfig& config_data = QMappingConfig(CONFIG_PATH));

QPanda::QProg OBMT_mapping(QPanda::QProg prog, QPanda::QuantumMachine *quantum_machine,
	std::map<uint32_t, Qubit*> &final_map, bool b_enable_fidelity =false,
	uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(),
	uint32_t max_children = (std::numeric_limits<uint32_t>::max)(), 
    const QMappingConfig& config_data = QMappingConfig(CONFIG_PATH));

QPanda::QProg OBMT_mapping(QPanda::QProg prog, QPanda::QuantumMachine *quantum_machine,
	QVec &final_map, bool b_enable_fidelity = false,
	uint32_t max_partial = (std::numeric_limits<uint32_t>::max)(),
	uint32_t max_children = (std::numeric_limits<uint32_t>::max)(),
    const QMappingConfig& config_data = QMappingConfig(CONFIG_PATH));

QPANDA_END
#endif
