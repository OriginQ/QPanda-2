#ifndef __EFD_OPT_BMT_QALLOCATOR_H__
#define __EFD_OPT_BMT_QALLOCATOR_H__

#include "QbitAllocator.h"
#include "BFSCachedDistance.h"
#include "TokenSwapFinder.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include <random>
#include <queue>

namespace BMT {
	/// \brief Structure for abstracting dependencies.
	struct Dep {
		uint32_t mFrom;
		uint32_t mTo;
	};

	/// \brief Represents a sequence of dependencies (should be treated as
	/// parallel dependencies) for each node.
	struct Dependencies {
		typedef std::vector<Dep>::iterator Iterator;
		typedef std::vector<Dep>::const_iterator ConstIterator;

		std::vector<Dep> mDeps;
		QNodeRef mCallPoint;

		/// \brief Forwards to the \em mDeps attribute.
		const Dep& operator[](uint32_t i) const { return mDeps[i]; }
		/// \brief Forwards to the \em mDeps attribute.
		Dep& operator[](uint32_t i) { return mDeps[i]; }

		/// \brief Forwards to the \em mDeps attribute.
		bool empty() const { return mDeps.empty(); }

		/// \brief Forwards to the \em mDeps attribute.
		uint32_t size() const { return mDeps.size(); }

		/// \brief Forwards to the \em mDeps attribute.
		Iterator begin() { return mDeps.begin(); }
		/// \brief Forwards to the \em mDeps attribute.
		ConstIterator begin() const { return mDeps.begin(); }
		/// \brief Forwards to the \em mDeps attribute.
		Iterator end() { return mDeps.end(); }
		/// \brief Forwards to the \em mDeps attribute.
		ConstIterator end() const { return mDeps.end(); }
	};

	/// \brief Composition of each candidate in phase 1.
	struct CNodeCandidate {
		Dep dep;
		QNodeRef cNode;
		uint32_t weight;
		std::vector<QPanda::pOptimizerNodeInfo> m_relation_pre_nodes;
		std::vector<QPanda::pOptimizerNodeInfo> m_relation_successor_nodes;

		bool operator>(const CNodeCandidate& rhs) const {
			return weight > rhs.weight;
		}
	};

	typedef std::vector<uint32_t> Vector;
	typedef std::vector<Vector> Matrix;

	typedef std::vector<Mapping> MappingVector;
	typedef std::vector<std::vector<Mapping>> MappingVectorCollection;

	/// \brief Keep track of the sequence of `Mapping`s and its cost.
	struct MappingSeq {
		MappingVector mappingV;
		uint32_t mappingCost;
	};

	typedef std::vector<SwapSeq> SwapSeqVector;

	typedef std::vector<QNodeRef> PPartition;
	typedef std::vector<PPartition> PPartitionCollection;

namespace opt_bmt {
	/// \brief Composition of each candidate in phase 1.
	struct MappingCandidate {
		Mapping m;
		uint32_t cost;
		double reliability = 1.0;
		uint32_t weight;

		bool operator>(const MappingCandidate& rhs) const {
			return weight > rhs.weight;
		}
	};

	/// \brief Holds the sequence of `Mapping`s and `Swaps`to be executed.
	struct MappingSwapSequence {
		std::vector<Mapping> mappings;
		std::vector<SwapSeq> swapSeqs;
		uint32_t cost;
		double reliability;
	};

	/// \brief Interface for selecting candidates (if they are greater than
	/// a max) in phase 1.
	struct CandidateSelector 
	{
		typedef CandidateSelector* Ref;
		typedef std::unique_ptr<CandidateSelector> uRef;
		virtual ~CandidateSelector() = default;
		/// \brief Selects \em maxCandidates from \em candidates.
		virtual std::vector<MappingCandidate> select(uint32_t maxCandidates, const std::vector<MappingCandidate>& candidates) = 0;
	};


	/// \brief Necessary information for getting the combinations in phase 2.
	struct TracebackInfo {
		Mapping m;
		uint32_t parent;
		uint32_t mappingCost;
		uint32_t swapEstimatedCost;

		double mappingReliability;
		double swapEstimatedReliability;
	};
}
    /// \brief Subgraph Isomorphism based Qubit Allocator.
    ///
    /// This QAllocator is split into 3 phases:
    ///     1. Partitions the program into a number of smaller programs,
    ///         and find all* subgraph isomorphisms from the graph of that
    ///         program to the coupling graph (architecture);
    ///     2. Dynamic programming that tests all combinations of subgraph
    ///         isomorphisms, while estimating the cost of glueing them
    ///         together;
    ///     3. Reconstructs the selected sequence of subgraph isomorphisms
    ///         into a program.
    class OptBMTQAllocator : public QbitAllocator {
        public:
            typedef OptBMTQAllocator* Ref;
            typedef std::unique_ptr<OptBMTQAllocator> uRef;

            std::vector<std::vector<double>> mCnotReliability;
			std::vector<std::vector<double>> mSwapDist;
            std::vector<double> mMeaReliability;
            double mCirReliability = 1.0;
        protected:
            uint32_t mMaxPartial;
			uint32_t mMaxChildren;
			QPanda::BFSCachedDistance mBFSDistance;

            std::vector<std::vector<QNodeRef>> mPP;
            std::vector<std::vector<uint32_t>> mDistance;

            TokenSwapFinder::uRef mTSFinder;

            std::mt19937 mGen;
            std::uniform_real_distribution<double> mDistribution;
			opt_bmt::CandidateSelector::uRef mChildrenCSelector;

        private:
            std::vector<std::vector<opt_bmt::MappingCandidate>> phase1(QPanda::QProg prog, QPanda::LayeredTopoSeq& layer_info);
			std::vector<std::vector<opt_bmt::MappingCandidate>> phase1(QPanda::PressedTopoSeq& pressed_layer_info);
			opt_bmt::MappingSwapSequence phase2(const std::vector<std::vector<opt_bmt::MappingCandidate>>& collection);
            Mapping phase3(const opt_bmt::MappingSwapSequence& mss, QPanda::QuantumMachine *qvm);

            std::vector<opt_bmt::MappingCandidate> extendCandidates(const Dep& dep, const std::vector<bool>& mapped,
				const std::vector<opt_bmt::MappingCandidate>& candidates);

            void setCandidatesWeight(std::vector<opt_bmt::MappingCandidate>& candidates,
				QPanda::Graph& lastPartitionGraph);

            std::vector<opt_bmt::MappingCandidate> filterCandidates(const std::vector<opt_bmt::MappingCandidate>& candidates);

            uint32_t getNearest(uint32_t u, const InverseMap& inv);

            void propagateLiveQubits(const Mapping& fromM, Mapping& toM);

            double estimateSwapReliability(const Mapping& fromM, const Mapping& toM);
            uint32_t estimateSwapCost(const Mapping& fromM, const Mapping& toM);

            std::vector<Mapping> tracebackPath(const std::vector<std::vector<opt_bmt::TracebackInfo>>& mem, uint32_t idx);

            SwapSeq getTransformingSwapsFor(const Mapping& fromM, Mapping toM);

            void normalize(opt_bmt::MappingSwapSequence& mss);

            void init(QPanda::QProg);

            OptBMTQAllocator(QPanda::ArchGraph::sRef ag);
            Mapping allocate(QPanda::QProg qmod, QPanda::QuantumMachine *qvm) override;

			void calc_node_weight(CNodeCandidate& cNCand, const QPanda::Graph& partitionGraph,
				const std::vector<bool>& mapped);
            void createSwapReliabilityDist();
        public:
            static uRef Create(QPanda::ArchGraph::sRef ag);

			void setChildrenSelector(opt_bmt::CandidateSelector::uRef sel) {
				mChildrenCSelector = std::move(sel);
			}
    };

	uint32_t get_dependencies_cnt(QPanda::QProg prog);
	Dependencies build_deps(QNodeRef pnode);
}

#endif
