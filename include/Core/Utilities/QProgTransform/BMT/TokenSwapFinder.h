#ifndef __EFD_TOKEN_SWAP_FINDER_H__
#define __EFD_TOKEN_SWAP_FINDER_H__

#include "Core/Utilities/Tools/Graph.h"
#include "QbitAllocator.h"
#include "Core/Utilities/Tools/QPandaException.h"

namespace BMT {
    /// \brief Interface for solving the Token Swap Problem.
    class TokenSwapFinder {
        public:
            typedef TokenSwapFinder* Ref;
            typedef std::unique_ptr<TokenSwapFinder> uRef;

        protected:
			QPanda::Graph::Ref mG;
			TokenSwapFinder() :mG(nullptr) {}

            void checkGraphSet() {
				if (nullptr == mG)
				{
					QCERR_AND_THROW(QPanda::run_fail, "Error: please set the `Graph` for TokenSwapFinder..");
				}
			}
            virtual void preprocess() = 0;
            virtual SwapSeq findImpl(const InverseMap& from, const InverseMap& to) = 0;

        public:
            virtual ~TokenSwapFinder() = default;

            /// \brief Sets the `Graph`.
            void setGraph(QPanda::Graph::Ref graph) {
				mG = graph;
				preprocess();
			}

            /// \brief Finds a swap sequence to reach \p to from \p from.
            SwapSeq find(const InverseMap& from, const InverseMap& to) {
				checkGraphSet();
				return findImpl(from, to);
			}
    };

	/// \brief 4-Approximative polynomial algorithm.
	///
	/// Miltzow et al.
	/// DOI: 10.4230/LIPIcs.ESA.2016.66
	class ApproxTSFinder : public TokenSwapFinder {
	public:
		typedef ApproxTSFinder* Ref;
		typedef std::unique_ptr<ApproxTSFinder> uRef;

	private:
		typedef std::vector<uint32_t> GoodVertices;
		typedef std::vector<std::vector<GoodVertices>> GoodVerticesMatrix;
		GoodVerticesMatrix mMatrix;

	protected:
		void preprocess() override;
		SwapSeq findImpl(const InverseMap& from, const InverseMap& to) override;

	public:
		/// \brief Creates an instance of this class.
		static uRef Create();
	};

	/// \brief Simplified 4-Approximative polynomial algorithm.
	///
	/// Miltzow et al.
	/// DOI: 10.4230/LIPIcs.ESA.2016.66
	class SimplifiedApproxTSFinder : public TokenSwapFinder {
	public:
		typedef SimplifiedApproxTSFinder* Ref;
		typedef std::unique_ptr<SimplifiedApproxTSFinder> uRef;

	private:
		typedef std::vector<uint32_t> GoodVertices;
		typedef std::vector<std::vector<GoodVertices>> GoodVerticesMatrix;
		GoodVerticesMatrix mMatrix;

	protected:
		void preprocess() override;
		SwapSeq findImpl(const InverseMap& from, const InverseMap& to) override;

	public:
		/// \brief Creates an instance of this class.
		static uRef Create();
	};
}

#endif
