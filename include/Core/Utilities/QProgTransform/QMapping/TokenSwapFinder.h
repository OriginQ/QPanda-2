#ifndef TOKEN_SWAP_FINDER_H
#define TOKEN_SWAP_FINDER_H

#include "Core/Utilities/Tools/Graph.h"
#include "QubitMapping.h"
#include "Core/Utilities/Tools/QPandaException.h"

QPANDA_BEGIN

// Interface for solving the Token Swap Problem.
class TokenSwapFinder {
public:
	typedef TokenSwapFinder* Ref;
	typedef std::unique_ptr<TokenSwapFinder> uRef;

public:
	virtual ~TokenSwapFinder() = default;

	// Sets the `Graph`.
	void set_graph(QPanda::Graph::Ref graph) {
		m_graph = graph;
		pre_process();
	}

	// Finds a swap sequence to reach \p to from \p from.
	SwapSeq find(const InverseMap& from, const InverseMap& to) {
		check_graph_set();
		return find_impl(from, to);
	}

protected:
	TokenSwapFinder() :m_graph(nullptr) {}

	void check_graph_set() {
		if (nullptr == m_graph)
		{
			QCERR_AND_THROW(QPanda::run_fail, "Error: please set the `Graph` for TokenSwapFinder..");
		}
	}
	virtual void pre_process() = 0;
	virtual SwapSeq find_impl(const InverseMap& from, const InverseMap& to) = 0;

protected:
	QPanda::Graph::Ref m_graph;
};

/**
* @brief 4-Approximative polynomial algorithm.
   Miltzow et al.
   DOI: 10.4230/LIPIcs.ESA.2016.66
*/
class ApproxTSFinder : public TokenSwapFinder {
public:
	typedef ApproxTSFinder* Ref;
	typedef std::unique_ptr<ApproxTSFinder> uRef;

public:
	// Creates an instance of this class.
	static uRef Create();

protected:
	void pre_process() override;
	SwapSeq find_impl(const InverseMap& from, const InverseMap& to) override;

private:
	typedef std::vector<uint32_t> GoodVertices;
	typedef std::vector<std::vector<GoodVertices>> GoodVerticesMatrix;
	GoodVerticesMatrix m_matrix;
};

/**
* @brief Simplified 4-Approximative polynomial algorithm.
   Miltzow et al.
   DOI: 10.4230/LIPIcs.ESA.2016.66
*/
class SimplifiedApproxTSFinder : public TokenSwapFinder {
public:
	typedef SimplifiedApproxTSFinder* Ref;
	typedef std::unique_ptr<SimplifiedApproxTSFinder> uRef;

public:
	// Creates an instance of this class.
	static uRef Create();

protected:
	void pre_process() override;
	SwapSeq find_impl(const InverseMap& from, const InverseMap& to) override;

private:
	typedef std::vector<uint32_t> GoodVertices;
	typedef std::vector<std::vector<GoodVertices>> GoodVerticesMatrix;
	GoodVerticesMatrix m_matrix;
};

QPANDA_END

#endif
