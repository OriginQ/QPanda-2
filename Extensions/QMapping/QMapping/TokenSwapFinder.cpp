#include "Core/Utilities/Tools/Graph.h"
#include "TokenSwapFinder.h"


#include <limits>
#include <queue>
#include <stack>
#include <set>
#include <map>

using namespace QPanda;

static const uint32_t _white  = 0;
static const uint32_t _silver = 1;
static const uint32_t _gray   = 2;
static const uint32_t _black  = 3;

static inline uint32_t max(uint32_t a, uint32_t b) { return (a > b) ? a : b; }
static inline uint32_t min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

static uint32_t getUndefVertex(uint32_t start, uint32_t end, std::vector<uint32_t> M) {
    for (uint32_t i = start; i < end; ++i)
        if (M[i] == UNDEF_UINT32) return i;
    return UNDEF_UINT32;
}

static void fixUndefAssignments(Graph::Ref graph, 
                                InverseMap& from, InverseMap& to) {
    uint32_t size = graph->size();
    std::vector<uint32_t> fromUndefvs;
    std::vector<uint32_t> toUndefvs;
    std::vector<bool> isnotundef(size, false);

    for (uint32_t i = 0; i < size; ++i) {
        if (from[i] == UNDEF_UINT32) { fromUndefvs.push_back(i); }
        if (to[i] == UNDEF_UINT32) { toUndefvs.push_back(i); }
        else { isnotundef[to[i]] = true; }
    }

    if (fromUndefvs.empty()) return;

    uint32_t xsize = fromUndefvs.size();
    uint32_t ysize = xsize;
    uint32_t bsize = xsize + ysize;
    WeightedGraph<uint32_t> bgraph(bsize);
    std::vector<uint32_t> M(bsize, UNDEF_UINT32);
    std::vector<uint32_t> l(bsize, 0);

    for (uint32_t i = 0; i < xsize; ++i) {
        uint32_t src = fromUndefvs[i];

        std::vector<uint32_t> d(size, UNDEF_UINT32);
        std::vector<bool> visited(size, false);
        std::queue<uint32_t> q;

        q.push(src);
        d[src] = 0;
        visited[src] = true;

        while (!q.empty()) {
            uint32_t u = q.front();
            q.pop();

            for (auto v : graph->adj(u)) {
                if (!visited[v]) {
                    d[v] = d[u] + 1;
                    visited[v] = true;
                    q.push(v);
                }
            }
        }

        for (uint32_t j = 0; j < ysize; ++j) {
            uint32_t tgt = toUndefvs[j];
            if (d[tgt] != UNDEF_UINT32)
                // The id of 'tgt' in the 'bgraph' is 'j + xsize'.
                bgraph.putEdge(i, j + xsize, d[tgt]);
        }
    }

    uint32_t maxw = 0;

    for (uint32_t i = 0; i < xsize; ++i) {
        for (uint32_t j : bgraph.succ(i)) {
            uint32_t w = bgraph.getW(i, j);
            if (maxw < w) maxw = w;
        }
    }

    for (uint32_t i = 0; i < xsize; ++i) {
        for (uint32_t j : bgraph.succ(i)) {
            uint32_t neww = maxw - bgraph.getW(i, j);
            bgraph.setW(i, j, neww);
            bgraph.setW(j, i, neww);
            l[i] = max(l[i], bgraph.getW(i, j));
        }
    }
 
    uint32_t u;
    while ((u = getUndefVertex(0, xsize, M)) != UNDEF_UINT32) {
        std::vector<uint32_t> slack(ysize, UNDEF_UINT32), slackx(ysize, u);
        std::vector<bool> S(xsize, false), T(ysize, false), NS(ysize, false);

        S[u] = true;

        for (uint32_t y : bgraph.succ(u))
            slack[y - xsize] = l[u] + l[y] - bgraph.getW(u, y);

        bool reset = false;

        for (uint32_t y : bgraph.succ(u))
            NS[y - xsize] = l[u] + l[y] == bgraph.getW(u, y);

        do {
            if (NS == T) {
                uint32_t alpha = UNDEF_UINT32;

                for (uint32_t i = 0; i < ysize; ++i)
                    if (!T[i]) alpha = min(alpha, slack[i]);

                for (uint32_t i = 0; i < xsize; ++i)
                    if (S[i]) l[i] -= alpha;
                for (uint32_t i = 0; i < ysize; ++i)
                    if (T[i]) l[i + xsize] += alpha;

                for (uint32_t i = 0; i < ysize; ++i)
                    if (!T[i]) slack[i] -= alpha;

                for (uint32_t i = 0; i < ysize; ++i) {
                    uint32_t y = i + xsize;
                    NS[i] = NS[i] || (l[slackx[i]] + l[y]) == bgraph.getW(slackx[i], y);
                }
            }

            else {
                uint32_t v = xsize;

                for (uint32_t i = 0; i < ysize; ++i)
                    if (!T[i] && NS[i]) { v += i; break; }

                if (M[v] == UNDEF_UINT32) {
                    Graph altGraph(bsize, Graph::Directed);

                    for (uint32_t x = 0; x < xsize; ++x)
                        for (uint32_t y : bgraph.succ(x))
                            if (l[x] + l[y] == bgraph.getW(x, y)) {
                                if (M[x] == y) { altGraph.putEdge(y, x); }
                                else { altGraph.putEdge(x, y); }
                            }

                    std::queue<uint32_t> q;
                    std::vector<uint32_t> pi(bsize, UNDEF_UINT32);
                    std::vector<bool> visited(bsize, false);

                    q.push(u);
                    visited[u] = true;

                    while (!q.empty()) {
                        uint32_t a = q.front();
                        q.pop();

                        if (a == v) break;

                        for (uint32_t b : altGraph.succ(a)) {
                            if (!visited[b]) {
                                pi[b] = a;
                                q.push(b);
                                visited[b] = true;
                            }
                        }
                    }

                    do {
                        M[v] = pi[v];
                        M[pi[v]] = v;
                        v = pi[pi[v]];
                    } while (v != UNDEF_UINT32);

                    reset = true;
                } else {
                    uint32_t z = M[v];
                    S[z] = true;
                    T[v - xsize] = true;

                    for (uint32_t y : bgraph.succ(z)) {
                        uint32_t i = y - xsize;
                        uint32_t newSlack = l[z] + l[y] - bgraph.getW(z, y);

                        if (slack[i] > newSlack) {
                            slack[i] = newSlack;
                            slackx[i] = z;
                        }
                    }
                }
            }
        } while (!reset);
    }

    std::vector<uint32_t> logicalUndefs;

    for (uint32_t i = 0; i < size; ++i)
        if (!isnotundef[i]) { logicalUndefs.push_back(i); }

    for (uint32_t i = 0; i < xsize; ++i)
        from[fromUndefvs[i]] = logicalUndefs[i];
    for (uint32_t i = 0; i < ysize; ++i)
        to[toUndefvs[i]] = logicalUndefs[M[i + xsize]];
}

static std::vector<uint32_t> findCycleDFS(uint32_t src,
                                          std::vector<std::vector<uint32_t>>& adj) {
    std::vector<uint32_t> color(adj.size(), _white);
    std::vector<uint32_t> pi(adj.size(), UNDEF_UINT32);
    std::stack<uint32_t> stack;

    stack.push(src);
    color[src] = _silver;

    uint32_t from, to;
    bool cyclefound = false;
    while (!cyclefound && !stack.empty()) {
        uint32_t u = stack.top();
        if (color[u] == _gray) { color[u] = _black; stack.pop(); continue; }
        color[u] = _gray;

        for (auto v : adj[u]) {
            if (color[v] == _white) {
                pi[v] = u;
                color[v] = _silver;
                stack.push(v);
            } else if (color[v] == _gray) {
                from = u; to = v;
                cyclefound = true;
                break;
            }
        }
    }

    std::vector<uint32_t> cycle;

    if (cyclefound) {
        cycle.push_back(from);

        do {
            from = pi[from];
            cycle.push_back(from);
        } while (from != to);
    }

    return cycle;
}

static std::vector<std::vector<uint32_t>>
findGoodVerticesBFS(Graph::Ref graph, uint32_t src) {
    uint32_t size = graph->size();
    const uint32_t inf = std::numeric_limits<uint32_t>::max();
    std::vector<std::vector<uint8_t>> goodvlist(size, std::vector<uint8_t>(size, false));
    std::vector<uint32_t> d(size, inf);
    std::queue<uint32_t> q;

    d[src] = 0;
    q.push(src);
    while (!q.empty()) {
        uint32_t u = q.front();
        q.pop();
        for (auto v : graph->adj(u)) {
            if (d[v] == inf) {
                q.push(v);
                d[v] = d[u] + 1;
            }

            if (d[v] == d[u] + 1) {
                goodvlist[v][v] = true;
                for (uint32_t i = 0; i < size; ++i) {
                    goodvlist[v][i] |= goodvlist[u][i];
                }
            }
        }
    }

    std::vector<std::vector<uint32_t>> goodv(size, std::vector<uint32_t>());
    for (uint32_t u = 0; u < size; ++u) {
        for (auto v : graph->adj(src)) {
            if (goodvlist[u][v])
                goodv[u].push_back(v);
        }
    }

    return goodv;
}

/*******************************************************************
*                      class ApproxTSFinder
********************************************************************/
SwapSeq ApproxTSFinder::find_impl(const InverseMap& from, const InverseMap& to) {
    auto fromInv = from;
    auto toInv = to;

    fixUndefAssignments(m_graph, fromInv, toInv);

    uint32_t size = m_graph->size();
    std::vector<std::vector<uint32_t>> gprime(size, std::vector<uint32_t>());
    std::vector<bool> inplace(size, false);
    SwapSeq swapseq;

    Mapping toMap(size, 0);
	for (uint32_t i = 0; i < size; ++i){
		toMap[toInv[i]] = i;
	}
   
	for (uint32_t i = 0; i < size; ++i){
		if (fromInv[i] == toInv[i]) inplace[i] = true;
		else inplace[i] = false;
	}

    for (uint32_t i = 0; i < size; ++i) {
        gprime[i] = m_matrix[i][toMap[fromInv[i]]];
    }

    do {
        std::vector<uint32_t> swappath;
        for (uint32_t i = 0; i < size; ++i)
            if (!inplace[i]) {
                swappath = findCycleDFS(i, gprime);
                if (!swappath.empty()) break;
            }

        if (swappath.empty()) {
            for (uint32_t u = 0; u < size; ++u) {
                if (!inplace[u]) {
                    bool found = false;

                    for (auto v : gprime[u])
                        if (inplace[v]) {
                            found = true;
                            swappath = { u, v };
                            break;
                        }

                    if (found) break;
                }
            }
        }

        if (!swappath.empty()) {
            for (uint32_t i = 1, e = swappath.size(); i < e; ++i) {
                auto u = swappath[i-1], v = swappath[i];
                swapseq.push_back({ u, v });
                std::swap(fromInv[u], fromInv[v]);
            }

            for (uint32_t i = 0, e = swappath.size(); i < e; ++i) {
                auto u = swappath[i];

                if (fromInv[u] == toInv[u]) inplace[u] = true;
                else inplace[u] = false;

                gprime[u] = m_matrix[u][toMap[fromInv[u]]];
            }
        } else {
            break;
        }
    } while (true);

    return swapseq;
}

void ApproxTSFinder::pre_process() {
    for (uint32_t u = 0; u < m_graph->size(); ++u) {
        m_matrix.push_back(findGoodVerticesBFS(m_graph, u));
    }
}

ApproxTSFinder::uRef ApproxTSFinder::Create() {
    return uRef(new ApproxTSFinder());
}

/*******************************************************************
*                      class SimplifiedApproxTSFinder
********************************************************************/
SwapSeq SimplifiedApproxTSFinder::find_impl(const InverseMap& from, const InverseMap& to) {
	auto fromInv = from;
	auto toInv = to;

	uint32_t size = m_graph->size();
	std::vector<std::vector<uint32_t>> gprime(size, std::vector<uint32_t>());
	std::vector<bool> inplace(size, true);
	SwapSeq swapseq;

	Mapping toMap(size, UNDEF_UINT32);
	for (uint32_t i = 0; i < size; ++i) {
		if (toInv[i] != UNDEF_UINT32)
			toMap[toInv[i]] = i;
	}

	for (uint32_t i = 0; i < size; ++i)
		if (fromInv[i] == UNDEF_UINT32) inplace[i] = true;
		else inplace[i] = fromInv[i] == toInv[i];

	for (uint32_t i = 0; i < size; ++i) {
		if (fromInv[i] != UNDEF_UINT32)
			gprime[i] = m_matrix[i][toMap[fromInv[i]]];
	}

	do {
		std::vector<uint32_t> swappath;
		for (uint32_t i = 0; i < size; ++i)
			if (!inplace[i]) {
				swappath = findCycleDFS(i, gprime);
				if (!swappath.empty()) break;
			}

		if (swappath.empty()) {
			for (uint32_t u = 0; u < size; ++u) {
				if (!inplace[u]) {
					bool found = false;

					for (auto v : gprime[u])
						if (inplace[v]) {
							found = true;
							swappath = { u, v };
							break;
						}

					if (found) break;
				}
			}
		}

		if (!swappath.empty()) {
			for (uint32_t i = 1, e = swappath.size(); i < e; ++i) {
				auto u = swappath[i - 1], v = swappath[i];
				swapseq.push_back({ u, v });
				std::swap(fromInv[u], fromInv[v]);
			}

			for (uint32_t i = 0, e = swappath.size(); i < e; ++i) {
				auto u = swappath[i];

				if (fromInv[u] == UNDEF_UINT32) {
					inplace[u] = true;
					gprime[u].clear();
					continue;
				}

				if (fromInv[u] == toInv[u]) inplace[u] = true;
				else inplace[u] = false;

				gprime[u] = m_matrix[u][toMap[fromInv[u]]];
			}
		}
		else {
			break;
		}
	} while (true);

	return swapseq;
}

void SimplifiedApproxTSFinder::pre_process() {
	for (uint32_t u = 0; u < m_graph->size(); ++u) {
		m_matrix.push_back(findGoodVerticesBFS(m_graph, u));
	}
}

SimplifiedApproxTSFinder::uRef SimplifiedApproxTSFinder::Create() {
	return uRef(new SimplifiedApproxTSFinder());
}
