#include "Core/Utilities/QProgTransform/BMT//TokenSwapFinder.h"
#include "Core/Utilities/Tools/Graph.h"

#include <limits>
#include <queue>
#include <stack>
#include <set>
#include <map>

using namespace QPanda;

// White, gray and black are the usual dfs guys.
// Silver is for marking when it is already in the stack
// (for iterative versions).
static const uint32_t _white  = 0;
static const uint32_t _silver = 1;
static const uint32_t _gray   = 2;
static const uint32_t _black  = 3;

static inline uint32_t max(uint32_t a, uint32_t b) { return (a > b) ? a : b; }
static inline uint32_t min(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

static uint32_t getUndefVertex(uint32_t start, uint32_t end, std::vector<uint32_t> M) {
    for (uint32_t i = start; i < end; ++i)
        if (M[i] == BMT::_undef) return i;
    return BMT::_undef;
}

static void fixUndefAssignments(Graph::Ref graph, 
                                BMT::InverseMap& from, BMT::InverseMap& to) {
    uint32_t size = graph->size();
    std::vector<uint32_t> fromUndefvs;
    std::vector<uint32_t> toUndefvs;
    std::vector<bool> isnotundef(size, false);

    for (uint32_t i = 0; i < size; ++i) {
        if (from[i] == BMT::_undef) { fromUndefvs.push_back(i); }
        if (to[i] == BMT::_undef) { toUndefvs.push_back(i); }
        else { isnotundef[to[i]] = true; }
    }

    // If this assignment does not have an '_undef', we don't have to do nothing.
    if (fromUndefvs.empty()) return;

    // Bipartite graph 'G = (X U Y, E)' in which we want to find a matching.
    // Mx: X -> Y (matching of each vertex from X to Y)
    // My: Y -> X (matching of each vertex from Y to X)
    // lx: from -> R ('label' for each element of 'from')
    // ly: to -> R ('label' for each element of 'to')
    uint32_t xsize = fromUndefvs.size();
    uint32_t ysize = xsize;
    uint32_t bsize = xsize + ysize;
    WeightedGraph<uint32_t> bgraph(bsize);
    std::vector<uint32_t> M(bsize, BMT::_undef);
    std::vector<uint32_t> l(bsize, 0);

    /*
     * THE HUNGARY ALGORITHM
     * 1. Initialization:
     *     1.1. Construct 'bgraph' (a complete bipartite graph) 'G = (V, E)' by applying
     *          BFS from all '_undef' vertices to all '_undef's;
     *     1.2. Construct a weight function 'w: E -> R';
     *     1.3. Set the initial labels 'lx';
     */
    for (uint32_t i = 0; i < xsize; ++i) {
        uint32_t src = fromUndefvs[i];

        std::vector<uint32_t> d(size, BMT::_undef);
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
            if (d[tgt] != BMT::_undef)
                // The id of 'tgt' in the 'bgraph' is 'j + xsize'.
                bgraph.putEdge(i, j + xsize, d[tgt]);
        }
    }

    /*
     *  Inverting the weights.
     *  This solves finds the max-weight assignment problem. As we want the minimum number
     *  of swaps, we have to find the minimum weight matching.
     *  We do this by subtracting all weights by the bigger weight.
     */
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
    /*
     *     1.4. Construct 'eqgraph' (the equality graph) 'H = (V, El)', where an edge
     *          '(u, v)' from 'E' is in 'El' iff 'w(u, v) = lx(u) + ly(v)';
     *  (NEEDED??)
     */

    /*
     * 2. Main Loop:
     *     2.1. Pick a vertex 'u' outside 'M';       <<-------------------------------|
     *     2.2. Set 'S = {u}' and 'T = 0' (empty);                                    |
     *     2.3. Initialize the 'slack' structure for computing '@' (needed on 2.4.);  |
     */
    uint32_t u;
    while ((u = getUndefVertex(0, xsize, M)) != BMT::_undef) {
        std::vector<uint32_t> slack(ysize, BMT::_undef), slackx(ysize, u);
        std::vector<bool> S(xsize, false), T(ysize, false), NS(ysize, false);

        S[u] = true;

        for (uint32_t y : bgraph.succ(u))
            slack[y - xsize] = l[u] + l[y] - bgraph.getW(u, y);
        /*
         *     2.4. If 'T' equals the neighbors of 'S':  <<------------------------|      |
         *         2.4.1. Get the minimum difference '@' from 'lx(u) + ly(v)' and 'w(u, v)',
         *                for all 'u' inside 'S' and 'v' outside 'T';              |      |
         *         2.4.2. Sum '@' to all 'lx(u)', where 'u' is inside 'S';         |      |
         *         2.4.3. Subtract '@' to all 'ly(v)', where 'v' is inside 'T'.    |      |
         */

        bool reset = false;

        for (uint32_t y : bgraph.succ(u))
            NS[y - xsize] = l[u] + l[y] == bgraph.getW(u, y);

        do {
            if (NS == T) {
                uint32_t alpha = BMT::_undef;

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

            /*
             *     2.5. Else (if 'T' does not equals the neighbors of 'S'):            |      |
             *         2.5.1. Pick a vertex 'v' that is not in 'T' but in the neighbors of 'S';
             *         2.5.2. If 'v' is outside 'M':                                   |      |
             *             2.5.2.1. 'u -> v' is an augmenting path, so augment 'M';    |      |
             *             2.5.2.1. Update the 'slack' structure;                      |      |
             *             2.5.2.1. Goto 2.1.  ----------------------------------------|------|
             *         2.5.3. Else if '(v,z)' is in 'M':                               |
             *             2.5.3.1. Set 'S = S U {z}';                                 |
             *             2.5.3.2. Set 'T = T U {v}';                                 |
             *             2.5.3.3. Update the 'slack' structure;                      |
             *             2.5.3.4. Goto 2.4.  ----------------------------------------|
             */
            else {
                uint32_t v = xsize;

                for (uint32_t i = 0; i < ysize; ++i)
                    if (!T[i] && NS[i]) { v += i; break; }

                if (M[v] == BMT::_undef) {
                    // Finding an alternate path from u -> v.
                    // 1. We create a directed graph similar to 'bgraph'. Every unmatched edge
                    // corresponds to an edge from X to Y vertices set. The oposite is true for
                    // matched edges.
                    Graph altGraph(bsize, Graph::Directed);

                    for (uint32_t x = 0; x < xsize; ++x)
                        for (uint32_t y : bgraph.succ(x))
                            if (l[x] + l[y] == bgraph.getW(x, y)) {
                                if (M[x] == y) { altGraph.putEdge(y, x); }
                                else { altGraph.putEdge(x, y); }
                            }

                    // 2. BFS through 'altGraph' until we find 'v'.
                    std::queue<uint32_t> q;
                    std::vector<uint32_t> pi(bsize, BMT::_undef);
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
                    } while (v != BMT::_undef);

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
    std::vector<uint32_t> pi(adj.size(), BMT::_undef);
    std::stack<uint32_t> stack;

    stack.push(src);
    color[src] = _silver;

    uint32_t from, to;
    bool cyclefound = false;

    // The color "hierarchy" goes:
    // white -> silver -> gray -> black
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
    // List of good vertices used to reach the 'i'-th vertex.
    // We say 'u' is a good vertex of 'v' iff the path 'src -> u -> v' results in the
    // smallest path from 'src' to 'v'.
    std::vector<std::vector<uint8_t>> goodvlist(size, std::vector<uint8_t>(size, false));
    // Distance from the source.
    std::vector<uint32_t> d(size, inf);
    std::queue<uint32_t> q;

    d[src] = 0;
    q.push(src);
    // Complexity: O(E(G) * V(G))
    while (!q.empty()) {
        uint32_t u = q.front();
        q.pop();

        // Complexity: O(Adj(u) * V(G))
        for (auto v : graph->adj(u)) {
            // If it is our first time visiting 'v' or the distance of 'src -> u -> v'
            // is equal the best distance of 'v' ('d[v]'), then 'u' is a good vertex of
            // 'v'.
            // Complexity: O(1)
            if (d[v] == inf) {
                q.push(v);
                d[v] = d[u] + 1;
            }

            if (d[v] == d[u] + 1) {
                goodvlist[v][v] = true;

                // Every good vertex of 'u' is also a good vertex of 'v'.
                // So, goodvlist[v] should be the union of goodvlist[v] and goodvlist[u],
                // since we can have multiple shortest paths reaching 'v'.
                // Complexity: O(V(G))
                for (uint32_t i = 0; i < size; ++i) {
                    goodvlist[v][i] |= goodvlist[u][i];
                }
            }
        }
    }

    std::vector<std::vector<uint32_t>> goodv(size, std::vector<uint32_t>());

    // Complexity: O(V(G) * V(G))
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
BMT::SwapSeq BMT::ApproxTSFinder::findImpl(const InverseMap& from, const InverseMap& to) {
    auto fromInv = from;
    auto toInv = to;

    fixUndefAssignments(mG, fromInv, toInv);

    uint32_t size = mG->size();
    std::vector<std::vector<uint32_t>> gprime(size, std::vector<uint32_t>());
    std::vector<bool> inplace(size, false);
    SwapSeq swapseq;

    // Constructing the inverse for 'to' -----------------------
    Mapping toMap(size, 0);
    for (uint32_t i = 0; i < size; ++i)
        toMap[toInv[i]] = i;
    // ---------------------------------------------------------

    // Initializing data ---------------------------------------
    // 1. Checking which vertices are inplace.
    for (uint32_t i = 0; i < size; ++i)
        if (fromInv[i] == toInv[i]) inplace[i] = true;
        else inplace[i] = false;

    // 2. Constructing the graph with the good neighbors.
    for (uint32_t i = 0; i < size; ++i) {
        gprime[i] = mMatrix[i][toMap[fromInv[i]]];
    }
    // ---------------------------------------------------------

    // Main Loop -----------------------------------------------
    do {
        std::vector<uint32_t> swappath;

        // 1. Trying to find a 'happy chain'
        for (uint32_t i = 0; i < size; ++i)
            if (!inplace[i]) {
                swappath = findCycleDFS(i, gprime);
                if (!swappath.empty()) break;
            }

        // 2. If we failed, we want a unhappy swap
        if (swappath.empty()) {
            // We search for an edge (u, v), such that 'u' has a label that
            // is out of place, and 'v' has a label in place.
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

        // 3. Swap what we found
        if (!swappath.empty()) {
            for (uint32_t i = 1, e = swappath.size(); i < e; ++i) {
                auto u = swappath[i-1], v = swappath[i];
                swapseq.push_back({ u, v });
                std::swap(fromInv[u], fromInv[v]);
            }

            // Updating those vertices that were swapped.
            // The others neither were magically put into place nor changed 'their mind'
            // about where to go (which are good neighbors).
            for (uint32_t i = 0, e = swappath.size(); i < e; ++i) {
                // Updating vertex u.
                auto u = swappath[i];

                if (fromInv[u] == toInv[u]) inplace[u] = true;
                else inplace[u] = false;

                gprime[u] = mMatrix[u][toMap[fromInv[u]]];
            }
        } else {
            break;
        }
    } while (true);
    // ---------------------------------------------------------

    return swapseq;
}

void BMT::ApproxTSFinder::preprocess() {
    for (uint32_t u = 0; u < mG->size(); ++u) {
        mMatrix.push_back(findGoodVerticesBFS(mG, u));
    }
}

BMT::ApproxTSFinder::uRef BMT::ApproxTSFinder::Create() {
    return uRef(new ApproxTSFinder());
}

/*******************************************************************
*                      class SimplifiedApproxTSFinder
********************************************************************/
BMT::SwapSeq BMT::SimplifiedApproxTSFinder::findImpl(const InverseMap& from, const InverseMap& to) {
	auto fromInv = from;
	auto toInv = to;

	uint32_t size = mG->size();
	std::vector<std::vector<uint32_t>> gprime(size, std::vector<uint32_t>());
	std::vector<bool> inplace(size, true);
	SwapSeq swapseq;

	// Constructing the inverse for 'to' -----------------------
	Mapping toMap(size, _undef);
	for (uint32_t i = 0; i < size; ++i) {
		if (toInv[i] != _undef)
			toMap[toInv[i]] = i;
	}
	// ---------------------------------------------------------

	// Initializing data ---------------------------------------
	// 1. Checking which vertices are inplace.
	for (uint32_t i = 0; i < size; ++i)
		if (fromInv[i] == _undef) inplace[i] = true;
		else inplace[i] = fromInv[i] == toInv[i];

	// 2. Constructing the graph with the good neighbors.
	for (uint32_t i = 0; i < size; ++i) {
		if (fromInv[i] != _undef)
			gprime[i] = mMatrix[i][toMap[fromInv[i]]];
	}
	// ---------------------------------------------------------

	// Main Loop -----------------------------------------------
	do {
		std::vector<uint32_t> swappath;

		// 1. Trying to find a 'happy chain'
		for (uint32_t i = 0; i < size; ++i)
			if (!inplace[i]) {
				swappath = findCycleDFS(i, gprime);
				if (!swappath.empty()) break;
			}

		// 2. If we failed, we want a unhappy swap
		if (swappath.empty()) {
			// We search for an edge (u, v), such that 'u' has a label that
			// is out of place, and 'v' has a label in place.
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

		// 3. Swap what we found
		if (!swappath.empty()) {
			for (uint32_t i = 1, e = swappath.size(); i < e; ++i) {
				auto u = swappath[i - 1], v = swappath[i];
				swapseq.push_back({ u, v });
				std::swap(fromInv[u], fromInv[v]);
			}

			// Updating those vertices that were swapped.
			// The others neither were magically put into place nor changed 'their mind'
			// about where to go (which are good neighbors).
			for (uint32_t i = 0, e = swappath.size(); i < e; ++i) {
				// Updating vertex u.
				auto u = swappath[i];

				if (fromInv[u] == _undef) {
					inplace[u] = true;
					gprime[u].clear();
					continue;
				}

				if (fromInv[u] == toInv[u]) inplace[u] = true;
				else inplace[u] = false;

				gprime[u] = mMatrix[u][toMap[fromInv[u]]];
			}
		}
		else {
			break;
		}
	} while (true);
	// ---------------------------------------------------------

	return swapseq;
}

void BMT::SimplifiedApproxTSFinder::preprocess() {
	for (uint32_t u = 0; u < mG->size(); ++u) {
		mMatrix.push_back(findGoodVerticesBFS(mG, u));
	}
}

BMT::SimplifiedApproxTSFinder::uRef BMT::SimplifiedApproxTSFinder::Create() {
	return uRef(new SimplifiedApproxTSFinder());
}