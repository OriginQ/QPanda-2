#ifndef SHORTEST_DISTANCE_BY_BFS_H
#define SHORTEST_DISTANCE_BY_BFS_H

#include "Core/Utilities/Tools/Graph.h"
#include <queue>

QPANDA_BEGIN

extern const uint32_t UNDEF_UINT32;

/**
* @brief Calculates the distance between two vertices by applying BFS.
*/
class ShortestDistanceByBFS
{
public:
    typedef ShortestDistanceByBFS* Ref;
    typedef std::shared_ptr<ShortestDistanceByBFS> sRef;
    typedef std::unique_ptr<ShortestDistanceByBFS> uRef;
	typedef std::vector<uint32_t> VecUInt32;
	typedef std::vector<VecUInt32> MatrixUInt32;

public:
	ShortestDistanceByBFS() {}

	/**
	* @brief Instantiate one object of this type.
	*/
	static uRef create() {
		return uRef(new ShortestDistanceByBFS());
	}

	void init(Graph::Ref graph) {
		if (graph != nullptr) {
			m_graph = graph;
			m_distance.assign(m_graph->size(), VecUInt32());
		}
	}

	uint32_t get(uint32_t u, uint32_t v) {
		check_vertex(u);
		check_vertex(v);
		if (!m_distance[u].empty()) return m_distance[u][v];
		if (!m_distance[v].empty()) return m_distance[v][u];
		get_distance_from(u);
		return m_distance[u][v];
	}

protected:
    void get_distance_from(uint32_t u) {
		auto& distance = m_distance[u];
		distance.assign(m_graph->size(), UNDEF_UINT32);

		std::queue<uint32_t> q;
		std::vector<bool> visited(m_graph->size(), false);

		q.push(u);
		visited[u] = true;
		distance[u] = 0;

		while (!q.empty()) {
			uint32_t u = q.front();
			q.pop();

			for (uint32_t v : m_graph->adj(u)) {
				if (!visited[v]) {
					visited[v] = true;
					distance[v] = distance[u] + 1;
					q.push(v);
				}
			}
		}
	}

	void check_vertex(uint32_t u) {
		if (nullptr == m_graph)
		{
			QCERR_AND_THROW(run_fail, "Set `Graph` for the DistanceGetter!");
		}

		if (m_graph->size() <= u)
		{
			QCERR_AND_THROW(run_fail, "Out of Bounds: can't calculate distance for: `" << u << "`");
		}
	}

private:
    MatrixUInt32 m_distance;
	Graph::Ref m_graph;
};

QPANDA_END
#endif
