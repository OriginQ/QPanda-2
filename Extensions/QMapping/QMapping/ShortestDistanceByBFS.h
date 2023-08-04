#ifndef SHORTEST_DISTANCE_BY_BFS_H
#define SHORTEST_DISTANCE_BY_BFS_H

#include "Core/Utilities/Tools/Graph.h"
#include <queue>
#include <map>
#include <vector>

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
	typedef std::vector<MatrixUInt32> TripleUInt32;
	typedef std::vector<double> VecUDouble;
	typedef std::vector<VecUDouble> MatrixUDouble;

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
			m_fidelity.assign(m_graph->size(), VecUDouble());
			m_shortest_path.resize(m_graph->size());
			for (int i = 0; i < m_graph->size(); ++i)
			{
				m_shortest_path[i].assign(m_graph->size(), VecUInt32());
			}
			m_shortest_flag = false;
		}
	}

	uint32_t get(uint32_t u, uint32_t v) {
		check_vertex(u, v);
		if (!m_distance[u].empty()) return m_distance[u][v];
		if (!m_distance[v].empty()) return m_distance[v][u];
		//get_distance_from(u);
        get_short_path_from(u);
		return m_distance[u][v];
	}

	/*
	 * @brief Gets the maximum fidelity value in the shortest path set between two qubits
	 */
	double get_fidelity(uint32_t u, uint32_t v)
	{
		check_vertex(u, v);
		if (!m_fidelity[u].empty()) return m_fidelity[u][v];
		get_fidelity_from(u);
		return m_fidelity[u][v];
	}

	/*
	 * @brief Gets the overall dispersion of the mapping
	 */
    double get_overall_dispersion(Mapping mapping)
	{
        double dis_sum = 0.0;
		for (int i = 0; i < mapping.size(); ++i)
		{
			for (int j = i + 1; j < mapping.size(); ++j)
			{
                dis_sum += get_fidelity(mapping[i], mapping[j]);
			}
		}
		return dis_sum;
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

    void get_short_path_from(uint32_t u) {
        auto& distance = m_distance[u];
        if (distance.size() > 0){
            /* Indicates that the shortest path from the current point to other point has been traversed */
            return;
        }

        distance.assign(m_graph->size(), UNDEF_UINT32);

        std::queue<uint32_t> q;
        std::vector<bool> visited(m_graph->size(), false);

        q.push(u);
        visited[u] = true;
        distance[u] = 0;

        auto append_short_path_func =
            [&](const std::pair<uint32_t, uint32_t>& target_pair, const uint32_t& tmp_u) {
            const auto tmp_point_pair = std::make_pair(u, tmp_u);
            if (m_short_path.find(tmp_point_pair) != m_short_path.end())
            {
                for (const auto& _tmp_path : m_short_path.at(tmp_point_pair))
                {
                    m_short_path.at(target_pair).emplace_back(_tmp_path);
                    m_short_path.at(target_pair).back().emplace_back(tmp_u);
                }
            }
            else
            {
                m_short_path.at(target_pair).emplace_back(std::vector<uint32_t>{tmp_u});
            }
        };
        std::set<uint32_t> passed_node;
        while (!q.empty())
        {
            uint32_t _u = q.front();
            passed_node.emplace(_u);
            q.pop();

            for (uint32_t v : m_graph->adj(_u))
            {
                if (!visited[v]) {
                    visited[v] = true;
                    distance[v] = distance[_u] + 1; /* get the distance from u to v */
                    q.push(v);
                }

                if ((u == v) || (passed_node.find(v) != passed_node.end())){
                    continue;
                }

                const auto point_pair = std::make_pair(u, v);
                if (m_short_path.find(point_pair) == m_short_path.end())
                {
                    m_short_path.emplace(std::make_pair(point_pair, std::vector<std::vector<uint32_t>>()));

                    append_short_path_func(point_pair, _u);
                }
                else
                {
                    auto& cur_shot_paths = m_short_path.at(point_pair);
                    if ((cur_shot_paths.size() < 5)){
                        append_short_path_func(point_pair, _u);
                    }
                }
            }
        }
    }

    double calc_fidelity_by_short_path(uint32_t u, uint32_t v)
    {
        if (v == u) {
            return 1.0;
        }

        const auto point_pair = std::make_pair(u, v);
        if (m_short_path.find(point_pair) == m_short_path.end()){
            return .0;
        }

        double max_fidelity = .0;
        std::vector<uint32_t> max_fidelity_short_path;
        for (const auto& _short_path : m_short_path.at(point_pair))
        {
            if (_short_path.size() == 0){
                std::cerr << "Error: short path is empty !" << std::endl;
                continue;
            }

            double temp_fidelity = 1.0;
            for (size_t i = 0; i < _short_path.size() - 1; ++i){
                temp_fidelity *= m_weight_matrix[_short_path[i]][_short_path[i + 1]];
            }
            temp_fidelity *= m_weight_matrix[_short_path.back()][v];
            temp_fidelity *= (1.0 / ((float)_short_path.size())); /* 增强最短路径权重信息 */

            if (temp_fidelity > max_fidelity)
            {
                max_fidelity = temp_fidelity;
                max_fidelity_short_path = _short_path;
            }
        }

        return max_fidelity;
    }

	/*
	 * @brief Gets the multi-source shortest path
	 */
	void get_shortest_path()
	{
		std::vector<std::vector<int>> distance_matrix(m_weight_matrix.size(), std::vector<int>(m_weight_matrix[0].size(), 10000));
		/* Initialize the distance matrix */
		for (int i = 0; i < m_weight_matrix.size(); ++i)
		{
			for (int j = 0; j < m_weight_matrix[0].size(); ++j)
			{
				if (i == j)
				{
					distance_matrix[i][j] = 0;
				}
				else if (m_weight_matrix[i][j] > 0)
				{
					distance_matrix[i][j] = 1;
					if (m_shortest_path[i].empty())
					{
						m_shortest_path[i].clear();
					}
					m_shortest_path[i][j].emplace_back(i);
				}
				else
				{
					distance_matrix[i][j] = 10000;
				}
			}
		}

		/*
		 * Floyid
		 */
		int temp_dis = 0;		/**< Record the distance traveled through transit */
		/* Traverse turning points */
		for (int k = 0; k < m_weight_matrix.size(); ++k)
		{
			/* Traverse the source point */
			for (int i = 0; i < m_weight_matrix.size(); ++i)
			{
				if (k == i)
					continue;
				/* Traverse the end point */
				for (int j = 0; j < m_weight_matrix[0].size(); ++j)
				{
					if (j == i || j == k || m_weight_matrix[i][j] == 1)
						continue;

					/* Calculate the transfer distance */
					temp_dis = distance_matrix[i][k] + distance_matrix[j][k];
					if (distance_matrix[i][j] == temp_dis)		/* If the transit distance is equal to the current shortest distance, the transit qubit is added to the shortest path */
					{
						m_shortest_path[i][j].emplace_back(k);
					}
					else if (distance_matrix[i][j] > temp_dis)	/* If the transfer distance is less than the current minimum distance, the shortest distance and shortest path are updated */
					{
						m_shortest_path[i][j].clear();
						m_shortest_path[i][j].emplace_back(k);
						distance_matrix[i][j] = temp_dis;
					}
				}
			}
		}

		m_shortest_flag = true;
	}

	/*
	 * @brief Recursively calculates maximum fidelity
	 */
	double cal_fidelity(uint32_t u, uint32_t v)
	{
        if (v == u){
            return 1.0;
        }
		else if ((m_shortest_path[u][v].size() != 0) && (u == m_shortest_path[u][v][0])){
			return m_weight_matrix[u][v];
		}
		else
		{
			double fidelity = 0.0;
			for (int i = 0; i < m_shortest_path[u][v].size(); ++i)
			{
				const double temp_fidelity = m_weight_matrix[m_shortest_path[u][v][i]][v] * cal_fidelity(u, m_shortest_path[u][v][i]);
				fidelity = std::max(fidelity, temp_fidelity);
			}
			return fidelity;
		}
	}

	/*
	 * @brief Get the fidelity of a quantum circuit with a fixed source
	 */
	void get_fidelity_from(uint32_t u)
	{
		/* Convert to weighted (fidelity) undirected graphs */
		ArchGraph* arch_graph = dynamic_cast<ArchGraph*>(m_graph);
		m_weight_matrix = arch_graph->get_adj_weight_matrix();

		/*if (!m_shortest_flag)
		{
			get_shortest_path();
		}*/
        get_short_path_from(u);
			
		auto& fidelity = m_fidelity[u];
		fidelity.assign(arch_graph->size(), 0.0);
		fidelity[u] = 1.0;

		for (int i = 0; i < m_fidelity.size(); ++i)
		{
            //fidelity[i] = cal_fidelity(u, i);
            fidelity[i] = calc_fidelity_by_short_path(u, i);
		}
	}

	template<typename ...Args>
	void check_vertex(Args... args)
	{
		if constexpr (sizeof...(args) == 0) {
			QCERR_AND_THROW(run_fail, "Parameter is empty!");
		}
		if (nullptr == m_graph)
		{
			QCERR_AND_THROW(run_fail, "Set `Graph` for the DistanceGetter!");
		}

		if ((std::is_integral_v<Args> && ...)) {
			if (((m_graph->size() < args) || ...)) {
				QCERR_AND_THROW(run_fail, "Out of Bounds: can't calculate distance");
			}
		}
		else {
			QCERR_AND_THROW(run_fail, "Invalid type!");
		}
	}

private:
	MatrixUInt32 m_distance;
	Graph::Ref m_graph;
	TripleUInt32 m_shortest_path;	/**< 最短路径 */
	MatrixUDouble m_fidelity;		/**< 最大保真度值 */
	MatrixUDouble m_weight_matrix;	/**< 权重值 */
	bool m_shortest_flag;			/**< 最短路径获取成功标志 */
    std::map<std::pair<uint32_t, uint32_t>, std::vector<std::vector<uint32_t>>> m_short_path; /**< the shortest path for any point-pair */
};

QPANDA_END
#endif
