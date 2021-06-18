#ifndef _QUICKBB_HPP_
#define _QUICKBB_HPP_
#include <set>
#include <chrono>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
typedef size_t vertex_index_t;
typedef std::vector<vertex_index_t> adj_arr_t;
typedef std::map<vertex_index_t, adj_arr_t> graph_data_t;

class QuickBB
{
public:
	QuickBB() = default;

	class Graph
	{
		graph_data_t m_data_;

	public:
		Graph() = default;
		const adj_arr_t& get_neighborhood(vertex_index_t nodeIndex) const
		{
			return m_data_.at(nodeIndex);
		}

		auto begin() const { return m_data_.begin(); }
		auto end() const { return m_data_.end(); }

		void remove_vertex(vertex_index_t vertexIndex)
		{
			std::vector<vertex_index_t> to_be_removed;
			to_be_removed.emplace_back(vertexIndex);
			for (const auto& v : m_data_)
			{
				if (v.first == vertexIndex)
					continue;

				auto pred = [vertexIndex](vertex_index_t v) { return vertexIndex == v; };
				auto remove_it = std::remove_if(m_data_[v.first].begin(), m_data_[v.first].end(), pred);
				m_data_[v.first].erase(remove_it, m_data_[v.first].end());
				if (degree(v.first) == 0)
					to_be_removed.emplace_back(v.first);
			}

			for (const auto& v : to_be_removed)
				m_data_.erase(m_data_.find(v));
		}

		bool has_edge(vertex_index_t u, vertex_index_t v) const
		{
			if (m_data_.count(v) == 0 || m_data_.count(u) == 0) return false;
			auto result = std::find(m_data_.at(u).begin(), m_data_.at(u).end(), v);
			return result != m_data_.at(u).end();
		}

		bool add_edge(vertex_index_t u, vertex_index_t v)
		{
			if (has_edge(u, v)) return false;
			m_data_[u].emplace_back(v);
			m_data_[v].emplace_back(u);
			return true;
		}

		bool remove_edge(vertex_index_t u, vertex_index_t v)
		{
			if (!has_edge(u, v))
				return false;

			auto u_begin = m_data_[u].begin();
			auto u_end = m_data_[u].end();
			m_data_[u].erase(std::remove(u_begin, u_end, v), u_end);
			if (degree(u) == 0)
				m_data_.erase(m_data_.find(u));

			auto v_begin = m_data_[v].begin();
			auto v_end = m_data_[v].end();
			m_data_[v].erase(std::remove(v_begin, v_end, u), v_end);
			if (degree(v) == 0)
				m_data_.erase(m_data_.find(v));

			return true;
		}

		void contract_edge(vertex_index_t u, vertex_index_t v)
		{
			m_data_[u].erase(std::find(m_data_[u].begin(), m_data_[u].end(), v));
			const auto u_deg = m_data_[u].size();
			for (auto n : m_data_[v])
			{
				if (n == u)
					continue;

				auto last = m_data_[u].begin() + u_deg;
				auto found = std::find(m_data_[u].begin(), last, n);

				if (found != last)
				{
					m_data_[n].erase(std::find(m_data_[n].begin(), m_data_[n].end(), v));
				}
				else
				{
					m_data_[u].emplace_back(n);
					for (auto& a : m_data_[n])
					{
						if (a == v)
						{
							a = u;
							break;
						}
					}
				}
			}
			m_data_.erase(v);
		}

		vertex_index_t degree(vertex_index_t nodeIndex) const
		{
			return m_data_.at(nodeIndex).size();
		}

		vertex_index_t order() const { return m_data_.size(); }
	};

	/**
	* @brief  compute the optimal order
	* @param[in]  Graph  QuickBB graph
	* @param[in]  size_t   alloted compute time
	* @return  std::pair<size_t, adj_arr_t>  first :  tree width  , second : order
	*/
	static std::pair<size_t, adj_arr_t> compute(Graph &graph, size_t alloted_time)
	{
		auto start = std::chrono::steady_clock::now();

		auto upper_bound_pair = upper_bound(graph);
		auto lb = lower_bound(graph);

		auto best_order = upper_bound_pair.first;
		auto best_upper_bound = upper_bound_pair.second;

		adj_arr_t order;

		std::function<void(Graph&, adj_arr_t, size_t, size_t)> bb;

		bb = [alloted_time, start, &bb, &best_upper_bound, &best_order, lb]
		(Graph& graph, adj_arr_t order, size_t f, size_t g) mutable
		{
			auto time = std::chrono::steady_clock::now() - start;
			auto time_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(time).count();
			if (time_in_seconds > alloted_time)
			{
				return;
			}

			if (graph.order() < 2 && f < best_upper_bound)
			{
				assert(f == g);
				best_upper_bound = f;
				//std::cout << "found new best upperbound: " << best_upper_bound << std::endl;
				best_order = adj_arr_t(order);
				for (const auto& v : graph)
				{
					best_order.emplace_back(v.first);
				}
			}
			else
			{
				adj_arr_t vertices;
				for (const auto& a : graph)
				{
					if (simplicial(graph, a.first) ||
						(almost_simplicial(graph, a.first) && a.second.size() <= lb))
					{
						vertices.clear();
						vertices.push_back(a.first);
						break;
					}
					else
					{
						vertices.push_back(a.first);
					}
				}

				for (auto v : vertices)
				{
					auto next_graph(graph);
					eliminate(next_graph, v);
					auto next_order(order);
					next_order.emplace_back(v);
					auto next_g = std::max(g, graph.get_neighborhood(v).size());
					auto next_f = std::max(g, lower_bound(next_graph));
					if (next_f < best_upper_bound)
					{
						bb(next_graph, next_order, next_f, next_g);
					}
				}
			}
		};

		if (lb < best_upper_bound)
		{
			bb(graph, order, lb, 0);
		}

		auto time = std::chrono::steady_clock::now() - start;
		auto time_in_seconds = std::chrono::duration_cast<std::chrono::seconds>(time).count();
		//std::cout << "found elimination order with width " << best_upper_bound << " in " << time_in_seconds << " seconds." << std::endl;
		return { best_upper_bound, best_order };
	}

	/**
	* @brief  compute the optimal order
	* @param[in]  std::vector<std::pair<size_t, size_t>>  QuickBB graph
	* @param[in]  size_t   alloted compute time
	* @return  std::pair<size_t, adj_arr_t>  first :  tree width  , second : order
	*/
	static std::pair<size_t, adj_arr_t> compute(const std::vector<std::pair<size_t, size_t>>& vertice_vect, size_t alloted_time)
	{
		Graph graph;
		for (auto iter : vertice_vect)
			graph.add_edge(iter.first, iter.second);

		return compute(graph, alloted_time);
	}

	static void make_clique(Graph& graph, const adj_arr_t& vertices)
	{
		for (auto u : vertices)
		{
			for (auto v : vertices)
			{
				if (u != v)
					graph.add_edge(u, v);
			}
		}
	}

	static bool is_clique(const Graph& graph, const adj_arr_t& vertices)
	{
		for (auto u : vertices)
		{
			for (auto v : vertices)
			{
				if (u != v && !graph.has_edge(u, v))
					return false;
			}
		}
		return true;
	}

	static bool simplicial(const Graph& graph, vertex_index_t vertex)
	{
		return is_clique(graph, graph.get_neighborhood(vertex));
	}

	static bool almost_simplicial(const Graph& graph, vertex_index_t vertex)
	{
		const adj_arr_t& neighbors = graph.get_neighborhood(vertex);

		for (auto v : neighbors)
		{
			adj_arr_t temp(neighbors);
			temp.erase(std::find(temp.begin(), temp.end(), v));
			if (is_clique(graph, temp))
				return true;
		}
		return false;
	}

	static void eliminate(Graph& graph, vertex_index_t vertex)
	{
		make_clique(graph, graph.get_neighborhood(vertex));
		graph.remove_vertex(vertex);
	}

	static size_t count_fillin(const Graph& graph, adj_arr_t vertices)
	{
		size_t count = 0;
		for (auto u : vertices)
		{
			for (auto v : vertices)
			{
				if (u != v)
				{
					auto u_nb = graph.get_neighborhood(u);
					auto found = std::find(std::begin(u_nb), std::end(u_nb), v);
					if (found == std::end(u_nb)) 
						count++;
				}
			}
		}
		return count / 2;
	}

	static std::pair<adj_arr_t, size_t> upper_bound(const Graph& graph)
	{
		Graph graph_copy(graph);
		size_t max_degree(0);
		adj_arr_t ordered_vertices;
		using value_t = decltype(graph_copy.begin())::value_type;

		while (graph_copy.order() > 0)
		{
			auto cmp = [&, graph_copy](const value_t& u, const value_t& v)
			{
				return count_fillin(graph_copy, u.second) < count_fillin(graph_copy, v.second);
			};

			auto u = std::min_element(std::begin(graph_copy), std::end(graph_copy), cmp);
			max_degree = std::max(u->second.size(), max_degree);
			ordered_vertices.emplace_back(u->first);
			eliminate(graph_copy, u->first);
		}
		return { ordered_vertices, max_degree };
	}

	static size_t lower_bound(const Graph& graph)
	{
		Graph graph_copy(graph);
		size_t max_degree(0);
		using value_t = decltype(graph_copy.begin())::value_type;

		while (graph_copy.order() > 0)
		{
			auto u = std::min_element(std::begin(graph_copy), std::end(graph_copy),
				[](const value_t& u, const value_t& v) {return u.second.size() < v.second.size(); });
			
			max_degree = std::max(u->second.size(), max_degree);
			auto neighbors = graph_copy.get_neighborhood(u->first);

			auto cmp = [graph_copy, neighbors](vertex_index_t u, vertex_index_t v)
			{
				auto u_nb = graph_copy.get_neighborhood(u);
				auto v_nb = graph_copy.get_neighborhood(v);
				return u_nb.size() < v_nb.size();
			};

			if (!neighbors.empty())
			{
				auto v = std::min_element(std::begin(neighbors), std::end(neighbors), cmp);
				graph_copy.contract_edge(u->first, *v);
			}
			else
			{
				graph_copy.remove_vertex(u->first);
			}
		}
		return max_degree;
	}
};
#endif //!_QUICKBB_HPP_
