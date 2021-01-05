#include "Core/Utilities/Tools/GetQubitTopology.h"
#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/QProgInfo/QuantumMetadata.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <memory>
#include <cassert>

using namespace std;
USING_QPANDA

#define KMETADATA_GATE_TYPE_COUNT 2
#define DISTANCE_PRECISION 1e-7
#define INF 0x1effff

#define PRINT_TRACE 1
#if PRINT_TRACE
#define PTrace printf
#define PTraceMat(mat) (std::cout << (mat) << endl)
#else
#define PTrace
#define PTraceMat(mat)
#endif

/*******************************************************************
*                      Planarity test algorithm
********************************************************************/
class Graph
{
public:
	Graph(int numberOfVertexes) : m_adj_list(numberOfVertexes), m_edges_cnt(0) {}
	~Graph() {}

	const size_t size() const { return m_adj_list.size(); }
	const size_t EdgesCount() const { return m_edges_cnt; }
	const std::vector<int>& getAdjacentVertexes(size_t v) const { return m_adj_list.at(v); }
	void addEdge(int from, int to) {
		m_adj_list.at(from).push_back(to);
		m_adj_list.at(to).push_back(from);
		++m_edges_cnt;
	}

	void addOrientedEdge(int from, int to) {
		m_adj_list.at(from).push_back(to);
		++m_edges_cnt;
	}

	bool connected(int v1, int v2){
		for (int w : m_adj_list[v1])
		{
			if (w == v2)
			{
				return 1;
			}
		}
		return 0;
	}

private:
	std::vector<std::vector<int>> m_adj_list;
	size_t m_edges_cnt;
};

class PlanarityTest
{
	struct Edge : public std::pair<int, int>
	{
		Edge() : std::pair<int, int>(-1, -1) {}
		Edge(int from, int to) : std::pair<int, int>(from, to) {}

		bool IsEmpty() const { return first == -1 && second == -1; }
		void MakeEmpty() { first = -1; second = -1; }
	};

	struct EdgeHash
	{
		std::size_t operator() (const Edge& e) const
		{
			std::hash<int> intHash;
			return (intHash(e.first) << 4) + intHash(e.second);
		}
	};

	template <typename T>
	using EdgeDict = std::unordered_map<Edge, T, EdgeHash>;

	// Edge pair
	struct Interval : public std::pair<Edge, Edge>
	{
		Interval() = default;
		Interval(Edge low, Edge high) : std::pair<Edge, Edge>(low, high) {}

		Edge& low() { return first; }
		const Edge& low() const { return first; }
		Edge& high() { return second; }
		const Edge& high() const { return second; }
		bool IsEmpty() const { return first.IsEmpty() && second.IsEmpty(); }
	};

	// Interval pair
	struct ConflictPair : public std::pair<Interval, Interval>
	{
		ConflictPair() = default;
		ConflictPair(Interval L, Interval R) : std::pair<Interval, Interval>(L, R) {}

		Interval& L() { return first; }
		const Interval& L() const { return first; }
		Interval& R() { return second; }
		const Interval& R() const { return second; }
		bool IsEmpty() const { return first.IsEmpty() && second.IsEmpty(); }
	};

	class not_planar_exception : public std::exception
	{};

public:
	PlanarityTest(const Graph& g) : m_graph(g) {}

	bool is_planar() {
		const size_t vertexesCount = m_graph.size();
		const size_t _cntEdges = m_graph.EdgesCount();
		// 1. Euler planarity check
		if (vertexesCount > 2 && _cntEdges > (3 * vertexesCount - 6))
			return false;

		// 2. DFS orientation
		std::vector<int> Roots;

		m_height.assign(m_graph.size(), INF);
		m_parent.assign(m_graph.size(), -1);

		for (int v = 0; v < vertexesCount; ++v)
		{
			if (!isReached(v))
			{
				Roots.push_back(v);
				DFS1(v);
			}
		}
		// 3. Testing
		try
		{
			//sort adjacency lists according to non-decreasing nesting depth
			sortAdjacencyList(); // + create new oriented graph 

			for (int i = 0; i < Roots.size(); i++)
			{
				DFS2(Roots[i]);
			}
		}
		catch (not_planar_exception& e)
		{
			return false;
		}

		return true;
	}

protected:
	void doDfs1(int v) {
		//let u be parent in dfs order to v:
		Edge u_v(m_parent[v], v);
		for (const int w : m_graph.getAdjacentVertexes(v))
		{
			// Check if we came from w
			if (m_orientation.find(Edge(w, v)) != m_orientation.end()) continue;

			Edge v_w(v, w);
			m_orientation.insert(v_w);

			// tree edge
			if (m_height[w] == INF)
			{
				m_parent[w] = v;
				m_height[w] = m_height[v] + 1;
				doDfs1(w);
			}
			// back edge
			else
			{
				m_lowpt[v_w] = m_height[w];
			}
			// determine nesting depth
			if (m_lowpt.find(v_w) != m_lowpt.end())
			{
				m_nesting_depth[v_w] = 2 * m_lowpt[v_w];
				auto lp2Result = m_lowpt2.find(v_w);


				// chordal case
				if (lp2Result != m_lowpt2.end() && lp2Result->second < m_height[v])
				{
					m_nesting_depth[v_w] += 1;
				}
			}
			// Update lowpts of parent edge u_v.

			// If not root
			if (u_v.first != -1)
			{
				if (m_lowpt.find(u_v) == m_lowpt.end())
				{
					m_lowpt[u_v] = m_lowpt[v_w];
					if (m_lowpt2.find(v_w) != m_lowpt2.end())
					{
						m_lowpt2[u_v] = m_lowpt2[v_w];
					}
				}
				else
				{
					if (m_lowpt[v_w] < m_lowpt[u_v])
					{
						if (m_lowpt2.find(v_w) != m_lowpt2.end())
						{
							m_lowpt2[u_v] = std::min(m_lowpt[u_v], m_lowpt2[v_w]);
						}
						else
						{
							m_lowpt2[u_v] = m_lowpt[u_v];
						}

						m_lowpt[u_v] = m_lowpt[v_w];
					}
					else if (m_lowpt[v_w] > m_lowpt[u_v])
					{
						if (m_lowpt2.find(u_v) != m_lowpt2.end())
						{
							m_lowpt2[u_v] = std::min(m_lowpt2[u_v], m_lowpt[v_w]);
						}
						else
						{
							m_lowpt2[u_v] = m_lowpt[v_w];
						}
					}
					else
					{
						if (m_lowpt2.find(u_v) != m_lowpt2.end())
						{
							if (m_lowpt2.find(v_w) != m_lowpt2.end())
							{
								m_lowpt2[u_v] = std::min(m_lowpt2[u_v], m_lowpt2[v_w]);
							}
						}
						else if (m_lowpt2.find(v_w) != m_lowpt2.end())
						{
							m_lowpt2[u_v] = m_lowpt2[v_w];
						}
					}
				}
			}
		}
	}

	// Check if was in dfs
	bool isReached(int v) const { return m_height[v] != INF; }

	// Wrapper to set root height in each connection comp to 0
	void DFS1(int v) {
		m_height[v] = 0;
		doDfs1(v);
	}

	void sortAdjacencyList() {
		m_reordered_graph.reset(new Graph(m_graph.size())); // uptr
		vector<vector<pair<int, int>>> newEdges(m_graph.size());
		for (const auto& v : m_nesting_depth)
		{
			const int from = v.first.first;
			const int to = v.first.second;
			const int order = v.second;
			newEdges[from].push_back(make_pair(order, to));
		}
		for (int i = 0; i < newEdges.size(); ++i)
		{
			std::sort(newEdges[i].begin(), newEdges[i].end(), less<std::pair<int, int>>());
			for (const auto& v : newEdges[i])
			{
				m_reordered_graph->addOrientedEdge(i, v.second);
			}
		}
		m_orientation.clear();
	}

	void DFS2(int v) {
		const int u = m_parent[v];
		Edge u_v(u, v);
		for (int i = 0; i < m_reordered_graph->getAdjacentVertexes(v).size(); ++i)
		{
			const int w = m_reordered_graph->getAdjacentVertexes(v)[i];
			Edge v_w(v, w);
			if (!m_S.empty())
			{
				m_stack_bottom[v_w] = m_S.top();
			}
			if (m_parent[w] == v)
			{
				//tree edge
				DFS2(w);
			}
			else
			{
				// back edge
				m_lowpt_edge[v_w] = v_w;
				m_S.push(ConflictPair(Interval(), Interval(v_w, v_w)));
			}
			// integrate new return edges
			if (m_lowpt[v_w] < m_height[v])
			{
				// v_w has return edge
				if (i == 0)
				{
					m_lowpt_edge[u_v] = v_w;
				}
				else
				{
					addEdgeConstraint(v_w);
				}
			}
		}
		// remove back edges returning to parent
		if (u != -1)
		{
			trimBackEdgesEndingAtParent(u);
			// side of u_v is side of a highest return edge
			if (m_lowpt[u_v] < m_height[u] && !m_S.empty())
			{
				/* e has return edge */
				Edge h_L = m_S.top().L().high();
				Edge h_R = m_S.top().R().high();
				if (h_L.first != -1 && (h_R.first != -1 || m_lowpt[h_L] > m_lowpt[h_R]))
				{
					m_ref[u_v] = h_L;
				}
				else
				{
					m_ref[u_v] = h_R;
				}
			}
		}
	}

	void addEdgeConstraint(const Edge& v_w) {
		const int v = v_w.first;
		const Edge u_v(m_parent[v], v);
		ConflictPair P;
		// merge return edges of e into P.R()
		do {
			ConflictPair Q = m_S.top();
			m_S.pop();

			if (!Q.L().IsEmpty())
			{
				std::swap(Q.L(), Q.R());
			}
			if (!Q.L().IsEmpty())
			{
				////HALT//////
				throw not_planar_exception();
			}
			else {
				if (m_lowpt[Q.R().low()] > m_lowpt[u_v])
				{
					// merge intevals
					if (P.R().IsEmpty())
					{
						// topmost interval
						P.R().high() = Q.R().high();
					}
					else
					{
						m_ref[P.R().low()] = Q.R().high();
					}
					P.R().low() = Q.R().low();
				}
				else
				{
					// align
					m_ref[Q.R().low()] = m_lowpt_edge[u_v];
				}
			}
		} while (!m_S.empty() && m_S.top() != m_stack_bottom[v_w]);

		// merge conflicting return edges of e1,...,ei−1 into P.L()
		while (!m_S.empty() && (conflicting(m_S.top().L(), v_w) || conflicting(m_S.top().R(), v_w)))
		{
			ConflictPair Q = m_S.top(); m_S.pop();
			if (conflicting(Q.R(), v_w))
			{
				std::swap(Q.L(), Q.R());
			}
			if (conflicting(Q.R(), v_w))
			{
				////HALT//////
				throw not_planar_exception();
			}
			else
			{
				// merge interval below lowpt(e) into P.R
				m_ref[P.R().low()] = Q.R().high();
				if (!Q.R().low().IsEmpty())
				{
					P.R().low() = Q.R().low();
				}
			}
			if (P.L().IsEmpty())
			{
				// topmost interval
				P.L().high() = Q.L().high();
			}
			else
			{
				m_ref[P.L().low()] = Q.L().high();
			}
			P.L().low() = Q.L().low();
		}
		if (!P.IsEmpty())
		{
			m_S.push(P);
		}

	}

	bool conflicting(const Interval& I, const Edge& b) {
		return (!I.IsEmpty() && m_lowpt[I.high()] > m_lowpt[b]);
	}

	void trimBackEdgesEndingAtParent(int u) {
		//drop entire conflict pairs
		while (!m_S.empty() && lowest(m_S.top()) == m_height[u])
		{
			ConflictPair P = m_S.top(); m_S.pop();
			if (!P.L().low().IsEmpty())
			{
				m_side[P.L().low()] = -1;
			}
		}
		if (!m_S.empty())
		{
			// one more conflict pair to consider
			ConflictPair P = m_S.top(); m_S.pop();
			// trim left interval
			while (!P.L().high().IsEmpty() && P.L().high().second == u)
			{
				P.L().high() = m_ref[P.L().high()];
			}
			if (P.L().high().IsEmpty() && !P.L().low().IsEmpty())
			{
				// just emptied
				m_ref[P.L().low()] = P.R().low();
				m_side[P.L().low()] = -1;
				P.L().low().MakeEmpty();
			}
			// trim right interval
			m_S.push(P);
		}
	}

	int lowest(const ConflictPair& P) {
		if (P.L().IsEmpty())
		{
			return m_lowpt[P.R().low()];
		}
		if (P.R().IsEmpty())
		{
			return m_lowpt[P.L().low()];
		}
		return std::min(m_lowpt[P.L().low()], m_lowpt[P.R().low()]);
	}

private:
	const Graph& m_graph;

	/* orientation phase */
	std::unordered_set<Edge, EdgeHash> m_orientation; /**< Orientation of dfs */
	std::vector<int> m_height;/**< Distance to the root in DFS tree for each vert */
	std::vector<int> m_parent;/**< Parent vertex in dfs orientation */
	EdgeDict<int> m_lowpt;/**< height of lowest return point */
	EdgeDict<int> m_lowpt2;/**< height of next-to-lowest return point (tree edges only) */
	EdgeDict<int> m_nesting_depth;/**< nesting depth, proxy for nesting order given by twice lowpt (plus 1 if chordal) */

	/* testing phase */
	EdgeDict<ConflictPair> m_stack_bottom;/**< top of stack S when traversing the edge(tree edges only) */
	std::stack<ConflictPair> m_S;/**< conflict pairs consisting of current return edges */
	EdgeDict<Edge> m_ref;/**< edge relative to which side is defined */
	EdgeDict<int> m_side;/**< side of edge, or modier for side of reference edge (-1, 1) */
	EdgeDict<Edge> m_lowpt_edge;/**< next back edge in traversal (i.e.with lowest return point) */

	std::unique_ptr<Graph> m_reordered_graph; /**< DFS2 new graph */
};

/*******************************************************************
*                      class GetQubitTopology
********************************************************************/
GetQubitTopology::GetQubitTopology()
{
	init();
}

GetQubitTopology::~GetQubitTopology()
{}

void GetQubitTopology::init(){}

void GetQubitTopology::get_all_double_gate_qubits(QProg prog)
{
	LayeredTopoSeq layer_info = prog_layer(prog);
	std::vector<QubitPair> last_layer_qubit_pairs; // pre-layer
	for (const auto layer : layer_info)
	{
		std::vector<QubitPair> cur_layer_qubit_pairs;
		for (const auto node : layer)
		{
			auto cur_node_qubits = node.first->m_target_qubits;
			cur_node_qubits += node.first->m_control_qubits;
			for (size_t i = 0; i < cur_node_qubits.size() - 1; ++i)
			{
				for (size_t j = i + 1; j < cur_node_qubits.size(); ++j)
				{
					QubitPair tmp_pair;
					const size_t first_qubit = cur_node_qubits[i]->get_phy_addr();
					const size_t second_qubit = cur_node_qubits[j]->get_phy_addr();
					if (first_qubit < second_qubit)
					{
						tmp_pair = std::make_pair(first_qubit, second_qubit);
					}
					else if (first_qubit > second_qubit)
					{
						tmp_pair = std::make_pair(second_qubit, first_qubit);
					}
					else
					{
						QCERR_AND_THROW(runtime_error, "Error: qubits error on double gate.");
					}

					cur_layer_qubit_pairs.push_back(tmp_pair);

					auto item = m_double_gate_qubits.find(tmp_pair);
					if (m_double_gate_qubits.end() == item)
					{
						m_double_gate_qubits.insert(std::make_pair(tmp_pair, 1));
					}
					else
					{
						item->second += 1;
						for (const auto last_layer_qubit_pair : last_layer_qubit_pairs)
						{
							if (last_layer_qubit_pair == tmp_pair)
							{
								item->second -= 1;
								break;
							}
						}
					}
				}
			}
		}

		last_layer_qubit_pairs.swap(cur_layer_qubit_pairs);
	}
}

const TopologyData& GetQubitTopology::get_src_adjaccent_matrix(QProg prog)
{
	std::vector<int> qubits;
	get_all_used_qubits(prog, qubits);
	const size_t qubit_num = qubits.size();

	std::map<size_t, size_t> qubit_map;
	for (size_t i = 0; i < qubit_num; ++i)
	{
		qubit_map.insert(std::make_pair(qubits[i], i));
	}

	get_all_double_gate_qubits(prog);

	m_topo_data.resize(qubit_num, std::vector<int>(qubit_num, 0));
	for (const auto& qubit_pair_item : m_double_gate_qubits)
	{
		const auto& qubit_pair = qubit_pair_item.first;
		m_topo_data[qubit_map[qubit_pair.first]][qubit_map[qubit_pair.second]] += qubit_pair_item.second;
		m_topo_data[qubit_map[qubit_pair.second]][qubit_map[qubit_pair.first]] += qubit_pair_item.second;
	}

	return m_topo_data;
}

/*******************************************************************
*                     Optimization algorithm interface 
********************************************************************/
/**
* @brief Gets the total weight of all the edge
*/
static size_t get_total_weights(const TopologyData& topo_data, size_t& valid_edge_cnt)
{
	const size_t qubit_num = topo_data.size();
	size_t all_edges_weight = 0;
	valid_edge_cnt = 0;
	for (size_t i = 0; i < qubit_num; ++i)
	{
		for (size_t j = i; j < qubit_num; ++j)
		{
			all_edges_weight += topo_data[i][j];
			if (0 != topo_data[i][j])
			{
				++valid_edge_cnt;
			}
		}
	}

	return all_edges_weight;
}

static std::vector<size_t> get_qubits_weight(const TopologyData& topo_data)
{
	std::vector<size_t> weight_vec(topo_data.size());
	for (size_t i = 0; i < topo_data.size(); ++i)
	{
		size_t m = 0;
		for (auto j : topo_data.at(i))
		{
			m += j;
		}

		weight_vec.at(i) = m;
	}

	return weight_vec;
}

static std::vector<size_t> get_qubits_connectivity_degree(TopologyData& topo_data)
{
	std::vector<size_t> connectivity_degree_vec(topo_data.size());
	for (size_t i = 0; i < topo_data.size(); ++i)
	{
		size_t m = 0;
		for (const auto j : topo_data.at(i))
		{
			if (0 != j)
			{
				++m;
			}
		}

		connectivity_degree_vec.at(i) = m;
	}

	return connectivity_degree_vec;
}

static double get_variance(std::vector<int>& data)
{
	const size_t n = data.size();

	size_t sum = 0;
	for (const auto& i : data)
	{
		sum += i;
	}

	const double average_value = (double)sum / (double)n;
	double tmp_val = 0.0;
	for (const auto& i : data)
	{
		tmp_val += pow((i - average_value), 2);
	}

	return tmp_val / (double)n;
}

static std::vector<double> get_qubits_dispersion_degree(TopologyData& topo_data)
{
	std::vector<double> dispersion_degree_vec(topo_data.size());
	for (size_t i = 0; i < topo_data.size(); ++i)
	{
		std::vector<int> weight_edge;
		for (const auto j : topo_data.at(i))
		{
			if (0 != j)
			{
				weight_edge.push_back(j);
			}
		}

		dispersion_degree_vec.at(i) = get_variance(weight_edge);
	}

	return dispersion_degree_vec;
}

static size_t get_intermediary_point_num(const std::vector<int>& points, const size_t max_connect_degree)
{
	size_t points_num = points.size();
	if (points_num <= (max_connect_degree + 1))
	{
		return 1;
	}
	else if (points_num <= (2 * max_connect_degree))
	{
		return 2;
	}

	return ceil(((float)points_num) / ((float)max_connect_degree - 1.0));
}

template <typename T>
std::vector<int> get_candidate_points(std::vector<int>& intermediary_points, T& factor_sort_vec)
{
	std::vector<int> candidate_points;
	candidate_points.push_back(intermediary_points.back());
	intermediary_points.pop_back();
	for (size_t i = intermediary_points.size(); i > 0; --i)
	{
		if ((factor_sort_vec[candidate_points[0]].second == factor_sort_vec[intermediary_points.back()].second))
		{
			candidate_points.push_back(intermediary_points.back());
			intermediary_points.pop_back();
		}
		else
		{
			break;
		}
	}

	return candidate_points;
}

static std::vector<int> check_connectivity_degree(const std::vector<int>& points, std::vector<int>& intermediary_points,
	const size_t intermediary_point_num, const std::vector<size_t>& connectivity_degree)
{
	std::vector<std::pair<int, size_t>> connectivity_degree_sort_vec;
	for (const auto& i : points)
	{
		connectivity_degree_sort_vec.push_back(std::make_pair(i, connectivity_degree[i]));
	}
	sort(connectivity_degree_sort_vec.begin(), connectivity_degree_sort_vec.end(),
		[](std::pair<int, size_t>& a, std::pair<int, size_t>& b) {return a.second > b.second; });

	const size_t need_candidate_point_num = intermediary_point_num - intermediary_points.size();
	for (size_t i = 0; i < connectivity_degree_sort_vec.size(); ++i)
	{
		if (i < need_candidate_point_num)
		{
			intermediary_points.push_back(connectivity_degree_sort_vec[i].first);
			continue;
		}

		if (connectivity_degree_sort_vec[i - 1].second == connectivity_degree_sort_vec[i].second)
		{
			intermediary_points.push_back(connectivity_degree_sort_vec[i].first);
		}
		else
		{
			break;
		}
	}

	// candidate points
	if (intermediary_points.size() == intermediary_point_num)
	{
		return std::vector<int>();
	}

	return get_candidate_points(intermediary_points, connectivity_degree_sort_vec);
}

static std::vector<int> check_weight(const std::vector<int>& points, std::vector<int>& intermediary_points,
	const size_t intermediary_point_num, const std::vector<size_t>& weight)
{
	std::vector<std::pair<int, size_t>> weight_sort_vec;
	for (size_t i = 0; i < points.size(); ++i)
	{
		const auto& tmp_qubit = points[i];
		weight_sort_vec.push_back(std::make_pair(tmp_qubit, weight[tmp_qubit]));
	}

	sort(weight_sort_vec.begin(), weight_sort_vec.end(), [](std::pair<int, size_t>& a, std::pair<int, size_t>& b) {return a.second > b.second; });

	const size_t need_candidate_point_num = intermediary_point_num - intermediary_points.size();
	for (size_t i = 0; i < points.size(); ++i)
	{
		if (i < need_candidate_point_num)
		{
			intermediary_points.push_back(weight_sort_vec[i].first);
			continue;
		}

		if (weight_sort_vec[i - 1].second == weight_sort_vec[i].second)
		{
			intermediary_points.push_back(weight_sort_vec[i].first);
		}
		else
		{
			break;
		}
	}

	// candidate points
	if (intermediary_points.size() == intermediary_point_num)
	{
		return std::vector<int>();
	}

	return get_candidate_points(intermediary_points, weight_sort_vec);
}

static std::vector<int> check_dispersion_degree(const std::vector<int>& points, std::vector<int>& intermediary_points,
	const size_t intermediary_point_num, const std::vector<double>& dispersion_degree)
{
	std::vector<std::pair<int, double>> dispersion_degree_sort_vec;
	for (size_t i = 0; i < points.size(); ++i)
	{
		const auto tmp_qubit = points[i];
		dispersion_degree_sort_vec.push_back(std::make_pair(tmp_qubit, dispersion_degree[tmp_qubit]));
	}
	sort(dispersion_degree_sort_vec.begin(), dispersion_degree_sort_vec.end(),
		[](std::pair<int, double>& a, std::pair<int, double>& b) {return a.second > b.second; });

	const auto need_candidate_point_num = intermediary_point_num - intermediary_points.size();
	for (size_t i = 0; i < points.size(); ++i)
	{
		if (i < need_candidate_point_num)
		{
			intermediary_points.push_back(dispersion_degree_sort_vec[i].first);
			continue;
		}

		/*if (dispersion_degree_sort_vec[i - 1].second == dispersion_degree_sort_vec[i].second)
		{
			intermediary_points.push_back(dispersion_degree_sort_vec[i].first);
		}
		else
		{
			break;
		}*/
	}

	// candidate points
	if (intermediary_points.size() == intermediary_point_num)
	{
		return std::vector<int>();
	}

	return get_candidate_points(intermediary_points, dispersion_degree_sort_vec);
}

/**
* @brief We calculate these three indicators for each vertex and then sort them in lexicographic order in order of degree,
		 total weight, and dispersion
*/
static std::vector<int> get_intermediary_points(std::vector<int>& points, size_t max_connect_degree, std::vector<size_t>& weight,
	std::vector<size_t>& connectivity_degree, std::vector<double>& dispersion_degree)
{
	if (points.size() == 0)
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to get intermediary point, the input points is empty.");
	}
	else if (points.size() == 1)
	{
		return points;
	}

	const auto intermediary_point_num = get_intermediary_point_num(points, max_connect_degree);
	std::vector<int> intermediary_points;

	//check weight
	std::vector<int> candidate_points = check_weight(points, intermediary_points, intermediary_point_num, weight);
	if (candidate_points.size() == 0)
	{
		return intermediary_points;
	}

	//check connectivity_degree
	candidate_points = check_connectivity_degree(candidate_points, intermediary_points, intermediary_point_num, connectivity_degree);
	if (candidate_points.size() == 0)
	{
		return intermediary_points;
	}

	//check dispersion_degree
	candidate_points = check_dispersion_degree(candidate_points, intermediary_points, intermediary_point_num, dispersion_degree);

	if (intermediary_points.size() == intermediary_point_num)
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to get intermediary point, algorithm error.");
	}

	return intermediary_points;
}

static int get_intermediary_points(std::vector<int>& points, std::vector<size_t>& weight,
	std::vector<size_t>& connectivity_degree, std::vector<double>& dispersion_degree,
	const double lamda1, const double lamda2, const double lamda3)
{
	if (points.size() == 0)
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to get intermediary point, the input points is empty.");
	}
	else if (points.size() == 1)
	{
		return points[0];
	}

	std::vector<std::pair<int, double>> qubits_score;
	double score = 0.0;
	for (const auto qubit : points)
	{
		score = (double)(weight[qubit]) * lamda1 + (double)(connectivity_degree[qubit]) * lamda2 + (dispersion_degree[qubit]) * lamda3;
		qubits_score.push_back(std::make_pair(qubit, score));
	}

	sort(qubits_score.begin(), qubits_score.end(), [](std::pair<int, double>& a, std::pair<int, double>& b) {return a.second > b.second; });

	return qubits_score.front().first;
}

static bool is_intermediary_point(const int point, const std::vector<int>& intermediary_points)
{
	for (const auto& m : intermediary_points)
	{
		if ((m == point))
		{
			return true;
		}
	}

	return false;
}

/**
* @brief Check whether the degree of the non intermediate point violates the constraint
*/
static void check_non_intermediary_points(TopologyData& topo_data, const std::vector<int>& intermediary_points, const size_t max_connect_degree)
{
	const size_t topo_size = topo_data.size();
	for (int i = 0; i < topo_data.size(); ++i)
	{
		if (is_intermediary_point(i, intermediary_points))
		{
			continue;
		}

		std::vector<std::pair<int, size_t>> tmp_connectivity;
		for (size_t j = 0; j < topo_data[i].size(); ++j)
		{
			if (is_intermediary_point(j, intermediary_points))
			{
				continue;
			}
			else if (0 != topo_data[i][j])
			{
				tmp_connectivity.push_back(std::make_pair(j, topo_data[i][j]));
			}
		}

		if (tmp_connectivity.size() > max_connect_degree)
		{
			sort(tmp_connectivity.begin(), tmp_connectivity.end(), [](std::pair<int, size_t>& a, std::pair<int, size_t>& b) {return a.second > b.second; });
			while (tmp_connectivity.size() > max_connect_degree)
			{
				topo_data[i][tmp_connectivity.back().first] = 0;
				topo_data[tmp_connectivity.back().first][i] = 0;
				tmp_connectivity.pop_back();
			}
		}
	}
}

static int get_intermediary_point(std::vector<int>& points, std::vector<size_t>& weight,
	std::vector<size_t>& connectivity_degree, std::vector<double>& dispersion_degree)
{
	if (points.size() == 0)
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to get intermediary point, the input points is empty.");
	}
	else if (points.size() == 1)
	{
		return points[0];
	}

	//check weight
	std::vector<std::pair<int, size_t>> tmp_vec;
	for (const auto& i : points)
	{
		tmp_vec.push_back(std::make_pair(i, weight[i]));
	}
	sort(tmp_vec.begin(), tmp_vec.end(), [](std::pair<int, size_t>& a, std::pair<int, size_t>& b) {return a.second > b.second; });

	if (tmp_vec[0].second > tmp_vec[1].second)
	{
		return tmp_vec[0].first;
	}

	//check connectivity_degree
	int tmp_qubit = tmp_vec[0].first;
	std::vector<std::pair<int, size_t>> tmp_vec2{std::make_pair(tmp_qubit, connectivity_degree[tmp_qubit])};
	for (size_t i = 1; i < tmp_vec.size(); ++i)
	{
		if (tmp_vec[i-1].second == tmp_vec[i].second)
		{
			tmp_qubit = tmp_vec[i].first;
			tmp_vec2.push_back(std::make_pair(tmp_qubit, connectivity_degree[tmp_qubit]));
		}
		else
		{
			break;
		}
	}

	sort(tmp_vec2.begin(), tmp_vec2.end(), [](std::pair<int, size_t>& a, std::pair<int, size_t>& b) {return a.second > b.second; });
	if (tmp_vec2[0].second > tmp_vec2[1].second)
	{
		return tmp_vec2[0].first;
	}

	//check dispersion_degree
	tmp_qubit = tmp_vec2[0].first;
	std::vector<std::pair<int, double>> tmp_vec3{ std::make_pair(tmp_qubit, dispersion_degree[tmp_qubit]) };
	for (size_t i = 1; i < tmp_vec2.size(); ++i)
	{
		if (tmp_vec2[i - 1].second == tmp_vec2[i].second)
		{
			tmp_qubit = tmp_vec2[i].first;
			tmp_vec3.push_back(std::make_pair(tmp_qubit, dispersion_degree[tmp_qubit]));
		}
		else
		{
			break;
		}
	}
	sort(tmp_vec3.begin(), tmp_vec3.end(), [](std::pair<int, double>& a, std::pair<int, double>& b) {return a.second > b.second; });
	if (tmp_vec3[0].second > tmp_vec3[1].second)
	{
		return tmp_vec3[0].first;
	}
	else
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to get intermediary point, algorithm error.");
	}

	return -1;
}

template <typename T>
void to_symmetric_matrix(MatData<T>& mat_data) {
	const size_t mat_size = mat_data.size();
	for (size_t i = 0; i < mat_size; ++i)
	{
		for (size_t j = 0; j < mat_size; ++j)
		{
			if (mat_data[i][j] != mat_data[j][i])
			{
				if ((mat_data[i][j] - 0.0) < 0.000001)
				{
					mat_data[i][j] = mat_data[j][i];
				}
				else
				{
					mat_data[j][i] = mat_data[i][j];
				}
			}
		}
	}
}

static void build_topology(TopologyData& topo, int split_method)
{
	const size_t topo_size = topo.size();
	switch (split_method)
	{
	case 0: /* 环形 */
	{
		for (size_t i = 0; i < topo_size - 1; ++i)
		{
			topo[i].resize(topo_size, 0);
			topo[i][i + 1] = 1;
		}
		topo[topo_size - 1].resize(topo_size, 0);
		topo[topo_size - 1][0] = 1;
	}
	break;

	case 1: /* 环形全连接 */
	{
		for (size_t i = 0; i < topo_size; ++i)
		{
			topo[i].resize(topo_size, 0);
			for (size_t j = 0; j < topo_size; ++j)
			{
				if (i != j)
				{
					topo[i][j] = 1;
				}
			}
		}
	}
	break;

	case 2: /* 星型结构 */

		break;

	default:
		QCERR_AND_THROW(runtime_error, "Error: failed to build topology structure, the splitting method is error.");
		break;
	}
}

static TopologyData extension_single_point(const int complex_point, const int need_increase_qubit, int split_method)
{
	TopologyData ret_topo;
	switch (need_increase_qubit)
	{
	case 0:
		QCERR_AND_THROW(runtime_error, "Error: failed to extension point, the increase qubit's number cann't be zero.");
		break;

	case 1:
	{
		ret_topo.resize(2);
		ret_topo[0] = { 0,1 };
		ret_topo[1] = { 1,0 };
	}
	break;

	case 2:
	{
		ret_topo.resize(3);
		build_topology(ret_topo, split_method);
	}
	break;

	case 3:
	{
		ret_topo.resize(4);
		build_topology(ret_topo, split_method);
	}
	break;

	case 4:
	{
		ret_topo.resize(5);
		build_topology(ret_topo, split_method);
	}
	break;

	case 5:
	{
		ret_topo.resize(6);
		build_topology(ret_topo, split_method);
	}
	break;

	default:
		QCERR_AND_THROW(runtime_error, "Error: failed to extension point, the increase qubit's number error.");
		break;
	}

	return ret_topo;
}

static TopologyData extension_single_point2(const int complex_point, const size_t connectivity,
	const size_t max_connect_degree, int split_method)
{
	if (connectivity < max_connect_degree)
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to extension point, connectivity error.");
	}

	TopologyData ret_topo;
	if (((max_connect_degree - 1) * 2) >= connectivity)
	{
		ret_topo.resize(2);
		ret_topo[0] = { 0,1 };
		ret_topo[1] = { 1,0 };
		return ret_topo;
	}

	switch (split_method)
	{
	case LINEAR:
	{
		//linear structure
		/** (MAX_CONNECTIVITY - 1)*2 + (MAX_CONNECTIVITY - 2)*x = connectivity
		*/
		int x = ceil((connectivity - ((max_connect_degree - 1) * 2)) / (max_connect_degree - 2));
		x += 2;
		ret_topo.resize(x);
		for (size_t i = 0; i < (x - 1); ++i)
		{
			ret_topo[i].resize(x, 0);
			ret_topo[i][i + 1] = 1;
		}
		ret_topo[x - 1].resize(x, 0);
	}
	break;

	case RING:
	{
		//ring architecture
		/** (MAX_CONNECTIVITY - 2)*x = connectivity
		*/
		int x = ceil(connectivity / (max_connect_degree - 2));
		ret_topo.resize(x);
		for (size_t i = 0; i < (x - 1); ++i)
		{
			ret_topo[i].resize(x, 0);
			ret_topo[i][i + 1] = 1;
		}
		ret_topo[x - 1].resize(x, 0);
		ret_topo[x - 1][0] = 1;
	}
	break;

	//case RING_FULL_CONNECT:
	//	//Ring full connection
	//	break;

	//case STAR:
	//{
	//	//star topology
	//}
	//break;

	default:
		QCERR_AND_THROW(runtime_error, "Error: failed to build topology structure, the splitting method is error.");
		break;
	}

	return ret_topo;
}

static std::vector<std::pair<int, size_t>> sort_sub_graph_node(const TopologyData& sub_topo_data)
{
	const size_t sub_topo_node_num = sub_topo_data.size();
	std::vector<std::pair<int, size_t>> qubits;

	//sort by weight
	std::map<int, size_t> node_connectivity;
	std::vector<std::pair<int, size_t>> weight_vec(sub_topo_node_num);
	for (int i = 0; i < sub_topo_data.size(); ++i)
	{
		size_t m = 0;
		size_t tmp_connectivity = 0;
		for (const auto j : sub_topo_data.at(i))
		{
			if (j != 0)
			{
				m += j;
				++tmp_connectivity;
			}
		}
		node_connectivity.insert(std::make_pair(i, tmp_connectivity));
		weight_vec.at(i) = std::make_pair(i, m);
	}

	sort(weight_vec.begin(), weight_vec.end(), [](std::pair<int, size_t>& a, std::pair<int, size_t>& b) {return a.second > b.second; });

	qubits.push_back(std::make_pair(weight_vec.front().first, node_connectivity.at(weight_vec.front().first)));
	weight_vec.erase(weight_vec.begin());
	while (qubits.size() < sub_topo_node_num)
	{
		if (weight_vec.size() == 1)
		{
			qubits.push_back(std::make_pair(weight_vec.front().first, node_connectivity.at(weight_vec.front().first)));
			weight_vec.clear();
			break;
		}

		//get distance for every qubit
		std::vector<std::pair<int, double>> qubit_distance;
		for (const auto& tmp_qubit : weight_vec)
		{
			double distance = 1.0;
			for (const auto already_qubit : qubits)
			{
				if (sub_topo_data[tmp_qubit.first][already_qubit.first] != 0)
				{
					distance /= 2.0;
				}
			}

			qubit_distance.push_back(std::make_pair(tmp_qubit.first, distance));
		}

		sort(qubit_distance.begin(), qubit_distance.end(), [](std::pair<int, double>& a, std::pair<int, double>& b) {return a.second < b.second; });

		if ((qubit_distance.at(0).second - qubit_distance.at(1).second < DISTANCE_PRECISION))
		{
			//get all the same distance qubits
			std::vector<std::pair<int, double>> qubit_same_distance;
			qubit_same_distance.push_back(qubit_distance.at(0));
			qubit_same_distance.push_back(qubit_distance.at(1));
			for (size_t i = 2; i < qubit_distance.size(); ++i)
			{
				if ((qubit_distance.at(0).second - qubit_distance.at(i).second < DISTANCE_PRECISION))
				{
					qubit_same_distance.push_back(qubit_distance.at(i));
				}
			}

			//find the qubit with the largest weight in the same distance
			for (auto itr = weight_vec.begin(); itr != weight_vec.end(); ++itr)
			{
				bool b_find_node = false;
				for (auto itr_same_distance = qubit_same_distance.begin(); itr_same_distance != qubit_same_distance.end(); ++itr_same_distance)
				{
					if (itr->first == itr_same_distance->first)
					{
						qubits.push_back(std::make_pair(itr->first, node_connectivity.at(itr->first)));
						weight_vec.erase(itr);
						b_find_node = true;
						break;
					}
				}
				if (b_find_node)
				{
					break;
				}
			}
		}
		else
		{
			qubits.push_back(std::make_pair(qubit_distance.front().first, node_connectivity.at(qubit_distance.front().first)));
			for (auto itr = weight_vec.begin(); itr != weight_vec.end(); ++itr)
			{
				if (itr->first == qubit_distance.front().first)
				{
					weight_vec.erase(itr);
					break;
				}
			}
		}
	}

	return qubits;
}

static std::vector<std::pair<int, size_t>> get_all_adjacent_node(const int target_qubit, const TopologyData& topo_mat)
{
	if (target_qubit >= topo_mat.size())
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to get adjacent node, the target qubit is error.");
	}

	std::vector<std::pair<int, size_t>> ret_data;
	for (int i = 0; i < topo_mat[target_qubit].size(); ++i)
	{
		if (topo_mat[target_qubit][i] != 0)
		{
			ret_data.push_back(std::make_pair(i, (size_t)(topo_mat[target_qubit][i])));
		}
	}

	return ret_data;
}

static std::map<std::pair<int, int>, std::pair<int, int>> dispatch_node(const int target_qubit, const size_t max_connect_degree,
	std::vector<std::pair<int, size_t>>& src_adjacent_node, std::vector<std::pair<int, size_t>>& sub_graph_node)
{
#if PRINT_TRACE
	PTrace("src_adjacent_node\n");
	for (auto& item : src_adjacent_node)
	{
		PTrace("(%d, %d)\n", item.first, item.second);
	}
	PTrace("-------src_adjacent_node end----------\n");
#endif // PRINT_TRACE

	std::map<std::pair<int, int>, std::pair<int, int>> new_edge_map;
	size_t sub_graph_node_index = 0;
	size_t dispatch_cnt = 0;
	for (const auto& adjacent_node : src_adjacent_node)
	{
		if ((max_connect_degree - sub_graph_node[sub_graph_node_index].second) <= dispatch_cnt)
		{
			++sub_graph_node_index;
			dispatch_cnt = 0;
		}

		std::pair<int, int> src_pair(target_qubit, adjacent_node.first);
		std::pair<int, int> dst_pair(sub_graph_node[sub_graph_node_index].first, adjacent_node.first);
		new_edge_map.insert(std::make_pair(src_pair, dst_pair));
		++dispatch_cnt;
	}

	return new_edge_map;
}

static double get_distance(const int qubit_1, const int qubit_2, const TopologyData& topo_data)
{
	const size_t topo_size = topo_data.size();
	if ((topo_size <= qubit_1) || (topo_size <= qubit_2))
	{
		QCERR_AND_THROW(runtime_error, "Error: failed to get the distance between the target qubits, qubit index error.");
	}

	double distance = 0.0;
	std::vector<int> adjacent_qubits = { qubit_1 };
	while (adjacent_qubits.size() > 0)
	{
		++distance;
		std::vector<int> tmp_adjacent_qubits;
		for (const auto qubit : adjacent_qubits)
		{
			if (0 != topo_data[qubit][qubit_2])
			{
				return distance;
			}

			for (size_t i = 0; i < topo_data[qubit].size(); ++i)
			{
				if (0 != topo_data[qubit][i])
				{
					tmp_adjacent_qubits.push_back(i);
				}
			}
		}

		adjacent_qubits.swap(tmp_adjacent_qubits);
	}

	QCERR_AND_THROW(runtime_error, "Error: failed to get the distance between the target qubits, unknow error.");
	return -1.0;
}

static MatData<double> get_diatance_matrix(const TopologyData& src_topo_data)
{
	const size_t topo_size = src_topo_data.size();
	MatData<double> ret_mat(topo_size, std::vector<double>(topo_size, 0.0));
	for (size_t i = 0; i < topo_size; ++i)
	{
		for (size_t j = i +1; j < topo_size; ++j)
		{
			ret_mat[i][j] = get_distance(i, j, src_topo_data);
		}
	}

	to_symmetric_matrix(ret_mat);

	return ret_mat;
}

static double calc_relative_weight(const int qubit_1, const int qubit_2, const std::vector<size_t>& qubit_weight_vec)
{
	return abs((double)(qubit_weight_vec[qubit_1] - qubit_weight_vec[qubit_2] + 1)/(double)(qubit_weight_vec[qubit_1] - qubit_weight_vec[qubit_2]));
}

static MatData<double> get_relative_weight_matrix(const TopologyData& src_topo_data)
{
	const size_t topo_size = src_topo_data.size();
	MatData<double> ret_mat(topo_size, std::vector<double>(topo_size, 0.0));
	std::vector<size_t> qubit_weight_vec = get_qubits_weight(src_topo_data);
	for (size_t i = 0; i < topo_size; ++i)
	{
		for (size_t j = i + 1; j < topo_size; ++j)
		{
			ret_mat[i][j] = calc_relative_weight(i, j, qubit_weight_vec);
		}
	}

	to_symmetric_matrix(ret_mat);

	return ret_mat;
}

/*******************************************************************
*                      public interface
********************************************************************/
TopologyData QPanda::get_double_gate_block_topology(QProg prog)
{
	GetQubitTopology get_topology;
	return get_topology.get_src_adjaccent_matrix(prog);
}

std::vector<int> QPanda::get_sub_graph(const TopologyData& topo_data)
{
	const size_t qubit_num = topo_data.size();
	std::vector<int> ret_vec(qubit_num, 0);

	//test val
	/*for (size_t i = 2; i < qubit_num; ++i)
	{
		ret_vec[i] = 1;
	}*/

	return ret_vec;
}

void QPanda::del_weak_edge(TopologyData& topo_data)
{
	//get the import nodes
	const size_t qubit_num = topo_data.size();
	const size_t import_nodes_num = 1;
	std::vector<size_t> weight_vec = get_qubits_weight(topo_data);
	std::vector<std::pair<size_t, size_t>> weight_map;
	for (size_t i = 0; i < weight_vec.size(); ++i)
	{
		weight_map.push_back(std::make_pair(i, weight_vec[i]));
	}
	sort(weight_map.begin(), weight_map.end(), [](std::pair<size_t, size_t>& a, std::pair<size_t, size_t>& b) {return a.second > b.second; });
	std::vector<size_t> improt_nodes;
	for (size_t i = 0; i < import_nodes_num; ++i)
	{
		improt_nodes.push_back(weight_map.at(i).first);
	}

	//get all the edge weight
	size_t valid_edge_cnt = 0;
	const size_t all_edges_weight = get_total_weights(topo_data, valid_edge_cnt);

	//get average value
	double average_weight = ((double)all_edges_weight) / valid_edge_cnt;

	//Remove edges less than average
	for (size_t i = 0; i < qubit_num; ++i)
	{
		for (size_t j = 0; j < qubit_num; ++j)
		{
			if (topo_data[i][j] <= average_weight)
			{
				{
					topo_data[i][j] = 0;
				}
			}
		}
	}
}

std::vector<int> QPanda::del_weak_edge(TopologyData& topo_data, const size_t max_connect_degree,
	std::vector<int>& sub_graph_set, std::vector<weight_edge>& candidate_edges)
{
	//sub graph to map
	std::map<int, std::vector<int>> sub_graph_qubit_map;//key:聚团id，val:聚团包含的节点id
	for (int i = 0; i < sub_graph_set.size(); ++i)
	{
		const int sub_graph_index = sub_graph_set[i];
		auto itr = sub_graph_qubit_map.find(sub_graph_index);
		if (sub_graph_qubit_map.end() == itr)
		{
			sub_graph_qubit_map.insert(std::make_pair(sub_graph_index, std::vector<int>{i}));
		}
		else
		{
			itr->second.push_back(i);
		}
	}

	//get intermediary points
	std::vector<int> intermediary_points;
	auto weight_vec = get_qubits_weight(topo_data);
	auto connectivity_vec = get_qubits_connectivity_degree(topo_data);
	auto dispersion_degrees = get_qubits_dispersion_degree(topo_data);
	for (auto& item : sub_graph_qubit_map)
	{
		auto tmp_intermediary_points = get_intermediary_points(item.second, max_connect_degree,
			weight_vec, connectivity_vec, dispersion_degrees);
		if (tmp_intermediary_points.size() == 0)
		{
			QCERR_AND_THROW(runtime_error, "Error: failed to get intermediary point");
		}

		intermediary_points.insert(intermediary_points.end(), tmp_intermediary_points.begin(), tmp_intermediary_points.end());
	}

	//del weak edges
	size_t valid_edge_cnt = 0;
	const size_t all_edges_weight = get_total_weights(topo_data, valid_edge_cnt);
	const double average_weight = ((double)all_edges_weight) / (double)valid_edge_cnt;
	std::vector<weight_edge> all_weight_edges;
	for (int i = 0; i < topo_data.size(); ++i)
	{
		for (int j = i; j < topo_data[i].size(); ++j)
		{
			if (0 < topo_data[i][j])
			{
				all_weight_edges.push_back(weight_edge(topo_data[i][j], std::vector<int>({ i, j })));
			}
		}
	}

	sort(all_weight_edges.begin(), all_weight_edges.end(), [](const weight_edge& a, const weight_edge& b) {
		return a.first < b.first;
	});

	auto tmp_connectivity_vec = connectivity_vec;
	for (auto edge_itr = all_weight_edges.begin(); edge_itr != all_weight_edges.end(); ++edge_itr)
	{
		if ((edge_itr->first <= average_weight) &&
			(1 < tmp_connectivity_vec[edge_itr->second.front()]) &&
			(1 < tmp_connectivity_vec[edge_itr->second.back()]))
		{
			const auto& point_pair = edge_itr->second;
			bool b_intermediary_point = false;
			for (const auto& m : intermediary_points)
			{
				if ((m == point_pair.front()) || (m == point_pair.back()))
				{
					b_intermediary_point = true;
					break;
				}
			}

			if (!b_intermediary_point)
			{
				candidate_edges.push_back(weight_edge(edge_itr->first, point_pair));
				tmp_connectivity_vec[point_pair.front()] -= 1;
				tmp_connectivity_vec[point_pair.back()] -= 1;
				topo_data[point_pair.front()][point_pair.back()] = 0;
				topo_data[point_pair.back()][point_pair.front()] = 0;
			}

		}
		else
		{
			break;
		}
	}

	check_non_intermediary_points(topo_data, intermediary_points, max_connect_degree);

	return intermediary_points;
}

std::vector<int> QPanda::del_weak_edge(TopologyData& topo_data, std::vector<int>& sub_graph_set, const size_t max_connect_degree,
	const double lamda1, const double lamda2, const double lamda3)
{
	//sub graph to map
	std::map<int, std::vector<int>> sub_graph_qubit_map;
	for (int i = 0; i < sub_graph_set.size(); ++i)
	{
		const int sub_graph_index = sub_graph_set[i];
		auto itr = sub_graph_qubit_map.find(sub_graph_index);
		if (sub_graph_qubit_map.end() == itr)
		{
			sub_graph_qubit_map.insert(std::make_pair(sub_graph_index, std::vector<int>{i}));
		}
		else
		{
			itr->second.push_back(i);
		}
	}

	//get intermediary points
	std::vector<int> intermediary_points;
	auto weight_vec = get_qubits_weight(topo_data);
	auto connectivity_vec = get_qubits_connectivity_degree(topo_data);
	auto dispersion_degrees = get_qubits_dispersion_degree(topo_data);
	for (auto& item : sub_graph_qubit_map)
	{
		int p = get_intermediary_points(item.second, weight_vec, connectivity_vec, dispersion_degrees, lamda1, lamda2, lamda3);
		if (p < 0)
		{
			QCERR_AND_THROW(runtime_error, "Error: failed to get intermediary point");
		}

		intermediary_points.push_back(p);
	}

	size_t valid_edge_cnt = 0;
	const size_t all_edges_weight = get_total_weights(topo_data, valid_edge_cnt);
	const double average_weight = ((double)all_edges_weight) / (double)valid_edge_cnt;

	//del weak edges
	for (int i = 0; i < topo_data.size(); ++i)
	{
		for (size_t j = 0; j < topo_data[i].size(); ++j)
		{
			if (average_weight < topo_data[i][j])
			{
				continue;
			}

			bool b_intermediary_point = false;
			for (const auto& m : intermediary_points)
			{
				if ((m == i) || (m == j))
				{
					b_intermediary_point = true;
					break;
				}
			}

			if (b_intermediary_point)
			{
				continue;
			}
			else
			{
				topo_data[i][j] = 0;
			}
		}
	}

	check_non_intermediary_points(topo_data, intermediary_points, max_connect_degree);

	return intermediary_points;
}

std::vector<int> QPanda::get_complex_points(const TopologyData& topo_data, const size_t max_connect_degree)
{
	const size_t qubit_num = topo_data.size();
	std::vector<int> complex_points;
	for (size_t i = 0; i < qubit_num; ++i)
	{
		size_t tmp_connectivity = 0;
		for (size_t j = 0; j < qubit_num; ++j)
		{
			if (0 != topo_data[i][j])
			{
				++tmp_connectivity;
			}
		}

		if (max_connect_degree < tmp_connectivity)
		{
			complex_points.push_back(i);
		}
	}

	return complex_points;
}

std::vector<std::pair<int, TopologyData>> QPanda::split_complex_points(std::vector<int>& complex_points, const size_t max_connect_degree,
	const TopologyData& topo_data, const ComplexVertexSplitMethod split_method /*= LINEAR*/)
{
	std::vector<std::pair<int, TopologyData>> extension_qubit;
	for (const auto complex_qubit : complex_points)
	{
		// get connectivity
		size_t tmp_connectivity = 0;
		for (size_t i = 0; i < topo_data[complex_qubit].size(); ++i)
		{
			if (topo_data[complex_qubit][i] > 0)
			{
				++tmp_connectivity;
			}
		}

		//int need_increase_qubit = (tmp_connectivity / MAX_CONNECTIVITY) /*- 1*/;
		//if ((tmp_connectivity % MAX_CONNECTIVITY) > 0)
		//{
		//	++need_increase_qubit;
		//}
		//TopologyData t = extension_single_point(complex_qubit, need_increase_qubit, split_method);
		TopologyData t = extension_single_point2(complex_qubit, tmp_connectivity, max_connect_degree, split_method);

		// to undirected graph
		to_symmetric_matrix(t);

		extension_qubit.push_back(std::make_pair(complex_qubit, t));
	}

	return extension_qubit;
}

void QPanda::replace_complex_points(TopologyData& src_topo_data, const size_t max_connect_degree,
	const std::vector<std::pair<int, TopologyData>>& sub_topo_vec)
{
	const size_t src_topo_size = src_topo_data.size();
	size_t new_topo_size = src_topo_size;

	std::map<std::pair<int, int>, std::pair<int, int>> new_edge_map;
	std::vector<std::pair<int, int>> new_edges;
	for (const auto& sub_topo : sub_topo_vec)
	{
		std::map<int, int> sub_graph_edge_map;
		int target_qubit = sub_topo.first;
		std::vector<std::pair<int, size_t>> sorted_nodes = sort_sub_graph_node(sub_topo.second);

		sub_graph_edge_map.insert(std::make_pair(sorted_nodes[0].first, target_qubit));
		sorted_nodes[0].first = target_qubit;
		size_t new_qubit_index = 0;
		for (size_t i = 1; i < sorted_nodes.size(); ++i)
		{
			new_qubit_index = new_topo_size + i - 1;
			sub_graph_edge_map.insert(std::make_pair(sorted_nodes[i].first, new_qubit_index));
			sorted_nodes[i].first = new_qubit_index;
		}

		new_topo_size += (sorted_nodes.size() - 1);

		std::vector<std::pair<int, size_t>> src_adjacent_node = get_all_adjacent_node(target_qubit, src_topo_data);
		sort(src_adjacent_node.begin(), src_adjacent_node.end(), [](std::pair<int, size_t>& a, std::pair<int, size_t>& b) {return a.second > b.second; });

		std::map<std::pair<int, int>, std::pair<int, int>> tmp_edge_map = dispatch_node(target_qubit, max_connect_degree, src_adjacent_node, sorted_nodes);
		new_edge_map.insert(tmp_edge_map.begin(), tmp_edge_map.end());

		for (size_t i = 0; i < sub_topo.second.size(); ++i)
		{
			for (size_t j = 0; j < sub_topo.second.size(); ++j)
			{
				if (0 != sub_topo.second[i][j])
				{
					new_edges.push_back(std::make_pair(sub_graph_edge_map.at(i), sub_graph_edge_map.at(j)));
				}
			}
		}
	}

#if PRINT_TRACE
	PTrace("new_edge_map\n");
	for (auto& item : new_edge_map)
	{
		PTrace("(%d, %d)->(%d, %d)\n", item.first.first, item.first.second, item.second.first, item.second.second);
	}
	PTrace("-----------------\n");
#endif // PRINT_TRACE

	for (auto iter_1 = new_edge_map.begin(); iter_1 != new_edge_map.end(); ++iter_1)
	{
		for (auto iter_2 = iter_1; iter_2 != new_edge_map.end(); ++iter_2)
		{
			if ((iter_1->first.first == iter_2->first.second) &&
				(iter_1->first.second == iter_2->first.first))
			{
				iter_1->second.second = iter_2->second.first;
				iter_2->second.second = iter_1->second.first;
			}
		}
	}

#if PRINT_TRACE
	PTrace("remap: new_edge_map\n");
	for (auto& item : new_edge_map)
	{
		PTrace("(%d, %d)->(%d, %d)\n", item.first.first, item.first.second, item.second.first, item.second.second);
	}
	PTrace("-----------------\n");
#endif // PRINT_TRACE

	//build new topology by new_edge_map
	TopologyData new_topo_data(new_topo_size, std::vector<int>(new_topo_size, 0));
	for (size_t i = 0; i < src_topo_size; ++i)
	{
		for (size_t j = 0; j < src_topo_size; ++j)
		{
			new_topo_data[i][j] = src_topo_data[i][j];
		}
	}

	size_t tmp_edge_weight = 0;
	for (auto& item : new_edge_map)
	{
		const auto& src_edge = item.first;
		tmp_edge_weight = src_topo_data[src_edge.first][src_edge.second];
		new_topo_data[src_edge.first][src_edge.second] = 0;
		new_topo_data[src_edge.second][src_edge.first] = 0;

		const auto& new_edge = item.second;
		new_topo_data[new_edge.first][new_edge.second] = tmp_edge_weight;
		new_topo_data[new_edge.second][new_edge.first] = tmp_edge_weight;
	}

	for (auto& new_edge_item : new_edges)
	{
		new_topo_data[new_edge_item.first][new_edge_item.second] = 1;
	}

	src_topo_data.swap(new_topo_data);
}

bool QPanda::planarity_testing(const TopologyData& graph)
{
	const size_t topo_size = graph.size();
	Graph g(topo_size);
	for (size_t i = 0; i < topo_size; ++i)
	{
		for (size_t j = i + 1; j < topo_size; ++j)
		{	
			if (0 != graph[i][j])
			{
				g.addEdge(i, j);
			}
		}	
	}

	return PlanarityTest(g).is_planar();
}

void QPanda::recover_edges(TopologyData& topo_data, const size_t max_connect_degree, std::vector<weight_edge>& candidate_edges)
{
	sort(candidate_edges.begin(), candidate_edges.end(), [](const weight_edge& a, const weight_edge& b) {
		return a.first > b.first;
	});
	auto connectivity_vec = get_qubits_connectivity_degree(topo_data);
	for (const auto& edge_item : candidate_edges)
	{
		const auto& point_pair = edge_item.second;
		cout << "on edge:(" << point_pair.front() << ", " << point_pair.back() << "), weight = " << edge_item.first << endl;
		if ((connectivity_vec[point_pair.front()] >= max_connect_degree) || (connectivity_vec[point_pair.back()] >= max_connect_degree))
		{
			cout << "False on max connect degree" << endl;
			continue;
		}

		TopologyData tmp_topo_data = topo_data;
		tmp_topo_data[point_pair.front()][point_pair.back()] = edge_item.first;
		tmp_topo_data[point_pair.back()][point_pair.front()] = edge_item.first;
		cout << "planarity_testing: " << planarity_testing(tmp_topo_data) << endl;
		if (planarity_testing(tmp_topo_data))
		{
			connectivity_vec[point_pair.front()] += 1;
			connectivity_vec[point_pair.back()] += 1;
			topo_data[point_pair.front()][point_pair.back()] = edge_item.first;
			topo_data[point_pair.back()][point_pair.front()] = edge_item.first;
			cout << "^^^^^add ok^^^^^^^." << endl;
		}
		else
		{
			cout << "False on planarity_testing......." << endl;
		}
	}
}

double QPanda::estimate_topology(const TopologyData& src_topo_data)
{
	auto mat_d = get_diatance_matrix(src_topo_data);
	auto mat_w = get_relative_weight_matrix(src_topo_data);

	double ret_data = 0.0;
	const size_t topo_size = src_topo_data.size();
	for (size_t i = 0; i < topo_size; ++i)
	{
		for (size_t j = i + 1; j < topo_size; ++j)
		{
			ret_data += (mat_d[i][j] - mat_w[i][j]);
		}
	}

	return ret_data;
}

TopologyData QPanda::get_circuit_optimal_topology(QProg prog, QuantumMachine* quantum_machine, 
	const size_t max_connect_degree, const std::string& config_data /*= CONFIG_PATH*/)
{
	decompose_multiple_control_qgate(prog, quantum_machine, config_data);

	TopologyData toto_data = get_double_gate_block_topology(prog);

	std::vector<int> sub_graph = get_sub_graph(toto_data);
	std::vector<weight_edge> candidate_edges;
	std::vector<int> intermediary_points = del_weak_edge(toto_data, max_connect_degree, sub_graph, candidate_edges);
	//std::vector<int> intermediary_points = del_weak_edge(toto_data, sub_graph, max_connect_degree, 0.5, 0.5, 0.5);

	std::vector<int> complex_points = get_complex_points(toto_data, max_connect_degree);

	std::vector<std::pair<int, TopologyData>> complex_point_sub_graph = split_complex_points(
		complex_points, max_connect_degree, toto_data);

	replace_complex_points(toto_data, max_connect_degree, complex_point_sub_graph);

	recover_edges(toto_data, max_connect_degree, candidate_edges);

	return toto_data;
}

std::map<size_t, size_t> QPanda::map_to_continues_qubits(QProg prog, QuantumMachine *quantum_machine)
{
	std::vector<int> used_qv;
	const auto qubit_cnt = get_all_used_qubits(prog, used_qv);
	std::map<size_t, size_t> qubit_map;
	std::map<size_t, Qubit*> addr_qv_map;
	for (size_t i = 0; i < qubit_cnt; ++i)
	{
		qubit_map.insert(std::make_pair(used_qv[i], i));
		addr_qv_map.insert(std::make_pair(i, quantum_machine->allocateQubitThroughPhyAddress(i)));
	}

	LayeredTopoSeq layer_info = prog_layer(prog);
	for (const auto layer : layer_info)
	{
		for (const auto& node : layer)
		{
			bool b_need_remap = false;
			auto& cur_control_qubits = node.first->m_control_qubits;
			for (auto iter = cur_control_qubits.begin(); iter != cur_control_qubits.end(); ++iter)
			{
				const auto tmp_addr = (*iter)->get_phy_addr();
				const auto mapped_addr = qubit_map.find(tmp_addr)->second;
				if (tmp_addr != mapped_addr)
				{
					b_need_remap = true;
					*iter = addr_qv_map.at(mapped_addr);
				}
			}

			if (b_need_remap)
			{
				auto pgate = std::dynamic_pointer_cast<AbstractQGateNode>(*(node.first->m_iter));
				pgate->clear_control();
				pgate->setControl(cur_control_qubits);
			}

			auto& cur_node_qubits = node.first->m_target_qubits;
			b_need_remap = false;
			QVec new_gate_qv;
			for (auto iter = cur_node_qubits.begin(); iter != cur_node_qubits.end(); ++iter)
			{
				const auto tmp_addr = (*iter)->get_phy_addr();
				const auto mapped_addr = qubit_map.find(tmp_addr)->second;
				if (tmp_addr != mapped_addr)
				{
					b_need_remap = true;
					new_gate_qv.push_back(addr_qv_map.at(mapped_addr));
				}
				else
				{
					new_gate_qv.push_back(*iter);
				}
			}

			if (b_need_remap)
			{
				if (MEASURE_GATE == node.first->m_node_type)
				{
					//measure
					auto old_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*(node.first->m_iter));
					auto new_measure = Measure(new_gate_qv.front(), old_measure->getCBit());
					auto node_manager = std::dynamic_pointer_cast<AbstractNodeManager>(node.first->m_parent_node);
					auto new_iter = node_manager->insertQNode(node.first->m_iter,
						std::dynamic_pointer_cast<QNode>(new_measure.getImplementationPtr()));
					node_manager->deleteQNode(node.first->m_iter);
					node.first->m_iter = new_iter;
				}
				else if (RESET_NODE == node.first->m_node_type)
				{
					//reset
					auto new_reset_gate = Reset(new_gate_qv.front());
					auto node_manager = std::dynamic_pointer_cast<AbstractNodeManager>(node.first->m_parent_node);
					auto new_iter = node_manager->insertQNode(node.first->m_iter,
						std::dynamic_pointer_cast<QNode>(new_reset_gate.getImplementationPtr()));
					node_manager->deleteQNode(node.first->m_iter);
					node.first->m_iter = new_iter;
				}
				else
				{
					//gate
					auto pgate = std::dynamic_pointer_cast<AbstractQGateNode>(*(node.first->m_iter));
					pgate->remap(new_gate_qv);
				}
			}
		}
	}

	return qubit_map;
}