#ifndef GET_QUBIT_TOPOLOGY_H
#define GET_QUBIT_TOPOLOGY_H

#include <vector>
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include <map>

QPANDA_BEGIN

template <class T = int>
using MatData = std::vector<std::vector<T>>;

using TopologyData = MatData<int>;
using QubitPair = std::pair<size_t, size_t>;
using weight_edge = std::pair<size_t, std::vector<int>>;

enum ComplexVertexSplitMethod
{
	METHOD_UNDEFINED = -1,
	LINEAR = 0,
	RING
	/*RING_FULL_CONNECT,
	STAR*/

};

class GetQubitTopology
{
public:
	GetQubitTopology();
	~GetQubitTopology();

	void init();

	const TopologyData& get_src_adjaccent_matrix(QProg prog);

protected:
	void get_all_double_gate_qubits(QProg prog);

private:
	std::shared_ptr<TransformDecomposition> m_p_transf_decompos;
	std::map<QubitPair, size_t> m_double_gate_qubits;
	TopologyData m_topo_data;
};

/* public interface */

/**
* @brief Gets sub-graph from the input graph
* @ingroup Utilities
* @param[in] const TopologyData& the target graph
* @return std::vector<int> Returns the sub graph information of each vertex.
*/
std::vector<int> get_sub_graph(const TopologyData& topo_data);

/**
* @brief Recover edges from the candidate edges
* @ingroup Utilities
* @param[in] const TopologyData& the target graph
* @param[in] const size_t The max connect-degree
* @param[in] std::vector<weight_edge>& Thecandidate edges
* @return 
*/
void recover_edges(TopologyData& topo_data, const size_t max_connect_degree, std::vector<weight_edge>& candidate_edges);

/**
* @brief Gets complex points
* @ingroup Utilities
* @param[in] const TopologyData& the target graph
* @param[in] const size_t The max connect-degree
* @return std::vector<int> Complex point set
*/
std::vector<int> get_complex_points(const TopologyData& topo_data, const size_t max_connect_degree);

/**
* @brief splitting complex points
* @ingroup Utilities
* @param[in] std::vector<int>& Complex point set
* @param[in] const size_t The max connect-degree
* @param[in] const TopologyData& the target graph
* @param[in] const ComplexVertexSplitMethod The split-method, default is LINEAR
* @return std::vector<std::pair<int, TopologyData>> Complex points and their corresponding splitting structures
*/
std::vector<std::pair<int, TopologyData>> split_complex_points(std::vector<int>& complex_points, const size_t max_connect_degree, 
	const TopologyData& topo_data, const ComplexVertexSplitMethod split_method = LINEAR);

/**
* @brief replace complex points in target graph
* @ingroup Utilities
* @param[in] TopologyData& the target graph
* @param[in] const size_t The max connect-degree
* @param[in] const std::vector<std::pair<int, TopologyData>>& Complex points and their corresponding splitting structures
* @return
*/
void replace_complex_points(TopologyData& src_topo_data, const size_t max_connect_degree, 
	const std::vector<std::pair<int, TopologyData>>& sub_topo_vec);

/**
* @brief Evaluate topology performance
* @ingroup Utilities
* @param[in] const TopologyData& the target graph
* @return double Topological structure score
*/
double estimate_topology(const TopologyData& src_topo_data);

/**
* @brief del weak edges(Edges with weight less than average)
* @ingroup Utilities
* @param[in] TopologyData& the target graph
* @return
*/
void del_weak_edge(TopologyData& topo_data);

/**
* @brief del weak edges
* @ingroup Utilities
* @param[in] TopologyData& the target graph
* @param[in] const size_t The max connect-degree
* @param[in] std::vector<int>& The sub-graph set
* @param[out] std::vector<weight_edge>& The candidate-edges(Temporarily deleted edges)
* @return std::vector<int> Intermediary points
*/
std::vector<int> del_weak_edge(TopologyData& topo_data, const size_t max_connect_degree,
	std::vector<int>& sub_graph_set, std::vector<weight_edge>& candidate_edges);

/**
* @brief del weak edges
* @ingroup Utilities
* @param[in] TopologyData& the target graph
* @param[in] std::vector<int>& The sub-graph set
* @param[in] const size_t The max connect-degree
* @param[in] const double weight factor
* @param[in] const double connectivity-degree factor
* @param[in] const double dispersion-degree factor
* @return std::vector<int> Intermediary points
*/
std::vector<int> del_weak_edge(TopologyData& topo_data, std::vector<int>& sub_graph_set, const size_t max_connect_degree,
	const double lamda1, const double lamda2, const double lamda3);

/**
* @brief get double gate block topology
* @ingroup Utilities
* @param[in] prog  the target circuit/prog
* @return TopologyData Topological structure composed of two gate blocks
*/
TopologyData get_double_gate_block_topology(QProg prog);

/**
* @brief  planarity testing
* @ingroup Utilities
* @param[in] const TopologyData& the target graph
* @return bool If the input graph is planarity, return true, otherwise retuen false.
*/
bool planarity_testing(const TopologyData& graph);

/**
* @brief  Get the optimal topology of the input circuit
* @ingroup Utilities
* @param[in] prog  the target circuit/prog
* @param[in] QuantumMachine* Quantum Machine
* @param[in] const size_t The max connect-degree
* @param[in] const std::string& It can be configuration file or configuration data, which can be distinguished by file suffix,
			 so the configuration file must be end with ".json", default is CONFIG_PATH
* @return TopologyData Return the optimal topology of the target circuit
*/
TopologyData get_circuit_optimal_topology(QProg prog, QuantumMachine* quantum_machine, 
	const size_t max_connect_degree, const std::string& config_data = CONFIG_PATH);

std::map<size_t, size_t> map_to_continues_qubits(QProg prog, QuantumMachine *quantum_machine);

QPANDA_END

#endif // GET_QUBIT_TOPOLOGY_H