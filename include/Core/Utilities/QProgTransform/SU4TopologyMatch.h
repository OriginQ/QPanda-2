/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file SU4TopologyMatch.h */
#ifndef _SU4_TOPOLOGY_MATCH_H_
#define _SU4_TOPOLOGY_MATCH_H_

#include <set>
#include <queue>
#include <vector>
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/QProgTransform/TopologyMatch.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/GraphMatch.h"
QPANDA_BEGIN


/**
* @class SU4TopologyMatch
* @ingroup Utilities
* @brief Su4 quantum circuit matches the topology of the physical qubits
*/
class SU4TopologyMatch
{
private:
	/**
	* @brief Store quantum gate information
	*/
	struct gate
	{
		int target;
		int control;
		int type;
		bool is_dagger;
		bool is_flip;
		std::vector<double> param;
	};

	/**
	* @brief define struct for nodes in the A* search;
	*/
	struct node
	{
		int cost_fixed;			/**< fixed cost of the current permutation */
		int cost_heur;			/** heuristic cost of the current permutation*/
		std::vector<int> locations;		/** location (i.e. pysical qubit) of a logical qubit*/
		std::vector<int> qubits;			/** logical qubits that are mapped to the physical ones */
		bool is_goal;							/** true if the node is a goal node;*/
		std::vector<std::pair<int, int>> swaps;						/** a sequence of swap operations that have been applied */
		std::vector<std::pair<int, int>> remaining_gates;		/** vector holding the initial gates that have to be mapped to an edge of the coupling map*/
	};

	/**
	* @brief define struct for priority queue
	*/
	struct node_cmp
	{
		bool operator()(node &x, node &y) const
		{
			return (x.cost_fixed + x.cost_heur) > (y.cost_fixed + y.cost_heur);
		}
	};

	/**
	* @brief Simple digraph, used to group all gates
	*/
	struct gates_digraph
	{
		std::map<size_t, std::pair<std::vector<gate>, std::vector<int> > > vertexs;
		std::vector<std::pair<size_t, size_t> > edges; /**< in --> out  */
		size_t id = 0;
		size_t add_vertex(std::pair<std::vector<gate>, std::vector<int>> info)
		{
			vertexs.insert({ id, info });
			return id++;
		}

		bool add_edge(size_t u, size_t v)
		{
			if (vertexs.find(u) == vertexs.end()
				|| vertexs.find(v) == vertexs.end())
			{
				return false;
			}
			edges.push_back({ u, v });
			return true;
		}

		bool remove_vertex(size_t id)
		{
			auto it = vertexs.find(id);
			if (it == vertexs.end())
				return false;

			auto iter = edges.begin();
			while (iter != edges.end())
			{
				if (iter->first == id || iter->second == id)
					iter = edges.erase(iter);
				else
					iter++;
			}
			vertexs.erase(it);
			return true;
		}

		size_t in_degree(size_t id)
		{
			int degree = 0;
			for (auto edge : edges)
			{
				if (edge.second == id)
					degree++;
			}
			return degree;
		}
	};


	QuantumMachine *m_qvm;
	QVec &m_qv;
	size_t m_nqubits;				/**< The number of physical quanta in the topology */
	size_t m_used_qubits;		/**< The number of qubits used in a quantum program */
	std::set<std::pair<int, int >> m_coupling_map;
	std::vector< std::vector<int> > m_dist;   /**<Table of minimal distances between physical qubits */
	std::map<int, int > m_gate_costs;
	std::vector<int>  m_locations;   /**< location (i.e. pysical qubit) of a logical qubit  result */
	std::vector<int>  m_qubits;		/**< logical qubits that are mapped to the physical ones result */

public:

	SU4TopologyMatch(QuantumMachine * machine, QVec &qv);
	~SU4TopologyMatch() {}

	/**
	* @brief  Mapping qubits in a quantum program
	* @param[in]  Qprog  quantum program
	* @param[out]  Qprog&  the mapped quantum program
	* @return   void
	**/
	void mapping_qprog(QProg prog, QProg &mapped_prog);

private:

	/**
	* @brief Transform quantum program
	**/
	void transform_qprog(QProg prog, std::vector<gate> &circuit);

	/**
	* @brief Group all gate that are applied to two certain qubits
	**/
	void pre_processing(std::vector<gate> circuit, gates_digraph &grouped_gates);

	/**
	* @brief Build coupling map by topological type
	**/
	void build_coupling_map();

	/**
	* @brief Breadth-first search algorithm to find minimal distances between physical qubits
	**/
	void bfs(int start, std::vector< std::vector<int> > &dist);

	/**
	* @brief Main method for performing the mapping algorithm
	**/
	void a_star_mapper(gates_digraph grouped_gates, std::vector<gate> &compiled_circuit);

	/**
	* @brief A* search algorithm to find a sequence of swap gates such that at least one gate in gates can be applied
	**/
	void a_star_search(std::set<std::pair<int, int>>& applicable_gates, std::vector<int> & map,
		std::vector<int> & loc, std::set<std::pair<int, int>>& free_swaps, node &result);

	/**
	* @brief  Function to rewrite gates with the current mapping and add them to the compiled circuit
	**/
	void add_rewritten_gates(std::vector<gate> gates_original, std::vector<int> locations, std::vector<gate> &compiled_circuit);


	/**
	* @brief Call an A* algorithm to determe the best initial mapping   
	**/
	void  find_initial_permutation(gates_digraph grouped_gates, node &result);

	/**
	* @brief Generate a quantum program by the mapping results
	**/
	void build_qprog(std::vector<gate>  compiled_circuit, QProg & mapped_prog);
};



/**
* @brief  Su4 quantum circuit matches the topology of the physical qubits
* @ingroup Utilities
* @param[in]  QProg  quantum program
* @param[in]  QVec  qubit  vector
* @param[in]  QuantumMachine *  quantum machine
* @return    QProg   mapped  quantum program
*/
QProg  su4_circiut_topology_match(QProg prog, QVec &qv, QuantumMachine *machine);


QPANDA_END


#endif // ! _SU4_TOPOLOGY_MATCH_H_