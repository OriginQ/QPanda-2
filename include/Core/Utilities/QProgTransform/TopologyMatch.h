#ifndef _TOPOLOGY_MATCH_H_
#define _TOPOLOGY_MATCH_H_
#include <queue>
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"

QPANDA_BEGIN

/**
* @class TopologyMatch
* @ingroup Utilities
* @brief QProg matches the topology of the physical qubits
*/
class TopologyMatch
{
private:
	struct edge
	{
		int v1;
		int v2;
		bool operator<(const struct edge& right)const 
		{
			if (this->v1 != right.v1)
			{
				return this->v1 < right.v1;
			}
			return this->v2 < right.v2;
		}
	};

	struct gate
	{
		int target;
		int control;
		int type;
		int barrier_id;
		int barrier_size;

		size_t vertex_id;
	};

	struct node
	{
		int cost_fixed;
		int cost_heur;
		int cost_heur2;
		int depth;
		std::vector<int> qubits;// get qubit of location -> -1 indicates that there is "no" qubit at a certain location
		std::vector<int> locations;// get location of qubits -> -1 indicates that a qubit does not have a location -> shall only occur for i > nqubits
		int nswaps;
		int done;
		std::vector<std::vector<edge> > swaps;
	};

	struct node_cmp
	{
		bool operator()(node &x, node &y) const
		{
			if ((x.cost_fixed + x.cost_heur + x.cost_heur2) != (y.cost_fixed + y.cost_heur + y.cost_heur2))
			{
				return (x.cost_fixed + x.cost_heur + x.cost_heur2) > (y.cost_fixed + y.cost_heur + y.cost_heur2);
			}

			if (x.done == 1)
			{
				return false;
			}
			if (y.done == 1)
			{
				return true;
			}
			if (x.cost_heur + x.cost_heur2 != y.cost_heur + y.cost_heur2)
			{
				return x.cost_heur + x.cost_heur2 > y.cost_heur + y.cost_heur2;
			}
			else
			{
				for (int i = 0; i < x.qubits.size(); i++)
				{
					if (x.qubits[i] != y.qubits[i])
					{
						return x.qubits[i] < y.qubits[i];
					}
				}
				return false;
			}
		}
	};

	size_t m_positions;  /**< physical  qubits  number   */
	size_t m_nqubits;    /**< quantum machine allocate qubits */
	const size_t m_swap_cost = 7;
	const size_t m_flip_cost = 0;
	std::vector<std::vector<int> > m_dist;
	std::set<edge> m_graph;		  /**< topological graph */
	std::vector<std::vector<gate> > m_layers;    /**< qprog layered results */
	std::priority_queue<node, std::vector<node>, node_cmp> m_nodes;     /**< priority_queue of searched nodes */
	QuantumMachine *m_qvm;
	std::shared_ptr<QProgDAG> m_dag;
	QProg m_prog;

public:
	TopologyMatch(QuantumMachine * machine, QProg prog,  const std::string conf = CONFIG_PATH);

	~TopologyMatch();
	
	/**
	* @brief  Mapping qubits in a quantum program
	* @param[in]  Qprog  quantum program
	* @param[out]  Qprog&  the mapped quantum program
	* @return   void
	**/
	void mappingQProg(QVec &qv, QProg &mapped_prog);

private:

	void traversalQProgToLayers();

	void buildGraph(std::set<edge> &graph, size_t &positions);

	int breadthFirstSearch(int start, int goal, 
		const std::set<edge>& graph, size_t swap_cost, size_t flip_cost);

	std::vector<std::vector<int> > buildDistTable(int positions, 
		const std::set<edge> &graph, size_t swap_cost, size_t flip_cost);

	void createNodeFromBase(node base_node,
		std::vector<edge> &swaps, int nswaps, node &new_node);

	void calculateHeurCostForNextLayer(int next_layer, node &new_node);

	void expandNode(const std::vector<int> &qubits, int qubit, 
		std::vector<edge> &swaps, int nswaps, std::vector<int> &used, 
		node base_node, const std::vector<gate> &layer_gates, int next_layer);

	int getNextLayer(int layer);

	node fixLayerByAStar(int layer, std::vector<int> &map, std::vector<int> &loc);

	void buildResultingQProg(const std::vector<gate> &resulting_gates, 
		std::vector<int> loc, QVec &qv, QProg &prog);

	bool isContains(std::vector<int> v, int e);

	std::vector<int> getGateQaddrs(const QProgDAGVertex& _v);
};

/**
* @brief  QProg/QCircuit matches the topology of the physical qubits
* @ingroup Utilities
* @param[in]  QProg  quantum program
* @param[in|out]  QVec& Mapped bit sequence
* @param[in]  QuantumMachine *  quantum machine
* @param[in]  const std::string : the  config data, @see JsonConfigParam::load_config
* @return    QProg   mapped  quantum program
*/
QProg  topology_match(QProg prog, QVec &qv, QuantumMachine *machine,  
	const std::string& conf = CONFIG_PATH);


QPANDA_END


#endif // ! _TOPOLOGY_MATCH_H_