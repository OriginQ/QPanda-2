#ifndef _TOPOLOGY_MATCH_H_
#define _TOPOLOGY_MATCH_H_
#include <set>
#include <queue>
#include <vector>
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"

QPANDA_BEGIN


enum SwapQubitsMethod
{
	ISWAP_GATE_METHOD = 0,
	CZ_GATE_METHOD,
	CNOT_GATE_METHOD,
	SWAP_GATE_METHOD,
};

enum ArchType
{
	IBM_QX5_ARCH = 0,
	ORIGIN_VIRTUAL_ARCH,
};

/**
* @brief swap qubit location algorithm abstract class 
* @ingroup Utilities
*/
class TransformSwapAlg
{
public:
	virtual void transform(Qubit *control_qubit, Qubit *target_qubit, QProg &prog) = 0;
	virtual int getSwapCost() = 0;
	virtual int getFlipCost() = 0;
};

/**
* @brief swap qubit location by CNOT quantum gate
* @ingroup Utilities
*/
class TransformByCNOT : public TransformSwapAlg
{
public:
	TransformByCNOT() {}
	void transform(Qubit *control_qubit, Qubit *target_qubit, QProg &prog)
	{
		prog << CNOT(control_qubit, target_qubit)
			<< H(control_qubit)
			<< H(target_qubit)
			<< CNOT(control_qubit, target_qubit)
			<< H(control_qubit)
			<< H(target_qubit)
			<< CNOT(control_qubit, target_qubit);
	}
	inline int getSwapCost() { return 7; }
	inline int getFlipCost() { return 4; }
};

/**
* @brief swap qubit location by CZ quantum gate
* @ingroup Utilities
*/
class TransformByCZ : public TransformSwapAlg
{
public:
	TransformByCZ() {}

	void transform(Qubit *control_qubit, Qubit *target_qubit, QProg &prog)
	{
		prog << H(control_qubit) 
			<< CZ(control_qubit, target_qubit)
			<< H(control_qubit)
			<< H(target_qubit)
			<< CZ(control_qubit, target_qubit)
			<< H(control_qubit)
			<< H(target_qubit)
			<< CZ(control_qubit, target_qubit)
			<< H(control_qubit);
	}
	inline int getSwapCost() { return 9; }
	inline int getFlipCost() { return 0; }
};

/**
* @brief swap qubit location by ISWAP quantum gate
* @ingroup Utilities
*/
class TransformByISWAP : public TransformSwapAlg
{
public:
	TransformByISWAP() {}

	void transform(Qubit *control_qubit, Qubit *target_qubit, QProg &prog)
	{
		QGate z_gate = Z1(control_qubit);
		z_gate.setDagger(true);
		prog << z_gate
			<< X1(target_qubit)
			<< Z1(target_qubit)
			<< iSWAP(control_qubit, target_qubit)
			<< X1(control_qubit)
			<< iSWAP(control_qubit, target_qubit)
			<< X1(control_qubit)
			<< Z1(control_qubit)
			<< iSWAP(control_qubit, target_qubit)
			<< X1(target_qubit)
			<< iSWAP(control_qubit, target_qubit)
			<< X1(target_qubit)
			<< Z1(target_qubit)
			<< iSWAP(control_qubit, target_qubit)
			<< X1(control_qubit)
			<< iSWAP(control_qubit, target_qubit)
			<< Z1(target_qubit);

	}
	inline int getSwapCost() { return 17; }
	inline int getFlipCost() { return 0; }
};

/**
* @brief swap qubit location by SWAP quantum gate
* @ingroup Utilities
*/
class TransformBySWAP : public TransformSwapAlg
{
public:
	TransformBySWAP() {}

	void transform(Qubit *control_qubit, Qubit *target_qubit, QProg &prog)
	{
		prog << SWAP(control_qubit, target_qubit);
	}
	int getSwapCost() { return 7; }
	int getFlipCost() { return 0; }
};

/** 
* @brief swap qubit location algorithm factory
* @ingroup Utilities
*/
class TransformSwapAlgFactory
{
public:
	TransformSwapAlgFactory() {}

	static TransformSwapAlgFactory &GetFactoryInstance()
	{
		static TransformSwapAlgFactory fac;
		return fac;
	}
	TransformSwapAlg* CreateByType(SwapQubitsMethod type)
	{
		TransformSwapAlg* p_trans = nullptr;
		switch (type)
		{
		case QPanda::ISWAP_GATE_METHOD:
			p_trans = new TransformByISWAP();
			break;
		case QPanda::CZ_GATE_METHOD:
			p_trans = new TransformByCZ();
			break;
		case QPanda::CNOT_GATE_METHOD:
			p_trans = new TransformByCNOT();
			break;
		case QPanda::SWAP_GATE_METHOD:
			p_trans = new TransformBySWAP();
			break;
		default:
			break;
		}
		return p_trans;
	}
};

/**
* @class TopologyMatch
* @ingroup Utilities
* @brief QProg/QCircuit matches the topology of the physical qubits
*/
class TopologyMatch :public TraversalInterface<bool&>
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
		bool is_dagger;
		bool is_flip;
		std::vector<double> param;
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

	size_t m_swap_cost;
	size_t m_flip_cost;
	int m_swap_qubits_method;

	std::set<edge> m_graph;		  /**< topological graph */
	std::vector<std::vector<gate> > m_layers;    /**< qprog layered results */
	std::vector<int> m_last_layer;
	std::priority_queue<node, std::vector<node>, node_cmp> m_nodes;     /**< priority_queue of searched nodes */
	TransformSwapAlg *m_pTransformSwap;
	QuantumMachine *m_qvm;
	std::map<int, std::vector<std::vector<int> > > m_gate_dist_map;

	std::map<int, std::function<QGate(Qubit *)> > m_singleGateFunc;
	std::map<int, std::function<QGate(Qubit *, double)> > m_singleAngleGateFunc;
	std::map<int, std::function<QGate(Qubit *, Qubit*)> > m_doubleGateFunc;
	std::map<int, std::function<QGate(Qubit *, Qubit*, double)> > m_doubleAngleGateFunc;
	QProg m_prog;

public:
	TopologyMatch(QuantumMachine * machine, QProg prog, SwapQubitsMethod method = CNOT_GATE_METHOD, 
		ArchType arch_type = IBM_QX5_ARCH, const std::string conf = CONFIG_PATH);

	~TopologyMatch();
	/**
	* @brief  Mapping qubits in a quantum program
	* @param[in]  Qprog  quantum program
	* @param[out]  Qprog&  the mapped quantum program
	* @return   void
	**/
	void mappingQProg(QVec &qv, QProg &mapped_prog);


	virtual void execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, bool &);
	virtual void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, bool &);

private:

	void traversalQProgToLayers(QProg *prog);

	void buildGraph(ArchType type, std::set<edge> &graph, size_t &positions);

	int breadthFirstSearch(int start, int goal, const std::set<edge>& graph, size_t swap_cost, size_t flip_cost);

	std::vector<std::vector<int> > buildDistTable(int positions, const std::set<edge> &graph, size_t swap_cost, size_t flip_cost);

	std::vector<std::vector<int> > getGateDistTable(int gate_type);

	void createNodeFromBase(node base_node, std::vector<edge> &swaps, int nswaps, node &new_node);

	void calculateHeurCostForNextLayer(int next_layer, node &new_node);

	void expandNode(const std::vector<int> &qubits, int qubit, std::vector<edge> &swaps, int nswaps,
		std::vector<int> &used, node base_node, const std::vector<gate> &layer_gates, int next_layer);

	int getNextLayer(int layer);

	node fixLayerByAStar(int layer, std::vector<int> &map, std::vector<int> &loc);

	void buildResultingQProg(const std::vector<gate> &resulting_gates, std::vector<int> loc, QVec &qv, QProg &prog);

	bool isContains(std::vector<int> v, int e);

	bool isReversed(std::set<edge> graph, edge det_edge);

};

/**
* @brief  QProg/QCircuit matches the topology of the physical qubits
* @ingroup Utilities
* @param[in]  QProg  quantum program
* @param[out]  QVec& Mapped bit sequence
* @param[in]  QuantumMachine *  quantum machine
* @param[in]  SwapQubitsMethod   swap qubits by CNOT/CZ/SWAP/iSWAP gate
* @param[in]  ArchType    architectures type
* @param[in]  const std::string : the  config data, @see JsonConfigParam::load_config
* @return    QProg   mapped  quantum program
* @exception
* @note
*/
QProg  topology_match(QProg prog, QVec &qv, QuantumMachine *machine, SwapQubitsMethod method = CNOT_GATE_METHOD, 
	ArchType arch_type = IBM_QX5_ARCH, const std::string& conf = CONFIG_PATH);


QPANDA_END


#endif // ! _TOPOLOGY_MATCH_H_