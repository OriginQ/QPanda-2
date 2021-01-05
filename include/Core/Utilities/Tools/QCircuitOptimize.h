#ifndef QCIRCUIT_OPTIMIZE_H
#define QCIRCUIT_OPTIMIZE_H
#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <vector>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/TopologSequence.h"
#include <memory>
#include "Core/Utilities/QProgInfo/QGateCounter.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include <fstream>

QPANDA_BEGIN

enum QCircuitOPtimizerMode
{
	Undefine_mode = -1,
	Merge_H_X = 1,
	Merge_U3 = (1 << 1),
    Merge_RX = (1 << 2),
	Merge_RY = (1 << 3),
	Merge_RZ = (1 << 4)
};

class AbstractCirOptimizer
{
public:
	virtual void do_optimize(QProg src_prog, OptimizerSink &gates_sink, SinkPos& sink_size, std::vector<QCircuit>& replace_to_cir_vec) = 0;

	virtual bool is_same_controled(pOptimizerNodeInfo first_node, pOptimizerNodeInfo second_node){
		QVec& control_vec1 = first_node->m_control_qubits;
		QVec& control_vec2 = second_node->m_control_qubits;
		QVec result_vec1 = control_vec1 - control_vec2;
		QVec result_vec2 = control_vec2 - control_vec1;

		if ((result_vec1.size() != 0) || (result_vec2.size() != 0))
		{
			return false;
		}

		if (1 != (second_node->m_layer - first_node->m_layer))
		{
			return false;
		}
		return true;
	}
};

struct OptimizerSubCir
{
	QCircuit target_sub_cir;
	QCircuit replace_to_sub_cir;
};

class FindSubCircuit
{
public:
	template <class T>
	using MatchNode = SeqNode<T>;

	template <class T>
	using MatchNodeVec = std::vector<MatchNode<T>>;

	using MatchNodeTable = std::vector<std::pair<pOptimizerNodeInfo, MatchNodeVec<pOptimizerNodeInfo>>>;

public:
	FindSubCircuit(LayeredTopoSeq& topolog_sequence) 
		:m_topolog_sequence(topolog_sequence)
	{}
	virtual ~FindSubCircuit() {}

	/**
	* @brief  Query the subgraph and store the query results in query_Result
	* @ingroup Utilities
	* @param[in] LayeredTopoSeq& store the query results
	* @return
	* @note
	*/
	void sub_cir_query(LayeredTopoSeq& sub_sequence);

	bool node_match(const SeqNode<pOptimizerNodeInfo>& target_seq_node, const SeqNode<pOptimizerNodeInfo>& graph_node);
	bool check_angle(const pOptimizerNodeInfo node_1, const pOptimizerNodeInfo node_2);

	/**
	* @brief Layer matching: matching and combining the nodes of each layer of the sub graph
	* @ingroup Utilities
	* @param[in] SeqLayer<pOptimizerNodeInfo>& the target matching sub-seq-layer
	* @param[in] const size_t the current matching layer
	* @param[in] std::vector<LayeredTopoSeq>& sub-graph vector
	* @return
	* @note
	*/
	void match_layer(SeqLayer<pOptimizerNodeInfo>& sub_seq_layer, const size_t match_layer, std::vector<LayeredTopoSeq>& sub_graph_vec);

	/**
	* @brief Merge incomplete subgraphs
		 Implementation method: get the node set of the next layer of each subgraph of the matching subgraph set.
		 If the node set of the next layer of the two subgraphs has duplicate elements, merge the two subgraphs
	* @ingroup Utilities
	* @param[in] std::vector<LayeredTopoSeq>& the sub graph vector
	* @param[in] const size_t the target layer
	* @param[in] LayeredTopoSeq& the target sub-sequence
	* @return
	* @note
	*/
	void merge_sub_graph_vec(std::vector<LayeredTopoSeq>& sub_graph_vec, const size_t match_layer, LayeredTopoSeq& target_sub_sequence);

	/**
	* @brief Clean up the result set of matching subgraphs and delete the wrong matches
	* @ingroup Utilities
	* @param[in] std::vector<LayeredTopoSeq>& the result set of matching subgraphs
	* @param[in] LayeredTopoSeq& the target sub-sequence
	* @return
	* @note
	*/
	void clean_sub_graph_vec(std::vector<LayeredTopoSeq>& sub_graph_vec, LayeredTopoSeq& target_sub_sequence);

	/**
	* @brief merge sub-graph: merging src_seq into dst_seq by layer
	* @ingroup Utilities
	* @param[in] LayeredTopoSeq& the src_seq
	* @param[in] LayeredTopoSeq& dst_seq
	* @return
	* @note
	*/
	void merge_topolog_sequence(LayeredTopoSeq& src_seq, LayeredTopoSeq& dst_seq);

	const std::vector<LayeredTopoSeq>& get_sub_graph_vec() { return m_sub_graph_vec; }

	void clear() {
		m_node_match_vector.clear();
		m_sub_graph_vec.clear();
	}

private:
	LayeredTopoSeq& m_topolog_sequence;
	std::vector<LayeredTopoSeq> m_sub_graph_vec; /* Multiple possible matching subgraphs, each of which is stored in the form of topological sequence */
	MatchNodeTable m_node_match_vector;
};

class QCircuitOPtimizer : public ProcessOnTraversing
{
public:
	QCircuitOPtimizer();
	~QCircuitOPtimizer();

	void process(const bool on_travel_end = false) override;
	void register_single_gate_optimizer(const int mode);
	void register_optimize_sub_cir(QCircuit sub_cir, QCircuit replase_to_cir);
	void run_optimize(QProg src_prog, const QVec qubits = {}, bool b_enable_I = false);

	/**
    * @brief  replace sub circuit
    * @ingroup Utilities
    * @param[in] std::function<QCircuit(const size_t)> the function to get a new quantum circuit
    * @return the new quantum prog
    * @note
    */
	QProg replase_sub_cir(std::function<QCircuit(const size_t)> get_cir_fun);
	void sub_cir_optimizer(const size_t optimizer_sub_cir_index);
	void do_optimizer();

protected:
	/**
    * @brief Mark each node in the sub graph
    * @ingroup Utilities
    * @param[in] std::vector<LayeredTopoSeq>& the sub graph vector
    * @return
    * @note
    */
	void mark_sug_graph(const std::vector<LayeredTopoSeq>& sub_graph_vec);
	
	/*
    Note: No nesting in src circuit or prog
    */
	void cir_to_gate_buffer(QProg& src_node);

	void check_bit_map(LayeredTopoSeq&);
	QCircuit remap_cir(QCircuit src_cir, const size_t target_graph_index);
	QProg gate_sink_to_cir(std::vector<QCircuit>& replace_to_cir_vec);
	void check_angle_param(pOptimizerNodeInfo target_gate, pOptimizerNodeInfo matched_gate, std::vector<double>& angle_vec);
	void set_angle_param(std::shared_ptr<AbstractQGateNode> p_gate, const size_t target_graph_index);
	bool check_same_gate_type(SeqLayer<pOptimizerNodeInfo>& layer);

public:
	QProg m_new_prog;

private:
	QProg m_src_prog;
	std::vector<std::shared_ptr<AbstractCirOptimizer>> m_optimizers;
	std::vector<OptimizerSubCir> m_optimizer_cir_vec;
	LayeredTopoSeq m_topolog_sequence;
	FindSubCircuit m_sub_cir_finder;
	std::vector<std::map<size_t, Qubit*>> m_sub_graph_qubit_map_vec;/* qubit mapping of each sub-graph */
	size_t m_cur_optimizer_sub_cir_index;
	std::vector<QProg> m_tmp_cir_vec; /* Temporary QCircuits for storing QGate nodes */
	std::vector<std::vector<double>> m_angle_vec; /* Angle variable of each sub-graph */ 
	bool m_b_enable_I;
};

/**
* @brief QCircuit optimizer
* @ingroup Utilities
* @param[in,out]  QProg&(or	QCircuit&) the source prog(or circuit)
* @param[in] std::vector<std::pair<QCircuit, QCircuit>> 
* @param[in] const int Optimise mode(see QCircuitOPtimizerMode), Support several models: 
                  Merge_H_X: Optimizing continuous H or X gate
				  Merge_U3: merge continues single gates to a U3 gate
				  Merge_RX: merge continues RX gates
				  Merge_RY: merge continues RY gates
				  Merge_RZ: merge continues RZ gates
* @return     void
*/
void sub_cir_optimizer(QCircuit& src_cir, std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec = {}, 
	const int mode = QCircuitOPtimizerMode::Merge_H_X);
void sub_cir_optimizer(QProg& src_prog, std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec = {},
	const int mode = QCircuitOPtimizerMode::Merge_H_X);

template <typename T>
void cir_optimizer_by_config(T &src_cir, const std::string config_data = CONFIG_PATH, 
	const int mode = QCircuitOPtimizerMode::Merge_H_X) {
	std::vector<std::pair<QCircuit, QCircuit>> optimitzer_cir;
	QCircuitOptimizerConfig tmp_config_reader(config_data);
	tmp_config_reader.get_replace_cir(optimitzer_cir);

	sub_cir_optimizer(src_cir, optimitzer_cir, mode);
}

QPANDA_END
#endif // QCIRCUIT_OPTIMIZE_H