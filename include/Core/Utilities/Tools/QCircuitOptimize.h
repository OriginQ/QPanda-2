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

using OptimizerFlag = unsigned char;

class AbstractCirOptimizer
{
public:
	virtual void do_optimize(QProg src_prog, OptimizerSink &gates_sink, std::vector<QCircuit>& replace_to_cir_vec) = 0;

	virtual bool is_same_controled(pOptimizerNodeInfo first_node, pOptimizerNodeInfo second_node){
		QVec& control_vec1 = first_node->m_ctrl_qubits;
		QVec& control_vec2 = second_node->m_ctrl_qubits;
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
	OptimizerSink m_sub_cir_gates_buffer;
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
	FindSubCircuit(TopologSequence<pOptimizerNodeInfo>& topolog_sequence) 
		:m_topolog_sequence(topolog_sequence)
	{}
	virtual ~FindSubCircuit() {}

	/**
	* @brief  Query the subgraph and store the query results in query_Result
	* @ingroup Utilities
	* @param[in] TopologSequence<pOptimizerNodeInfo>& store the query results
	* @return
	* @note
	*/
	void sub_cir_query(TopologSequence<pOptimizerNodeInfo>& sub_sequence);

	bool node_match(const SeqNode<pOptimizerNodeInfo>& target_seq_node, const SeqNode<pOptimizerNodeInfo>& graph_node);
	bool check_angle(const pOptimizerNodeInfo node_1, const pOptimizerNodeInfo node_2);

	/**
	* @brief Layer matching: matching and combining the nodes of each layer of the sub graph
	* @ingroup Utilities
	* @param[in] SeqLayer<pOptimizerNodeInfo>& the target matching sub-seq-layer
	* @param[in] const size_t the current matching layer
	* @param[in] std::vector<TopologSequence<pOptimizerNodeInfo>>& sub-graph vector
	* @return
	* @note
	*/
	void match_layer(SeqLayer<pOptimizerNodeInfo>& sub_seq_layer, const size_t match_layer, std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec);

	/**
	* @brief Merge incomplete subgraphs
		 Implementation method: get the node set of the next layer of each subgraph of the matching subgraph set.
		 If the node set of the next layer of the two subgraphs has duplicate elements, merge the two subgraphs
	* @ingroup Utilities
	* @param[in] std::vector<TopologSequence<pOptimizerNodeInfo>>& the sub graph vector
	* @param[in] const size_t the target layer
	* @param[in] TopologSequence<pOptimizerNodeInfo>& the target sub-sequence
	* @return
	* @note
	*/
	void merge_sub_graph_vec(std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec, const size_t match_layer, TopologSequence<pOptimizerNodeInfo>& target_sub_sequence);

	/**
	* @brief Clean up the result set of matching subgraphs and delete the wrong matches
	* @ingroup Utilities
	* @param[in] std::vector<TopologSequence<pOptimizerNodeInfo>>& the result set of matching subgraphs
	* @param[in] TopologSequence<pOptimizerNodeInfo>& the target sub-sequence
	* @return
	* @note
	*/
	void clean_sub_graph_vec(std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec, TopologSequence<pOptimizerNodeInfo>& target_sub_sequence);

	/**
	* @brief merge sub-graph: merging src_seq into dst_seq by layer
	* @ingroup Utilities
	* @param[in] TopologSequence<pOptimizerNodeInfo>& the src_seq
	* @param[in] TopologSequence<pOptimizerNodeInfo>& dst_seq
	* @return
	* @note
	*/
	void merge_topolog_sequence(TopologSequence<pOptimizerNodeInfo>& src_seq, TopologSequence<pOptimizerNodeInfo>& dst_seq);

	const std::vector<TopologSequence<pOptimizerNodeInfo>>& get_sub_graph_vec() { return m_sub_graph_vec; }

	void clear() {
		m_node_match_vector.clear();
		m_sub_graph_vec.clear();
	}

private:
	TopologSequence<pOptimizerNodeInfo>& m_topolog_sequence;
	std::vector<TopologSequence<pOptimizerNodeInfo>> m_sub_graph_vec; /* Multiple possible matching subgraphs, each of which is stored in the form of topological sequence */
	MatchNodeTable m_node_match_vector;
};

class QCircuitOPtimizer : public ProcessOnTraversing
{
public:
	QCircuitOPtimizer();
	~QCircuitOPtimizer();

	void process(const bool on_travel_end = false) override;
	void register_single_gate_optimizer(const OptimizerFlag mode);
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
    * @param[in] std::vector<TopologSequence<pOptimizerNodeInfo>>& the sub graph vector
    * @return
    * @note
    */
	void mark_sug_graph(const std::vector<TopologSequence<pOptimizerNodeInfo>>& sub_graph_vec);
	
	/*
    Note: No nesting in src circuit or prog
    */
	template <class T>
	void cir_to_gate_buffer(T& src_node, OptimizerSink& gates_buffer) {
		gates_buffer.clear();


		std::vector<int> sub_cir_used_qubits;
		get_all_used_qubits(src_node, sub_cir_used_qubits);
		if (sub_cir_used_qubits.size() == 0)
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to transfer cir to gate_buffer, src_cir is null.");
		}

		for (size_t i = 0; i < sub_cir_used_qubits.size(); ++i)
		{
			gates_buffer.insert(GatesBufferType(sub_cir_used_qubits.at(i), std::list<pOptimizerNodeInfo>()));
		}

		std::shared_ptr<QNode> parent_node = std::dynamic_pointer_cast<QNode>(src_node.getImplementationPtr());
		QCircuitParam p;
		for (auto gate_itr = src_node.getFirstNodeIter(); gate_itr != src_node.getEndNodeIter(); ++gate_itr)
		{
			auto tmp_node = (*gate_itr);
			if (GATE_NODE == tmp_node->getNodeType())
			{
				add_gate_to_buffer(gate_itr, p, std::dynamic_pointer_cast<QNode>(src_node.getImplementationPtr()), gates_buffer);
			}
			else if (MEASURE_GATE == tmp_node->getNodeType())
			{
				auto measure_node = std::dynamic_pointer_cast<AbstractQuantumMeasure>(tmp_node);
				add_non_gate_to_buffer(gate_itr, tmp_node->getNodeType(), { measure_node->getQuBit() }, p, gates_buffer);
			}
			else if (RESET_NODE == tmp_node->getNodeType())
			{
				auto reset_node = std::dynamic_pointer_cast<AbstractQuantumReset>(tmp_node);
				add_non_gate_to_buffer(gate_itr, tmp_node->getNodeType(), { reset_node->getQuBit() }, p, gates_buffer);
			}
			else
			{
				QCERR_AND_THROW_ERRSTR(run_fail, "Error: unsupport node type.");
			}
		}
	}

	template <class T>
	void cir_to_topolog_sequence(T& src_cir, OptimizerSink& gate_buf, TopologSequence<pOptimizerNodeInfo>& sub_cir_sequence) {
		gate_buf.clear();
		cir_to_gate_buffer(src_cir, gate_buf);
		gates_sink_to_topolog_sequence(gate_buf, sub_cir_sequence);
	}

	void check_bit_map(TopologSequence<pOptimizerNodeInfo>&);
	QCircuit remap_cir(QCircuit src_cir, const size_t target_graph_index);
	QProg gate_sink_to_cir(std::vector<QCircuit>& replace_to_cir_vec);
	void check_angle_param(pOptimizerNodeInfo target_gate, pOptimizerNodeInfo matched_gate, std::vector<double>& angle_vec);
	void set_angle_param(std::shared_ptr<AbstractQGateNode> p_gate, const size_t target_graph_index);
	bool check_same_gate_type(SeqLayer<pOptimizerNodeInfo>& layer);

public:
	QProg m_new_prog;

	/* Optimise mode */
	static const unsigned char Merge_H_X = 1;
	static const unsigned char Merge_U3 = (1 << 1);
	static const unsigned char Merge_RX = (1 << 2);
	static const unsigned char Merge_RY = (1 << 3);
	static const unsigned char Merge_RZ = (1 << 4);

private:
	QProg m_src_prog;
	std::vector<std::shared_ptr<AbstractCirOptimizer>> m_optimizers;
	std::vector<OptimizerSubCir> m_optimizer_cir_vec;
	TopologSequence<pOptimizerNodeInfo> m_topolog_sequence;
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
* @param[in] const OptimizerFlag Optimise mode, Support several models: 
                  Merge_H_X: Optimizing continuous H or X gate
				  Merge_U3: merge continues single gates to a U3 gate
				  Merge_RX: merge continues RX gates
				  Merge_RY: merge continues RY gates
				  Merge_RZ: merge continues RZ gates
* @return     void
*/
void sub_cir_optimizer(QCircuit& src_cir, std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec, const OptimizerFlag mode = QCircuitOPtimizer::Merge_H_X);
void sub_cir_optimizer(QProg& src_prog, std::vector<std::pair<QCircuit, QCircuit>> optimizer_cir_vec, const OptimizerFlag mode = QCircuitOPtimizer::Merge_H_X);

template <typename T>
void cir_optimizer_by_config(T &src_cir, const std::string config_data = CONFIG_PATH, const OptimizerFlag mode = QCircuitOPtimizer::Merge_H_X) {
	std::vector<std::pair<QCircuit, QCircuit>> optimitzer_cir;
	QCircuitOptimizerConfig tmp_config_reader(config_data);
	tmp_config_reader.get_replace_cir(optimitzer_cir);

	sub_cir_optimizer(src_cir, optimitzer_cir, mode);
}

QPANDA_END
#endif // QCIRCUIT_OPTIMIZE_H