#ifndef PROCESS_ON_TRAVERSING_H
#define PROCESS_ON_TRAVERSING_H
#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <vector>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/TopologSequence.h"
#include <memory>
#include "Core/Utilities/QProgInfo/QGateCounter.h"
#include "Core/Utilities//Tools/JsonConfigParam.h"

QPANDA_BEGIN

#define MAX_LAYER 0xEFFFFFFFFFFFFFFF

struct OptimizerNodeInfo : public NodeInfo
{
	size_t m_layer;
	int m_type;
	std::shared_ptr<QNode> m_parent_node;
	int m_sub_graph_index;

	OptimizerNodeInfo(const NodeIter iter, size_t layer, QVec target_qubits, QVec control_qubits, 
		int type, std::shared_ptr<QNode> parent_node, const bool dagger)
		:NodeInfo(iter, target_qubits, control_qubits, type, dagger), m_layer(layer)
		, m_parent_node(parent_node), m_sub_graph_index(-1), m_type(type)
	{}

	void reset() {
		/*m_parent_node will not be nullptr for gate node */
		auto node_type = m_parent_node->getNodeType();
		switch (node_type)
		{
		case CIRCUIT_NODE:
			(std::dynamic_pointer_cast<AbstractQuantumCircuit>(m_parent_node))->deleteQNode(m_iter);
			break;

		case PROG_NODE:
			(std::dynamic_pointer_cast<AbstractQuantumProgram>(m_parent_node))->deleteQNode(m_iter);
			break;

		default:
			QCERR_AND_THROW(run_fail, "Error: failed to delete target QNode, Node type error.");
			break;
		}
		
		NodeInfo::reset();
	}

	void insert_QNode(std::shared_ptr<QNode> node) {
		/*m_parent_node will not be nullptr for gate node */
		auto node_type = m_parent_node->getNodeType();
		switch (node_type)
		{
		case CIRCUIT_NODE:
			(std::dynamic_pointer_cast<AbstractQuantumCircuit>(m_parent_node))->insertQNode(m_iter, node);
			break;

		case PROG_NODE:
			(std::dynamic_pointer_cast<AbstractQuantumProgram>(m_parent_node))->insertQNode(m_iter, node);
			break;

		default:
			QCERR_AND_THROW(run_fail, "Error: failed to delete target QNode, Node type error.");
			break;
		}
	}

	bool operator== (const OptimizerNodeInfo& other) const {
		return ((other.m_iter == m_iter)
			&& (other.m_layer == m_layer)
			&& (other.m_target_qubits == m_target_qubits)
			&& (other.m_control_qubits == m_control_qubits)
			&& (other.m_gate_type == m_gate_type)
			&& (other.m_node_type == m_node_type)
			&& (other.m_parent_node == m_parent_node));
	}

	bool is_empty() const { return nullptr == m_iter.getPCur(); }
};

using pOptimizerNodeInfo = std::shared_ptr<OptimizerNodeInfo>;
using GatesBufferType = std::pair<size_t, std::vector<pOptimizerNodeInfo>>;
using OptimizerSink = std::map<size_t, std::vector<pOptimizerNodeInfo>>;
using SinkPos = std::map<size_t, size_t>;
using LayeredTopoSeq = TopologSequence<pOptimizerNodeInfo>;

class ProcessOnTraversing : protected TraverseByNodeIter
{
public:
	using layer_iter_seq = TopologSequence<std::pair<size_t, NodeIter>>; //size_t: layer index

public:
	ProcessOnTraversing()
		:m_min_layer(0)
	{
	}
	virtual ~ProcessOnTraversing() {}

	virtual void process(const bool on_travel_end) = 0;
	virtual void run_traversal(QProg src_prog, const QVec qubits = {});
	virtual void do_process(const bool on_travel_end) {
		if (m_cur_gates_buffer.size() == 0)
		{
			return;
		}

		process(on_travel_end);
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	virtual void gates_sink_to_topolog_sequence(OptimizerSink& gate_buf, LayeredTopoSeq& seq, const size_t max_output_layer = MAX_LAYER);

	/**
	* @brief pop gate buf to circuit
	* @ingroup Utilities
	* @param[in] QProg & the output prog
	* @param[in] bool whether output all the gate node in gate buf
	* @return
	* @note
	*/
	virtual void clean_gate_buf_to_cir(QProg &cir, bool b_clean_all_buf = false);
	virtual void clean_gate_buf(bool b_clean_all_buf = false);
	virtual void drop_gates(const size_t max_drop_layer);

	virtual void seq_to_cir(layer_iter_seq &tmp_seq, QProg& prog, const size_t start_layer_to_cir, const size_t max_output_layer);
	virtual void seq_to_cir(layer_iter_seq &tmp_seq, QProg& prog);
	virtual void add_node_to_seq(layer_iter_seq &tmp_seq, NodeIter node_iter, const size_t layer);

protected:
	virtual void add_gate_to_buffer(NodeIter iter, QCircuitParam &cir_param, std::shared_ptr<QNode> parent_node, OptimizerSink& gates_buffer);
	virtual void add_non_gate_to_buffer(NodeIter iter, NodeType node_type, QVec gate_qubits, QCircuitParam &cir_param,
		OptimizerSink& gates_buffer, std::shared_ptr<QNode> parent_node = nullptr);
	size_t get_node_layer(QVec gate_qubits, OptimizerSink& gate_buffer);
	size_t get_node_layer(const std::vector<int>& gate_qubits, OptimizerSink& gate_buffer);
	virtual size_t get_min_include_layers();
	void init_gate_buf();
	virtual void append_data_to_gate_buf(std::vector<pOptimizerNodeInfo>& gate_buf,
		pOptimizerNodeInfo p_node, const size_t qubit_i);

protected:
	QVec m_qubits;
	OptimizerSink m_cur_gates_buffer;
	SinkPos m_cur_buffer_pos;
	size_t m_min_layer;
};

struct PressedCirNode
{
	pOptimizerNodeInfo m_cur_node;
	std::vector<pOptimizerNodeInfo> m_relation_pre_nodes;
	std::vector<pOptimizerNodeInfo> m_relation_successor_nodes;
};

using PressedTopoSeq = TopologSequence<PressedCirNode>;
using PressedLayer = SeqLayer<PressedCirNode>;
using PressedNode = SeqNode<PressedCirNode>;

PressedTopoSeq get_pressed_layer(QProg src_prog);

/**
* @brief Program layering.
* @ingroup Utilities
* @param[in] prog  the source prog
* @param[in] bool Whether to enable low-frequency qubit compensation, default is false
* @param[in] const std::string config data, @See JsonConfigParam::load_config()
* @return the TopologSequence
*/
LayeredTopoSeq prog_layer(QProg src_prog, const bool b_enable_qubit_compensation = false, const std::string config_data = CONFIG_PATH);

QPANDA_END
#endif // PROCESS_ON_TRAVERSING_H