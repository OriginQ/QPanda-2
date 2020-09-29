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

QPANDA_BEGIN

#define MAX_LAYER 0xEFFFFFFFFFFFFFFF

struct OptimizerNodeInfo
{
	NodeIter m_iter;
	size_t m_layer;
	QVec m_target_qubits;
	QVec m_ctrl_qubits;
	GateType m_type;
	std::shared_ptr<QNode> m_parent_node;
	int m_sub_graph_index;
	bool m_dagger;

	OptimizerNodeInfo(const NodeIter iter, size_t layer, QVec target_qubits, QVec control_qubits, 
		GateType type, std::shared_ptr<QNode> parent_node, const bool dagger)
		:m_iter(iter), m_layer(layer), m_target_qubits(target_qubits) 
		, m_ctrl_qubits(control_qubits), m_type(type), m_parent_node(parent_node)
		, m_sub_graph_index(-1), m_dagger(dagger)
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
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to delete target QNode, Node type error.");
			break;
		}
		m_type = GATE_UNDEFINED;
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
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to delete target QNode, Node type error.");
			break;
		}
	}

	bool operator== (const OptimizerNodeInfo& other) const {
		return ((other.m_iter == m_iter)
			&& (other.m_layer == m_layer)
			&& (other.m_target_qubits == m_target_qubits)
			&& (other.m_ctrl_qubits == m_ctrl_qubits)
			&& (other.m_type == m_type)
			&& (other.m_parent_node == m_parent_node));
	}
};

using pOptimizerNodeInfo = std::shared_ptr<OptimizerNodeInfo>;
using GatesBufferType = std::pair<size_t, std::list<pOptimizerNodeInfo>>;
using OptimizerSink = std::map<size_t, std::list<pOptimizerNodeInfo>>;

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

	virtual void gates_sink_to_topolog_sequence(OptimizerSink& gate_buf, TopologSequence<pOptimizerNodeInfo>& seq, const size_t max_output_layer = MAX_LAYER);

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
	virtual size_t get_node_layer(QVec gate_qubits, OptimizerSink& gate_buffer);
	virtual size_t get_min_include_layers();
	void init_gate_buf() {
		for (const auto& item : m_qubits)
		{
			m_cur_gates_buffer.insert(GatesBufferType(item->getPhysicalQubitPtr()->getQubitAddr(), std::list<pOptimizerNodeInfo>()));
		}
	}

protected:
	QVec m_qubits;
	OptimizerSink m_cur_gates_buffer;
	size_t m_min_layer;
};

class QProgLayer : protected ProcessOnTraversing
{
public:
	QProgLayer(){}
	~QProgLayer() {}

	void layer(QProg src_prog) { run_traversal(src_prog); }
	void process(const bool on_travel_end = false) override;
	void append_topolog_seq(TopologSequence<pOptimizerNodeInfo>& tmp_seq);

	const TopologSequence<pOptimizerNodeInfo>& get_topo_seq() { return m_topolog_sequence; }

private:
	TopologSequence<pOptimizerNodeInfo> m_topolog_sequence;
};

/**
* @brief Program layering.
* @ingroup Utilities
* @param[in] prog  the source prog
* @param[in] bool Whether to start single gate exclusive layer, default is false
* @return the TopologSequence
*/
const TopologSequence<pOptimizerNodeInfo> prog_layer(QProg src_prog, const bool b_double_gate_one_layer = false);

QPANDA_END
#endif // PROCESS_ON_TRAVERSING_H