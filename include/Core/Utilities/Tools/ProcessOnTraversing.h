#ifndef PROCESS_ON_TRAVERSING_H
#define PROCESS_ON_TRAVERSING_H
#include "Core/Utilities/QPandaNamespace.h"
#include <iostream>
#include <complex>
#include <string>
#include <vector>
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/TopologSequence.h"
#include <memory>
#include "Core/Utilities/QProgInfo/QGateCounter.h"
#include "Core/Utilities//Tools/JsonConfigParam.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
enum  ChipID
{
    Simulation = 0,
    WUYUAN_1 = 1,
    WUYUAN_2 = 2,
    WUYUAN_3 = 3
};
QPANDA_BEGIN

#define MAX_LAYER (std::numeric_limits<uint32_t>::max)()

struct OptimizerNodeInfo : public NodeInfo
{
	size_t m_layer;
	int m_type; /**< @see DAGNodeType */
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
using SinkPos = std::map<size_t, size_t>;
using GatesBufferType = std::pair<size_t, std::vector<pOptimizerNodeInfo>>;
class QubitNodesSink : public std::map<size_t, std::vector<pOptimizerNodeInfo>, std::less<size_t>>
{
public:
	using QubitNodesSinkItr = std::map<size_t, std::vector<pOptimizerNodeInfo>>::iterator;
	using QubitNodesVecItr = std::vector<pOptimizerNodeInfo>::iterator;

	void append_data(pOptimizerNodeInfo p_node, const size_t qubit_i) {
		std::vector<pOptimizerNodeInfo>& gate_buf = at(qubit_i);
		auto &tmp_pos = m_data_size.at(qubit_i);
		if (gate_buf.size() <= (tmp_pos)){
			gate_buf.emplace_back(p_node);
		}else{
			gate_buf[tmp_pos] = p_node;
		}
		++tmp_pos;
	}

	void insert(GatesBufferType qubit_nodes) {
		std::map<size_t, std::vector<pOptimizerNodeInfo>>::insert(qubit_nodes);
		m_data_size.insert(std::make_pair(qubit_nodes.first, 0));
	}

	const size_t& get_target_qubit_sink_size(size_t q) const { return m_data_size.at(q); }
	size_t& get_target_qubit_sink_size(size_t q) { return m_data_size.at(q); }

	SinkPos& get_sink_pos() { return m_data_size; }

	/*void insert_data(size_t qubit, pOptimizerNodeInfo node) {
		at(qubit).push_back(node);
		++m_data_size[qubit];
	}

	void insert_data(size_t qubit, const std::vector<pOptimizerNodeInfo>& node_vec) {
		at(qubit).insert(at(qubit).end(), node_vec.begin(), node_vec.end());
		m_data_size[qubit] += node_vec.size();
	}*/

	/** note: not include it_end
	*/
	void remove(size_t qubit, QubitNodesVecItr it_first, QubitNodesVecItr it_end) {
		const auto remove_size = it_end - it_first;
		if (remove_size > m_data_size[qubit]){
			QCERR_AND_THROW(run_fail, "Error: Iterator error deleting element from target sink.");
		}

		for (auto _itr = it_first; _itr != it_end; ++_itr){
			_itr->reset();
		}

		if (remove_size != m_data_size[qubit]){
			std::rotate(it_first, it_end, at(qubit).end());
		}
		
		m_data_size[qubit] -= remove_size;
	}

	void remove(size_t qubit, QubitNodesVecItr it_first) {
		remove(qubit, it_first, it_first + 1);
	}

protected:
	SinkPos m_data_size; /**< Number of logic gates on each qubit */
};


//using OptimizerSink = std::map<size_t, std::vector<pOptimizerNodeInfo>>;
using OptimizerSink = QubitNodesSink;
using LayeredTopoSeq = TopologSequence<pOptimizerNodeInfo>;
/* gcc can't get protected base class, change protected to public */
class ProcessOnTraversing : public TraverseByNodeIter
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
	virtual size_t get_max_buf_size();
	void init_gate_buf();
	/*virtual void append_data_to_gate_buf(std::vector<pOptimizerNodeInfo>& gate_buf,
		pOptimizerNodeInfo p_node, const size_t qubit_i);*/

protected:
	QVec m_qubits;
	OptimizerSink m_cur_gates_buffer;
	//SinkPos m_cur_buffer_pos;
	size_t m_min_layer;
};

struct PressedCirNode
{
	pOptimizerNodeInfo m_cur_node;
	std::vector<pOptimizerNodeInfo> m_relation_pre_nodes;
	std::vector<pOptimizerNodeInfo> m_relation_successor_nodes;
};

using pPressedCirNode = std::shared_ptr<PressedCirNode>;
using PressedTopoSeq = TopologSequence<pPressedCirNode>;
using PressedLayer = SeqLayer<pPressedCirNode>;
using PressedNode = SeqNode<pPressedCirNode>;

class QProgLayer : protected ProcessOnTraversing
{
public:
	QProgLayer() {}
	virtual ~QProgLayer() {}

	virtual void init() {}

	virtual void layer(QProg src_prog);

	virtual const LayeredTopoSeq& get_topo_seq() { return m_topolog_sequence; }

    void move_measure_to_last(LayeredTopoSeq& seq);

    //void prog_layer_by_double_gate(LayeredTopoSeq& seq);

protected:
	void process(const bool on_travel_end = false) override;
	void append_topolog_seq(LayeredTopoSeq& tmp_seq);
	void add_gate_to_buffer(NodeIter iter, QCircuitParam &cir_param,
		std::shared_ptr<QNode> parent_node, OptimizerSink& gates_buffer) override;

private:
	LayeredTopoSeq m_topolog_sequence;
	std::vector<std::vector<int>> m_qubit_topo_matrix;
	std::vector<int> m_high_frequency_qubits;
	size_t m_qubit_size;
};

PressedTopoSeq get_pressed_layer(QProg src_prog);

/**
* @brief Program layering.
* @ingroup Utilities
* @param[in] prog  the source prog
* @return the TopologSequence
*/
LayeredTopoSeq prog_layer(QProg src_prog);

/**
* @brief Program layering.
* @ingroup Utilities
* @param[in] prog  the source prog
* @return the TopologSequence
*/
//LayeredTopoSeq get_chip_layer(QProg src_prog, ChipID chip_id = ChipID::Simulation, QuantumMachine *quantum_machine = nullptr);
LayeredTopoSeq get_clock_layer(QProg src_prog, const std::string config_data = CONFIG_PATH);

/*new interface*/
/**
* @brief circuit_layer
* @ingroup Utilities
* @param[in] prog  the source prog
* @return  LayerInfo include circuit  total layer and every layer NodeInfo
*/

using LayerInfo = std::pair <size_t, std::vector<std::vector<NodeInfo>>>;

LayerInfo circuit_layer(QProg src_prog);


QPANDA_END
#endif // PROCESS_ON_TRAVERSING_H