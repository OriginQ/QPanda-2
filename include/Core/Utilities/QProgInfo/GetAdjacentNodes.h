#ifndef _GET_ADJACENT_NODES_H
#define _GET_ADJACENT_NODES_H

#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

/**
* @brief Get information about adjacent nodes
* @ingroup Utilities
*/
class AdjacentQGates : public TraverseByNodeIter
{
	enum TraversalStatue
	{
		HAVE_NOT_FOUND_TARGET_NODE = 0, // 0: init satue(haven't found the target node)
		TO_FIND_BACK_NODE, // 1: found the target node,
		FOUND_ALL_ADJACENT_NODE //  2: found enough
	};

	class AbstractTraversalStatueInterface
	{
	public:
		virtual void handle_QGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handle_QMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handle_QReset(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void on_enter_QIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void on_leave_QIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void on_enter_QWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void on_leave_QWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handle_classical_prog(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual TraversalStatue get_statue() const = 0;
	};

	class HaveNotFoundTargetNode : public AbstractTraversalStatueInterface
	{
	public:
		HaveNotFoundTargetNode(AdjacentQGates &parent, TraversalStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~HaveNotFoundTargetNode() {}

		void handle_QGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

		void handle_QMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
		void handle_QReset(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

		void on_enter_QIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//do nothing
		}

		void on_leave_QIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//clear front nodeIter
			m_parent.update_front_iter(NodeIter(), cir_param);
		}

		void on_enter_QWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//clear front nodeIter
			m_parent.update_front_iter(NodeIter(), cir_param);
		}
		void on_leave_QWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//clear front nodeIter
			m_parent.update_front_iter(NodeIter(), cir_param);
		}

		void handle_classical_prog(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//do nothing
		}

		TraversalStatue get_statue() const { return m_statue; }

	private:
		AdjacentQGates &m_parent;
		const TraversalStatue m_statue;
	};

	class ToFindBackNode : public AbstractTraversalStatueInterface
	{
	public:
		ToFindBackNode(AdjacentQGates &parent, TraversalStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~ToFindBackNode() {}

		void handle_QGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.update_back_iter(cur_node_iter, cir_param);
			m_parent.change_traversal_statue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void handle_QMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.update_back_iter(cur_node_iter, cir_param);
			m_parent.change_traversal_statue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}

		void handle_QReset(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.update_back_iter(cur_node_iter, cir_param);
			m_parent.change_traversal_statue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}

		void on_enter_QIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.update_back_iter(NodeIter(), cir_param);
			m_parent.change_traversal_statue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void on_leave_QIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			// do nothing
		}
		void on_enter_QWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.update_back_iter(NodeIter(), cir_param);
			m_parent.change_traversal_statue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void on_leave_QWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.update_back_iter(NodeIter(), cir_param);
			m_parent.change_traversal_statue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void handle_classical_prog(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//do nothing
		}

		TraversalStatue get_statue() const { return m_statue; }

	private:
		AdjacentQGates &m_parent;
		const TraversalStatue m_statue;
	};

	class FoundAllAdjacentNode : public AbstractTraversalStatueInterface
	{
	public:
		FoundAllAdjacentNode(AdjacentQGates &parent, TraversalStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~FoundAllAdjacentNode() {}

		TraversalStatue get_statue() const { return m_statue; }

	private:
		AdjacentQGates &m_parent;
		const TraversalStatue m_statue;
	};

public:
	AdjacentQGates(QProg prog, NodeIter &nodeItr)
		:m_prog(prog)
		, m_target_node_itr(nodeItr)
		, m_traversal_statue(nullptr)
	{}
	~AdjacentQGates() {
		if (nullptr != m_traversal_statue)
		{
			delete m_traversal_statue;
		}
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		m_traversal_statue->handle_QGate(cur_node, parent_node, cir_param, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		m_traversal_statue->handle_QMeasure(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		m_traversal_statue->handle_QReset(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		// handle classical prog
		m_traversal_statue->handle_classical_prog(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}

	virtual void traverse_qprog();

	void update_front_iter(const NodeIter &itr, const QCircuitParam &cir_param) { 
		_update_node_info(m_front_node, itr, cir_param);
	}

	void update_back_iter(const NodeIter &itr, const QCircuitParam &cir_param) {
		_update_node_info(m_back_node, itr, cir_param);
	}

	GateType get_node_ype(const NodeIter &ter);

	std::string get_node_type_str(const NodeIter &ter);

	std::string get_back_node_type_str() { 
		if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_statue->get_statue())
		{
			return std::string("Null");
		}
		return get_node_type_str(m_back_node.m_iter); 
	}

	std::string get_front_node_type_str() {
		if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_statue->get_statue())
		{
			return std::string("Null");
		}
		return get_node_type_str(m_front_node.m_iter);
	}

	static bool is_sub_prog_node(const std::shared_ptr<QNode> &node) {
		const NodeType t = node->getNodeType();
		return  ((t == CIRCUIT_NODE) || (t == PROG_NODE));
	}
	static bool is_flow_ctrl_node(const std::shared_ptr<QNode> &node) {
		const NodeType t = node->getNodeType();
		return  ((t == WHILE_START_NODE) || (t == QIF_START_NODE));
	}

	bool is_valid_node_type(const NodeIter &itr) { return is_valid_node_type((*itr)->getNodeType()); }
	bool is_valid_node_type(const NodeType t) { return ((GATE_NODE == t) || (MEASURE_GATE == t)); }

	const NodeInfo& get_front_node() { return m_front_node; }
	const NodeInfo& get_back_node() { return m_back_node; }

	void change_traversal_statue(AbstractTraversalStatueInterface* s) {
		if (nullptr != m_traversal_statue)
		{
			delete m_traversal_statue;
		}
		m_traversal_statue = s;
	}

protected:
	void _update_node_info(NodeInfo& node_info, const NodeIter &itr, const QCircuitParam &cir_param) {
		if (nullptr == itr.getPCur())
		{
			node_info.reset();
			return;
		}

		node_info.m_iter = itr;
		std::shared_ptr<QNode> p_node = *itr;
		node_info.m_node_type = p_node->getNodeType();
		if (GATE_NODE == node_info.m_node_type)
		{
			auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(p_node);
			node_info.m_gate_type = (GateType)(p_gate->getQGate()->getGateType());
			node_info.m_is_dagger = p_gate->isDagger() ^ (cir_param.m_is_dagger);
			p_gate->getQuBitVector(node_info.m_target_qubits);
			p_gate->getControlVector(node_info.m_control_qubits);
		}
		else if (CIRCUIT_NODE == node_info.m_node_type)
		{
			auto p_circuit = std::dynamic_pointer_cast<AbstractQuantumCircuit>(p_node);
			node_info.m_is_dagger = p_circuit->isDagger() ^ (cir_param.m_is_dagger);
			p_circuit->getControlVector(node_info.m_control_qubits);
		}
		else if (MEASURE_GATE == node_info.m_node_type)
		{
			auto p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(p_node);
			node_info.m_target_qubits.push_back(p_measure->getQuBit());
		}
		else if (RESET_NODE == node_info.m_node_type)
		{
			auto p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(p_node);
			node_info.m_target_qubits.push_back(p_reset->getQuBit());
		}
	}

private:
	QProg m_prog;
	const NodeIter m_target_node_itr;
	NodeInfo m_front_node;
	NodeIter m_cur_iter;
	NodeInfo m_back_node;
	std::shared_ptr<QNode> m_last_parent_node_itr;
	AbstractTraversalStatueInterface* m_traversal_statue;
};

QPANDA_END

#endif