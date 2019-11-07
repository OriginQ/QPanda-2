/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

add by zhaody

*/

#ifndef _QCIRCUIT_INFO_H
#define _QCIRCUIT_INFO_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgDAG.h"
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include "Core/QuantumCircuit/QuantumGate.h"

QPANDA_BEGIN

class QCircuitParam
{
public:
	QCircuitParam(){
		m_is_dagger = false;
	}

	QCircuitParam(const QCircuitParam& rhs){}


	std::shared_ptr<QCircuitParam> clone() {
		return std::make_shared<QCircuitParam>(*this);
	}

	void append_control_qubits(const QVec &ctrl_qubits) {
		m_control_qubits.insert(m_control_qubits.end(), ctrl_qubits.begin(), ctrl_qubits.end());
	}

	// real_append_qubits = append_qubits - target_qubits
	static QVec get_real_append_qubits(QVec append_qubits, QVec target_qubits) {
		if (0 == target_qubits.size())
		{
			return append_qubits;
		}

		std::sort(append_qubits.begin(), append_qubits.end());
		std::sort(target_qubits.begin(), target_qubits.end());

		QVec result_vec;
		set_difference(append_qubits.begin(), append_qubits.end(), target_qubits.begin(), target_qubits.end(), std::back_inserter(result_vec));
		return result_vec;
	}

	bool m_is_dagger;
	std::vector<QPanda::Qubit*> m_control_qubits;
};

class TraverseByNodeIter : public TraversalInterface<QCircuitParam&, NodeIter&>
{
public:
	TraverseByNodeIter(QProg &prog)
		:m_prog(prog)
	{}
	~TraverseByNodeIter() {}

public:
	virtual void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		//handle QGate node
	}

	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		//handle measure node
	}

	virtual void execute(std::shared_ptr<AbstractClassicalProg> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		// handle classical prog
	}

	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		// handle flow control node
		Traversal::traversal(cur_node, *this, cir_param, cur_node_iter);
	}

	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	virtual void execute(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	virtual void traverse_qprog();

protected:
	QProg &m_prog;
};

class GetAllNodeType : public TraverseByNodeIter
{
public:
	GetAllNodeType(QProg &src_prog)
		:TraverseByNodeIter(src_prog), m_indent_cnt(0)
	{
	}
	~GetAllNodeType() {}

	std::string printNodesType() {
		std::cout << m_output_str << std::endl;
		return m_output_str;
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		// handle classical prog
		sub_circuit_indent();
		m_output_str.append(">>ClassicalProgNode ");
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override ;
	void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override ;

protected:
	std::string get_indent_str() {
		std::string ret_str = "\n";
		for (size_t i = 0; i < m_indent_cnt; ++i)
		{
			ret_str.append("  ");
		}

		return ret_str;
	}

	void sub_circuit_indent() {
		size_t line_start = m_output_str.find_last_of('\n');
		if ((m_output_str.length() - line_start) > 80)
		{
			m_output_str.append(get_indent_str());
		}
	}

private:
	size_t m_indent_cnt;
	std::string m_output_str;
};

class PickUpNodes : public TraverseByNodeIter
{
public:
	PickUpNodes(QProg &output_prog, QProg &src_prog, const NodeIter &node_itr_start, const NodeIter &node_itr_end)
		:TraverseByNodeIter(src_prog), m_output_prog(output_prog),
		m_start_iter(node_itr_start),
		m_end_iter(node_itr_end),
		m_b_picking(false), m_b_pickup_end(false), m_b_pick_measure_node(false), m_b_need_dagger(false)
	{}
	~PickUpNodes() {}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		// handle classical prog
		judgeSubCirNodeIter(cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		if (judgeSubCirNodeIter(cur_node_iter))
		{
			TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
		}
	}

	void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		if (judgeSubCirNodeIter(cur_node_iter))
		{
			TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
		}
	}

	void setPickUpMeasureNode(bool b) { m_b_pick_measure_node = b; }
	void setDaggerFlag(bool b) { m_b_need_dagger = b; }
	void reverse_dagger_circuit();

	//return true on Legal
	static bool check_control_qubits(QGate& gate);
	static void no_dagger_gate(QGate& gate);

protected:
	void pickQGateNode(const NodeIter cur_node_iter, QCircuitParam &cir_param);
	void pickQMeasureNode(const NodeIter cur_node_iter);

	template<class Function, class... Args>
	inline auto pickUp(const NodeIter cur_node_iter, Function &&func, Args && ... args) {
		if (m_b_pickup_end)
		{
			return;
		}
		else if (m_b_picking)
		{
			func(std::forward<Args>(args)...);
		}
		else
		{
			if (cur_node_iter == m_start_iter)
			{
				m_b_picking = true;
				func(std::forward<Args>(args)...);
			}
			else if (cur_node_iter == m_end_iter)
			{
				m_b_picking = true;
				m_end_iter = m_start_iter;
				func(std::forward<Args>(args)...);
			}
		}
	}

	inline bool judgeSubCirNodeIter(const NodeIter& sub_circuit_node_iter) {
		if (m_b_pickup_end)
		{
			return false;
		}
		else if (m_b_picking)
		{
			if (sub_circuit_node_iter == m_end_iter)
			{
				m_b_pickup_end = true;
			}
		}
		else
		{
			if (sub_circuit_node_iter == m_start_iter)
			{
				m_b_picking = true;
			}
		}
		return true;
	}

private:
	bool m_b_pick_measure_node;
	QProg &m_output_prog;
	NodeIter m_start_iter;
	NodeIter m_end_iter;
	bool m_b_picking;
	bool m_b_pickup_end;
	bool m_b_need_dagger;
};

class JudgeTwoNodeIterIsSwappable : public TraverseByNodeIter
{
	enum ResultStatue
	{
		INIT = 0,
		JUST_FOUND_ONE_NODE,
		NEED_JUDGE_LAYER,
		CAN_NOT_BE_EXCHANGED,
		COULD_BE_EXCHANGED
	};

	class AbstractJudgeStatueInterface
	{
	public:
		virtual void handleQGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handleQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void onEnterFlowCtrlNode() {}
		virtual void onLeaveFlowCtrlNode() {}
		virtual void onEnterCircuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) {}
		virtual void onLeaveCircuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) {}
		virtual void onTraversalEnd() {}
		virtual ResultStatue getStatue() const = 0;
	};

	class CoubleBeExchange : public AbstractJudgeStatueInterface
	{
	public:
		CoubleBeExchange(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~CoubleBeExchange() {}

		ResultStatue getStatue() const { return m_statue; }

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
	};

	class CanNotBeExchange : public AbstractJudgeStatueInterface
	{
	public:
		CanNotBeExchange(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~CanNotBeExchange() {}

		ResultStatue getStatue() const { return m_statue; }

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
	};

	class OnJudgeLayerInfo : public AbstractJudgeStatueInterface
	{
	public:
		OnJudgeLayerInfo(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~OnJudgeLayerInfo() {}

		void onTraversalEnd() override {
			//layer
			m_parent.judgeLayerInfo();
		}
		ResultStatue getStatue() const { return m_statue; }

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
	};

	class OnPickUpNode : public AbstractJudgeStatueInterface
	{
	public:
		OnPickUpNode(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~OnPickUpNode() {}

		void handleQGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			m_parent.pickNode(cur_node_iter);
		}
		void handleQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			m_parent.pickNode(cur_node_iter);
		}
		void onEnterFlowCtrlNode() override {
			judgePickStatue();
		}
		void onLeaveFlowCtrlNode() override {
			judgePickStatue();
		}
		void onEnterCircuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) override {
			on_circuit(cur_node, cir_param);
		}
		void onLeaveCircuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) override {
			on_circuit(cur_node, cir_param);
		}

		void onTraversalEnd() override {
			judgePickStatue();
		}
		ResultStatue getStatue() const { return m_statue; }

	protected:
		void judgePickStatue() {
			if (m_parent.isFoundAllNodeIter())
			{
				m_parent.changeStatue(new OnJudgeLayerInfo(m_parent, NEED_JUDGE_LAYER));
			}
			else
			{
				m_parent.changeStatue(new CanNotBeExchange(m_parent, CAN_NOT_BE_EXCHANGED));
			}
		}

		void on_circuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) {
			//if current node is dagger or be controled, clear che m_parent.pickProg;
			bool cur_node_is_dagger = cur_node->isDagger();
			QVec control_qubit_vector;
			bool cur_node_is_controled = cur_node->getControlVector(control_qubit_vector);
			if (cur_node_is_dagger || (cur_node_is_controled))
			{
				judgePickStatue();
			}
		}

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
	};

	class OnInitStatue : public AbstractJudgeStatueInterface
	{
	public:
		OnInitStatue(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s), m_need_update_layer_start_iter(true)
		{}
		void handleQGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			onPick(cur_node_iter);
		}
		void handleQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			onPick(cur_node_iter);
		}
		void onEnterFlowCtrlNode() override {
			m_parent.clearPickProg();
		}
		void onLeaveFlowCtrlNode() override {
			m_parent.clearPickProg();
		}
		void onEnterCircuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) override {
			on_circuit(cur_node, cir_param);
		}
		void onLeaveCircuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) override {
			on_circuit(cur_node, cir_param);
		}

		ResultStatue getStatue() const { return m_statue; }

	protected:
		void onPick(NodeIter cur_node_iter) {
			m_parent.pickNode(cur_node_iter);

			if (m_parent.m_b_found_first_iter || m_parent.m_b_found_first_iter)
			{
				//change statue
				m_parent.changeStatue(new OnPickUpNode(m_parent, JUST_FOUND_ONE_NODE));
			}
		}

		void on_circuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) {
			//if current node is dagger or be controled, clear che m_parent.pickProg;
			bool cur_node_is_dagger = cur_node->isDagger();
			QVec control_qubit_vector;
			bool cur_node_is_controled = cur_node->getControlVector(control_qubit_vector);
			if (cur_node_is_dagger || (cur_node_is_controled))
			{
				m_parent.clearPickProg();
			}
		}

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		NodeIter m_layer_start_iter;
		const ResultStatue m_statue;
		bool m_need_update_layer_start_iter;
	};

public:
	JudgeTwoNodeIterIsSwappable(QProg &prog, NodeIter &nodeItr_1, NodeIter &nodeItr_2)
		: TraverseByNodeIter(prog), m_nodeItr1(nodeItr_1), m_nodeItr2(nodeItr_2), m_judge_statue(nullptr),
		m_result(INIT), m_b_found_first_iter(false), m_b_found_second_iter(false), m_b_dagger_circuit(false)
	{
	}
	~JudgeTwoNodeIterIsSwappable() {
		if (nullptr != m_judge_statue)
		{
			delete m_judge_statue;
		}
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		m_judge_statue->handleQGate(cur_node, parent_node, cir_param, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		m_judge_statue->handleQMeasure(cur_node, parent_node, cir_param, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		// handle classical prog
		if ((cur_node_iter == m_nodeItr1) || (cur_node_iter == m_nodeItr2))
		{
			changeStatue(new CanNotBeExchange(*this, CAN_NOT_BE_EXCHANGED));
		}
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}

	bool getResult();

	/**
	* @brief
	* @param[out] result if the two Iters on the same layer and could be exchanged, result=true, or else false.
	* @return if any any error happened, return <0 ,else return 0
	*/
	int judgeLayerInfo();

	void changeStatue(AbstractJudgeStatueInterface* s) {
		if (nullptr != m_judge_statue)
		{
			delete m_judge_statue;
		}
		m_judge_statue = s;

		if (NEED_JUDGE_LAYER == m_judge_statue->getStatue())
		{
			//layer
			judgeLayerInfo();
		}

		m_result = m_judge_statue->getStatue();
	}

	void traverse_qprog() override;

	void clearPickProg() { m_pick_prog.clear(); }

	void pickNode(const NodeIter iter);

	bool isFoundAllNodeIter() { return (m_b_found_first_iter && m_b_found_second_iter); }

	friend class JudgeStatueInterface;

private:
	ResultStatue m_result;
	NodeIter m_nodeItr1;
	NodeIter m_nodeItr2;
	bool m_b_found_first_iter;
	bool m_b_found_second_iter;
	bool m_b_dagger_circuit;
	QProg m_pick_prog;
	AbstractJudgeStatueInterface *m_judge_statue;
};

#if 1
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
		virtual void handleQGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handleQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void onEnterQIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void onLeaveQIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void onEnterQWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void onLeaveQWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handleQClassicalProg(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual TraversalStatue getStatue() const = 0;
	};

	class HaveNotFoundTargetNode : public AbstractTraversalStatueInterface
	{
	public:
		HaveNotFoundTargetNode(AdjacentQGates &parent, TraversalStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~HaveNotFoundTargetNode() {}

		void handleQGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

		void handleQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

		void onEnterQIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//do nothing
		}

		void onLeaveQIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//clear front nodeIter
			m_parent.updateFrontIter(NodeIter());
		}

		void onEnterQWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//clear front nodeIter
			m_parent.updateFrontIter(NodeIter());
		}
		void onLeaveQWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//clear front nodeIter
			m_parent.updateFrontIter(NodeIter());
		}

		void handleQClassicalProg(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//do nothing
		}

		TraversalStatue getStatue() const { return m_statue; }

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

		void handleQGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.updateBackIter(cur_node_iter);
			m_parent.changeTraversalStatue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void handleQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.updateBackIter(cur_node_iter);
			m_parent.changeTraversalStatue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void onEnterQIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.updateBackIter(NodeIter());
			m_parent.changeTraversalStatue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void onLeaveQIf(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			// do nothing
		}
		void onEnterQWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.updateBackIter(NodeIter());
			m_parent.changeTraversalStatue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void onLeaveQWhile(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			m_parent.updateBackIter(NodeIter());
			m_parent.changeTraversalStatue(new FoundAllAdjacentNode(m_parent, FOUND_ALL_ADJACENT_NODE));
		}
		void handleQClassicalProg(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//do nothing
		}

		TraversalStatue getStatue() const { return m_statue; }

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

		TraversalStatue getStatue() const { return m_statue; }

	private:
		AdjacentQGates &m_parent;
		const TraversalStatue m_statue;
	};

public:
	AdjacentQGates(QProg &prog, NodeIter &nodeItr)
		:TraverseByNodeIter(prog)
		, m_target_node_itr(nodeItr)
		, m_prog(prog), m_traversal_statue(nullptr)
	{}
	~AdjacentQGates() {
		if (nullptr != m_traversal_statue)
		{
			delete m_traversal_statue;
		}
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		m_traversal_statue->handleQGate(cur_node, parent_node, cir_param, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		m_traversal_statue->handleQMeasure(cur_node, parent_node, cir_param, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		// handle classical prog
		m_traversal_statue->handleQClassicalProg(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void traverse_qprog() override;

	void updateFrontIter(const NodeIter &itr) { m_front_iter = itr; }
	void updateBackIter(const NodeIter &itr) { m_back_iter = itr; }
	GateType getFrontIterNodeType() {
		if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_statue->getStatue())
		{
			return GATE_UNDEFINED;
		}
		return getItrNodeType(m_front_iter);
	}
	GateType getBackIterNodeType() { return getItrNodeType(m_back_iter); }

	GateType getItrNodeType(const NodeIter &ter);

	std::string getItrNodeTypeStr(const NodeIter &ter);
	std::string getBackIterNodeTypeStr() { return getItrNodeTypeStr(m_back_iter); }
	std::string getFrontIterNodeTypeStr() {
		if (HAVE_NOT_FOUND_TARGET_NODE == m_traversal_statue->getStatue())
		{
			return std::string("Null");
		}
		return getItrNodeTypeStr(m_front_iter);
	}

	static bool isSubProgNode(const std::shared_ptr<QNode> &node) {
		const NodeType t = node->getNodeType();
		return  ((t == CIRCUIT_NODE) || (t == PROG_NODE));
	}
	static bool isFlowCtrlNode(const std::shared_ptr<QNode> &node) {
		const NodeType t = node->getNodeType();
		return  ((t == WHILE_START_NODE) || (t == QIF_START_NODE));
	}

	bool isValidNodeType(const NodeIter &itr) { return isValidNodeType((*itr)->getNodeType()); }
	bool isValidNodeType(const NodeType t) { return ((GATE_NODE == t) || (MEASURE_GATE == t)); }

	const NodeIter& getFrontIter() { return m_front_iter; }
	const NodeIter& getBackIter() { return m_back_iter; }

public:
	void changeTraversalStatue(AbstractTraversalStatueInterface* s) {
		if (nullptr != m_traversal_statue)
		{
			delete m_traversal_statue;
		}
		m_traversal_statue = s;
	}

private:
	QProg &m_prog;
	const NodeIter m_target_node_itr;
	NodeIter m_front_iter;
	NodeIter m_cur_iter;
	NodeIter m_back_iter;
	std::shared_ptr<QNode> m_last_parent_node_itr;
	AbstractTraversalStatueInterface* m_traversal_statue;
};
#endif

class QprogToMatrix
{
	using gateAndQubitsItem_t = std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>;
	using gateQubitInfo_t = std::vector<gateAndQubitsItem_t>;
	using calcUintItem_t = std::pair<QStat, std::vector<int>>;
	using calcUnitVec_t = std::vector<calcUintItem_t>;
	
	class MatrixOfOneLayer
	{
	public:
		MatrixOfOneLayer(SequenceLayer& layer, const QProgDAG& prog_dag, std::vector<int> &qubits_in_use);
		void merge_double_gate();
		void merge_calc_unit();
		void merge_controled_gate();
		void merge_sing_gate();

	protected:
		void reverse_ctrl_gate_matrix(QStat& src_mat, const GateType &gate_T);
		QStat reverseCtrlGateMatrixCX(QStat& src_mat);
		QStat reverseCtrlGateMatrixCU(QStat& src_mat);
		void mergeToCalcUnit(std::vector<int>& qubits, QStat& gate_mat, calcUnitVec_t &calc_unit_vec, gateQubitInfo_t &single_qubit_gates);
		void getStrideOverQubits(const std::vector<int> &qgate_used_qubits, std::vector<int> &stride_over_qubits);
		void tensorByMatrix(QStat& src_mat, const QStat& tensor_mat);
		void tensorByQGate(QStat& src_mat, std::shared_ptr<AbstractQGateNode> &pGate);
		bool check_cross_calc_unit(calcUnitVec_t& calc_unit_vec, calcUnitVec_t::iterator target_calc_unit_itr);
		void merge_two_crossed_matrix(const calcUintItem_t& calc_unit_1, const calcUintItem_t& calc_unit_2, calcUintItem_t& result);
		void build_standard_control_gate_matrix(const QStat& src_mat, const int qubit_number, QStat& result_mat);
		void swap_two_qubit_on_matrix(QStat& src_mat, const int mat_qubit_start, const int mat_qubit_end, const int qubit_1, const int qubit_2);

	public:
		QStat m_current_layer_mat;
		gateQubitInfo_t m_double_qubit_gates;//double qubit gate vector
		gateQubitInfo_t m_single_qubit_gates;//single qubit gate vector
		gateQubitInfo_t m_controled_gates;//controled qubit gate vector
		calcUnitVec_t m_calc_unit_vec;
		const QStat m_mat_I;
		std::vector<int> &m_qubits_in_use; //the number of all the qubits in the target QCircuit.
	};

public:
	QprogToMatrix(QProg& p)
		:m_prog(p)
	{}
	~QprogToMatrix() {}

	QStat getMatrix();
	QStat getMatrixOfOneLayer(SequenceLayer& layer, const QProgDAG& prog_dag);

private:
	QProg& m_prog;
	std::vector<int> m_qubits_in_use;
};

/**
* @brief  judge the Qgate if match the target topologic structure of quantum circuit
* @param[in]  vector<vector<int>>& the target topologic structure of quantum circuit
* @return     if the Qgate match the target topologic structure return true, or else return false
* @see XmlConfigParam::readAdjacentMatrix(TiXmlElement *, int&, std::vector<std::vector<int>>&)
*/
bool isMatchTopology(const QGate& gate, const std::vector<std::vector<int>>& vecTopoSt);

/**
* @brief  get the adjacent quantum gates's(the front one and the back one) type
* @param[in] nodeItr  the specialed NodeIter
* @param[out] std::vector<NodeIter> frontAndBackIter the front iter and the back iter
* @return result string.
* @see
*/
std::string getAdjacentQGateType(QProg &prog, NodeIter &nodeItr, std::vector<NodeIter>& frontAndBackIter);

/**
* @brief  judge the specialed two NodeIters whether can be exchanged
* @param[in] nodeItr1 the first NodeIter
* @param[in] nodeItr2 the second NodeIter
* @return if the two NodeIters can be exchanged, return true, otherwise retuen false.
* @see
*/
bool isSwappable(QProg &prog, NodeIter &nodeItr1, NodeIter &nodeItr2);

/**
* @brief  judge if the target node is a base QGate type
* @param[in] nodeItr the target NodeIter
* @return if the target node is a base QGate type, return true, otherwise retuen false.
* @see
*/
bool isSupportedGateType(const NodeIter &nodeItr);

/**
* @brief  get the target matrix between the input two Nodeiters
* @param[in] nodeItrStart the start NodeIter
* @param[in] nodeItrEnd the end NodeIter
* @return the target matrix include all the QGate's matrix (multiply).
* @see
*/
QStat getCircuitMatrix(QProg srcProg, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter());

/**
* @brief  pick up the nodes of srcProg between nodeItrStart and  nodeItrEnd to outPutProg
* @param[out] outPutProg  the output prog
* @param[in] srcProg The source prog
* @param[in] nodeItrStart The start pos of source prog
* @param[in] nodeItrEnd The end pos of source prog
* @param[in] bPickMeasure if bPickMeasure is true pick up measure node, or else ignore.
* @param[in] bDagger daggger flag
* @ Note: If there are any Qif/Qwhile nodes between nodeItrStart and nodeItrEnd,
		  Or the nodeItrStart and the nodeItrEnd are in different sub-circuit, an exception will be throw.
*/
void pickUpNode(QProg &outPutProg, QProg &srcProg, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter(), bool bPickMeasure = false, bool bDagger = false);

/**
* @brief  Get all the used  quantum bits in the input prog
* @param[in] prog  the input prog
* @param[out] vecQuBitsInUse The vector of used quantum bits
* @ Note: All the Qif/Qwhile or other sub-circuit nodes in the input prog will be ignored.
*/
void get_all_used_qubits(QProg &prog, std::vector<int> &vecQuBitsInUse);
void get_all_used_qubits(QProg &prog, QVec &vecQuBitsInUse);

/**
* @brief  Get all the used  class bits in the input prog
* @param[in] prog  the input prog
* @param[out] vecClBitsInUse The vector of used class bits
* @ Note: All the Qif/Qwhile or other sub-circuit nodes in the input prog will be ignored.
*/
void get_all_used_class_bits(QProg &prog, std::vector<int> &vecClBitsInUse);

/**
* @brief  output all the node type of the target prog
* @param[in] the target prog
* @return return the output string
*/
std::string printAllNodeType(QProg &prog);

/**
* @brief  get gate parameter
* @param[in] pGate the target gate pointer
* @param[out] para_str parameter string
* @return
*/
void get_gate_parameter(std::shared_ptr<AbstractQGateNode> pGate, std::string& para_str);

QPANDA_END
#endif