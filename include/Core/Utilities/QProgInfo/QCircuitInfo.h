/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.

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

/**
* @brief Detailed information of a QProg node.
* @ingroup Utilities
*/
class NodeInfo
{
public:

	/**
	* @brief Constructor of NodeInfo
	*/
	NodeInfo()
		:m_gate_type(GATE_UNDEFINED)
		, m_node_type(NODE_UNDEFINED)
		, m_is_dagger(false)
	{}

	NodeInfo(const NodeIter iter, QVec target_qubits, QVec control_qubits,
		int type, const bool dagger)
		:m_iter(iter), m_target_qubits(target_qubits), m_control_qubits(control_qubits)
		, m_is_dagger(dagger), m_node_type(NODE_UNDEFINED), m_gate_type(GATE_UNDEFINED)
	{
		if (nullptr != iter.getPCur())
		{
			init(type, target_qubits, control_qubits);
		}
	}

	/**
	* @brief reset the node information
	*/
	virtual void reset();

private:
	virtual void init(const int type, const QVec& target_qubits, const QVec& control_qubits);

public:
	NodeIter m_iter;                 /**< the NodeIter of the node */
	NodeType m_node_type;            /**< the node type */
	GateType m_gate_type;            /**< the gate type (if the node type is gate_node) */
	bool m_is_dagger;                /**< dagger information */
	QVec m_target_qubits;            /**< Quantum bits of current node. */
	QVec m_control_qubits;           /**< control Quantum bits. */
	std::vector<int> m_cbits;
	std::vector<double> m_params;
	std::string m_name;
};

/**
* @brief Circuit Parameter information
* @ingroup Utilities
*/
class QCircuitParam
{
public:
	/**
	* @brief Constructor of QCircuitParam
	*/
	QCircuitParam(){
		m_is_dagger = false;
	}

	virtual ~QCircuitParam() {}

	/**
	* @brief copy constructor
	*/
	QCircuitParam(const QCircuitParam& rhs){
		m_is_dagger = rhs.m_is_dagger;
		m_control_qubits = rhs.m_control_qubits;
	}

	/**
	* @brief clone
	*/
	virtual std::shared_ptr<QCircuitParam> clone() {
		return std::make_shared<QCircuitParam>(*this);
	}

	/**
	* @brief append control qubits
	* @param[in] QVec&  control qubits
	*/
	void append_control_qubits(const QVec &ctrl_qubits) {
		m_control_qubits.insert(m_control_qubits.end(), ctrl_qubits.begin(), ctrl_qubits.end());
	}

	/**
	* @brief get the real increased control qubits
	* @param[in] QVec   increased control qubits, maybe some repeat exist
	* @param[in] QVec  already controled qubits
	* @return QVec the real increased control qubits
	*/
	static QVec get_real_append_qubits(QVec append_qubits, QVec target_qubits) {
		if (0 == target_qubits.size())
		{
			return append_qubits;
		}

		if (0 == append_qubits.size())
		{
			return QVec();
		}

		QVec result_vec = append_qubits - target_qubits;
		return result_vec;
	}

	bool m_is_dagger;  /**< dagger information */
	QVec m_control_qubits;/**< control Quantum bits */
};

/**
* @brief Traverse QProg By NodeIter
* @ingroup Utilities
*/
class TraverseByNodeIter : public TraversalInterface<QCircuitParam&, NodeIter&>
{
public:
	/**
	* @brief Constructor of TraverseByNodeIter
	*/
	TraverseByNodeIter() {}
	~TraverseByNodeIter() {}

public:
	virtual void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		//handle QGate node
	}

	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		//handle measure node
	}

	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		//handle reset node
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

	/**
	* @brief start traverse a quantum prog
	*/
	virtual void traverse_qprog(QProg prog);

protected:
};

/**
* @brief get all the node type of the target QProg
* @ingroup Utilities
*/
class GetAllNodeType : public TraverseByNodeIter
{
public:
	/**
	* @brief Constructor of GetAllNodeType
	*/
	GetAllNodeType()
		:m_indent_cnt(0)
	{
	}
	~GetAllNodeType() {}

	/**
	* @brief output the node type string
	* @return std::string node type string
	*/
	std::string printNodesType() {
		std::cout << m_output_str << std::endl;
		return m_output_str;
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		// handle classical prog
		sub_circuit_indent();
		m_output_str.append(">>ClassicalProgNode ");
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override ;
	void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override ;

private:
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

/**
* @brief Pick Up all the Nodes between the two NodeIters
* @ingroup QProgInfo
*/
class PickUpNodes : public TraverseByNodeIter
{
public:
	/**
	* @brief Constructor of PickUpNodes
	*/
	PickUpNodes(QProg &output_prog, QProg src_prog, const std::vector<NodeType> &reject_node_types, const NodeIter &node_itr_start, const NodeIter &node_itr_end)
		:m_src_prog(src_prog), m_output_prog(output_prog), m_reject_node_type(reject_node_types),
		m_start_iter(node_itr_start),
		m_end_iter(node_itr_end),
		m_b_picking(false), m_b_pickup_end(false), m_b_need_dagger(false)
	{}
	~PickUpNodes() {}

	virtual void traverse_qprog() {
		TraverseByNodeIter::traverse_qprog(m_src_prog);
	}
	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
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

	/**
	* @brief set dagger flag
	* @param[in] bool
	*/
	void setDaggerFlag(bool b) { m_b_need_dagger = b; }

	/**
	* @brief reverse the dagger circuit
	*/
	void reverse_dagger_circuit();

	/**
	* @brief check whether the control qubit is same as the target qubit
	* @return bool return true if the control qubit is same as the target qubit, otherwise return false
	*/
	static bool check_control_qubits(QGate& gate);

	/**
	* @brief check no dagger gate
	*/
	static void no_dagger_gate(QGate& gate);

private:
	void pickQGateNode(const NodeIter cur_node_iter, QCircuitParam &cir_param);
	void pickQMeasureNode(const NodeIter cur_node_iter);
	void pickQResetNode(const NodeIter cur_node_iter);
	bool is_valid_pick_up_node_type(const NodeType node_type) {
		for (auto &itr : m_reject_node_type)
		{
			if (node_type == itr)
			{
				return false;
			}
		}

		return true;
	}

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
	QProg m_src_prog;
	const std::vector<NodeType> &m_reject_node_type;
	QProg &m_output_prog;
	NodeIter m_start_iter;
	NodeIter m_end_iter;
	bool m_b_picking;
	bool m_b_pickup_end;
	bool m_b_need_dagger;
};

/**
* @brief  judge the Qgate if match the target topologic structure of quantum circuit
* @ingroup Utilities
* @param[in]  vector<vector<double>>& the target topologic structure of quantum circuit
* @return     if the Qgate match the target topologic structure return true, or else return false
* @see JsonConfigParam::readAdjacentMatrix(TiXmlElement *, int&, std::vector<std::vector<int>>&)
*/
bool isMatchTopology(const QGate& gate, const std::vector<std::vector<double>>& vecTopoSt);

/**
* @brief  get the adjacent quantum gates's(the front one and the back one) type
* @ingroup Utilities
* @param[in] nodeItr  the specialed NodeIter
* @param[out] std::vector<NodeInfo>& adjacentNodes the front node and the back node
* @return result string.
* @see
*/
std::string getAdjacentQGateType(QProg prog, NodeIter &nodeItr, std::vector<NodeInfo>& adjacentNodes);

/**
* @brief  judge the specialed two NodeIters whether can be exchanged
* @ingroup Utilities
* @param[in] nodeItr1 the first NodeIter
* @param[in] nodeItr2 the second NodeIter
* @return if the two NodeIters can be exchanged, return true, otherwise retuen false.
* @note If the two input nodeIters are in different sub-prog, they are unswappable.
*/
bool isSwappable(QProg prog, NodeIter &nodeItr1, NodeIter &nodeItr2);

/**
* @brief  judge if the target node is a base QGate type
* @ingroup Utilities
* @param[in] nodeItr the target NodeIter
* @return if the target node is a base QGate type, return true, otherwise retuen false.
* @see
*/
bool isSupportedGateType(const NodeIter &nodeItr);

/**
* @brief  get the target matrix between the input two Nodeiters
* @ingroup Utilities
* @param[in] const bool Qubit order mark of output matrix, 
             true for positive sequence(Bid Endian), false for inverted order(Little Endian), default is false
* @param[in] nodeItrStart the start NodeIter
* @param[in] nodeItrEnd the end NodeIter
* @return the target matrix include all the QGate's matrix (multiply).
* @see
*/
QStat getCircuitMatrix(QProg srcProg, const bool b_bid_endian = false, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter());

/**
* @brief  pick up the nodes of srcProg between nodeItrStart and  nodeItrEnd to outPutProg
* @ingroup Utilities
* @param[out] outPutProg  the output prog
* @param[in] srcProg The source prog
* @param[in] nodeItrStart The start pos of source prog
* @param[in] nodeItrEnd The end pos of source prog
* @param[in] reject_node_types vector of the reject node  types.
* @param[in] bDagger daggger flag
* @ Note: If there are any Qif/Qwhile nodes between nodeItrStart and nodeItrEnd,
		  Or the nodeItrStart and the nodeItrEnd are in different sub-circuit, an exception will be throw.
*/
void pickUpNode(QProg &outPutProg, QProg srcProg, const std::vector<NodeType> reject_node_types, const NodeIter nodeItrStart = NodeIter(), const NodeIter nodeItrEnd = NodeIter(), bool bDagger = false);

/**
* @brief  Get all the used  quantum bits in the input prog
* @ingroup Utilities
* @param[in] prog  the input prog
* @param[out] vecQuBitsInUse The vector of used quantum bits, sorted from small to large;
* @return return the size of used qubits,sorted by physical address, in descending order
*/
size_t get_all_used_qubits(QProg prog, std::vector<int> &vecQuBitsInUse);
size_t get_all_used_qubits(QProg prog, QVec &vecQuBitsInUse);

/**
* @brief  Get all the used  class bits in the input prog
* @ingroup Utilities
* @param[in] prog  the input prog
* @param[out] vecClBitsInUse The vector of used class bits, sorted from small to large;
* @return return the size of used class bits
*/
size_t get_all_used_class_bits(QProg prog, std::vector<int> &vecClBitsInUse);

/**
* @brief  output all the node type of the target prog
* @ingroup Utilities
* @param[in] the target prog
* @return return the output string
*/
std::string printAllNodeType(QProg prog);

/**
* @brief  get gate parameter
* @ingroup Utilities
* @param[in] pGate the target gate pointer
* @param[out] para_str parameter string
* @return
*/
void get_gate_parameter(std::shared_ptr<AbstractQGateNode> pGate, std::string& para_str);
std::vector<double> get_gate_parameter(std::shared_ptr<AbstractQGateNode> pGate);

/**
* @brief  Check if it is a valid dagger
* @ingroup Utilities
* @param[in] bool 
* @return bool true for valid dagger
*/
bool check_dagger(std::shared_ptr<AbstractQGateNode> p_gate, const bool& b_dagger);

QPANDA_END
#endif