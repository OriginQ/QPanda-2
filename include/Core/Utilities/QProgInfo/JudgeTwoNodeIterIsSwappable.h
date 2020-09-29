#ifndef _JUDGE_TWO_NODEITER_IS_SWAPPABLE_H
#define _JUDGE_TWO_NODEITER_IS_SWAPPABLE_H

#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include <stdexcept>

QPANDA_BEGIN

#define SAFE_DELETE_PTR(p) { if(nullptr != p){ delete p; p = nullptr; } }

/**
* @brief Judge whether the prog is related to the target qubits
* @ingroup QProgInfo
* @param[in] QProg the target prog
* @param[in] cir_param& QCircuit param
* @param[in] std::vector<int>& the appointed qubits vector
* @return if the prog have operated any qubit in qubits_vec return true, or else return false
*/
bool judge_prog_operate_target_qubts(QProg prog, const QCircuitParam &cir_param, const std::vector<int>& qubits_vec);

/**
* @brief Judge two node is swappable
* @ingroup QProgInfo
*/
class JudgeTwoNodeIterIsSwappable : public TraverseByNodeIter
{
	enum ResultStatue
	{
		INIT = 0,
		JUST_FOUND_ONE_NODE,
		FOUND_ALL_NODES,
		JUDGE_MATRIX,
		CAN_NOT_BE_EXCHANGED,
		COULD_BE_EXCHANGED
	};

	struct NodeInCircuitInfo
	{
		NodeInCircuitInfo() 
			:m_in_circuit(false), m_dagger(false)
		{}

		bool m_in_circuit;
		bool m_dagger;
	};

	class AbstractJudgeStatueInterface
	{
	public:
		virtual void handle_QGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handle_QMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void handle_QReset(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {}
		virtual void enter_flow_ctrl_node() {}
		virtual void leave_flow_ctrl_node() {}
		virtual void on_enter_circuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) {}
		virtual void on_leave_circuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) {}
		virtual void on_enter_prog(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param) {}
		virtual void on_leave_prog(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param) {}
		virtual void on_traversal_end() {}
		virtual ResultStatue get_statue() const = 0;
	};

	friend class JudgeStatueInterface;

public:
	/**
	* @brief  Constructor of JudgeTwoNodeIterIsSwappable
	*/
	JudgeTwoNodeIterIsSwappable(QProg prog, NodeIter &nodeItr_1, NodeIter &nodeItr_2)
		:m_prog(prog), m_nodeItr1(nodeItr_1), m_nodeItr2(nodeItr_2), m_judge_statue(nullptr), m_last_statue(nullptr),
		m_result(INIT), m_b_found_first_iter(false), m_b_found_second_iter(false), m_b_dagger_circuit(false)
	{
	}
	~JudgeTwoNodeIterIsSwappable() {
		SAFE_DELETE_PTR(m_last_statue);
		SAFE_DELETE_PTR(m_judge_statue);
	}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		m_judge_statue->handle_QGate(cur_node, parent_node, cir_param, cur_node_iter);
	}
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		m_judge_statue->handle_QMeasure(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		m_judge_statue->handle_QReset(cur_node, parent_node, cir_param, cur_node_iter);
	}

	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) {
		// handle classical prog
		if ((cur_node_iter == m_nodeItr1) || (cur_node_iter == m_nodeItr2))
		{
			_change_statue(new CanNotBeExchange(*this, CAN_NOT_BE_EXCHANGED));
		}
	}

	void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);
	void execute(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter);

	/**
    * @brief get final judge result
	* @return  bool
    */
	bool get_result();

	/**
	* @brief start traverse a quantum prog
	*/
	virtual void traverse_qprog();

	/**
	* @brief judge the input two node type
	* @return if any input node is unswappable type, return false, else return true.
	*/
	bool judge_node_type();

private:
	void _change_statue(AbstractJudgeStatueInterface* s);
	void _clear_picked_prog() { m_pick_prog.clear(); }
	void _pick_node(const NodeIter iter, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param);
	bool _have_found_all_nodeIter() { return (m_b_found_first_iter && m_b_found_second_iter); }
	void _check_picked_prog_matrix();

private:
	QProg m_prog;
	ResultStatue m_result;
	NodeIter m_nodeItr1;
	NodeIter m_nodeItr2;
	bool m_b_found_first_iter;
	bool m_b_found_second_iter;
	bool m_b_dagger_circuit;
	QProg m_pick_prog;
	AbstractJudgeStatueInterface *m_judge_statue;
	AbstractJudgeStatueInterface *m_last_statue;
	std::vector<int> m_correlated_qubits; /**< Correlated qubits of the input two NodeIter.*/
	std::vector<NodeInCircuitInfo> m_node_circuit_info = {NodeInCircuitInfo(), NodeInCircuitInfo()};/**< whether the circuit which include the appointed node is dagger.*/

private:
	class CoubleBeExchange : public AbstractJudgeStatueInterface
	{
	public:
		CoubleBeExchange(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~CoubleBeExchange() {}

		ResultStatue get_statue() const { return m_statue; }

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

		ResultStatue get_statue() const { return m_statue; }

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
	};

	class OnJudgeMatrix : public AbstractJudgeStatueInterface
	{
	public:
		OnJudgeMatrix(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~OnJudgeMatrix() {}

		void on_traversal_end() override {
			m_parent._check_picked_prog_matrix();
		}
		ResultStatue get_statue() const { return m_statue; }

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
	};

	class OnFoundAllNodes : public AbstractJudgeStatueInterface
	{
	public:
		OnFoundAllNodes(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s)
		{}
		~OnFoundAllNodes() {}

		void on_traversal_end() override {
			m_parent._check_picked_prog_matrix();
		}
		ResultStatue get_statue() const { return m_statue; }

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
	};

	class OnPickUpNode : public AbstractJudgeStatueInterface
	{
	public:
		OnPickUpNode(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s), m_nesting_rank(0), m_in_flow_control_node(0)
		{}
		~OnPickUpNode() {}

		void handle_QGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			_on_pick(cur_node_iter, parent_node, cir_param);
		}
		void handle_QMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			_on_pick(cur_node_iter, parent_node, cir_param);
		}

		void handle_QReset(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			// handle reset node
			_on_pick(cur_node_iter, parent_node, cir_param);
		}

		void enter_flow_ctrl_node() override {
			++m_in_flow_control_node;
		}
		void leave_flow_ctrl_node() override {
			--m_in_flow_control_node;
		}
		void on_enter_circuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) override {
			if (0 < m_in_flow_control_node)
			{
				QCircuit tmp_circuit(cur_node);
				if (judge_prog_operate_target_qubts(tmp_circuit, cir_param, m_parent.m_correlated_qubits))
				{
					m_parent._change_statue(new CanNotBeExchange(m_parent, CAN_NOT_BE_EXCHANGED));
				}
			}
			else
			{
				++m_nesting_rank;
			}
		}
		void on_leave_circuit(std::shared_ptr<AbstractQuantumCircuit> cur_node, QCircuitParam &cir_param) override {
			if (0 < m_in_flow_control_node) return;

			if (0 < m_nesting_rank)
			{
				--m_nesting_rank;
			}
			else
			{
				m_parent._change_statue(new CanNotBeExchange(m_parent, CAN_NOT_BE_EXCHANGED));
			}
		}

		void on_enter_prog(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param) override {
			if (0 < m_in_flow_control_node)
			{
				QProg tmp_prog(cur_node);
				if (judge_prog_operate_target_qubts(tmp_prog, cir_param, m_parent.m_correlated_qubits))
				{
					m_parent._change_statue(new CanNotBeExchange(m_parent, CAN_NOT_BE_EXCHANGED));
				}
			}
			else
			{
				++m_nesting_rank;
			}
		}
		void on_leave_prog(std::shared_ptr<AbstractQuantumProgram> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param) override {
			if (0 < m_in_flow_control_node) return;

			if (0 < m_nesting_rank)
			{
				--m_nesting_rank;
			}
			else
			{
				m_parent._change_statue(new CanNotBeExchange(m_parent, CAN_NOT_BE_EXCHANGED));
			}
		}

		ResultStatue get_statue() const { return m_statue; }

	protected:
		void _on_pick(NodeIter cur_node_iter, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param) {
			if (0 < m_in_flow_control_node)
			{
				return;
			}

			_pick_node(cur_node_iter, parent_node, cir_param);

			if (m_parent.m_b_found_first_iter && m_parent.m_b_found_second_iter)
			{
				//change statue
				if (0 != m_nesting_rank)
				{
					m_parent._change_statue(new CanNotBeExchange(m_parent, CAN_NOT_BE_EXCHANGED));
				}
				else
				{
					m_parent._change_statue(new OnFoundAllNodes(m_parent, FOUND_ALL_NODES));
				}
			}
		}

		void _pick_node(const NodeIter iter, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param)
		{
			//relevance judgement
			QVec tmp_vec;
			std::vector<int> qubits_val_vec;
			switch ((*iter)->getNodeType())
			{
			case GATE_NODE:
			{
				auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*iter);
				p_gate->getQuBitVector(tmp_vec);
				p_gate->getControlVector(tmp_vec);
				for (auto& qubit_item : tmp_vec)
				{
					qubits_val_vec.push_back(qubit_item->getPhysicalQubitPtr()->getQubitAddr());
				}
			}
			break;

			case MEASURE_GATE:
			{
				auto p_measure = std::dynamic_pointer_cast<AbstractQuantumMeasure>(*iter);
				qubits_val_vec.push_back(p_measure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
			}
			break;

			case RESET_NODE:
			{
				auto p_reset = std::dynamic_pointer_cast<AbstractQuantumReset>(*iter);
				qubits_val_vec.push_back(p_reset->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
			}
			break;

			default:
				QCERR_AND_THROW_ERRSTR(std::runtime_error, "Error: Node type error.");
				break;
			}

			sort(qubits_val_vec.begin(), qubits_val_vec.end());
			qubits_val_vec.erase(unique(qubits_val_vec.begin(), qubits_val_vec.end()), qubits_val_vec.end());
			std::vector<int> result_vec;
			std::set_intersection(m_parent.m_correlated_qubits.begin(), m_parent.m_correlated_qubits.end(), 
				qubits_val_vec.begin(), qubits_val_vec.end(), 
				std::back_inserter(result_vec));
			if (result_vec.size() != 0)
			{
				if (iter == m_parent.m_nodeItr1)
				{
					m_parent.m_b_found_first_iter = true;
				}
				else if (iter == m_parent.m_nodeItr2)
				{
					m_parent.m_b_found_second_iter = true;
				}

				m_parent._pick_node(iter, parent_node, cir_param);
			}
		}

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		const ResultStatue m_statue;
		int m_nesting_rank;
		int m_in_flow_control_node;
	};

	class OnInitStatue : public AbstractJudgeStatueInterface
	{
	public:
		OnInitStatue(JudgeTwoNodeIterIsSwappable& parent, ResultStatue s)
			:m_parent(parent), m_statue(s), m_need_update_layer_start_iter(true)
		{}
		void handle_QGate(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			_on_pick(cur_node_iter, parent_node, cir_param);
		}
		void handle_QMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			//pickup
			_on_pick(cur_node_iter, parent_node, cir_param);
		}

		void handle_QReset(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
			// handle reset node
			_on_pick(cur_node_iter, parent_node, cir_param);
		}

		ResultStatue get_statue() const { return m_statue; }

	private:
		void _on_pick(NodeIter cur_node_iter, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param) {
			_pick_node(cur_node_iter, parent_node, cir_param);

			if (m_parent.m_b_found_first_iter || m_parent.m_b_found_second_iter)
			{
				//change statue
				m_parent._change_statue(new OnPickUpNode(m_parent, JUST_FOUND_ONE_NODE));
			}
		}

		void _pick_node(const NodeIter iter, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param){
			if (iter == m_parent.m_nodeItr1)
			{
				m_parent.m_b_found_first_iter = true;
				m_parent._pick_node(iter, parent_node, cir_param);
			}
			else if (iter == m_parent.m_nodeItr2)
			{
				m_parent.m_b_found_second_iter = true;
				m_parent._pick_node(iter, parent_node, cir_param);
			}
		}

	private:
		JudgeTwoNodeIterIsSwappable& m_parent;
		NodeIter m_layer_start_iter;
		const ResultStatue m_statue;
		bool m_need_update_layer_start_iter;
	};
};

QPANDA_END

#endif