#ifndef _QPROG_TO_MATRIX_H
#define _QPROG_TO_MATRIX_H

#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/TopologSequence.h"

QPANDA_BEGIN
/**
* @brief get the matrix of a QProg
* @ingroup Utilities
*/
class QProgToMatrix
{
	using gateAndQubitsItem_t = std::pair<std::shared_ptr<AbstractQGateNode>, std::vector<int>>;
	using gateQubitInfo_t = std::vector<gateAndQubitsItem_t>;
	using calcUintItem_t = std::pair<qmatrix_t, std::vector<int>>;
	using calcUnitVec_t = std::vector<calcUintItem_t>;

	class MatrixOfOneLayer
	{
	public:
		MatrixOfOneLayer(QProgToMatrix& parent, SeqLayer<SequenceNode>& layer,
			const QProgDAG<GateNodeInfo>& prog_dag, std::vector<int> &qubits_in_use);
		void merge_double_gate();
		void merge_calc_unit();
		void merge_controled_gate();
		void merge_sing_gate();

	protected:
		void reverse_ctrl_gate_matrix(qmatrix_t& src_mat, const GateType &gate_T);
		qmatrix_t reverse_ctrl_gate_matrix_CX(qmatrix_t& src_mat);
		qmatrix_t reverse_ctrl_gate_matrix_CU(qmatrix_t& src_mat);
		void merge_to_calc_unit(std::vector<int>& qubits, qmatrix_t& gate_mat, calcUnitVec_t &calc_unit_vec, gateQubitInfo_t &single_qubit_gates);
		void get_stride_over_qubits(const std::vector<int> &qgate_used_qubits, std::vector<int> &stride_over_qubits);
		void tensor_by_matrix(qmatrix_t& src_mat, const qmatrix_t& tensor_mat);
		void tensor_by_QGate(qmatrix_t& src_mat, std::shared_ptr<AbstractQGateNode> &pGate);
		bool check_cross_calc_unit(calcUnitVec_t& calc_unit_vec, calcUnitVec_t::iterator target_calc_unit_itr);
		void merge_two_crossed_matrix(const calcUintItem_t& calc_unit_1, const calcUintItem_t& calc_unit_2, calcUintItem_t& result);
		void build_standard_control_gate_matrix(const qmatrix_t& src_mat, const int qubit_number, qmatrix_t& result_mat);
		void swap_two_qubit_on_matrix(qmatrix_t& src_mat, const int mat_qubit_start, const int mat_qubit_end, const int qubit_1, const int qubit_2);
		void remove_same_control_qubits(std::vector<int>& qubits_in_standard_mat, std::vector<int>& gate_qubits, const size_t control_qubits_cnt);

	public:
		QProgToMatrix& m_parent;
		qmatrix_t m_current_layer_mat;
		gateQubitInfo_t m_double_qubit_gates;  /**< double qubit gate vector */
		gateQubitInfo_t m_single_qubit_gates;	  /**< single qubit gate vector */
		gateQubitInfo_t m_controled_gates;       /**< controled qubit gate vector */
		calcUnitVec_t m_calc_unit_vec;
		qmatrix_t m_mat_I;
		std::vector<int> &m_qubits_in_use;		 /**< the number of all the qubits in the target QCircuit. */
	};

	friend class MatrixOfOneLayer;

public:
	QProgToMatrix(QProg& p, const bool b_bid_endian = false)
		:m_prog(p), m_b_bid_endian(b_bid_endian)
	{
		m_qvm.init();
	}
	~QProgToMatrix() {
		m_qvm.finalize();
	}

	/**
    * @brief calc the matrix of the input QProg
    * @return QStat the matrix of the input QProg
    */
	QStat  get_matrix();

	/**
	* @brief calc the matrix of nodes in one layer
	* @param[in] SeqLayer<SequenceNode>&  layer nodes
	* @param[in] QProgDAG&  DAG algorithm object
	* @return qmatrix_t the matrix of the layer
	*/
	qmatrix_t get_matrix_of_one_layer(SeqLayer<SequenceNode>& layer, const QProgDAG<GateNodeInfo>& prog_dag);

protected:
	QVec& allocate_qubits(const size_t cnt);

private:
	QProg& m_prog;
	const bool m_b_bid_endian;
	std::vector<int> m_qubits_in_use;
	CPUQVM m_qvm; 
	QVec m_allocate_qubits;
};

QPANDA_END

#endif
