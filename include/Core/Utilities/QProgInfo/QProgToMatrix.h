#ifndef _QPROG_TO_MATRIX_H
#define _QPROG_TO_MATRIX_H

#include "Core/Utilities/QProgInfo/QCircuitInfo.h"

QPANDA_BEGIN

/**
* @brief get the matrix of a QProg
* @ingroup Utilities
*/
class QProgToMatrix
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
		QStat reverse_ctrl_gate_matrix_CX(QStat& src_mat);
		QStat reverse_ctrl_gate_matrix_CU(QStat& src_mat);
		void merge_to_calc_unit(std::vector<int>& qubits, QStat& gate_mat, calcUnitVec_t &calc_unit_vec, gateQubitInfo_t &single_qubit_gates);
		void get_stride_over_qubits(const std::vector<int> &qgate_used_qubits, std::vector<int> &stride_over_qubits);
		void tensor_by_matrix(QStat& src_mat, const QStat& tensor_mat);
		void tensor_by_QGate(QStat& src_mat, std::shared_ptr<AbstractQGateNode> &pGate);
		bool check_cross_calc_unit(calcUnitVec_t& calc_unit_vec, calcUnitVec_t::iterator target_calc_unit_itr);
		void merge_two_crossed_matrix(const calcUintItem_t& calc_unit_1, const calcUintItem_t& calc_unit_2, calcUintItem_t& result);
		void build_standard_control_gate_matrix(const QStat& src_mat, const int qubit_number, QStat& result_mat);
		void swap_two_qubit_on_matrix(QStat& src_mat, const int mat_qubit_start, const int mat_qubit_end, const int qubit_1, const int qubit_2);
		void remove_same_control_qubits(std::vector<int>& qubits_in_standard_mat, std::vector<int>& gate_qubits, const size_t control_qubits_cnt);

	public:
		QStat m_current_layer_mat;
		gateQubitInfo_t m_double_qubit_gates;  /**< double qubit gate vector */
		gateQubitInfo_t m_single_qubit_gates;	  /**< single qubit gate vector */
		gateQubitInfo_t m_controled_gates;       /**< controled qubit gate vector */
		calcUnitVec_t m_calc_unit_vec;
		const QStat m_mat_I;
		std::vector<int> &m_qubits_in_use;		 /**< the number of all the qubits in the target QCircuit. */
	};

public:
	QProgToMatrix(QProg& p)
		:m_prog(p)
	{}
	~QProgToMatrix() {}

	/**
    * @brief calc the matrix of the input QProg
    * @return QStat the matrix of the input QProg
    */
	QStat get_matrix();

	/**
	* @brief calc the matrix of nodes in one layer
	* @param[in] SequenceLayer&  layer nodes
	* @param[in] QProgDAG&  DAG algorithm object
	* @return QStat the matrix of the layer
	*/
	QStat get_matrix_of_one_layer(SequenceLayer& layer, const QProgDAG& prog_dag);

private:
	QProg& m_prog;
	std::vector<int> m_qubits_in_use;
};

QPANDA_END

#endif
