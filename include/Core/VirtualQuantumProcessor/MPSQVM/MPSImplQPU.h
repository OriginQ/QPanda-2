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

#ifndef _MPS_IMPLQPU_H_
#define _MPS_IMPLQPU_H_
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/MPSQVM/MPSTensor.h"

QPANDA_BEGIN

struct QCircuitConfig
{
    bool _is_dagger = false;
    QVec _contorls;
    bool _can_optimize_measure = true;
};

/**
* @brief QPU implementation by MPS model
* @ingroup VirtualQuantumProcessor
*/
class MPSImplQPU : public QPUImpl
{
public:

    size_t get_qubit_num() { return m_qubits_num; }

	bool qubitMeasure(size_t qn);

	QError pMeasure(Qnum& qnum, prob_vec &mResult);

	QError initState(size_t head_rank, size_t rank_size, size_t qubit_num);

    void initState(const MPSImplQPU &other);

    QError initState(size_t qubit_num, const QStat &state = {});

	/**
	* @brief  init state from matrix
	* @param[in]  size_t number of qubits
	* @param[in]  cmatrix_t  matrix
	*/
    void initState_from_matrix(size_t num_qubits, const cmatrix_t &mat);

	/**
	* @brief  unitary single qubit gate
	* @param[in]  size_t  qubit address
	* @param[in]  QStat&  matrix
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
	QError unitarySingleQubitGate(size_t qn, QStat& matrix,
		bool isConjugate,
		GateType);

	/**
	* @brief  controlunitary single qubit gate
	* @param[in]  size_t  qubit address
	* @param[in]  Qnum&  control qubit addresses
	* @param[in]  QStat &  matrix
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
	QError controlunitarySingleQubitGate(size_t qn, Qnum& qnum,
		QStat& matrix,
		bool isConjugate,
		GateType);

	/**
	* @brief unitary double qubit gate
	* @param[in]  size_t  first qubit address
	* @param[in]  size_t  second qubit address
	* @param[in]  QStat&  matrix
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
	QError unitaryDoubleQubitGate(size_t qn_0, size_t qn_1,
		QStat& matrix,
		bool isConjugate,
		GateType);

	/**
	* @brief  controlunitary double qubit gate
	* @param[in]  size_t  first qubit address
	* @param[in]  size_t  second qubit address
	* @param[in]  Qnum&  control qubit addresses
	* @param[in]  QStat&  quantum states
	* @param[in]  bool   state of conjugate
	* @param[in]  GateType    gate type
	* @return    QError
	*/
	QError controlunitaryDoubleQubitGate(size_t qn_0,
		size_t qn_1,
		Qnum& qnum,
		QStat& matrix,
		bool isConjugate,
		GateType);
	
	/**
	* @brief get quantum states
	* @return QStat  quantum states
	*/
	QStat getQState();


	/**
	* @brief reset qubit
	* @param[in]  size_t  qubit address
	*/
	QError Reset(size_t qn);


	/** 
	* @brief gets the position of the qubits in MPS form
	* @param[in]  size_t  qubits index
	* @return size_t  the position on m_qubits_location
	*/
	size_t get_qubit_index(size_t index) const
	{
		return m_qubits_location[index];
	}

	/**
	* @brief change two qubits 
	* @param[in]  size_t  src qubit location
	* @param[in]  size_t  dst qubit location
	*/
	void change_qubits_location(size_t src, size_t dst);

	/**
	* @brief execute SWAP gate, the state of swapping two qubits
	* @param[in]  size_t  A qubit index
	* @param[in]  size_t  B qubit index
	*/
	void swap_qubits_location(size_t index_A, size_t index_B);

	/**
	* @brief measure one qubit collapsing
	* @param[in]  size_t  the qubit position of the measurement
	* @return bool the measurement results
	*/
    bool measure_one_collapsing(size_t qubit);

	/**
	* @brief measure all qubits collapsing
	* @param[in]  size_t  the qubit position of the measurement
	* @return std::vector<std::vector<size_t>> the measurement results
	*/
	std::vector<std::vector<size_t>> measure_all_noncollapsing(Qnum measure_qubits, int shots);

	/**
	* @brief after the SVD decomposition , The product of S and V
	* @param[in]  cmatrix_t  V matrix
	* @param[in]  rvector_t  S vector
	* @return cmatrix_t product
	*/
	cmatrix_t mul_v_by_s(const cmatrix_t &mat, const rvector_t &lambda);


	/**
	* @brief convert to MPS form
	* @param[in]  size_t  starting position 
	* @param[in]  size_t end position
	* @return MPS_Tensor  MPS form  tensor
	*/
	MPS_Tensor convert_qstate_to_mps_form(size_t first_index, size_t last_index);

	/**
	* @brief sort qubits location, and centralize qubits locations
	* @param[in]  Qnum original qubits location
	* @param[in]  Qnum sorted indices
	* @param[out]  Qnum centralized qubits location
	*/
	void centralize_and_sort_qubits(const Qnum &qubits, Qnum &sorted_indices, Qnum &centralized_qubits);

	/**
	* @brief move all qubits to sorted ordering 
	*/
	void move_all_qubits_to_sorted_ordering();

	/**
	* @brief move qubits to right_end location
	* @param[in]  Qnum original qubits location
	* @param[out]  Qnum target qubits location
	* @param[out]  actual indices
	*/
	void move_qubits_to_right_end(const Qnum &qubits, Qnum &target_qubits, Qnum &actual_indices);
	

	/**
	* @brief execute one qubit gate
	* @param[in]  size_t target qubit
	* @param[in]  cmatrix_t gate matrix
	*/
	void execute_one_qubit_gate(size_t qn, const cmatrix_t &mat);

	/**
	* @brief execute two qubits gate
	* @param[in]  size_t control qubit
	* @param[in]  size_t target qubit
	* @param[in]  cmatrix_t gate matrix
	*/
	void execute_two_qubit_gate(size_t qn_0, size_t qn_1, const cmatrix_t &mat);

	/**
	* @brief execute multi qubits gate
	* @param[in]  size_t control and target qubits, target qubit  in the tail
	* @param[in]  cmatrix_t gate matrix
	*/
	void execute_multi_qubit_gate(const Qnum &qubits, const cmatrix_t &mat);
	
	qcomplex_t expectation_value_pauli(const Qnum &qubits);

	qcomplex_t expectation_value_pauli_internal(const Qnum &qubits,
		const std::vector<GateType> &matrices, size_t first_index, size_t last_index, size_t num_Is);

	bool apply_measure(size_t qubit);

	Qnum apply_measure(Qnum qubits);

    cmatrix_t density_matrix(const Qnum &qubits);

    double expectation_value(const Qnum &qubits, const cmatrix_t &matrix);

    double single_expectation_value(const Qnum &qubits, const cmatrix_t &matrix);
    double double_expectation_value(const Qnum &qubits, const cmatrix_t &matrix);

    void unitaryQubitGate(Qnum qubits, QStat matrix, bool isConjugate);

    qcomplex_t pmeasure_bin_index(std::string str);

    qcomplex_t pmeasure_dec_index(std::string str);

private:
    
    QStat m_init_state;

	Qnum m_qubits_order;  /**< sstores the current ordering of the qubitsr.   */
	
	Qnum m_qubits_location; /**< stores the location of each qubit in the vector.   */

	size_t m_qubits_num;   /**< number of qubits.   */

public:

	std::vector<MPS_Tensor> m_qubits_tensor;  /**< the tensor of qubits.   */

	std::vector<rvector_t> m_lambdas;  /**< lambdas between tensors.   */
};

QPANDA_END

#endif  //!_MPS_IMPLQPU_H_
