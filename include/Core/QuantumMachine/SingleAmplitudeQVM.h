/*! \file SingleAmplitudeQVM.h */
#ifndef  _SINGLEAMPLITUDE_H_
#define  _SINGLEAMPLITUDE_H_
#include "Core/Utilities/Tools/Uinteger.h"
#include "Core/VirtualQuantumProcessor/SingleAmplitude/QuantumGates.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/VirtualQuantumProcessor/SingleAmplitude/Tensor.h"
QPANDA_BEGIN

/**
* @class SingleAmplitudeQVM
* @ingroup QuantumMachine
* @see QuantumMachine
* @brief Quantum machine for single amplitude simulation
*/
class SingleAmplitudeQVM : public QVM, public TraversalInterface<bool&>
{
	using SingleGateNoneAngle_cb = std::function<void(qstate_t&, bool)>;
	using DoubleGateNoneAngle_cb = std::function<void(qstate_t&, bool)>;
	using SingleGateAndAngle_cb = std::function<void(qstate_t&, double, bool)>;
	using DoubleGateAndAngle_cb = std::function<void(qstate_t&, double, bool)>;

public:
	SingleAmplitudeQVM();
	~SingleAmplitudeQVM() {};

	void init();

	/**
	* @brief  run 
	* @param[in]  QProg& quantum program
	* @param[in]  QVec& qubits vector
	* @param[in]  size_t  rank number 
	* @param[in]  size_t  run QuickBB alloted time
	*/
	void run(QProg& prog, QVec& qv, size_t max_rank = 30, size_t alloted_time = 5);

	/**
	* @brief  run
	* @param[in]  QProg& quantum program
	* @param[in]  QVec& qubits vector
	* @param[in]  size_t  rank number
	* @param[in]  size_t quantum program contraction sequence
	*/
	void run(QProg& prog, QVec& qv, size_t max_rank, const std::vector<qprog_sequence_t>& sequences);

	/**
	* @brief  get quantum program contraction sequence
	* @param[in]  const std::vector<size_t>& quickbb vertice
	* @param[out]  std::vector<qprog_sequence_t>& quantum program contraction sequence
	* @return  size_t sequence number
	*/
	size_t getSequence(const std::vector<size_t>& quickbb_vertice, std::vector<qprog_sequence_t>& sequence_vec);

	void getQuickMapVertice(std::vector<std::pair<size_t, size_t>>& map_vector);

	/**
	* @brief  PMeasure by binary index
	* @param[in]  std::string  binary index
	* @return     qstate_type double
	* @note  example: pMeasureBinindex("0000000000")
	*/
	qstate_type pMeasureBinindex(std::string index);

	/**
	* @brief  PMeasure by decimal  index
	* @param[in]  std::string  decimal index
	* @return     qstate_type double
	* @note  example: pMeasureDecindex("1")
	*/
	qstate_type pMeasureDecindex(std::string index);

	/**
	* @brief  get probability by qubits 
	* @param[in] const QVec&  qubits vector
	* @return prob_dict 
	*/
	prob_dict getProbDict(QVec qlist);

	/**
	* @brief  get probability by qubits 
	* @param[in] QProg&  quantum program
	* @param[in] QVec&  qubits vector
	* @return prob_dict 
	*/
	prob_dict probRunDict(QProg& prog, QVec qlist);

	prob_dict getProbDict(const std::vector<int>& qaddrs_list)
	{
		return getProbDict(get_qubits_by_phyaddrs(qaddrs_list));
	}

	prob_dict probRunDict(QProg& prog, const std::vector<int>& qaddrs_list)
	{
		return probRunDict(prog, get_qubits_by_phyaddrs(qaddrs_list));
	}

	virtual void execute(std::shared_ptr<AbstractQuantumProgram> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger);
	virtual void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger);
	virtual void execute(std::shared_ptr<AbstractControlFlowNode> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger);
	virtual void execute(std::shared_ptr<AbstractClassicalProg> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger);
	virtual void execute(std::shared_ptr<AbstractQGateNode> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger);
	virtual void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger);
	virtual void execute(std::shared_ptr<AbstractQuantumReset> cur_node,
		std::shared_ptr<QNode> parent_node, bool& is_dagger);

protected:
	qstate_type singleAmpBackEnd(const std::string  &bin_index);

	void addVerticeAndEdge(qstate_t& gate_tensor, GateType gate_type,
		qsize_t qubit1, qsize_t qubit2 = -1);

	void addSingleGateDiagonalVerticeAndEdge(qstate_t& gate_tensor, qsize_t qubit);

	void addSingleGateNonDiagonalVerticeAndEdge(qstate_t& gate_tensor,
		qsize_t qubit);

	void addDoubleDiagonalGateVerticeAndEdge(qstate_t& gate_tensor,
		qsize_t qubit1, qsize_t qubit2);

	void addDoubleNonDiagonalGateVerticeAndEdge(qstate_t& gate_tensor,
		qsize_t qubit1, qsize_t qubit2);

	void addThreeNonDiagonalGateVerticeAndEdge(qstate_t& gate_tensor,
		qsize_t qubit1,
		qsize_t qubit2,
		qsize_t qubit3);
private:
	QProg m_prog;
	QProgMap m_prog_map;
	qsize_t m_edge_count{ 0 };
	std::vector<qprog_sequence_t> m_sequences;
	ComputeBackend m_backend{ ComputeBackend::CPU };

	std::map<GateType, SingleGateNoneAngle_cb> m_single_gate_none_angle;
	std::map<GateType, DoubleGateNoneAngle_cb> m_double_gate_none_angle;
	std::map<GateType, SingleGateAndAngle_cb> m_single_gate_and_angle;
	std::map<GateType, DoubleGateAndAngle_cb> m_double_gate_and_angle;
};


QPANDA_END
#endif
