#ifndef REMAP_QPROG_H
#define REMAP_QPROG_H

#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumCircuit/QNode.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include "Core/QuantumCircuit/QuantumGate.h"

QPANDA_BEGIN

/**
* @brief remap a QProg to new qubits
* @ingroup Utilities   
*/
class RemapQProg : public TraverseByNodeIter
{
public:
	RemapQProg() {}
	~RemapQProg() {}

	void execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	void execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		QCERR_AND_THROW(run_fail, "Error: Unsupport ControlFlowNode on RemapQProg.");
	}

	void execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override {
		// handle classical prog
		QCERR_AND_THROW(run_fail, "Error: Unsupport ClassicalProgNode on RemapQProg.");
	}

	//void execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;
	//void execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter) override;

	QProg remap(QProg src_prog, QVec target_qv, std::vector<ClassicalCondition> target_cv);

	const std::map<size_t, Qubit*>& get_qubit_map() const { return m_qubit_map; };
	const std::map<size_t, ClassicalCondition>& get_cbit_map() const { return m_cbit_map; };

protected:
	QVec remap_qv(const QVec& src_qv);

private:
	QProg m_out_prog;
	std::map<size_t, Qubit*> m_qubit_map;
	std::map<size_t, ClassicalCondition> m_cbit_map;
};

QProg remap(QProg src_prog, const QVec& target_qv, const std::vector<ClassicalCondition>& target_cv);
QCircuit remap(QCircuit src_prog, const QVec& target_qv);

QPANDA_END
#endif