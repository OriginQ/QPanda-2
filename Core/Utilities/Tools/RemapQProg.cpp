#include "Core/Utilities/Tools/RemapQProg.h"
#include "Core/Utilities/QProgInfo/Visualization/QVisualization.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgInfo/GetAllUsedQubitAndCBit.h"
#include "Core/Utilities/Tools/QProgFlattening.h"

USING_QPANDA
using namespace std;

/*******************************************************************
*                      class RemapQProg
********************************************************************/
QVec RemapQProg::remap_qv(const QVec& src_qv)
{
	QVec new_qv;
	for (const auto& q : src_qv){
		new_qv.emplace_back(m_qubit_map.at(q->get_phy_addr()));
	}

	return new_qv;
}

void RemapQProg::execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	QVec gate_qubits;
	cur_node->getQuBitVector(gate_qubits);

	QVec ctrl_qubits;
	cur_node->getControlVector(ctrl_qubits);
	ctrl_qubits.insert(ctrl_qubits.end(), cir_param.m_control_qubits.begin(), cir_param.m_control_qubits.end());

    auto gate = QGate(cur_node);
    QGate new_gate = deepCopy(gate);
	new_gate.clear_control();
	new_gate.remap(remap_qv(gate_qubits));
	new_gate.setControl(remap_qv(ctrl_qubits));
	m_out_prog << new_gate;
}

void RemapQProg::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	const auto src_qubit_i = cur_node->getQuBit()->get_phy_addr();
	const auto src_cbit_i = cur_node->getCBit()->get_addr();
	auto new_measure = Measure(m_qubit_map.at(src_qubit_i), m_cbit_map.at(src_cbit_i));
	m_out_prog << new_measure;
}

void RemapQProg::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	const auto src_qubit_i = cur_node->getQuBit()->get_phy_addr();
	auto new_reset = Reset(m_qubit_map.at(src_qubit_i));
	m_out_prog << new_reset;
}

//void RemapQProg::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
//{
//	
//}

//void RemapQProg::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
//{}
//
//void RemapQProg::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
//{}

QProg RemapQProg::remap(QProg src_prog, QVec target_qv, std::vector<ClassicalCondition> target_cv)
{
    GetAllUsedQubitAndCBit get_qubit_object;
    get_qubit_object.traversal(src_prog);

    const auto src_qubit_vec = get_qubit_object.get_used_qubits();
    const auto src_cbit_vec_i = get_qubit_object.get_used_cbits_i();
    const auto src_cbit_vec = get_qubit_object.get_used_cbits();

    const auto qubit_cnt = src_qubit_vec.size();
    const auto cbit_cnt = src_cbit_vec_i.size();

    if (target_qv.size() < qubit_cnt) {
        QCERR_AND_THROW(run_fail, "Error: The number of target qubit is error.");
    }

    if (target_cv.size() != 0 && target_cv.size() < cbit_cnt)
    {
        QCERR_AND_THROW(run_fail, "Error: The number of target cbit is error.");
    }

    for (size_t i = 0; i < qubit_cnt; ++i) {
        m_qubit_map.insert(std::make_pair(src_qubit_vec[i]->get_phy_addr(), target_qv.at(i)));
    }

    for (auto i = 0; i < cbit_cnt; ++i) {
        if (target_cv.size() != 0)
        {
            m_cbit_map.insert(std::make_pair(src_cbit_vec_i[i], target_cv.at(i)));
        }
        else
        {
            m_cbit_map.insert(std::make_pair(src_cbit_vec_i[i], src_cbit_vec[i]));
        }
    }

    traverse_qprog(src_prog);
    return m_out_prog;
}

/*******************************************************************
*                      public interface
********************************************************************/
QProg QPanda::remap(QProg src_prog, const QVec& target_qv, const std::vector<ClassicalCondition>& target_cv = {})
{
	return RemapQProg().remap(src_prog, target_qv, target_cv);
}

QCircuit QPanda::remap(QCircuit src_cir, const QVec& target_qv)
{
	auto new_prog = remap(src_cir, target_qv, {});
	return QProgFlattening::prog_flatten_to_cir(new_prog);
}