#include "Core/Utilities/Tools/QCircuitFusion.h"
#include "Core/VirtualQuantumProcessor//CPUImplQPU.h"
#include "Core/Utilities/Tools/QStatMatrix.h"
#include "Core/QuantumCircuit/QuantumGate.h"
#include "Core/Utilities/Tools/QProgFlattening.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/Utilities/Tools/Traversal.h"
#include <atomic>
#include <ctime>
#include <vector>
#include <thread>
#include <set>
USING_QPANDA
QGate Fusion::_generate_operation_internal(const std::vector<QGate> &fusion_gates,
	const std::vector<int> &qubits, QuantumMachine *qvm)
{
	CPUImplQPU cpu;
	QStat state;
	cpu.initMatrixState(qubits.size() * 2, state);
	for (auto i = 0; i < fusion_gates.size(); i++)
	{
		QStat matrix;
		fusion_gates[i].getQGate()->getMatrix(matrix);
		if (fusion_gates[i].isDagger()) {
			dagger(matrix);
		}

		QVec qubit_vector;
		fusion_gates[i].getQuBitVector(qubit_vector);
		Qubit* qubit = *(qubit_vector.begin());
		size_t bit = qubit->getPhysicalQubitPtr()->getQubitAddr();
		auto gate_type = fusion_gates[i].getQGate()->getGateType();

		std::vector<size_t> phy_qv;
		for (auto &i : qubit_vector) {
			phy_qv.push_back(i->get_phy_addr());
		}

		if (gate_type == GateType::ORACLE_GATE) {
			cpu.OracleGate(phy_qv, matrix, false);
		}
		else
		{
			if (qubit_vector.size() > 1) {
				cpu.unitaryDoubleQubitGate(qubits[0], qubits[1], matrix, false, static_cast<GateType>(gate_type));
			}
			else {
				cpu.unitarySingleQubitGate(bit, matrix, false, static_cast<GateType>(gate_type));
			}
		}

	}

	QStat data = cpu.getQState();
	QVec gate_qv;
	gate_qv.resize(qubits.size(), 0);
	std::map<int, Qubit*> tmp_map;
	QVec used_qv;
	qvm->get_allocate_qubits(used_qv);

	for (auto &qv : used_qv) {
		tmp_map[qv->get_phy_addr()] = qv;
	}

	for (int i = 0; i < qubits.size(); i++) {
		gate_qv[i] = tmp_map[qubits[i]];
	}

	if (gate_qv.size() > 1)
	{
		return QDouble(gate_qv[0], gate_qv[1], data);
	}
	else
	{
		return U4(gate_qv[0], data);
	}

}

bool Fusion::_exclude_escaped_qubits(std::vector<int>& fusion_qubits,
	const QGate& tgt_op)  const
{

	bool included = true;
	QVec used_qv;
	tgt_op.getQuBitVector(used_qv);
	if (tgt_op.getControlQubitNum() > 0)
		return true;
	for (const auto qubit : used_qv) {
		included &= (std::find(fusion_qubits.begin(), fusion_qubits.end(), qubit->get_phy_addr()) != fusion_qubits.end());

	}

	if (included) {
		return false;
	}

	for (const auto op_qubit : used_qv) {
		auto found = std::find(fusion_qubits.begin(), fusion_qubits.end(), op_qubit->get_phy_addr());
		if (found != fusion_qubits.end())
			fusion_qubits.erase(found);
	}
	return true;
}


void Fusion::aggregate_operations(QCircuit &cir, QuantumMachine *qvm)
{
	if (cir.is_empty())
		return ;

	flatten(cir);
	_fusion_gate(cir, 1, qvm);
	_fusion_gate(cir, 2, qvm);
}

void  Fusion::aggregate_operations(QProg& src_prog, QuantumMachine* qvm)
{

	if (src_prog.is_empty()) {
		return;
	}

	flatten(src_prog, true);
	_fusion_gate(src_prog, 1, qvm);
	_fusion_gate(src_prog, 2, qvm);

}

template<class T>
void Fusion::_fusion_gate(T& src_prog, const int fusion_bit, QuantumMachine* qvm)
{
	auto prog_node = src_prog.getImplementationPtr();
	for (auto itr = prog_node->getFirstNodeIter(); itr != prog_node->getEndNodeIter(); itr++)
	{
		if (itr == nullptr) {
			break;
		}

		auto gate_tmp = std::dynamic_pointer_cast<QNode>(*itr);
		if ((*gate_tmp).getNodeType() != NodeType::GATE_NODE) {
			continue;
		}

		auto gate_node = std::dynamic_pointer_cast<AbstractQGateNode>(gate_tmp);
		if (gate_node->getControlQubitNum() > 0) {
			continue;
		}

		QVec qubit_vec;
		gate_node->getQuBitVector(qubit_vec);
		if (qubit_vec.size() != fusion_bit) {
			continue;
		}

		std::vector<NodeIter> fusing_gate_idxs = { itr };
		std::vector<int> fusing_qubits;
		for (const auto qbit : qubit_vec) {
			fusing_qubits.insert(fusing_qubits.end(), qbit->get_phy_addr());
		}

		/*2.Fuse gate with backwarding*/
		if (itr != prog_node->getFirstNodeIter())
		{
			auto fusion_gate_itr = itr;
			--fusion_gate_itr;
			for (; fusion_gate_itr != prog_node->getHeadNodeIter(); --fusion_gate_itr)
			{
				auto q_gate = std::dynamic_pointer_cast<QNode>(*fusion_gate_itr);
				if (q_gate->getNodeType() != NodeType::GATE_NODE) {
					continue;
				}

				auto gate_tmp = std::dynamic_pointer_cast<AbstractQGateNode>(q_gate);
				if (gate_tmp->getControlQubitNum() > 0) {
					break;
				}

				auto &t_gate = gate_tmp;
				if (!_exclude_escaped_qubits(fusing_qubits, t_gate)) {
					fusing_gate_idxs.push_back(fusion_gate_itr); /*All the qubits of tgt_op are in fusing_qubits*/
				}

				else if (fusing_qubits.empty()) {
					break;
				}
			}
		}

		std::reverse(fusing_gate_idxs.begin(), fusing_gate_idxs.end());
		fusing_qubits.clear();
		for (auto &qbit : qubit_vec) {
			fusing_qubits.insert(fusing_qubits.end(), qbit->get_phy_addr());
		}

		/*3.fuse gate with forwarding */
		if (itr != prog_node->getLastNodeIter())
		{
			auto fusion_gate_itr = itr;
			++fusion_gate_itr;
			for (; fusion_gate_itr != prog_node->getEndNodeIter(); ++fusion_gate_itr)
			{
				auto q_gate = std::dynamic_pointer_cast<QNode>(*fusion_gate_itr);
				if (q_gate->getNodeType() != NodeType::GATE_NODE) {
					continue;
				}
				auto gate_tmp = std::dynamic_pointer_cast<AbstractQGateNode>(q_gate);
				if (gate_tmp->getControlQubitNum() > 0) {
					break;
				}

				auto &t_gate = gate_tmp;
				if (!_exclude_escaped_qubits(fusing_qubits, t_gate)) {
					fusing_gate_idxs.push_back(fusion_gate_itr); /*All the qubits of tgt_op are in fusing_qubits*/
				}
				else if (fusing_qubits.empty()) {
					break;
				}
			}
		}

		if (fusing_gate_idxs.size() <= 1) {
			continue;
		}

		/*4.generate a fused gate*/
		_allocate_new_operation(src_prog, itr, fusing_gate_idxs, qvm);
	}
}

template<class T>
void Fusion::_allocate_new_operation(T& prog, NodeIter& index_itr,
	std::vector<NodeIter>& fusing_gate_itrs, QuantumMachine* qvm)
{
	std::vector<QGate> fusion_gates;
	for (auto& itr : fusing_gate_itrs) {
		auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*itr);
		fusion_gates.push_back(p_gate);
	}

	auto q_gate = _generate_operation(fusion_gates, qvm);
	prog.insertQNode(index_itr, std::dynamic_pointer_cast<QNode>(q_gate.getImplementationPtr()));
	index_itr++;
	for (auto &itr : fusing_gate_itrs) {
		prog.deleteQNode(itr);
	}
}

QGate Fusion::_generate_operation(std::vector<QGate>& fusion_gates, QuantumMachine* qvm)
{
	std::set<int> fusioned_qubits;
	std::vector<QVec> tmp;
	for (auto &t_gate : fusion_gates)
	{
		QVec t_vec;
		t_gate.getQuBitVector(t_vec);
		for (int i = 0; i < t_vec.size(); i++)
			fusioned_qubits.insert(t_vec[i]->get_phy_addr());
	}

	std::vector<int> remapped2orig(fusioned_qubits.begin(), fusioned_qubits.end());
	std::unordered_map<int, int> orig2remapped;
	std::vector<int> arg_qubits;

	arg_qubits.resize(fusioned_qubits.size(), 0);

	for (int i = 0; i < remapped2orig.size(); i++)
	{
		orig2remapped[remapped2orig[i]] = i;
		arg_qubits[i] = i;
	}
	std::map<int, Qubit*> tmp_map;
	QVec used_qv;
	qvm->get_allocate_qubits(used_qv);

	for (auto &it : used_qv)
	{
		tmp_map[it->get_phy_addr()] = it;
	}
	for (auto &op : fusion_gates)
	{
		QVec tmp_qv;
		op.getQuBitVector(tmp_qv);
		for (int i = 0; i < tmp_qv.size(); i++)
		{
			tmp_qv[i] = tmp_map[i];

		}
		op.remap(tmp_qv);
	}

	auto fusioned_op = _generate_operation_internal(fusion_gates, arg_qubits, qvm);

	QVec gate_qv;
	fusioned_op.getQuBitVector(gate_qv);

	for (auto &it : used_qv)
	{
		tmp_map[it->get_phy_addr()] = it;
	}
	for (size_t i = 0; i < gate_qv.size(); i++)
	{

		gate_qv[i] = tmp_map[remapped2orig[i]];
	}

	fusioned_op.remap(gate_qv);
	return fusioned_op;
}