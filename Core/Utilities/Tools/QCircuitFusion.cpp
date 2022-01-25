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

double Fusion::distance_cost(const std::vector<QGate>& ops,
    const int from,
    const int until) const
{
    std::vector<int> fusion_qubits;
    for (int i = from; i <= until; ++i)
        add_optimize_qubits(fusion_qubits, ops[i]);
    auto configured_cost = distances_[fusion_qubits.size() - 1];
    if (configured_cost > 0)
        return configured_cost;
    switch (fusion_qubits.size()) {
    case 1:
        /*bull*/
    case 2:
        return 1.0;
    case 3:
        return 1.1;
    case 4:
        return 3;
    default:
        return pow(distance_factor, (double)std::max(fusion_qubits.size() - 2, size_t(1)));
    }
}

void Fusion::add_optimize_qubits(std::vector<int>& fusion_qubits, const QGate& gate) const
{
    QVec used_qv;
    gate.getQuBitVector(used_qv);
    for (const auto &op_qubit : used_qv) {
        if (find(fusion_qubits.begin(), fusion_qubits.end(), op_qubit->get_phy_addr()) == fusion_qubits.end())
            fusion_qubits.push_back(op_qubit->get_phy_addr());
    }
}

bool Fusion::aggreate(std::vector<QGate>& gate_v, QuantumMachine* qvm)
{
    std::vector<double> distances;
    std::vector<int> optimize_index;
    bool flag = false;
    optimize_index.push_back(0);
    distances.push_back(distance_factor);
    for (int i = 1; i < gate_v.size(); i++)
    {
        optimize_index.push_back(i);
        distances.push_back(distances[i - 1] + distance_factor);
        for (int num_fusion = 2; num_fusion <= 5; ++num_fusion)
        {
            std::vector<int> fusion_qubits;
            add_optimize_qubits(fusion_qubits, gate_v[i]);

            for (int j = i - 1; j >= 0; --j)
            {
                add_optimize_qubits(fusion_qubits, gate_v[j]);
                if (fusion_qubits.size() > num_fusion)
                    break;
                /*optimize gate from j - i*/
                double distance = distance_cost(gate_v, j, i) + (j == 0 ? 0.0 : distances[j - 1]);
                if (distance <= distances[i])
                {
                    distances[i] = distance;
                    optimize_index[i] = j;
                    flag = true;
                }
            }
        }

    }
    if (!flag)
        return false;

    for (int i = gate_v.size() - 2; i >= 0;) {
        int start = optimize_index[i];
        if (start != i) {
            std::vector<int> opt_gate_idxs;
            for (int j = start; j <= i; ++j)
                opt_gate_idxs.push_back(j);
            if (!opt_gate_idxs.empty())
                _allocate_new_gate(gate_v, i, opt_gate_idxs, qvm);
        }
        i = start - 1;
    }
    return true;
}

QGate Fusion::_generate_oracle_gate(const std::vector<QGate>& fusion_gates,
    const std::vector<int>& qubits, QuantumMachine* qvm)
{
    CPUImplQPU cpu;
    QStat state;
    cpu.initMatrixState(qubits.size() * 2, state);
    for (int i = fusion_gates.size() - 1; i >= 0; i--)
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
            if (qubit_vector.size() == 1)
            {
                cpu.unitarySingleQubitGate(phy_qv[0], matrix, false, (GateType)gate_type);
            }
            else
            {
                cpu.three_qubit_gate_fusion(phy_qv[0], phy_qv[1], matrix);
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

    return QOracle(gate_qv, data);
}



QGate Fusion::_generate_operation_internal(const std::vector<QGate> &fusion_gates,
	const std::vector<int> &qubits, QuantumMachine *qvm)
{
	CPUImplQPU cpu;
	QStat state;
	cpu.initMatrixState(qubits.size() * 2, state);
	for (int i = 0; i < fusion_gates.size(); i++)
	{
		QStat matrix;
		fusion_gates[i].getQGate()->getMatrix(matrix);
		if (fusion_gates[i].isDagger()) {
			dagger(matrix);
		}

		QStat tmp_matrix;
		tmp_matrix.resize(16);
		QStat temp_init = { qcomplex_t(1,0), qcomplex_t(0,0),qcomplex_t(0,0),
		qcomplex_t(1,0) };

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
				cpu.double_qubit_gate_fusion(phy_qv[0], phy_qv[1], matrix);
			}
			else 
			{
				if (qubit_vector.size() > 1) {
					cpu.double_qubit_gate_fusion(qubits[0], qubits[1], matrix);
				}
				else
				{
					if (qubits.size() > 1)
					{
						if (bit == qubits[0])
						{
							tmp_matrix = tensor(matrix, temp_init);
						}
						else
						{
							tmp_matrix = tensor(temp_init, matrix);
						}
						cpu.double_qubit_gate_fusion(qubits[0], qubits[1], tmp_matrix);
					}
					else
					{
						cpu.single_qubit_gate_fusion(bit, matrix);
					}
				}
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
		for (int j = 0; j < 4; j++)
		{
			qcomplex_t tmp = data[j * 4 + 1];
			data[j * 4 + 1] = data[j * 4 + 2];
			data[j * 4 + 2] = tmp;
		}
		for (int j = 0; j < 4; j++)
		{
			qcomplex_t tmp = data[4 + j];
			data[4 + j] = data[8 + j];
			data[8 + j] = tmp;
		}
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

    std::vector<QGate> gate_list;
    for (auto gate_itr = src_prog.getFirstNodeIter(); gate_itr != src_prog.getEndNodeIter(); ++gate_itr)
    {
        auto gate_tmp = std::dynamic_pointer_cast<QNode>(*gate_itr);
        if ((*gate_tmp).getNodeType() != NodeType::GATE_NODE) {
            continue;
        }

        auto p_gate = std::dynamic_pointer_cast<AbstractQGateNode>(*gate_itr);
        gate_list.push_back(p_gate);

    }
    src_prog.clear();
    aggreate(gate_list, qvm);

    for (int i = 0; i < gate_list.size(); i++)
    {
        if (gate_list[i].getQGate()->getGateType() == GateType::GATE_NOP)
            continue;
        src_prog.insertQNode(src_prog.getLastNodeIter(), std::dynamic_pointer_cast<QNode>(gate_list[i].getImplementationPtr()));
    }

}

template<class T>
void Fusion::_fusion_gate(T& src_prog, const int fusion_bit, QuantumMachine* qvm)
{
	auto prog_node = src_prog.getImplementationPtr();
	for (auto itr = prog_node->getLastNodeIter(); itr != prog_node->getHeadNodeIter(); --itr)
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

		std::reverse(fusing_gate_idxs.begin(), fusing_gate_idxs.end());
		fusing_qubits.clear();
		for (auto &qbit : qubit_vec) {
			fusing_qubits.insert(fusing_qubits.end(), qbit->get_phy_addr());
		}

		/*3.fuse gate with forwarding */
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

void  Fusion::_allocate_new_gate(std::vector<QGate>& gate_v, int index,
    std::vector<int>& fusing_op_indexs, QuantumMachine* qvm)
{
    std::vector<QGate> fusion_gates;
    for (auto itr : fusing_op_indexs) {
        fusion_gates.push_back(gate_v[itr]);
    }

    gate_v[index] = _generate_operation(fusion_gates, qvm);
    for (auto i : fusing_op_indexs)
        if (i != index)
            gate_v[i].getQGate()->setGateType(GateType::GATE_NOP);

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
			tmp_qv[i] = tmp_map[orig2remapped[tmp_qv[i]->get_phy_addr()]];

		}
		op.remap(tmp_qv);
	}

    if (arg_qubits.size() > 2)
    {
        auto fusioned_op = _generate_oracle_gate(fusion_gates, arg_qubits, qvm);
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
    else
    {
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
}