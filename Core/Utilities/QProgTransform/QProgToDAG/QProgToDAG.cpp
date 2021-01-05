#include <memory>
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "Core/Utilities/QProgTransform/QProgToDAG/QProgToDAG.h"
USING_QPANDA
using namespace std;

void QProgToDAG::transformQGate(shared_ptr<AbstractQGateNode> gate_node, QProgDAG<GateNodeInfo> &prog_dag, const QCircuitParam& parm, NodeIter& cur_iter)
{
    if (nullptr == gate_node || nullptr == gate_node->getQGate())
    {
        QCERR("gate_node is null");
        throw invalid_argument("gate_node is null");
    }

	GateNodeInfo node_info(cur_iter);
	gate_node->getQuBitVector(node_info.m_qubits_vec);
	gate_node->getControlVector(node_info.m_control_vec);

    for_each(parm.m_control_qubits.begin(), parm.m_control_qubits.end(), [&](Qubit* src_qubit)
    {
        bool find{ false };
        for (auto const &dst_qubit : node_info.m_control_vec)
        {
            if (src_qubit->getPhysicalQubitPtr()->getQubitAddr() ==
                dst_qubit->getPhysicalQubitPtr()->getQubitAddr())
            {
                find == true;
                break;
            }
        }

        if (!find)
        {
			node_info.m_control_vec.emplace_back(src_qubit);
        }
    });

	//check control qubits
	std::vector<int> gate_used_qubits;
	for (auto &itr : node_info.m_qubits_vec)
	{
		gate_used_qubits.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
	}
	for (auto &itr : node_info.m_control_vec)
	{
		gate_used_qubits.push_back(itr->getPhysicalQubitPtr()->getQubitAddr());
	}
	std::sort(gate_used_qubits.begin(), gate_used_qubits.end());
	gate_used_qubits.erase(unique(gate_used_qubits.begin(), gate_used_qubits.end()), gate_used_qubits.end());
	if (gate_used_qubits.size() != (node_info.m_qubits_vec.size() + node_info.m_control_vec.size()))
	{
		QCERR_AND_THROW(runtime_error, "Control gate Error: Illegal control qubits.");
	}

	node_info.m_dagger = gate_node->isDagger() ^ parm.m_is_dagger;

    auto vertice_num = prog_dag.add_vertex(node_info);

    for_each(node_info.m_qubits_vec.begin(), node_info.m_qubits_vec.end(), [&](Qubit* qubit)
    {
        prog_dag.add_qubit_map(qubit->getPhysicalQubitPtr()->getQubitAddr(), vertice_num);
    });

    for_each(node_info.m_control_vec.begin(), node_info.m_control_vec.end(), [&](Qubit* qubit)
    {
        prog_dag.add_qubit_map(qubit->getPhysicalQubitPtr()->getQubitAddr(), vertice_num);
    });
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG<GateNodeInfo> & prog_dag, QCircuitParam&, NodeIter& cur_iter)
{
    transformQMeasure(cur_node, prog_dag, cur_iter);
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QProgDAG<GateNodeInfo> &prog_dag, QCircuitParam&, NodeIter& cur_iter)
{
	transformQReset(cur_node, prog_dag, cur_iter);
}

void QProgToDAG::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG<GateNodeInfo> & prog_dag, QCircuitParam& parm, NodeIter& cur_iter)
{
    transformQGate(cur_node, prog_dag, parm, cur_iter);
}

void QProgToDAG::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG<GateNodeInfo> &prog_dag, QCircuitParam&, NodeIter& cur_iter)
{
    QCERR("ignore classical prog node");
}

void QProgToDAG::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QProgDAG<GateNodeInfo> &prog_dag, QCircuitParam&, NodeIter& cur_iter)
{
    QCERR("ignore controlflow node");
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG<GateNodeInfo> &prog_dag, QCircuitParam& parm, NodeIter& cur_iter)
{
    if (nullptr == cur_node)
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    auto aiter = cur_node->getFirstNodeIter();
    if (aiter == cur_node->getEndNodeIter())
        return;

    auto node = std::dynamic_pointer_cast<QNode>(cur_node);
    if (nullptr == node)
    {
        QCERR("node is nullptr");
        throw std::invalid_argument("node is nullptr");
    }

    while (aiter != cur_node->getEndNodeIter())
    {
        auto next = aiter.getNextIter();
        Traversal::traversalByType(*aiter, node, *this, prog_dag, parm, aiter);
        aiter = next;
    }
}

void QProgToDAG::transformQMeasure(std::shared_ptr<AbstractQuantumMeasure> cur_node, QProgDAG<GateNodeInfo> &prog_dag, NodeIter& cur_iter)
{
    if (nullptr == cur_node)
    {
        QCERR("measure_node is null");
        throw invalid_argument("measure_node is null");
    }

    size_t vertice_num = prog_dag.add_vertex(cur_iter);
    auto tar_qubit = cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    prog_dag.add_qubit_map(tar_qubit, vertice_num);
}

void QProgToDAG::transformQReset(std::shared_ptr<AbstractQuantumReset> cur_node, QProgDAG<GateNodeInfo> &prog_dag, NodeIter& cur_iter)
{
	if (nullptr == cur_node)
	{
		QCERR("reset_node is null");
		throw invalid_argument("reset_node is null");
	}

	size_t vertice_num = prog_dag.add_vertex(cur_iter);
	auto tar_qubit = cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
	prog_dag.add_qubit_map(tar_qubit, vertice_num);
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QProgDAG<GateNodeInfo> &prog_dag, QCircuitParam& cir_parm, NodeIter& cur_iter)
{
    if (nullptr == cur_node)
    {
        QCERR("pQCircuit is nullptr");
        throw std::invalid_argument("pQCircuit is nullptr");
    }

    auto aiter = cur_node->getFirstNodeIter();
    if (aiter == cur_node->getEndNodeIter())
        return;

    auto node = std::dynamic_pointer_cast<QNode>(cur_node);
    if (nullptr == node)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }

    bool cur_node_is_dagger = cur_node->isDagger() ^ (cir_parm.m_is_dagger);
    QVec ctrl_qubits;
    cur_node->getControlVector(ctrl_qubits);

    auto tmp_param = cir_parm.clone();
    tmp_param->m_is_dagger = cur_node_is_dagger;
    tmp_param->append_control_qubits(QCircuitParam::get_real_append_qubits(ctrl_qubits, cir_parm.m_control_qubits));
    if (cur_node_is_dagger)
    {
        auto aiter = cur_node->getLastNodeIter();
        if (nullptr == *aiter)
        {
            return;
        }
        while (aiter != cur_node->getHeadNodeIter())
        {
            if (aiter == nullptr)
            {
                break;
            }
            Traversal::traversalByType(*aiter, node, *this, prog_dag, *tmp_param, aiter);
            --aiter;
        }

    }
    else
    {
        auto aiter = cur_node->getFirstNodeIter();
        while (aiter != cur_node->getEndNodeIter())
        {
            auto next = aiter.getNextIter();
            Traversal::traversalByType(*aiter, node, *this, prog_dag, *tmp_param, aiter);
            aiter = next;
        }
    }
}
