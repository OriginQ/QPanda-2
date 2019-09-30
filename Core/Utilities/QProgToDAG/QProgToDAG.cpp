#include "Core/Utilities/QProgToDAG/QProgToDAG.h"
#include <memory>
USING_QPANDA
using namespace std;

void QProgToDAG::construct(size_t tar_qubit, size_t vertice_num, QProgDAG &prog_dag)
{
    auto iter = qubit_vertices_map.find(tar_qubit);
    if (iter != qubit_vertices_map.end())
    {
        size_t in_vertex_num = iter->second;
        prog_dag.addEgde(in_vertex_num, vertice_num);
        qubit_vertices_map[iter->first] = vertice_num;
    }
    else
    {
        qubit_vertices_map.insert(make_pair(tar_qubit, vertice_num));
    }
}

void QProgToDAG::transformQGate(shared_ptr<AbstractQGateNode> pQGate, QProgDAG &prog_dag, NodeIter& curIter)
{
    if (nullptr == pQGate || nullptr == pQGate->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    QVec qubits_vector;
    pQGate->getQuBitVector(qubits_vector);

    /*auto gate_ptr = dynamic_pointer_cast<QNode>(pQGate);
    auto vertice_num = prog_dag.addVertex(gate_ptr);*/
	auto vertice_num = prog_dag.addVertex(curIter);

    switch (pQGate->getQGate()->getGateType())
    {
        case GateType::P0_GATE:
        case GateType::P1_GATE:
        case GateType::PAULI_X_GATE:
        case GateType::PAULI_Y_GATE:
        case GateType::PAULI_Z_GATE:
        case GateType::HADAMARD_GATE:
        case GateType::T_GATE:
        case GateType::S_GATE:
        case GateType::RX_GATE:
        case GateType::RY_GATE:
        case GateType::RZ_GATE:
        case GateType::U1_GATE:
        case GateType::X_HALF_PI:
        case GateType::Y_HALF_PI:
        case GateType::Z_HALF_PI:
            {
                construct(qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(), vertice_num, prog_dag);
            }
            break;

        case GateType::CNOT_GATE:
        case GateType::CZ_GATE:
        case GateType::ISWAP_GATE:
        case GateType::SQISWAP_GATE:
        case GateType::CPHASE_GATE:
		case GateType::SWAP_GATE:
            {
                construct(qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(), vertice_num, prog_dag);
                construct(qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr(), vertice_num, prog_dag);
            }
            break;

        default:
            QCERR("do not support this gate type");
            throw invalid_argument("do not support this gate type");
            break;
    }
}


void QProgToDAG::transformQMeasure(std::shared_ptr<AbstractQuantumMeasure> pQMeasure, QProgDAG &prog_dag, NodeIter& curIter)
{
    if (nullptr == pQMeasure)
    {
        QCERR("measure_node is null");
        throw invalid_argument("measure_node is null");
    }

    /*auto measure_ptr = dynamic_pointer_cast<QNode>(pQMeasure);
    size_t vertice_num = prog_dag.addVertex(measure_ptr);*/
	size_t vertice_num = prog_dag.addVertex(curIter);
    auto tar_qubit = pQMeasure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    construct(tar_qubit, vertice_num, prog_dag);
}


void QProgToDAG::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG & prog_dag, NodeIter& curIter)
{
    transformQGate(cur_node, prog_dag, curIter);
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumMeasure>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG & prog_dag, NodeIter& curIter)
{
    transformQMeasure(cur_node, prog_dag, curIter);
}

void QProgToDAG::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QProgDAG &prog_dag, NodeIter& curIter)
{
    QCERR("ignore ControlFlowNode.");
    /*throw std::runtime_error("Does not support ControlFlowNode"); modified by zhaodongyi*/
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QProgDAG &prog_dag, NodeIter& curIter)
{
    //Traversal::traversal(cur_node, false, *this, prog_dag);

	if (nullptr == cur_node)
	{
		QCERR("pQCircuit is nullptr");
		throw std::invalid_argument("pQCircuit is nullptr");
	}

	auto aiter = cur_node->getFirstNodeIter();

	if (aiter == cur_node->getEndNodeIter())
		return;

	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

	if (nullptr == pNode)
	{
		QCERR("Unknown internal error");
		throw std::runtime_error("Unknown internal error");
	}
	auto is_dagger = false;
	if (false)
	{
		is_dagger = cur_node->isDagger();
	}

	if (is_dagger)
	{
		auto aiter = cur_node->getLastNodeIter();
		if (nullptr == *aiter)
		{
			return;
		}
		while (aiter != cur_node->getHeadNodeIter())
		{
			//auto next = --aiter;
			if (aiter == nullptr)
			{
				break;
			}
			Traversal::traversalByType(*aiter, pNode, *this, prog_dag, aiter);
			//aiter = next;
			--aiter;
		}

	}
	else
	{
		auto aiter = cur_node->getFirstNodeIter();
		while (aiter != cur_node->getEndNodeIter())
		{
			auto next = aiter.getNextIter();
			Traversal::traversalByType(*aiter, pNode, *this, prog_dag, aiter);
			aiter = next;
		}
	}
}

void QProgToDAG::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG &prog_dag, NodeIter& curIter)
{
    //Traversal::traversal(cur_node, *this, prog_dag);

	if (nullptr == cur_node)
	{
		QCERR("param error");
		throw std::invalid_argument("param error");
	}

	auto aiter = cur_node->getFirstNodeIter();

	if (aiter == cur_node->getEndNodeIter())
		return;


	auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

	if (nullptr == pNode)
	{
		QCERR("pNode is nullptr");
		throw std::invalid_argument("pNode is nullptr");
	}

	while (aiter != cur_node->getEndNodeIter())
	{
		auto next = aiter.getNextIter();
		Traversal::traversalByType(*aiter, pNode, *this, prog_dag, aiter);
		aiter = next;
	}
}

void QProgToDAG::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QProgDAG &prog_dag, NodeIter& curIter)
{
	QCERR("ignore ClassicalProg.");
   /* QCERR("Does not support ClassicalProg ");
    throw std::runtime_error("Does not support ClassicalProg"); modified by zhaodongyi*/
}
