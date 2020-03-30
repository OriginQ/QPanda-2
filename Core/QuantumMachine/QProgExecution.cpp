#include "Core/QuantumMachine/QProgExecution.h"

USING_QPANDA
using namespace std;
static bool compareQubit(Qubit * a, Qubit * b)
{
    return a->getPhysicalQubitPtr()->getQubitAddr() <
        b->getPhysicalQubitPtr()->getQubitAddr();
}

static bool Qubitequal(Qubit * a, Qubit * b)
{
    return a->getPhysicalQubitPtr()->getQubitAddr() == 
        b->getPhysicalQubitPtr()->getQubitAddr();
}

void QProgExecution::execute(std::shared_ptr<AbstractQGateNode> cur_node,
    std::shared_ptr<QNode> parent_node,
    TraversalConfig & param,
	QPUImpl* qpu)
{
	bool dagger = cur_node->isDagger() ^ param.m_is_dagger;
    if (cur_node->getTargetQubitNum() <= 0)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    QVec control_qubit_vector;
    for (auto aiter : param.m_control_qubit_vector)
    {
        control_qubit_vector.push_back(aiter);
    }

	cur_node->getControlVector(control_qubit_vector);

    if (control_qubit_vector.size() > 0)
    {
        sort(control_qubit_vector.begin(), 
            control_qubit_vector.end(),
            compareQubit);

        control_qubit_vector.erase(unique(control_qubit_vector.begin(),
                                         control_qubit_vector.end(), Qubitequal),
                                  control_qubit_vector.end());
    }

	QVec target_qubit;
	cur_node->getQuBitVector(target_qubit);

    for (auto aQIter : target_qubit)
    {
        for (auto aCIter : control_qubit_vector)
        {
            if (Qubitequal(aQIter, aCIter))
            {
                QCERR("targitQubit == controlQubit");
                throw invalid_argument("targitQubit == controlQubit");
            }
        }
    }

	auto qgate = cur_node->getQGate();

    auto aiter = QGateParseMap::getFunction(qgate->getOperationNum());
    if (nullptr == aiter)
    {
        stringstream error;
        error << "gate operation num error ";
        QCERR(error.str());
        throw run_fail(error.str());
    }
    aiter(qgate, target_qubit, qpu, dagger, control_qubit_vector, (GateType)qgate->getGateType());
}


void QProgExecution::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node,
	std::shared_ptr<QNode> parent_node,
	TraversalConfig & param,
	QPUImpl* qpu)
{
	 int iResult = qpu->qubitMeasure(cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
    if (iResult < 0)
    {
        QCERR("result error");
        throw runtime_error("result error");
    }
    CBit * cexpr = cur_node->getCBit();
    if (nullptr == cexpr)
    {
        QCERR("unknow error");
        throw runtime_error("unknow error");
    }

    cexpr->setValue(iResult);
    string name = cexpr->getName();
    auto aiter = m_result.find(name);
    if (aiter != m_result.end())
    {
        aiter->second = (bool)iResult;
    }
    else
    {
		m_result.insert(pair<string, bool>(name, (bool)iResult));
    }
}

void QProgExecution::execute(std::shared_ptr<AbstractQuantumReset> cur_node,
	std::shared_ptr<QNode> parent_node,
	TraversalConfig & param,
	QPUImpl* qpu)
{
    qpu->Reset(cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
}

void QProgExecution::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, TraversalConfig &param, QPUImpl *qpu)
{
    if (nullptr == cur_node)
    {
        QCERR("control_flow_node is nullptr");
        throw std::invalid_argument("control_flow_node is nullptr");
    }

    auto node = std::dynamic_pointer_cast<QNode>(cur_node);
    if (nullptr == node)
    {
        QCERR("Unknown internal error");
        throw std::runtime_error("Unknown internal error");
    }
    auto node_type = node->getNodeType();

    auto cexpr = cur_node->getCExpr();
    if (WHILE_START_NODE == node_type)
    {
        while(cexpr.eval())
        {
            auto true_branch_node = cur_node->getTrueBranch();
            Traversal::traversalByType(true_branch_node, node, *this,param,qpu);
        }
    }
    else if (QIF_START_NODE == node_type)
    {
        if (cexpr.eval())
        {
            auto true_branch_node = cur_node->getTrueBranch();
            Traversal::traversalByType(true_branch_node, node, *this,param,qpu);
        }
        else
        {
            auto false_branch_node = cur_node->getFalseBranch();
            if (nullptr != false_branch_node)
            {
                Traversal::traversalByType(false_branch_node, node, *this,param,qpu);
            }
        }
    }

    return ;
}

void QProgExecution::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node,
    std::shared_ptr<QNode> parent_node,
    TraversalConfig & param,
    QPUImpl* qpu)
{
    bool save_dagger = param.m_is_dagger;
    size_t control_qubit_count = 0;

    param.m_is_dagger = cur_node->isDagger() ^ param.m_is_dagger;

    QVec control_qubit_vector;

    cur_node->getControlVector(control_qubit_vector);
    for (auto aiter : control_qubit_vector)
    {
        param.m_control_qubit_vector.push_back(aiter);
        control_qubit_count++;
    }

    if (param.m_is_dagger)
    {
        auto aiter = cur_node->getLastNodeIter();
        if (nullptr == *aiter)
        {
            return ;
        }
        for (; aiter != cur_node->getHeadNodeIter(); --aiter)
        {
            auto node = *aiter;
            if (nullptr == node)
            {
                QCERR("node is null");
                std::runtime_error("node is null");
            }
            
            Traversal::traversalByType(node,dynamic_pointer_cast<QNode>(cur_node),*this,param,qpu);
        }

    }
    else
    {
        auto aiter = cur_node->getFirstNodeIter();
        if (nullptr == *aiter)
        {
            return ;
        }
        for (; aiter != cur_node->getEndNodeIter(); ++aiter)
        {
            auto node = *aiter;
            if (nullptr == node)
            {
                QCERR("node is null");
                std::runtime_error("node is null");
            }

            Traversal::traversalByType(node,dynamic_pointer_cast<QNode>(cur_node),*this,param,qpu);
        }
    }

    param.m_is_dagger = save_dagger;

    for (size_t i = 0; i < control_qubit_count; i++)
    {
        param.m_control_qubit_vector.pop_back();
    }

}
