#include "Core/QuantumMachine/QProgCheck.h"

USING_QPANDA
using namespace std;


QProgCheck::QProgCheck()
{

}

void QProgCheck::execute(std::shared_ptr<AbstractQGateNode> cur_node, std::shared_ptr<QNode> parent_node, TraversalConfig &param)
{
    bool dagger = cur_node->isDagger() ^ param.m_is_dagger;
    if (cur_node->getTargetQubitNum() <= 0)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    QVec control_qubit_vector;
    cur_node->getControlVector(control_qubit_vector);

    if (param.m_control_qubit_vector.size() > 0)
    {
        control_qubit_vector.insert(control_qubit_vector.end(),
            param.m_control_qubit_vector.begin(), param.m_control_qubit_vector.end());
    }
    
    QVec target_qubit;
    cur_node->getQuBitVector(target_qubit);
    is_can_optimize_measure(control_qubit_vector, target_qubit, param);

    return ;
}

void QProgCheck::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, TraversalConfig &param)
{
    auto qubit_addr = cur_node->getQuBit()->getPhysicalQubitPtr()->getQubitAddr();
    param.m_measure_qubits.push_back(qubit_addr);

    auto cbit_name = cur_node->getCBit()->getName();
    auto cbit = cur_node->getCBit();
    string cbit_number_str = cbit_name.substr(1);
    size_t cbit_addr = stoul(cbit_number_str);
    param.m_measure_cc.push_back(cbit);

    return ;
}

void QProgCheck::execute(std::shared_ptr<AbstractQuantumReset> cur_node,
                         std::shared_ptr<QNode> parent_node, TraversalConfig &param)
{
    param.m_can_optimize_measure = false;
    return ;
}

void QProgCheck::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, TraversalConfig &param)
{
    param.m_can_optimize_measure = false;
    return ;
}

void QProgCheck::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node,
                         TraversalConfig &param)
{
    bool save_dagger = param.m_is_dagger;
    param.m_is_dagger = cur_node->isDagger() ^ param.m_is_dagger;
    QVec control_qubit_vector;
    cur_node->getControlVector(control_qubit_vector);
    auto size_bak = param.m_control_qubit_vector.size();
    param.m_control_qubit_vector.insert(param.m_control_qubit_vector.end(),
                                        control_qubit_vector.begin(), control_qubit_vector.end());

    if (param.m_is_dagger)
    {
        auto iter = cur_node->getLastNodeIter();
        for (; iter != cur_node->getHeadNodeIter() && param.m_can_optimize_measure; --iter)
        {
            auto node = *iter;
            if (nullptr == node)
            {
                QCERR("node is null");
                std::runtime_error("node is null");
            }

            Traversal::traversalByType(node,dynamic_pointer_cast<QNode>(cur_node),*this,param);
        }

    }
    else
    {
        auto iter = cur_node->getFirstNodeIter();
        for (; iter != cur_node->getEndNodeIter() && param.m_can_optimize_measure; ++iter)
        {
            auto node = *iter;
            if (nullptr == node)
            {
                QCERR("node is null");
                std::runtime_error("node is null");
            }

            Traversal::traversalByType(node,dynamic_pointer_cast<QNode>(cur_node),*this,param);
        }
    }

    param.m_is_dagger = save_dagger;
    param.m_control_qubit_vector.erase(param.m_control_qubit_vector.begin() + size_bak,
                                       param.m_control_qubit_vector.end());

    return ;
}

void QProgCheck::execute(std::shared_ptr<AbstractQuantumProgram> cur_node,
                         std::shared_ptr<QNode> parent_node, TraversalConfig &paramu)
{
    if (nullptr == cur_node)
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }

    if (paramu.m_rotation_angle_error > DBL_EPSILON ||
        paramu.m_rotation_angle_error < -DBL_EPSILON)
    {
        paramu.m_can_optimize_measure = false;
        return;
    }

    auto aiter = cur_node->getFirstNodeIter();
    auto end_iter = cur_node->getEndNodeIter();
    if (aiter == cur_node->getEndNodeIter())
        return;


    auto pNode = std::dynamic_pointer_cast<QNode>(cur_node);

    if (nullptr == pNode)
    {
        QCERR("pNode is nullptr");
        throw std::invalid_argument("pNode is nullptr");
    }

    while (aiter != end_iter && paramu.m_can_optimize_measure)
    {
        auto next = aiter.getNextIter();
        Traversal::traversalByType(*aiter,dynamic_pointer_cast<QNode>(cur_node),*this, paramu);
        aiter = next;
    }

    return ;
}


void QProgCheck::execute(std::shared_ptr<AbstractQNoiseNode> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param)
{
    param.m_can_optimize_measure = false;
    return ;
}

void QProgCheck::execute(std::shared_ptr<AbstractQDebugNode> cur_node,
        std::shared_ptr<QNode> parent_node,
        TraversalConfig & param)
{
    param.m_can_optimize_measure = false;
    return ;
}

void QProgCheck::is_can_optimize_measure(const QVec &controls, const QVec &targets, TraversalConfig &param)
{
    if (0 == param.m_measure_qubits.size() || !param.m_can_optimize_measure)
        return ;

    for (auto &control : controls)
    {
        auto addr = control->getPhysicalQubitPtr()->getQubitAddr();
        auto iter = std::find(param.m_measure_qubits.begin(), param.m_measure_qubits.end(), addr);
        if (param.m_measure_qubits.end() != iter)
        {
            param.m_can_optimize_measure = false;
            break;
        }
    }

    for (auto &target : targets)
    {
        auto addr = target->getPhysicalQubitPtr()->getQubitAddr();
        auto iter = std::find(param.m_measure_qubits.begin(), param.m_measure_qubits.end(), addr);
        if (param.m_measure_qubits.end() != iter)
        {
            param.m_can_optimize_measure = false;
            break;
        }
    }

    return ;
}
