#include "QCircuit.h"
#include "Utilities/ConfigMap.h"
#include "QGate.h"
USING_QPANDA
using namespace std;
QCircuit  QPanda::CreateEmptyCircuit()
{
    QCircuit temp;
    return temp;
}

QCircuit::QCircuit(std::shared_ptr<AbstractQuantumCircuit> node)
{
    if (!node)
    {
        QCERR("node is null shared_ptr");
        throw invalid_argument("node is null shared_ptr");
    }

    m_pQuantumCircuit = node;
}

QCircuit::QCircuit()
{
    auto class_name = ConfigMap::getInstance()["QCircuit"];
    auto qcircuit = QuantumCircuitFactory::getInstance().getQuantumCircuit(class_name);

    m_pQuantumCircuit.reset(qcircuit);
}


QCircuit::QCircuit(QGate & gate)
{
    auto class_name = ConfigMap::getInstance()["QCircuit"];
    auto qcircuit = QuantumCircuitFactory::getInstance().getQuantumCircuit(class_name);

    m_pQuantumCircuit.reset(qcircuit);
    m_pQuantumCircuit->pushBackNode(gate.getImplementationPtr());
}

QCircuit::QCircuit(const QCircuit & old_qcircuit)
{
    m_pQuantumCircuit = old_qcircuit.m_pQuantumCircuit;
}


QCircuit::~QCircuit()
{
    if (m_pQuantumCircuit)
    {
        m_pQuantumCircuit.reset();
    }
}

std::shared_ptr<QNode> QCircuit::getImplementationPtr()
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return dynamic_pointer_cast<QNode>(m_pQuantumCircuit);
}


void QCircuit::pushBackNode(QNode * node)
{
    if (nullptr == node)
    {
        QCERR("node is nullptr");
        throw runtime_error("node is nullptr");
    }
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQuantumCircuit->pushBackNode(node);
}

void QCircuit::pushBackNode(std::shared_ptr<QNode> node)
{
    if (!node)
    {
        QCERR("node is null");
        throw runtime_error("node is null");
    }
    m_pQuantumCircuit->pushBackNode(node);
}

QCircuit QCircuit::dagger()
{
    QCircuit qCircuit;
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aiter = m_pQuantumCircuit->getFirstNodeIter();
    if (aiter == m_pQuantumCircuit->getEndNodeIter())
    {
        return qCircuit;
    }

    for (; aiter != m_pQuantumCircuit->getEndNodeIter(); ++aiter)
    {
        qCircuit.pushBackNode(*aiter);
    }

    qCircuit.setDagger(true ^ this->isDagger());
    return qCircuit;
}

QCircuit  QCircuit::control(QVec qubit_vector)
{
    QCircuit qcircuit;
    if (nullptr == m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto aiter = m_pQuantumCircuit->getFirstNodeIter();
    if (aiter == m_pQuantumCircuit->getEndNodeIter())
    {
        return qcircuit;
    }
    for (; aiter != m_pQuantumCircuit->getEndNodeIter(); ++aiter)
    {
        qcircuit.pushBackNode(*aiter);
    }

    qcircuit.setControl(qubit_vector);
    return qcircuit;
}


NodeType QCircuit::getNodeType() const
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return dynamic_pointer_cast<QNode>(m_pQuantumCircuit)->getNodeType();
}

bool QCircuit::isDagger() const
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->isDagger();
}

bool QCircuit::getControlVector(QVec& qubit_vector)
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getControlVector(qubit_vector);
}

NodeIter  QCircuit::getFirstNodeIter()
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getFirstNodeIter();
}

NodeIter  QCircuit::getLastNodeIter()
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getLastNodeIter();
}

NodeIter QCircuit::getEndNodeIter()
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getEndNodeIter();
}

NodeIter QCircuit::getHeadNodeIter()
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->getHeadNodeIter();
}

NodeIter QCircuit::insertQNode(NodeIter & iter, QNode * node)
{
    if (nullptr == node)
    {
        QCERR("node is nullptr");
        throw runtime_error("node is nullptr");
    }

    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->insertQNode(iter, node);
}

NodeIter QCircuit::deleteQNode(NodeIter & iter)
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_pQuantumCircuit->deleteQNode(iter);
}

void QCircuit::setDagger(bool is_dagger)
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQuantumCircuit->setDagger(is_dagger);
}

void QCircuit::setControl(QVec control_qubit_vector)
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    m_pQuantumCircuit->setControl(control_qubit_vector);
}

OriginCircuit::~OriginCircuit()
{
    Item *temp;

    while (m_head != m_end)
    {
        m_head->setPre(nullptr);
        temp = m_head;
        m_head = m_head->getNext();
        delete temp;
    }
    delete m_head;
    m_head = nullptr;
    m_end = nullptr;
}

void OriginCircuit::pushBackNode(QNode * node)
{
    if (nullptr == node)
    {
        QCERR("node is null");
        throw invalid_argument("node is null");
    }
    auto temp = node->getImplementationPtr();
    pushBackNode(temp);
}

void OriginCircuit::pushBackNode(std::shared_ptr<QNode> node)
{
    if (!node)
    {
        QCERR("node is null");
        throw invalid_argument("node is null");
    }
    WriteLock wl(m_sm);
    try
    {
        if (m_end == m_head)
        {
            Item *end_iter = new OriginItem();
            m_head->setNode(node);
            end_iter->setNext(nullptr);
            end_iter->setPre(m_head);
            m_head->setNext(end_iter);
            m_end = end_iter;
        }
        else
        {
            Item *iter = new OriginItem();
            iter->setNext(nullptr);
            iter->setPre(m_end);
            m_end->setNext(iter);
            m_end->setNode(node);
            m_end = iter;
        }
    }
    catch (exception &memExp)
    {
        QCERR(memExp.what());
        throw memExp;
    }
}

void OriginCircuit::setDagger(bool is_dagger)
{
    m_Is_dagger = is_dagger;
}

void OriginCircuit::setControl(QVec qubit_vector)
{
    for (auto aiter : qubit_vector)
    {
        m_control_qubit_vector.push_back(aiter);
    }
}

NodeType OriginCircuit::getNodeType() const
{
    return m_node_type;
}

bool OriginCircuit::isDagger() const
{
    return m_Is_dagger;
}

bool OriginCircuit::getControlVector(QVec& qubit_vector)
{
    for (auto aiter : m_control_qubit_vector)
    {
        qubit_vector.push_back(aiter);
    }
    return m_control_qubit_vector.size();
}

NodeIter OriginCircuit::getFirstNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_head);
    return temp;
}

NodeIter OriginCircuit::getLastNodeIter()
{
    ReadLock rl(m_sm);
    NodeIter temp(m_end->getPre());
    return temp;
}

NodeIter OriginCircuit::getEndNodeIter()
{
    NodeIter temp(m_end);
    return temp;
}

NodeIter OriginCircuit::getHeadNodeIter()
{
    NodeIter temp;
    return temp;
}

NodeIter OriginCircuit::insertQNode(NodeIter & perIter, QNode * node)
{
    ReadLock * rl = new ReadLock(m_sm);
    Item * perItem = perIter.getPCur();
    if (nullptr == perItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    if (m_end == m_head)
    {
        QCERR("The perIter is not in the prog");
        throw runtime_error("The perIter is not in the prog");
    }

    auto aiter = this->getFirstNodeIter();

    for (; aiter != this->getEndNodeIter(); aiter++)
    {
        if (perItem == aiter.getPCur())
        {
            break;
        }
    }
    if (aiter == this->getEndNodeIter())
    {
        QCERR("The perIter is not in the qprog");
        throw runtime_error("The perIter is not in the qprog");
    }

    delete rl;
    WriteLock wl(m_sm);
    Item *curItem = new OriginItem();
    auto ptemp = node->getImplementationPtr();

    if (m_end != perItem->getNext())
    {
        curItem->setNode(ptemp);
        perItem->getNext()->setPre(curItem);
        curItem->setNext(perItem->getNext());
        perItem->setNext(curItem);
        curItem->setPre(perItem);
        NodeIter temp(curItem);
        return temp;
    }
    else
    {
        m_end->setNode(ptemp);
        m_end->setNext(curItem);
        curItem->setPre(m_end);
        curItem->setNext(nullptr);
        NodeIter temp(m_end);
        m_end = curItem;
        return temp;
    }
}

NodeIter OriginCircuit::deleteQNode(NodeIter & target_iter)
{
    ReadLock *rl = new ReadLock(m_sm);
    Item * target_item = target_iter.getPCur();
    if (nullptr == target_item)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }


    if (m_end == m_head)
    {
        QCERR("The target_iter is not in the qprogget_iter");
        throw runtime_error("The target_iter is not in the qprog");
    }

    auto aiter = this->getFirstNodeIter();
    for (; aiter != this->getEndNodeIter(); aiter++)
    {
        if (target_item == aiter.getPCur())
        {
            break;
        }
    }
    if (aiter == this->getEndNodeIter())
    {
        QCERR("The target_iter is not in the qprogget_iter");
        throw runtime_error("The target_iter is not in the qprogget_iter");
    }


    delete rl;
    WriteLock wl(m_sm);

    if (m_head == target_item)
    {

        m_head = target_item->getNext();
        m_head->setPre(nullptr);
        delete target_item;
        target_iter.setPCur(nullptr);
        NodeIter temp(m_head);
        return temp;
    }

    Item * perItem = target_item->getPre();
    if (nullptr == perItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    Item * nextItem = target_item->getNext();
    if (nullptr == nextItem)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    perItem->setNext(nextItem);
    nextItem->setPre(perItem);
    delete target_item;
    target_iter.setPCur(nullptr);
    NodeIter temp(perItem);
    return temp;
}

void OriginCircuit::clearControl()
{
    m_control_qubit_vector.clear();
    m_control_qubit_vector.resize(0);
}

void OriginCircuit::execute(QPUImpl * quantum_gates, QuantumGateParam * param)
{
    bool save_dagger = param->m_is_dagger;
    size_t control_qubit_count = 0;

    param->m_is_dagger = isDagger() ^ param->m_is_dagger;

    for (auto aiter : m_control_qubit_vector)
    {
        param->m_control_qubit_vector.push_back(aiter);
        control_qubit_count++;
    }

    if (param->m_is_dagger)
    {
        auto aiter = getLastNodeIter();
        if (nullptr == *aiter)
        {
            return ;
        }
        for (; aiter != getHeadNodeIter(); --aiter)
        {
            auto node = *aiter;
            if (nullptr == node)
            {
                QCERR("node is null");
                std::runtime_error("node is null");
            }
            
            node->execute(quantum_gates, param);
        }

    }
    else
    {
        auto aiter = getFirstNodeIter();
        if (nullptr == *aiter)
        {
            return ;
        }
        for (; aiter != getEndNodeIter(); ++aiter)
        {
            auto node = *aiter;
            if (nullptr == node)
            {
                QCERR("node is null");
                std::runtime_error("node is null");
            }

            node->execute(quantum_gates, param);
        }
    }

    param->m_is_dagger = save_dagger;

    for (size_t i = 0; i < control_qubit_count; i++)
    {
        param->m_control_qubit_vector.pop_back();
    }
}

void QuantumCircuitFactory::registClass(string name, CreateQCircuit method)
{
    if ((name.size() <= 0) || (nullptr == method))
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    m_QCirciutMap.insert(pair<string, CreateQCircuit>(name, method));
}

AbstractQuantumCircuit * QuantumCircuitFactory::getQuantumCircuit(std::string & name)
{
    if (name.size() <= 0)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    auto aiter = m_QCirciutMap.find(name);
    if (aiter != m_QCirciutMap.end())
    {
        return aiter->second();
    }
    return nullptr;
}
REGISTER_QCIRCUIT(OriginCircuit);



HadamardQCircuit::HadamardQCircuit(QVec& qubit_vector)
{
    for (auto aiter : qubit_vector)
    {
        auto  temp = H(aiter);
        m_pQuantumCircuit->pushBackNode((QNode *)&temp);
    }
}

HadamardQCircuit QPanda::CreateHadamardQCircuit(QVec & qubit_vector)
{
    HadamardQCircuit temp(qubit_vector);
    return temp;
}
