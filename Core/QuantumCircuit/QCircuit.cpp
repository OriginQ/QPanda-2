#include "QCircuit.h"
#include "Core/Utilities/QProgInfo/ConfigMap.h"
#include "QGate.h"
USING_QPANDA
using namespace std;
QCircuit  QPanda::CreateEmptyCircuit()
{
    QCircuit temp;
    return temp;
}

QCircuit  QPanda::createEmptyCircuit()
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
    m_pQuantumCircuit->pushBackNode(dynamic_pointer_cast<QNode>(gate.getImplementationPtr()));
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

std::shared_ptr<AbstractQuantumCircuit> QCircuit::getImplementationPtr()
{
    if (!m_pQuantumCircuit)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_pQuantumCircuit;
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

QCircuit  QCircuit::control(const QVec qubit_vector)
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

NodeIter QCircuit::insertQNode(const NodeIter & iter, shared_ptr<QNode> node)
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

void QCircuit::setControl(const QVec control_qubit_vector)
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
}

void OriginCircuit::setDagger(bool is_dagger)
{
    m_Is_dagger = is_dagger;
}

void OriginCircuit::setControl(const QVec qubit_vector)
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

void OriginCircuit::clearControl()
{
    m_control_qubit_vector.clear();
    m_control_qubit_vector.resize(0);
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
        m_pQuantumCircuit->pushBackNode(std::dynamic_pointer_cast<QNode>(temp.getImplementationPtr()));
    }
}

HadamardQCircuit QPanda::CreateHadamardQCircuit(QVec & qubit_vector)
{
    HadamardQCircuit temp(qubit_vector);
    return temp;
}

HadamardQCircuit QPanda::createHadamardQCircuit(QVec & qubit_vector)
{
    HadamardQCircuit temp(qubit_vector);
    return temp;
}
