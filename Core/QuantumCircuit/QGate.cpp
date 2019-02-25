#include "QGate.h"
#include "Utilities/ConfigMap.h"
using namespace QGATE_SPACE;
using namespace std;
USING_QPANDA
static QGateNodeFactory * _gs_pGateNodeFactory = QGateNodeFactory::getInstance();
QGATE_FUN_MAP QGateParseMap::m_qgate_function_map = {};
QGate::~QGate()
{
    m_qgate_node.reset();
}

QGate::QGate(const QGate & old_Gate)
{
    m_qgate_node = old_Gate.m_qgate_node;
}

QGate::QGate(Qubit * qbit, QuantumGate *QGate)
{
    if (nullptr == QGate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == qbit)
    {
        QCERR("qbit param err");
        throw invalid_argument("qbit param err");
    }
    m_qgate_node.reset(new OriginQGate(qbit, QGate));
}

QGate::QGate(Qubit *  control_qbit, Qubit * target_qbit, QuantumGate *QGate)
{
    if (nullptr == QGate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == target_qbit)
    {
        QCERR("target_qbit param err");
        throw invalid_argument("target_qbit param err");
    }
    if (nullptr == control_qbit)
    {
        QCERR("control_qbit param err");
        throw invalid_argument("control_qbit param err");
    }

    m_qgate_node.reset(new OriginQGate(control_qbit, target_qbit, QGate));
}


NodeType QGate::getNodeType() const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    auto temp = dynamic_pointer_cast<QNode>(m_qgate_node);
    return temp->getNodeType();
}

size_t QGate::getQuBitVector(QVec& vector) const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_qgate_node->getQuBitVector(vector);
}

size_t QGate::getQuBitNum() const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_qgate_node->getQuBitNum();
}

QuantumGate * QGate::getQGate() const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_qgate_node->getQGate();
}

bool QGate::setDagger(bool is_dagger)
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_qgate_node->setDagger(is_dagger);
}

bool QGate::setControl(QVec qbit_vector)
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_qgate_node->setControl(qbit_vector);
}

std::shared_ptr<QNode> QGate::getImplementationPtr()
{
    return dynamic_pointer_cast<QNode>( m_qgate_node);
}

/*****************************************************************
Name        : dagger
Description : dagger the QGate
argin       :
argout      :
Return      : new QGate
*****************************************************************/
QGate QGate::dagger()
{
    QVec qbit_vector;
    this->getQuBitVector(qbit_vector);
    QVec control_qbit_vector;
    this->getControlVector(control_qbit_vector);

    QStat matrix;
    auto pQgate = this->m_qgate_node->getQGate();
    pQgate->getMatrix(matrix);

    if (qbit_vector.size() == 1)
    {
        string name = "U4";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qbit_vector[0]);
        temp_gate.setControl(control_qbit_vector);
        temp_gate.setDagger(this->isDagger() ^ true);
        return temp_gate;
    }
    else
    {
        string name = "QDoubleGate";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qbit_vector[0], qbit_vector[1]);
        temp_gate.setControl(control_qbit_vector);
        temp_gate.setDagger(this->isDagger() ^ true);
        return temp_gate;
    }
}

/*****************************************************************
Name        : dagger
Description : set controlQubit to QGate
argin       :
argout      :
Return      : new QGate
*****************************************************************/
QGate QGate::control(QVec control_qbit_vector)
{
    QVec qbit_vector;
    this->getQuBitVector(qbit_vector);
    this->getControlVector(control_qbit_vector);

    QStat matrix;
    auto pQgate = this->m_qgate_node->getQGate();

    pQgate->getMatrix(matrix);

    if (qbit_vector.size() == 1)
    {
        string name = "U4";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qbit_vector[0]);
        temp_gate.setControl(control_qbit_vector);
        temp_gate.setDagger(this->isDagger());
        return temp_gate;
    }
    else if(qbit_vector.size() == 2)
    {
        string name = "QDoubleGate";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qbit_vector[0], qbit_vector[1]);
        temp_gate.setControl(control_qbit_vector);
        temp_gate.setDagger(this->isDagger());
        return temp_gate;
    }
    else
    {
        QCERR("qbit_vector is too long");
        throw runtime_error("qbit_vector is too long");
    }
}

bool QGate::isDagger() const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_qgate_node->isDagger();
}

size_t QGate::getControlVector(QVec& qbit_vector) const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_qgate_node->getControlVector(qbit_vector);
}



OriginQGate::~OriginQGate()
{
    if (nullptr != m_qgate)
    {
        delete m_qgate;
    }
}

OriginQGate::OriginQGate(Qubit * qbit, QuantumGate *qgate) :m_Is_dagger(false)
{
    if (nullptr == qgate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == qbit)
    {
        QCERR("qbit param is null");
        throw invalid_argument("qbit param s null");
    }
    m_qgate = qgate;
    m_qbit_vector.push_back(qbit);
    m_node_type = GATE_NODE;
}

OriginQGate::OriginQGate(Qubit * control_qbit, Qubit * target_qbit, QuantumGate * qgate) :m_Is_dagger(false)
{
    if (nullptr == qgate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == target_qbit)
    {
        QCERR("target_qbit param is null");
        throw invalid_argument("target_qbit param s null");
    }
    if (nullptr == control_qbit)
    {
        QCERR("control_qbit param is null");
        throw invalid_argument("control_qbit param s null");
    }
    m_qgate = qgate;
    m_qbit_vector.push_back(control_qbit);
    m_qbit_vector.push_back(target_qbit);
    m_node_type = GATE_NODE;
}

OriginQGate::OriginQGate(QVec &qbit_vector, QuantumGate *qgate) :m_Is_dagger(false)
{
    if (nullptr == qgate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (0 == qbit_vector.size())
    {
        QCERR("qbit_vector err");
        throw invalid_argument("qbit_vector err");
    }
        m_qgate = qgate;
    for (auto aiter = qbit_vector.begin(); aiter != qbit_vector.end(); ++aiter)
    {
        m_qbit_vector.push_back(*aiter);
    }
    m_node_type = GATE_NODE;
}

NodeType OriginQGate::getNodeType() const
{
    return m_node_type;
}

size_t OriginQGate::getQuBitVector(QVec& vector) const
{
    for (auto aiter : m_qbit_vector)
    {
        vector.push_back(aiter);
    }
    return m_qbit_vector.size();
}

size_t OriginQGate::getQuBitNum() const
{
    return m_qbit_vector.size();
}

Qubit * OriginQGate::popBackQuBit()
{
    auto temp = m_qbit_vector.back();
    m_qbit_vector.pop_back();
    return temp;
}

QuantumGate * OriginQGate::getQGate() const
{
    if (nullptr == m_qgate)
    {
        QCERR("m_qgate is null");
        throw runtime_error("m_qgate is null");
    }
    return m_qgate;
}

void OriginQGate::setQGate(QuantumGate * qgate)
{
    m_qgate = qgate;
}

bool OriginQGate::setDagger(bool is_dagger)
{
    m_Is_dagger = is_dagger;
    return m_Is_dagger;
}

bool OriginQGate::setControl(QVec qbit_vector)
{
    for (auto aiter : qbit_vector)
    {
        m_control_qbit_vector.push_back(aiter);
    }
    return true;
}

bool OriginQGate::isDagger() const
{
    return m_Is_dagger;
}

size_t OriginQGate::getControlVector(QVec& qbit_vector) const
{
    for (auto aiter : m_control_qbit_vector)
    {
        qbit_vector.push_back(aiter);
    }
    return qbit_vector.size();
}

void OriginQGate::PushBackQuBit(Qubit * qbit)
{

    if (nullptr == qbit)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    m_qbit_vector.push_back(qbit);

}

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


void OriginQGate::execute(QPUImpl * quantum_gates, QuantumGateParam * param)
{
    bool dagger = m_Is_dagger ^ param->m_is_dagger;
    if (m_qbit_vector.size() <= 0)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    QVec control_qbit_vector;
    for (auto aiter : param->m_control_qbit_vector)
    {
        control_qbit_vector.push_back(aiter);
    }

    for (auto aiter : m_control_qbit_vector)
    {
        control_qbit_vector.push_back(aiter);
    }
    if (control_qbit_vector.size() > 0)
    {
        sort(control_qbit_vector.begin(), 
            control_qbit_vector.end(),
            compareQubit);

        control_qbit_vector.erase(unique(control_qbit_vector.begin(),
                                         control_qbit_vector.end(), Qubitequal),
                                  control_qbit_vector.end());
    }

    for (auto aQIter : m_qbit_vector)
    {
        for (auto aCIter : control_qbit_vector)
        {
            if (Qubitequal(aQIter, aCIter))
            {
                QCERR("targitQubit == controlQubit");
                throw invalid_argument("targitQubit == controlQubit");
            }
        }
    }
    auto aiter = QGateParseMap::getFunction(m_qgate->getOperationNum());
    aiter(m_qgate, m_qbit_vector, quantum_gates, dagger, control_qbit_vector);
}


QGate QGateNodeFactory::getGateNode(const string & name, Qubit * qbit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string & name, Qubit * qbit, double angle)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, angle);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string & name, 
    Qubit * control_qbit, 
    Qubit * target_qbit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);
    QGate  QGateNode(control_qbit, target_qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string & name,
    Qubit * control_qbit,
    Qubit * target_qbit, 
    double theta)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, theta);
    QGate  QGateNode(control_qbit, target_qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(double alpha,
    double beta, 
    double gamma,
    double delta,
    Qubit * qbit)
{
    string name = "U4";
    QuantumGate * pGate = m_pGateFact->getGateNode(name, alpha, beta, gamma, delta);
    QGate  QGateNode(qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(double alpha,
    double beta,
    double gamma, 
    double delta, 
    Qubit * control_qbit, Qubit * target_qbit)
{
    string name = "CU";
    QuantumGate * pGate = m_pGateFact->getGateNode(name, alpha, beta, gamma, delta);
    QGate  QGateNode(control_qbit, target_qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string &name,
    QStat matrix,
    Qubit * control_qbit,
    Qubit * target_qbit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, matrix);
    QGate  QGateNode(control_qbit, target_qbit, pGate);
    return QGateNode;
}

QGate QGateNodeFactory::getGateNode(const string &name,
    QStat matrix,
    Qubit * target_qbit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, matrix);
    QGate  QGateNode(target_qbit, pGate);
    return QGateNode;
}

void QGateParseSingleBit(QuantumGate * qgate, 
    QVec & qbit_vector, 
    QPUImpl* qgates,
    bool is_dagger,
    QVec & control_qbit_vector)
{
    if (nullptr == qgate)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    QStat matrix;
    qgate->getMatrix(matrix);
    Qubit * qbit = *(qbit_vector.begin());
    size_t bit = qbit->getPhysicalQubitPtr()->getQubitAddr();
    if (control_qbit_vector.size() == 0)
    {
        qgates->unitarySingleQubitGate(bit, matrix, is_dagger, 0);
    }
    else
    {
        size_t temp;
        vector<size_t> bit_num_vector;
        for (auto aiter : control_qbit_vector)
        {
            temp = aiter->getPhysicalQubitPtr()->getQubitAddr();
            bit_num_vector.push_back(temp);
        }
        bit_num_vector.push_back(bit);
        qgates->controlunitarySingleQubitGate(bit, bit_num_vector, matrix, is_dagger, 0);
    }

}

void QGateParseDoubleBit(QuantumGate * qgate, 
    QVec & qbit_vector,
    QPUImpl* qgates,
    bool is_dagger,
    QVec & control_qbit_vector)
{
    QStat matrix;
    qgate->getMatrix(matrix);
    auto aiter = qbit_vector.begin();
    Qubit * qbit = *aiter;
    aiter++;
    Qubit * qbit2 = *aiter;
    size_t bit = qbit->getPhysicalQubitPtr()->getQubitAddr();
    size_t bit2 = qbit2->getPhysicalQubitPtr()->getQubitAddr();

    if (control_qbit_vector.size() == 0)
    {
        qgates->unitaryDoubleQubitGate(bit, bit2, matrix, is_dagger, 0);
    }
    else
    {
        size_t temp;
        vector<size_t> bit_num_vector;
        for (auto aiter : control_qbit_vector)
        {
            temp = aiter->getPhysicalQubitPtr()->getQubitAddr();
            bit_num_vector.push_back(temp);
        }
        bit_num_vector.push_back(bit2);
        bit_num_vector.push_back(bit);
        qgates->controlunitaryDoubleQubitGate(bit, bit2, bit_num_vector, matrix, is_dagger, 0);
    }
}

#define REGISTER_QGATE_PARSE(BitCount,FunctionName) \
class insertQGateMapHelper_##FunctionName \
{ \
public: \
     inline insertQGateMapHelper_##FunctionName(int bitCount,QGATE_FUN pFunction) \
    { \
        QGateParseMap::insertMap(bitCount, pFunction); \
    } \
};\
insertQGateMapHelper_##FunctionName _G_insertQGateHelper##FunctionName(BitCount, FunctionName)

REGISTER_QGATE_PARSE(1, QGateParseSingleBit);
REGISTER_QGATE_PARSE(2, QGateParseDoubleBit);


QGate QPanda::X(Qubit * qbit)
{
    string name = "X";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::X1(Qubit * qbit)
{
    string name = "X1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::RX(Qubit * qbit, double angle)
{
    string name = "RX";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate QPanda::U1(Qubit * qbit, double angle)
{
    string name = "U1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate QPanda::Y(Qubit * qbit)
{
    string name = "Y";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::Y1(Qubit * qbit)
{
    string name = "Y1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::RY(Qubit * qbit, double angle)
{
    string name = "RY";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}
QGate QPanda::Z(Qubit * qbit)
{
    string name = "Z";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}
QGate QPanda::Z1(Qubit * qbit)
{
    string name = "Z1";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::RZ(Qubit * qbit, double angle)
{
    string name = "RZ";
    return _gs_pGateNodeFactory->getGateNode(name, qbit, angle);
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "ISWAP";
    return _gs_pGateNodeFactory->getGateNode(name, targitBit_fisrt, targitBit_second);
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second, double theta)
{
    string name = "ISWAP";
    return _gs_pGateNodeFactory->getGateNode(name,
        targitBit_fisrt,
        targitBit_second,
        theta);
}

QGate QPanda::CR(Qubit * control_qbit, Qubit * targit_qbit, double theta)
{
    string name = "CPhaseGate";
    return _gs_pGateNodeFactory->getGateNode(name, control_qbit, targit_qbit, theta);
}

QGate QPanda::SqiSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "SQISWAP";
    return _gs_pGateNodeFactory->getGateNode(name,
        targitBit_fisrt,
        targitBit_second);
}

QGate QPanda::S(Qubit * qbit)
{
    string name = "S";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate QPanda::T(Qubit * qbit)
{
    string name = "T";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate  QPanda::H(Qubit * qbit)
{
    string name = "H";
    return _gs_pGateNodeFactory->getGateNode(name, qbit);
}

QGate  QPanda::CNOT(Qubit * control_qbit, Qubit * target_qbit)
{
    string name = "CNOT";
    return _gs_pGateNodeFactory->getGateNode(name, control_qbit, target_qbit);
}

QGate QPanda::CZ(Qubit * control_qbit, Qubit *target_qbit)
{
    string name = "CZ";
    return _gs_pGateNodeFactory->getGateNode(name, control_qbit, target_qbit);
}

QGate QPanda::U4(double alpha, double beta, double gamma, double delta, Qubit * qbit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, qbit);
}

QGate QPanda::U4(QStat & matrix, Qubit *qubit)
{
    string name = "U4";
    return _gs_pGateNodeFactory->getGateNode(name, matrix, qubit);
}

QGate QPanda::CU(double alpha, 
    double beta,
    double gamma,
    double delta,
    Qubit * control_qbit, 
    Qubit * target_qbit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, control_qbit, target_qbit);
}

QGate QPanda::CU(QStat & matrix, Qubit * control_qbit, Qubit * target_qbit)
{
    string name = "CU";
    return _gs_pGateNodeFactory->getGateNode(name, matrix, control_qbit, target_qbit);
}

QGate QPanda::QDouble(QStat matrix, Qubit * qbit1, Qubit * qbit2)
{
    string name = "QDoubleGate";
    return _gs_pGateNodeFactory->getGateNode(name, matrix, qbit1, qbit2);
}
