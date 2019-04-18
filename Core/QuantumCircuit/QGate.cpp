#include "QGate.h"
#include "Utilities/ConfigMap.h"
#include "QPandaException.h"
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

QGate::QGate(Qubit * qubit, QuantumGate *QGate)
{
    if (nullptr == QGate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == qubit)
    {
        QCERR("qubit param err");
        throw invalid_argument("qubit param err");
    }
    m_qgate_node.reset(new OriginQGate(qubit, QGate));
}

QGate::QGate(Qubit *  control_qubit, Qubit * target_qubit, QuantumGate *QGate)
{
    if (nullptr == QGate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == target_qubit)
    {
        QCERR("target_qubit param err");
        throw invalid_argument("target_qubit param err");
    }
    if (nullptr == control_qubit)
    {
        QCERR("control_qubit param err");
        throw invalid_argument("control_qubit param err");
    }

    m_qgate_node.reset(new OriginQGate(control_qubit, target_qubit, QGate));
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

size_t QGate::getTargetQubitNum() const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_qgate_node->getTargetQubitNum();
}

size_t QGate::getControlQubitNum() const
{
        if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_qgate_node->getControlQubitNum();
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

bool QGate::setControl(QVec qubit_vector)
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }
    return m_qgate_node->setControl(qubit_vector);
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
    QVec qubit_vector;
    this->getQuBitVector(qubit_vector);
    QVec control_qubit_vector;
    this->getControlVector(control_qubit_vector);

    QStat matrix;
    auto pQgate = this->m_qgate_node->getQGate();
    pQgate->getMatrix(matrix);

    if (qubit_vector.size() == 1)
    {
        string name = "U4";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubit_vector[0]);
        temp_gate.setControl(control_qubit_vector);
        temp_gate.setDagger(this->isDagger() ^ true);
        return temp_gate;
    }
    else
    {
        string name = "QDoubleGate";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubit_vector[0], qubit_vector[1]);
        temp_gate.setControl(control_qubit_vector);
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
QGate QGate::control(QVec control_qubit_vector)
{
    QVec qubit_vector;
    this->getQuBitVector(qubit_vector);
    this->getControlVector(control_qubit_vector);

    QStat matrix;
    auto pQgate = this->m_qgate_node->getQGate();

    pQgate->getMatrix(matrix);

    if (qubit_vector.size() == 1)
    {
        string name = "U4";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubit_vector[0]);
        temp_gate.setControl(control_qubit_vector);
        temp_gate.setDagger(this->isDagger());
        return temp_gate;
    }
    else if(qubit_vector.size() == 2)
    {
        string name = "QDoubleGate";
        auto temp_gate = _gs_pGateNodeFactory->getGateNode(name, matrix, qubit_vector[0], qubit_vector[1]);
        temp_gate.setControl(control_qubit_vector);
        temp_gate.setDagger(this->isDagger());
        return temp_gate;
    }
    else
    {
        QCERR("qubit_vector is too long");
        throw runtime_error("qubit_vector is too long");
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

size_t QGate::getControlVector(QVec& qubit_vector) const
{
    if (!m_qgate_node)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return m_qgate_node->getControlVector(qubit_vector);
}



OriginQGate::~OriginQGate()
{
    if (nullptr != m_qgate)
    {
        delete m_qgate;
    }
}

OriginQGate::OriginQGate(Qubit * qubit, QuantumGate *qgate) :m_Is_dagger(false)
{
    if (nullptr == qgate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == qubit)
    {
        QCERR("qubit param is null");
        throw invalid_argument("qubit param s null");
    }
    m_qgate = qgate;
    m_qubit_vector.push_back(qubit);
    m_node_type = GATE_NODE;
}

OriginQGate::OriginQGate(Qubit * control_qubit, Qubit * target_qubit, QuantumGate * qgate) :m_Is_dagger(false)
{
    if (nullptr == qgate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (nullptr == target_qubit)
    {
        QCERR("target_qubit param is null");
        throw invalid_argument("target_qubit param s null");
    }
    if (nullptr == control_qubit)
    {
        QCERR("control_qubit param is null");
        throw invalid_argument("control_qubit param s null");
    }
    m_qgate = qgate;
    m_qubit_vector.push_back(control_qubit);
    m_qubit_vector.push_back(target_qubit);
    m_node_type = GATE_NODE;
}

OriginQGate::OriginQGate(QVec &qubit_vector, QuantumGate *qgate) :m_Is_dagger(false)
{
    if (nullptr == qgate)
    {
        QCERR("qgate param err");
        throw invalid_argument("qgate param err");
    }
    if (0 == qubit_vector.size())
    {
        QCERR("qubit_vector err");
        throw invalid_argument("qubit_vector err");
    }
        m_qgate = qgate;
    for (auto aiter = qubit_vector.begin(); aiter != qubit_vector.end(); ++aiter)
    {
        m_qubit_vector.push_back(*aiter);
    }
    m_node_type = GATE_NODE;
}

NodeType OriginQGate::getNodeType() const
{
    return m_node_type;
}

size_t OriginQGate::getQuBitVector(QVec& vector) const
{
    for (auto aiter : m_qubit_vector)
    {
        vector.push_back(aiter);
    }
    return m_qubit_vector.size();
}

size_t OriginQGate::getTargetQubitNum() const
{
    return m_qubit_vector.size();
}

size_t OriginQGate::getControlQubitNum() const
{
        return m_control_qubit_vector.size();
}

Qubit * OriginQGate::popBackQuBit()
{
    auto temp = m_qubit_vector.back();
    m_qubit_vector.pop_back();
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

bool OriginQGate::setControl(QVec qubit_vector)
{
    for (auto aiter : qubit_vector)
    {
        m_control_qubit_vector.push_back(aiter);
    }
    return true;
}

bool OriginQGate::isDagger() const
{
    return m_Is_dagger;
}

size_t OriginQGate::getControlVector(QVec& qubit_vector) const
{
    for (auto aiter : m_control_qubit_vector)
    {
        qubit_vector.push_back(aiter);
    }
    return m_control_qubit_vector.size();
}

void OriginQGate::PushBackQuBit(Qubit * qubit)
{

    if (nullptr == qubit)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    m_qubit_vector.push_back(qubit);

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
    if (m_qubit_vector.size() <= 0)
    {
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    QVec control_qubit_vector;
    for (auto aiter : param->m_control_qubit_vector)
    {
        control_qubit_vector.push_back(aiter);
    }

    for (auto aiter : m_control_qubit_vector)
    {
        control_qubit_vector.push_back(aiter);
    }
    if (control_qubit_vector.size() > 0)
    {
        sort(control_qubit_vector.begin(), 
            control_qubit_vector.end(),
            compareQubit);

        control_qubit_vector.erase(unique(control_qubit_vector.begin(),
                                         control_qubit_vector.end(), Qubitequal),
                                  control_qubit_vector.end());
    }

    for (auto aQIter : m_qubit_vector)
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
    auto aiter = QGateParseMap::getFunction(m_qgate->getOperationNum());
    if (nullptr == aiter)
    {
        stringstream error;
        error << "gate operation num error ";
        QCERR(error.str());
        throw run_fail(error.str());
    }
    aiter(m_qgate, m_qubit_vector, quantum_gates, dagger, control_qubit_vector, (GateType)m_qgate->getGateType());
}


QGate QGateNodeFactory::getGateNode(const string & name, Qubit * qubit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);

    try
    {
        QGate  QGateNode(qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }

}

QGate QGateNodeFactory::getGateNode(const string & name, Qubit * qubit, double angle)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, angle);

    try
    {
        QGate  QGateNode(qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }


}

QGate QGateNodeFactory::getGateNode(const string & name, 
    Qubit * control_qubit, 
    Qubit * target_qubit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name);

    try
    {
        QGate  QGateNode(control_qubit, target_qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }
}

QGate QGateNodeFactory::getGateNode(const string & name,
    Qubit * control_qubit,
    Qubit * target_qubit, 
    double theta)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, theta);
    try
    {
        QGate  QGateNode(control_qubit, target_qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }
}

QGate QGateNodeFactory::getGateNode(double alpha,
    double beta, 
    double gamma,
    double delta,
    Qubit * qubit)
{
    string name = "U4";
    QuantumGate * pGate = m_pGateFact->getGateNode(name, alpha, beta, gamma, delta);

    try
    {
        QGate  QGateNode(qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }
}

QGate QGateNodeFactory::getGateNode(double alpha,
    double beta,
    double gamma, 
    double delta, 
    Qubit * control_qubit, Qubit * target_qubit)
{
    string name = "CU";
    QuantumGate * pGate = m_pGateFact->getGateNode(name, alpha, beta, gamma, delta);

    try
    {
        QGate  QGateNode(control_qubit, target_qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }

}

QGate QGateNodeFactory::getGateNode(const string &name,
    QStat matrix,
    Qubit * control_qubit,
    Qubit * target_qubit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, matrix);

    try
    {
        QGate  QGateNode(control_qubit, target_qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }
}

QGate QGateNodeFactory::getGateNode(const string &name,
    QStat matrix,
    Qubit * target_qubit)
{
    QuantumGate * pGate = m_pGateFact->getGateNode(name, matrix);

    try
    {
        QGate  QGateNode(target_qubit, pGate);
        return QGateNode;
    }
    catch (const std::exception& e)
    {
        QCERR(e.what());
        throw gate_alloc_fail(e.what());
    }
}

void QGateParseSingleBit(QuantumGate * qgate, 
    QVec & qubit_vector, 
    QPUImpl* qgates,
    bool is_dagger,
    QVec & control_qubit_vector,
    GateType type)
{
    if (nullptr == qgate)
    {
        QCERR("param error");
        throw invalid_argument("param error");
    }

    QStat matrix;
    qgate->getMatrix(matrix);
    Qubit * qubit = *(qubit_vector.begin());
    size_t bit = qubit->getPhysicalQubitPtr()->getQubitAddr();
    if (control_qubit_vector.size() == 0)
    {
        qgates->unitarySingleQubitGate(bit, matrix, is_dagger, 0, type);
    }
    else
    {
        size_t temp;
        vector<size_t> bit_num_vector;
        for (auto aiter : control_qubit_vector)
        {
            temp = aiter->getPhysicalQubitPtr()->getQubitAddr();
            bit_num_vector.push_back(temp);
        }
        bit_num_vector.push_back(bit);
        qgates->controlunitarySingleQubitGate(bit, bit_num_vector, matrix, is_dagger, 0, type);
    }

}

void QGateParseDoubleBit(QuantumGate * qgate, 
    QVec & qubit_vector,
    QPUImpl* qgates,
    bool is_dagger,
    QVec & control_qubit_vector,
    GateType type)
{
    QStat matrix;
    qgate->getMatrix(matrix);
    auto aiter = qubit_vector.begin();
    Qubit * qubit = *aiter;
    aiter++;
    Qubit * qubit2 = *aiter;
    size_t bit = qubit->getPhysicalQubitPtr()->getQubitAddr();
    size_t bit2 = qubit2->getPhysicalQubitPtr()->getQubitAddr();

    if (control_qubit_vector.size() == 0)
    {
        qgates->unitaryDoubleQubitGate(bit, bit2, matrix, is_dagger, 0, type);
    }
    else
    {
        size_t temp;
        vector<size_t> bit_num_vector;
        for (auto aiter : control_qubit_vector)
        {
            temp = aiter->getPhysicalQubitPtr()->getQubitAddr();
            bit_num_vector.push_back(temp);
        }
        bit_num_vector.push_back(bit2);
        bit_num_vector.push_back(bit);
        qgates->controlunitaryDoubleQubitGate(bit, bit2, bit_num_vector, matrix, is_dagger, 0, type);
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


QGate QPanda::X(Qubit * qubit)
{
    string name = "X";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate QPanda::X1(Qubit * qubit)
{
    string name = "X1";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate QPanda::RX(Qubit * qubit, double angle)
{
    string name = "RX";
    return _gs_pGateNodeFactory->getGateNode(name, qubit, angle);
}

QGate QPanda::U1(Qubit * qubit, double angle)
{
    string name = "U1";
    return _gs_pGateNodeFactory->getGateNode(name, qubit, angle);
}

QGate QPanda::Y(Qubit * qubit)
{
    string name = "Y";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate QPanda::Y1(Qubit * qubit)
{
    string name = "Y1";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate QPanda::RY(Qubit * qubit, double angle)
{
    string name = "RY";
    return _gs_pGateNodeFactory->getGateNode(name, qubit, angle);
}
QGate QPanda::Z(Qubit * qubit)
{
    string name = "Z";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}
QGate QPanda::Z1(Qubit * qubit)
{
    string name = "Z1";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate QPanda::RZ(Qubit * qubit, double angle)
{
    string name = "RZ";
    return _gs_pGateNodeFactory->getGateNode(name, qubit, angle);
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "ISWAP";
    return _gs_pGateNodeFactory->getGateNode(name, targitBit_fisrt, targitBit_second);
}

QGate QPanda::iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second, double theta)
{
    string name = "ISWAPTheta";
    return _gs_pGateNodeFactory->getGateNode(name,
        targitBit_fisrt,
        targitBit_second,
        theta);
}

QGate QPanda::CR(Qubit * control_qubit, Qubit * targit_qubit, double theta)
{
    string name = "CPhaseGate";
    return _gs_pGateNodeFactory->getGateNode(name, control_qubit, targit_qubit, theta);
}

QGate QPanda::SqiSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "SQISWAP";
    return _gs_pGateNodeFactory->getGateNode(name,
        targitBit_fisrt,
        targitBit_second);
}

QGate QPanda::SWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    string name = "SWAP";
    return _gs_pGateNodeFactory->getGateNode(name,
        targitBit_fisrt,
        targitBit_second);
}

QGate QPanda::S(Qubit * qubit)
{
    string name = "S";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate QPanda::T(Qubit * qubit)
{
    string name = "T";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate  QPanda::H(Qubit * qubit)
{
    string name = "H";
    return _gs_pGateNodeFactory->getGateNode(name, qubit);
}

QGate  QPanda::CNOT(Qubit * control_qubit, Qubit * target_qubit)
{
    string name = "CNOT";
    return _gs_pGateNodeFactory->getGateNode(name, control_qubit, target_qubit);
}

QGate QPanda::CZ(Qubit * control_qubit, Qubit *target_qubit)
{
    string name = "CZ";
    return _gs_pGateNodeFactory->getGateNode(name, control_qubit, target_qubit);
}

QGate QPanda::U4(double alpha, double beta, double gamma, double delta, Qubit * qubit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, qubit);
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
    Qubit * control_qubit, 
    Qubit * target_qubit)
{
    return _gs_pGateNodeFactory->getGateNode(alpha, beta, gamma, delta, control_qubit, target_qubit);
}

QGate QPanda::CU(QStat & matrix, Qubit * control_qubit, Qubit * target_qubit)
{
    string name = "CU";
    return _gs_pGateNodeFactory->getGateNode(name, matrix, control_qubit, target_qubit);
}

QGate QPanda::QDouble(QStat matrix, Qubit * qubit1, Qubit * qubit2)
{
    string name = "QDoubleGate";
    return _gs_pGateNodeFactory->getGateNode(name, matrix, qubit1, qubit2);
}
