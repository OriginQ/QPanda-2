#ifndef _QGATE_H
#define _QGATE_H

#include <complex>
#include <vector>
#include <iterator>
#include <map>

#include "Core/QuantumCircuit/QNode.h"
#include "Core/QuantumCircuit/QuantumGate.h"
#include "Core/QuantumMachine/QubitFactory.h"
#include "Core/QuantumMachine/QVec.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
QPANDA_BEGIN
using QGATE_SPACE::QuantumGate;
using QGATE_SPACE::QGateFactory;

class AbstractQGateNode
{
public:
    virtual size_t getQuBitVector(QVec &) const = 0;
    virtual Qubit * popBackQuBit() = 0;
    virtual void PushBackQuBit(Qubit *) = 0;
    virtual size_t getQuBitNum() const = 0;
    virtual QuantumGate * getQGate() const = 0;
    virtual void setQGate(QuantumGate *) = 0;
    virtual bool isDagger() const = 0;
    virtual size_t getControlVector(QVec &) const = 0;
    virtual bool setDagger(bool) = 0;
    virtual bool setControl(QVec) = 0;
    virtual ~AbstractQGateNode() {}
};

/*
*  Quantum single gate node: RX_GATE,RY_GATE,RZ_GATE,H,S_GATE,      CAN ADD OTHER GATES
*  gate:  gate type
*  opQuBit: qubit number
*
*/
class QGateNodeFactory;

class QGate : public QNode, public AbstractQGateNode
{
private:
    std::shared_ptr<AbstractQGateNode>  m_qgate_node;

public:
    ~QGate();
    QGate(const QGate&);
    QGate(Qubit*, QuantumGate*);
    QGate(Qubit*, Qubit*, QuantumGate*);
    NodeType getNodeType() const;
    size_t getQuBitVector(QVec &) const;
    size_t getQuBitNum() const;
    QuantumGate *getQGate() const;
    bool setDagger(bool);
    bool setControl(QVec);
    std::shared_ptr<QNode> getImplementationPtr();
    QGate dagger();
    QGate control(QVec);
    bool isDagger() const;
    size_t getControlVector(QVec &) const;
private:
    Qubit * popBackQuBit() { return nullptr; };
    void setQGate(QuantumGate *) {};
    void PushBackQuBit(Qubit *) {};
    void execute(QPUImpl *, QuantumGateParam *) {};
};

class OriginQGate : public QNode, public AbstractQGateNode
{
private:
    QVec m_qubit_vector;
    QuantumGate *m_qgate;
    NodeType m_node_type;
    bool m_Is_dagger;
    std::vector<Qubit*> m_control_qubit_vector;
    std::shared_ptr<QNode> getImplementationPtr()
    {
        QCERR("Can't use this function");
        throw std::runtime_error("Can't use this function");
    };
public:
    ~OriginQGate();
    OriginQGate(Qubit*, QuantumGate *);
    OriginQGate(Qubit*, Qubit *, QuantumGate *);
    OriginQGate(QVec &, QuantumGate *);
    NodeType getNodeType() const;
    size_t getQuBitVector(QVec &) const;
    size_t getQuBitNum() const;
    Qubit *popBackQuBit();
    QuantumGate *getQGate() const;
    void setQGate(QuantumGate *);
    bool setDagger(bool);
    bool setControl(QVec);
    bool isDagger() const;
    size_t getControlVector(QVec &) const;
    void PushBackQuBit(Qubit *);

    void execute(QPUImpl *, QuantumGateParam *);
};

class QGateNodeFactory
{
public:
    static QGateNodeFactory * getInstance()
    {
        static QGateNodeFactory s_gateNodeFactory;
        return &s_gateNodeFactory;
    }

    QGate getGateNode(const std::string & name, Qubit *);
    QGate getGateNode(const std::string & name, Qubit *, double);
    QGate getGateNode(const std::string & name, Qubit *, Qubit*);
    QGate getGateNode(const std::string & name, Qubit * control_qubit, Qubit * target_qubit, double theta);
    QGate getGateNode(double alpha, double beta, double gamma, double delta, Qubit *);
    QGate getGateNode(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);
    QGate getGateNode(const std::string &name, QStat matrix, Qubit *, Qubit *);
    QGate getGateNode(const std::string &name, QStat matrix, Qubit *);
private:
    QGateNodeFactory()
    {
        m_pGateFact = QGateFactory::getInstance();
    }
    QGateFactory * m_pGateFact;
};

typedef void(*QGATE_FUN)(QuantumGate *,
    QVec &,
    QPUImpl*,
    bool,
    QVec &,
    GateType);
typedef std::map<int, QGATE_FUN> QGATE_FUN_MAP;

class QGateParseMap
{

    static QGATE_FUN_MAP m_qgate_function_map;
public:

    static void insertMap(int opNum, QGATE_FUN function)
    {
        m_qgate_function_map.insert(std::pair<int, QGATE_FUN>(opNum, function));
    }

    static QGATE_FUN getFunction(int iOpNum)
    {
        auto aiter = m_qgate_function_map.find(iOpNum);
        if (aiter == m_qgate_function_map.end())
        {
            return nullptr;
        }

        return aiter->second;
    }


};


QGate X(Qubit* qubit);
QGate X1(Qubit* qubit);
QGate RX(Qubit*, double angle);
QGate U1(Qubit*, double angle);
QGate Y(Qubit* qubit);
QGate Y1(Qubit* qubit);
QGate RY(Qubit*, double angle);
QGate Z(Qubit* qubit);
QGate Z1(Qubit* qubit);
QGate RZ(Qubit*, double angle);
QGate S(Qubit* qubit);
QGate T(Qubit*);
QGate H(Qubit* qubit);
QGate CNOT(Qubit* control_qubit, Qubit* target_qubit);
QGate CZ(Qubit*  control_qubit, Qubit* target_qubit);
QGate U4(double alpha, double beta, double gamma, double delta, Qubit*);
QGate U4(QStat& matrix, Qubit*);
QGate QDouble(QStat matrix, Qubit * qubit1, Qubit * qubit2);
QGate CU(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);
QGate CU(QStat& matrix, Qubit*, Qubit*);
QGate iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);
QGate iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second, double theta);
QGate CR(Qubit * control_qubit, Qubit * targit_qubit, double theta);
QGate SqiSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);
QGate SWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);
QPANDA_END
#endif // !_QGATE_H
