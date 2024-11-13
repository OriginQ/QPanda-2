/*! \file QGate.h */
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
#include "Core/Utilities/Tools/TranformQGateTypeStringAndEnum.h"
#include <unordered_set>


QPANDA_BEGIN

using QGATE_SPACE::QuantumGate;
using QGATE_SPACE::QGateFactory;
class QCircuit;

/**
* @brief   Quantum gate basic abstract class
* @ingroup QuantumCircuit
*/
class AbstractQGateNode
{
public:
    /**
    * @brief  Get qubit vector inside this quantum gate
    * @param[in]  QVec&  qubit vector
    * @return     size_t
    * @see   GateType
    */
    virtual size_t getQuBitVector(QVec &) const = 0;

    /**
    * @brief  erase qubit vector element at end
    * @return     Qubit*
    */
    virtual Qubit * popBackQuBit() = 0;

    /**
    * @brief  insert qubit vector element at end
    * @param[in]  Qubit*  Qubit pointer
    */
    virtual void PushBackQuBit(Qubit *) = 0;

    /**
    * @brief  Get target qubit num inside this quantum gate
    * @return     size_t  qubit num
    */
    virtual size_t getTargetQubitNum() const = 0;

    /**
    * @brief  Get control qubit num inside this quantum gate
    * @return     size_t  qubit num
    */
    virtual size_t getControlQubitNum() const = 0;

    /**
    * @brief  Get Quantum Gate
    * @return     QuantumGate *
    */
    virtual QuantumGate * getQGate() const = 0;

    /**
    * @brief  Set Quantum Gate
    * @param[in]  QuantumGate*  QuantumGate pointer
    */
    virtual void setQGate(QuantumGate *) = 0;

    /**
    * @brief  Judge current quantum gate is dagger
    * @return  bool
    */
    virtual bool isDagger() const = 0;

    /**
    * @brief  Get control vector fron current quantum gate node
    * @param[in]  QVec& qubits  vector
    * @return     size_t
    * @see QVec
    */
    virtual size_t getControlVector(QVec &) const = 0;

    /**
    * @brief  Clear the control qubits for current quantum gate
    * @see QVec
    */
    virtual void clear_control() = 0;

    /**
    * @brief  Set dagger to current quantum gate
    * @param[in]  bool is dagger
    * @return  bool
    */
    virtual bool setDagger(bool) = 0;

    /**
    * @brief  Set control qubits to current quantum gate
    * @param[in]  QVec  control qubits  vector
    * @return  bool
    * @see QVec
    */
    virtual bool setControl(QVec) = 0;

    /**
    * @brief  remap qubit
    * @return
    */
    virtual void remap(QVec) = 0;
    virtual void clear_qubits() = 0;

    virtual ~AbstractQGateNode() {}
};


class QGateNodeFactory;



/**
* @brief    QPanda2 quantum gate  basic classs
* @ingroup  QuantumCircuit
*/
class QGate : public AbstractQGateNode
{
private:
    std::shared_ptr<AbstractQGateNode>  m_qgate_node;

public:
    ~QGate();
    QGate(const QGate&);
    QGate(QVec &, QuantumGate*);
    QGate(std::shared_ptr<AbstractQGateNode> node);

    /**
    * @brief  Get current node type
    * @return     NodeType  current node type
    * @see  NodeType
    */
    NodeType getNodeType() const;


    /**
    * @brief  Get qubit vector inside this quantum gate
    * @param[in]  QVec&  qubit vector
    * @return     size_t
    * @see   GateType
    */
    size_t getQuBitVector(QVec &) const;

    /**
    * @brief  Get qubit num inside this quantum gate
    * @return size_t  qubit num
    */
    size_t getTargetQubitNum() const;

    size_t getControlQubitNum() const;

    QuantumGate *getQGate() const;

    /**
    * @brief  Set dagger to current quantum gate
    * @param[in]  bool is dagger
    * @return  bool
    */
    bool setDagger(bool);

    /**
    * @brief  Set control qubits to current quantum gate
    * @param[in]  QVec  control qubits  vector
    * @return  bool
    * @see QVec
    */
    bool setControl(QVec);
    std::shared_ptr<AbstractQGateNode> getImplementationPtr();


    /**
    * @brief  Get a dagger quantumgate  base on current quantum gate node
    * @return     QPanda::QGate  quantum gate
    */
    QGate dagger();
    /**
    * @brief  Get a control quantumgate  base on current quantum gate node
    * @param[in]  QVec control qubits  vector
    * @return     QPanda::QGate  quantum gate
    * @see QVec
    */
    QGate control(QVec);

    /**
    * @brief  Clear the control qubits for current quantum gate
    * @return
    */
    void clear_control();

    /**
    * @brief  remap qubit
    * @return
    */
    void remap(QVec) override;

    void clear_qubits() override;

    /**
    * @brief  Judge current quantum gate is dagger
    * @return  bool
    */
    bool isDagger() const;

    /**
    * @brief  Get control vector fron current quantum gate node
    * @param[in]  QVec& qubits  vector
    * @return     size_t
    * @see QVec
    */
    size_t getControlVector(QVec &) const;

    QGate exp(double exponent);

    QGate operator*(const QGate &other);

private:
    Qubit * popBackQuBit() { return nullptr; };
    void setQGate(QuantumGate *) {};
    void PushBackQuBit(Qubit *) {};
};

/**
* @brief Implementation  class of QGate
* @ingroup QuantumCircuit
*/
class OriginQGate : public QNode, public AbstractQGateNode
{
private:
    QVec m_qubit_vector;
    QuantumGate *m_qgate;
    NodeType m_node_type;
    bool m_Is_dagger;
    std::vector<Qubit*> m_control_qubit_vector;
    std::unordered_set<int> m_check_qubits;

    bool _check_duplicate(const QVec &add_qubits);
public:
    ~OriginQGate();
    OriginQGate(QVec &, QuantumGate *);
    NodeType getNodeType() const;
    size_t getQuBitVector(QVec &) const;
    size_t getTargetQubitNum() const;
    size_t getControlQubitNum() const;
    Qubit *popBackQuBit();
    QuantumGate *getQGate() const;
    void setQGate(QuantumGate *);
    bool setDagger(bool);
    bool setControl(QVec);
    bool isDagger() const;
    size_t getControlVector(QVec &) const;
    void PushBackQuBit(Qubit *);
    void remap(QVec) override;
    void clear_control();
    void clear_qubits();
};

/**
 * @brief Factory for class QGate
 * @ingroup QuantumCircuit
 */
class QGateNodeFactory
{
public:
    /**
     * @brief Get the static instance of factory
     * @return QGateNodeFactory *
     */
    static QGateNodeFactory * getInstance()
    {
        static QGateNodeFactory s_gateNodeFactory;
        return &s_gateNodeFactory;
    }
    template<typename ...Targs>
    QGate getGateNode(const std::string & name, QVec qs, Targs&&... args)
    {
        QuantumGate * pGate = QGATE_SPACE::create_quantum_gate(name, std::forward<Targs>(args)...);
        try
        {
            QGate  QGateNode(qs, pGate);
            return QGateNode;
        }
        catch (const std::exception& e)
        {
            QCERR(e.what());
            throw std::runtime_error(e.what());
        }
    }

private:
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

/**
* @brief  Construct a new I gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate I(Qubit* qubit);

/**
* @brief  Construct qubits.size() new I gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit I(const QVec& qubits);

/**
* @brief  Construct a new I gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate I(int  qaddr);

/**
* @brief  Construct qaddrs.size() new I gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit I(const std::vector<int>& qaddrs);

/**
* @brief  Construct a new quantum X gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate X(Qubit* qubit);

/**
* @brief  Construct qubits.size() new X gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit X(const QVec& qubits);

/**
* @brief  Construct a new quantum X gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate X(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum X gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit X(const std::vector<int>& qaddrs);

/**
* @brief  Construct a new quantum X1 gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate X1(Qubit* qubit);

/**
* @brief  Construct qubits.size() new X1 gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit X1(const QVec& qubits);

/**
* @brief  Construct a new quantum X1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate X1(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum X1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit X1(const std::vector<int> &qaddrs);

/**
* @brief  Construct a new quantum RX gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RX(Qubit* qaddr, double angle);

/**
* @brief  Construct qubits.size() new quantum RX gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RX(const QVec& qubits, double angle);

/**
* @brief  Construct a new quantum RX gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RX(int  qaddr, double angle);

/**
* @brief  Construct qaddrs.size() new quantum RX gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RX(const std::vector<int>& qaddrs, double angle);

/**
* @brief  Construct a new quantum U1 gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U1(Qubit*, double angle);

/**
* @brief  Construct qubits.size() new U1 gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit U1(const QVec& qubits, double angle);

/**
* @brief  Construct a new quantum U1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U1(int  qaddr, double angle);

/**
* @brief  Construct qaddrs.size() new quantum U1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit U1(const std::vector<int>& qaddrs, double angle);

/**
* @brief  Construct a new quantum P gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate P(Qubit*, double angle);

/**
* @brief  Construct qubits.size() new P gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit P(const QVec& qubites, double angle);

/**
* @brief  Construct a new quantum P gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate P(int qaddr, double angle);

/**
* @brief  Construct qaddrs.size() new quantum P gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit P(const std::vector<int>& qaddrs, double angle);


/**
* @brief  Construct a new quantum U2 gate
* @param[in]  Qubit* target qubit
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U2(Qubit * qubit, double phi, double lambda);

/**
* @brief  Construct qubits.size() new U2 gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit U2(const QVec& qubits, double phi, double lambda);


/**
* @brief  Construct a new quantum U2 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U2(int  qaddr, double phi, double lambda);

/**
* @brief  Construct qaddrs.size() new quantum U2 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit U2(const std::vector<int>& qaddrs, double phi, double lambda);

/**
* @brief  Construct a new quantum U3 gate
* @param[in]  Qubit* target qubit
* @param[in]  double theta
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U3(Qubit * qubit, double theta, double phi, double lambda);

/**
* @brief  Construct qubits.size() new U3 gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit U3(const QVec& qubits, double theta, double phi, double lambda);

/**
* @brief  Construct a new quantum U3 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U3(int  qaddr, double theta, double phi, double lambda);

/**
* @brief  Construct qaddrs.size() new quantum U3 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @param[in]  double phi
* @param[in]  double lambda
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit U3(const std::vector<int>& qaddrs, double theta, double phi, double lambda);

/**
* @brief  Construct a new quantum U3 gate
* @param[in]  Qubit* target qubit
* @param[in]  QStat& matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U3(Qubit* qubit, QStat& matrix);

/**
* @brief  Construct qubits.size() new U3 gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit U3(const QVec& qubit, QStat & matrix);

/**
* @brief  Construct a new quantum Y gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Y(Qubit* qubit);

/**
* @brief  Construct qubits.size() new Y gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit Y(const QVec& quibits);

/**
* @brief  Construct a new quantum Y gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Y(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum Y gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit Y(const std::vector<int>& qaddrs);

/**
* @brief  Construct a new quantum Y1 gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Y1(Qubit* qubit);

/**
* @brief  Construct qubits.size() new Y1 gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit Y1(const QVec& quibits);

/**
* @brief  Construct a new quantum Y1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Y1(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum Y1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit Y1(const std::vector<int>& qaddrs);

/**
* @brief  Construct a new quantum RY gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RY(Qubit*, double angle);

/**
* @brief  Construct qubits.size() new RY gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit RY(const QVec& quibits, double angle);

/**
* @brief  Construct a new quantum RY gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RY(int  qaddr, double angle);

/**
* @brief  Construct qaddrs.size() new quantum RY gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RY(const std::vector<int>& qaddrs, double angle);

/**
* @brief  Construct a new quantum Z gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Z(Qubit* qubit);

/**
* @brief  Construct qubits.size() new Z gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit Z(const QVec& quibits);

/**
* @brief  Construct a new quantum Z gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Z(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum Z gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit Z(const std::vector<int>& qaddrs);

/**
* @brief  Construct a new quantum Z1 gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Z1(Qubit* qubit);

/**
* @brief  Construct qubits.size() new Z1 gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit Z1(const QVec& qubits);

/**
* @brief  Construct a new quantum Z1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate Z1(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum Z1 gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit Z1(const std::vector<int>& qaddrs);


/**
* @brief  Construct a new quantum RZ gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RZ(Qubit*, double angle);

/**
* @brief  Construct qubits.size() new RZ gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit RZ(const QVec& qubits, double angle);

/**
* @brief  Construct a new quantum RZ gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RZ(int  qaddr, double angle);

/**
* @brief  Construct qaddrs.size() new quantum RZ gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double angle
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RZ(const std::vector<int>& qaddrs, double angle);

/**
* @brief  Construct a new quantum RZPhi gate
* @param[in]  Qubit* target qubit
* @param[in]  double angle
* @param[in]  double phi
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RPhi(Qubit * qubit, double angle, double phi);

/**
* @brief  Construct qubits.size() new RZPhi gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit RPhi(const QVec& quibits, double angle, double phi);

/**
* @brief  Construct a new quantum RZPhi gate
* @param[in] int  qaddr  target qubit phy addr
* @param[in]  double angle
* @param[in]  double phi
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RPhi(int  qaddr, double angle, double phi);

/**
* @brief  Construct qaddrs.size() new quantum RZPhi gate
* @param[in] int  qaddr  target qubit phy addr
* @param[in]  double angle
* @param[in]  double phi
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RPhi(const std::vector<int>& qaddrs, double angle, double phi);

/**
* @brief  Construct a new quantum S gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate S(Qubit* qubit);

/**
* @brief  Construct qubits.size() new S gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit S(const QVec& quibits);

/**
* @brief  Construct a new quantum S gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate S(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum S gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit S(const std::vector<int>& qaddrs);

/**
* @brief  Construct a new quantum T gate
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate T(Qubit*);

/**
* @brief  Construct qubits.size() new T gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit T(const QVec& qubits);

/**
* @brief  Construct a new quantum T gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate T(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum T gate
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit T(const std::vector<int>& qaddrs);

/**
* @brief  Construct a new quantum H gate
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate H(Qubit* qubit);

/**
* @brief  Construct qubits.size() new H gate
* @param[in] const QVec& qubits target qubits vector
* @return     QPanda::QCircuit  quantum circuit
* @ingroup QuantumCircuit
*/
QCircuit H(const QVec& qubits);

/**
* @brief  Construct a new quantum H gate
* @param[in] int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate H(int  qaddr);
/**
* @brief  Construct qaddrs.size() new quantum H gate
* @param[in] int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit H(const std::vector<int>& qaddrs);


/**
* @brief  Construct a new quantum ECHO gate; Only for 6 qubits online projects !
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate ECHO(Qubit* qubit);

/**
* @brief  Construct qubits.size() new quantum ECHO gate; Only for 6 qubits online projects !
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit ECHO(const QVec& qubits);

/**
* @brief  Construct a new quantum ECHO gate; Only for 6 qubits online projects !
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate ECHO(int  qaddr);

/**
* @brief  Construct qaddrs.size() new quantum ECHO gate; Only for 6 qubits online projects !
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit ECHO(const std::vector<int>& qaddrs);


/**
* @brief  Construct a new quantum CNOT gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CNOT(Qubit* control_qubit, Qubit* target_qubit);

/**
* @brief  Construct control_qubits.size() new quantum CNOT gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CNOT(const QVec &control_qubits, const QVec &target_qubits);



/**
* @brief  Construct a new quantum CNOT gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CNOT(int control_qaddr, int target_qaddr);

/**
* @brief  Construct control_qaddrs.size() new quantum CNOT gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CNOT(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs);

/**
* @brief  Construct a new quantum CZ gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CZ(Qubit*  control_qubit, Qubit* target_qubit);

/**
* @brief  Construct control_qubits.size() new quantum CZ gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CZ(const QVec& control_qubits, const QVec &target_qubits);

/**
* @brief  Construct a new quantum CZ gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CZ(int control_qaddr, int target_qaddr);

/**
* @brief  Construct control_qaddrs.size() new quantum CZ gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CZ(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs);

/**
* @brief  Construct a new quantum U4 gate
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U4(double alpha, double beta, double gamma, double delta, Qubit*);

/**
* @brief  Construct a new quantum U4 gate
* @param[in]  QStat& matrix
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U4(QStat& matrix, Qubit*);

/**
* @brief  Construct a new quantum QDouble gate
* @param[in]  QStat matrix
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate QDouble(QStat& matrix, Qubit * qubit1, Qubit * qubit2);

/**
* @brief  Construct a new quantum CU gate
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @param[in]  Qubit*   control qubit
* @param[in]  Qubit*   target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CU(double alpha, double beta, double gamma, double delta, Qubit *, Qubit *);

/**
* @brief  Construct control_qubits.size() new quantum CU gate
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @param[in]  Qubit*   control qubit
* @param[in]  Qubit*   target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CU(double alpha, double beta, double gamma, double delta, const QVec& control_qubits, const QVec& target_qubits);

/**
* @brief  Construct a new quantum CU gate
* @param[in]  QStat & matrix
* @param[in]  Qubit*  target qubit
* @param[in]  Qubit*  control qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CU(QStat& matrix, Qubit*, Qubit*);

/**
* @brief  Construct control_qubits.size() new quantum CU gate
* @param[in]  QStat & matrix
* @param[in]  Qubit*  target qubit
* @param[in]  Qubit*  control qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CU(QStat& matrix, const QVec& control_qubits, const QVec& target_qubits);

/**
* @brief  Construct a new quantum CU gate
* @param[in]  Qubit*  target qubit
* @param[in]  Qubit*  control qubit
* @param[in]  QStat & matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CU(Qubit*, Qubit*, QStat& matrix);

/**
* @brief  Construct control_qubits.size() new quantum CU gate
* @param[in]  Qubit*  target qubit
* @param[in]  Qubit*  control qubit
* @param[in]  QStat & matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CU(const QVec& control_qubits, const QVec& target_qubits, QStat& matrix);



/**
* @brief  Construct a new quantum CU gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  QStat & matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CU(int control_qaddr, int target_qaddr, QStat& matrix);

/**
* @brief  Construct control_qaddrs.size() new quantum CU gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  QStat & matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CU(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, QStat& matrix);

/**
* @brief  Construct a new quantum CU gate
* @param[in]  Qubit*   control qubit
* @param[in]  Qubit*   target qubit
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CU(Qubit *, Qubit *, double alpha, double beta, double gamma, double delta);

/**
* @brief  Construct control_qaddrs.size() new quantum CU gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CU(const QVec& control_qaddrs, const QVec& target_qaddrs, double alpha, double beta, double gamma, double delta);

/**
* @brief  Construct a new quantum CU gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CU(int control_qaddr, int target_qaddr, double alpha, double beta, double gamma, double delta);

/**
* @brief  Construct control_qaddrs.size() new quantum CU gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CU(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double alpha, double beta, double gamma, double delta);


/**
* @brief  Construct a new quantum iSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup CQuantumCircuitore
*/
QGate iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);

/**
* @brief  Construct targitBit_first.size() new quantum iSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup CQuantumCircuitore
*/
QCircuit iSWAP(const QVec& targitBit_first, const QVec& targitBit_second);

/**
* @brief  Construct a new quantum iSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second, double theta);

/**
* @brief  Construct targitBit_first.size() new quantum iSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit iSWAP(const QVec& targitBit_first, const QVec& targitBit_second, double theta);

/**
* @brief  Construct a new quantum iSWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup CQuantumCircuitore
*/
QGate iSWAP(int control_qaddr, int target_qaddr);

/**
* @brief  Construct control_qaddrs.size() new quantum iSWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup CQuantumCircuitore
*/
QCircuit iSWAP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs);

/**
* @brief  Construct a new quantum iSWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate iSWAP(int control_qaddr, int target_qaddr, double theta);

/**
* @brief  Construct control_qaddrs.size() new quantum iSWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit iSWAP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double theta);

/**
* @brief  Construct a new quantum CR gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* targit qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CR(Qubit * control_qubit, Qubit * targit_qubit, double theta);

/**
* @brief  Construct control_qubits.size() new quantum CR gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* targit qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CR(const QVec& control_qubits, const QVec& targit_qubits, double theta);

/**
* @brief  Construct a new quantum CR gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CR(int control_qaddr, int target_qaddr, double theta);

/**
* @brief  Construct control_qaddrs.size() new quantum CR gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CR(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double theta);

/**
* @brief  Construct a new quantum CP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* targit qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CP(Qubit * control_qubit, Qubit * targit_qubit, double theta);

/**
* @brief  Construct control_qubits.size() new quantum CP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* targit qubit
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CP(const QVec& control_qubits, const QVec& targit_qubits, double theta);

/**
* @brief  Construct a new quantum CP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate CP(int control_qaddr, int target_qaddr, double theta);

/**
* @brief  Construct control_qaddrs.size() new quantum CP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit CP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, double theta);

/**
* @brief  Construct a new quantum RXX gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RXX(Qubit*, Qubit*, double angle);

/**
* @brief  Construct a new quantum RXX gate
* @param[in]  QVec& control qubit
* @param[in]  QVec& target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RXX(const QVec&, const QVec&, double angle);

/**
* @brief  Construct a new quantum RXX gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RXX(int, int, double angle);

/**
* @brief  Construct control_qaddrs.size() new quantum RXX gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RXX(const std::vector<int>&, const std::vector<int>&, double angle);

/**
* @brief  Construct a new quantum RYY gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RYY(Qubit*, Qubit*, double angle);

/**
* @brief  Construct a new quantum RYY gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RYY(int, int, double angle);

/**
* @brief  Construct a new quantum RYY gate
* @param[in]  QVec& control qubit
* @param[in]  QVec& target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RYY(const QVec&, const QVec&, double angle);

/**
* @brief  Construct control_qaddrs.size() new quantum RYY gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RYY(const std::vector<int>&, const std::vector<int>&, double angle);

/**
* @brief  Construct a new quantum RZZ gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RZZ(Qubit*, Qubit*, double angle);

/**
* @brief  Construct a new quantum RZZ gate
* @param[in]  QVec& control qubit
* @param[in]  QVec& target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RZZ(const QVec&, const QVec&, double angle);


/**
* @brief  Construct a new quantum RZZ gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RZZ(int, int, double angle);

/**
* @brief  Construct control_qaddrs.size() new quantum RZZ gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RZZ(const std::vector<int>&, const std::vector<int>&, double angle);

/**
* @brief  Construct a new quantum RZX gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RZX(Qubit*, Qubit*, double angle);

/**
* @brief  Construct a new quantum RZX gate
* @param[in]  QVec& control qubit
* @param[in]  QVec& target qubit
* @param[in]  double angle target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RZX(const QVec&, const QVec&, double angle);

/**
* @brief  Construct a new quantum SqiSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/

/**
* @brief  Construct a new quantum RZX gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate RZX(int, int, double angle);

/**
* @brief  Construct control_qaddrs.size() new quantum RZZ gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double theta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit RZX(const std::vector<int>&, const std::vector<int>&, double angle);

QGate SqiSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);

/**
* @brief  Construct targitBits_fisrt.size() new quantum SqiSWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit SqiSWAP(const QVec& targitBits_fisrt, const QVec& targitBits_second);

/**
* @brief  Construct a new quantum SqiSWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate SqiSWAP(int control_qaddr, int target_qaddr);

/**
* @brief  Construct control_qaddrs.size() new quantum SqiSWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit SqiSWAP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs);

/**
* @brief  Construct a new quantum SWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate SWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second);

/**
* @brief  Construct targitBits_first.size() new quantum SWAP gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit SWAP(const QVec& targitBits_first, const QVec& targitBits_second);

/**
* @brief  Construct a new quantum SWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate SWAP(int control_qaddr, int target_qaddr);

/**
* @brief  Construct control_qaddrs new quantum SWAP gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit SWAP(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs);

/**
* @brief  Construct a new quantum CNOT gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate MS(Qubit* control_qubit, Qubit* target_qubit);

/**
* @brief  Construct control_qubits.size() new quantum CNOT gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit MS(const QVec &control_qubits, const QVec &target_qubits);



/**
* @brief  Construct a new quantum CNOT gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate MS(int control_qaddr, int target_qaddr);

/**
* @brief  Construct control_qaddrs.size() new quantum CNOT gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit MS(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs);

QGate oracle(QVec qubits, std::string oracle_name);
QGate oracle(QVec qubits, std::string oracle_name, std::vector<double> &user_data);
QGate oracle(QVec qubits, std::string oracle_name, std::vector<std::vector<size_t>>& user_data); //Ol
QGate oracle(QVec qubits, std::string oracle_name, std::vector<std::vector<double>>& user_data);//OM

/**
* @brief  Construct a new quantum U4 gate
* @param[in]  Qubit* target qubit
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U4(Qubit*, double alpha, double beta, double gamma, double delta);

/**
* @brief  Construct qubits.size() new quantum U4 gate
* @param[in]  Qubit* target qubit
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit U4(const QVec& qubits, double alpha, double beta, double gamma, double delta);

/**
* @brief  Construct a new quantum U4 gate
* @param[in]  Qubit* target qubit
* @param[in]  QStat& matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U4(Qubit*, QStat& matrix);

/**
* @brief  Construct qbits.size() new quantum U4 gate
* @param[in]  Qubit* target qubit
* @param[in]  QStat& matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit U4(const QVec& qbits, QStat& matrix);

/**
* @brief  Construct a new quantum U4 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U4(int qaddr, double alpha, double beta, double gamma, double delta);

/**
* @brief  Construct qaddrs.size() new quantum U4 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  double alpha
* @param[in]  double beta
* @param[in]  double gamma
* @param[in]  double delta
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit U4(const std::vector<int>& qaddrs, double alpha, double beta, double gamma, double delta);

/**
* @brief  Construct a new quantum U4 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  QStat& matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate U4(int qaddr, QStat& matrix);

/**
* @brief  Construct qaddrs.size() new quantum U4 gate
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  QStat& matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit U4(const std::vector<int>& qaddrs, QStat& martix);

/**
* @brief  Construct a new quantum QDouble gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  QStat matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate QDouble(Qubit * qubit1, Qubit * qubit2, QStat& matrix);

/**
* @brief  Construct qubit1.size() new quantum QDouble gate
* @param[in]  Qubit* control qubit
* @param[in]  Qubit* target qubit
* @param[in]  QStat matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit QDouble(const QVec& qubit1, const QVec& qubit2, QStat& matrix);

/**
* @brief  Construct a new quantum QDouble gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  QStat matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate QDouble(int control_qaddr, int target_qaddr, QStat& matrix);

/**
* @brief  Construct control_qaddrs.size.() new quantum QDouble gate
* @param[in]  int  qaddr  control qubit phy addr
* @param[in]  int  qaddr  target qubit phy addr
* @param[in]  QStat matrix
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QCircuit QDouble(const std::vector<int>& control_qaddrs, const std::vector<int>& target_qaddrs, QStat& matrix);


/**
* @brief  Construct a new quantum BARRIER gate; Only for 6 qubits online projects !
* @param[in]  int  qaddr  target qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate BARRIER(int  qaddr);

/**
* @brief  Construct a new quantum BARRIER gate; Only for 6 qubits online projects !
* @param[in]  std::vector<int>  qaddrs  all qubit phy addr
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate BARRIER(std::vector<int> qaddrs);


/**
* @brief  Construct a new quantum BARRIER gate; Only for 6 qubits online projects !
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate BARRIER(Qubit* qubit);

/**
* @brief  Construct a new quantum BARRIER gate; Only for 6 qubits online projects !
* @param[in]  Qubit* qubit target qubit
* @return     QPanda::QGate  quantum gate
* @ingroup QuantumCircuit
*/
QGate BARRIER(QVec qubits);

/**
* @brief  Toffoli Quantum Gate
* @ingroup Utilities
* @param[in]  int  first control qubit addr
* @param[in]  int  second control addr
* @param[in]  int  target qubit addr
* @return     QGate
*/
QGate Toffoli(int qaddr0, int qaddr1, int target_qaddr);

/**
* @brief  Toffoli Quantum Gate
* @ingroup Utilities
* @param[in]  Qubit*  first control qubit
* @param[in]  Qubit*  second control qubit
* @param[in]  Qubit*  target qubit
* @return     QGate
*/
QGate Toffoli(Qubit * control_fisrt, Qubit * control_second, Qubit * target);

inline QGate copy_qgate(QuantumGate *  qgate_old, QVec qubit_vector)
{
    if (nullptr == qgate_old)
    {
        QCERR("param error");
        throw std::invalid_argument("param error");
    }
    auto gate_type = (GateType)qgate_old->getGateType();
    auto class_name = TransformQGateType::getInstance()[gate_type];

    auto temp_gate = QGateNodeFactory::getInstance()->getGateNode(class_name, qubit_vector, std::move(qgate_old));
    return temp_gate;
}

QGate QOracle(const QVec& qubits, const QStat& matrix, const double tolerance = 1e-10);

inline QGate copy_qgate(QGate &qgate, QVec qubit_vector)
{
    return copy_qgate(qgate.getQGate(), qubit_vector);
}

inline QGate copy_qgate(QGate *qgate, QVec qubit_vector)
{
    return copy_qgate(qgate->getQGate(), qubit_vector);
}


QPANDA_END
#endif // !_QGATE_H

