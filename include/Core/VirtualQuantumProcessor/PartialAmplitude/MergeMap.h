#ifndef  _MERGEMAP_H_
#define  _MERGEMAP_H_
#include "QPanda.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/PartialAmplitude/TraversalQProg.h"
QPANDA_BEGIN


#define SINGLE_GATE(NAME) \
void NAME##_Gate(QGateNode &node, QPUImpl *pQGate)\
{\
    pQGate->NAME(node.tar_qubit, node.isConjugate, 0);\
}\

#define SINGLE_ANGLE_GATE(NAME) \
void NAME##_Gate(QGateNode &node, QPUImpl *pQGate)\
{\
    pQGate->NAME(node.tar_qubit,node.gate_parm,node.isConjugate, 0);\
}\

#define DOUBLE_GATE(NAME) \
void NAME##_Gate(QGateNode &node, QPUImpl *pQGate)\
{\
    pQGate->NAME(node.ctr_qubit,node.tar_qubit,node.isConjugate, 0);\
}\

#define DOUBLE_ANGLE_GATE(NAME) \
void NAME##_Gate(QGateNode &node, QPUImpl *pQGate)\
{\
    pQGate->NAME(node.ctr_qubit,node.tar_qubit,node.gate_parm,node.isConjugate, 0);\
}\

struct QGateNode
{
    unsigned short gate_type;
    bool isConjugate;
    size_t tar_qubit;
    size_t ctr_qubit;
    double gate_parm;
    //std::vector<size_t> ctr_list;
};

class MergeMap : public TraversalQProg 
{
public:      
    std::vector<std::map<bool, std::vector<QGateNode>>> m_circuit_vec;
    MergeMap();
 
    void traversalAll(AbstractQuantumProgram *) ;
    void traversalMap(std::vector<QGateNode> &, QPUImpl *, QuantumGateParam*);


    inline size_t getMapVecSize() noexcept
    {
        return m_circuit_vec.size();
    }

    inline void clear() noexcept
    {
        m_circuit.clear();
        m_circuit_vec.clear();
    }

private:
    bool isCorssNode(size_t, size_t);
    void traversalQlist(std::vector<QGateNode> &);
    void splitQlist(std::vector<QGateNode> &);
    void traversal(AbstractQGateNode *);


    std::map<size_t, size_t> m_key_map;
    std::vector<QGateNode> m_circuit;
    std::map<unsigned short, std::function<void(QGateNode&, QPUImpl*)> > m_GateFunc;
};

QPANDA_END
#endif