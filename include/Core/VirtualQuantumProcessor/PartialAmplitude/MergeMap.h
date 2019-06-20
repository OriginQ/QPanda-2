#ifndef  _MERGEMAP_H_
#define  _MERGEMAP_H_
#include "QPanda.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/PartialAmplitude/TraversalQProg.h"
QPANDA_BEGIN


#define _SINGLE_GATE(NAME) \
void _##NAME(QGateNode &node, QPUImpl *pQGate)\
{\
    pQGate->NAME(node.tar_qubit, node.isConjugate, 0);\
}\

#define _SINGLE_ANGLE_GATE(NAME) \
void _##NAME(QGateNode &node, QPUImpl *pQGate)\
{\
    pQGate->NAME(node.tar_qubit,node.gate_parm,node.isConjugate, 0);\
}\

#define _DOUBLE_GATE(NAME) \
void _##NAME(QGateNode &node, QPUImpl *pQGate)\
{\
    pQGate->NAME(node.ctr_qubit,node.tar_qubit,node.isConjugate, 0);\
}\

#define _DOUBLE_ANGLE_GATE(NAME) \
void _##NAME(QGateNode &node, QPUImpl *pQGate)\
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
};

class MergeMap
{
public:
    size_t m_qubit_num;
    std::vector<QGateNode> m_circuit;
    std::vector<std::map<bool, std::vector<QGateNode>>> m_circuit_vec;
    MergeMap();

    inline size_t getMapVecSize() noexcept
    {
        return m_circuit_vec.size();
    }

    inline void clear() noexcept
    {
        m_circuit.clear();
        m_circuit_vec.clear();
    }
    void traversalMap(std::vector<QGateNode> &, QPUImpl *, QuantumGateParam*);
    bool isCorssNode(size_t, size_t);
    void traversalQlist(std::vector<QGateNode> &);
    void splitQlist(std::vector<QGateNode> &);

private:
    std::map<unsigned short, unsigned short> m_key_map;
    std::map<unsigned short, std::function<void(QGateNode&, QPUImpl*)> > m_GateFunc;
};

QPANDA_END
#endif
