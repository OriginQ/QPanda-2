#ifndef  _PARTIALAMPLITUDEGRAPH_H_
#define  _PARTIALAMPLITUDEGRAPH_H_
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/CPUImplQPU.h"
QPANDA_BEGIN


#define _SINGLE_GATE(NAME) \
void _##NAME(QGateNode &node, CPUImplQPU *pQGate)\
{\
    pQGate->NAME(node.tar_qubit, node.isConjugate, 0);\
}\

#define _SINGLE_ANGLE_GATE(NAME) \
void _##NAME(QGateNode &node, CPUImplQPU *pQGate)\
{\
    pQGate->NAME(node.tar_qubit,node.gate_parm,node.isConjugate, 0);\
}\

#define _DOUBLE_GATE(NAME) \
void _##NAME(QGateNode &node, CPUImplQPU *pQGate)\
{\
    pQGate->NAME(node.ctr_qubit,node.tar_qubit,node.isConjugate, 0);\
}\

#define _DOUBLE_ANGLE_GATE(NAME) \
void _##NAME(QGateNode &node, CPUImplQPU *pQGate)\
{\
    pQGate->NAME(node.ctr_qubit,node.tar_qubit,node.gate_parm,node.isConjugate, 0);\
}\

#define _TRIPLE_GATE(NAME) \
void _##NAME(QGateNode &node, CPUImplQPU *pQGate)\
{\
    Qnum ctr_qubits={node.ctr_qubit,node.tof_qubit};\
    pQGate->X(node.tar_qubit,ctr_qubits,node.isConjugate, 0);\
}\


struct QGateNode
{
    unsigned short gate_type;
    bool isConjugate;
    uint32_t tar_qubit;
    uint32_t ctr_qubit;
    float gate_parm;
    uint32_t tof_qubit;
};

/**
* @brief  Partial Amplitude Graph
* @ingroup VirtualQuantumProcessor
*/
class PartialAmplitudeGraph
{
public:
    uint32_t m_qubit_num, m_spilt_num;
    std::vector<QGateNode> m_circuit;
    std::vector<std::map<bool, std::vector<QGateNode>>> m_circuit_vec;
    PartialAmplitudeGraph();

    inline size_t getMapVecSize() noexcept
    {
        return m_circuit_vec.size();
    }

    void init(uint32_t qubit_num) noexcept
    {
        m_circuit.clear();
        m_circuit_vec.clear();
        m_qubit_num = qubit_num;
    }

    void traversalMap(std::vector<QGateNode> &, QPUImpl *);
    bool isCorssNode(size_t, size_t);
    void traversalQlist(std::vector<QGateNode> &);
    void splitQlist(std::vector<QGateNode> &);

private:
    std::map<unsigned short, unsigned short> m_key_map;
    std::map<unsigned short, std::function<void(QGateNode&, CPUImplQPU*)> > m_GateFunc;
};

QPANDA_END
#endif
