#ifndef  _PARTIALAMPLITUDEGRAPH_H_
#define  _PARTIALAMPLITUDEGRAPH_H_
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/VirtualQuantumProcessor/QPUImpl.h"
#include "Core/VirtualQuantumProcessor/CPUImplQPU.h"
QPANDA_BEGIN

struct QGateNode
{
    int gate_type;
    bool is_dagger;
    std::vector<uint32_t> qubits; //first target, other controls
    std::vector<double> params;
};

using cir_type = std::vector<QGateNode>;

/**
* @brief  Partial Amplitude Graph
* @ingroup VirtualQuantumProcessor
*/
class PartialAmplitudeGraph
{
public:
	uint32_t m_spilt_num;
	uint32_t m_qubit_num;

    std::vector<QGateNode> m_circuit;
	std::vector<std::vector<cir_type>> m_sub_graph;

    PartialAmplitudeGraph();

	inline void reset(size_t qubit_num) noexcept
	{
        m_spilt_num = 0;
		m_qubit_num = qubit_num;
		m_circuit.clear();
		m_sub_graph.clear();
	}

    void computing_graph(const cir_type &, std::shared_ptr<QPUImpl>);
    bool is_corss_node(size_t, size_t);
    void traversal(std::vector<QGateNode> &);
    void split_circuit(std::vector<QGateNode> &);

private:
	std::unordered_map<unsigned short, unsigned short> m_key_map;
    std::unordered_map<unsigned short, std::function<void(QGateNode&, QPUImpl*)> > m_function_mapping;
};

QPANDA_END
#endif
