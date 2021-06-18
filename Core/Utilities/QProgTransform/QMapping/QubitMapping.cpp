#include "Core/Utilities/QProgTransform/QMapping/QubitMapping.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include <iterator>

using namespace std;
using namespace QPanda;

const uint32_t QPanda::UNDEF_UINT32 = (std::numeric_limits<uint32_t>::max)();

InverseMap QPanda::InvertMapping(uint32_t archQ, Mapping mapping, bool fill) {
    uint32_t progQ = mapping.size();
    // 'archQ' is the number of qubits from the architecture.
    std::vector<uint32_t> inv(archQ, UNDEF_UINT32);

    // for 'u' in arch; and 'a' in prog:
    // if 'a' -> 'u', then 'u' -> 'a'
    for (uint32_t i = 0; i < progQ; ++i)
        if (mapping[i] != UNDEF_UINT32)
            inv[mapping[i]] = i;

    if (fill) {
        // Fill the qubits in the architecture that were not mapped.
        Fill(mapping, inv);
    }

    return inv;
}

void QPanda::Fill(Mapping& mapping, InverseMap& inv) {
    uint32_t progQ = mapping.size(), archQ = inv.size();
    uint32_t a = 0, u = 0;

    do {
        while (a < progQ && mapping[a] != UNDEF_UINT32) ++a;
        while (u < archQ && inv[u] != UNDEF_UINT32) ++u;

        if (u < archQ && a < progQ) {
            mapping[a] = u;
            inv[u] = a;
            ++u; ++a;
        } else {
            break;
        }
    } while (true);
}

void QPanda::Fill(uint32_t archQ, Mapping& mapping) {
    auto inv = InvertMapping(archQ, mapping, false);
    Fill(mapping, inv);
}

Mapping QPanda::IdentityMapping(uint32_t progQ) {
    Mapping mapping(progQ, UNDEF_UINT32);

    for (uint32_t i = 0; i < progQ; ++i) {
        mapping[i] = i;
    }

    return mapping;
}

std::string QPanda::MappingToString(Mapping m) {
    std::string s = "[";
    for (uint32_t i = 0, e = m.size(); i < e; ++i) {
        s = s + std::to_string(i) + " => ";
        if (m[i] == UNDEF_UINT32) s = s + "UNDEF_UINT32";
        else s = s + std::to_string(m[i]);
        s = s + ";";
        if (i != e - 1) s = s + " ";
    }
    s = s + "]";
    return s;
}

/*******************************************************************
*                      class AbstractQubitMapping
********************************************************************/
AbstractQubitMapping::AbstractQubitMapping(ArchGraph::sRef archGraph)
	: mArchGraph(archGraph), m_CX_cost(10), m_CZ_cost(10), m_u3_cost(1)
{
    mGateWeightMap = { {"U", 1}, {"CX", 10}, {"CZ", 10} };
}

uint32_t AbstractQubitMapping::get_CX_cost(uint32_t u, uint32_t v)
{
	if (mArchGraph->hasEdge(u, v)) return m_CX_cost;
	if (mArchGraph->hasEdge(v, u)) return m_CX_cost + (4 * m_u3_cost);
}

uint32_t AbstractQubitMapping::get_CZ_cost(uint32_t u, uint32_t v)
{
	if ((mArchGraph->hasEdge(u, v)) || (mArchGraph->hasEdge(v, u))) return m_CZ_cost;
}

uint32_t AbstractQubitMapping::getSwapCost(uint32_t u, uint32_t v) {
    uint32_t uvCost = get_CZ_cost(u, v);
    return uvCost * 3;
}

bool AbstractQubitMapping::run(QPanda::QProg prog, QuantumMachine *qvm)
{
    // Filling Qubit information.
	QVec used_qubits;
    mVQubits = get_all_used_qubits(prog, used_qubits);
    mPQubits = mArchGraph->size();

	if (mVQubits > mPQubits)
	{
		QCERR_AND_THROW(run_fail, "Error: The number of qubits used in target QPorg exceeds the number of qubits of physical chips.");
	}

	m_final_mapping = allocate(prog, qvm);
    return true;
}

/*******************************************************************
*                      class FrontLayer
********************************************************************/
void FrontLayer::remove_node(pPressedCirNode p_node)
{
	update_cur_layer_qubits(p_node);
	for (auto itr = m_front_layer_nodes.begin(); itr != m_front_layer_nodes.end(); ++itr) {
		if (*itr == p_node) {
			m_front_layer_nodes.erase(itr);
			return;
		}
	}

	QCERR_AND_THROW(run_fail, "unknow error on FrontLayer::remove_node.");
}

/*******************************************************************
*                      class DynamicQCircuitGraph
********************************************************************/
DynamicQCircuitGraph::DynamicQCircuitGraph(QProg prog)
	:m_prog(prog) {
	init();
}

DynamicQCircuitGraph::DynamicQCircuitGraph(const DynamicQCircuitGraph& c) 
{
	m_prog = c.m_prog;
	m_layer_info = c.m_layer_info;
	m_cur_layer_iter = m_layer_info.begin();
}

void DynamicQCircuitGraph::init() 
{
	auto start = chrono::system_clock::now();
	m_layer_info = get_pressed_layer(m_prog);
#if PRINT_TRACE
	auto end = chrono::system_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
	cout << "The layer takes "
		<< double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den
		<< "seconds. layers_cnt:" << m_layer_info.size() << endl;
	{
		size_t gate_cnt = 0;
		for (auto& layer : m_layer_info) {
			gate_cnt += layer.size();
			for (const auto& node : layer) {
				gate_cnt += node.first->m_relation_pre_nodes.size();
				gate_cnt += node.first->m_relation_successor_nodes.size();
			}
		}
		cout << "total gate_cnt=" << gate_cnt << endl;
		/*cout << "Press enter to continue." << endl;
		getchar();*/
	}
#endif
	m_cur_layer_iter = m_layer_info.begin();
}

FrontLayer& DynamicQCircuitGraph::get_front_layer()
{
	bool b_stay_cur_layer = false;
	if (m_cur_layer_iter != m_layer_info.end())
	{
		auto& cur_layer = *m_cur_layer_iter;
		for (auto node_iter = cur_layer.begin(); node_iter != cur_layer.end(); )
		{
			const auto tmp_node = node_iter->first->m_cur_node;
			if (BARRIER_GATE == tmp_node->m_gate_type)
			{
				m_front_layer.m_front_layer_nodes.push_back(node_iter->first);
				m_front_layer.m_cur_layer_qubits += (tmp_node->m_target_qubits + tmp_node->m_control_qubits);
				node_iter = cur_layer.erase(node_iter);
				continue;
			}

			auto q = tmp_node->m_target_qubits - m_front_layer.m_cur_layer_qubits;
			bool b_qubit_multiplex = (q.size() != tmp_node->m_target_qubits.size());
			b_stay_cur_layer = (b_stay_cur_layer || b_qubit_multiplex);
			if (!b_qubit_multiplex)
			{
				m_front_layer.m_front_layer_nodes.push_back(node_iter->first);
				m_front_layer.m_cur_layer_qubits += tmp_node->m_target_qubits;
				node_iter = cur_layer.erase(node_iter);
				continue;
			}

			++node_iter;
		}

		if (cur_layer.size() == 0) {
			++m_cur_layer_iter;
		}
	}

	return m_front_layer;
}