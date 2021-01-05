#include "Core/Utilities/QProgTransform/BMT/QbitAllocator.h"
#include "Core/Utilities/QProgInfo/QCircuitInfo.h"
#include <iterator>

using namespace BMT;
using namespace std;
using namespace QPanda;

InverseMap BMT::InvertMapping(uint32_t archQ, Mapping mapping, bool fill) {
    uint32_t progQ = mapping.size();
    // 'archQ' is the number of qubits from the architecture.
    std::vector<uint32_t> inv(archQ, _undef);

    // for 'u' in arch; and 'a' in prog:
    // if 'a' -> 'u', then 'u' -> 'a'
    for (uint32_t i = 0; i < progQ; ++i)
        if (mapping[i] != _undef)
            inv[mapping[i]] = i;

    if (fill) {
        // Fill the qubits in the architecture that were not mapped.
        Fill(mapping, inv);
    }

    return inv;
}

void BMT::Fill(Mapping& mapping, InverseMap& inv) {
    uint32_t progQ = mapping.size(), archQ = inv.size();
    uint32_t a = 0, u = 0;

    do {
        while (a < progQ && mapping[a] != _undef) ++a;
        while (u < archQ && inv[u] != _undef) ++u;

        if (u < archQ && a < progQ) {
            mapping[a] = u;
            inv[u] = a;
            ++u; ++a;
        } else {
            break;
        }
    } while (true);
}

void BMT::Fill(uint32_t archQ, Mapping& mapping) {
    auto inv = InvertMapping(archQ, mapping, false);
    Fill(mapping, inv);
}

Mapping BMT::IdentityMapping(uint32_t progQ) {
    Mapping mapping(progQ, _undef);

    for (uint32_t i = 0; i < progQ; ++i) {
        mapping[i] = i;
    }

    return mapping;
}

std::string BMT::MappingToString(Mapping m) {
    std::string s = "[";
    for (uint32_t i = 0, e = m.size(); i < e; ++i) {
        s = s + std::to_string(i) + " => ";
        if (m[i] == _undef) s = s + "_undef";
        else s = s + std::to_string(m[i]);
        s = s + ";";
        if (i != e - 1) s = s + " ";
    }
    s = s + "]";
    return s;
}

// ------------------ QbitAllocator ----------------------
QbitAllocator::QbitAllocator(ArchGraph::sRef archGraph) 
	: mArchGraph(archGraph), m_CX_cost(10), m_CZ_cost(10), m_u3_cost(1)
{
    mGateWeightMap = { {"U", 1}, {"CX", 10}, {"CZ", 10} };
}

uint32_t QbitAllocator::get_CX_cost(uint32_t u, uint32_t v)
{
	if (mArchGraph->hasEdge(u, v)) return m_CX_cost;
	if (mArchGraph->hasEdge(v, u)) return m_CX_cost + (4 * m_u3_cost);
}

uint32_t QbitAllocator::get_CZ_cost(uint32_t u, uint32_t v)
{
	if ((mArchGraph->hasEdge(u, v)) || (mArchGraph->hasEdge(v, u))) return m_CZ_cost;
}

uint32_t QbitAllocator::getSwapCost(uint32_t u, uint32_t v) {
    uint32_t uvCost = get_CZ_cost(u, v);
    return uvCost * 3;
}

bool QbitAllocator::run(QPanda::QProg prog, QuantumMachine *qvm) 
{
    // Filling Qubit information.
	QVec used_qubits;
    mVQubits = get_all_used_qubits(prog, used_qubits);
    mPQubits = mArchGraph->size();

	cout << "start allocate-----." << endl;
	m_mapping = allocate(prog, qvm);
	cout << "------finished allocate." << endl;

    std::cout << "Initial Configuration: " << MappingToString(m_mapping) << std::endl;
    return true;
}
