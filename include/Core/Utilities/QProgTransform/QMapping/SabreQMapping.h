#ifndef SABRE_MAPPING_H
#define SABRE_MAPPING_H

#include "QubitMapping.h"
#include "ShortestDistanceByBFS.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/Utilities/QProgTransform/QMapping//ShortestDistanceByBFS.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include <random>
#include <queue>

QPANDA_BEGIN

class DynamicQCircuitGraph;

/**
* @brief SABRE qubit mapping
   Implemented from Gushu et. al.:
   Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices
*/
class SabreQAllocator : public AbstractQubitMapping {
public:
	typedef SabreQAllocator* Ref;
	typedef std::unique_ptr<SabreQAllocator> uRef;

public:
	static uRef Create(QPanda::ArchGraph::sRef ag, uint32_t max_look_ahead = 20, uint32_t max_iterations = 10);
	uint32_t get_swap_cnt() const { return m_swap_cnt; }

private:
	typedef std::pair<Mapping, uint32_t> MappingAndNSwaps;

	uint32_t mLookAhead; /**< Sets the number of instructions to peek. */
	uint32_t mIterations; /**< Sets the number of times to run SABRE. */
	uint32_t m_swap_cnt; /**< the count of SWAP for currently best mapping. */
	QPanda::ShortestDistanceByBFS m_shortest_distance;

	MappingAndNSwaps allocateWithInitialMapping(const Mapping& initialMapping, DynamicQCircuitGraph cir_graph,
		QPanda::QuantumMachine *qvm, bool issueInstructions);

protected:
	SabreQAllocator(QPanda::ArchGraph::sRef ag, uint32_t max_look_ahead, uint32_t max_iterations);
	Mapping allocate(QPanda::QProg qmod, QPanda::QuantumMachine *qvm) override;
	std::vector<QNodeRef> get_candidates_nodes();
	std::vector<QNodeRef> candidates_node_vector;
	void remap_node_to_new_prog(QNodeRef node, const Mapping& mapping, QPanda::QuantumMachine *qvm);
};

/**
* @brief SABRE QAllocator
* @ingroup Utilities
* @param[in] prog  the target prog
* @param[in] QuantumMachine *  quantum machine
* @param[out] QVec& Mapped bit sequence
* @param[in] uint32_t The number of instructions to peek, default is 20
* @param[in] uint32_t The number of times to run SABRE, default is 10
* @param[in] const std::string config data, @See JsonConfigParam::load_config()
* @return QProg   mapped  quantum program
*/
QProg SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv, uint32_t max_look_ahead = 20, 
	uint32_t max_iterations = 10, const std::string& config_data = CONFIG_PATH);

QProg SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv, std::vector<uint32_t>& init_map,
	uint32_t max_look_ahead = 20, uint32_t max_iterations = 10, const std::string& config_data = CONFIG_PATH);

QPANDA_END
#endif
