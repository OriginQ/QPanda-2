#ifndef SABRE_MAPPING_H
#define SABRE_MAPPING_H

#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include <random>
#include <queue>
#include <optional>

#include "QMapping/QubitMapping.h"
#include "QMapping/ShortestDistanceByBFS.h"

QPANDA_BEGIN

class DynamicQCircuitGraph;

/**
 * @brief SABRE qubit mapping
 *  Implemented from Gushu et. al.:
 *  Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices
 */
class SabreQAllocator : public AbstractQubitMapping {
public:
	typedef SabreQAllocator* Ref;
	typedef std::unique_ptr<SabreQAllocator> uRef;

public:
	static uRef Create(QPanda::ArchGraph::sRef ag, uint32_t mLookAhead = 20, uint32_t mIterations = 0, uint32_t mRandomMappings = 10000);
	std::map<double, std::vector<Mapping>, std::greater<double>>
	select_best_qubits_blocks(QPanda::ArchGraph::sRef& arch_graph, QProg& qmod);
	uint32_t get_swap_cnt() const { return m_swap_cnt; }

private:
	typedef std::pair<Mapping, uint32_t> MappingAndNSwaps;

	uint32_t m_look_ahead;				/**< Sets the number of instructions to peek. */
	uint32_t m_max_iterations;			/**< Sets the number of times to run SABRE. */
	uint32_t m_max_random_mappings;		/**< Set the maximum number of random mappings */
	uint32_t m_swap_cnt;				/**< the count of SWAP for currently best mapping. */
	QPanda::ShortestDistanceByBFS m_shortest_distance;

	std::optional<MappingAndNSwaps> allocateWithInitialMapping(const Mapping& initialMapping, DynamicQCircuitGraph cir_graph,
		QPanda::QuantumMachine *qvm, ArchGraph::sRef arch_graph, bool issueInstructions);

protected:
	SabreQAllocator(QPanda::ArchGraph::sRef ag, uint32_t mLookAhead, uint32_t mIterations, uint32_t mRandomMappings);
	Mapping allocate(QPanda::QProg qmod, QPanda::QuantumMachine *qvm) override;
	std::vector<QNodeRef> get_candidates_nodes();
	std::vector<QNodeRef> candidates_node_vector;
	void remap_node_to_new_prog(QNodeRef node, const Mapping& mapping, QPanda::QuantumMachine *qvm);
    ArchGraph::sRef m_target_sub_graph; /* subgraph for the target qubit block */
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
	uint32_t max_iterations = 0, const QMappingConfig& config_data = QMappingConfig(CONFIG_PATH),
	uint32_t m_max_random_mappings = 10000, uint32_t hops = 1);

QProg SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv,
	std::map<double, std::vector<std::vector<uint32_t>>, std::greater<double>>& specified_blocks,
	uint32_t max_look_ahead = 20, uint32_t max_iterations = 0,
	const QMappingConfig& config_data = QMappingConfig(CONFIG_PATH),
	uint32_t m_max_random_mappings = 10000, uint32_t hops = 1);

QProg SABRE_mapping(QProg prog, QuantumMachine *quantum_machine, QVec &qv, std::vector<uint32_t>& init_map,
	std::map<double, std::vector<std::vector<uint32_t>>, std::greater<double>>& specified_blocks,
	uint32_t max_look_ahead = 20, uint32_t max_iterations = 0,
	const QMappingConfig& config_data = QMappingConfig(CONFIG_PATH),
	uint32_t m_max_random_mappings = 10000, uint32_t hops = 1);

QPANDA_END
#endif
