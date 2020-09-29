/*! \file QuantumChipAdapter.h */
#ifndef  QUANTUM_CHIP_ADAPTER_H
#define  QUANTUM_CHIP_ADAPTER_H

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"

QPANDA_BEGIN

class QuantumChipAdapter
{
public:
	QuantumChipAdapter(QuantumMachine *quantum_machine, bool b_mapping = true, const std::string config_data = CONFIG_PATH);
	~QuantumChipAdapter() {}

	void init();
	void adapter_conversion(QProg& prog, QVec &new_qvec);
	void mapping(QProg &prog);

private:
	QuantumMachine * m_quantum_machine;
	bool m_b_enable_mapping;
	const std::string m_config_data;
	std::vector<std::vector<std::string>> m_valid_gate;
	std::vector<std::vector<std::string>> m_gates;
	std::shared_ptr<TransformDecomposition> m_p_transf_decompos;
	QVec m_new_qvec;
};

/**
* @brief  Quantum chip adaptive conversion
* @ingroup Utilities
* @param[in]  QProg&   Quantum Program
* @param[in]  QuantumMachine*  quantum machine pointer
* @param[out] QVec& Quantum bits after mapping.
              Note: if b_mapping is false, the input QVec will be misoperated.
* @param[in]  bool whether or not perform the mapping operation.
* @param[in] const std::string It can be configuration file or configuration data, which can be distinguished by file suffix, 
             so the configuration file must be end with ".json", default is CONFIG_PATH
* @return
*/
void quantum_chip_adapter(QProg& prog, QuantumMachine *quantum_machine, QVec &new_qvec, bool b_mapping = true, const std::string config_data = CONFIG_PATH);
void quantum_chip_adapter(QCircuit& cir, QuantumMachine *quantum_machine, QVec &new_qvec, bool b_mapping = true, const std::string config_data = CONFIG_PATH);

QPANDA_END

#endif