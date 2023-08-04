#ifndef VIRTUAL_Z_TRANSFER_H
#define VIRTUAL_Z_TRANSFER_H
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/ProcessOnTraversing.h"
#include <vector>
#include "Core/Utilities/Tools/ThreadPool.h"
#include "Core/Utilities/Tools/JsonConfigParam.h"
#include <atomic>
#include <ctime>
#include <thread>

QPANDA_BEGIN

/**
* @brief  transfer QGate to RX + RZ + CZ
* @ingroup Utilities
* @param[in]  QProg&  Quantum Program
* @param[in]  QuantumMachine*  quantum machine pointer
* @param[in] const std::string config data, @See JsonConfigParam::load_config()
* @return
*/
void transfer_to_rotating_gate(QProg& prog, QuantumMachine* quantum_machine, 
	const std::string& config_data = CONFIG_PATH);

/**
* @brief Z-direction crosstalk compensation
* @ingroup Utilities
* @param[in]  QProg&  Quantum Program
* @param[in] const std::string config data, @See JsonConfigParam::load_config()
* @return
*/
void cir_crosstalk_compensation(QProg& prog, const std::string& config_data = CONFIG_PATH);

/**
* @brief Virtual z-gate conversion
* @ingroup Utilities
* @param[in]  QProg& or QCircuit&  Quantum Program
* @param[in]  QuantumMachine*  quantum machine pointer
* @param[in] const std::string config data, @See JsonConfigParam::load_config()
* @return
*/
void virtual_z_transform(QCircuit& cir, QuantumMachine* quantum_machine, const bool b_del_rz_gate = false, const std::string& config_data = CONFIG_PATH);
void virtual_z_transform(QProg& prog, QuantumMachine* quantum_machine, const bool b_del_rz_gate = false, const std::string& config_data = CONFIG_PATH);

void move_rz_backward(QProg& prog, const bool b_del_rz_gate = false);

void transfer_to_u3_gate(QProg& prog, QuantumMachine* quantum_machine);
void transfer_to_u3_gate(QCircuit& circuit, QuantumMachine* quantum_machine);

//decompose U3 gate
void decompose_U3(QProg& prog, const std::string& config_data = CONFIG_PATH);

QPANDA_END
#endif