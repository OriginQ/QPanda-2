#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>
QPANDA_BEGIN
using grover_oracle = Oracle<QVec, Qubit*>;

/**
* @brief  Grover Algorithm
* @ingroup Grover_Algorithm
* @param[in] size_t target number
* @param[in] size_t search range
* @param[in] QuantumMachine* Quantum machine ptr
* @param[in] grover_oracle Grover Algorithm oracle
* @return    QProg
* @note  
*/
QProg groverAlgorithm(size_t target,
	size_t search_range,
	QuantumMachine * qvm,
	grover_oracle oracle);
QPANDA_END