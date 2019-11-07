#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Utils.h"
#include "Core/Utilities/QPandaNamespace.h"
#include <vector>
QPANDA_BEGIN
using grover_oracle = Oracle<QVec, Qubit*>;

/**
* @brief  Grover Algorithm
* @ingroup QAlg
* @param[in]  target number
* @param[in]  search range
* @param[in]  Quantum machine ptr
* @param[in]  Quantum machine ptr
* @param[in]  Grover Algorithm oracle
* @return    QProg
* @note  
*/
QProg groverAlgorithm(size_t target,
	size_t search_range,
	QuantumMachine * qvm,
	grover_oracle oracle);
QPANDA_END