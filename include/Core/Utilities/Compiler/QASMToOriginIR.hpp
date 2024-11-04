#ifndef  _QASM_TO_ORIGINIR_H
#define  _QASM_TO_ORIGINIR_H

#include "Core/Utilities/Compiler/QuantumComputation.hpp"
#include "Core/Utilities/QPandaNamespace.h"


QPANDA_BEGIN
using namespace qc;
std::string qasmfile2str(const std::string& filename);

std::string convert_qasm_to_originir(std::string file_path);
std::string convert_qasm_string_to_originir(std::string qasm_str);

QPANDA_END
#endif //!_QASMTOQPORG_H