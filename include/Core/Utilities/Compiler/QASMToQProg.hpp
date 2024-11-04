#ifndef  _QASMTOQPORG_H
#define  _QASMTOQPORG_H

//#include "Core/Utilities/Compiler/QuantumComputation.hpp"

#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/Utilities/Compiler/QASMToOriginIR.hpp"


//using namespace qc;


QPANDA_BEGIN


QProg convert_qasm_to_qprog(std::string file_path, QuantumMachine* qvm);
QProg convert_qasm_to_qprog(std::string file_path, QuantumMachine* qvm, QVec& qv, std::vector<ClassicalCondition>& cv);

QProg convert_qasm_string_to_qprog(std::string qasm_str, QuantumMachine* qvm);
QProg convert_qasm_string_to_qprog(std::string qasm_str, QuantumMachine* qvm, QVec& qv, std::vector<ClassicalCondition>& cv);
QPANDA_END
#endif //!_QASMTOQPORG_H