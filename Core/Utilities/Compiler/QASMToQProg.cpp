
#include "Core/Utilities/Compiler/QASMToQProg.hpp"

QPANDA_BEGIN


extern QProg convert_originir_to_qprog(std::string file_path, QuantumMachine* qm);
extern QProg convert_originir_to_qprog(std::string file_path, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);

extern QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);

extern QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine* qm);

extern std::string convert_qasm_to_originir(std::string file_path);
extern std::string convert_qasm_string_to_originir(std::string qasm_str);

QProg convert_qasm_string_to_qprog(std::string qasm_str, QuantumMachine* qvm)
{
    return convert_originir_string_to_qprog(convert_qasm_string_to_originir(qasm_str), qvm);
}

QProg convert_qasm_string_to_qprog(std::string qasm_str,QuantumMachine* qvm, QVec& qv, std::vector<ClassicalCondition>& cv)
{
	return convert_originir_string_to_qprog(convert_qasm_string_to_originir(qasm_str), qvm, qv, cv);
}

QProg convert_qasm_to_qprog(std::string qasm_filepath, QuantumMachine* qvm, QVec& qv, std::vector<ClassicalCondition>& cv)
{   
    return convert_originir_string_to_qprog(convert_qasm_to_originir(qasm_filepath), qvm,qv,cv);
}

QProg convert_qasm_to_qprog(std::string qasm_filepath, QuantumMachine* qvm)
{
    return convert_originir_string_to_qprog(convert_qasm_to_originir(qasm_filepath), qvm);
}
QPANDA_END