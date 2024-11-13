#include "Core/Utilities/Compiler/QASMToOriginIR.hpp"


QPANDA_BEGIN

std::string qasmfile2str(const std::string& filename) {
    std::stringstream ss;
    std::ifstream ifs;
    ifs.open(filename);
    if (ifs.is_open()) {
        std::string line;
        std::cout << "### opened qasm file:" << filename << std::endl;
        while (std::getline(ifs, line)) {
            ss << line;
        }
        ifs.close();
    }
    else {
        std::cerr << "###Error: qasmfile2str open " << filename << "failed." << std::endl;
        exit(-1);
        return {};
    }
    return ss.str();
}

std::string convert_qasm_to_originir(std::string qasm_filepath) {
    return QuantumComputation::fromQASM(qasmfile2str(qasm_filepath)).toOriginIR();
}
std::string convert_qasm_string_to_originir(std::string qasm_str) {
    return QuantumComputation::fromQASM(qasm_str).toOriginIR();
}

QPANDA_END
