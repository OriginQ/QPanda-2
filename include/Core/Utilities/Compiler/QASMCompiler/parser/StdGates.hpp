#pragma once

#include "Core/Utilities/Compiler/QASMCompiler/parser/Gate.hpp"

#include <string>

namespace qasm {
// Non-natively supported gates from
// https://github.com/Qiskit/qiskit/blob/main/qiskit/qasm/libs/stdgates.inc
const std::string STDGATES =
    "// four parameter controlled-U gate with relative phase\n"
    "gate cu(theta, phi, lambda, gamma) c, t { p(gamma) c; ctrl @ U(theta, "
    "phi, lambda) c, t; }\n";

// Non-natively supported gates from
// https://github.com/Qiskit/qiskit/blob/main/qiskit/qasm/libs/qelib1.inc
const std::string QE1LIB = "gate rccx a, b, c {\n"
                           "  u2(0, pi) c; u1(pi/4) c; \n"
                           "  cx b, c; u1(-pi/4) c; \n"
                           "  cx a, c; u1(pi/4) c; \n"
                           "  cx b, c; u1(-pi/4) c; \n"
                           "  u2(0, pi) c; \n"
                           "}\n"
                           "gate rc3x a,b,c,d {\n"
                           "  u2(0,pi) d; u1(pi/4) d; \n"
                           "  cx c,d; u1(-pi/4) d; u2(0,pi) d; \n"
                           "  cx a,d; u1(pi/4) d; \n"
                           "  cx b,d; u1(-pi/4) d; \n"
                           "  cx a,d; u1(pi/4) d; \n"
                           "  cx b,d; u1(-pi/4) d; \n"
                           "  u2(0,pi) d; u1(pi/4) d; \n"
                           "  cx c,d; u1(-pi/4) d; \n"
                           "  u2(0,pi) d; \n"
                           "}\n"
                           "gate c3x a,b,c,d {\n"
                           "  h d; cu1(-pi/4) a,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(pi/4) b,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(-pi/4) b,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/4) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/4) c,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/4) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/4) c,d; h d; \n"
                           "}\n"
                           "gate c3sqrtx a,b,c,d {\n"
                           "  h d; cu1(-pi/8) a,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(pi/8) b,d; h d; \n"
                           "  cx a,b; \n"
                           "  h d; cu1(-pi/8) b,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/8) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/8) c,d; h d; \n"
                           "  cx b,c; \n"
                           "  h d; cu1(pi/8) c,d; h d; \n"
                           "  cx a,c; \n"
                           "  h d; cu1(-pi/8) c,d; h d; \n"
                           "}\n"
                           "gate c4x a,b,c,d,e {\n"
                           "  h e; cu1(-pi/2) d,e; h e; \n"
                           "  c3x a,b,c,d; \n"
                           "  h e; cu1(pi/2) d,e; h e; \n"
                           "  c3x a,b,c,d; \n"
                           "  c3sqrtx a,b,c,e; \n"
                           "}\n";

const std::map<std::string, std::shared_ptr<Gate>> STANDARD_GATES = {
    // gates from which all other gates can be constructed.
    {"gphase",
     std::make_shared<StandardGate>(StandardGate({0, 0, 1, qc::otGPhase}))},

    {"u", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::otU3}))},
    {"U", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::otU3}))},
    {"u3", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::otU3}))},
    {"U3", std::make_shared<StandardGate>(StandardGate({0, 1, 3, qc::otU3}))},

    // natively supported gates
    {"p", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otP}))},
    {"P", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otP}))},
    {"u1", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otP}))},
    {"U1", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otP}))},
    {"phase", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otP}))},

    {"cu", std::make_shared<StandardGate>(StandardGate({1, 1, 4, qc::otCU}))},
    {"CU", std::make_shared<StandardGate>(StandardGate({1, 1, 4, qc::otCU}))},

    {"cu3", std::make_shared<StandardGate>(StandardGate({1, 1, 3, qc::otCU3}))},
    {"CU3", std::make_shared<StandardGate>(StandardGate({1, 1, 3, qc::otCU3}))},



    {"cphase", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCP}))},
    {"cp", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCP}))},
    {"cu1", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCP}))},
    {"CPHASE", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCP}))},
    {"CP", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCP}))},
    {"CU1", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCP}))},

    {"id", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otI}))},
    {"i", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otI}))},
    {"I", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otI}))},
    {"u0", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otI}))},

    {"u2", std::make_shared<StandardGate>(StandardGate({0, 1, 2, qc::otU2}))},
    {"U2", std::make_shared<StandardGate>(StandardGate({0, 1, 2, qc::otU2}))},

    {"x", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otX}))},
    {"X", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otX}))},

    {"cx", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCNOT}))},
    {"CX", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCNOT}))},
    {"cnot", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCNOT}))},
    {"CNOT", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCNOT}))},

    {"ccx", std::make_shared<StandardGate>(StandardGate({2, 1, 0, qc::otTOFFOLI}))},
    {"CCX", std::make_shared<StandardGate>(StandardGate({2, 1, 0, qc::otTOFFOLI}))},
    {"toffoli", std::make_shared<StandardGate>(StandardGate({2, 1, 0, qc::otTOFFOLI}))},
    {"TOFFOLI", std::make_shared<StandardGate>(StandardGate({2, 1, 0, qc::otTOFFOLI}))},

    {"ccz", std::make_shared<StandardGate>(StandardGate({2, 1, 0, qc::otCCZ}))},
    {"CCZ", std::make_shared<StandardGate>(StandardGate({2, 1, 0, qc::otCCZ}))},

    {"c4x", std::make_shared<StandardGate>(StandardGate({1, 4, 0, qc::otC4X}))},
    {"C4X", std::make_shared<StandardGate>(StandardGate({1, 4, 0, qc::otC4X}))},

    {"rx", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otRX}))},
    {"RX", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otRX}))},

    {"crx", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCRX}))},
    {"CRX", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCRX}))},

    {"y", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otY}))},
    {"Y", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otY}))},

    {"cy", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCY}))},
    {"CY", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCY}))},

    {"ry", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otRY}))},
    {"RY", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otRY}))},

    {"cry", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCRY}))},
    {"CRY", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCRY}))},

    {"z", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otZ}))},
    {"Z", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otZ}))},

    {"cz", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCZ}))},
    {"CZ", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCZ}))},

    {"rz", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otRZ}))},
     {"RZ", std::make_shared<StandardGate>(StandardGate({0, 1, 1, qc::otRZ}))},

    {"crz", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCRZ}))},
    {"CRZ", std::make_shared<StandardGate>(StandardGate({1, 1, 1, qc::otCRZ}))},

    {"h", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otH}))},
     {"H", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otH}))},

    {"ch", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCH}))},
    {"CH", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCH}))},

    {"cy", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCY}))},
    {"CY", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCY}))},

    {"s", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otS}))},
    {"S", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otS}))},

    {"cs", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCS}))},
    {"CS", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCS}))},

    {"csdg", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCSdg}))},
    {"CSdg", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCSdg}))},

    {"sdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSdg}))},
    {"Sdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSdg}))},
    {"SDG", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSdg}))},

    {"t", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otT}))},
    {"T", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otT}))},

    {"tdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otTdg}))},
    {"Tdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otTdg}))},
    {"TDG", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otTdg}))},

    {"sx", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSX}))},
    { "SX", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSX})) },

    { "csx", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCSX})) },
    { "CSX", std::make_shared<StandardGate>(StandardGate({1, 1, 0, qc::otCSX})) },

    {"sxdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSXdg}))},
    {"SXdg", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSXdg}))},
    {"SXDG", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otSXdg}))},

    {"teleport", std::make_shared<StandardGate>(
                     StandardGate({0, 3, 0, qc::otTeleportation}))},

    {"swap", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::otSWAP}))},
    {"SWAP", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::otSWAP}))},

    {"cswap",std::make_shared<StandardGate>(StandardGate({1, 2, 0, qc::otCSWAP}))},
    { "CSWAP",std::make_shared<StandardGate>(StandardGate({1, 2, 0, qc::otCSWAP})) },

    { "c3sqrtx",std::make_shared<StandardGate>(StandardGate({1, 3, 0, qc::otC3SQRTX})) },
    { "C3SQRTX",std::make_shared<StandardGate>(StandardGate({1, 3, 0, qc::otC3SQRTX})) },

    {"iswap",
     std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::ot_iSWAP}))},
    {"iswapdg",
     std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::ot_iSWAPdg}))},

    {"rxx", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRXX}))},
    {"RXX", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRXX}))},

    {"ryy", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRYY}))},
    {"RYY", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRYY}))},

    {"rzz", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRZZ}))},
    {"RZZ", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRZZ}))},

    {"rzx", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRZX}))},
    {"RZX", std::make_shared<StandardGate>(StandardGate({0, 2, 1, qc::otRZX}))},

    { "RCCX", std::make_shared<StandardGate>(StandardGate({0, 3, 0, qc::otRCCX})) },
    { "rccx", std::make_shared<StandardGate>(StandardGate({0, 3, 0, qc::otRCCX})) },

    { "RC3X", std::make_shared<StandardGate>(StandardGate({0, 4, 0, qc::otRC3X})) },
    { "rc3x", std::make_shared<StandardGate>(StandardGate({0, 4, 0, qc::otRC3X})) },

    { "C3X", std::make_shared<StandardGate>(StandardGate({0, 4, 0, qc::otC3X})) },
    { "c3x", std::make_shared<StandardGate>(StandardGate({0, 4, 0, qc::otC3X})) },

    {"dcx", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::otDCX}))},
    { "DCX", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::otDCX})) },
    {"ecr", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::otECR}))},
    { "ECR", std::make_shared<StandardGate>(StandardGate({0, 2, 0, qc::otECR})) },

    { "R", std::make_shared<StandardGate>(StandardGate({0, 1, 2, qc::otR})) },
    { "r", std::make_shared<StandardGate>(StandardGate({0, 1, 2, qc::otR})) },

    {"xx_minus_yy",
     std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::otXXminusYY}))},
    { "XXMinusYY",
 std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::otXXminusYY})) },
    {"xx_plus_yy",
     std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::otXXplusYY}))},
    { "XXPlusYY",
 std::make_shared<StandardGate>(StandardGate({0, 2, 2, qc::otXXplusYY})) },

    { "V", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otV})) },
     { "v", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otV})) },

    { "W", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otW})) },
    { "w", std::make_shared<StandardGate>(StandardGate({0, 1, 0, qc::otW})) },
};
} // namespace qasm
