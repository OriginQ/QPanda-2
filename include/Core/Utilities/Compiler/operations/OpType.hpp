#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

namespace qc {
// Natively supported operations of the QFR library
enum OpType : std::uint8_t {
  otNone,
  // Standard Operations
  otGPhase,
  otI,
  otBarrier,
  otH,
  otX,
  otY,
  otZ,
  otS,
  otSdg,
  otT,
  otTdg,
  otV,
  otVdg,
  otU,
  otU2,
  otP,
  otSX,
  otSXdg,
  otRX,
  otRY,
  otRZ,
  otSWAP,
  ot_iSWAP,   // NOLINT (readability-identifier-naming)
  ot_iSWAPdg, // NOLINT (readability-identifier-naming)
  otPeres,
  otPeresdg,
  otDCX,
  otECR,
  otRXX,
  otRYY,
  otRZZ,
  otRZX,
  otXXminusYY,
  otXXplusYY,
  // Compound Operation
  otCompound,
  // Non Unitary Operations
  otMeasure,
  otReset,
  otTeleportation,
  // Classically-controlled Operation
  otClassicControlled,
  // Noise operations
  otATrue,
  otAFalse,
  otMultiATrue,
  otMultiAFalse,
  // Number of OpTypes
  otOpCount,
  /*fj add *************************/
    otCNOT,
    otTOFFOLI,
    otCZ,
    otU3,
    otCU,
    otU1,
    otCH,
    otCRX,
    otCRY,
    otCRZ,
    otRCCX,
    otRC3X,
    otCP,
    otCSWAP,
    otC3X,
    otCY,
    otCSX,
    otC3SQRTX,
    otCU3,
    otC4X,
    otCS,
    otCSdg,
    otCCZ,
    otR,
    otW
    /******************************/
};

inline std::string toString(const OpType& opType) {
  switch (opType) {
  case otNone:
    return "none";
  case otGPhase:
    return "gphase";
  case otI:
    return "i";
  case otH:
    return "h";
  case otX:
    return "x";
  case otY:
    return "y";
  case otZ:
    return "z";
  case otS:
    return "s";
  case otSdg:
    return "sdg";
  case otT:
    return "t";
  case otTdg:
    return "tdg";
  case otV:
    return "v";
  case otVdg:
    return "vdg";
  case otU:
    return "u";
  case otU2:
    return "u2";
  case otP:
    return "p";
  case otSX:
    return "sx";
  case otSXdg:
    return "sxdg";
  case otRX:
    return "rx";
  case otRY:
    return "ry";
  case otRZ:
    return "rz";
  case otSWAP:
    return "swap";
  case ot_iSWAP:
    return "iswap";
  case ot_iSWAPdg:
    return "iswapdg";
  case otPeres:
    return "peres";
  case otPeresdg:
    return "peresdg";
  case otDCX:
    return "dcx";
  case otECR:
    return "ecr";
  case otRXX:
    return "rxx";
  case otRYY:
    return "ryy";
  case otRZZ:
    return "rzz";
  case otRZX:
    return "rzx";
  case otXXminusYY:
    return "xx_minus_yy";
  case otXXplusYY:
    return "xx_plus_yy";
  case otCompound:
    return "compound";
  case otMeasure:
    return "MEASURE";
  case otReset:
    return "RESET";
  case otBarrier:
    return "BARRIER";
  case otTeleportation:
    return "teleportation";
  case otClassicControlled:
    return "classic_controlled";
    //fj add
  case otCNOT:
      return "CNOT";
  case otU3:
      return "U3";
  case otCU:
      return "CU";
  case otCZ:
      return "CZ";
  case otTOFFOLI:
      return "TOFFOLI";
  case otCRX:
      return "CRX";
  case otCRY:
      return "CRY";
  case otCRZ:
      return "CRZ";
  case otRCCX:
      return "RCCX";
  case otRC3X:
      return "RC3X";
  case otCP:
      return "CP";
  case otCSWAP:
      return "CSWAP";
  case otC3X:
      return "C3X";
  case otCH:
      return "CH";
  case otCY:
      return "CY";
  case otCSX:
      return "CSX";
  case otC3SQRTX:
      return "C3SQRTX";
  case otCU3:
      return "CU3";
  case otC4X:
      return "C4X";
  case otCS:
      return "CS";
  case otCSdg:
      return "CSdg";
  case otCCZ:
      return "CCZ";
  case otR:
      return "R";
  case otW:
      return "W";
    // 
  // GCOV_EXCL_START
  default:
    throw std::invalid_argument("Invalid OpType!");
    // GCOV_EXCL_STOP
  }
}

/**
 * @brief Gives a short name for the given OpType (at most 3 characters)
 * @param opType OpType to get the short name for
 * @return Short name for the given OpType
 */
inline std::string shortName(const OpType& opType) {
  switch (opType) {
  case otGPhase:
    return "GPh";
  case otSXdg:
    return "sxd";
  case otSWAP:
    return "sw";
  case ot_iSWAP:
    return "isw";
  case ot_iSWAPdg:
    return "isd";
  case otPeres:
    return "pr";
  case otPeresdg:
    return "prd";
  case otXXminusYY:
    return "x-y";
  case otXXplusYY:
    return "x+y";
  case otBarrier:
    return "====";
  case otMeasure:
    return "msr";
  case otReset:
    return "rst";
  case otTeleportation:
    return "tel";
  default:
    return toString(opType);
  }
}

inline bool isTwoQubitGate(const OpType& opType) {
  switch (opType) {
  case otSWAP:
  case ot_iSWAP:
  case ot_iSWAPdg:
  case otPeres:
  case otPeresdg:
  case otDCX:
  case otECR:
  case otRXX:
  case otRYY:
  case otRZZ:
  case otRZX:
  case otXXminusYY:
  case otXXplusYY:
    return true;
  default:
    return false;
  }
}

inline std::ostream& operator<<(std::ostream& out, OpType& opType) {
  out << toString(opType);
  return out;
}

const inline static std::unordered_map<std::string, qc::OpType>
    OP_NAME_TO_TYPE = {
        {"none", OpType::otNone},
        {"gphase", OpType::otGPhase},
        {"i", OpType::otI},
        {"id", OpType::otI},
        {"h", OpType::otH},
        {"H", OpType::otH},
        {"ch", OpType::otCH},
        {"CH", OpType::otCH},
        {"x", OpType::otX},
        {"cnot", OpType::otCNOT},
        {"cx", OpType::otCNOT},
        {"CNOT", OpType::otCNOT},
        {"CX", OpType::otCNOT},
        {"mcx", OpType::otX},
        {"y", OpType::otY},
        {"cy", OpType::otY},
        {"z", OpType::otZ},
        {"cz", OpType::otZ},
        {"s", OpType::otS},
        {"cs", OpType::otCS},
        {"CS", OpType::otCS},
        {"sdg", OpType::otSdg},
        {"Sdg", OpType::otSdg},
        {"SDG", OpType::otSdg},
        {"csdg", OpType::otCSdg},
        {"CSdg", OpType::otCSdg },
        {"t", OpType::otT},
        {"ct", OpType::otT},
        {"tdg", OpType::otTdg},
        {"Tdg", OpType::otTdg},
        {"TDG", OpType::otTdg},
        {"ctdg", OpType::otTdg},
        {"v", OpType::otV},
        {"vdg", OpType::otVdg},
        {"u", OpType::otU},
        {"cu", OpType::otCU},
        {"u3", OpType::otU},
        {"cu3", OpType::otCU3},
        {"u2", OpType::otU2},
        {"cu2", OpType::otU2},
        {"p", OpType::otP},
        {"cp", OpType::otCP},
        {"cu1", OpType::otCP},
        {"cphase", OpType::otCP},
        {"CP", OpType::otCP},
        {"CU1", OpType::otCP},
        {"CPHASE", OpType::otCP},

        {"mcp", OpType::otP},
        {"phase", OpType::otP},
        {"mcphase", OpType::otP},
        {"u1", OpType::otP},
        {"sx", OpType::otSX},
        {"csx", OpType::otSX},
        {"sxdg", OpType::otSXdg},
        {"SXdg", OpType::otSXdg},
        {"SXDG", OpType::otSXdg},
        {"csxdg", OpType::otSXdg},
        {"rx", OpType::otRX},
        {"crx", OpType::otCRX},
        {"ry", OpType::otRY},
        {"cry", OpType::otCRY},
        {"rz", OpType::otRZ},
        {"crz", OpType::otCRZ},
        {"swap", OpType::otSWAP},
        {"cswap", OpType::otCSWAP},
        {"CSWAP", OpType::otCSWAP},
        {"iswap", OpType::ot_iSWAP},
        {"iswapdg", OpType::ot_iSWAPdg},
        {"peres", OpType::otPeres},
        {"peresdg", OpType::otPeresdg},
        {"dcx", OpType::otDCX},
        {"DCX", OpType::otDCX},
        {"ecr", OpType::otECR},
        { "ECR", OpType::otECR },
        {"rxx", OpType::otRXX},
        {"ryy", OpType::otRYY},
        {"rzz", OpType::otRZZ},
        {"rzx", OpType::otRZX},
        {"rccx",OpType::otRCCX},
        {"RCCX",OpType::otRCCX},
        {"rc3x",OpType::otRC3X},
        {"RC3X",OpType::otRC3X},
        {"xx_minus_yy", OpType::otXXminusYY},
        {"XXminusYY", OpType::otXXminusYY},
        {"XXMinusYY", OpType::otXXminusYY},
        {"xx_plus_yy", OpType::otXXplusYY},
        {"XXPlusYY", OpType::otXXplusYY},
        {"XXplusYY", OpType::otXXplusYY},
        {"measure", OpType::otMeasure},
        {"reset", OpType::otReset},
        {"barrier", OpType::otBarrier},
        {"teleportation", OpType::otTeleportation},
        {"classic_controlled", OpType::otClassicControlled},
        {"compound", OpType::otCompound},
        {"c3x",OpType::otC3X},
        {"C3X",OpType::otC3X},
        {"c4x",OpType::otC4X},
        {"C4X",OpType::otC4X},
        { "CCZ",OpType::otCCZ },
        { "r",OpType::otR },
        { "R",OpType::otR },
        { "W",OpType::otW },
        { "w",OpType::otW },
};

[[nodiscard]] inline OpType opTypeFromString(const std::string& opType) {
  // try to find the operation type in the map of known operation types and
  // return it if found or throw an exception otherwise.
  if (const auto it = OP_NAME_TO_TYPE.find(opType);
      it != OP_NAME_TO_TYPE.end()) {
    return OP_NAME_TO_TYPE.at(opType);
  }
  throw std::invalid_argument("Unsupported operation type: " + opType);
}

inline std::istream& operator>>(std::istream& in, OpType& opType) {
  std::string opTypeStr;
  in >> opTypeStr;

  if (opTypeStr.empty()) {
    in.setstate(std::istream::failbit);
    return in;
  }

  opType = opTypeFromString(opTypeStr);
  return in;
}

} // namespace qc
