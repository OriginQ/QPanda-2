
#include "Core/Utilities/Compiler/operations/StandardOperation.hpp"
#include <algorithm>
#include <cassert>
#include <sstream>
#include <variant>

namespace qc {
/***
 * Protected Methods
 ***/
OpType StandardOperation::parseU3(fp& theta, fp& phi, fp& lambda) {
  if (std::abs(theta) < PARAMETER_TOLERANCE &&
      std::abs(phi) < PARAMETER_TOLERANCE) {
    parameter = {lambda};
    return parseU1(parameter[0]);
  }

  if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
    parameter = {phi, lambda};
    return parseU2(parameter[0], parameter[1]);
  }

  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    lambda = 0.L;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      checkInteger(theta);
      checkFractionPi(theta);
      parameter = {theta};
      return otRY;
    }
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
    if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
      checkInteger(theta);
      checkFractionPi(theta);
      parameter = {theta};
      return otRX;
    }

    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      phi = PI_2;
      if (std::abs(theta - qcPI) < PARAMETER_TOLERANCE) {
        parameter.clear();
        return otY;
      }
    }
  }

  if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
    lambda = -PI_2;
    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      phi = PI_2;
      parameter = {-theta};
      return otRX;
    }
  }

  if (std::abs(lambda - qcPI) < PARAMETER_TOLERANCE) {
    lambda = qcPI;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      phi = 0.L;
      if (std::abs(theta - qcPI) < PARAMETER_TOLERANCE) {
        parameter.clear();
        return otX;
      }
    }
  }

  // parse a real u3 gate
  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(phi);
  checkFractionPi(phi);
  checkInteger(theta);
  checkFractionPi(theta);

  return otU;
}

OpType StandardOperation::parseU2(fp& phi, fp& lambda) {
  if (std::abs(phi) < PARAMETER_TOLERANCE) {
    phi = 0.L;
    if (std::abs(std::abs(lambda) - qcPI) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return otH;
    }
    if (std::abs(lambda) < PARAMETER_TOLERANCE) {
      parameter = {PI_2};
      return otRY;
    }
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
    if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return otV;
    }
  }

  if (std::abs(lambda + PI_2) < PARAMETER_TOLERANCE) {
    lambda = -PI_2;
    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      parameter.clear();
      return otVdg;
    }
  }

  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(phi);
  checkFractionPi(phi);

  return otU2;
}

OpType StandardOperation::parseU1(fp& lambda) {
  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return otI;
  }
  const bool sign = std::signbit(lambda);

  if (std::abs(std::abs(lambda) - qcPI) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return otZ;
  }

  if (std::abs(std::abs(lambda) - PI_2) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return sign ? otSdg : otS;
  }

  if (std::abs(std::abs(lambda) - PI_4) < PARAMETER_TOLERANCE) {
    parameter.clear();
    return sign ? otTdg : otT;
  }

  checkInteger(lambda);
  checkFractionPi(lambda);

  return otP;
}

void StandardOperation::checkUgate() {
  if (parameter.empty()) {
    return;
  }
  if (type == otP) {
    assert(parameter.size() == 1);
    type = parseU1(parameter.at(0));
  } else if (type == otU2) {
    assert(parameter.size() == 2);
    type = parseU2(parameter.at(0), parameter.at(1));
  } else if (type == otU) {
    assert(parameter.size() == 3);
    type = parseU3(parameter.at(0), parameter.at(1), parameter.at(2));
  }
}

void StandardOperation::setup() {
  checkUgate();
  name = toString(type);
}

/***
 * Constructors
 ***/
StandardOperation::StandardOperation(const QBit target, const OpType g,
                                     std::vector<fp> params) {
  type = g;
  parameter = std::move(params);
  setup();
  targets.emplace_back(target);
}

StandardOperation::StandardOperation(const Targets& targ, const OpType g,
                                     std::vector<fp> params) {
  type = g;
  parameter = std::move(params);
  setup();
  targets = targ;
}

StandardOperation::StandardOperation(const Control control, const QBit target,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(target, g, params) {
  controls.insert(control);
}

StandardOperation::StandardOperation(const Control control, const Targets& targ,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(targ, g, params) {
  controls.insert(control);
}

StandardOperation::StandardOperation(const Controls& c, const QBit target,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(target, g, params) {
  controls = c;
}

StandardOperation::StandardOperation(const Controls& c, const Targets& targ,
                                     const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(targ, g, params) {
  controls = c;
}

// MCF (cSWAP), Peres, parameterized two target Constructor
StandardOperation::StandardOperation(const Controls& c, const QBit target0,
                                     const QBit target1, const OpType g,
                                     const std::vector<fp>& params)
    : StandardOperation(c, {target0, target1}, g, params) {}

/***
 * Public Methods
 ***/
void StandardOperation::dumpOpenQASM(std::ostream& of,
                                     const RegisterNames& qreg,
                                     [[maybe_unused]] const RegisterNames& creg,
                                     size_t indent, bool openQASM3) const {
  std::ostringstream op;
  op << std::setprecision(std::numeric_limits<fp>::digits10);

  op << std::string(indent * OUTPUT_INDENT_SIZE, ' ');

  if (openQASM3) {
    dumpOpenQASM3(of, op, qreg);
  } else {
    dumpOpenQASM2(of, op, qreg);
  }
}

void checkControlFlags(const Controls& controls) {

}

void StandardOperation::dumpOriginIR_bak(std::ostream& of,
    const RegisterNames& qreg,
    [[maybe_unused]] const RegisterNames& creg,
    size_t indent) const 
{
    std::ostringstream op;
    op << std::setprecision(std::numeric_limits<fp>::digits10);
    op << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
    
    if ((controls.size() > 1 && type != otX) || controls.size() > 2) 
    {
        std::cout << "[WARNING] Multiple controlled gates are not natively "
            "supported by OpenQASM. "
            << "However, this library can parse .qasm files with multiple "
            "controlled gates (e.g., cccx) correctly. "
            << "Thus, while not valid vanilla OpenQASM, the dumped file will "
            "work with this library.\n";
    }

    // safe the numbers of controls as a prefix to the operation name
    //op << std::string(controls.size(), 'c');
    /*fj 240724注释 
    if (!controls.empty())
    {
        of << "CONTROL ";
        for (auto it = controls.begin(); it != controls.end();) 
        {
            std::cout << "Debug: StandardOperation::dumpOriginIR for() contorls" << std::endl;
            std::cout << "### it:" << it->toString() << std::endl;
            of << qreg[it->qubit].second;
           // std::cout<< qreg[it->qubit].second;
            if (++it != controls.end()) {
                std::cout << ",";
                of << ",";
            }
           
            
            
        }
        of << "\n";

    }
    
    const bool isSpecialGate = 
        type == Peres || type == Peresdg || type == Teleportation;

    if (!isSpecialGate) 
    {
        // apply X operations to negate the respective controls
        for (const auto& c : controls) {
            if (c.type == Control::Type::Neg)
                of << "X " << qreg[c.qubit].second << "\n";
        }
    }
    */
    dumpOriginGateType(of, op, qreg);

    /*fj 240724注释
    if (!isSpecialGate) 
    {
        // apply X operations to negate the respective controls again
        for (const auto& c : controls) {
            if (c.type == Control::Type::Neg)
                of << "X " << qreg[c.qubit].second << "\n";
        }
    }
    */
    /*fj 240724注释
    if (!controls.empty())
    {
        of << "ENDCONTROL ";
        for (auto it = controls.begin(); it != controls.end();)
        {
            of << qreg[it->qubit].second;
            if (++it != controls.end())
                of << ",";
        }
        of << "\n";
    }
    */
    return;
}


void StandardOperation::dumpOpenQASM2(std::ostream& of, std::ostringstream& op,
                                      const RegisterNames& qreg) const {
  if ((controls.size() > 1 && type != otX) || controls.size() > 2) {
    std::cout << "[WARNING] Multiple controlled gates are not natively "
                 "supported by OpenQASM. "
              << "However, this library can parse .qasm files with multiple "
                 "controlled gates (e.g., cccx) correctly. "
              << "Thus, while not valid vanilla OpenQASM, the dumped file will "
                 "work with this library.\n";
  }

  // safe the numbers of controls as a prefix to the operation name
  op << std::string(controls.size(), 'c');

  const bool isSpecialGate = type == otPeres || type == otPeresdg || type == otTeleportation;

  if (!isSpecialGate) {
    // apply X operations to negate the respective controls
    for (const auto& c : controls) {
      if (c.type == Control::Type::Neg) {
        of << "x " << qreg[c.qubit].second << ";\n";
      }
    }
  }

  dumpGateType(of, op, qreg);

  if (!isSpecialGate) {
    // apply X operations to negate the respective controls again
    for (const auto& c : controls) {
      if (c.type == Control::Type::Neg) {
        of << "x " << qreg[c.qubit].second << ";\n";
      }
    }
  }
}

void StandardOperation::dumpOpenQASM3(std::ostream& of, std::ostringstream& op,
                                      const RegisterNames& qreg) const {
  dumpControls(op);

  dumpGateType(of, op, qreg);
}

void StandardOperation::dumpGateType(std::ostream& of, std::ostringstream& op,
                                     const RegisterNames& qreg) const {
  // Dump the operation name and parameters.
  switch (type) {
  case otGPhase:
    op << "gphase(" << parameter.at(0) << ")";
    break;
  case otI:
    op << "id";
    break;
  case otBarrier:
    assert(controls.empty());
    op << "barrier";
    break;
  case otH:
    op << "H";
    break;
  case otX:
    op << "X";
    break;
  case otY:
    op << "Y";
    break;
  case otZ:
    op << "Z";
    break;
  case otRCCX:
    op << "RCCX";
    break;
  case otRC3X:
      op << "RC3X";
      break;
  case otS:
    if (!controls.empty()) {
      op << "p(pi/2)";
    } else {
      op << "s";
    }
    break;
  case otSdg:
    if (!controls.empty()) {
      op << "p(-pi/2)";
    } else {
      op << "sdg";
    }
    break;
  case otT:
    if (!controls.empty()) {
      op << "p(pi/4)";
    } else {
      op << "t";
    }
    break;
  case otTdg:
    if (!controls.empty()) {
      op << "p(-pi/4)";
    } else {
      op << "tdg";
    }
    break;
  case otV:
    op << "U(pi/2,-pi/2,pi/2)";
    break;
  case otVdg:
    op << "U(pi/2,pi/2,-pi/2)";
    break;
  case otU:
    op << "U(" << parameter[0] << "," << parameter[1] << "," << parameter[2]
       << ")";
    break;
  case otU2:
    op << "U(pi/2," << parameter[0] << "," << parameter[1] << ")";
    break;
  case otP:
    op << "p(" << parameter[0] << ")";
    break;
  case otSX:
    op << "sx";
    break;
  case otSXdg:
    op << "sxdg";
    break;
  case otRX:
    op << "rx(" << parameter[0] << ")";
    break;
  case otRY:
    op << "ry(" << parameter[0] << ")";
    break;
  case otRZ:
    op << "rz(" << parameter[0] << ")";
    break;
  case otDCX:
    op << "dcx";
    break;
  case otECR:
    op << "ecr";
    break;
  case otRXX:
    op << "rxx(" << parameter[0] << ")";
    break;
  case otRYY:
    op << "ryy(" << parameter[0] << ")";
    break;
  case otRZZ:
    op << "rzz(" << parameter[0] << ")";
    break;
  case otRZX:
    op << "rzx(" << parameter[0] << ")";
    break;
  case otXXminusYY:
    op << "xx_minus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case otXXplusYY:
    op << "xx_plus_yy(" << parameter[0] << "," << parameter[1] << ")";
    break;
  case otSWAP:
    op << "swap";
    break;
  case ot_iSWAP:
    op << "iswap";
    break;
  case ot_iSWAPdg:
    op << "iswapdg";
    break;
  case otPeres:
    of << op.str() << "cx";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ", " << qreg[targets[0]].second
       << ";\n";

    of << op.str() << "x";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ";\n";
    return;
  case otPeresdg:
    of << op.str() << "x";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ";\n";

    of << op.str() << "cx";
    for (const auto& c : controls) {
      of << " " << qreg[c.qubit].second << ",";
    }
    of << " " << qreg[targets[1]].second << ", " << qreg[targets[0]].second
       << ";\n";
    return;
  case otTeleportation:
    dumpOpenQASMTeleportation(of, qreg);
    return;
  default:
    std::cerr << "gate type " << toString(type)
              << " could not be converted to OpenQASM\n.";
  }

  // apply the operation
  of << op.str();

  // First print control qubits.
  for (auto it = controls.begin(); it != controls.end();) {
    of << " " << qreg[it->qubit].second;
    // we only print a comma if there are more controls or targets.
    if (++it != controls.end() || !targets.empty()) {
      of << ",";
    }
  }
  // Print target qubits.
  if (!targets.empty() && type == otBarrier &&
      isWholeQubitRegister(qreg, targets.front(), targets.back())) {
    of << " " << qreg[targets.front()].first;
  } else {
    for (auto it = targets.begin(); it != targets.end();) {
      of << " " << qreg[*it].second;
      // only print comma if there are more targets
      if (++it != targets.end()) {
        of << ",";
      }
    }
  }
  of << ";\n";
}

bool StandardOperation::isOrigin1levelCombineGateType() const {
    switch (type) {
    case otCRX:
    case otCRY:
    case otCRZ:
    case otRCCX:
    case otRC3X:
    case otSXdg:
    case otCP:
    case otSdg:
    case otTdg:
    case otCSWAP:
    case otC3X:
        return true;
    default:
        return false;
    }
}

void StandardOperation::dumpOrigin1levelCombineGateType(std::ostream& of) const {
    of << "QGATE ";
    switch (type) {
    case otCRX:
        of << "CRX a,b,(lambda)\n";
        of << "P b,(PI/2)\n";
        of << "CNOT a,b\n";
        of << "U3 b,(-lambda/2,0,0)\n";
        of << "CNOT a,b\n";
        of << "U3 b,(lambda/2,-PI/2,0)\n";
        break;
    case otCRY:
        of << "CRY a,b,(lambda)\n";
        of << "RY b,(lambda/2)\n";
        of << "CNOT a,b\n";
        of << "RY b,(-lambda/2)\n";
        of << "CNOT a,b\n";
        break;
    case otCRZ:
        of << "CRZ a,b,(lambda)\n";
        of << "RZ b,(lambda/2)\n";
        of << "CNOT a,b\n";
        of << "RZ b,(-lambda/2)\n";
        of << "CNOT a,b\n";
        break;
    case otRCCX:
        of << "RCCX a,b,c\n";
        of << "U2 c,(0,PI)\n";
        of << "P c,(PI/4)\n";
        of << "CNOT b,c\n";
        of << "U1 c,(-PI/4)\n";
        of << "CNOT a,c\n";
        of << "P c,(PI/4)\n";
        of << "CNOT b,c\n";
        of << "P c,(-PI/4)\n";
        of << "U2 c,(0,PI)\n";
        break;
    case otRC3X:
        of << "RC3X a,b,c,d\n";
        of << "U2 d,(0,PI)\n";
        of << "P d,(PI/4)\n";
        of << "CNOT c,d\n";
        of << "P d,(-PI/4)\n";
        of << "U2 d,(0,PI)\n";
        of << "CNOT a,d\n";
        of << "P d,(PI/4)\n";
        of << "CNOT b,d\n";
        of << "P d,(-PI/4)\n";
        of << "CNOT a,d\n";
        of << "P d,(PI/4)\n";
        of << "CNOT b,d\n";
        of << "P d,(-PI/4)\n";
        of << "U2 d,(0,PI)\n";
        of << "P d,(PI/4)\n";
        of << "CNOT c,d\n";
        of << "P d,(-PI/4)\n";
        of << "U2 d,(0,PI)\n";
        break;
    case otSXdg:
        of << "SXDG a\n";
        of << "S a\n";
        of << "H a\n";
        of << "S a\n";
        break;
    case otCP:
        of << "CP a,b,(lambda)\n";
        of << "P a,(lambda/2)\n";
        of << "CNOT a,b\n";
        of << "P b,(-lambda/2)\n";
        of << "CNOT a,b\n";
        of << "P b,(lambda/2)\n";
        break;
    case otSdg:
        of << "SDG a\n";
        of << "P a,(-PI/2)\n";
        break;
    case otTdg:
        of << "TDG a\n";
        of << "P a,(-PI/4)\n";
        break;
    case otCSWAP:
        of << "CSWAP a,b,c\n";
        of << "CNOT c,b\n";
        of << "TOFFOLI a,b,c\n";
        of << "CNOT c,b\n";
        break;
    case otC3X:
        of << "C3X a,b,c,d\n";
        of << "H d\n";
        of << "P a,(PI/8)\n";
        of << "P b,(PI/8)\n";
        of << "P c,(PI/8)\n";
        of<<"P d,(PI/8)\n";
        of<<"CNOT a,b\n";
        of<<"P b,(-PI/8)\n";
        of<<"CNOT a,b\n";
        of<<"CNOT b,c\n";
        of<<"P c,(-PI/8)\n";
        of<<"CNOT a,c\n";
        of<<"P c,(PI/8)\n";
        of<<"CNOT b,c\n";
        of<<"P c,(-PI/8)\n";
        of<<"CNOT a,c\n";
        of<<"CNOT c,d\n";
        of<<"P d,(-PI/8)\n";
        of<<"CNOT b,d\n";
        of<<"P d,(PI/8)\n";
        of<<"CNOT c,d\n";
        of<<"P d,(-PI/8)\n";
        of<<"CNOT a,d\n";
        of<<"P d,(PI/8)\n";
        of<<"CNOT c,d\n";
        of<<"P d,(-PI/8)\n";
        of<<"CNOT b,d\n";
        of<<"P d,(PI/8)\n";
        of<<"CNOT c,d\n";
        of<<"P d,(-PI/8)\n";
        of<<"CNOT a,d\n";
        of << "H d\n";
        break;
    default:
        std::cerr << "StandardOperation::dumpOrigin1levelCombineGateType\n";
    }
    of << "ENDQGATE\n";
}

void StandardOperation::dumpOriginGateType(std::ostream& of, std::ostringstream& op,
    const RegisterNames& qreg) const 
{
    bool is_dagger = false;

    // Dump the operation name and parameters.
    switch (type) {
    case otGPhase:
        op << "GPHASE(udef)" ;
        break;
    case otI:
        op << "I";
        break;
    case otBarrier:
        assert(controls.empty());
        op << "BARRIER";
        break;
    case otH:
        op << "H";
        break;
    case otX:
        op << "X";
        break;
        //fj 240724 add
    case otCNOT:
        op << "CNOT";
        break;
    case otTOFFOLI:
        op << "TOFFOLI";
        break;
        //fj 240724 add//////////////////
    case otY:
        op << "Y";
        break;
    case otZ:
        op << "Z";
        break;
    case otCZ://fj 240724 add
        op << "CZ";
        break;
    case otS:
        op << "S";
        break;
    case otSdg:
        is_dagger = true;
        // fj 240724 注释//////////
        //op << "DAGGER\n";
        //op << "S";
        /////////////
        //fj 240724 add
        op << "SDG";
        break;
    case otT:
        op << "T";
        break;
    case otTdg:
        is_dagger = true;
        //fj 240724注释
//        op << "DAGGER\n";
//        op << "T";
        ////////
        op << "TDG";// fj 240724 add
        break;
    case otV:
       // op << "U";
        op << "V(Udef)";
        break;
    case otVdg:
        //op << "U";
        op << "Vdg(udef)";
        break;
    case otU:
        //op << "U" ;
        op << "U3";
        break;
    case otU2:
        //op << "U" ;
        op << "U2";
        break;
    case otU3:
        op << "U3";
        break;
    case otCU:
        op << "CU";
        break;
    case otP:
        op << "P";
        break;
    case otU1:
        op << "U1";
        break;
    case otSX:
        //fj 240724注释 op << "SX";
        op << "SX(udef)";
        break;
    case otSXdg:
        is_dagger = true;
        /*fj 240724注释
        op << "SXDG";
        */
        op << "SXDG";
        break;
    case otRX:
        op << "RX";
        break;
    case otCRX:
        op << "CRX";
        break;
    case otRY:
        op << "RY";
        break;
    case otCRY:
        op << "CRY";
        break;
    case otRZ:
        op << "RZ";
        break;
    case otCRZ:
        op << "CRZ";
        break;
    case otDCX:
        op << "DCX(Udef)";
        break;
    case otECR:
        op << "ECR(Udef)";
        break;
    case otRXX:
        op << "RXX";
        break;
    case otRYY:
        op << "RYY";
        break;
    case otRZZ:
        op << "RZZ";
        break;
    case otRZX:
        op << "RZX";
        break;
    case otRCCX:
        op << "RCCX";
        break;
    case otRC3X:
        op << "RC3X";
        break;
    case otXXminusYY:
        op << "XX_MINUS_YY(udef)";
        break;
    case otXXplusYY:
        op << "XX_PLUS_YY(udef)";
        break;
    case otSWAP:
        op << "SWAP";
        break;
    case ot_iSWAP:
        op << "ISWAP(udef)";
        break;
    case otCP:
        op << "CP";
        break;
    case ot_iSWAPdg:
        is_dagger = true;
        /*fj 240724 注释
        op << "DAGGER\n";
        op << "ISWAP";
        */
        op << "iSWAPdg(udef)";
        break;
        //case Peres:
        //    break;
        //case Peresdg:
        //    break;
        //case Teleportation:
        //    dumpOpenQASMTeleportation(of, qreg);
        //    return;
    case otCSWAP:
        op << "CSWAP";
        break;
    case otC3X:
        op << "C3X";
        break;
    default:
        std::cout << "StandardOperation::dumpOriginGateType" << std::endl;
        std::cerr << "gate type " << toString(type)
            << " could not be converted to OriginIR\n.";
    }

    // apply the operation
    of << op.str();

    //print control qubits
    if (controls.size() > 0) {
        of << " ";
    }
    for (auto& ite:controls) {
        of << qreg[ite.qubit].second << ",";
    }

    // Print target qubits.
    if (!targets.empty() && type == otBarrier &&
        isWholeQubitRegister(qreg, targets.front(), targets.back())) 
    {
        of << " " << qreg[targets.front()].first;
    }
    else 
    {
        for (auto it = targets.begin(); it != targets.end();) 
        {
            of << " " << qreg[*it].second;
            if (++it != targets.end())
                of << ",";
        }
    }
    

    if (parameter.size() > 0)
        of << ",";

    // Print parameter
    switch (type) {
    case otGPhase:
        of << "(" << parameter.at(0) << ")";
        break;
    case otI:
    case otBarrier:
    case otH:
    case otX:
    case otY:
    case otZ:
    case otS:
    case otSdg:
    case otT:
    case otCNOT:
    case otCZ:
    case otRCCX:
    case otRC3X:
    case otTOFFOLI:
    case otSXdg:
    case otTdg:
    case otCSWAP:
    case otC3X:
        break;
    case otV:
        of << "(pi/2,-pi/2,pi/2)";
        break;
    case otVdg:
        of << "(pi/2,pi/2,-pi/2)";
        break;
    case otU3:
        of << "(" << parameter[0] << "," << parameter[1] << "," << parameter[2]
            << ")";
        break;
    case otCU:
        of << "(" << parameter[0] << "," << parameter[1] << "," << parameter[2]
            << ","<<parameter[3]<<")";
        break;
    case otU2:
        //of << "(pi/2," << parameter[0] << "," << parameter[1] << ")";
        of << "(" << parameter[0] << "," << parameter[1] << ")";
        break;
    case otP:
    case otCP:
        of << "(" << parameter[0] << ")";
        break;
    case otSX:
        break;
    case otRX:
    case otRY:
    case otRZ:
        of << "(" << parameter[0] << ")";
        break;
    case otDCX:
        break;
    case otECR:
        break;
    case otRXX:
    case otRYY:
    case otRZZ:
    case otRZX:
    case otCRX:
    case otCRY:
    case otCRZ:
        of << "(" << parameter[0] << ")";
        break;
    case otXXminusYY:
    case otXXplusYY:
        of << "(" << parameter[0] << "," << parameter[1] << ")";
        break;
    case otSWAP:
    case ot_iSWAP:
    case ot_iSWAPdg:
        break;
    case otPeres:
    case otPeresdg:
    case otTeleportation:
        return;
    default:
        std::cout <<  "StandardOperation::dumpOriginGateType //print paramerters"  << std::endl;
        std::cerr << "gate type " << toString(type)
            << " could not be converted to originir\n.";
    }

   
    /*fj 240724 注释
    if (is_dagger)
        of << "\nENDDAGGER\n";
    else
        of << "\n";
    */
    of << "\n";//fj 240724add 

    return;
}

void StandardOperation::dumpOriginSpecialGate(std::ostream& of, std::ostringstream& op,
    const RegisterNames& qreg) const
{

}

void StandardOperation::dumpOpenQASMTeleportation(
    std::ostream& of, const RegisterNames& qreg) const {
  if (!controls.empty() || targets.size() != 3) {
    std::cerr << "controls = ";
    for (const auto& c : controls) {
      std::cerr << qreg.at(c.qubit).second << " ";
    }
    std::cerr << "\ntargets = ";
    for (const auto& t : targets) {
      std::cerr << qreg.at(t).second << " ";
    }
    std::cerr << "\n";

    throw QFRException("Teleportation needs three targets");
  }
  /*
                                      ░      ┌───┐ ░ ┌─┐    ░
                  |ψ⟩ q_0: ───────────░───■──┤ H ├─░─┤M├────░─────────────── |0⟩
     or |1⟩ ┌───┐      ░ ┌─┴─┐└───┘ ░ └╥┘┌─┐ ░ |0⟩ a_0: ┤ H ├──■───░─┤ X
     ├──────░──╫─┤M├─░─────────────── |0⟩ or |1⟩ └───┘┌─┴─┐ ░ └───┘      ░  ║
     └╥┘ ░  ┌───┐  ┌───┐ |0⟩ a_1: ─────┤ X ├─░────────────░──╫──╫──░──┤ X ├──┤ Z
     ├─ |ψ⟩ └───┘ ░            ░  ║  ║  ░  └─┬─┘  └─┬─┘ ║  ║    ┌──┴──┐   │
                bitflip: 1/═══════════════════════════╩══╬════╡ = 1 ╞═══╪═══
                                                      0  ║    └─────┘┌──┴──┐
              phaseflip: 1/══════════════════════════════╩═══════════╡ = 1 ╞
                                                         0           └─────┘
          */
  of << "// teleport q_0, a_0, a_1; q_0 --> a_1  via a_0\n";
  of << "teleport " << qreg[targets[0]].second << ", "
     << qreg[targets[1]].second << ", " << qreg[targets[2]].second << ";\n";
}

void StandardOperation::invert() {
  switch (type) {
  // self-inverting gates
  case otI:
  case otX:
  case otY:
  case otZ:
  case otH:
  case otSWAP:
  case otECR:
  case otBarrier:
    break;
  // gates where we just update parameters
  case otGPhase:
  case otP:
  case otRX:
  case otRY:
  case otRZ:
  case otRXX:
  case otRYY:
  case otRZZ:
  case otRZX:
    parameter[0] = -parameter[0];
    break;
  case otU2:
    std::swap(parameter[0], parameter[1]);
    parameter[0] = -parameter[0] + qcPI;
    parameter[1] = -parameter[1] - qcPI;
    break;
  case otU:
    parameter[0] = -parameter[0];
    parameter[1] = -parameter[1];
    parameter[2] = -parameter[2];
    std::swap(parameter[1], parameter[2]);
    break;
  case otXXminusYY:
  case otXXplusYY:
    parameter[0] = -parameter[0];
    break;
  case otDCX:
    std::swap(targets[0], targets[1]);
    break;
  // gates where we have specialized inverted operation types
  case otS:
    type = otSdg;
    break;
  case otSdg:
    type = otS;
    break;
  case otT:
    type = otTdg;
    break;
  case otTdg:
    type = otT;
    break;
  case otV:
    type = otVdg;
    break;
  case otVdg:
    type = otV;
    break;
  case otSX:
    type = otSXdg;
    break;
  case otSXdg:
    type = otSX;
    break;
  case otPeres:
    type = otPeresdg;
    break;
  case otPeresdg:
    type = otPeres;
    break;
  case ot_iSWAP:
    type = ot_iSWAPdg;
    break;
  case ot_iSWAPdg:
    type = ot_iSWAP;
    break;
  case otNone:
  //case ot_Compound:
  //case ot_Measure:
  case otReset:
  case otTeleportation:
  case otClassicControlled:
  case otATrue:
  case otAFalse:
  case otMultiATrue:
  case otMultiAFalse:
  case otOpCount:
    throw QFRException("Inverting gate" + toString(type) +
                       " is not supported.");
  }
}

void StandardOperation::dumpControls(std::ostringstream& op) const {
  if (controls.empty()) {
    return;
  }

  // if operation is in stdgates.inc, we print a c prefix instead of ctrl @
  if (bool printBuiltin = std::none_of(
          controls.begin(), controls.end(),
          [](const Control& c) { return c.type == Control::Type::Neg; });
      printBuiltin) {
    const auto numControls = controls.size();
    switch (type) {
    case otP:
    case otRX:
    case otY:
    case otRY:
    case otZ:
    case otRZ:
    case otH:
    case otSWAP:
      printBuiltin = numControls == 1;
      break;
    case otX:
      printBuiltin = numControls == 1 || numControls == 2;
      break;
    default:
      printBuiltin = false;
    }
    if (printBuiltin) {
      op << std::string(numControls, 'c');
      return;
    }
  }

  Control::Type currentType = controls.begin()->type;
  int count = 0;

  for (const auto& control : controls) {
    if (control.type == currentType) {
      ++count;
    } else {
      op << (currentType == Control::Type::Neg ? "negctrl" : "ctrl");
      if (count > 1) {
        op << "(" << count << ")";
      }
      op << " @ ";
      currentType = control.type;
      count = 1;
    }
  }

  op << (currentType == Control::Type::Neg ? "negctrl" : "ctrl");
  if (count > 1) {
    op << "(" << count << ")";
  }
  op << " @ ";
}

void StandardOperation::dumpOriginIR_controlqs_targetqs_param(
    std::ostream& of, const std::vector<std::string> &controls, 
    const std::vector<std::string>& targets, const std::vector<double>& parameters)const{

    if (controls.size() > 0) {
        of << controls[0];
    }
    for(int i=1;i<controls.size();i++){
        of << ","<<controls[i];
    }
    if (controls.size() > 0) {
        of << ",";
    }
    
    if (targets.size() > 0) {
        of << targets[0];
    }
    for (int i = 1; i < targets.size(); i++) {
        of << "," << targets[i];
    }
    
    if (!parameters.empty()) {
        of << ",(" << parameters[0];
    }
    for (int i = 1; i < parameters.size(); i++) {
        of << "," << parameters[i];
    }
    if (!parameters.empty()) {
        of << ")";
    }
    of << "\n";
}
void StandardOperation::I_dump2originIR(std::ostream& of,std::string tqbit)const {
    of << "I" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit }, {});
}
void StandardOperation::H_dump2originIR(std::ostream& of, std::string tqbit)const {
    of << "H" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit }, {});
}
void StandardOperation::X_dump2originIR(std::ostream& of, std::string tqbit)const {
    of << "X" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit }, {});
}
void StandardOperation::Y_dump2originIR(std::ostream& of, std::string tqbit)const {
    of << "Y" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit }, {});
}
void StandardOperation::Z_dump2originIR(std::ostream& of, std::string tqbit)const {
    of << "Z" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit }, {});
}
void StandardOperation::S_dump2originIR(std::ostream& of, std::string tqbit)const {
    of << "S" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit }, {});
}
void StandardOperation::T_dump2originIR(std::ostream& of, std::string tqbit)const {
    of << "T" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit }, {});
}
void StandardOperation::CNOT_dump2originIR(std::ostream& of,std::string cqbit, std::string tqbit)const {
    of << "CNOT" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {cqbit}, { tqbit }, {});
}
void StandardOperation::SWAP_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2)const {
    of << "SWAP" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, {}, { tqbit1,tqbit2 }, {});
}
void StandardOperation::CZ_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit)const {
    of << "CZ" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { cqbit }, { tqbit }, {});
}
void StandardOperation::RX_dump2originIR(std::ostream& of, std::string tqbit, fp theta)const {
    of << "RX" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, {tqbit }, {theta});
}
void StandardOperation::RY_dump2originIR(std::ostream& of, std::string tqbit, fp theta)const {
    of << "RY" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit }, { theta });
}
void StandardOperation::RZ_dump2originIR(std::ostream& of, std::string tqbit, fp theta)const {
    of << "RZ" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit }, { theta });
}
void StandardOperation::P_dump2originIR(std::ostream& of, std::string tqbit, fp theta)const {
    of << "P" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit }, { theta });
}
void StandardOperation::RXX_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta)const {
    of << "RXX" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit1,tqbit2 }, { theta });
}
void StandardOperation::RYY_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta)const {
    of << "RYY" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit1,tqbit2 }, { theta });
}
void StandardOperation::RZZ_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta)const {
    of << "RZZ" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit1,tqbit2 }, { theta });
}
void StandardOperation::RZX_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta)const {
    of << "RZX" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit1,tqbit2 }, { theta });
}
void StandardOperation::TOFFOLI_dump2originIR(std::ostream& of,std::string cqbit1,std::string cqbit2,std::string tqbit)const {
    of << "TOFFOLI" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { cqbit1,cqbit2}, { tqbit}, {});
}

void StandardOperation::U2_dump2originIR(std::ostream& of, std::string tqbit, fp phi, fp lambda)const {
    of << "U2" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit }, {phi,lambda});
}
void StandardOperation::U3_dump2originIR(std::ostream& of, std::string tqbit, fp theta, fp phi, fp lambda)const {
    of << "U3" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { }, { tqbit }, { theta,phi,lambda });
}

void StandardOperation::CU_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit, fp a, fp b, fp c,fp d) const {
    of << "CU" << " ";
    dumpOriginIR_controlqs_targetqs_param(of, { cqbit}, { tqbit }, { d + (b + c) / 2,b,a,c });
}

void StandardOperation::C3X_dump2originIR(std::ostream& of, std::string a, std::string b,std::string c,std::string d) const {
    /*
        h d;
    p(pi/8) a;
    p(pi/8) b;
    p(pi/8) c;
    p(pi/8) d;
    cx a, b;
    p(-pi/8) b;
    cx a, b;
    cx b, c;
    p(-pi/8) c;
    cx a, c;
    p(pi/8) c;
    cx b, c;
    p(-pi/8) c;
    cx a, c;
    cx c, d;
    p(-pi/8) d;
    cx b, d;
    p(pi/8) d;
    cx c, d;
    p(-pi/8) d;
    cx a, d;
    p(pi/8) d;
    cx c, d;
    p(-pi/8) d;
    cx b, d;
    p(pi/8) d;
    cx c, d;
    p(-pi/8) d;
    cx a, d;
    h d;

    */
    H_dump2originIR(of,d);
    P_dump2originIR(of, a, qcPI / 8);
    P_dump2originIR(of, b, qcPI / 8);
    P_dump2originIR(of, c, qcPI / 8);
    P_dump2originIR(of, d, qcPI / 8);
    CNOT_dump2originIR(of, a, b);
    P_dump2originIR(of, b, -qcPI / 8);
    CNOT_dump2originIR(of,a, b);
    CNOT_dump2originIR(of, b, c);
    P_dump2originIR(of, c, -qcPI / 8);
    CNOT_dump2originIR(of, a, c);
    P_dump2originIR(of, c, qcPI / 8);
    CNOT_dump2originIR(of, b, c);
    P_dump2originIR(of, c, -qcPI / 8);
    CNOT_dump2originIR(of, a, c);
    CNOT_dump2originIR(of, c, d);
    P_dump2originIR(of, d, -qcPI / 8);
    CNOT_dump2originIR(of, b, d);
    P_dump2originIR(of, d, qcPI / 8);
    CNOT_dump2originIR(of, c, d);
    P_dump2originIR(of, d, -qcPI / 8);
    CNOT_dump2originIR(of, a, d);
    P_dump2originIR(of, d, qcPI / 8);
    CNOT_dump2originIR(of, c, d);
    P_dump2originIR(of, d, -qcPI / 8);
    CNOT_dump2originIR(of, b, d);
    P_dump2originIR(of, d, qcPI / 8);
    CNOT_dump2originIR(of, c, d);
    P_dump2originIR(of, d, -qcPI / 8);
    CNOT_dump2originIR(of, a, d);
    H_dump2originIR(of, d); 
}

void StandardOperation::SDG_dump2originIR(std::ostream& of, std::string tqbit)const {
    /*
    gate sdg a { u1(-pi/2) a; }
    */
    P_dump2originIR(of, tqbit, -qcPI / 2);
}

void StandardOperation::CP_dump2originIR(std::ostream& of, std::string a, std::string b, fp lambda) const {
    /*
    gate cu1(lambda) a,b
{
  u1(lambda/2) a;
  cx a,b;
  u1(-lambda/2) b;
  cx a,b;
  u1(lambda/2) b;
}
    */
    P_dump2originIR(of, a, lambda / 2);
    CNOT_dump2originIR(of, a, b);
    P_dump2originIR(of, b, -lambda / 2);
    CNOT_dump2originIR(of, a, b);
    P_dump2originIR(of, b, lambda / 2);
}

void StandardOperation::CRX_dump2originIR(std::ostream& of, std::string a, std::string b, fp lambda) const {
  /*
  gate crx(lambda) a,b
{
  u1(pi/2) b;
  cx a,b;
  u3(-lambda/2,0,0) b;
  cx a,b;
  u3(lambda/2,-pi/2,0) b;
}
  */
    P_dump2originIR(of, b, qcPI/2);
    CNOT_dump2originIR(of, a, b);
    U3_dump2originIR(of, b, -lambda / 2, 0, 0);
    CNOT_dump2originIR(of, a, b);
    U3_dump2originIR(of, b, lambda / 2, -qcPI/2, 0);
}

void StandardOperation::RCCX_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c)const {
    /*
     u2(0,pi) c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  cx a, c;
  u1(pi/4) c;
  cx b, c;
  u1(-pi/4) c;
  u2(0,pi) c;
    */
    U2_dump2originIR(of, c, 0, qcPI);
    P_dump2originIR(of, c, qcPI / 4);
    CNOT_dump2originIR(of, b, c);
    P_dump2originIR(of, c, -qcPI / 4);
    CNOT_dump2originIR(of, a, c);
    P_dump2originIR(of, c, qcPI / 4);
    CNOT_dump2originIR(of, b, c);
    P_dump2originIR(of, c, -qcPI / 4);
    U2_dump2originIR(of, c, 0, qcPI);
}

void StandardOperation::RC3X_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c,std::string d)const {
    /*

    */
    U2_dump2originIR(of, d, 0, qcPI);// u2(0, pi) d;
    P_dump2originIR(of, d, qcPI / 4);// u1(pi / 4) d;
    CNOT_dump2originIR(of, c, d);// cx c, d;
    P_dump2originIR(of, d, -qcPI / 4);// u1(-pi / 4) d;
    U2_dump2originIR(of, d, 0, qcPI);// u2(0, pi) d;
    CNOT_dump2originIR(of, a, d);// cx a, d;
    P_dump2originIR(of,d, qcPI / 4);// u1(pi / 4) d;
    CNOT_dump2originIR(of, b, d);// cx b, d;
    P_dump2originIR(of,d, -qcPI / 4);// u1(-pi / 4) d;
    CNOT_dump2originIR(of, a, d);// cx a, d;
    P_dump2originIR(of, d, qcPI / 4);// u1(pi / 4) d;
    CNOT_dump2originIR(of, b, d);// cx b, d;
    P_dump2originIR(of, d, -qcPI / 4);// u1(-pi / 4) d;
    U2_dump2originIR(of, d, 0, qcPI);// u2(0, pi) d;
    P_dump2originIR(of, d, qcPI / 4);// u1(pi / 4) d;
    CNOT_dump2originIR(of, c, d);// cx c, d;
    P_dump2originIR(of, d, -qcPI / 4);// u1(-pi / 4) d;
    U2_dump2originIR(of, d, 0, qcPI);// u2(0, pi) d;
}

void StandardOperation::TDG_dump2originIR(std::ostream& of, std::string tqbit)const {
   //u1(-pi/4) a;
    P_dump2originIR(of, tqbit, -qcPI / 4);

}

void StandardOperation::CSWAP_dump2originIR(std::ostream& of, std::string a, std::string b,std::string c)const {
    CNOT_dump2originIR(of, c, b);//cx c, b;
    TOFFOLI_dump2originIR(of, a, b, c);//ccx a, b, c;
    CNOT_dump2originIR(of, c, b);//cx c, b;
}

void StandardOperation::CRY_dump2originIR(std::ostream& of, std::string a, std::string b, fp lambda) const {
    RY_dump2originIR(of, b, lambda / 2);//ry(lambda / 2) b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    RY_dump2originIR(of, b, -lambda / 2);//ry(-lambda / 2) b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
}

void StandardOperation::CRZ_dump2originIR(std::ostream& of, std::string a, std::string b, fp lambda) const {
    RZ_dump2originIR(of, b, lambda / 2);//rz(lambda / 2) b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    RZ_dump2originIR(of, b, -lambda / 2);//rz(-lambda / 2) b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
}

void StandardOperation::SXDG_dump2originIR(std::ostream& of, std::string tqbit)const {
    S_dump2originIR(of, tqbit);// s a; 
    H_dump2originIR(of, tqbit);// h a; 
    S_dump2originIR(of, tqbit);// s a;
}

void StandardOperation::CH_dump2originIR(std::ostream& of, std::string a, std::string b)const {
    H_dump2originIR(of, b);//h b;
    SDG_dump2originIR(of, b);// sdg b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    H_dump2originIR(of, b);//h b; 
    T_dump2originIR(of, b);// t b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    T_dump2originIR(of, b);//t b;
    H_dump2originIR(of, b);// h b;
    S_dump2originIR(of, b);// s b;
    X_dump2originIR(of, b);// x b; 
    S_dump2originIR(of, a);// s a;
}

void StandardOperation::CY_dump2originIR(std::ostream& of, std::string a, std::string b)const {
    SDG_dump2originIR(of, b);//sdg b; 
    CNOT_dump2originIR(of, a, b);// cx a, b; 
    S_dump2originIR(of, b);// s b;
}

void StandardOperation::SX_dump2originIR(std::ostream& of, std::string a)const {
    SDG_dump2originIR(of, a);//sdg a; 
    H_dump2originIR(of, a);// h a; 
    SDG_dump2originIR(of, a);// sdg a;
}

void StandardOperation::CSX_dump2originIR(std::ostream& of, std::string a, std::string b)const {
    H_dump2originIR(of, b);//h b; 
    CP_dump2originIR(of, a, b, qcPI / 2);// cu1(pi / 2) a, b; 
    H_dump2originIR(of, b);// h b;
}

void StandardOperation::C3SQRTX_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c,std::string d)const {
    H_dump2originIR(of, d);//h d; 
    CP_dump2originIR(of, a, d, qcPI / 8);// cu1(pi / 8) a, d; 
    H_dump2originIR(of, d);// h d;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    H_dump2originIR(of, d);//h d; 
    CP_dump2originIR(of, b, d, -qcPI / 8);// cu1(-pi / 8) b, d; 
    H_dump2originIR(of, d);// h d;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    H_dump2originIR(of, d);//h d; 
    CP_dump2originIR(of, b, d, qcPI / 8);// cu1(pi / 8) b, d; 
    H_dump2originIR(of, d);// h d;
    CNOT_dump2originIR(of, b, c);//cx b, c;
    H_dump2originIR(of, d);//h d; 
    CP_dump2originIR(of, c, d, -qcPI / 8);// cu1(-pi / 8) c, d; 
    H_dump2originIR(of, d);// h d;
    CNOT_dump2originIR(of, a, c);//cx a, c;
    H_dump2originIR(of, d);//h d; 
    CP_dump2originIR(of, c, d, qcPI / 8);// cu1(pi / 8) c, d; 
    H_dump2originIR(of, d);// h d;
    CNOT_dump2originIR(of, b, c);//cx b, c;
    H_dump2originIR(of, d);//h d; 
    CP_dump2originIR(of, c, d, -qcPI / 8);// cu1(-pi / 8) c, d; 
    H_dump2originIR(of, d);// h d;
    CNOT_dump2originIR(of, a, c);//cx a, c;
    H_dump2originIR(of, d);//h d; 
    CP_dump2originIR(of, c, d, qcPI / 8);// cu1(pi / 8) c, d; 
    H_dump2originIR(of, d);// h d;
}

void StandardOperation::CU3_dump2originIR(std::ostream& of, std::string c, std::string t, fp theta, fp phi, fp lambda) const {
    P_dump2originIR(of, c, (lambda+phi) / 2);//u1((lambda + phi) / 2) c;
    P_dump2originIR(of, t, (lambda-phi) / 2);//u1((lambda - phi) / 2) t;
    CNOT_dump2originIR(of, c, t);//cx c, t;
    U3_dump2originIR(of, t, -theta / 2, 0, -(phi + lambda) / 2);//u3(-theta / 2, 0, -(phi + lambda) / 2) t;
    CNOT_dump2originIR(of, c, t);//cx c, t;
    U3_dump2originIR(of, t, theta / 2, phi, 0);// u3(theta / 2, phi, 0) t;
}

void StandardOperation::C4X_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c, std::string d,std::string e) const {
    H_dump2originIR(of, e);//h e;
    CP_dump2originIR(of, d, e, qcPI / 2);// cu1(pi / 2) d, e;
    H_dump2originIR(of, e);// h e;
    C3X_dump2originIR(of,a, b, c, d);//c3x a, b, c, d;
    H_dump2originIR(of, e);//h e;
    CP_dump2originIR(of, d, e, -qcPI / 2);// cu1(-pi / 2) d, e;
    H_dump2originIR(of, e);// h e;
    C3X_dump2originIR(of, a, b, c, d);//c3x a, b, c, d;
    C3SQRTX_dump2originIR(of, a, b, c, e);//c3sqrtx a, b, c, e;
}

void StandardOperation::iSWAP_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2)const {
    S_dump2originIR(of, tqbit1);
    S_dump2originIR(of, tqbit2);
    H_dump2originIR(of, tqbit1);
    CNOT_dump2originIR(of, tqbit1, tqbit2);
    CNOT_dump2originIR(of, tqbit2, tqbit1);
    H_dump2originIR(of,tqbit2);
}

void StandardOperation::DCX_dump2originIR(std::ostream& of, std::string a, std::string b)const {
    //gate dcx a, b { cx a, b; cx b, a; }
    CNOT_dump2originIR(of,a,b);
    CNOT_dump2originIR(of,b, a);
}

void StandardOperation::CS_dump2originIR(std::ostream& of, std::string a, std::string b)const {
   //gate cs a,b { h b; cp(pi/2) a,b; h b; }
    H_dump2originIR(of, b);
    CP_dump2originIR(of,  a, b,qcPI/2);
    H_dump2originIR(of, b);
}

void StandardOperation::CSdg_dump2originIR(std::ostream& of, std::string a, std::string b)const {
    // gate csdg a,b { h b; cp(-pi/2) a,b; h b; }
    H_dump2originIR(of, b);
    CP_dump2originIR(of, a, b, -qcPI / 2);
    H_dump2originIR(of, b);
    
}

void StandardOperation::CCZ_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c)const {
   //gate ccz a,b,c { h c; ccx a,b,c; h c; }
    H_dump2originIR(of, c);
    TOFFOLI_dump2originIR(of, a, b, c);
    H_dump2originIR(of, c);

}

void StandardOperation::ECR_dump2originIR(std::ostream& of, std::string a, std::string b)const {
    //gate ecr a, b { 
    // rzx(pi/4) a, b; 
    // x a;
    // rzx(-pi/4) a, b;}
    RZX_dump2originIR(of, a, b, qcPI / 4);
    X_dump2originIR(of, a);
    RZX_dump2originIR(of, a, b, -qcPI / 4);
   
}

void StandardOperation::R_dump2originIR(std::ostream& of, std::string tqbit,fp theta, fp phi)const {
    //gate r(θ, φ) a {
    // u3(θ, φ - π/2, -φ + π/2) a;}
    U3_dump2originIR(of, tqbit, theta, phi - qcPI / 2, qcPI / 2 - phi);
    
}


void StandardOperation::XXMinusYY_dump2originIR(std::ostream& of, std::string a, std::string b, fp theta, fp beta) const {
    RZ_dump2originIR(of, b, -beta);//rz(-beta) b;
    RZ_dump2originIR(of, a, -qcPI / 2);//rz(-pi / 2) a;
    SX_dump2originIR(of, a);//sx a;
    RZ_dump2originIR(of, a, qcPI / 2);//rz(pi / 2) a;
    S_dump2originIR(of, b);//s b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    RY_dump2originIR(of, a, theta / 2);// ry(theta / 2) a;
    RY_dump2originIR(of, b, -theta / 2);//ry(-theta / 2) b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    SDG_dump2originIR(of, b);//sdg b;
    RZ_dump2originIR(of, a, -qcPI / 2);//rz(-pi / 2) a;
    SXDG_dump2originIR(of, a);//sxdg a;
    RZ_dump2originIR(of, a, qcPI / 2);//rz(pi / 2) a;
    RZ_dump2originIR(of, b, beta);//rz(beta) b;
}

void StandardOperation::XXPlusYY_dump2originIR(std::ostream& of, std::string a, std::string b, fp theta, fp beta) const {
    //xx_plus_yy(theta, beta) a, b
    RZ_dump2originIR(of, b, beta);//rz(beta) b;
    RZ_dump2originIR(of, a, -qcPI / 2);//rz(-pi / 2) a;
    SX_dump2originIR(of, a);//sx a;
    RZ_dump2originIR(of, a, qcPI / 2);//rz(pi / 2) a;
    S_dump2originIR(of, b);//s b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    RY_dump2originIR(of, a, theta / 2);//ry(theta / 2) a;
    RY_dump2originIR(of, b, theta / 2);//ry(theta / 2) b;
    CNOT_dump2originIR(of, a, b);//cx a, b;
    SDG_dump2originIR(of, b);//sdg b;
    RZ_dump2originIR(of, a, -qcPI / 2);//rz(-pi / 2) a;
    SXDG_dump2originIR(of, a);//sxdg a;
    RZ_dump2originIR(of, a, qcPI / 2);//rz(pi / 2) a;
    RZ_dump2originIR(of, b, -beta);//rz(-beta) b;
}

void StandardOperation::V_dump2originIR(std::ostream& of, std::string tqbit)const {
    SDG_dump2originIR(of, tqbit);
    H_dump2originIR(of,tqbit);
}

void StandardOperation::W_dump2originIR(std::ostream& of, std::string tqbit)const {
    H_dump2originIR(of, tqbit);
    S_dump2originIR(of, tqbit);
}

void StandardOperation::BARRIER_dump2originIR(std::ostream& of, const RegisterNames& qreg)const {
    of << "BARRIER"<<" ";
    if (targets.size() == 0) {
        for (auto i = 0; i < qreg.size();i++) {
            if (i > 0) {
                of << ",";
            }
            of << qreg[i].second;
        }
    }
    else {
        for(auto i=0;i<targets.size();i++){
            std::stringstream ss;
            if (i > 0) {
                of << ",";
            }
            of << qreg[targets.at(i)].second;
        }
    }
    of << "\n";
}

void StandardOperation::dumpOriginIR(std::ostream& of,
    const RegisterNames& qreg,
    [[maybe_unused]] const RegisterNames& creg,
    size_t indent) const
{
    std::ostringstream op;
    op << std::setprecision(std::numeric_limits<fp>::digits10);
    op << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
    std::vector<std::string> cqbit_list;
    switch (type) {
    case otI:
        I_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otH:
        H_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otX:
        X_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otY:
        Y_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otZ:
        Z_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otS:
        S_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otT:
        T_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otCNOT:
        for (auto&ite: controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        
        CNOT_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second);
        break;
    case otCZ:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CZ_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second);
        break;
    case otSWAP:
        SWAP_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second);
        break;
    case otRX:
        RX_dump2originIR(of, qreg[targets[0]].second, parameter[0]);
        break;
    case otRY:
        RY_dump2originIR(of, qreg[targets[0]].second, parameter[0]);
        break;
    case otRZ:
        RZ_dump2originIR(of, qreg[targets[0]].second, parameter[0]);
        break;
    case otP:
        P_dump2originIR(of, qreg[targets[0]].second, parameter[0]);
        break;
    case otRXX:
        RXX_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, parameter[0]);
        break;
    case otRYY:
        RYY_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, parameter[0]);
        break;
    case otRZZ:
        RZZ_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, parameter[0]);
        break;
    case otRZX:
        RZX_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, parameter[0]);
        break;
    case otTOFFOLI:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        TOFFOLI_dump2originIR(of,cqbit_list[0], cqbit_list[1], qreg[targets[0]].second);
        break;
    case otU2:
        U2_dump2originIR(of, qreg[targets[0]].second, parameter[0], parameter[1]);
        break;
    case otU3:
        U3_dump2originIR(of, qreg[targets[0]].second, parameter[0], parameter[1], parameter[2]);
        break;
    case otCU:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CU_dump2originIR(of,cqbit_list[0], qreg[targets[0]].second, parameter[0], parameter[1], parameter[2], parameter[3]);
        break;
    case otCU3:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CU3_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second, parameter[0], parameter[1], parameter[2]);
        break;
    case otC3X:
        C3X_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, qreg[targets[2]].second, qreg[targets[3]].second);
        break;
    case otSdg:
        SDG_dump2originIR(of,qreg[targets[0]].second);
        break;
    case otCP:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CP_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second, parameter[0]);
        break;
    case otCRX:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CRX_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second, parameter[0]);
        break;
    case otRCCX:
        RCCX_dump2originIR(of,qreg[targets[0]].second, qreg[targets[1]].second, qreg[targets[2]].second);
        break;
    case otRC3X:
        RC3X_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, qreg[targets[2]].second, qreg[targets[3]].second);
        break;
    case otTdg:
        TDG_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otCSWAP:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CSWAP_dump2originIR(of, cqbit_list[0],qreg[targets[0]].second, qreg[targets[1]].second);
        break;
    case otCRY:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CRY_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second, parameter[0]);
        break;
    case otCRZ:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CRZ_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second, parameter[0]);
        break;
    case otSXdg:
        SXDG_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otCH:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CH_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second);
        break;
    case otCY:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CY_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second);
        break;
    case otSX:
        SX_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otCSX:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CSX_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second);
        break;
    case otC3SQRTX:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        C3SQRTX_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second, qreg[targets[1]].second, qreg[targets[2]].second);
        break;
    case otC4X:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        C4X_dump2originIR(of,cqbit_list[0], qreg[targets[0]].second, qreg[targets[1]].second, qreg[targets[2]].second, qreg[targets[3]].second);
        break;
    case ot_iSWAP:
        iSWAP_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second);
        break;
    case otDCX:
        DCX_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second);
        break;
    case otCS:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CS_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second);
        break;
    case otCSdg:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CSdg_dump2originIR(of, cqbit_list[0], qreg[targets[0]].second);
        break;
    case otCCZ:
        for (auto& ite : controls) {
            cqbit_list.push_back(qreg[ite.qubit].second);
        }
        CCZ_dump2originIR(of, cqbit_list[0], cqbit_list[1], qreg[targets[0]].second);
        break;
    case otECR:
        ECR_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second);
        break;
    case otR:
        R_dump2originIR(of, qreg[targets[0]].second, parameter[0], parameter[1]);
        break;
    case otXXminusYY:
        XXMinusYY_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, parameter[0], parameter[1]);
        break;
    case otXXplusYY:
        XXPlusYY_dump2originIR(of, qreg[targets[0]].second, qreg[targets[1]].second, parameter[0], parameter[1]);
        break;
    case otV:
        V_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otW:
        W_dump2originIR(of, qreg[targets[0]].second);
        break;
    case otBarrier:
        BARRIER_dump2originIR(of, qreg);
        break;
    default:
        std::cerr << "Error:StandardOperation::dumpOriginIR" << std::endl;
    }
    return;
}
} // namespace qc
