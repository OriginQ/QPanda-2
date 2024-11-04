#pragma once

#include "Core/Utilities/Compiler/operations/Operation.hpp"

#include <cmath>

namespace qc 
{
class StandardOperation : public Operation {
protected:
  static void checkInteger(fp& ld) {
    const fp nearest = std::nearbyint(ld);
    if (std::abs(ld - nearest) < PARAMETER_TOLERANCE) {
      ld = nearest;
    }
  }

  static void checkFractionPi(fp& ld) {
    const fp div = qcPI / ld;
    const fp nearest = std::nearbyint(div);
    if (std::abs(div - nearest) < PARAMETER_TOLERANCE) {
      ld = qcPI / nearest;
    }
  }

  OpType parseU3(fp& theta, fp& phi, fp& lambda);
  OpType parseU2(fp& phi, fp& lambda);
  OpType parseU1(fp& lambda);

  void checkUgate();
  void setup();

  void dumpOpenQASMTeleportation(std::ostream& of,
                                 const RegisterNames& qreg) const;

public:
  StandardOperation() = default;

  // Standard Constructors
  StandardOperation(QBit target, OpType g, std::vector<fp> params = {});
  StandardOperation(const Targets& targ, OpType g, std::vector<fp> params = {});

  StandardOperation(Control control, QBit target, OpType g,
                    const std::vector<fp>& params = {});
  StandardOperation(Control control, const Targets& targ, OpType g,
                    const std::vector<fp>& params = {});

  StandardOperation(const Controls& c, QBit target, OpType g,
                    const std::vector<fp>& params = {});
  StandardOperation(const Controls& c, const Targets& targ, OpType g,
                    const std::vector<fp>& params = {});

  // MCF (cSWAP), Peres, parameterized two target Constructor
  StandardOperation(const Controls& c, QBit target0, QBit target1, OpType g,
                    const std::vector<fp>& params = {});

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<StandardOperation>(*this);
  }

  [[nodiscard]] bool isStandardOperation() const override { return true; }

  void addControl(const Control c) override {
    if (actsOn(c.qubit)) {
      throw QFRException("Cannot add control on qubit " +
                         std::to_string(c.qubit) +
                         " to operation it already acts on the qubit.");
    }

    controls.emplace(c);
  }

  void clearControls() override { controls.clear(); }

  void removeControl(const Control c) override {
    if (controls.erase(c) == 0) {
      throw QFRException("Cannot remove control on qubit " +
                         std::to_string(c.qubit) +
                         " from operation as it is not a control.");
    }
  }

  Controls::iterator removeControl(const Controls::iterator it) override {
    return controls.erase(it);
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override {
    return Operation::equals(op, perm1, perm2);
  }
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg, size_t indent,
                    bool openQASM3) const override;
  void dumpOriginIR_bak(std::ostream& of, const RegisterNames& qreg,
      const RegisterNames& creg, size_t indent) const;
  void dumpOriginIR(std::ostream& of, const RegisterNames& qreg,
      const RegisterNames& creg, size_t indent) const override;

  void invert() override;

protected:
  void dumpOpenQASM2(std::ostream& of, std::ostringstream& op,
                     const RegisterNames& qreg) const;
  void dumpOpenQASM3(std::ostream& of, std::ostringstream& op,
                     const RegisterNames& qreg) const;

  void dumpGateType(std::ostream& of, std::ostringstream& op,
                    const RegisterNames& qreg) const;
  bool isOrigin1levelCombineGateType() const;
  void dumpOrigin1levelCombineGateType(std::ostream& of) const;
  void dumpOriginGateType(std::ostream& of, std::ostringstream& op,
      const RegisterNames& qreg) const;

  void dumpOriginSpecialGate(std::ostream& of, std::ostringstream& op,
      const RegisterNames& qreg) const;

  void dumpControls(std::ostringstream& op) const;

  //fj add
  //递归方式将指令解析为OriginIR形式
  //递归终止操作：打印控制比特、目标比特、参数列表
 // void dump_controlqs_targetqs_param(std::ostream& of, const Controls& controls,const Targets& targets,const std::vector<fp>& parameters)const;
  void dumpOriginIR_controlqs_targetqs_param(std::ostream& of, const std::vector<std::string>& controls, const std::vector<std::string>& targets, const std::vector<double>& parameters) const;
 //递归终止条件：OriginIR原生支持的门
  void I_dump2originIR(std::ostream& of, std::string tqbit) const;
  void H_dump2originIR(std::ostream& of, std::string tqbit) const;
  void X_dump2originIR(std::ostream& of, std::string tqbit) const;
  void Y_dump2originIR(std::ostream& of, std::string tqbit) const;
  void Z_dump2originIR(std::ostream& of, std::string tqbit) const;
  void S_dump2originIR(std::ostream& of, std::string tqbit) const;
  void T_dump2originIR(std::ostream& of, std::string tqbit) const;
  void CNOT_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit) const;
  void SWAP_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit) const;
  void CZ_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit) const;
  void RX_dump2originIR(std::ostream& of, std::string tqbit, fp theta) const;
  void RY_dump2originIR(std::ostream& of, std::string tqbit, fp theta) const;
  void RZ_dump2originIR(std::ostream& of, std::string tqbit, fp theta) const;
  void P_dump2originIR(std::ostream& of, std::string tqbit, fp theta) const;
  void RXX_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta) const;
  void RYY_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta) const;
  void RZZ_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta) const;
  void RZX_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2, fp theta) const;
  void TOFFOLI_dump2originIR(std::ostream& of, std::string cqbit1, std::string cqbit2, std::string tqbit) const;
  void U2_dump2originIR(std::ostream& of, std::string tqbit, fp phi, fp lambda) const;
  void U3_dump2originIR(std::ostream& of, std::string tqbit, fp theta, fp phi, fp lambda) const;
  void CU_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit, fp theta, fp phi, fp lambda, fp gamma) const;
  //递归生成的门
  void C3X_dump2originIR(std::ostream& of, std::string qbit1, std::string qbit2, std::string qbit3, std::string qbit4) const;
  void SDG_dump2originIR(std::ostream& of, std::string tqbit) const;
  void CP_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit, fp lambda) const;
  void CRX_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit, fp lambda) const;
  void RCCX_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c)const;
  void RC3X_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c, std::string d) const;
  void TDG_dump2originIR(std::ostream& of, std::string tqbit) const;
  void CSWAP_dump2originIR(std::ostream& of, std::string a, std::string b,std::string c)const;
  void CRY_dump2originIR(std::ostream& of, std::string a, std::string b, fp lambda) const;
  void CRZ_dump2originIR(std::ostream& of, std::string a, std::string b, fp lambda) const;
  void SXDG_dump2originIR(std::ostream& of, std::string tqbit) const;
  void CH_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit) const;
  void CY_dump2originIR(std::ostream& of, std::string a, std::string b) const;
  void SX_dump2originIR(std::ostream& of, std::string tqbit) const;
  void CSX_dump2originIR(std::ostream& of, std::string a, std::string b) const;
  void C3SQRTX_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c, std::string d) const;
  void CU3_dump2originIR(std::ostream& of, std::string cqbit, std::string tqbit, fp theta, fp phi, fp lambda) const;
  void C4X_dump2originIR(std::ostream& of, std::string a, std::string b, std::string c, std::string d, std::string e) const;
  void iSWAP_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2) const;
  void DCX_dump2originIR(std::ostream& of, std::string tqbit1, std::string tqbit2) const;
  void CS_dump2originIR(std::ostream& of, std::string a, std::string b) const;
  void CSdg_dump2originIR(std::ostream& of, std::string a, std::string b) const;
  void CCZ_dump2originIR(std::ostream& of, std::string cqbit1, std::string cqbit2, std::string tqbit) const;
  void ECR_dump2originIR(std::ostream& of, std::string a, std::string b) const;
  void R_dump2originIR(std::ostream& of, std::string tqbit, fp theta, fp phi) const;
  void XXMinusYY_dump2originIR(std::ostream& of, std::string a, std::string b, fp theta, fp beta) const;
  void XXPlusYY_dump2originIR(std::ostream& of, std::string a, std::string b, fp theta, fp beta) const;
  void V_dump2originIR(std::ostream& of, std::string tqbit) const;
  void W_dump2originIR(std::ostream& of, std::string tqbit) const;
  void BARRIER_dump2originIR(std::ostream& of, const RegisterNames& qreg) const;
  /*
  //
  */

};

} // namespace qc
