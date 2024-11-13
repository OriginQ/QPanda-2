#pragma once

#include "Core/Utilities/Compiler/operations/Operation.hpp"

namespace qc {

class NonUnitaryOperation final : public Operation {
protected:
  std::vector<Bit> classics{}; // vector for the classical bits to measure into

  static void printMeasurement(std::ostream& os, const std::vector<QBit>& q,
                               const std::vector<Bit>& c,
                               const Permutation& permutation,
                               std::size_t nqubits);
  void printReset(std::ostream& os, const std::vector<QBit>& q,
                  const Permutation& permutation, std::size_t nqubits) const;

public:
  // Measurement constructor
  NonUnitaryOperation(std::vector<QBit> qubitRegister,
                      std::vector<Bit> classicalRegister);
  NonUnitaryOperation(QBit qubit, Bit cbit);

  // General constructor
  explicit NonUnitaryOperation(Targets qubits, OpType op = otReset);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<NonUnitaryOperation>(*this);
  }

  [[nodiscard]] bool isUnitary() const override { return false; }

  [[nodiscard]] bool isNonUnitaryOperation() const override { return true; }

  [[nodiscard]] const std::vector<Bit>& getClassics() const { return classics; }
  std::vector<Bit>& getClassics() { return classics; }
  [[nodiscard]] size_t getNclassics() const { return classics.size(); }

  [[nodiscard]] std::set<QBit> getUsedQubits() const override {
    const auto& opTargets = getTargets();
    return {opTargets.begin(), opTargets.end()};
  }

  [[nodiscard]] const Controls& getControls() const override {
    throw QFRException("Cannot get controls from non-unitary operation.");
  }

  [[nodiscard]] Controls& getControls() override {
    throw QFRException("Cannot get controls from non-unitary operation.");
  }

  void addDepthContribution(std::vector<std::size_t>& depths) const override;

  void addControl(const Control /*c*/) override {
    throw QFRException("Cannot add control to non-unitary operation.");
  }

  void clearControls() override {
    throw QFRException("Cannot clear controls from non-unitary operation.");
  }

  void removeControl(const Control /*c*/) override {
    throw QFRException("Cannot remove controls from non-unitary operation.");
  }

  Controls::iterator removeControl(const Controls::iterator /*it*/) override {
    throw QFRException("Cannot remove controls from non-unitary operation.");
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  std::ostream& print(std::ostream& os, const Permutation& permutation,
                      std::size_t prefixWidth,
                      std::size_t nqubits) const override;

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg, size_t indent,
                    bool openQASM3) const override;

  void MEASURE_dumpOriginIR(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, size_t indent) const;
  void RESET_dumpOriginIR(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, size_t indent) const;
  void BARRIER_dumpOriginIR(std::ostream& of, const RegisterNames& qreg, const RegisterNames& creg, size_t indent) const;
  void dumpOriginIR(std::ostream& of, const RegisterNames& qreg,
      const RegisterNames& creg, size_t indent) const override;

  void invert() override {
    throw QFRException("Inverting a non-unitary operation is not supported.");
  }

  void apply(const Permutation& permutation) override;
};
} // namespace qc

namespace std {
template <> struct hash<qc::NonUnitaryOperation> {
  std::size_t operator()(qc::NonUnitaryOperation const& op) const noexcept {
    std::size_t seed = 0U;
    qc::hashCombine(seed, op.getType());
    for (const auto& q : op.getTargets()) {
      qc::hashCombine(seed, q);
    }
    for (const auto& c : op.getClassics()) {
      qc::hashCombine(seed, c);
    }
    return seed;
  }
};
} // namespace std
