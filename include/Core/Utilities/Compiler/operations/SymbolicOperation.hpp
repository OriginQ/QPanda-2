#pragma once

#include "Core/Utilities/Compiler/Definitions.hpp"
#include "Core/Utilities/Compiler/operations/Expression.hpp"
#include "Core/Utilities/Compiler/Permutation.hpp"
#include "Core/Utilities/Compiler/operations/StandardOperation.hpp"
#include "Core/Utilities/Compiler/operations/Control.hpp"
#include "Core/Utilities/Compiler/operations/OpType.hpp"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <variant>
#include <vector>

namespace qc {

// Overload pattern for std::visit
template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Overload(Ts...) -> Overload<Ts...>;

class SymbolicOperation final : public StandardOperation {
protected:
  std::vector<std::optional<Symbolic>> symbolicParameter{};

  static OpType parseU3(const Symbolic& theta, fp& phi, fp& lambda);
  static OpType parseU3(fp& theta, const Symbolic& phi, fp& lambda);
  static OpType parseU3(fp& theta, fp& phi, const Symbolic& lambda);
  static OpType parseU3(const Symbolic& theta, const Symbolic& phi, fp& lambda);
  static OpType parseU3(const Symbolic& theta, fp& phi, const Symbolic& lambda);
  static OpType parseU3(fp& theta, const Symbolic& phi, const Symbolic& lambda);

  static OpType parseU2(const Symbolic& phi, const Symbolic& lambda);
  static OpType parseU2(const Symbolic& phi, fp& lambda);
  static OpType parseU2(fp& phi, const Symbolic& lambda);

  static OpType parseU1(const Symbolic& lambda);

  void checkSymbolicUgate();

  void storeSymbolOrNumber(const SymbolOrNumber& param, std::size_t i);

  [[nodiscard]] bool isSymbolicParameter(const std::size_t i) const {
    return symbolicParameter.at(i).has_value();
  }

  static bool isSymbol(const SymbolOrNumber& param) {
    return std::holds_alternative<Symbolic>(param);
  }

  static Symbolic& getSymbol(SymbolOrNumber& param) {
    return std::get<Symbolic>(param);
  }

  static fp& getNumber(SymbolOrNumber& param) { return std::get<fp>(param); }

  void setup(const std::vector<SymbolOrNumber>& params);

  [[nodiscard]] static fp
  getInstantiation(const SymbolOrNumber& symOrNum,
                   const VariableAssignment& assignment);

  void negateSymbolicParameter(std::size_t index);

  void addToSymbolicParameter(std::size_t index, fp value);

public:
  SymbolicOperation() = default;

  [[nodiscard]] SymbolOrNumber getParameter(const std::size_t i) const {
    const auto& param = symbolicParameter.at(i);
    if (param.has_value()) {
      return *param;
    }
    return parameter.at(i);
  }

  [[nodiscard]] std::vector<SymbolOrNumber> getParameters() const {
    std::vector<SymbolOrNumber> params{};
    for (std::size_t i = 0; i < parameter.size(); ++i) {
      params.emplace_back(getParameter(i));
    }
    return params;
  }

  void setSymbolicParameter(const Symbolic& par, const std::size_t i) {
    symbolicParameter.at(i) = par;
  }

  // Standard Constructors
  SymbolicOperation(QBit target, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});
  SymbolicOperation(const Targets& targ, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  SymbolicOperation(Control control, QBit target, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});
  SymbolicOperation(Control control, const Targets& targ, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  SymbolicOperation(const Controls& c, QBit target, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});
  SymbolicOperation(const Controls& c, const Targets& targ, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  // MCF (cSWAP), Peres, parameterized two target Constructor
  SymbolicOperation(const Controls& c, QBit target0, QBit target1, OpType g,
                    const std::vector<SymbolOrNumber>& params = {});

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<SymbolicOperation>(*this);
  }

  [[nodiscard]] inline bool isSymbolicOperation() const override {
    return std::any_of(symbolicParameter.begin(), symbolicParameter.end(),
                       [](const auto& sym) { return sym.has_value(); });
  }

  [[nodiscard]] inline bool isStandardOperation() const override {
    return std::all_of(symbolicParameter.begin(), symbolicParameter.end(),
                       [](const auto& sym) { return !sym.has_value(); });
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] inline bool equals(const Operation& op) const override {
    return equals(op, {}, {});
  }

  [[noreturn]] void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                                 const RegisterNames& creg, size_t indent,
                                 bool openQASM3) const override;

  [[noreturn]] void dumpOriginIR(std::ostream& of, const RegisterNames& qreg,
      const RegisterNames& creg, size_t indent) const override
  {
      throw QFRException("OriginIR doesn't support parametrized gates!");
  };

  [[nodiscard]] StandardOperation
  getInstantiatedOperation(const VariableAssignment& assignment) const;

  // Instantiates this Operation
  // Afterwards casting to StandardOperation can be done if assignment is total
  void instantiate(const VariableAssignment& assignment);

  void invert() override;
};
} // namespace qc

namespace std {
template <> struct hash<qc::SymbolicOperation> {
  std::size_t operator()(qc::SymbolicOperation const& op) const noexcept {
    std::size_t seed = 0U;
    qc::hashCombine(seed, std::hash<qc::Operation>{}(op));
    for (const auto& param : op.getParameters()) {
      if (std::holds_alternative<qc::fp>(param)) {
        qc::hashCombine(seed, hash<qc::fp>{}(get<qc::fp>(param)));
      } else {
        qc::hashCombine(seed, hash<qc::Symbolic>{}(get<qc::Symbolic>(param)));
      }
    }
    return seed;
  }
};
} // namespace std
