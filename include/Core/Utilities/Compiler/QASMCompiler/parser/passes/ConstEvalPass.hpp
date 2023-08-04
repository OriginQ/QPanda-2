#pragma once

#include "Core/Utilities/Compiler/QASMCompiler/parser/passes/CompilerPass.hpp"
#include "Core/Utilities/Compiler/Definitions.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/Exception.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/NestedEnvironment.hpp"

namespace qasm::const_eval {
struct ConstEvalValue {
  enum Type {
    ConstInt,
    ConstUint,
    ConstFloat,
    ConstBool,
  } type;
  std::variant<int64_t, double, bool> value;
  size_t width;

  explicit ConstEvalValue(double val, const size_t w = 64)
      : type(ConstFloat), value(val), width(w) {}
  explicit ConstEvalValue(int64_t val, const bool isSigned, const size_t w = 64)
      : type(isSigned ? ConstInt : ConstUint), value(val), width(w) {}
  explicit ConstEvalValue(bool val) : type(ConstBool), value(val), width(1) {}

  [[nodiscard]] std::shared_ptr<Constant> toExpr() const {
    switch (type) {
    case ConstInt:
      return std::make_shared<Constant>(Constant(std::get<0>(value), true));
    case ConstUint:
      return std::make_shared<Constant>(Constant(std::get<0>(value), false));
    case ConstFloat:
      return std::make_shared<Constant>(Constant(std::get<1>(value)));
    case ConstBool:
      return std::make_shared<Constant>(
          Constant(static_cast<int64_t>(std::get<2>(value)), false));
    default:
      qc::unreachable();
    }
  }

  bool operator==(const ConstEvalValue& rhs) const {
    if (type != rhs.type) {
      return false;
    }

    switch (type) {
    case ConstInt:
    case ConstUint:
      return std::get<0>(value) == std::get<0>(rhs.value);
    case ConstFloat:
      return std::get<1>(value) == std::get<1>(rhs.value);
    case ConstBool:
      return std::get<2>(value) == std::get<2>(rhs.value);
    }

    return false;
  }

  bool operator!=(const ConstEvalValue& rhs) const { return !(*this == rhs); }

  [[nodiscard]] std::string toString() const {
    std::stringstream ss{};
    switch (type) {
    case ConstInt:
      ss << "ConstInt(" << std::get<0>(value) << ")";
      break;
    case ConstUint:
      ss << "ConstUint(" << std::get<0>(value) << ")";
      break;
    case ConstFloat:
      ss << "ConstFloat(" << std::get<1>(value) << ")";
      break;
    case ConstBool:
      ss << "ConstBool(" << std::get<2>(value) << ")";
      break;
    }

    return ss.str();
  }
};

class ConstEvalPass : public CompilerPass,
                      public DefaultInstVisitor,
                      public ExpressionVisitor<std::optional<ConstEvalValue>>,
                      public TypeVisitor<std::shared_ptr<Expression>> {
  NestedEnvironment<ConstEvalValue> env{};

  template <typename T> static int64_t castToWidth(int64_t value) {
    return static_cast<int64_t>(static_cast<T>(value));
  }

  static ConstEvalValue evalIntExpression(BinaryExpression::Op op, int64_t lhs,
                                          int64_t rhs, size_t width,
                                          bool isSigned);
  static ConstEvalValue evalFloatExpression(BinaryExpression::Op op, double lhs,
                                            double rhs);
  static ConstEvalValue evalBoolExpression(BinaryExpression::Op op, bool lhs,
                                           bool rhs);

public:
  ConstEvalPass() = default;
  ~ConstEvalPass() override = default;

  void addConst(const std::string& identifier, const ConstEvalValue& val) {
    env.emplace(identifier, val);
  }

  void addConst(const std::string& identifier, const double val) {
    env.emplace(identifier, ConstEvalValue(val));
  }

  void processStatement(Statement& statement) override {
    try {
      statement.accept(this);
    } catch (const ConstEvalError& e) {
      throw CompilerError(e.toString(), statement.debugInfo);
    }
  }

  void pushEnv() { env.push(); }
  void popEnv() { env.pop(); }

  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) override;
  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) override;

  std::optional<ConstEvalValue> visitBinaryExpression(
      std::shared_ptr<BinaryExpression> binaryExpression) override;
  std::optional<ConstEvalValue> visitUnaryExpression(
      std::shared_ptr<UnaryExpression> unaryExpression) override;
  std::optional<ConstEvalValue>
  visitConstantExpression(std::shared_ptr<Constant> constant) override;
  std::optional<ConstEvalValue> visitIdentifierExpression(
      std::shared_ptr<IdentifierExpression> identifierExpression) override;
  std::optional<ConstEvalValue>
  visitIdentifierList(std::shared_ptr<IdentifierList> identifierList) override;
  std::optional<ConstEvalValue> visitMeasureExpression(
      std::shared_ptr<MeasureExpression> measureExpression) override;

  std::shared_ptr<ResolvedType> visitDesignatedType(
      DesignatedType<std::shared_ptr<Expression>>* designatedType) override;
  std::shared_ptr<ResolvedType> visitUnsizedType(
      UnsizedType<std::shared_ptr<Expression>>* unsizedType) override;
  std::shared_ptr<ResolvedType>
  visitArrayType(ArrayType<std::shared_ptr<Expression>>* arrayType) override;
};
} // namespace qasm::const_eval
