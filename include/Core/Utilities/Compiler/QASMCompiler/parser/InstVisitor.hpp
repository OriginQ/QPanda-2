#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace qasm {
class GateDeclaration;
class GateCallStatement;
class VersionDeclaration;
class DeclarationStatement;
class InitialLayout;
class OutputPermutation;
class AssignmentStatement;
class BarrierStatement;
class ResetStatement;

class Expression;
class BinaryExpression;
class UnaryExpression;
class IdentifierExpression;
class IdentifierList;
class Constant;
class MeasureExpression;
class IfStatement;

class InstVisitor {
public:
  virtual void
  visitGateStatement(std::shared_ptr<GateDeclaration> gateStatement) = 0;
  virtual void visitVersionDeclaration(
      std::shared_ptr<VersionDeclaration> versionDeclaration) = 0;
  virtual void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> declarationStatement) = 0;
  virtual void
  visitInitialLayout(std::shared_ptr<InitialLayout> initialLayout) = 0;
  virtual void visitOutputPermutation(
      std::shared_ptr<OutputPermutation> outputPermutation) = 0;
  virtual void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> gateCallStatement) = 0;
  virtual void visitAssignmentStatement(
      std::shared_ptr<AssignmentStatement> assignmentStatement) = 0;
  virtual void
  visitBarrierStatement(std::shared_ptr<BarrierStatement> barrierStatement) = 0;
  virtual void
  visitResetStatement(std::shared_ptr<ResetStatement> resetStatement) = 0;
  virtual void visitIfStatement(std::shared_ptr<IfStatement> ifStatement) = 0;

  virtual ~InstVisitor() = default;
};

class DefaultInstVisitor : public InstVisitor {
public:
  void visitGateStatement(
      std::shared_ptr<GateDeclaration> /*gateStatement*/) override {}
  void visitVersionDeclaration(
      std::shared_ptr<VersionDeclaration> /*versionDeclaration*/) override {}
  void visitDeclarationStatement(
      std::shared_ptr<DeclarationStatement> /*declarationStatement*/) override {
  }
  void visitInitialLayout(
      std::shared_ptr<InitialLayout> /*initialLayout*/) override {}
  void visitOutputPermutation(
      std::shared_ptr<OutputPermutation> /*outputPermutation*/) override {}
  void visitGateCallStatement(
      std::shared_ptr<GateCallStatement> /*gateCallStatement*/) override {}
  void visitAssignmentStatement(
      std::shared_ptr<AssignmentStatement> /*assignmentStatement*/) override {}
  void visitBarrierStatement(
      std::shared_ptr<BarrierStatement> /*barrierStatement*/) override {}
  void visitResetStatement(
      std::shared_ptr<ResetStatement> /*resetStatement*/) override {}
  void visitIfStatement(std::shared_ptr<IfStatement> /*ifStatement*/) override {
  }
};

template <typename T> class ExpressionVisitor {
public:
  virtual T
  visitBinaryExpression(std::shared_ptr<BinaryExpression> binaryExpression) = 0;
  virtual T
  visitUnaryExpression(std::shared_ptr<UnaryExpression> unaryExpression) = 0;
  virtual T visitConstantExpression(std::shared_ptr<Constant> constant) = 0;
  virtual T visitIdentifierExpression(
      std::shared_ptr<IdentifierExpression> identifierExpression) = 0;
  virtual T
  visitIdentifierList(std::shared_ptr<IdentifierList> identifierList) = 0;
  virtual T visitMeasureExpression(
      std::shared_ptr<MeasureExpression> measureExpression) = 0;

  // A manually implemented visitor function with a templated return type.
  // This is not possible as a virtual function in expression, which is why
  // we define it manually.
  T visit(const std::shared_ptr<Expression>& expression) {
    if (expression == nullptr) {
      throw std::runtime_error("Expression is null");
    }
    if (const auto binaryExpression =
            std::dynamic_pointer_cast<BinaryExpression>(expression)) {
      return visitBinaryExpression(binaryExpression);
    }
    if (const auto unaryExpression =
            std::dynamic_pointer_cast<UnaryExpression>(expression)) {
      return visitUnaryExpression(unaryExpression);
    }
    if (const auto constantInt =
            std::dynamic_pointer_cast<Constant>(expression)) {
      return visitConstantExpression(constantInt);
    }
    if (const auto identifierExpression =
            std::dynamic_pointer_cast<IdentifierExpression>(expression)) {
      return visitIdentifierExpression(identifierExpression);
    }
    if (const auto identifierList =
            std::dynamic_pointer_cast<IdentifierList>(expression)) {
      return visitIdentifierList(identifierList);
    }
    if (const auto measureExpression =
            std::dynamic_pointer_cast<MeasureExpression>(expression)) {
      return visitMeasureExpression(measureExpression);
    }
    throw std::runtime_error("Unhandled expression type.");
  }

  virtual ~ExpressionVisitor() = default;
};

template <typename T> class Type;
using ResolvedType = Type<uint64_t>;
template <typename T> class DesignatedType;
template <typename T> class UnsizedType;
template <typename T> class ArrayType;

template <typename T> class TypeVisitor {
public:
  virtual ~TypeVisitor() = default;

  virtual std::shared_ptr<ResolvedType>
  visitDesignatedType(DesignatedType<T>* designatedType) = 0;
  virtual std::shared_ptr<ResolvedType>
  visitUnsizedType(UnsizedType<T>* unsizedType) = 0;
  virtual std::shared_ptr<ResolvedType>
  visitArrayType(ArrayType<T>* arrayType) = 0;
};
} // namespace qasm
