
#include "Core/Utilities/Compiler/QASMCompiler/parser/Statement.hpp"

namespace qasm {

std::optional<qc::ComparisonKind> getComparisonKind(BinaryExpression::Op op) {
  switch (op) {
  case BinaryExpression::Op::LessThan:
    return qc::ComparisonKind::Lt;
  case BinaryExpression::Op::LessThanOrEqual:
    return qc::ComparisonKind::Leq;
  case BinaryExpression::Op::GreaterThan:
    return qc::ComparisonKind::Gt;
  case BinaryExpression::Op::GreaterThanOrEqual:
    return qc::ComparisonKind::Geq;
  case BinaryExpression::Op::Equal:
    return qc::ComparisonKind::Eq;
  case BinaryExpression::Op::NotEqual:
    return qc::ComparisonKind::Neq;
  default:
    return std::nullopt;
  }
}
} // namespace qasm
