
#include "Core/Utilities/Compiler/operations/ClassicControlledOperation.hpp"

namespace qc {

std::string toString(const ComparisonKind& kind) {
  switch (kind) {
  case ComparisonKind::Eq:
    return "==";
  case ComparisonKind::Neq:
    return "!=";
  case ComparisonKind::Lt:
    return "<";
  case ComparisonKind::Leq:
    return "<=";
  case ComparisonKind::Gt:
    return ">";
  case ComparisonKind::Geq:
    return ">=";
  default:
    unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const ComparisonKind& kind) {
  os << toString(kind);
  return os;
}

ComparisonKind getInvertedComparsionKind(const ComparisonKind kind) {
  switch (kind) {
  case Lt:
    return Geq;
  case Leq:
    return Gt;
  case Gt:
    return Leq;
  case Geq:
    return Lt;
  case Eq:
    return Neq;
  case Neq:
    return Eq;
  default:
    unreachable();
  }
}
} // namespace qc
