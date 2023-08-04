#pragma once

#include "Core/Utilities/Compiler/Definitions.hpp"

#include <cstddef>
#include <functional>
#include <set>
#include <sstream>
#include <string>

namespace qc {
struct Control {
  enum class Type : bool { Pos = true, Neg = false };

  QBit qubit{};
  Type type = Type::Pos;

  [[nodiscard]] std::string toString() const {
    std::ostringstream oss{};
    oss << "Control(qubit=" << qubit << ", type_=\"";
    if (type == Type::Pos) {
      oss << "Pos";
    } else {
      oss << "Neg";
    }
    oss << "\")";
    return oss.str();
  }

  // Explicitly allow implicit conversion from `QBit` to `Control`
  // NOLINTNEXTLINE(google-explicit-constructor)
  Control(const QBit q = {}, const Type t = Type::Pos) : qubit(q), type(t) {}
};

inline bool operator<(const Control& lhs, const Control& rhs) {
  return lhs.qubit < rhs.qubit ||
         (lhs.qubit == rhs.qubit && lhs.type < rhs.type);
}

inline bool operator==(const Control& lhs, const Control& rhs) {
  return lhs.qubit == rhs.qubit && lhs.type == rhs.type;
}

inline bool operator!=(const Control& lhs, const Control& rhs) {
  return !(lhs == rhs);
}

// this allows a set of controls to be indexed by a `QBit`
struct CompareControl {
  using is_transparent [[maybe_unused]] = void;

  bool operator()(const Control& lhs, const Control& rhs) const {
    return lhs < rhs;
  }

  bool operator()(QBit lhs, const Control& rhs) const {
    return lhs < rhs.qubit;
  }

  bool operator()(const Control& lhs, QBit rhs) const {
    return lhs.qubit < rhs;
  }
};
using Controls = std::set<Control, CompareControl>;

inline namespace literals {
// User-defined literals require unsigned long long int
// NOLINTNEXTLINE(google-runtime-int)
inline Control operator""_pc(unsigned long long int q) {
  return {static_cast<QBit>(q)};
}
// User-defined literals require unsigned long long int
// NOLINTNEXTLINE(google-runtime-int)
inline Control operator""_nc(unsigned long long int q) {
  return {static_cast<QBit>(q), Control::Type::Neg};
}
} // namespace literals
} // namespace qc

namespace std {
template <> struct hash<qc::Control> {
  std::size_t operator()(const qc::Control& c) const {
    return std::hash<qc::QBit>{}(c.qubit) ^
           std::hash<qc::Control::Type>{}(c.type);
  }
};
} // namespace std
