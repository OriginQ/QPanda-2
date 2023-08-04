#pragma once

#include "Core/Utilities/Compiler/Definitions.hpp"
#include "Core/Utilities/Compiler/operations/Control.hpp"

#include <cstddef>
#include <functional>
#include <map>

namespace qc {
class Permutation : public std::map<QBit, QBit> {
public:
  [[nodiscard]] inline Controls apply(const Controls& controls) const {
    if (empty()) {
      return controls;
    }
    Controls c{};
    for (const auto& control : controls) {
      c.emplace(at(control.qubit), control.type);
    }
    return c;
  }
  [[nodiscard]] inline Targets apply(const Targets& targets) const {
    if (empty()) {
      return targets;
    }
    Targets t{};
    for (const auto& target : targets) {
      t.emplace_back(at(target));
    }
    return t;
  }
};
} // namespace qc

// define hash function for Permutation
namespace std {
template <> struct hash<qc::Permutation> {
  std::size_t operator()(const qc::Permutation& p) const {
    std::size_t seed = 0;
    for (const auto& [k, v] : p) {
      qc::hashCombine(seed, k);
      qc::hashCombine(seed, v);
    }
    return seed;
  }
};
} // namespace std
