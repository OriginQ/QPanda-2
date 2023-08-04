/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */


#include "Core/Utilities/Compiler/operations/NonUnitaryOperation.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

namespace qc {
// Measurement constructor
NonUnitaryOperation::NonUnitaryOperation(std::vector<QBit> qubitRegister,
                                         std::vector<Bit> classicalRegister)
    : classics(std::move(classicalRegister)) {
  type = otMeasure;
  targets = std::move(qubitRegister);
  name = toString(type);
  if (targets.size() != classics.size()) {
    throw std::invalid_argument(
        "Sizes of qubit register and classical register do not match.");
  }
}
NonUnitaryOperation::NonUnitaryOperation(const QBit qubit, const Bit cbit)
    : classics({cbit}) {
  type = otMeasure;
  targets = {qubit};
  name = toString(type);
}

// General constructor
NonUnitaryOperation::NonUnitaryOperation(Targets qubits, OpType op) {
  type = op;
  targets = std::move(qubits);
  std::sort(targets.begin(), targets.end());
  name = toString(type);
}

std::ostream&
NonUnitaryOperation::print(std::ostream& os, const Permutation& permutation,
                           [[maybe_unused]] const std::size_t prefixWidth,
                           const std::size_t nqubits) const {
  switch (type) {
  case otMeasure:
    printMeasurement(os, targets, classics, permutation, nqubits);
    break;
  case otReset:
    printReset(os, targets, permutation, nqubits);
    break;
  default:
    break;
  }
  return os;
}

void NonUnitaryOperation::dumpOpenQASM(std::ostream& of,
                                       const RegisterNames& qreg,
                                       const RegisterNames& creg, size_t indent,
                                       bool openQASM3) const {
  of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');

  if (isWholeQubitRegister(qreg, targets.front(), targets.back()) &&
      (type != otMeasure ||
       isWholeQubitRegister(creg, classics.front(), classics.back()))) {
    if (type == otMeasure && openQASM3) {
      of << creg[classics.front()].first << " = ";
    }
    of << toString(type) << " " << qreg[targets.front()].first;
    if (type == otMeasure && !openQASM3) {
      of << " -> ";
      of << creg[classics.front()].first;
    }
    of << ";\n";
    return;
  }
  auto classicsIt = classics.cbegin();
  for (const auto& q : targets) {
    if (type == otMeasure && openQASM3) {
      of << creg[*classicsIt].second << " = ";
    }
    of << toString(type) << " " << qreg[q].second;
    if (type == otMeasure && !openQASM3) {
      of << " -> " << creg[*classicsIt].second;
      ++classicsIt;
    }
    of << ";\n";
  }
}
void NonUnitaryOperation::MEASURE_dumpOriginIR(
    std::ostream& of,
    const RegisterNames& qreg,
    const RegisterNames& creg, size_t indent)const  {
    auto classicsIt = classics.cbegin();
    for (const auto& q : targets) {
        std::stringstream ss;
        of << toString(type) << " " << qreg[q].second;
        ss << toString(type) << " " << qreg[q].second;
        of << "," << creg[*classicsIt].second;
        ss << "," << creg[*classicsIt].second;
        ++classicsIt;
        of << "\n";
        ss << "\n";
    }
}

void NonUnitaryOperation::RESET_dumpOriginIR(
    std::ostream& of,
    const RegisterNames& qreg,
    const RegisterNames& creg, size_t indent)const {
    auto classicsIt = classics.cbegin();
    of << toString(type);
    for (const auto& q : targets) {
        std::stringstream ss;
        of <<" " << qreg[q].second;
        of << "\n";
    }
    
}

void NonUnitaryOperation::BARRIER_dumpOriginIR(
    std::ostream& of,
    const RegisterNames& qreg,
    const RegisterNames& creg, size_t indent)const {
    auto classicsIt = classics.cbegin();
    if (targets.size() == 0) {
        for (const auto& q : qreg) {
            of << " " << q.second;
        }
    }
    else {
        for (const auto& q : targets) {
            std::stringstream ss;
            of << " " << qreg[q].second;
        }
    }
    of << "\n";
}

void NonUnitaryOperation::dumpOriginIR(std::ostream& of,
    const RegisterNames& qreg,
    const RegisterNames& creg, size_t indent) const {
    of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
    std::stringstream ss;
    if (isWholeQubitRegister(qreg, targets.front(), targets.back()) &&
        (type != otMeasure ||
            isWholeQubitRegister(creg, classics.front(), classics.back()))) 
    {
        of << toString(type) << " " << qreg[targets.front()].first;
        ss << toString(type) << " " << qreg[targets.front()].first;
        if (type == otMeasure ) {
            of << ",";
            ss << ",";
            of << creg[classics.front()].first;
            ss << creg[classics.front()].first;
        }
        of << "\n";
        ss << "\n";
        return;
    }
    /*
    auto classicsIt = classics.cbegin();
    for (const auto& q : targets) {
        std::stringstream ss;
        of << toString(type) << " " << qreg[q].second;
        ss << toString(type) << " " << qreg[q].second;
        std::cout << "NonUnitaryOperation.cpp¡·NonUnitaryOperation::dumpOriginIR>>3>>1ss:" << ss.str() << std::endl;
        if (type == otMeasure) {
            of << "," << creg[*classicsIt].second;
            ss << "," << creg[*classicsIt].second;
            std::cout << "NonUnitaryOperation.cpp¡·NonUnitaryOperation::dumpOriginIR>>3>>2ss:" << ss.str() << std::endl;
            ++classicsIt;
        }
        of << "\n";
        ss << "\n";
        std::cout << "NonUnitaryOperation.cpp¡·NonUnitaryOperation::dumpOriginIR>>3>>3ss:"<<ss.str() << std::endl;
    }
    */
    if (type == otMeasure) {
        MEASURE_dumpOriginIR(of, qreg, creg, indent);
    }
    else if (type == otReset) {
        RESET_dumpOriginIR(of, qreg, creg, indent);
    }
    else if (type == otBarrier) {
        BARRIER_dumpOriginIR(of, qreg, creg, indent);
    }
}

bool NonUnitaryOperation::equals(const Operation& op, const Permutation& perm1,
                                 const Permutation& perm2) const {
  if (const auto* nonunitary = dynamic_cast<const NonUnitaryOperation*>(&op)) {
    if (getType() != nonunitary->getType()) {
      return false;
    }

    if (getType() == otMeasure) {
      // check number of qubits to be measured
      const auto nq1 = targets.size();
      const auto nq2 = nonunitary->targets.size();
      if (nq1 != nq2) {
        return false;
      }

      // these are just sanity checks and should always be fulfilled
      assert(targets.size() == classics.size());
      assert(nonunitary->targets.size() == nonunitary->classics.size());

      std::set<std::pair<QBit, Bit>> measurements1{};
      auto qubitIt1 = targets.cbegin();
      auto classicIt1 = classics.cbegin();
      while (qubitIt1 != targets.cend()) {
        if (perm1.empty()) {
          measurements1.emplace(*qubitIt1, *classicIt1);
        } else {
          measurements1.emplace(perm1.at(*qubitIt1), *classicIt1);
        }
        ++qubitIt1;
        ++classicIt1;
      }

      std::set<std::pair<QBit, Bit>> measurements2{};
      auto qubitIt2 = nonunitary->targets.cbegin();
      auto classicIt2 = nonunitary->classics.cbegin();
      while (qubitIt2 != nonunitary->targets.cend()) {
        if (perm2.empty()) {
          measurements2.emplace(*qubitIt2, *classicIt2);
        } else {
          measurements2.emplace(perm2.at(*qubitIt2), *classicIt2);
        }
        ++qubitIt2;
        ++classicIt2;
      }

      return measurements1 == measurements2;
    }
    return Operation::equals(op, perm1, perm2);
  }
  return false;
}

void NonUnitaryOperation::printMeasurement(std::ostream& os,
                                           const std::vector<QBit>& q,
                                           const std::vector<Bit>& c,
                                           const Permutation& permutation,
                                           const std::size_t nqubits) {
  auto qubitIt = q.cbegin();
  auto classicIt = c.cbegin();
  if (permutation.empty()) {
    for (std::size_t i = 0; i < nqubits; ++i) {
      if (qubitIt != q.cend() && *qubitIt == i) {
        os << "\033[34m" << std::setw(4) << *classicIt << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << std::setw(4) << "|";
      }
    }
  } else {
    for (const auto& [physical, logical] : permutation) {
      if (qubitIt != q.cend() && *qubitIt == physical) {
        os << "\033[34m" << std::setw(4) << *classicIt << "\033[0m";
        ++qubitIt;
        ++classicIt;
      } else {
        os << std::setw(4) << "|";
      }
    }
  }
}

void NonUnitaryOperation::printReset(std::ostream& os,
                                     const std::vector<QBit>& q,
                                     const Permutation& permutation,
                                     const std::size_t nqubits) const {
  const auto actualTargets = permutation.apply(q);
  for (std::size_t i = 0; i < nqubits; ++i) {
    if (std::find(actualTargets.cbegin(), actualTargets.cend(), i) !=
        actualTargets.cend()) {
      os << "\033[31m" << std::setw(4) << shortName(type) << "\033[0m";
      continue;
    }
    os << std::setw(4) << "|";
  }
}

void NonUnitaryOperation::addDepthContribution(
    std::vector<std::size_t>& depths) const {
  for (const auto& target : getTargets()) {
    depths[target] += 1;
  }
}

void NonUnitaryOperation::apply(const Permutation& permutation) {
  getTargets() = permutation.apply(getTargets());
}
} // namespace qc
