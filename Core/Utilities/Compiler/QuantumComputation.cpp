
#include "Core/Utilities/Compiler/QuantumComputation.hpp"
#include <cassert>
#include <memory>

namespace qc {

/***
 * Public Methods
 ***/
std::size_t QuantumComputation::getNindividualOps() const {
  std::size_t nops = 0;
  for (const auto& op : ops) {
    if (const auto* const comp =
            dynamic_cast<const CompoundOperation*>(op.get());
        comp != nullptr) {
      nops += comp->size();
    } else {
      ++nops;
    }
  }

  return nops;
}

std::size_t QuantumComputation::getNsingleQubitOps() const {
  std::size_t nops = 0;
  for (const auto& op : ops) {
    if (!op->isUnitary()) {
      continue;
    }

    if (const auto* const comp =
            dynamic_cast<const CompoundOperation*>(op.get());
        comp != nullptr) {
      for (const auto& subop : *comp) {
        if (subop->isUnitary() && !subop->isControlled() &&
            subop->getNtargets() == 1U) {
          ++nops;
        }
      }
    } else {
      if (!op->isControlled() && op->getNtargets() == 1U) {
        ++nops;
      }
    }
  }
  return nops;
}

std::size_t QuantumComputation::getDepth() const {
  if (empty()) {
    return 0U;
  }

  std::vector<std::size_t> depths(getNqubits(), 0U);
  for (const auto& op : ops) {
    op->addDepthContribution(depths);
  }

  return *std::max_element(depths.begin(), depths.end());
}

void QuantumComputation::import(const std::string& filename) {
  const std::size_t dot = filename.find_last_of('.');
  std::string extension = filename.substr(dot + 1);
  std::transform(
      extension.begin(), extension.end(), extension.begin(),
      [](unsigned char ch) { return static_cast<char>(::tolower(ch)); });
  if (extension == "real") {
    import(filename, Format::Real);
  } else if (extension == "qasm") {
    import(filename, Format::OpenQASM3);
  } else if (extension == "txt") {
    import(filename, Format::GRCS);
  } else if (extension == "tfc") {
    import(filename, Format::TFC);
  } else if (extension == "qc") {
    import(filename, Format::QC);
  } else {
    throw QFRException("[import] extension " + extension + " not recognized");
  }
}

void QuantumComputation::import(const std::string& filename, Format format) {
  const std::size_t slash = filename.find_last_of('/');
  const std::size_t dot = filename.find_last_of('.');
  name = filename.substr(slash + 1, dot - slash - 1);

  auto ifs = std::ifstream(filename);
  if (ifs.good()) {
    import(ifs, format);
  } else {
    throw QFRException("[import] Error processing input stream: " + name);
  }
}

void QuantumComputation::import(std::istream&& is, Format format) {
  // reset circuit before importing
  reset();

  switch (format) {
  case Format::Real:
    //importReal(is);
    break;
  case Format::OpenQASM2:
  case Format::OpenQASM3:
    importOpenQASM(is);
    break;
  case Format::GRCS:
    //importGRCS(is);
    break;
  case Format::TFC:
    //importTFC(is);
    break;
  case Format::QC:
    //importQC(is);
    break;
  default:
    throw QFRException("[import] format not recognized");
  }

  // initialize the initial layout and output permutation
  initializeIOMapping();
}

void QuantumComputation::initializeIOMapping() {
  // if no initial layout was found during parsing the identity mapping is
  // assumed
  if (initialLayout.empty()) {
    for (QBit i = 0; i < nqubits; ++i) {
      initialLayout.emplace(i, i);
    }
  }

  // try gathering (additional) output permutation information from
  // measurements, e.g., a measurement
  //      `measure q[i] -> c[j];`
  // implies that the j-th (logical) output is obtained from measuring the i-th
  // physical qubit.
  const bool outputPermutationFound = !outputPermutation.empty();

  // track whether the circuit contains measurements at the end of the circuit
  // if it does, then all qubits that are not measured shall be considered
  // garbage outputs
  bool outputPermutationFromMeasurements = false;
  std::set<QBit> measuredQubits{};

  for (const auto& opIt : ops) {
    if (const auto* const op = dynamic_cast<NonUnitaryOperation*>(opIt.get());
        op != nullptr && op->getType() == otMeasure) {
      outputPermutationFromMeasurements = true;
      assert(op->getTargets().size() == op->getClassics().size());
      auto classicIt = op->getClassics().cbegin();
      for (const auto& q : op->getTargets()) {
        const auto qubitidx = q;
        // only the first measurement of a qubit is used to determine the output
        // permutation
        if (measuredQubits.count(qubitidx) != 0) {
          continue;
        }

        const auto bitidx = *classicIt;
        if (outputPermutationFound) {
          // output permutation was already set before -> permute existing
          // values
          const auto current = outputPermutation.at(qubitidx);
          if (static_cast<std::size_t>(current) != bitidx) {
            for (auto& p : outputPermutation) {
              if (static_cast<std::size_t>(p.second) == bitidx) {
                p.second = current;
                break;
              }
            }
            outputPermutation.at(qubitidx) = static_cast<QBit>(bitidx);
          }
        } else {
          // directly set permutation if none was set beforehand
          outputPermutation[qubitidx] = static_cast<QBit>(bitidx);
        }
        measuredQubits.emplace(qubitidx);
        ++classicIt;
      }
    }
  }

  // clear any qubits that were not measured from the output permutation
  // these will be marked garbage further down below
  if (outputPermutationFromMeasurements) {
    auto it = outputPermutation.begin();
    while (it != outputPermutation.end()) {
      if (measuredQubits.find(it->first) == measuredQubits.end()) {
        it = outputPermutation.erase(it);
      } else {
        ++it;
      }
    }
  }

  const bool buildOutputPermutation = outputPermutation.empty();
  garbage.assign(nqubits + nancillae, false);
  for (const auto& [physicalIn, logicalIn] : initialLayout) {
    const bool isIdle = isIdleQubit(physicalIn);

    // if no output permutation was found, build it from the initial layout
    if (buildOutputPermutation && !isIdle) {
      outputPermutation.insert({physicalIn, logicalIn});
    }

    // if the qubit is not an output, mark it as garbage
    const bool isOutput = std::any_of(
        outputPermutation.begin(), outputPermutation.end(),
        [&logicIn = logicalIn](const auto& p) { return p.second == logicIn; });
    if (!isOutput) {
      setLogicalQubitGarbage(logicalIn);
    }

    // if the qubit is an ancillary and idle, mark it as garbage
    if (logicalQubitIsAncillary(logicalIn) && isIdle) {
      setLogicalQubitGarbage(logicalIn);
    }
  }
}

void QuantumComputation::addQubitRegister(std::size_t nq,
                                          const std::string& regName) {
  if (qregs.count(regName) != 0) {
    auto& reg = qregs.at(regName);
    if (reg.first + reg.second == nqubits + nancillae) {
      reg.second += nq;
    } else {
      throw QFRException(
          "[addQubitRegister] Augmenting existing qubit registers is only "
          "supported for the last register in a circuit");
    }
  } else {
    qregs.try_emplace(regName, static_cast<QBit>(nqubits), nq);
  }
  assert(nancillae ==
         0); // should only reach this point if no ancillae are present

  for (std::size_t i = 0; i < nq; ++i) {
    auto j = static_cast<QBit>(nqubits + i);
    initialLayout.insert({j, j});
    outputPermutation.insert({j, j});
  }
  nqubits += nq;
  ancillary.resize(nqubits + nancillae);
  garbage.resize(nqubits + nancillae);
}

void QuantumComputation::addClassicalRegister(std::size_t nc,
                                              const std::string& regName) {
  if (cregs.count(regName) != 0) {
    throw QFRException("[addClassicalRegister] Augmenting existing classical "
                       "registers is currently not supported");
  }
  if (nc == 0) {
    throw QFRException(
        "[addClassicalRegister] New register size must be larger than 0");
  }

  cregs.try_emplace(regName, nclassics, nc);
  nclassics += nc;
}

void QuantumComputation::addAncillaryRegister(std::size_t nq,
                                              const std::string& regName) {
  const auto totalqubits = nqubits + nancillae;
  if (ancregs.count(regName) != 0) {
    auto& reg = ancregs.at(regName);
    if (reg.first + reg.second == totalqubits) {
      reg.second += nq;
    } else {
      throw QFRException(
          "[addAncillaryRegister] Augmenting existing ancillary registers is "
          "only supported for the last register in a circuit");
    }
  } else {
    ancregs.try_emplace(regName, static_cast<QBit>(totalqubits), nq);
  }

  ancillary.resize(totalqubits + nq);
  garbage.resize(totalqubits + nq);
  for (std::size_t i = 0; i < nq; ++i) {
    auto j = static_cast<QBit>(totalqubits + i);
    initialLayout.insert({j, j});
    outputPermutation.insert({j, j});
    ancillary[j] = true;
  }
  nancillae += nq;
}

void QuantumComputation::removeQubitfromQubitRegister(QuantumRegisterMap& regs,
                                                      const std::string& reg,
                                                      const QBit idx) {
  if (idx == 0) {
    // last remaining qubit of register
    if (regs[reg].second == 1) {
      // delete register
      regs.erase(reg);
    }
    // first qubit of register
    else {
      regs[reg].first++;
      regs[reg].second--;
    }
    // last index
  } else if (idx == regs[reg].second - 1) {
    // reduce count of register
    regs[reg].second--;
  } else {
    auto qreg = regs.at(reg);
    auto lowPart = reg + "_l";
    auto lowIndex = qreg.first;
    auto lowCount = idx;
    auto highPart = reg + "_h";
    auto highIndex = qreg.first + idx + 1;
    auto highCount = qreg.second - idx - 1;

    regs.erase(reg);
    regs.try_emplace(lowPart, lowIndex, lowCount);
    regs.try_emplace(highPart, highIndex, highCount);
  }
}

void QuantumComputation::addQubitToQubitRegister(
    QuantumRegisterMap& regs, const QBit physicalQubitIndex,
    const std::string& defaultRegName) {
  bool fusionPossible = false;
  for (auto& reg : regs) {
    auto& startIndex = reg.second.first;
    auto& count = reg.second.second;
    // 1st case: can append to start of existing register
    if (startIndex == physicalQubitIndex + 1) {
      startIndex--;
      count++;
      fusionPossible = true;
      break;
    }
    // 2nd case: can append to end of existing register
    if (startIndex + count == physicalQubitIndex) {
      count++;
      fusionPossible = true;
      break;
    }
  }

  consolidateRegister(regs);

  if (regs.empty()) {
    regs.try_emplace(defaultRegName, physicalQubitIndex, 1);
  } else if (!fusionPossible) {
    auto newRegName = defaultRegName + "_" + std::to_string(physicalQubitIndex);
    regs.try_emplace(newRegName, physicalQubitIndex, 1);
  }
}

// removes the i-th logical qubit and returns the index j it was assigned to in
// the initial layout i.e., initialLayout[j] = i
std::pair<QBit, std::optional<QBit>>
QuantumComputation::removeQubit(const QBit logicalQubitIndex) {
  // Find index of the physical qubit i is assigned to
  const auto physicalQubitIndex = getPhysicalQubitIndex(logicalQubitIndex);

  // get register and register-index of the corresponding qubit
  const auto [reg, idx] = getQubitRegisterAndIndex(physicalQubitIndex);

  if (physicalQubitIsAncillary(physicalQubitIndex)) {
    removeQubitfromQubitRegister(ancregs, reg, idx);
    // reduce ancilla count
    nancillae--;
  } else {
    removeQubitfromQubitRegister(qregs, reg, idx);
    // reduce qubit count
    if (ancillary.at(logicalQubitIndex)) {
      // if the qubit is ancillary, it is not counted as a qubit
      nancillae--;
    } else {
      nqubits--;
    }
  }

  // adjust initial layout permutation
  initialLayout.erase(physicalQubitIndex);

  // remove potential output permutation entry
  std::optional<QBit> outputQubitIndex{};
  if (const auto it = outputPermutation.find(physicalQubitIndex);
      it != outputPermutation.end()) {
    outputQubitIndex = it->second;
    // erasing entry
    outputPermutation.erase(physicalQubitIndex);
  }

  // update ancillary and garbage tracking
  const auto totalQubits = nqubits + nancillae;
  for (std::size_t i = logicalQubitIndex; i < totalQubits; ++i) {
    ancillary[i] = ancillary[i + 1];
    garbage[i] = garbage[i + 1];
  }
  // unset last entry
  ancillary[totalQubits] = false;
  garbage[totalQubits] = false;

  return {physicalQubitIndex, outputQubitIndex};
}

// adds j-th physical qubit as ancilla to the end of reg or creates the register
// if necessary
void QuantumComputation::addAncillaryQubit(
    QBit physicalQubitIndex, std::optional<QBit> outputQubitIndex) {
  if (initialLayout.count(physicalQubitIndex) > 0 ||
      outputPermutation.count(physicalQubitIndex) > 0) {
    throw QFRException("[addAncillaryQubit] Attempting to insert physical "
                       "qubit that is already assigned");
  }

  addQubitToQubitRegister(ancregs, physicalQubitIndex, "anc");

  // index of logical qubit
  const auto logicalQubitIndex = nqubits + nancillae;

  // resize ancillary and garbage tracking vectors
  ancillary.resize(logicalQubitIndex + 1U);
  garbage.resize(logicalQubitIndex + 1U);

  // increase ancillae count and mark as ancillary
  nancillae++;
  ancillary[logicalQubitIndex] = true;

  // adjust initial layout
  initialLayout.insert(
      {physicalQubitIndex, static_cast<QBit>(logicalQubitIndex)});

  // adjust output permutation
  if (outputQubitIndex.has_value()) {
    outputPermutation.insert({physicalQubitIndex, *outputQubitIndex});
  } else {
    // if a qubit is not relevant for the output, it is considered garbage
    garbage[logicalQubitIndex] = true;
  }
}

void QuantumComputation::addQubit(const QBit logicalQubitIndex,
                                  const QBit physicalQubitIndex,
                                  const std::optional<QBit> outputQubitIndex) {
  if (initialLayout.count(physicalQubitIndex) > 0 ||
      outputPermutation.count(physicalQubitIndex) > 0) {
    throw QFRException("[addQubit] Attempting to insert physical qubit that is "
                       "already assigned");
  }

  if (logicalQubitIndex > nqubits) {
    throw QFRException(
        "[addQubit] There are currently only " + std::to_string(nqubits) +
        " qubits in the circuit. Adding " + std::to_string(logicalQubitIndex) +
        " is therefore not possible at the moment.");
    // TODO: this does not necessarily have to lead to an error. A new qubit
    // register could be created and all ancillaries shifted
  }

  addQubitToQubitRegister(qregs, physicalQubitIndex, "q");

  // increase qubit count
  nqubits++;
  // adjust initial layout
  initialLayout.insert({physicalQubitIndex, logicalQubitIndex});
  if (outputQubitIndex.has_value()) {
    // adjust output permutation
    outputPermutation.insert({physicalQubitIndex, *outputQubitIndex});
  }

  // update ancillary and garbage tracking
  const auto totalQubits = nqubits + nancillae;
  ancillary.resize(totalQubits);
  garbage.resize(totalQubits);
  for (auto i = totalQubits - 1; i > logicalQubitIndex; --i) {
    ancillary[i] = ancillary[i - 1];
    garbage[i] = garbage[i - 1];
  }
  // unset new entry
  ancillary[logicalQubitIndex] = false;
  garbage[logicalQubitIndex] = false;
}

std::ostream& QuantumComputation::print(std::ostream& os) const {
  os << name << "\n";
  const auto width =
      ops.empty() ? 1 : static_cast<int>(std::log10(ops.size()) + 1.);

  os << std::setw(width + 1) << "i:";
  for (const auto& [physical, logical] : initialLayout) {
    if (ancillary[logical]) {
      os << "\033[31m";
    }
    os << std::setw(4) << logical << "\033[0m";
  }
  os << "\n";

  size_t i = 0U;
  for (const auto& op : ops) {
    os << std::setw(width) << ++i << ":";
    op->print(os, initialLayout, static_cast<std::size_t>(width) + 1U,
              getNqubits());
    os << "\n";
  }

  os << std::setw(width + 1) << "o:";
  for (const auto& physicalQubit : initialLayout) {
    auto it = outputPermutation.find(physicalQubit.first);
    if (it == outputPermutation.end()) {
      os << "\033[31m" << std::setw(4) << "|" << "\033[0m";
    } else {
      os << std::setw(4) << it->second;
    }
  }
  os << "\n";
  return os;
}

void QuantumComputation::printBin(std::size_t n, std::stringstream& ss) {
  if (n > 1) {
    printBin(n / 2, ss);
  }
  ss << n % 2;
}

std::ostream& QuantumComputation::printStatistics(std::ostream& os) const {
  os << "QC Statistics:";
  os << "\n\tn: " << static_cast<std::size_t>(nqubits);
  os << "\n\tanc: " << static_cast<std::size_t>(nancillae);
  os << "\n\tm: " << ops.size();
  os << "\n--------------\n";
  return os;
}

void QuantumComputation::dump(const std::string& filename) {
  const std::size_t dot = filename.find_last_of('.');
  assert(dot != std::string::npos);
  std::string extension = filename.substr(dot + 1);
  std::transform(
      extension.begin(), extension.end(), extension.begin(),
      [](unsigned char c) { return static_cast<char>(::tolower(c)); });
  if (extension == "real") {
    dump(filename, Format::Real);
  } else if (extension == "qasm") {
    dump(filename, Format::OpenQASM3);
  } else if (extension == "qc") {
    dump(filename, Format::QC);
  } else if (extension == "tfc") {
    dump(filename, Format::TFC);
  } else if (extension == "tensor") {
    dump(filename, Format::Tensor);
  } else {
    throw QFRException("[dump] Extension " + extension +
                       " not recognized/supported for dumping.");
  }
}

void QuantumComputation::dumpOpenQASM(std::ostream& of, bool openQASM3) {
  // Add missing physical qubits
  if (!qregs.empty()) {
    for (QBit physicalQubit = 0; physicalQubit < initialLayout.rbegin()->first;
         ++physicalQubit) {
      if (initialLayout.count(physicalQubit) == 0) {
        const auto logicalQubit = getHighestLogicalQubitIndex() + 1;
        addQubit(logicalQubit, physicalQubit, std::nullopt);
      }
    }
  }

  // dump initial layout and output permutation
  Permutation inverseInitialLayout{};
  for (const auto& q : initialLayout) {
    inverseInitialLayout.insert({q.second, q.first});
  }
  of << "// i";
  for (const auto& q : inverseInitialLayout) {
    of << " " << static_cast<std::size_t>(q.second);
  }
  of << "\n";

  Permutation inverseOutputPermutation{};
  for (const auto& q : outputPermutation) {
    inverseOutputPermutation.insert({q.second, q.first});
  }
  of << "// o";
  for (const auto& q : inverseOutputPermutation) {
    of << " " << q.second;
  }
  of << "\n";

  if (openQASM3) {
    of << "OPENQASM 3.0;\n";
    of << "include \"stdgates.inc\";\n";
  } else {
    of << "OPENQASM 2.0;\n";
    of << "include \"qelib1.inc\";\n";
  }
  if (std::any_of(std::begin(ops), std::end(ops), [](const auto& op) {
        return op->getType() == OpType::otTeleportation;
      })) {
    of << "opaque teleport src, anc, tgt;\n";
  }

  // combine qregs and ancregs
  QuantumRegisterMap combinedRegs = qregs;
  for (const auto& [regName, reg] : ancregs) {
    combinedRegs.try_emplace(regName, reg.first, reg.second);
  }
  printSortedRegisters(combinedRegs, openQASM3 ? "qubit" : "qreg", of,
                       openQASM3);
  RegisterNames combinedRegNames{};
  createRegisterArray(combinedRegs, combinedRegNames);
  assert(combinedRegNames.size() == nqubits + nancillae);

  printSortedRegisters(cregs, openQASM3 ? "bit" : "creg", of, openQASM3);
  RegisterNames cregnames{};
  createRegisterArray(cregs, cregnames);
  assert(cregnames.size() == nclassics);

  for (const auto& op : ops) {
    op->dumpOpenQASM(of, combinedRegNames, cregnames, 0, openQASM3);
  }
}

void QuantumComputation::dumpOriginIR(std::ostream& of) {
    // Add missing physical qubits
    if (!qregs.empty()) {
        for (QBit physicalQubit = 0; physicalQubit < initialLayout.rbegin()->first;
            ++physicalQubit) {
            if (initialLayout.count(physicalQubit) == 0) {
                const auto logicalQubit = getHighestLogicalQubitIndex() + 1;
                addQubit(logicalQubit, physicalQubit, std::nullopt);
            }
        }
    }

#if 0
    // dump initial layout and output permutation
    Permutation inverseInitialLayout{};
    for (const auto& q : initialLayout) {
        inverseInitialLayout.insert({ q.second, q.first });
    }
    of << "// i";
    for (const auto& q : inverseInitialLayout) {
        of << " " << static_cast<std::size_t>(q.second);
    }
    of << "\n";

    Permutation inverseOutputPermutation{};
    for (const auto& q : outputPermutation) {
        inverseOutputPermutation.insert({ q.second, q.first });
    }
    of << "// o";
    for (const auto& q : inverseOutputPermutation) {
        of << " " << q.second;
    }
    of << "\n";

    if (std::any_of(std::begin(ops), std::end(ops), [](const auto& op) {
        return op->getType() == OpType::Teleportation;
        })) 
    {
        of << "opaque teleport src, anc, tgt;\n";
    }

#endif

    // combine qregs and ancregs
    QuantumRegisterMap combinedRegs = qregs;
    for (const auto& [regName, reg] : ancregs) {
        combinedRegs.try_emplace(regName, reg.first, reg.second);
    }
    //printSortedRegisters(combinedRegs, "QINIT", of,false);
    RegisterNames combinedRegNames{};
    createRegisterArray(combinedRegs, combinedRegNames);
    for (int i = 0; i < combinedRegNames.size(); i++)
    {
        combinedRegNames[i] = std::make_pair("q", "q[" + std::to_string(i) + "]");
    }
    assert(combinedRegNames.size() == nqubits + nancillae);

    //printSortedRegisters(cregs, "CREG", of, false);
    RegisterNames cregnames{};
    createRegisterArray(cregs, cregnames);
    for (int i = 0; i < cregnames.size(); i++)
    {
        cregnames[i] = std::make_pair("c", "c[" + std::to_string(i) + "]");
    }
    assert(cregnames.size() == nclassics);

    printOriginirQubitsCbits(combinedRegNames.size(), cregnames.size(), of);

    
    for (const auto& op : ops) {
        op->dumpOriginIR(of, combinedRegNames, cregnames, 0);
    }
        
}

void QuantumComputation::dump(const std::string& filename, Format format) {
  auto of = std::ofstream(filename);
  if (!of.good()) {
    throw QFRException("[dump] Error opening file: " + filename);
  }
  dump(of, format);
}

void QuantumComputation::dump(std::ostream&& of, Format format) {
  switch (format) {
  case Format::OpenQASM3:
    dumpOpenQASM(of, true);
    break;
  case Format::OpenQASM2:
    dumpOpenQASM(of, false);
    break;
  case Format::Real:
    std::cerr << "Dumping in real format currently not supported\n";
    break;
  case Format::GRCS:
    std::cerr << "Dumping in GRCS format currently not supported\n";
    break;
  case Format::TFC:
    std::cerr << "Dumping in TFC format currently not supported\n";
    break;
  case Format::QC:
    std::cerr << "Dumping in QC format currently not supported\n";
    break;
  default:
    throw QFRException("[dump] Format not recognized/supported for dumping.");
  }
}

bool QuantumComputation::isIdleQubit(const QBit physicalQubit) const {
  return !std::any_of(
      ops.cbegin(), ops.cend(),
      [&physicalQubit](const auto& op) { return op->actsOn(physicalQubit); });
}

void QuantumComputation::stripIdleQubits(bool force,
                                         bool reduceIOpermutations) {
  auto layoutCopy = initialLayout;
  for (auto physicalQubitIt = layoutCopy.rbegin();
       physicalQubitIt != layoutCopy.rend(); ++physicalQubitIt) {
    auto physicalQubitIndex = physicalQubitIt->first;
    if (isIdleQubit(physicalQubitIndex)) {
      if (auto it = outputPermutation.find(physicalQubitIndex);
          it != outputPermutation.end() && !force) {
        continue;
      }

      auto logicalQubitIndex = initialLayout.at(physicalQubitIndex);
      // check whether the logical qubit is used in the output permutation
      bool usedInOutputPermutation = false;
      for (const auto& [physical, logical] : outputPermutation) {
        if (logical == logicalQubitIndex) {
          usedInOutputPermutation = true;
          break;
        }
      }
      if (usedInOutputPermutation && !force) {
        // cannot strip a logical qubit that is used in the output permutation
        continue;
      }

      removeQubit(logicalQubitIndex);

      if (reduceIOpermutations && (logicalQubitIndex < nqubits + nancillae)) {
        for (auto& [physical, logical] : initialLayout) {
          if (logical > logicalQubitIndex) {
            --logical;
          }
        }

        for (auto& [physical, logical] : outputPermutation) {
          if (logical > logicalQubitIndex) {
            --logical;
          }
        }
      }
    }
  }
}

std::string
QuantumComputation::getQubitRegister(const QBit physicalQubitIndex) const {
  for (const auto& reg : qregs) {
    auto startIdx = reg.second.first;
    auto count = reg.second.second;
    if (physicalQubitIndex < startIdx) {
      continue;
    }
    if (physicalQubitIndex >= startIdx + count) {
      continue;
    }
    return reg.first;
  }
  for (const auto& reg : ancregs) {
    auto startIdx = reg.second.first;
    auto count = reg.second.second;
    if (physicalQubitIndex < startIdx) {
      continue;
    }
    if (physicalQubitIndex >= startIdx + count) {
      continue;
    }
    return reg.first;
  }

  throw QFRException("[getQubitRegister] QBit index " +
                     std::to_string(physicalQubitIndex) +
                     " not found in any register");
}

std::pair<std::string, QBit> QuantumComputation::getQubitRegisterAndIndex(
    const QBit physicalQubitIndex) const {
  const std::string regName = getQubitRegister(physicalQubitIndex);
  QBit index = 0;
  auto it = qregs.find(regName);
  if (it != qregs.end()) {
    index = physicalQubitIndex - it->second.first;
  } else {
    auto itAnc = ancregs.find(regName);
    if (itAnc != ancregs.end()) {
      index = physicalQubitIndex - itAnc->second.first;
    }
    // no else branch needed here, since error would have already shown in
    // getQubitRegister(physicalQubitIndex)
  }
  return {regName, index};
}

std::string
QuantumComputation::getClassicalRegister(const Bit classicalIndex) const {
  for (const auto& reg : cregs) {
    auto startIdx = reg.second.first;
    auto count = reg.second.second;
    if (classicalIndex < startIdx) {
      continue;
    }
    if (classicalIndex >= startIdx + count) {
      continue;
    }
    return reg.first;
  }

  throw QFRException("[getClassicalRegister] Classical index " +
                     std::to_string(classicalIndex) +
                     " not found in any register");
}

std::pair<std::string, Bit> QuantumComputation::getClassicalRegisterAndIndex(
    const Bit classicalIndex) const {
  const std::string regName = getClassicalRegister(classicalIndex);
  std::size_t index = 0;
  auto it = cregs.find(regName);
  if (it != cregs.end()) {
    index = classicalIndex - it->second.first;
  } // else branch not needed since getClassicalRegister covers this case
  return {regName, index};
}

QBit QuantumComputation::getPhysicalQubitIndex(const QBit logicalQubitIndex) {
  for (const auto& [physical, logical] : initialLayout) {
    if (logical == logicalQubitIndex) {
      return physical;
    }
  }
  throw QFRException("[getPhysicalQubitIndex] Logical qubit index " +
                     std::to_string(logicalQubitIndex) +
                     " not found in initial layout");
}

QBit QuantumComputation::getIndexFromQubitRegister(
    const std::pair<std::string, QBit>& qubit) const {
  // no range check is performed here!
  return qregs.at(qubit.first).first + qubit.second;
}
Bit QuantumComputation::getIndexFromClassicalRegister(
    const std::pair<std::string, std::size_t>& clbit) const {
  // no range check is performed here!
  return cregs.at(clbit.first).first + clbit.second;
}

std::ostream&
QuantumComputation::printPermutation(const Permutation& permutation,
                                     std::ostream& os) {
  for (const auto& [physical, logical] : permutation) {
    os << "\t" << physical << ": " << logical << "\n";
  }
  return os;
}

std::ostream& QuantumComputation::printRegisters(std::ostream& os) const {
  os << "qregs:";
  for (const auto& qreg : qregs) {
    os << " {" << qreg.first << ", {" << qreg.second.first << ", "
       << qreg.second.second << "}}";
  }
  os << "\n";
  if (!ancregs.empty()) {
    os << "ancregs:";
    for (const auto& ancreg : ancregs) {
      os << " {" << ancreg.first << ", {" << ancreg.second.first << ", "
         << ancreg.second.second << "}}";
    }
    os << "\n";
  }
  os << "cregs:";
  for (const auto& creg : cregs) {
    os << " {" << creg.first << ", {" << creg.second.first << ", "
       << creg.second.second << "}}";
  }
  os << "\n";
  return os;
}

QBit QuantumComputation::getHighestLogicalQubitIndex(
    const Permutation& permutation) {
  QBit maxIndex = 0;
  for (const auto& [physical, logical] : permutation) {
    maxIndex = std::max(maxIndex, logical);
  }
  return maxIndex;
}

bool QuantumComputation::physicalQubitIsAncillary(
    const QBit physicalQubitIndex) const {
  return std::any_of(ancregs.cbegin(), ancregs.cend(),
                     [&physicalQubitIndex](const auto& ancreg) {
                       return ancreg.second.first <= physicalQubitIndex &&
                              physicalQubitIndex <
                                  ancreg.second.first + ancreg.second.second;
                     });
}

void QuantumComputation::setLogicalQubitAncillary(
    const QBit logicalQubitIndex) {
  if (logicalQubitIsAncillary(logicalQubitIndex)) {
    return;
  }

  nqubits--;
  nancillae++;
  ancillary[logicalQubitIndex] = true;
}

void QuantumComputation::setLogicalQubitsAncillary(
    const QBit minLogicalQubitIndex, const QBit maxLogicalQubitIndex) {
  for (QBit i = minLogicalQubitIndex; i <= maxLogicalQubitIndex; i++) {
    setLogicalQubitAncillary(i);
  }
}

void QuantumComputation::setLogicalQubitGarbage(const QBit logicalQubitIndex) {
  garbage[logicalQubitIndex] = true;
  // setting a logical qubit garbage also means removing it from the output
  // permutation if it was present before
  for (auto it = outputPermutation.begin(); it != outputPermutation.end();
       ++it) {
    if (it->second == logicalQubitIndex) {
      outputPermutation.erase(it);
      break;
    }
  }
}

void QuantumComputation::setLogicalQubitsGarbage(
    const QBit minLogicalQubitIndex, const QBit maxLogicalQubitIndex) {
  for (QBit i = minLogicalQubitIndex; i <= maxLogicalQubitIndex; i++) {
    setLogicalQubitGarbage(i);
  }
}

[[nodiscard]] std::pair<bool, std::optional<QBit>>
QuantumComputation::containsLogicalQubit(const QBit logicalQubitIndex) const {
  if (const auto it = std::find_if(initialLayout.cbegin(), initialLayout.cend(),
                                   [&logicalQubitIndex](const auto& mapping) {
                                     return mapping.second == logicalQubitIndex;
                                   });
      it != initialLayout.cend()) {
    return {true, it->first};
  }
  return {false, std::nullopt};
}

bool QuantumComputation::isLastOperationOnQubit(
    const const_iterator& opIt, const const_iterator& end) const {
  if (opIt == end) {
    return true;
  }

  // determine which qubits the gate acts on
  std::vector<bool> actson(nqubits + nancillae);
  for (std::size_t i = 0; i < actson.size(); ++i) {
    if ((*opIt)->actsOn(static_cast<QBit>(i))) {
      actson[i] = true;
    }
  }

  // iterate over remaining gates and check if any act on qubits overlapping
  // with the target gate
  auto atEnd = opIt;
  std::advance(atEnd, 1);
  while (atEnd != end) {
    for (std::size_t i = 0; i < actson.size(); ++i) {
      if (actson[i] && (*atEnd)->actsOn(static_cast<QBit>(i))) {
        return false;
      }
    }
    ++atEnd;
  }
  return true;
}

void QuantumComputation::unifyQuantumRegisters(const std::string& regName) {
  ancregs.clear();
  qregs.clear();
  nqubits += nancillae;
  nancillae = 0;
  qregs[regName] = {0, nqubits};
}

void QuantumComputation::appendMeasurementsAccordingToOutputPermutation(
    const std::string& registerName) {
  // ensure that the circuit contains enough classical registers
  if (cregs.empty()) {
    // in case there are no registers, create a new one
    addClassicalRegister(outputPermutation.size(), registerName);
  } else if (nclassics < outputPermutation.size()) {
    if (cregs.find(registerName) == cregs.end()) {
      // in case there are registers but not enough, add a new one
      addClassicalRegister(outputPermutation.size() - nclassics, registerName);
    } else {
      // in case the register already exists, augment it
      nclassics += outputPermutation.size() - nclassics;
      cregs[registerName].second = outputPermutation.size();
    }
  }
  auto targets = std::vector<qc::QBit>{};
  for (std::size_t q = 0; q < getNqubits(); ++q) {
    targets.emplace_back(static_cast<QBit>(q));
  }
  barrier(targets);
  // append measurements according to output permutation
  for (const auto& [qubit, clbit] : outputPermutation) {
    measure(qubit, clbit);
  }
}

void QuantumComputation::checkQubitRange(const QBit qubit) const {
  if (const auto it = initialLayout.find(qubit);
      it == initialLayout.end() || it->second >= getNqubits()) {
    throw QFRException("QBit index out of range: " + std::to_string(qubit));
  }
}
void QuantumComputation::checkQubitRange(const QBit qubit,
                                         const Controls& controls) const {
  checkQubitRange(qubit);
  for (const auto& [ctrl, _] : controls) {
    checkQubitRange(ctrl);
  }
}

void QuantumComputation::checkQubitRange(const QBit qubit0, const QBit qubit1,
                                         const Controls& controls) const {
  checkQubitRange(qubit0, controls);
  checkQubitRange(qubit1);
}

void QuantumComputation::checkQubitRange(
    const std::vector<QBit>& qubits) const {
  for (const auto& qubit : qubits) {
    checkQubitRange(qubit);
  }
}

void QuantumComputation::checkBitRange(const qc::Bit bit) const {
  if (bit >= nclassics) {
    std::stringstream ss{};
    ss << "Classical bit index " << bit << " not found in any register";
    throw QFRException(ss.str());
  }
}

void QuantumComputation::checkBitRange(const std::vector<Bit>& bits) const {
  for (const auto& bit : bits) {
    checkBitRange(bit);
  }
}

void QuantumComputation::checkClassicalRegister(
    const ClassicalRegister& creg) const {
  if (creg.first + creg.second > nclassics) {
    std::stringstream ss{};
    ss << "Classical register starting at index " << creg.first << " with "
       << creg.second << " bits is too large! The circuit has " << nclassics
       << " classical bits.";
    throw QFRException(ss.str());
  }
}

void QuantumComputation::addVariable(const SymbolOrNumber& expr) {
  if (std::holds_alternative<Symbolic>(expr)) {
    const auto& sym = std::get<Symbolic>(expr);
    for (const auto& term : sym) {
      occuringVariables.insert(term.getVar());
    }
  }
}

// Instantiates this computation
void QuantumComputation::instantiateInplace(
    const VariableAssignment& assignment) {
  for (auto& op : ops) {
    if (auto* symOp = dynamic_cast<SymbolicOperation*>(op.get());
        symOp != nullptr) {
      symOp->instantiate(assignment);
      // if the operation is fully instantiated, it can be replaced by the
      // corresponding standard operation
      if (symOp->isStandardOperation()) {
        op = std::make_unique<StandardOperation>(
            *dynamic_cast<StandardOperation*>(symOp));
      }
    }
  }
  // after an operation is instantiated, the respective parameters can be
  // removed from the circuit
  for (const auto& [var, _] : assignment) {
    occuringVariables.erase(var);
  }
}

void QuantumComputation::measure(
    const QBit qubit, const std::pair<std::string, Bit>& registerBit) {
  checkQubitRange(qubit);
  if (const auto cRegister = cregs.find(registerBit.first);
      cRegister != cregs.end()) {
    if (registerBit.second >= cRegister->second.second) {
      std::stringstream ss{};
      ss << "The classical register \"" << registerBit.first
         << "\" is too small! (" << registerBit.second
         << " >= " << cRegister->second.second << ")";
      throw QFRException(ss.str());
    }
    emplace_back<NonUnitaryOperation>(qubit, cRegister->second.first +
                                                 registerBit.second);

  } else {
    std::stringstream ss{};
    ss << "The classical register \"" << registerBit.first
       << "\" does not exist!";
    throw QFRException(ss.str());
  }
}

void QuantumComputation::measureAll(const bool addBits) {
  if (addBits) {
    addClassicalRegister(getNqubits(), "meas");
  }

  if (nclassics < getNqubits()) {
    std::stringstream ss{};
    ss << "The number of classical bits (" << nclassics
       << ") is smaller than the number of qubits (" << getNqubits() << ")!";
    throw QFRException(ss.str());
  }

  barrier();
  QBit start = 0U;
  if (addBits) {
    start = static_cast<QBit>(cregs.at("meas").first);
  }
  // measure i -> (start+i) in descending order
  // (this is an optimization for the simulator)
  for (std::size_t i = getNqubits(); i > 0; --i) {
    const auto q = static_cast<QBit>(i - 1);
    measure(q, start + q);
  }
}
} // namespace qc
