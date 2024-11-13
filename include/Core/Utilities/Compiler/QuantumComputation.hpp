#pragma once

//#include "QPanda.h"
#include "Core/Utilities/Compiler/operations/ClassicControlledOperation.hpp"
#include "Core/Utilities/Compiler/operations/CompoundOperation.hpp"
#include "Core/Utilities/Compiler/operations/NonUnitaryOperation.hpp"
#include "Core/Utilities/Compiler/operations/StandardOperation.hpp"
#include "Core/Utilities/Compiler/operations/SymbolicOperation.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace qc 
{
    class CircuitOptimizer;

    class QuantumComputation 
    {
    public:
        using iterator = typename std::vector<std::unique_ptr<Operation>>::iterator;
        using const_iterator =
            typename std::vector<std::unique_ptr<Operation>>::const_iterator;

        friend class CircuitOptimizer;

    protected:

        //QProg ops;
        std::vector<std::unique_ptr<Operation>> ops{};

        std::size_t nqubits = 0;
        std::size_t nclassics = 0;
        std::size_t nancillae = 0;
        std::string name;

        // register names are used as keys, while the values are `{startIndex,
        // length}` pairs
        QuantumRegisterMap qregs{};
        ClassicalRegisterMap cregs{};
        QuantumRegisterMap ancregs{};

        std::mt19937_64 mt;
        std::size_t seed = 0;

        fp globalPhase = 0.;

        std::unordered_set<sym::Variable> occuringVariables;

        void importOpenQASM(std::istream& is);
        std::string importOpenQASMToIR(std::istream& is);

        //void importReal(std::istream& is);
        int readRealHeader(std::istream& is);
        void readRealGateDescriptions(std::istream& is, int line);
        //void importTFC(std::istream& is);
        int readTFCHeader(std::istream& is, std::map<std::string, QBit>& varMap);
        void readTFCGateDescriptions(std::istream& is, int line,
            std::map<std::string, QBit>& varMap);
        //void importQC(std::istream& is);
        //int readQCHeader(std::istream& is, std::map<std::string, QBit>& varMap);
        void readQCGateDescriptions(std::istream& is, int line,
            std::map<std::string, QBit>& varMap);
        //void importGRCS(std::istream& is);

        template <class RegisterType>
        static void printSortedRegisters(const RegisterMap<RegisterType>& regmap,
            const std::string& identifier,
            std::ostream& of, bool openQASM3 = false) {
            // sort regs by start index
            std::map<decltype(RegisterType::first),
                std::pair<std::string, RegisterType>>
                sortedRegs{};
            for (const auto& reg : regmap) {
                sortedRegs.insert({ reg.second.first, reg });
            }

            for (const auto& reg : sortedRegs) {
                if (openQASM3) {
                    of << identifier << "[" << reg.second.second.second << "] "
                        << reg.second.first << ";" << std::endl;
                }
                else {
                    of << identifier << " " 
                        << reg.second.second.second  << std::endl;
                }
            }
        }

        static void printOriginirQubitsCbits(size_t qubits_num, size_t cbits_num, std::ostream& of)
        {
            of << "QINIT " << qubits_num << std::endl;
            of << "CREG " << cbits_num << std::endl;
            return;
        }

        template <class RegisterType>
        static void consolidateRegister(RegisterMap<RegisterType>& regs) {
            bool finished = regs.empty();
            while (!finished) {
                for (const auto& qreg : regs) {
                    finished = true;
                    auto regname = qreg.first;
                    // check if lower part of register
                    if (regname.length() > 2 &&
                        regname.compare(regname.size() - 2, 2, "_l") == 0) {
                        auto lowidx = qreg.second.first;
                        auto lownum = qreg.second.second;
                        // search for higher part of register
                        auto highname = regname.substr(0, regname.size() - 1) + 'h';
                        auto it = regs.find(highname);
                        if (it != regs.end()) {
                            auto highidx = it->second.first;
                            auto highnum = it->second.second;
                            // fusion of registers possible
                            if (lowidx + lownum == highidx) {
                                finished = false;
                                auto targetname = regname.substr(0, regname.size() - 2);
                                auto targetidx = lowidx;
                                auto targetnum = lownum + highnum;
                                regs.insert({ targetname, {targetidx, targetnum} });
                                regs.erase(regname);
                                regs.erase(highname);
                            }
                        }
                        break;
                    }
                }
            }
        }

        /**
         * @brief Removes a certain qubit in a register from the register map
         * @details If this was the last qubit in the register, the register is
         * deleted. Removals at the beginning or the end of a register just modify the
         * existing register. Removals in the middle of a register split the register
         * into two new registers. The new registers are named by appending "_l" and
         * "_h" to the original register name.
         * @param regs A collection of all the registers
         * @param reg The name of the register containing the qubit to be removed
         * @param idx The index of the qubit in the register to be removed
         */
        static void removeQubitfromQubitRegister(QuantumRegisterMap& regs,
            const std::string& reg, QBit idx);

        /**
         * @brief Adds a qubit to a register in the register map
         * @details If the register map is empty, a new register is created with the
         * default name. If the qubit can be appended to the start or the end of an
         * existing register, it is appended. Otherwise a new register is created with
         * the default name and the qubit index appended.
         * @param regs A collection of all the registers
         * @param physicalQubitIndex The index of the qubit to be added
         * @param defaultRegName The default name of the register to be created
         */
        static void addQubitToQubitRegister(QuantumRegisterMap& regs,
            QBit physicalQubitIndex,
            const std::string& defaultRegName);

        template <class RegisterType>
        static void createRegisterArray(const RegisterMap<RegisterType>& regs,
            RegisterNames& regnames) {
            regnames.clear();
            std::stringstream ss;
            // sort regs by start index
            std::map<decltype(RegisterType::first),
                std::pair<std::string, RegisterType>>
                sortedRegs{};
            for (const auto& reg : regs) {
                sortedRegs.insert({ reg.second.first, reg });
            }

            for (const auto& reg : sortedRegs) {
                for (decltype(RegisterType::second) i = 0; i < reg.second.second.second;
                    i++) {
                    ss << reg.second.first << "[" << i << "]";
                    regnames.push_back(std::make_pair(reg.second.first, ss.str()));
                    ss.str(std::string());
                }
            }
        }

        [[nodiscard]] std::size_t getSmallestAncillary() const {
            for (std::size_t i = 0; i < ancillary.size(); ++i) {
                if (ancillary[i]) {
                    return i;
                }
            }
            return ancillary.size();
        }

        [[nodiscard]] std::size_t getSmallestGarbage() const {
            for (std::size_t i = 0; i < garbage.size(); ++i) {
                if (garbage[i]) {
                    return i;
                }
            }
            return garbage.size();
        }
        [[nodiscard]] bool isLastOperationOnQubit(const const_iterator& opIt) const {
            const auto end = ops.cend();
            return isLastOperationOnQubit(opIt, end);
        }
        void checkQubitRange(QBit qubit) const;
        void checkQubitRange(QBit qubit, const Controls& controls) const;
        void checkQubitRange(QBit qubit0, QBit qubit1,
            const Controls& controls) const;
        void checkQubitRange(const std::vector<QBit>& qubits) const;
        void checkBitRange(Bit bit) const;
        void checkBitRange(const std::vector<Bit>& bits) const;
        void checkClassicalRegister(const ClassicalRegister& creg) const;

    public:
        QuantumComputation() = default;
        explicit QuantumComputation(const std::size_t nq, const std::size_t nc = 0U,
            const std::size_t s = 0)
            : seed(s) {
            addQubitRegister(nq);
            if (nc > 0) {
                addClassicalRegister(nc);
            }
            if (seed != 0) {
                mt.seed(seed);
            }
            else {
                // create and properly seed rng
                std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
                    randomData{};
                std::random_device rd;
                std::generate(std::begin(randomData), std::end(randomData),
                    [&rd]() { return rd(); });
                std::seed_seq seeds(std::begin(randomData), std::end(randomData));
                mt.seed(seeds);
            }
        }
        explicit QuantumComputation(const std::string& filename,
            const std::size_t s = 0U)
            : seed(s) {
            import(filename);
            if (seed != 0U) {
                mt.seed(seed);
            }
            else {
                // create and properly seed rng
                std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
                    randomData{};
                std::random_device rd;
                std::generate(std::begin(randomData), std::end(randomData),
                    [&rd]() { return rd(); });
                std::seed_seq seeds(std::begin(randomData), std::end(randomData));
                mt.seed(seeds);
            }
        }
        QuantumComputation(QuantumComputation&& qc) noexcept = default;
        QuantumComputation& operator=(QuantumComputation&& qc) noexcept = default;
        QuantumComputation(const QuantumComputation& qc)
            : nqubits(qc.nqubits), nclassics(qc.nclassics), nancillae(qc.nancillae),
            name(qc.name), qregs(qc.qregs), cregs(qc.cregs), ancregs(qc.ancregs),
            mt(qc.mt), seed(qc.seed), globalPhase(qc.globalPhase),
            occuringVariables(qc.occuringVariables),
            initialLayout(qc.initialLayout),
            outputPermutation(qc.outputPermutation), ancillary(qc.ancillary),
            garbage(qc.garbage) {
            ops.reserve(qc.ops.size());
            for (const auto& op : qc.ops) {
                emplace_back(op->clone());
            }
        }
        QuantumComputation& operator=(const QuantumComputation& qc) {
            if (this != &qc) {
                nqubits = qc.nqubits;
                nclassics = qc.nclassics;
                nancillae = qc.nancillae;
                name = qc.name;
                qregs = qc.qregs;
                cregs = qc.cregs;
                ancregs = qc.ancregs;
                mt = qc.mt;
                seed = qc.seed;
                globalPhase = qc.globalPhase;
                occuringVariables = qc.occuringVariables;
                initialLayout = qc.initialLayout;
                outputPermutation = qc.outputPermutation;
                ancillary = qc.ancillary;
                garbage = qc.garbage;

                ops.clear();
                ops.reserve(qc.ops.size());
                for (const auto& op : qc.ops) {
                    emplace_back(op->clone());
                }
            }
            return *this;
        }
        virtual ~QuantumComputation() = default;

        /**
         * @brief Construct a QuantumComputation from an OpenQASM string
         * @param qasm The OpenQASM 2.0 or 3.0 string
         * @return The constructed QuantumComputation
         */
        [[nodiscard]] static QuantumComputation fromQASM(const std::string& qasm) {
            std::stringstream ss{};
            ss << qasm;
            QuantumComputation qc{};
            qc.importOpenQASM(ss);
            qc.initializeIOMapping();
            return qc;
        }

        /**
         * @brief Construct a QuantumComputation from an OpenQASM string To originir
         * @param qasm The OpenQASM 2.0 or 3.0 string
         * @return The constructed originir
         */
        [[nodiscard]] static std::string convertOriginIR(const std::string& qasm) 
        {
            std::stringstream ss{};
            ss << qasm;

            return "";
        }

        [[nodiscard]] static QuantumComputation fromQASMFile(const std::string& file_path) {
            std::ifstream file(file_path);
            file.open(file_path);

            if (!file.is_open())
                throw std::runtime_error(std::string("Could not open originir file at:") + file_path);
            std::stringstream ss;
            ss << file.rdbuf();
            // std::stringstream ss{};
            // ss << qasm;
            QuantumComputation qc{};
            qc.importOpenQASM(ss);
            qc.initializeIOMapping();
            return qc;
        }

        [[nodiscard]] virtual std::size_t getNops() const { return ops.size(); }
        [[nodiscard]] std::size_t getNqubits() const { return nqubits + nancillae; }
        [[nodiscard]] std::size_t getNancillae() const { return nancillae; }
        [[nodiscard]] std::size_t getNqubitsWithoutAncillae() const {
            return nqubits;
        }
        [[nodiscard]] std::size_t getNmeasuredQubits() const {
            return getNqubits() - getNgarbageQubits();
        }
        [[nodiscard]] std::size_t getNgarbageQubits() const {
            return static_cast<std::size_t>(
                std::count(getGarbage().begin(), getGarbage().end(), true));
        }
        [[nodiscard]] std::size_t getNcbits() const { return nclassics; }
        [[nodiscard]] std::string getName() const { return name; }
        [[nodiscard]] const QuantumRegisterMap& getQregs() const { return qregs; }
        [[nodiscard]] const ClassicalRegisterMap& getCregs() const { return cregs; }
        [[nodiscard]] const QuantumRegisterMap& getANCregs() const { return ancregs; }
        [[nodiscard]] decltype(mt)& getGenerator() { return mt; }

        [[nodiscard]] fp getGlobalPhase() const { return globalPhase; }

        void setName(const std::string& n) { name = n; }

        // physical qubits are used as keys, logical qubits as values
        Permutation initialLayout{};
        Permutation outputPermutation{};

        std::vector<bool> ancillary{};
        std::vector<bool> garbage{};

        [[nodiscard]] std::size_t getNindividualOps() const;
        [[nodiscard]] std::size_t getNsingleQubitOps() const;
        [[nodiscard]] std::size_t getDepth() const;

        [[nodiscard]] std::string getQubitRegister(QBit physicalQubitIndex) const;
        [[nodiscard]] std::string getClassicalRegister(Bit classicalIndex) const;
        static QBit getHighestLogicalQubitIndex(const Permutation& permutation);
        [[nodiscard]] QBit getHighestLogicalQubitIndex() const {
            return getHighestLogicalQubitIndex(initialLayout);
        };
        [[nodiscard]] std::pair<std::string, QBit>
            getQubitRegisterAndIndex(QBit physicalQubitIndex) const;
        [[nodiscard]] std::pair<std::string, Bit>
            getClassicalRegisterAndIndex(Bit classicalIndex) const;
        /**
         * @brief Returns the physical qubit index of the given logical qubit index
         * @details Iterates over the initial layout dictionary and returns the key
         * corresponding to the given value.
         * @param logicalQubitIndex The logical qubit index to look for
         * @return The physical qubit index of the given logical qubit index
         */
        [[nodiscard]] QBit getPhysicalQubitIndex(QBit logicalQubitIndex);

        [[nodiscard]] QBit
            getIndexFromQubitRegister(const std::pair<std::string, QBit>& qubit) const;
        [[nodiscard]] Bit getIndexFromClassicalRegister(
            const std::pair<std::string, std::size_t>& clbit) const;
        [[nodiscard]] bool isIdleQubit(QBit physicalQubit) const;
        [[nodiscard]] bool isLastOperationOnQubit(const const_iterator& opIt,
            const const_iterator& end) const;
        [[nodiscard]] bool physicalQubitIsAncillary(QBit physicalQubitIndex) const;
        [[nodiscard]] bool
            logicalQubitIsAncillary(const QBit logicalQubitIndex) const {
            return ancillary[logicalQubitIndex];
        }
        /**
         * @brief Sets the given logical qubit to be ancillary
         * @details Removes the qubit from the qubit register and adds it to the
         * ancillary register, if such a register exists. Otherwise a new ancillary
         * register is created.
         * @param logicalQubitIndex
         */
        void setLogicalQubitAncillary(QBit logicalQubitIndex);
        /**
         * @brief Sets all logical qubits in the range [minLogicalQubitIndex,
         * maxLogicalQubitIndex] to be ancillary
         * @details Removes the qubits from the qubit register and adds it to the
         * ancillary register, if such a register exists. Otherwise a new ancillary
         * register is created.
         * @param minLogicalQubitIndex first qubit that is set to be ancillary
         * @param maxLogicalQubitIndex last qubit that is set to be ancillary
         */
        void setLogicalQubitsAncillary(QBit minLogicalQubitIndex,
            QBit maxLogicalQubitIndex);
        [[nodiscard]] bool
            logicalQubitIsGarbage(const QBit logicalQubitIndex) const {
            return garbage[logicalQubitIndex];
        }
        void setLogicalQubitGarbage(QBit logicalQubitIndex);
        /**
         * @brief Sets all logical qubits in the range [minLogicalQubitIndex,
         * maxLogicalQubitIndex] to be garbage
         * @param minLogicalQubitIndex first qubit that is set to be garbage
         * @param maxLogicalQubitIndex last qubit that is set to be garbage
         */
        void setLogicalQubitsGarbage(QBit minLogicalQubitIndex,
            QBit maxLogicalQubitIndex);
        [[nodiscard]] const std::vector<bool>& getAncillary() const {
            return ancillary;
        }
        [[nodiscard]] const std::vector<bool>& getGarbage() const { return garbage; }

        /// checks whether the given logical qubit exists in the initial layout.
        /// \param logicalQubitIndex the logical qubit index to check
        /// \return whether the given logical qubit exists in the initial layout and
        /// to which physical qubit it is mapped
        [[nodiscard]] std::pair<bool, std::optional<QBit>>
            containsLogicalQubit(QBit logicalQubitIndex) const;

        /// Adds a global phase to the quantum circuit.
        /// \param angle the angle to add
        void gphase(const fp& angle) {
            globalPhase += angle;
            // normalize to [0, 2pi)
            while (globalPhase < 0) {
                globalPhase += 2 * qcPI;
            }
            while (globalPhase >= 2 * qcPI) {
                globalPhase -= 2 * qcPI;
            }
        }

        ///---------------------------------------------------------------------------
        ///                            \n Operations \n
        ///---------------------------------------------------------------------------

#define DEFINE_SINGLE_TARGET_OPERATION(op)                                     \
  void op(const QBit target) { mc##op(Controls{}, target); }                  \
  void c##op(const Control& control, const QBit target) {                     \
    mc##op(Controls{control}, target);                                         \
  }                                                                            \
  void mc##op(const Controls& controls, const QBit target) {                  \
    checkQubitRange(target, controls);                                         \
    emplace_back<StandardOperation>(controls, target,                          \
                                    OP_NAME_TO_TYPE.at(#op));                  \
  }

        DEFINE_SINGLE_TARGET_OPERATION(i)
            DEFINE_SINGLE_TARGET_OPERATION(x)
            DEFINE_SINGLE_TARGET_OPERATION(y)
            DEFINE_SINGLE_TARGET_OPERATION(z)
            DEFINE_SINGLE_TARGET_OPERATION(h)
            DEFINE_SINGLE_TARGET_OPERATION(s)
            DEFINE_SINGLE_TARGET_OPERATION(sdg)
            DEFINE_SINGLE_TARGET_OPERATION(t)
            DEFINE_SINGLE_TARGET_OPERATION(tdg)
            DEFINE_SINGLE_TARGET_OPERATION(v)
            DEFINE_SINGLE_TARGET_OPERATION(vdg)
            DEFINE_SINGLE_TARGET_OPERATION(sx)
            DEFINE_SINGLE_TARGET_OPERATION(sxdg)

#define DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(op, param)             \
  void op(const SymbolOrNumber&(param), const QBit target) {                  \
    mc##op(param, Controls{}, target);                                         \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param), const Control& control,             \
             const QBit target) {                                             \
    mc##op(param, Controls{control}, target);                                  \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param), const Controls& controls,          \
              const QBit target) {                                            \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param)) {                                   \
      emplace_back<StandardOperation>(controls, target,                        \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{std::get<fp>(param)});       \
    } else {                                                                   \
      addVariables(param);                                                     \
      emplace_back<SymbolicOperation>(                                         \
          controls, target, OP_NAME_TO_TYPE.at(#op), std::vector{param});      \
    }                                                                          \
  }

            DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rx, theta)
            DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(ry, theta)
            DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rz, theta)
            DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(p, theta)

#define DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)       \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          const QBit target) {                                                \
    mc##op(param0, param1, Controls{}, target);                                \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const Control& control, const QBit target) {                     \
    mc##op(param0, param1, Controls{control}, target);                         \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const Controls& controls, const QBit target) {                  \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1)) {                                  \
      emplace_back<StandardOperation>(                                         \
          controls, target, OP_NAME_TO_TYPE.at(#op),                           \
          std::vector{std::get<fp>(param0), std::get<fp>(param1)});            \
    } else {                                                                   \
      addVariables(param0, param1);                                            \
      emplace_back<SymbolicOperation>(controls, target,                        \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{param0, param1});            \
    }                                                                          \
  }

            DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(u2, phi, lambda)

#define DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(op, param0, param1,     \
                                                       param2)                 \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          const SymbolOrNumber&(param2), const QBit target) {                 \
    mc##op(param0, param1, param2, Controls{}, target);                        \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const SymbolOrNumber&(param2), const Control& control,            \
             const QBit target) {                                             \
    mc##op(param0, param1, param2, Controls{control}, target);                 \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const SymbolOrNumber&(param2), const Controls& controls,         \
              const QBit target) {                                            \
    checkQubitRange(target, controls);                                         \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1) &&                                  \
        std::holds_alternative<fp>(param2)) {                                  \
      emplace_back<StandardOperation>(                                         \
          controls, target, OP_NAME_TO_TYPE.at(#op),                           \
          std::vector{std::get<fp>(param0), std::get<fp>(param1),              \
                      std::get<fp>(param2)});                                  \
    } else {                                                                   \
      addVariables(param0, param1, param2);                                    \
      emplace_back<SymbolicOperation>(controls, target,                        \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{param0, param1, param2});    \
    }                                                                          \
  }

            DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(u, theta, phi, lambda)

#define DEFINE_TWO_TARGET_OPERATION(op)                                        \
  void op(const QBit target0, const QBit target1) {                          \
    mc##op(Controls{}, target0, target1);                                      \
  }                                                                            \
  void c##op(const Control& control, const QBit target0,                      \
             const QBit target1) {                                            \
    mc##op(Controls{control}, target0, target1);                               \
  }                                                                            \
  void mc##op(const Controls& controls, const QBit target0,                   \
              const QBit target1) {                                           \
    checkQubitRange(target0, target1, controls);                               \
    emplace_back<StandardOperation>(controls, target0, target1,                \
                                    OP_NAME_TO_TYPE.at(#op));                  \
  }

            DEFINE_TWO_TARGET_OPERATION(swap)
            DEFINE_TWO_TARGET_OPERATION(dcx)
            DEFINE_TWO_TARGET_OPERATION(ecr)
            DEFINE_TWO_TARGET_OPERATION(iswap)
            DEFINE_TWO_TARGET_OPERATION(iswapdg)
            DEFINE_TWO_TARGET_OPERATION(peres)
            DEFINE_TWO_TARGET_OPERATION(peresdg)

#define DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(op, param)                \
  void op(const SymbolOrNumber&(param), const QBit target0,                   \
          const QBit target1) {                                               \
    mc##op(param, Controls{}, target0, target1);                               \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param), const Control& control,             \
             const QBit target0, const QBit target1) {                       \
    mc##op(param, Controls{control}, target0, target1);                        \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param), const Controls& controls,          \
              const QBit target0, const QBit target1) {                      \
    checkQubitRange(target0, target1, controls);                               \
    if (std::holds_alternative<fp>(param)) {                                   \
      emplace_back<StandardOperation>(controls, target0, target1,              \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{std::get<fp>(param)});       \
    } else {                                                                   \
      addVariables(param);                                                     \
      emplace_back<SymbolicOperation>(controls, target0, target1,              \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{param});                     \
    }                                                                          \
  }

            DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rxx, theta)
            DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(ryy, theta)
            DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzz, theta)
            DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzx, theta)

#define DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)          \
  void op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),        \
          const QBit target0, const QBit target1) {                          \
    mc##op(param0, param1, Controls{}, target0, target1);                      \
  }                                                                            \
  void c##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),     \
             const Control& control, const QBit target0,                      \
             const QBit target1) {                                            \
    mc##op(param0, param1, Controls{control}, target0, target1);               \
  }                                                                            \
  void mc##op(const SymbolOrNumber&(param0), const SymbolOrNumber&(param1),    \
              const Controls& controls, const QBit target0,                   \
              const QBit target1) {                                           \
    checkQubitRange(target0, target1, controls);                               \
    if (std::holds_alternative<fp>(param0) &&                                  \
        std::holds_alternative<fp>(param1)) {                                  \
      emplace_back<StandardOperation>(                                         \
          controls, target0, target1, OP_NAME_TO_TYPE.at(#op),                 \
          std::vector{std::get<fp>(param0), std::get<fp>(param1)});            \
    } else {                                                                   \
      addVariables(param0, param1);                                            \
      emplace_back<SymbolicOperation>(controls, target0, target1,              \
                                      OP_NAME_TO_TYPE.at(#op),                 \
                                      std::vector{param0, param1});            \
    }                                                                          \
  }

            DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_minus_yy, theta, beta)
            DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_plus_yy, theta, beta)

#undef DEFINE_SINGLE_TARGET_OPERATION
#undef DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_OPERATION
#undef DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION

            void measure(const QBit qubit, const std::size_t bit) {
            checkQubitRange(qubit);
            checkBitRange(bit);
            emplace_back<NonUnitaryOperation>(qubit, bit);
        }

        void measure(QBit qubit, const std::pair<std::string, Bit>& registerBit);

        void measure(const Targets& qubits, const std::vector<Bit>& bits) {
            checkQubitRange(qubits);
            checkBitRange(bits);
            emplace_back<NonUnitaryOperation>(qubits, bits);
        }

        /**
         * @brief Add measurements to all qubits
         * @param addBits Whether to add new classical bits to the circuit
         * @details This function adds measurements to all qubits in the circuit and
         * appends a new classical register (named "meas") to the circuit if addBits
         * is true. Otherwise, qubit q is measured into classical bit q.
         */
        void measureAll(bool addBits = true);

        void reset(const QBit target) {
            checkQubitRange(target);
            emplace_back<NonUnitaryOperation>(std::vector<QBit>{target}, qc::otReset);
        }
        void reset(const Targets& targets) {
            checkQubitRange(targets);
            emplace_back<NonUnitaryOperation>(targets, qc::otReset);
        }

        void barrier() {
            std::vector<QBit> targets(getNqubits());
            std::iota(targets.begin(), targets.end(), 0);
            emplace_back<StandardOperation>(targets, qc::otBarrier);
        }
        void barrier(const QBit target) {
            checkQubitRange(target);
            emplace_back<StandardOperation>(target, qc::otBarrier);
        }
        void barrier(const Targets& targets) {
            checkQubitRange(targets);
            emplace_back<StandardOperation>(targets, qc::otBarrier);
        }

        void classicControlled(const OpType op, const QBit target,
            const ClassicalRegister& controlRegister,
            const std::uint64_t expectedValue = 1U,
            const std::vector<fp>& params = {}) {
            classicControlled(op, target, Controls{}, controlRegister, expectedValue,
                params);
        }
        void classicControlled(const OpType op, const QBit target,
            const Control control,
            const ClassicalRegister& controlRegister,
            const std::uint64_t expectedValue = 1U,
            const std::vector<fp>& params = {}) {
            classicControlled(op, target, Controls{ control }, controlRegister,
                expectedValue, params);
        }
        void classicControlled(const OpType op, const QBit target,
            const Controls& controls,
            const ClassicalRegister& controlRegister,
            const std::uint64_t expectedValue = 1U,
            const std::vector<fp>& params = {}) {
            checkQubitRange(target, controls);
            checkClassicalRegister(controlRegister);
            std::unique_ptr<Operation> gate =
                std::make_unique<StandardOperation>(controls, target, op, params);
            emplace_back<ClassicControlledOperation>(std::move(gate), controlRegister,
                expectedValue);
        }

        /// strip away qubits with no operations applied to them and which do not pop
        /// up in the output permutation \param force if true, also strip away idle
        /// qubits occurring in the output permutation
        void stripIdleQubits(bool force = false, bool reduceIOpermutations = true);

        void import(const std::string& filename);
        void import(const std::string& filename, Format format);
        void import(std::istream& is, Format format) {
            import(std::move(is), format);
        }
        void import(std::istream&& is, Format format);
        void initializeIOMapping();
        // append measurements to the end of the circuit according to the tracked
        // output permutation
        void appendMeasurementsAccordingToOutputPermutation(
            const std::string& registerName = "c");
        // search for current position of target value in map and afterwards exchange
        // it with the value at new position
        static void findAndSWAP(QBit targetValue, QBit newPosition,
            Permutation& map) {
            for (const auto& q : map) {
                if (q.second == targetValue) {
                    std::swap(map.at(newPosition), map.at(q.first));
                    break;
                }
            }
        }

        // this function augments a given circuit by additional registers
        void addQubitRegister(std::size_t, const std::string& regName = "q");
        void addClassicalRegister(std::size_t nc, const std::string& regName = "c");
        void addAncillaryRegister(std::size_t nq, const std::string& regName = "anc");
        // a function to combine all quantum registers (qregs and ancregs) into a
        // single register (useful for circuits mapped to a device)
        void unifyQuantumRegisters(const std::string& regName = "q");

        // removes a specific logical qubit and returns the index of the physical
        // qubit in the initial layout as well as the index of the removed physical
        // qubit's output permutation i.e., initialLayout[physical_qubit] =
        // logical_qubit and outputPermutation[physicalQubit] = output_qubit
        std::pair<QBit, std::optional<QBit>> removeQubit(QBit logicalQubitIndex);

        // adds physical qubit as ancillary qubit and gives it the appropriate output
        // mapping
        void addAncillaryQubit(QBit physicalQubitIndex,
            std::optional<QBit> outputQubitIndex);
        // try to add logical qubit to circuit and assign it to physical qubit with
        // certain output permutation value
        void addQubit(QBit logicalQubitIndex, QBit physicalQubitIndex,
            std::optional<QBit> outputQubitIndex);

        QuantumComputation instantiate(const VariableAssignment& assignment) {
            QuantumComputation result(*this);
            result.instantiateInplace(assignment);
            return result;
        }
        void instantiateInplace(const VariableAssignment& assignment);

        void addVariable(const SymbolOrNumber& expr);

        template <typename... Vars> void addVariables(const Vars&... vars) {
            (addVariable(vars), ...);
        }

        [[nodiscard]] bool isVariableFree() const {
            return std::all_of(ops.begin(), ops.end(), [](const auto& op) {
                return !op->isSymbolicOperation();
            });
        }

        [[nodiscard]] const std::unordered_set<sym::Variable>& getVariables() const {
            return occuringVariables;
        }

        /**
         * @brief Invert the circuit
         * @details Inverts the circuit by inverting all operations and reversing the
         * order of the operations. Additionally, the initial layout and output
         * permutation are swapped. If the circuit has different initial
         * layout and output permutation sizes, the initial layout and output
         * permutation will not be swapped.
         */
        void invert() {
            for (auto& op : ops) {
                op->invert();
            }
            std::reverse(ops.begin(), ops.end());

            if (initialLayout.size() == outputPermutation.size()) {
                std::swap(initialLayout, outputPermutation);
            }
            else {
                std::cerr << "Warning: Inverting a circuit with different initial layout "
                    "and output permutation sizes. This is not supported yet.\n"
                    "The circuit will be inverted, but the initial layout and "
                    "output permutation will not be swapped.\n";
            }
        }

        /**
         * printing
         */
        virtual std::ostream& print(std::ostream& os) const;

        friend std::ostream& operator<<(std::ostream& os,
            const QuantumComputation& qc) {
            return qc.print(os);
        }

        static void printBin(std::size_t n, std::stringstream& ss);

        virtual std::ostream& printStatistics(std::ostream& os) const;

        std::ostream& printRegisters(std::ostream& os = std::cout) const;

        static std::ostream& printPermutation(const Permutation& permutation,
            std::ostream& os = std::cout);

        virtual void dump(const std::string& filename, Format format);
        virtual void dump(const std::string& filename);
        virtual void dump(std::ostream& of, Format format) {
            dump(std::move(of), format);
        }
        virtual void dump(std::ostream&& of, Format format);
        void dumpOpenQASM2(std::ostream& of) { dumpOpenQASM(of, false); }
        void dumpOpenQASM3(std::ostream& of) { dumpOpenQASM(of, true); }
        virtual void dumpOpenQASM(std::ostream& of, bool openQasm3);
        virtual void dumpOriginIR(std::ostream& of);
        /**
         * @brief Returns the OpenQASM representation of the circuit
         * @param qasm3 Whether to use OpenQASM 3.0 or 2.0
         * @return The OpenQASM representation of the circuit
         */
        [[nodiscard]] std::string toQASM(const bool qasm3 = true) 
        {
            std::stringstream ss;
            dumpOpenQASM(ss, qasm3);
            return ss.str();
        }

        [[nodiscard]] std::string toOriginIR()
        {
            std::stringstream ss;
            dumpOriginIR(ss);
            return ss.str();
        }

        // this convenience method allows to turn a circuit into a compound operation.
        std::unique_ptr<CompoundOperation> asCompoundOperation() {
            return std::make_unique<CompoundOperation>(std::move(ops));
        }

        // this convenience method allows to turn a circuit into an operation.
        std::unique_ptr<Operation> asOperation() {
            if (ops.empty()) {
                return {};
            }
            if (ops.size() == 1) {
                auto op = std::move(ops.front());
                ops.clear();
                return op;
            }
            return asCompoundOperation();
        }

        virtual void reset() {
            ops.clear();
            nqubits = 0;
            nclassics = 0;
            nancillae = 0;
            qregs.clear();
            cregs.clear();
            ancregs.clear();
            initialLayout.clear();
            outputPermutation.clear();
        }

        /**
         * Pass-Through
         */

         // Iterators (pass-through)
        auto begin() noexcept { return ops.begin(); }
        [[nodiscard]] auto begin() const noexcept { return ops.begin(); }
        [[nodiscard]] auto cbegin() const noexcept { return ops.cbegin(); }
        auto end() noexcept { return ops.end(); }
        [[nodiscard]] auto end() const noexcept { return ops.end(); }
        [[nodiscard]] auto cend() const noexcept { return ops.cend(); }
        auto rbegin() noexcept { return ops.rbegin(); }
        [[nodiscard]] auto rbegin() const noexcept { return ops.rbegin(); }
        [[nodiscard]] auto crbegin() const noexcept { return ops.crbegin(); }
        auto rend() noexcept { return ops.rend(); }
        [[nodiscard]] auto rend() const noexcept { return ops.rend(); }
        [[nodiscard]] auto crend() const noexcept { return ops.crend(); }

        // Capacity (pass-through)
        [[nodiscard]] bool empty() const noexcept { return ops.empty(); }
        [[nodiscard]] std::size_t size() const noexcept { return ops.size(); }
        // NOLINTNEXTLINE(readability-identifier-naming)
        [[nodiscard]] std::size_t max_size() const noexcept { return ops.max_size(); }
        [[nodiscard]] std::size_t capacity() const noexcept { return ops.capacity(); }

        void reserve(const std::size_t newCap) { ops.reserve(newCap); }
        // NOLINTNEXTLINE(readability-identifier-naming)
        void shrink_to_fit() { ops.shrink_to_fit(); }

        // Modifiers (pass-through)
        void clear() noexcept { ops.clear(); }
        // NOLINTNEXTLINE(readability-identifier-naming)
        void pop_back() { return ops.pop_back(); }
        void resize(std::size_t count) { ops.resize(count); }
        iterator erase(const_iterator pos) { return ops.erase(pos); }
        iterator erase(const_iterator first, const_iterator last) {
            return ops.erase(first, last);
        }

        // NOLINTNEXTLINE(readability-identifier-naming)
        template <class T> void push_back(const T& op) {
            if (!ops.empty() && !op.isControlled() && !ops.back()->isControlled()) {
                std::cerr << op.getName() << std::endl;
            }

            ops.push_back(std::make_unique<T>(op));
        }

        // NOLINTNEXTLINE(readability-identifier-naming)
        template <class T, class... Args> void emplace_back(Args&&... args) {
            ops.emplace_back(std::make_unique<T>(args...));
        }

        // NOLINTNEXTLINE(readability-identifier-naming)
        template <class T> void emplace_back(std::unique_ptr<T>& op) {
            ops.emplace_back(std::move(op));
        }

        // NOLINTNEXTLINE(readability-identifier-naming)
        template <class T> void emplace_back(std::unique_ptr<T>&& op) {
            ops.emplace_back(std::move(op));
        }

        template <class T> iterator insert(const_iterator pos, T&& op) {
            return ops.insert(pos, std::forward<T>(op));
        }

        [[nodiscard]] const auto& at(const std::size_t i) const { return ops.at(i); }
        [[nodiscard]] auto& at(const std::size_t i) { return ops.at(i); }
        [[nodiscard]] const auto& front() const { return ops.front(); }
        [[nodiscard]] const auto& back() const { return ops.back(); }

        // reverse
        void reverse() { std::reverse(ops.begin(), ops.end()); }
    };
} // namespace qc
