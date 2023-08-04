
#include "Core/Utilities/Compiler/operations/Operation.hpp"

#include <algorithm>
#include <cassert>

namespace qc {

    std::ostream& Operation::printParameters(std::ostream& os) const {
        if (isClassicControlledOperation()) {
            os << "  c[" << parameter[0];
            if (parameter[1] != 1) {
                os << " ... " << (parameter[0] + parameter[1] - 1);
            }
            os << "] == " << parameter[2];
            return os;
        }

        bool isZero = true;
        for (const auto& p : parameter) {
            if (p != static_cast<fp>(0)) {
                isZero = false;
                break;
            }
        }
        if (!isZero) {
            os << "  p: (" << parameter[0] << ") ";
            for (size_t j = 1; j < parameter.size(); ++j) {
                isZero = true;
                for (size_t i = j; i < parameter.size(); ++i) {
                    if (parameter.at(i) != static_cast<fp>(0)) {
                        isZero = false;
                        break;
                    }
                }
                if (isZero) {
                    break;
                }
                os << "(" << parameter.at(j) << ") ";
            }
        }

        return os;
    }

    std::ostream& Operation::print(std::ostream& os, const Permutation& permutation,
        [[maybe_unused]] const std::size_t prefixWidth,
        const std::size_t nqubits) const {
        const auto precBefore = std::cout.precision(20);
        const auto& actualControls = permutation.apply(getControls());
        const auto& actualTargets = permutation.apply(getTargets());

        for (std::size_t i = 0; i < nqubits; ++i) {
            const auto q = static_cast<QBit>(i);
            if (std::find(actualTargets.cbegin(), actualTargets.cend(), q) !=
                actualTargets.cend()) {
                if (type == otClassicControlled) {
                    const auto reducedName = name.substr(2);
                    os << "\033[1m\033[35m" << std::setw(4) << reducedName;
                }
                else if (type == otBarrier) {
                    os << "\033[1m\033[32m" << std::setw(4) << shortName(type);
                }
                else {
                    os << "\033[1m\033[36m" << std::setw(4) << shortName(type);
                }
                os << "\033[0m";
                continue;
            }

            if (const auto it =
                std::find(actualControls.cbegin(), actualControls.cend(), q);
                it != actualControls.cend()) {
                if (it->type == Control::Type::Pos) {
                    os << "\033[32m";
                }
                else {
                    os << "\033[31m";
                }
                os << std::setw(4) << "c" << "\033[0m";
                continue;
            }

            os << std::setw(4) << "|" << "\033[0m";
        }

        printParameters(os);

        std::cout.precision(precBefore);

        return os;
    }

    bool Operation::equals(const Operation& op, const Permutation& perm1,
        const Permutation& perm2) const {
        // check type
        if (getType() != op.getType()) {
            return false;
        }

        // check number of controls
        const auto nc1 = getNcontrols();
        const auto nc2 = op.getNcontrols();
        if (nc1 != nc2) {
            return false;
        }

        // check parameters
        const auto& param1 = getParameter();
        const auto& param2 = op.getParameter();
        if (param1 != param2) {
            return false;
        }

        // check controls
        if (nc1 != 0U) {
            Controls controls1{};
            if (perm1.empty()) {
                controls1 = getControls();
            }
            else {
                for (const auto& control : getControls()) {
                    controls1.emplace(perm1.at(control.qubit), control.type);
                }
            }

            Controls controls2{};
            if (perm2.empty()) {
                controls2 = op.getControls();
            }
            else {
                for (const auto& control : op.getControls()) {
                    controls2.emplace(perm2.at(control.qubit), control.type);
                }
            }

            if (controls1 != controls2) {
                return false;
            }
        }

        // check targets
        std::set<QBit> targets1{};
        if (perm1.empty()) {
            targets1 = { getTargets().begin(), getTargets().end() };
        }
        else {
            for (const auto& target : getTargets()) {
                targets1.emplace(perm1.at(target));
            }
        }

        std::set<QBit> targets2{};
        if (perm2.empty()) {
            targets2 = { op.getTargets().begin(), op.getTargets().end() };
        }
        else {
            for (const auto& target : op.getTargets()) {
                targets2.emplace(perm2.at(target));
            }
        }

        return targets1 == targets2;
    }

    void Operation::addDepthContribution(std::vector<std::size_t>& depths) const {
        if (type == otBarrier) {
            return;
        }

        std::size_t maxDepth = 0;
        for (const auto& target : getTargets()) {
            maxDepth = std::max(maxDepth, depths[target]);
        }
        for (const auto& control : getControls()) {
            maxDepth = std::max(maxDepth, depths[control.qubit]);
        }
        maxDepth += 1;
        for (const auto& target : getTargets()) {
            depths[target] = maxDepth;
        }
        for (const auto& control : getControls()) {
            depths[control.qubit] = maxDepth;
        }
    }

    void Operation::apply(const Permutation& permutation) {
        getTargets() = permutation.apply(getTargets());
        getControls() = permutation.apply(getControls());
    }
    bool Operation::isOrigin1levelCombineGateType()const { return false; }
    void Operation::dumpOrigin1levelCombineGateType(std::ostream& of)const{}

} // namespace qc
