#pragma once

#include "Core/Utilities/Compiler/QASMCompiler/parser/Statement.hpp"

namespace qasm {
class CompilerError final : public std::exception {
public:
  std::string message{};
  std::shared_ptr<DebugInfo> debugInfo{};

  CompilerError(std::string msg, std::shared_ptr<DebugInfo> debug)
      : message(std::move(msg)), debugInfo(std::move(debug)) {}

  [[nodiscard]] std::string toString() const {
    std::stringstream ss{};
    ss << debugInfo->toString();

    auto parentDebugInfo = debugInfo->parent;
    while (parentDebugInfo != nullptr) {
      ss << "\n  (included from " << parentDebugInfo->toString() << ")";
      parentDebugInfo = parentDebugInfo->parent;
    }

    ss << ":\n" << message;

    return ss.str();
  }
};
} // namespace qasm

class ConstEvalError final : public std::exception {
public:
  std::string message{};

  explicit ConstEvalError(std::string msg) : message(std::move(msg)) {}

  [[nodiscard]] std::string toString() const {
    return "Constant Evaluation: " + message;
  }
};

class TypeCheckError final : public std::exception {
public:
  std::string message{};

  explicit TypeCheckError(std::string msg) : message(std::move(msg)) {}

  [[nodiscard]] std::string toString() const {
    return "Type Check Error: " + message;
  }
};
