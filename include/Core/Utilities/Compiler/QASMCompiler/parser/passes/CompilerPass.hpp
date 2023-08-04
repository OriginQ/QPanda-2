#pragma once

#include "Core/Utilities/Compiler/QASMCompiler/parser/Statement.hpp"

namespace qasm {
class CompilerPass {
public:
  virtual ~CompilerPass() = default;

  virtual void processStatement(Statement& statement) = 0;
};
} // namespace qasm
