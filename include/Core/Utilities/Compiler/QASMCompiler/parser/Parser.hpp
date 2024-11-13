/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#pragma once

#include "Core/Utilities/Compiler/QASMCompiler/parser/Exception.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/Scanner.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/Statement.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/StdGates.hpp"

#include <iostream>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <vector>

namespace qasm {
class Parser {
  struct ScannerState {
  private:
    std::unique_ptr<std::istream> is;

  public:
    Token last{0, 0};
    Token t{0, 0};
    Token next{0, 0};
    std::unique_ptr<Scanner> scanner;
    std::optional<std::string> filename;
    bool isImplicitInclude;

    bool scan() {
      last = t;
      t = next;
      next = scanner->next();

      return t.kind != Token::Kind::Eof;
    }

    explicit ScannerState(
        std::istream* in,
        std::optional<std::string> debugFilename = std::nullopt,
        const bool implicitInclude = false)
        : scanner(std::make_unique<Scanner>(in)),
          filename(std::move(debugFilename)),
          isImplicitInclude(implicitInclude) {
      scan();
    }

    explicit ScannerState(
        std::unique_ptr<std::istream> in,
        std::optional<std::string> debugFilename = std::nullopt,
        const bool implicitInclude = false)
        : is(std::move(in)), scanner(std::make_unique<Scanner>(is.get())),
          filename(std::move(debugFilename)),
          isImplicitInclude(implicitInclude) {
      scan();
    }
  };

  std::stack<ScannerState> scanner{};
  std::shared_ptr<DebugInfo> includeDebugInfo{nullptr};

  [[noreturn]] void error(const Token& token, const std::string& msg) {
    std::cerr << "Error at line " << token.line << ", column " << token.col
              << ": " << msg << '\n';
    throw CompilerError(msg, makeDebugInfo(token));
  }

  [[nodiscard]] Token last() const {
    if (scanner.empty()) {
      throw std::runtime_error("No scanner available");
    }
    return scanner.top().last;
  }

  [[nodiscard]] Token current() const {
    if (scanner.empty()) {
      throw std::runtime_error("No scanner available");
    }
    return scanner.top().t;
  }

  [[nodiscard]] Token peek() const {
    if (scanner.empty()) {
      throw std::runtime_error("No scanner available");
    }
    return scanner.top().next;
  }

  Token expect(const Token::Kind& expected,
               const std::optional<std::string>& context = std::nullopt) {
    if (current().kind != expected) {
      std::string message = "Expected '" + Token::kindToString(expected) +
                            "', got '" + Token::kindToString(current().kind) +
                            "'.";
      if (context.has_value()) {
        message += " " + context.value();
      }
      error(current(), message);
    }

    auto token = current();
    scan();
    return token;
  }

public:
  explicit Parser(std::istream* is, bool implicitlyIncludeStdgates = true) {
    scanner.emplace(is);
    scan();
    if (implicitlyIncludeStdgates) {
      scanner.emplace(std::make_unique<std::istringstream>(STDGATES),
                      "stdgates.inc", true);
      scan();
    }
  }

  virtual ~Parser() = default;

  std::shared_ptr<VersionDeclaration> parseVersionDeclaration();

  std::vector<std::shared_ptr<Statement>> parseProgram();

  std::shared_ptr<Statement> parseStatement();

  std::shared_ptr<QuantumStatement> parseQuantumStatement();

  void parseInclude();

  std::shared_ptr<AssignmentStatement> parseAssignmentStatement();

  std::shared_ptr<AssignmentStatement> parseMeasureStatement();

  std::shared_ptr<ResetStatement> parseResetStatement();

  std::shared_ptr<BarrierStatement> parseBarrierStatement();

  std::shared_ptr<Statement> parseDeclaration(bool isConst);

  std::shared_ptr<GateDeclaration> parseGateDefinition();

  std::shared_ptr<GateDeclaration> parseOpaqueGateDefinition();

  std::shared_ptr<GateCallStatement> parseGateCallStatement();

  std::shared_ptr<GateModifier> parseGateModifier();

  std::shared_ptr<GateOperand> parseGateOperand();

  std::shared_ptr<DeclarationExpression> parseDeclarationExpression();

  std::shared_ptr<MeasureExpression> parseMeasureExpression();

  std::shared_ptr<Expression> exponentiation();

  std::shared_ptr<Expression> factor();

  std::shared_ptr<Expression> term();

  std::shared_ptr<Expression> comparison();

  std::shared_ptr<Expression> parseExpression();

  std::shared_ptr<IdentifierList> parseIdentifierList();

  std::pair<std::shared_ptr<TypeExpr>, bool> parseType();

  std::shared_ptr<Expression> parseTypeDesignator();

  static qc::Permutation parsePermutation(std::string s);

  void scan();

  std::shared_ptr<DebugInfo> makeDebugInfo(Token const& begin,
                                           Token const& /*end*/) {
    // Parameter `end` is currently not used.
    return std::make_shared<DebugInfo>(
        begin.line, begin.col, scanner.top().filename.value_or("<input>"),
        includeDebugInfo);
  }

  std::shared_ptr<DebugInfo> makeDebugInfo(Token const& token) {
    return std::make_shared<DebugInfo>(
        token.line, token.col, scanner.top().filename.value_or("<input>"),
        includeDebugInfo);
  }

  [[nodiscard]] bool isAtEnd() const {
    return current().kind == Token::Kind::Eof;
  }
  std::shared_ptr<IfStatement> parseIfStatement();
  std::vector<std::shared_ptr<Statement>> parseBlockOrStatement();
};

} // namespace qasm
