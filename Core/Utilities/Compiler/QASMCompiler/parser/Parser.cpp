
#include "Core/Utilities/Compiler/QASMCompiler/parser/Parser.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/StdGates.hpp"

#include <regex>

namespace qasm {
void Parser::scan() {
  if (scanner.empty()) {
    throw std::runtime_error("No scanner available");
  }

  if (!scanner.top().scan() && scanner.size() > 1) {
    scanner.pop();
    if (includeDebugInfo) {
      includeDebugInfo = includeDebugInfo->parent;
    }
  }
}

std::shared_ptr<VersionDeclaration> Parser::parseVersionDeclaration() {
  auto const tBegin = expect(Token::Kind::OpenQasm);
  Token const versionToken = expect(Token::Kind::FloatLiteral);
  auto const tEnd = expect(Token::Kind::Semicolon);
  return std::make_shared<VersionDeclaration>(makeDebugInfo(tBegin, tEnd),
                                              versionToken.valReal);
}

std::vector<std::shared_ptr<Statement>> Parser::parseProgram() {
  std::vector<std::shared_ptr<Statement>> statements{};

  bool versionDeclarationAllowed = true;

  while (!isAtEnd()) {
    if (!scanner.top().isImplicitInclude) {
      // We allow a version declaration at the beginning of the file.
      if (current().kind == Token::Kind::OpenQasm) {
        if (!versionDeclarationAllowed) {
          error(current(),
                "Version declaration must be at the beginning of the file.");
        }
        statements.emplace_back(parseVersionDeclaration());
        versionDeclarationAllowed = false;
        continue;
      }
      // Once we encounter a non-comment token, we don't allow a version
      // declaration anymore.
      if (current().kind != Token::Kind::InitialLayout &&
          current().kind != Token::Kind::OutputPermutation) {
        versionDeclarationAllowed = false;
      }
    }

    statements.push_back(parseStatement());
  }
  return statements;
}

std::shared_ptr<Statement> Parser::parseStatement() {
  if (current().kind == Token::Kind::Include) {
    // We parse include and then continue in parseStatement, as the include
    // statement just adds a new file to the scanner.
    parseInclude();
  }

  if (current().kind == Token::Kind::Const) {
    scan();
    return parseDeclaration(true);
  }

  if (current().kind == Token::Kind::Int ||
      current().kind == Token::Kind::Uint ||
      current().kind == Token::Kind::Bit ||
      current().kind == Token::Kind::QBit ||
      current().kind == Token::Kind::Float ||
      current().kind == Token::Kind::Angle ||
      current().kind == Token::Kind::Bool ||
      current().kind == Token::Kind::Duration ||
      current().kind == Token::Kind::CReg ||
      current().kind == Token::Kind::Qreg) {
    return parseDeclaration(false);
  }

  if (current().kind == Token::Kind::InitialLayout) {
    const auto tBegin = current();
    scan();
    return std::make_shared<InitialLayout>(
        InitialLayout{makeDebugInfo(tBegin), parsePermutation(tBegin.str)});
  }
  if (current().kind == Token::Kind::OutputPermutation) {
    const auto tBegin = current();
    scan();
    return std::make_shared<OutputPermutation>(
        OutputPermutation{makeDebugInfo(tBegin), parsePermutation(tBegin.str)});
  }

  if (current().kind == Token::Kind::Gate) {
    return parseGateDefinition();
  }
  if (current().kind == Token::Kind::Opaque) {
    return parseOpaqueGateDefinition();
  }

  if (current().kind == Token::Kind::Identifier) {
    // switch for readability
    switch (peek().kind) {
    case Token::Kind::LBracket:
    case Token::Kind::Equals:
    case Token::Kind::PlusEquals:
    case Token::Kind::MinusEquals:
    case Token::Kind::AsteriskEquals:
    case Token::Kind::SlashEquals:
    case Token::Kind::AmpersandEquals:
    case Token::Kind::PipeEquals:
    case Token::Kind::TildeEquals:
    case Token::Kind::CaretEquals:
    case Token::Kind::LeftShitEquals:
    case Token::Kind::RightShiftEquals:
    case Token::Kind::PercentEquals:
    case Token::Kind::DoubleAsteriskEquals:
      return parseAssignmentStatement();
    default:
      break;
    }
  }

  if (current().kind == Token::Kind::If) {
    return parseIfStatement();
  }
  if (current().kind == Token::Kind::Measure) {
    return parseMeasureStatement();
  }

  if (auto quantumStatement = parseQuantumStatement();
      quantumStatement != nullptr) {
    return quantumStatement;
  }

  error(current(), "Expected statement, got '" + current().toString() + "'.");
}

std::shared_ptr<QuantumStatement> Parser::parseQuantumStatement() {
  if (current().kind == Token::Kind::Inv ||
      current().kind == Token::Kind::Pow ||
      current().kind == Token::Kind::Ctrl ||
      current().kind == Token::Kind::NegCtrl ||
      current().kind == Token::Kind::Identifier ||
      current().kind == Token::Kind::Gphase ||
      current().kind == Token::Kind::S) {
    // TODO: since we do not support classical function calls yet, we can assume
    // that this is a gate statement
    return parseGateCallStatement();
  }

  if (current().kind == Token::Kind::Reset) {
    return parseResetStatement();
  }

  if (current().kind == Token::Kind::Barrier) {
    return parseBarrierStatement();
  }

  return nullptr;
}

void Parser::parseInclude() {
  auto const tBegin = expect(Token::Kind::Include);
  auto filename = expect(Token::Kind::StringLiteral).str;
  auto const tEnd = expect(Token::Kind::Semicolon);

  // we need to make sure to report errors across includes
  includeDebugInfo = makeDebugInfo(tBegin, tEnd);

  // Here we add a new scanner to our stack and then continue with that one

  auto in = std::make_unique<std::ifstream>(filename, std::ifstream::in);
  std::unique_ptr<std::istream> is{nullptr};
  if (in->fail()) {
    if (filename == "stdgates.inc") {
      // stdgates.inc has already been included implicitly, so we just return
      return;
    }
    if (filename == "qelib1.inc") {
      is = std::make_unique<std::istringstream>(QE1LIB);
    } else {
      error(current(), "Failed to open file " + filename + ".");
    }
  } else {
    is = std::move(in);
  }

  scanner.emplace(std::move(is), filename);
  scan();
}

std::shared_ptr<AssignmentStatement> Parser::parseAssignmentStatement() {
  auto identifierToken = expect(Token::Kind::Identifier);
  auto identifier = std::make_shared<IdentifierExpression>(identifierToken.str);
  std::shared_ptr<Expression> indexExpression{nullptr};

  if (current().kind == Token::Kind::LBracket) {
    scan();
    indexExpression = parseExpression();
    expect(Token::Kind::RBracket);
  }

  if (current().kind == Token::Kind::LBracket) {
    error(current(), "Multidimensional indexing not supported yet.");
  }

  AssignmentStatement::Type type{};
  switch (current().kind) {
  case Token::Kind::Equals:
    type = AssignmentStatement::Type::Assignment;
    break;
  case Token::Kind::PlusEquals:
    type = AssignmentStatement::Type::PlusAssignment;
    break;
  case Token::Kind::MinusEquals:
    type = AssignmentStatement::Type::MinusAssignment;
    break;
  case Token::Kind::AsteriskEquals:
    type = AssignmentStatement::Type::TimesAssignment;
    break;
  case Token::Kind::SlashEquals:
    type = AssignmentStatement::Type::DivAssignment;
    break;
  case Token::Kind::AmpersandEquals:
    type = AssignmentStatement::Type::BitwiseAndAssignment;
    break;
  case Token::Kind::PipeEquals:
    type = AssignmentStatement::Type::BitwiseOrAssignment;
    break;
  case Token::Kind::TildeEquals:
    type = AssignmentStatement::Type::BitwiseNotAssignment;
    break;
  case Token::Kind::CaretEquals:
    type = AssignmentStatement::Type::BitwiseXorAssignment;
    break;
  case Token::Kind::LeftShitEquals:
    type = AssignmentStatement::Type::LeftShiftAssignment;
    break;
  case Token::Kind::RightShiftEquals:
    type = AssignmentStatement::Type::RightShiftAssignment;
    break;
  case Token::Kind::PercentEquals:
    type = AssignmentStatement::Type::ModuloAssignment;
    break;
  case Token::Kind::DoubleAsteriskEquals:
    type = AssignmentStatement::Type::PowerAssignment;
    break;
  default:
    error(current(), "Expected assignment operator");
  }

  scan();

  auto declarationExpression = parseDeclarationExpression();

  auto const tEnd = expect(Token::Kind::Semicolon);

  return std::make_shared<AssignmentStatement>(
      makeDebugInfo(identifierToken, tEnd), type, identifier, indexExpression,
      declarationExpression);
}

std::shared_ptr<AssignmentStatement> Parser::parseMeasureStatement() {
  auto const tBegin = expect(Token::Kind::Measure);

  auto gateOperand = parseGateOperand();

  expect(Token::Kind::Arrow);

  auto cbitIdentifier = std::make_shared<IdentifierExpression>(
      expect(Token::Kind::Identifier).str);
  std::shared_ptr<Expression> cbitIndexExpr{nullptr};
  if (current().kind == Token::Kind::LBracket) {
    scan();
    cbitIndexExpr = parseExpression();
    expect(Token::Kind::RBracket);
  }

  auto const tEnd = expect(Token::Kind::Semicolon);

  std::shared_ptr<Expression> const gateOperandExpr{
      std::make_shared<MeasureExpression>(gateOperand)};
  return std::make_shared<AssignmentStatement>(
      makeDebugInfo(tBegin, tEnd), AssignmentStatement::Type::Assignment,
      cbitIdentifier, cbitIndexExpr,
      std::make_shared<DeclarationExpression>(gateOperandExpr));
}

std::shared_ptr<ResetStatement> Parser::parseResetStatement() {
  auto const tBegin = expect(Token::Kind::Reset);

  auto operand = parseGateOperand();

  auto const tEnd = expect(Token::Kind::Semicolon);

  return std::make_shared<ResetStatement>(makeDebugInfo(tBegin, tEnd), operand);
}

std::shared_ptr<BarrierStatement> Parser::parseBarrierStatement() {
  auto const tBegin = expect(Token::Kind::Barrier);

  std::vector<std::shared_ptr<GateOperand>> operands{};
  while (current().kind != Token::Kind::Semicolon) {
    operands.push_back(parseGateOperand());
    if (current().kind != Token::Kind::Semicolon) {
      expect(Token::Kind::Comma);
    }
  }

  auto const tEnd = expect(Token::Kind::Semicolon);

  return std::make_shared<BarrierStatement>(makeDebugInfo(tBegin, tEnd),
                                            operands);
}

std::shared_ptr<IfStatement> Parser::parseIfStatement() {
  const auto tBegin = expect(Token::Kind::If);
  expect(Token::Kind::LParen, "after if keyword.");
  auto condition = parseExpression();
  expect(Token::Kind::RParen, "after if condition.");

  std::vector<std::shared_ptr<Statement>> const thenStatements =
      parseBlockOrStatement();
  std::vector<std::shared_ptr<Statement>> elseStatements;

  if (current().kind == Token::Kind::Else) {
    expect(Token::Kind::Else);

    elseStatements = parseBlockOrStatement();
  }

  const auto tEnd = last();

  return std::make_shared<IfStatement>(std::move(condition), thenStatements,
                                       elseStatements,
                                       makeDebugInfo(tBegin, tEnd));
}

std::vector<std::shared_ptr<Statement>> Parser::parseBlockOrStatement() {
  std::vector<std::shared_ptr<Statement>> statements;

  if (current().kind == Token::Kind::LBrace) {
    scan();

    while (!isAtEnd() && current().kind != Token::Kind::RBrace) {
      statements.push_back(parseStatement());
    }

    expect(Token::Kind::RBrace);
  } else {
    statements.push_back(parseStatement());
  }

  return statements;
}

std::shared_ptr<GateCallStatement> Parser::parseGateCallStatement() {
  auto const tBegin = current();
  std::vector<std::shared_ptr<GateModifier>> modifiers{};

  while (current().kind == Token::Kind::Inv ||
         current().kind == Token::Kind::Pow ||
         current().kind == Token::Kind::Ctrl ||
         current().kind == Token::Kind::NegCtrl) {
    modifiers.push_back(parseGateModifier());
    expect(Token::Kind::At);
  }

  bool operandsOptional = false;
  std::string identifier;
  if (current().kind == Token::Kind::Gphase) {
    scan();
    identifier = "gphase";
    operandsOptional = true;
  } else if (current().kind == Token::Kind::S) {
    scan();
    identifier = "s";
  } else {
    identifier = expect(Token::Kind::Identifier).str;
  }

  std::vector<std::shared_ptr<Expression>> arguments{};
  if (current().kind == Token::Kind::LParen) {
    scan();
    while (current().kind != Token::Kind::RParen) {
      arguments.push_back(parseExpression());
      if (current().kind != Token::Kind::RParen) {
        expect(Token::Kind::Comma);
      }
    }
    expect(Token::Kind::RParen);
  }

  if (current().kind == Token::Kind::LBracket) {
    // TODO: support designator
    error(current(), "Designator not yet supported for gate call statements");
  }

  std::vector<std::shared_ptr<GateOperand>> operands{};
  while (current().kind != Token::Kind::Semicolon) {
    operands.push_back(parseGateOperand());
    if (current().kind != Token::Kind::Semicolon) {
      expect(Token::Kind::Comma);
    }
  }

  if (!operandsOptional && operands.empty()) {
    // operands are only optional for gphase
    error(current(), "Expected gate operands");
  }

  auto const tEnd = expect(Token::Kind::Semicolon);

  return std::make_shared<GateCallStatement>(
      GateCallStatement{makeDebugInfo(tBegin, tEnd), std::move(identifier),
                        modifiers, arguments, operands});
}

std::shared_ptr<GateModifier> Parser::parseGateModifier() {
  if (current().kind == Token::Kind::Inv) {
    scan();
    return std::make_shared<InvGateModifier>(InvGateModifier{});
  }
  if (current().kind == Token::Kind::Pow) {
    scan();
    expect(Token::Kind::LParen);
    auto modifier =
        std::make_shared<PowGateModifier>(PowGateModifier{parseExpression()});
    expect(Token::Kind::RParen);
    return modifier;
  }
  if (current().kind == Token::Kind::Ctrl ||
      current().kind == Token::Kind::NegCtrl) {
    bool const ctrlType = current().kind == Token::Kind::Ctrl;
    scan();

    std::shared_ptr<Expression> expression{nullptr};
    if (current().kind == Token::Kind::LParen) {
      scan();
      expression = parseExpression();
      expect(Token::Kind::RParen);
    }

    return std::make_shared<CtrlGateModifier>(
        CtrlGateModifier(ctrlType, expression));
  }

  error(current(), "Expected gate modifier");
}

std::shared_ptr<GateOperand> Parser::parseGateOperand() {
  // TODO: support hardware qubits
  const auto identifier = expect(Token::Kind::Identifier);

  std::shared_ptr<Expression> expression{nullptr};
  if (current().kind == Token::Kind::LBracket) {
    scan();
    expression = parseExpression();
    expect(Token::Kind::RBracket);
  }

  return std::make_shared<GateOperand>(GateOperand{identifier.str, expression});
}

std::shared_ptr<Statement> Parser::parseDeclaration(bool isConst) {
  auto const tBegin = current();
  auto [type, isOldStyleDeclaration] = parseType();
  Token const identifier = expect(Token::Kind::Identifier);

  auto const name = identifier.str;

  if (current().kind == Token::Kind::LBracket) {
    if (isOldStyleDeclaration) {
      if (!type->allowsDesignator()) {
        error(current(), "Type does not allow designator");
      }
      // in this case, the designator expression is after the identifier
      auto const designator = parseTypeDesignator();
      type->setDesignator(designator);
    } else {
      error(current(), "In OpenQASM 3.0, the designator has been changed to "
                       "`type[designator] identifier;`");
    }
  }

  std::shared_ptr<DeclarationExpression> expression{nullptr};
  if (current().kind == Token::Kind::Equals) {
    scan();
    expression = parseDeclarationExpression();
  }

  auto const tEnd = expect(Token::Kind::Semicolon);

  auto statement = std::make_shared<DeclarationStatement>(DeclarationStatement{
      makeDebugInfo(tBegin, tEnd), isConst, type, name, expression});

  return statement;
}

std::shared_ptr<GateDeclaration> Parser::parseGateDefinition() {
  auto const tBegin = expect(Token::Kind::Gate);
  auto const identifier = expect(Token::Kind::Identifier);

  std::shared_ptr<IdentifierList> parameters{nullptr};
  if (current().kind == Token::Kind::LParen) {
    scan();
    parameters = parseIdentifierList();
    expect(Token::Kind::RParen);
  } else {
    parameters = std::make_shared<IdentifierList>(IdentifierList{});
  }

  const auto qubits = parseIdentifierList();

  std::vector<std::shared_ptr<QuantumStatement>> statements{};
  expect(Token::Kind::LBrace);
  while (current().kind != Token::Kind::RBrace) {
    statements.emplace_back(parseQuantumStatement());
  }
  auto const tEnd = expect(Token::Kind::RBrace);

  return std::make_shared<GateDeclaration>(
      GateDeclaration(makeDebugInfo(tBegin, tEnd), identifier.str, parameters,
                      qubits, statements));
}

std::shared_ptr<GateDeclaration> Parser::parseOpaqueGateDefinition() {
  auto const tBegin = expect(Token::Kind::Opaque);
  auto const identifier = expect(Token::Kind::Identifier);

  std::shared_ptr<IdentifierList> parameters{nullptr};
  if (current().kind == Token::Kind::LParen) {
    scan();
    parameters = parseIdentifierList();
    expect(Token::Kind::RParen);
  } else {
    parameters = std::make_shared<IdentifierList>(IdentifierList{});
  }

  const auto qubits = parseIdentifierList();

  auto const tEnd = expect(Token::Kind::Semicolon);

  return std::make_shared<GateDeclaration>(
      GateDeclaration(makeDebugInfo(tBegin, tEnd), identifier.str, parameters,
                      qubits, {}, true));
}

std::shared_ptr<DeclarationExpression> Parser::parseDeclarationExpression() {
  if (current().kind == Token::Kind::Measure) {
    return std::make_shared<DeclarationExpression>(
        DeclarationExpression{parseMeasureExpression()});
  }

  if (current().kind == Token::Kind::LBracket) {
    error(current(), "Array expressions not supported yet");
  }

  return std::make_shared<DeclarationExpression>(
      DeclarationExpression{parseExpression()});
}

std::shared_ptr<MeasureExpression> Parser::parseMeasureExpression() {
  expect(Token::Kind::Measure);
  auto const gateOperand = parseGateOperand();
  return std::make_shared<MeasureExpression>(MeasureExpression{gateOperand});
}

std::shared_ptr<Expression> Parser::exponentiation() {
  switch (current().kind) {
  case Token::Kind::Minus: {
    scan();
    const auto x = exponentiation();
    return std::make_shared<UnaryExpression>(
        UnaryExpression{UnaryExpression::Op::Negate, x});
  }
  case Token::Kind::FloatLiteral: {
    const auto val = current().valReal;
    scan();
    return std::make_shared<Constant>(Constant{val});
  }
  case Token::Kind::IntegerLiteral: {
    auto const val = current().val;
    auto const isSigned = current().isSigned;
    scan();
    return std::make_shared<Constant>(Constant{val, isSigned});
  }
  case Token::Kind::Identifier: {
    auto const str = current().str;
    scan();
    return std::make_shared<IdentifierExpression>(IdentifierExpression{str});
  }
  case Token::Kind::False: {
    scan();
    return std::make_shared<Constant>(false);
  }
  case Token::Kind::True: {
    scan();
    return std::make_shared<Constant>(true);
  }
  case Token::Kind::LParen: {
    scan();
    auto x = parseExpression();
    expect(Token::Kind::RParen);
    return x;
  }
  case Token::Kind::Sin:
  case Token::Kind::Cos:
  case Token::Kind::Tan:
  case Token::Kind::Exp:
  case Token::Kind::Ln:
  case Token::Kind::Sqrt: {
    UnaryExpression::Op op = UnaryExpression::Op::Sin;
    switch (current().kind) {
    case Token::Kind::Sin:
      op = UnaryExpression::Op::Sin;
      break;
    case Token::Kind::Cos:
      op = UnaryExpression::Op::Cos;
      break;
    case Token::Kind::Tan:
      op = UnaryExpression::Op::Tan;
      break;
    case Token::Kind::Exp:
      op = UnaryExpression::Op::Exp;
      break;
    case Token::Kind::Ln:
      op = UnaryExpression::Op::Ln;
      break;
    case Token::Kind::Sqrt:
      op = UnaryExpression::Op::Sqrt;
      break;
    default:
      error(current(), "Expected unary operator");
    }
    scan();
    expect(Token::Kind::LParen);
    const auto x = parseExpression();
    expect(Token::Kind::RParen);
    return std::make_shared<UnaryExpression>(UnaryExpression{op, x});
  }
  default: {
    error(current(), "Expected expression, got " + current().toString() + ".");
  }
  }
}

std::shared_ptr<Expression> Parser::factor() {
  auto x = exponentiation();
  while (current().kind == Token::Kind::Caret) {
    scan();
    const auto y = exponentiation();
    x = std::make_shared<BinaryExpression>(
        BinaryExpression{BinaryExpression::Op::Power, x, y});
  }
  return x;
}

std::shared_ptr<Expression> Parser::term() {
  auto x = factor();
  while (current().kind == Token::Kind::Asterisk ||
         current().kind == Token::Kind::Slash) {
    auto const op = current().kind == Token::Kind::Asterisk
                        ? BinaryExpression::Op::Multiply
                        : BinaryExpression::Op::Divide;
    scan();
    const auto y = factor();
    x = std::make_shared<BinaryExpression>(BinaryExpression{op, x, y});
  }
  return x;
}

std::shared_ptr<Expression> Parser::comparison() {
  auto x = term();
  while (current().kind == Token::Kind::DoubleEquals ||
         current().kind == Token::Kind::NotEquals ||
         current().kind == Token::Kind::LessThan ||
         current().kind == Token::Kind::GreaterThan ||
         current().kind == Token::Kind::LessThanEquals ||
         current().kind == Token::Kind::GreaterThanEquals) {
    BinaryExpression::Op op = BinaryExpression::Op::Equal;
    switch (current().kind) {
    case Token::Kind::DoubleEquals:
      op = BinaryExpression::Op::Equal;
      break;
    case Token::Kind::NotEquals:
      op = BinaryExpression::Op::NotEqual;
      break;
    case Token::Kind::LessThan:
      op = BinaryExpression::Op::LessThan;
      break;
    case Token::Kind::GreaterThan:
      op = BinaryExpression::Op::GreaterThan;
      break;
    case Token::Kind::LessThanEquals:
      op = BinaryExpression::Op::LessThanOrEqual;
      break;
    case Token::Kind::GreaterThanEquals:
      op = BinaryExpression::Op::GreaterThanOrEqual;
      break;
    default:
      error(current(), "Expected comparison operator");
    }
    scan();
    const auto y = term();
    x = std::make_shared<BinaryExpression>(BinaryExpression{op, x, y});
  }
  return x;
}

std::shared_ptr<Expression> Parser::parseExpression() {
  std::shared_ptr<Expression> x{};
  if (current().kind == Token::Kind::Minus) {
    scan();
    x = std::make_shared<UnaryExpression>(
        UnaryExpression{UnaryExpression::Op::Negate, term()});
  } else if (current().kind == Token::Kind::ExclamationPoint) {
    scan();
    x = std::make_shared<UnaryExpression>(
        UnaryExpression{UnaryExpression::Op::LogicalNot, term()});
  } else if (current().kind == Token::Kind::Tilde) {
    scan();
    x = std::make_shared<UnaryExpression>(
        UnaryExpression{UnaryExpression::Op::BitwiseNot, term()});
  } else {
    x = comparison();
  }

  while (current().kind == Token::Kind::Plus ||
         current().kind == Token::Kind::Minus) {
    auto const op = current().kind == Token::Kind::Plus
                        ? BinaryExpression::Op::Add
                        : BinaryExpression::Op::Subtract;
    scan();
    const auto y = comparison();
    x = std::make_shared<BinaryExpression>(BinaryExpression{op, x, y});
  }

  return x;
}

std::shared_ptr<IdentifierList> Parser::parseIdentifierList() {
  std::vector<std::shared_ptr<IdentifierExpression>> identifierList{};

  identifierList.emplace_back(std::make_shared<IdentifierExpression>(
      IdentifierExpression{expect(Token::Kind::Identifier).str}));

  while (current().kind == Token::Kind::Comma) {
    scan();
    identifierList.emplace_back(std::make_shared<IdentifierExpression>(
        IdentifierExpression{expect(Token::Kind::Identifier).str}));
  }

  return std::make_shared<IdentifierList>(IdentifierList{identifierList});
}

std::pair<std::shared_ptr<TypeExpr>, bool> Parser::parseType() {
  std::shared_ptr<TypeExpr> type;
  bool isOldStyleDeclaration = false;

  switch (current().kind) {
  case Token::Kind::CReg:
    type = DesignatedType<std::shared_ptr<Expression>>::getBitTy(nullptr);
    isOldStyleDeclaration = true;
    break;
  case Token::Kind::Qreg:
    type = DesignatedType<std::shared_ptr<Expression>>::getQubitTy(nullptr);
    isOldStyleDeclaration = true;
    break;
  case Token::Kind::Int:
    type = DesignatedType<std::shared_ptr<Expression>>::getIntTy(nullptr);
    break;
  case Token::Kind::Uint:
    type = DesignatedType<std::shared_ptr<Expression>>::getUintTy(nullptr);
    break;
  case Token::Kind::Bit:
    type = DesignatedType<std::shared_ptr<Expression>>::getBitTy(nullptr);
    break;
  case Token::Kind::QBit:
    type = DesignatedType<std::shared_ptr<Expression>>::getQubitTy(nullptr);
    break;
  case Token::Kind::Float:
    type = DesignatedType<std::shared_ptr<Expression>>::getFloatTy(nullptr);
    break;
  case Token::Kind::Angle:
    type = DesignatedType<std::shared_ptr<Expression>>::getAngleTy(nullptr);
    break;
  case Token::Kind::Bool:
    type = UnsizedType<std::shared_ptr<Expression>>::getBoolTy();
    break;
  case Token::Kind::Duration:
    type = UnsizedType<std::shared_ptr<Expression>>::getDurationTy();
    break;
  default:
    error(peek(), "Expected type");
  }

  scan();

  if (!isOldStyleDeclaration && current().kind == Token::Kind::LBracket) {
    if (!type->allowsDesignator()) {
      error(peek(), "Type does not allow designator");
    }
    auto designator = parseTypeDesignator();
    type->setDesignator(std::move(designator));
    return std::pair{std::move(type), isOldStyleDeclaration};
  }

  return std::pair{std::move(type), isOldStyleDeclaration};
}

std::shared_ptr<Expression> Parser::parseTypeDesignator() {
  expect(Token::Kind::LBracket);
  auto expr = parseExpression();
  expect(Token::Kind::RBracket);
  return expr;
}

qc::Permutation Parser::parsePermutation(std::string s) {
  qc::Permutation permutation{};
  static const auto QUBIT_REGEX = std::regex("\\d+");
  qc::QBit logicalQubit = 0;
  for (std::smatch m; std::regex_search(s, m, QUBIT_REGEX); s = m.suffix()) {
    auto physicalQubit = static_cast<qc::QBit>(std::stoul(m.str()));
    permutation.insert({physicalQubit, logicalQubit});
    ++logicalQubit;
  }
  return permutation;
}
} // namespace qasm
