
#include "Core/Utilities/Compiler/QuantumComputation.hpp"
#include "Core/Utilities/Compiler/operations/Operation.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/Exception.hpp"

#include "Core/Utilities/Compiler/QASMCompiler/parser/Gate.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/NestedEnvironment.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/Parser.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/Statement.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/StdGates.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/passes/ConstEvalPass.hpp"
#include "Core/Utilities/Compiler/QASMCompiler/parser/passes/TypeCheckPass.hpp"

#include <iostream>
#include <utility>

using namespace qasm;
using const_eval::ConstEvalPass;
using const_eval::ConstEvalValue;
using type_checking::InferredType;
using type_checking::TypeCheckPass;

class OpenQasmParser final : public InstVisitor {
  ConstEvalPass constEvalPass;
  TypeCheckPass typeCheckPass;

  NestedEnvironment<std::shared_ptr<DeclarationStatement>> declarations{};
  qc::QuantumComputation* qc{};

  std::vector<std::unique_ptr<qc::Operation>> ops{};

  std::map<std::string, std::shared_ptr<Gate>> gates = STANDARD_GATES;

  bool openQASM2CompatMode{false};

  qc::Permutation initialLayout{};
  qc::Permutation outputPermutation{};

  [[noreturn]] static void error(const std::string& message,
                                 const std::shared_ptr<DebugInfo>& debugInfo) {
    throw CompilerError(message, debugInfo);
  }

  static std::map<std::string, std::pair<ConstEvalValue, InferredType>>
  initializeBuiltins() {
    std::map<std::string, std::pair<ConstEvalValue, InferredType>> builtins{};

    InferredType const floatTy{std::dynamic_pointer_cast<ResolvedType>(
        std::make_shared<DesignatedType<uint64_t>>(Float, 64))};

    builtins.emplace("pi", std::pair{ConstEvalValue(qc::qcPI), floatTy});
    builtins.emplace("π", std::pair{ConstEvalValue(qc::qcPI), floatTy});
    builtins.emplace("tau", std::pair{ConstEvalValue(qc::TAU), floatTy});
    builtins.emplace("τ", std::pair{ConstEvalValue(qc::TAU), floatTy});
    builtins.emplace("euler", std::pair{ConstEvalValue(qc::E), floatTy});
    //builtins.emplace("ℇ", std::pair{ConstEvalValue(qc::E), floatTy});

    return builtins;
  }

  static void
  translateGateOperand(const std::shared_ptr<GateOperand>& gateOperand,
                       std::vector<qc::QBit>& qubits,
                       const qc::QuantumRegisterMap& qregs,
                       const std::shared_ptr<DebugInfo>& debugInfo) {
    translateGateOperand(gateOperand->identifier, gateOperand->expression,
                         qubits, qregs, debugInfo);
  }

  static void
  translateGateOperand(const std::string& gateIdentifier,
                       const std::shared_ptr<Expression>& indexExpr,
                       std::vector<qc::QBit>& qubits,
                       const qc::QuantumRegisterMap& qregs,
                       const std::shared_ptr<DebugInfo>& debugInfo) {
    const auto qubitIter = qregs.find(gateIdentifier);
    if (qubitIter == qregs.end()) {
      error("Usage of unknown quantum register.", debugInfo);
    }
    auto qubit = qubitIter->second;

    if (indexExpr != nullptr) {
      const auto result = evaluatePositiveConstant(indexExpr, debugInfo);

      if (result >= qubit.second) {
        error("Index expression must be smaller than the width of the "
              "quantum register.",
              debugInfo);
      }
      qubit.first += static_cast<qc::QBit>(result);
      qubit.second = 1;
    }

    for (uint64_t i = 0; i < qubit.second; ++i) {
      qubits.emplace_back(qubit.first + i);
    }
  }

  void translateBitOperand(const std::string& bitIdentifier,
                           const std::shared_ptr<Expression>& indexExpr,
                           std::vector<qc::Bit>& bits,
                           const std::shared_ptr<DebugInfo>& debugInfo) const {
    const auto iter = qc->getCregs().find(bitIdentifier);
    if (iter == qc->getCregs().end()) {
      error("Usage of unknown classical register.", debugInfo);
    }
    auto creg = iter->second;

    if (indexExpr != nullptr) {
      const auto index = evaluatePositiveConstant(indexExpr, debugInfo);
      if (index >= creg.second) {
        error("Index expression must be smaller than the width of the "
              "classical register.",
              debugInfo);
      }

      creg.first += index;
      creg.second = 1;
    }

    for (uint64_t i = 0; i < creg.second; ++i) {
      bits.emplace_back(creg.first + i);
    }
  }

  static uint64_t
  evaluatePositiveConstant(const std::shared_ptr<Expression>& expr,
                           const std::shared_ptr<DebugInfo>& debugInfo,
                           const uint64_t defaultValue = 0) {
    if (expr == nullptr) {
      return defaultValue;
    }

    const auto constInt = std::dynamic_pointer_cast<Constant>(expr);
    if (!constInt) {
      error("Expected a constant integer expression.", debugInfo);
    }

    return constInt->getUInt();
  }

public:
  explicit OpenQasmParser(qc::QuantumComputation* quantumComputation)
      : typeCheckPass(&constEvalPass), qc(quantumComputation) {
    for (const auto& [identifier, builtin] : initializeBuiltins()) {
      constEvalPass.addConst(identifier, builtin.first);
      typeCheckPass.addBuiltin(identifier, builtin.second);
    }
  }

  ~OpenQasmParser() override = default;

  void visitProgram(const std::vector<std::shared_ptr<Statement>>& program) {
    // TODO: in the future, don't exit early, but collect all errors
    // To do this, we need to insert make sure that erroneous declarations
    // actually insert a dummy entry; also, we need to synchronize to the next
    // semicolon, to make sure we don't do some weird stuff and report false
    // errors.
    for (const auto& statement : program) {
      try {
        constEvalPass.processStatement(*statement);
        typeCheckPass.processStatement(*statement);
        statement->accept(this);
      } catch (CompilerError& e) {
        std::cerr << e.toString() << '\n';
        throw;
      }
    }

    // Finally, if we have a initial layout and output permutation specified,
    // apply them.
    if (!initialLayout.empty()) {
      qc->initialLayout = initialLayout;
    }
    if (!outputPermutation.empty()) {
      qc->outputPermutation = outputPermutation;
    }
  }

  void visitVersionDeclaration(
      const std::shared_ptr<VersionDeclaration> versionDeclaration) override {
    if (versionDeclaration->version < 3) {
      openQASM2CompatMode = true;
    }
  }

  void visitDeclarationStatement(const std::shared_ptr<DeclarationStatement>
                                     declarationStatement) override {
    const auto identifier = declarationStatement->identifier;
    if (declarations.find(identifier).has_value()) {
      // TODO: show the location of the previous declaration
      error("Identifier '" + identifier + "' already declared.",
            declarationStatement->debugInfo);
    }

    std::shared_ptr<ResolvedType> const ty =
        std::get<1>(declarationStatement->type);

    if (const auto sizedTy =
            std::dynamic_pointer_cast<DesignatedType<uint64_t>>(ty)) {
      const auto designator = sizedTy->getDesignator();
      switch (sizedTy->type) {
      case QBit:
        qc->addQubitRegister(designator, identifier);
        break;
      case Bit:
      case Int:
      case Uint:
        qc->addClassicalRegister(designator, identifier);
        break;
      case Float:
        // not adding to qc
        break;
      case Angle:
        error("Angle type is currently not supported.",
              declarationStatement->debugInfo);
      }
    } else {
      error("Only sized types are supported.", declarationStatement->debugInfo);
    }
    declarations.emplace(identifier, declarationStatement);

    if (declarationStatement->expression == nullptr) {
      // value is uninitialized
      return;
    }
    if (const auto measureExpression =
            std::dynamic_pointer_cast<MeasureExpression>(
                declarationStatement->expression->expression)) {
      assert(!declarationStatement->isConst &&
             "Type check pass should catch this");
      visitMeasureAssignment(identifier, nullptr, measureExpression,
                             declarationStatement->debugInfo);
      return;
    }
    if (declarationStatement->isConst) {
      // nothing to do
      return;
    }

    error("Only measure statements are supported for initialization.",
          declarationStatement->debugInfo);
  }

  void visitAssignmentStatement(
      const std::shared_ptr<AssignmentStatement> assignmentStatement) override {
    const auto identifier = assignmentStatement->identifier->identifier;
    const auto declaration = declarations.find(identifier);
    assert(declaration.has_value() && "Checked by type check pass");
    assert(!declaration->get()->isConst && "Checked by type check pass");

    if (const auto measureExpression =
            std::dynamic_pointer_cast<MeasureExpression>(
                assignmentStatement->expression->expression)) {
      visitMeasureAssignment(identifier, assignmentStatement->indexExpression,
                             measureExpression, assignmentStatement->debugInfo);
      return;
    }

    // In the future, handle classical computation.
    error("Classical computation not supported.",
          assignmentStatement->debugInfo);
  }

  void
  visitInitialLayout(const std::shared_ptr<InitialLayout> layout) override {
    if (!initialLayout.empty()) {
      error("Multiple initial layout specifications found.", layout->debugInfo);
    }
    initialLayout = layout->permutation;
  }

  void visitOutputPermutation(
      const std::shared_ptr<OutputPermutation> permutation) override {
    if (!outputPermutation.empty()) {
      error("Multiple output permutation specifications found.",
            permutation->debugInfo);
    }
    outputPermutation = permutation->permutation;
  }

  void visitGateStatement(
      const std::shared_ptr<GateDeclaration> gateStatement) override {
    auto identifier = gateStatement->identifier;
    if (gateStatement->isOpaque) {
      if (gates.find(identifier) == gates.end()) {
        // only builtin gates may be declared as opaque.
        error("Unsupported opaque gate '" + identifier + "'.",
              gateStatement->debugInfo);
      }

      return;
    }

    if (openQASM2CompatMode) {
      // we need to check if this is a standard gate
      identifier = parseGateIdentifierCompatMode(identifier).first;
    }

    if (auto prevDeclaration = gates.find(identifier);
        prevDeclaration != gates.end()) {
      if (std::dynamic_pointer_cast<StandardGate>(prevDeclaration->second)) {
        // we ignore redeclarations of standard gates
        return;
      }
      // TODO: print location of previous declaration
      error("Gate '" + identifier + "' already declared.",
            gateStatement->debugInfo);
    }

    const auto parameters = gateStatement->parameters;
    const auto qubits = gateStatement->qubits;

    // first we check that all parameters and qubits are unique
    std::vector<std::string> parameterIdentifiers{};
    for (const auto& parameter : parameters->identifiers) {
      if (std::find(parameterIdentifiers.begin(), parameterIdentifiers.end(),
                    parameter->identifier) != parameterIdentifiers.end()) {
        error("Parameter '" + parameter->identifier + "' already declared.",
              gateStatement->debugInfo);
      }
      parameterIdentifiers.emplace_back(parameter->identifier);
    }
    std::vector<std::string> qubitIdentifiers{};
    for (const auto& qubit : qubits->identifiers) {
      if (std::find(qubitIdentifiers.begin(), qubitIdentifiers.end(),
                    qubit->identifier) != qubitIdentifiers.end()) {
        error("QBit '" + qubit->identifier + "' already declared.",
              gateStatement->debugInfo);
      }
      qubitIdentifiers.emplace_back(qubit->identifier);
    }

    auto compoundGate = std::make_shared<CompoundGate>(CompoundGate(
        parameterIdentifiers, qubitIdentifiers, gateStatement->statements));

    gates.emplace(identifier, compoundGate);
  }

  void visitGateCallStatement(
      const std::shared_ptr<GateCallStatement> gateCallStatement) override {
    auto qregs = qc->getQregs();

    if (auto op = evaluateGateCall(
            gateCallStatement, gateCallStatement->identifier,
            gateCallStatement->arguments, gateCallStatement->operands, qregs);
        op != nullptr) {
      qc->emplace_back(std::move(op));
    }
  }

  std::unique_ptr<qc::Operation>
  evaluateGateCall(const std::shared_ptr<GateCallStatement>& gateCallStatement,
                   const std::string& identifier,
                   const std::vector<std::shared_ptr<Expression>>& parameters,
                   std::vector<std::shared_ptr<GateOperand>> targets,
                   const qc::QuantumRegisterMap& qregs) {
    auto iter = gates.find(identifier);
    std::shared_ptr<Gate> gate;
    size_t implicitControls{0};

    if (iter == gates.end()) {
      if (identifier == "mcx" || identifier == "mcx_gray" ||
          identifier == "mcx_vchain" || identifier == "mcx_recursive" ||
          identifier == "mcphase") {
        // we create a temp gate definition for these gates
        gate =
            getMcGateDefinition(identifier, gateCallStatement->operands.size(),
                                gateCallStatement->debugInfo);
      } else if (openQASM2CompatMode) {
        auto [updatedIdentifier, nControls] =
            parseGateIdentifierCompatMode(identifier);

        iter = gates.find(updatedIdentifier);
        if (iter == gates.end()) {
          error("Usage of unknown gate '" + identifier + "'.",
                gateCallStatement->debugInfo);
        }
        gate = iter->second;
        implicitControls = nControls;
      } else {
        error("Usage of unknown gate '" + identifier + "'.",
              gateCallStatement->debugInfo);
      }
    } else {
      gate = iter->second;
    }

    if (gate->getNParameters() != parameters.size()) {
      error("Gate '" + identifier + "' takes " +
                std::to_string(gate->getNParameters()) + " parameters, but " +
                std::to_string(parameters.size()) + " were supplied.",
            gateCallStatement->debugInfo);
    }

    // here we count the number of controls
    std::vector<std::pair<std::shared_ptr<GateOperand>, bool>> controls{};
    // since standard gates may define a number of control targets, we first
    // need to handle those
    size_t nControls{gate->getNControls() + implicitControls};
    if (targets.size() < nControls) {
      error("Gate '" + identifier + "' takes " + std::to_string(nControls) +
                " controls, but only " + std::to_string(targets.size()) +
                " qubits were supplied.",
            gateCallStatement->debugInfo);
    }

    for (size_t i = 0; i < nControls; ++i) {
      controls.emplace_back(targets[i], true);
    }

    bool invertOperation = false;
    for (const auto& modifier : gateCallStatement->modifiers) {
      if (auto ctrlModifier =
              std::dynamic_pointer_cast<CtrlGateModifier>(modifier);
          ctrlModifier != nullptr) {
        size_t const n = evaluatePositiveConstant(ctrlModifier->expression,
                                                  gateCallStatement->debugInfo,
                                                  /*defaultValue=*/1);
        if (targets.size() < n + nControls) {
          error("Gate '" + identifier + "' takes " +
                    std::to_string(n + nControls) + " controls, but only " +
                    std::to_string(targets.size()) + " were supplied.",
                gateCallStatement->debugInfo);
        }

        for (size_t i = 0; i < n; ++i) {
          controls.emplace_back(targets[nControls + i], ctrlModifier->ctrlType);
        }
        nControls += n;
      } else if (auto invModifier =
                     std::dynamic_pointer_cast<InvGateModifier>(modifier);
                 invModifier != nullptr) {
        // if we have an even number of inv modifiers, they cancel each other
        // out
        invertOperation = !invertOperation;
      } else {
        error("Only ctrl/negctrl/inv modifiers are supported.",
              gateCallStatement->debugInfo);
      }
    }
    targets.erase(targets.begin(),
                  targets.begin() + static_cast<int64_t>(nControls));

    if (gate->getNTargets() != targets.size()) {
      error("Gate '" + identifier + "' takes " +
                std::to_string(gate->getNTargets()) + " targets, but " +
                std::to_string(targets.size()) + " were supplied.",
            gateCallStatement->debugInfo);
    }

    // now evaluate all arguments; we only support const arguments.
    std::vector<qc::fp> evaluatedParameters{};
    for (const auto& param : parameters) {
      auto result = constEvalPass.visit(param);
      if (!result.has_value()) {
        error("Only const expressions are supported as gate parameters, but "
              "found '" +
                  param->getName() + "'.",
              gateCallStatement->debugInfo);
      }

      evaluatedParameters.emplace_back(result->toExpr()->asFP());
    }

    size_t broadcastingWidth{1};
    qc::Targets targetBits{};
    std::vector<size_t> targetBroadcastingIndices{};
    size_t i{0};
    for (const auto& target : targets) {
      qc::Targets t{};
      translateGateOperand(target, t, qregs, gateCallStatement->debugInfo);

      targetBits.emplace_back(t[0]);

      if (t.size() > 1) {
        if (broadcastingWidth != 1 && t.size() != broadcastingWidth) {
          error("When broadcasting, all registers must be of the same width.",
                gateCallStatement->debugInfo);
        }
        broadcastingWidth = t.size();

        targetBroadcastingIndices.emplace_back(i);
      }

      i++;
    }

    std::vector<qc::Control> controlBits{};
    std::vector<size_t> controlBroadcastingIndices{};
    i = 0;
    for (const auto& [control, type] : controls) {
      qc::Targets c{};
      translateGateOperand(control, c, qregs, gateCallStatement->debugInfo);

      controlBits.emplace_back(c[0], type ? qc::Control::Type::Pos
                                          : qc::Control::Type::Neg);

      if (c.size() > 1) {
        if (broadcastingWidth != 1 && c.size() != broadcastingWidth) {
          error("When broadcasting, all registers must be of the same width.",
                gateCallStatement->debugInfo);
        }
        broadcastingWidth = c.size();

        controlBroadcastingIndices.emplace_back(i);
      }

      i++;
    }

    // check if any of the bits are duplicate
    std::unordered_set<qc::QBit> allQubits;
    for (const auto& control : controlBits) {
      if (allQubits.find(control.qubit) != allQubits.end()) {
        error("Duplicate qubit in control list.", gateCallStatement->debugInfo);
      }
      allQubits.emplace(control.qubit);
    }
    for (const auto& qubit : targetBits) {
      if (allQubits.find(qubit) != allQubits.end()) {
        error("Duplicate qubit in target list.", gateCallStatement->debugInfo);
      }
      allQubits.emplace(qubit);
    }

    if (broadcastingWidth == 1) {
      return applyQuantumOperation(gate, targetBits, controlBits,
                                   evaluatedParameters, invertOperation,
                                   gateCallStatement->debugInfo);
    }

    // if we are broadcasting, we need to create a compound operation
    auto op = std::make_unique<qc::CompoundOperation>();
    for (size_t j = 0; j < broadcastingWidth; ++j) {
      // first we apply the operation
      auto nestedOp = applyQuantumOperation(
          gate, targetBits, controlBits, evaluatedParameters, invertOperation,
          gateCallStatement->debugInfo);
      if (nestedOp == nullptr) {
        return nullptr;
      }
      op->getOps().emplace_back(std::move(nestedOp));

      // after applying the operation, we update the broadcast bits
      for (auto index : targetBroadcastingIndices) {
        targetBits[index] = qc::QBit{targetBits[index] + 1};
      }
      for (auto index : controlBroadcastingIndices) {
        controlBits[index].qubit = qc::QBit{controlBits[index].qubit + 1};
      }
    }
    return op;
  }

  static std::shared_ptr<Gate>
  getMcGateDefinition(const std::string& identifier, size_t operandSize,
                      const std::shared_ptr<DebugInfo>& debugInfo) {
    std::vector<std::string> targetParams{};
    std::vector<std::shared_ptr<GateOperand>> operands;
    size_t nTargets = operandSize;
    if (identifier == "mcx_vchain") {
      nTargets -= (nTargets + 1) / 2 - 2;
    } else if (identifier == "mcx_recursive" && nTargets > 5) {
      nTargets -= 1;
    }
    for (size_t i = 0; i < operandSize; ++i) {
      targetParams.emplace_back("q" + std::to_string(i));
      if (i < nTargets) {
        operands.emplace_back(
            std::make_shared<GateOperand>("q" + std::to_string(i), nullptr));
      }
    }
    const size_t nControls = nTargets - 1;

    std::string nestedGateIdentifier = "x";
    std::vector<std::shared_ptr<Expression>> nestedParameters{};
    std::vector<std::string> nestedParameterNames{};
    if (identifier == "mcphase") {
      nestedGateIdentifier = "p";
      nestedParameters.emplace_back(
          std::make_shared<IdentifierExpression>("x"));
      nestedParameterNames.emplace_back("x");
    }

    // ctrl(nTargets - 1) @ x q0, ..., q(nTargets - 1)
    const auto gateCall = GateCallStatement(
        debugInfo, nestedGateIdentifier,
        std::vector<std::shared_ptr<GateModifier>>{
            std::make_shared<CtrlGateModifier>(
                true, std::make_shared<Constant>(nControls, false))},
        nestedParameters, operands);
    const auto inner = std::make_shared<GateCallStatement>(gateCall);

    const CompoundGate g{nestedParameterNames, targetParams, {inner}};
    return std::make_shared<CompoundGate>(g);
  }

  std::unique_ptr<qc::Operation> applyQuantumOperation(
      const std::shared_ptr<Gate>& gate, qc::Targets targetBits,
      std::vector<qc::Control> controlBits,
      std::vector<qc::fp> evaluatedParameters, bool invertOperation,
      const std::shared_ptr<DebugInfo>& debugInfo) {
    if (auto* standardGate = dynamic_cast<StandardGate*>(gate.get())) {
      auto op = std::make_unique<qc::StandardOperation>(
          qc::Controls{}, targetBits, standardGate->info.type,
          evaluatedParameters);
      if (invertOperation) {
        op->invert();
      }
      op->setControls(qc::Controls{controlBits.begin(), controlBits.end()});
      return op;
    }
    if (auto* compoundGate = dynamic_cast<CompoundGate*>(gate.get())) {
      constEvalPass.pushEnv();

      for (size_t i = 0; i < compoundGate->parameterNames.size(); ++i) {
        constEvalPass.addConst(compoundGate->parameterNames[i],
                               evaluatedParameters[i]);
      }

      auto nestedQubits = qc::QuantumRegisterMap{};
      size_t index = 0;
      for (const auto& qubitIdentifier : compoundGate->targetNames) {
        auto qubit = std::pair{targetBits[index], 1};

        nestedQubits.emplace(qubitIdentifier, qubit);
        index++;
      }

      auto op = std::make_unique<qc::CompoundOperation>();
      for (const auto& nestedGate : compoundGate->body) {
        if (auto barrierStatement =
                std::dynamic_pointer_cast<BarrierStatement>(nestedGate);
            barrierStatement != nullptr) {
          // nothing to do here for the simulator.
        } else if (auto resetStatement =
                       std::dynamic_pointer_cast<ResetStatement>(nestedGate);
                   resetStatement != nullptr) {
          op->emplace_back(getResetOp(resetStatement, nestedQubits));
        } else if (auto gateCallStatement =
                       std::dynamic_pointer_cast<GateCallStatement>(nestedGate);
                   gateCallStatement != nullptr) {
          for (const auto& operand : gateCallStatement->operands) {
            // OpenQASM 3.0 doesn't support indexing of gate arguments.
            if (operand->expression != nullptr &&
                std::find(compoundGate->targetNames.begin(),
                          compoundGate->targetNames.end(),
                          operand->identifier) !=
                    compoundGate->targetNames.end()) {
              error("Gate arguments cannot be indexed within gate body.",
                    debugInfo);
            }
          }

          auto nestedOp =
              evaluateGateCall(gateCallStatement, gateCallStatement->identifier,
                               gateCallStatement->arguments,
                               gateCallStatement->operands, nestedQubits);
          if (nestedOp == nullptr) {
            return nullptr;
          }
          op->getOps().emplace_back(std::move(nestedOp));
        } else {
          error("Unhandled quantum statement.", debugInfo);
        }
      }
      op->setControls(qc::Controls{controlBits.begin(), controlBits.end()});
      if (invertOperation) {
        op->invert();
      }

      constEvalPass.popEnv();

      if (op->getOps().size() == 1) {
        return std::move(op->getOps()[0]);
      }

      return op;
    }

    error("Unknown gate type.", debugInfo);
  }

  void visitMeasureAssignment(
      const std::string& identifier,
      const std::shared_ptr<Expression>& indexExpression,
      const std::shared_ptr<MeasureExpression>& measureExpression,
      const std::shared_ptr<DebugInfo>& debugInfo) {
    const auto decl = declarations.find(identifier);
    if (!decl.has_value()) {
      error("Usage of unknown identifier '" + identifier + "'.", debugInfo);
    }

    if (!std::get<1>(decl.value()->type)->isBit()) {
      error("Measure expression can only be assigned to a bit register.",
            debugInfo);
    }

    std::vector<qc::QBit> qubits{};
    std::vector<qc::Bit> bits{};
    translateGateOperand(measureExpression->gate, qubits, qc->getQregs(),
                         debugInfo);
    translateBitOperand(identifier, indexExpression, bits, debugInfo);

    if (qubits.size() != bits.size()) {
      error("Classical and quantum register must have the same width in "
            "measure statement. Classical register '" +
                identifier + "' has " + std::to_string(bits.size()) +
                " bits, but quantum register '" +
                measureExpression->gate->identifier + "' has " +
                std::to_string(qubits.size()) + " qubits.",
            debugInfo);
    }

    auto op = std::make_unique<qc::NonUnitaryOperation>(qubits, bits);
    qc->emplace_back(std::move(op));
  }

  void visitBarrierStatement(
      const std::shared_ptr<BarrierStatement> barrierStatement) override {
    qc->emplace_back(getBarrierOp(barrierStatement, qc->getQregs()));
  }

  void
  visitResetStatement(std::shared_ptr<ResetStatement> resetStatement) override {
    qc->emplace_back(getResetOp(resetStatement, qc->getQregs()));
  }

  void visitIfStatement(std::shared_ptr<IfStatement> ifStatement) override {
    // TODO: for now we only support statements comparing a classical bit reg
    // to a constant.
    const auto condition =
        std::dynamic_pointer_cast<BinaryExpression>(ifStatement->condition);
    if (condition == nullptr) {
      error("Condition not supported for if statement.",
            ifStatement->debugInfo);
    }

    const auto comparisonKind = getComparisonKind(condition->op);
    if (!comparisonKind) {
      error("Unsupported comparison operator.", ifStatement->debugInfo);
    }

    const auto lhs =
        std::dynamic_pointer_cast<IdentifierExpression>(condition->lhs);
    const auto rhs = std::dynamic_pointer_cast<Constant>(condition->rhs);

    if (lhs == nullptr) {
      error("Only classical registers are supported in conditions.",
            ifStatement->debugInfo);
    }
    if (rhs == nullptr) {
      error("Can only compare to constants.", ifStatement->debugInfo);
    }

    const auto creg = qc->getCregs().find(lhs->identifier);
    if (creg == qc->getCregs().end()) {
      error("Usage of unknown or invalid identifier '" + lhs->identifier +
                "' in condition.",
            ifStatement->debugInfo);
    }

    // translate statements in then/else blocks
    if (!ifStatement->thenStatements.empty()) {
      auto thenOps = translateBlockOperations(ifStatement->thenStatements);
      qc->emplace_back(std::make_unique<qc::ClassicControlledOperation>(
          thenOps, creg->second, rhs->getUInt(), *comparisonKind));
    }
    if (!ifStatement->elseStatements.empty()) {
      const auto invertedComparsionKind =
          qc::getInvertedComparsionKind(*comparisonKind);
      auto elseOps = translateBlockOperations(ifStatement->elseStatements);
      qc->emplace_back(std::make_unique<qc::ClassicControlledOperation>(
          elseOps, creg->second, rhs->getUInt(), invertedComparsionKind));
    }
  }

  [[nodiscard]] std::unique_ptr<qc::Operation> translateBlockOperations(
      const std::vector<std::shared_ptr<Statement>>& statements) {
    auto blockOps = std::make_unique<qc::CompoundOperation>();
    for (const auto& statement : statements) {
      auto gateCall = std::dynamic_pointer_cast<GateCallStatement>(statement);
      if (gateCall == nullptr) {
        error("Only quantum statements are supported in blocks.",
              statement->debugInfo);
      }
      const auto& qregs = qc->getQregs();

      auto op =
          evaluateGateCall(gateCall, gateCall->identifier, gateCall->arguments,
                           gateCall->operands, qregs);

      blockOps->emplace_back(std::move(op));
    }

    return blockOps;
  }

  [[nodiscard]] static std::unique_ptr<qc::Operation>
  getBarrierOp(const std::shared_ptr<BarrierStatement>& barrierStatement,
               const qc::QuantumRegisterMap& qregs) {
    std::vector<qc::QBit> qubits{};
    for (const auto& gate : barrierStatement->gates) {
      translateGateOperand(gate, qubits, qregs, barrierStatement->debugInfo);
    }

    return std::make_unique<qc::StandardOperation>(qubits, qc::otBarrier);
  }

  [[nodiscard]] static std::unique_ptr<qc::Operation>
  getResetOp(const std::shared_ptr<ResetStatement>& resetStatement,
             const qc::QuantumRegisterMap& qregs) {
    std::vector<qc::QBit> qubits{};
    translateGateOperand(resetStatement->gate, qubits, qregs,
                         resetStatement->debugInfo);
    return std::make_unique<qc::NonUnitaryOperation>(qubits, qc::otReset);
  }

  std::pair<std::string, size_t>
  parseGateIdentifierCompatMode(const std::string& identifier) {
    // we need to copy as we modify the string and need to return the original
    // string if we don't find a match.
    std::string gateIdentifier = identifier;
    size_t implicitControls = 0;
    while (!gateIdentifier.empty() && gateIdentifier[0] == 'c') {
      gateIdentifier = gateIdentifier.substr(1);
      implicitControls++;
    }

    if (gates.find(gateIdentifier) == gates.end()) {
      return std::pair{identifier, 0};
    }
    return std::pair{gateIdentifier, implicitControls};
  }
};

void qc::QuantumComputation::importOpenQASM(std::istream& is) {
  using namespace qasm;

  Parser p(&is);

  const auto program = p.parseProgram();
  OpenQasmParser parser{this};
  parser.visitProgram(program);
}
