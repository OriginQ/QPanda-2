
// Generated from .\originir.g4 by ANTLR 4.7.2


#include "Core/Utilities/Compiler/OriginIRCompiler/originirListener.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirVisitor.h"

#include "Core/Utilities/Compiler/OriginIRCompiler/originirParser.h"



using namespace antlrcpp;
using namespace antlr4;

originirParser::originirParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

originirParser::~originirParser() {
  delete _interpreter;
}

std::string originirParser::getGrammarFileName() const {
  return "originir.g4";
}

const std::vector<std::string>& originirParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& originirParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- TranslationunitContext ------------------------------------------------------------------

originirParser::TranslationunitContext::TranslationunitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::DeclarationContext* originirParser::TranslationunitContext::declaration() {
  return getRuleContext<originirParser::DeclarationContext>(0);
}

std::vector<originirParser::StatementContext *> originirParser::TranslationunitContext::statement() {
  return getRuleContexts<originirParser::StatementContext>();
}

originirParser::StatementContext* originirParser::TranslationunitContext::statement(size_t i) {
  return getRuleContext<originirParser::StatementContext>(i);
}


size_t originirParser::TranslationunitContext::getRuleIndex() const {
  return originirParser::RuleTranslationunit;
}

void originirParser::TranslationunitContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTranslationunit(this);
}

void originirParser::TranslationunitContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTranslationunit(this);
}


antlrcpp::Any originirParser::TranslationunitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitTranslationunit(this);
  else
    return visitor->visitChildren(this);
}

originirParser::TranslationunitContext* originirParser::translationunit() {
  TranslationunitContext *_localctx = _tracker.createInstance<TranslationunitContext>(_ctx, getState());
  enterRule(_localctx, 0, originirParser::RuleTranslationunit);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(96);
    declaration();
    setState(100);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::ECHO_GATE)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
      | (1ULL << originirParser::I_GATE)
      | (1ULL << originirParser::U2_GATE)
      | (1ULL << originirParser::RPHI_GATE)
      | (1ULL << originirParser::U3_GATE)
      | (1ULL << originirParser::U4_GATE)
      | (1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN))) != 0) || _la == originirParser::Integer_Literal

    || _la == originirParser::Double_Literal) {
      setState(97);
      statement();
      setState(102);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DeclarationContext ------------------------------------------------------------------

originirParser::DeclarationContext::DeclarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Qinit_declarationContext* originirParser::DeclarationContext::qinit_declaration() {
  return getRuleContext<originirParser::Qinit_declarationContext>(0);
}

originirParser::Cinit_declarationContext* originirParser::DeclarationContext::cinit_declaration() {
  return getRuleContext<originirParser::Cinit_declarationContext>(0);
}


size_t originirParser::DeclarationContext::getRuleIndex() const {
  return originirParser::RuleDeclaration;
}

void originirParser::DeclarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclaration(this);
}

void originirParser::DeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclaration(this);
}


antlrcpp::Any originirParser::DeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDeclaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::DeclarationContext* originirParser::declaration() {
  DeclarationContext *_localctx = _tracker.createInstance<DeclarationContext>(_ctx, getState());
  enterRule(_localctx, 2, originirParser::RuleDeclaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(103);
    qinit_declaration();
    setState(104);
    cinit_declaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Qinit_declarationContext ------------------------------------------------------------------

originirParser::Qinit_declarationContext::Qinit_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Qinit_declarationContext::QINIT_KEY() {
  return getToken(originirParser::QINIT_KEY, 0);
}

tree::TerminalNode* originirParser::Qinit_declarationContext::Integer_Literal() {
  return getToken(originirParser::Integer_Literal, 0);
}

tree::TerminalNode* originirParser::Qinit_declarationContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}


size_t originirParser::Qinit_declarationContext::getRuleIndex() const {
  return originirParser::RuleQinit_declaration;
}

void originirParser::Qinit_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQinit_declaration(this);
}

void originirParser::Qinit_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQinit_declaration(this);
}


antlrcpp::Any originirParser::Qinit_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitQinit_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Qinit_declarationContext* originirParser::qinit_declaration() {
  Qinit_declarationContext *_localctx = _tracker.createInstance<Qinit_declarationContext>(_ctx, getState());
  enterRule(_localctx, 4, originirParser::RuleQinit_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(106);
    match(originirParser::QINIT_KEY);
    setState(107);
    match(originirParser::Integer_Literal);
    setState(108);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Cinit_declarationContext ------------------------------------------------------------------

originirParser::Cinit_declarationContext::Cinit_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Cinit_declarationContext::CREG_KEY() {
  return getToken(originirParser::CREG_KEY, 0);
}

tree::TerminalNode* originirParser::Cinit_declarationContext::Integer_Literal() {
  return getToken(originirParser::Integer_Literal, 0);
}

tree::TerminalNode* originirParser::Cinit_declarationContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}


size_t originirParser::Cinit_declarationContext::getRuleIndex() const {
  return originirParser::RuleCinit_declaration;
}

void originirParser::Cinit_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCinit_declaration(this);
}

void originirParser::Cinit_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCinit_declaration(this);
}


antlrcpp::Any originirParser::Cinit_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitCinit_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Cinit_declarationContext* originirParser::cinit_declaration() {
  Cinit_declarationContext *_localctx = _tracker.createInstance<Cinit_declarationContext>(_ctx, getState());
  enterRule(_localctx, 6, originirParser::RuleCinit_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(110);
    match(originirParser::CREG_KEY);
    setState(111);
    match(originirParser::Integer_Literal);
    setState(112);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Quantum_gate_declarationContext ------------------------------------------------------------------

originirParser::Quantum_gate_declarationContext::Quantum_gate_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Single_gate_without_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::single_gate_without_parameter_declaration() {
  return getRuleContext<originirParser::Single_gate_without_parameter_declarationContext>(0);
}

originirParser::Single_gate_with_one_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::single_gate_with_one_parameter_declaration() {
  return getRuleContext<originirParser::Single_gate_with_one_parameter_declarationContext>(0);
}

originirParser::Single_gate_with_two_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::single_gate_with_two_parameter_declaration() {
  return getRuleContext<originirParser::Single_gate_with_two_parameter_declarationContext>(0);
}

originirParser::Single_gate_with_three_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::single_gate_with_three_parameter_declaration() {
  return getRuleContext<originirParser::Single_gate_with_three_parameter_declarationContext>(0);
}

originirParser::Single_gate_with_four_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::single_gate_with_four_parameter_declaration() {
  return getRuleContext<originirParser::Single_gate_with_four_parameter_declarationContext>(0);
}

originirParser::Double_gate_without_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::double_gate_without_parameter_declaration() {
  return getRuleContext<originirParser::Double_gate_without_parameter_declarationContext>(0);
}

originirParser::Double_gate_with_one_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::double_gate_with_one_parameter_declaration() {
  return getRuleContext<originirParser::Double_gate_with_one_parameter_declarationContext>(0);
}

originirParser::Double_gate_with_four_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::double_gate_with_four_parameter_declaration() {
  return getRuleContext<originirParser::Double_gate_with_four_parameter_declarationContext>(0);
}

originirParser::Triple_gate_without_parameter_declarationContext* originirParser::Quantum_gate_declarationContext::triple_gate_without_parameter_declaration() {
  return getRuleContext<originirParser::Triple_gate_without_parameter_declarationContext>(0);
}


size_t originirParser::Quantum_gate_declarationContext::getRuleIndex() const {
  return originirParser::RuleQuantum_gate_declaration;
}

void originirParser::Quantum_gate_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantum_gate_declaration(this);
}

void originirParser::Quantum_gate_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantum_gate_declaration(this);
}


antlrcpp::Any originirParser::Quantum_gate_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitQuantum_gate_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Quantum_gate_declarationContext* originirParser::quantum_gate_declaration() {
  Quantum_gate_declarationContext *_localctx = _tracker.createInstance<Quantum_gate_declarationContext>(_ctx, getState());
  enterRule(_localctx, 8, originirParser::RuleQuantum_gate_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(123);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::ECHO_GATE:
      case originirParser::H_GATE:
      case originirParser::X_GATE:
      case originirParser::T_GATE:
      case originirParser::S_GATE:
      case originirParser::Y_GATE:
      case originirParser::Z_GATE:
      case originirParser::X1_GATE:
      case originirParser::Y1_GATE:
      case originirParser::Z1_GATE:
      case originirParser::I_GATE: {
        enterOuterAlt(_localctx, 1);
        setState(114);
        single_gate_without_parameter_declaration();
        break;
      }

      case originirParser::RX_GATE:
      case originirParser::RY_GATE:
      case originirParser::RZ_GATE:
      case originirParser::U1_GATE: {
        enterOuterAlt(_localctx, 2);
        setState(115);
        single_gate_with_one_parameter_declaration();
        break;
      }

      case originirParser::U2_GATE:
      case originirParser::RPHI_GATE: {
        enterOuterAlt(_localctx, 3);
        setState(116);
        single_gate_with_two_parameter_declaration();
        break;
      }

      case originirParser::U3_GATE: {
        enterOuterAlt(_localctx, 4);
        setState(117);
        single_gate_with_three_parameter_declaration();
        break;
      }

      case originirParser::U4_GATE: {
        enterOuterAlt(_localctx, 5);
        setState(118);
        single_gate_with_four_parameter_declaration();
        break;
      }

      case originirParser::CNOT_GATE:
      case originirParser::CZ_GATE:
      case originirParser::ISWAP_GATE:
      case originirParser::SQISWAP_GATE:
      case originirParser::SWAPZ1_GATE: {
        enterOuterAlt(_localctx, 6);
        setState(119);
        double_gate_without_parameter_declaration();
        break;
      }

      case originirParser::ISWAPTHETA_GATE:
      case originirParser::CR_GATE: {
        enterOuterAlt(_localctx, 7);
        setState(120);
        double_gate_with_one_parameter_declaration();
        break;
      }

      case originirParser::CU_GATE: {
        enterOuterAlt(_localctx, 8);
        setState(121);
        double_gate_with_four_parameter_declaration();
        break;
      }

      case originirParser::TOFFOLI_GATE: {
        enterOuterAlt(_localctx, 9);
        setState(122);
        triple_gate_without_parameter_declaration();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IndexContext ------------------------------------------------------------------

originirParser::IndexContext::IndexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::IndexContext::LBRACK() {
  return getToken(originirParser::LBRACK, 0);
}

originirParser::ExpressionContext* originirParser::IndexContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

tree::TerminalNode* originirParser::IndexContext::RBRACK() {
  return getToken(originirParser::RBRACK, 0);
}


size_t originirParser::IndexContext::getRuleIndex() const {
  return originirParser::RuleIndex;
}

void originirParser::IndexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndex(this);
}

void originirParser::IndexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndex(this);
}


antlrcpp::Any originirParser::IndexContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitIndex(this);
  else
    return visitor->visitChildren(this);
}

originirParser::IndexContext* originirParser::index() {
  IndexContext *_localctx = _tracker.createInstance<IndexContext>(_ctx, getState());
  enterRule(_localctx, 10, originirParser::RuleIndex);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(125);
    match(originirParser::LBRACK);
    setState(126);
    expression();
    setState(127);
    match(originirParser::RBRACK);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- C_KEY_declarationContext ------------------------------------------------------------------

originirParser::C_KEY_declarationContext::C_KEY_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::C_KEY_declarationContext::C_KEY() {
  return getToken(originirParser::C_KEY, 0);
}

originirParser::IndexContext* originirParser::C_KEY_declarationContext::index() {
  return getRuleContext<originirParser::IndexContext>(0);
}


size_t originirParser::C_KEY_declarationContext::getRuleIndex() const {
  return originirParser::RuleC_KEY_declaration;
}

void originirParser::C_KEY_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterC_KEY_declaration(this);
}

void originirParser::C_KEY_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitC_KEY_declaration(this);
}


antlrcpp::Any originirParser::C_KEY_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitC_KEY_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::C_KEY_declarationContext* originirParser::c_KEY_declaration() {
  C_KEY_declarationContext *_localctx = _tracker.createInstance<C_KEY_declarationContext>(_ctx, getState());
  enterRule(_localctx, 12, originirParser::RuleC_KEY_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(129);
    match(originirParser::C_KEY);
    setState(130);
    index();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Q_KEY_declarationContext ------------------------------------------------------------------

originirParser::Q_KEY_declarationContext::Q_KEY_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Q_KEY_declarationContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}

originirParser::IndexContext* originirParser::Q_KEY_declarationContext::index() {
  return getRuleContext<originirParser::IndexContext>(0);
}


size_t originirParser::Q_KEY_declarationContext::getRuleIndex() const {
  return originirParser::RuleQ_KEY_declaration;
}

void originirParser::Q_KEY_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQ_KEY_declaration(this);
}

void originirParser::Q_KEY_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQ_KEY_declaration(this);
}


antlrcpp::Any originirParser::Q_KEY_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitQ_KEY_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Q_KEY_declarationContext* originirParser::q_KEY_declaration() {
  Q_KEY_declarationContext *_localctx = _tracker.createInstance<Q_KEY_declarationContext>(_ctx, getState());
  enterRule(_localctx, 14, originirParser::RuleQ_KEY_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(132);
    match(originirParser::Q_KEY);
    setState(133);
    index();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_without_parameter_declarationContext ------------------------------------------------------------------

originirParser::Single_gate_without_parameter_declarationContext::Single_gate_without_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Single_gate_without_parameter_typeContext* originirParser::Single_gate_without_parameter_declarationContext::single_gate_without_parameter_type() {
  return getRuleContext<originirParser::Single_gate_without_parameter_typeContext>(0);
}

originirParser::Q_KEY_declarationContext* originirParser::Single_gate_without_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(0);
}


size_t originirParser::Single_gate_without_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_without_parameter_declaration;
}

void originirParser::Single_gate_without_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_without_parameter_declaration(this);
}

void originirParser::Single_gate_without_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_without_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_without_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_without_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_without_parameter_declarationContext* originirParser::single_gate_without_parameter_declaration() {
  Single_gate_without_parameter_declarationContext *_localctx = _tracker.createInstance<Single_gate_without_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 16, originirParser::RuleSingle_gate_without_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(135);
    single_gate_without_parameter_type();
    setState(136);
    q_KEY_declaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_one_parameter_declarationContext ------------------------------------------------------------------

originirParser::Single_gate_with_one_parameter_declarationContext::Single_gate_with_one_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Single_gate_with_one_parameter_typeContext* originirParser::Single_gate_with_one_parameter_declarationContext::single_gate_with_one_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_one_parameter_typeContext>(0);
}

originirParser::Q_KEY_declarationContext* originirParser::Single_gate_with_one_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(0);
}

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_declarationContext::COMMA() {
  return getToken(originirParser::COMMA, 0);
}

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_declarationContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

originirParser::ExpressionContext* originirParser::Single_gate_with_one_parameter_declarationContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_declarationContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Single_gate_with_one_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_one_parameter_declaration;
}

void originirParser::Single_gate_with_one_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_one_parameter_declaration(this);
}

void originirParser::Single_gate_with_one_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_one_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_one_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_one_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_one_parameter_declarationContext* originirParser::single_gate_with_one_parameter_declaration() {
  Single_gate_with_one_parameter_declarationContext *_localctx = _tracker.createInstance<Single_gate_with_one_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 18, originirParser::RuleSingle_gate_with_one_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(138);
    single_gate_with_one_parameter_type();
    setState(139);
    q_KEY_declaration();
    setState(140);
    match(originirParser::COMMA);
    setState(141);
    match(originirParser::LPAREN);
    setState(142);
    expression();
    setState(143);
    match(originirParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_two_parameter_declarationContext ------------------------------------------------------------------

originirParser::Single_gate_with_two_parameter_declarationContext::Single_gate_with_two_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Single_gate_with_two_parameter_typeContext* originirParser::Single_gate_with_two_parameter_declarationContext::single_gate_with_two_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_two_parameter_typeContext>(0);
}

originirParser::Q_KEY_declarationContext* originirParser::Single_gate_with_two_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Single_gate_with_two_parameter_declarationContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Single_gate_with_two_parameter_declarationContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}

tree::TerminalNode* originirParser::Single_gate_with_two_parameter_declarationContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

std::vector<originirParser::ExpressionContext *> originirParser::Single_gate_with_two_parameter_declarationContext::expression() {
  return getRuleContexts<originirParser::ExpressionContext>();
}

originirParser::ExpressionContext* originirParser::Single_gate_with_two_parameter_declarationContext::expression(size_t i) {
  return getRuleContext<originirParser::ExpressionContext>(i);
}

tree::TerminalNode* originirParser::Single_gate_with_two_parameter_declarationContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Single_gate_with_two_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_two_parameter_declaration;
}

void originirParser::Single_gate_with_two_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_two_parameter_declaration(this);
}

void originirParser::Single_gate_with_two_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_two_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_two_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_two_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_two_parameter_declarationContext* originirParser::single_gate_with_two_parameter_declaration() {
  Single_gate_with_two_parameter_declarationContext *_localctx = _tracker.createInstance<Single_gate_with_two_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 20, originirParser::RuleSingle_gate_with_two_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(145);
    single_gate_with_two_parameter_type();
    setState(146);
    q_KEY_declaration();
    setState(147);
    match(originirParser::COMMA);
    setState(148);
    match(originirParser::LPAREN);
    setState(149);
    expression();
    setState(150);
    match(originirParser::COMMA);
    setState(151);
    expression();
    setState(152);
    match(originirParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_three_parameter_declarationContext ------------------------------------------------------------------

originirParser::Single_gate_with_three_parameter_declarationContext::Single_gate_with_three_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Single_gate_with_three_parameter_typeContext* originirParser::Single_gate_with_three_parameter_declarationContext::single_gate_with_three_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_three_parameter_typeContext>(0);
}

originirParser::Q_KEY_declarationContext* originirParser::Single_gate_with_three_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Single_gate_with_three_parameter_declarationContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Single_gate_with_three_parameter_declarationContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}

tree::TerminalNode* originirParser::Single_gate_with_three_parameter_declarationContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

std::vector<originirParser::ExpressionContext *> originirParser::Single_gate_with_three_parameter_declarationContext::expression() {
  return getRuleContexts<originirParser::ExpressionContext>();
}

originirParser::ExpressionContext* originirParser::Single_gate_with_three_parameter_declarationContext::expression(size_t i) {
  return getRuleContext<originirParser::ExpressionContext>(i);
}

tree::TerminalNode* originirParser::Single_gate_with_three_parameter_declarationContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Single_gate_with_three_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_three_parameter_declaration;
}

void originirParser::Single_gate_with_three_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_three_parameter_declaration(this);
}

void originirParser::Single_gate_with_three_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_three_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_three_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_three_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_three_parameter_declarationContext* originirParser::single_gate_with_three_parameter_declaration() {
  Single_gate_with_three_parameter_declarationContext *_localctx = _tracker.createInstance<Single_gate_with_three_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 22, originirParser::RuleSingle_gate_with_three_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(154);
    single_gate_with_three_parameter_type();
    setState(155);
    q_KEY_declaration();
    setState(156);
    match(originirParser::COMMA);
    setState(157);
    match(originirParser::LPAREN);
    setState(158);
    expression();
    setState(159);
    match(originirParser::COMMA);
    setState(160);
    expression();
    setState(161);
    match(originirParser::COMMA);
    setState(162);
    expression();
    setState(163);
    match(originirParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_four_parameter_declarationContext ------------------------------------------------------------------

originirParser::Single_gate_with_four_parameter_declarationContext::Single_gate_with_four_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Single_gate_with_four_parameter_typeContext* originirParser::Single_gate_with_four_parameter_declarationContext::single_gate_with_four_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_four_parameter_typeContext>(0);
}

originirParser::Q_KEY_declarationContext* originirParser::Single_gate_with_four_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Single_gate_with_four_parameter_declarationContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Single_gate_with_four_parameter_declarationContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}

tree::TerminalNode* originirParser::Single_gate_with_four_parameter_declarationContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

std::vector<originirParser::ExpressionContext *> originirParser::Single_gate_with_four_parameter_declarationContext::expression() {
  return getRuleContexts<originirParser::ExpressionContext>();
}

originirParser::ExpressionContext* originirParser::Single_gate_with_four_parameter_declarationContext::expression(size_t i) {
  return getRuleContext<originirParser::ExpressionContext>(i);
}

tree::TerminalNode* originirParser::Single_gate_with_four_parameter_declarationContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Single_gate_with_four_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_four_parameter_declaration;
}

void originirParser::Single_gate_with_four_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_four_parameter_declaration(this);
}

void originirParser::Single_gate_with_four_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_four_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_four_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_four_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_four_parameter_declarationContext* originirParser::single_gate_with_four_parameter_declaration() {
  Single_gate_with_four_parameter_declarationContext *_localctx = _tracker.createInstance<Single_gate_with_four_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 24, originirParser::RuleSingle_gate_with_four_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(165);
    single_gate_with_four_parameter_type();
    setState(166);
    q_KEY_declaration();
    setState(167);
    match(originirParser::COMMA);
    setState(168);
    match(originirParser::LPAREN);
    setState(169);
    expression();
    setState(170);
    match(originirParser::COMMA);
    setState(171);
    expression();
    setState(172);
    match(originirParser::COMMA);
    setState(173);
    expression();
    setState(174);
    match(originirParser::COMMA);
    setState(175);
    expression();
    setState(176);
    match(originirParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Double_gate_without_parameter_declarationContext ------------------------------------------------------------------

originirParser::Double_gate_without_parameter_declarationContext::Double_gate_without_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Double_gate_without_parameter_typeContext* originirParser::Double_gate_without_parameter_declarationContext::double_gate_without_parameter_type() {
  return getRuleContext<originirParser::Double_gate_without_parameter_typeContext>(0);
}

std::vector<originirParser::Q_KEY_declarationContext *> originirParser::Double_gate_without_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContexts<originirParser::Q_KEY_declarationContext>();
}

originirParser::Q_KEY_declarationContext* originirParser::Double_gate_without_parameter_declarationContext::q_KEY_declaration(size_t i) {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(i);
}

tree::TerminalNode* originirParser::Double_gate_without_parameter_declarationContext::COMMA() {
  return getToken(originirParser::COMMA, 0);
}


size_t originirParser::Double_gate_without_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleDouble_gate_without_parameter_declaration;
}

void originirParser::Double_gate_without_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_without_parameter_declaration(this);
}

void originirParser::Double_gate_without_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_without_parameter_declaration(this);
}


antlrcpp::Any originirParser::Double_gate_without_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_without_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_without_parameter_declarationContext* originirParser::double_gate_without_parameter_declaration() {
  Double_gate_without_parameter_declarationContext *_localctx = _tracker.createInstance<Double_gate_without_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 26, originirParser::RuleDouble_gate_without_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(178);
    double_gate_without_parameter_type();
    setState(179);
    q_KEY_declaration();
    setState(180);
    match(originirParser::COMMA);
    setState(181);
    q_KEY_declaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Double_gate_with_one_parameter_declarationContext ------------------------------------------------------------------

originirParser::Double_gate_with_one_parameter_declarationContext::Double_gate_with_one_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Double_gate_with_one_parameter_typeContext* originirParser::Double_gate_with_one_parameter_declarationContext::double_gate_with_one_parameter_type() {
  return getRuleContext<originirParser::Double_gate_with_one_parameter_typeContext>(0);
}

std::vector<originirParser::Q_KEY_declarationContext *> originirParser::Double_gate_with_one_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContexts<originirParser::Q_KEY_declarationContext>();
}

originirParser::Q_KEY_declarationContext* originirParser::Double_gate_with_one_parameter_declarationContext::q_KEY_declaration(size_t i) {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::Double_gate_with_one_parameter_declarationContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_declarationContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_declarationContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

originirParser::ExpressionContext* originirParser::Double_gate_with_one_parameter_declarationContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_declarationContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Double_gate_with_one_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleDouble_gate_with_one_parameter_declaration;
}

void originirParser::Double_gate_with_one_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_one_parameter_declaration(this);
}

void originirParser::Double_gate_with_one_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_one_parameter_declaration(this);
}


antlrcpp::Any originirParser::Double_gate_with_one_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_with_one_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_with_one_parameter_declarationContext* originirParser::double_gate_with_one_parameter_declaration() {
  Double_gate_with_one_parameter_declarationContext *_localctx = _tracker.createInstance<Double_gate_with_one_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 28, originirParser::RuleDouble_gate_with_one_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(183);
    double_gate_with_one_parameter_type();
    setState(184);
    q_KEY_declaration();
    setState(185);
    match(originirParser::COMMA);
    setState(186);
    q_KEY_declaration();
    setState(187);
    match(originirParser::COMMA);
    setState(188);
    match(originirParser::LPAREN);
    setState(189);
    expression();
    setState(190);
    match(originirParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Double_gate_with_four_parameter_declarationContext ------------------------------------------------------------------

originirParser::Double_gate_with_four_parameter_declarationContext::Double_gate_with_four_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Double_gate_with_four_parameter_typeContext* originirParser::Double_gate_with_four_parameter_declarationContext::double_gate_with_four_parameter_type() {
  return getRuleContext<originirParser::Double_gate_with_four_parameter_typeContext>(0);
}

std::vector<originirParser::Q_KEY_declarationContext *> originirParser::Double_gate_with_four_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContexts<originirParser::Q_KEY_declarationContext>();
}

originirParser::Q_KEY_declarationContext* originirParser::Double_gate_with_four_parameter_declarationContext::q_KEY_declaration(size_t i) {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::Double_gate_with_four_parameter_declarationContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Double_gate_with_four_parameter_declarationContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}

tree::TerminalNode* originirParser::Double_gate_with_four_parameter_declarationContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

std::vector<originirParser::ExpressionContext *> originirParser::Double_gate_with_four_parameter_declarationContext::expression() {
  return getRuleContexts<originirParser::ExpressionContext>();
}

originirParser::ExpressionContext* originirParser::Double_gate_with_four_parameter_declarationContext::expression(size_t i) {
  return getRuleContext<originirParser::ExpressionContext>(i);
}

tree::TerminalNode* originirParser::Double_gate_with_four_parameter_declarationContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Double_gate_with_four_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleDouble_gate_with_four_parameter_declaration;
}

void originirParser::Double_gate_with_four_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_four_parameter_declaration(this);
}

void originirParser::Double_gate_with_four_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_four_parameter_declaration(this);
}


antlrcpp::Any originirParser::Double_gate_with_four_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_with_four_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_with_four_parameter_declarationContext* originirParser::double_gate_with_four_parameter_declaration() {
  Double_gate_with_four_parameter_declarationContext *_localctx = _tracker.createInstance<Double_gate_with_four_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 30, originirParser::RuleDouble_gate_with_four_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(192);
    double_gate_with_four_parameter_type();
    setState(193);
    q_KEY_declaration();
    setState(194);
    match(originirParser::COMMA);
    setState(195);
    q_KEY_declaration();
    setState(196);
    match(originirParser::COMMA);
    setState(197);
    match(originirParser::LPAREN);
    setState(198);
    expression();
    setState(199);
    match(originirParser::COMMA);
    setState(200);
    expression();
    setState(201);
    match(originirParser::COMMA);
    setState(202);
    expression();
    setState(203);
    match(originirParser::COMMA);
    setState(204);
    expression();
    setState(205);
    match(originirParser::RPAREN);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Triple_gate_without_parameter_declarationContext ------------------------------------------------------------------

originirParser::Triple_gate_without_parameter_declarationContext::Triple_gate_without_parameter_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Triple_gate_without_parameter_typeContext* originirParser::Triple_gate_without_parameter_declarationContext::triple_gate_without_parameter_type() {
  return getRuleContext<originirParser::Triple_gate_without_parameter_typeContext>(0);
}

std::vector<originirParser::Q_KEY_declarationContext *> originirParser::Triple_gate_without_parameter_declarationContext::q_KEY_declaration() {
  return getRuleContexts<originirParser::Q_KEY_declarationContext>();
}

originirParser::Q_KEY_declarationContext* originirParser::Triple_gate_without_parameter_declarationContext::q_KEY_declaration(size_t i) {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::Triple_gate_without_parameter_declarationContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Triple_gate_without_parameter_declarationContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}


size_t originirParser::Triple_gate_without_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleTriple_gate_without_parameter_declaration;
}

void originirParser::Triple_gate_without_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTriple_gate_without_parameter_declaration(this);
}

void originirParser::Triple_gate_without_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTriple_gate_without_parameter_declaration(this);
}


antlrcpp::Any originirParser::Triple_gate_without_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitTriple_gate_without_parameter_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Triple_gate_without_parameter_declarationContext* originirParser::triple_gate_without_parameter_declaration() {
  Triple_gate_without_parameter_declarationContext *_localctx = _tracker.createInstance<Triple_gate_without_parameter_declarationContext>(_ctx, getState());
  enterRule(_localctx, 32, originirParser::RuleTriple_gate_without_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(207);
    triple_gate_without_parameter_type();
    setState(208);
    q_KEY_declaration();
    setState(209);
    match(originirParser::COMMA);
    setState(210);
    q_KEY_declaration();
    setState(211);
    match(originirParser::COMMA);
    setState(212);
    q_KEY_declaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_without_parameter_typeContext ------------------------------------------------------------------

originirParser::Single_gate_without_parameter_typeContext::Single_gate_without_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::H_GATE() {
  return getToken(originirParser::H_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::T_GATE() {
  return getToken(originirParser::T_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::S_GATE() {
  return getToken(originirParser::S_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::X_GATE() {
  return getToken(originirParser::X_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::Y_GATE() {
  return getToken(originirParser::Y_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::Z_GATE() {
  return getToken(originirParser::Z_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::X1_GATE() {
  return getToken(originirParser::X1_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::Y1_GATE() {
  return getToken(originirParser::Y1_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::Z1_GATE() {
  return getToken(originirParser::Z1_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::I_GATE() {
  return getToken(originirParser::I_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_without_parameter_typeContext::ECHO_GATE() {
  return getToken(originirParser::ECHO_GATE, 0);
}


size_t originirParser::Single_gate_without_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_without_parameter_type;
}

void originirParser::Single_gate_without_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_without_parameter_type(this);
}

void originirParser::Single_gate_without_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_without_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_without_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_without_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_without_parameter_typeContext* originirParser::single_gate_without_parameter_type() {
  Single_gate_without_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_without_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 34, originirParser::RuleSingle_gate_without_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(214);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::ECHO_GATE)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
      | (1ULL << originirParser::I_GATE))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_one_parameter_typeContext ------------------------------------------------------------------

originirParser::Single_gate_with_one_parameter_typeContext::Single_gate_with_one_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_typeContext::RX_GATE() {
  return getToken(originirParser::RX_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_typeContext::RY_GATE() {
  return getToken(originirParser::RY_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_typeContext::RZ_GATE() {
  return getToken(originirParser::RZ_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_typeContext::U1_GATE() {
  return getToken(originirParser::U1_GATE, 0);
}


size_t originirParser::Single_gate_with_one_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_one_parameter_type;
}

void originirParser::Single_gate_with_one_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_one_parameter_type(this);
}

void originirParser::Single_gate_with_one_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_one_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_one_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_one_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_one_parameter_typeContext* originirParser::single_gate_with_one_parameter_type() {
  Single_gate_with_one_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_one_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 36, originirParser::RuleSingle_gate_with_one_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(216);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_two_parameter_typeContext ------------------------------------------------------------------

originirParser::Single_gate_with_two_parameter_typeContext::Single_gate_with_two_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Single_gate_with_two_parameter_typeContext::U2_GATE() {
  return getToken(originirParser::U2_GATE, 0);
}

tree::TerminalNode* originirParser::Single_gate_with_two_parameter_typeContext::RPHI_GATE() {
  return getToken(originirParser::RPHI_GATE, 0);
}


size_t originirParser::Single_gate_with_two_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_two_parameter_type;
}

void originirParser::Single_gate_with_two_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_two_parameter_type(this);
}

void originirParser::Single_gate_with_two_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_two_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_two_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_two_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_two_parameter_typeContext* originirParser::single_gate_with_two_parameter_type() {
  Single_gate_with_two_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_two_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 38, originirParser::RuleSingle_gate_with_two_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(218);
    _la = _input->LA(1);
    if (!(_la == originirParser::U2_GATE

    || _la == originirParser::RPHI_GATE)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_three_parameter_typeContext ------------------------------------------------------------------

originirParser::Single_gate_with_three_parameter_typeContext::Single_gate_with_three_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Single_gate_with_three_parameter_typeContext::U3_GATE() {
  return getToken(originirParser::U3_GATE, 0);
}


size_t originirParser::Single_gate_with_three_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_three_parameter_type;
}

void originirParser::Single_gate_with_three_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_three_parameter_type(this);
}

void originirParser::Single_gate_with_three_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_three_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_three_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_three_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_three_parameter_typeContext* originirParser::single_gate_with_three_parameter_type() {
  Single_gate_with_three_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_three_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 40, originirParser::RuleSingle_gate_with_three_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(220);
    match(originirParser::U3_GATE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Single_gate_with_four_parameter_typeContext ------------------------------------------------------------------

originirParser::Single_gate_with_four_parameter_typeContext::Single_gate_with_four_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Single_gate_with_four_parameter_typeContext::U4_GATE() {
  return getToken(originirParser::U4_GATE, 0);
}


size_t originirParser::Single_gate_with_four_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_four_parameter_type;
}

void originirParser::Single_gate_with_four_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_four_parameter_type(this);
}

void originirParser::Single_gate_with_four_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_four_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_four_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_four_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_four_parameter_typeContext* originirParser::single_gate_with_four_parameter_type() {
  Single_gate_with_four_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_four_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 42, originirParser::RuleSingle_gate_with_four_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(222);
    match(originirParser::U4_GATE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Double_gate_without_parameter_typeContext ------------------------------------------------------------------

originirParser::Double_gate_without_parameter_typeContext::Double_gate_without_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Double_gate_without_parameter_typeContext::CNOT_GATE() {
  return getToken(originirParser::CNOT_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_without_parameter_typeContext::CZ_GATE() {
  return getToken(originirParser::CZ_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_without_parameter_typeContext::ISWAP_GATE() {
  return getToken(originirParser::ISWAP_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_without_parameter_typeContext::SQISWAP_GATE() {
  return getToken(originirParser::SQISWAP_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_without_parameter_typeContext::SWAPZ1_GATE() {
  return getToken(originirParser::SWAPZ1_GATE, 0);
}


size_t originirParser::Double_gate_without_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleDouble_gate_without_parameter_type;
}

void originirParser::Double_gate_without_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_without_parameter_type(this);
}

void originirParser::Double_gate_without_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_without_parameter_type(this);
}


antlrcpp::Any originirParser::Double_gate_without_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_without_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_without_parameter_typeContext* originirParser::double_gate_without_parameter_type() {
  Double_gate_without_parameter_typeContext *_localctx = _tracker.createInstance<Double_gate_without_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 44, originirParser::RuleDouble_gate_without_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(224);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Double_gate_with_one_parameter_typeContext ------------------------------------------------------------------

originirParser::Double_gate_with_one_parameter_typeContext::Double_gate_with_one_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_typeContext::ISWAPTHETA_GATE() {
  return getToken(originirParser::ISWAPTHETA_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_typeContext::CR_GATE() {
  return getToken(originirParser::CR_GATE, 0);
}


size_t originirParser::Double_gate_with_one_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleDouble_gate_with_one_parameter_type;
}

void originirParser::Double_gate_with_one_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_one_parameter_type(this);
}

void originirParser::Double_gate_with_one_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_one_parameter_type(this);
}


antlrcpp::Any originirParser::Double_gate_with_one_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_with_one_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_with_one_parameter_typeContext* originirParser::double_gate_with_one_parameter_type() {
  Double_gate_with_one_parameter_typeContext *_localctx = _tracker.createInstance<Double_gate_with_one_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 46, originirParser::RuleDouble_gate_with_one_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(226);
    _la = _input->LA(1);
    if (!(_la == originirParser::ISWAPTHETA_GATE

    || _la == originirParser::CR_GATE)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Double_gate_with_four_parameter_typeContext ------------------------------------------------------------------

originirParser::Double_gate_with_four_parameter_typeContext::Double_gate_with_four_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Double_gate_with_four_parameter_typeContext::CU_GATE() {
  return getToken(originirParser::CU_GATE, 0);
}


size_t originirParser::Double_gate_with_four_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleDouble_gate_with_four_parameter_type;
}

void originirParser::Double_gate_with_four_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_four_parameter_type(this);
}

void originirParser::Double_gate_with_four_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_four_parameter_type(this);
}


antlrcpp::Any originirParser::Double_gate_with_four_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_with_four_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_with_four_parameter_typeContext* originirParser::double_gate_with_four_parameter_type() {
  Double_gate_with_four_parameter_typeContext *_localctx = _tracker.createInstance<Double_gate_with_four_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 48, originirParser::RuleDouble_gate_with_four_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(228);
    match(originirParser::CU_GATE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Triple_gate_without_parameter_typeContext ------------------------------------------------------------------

originirParser::Triple_gate_without_parameter_typeContext::Triple_gate_without_parameter_typeContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Triple_gate_without_parameter_typeContext::TOFFOLI_GATE() {
  return getToken(originirParser::TOFFOLI_GATE, 0);
}


size_t originirParser::Triple_gate_without_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleTriple_gate_without_parameter_type;
}

void originirParser::Triple_gate_without_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTriple_gate_without_parameter_type(this);
}

void originirParser::Triple_gate_without_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTriple_gate_without_parameter_type(this);
}


antlrcpp::Any originirParser::Triple_gate_without_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitTriple_gate_without_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Triple_gate_without_parameter_typeContext* originirParser::triple_gate_without_parameter_type() {
  Triple_gate_without_parameter_typeContext *_localctx = _tracker.createInstance<Triple_gate_without_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 50, originirParser::RuleTriple_gate_without_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(230);
    match(originirParser::TOFFOLI_GATE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Primary_expressionContext ------------------------------------------------------------------

originirParser::Primary_expressionContext::Primary_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t originirParser::Primary_expressionContext::getRuleIndex() const {
  return originirParser::RulePrimary_expression;
}

void originirParser::Primary_expressionContext::copyFrom(Primary_expressionContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Pri_ckeyContext ------------------------------------------------------------------

originirParser::C_KEY_declarationContext* originirParser::Pri_ckeyContext::c_KEY_declaration() {
  return getRuleContext<originirParser::C_KEY_declarationContext>(0);
}

originirParser::Pri_ckeyContext::Pri_ckeyContext(Primary_expressionContext *ctx) { copyFrom(ctx); }

void originirParser::Pri_ckeyContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPri_ckey(this);
}
void originirParser::Pri_ckeyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPri_ckey(this);
}

antlrcpp::Any originirParser::Pri_ckeyContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitPri_ckey(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Pri_cstContext ------------------------------------------------------------------

originirParser::ConstantContext* originirParser::Pri_cstContext::constant() {
  return getRuleContext<originirParser::ConstantContext>(0);
}

originirParser::Pri_cstContext::Pri_cstContext(Primary_expressionContext *ctx) { copyFrom(ctx); }

void originirParser::Pri_cstContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPri_cst(this);
}
void originirParser::Pri_cstContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPri_cst(this);
}

antlrcpp::Any originirParser::Pri_cstContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitPri_cst(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Pri_exprContext ------------------------------------------------------------------

std::vector<tree::TerminalNode *> originirParser::Pri_exprContext::LPAREN() {
  return getTokens(originirParser::LPAREN);
}

tree::TerminalNode* originirParser::Pri_exprContext::LPAREN(size_t i) {
  return getToken(originirParser::LPAREN, i);
}

originirParser::ExpressionContext* originirParser::Pri_exprContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

originirParser::Pri_exprContext::Pri_exprContext(Primary_expressionContext *ctx) { copyFrom(ctx); }

void originirParser::Pri_exprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPri_expr(this);
}
void originirParser::Pri_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPri_expr(this);
}

antlrcpp::Any originirParser::Pri_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitPri_expr(this);
  else
    return visitor->visitChildren(this);
}
originirParser::Primary_expressionContext* originirParser::primary_expression() {
  Primary_expressionContext *_localctx = _tracker.createInstance<Primary_expressionContext>(_ctx, getState());
  enterRule(_localctx, 52, originirParser::RulePrimary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(238);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::C_KEY: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_ckeyContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(232);
        c_KEY_declaration();
        break;
      }

      case originirParser::PI:
      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_cstContext>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(233);
        constant();
        break;
      }

      case originirParser::LPAREN: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_exprContext>(_localctx));
        enterOuterAlt(_localctx, 3);
        setState(234);
        match(originirParser::LPAREN);
        setState(235);
        expression();
        setState(236);
        match(originirParser::LPAREN);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Unary_expressionContext ------------------------------------------------------------------

originirParser::Unary_expressionContext::Unary_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Primary_expressionContext* originirParser::Unary_expressionContext::primary_expression() {
  return getRuleContext<originirParser::Primary_expressionContext>(0);
}

tree::TerminalNode* originirParser::Unary_expressionContext::PLUS() {
  return getToken(originirParser::PLUS, 0);
}

tree::TerminalNode* originirParser::Unary_expressionContext::MINUS() {
  return getToken(originirParser::MINUS, 0);
}

tree::TerminalNode* originirParser::Unary_expressionContext::NOT() {
  return getToken(originirParser::NOT, 0);
}


size_t originirParser::Unary_expressionContext::getRuleIndex() const {
  return originirParser::RuleUnary_expression;
}

void originirParser::Unary_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnary_expression(this);
}

void originirParser::Unary_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnary_expression(this);
}


antlrcpp::Any originirParser::Unary_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitUnary_expression(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Unary_expressionContext* originirParser::unary_expression() {
  Unary_expressionContext *_localctx = _tracker.createInstance<Unary_expressionContext>(_ctx, getState());
  enterRule(_localctx, 54, originirParser::RuleUnary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(247);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::PI:
      case originirParser::C_KEY:
      case originirParser::LPAREN:
      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        enterOuterAlt(_localctx, 1);
        setState(240);
        primary_expression();
        break;
      }

      case originirParser::PLUS: {
        enterOuterAlt(_localctx, 2);
        setState(241);
        match(originirParser::PLUS);
        setState(242);
        primary_expression();
        break;
      }

      case originirParser::MINUS: {
        enterOuterAlt(_localctx, 3);
        setState(243);
        match(originirParser::MINUS);
        setState(244);
        primary_expression();
        break;
      }

      case originirParser::NOT: {
        enterOuterAlt(_localctx, 4);
        setState(245);
        match(originirParser::NOT);
        setState(246);
        primary_expression();
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Multiplicative_expressionContext ------------------------------------------------------------------

originirParser::Multiplicative_expressionContext::Multiplicative_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Unary_expressionContext* originirParser::Multiplicative_expressionContext::unary_expression() {
  return getRuleContext<originirParser::Unary_expressionContext>(0);
}

originirParser::Multiplicative_expressionContext* originirParser::Multiplicative_expressionContext::multiplicative_expression() {
  return getRuleContext<originirParser::Multiplicative_expressionContext>(0);
}

tree::TerminalNode* originirParser::Multiplicative_expressionContext::MUL() {
  return getToken(originirParser::MUL, 0);
}

tree::TerminalNode* originirParser::Multiplicative_expressionContext::DIV() {
  return getToken(originirParser::DIV, 0);
}


size_t originirParser::Multiplicative_expressionContext::getRuleIndex() const {
  return originirParser::RuleMultiplicative_expression;
}

void originirParser::Multiplicative_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiplicative_expression(this);
}

void originirParser::Multiplicative_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiplicative_expression(this);
}


antlrcpp::Any originirParser::Multiplicative_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitMultiplicative_expression(this);
  else
    return visitor->visitChildren(this);
}


originirParser::Multiplicative_expressionContext* originirParser::multiplicative_expression() {
   return multiplicative_expression(0);
}

originirParser::Multiplicative_expressionContext* originirParser::multiplicative_expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  originirParser::Multiplicative_expressionContext *_localctx = _tracker.createInstance<Multiplicative_expressionContext>(_ctx, parentState);
  originirParser::Multiplicative_expressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 56;
  enterRecursionRule(_localctx, 56, originirParser::RuleMultiplicative_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(250);
    unary_expression();
    _ctx->stop = _input->LT(-1);
    setState(260);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(258);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Multiplicative_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression);
          setState(252);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(253);
          match(originirParser::MUL);
          setState(254);
          unary_expression();
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Multiplicative_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression);
          setState(255);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(256);
          match(originirParser::DIV);
          setState(257);
          unary_expression();
          break;
        }

        } 
      }
      setState(262);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Addtive_expressionContext ------------------------------------------------------------------

originirParser::Addtive_expressionContext::Addtive_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Multiplicative_expressionContext* originirParser::Addtive_expressionContext::multiplicative_expression() {
  return getRuleContext<originirParser::Multiplicative_expressionContext>(0);
}

originirParser::Addtive_expressionContext* originirParser::Addtive_expressionContext::addtive_expression() {
  return getRuleContext<originirParser::Addtive_expressionContext>(0);
}

tree::TerminalNode* originirParser::Addtive_expressionContext::PLUS() {
  return getToken(originirParser::PLUS, 0);
}

tree::TerminalNode* originirParser::Addtive_expressionContext::MINUS() {
  return getToken(originirParser::MINUS, 0);
}


size_t originirParser::Addtive_expressionContext::getRuleIndex() const {
  return originirParser::RuleAddtive_expression;
}

void originirParser::Addtive_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAddtive_expression(this);
}

void originirParser::Addtive_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAddtive_expression(this);
}


antlrcpp::Any originirParser::Addtive_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitAddtive_expression(this);
  else
    return visitor->visitChildren(this);
}


originirParser::Addtive_expressionContext* originirParser::addtive_expression() {
   return addtive_expression(0);
}

originirParser::Addtive_expressionContext* originirParser::addtive_expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  originirParser::Addtive_expressionContext *_localctx = _tracker.createInstance<Addtive_expressionContext>(_ctx, parentState);
  originirParser::Addtive_expressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 58;
  enterRecursionRule(_localctx, 58, originirParser::RuleAddtive_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(264);
    multiplicative_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(274);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(272);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Addtive_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleAddtive_expression);
          setState(266);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(267);
          match(originirParser::PLUS);
          setState(268);
          multiplicative_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Addtive_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleAddtive_expression);
          setState(269);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(270);
          match(originirParser::MINUS);
          setState(271);
          multiplicative_expression(0);
          break;
        }

        } 
      }
      setState(276);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Relational_expressionContext ------------------------------------------------------------------

originirParser::Relational_expressionContext::Relational_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Addtive_expressionContext* originirParser::Relational_expressionContext::addtive_expression() {
  return getRuleContext<originirParser::Addtive_expressionContext>(0);
}

originirParser::Relational_expressionContext* originirParser::Relational_expressionContext::relational_expression() {
  return getRuleContext<originirParser::Relational_expressionContext>(0);
}

tree::TerminalNode* originirParser::Relational_expressionContext::LT() {
  return getToken(originirParser::LT, 0);
}

tree::TerminalNode* originirParser::Relational_expressionContext::GT() {
  return getToken(originirParser::GT, 0);
}

tree::TerminalNode* originirParser::Relational_expressionContext::LEQ() {
  return getToken(originirParser::LEQ, 0);
}

tree::TerminalNode* originirParser::Relational_expressionContext::GEQ() {
  return getToken(originirParser::GEQ, 0);
}


size_t originirParser::Relational_expressionContext::getRuleIndex() const {
  return originirParser::RuleRelational_expression;
}

void originirParser::Relational_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRelational_expression(this);
}

void originirParser::Relational_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRelational_expression(this);
}


antlrcpp::Any originirParser::Relational_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitRelational_expression(this);
  else
    return visitor->visitChildren(this);
}


originirParser::Relational_expressionContext* originirParser::relational_expression() {
   return relational_expression(0);
}

originirParser::Relational_expressionContext* originirParser::relational_expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  originirParser::Relational_expressionContext *_localctx = _tracker.createInstance<Relational_expressionContext>(_ctx, parentState);
  originirParser::Relational_expressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 60;
  enterRecursionRule(_localctx, 60, originirParser::RuleRelational_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(278);
    addtive_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(294);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(292);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(280);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(281);
          match(originirParser::LT);
          setState(282);
          addtive_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(283);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(284);
          match(originirParser::GT);
          setState(285);
          addtive_expression(0);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(286);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(287);
          match(originirParser::LEQ);
          setState(288);
          addtive_expression(0);
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(289);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(290);
          match(originirParser::GEQ);
          setState(291);
          addtive_expression(0);
          break;
        }

        } 
      }
      setState(296);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Equality_expressionContext ------------------------------------------------------------------

originirParser::Equality_expressionContext::Equality_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Relational_expressionContext* originirParser::Equality_expressionContext::relational_expression() {
  return getRuleContext<originirParser::Relational_expressionContext>(0);
}

originirParser::Equality_expressionContext* originirParser::Equality_expressionContext::equality_expression() {
  return getRuleContext<originirParser::Equality_expressionContext>(0);
}

tree::TerminalNode* originirParser::Equality_expressionContext::EQ() {
  return getToken(originirParser::EQ, 0);
}

tree::TerminalNode* originirParser::Equality_expressionContext::NE() {
  return getToken(originirParser::NE, 0);
}


size_t originirParser::Equality_expressionContext::getRuleIndex() const {
  return originirParser::RuleEquality_expression;
}

void originirParser::Equality_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEquality_expression(this);
}

void originirParser::Equality_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEquality_expression(this);
}


antlrcpp::Any originirParser::Equality_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitEquality_expression(this);
  else
    return visitor->visitChildren(this);
}


originirParser::Equality_expressionContext* originirParser::equality_expression() {
   return equality_expression(0);
}

originirParser::Equality_expressionContext* originirParser::equality_expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  originirParser::Equality_expressionContext *_localctx = _tracker.createInstance<Equality_expressionContext>(_ctx, parentState);
  originirParser::Equality_expressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 62;
  enterRecursionRule(_localctx, 62, originirParser::RuleEquality_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(298);
    relational_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(308);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(306);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Equality_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleEquality_expression);
          setState(300);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(301);
          match(originirParser::EQ);
          setState(302);
          relational_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Equality_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleEquality_expression);
          setState(303);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(304);
          match(originirParser::NE);
          setState(305);
          relational_expression(0);
          break;
        }

        } 
      }
      setState(310);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Logical_and_expressionContext ------------------------------------------------------------------

originirParser::Logical_and_expressionContext::Logical_and_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Equality_expressionContext* originirParser::Logical_and_expressionContext::equality_expression() {
  return getRuleContext<originirParser::Equality_expressionContext>(0);
}

originirParser::Logical_and_expressionContext* originirParser::Logical_and_expressionContext::logical_and_expression() {
  return getRuleContext<originirParser::Logical_and_expressionContext>(0);
}

tree::TerminalNode* originirParser::Logical_and_expressionContext::AND() {
  return getToken(originirParser::AND, 0);
}


size_t originirParser::Logical_and_expressionContext::getRuleIndex() const {
  return originirParser::RuleLogical_and_expression;
}

void originirParser::Logical_and_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogical_and_expression(this);
}

void originirParser::Logical_and_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogical_and_expression(this);
}


antlrcpp::Any originirParser::Logical_and_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitLogical_and_expression(this);
  else
    return visitor->visitChildren(this);
}


originirParser::Logical_and_expressionContext* originirParser::logical_and_expression() {
   return logical_and_expression(0);
}

originirParser::Logical_and_expressionContext* originirParser::logical_and_expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  originirParser::Logical_and_expressionContext *_localctx = _tracker.createInstance<Logical_and_expressionContext>(_ctx, parentState);
  originirParser::Logical_and_expressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 64;
  enterRecursionRule(_localctx, 64, originirParser::RuleLogical_and_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(312);
    equality_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(319);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<Logical_and_expressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLogical_and_expression);
        setState(314);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(315);
        match(originirParser::AND);
        setState(316);
        equality_expression(0); 
      }
      setState(321);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Logical_or_expressionContext ------------------------------------------------------------------

originirParser::Logical_or_expressionContext::Logical_or_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Logical_and_expressionContext* originirParser::Logical_or_expressionContext::logical_and_expression() {
  return getRuleContext<originirParser::Logical_and_expressionContext>(0);
}

originirParser::Logical_or_expressionContext* originirParser::Logical_or_expressionContext::logical_or_expression() {
  return getRuleContext<originirParser::Logical_or_expressionContext>(0);
}

tree::TerminalNode* originirParser::Logical_or_expressionContext::OR() {
  return getToken(originirParser::OR, 0);
}


size_t originirParser::Logical_or_expressionContext::getRuleIndex() const {
  return originirParser::RuleLogical_or_expression;
}

void originirParser::Logical_or_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogical_or_expression(this);
}

void originirParser::Logical_or_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogical_or_expression(this);
}


antlrcpp::Any originirParser::Logical_or_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitLogical_or_expression(this);
  else
    return visitor->visitChildren(this);
}


originirParser::Logical_or_expressionContext* originirParser::logical_or_expression() {
   return logical_or_expression(0);
}

originirParser::Logical_or_expressionContext* originirParser::logical_or_expression(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  originirParser::Logical_or_expressionContext *_localctx = _tracker.createInstance<Logical_or_expressionContext>(_ctx, parentState);
  originirParser::Logical_or_expressionContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 66;
  enterRecursionRule(_localctx, 66, originirParser::RuleLogical_or_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(323);
    logical_and_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(330);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<Logical_or_expressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLogical_or_expression);
        setState(325);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(326);
        match(originirParser::OR);
        setState(327);
        logical_and_expression(0); 
      }
      setState(332);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Assignment_expressionContext ------------------------------------------------------------------

originirParser::Assignment_expressionContext::Assignment_expressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Logical_or_expressionContext* originirParser::Assignment_expressionContext::logical_or_expression() {
  return getRuleContext<originirParser::Logical_or_expressionContext>(0);
}

originirParser::C_KEY_declarationContext* originirParser::Assignment_expressionContext::c_KEY_declaration() {
  return getRuleContext<originirParser::C_KEY_declarationContext>(0);
}

tree::TerminalNode* originirParser::Assignment_expressionContext::ASSIGN() {
  return getToken(originirParser::ASSIGN, 0);
}


size_t originirParser::Assignment_expressionContext::getRuleIndex() const {
  return originirParser::RuleAssignment_expression;
}

void originirParser::Assignment_expressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignment_expression(this);
}

void originirParser::Assignment_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignment_expression(this);
}


antlrcpp::Any originirParser::Assignment_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitAssignment_expression(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Assignment_expressionContext* originirParser::assignment_expression() {
  Assignment_expressionContext *_localctx = _tracker.createInstance<Assignment_expressionContext>(_ctx, getState());
  enterRule(_localctx, 68, originirParser::RuleAssignment_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(338);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(333);
      logical_or_expression(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(334);
      c_KEY_declaration();
      setState(335);
      match(originirParser::ASSIGN);
      setState(336);
      logical_or_expression(0);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExpressionContext ------------------------------------------------------------------

originirParser::ExpressionContext::ExpressionContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Assignment_expressionContext* originirParser::ExpressionContext::assignment_expression() {
  return getRuleContext<originirParser::Assignment_expressionContext>(0);
}


size_t originirParser::ExpressionContext::getRuleIndex() const {
  return originirParser::RuleExpression;
}

void originirParser::ExpressionContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression(this);
}

void originirParser::ExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression(this);
}


antlrcpp::Any originirParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

originirParser::ExpressionContext* originirParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 70, originirParser::RuleExpression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(340);
    assignment_expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Controlbit_listContext ------------------------------------------------------------------

originirParser::Controlbit_listContext::Controlbit_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<originirParser::Q_KEY_declarationContext *> originirParser::Controlbit_listContext::q_KEY_declaration() {
  return getRuleContexts<originirParser::Q_KEY_declarationContext>();
}

originirParser::Q_KEY_declarationContext* originirParser::Controlbit_listContext::q_KEY_declaration(size_t i) {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::Controlbit_listContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Controlbit_listContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}


size_t originirParser::Controlbit_listContext::getRuleIndex() const {
  return originirParser::RuleControlbit_list;
}

void originirParser::Controlbit_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterControlbit_list(this);
}

void originirParser::Controlbit_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitControlbit_list(this);
}


antlrcpp::Any originirParser::Controlbit_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitControlbit_list(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Controlbit_listContext* originirParser::controlbit_list() {
  Controlbit_listContext *_localctx = _tracker.createInstance<Controlbit_listContext>(_ctx, getState());
  enterRule(_localctx, 72, originirParser::RuleControlbit_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(342);
    q_KEY_declaration();
    setState(347);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == originirParser::COMMA) {
      setState(343);
      match(originirParser::COMMA);
      setState(344);
      q_KEY_declaration();
      setState(349);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

originirParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Quantum_gate_declarationContext* originirParser::StatementContext::quantum_gate_declaration() {
  return getRuleContext<originirParser::Quantum_gate_declarationContext>(0);
}

tree::TerminalNode* originirParser::StatementContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}

originirParser::Control_statementContext* originirParser::StatementContext::control_statement() {
  return getRuleContext<originirParser::Control_statementContext>(0);
}

originirParser::Qif_statementContext* originirParser::StatementContext::qif_statement() {
  return getRuleContext<originirParser::Qif_statementContext>(0);
}

originirParser::Qwhile_statementContext* originirParser::StatementContext::qwhile_statement() {
  return getRuleContext<originirParser::Qwhile_statementContext>(0);
}

originirParser::Dagger_statementContext* originirParser::StatementContext::dagger_statement() {
  return getRuleContext<originirParser::Dagger_statementContext>(0);
}

originirParser::Measure_statementContext* originirParser::StatementContext::measure_statement() {
  return getRuleContext<originirParser::Measure_statementContext>(0);
}

originirParser::Reset_statementContext* originirParser::StatementContext::reset_statement() {
  return getRuleContext<originirParser::Reset_statementContext>(0);
}

originirParser::Expression_statementContext* originirParser::StatementContext::expression_statement() {
  return getRuleContext<originirParser::Expression_statementContext>(0);
}

originirParser::Barrier_statementContext* originirParser::StatementContext::barrier_statement() {
  return getRuleContext<originirParser::Barrier_statementContext>(0);
}


size_t originirParser::StatementContext::getRuleIndex() const {
  return originirParser::RuleStatement;
}

void originirParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void originirParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}


antlrcpp::Any originirParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::StatementContext* originirParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 74, originirParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(369);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::ECHO_GATE:
      case originirParser::H_GATE:
      case originirParser::X_GATE:
      case originirParser::T_GATE:
      case originirParser::S_GATE:
      case originirParser::Y_GATE:
      case originirParser::Z_GATE:
      case originirParser::X1_GATE:
      case originirParser::Y1_GATE:
      case originirParser::Z1_GATE:
      case originirParser::I_GATE:
      case originirParser::U2_GATE:
      case originirParser::RPHI_GATE:
      case originirParser::U3_GATE:
      case originirParser::U4_GATE:
      case originirParser::RX_GATE:
      case originirParser::RY_GATE:
      case originirParser::RZ_GATE:
      case originirParser::U1_GATE:
      case originirParser::CNOT_GATE:
      case originirParser::CZ_GATE:
      case originirParser::CU_GATE:
      case originirParser::ISWAP_GATE:
      case originirParser::SQISWAP_GATE:
      case originirParser::SWAPZ1_GATE:
      case originirParser::ISWAPTHETA_GATE:
      case originirParser::CR_GATE:
      case originirParser::TOFFOLI_GATE: {
        enterOuterAlt(_localctx, 1);
        setState(350);
        quantum_gate_declaration();
        setState(351);
        match(originirParser::NEWLINE);
        break;
      }

      case originirParser::CONTROL_KEY: {
        enterOuterAlt(_localctx, 2);
        setState(353);
        control_statement();
        break;
      }

      case originirParser::QIF_KEY: {
        enterOuterAlt(_localctx, 3);
        setState(354);
        qif_statement();
        break;
      }

      case originirParser::QWHILE_KEY: {
        enterOuterAlt(_localctx, 4);
        setState(355);
        qwhile_statement();
        break;
      }

      case originirParser::DAGGER_KEY: {
        enterOuterAlt(_localctx, 5);
        setState(356);
        dagger_statement();
        break;
      }

      case originirParser::MEASURE_KEY: {
        enterOuterAlt(_localctx, 6);
        setState(357);
        measure_statement();
        setState(358);
        match(originirParser::NEWLINE);
        break;
      }

      case originirParser::RESET_KEY: {
        enterOuterAlt(_localctx, 7);
        setState(360);
        reset_statement();
        setState(361);
        match(originirParser::NEWLINE);
        break;
      }

      case originirParser::PI:
      case originirParser::C_KEY:
      case originirParser::NOT:
      case originirParser::PLUS:
      case originirParser::MINUS:
      case originirParser::LPAREN:
      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        enterOuterAlt(_localctx, 8);
        setState(363);
        expression_statement();
        setState(364);
        match(originirParser::NEWLINE);
        break;
      }

      case originirParser::BARRIER_KEY: {
        enterOuterAlt(_localctx, 9);
        setState(366);
        barrier_statement();
        setState(367);
        match(originirParser::NEWLINE);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Dagger_statementContext ------------------------------------------------------------------

originirParser::Dagger_statementContext::Dagger_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Dagger_statementContext::DAGGER_KEY() {
  return getToken(originirParser::DAGGER_KEY, 0);
}

std::vector<tree::TerminalNode *> originirParser::Dagger_statementContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Dagger_statementContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

tree::TerminalNode* originirParser::Dagger_statementContext::ENDDAGGER_KEY() {
  return getToken(originirParser::ENDDAGGER_KEY, 0);
}

std::vector<originirParser::StatementContext *> originirParser::Dagger_statementContext::statement() {
  return getRuleContexts<originirParser::StatementContext>();
}

originirParser::StatementContext* originirParser::Dagger_statementContext::statement(size_t i) {
  return getRuleContext<originirParser::StatementContext>(i);
}


size_t originirParser::Dagger_statementContext::getRuleIndex() const {
  return originirParser::RuleDagger_statement;
}

void originirParser::Dagger_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDagger_statement(this);
}

void originirParser::Dagger_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDagger_statement(this);
}


antlrcpp::Any originirParser::Dagger_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitDagger_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Dagger_statementContext* originirParser::dagger_statement() {
  Dagger_statementContext *_localctx = _tracker.createInstance<Dagger_statementContext>(_ctx, getState());
  enterRule(_localctx, 76, originirParser::RuleDagger_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(371);
    match(originirParser::DAGGER_KEY);
    setState(372);
    match(originirParser::NEWLINE);
    setState(376);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::ECHO_GATE)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
      | (1ULL << originirParser::I_GATE)
      | (1ULL << originirParser::U2_GATE)
      | (1ULL << originirParser::RPHI_GATE)
      | (1ULL << originirParser::U3_GATE)
      | (1ULL << originirParser::U4_GATE)
      | (1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN))) != 0) || _la == originirParser::Integer_Literal

    || _la == originirParser::Double_Literal) {
      setState(373);
      statement();
      setState(378);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(379);
    match(originirParser::ENDDAGGER_KEY);
    setState(380);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Control_statementContext ------------------------------------------------------------------

originirParser::Control_statementContext::Control_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Control_statementContext::CONTROL_KEY() {
  return getToken(originirParser::CONTROL_KEY, 0);
}

originirParser::Controlbit_listContext* originirParser::Control_statementContext::controlbit_list() {
  return getRuleContext<originirParser::Controlbit_listContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Control_statementContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Control_statementContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

tree::TerminalNode* originirParser::Control_statementContext::ENDCONTROL_KEY() {
  return getToken(originirParser::ENDCONTROL_KEY, 0);
}

std::vector<originirParser::StatementContext *> originirParser::Control_statementContext::statement() {
  return getRuleContexts<originirParser::StatementContext>();
}

originirParser::StatementContext* originirParser::Control_statementContext::statement(size_t i) {
  return getRuleContext<originirParser::StatementContext>(i);
}


size_t originirParser::Control_statementContext::getRuleIndex() const {
  return originirParser::RuleControl_statement;
}

void originirParser::Control_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterControl_statement(this);
}

void originirParser::Control_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitControl_statement(this);
}


antlrcpp::Any originirParser::Control_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitControl_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Control_statementContext* originirParser::control_statement() {
  Control_statementContext *_localctx = _tracker.createInstance<Control_statementContext>(_ctx, getState());
  enterRule(_localctx, 78, originirParser::RuleControl_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(382);
    match(originirParser::CONTROL_KEY);
    setState(383);
    controlbit_list();
    setState(384);
    match(originirParser::NEWLINE);
    setState(388);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::ECHO_GATE)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
      | (1ULL << originirParser::I_GATE)
      | (1ULL << originirParser::U2_GATE)
      | (1ULL << originirParser::RPHI_GATE)
      | (1ULL << originirParser::U3_GATE)
      | (1ULL << originirParser::U4_GATE)
      | (1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN))) != 0) || _la == originirParser::Integer_Literal

    || _la == originirParser::Double_Literal) {
      setState(385);
      statement();
      setState(390);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(391);
    match(originirParser::ENDCONTROL_KEY);
    setState(392);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Qelse_statement_fragmentContext ------------------------------------------------------------------

originirParser::Qelse_statement_fragmentContext::Qelse_statement_fragmentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Qelse_statement_fragmentContext::ELSE_KEY() {
  return getToken(originirParser::ELSE_KEY, 0);
}

tree::TerminalNode* originirParser::Qelse_statement_fragmentContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}

std::vector<originirParser::StatementContext *> originirParser::Qelse_statement_fragmentContext::statement() {
  return getRuleContexts<originirParser::StatementContext>();
}

originirParser::StatementContext* originirParser::Qelse_statement_fragmentContext::statement(size_t i) {
  return getRuleContext<originirParser::StatementContext>(i);
}


size_t originirParser::Qelse_statement_fragmentContext::getRuleIndex() const {
  return originirParser::RuleQelse_statement_fragment;
}

void originirParser::Qelse_statement_fragmentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQelse_statement_fragment(this);
}

void originirParser::Qelse_statement_fragmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQelse_statement_fragment(this);
}


antlrcpp::Any originirParser::Qelse_statement_fragmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitQelse_statement_fragment(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Qelse_statement_fragmentContext* originirParser::qelse_statement_fragment() {
  Qelse_statement_fragmentContext *_localctx = _tracker.createInstance<Qelse_statement_fragmentContext>(_ctx, getState());
  enterRule(_localctx, 80, originirParser::RuleQelse_statement_fragment);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(394);
    match(originirParser::ELSE_KEY);
    setState(395);
    match(originirParser::NEWLINE);
    setState(399);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::ECHO_GATE)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
      | (1ULL << originirParser::I_GATE)
      | (1ULL << originirParser::U2_GATE)
      | (1ULL << originirParser::RPHI_GATE)
      | (1ULL << originirParser::U3_GATE)
      | (1ULL << originirParser::U4_GATE)
      | (1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN))) != 0) || _la == originirParser::Integer_Literal

    || _la == originirParser::Double_Literal) {
      setState(396);
      statement();
      setState(401);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Qif_statementContext ------------------------------------------------------------------

originirParser::Qif_statementContext::Qif_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}


size_t originirParser::Qif_statementContext::getRuleIndex() const {
  return originirParser::RuleQif_statement;
}

void originirParser::Qif_statementContext::copyFrom(Qif_statementContext *ctx) {
  ParserRuleContext::copyFrom(ctx);
}

//----------------- Qif_ifContext ------------------------------------------------------------------

tree::TerminalNode* originirParser::Qif_ifContext::QIF_KEY() {
  return getToken(originirParser::QIF_KEY, 0);
}

originirParser::ExpressionContext* originirParser::Qif_ifContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Qif_ifContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Qif_ifContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

originirParser::Qelse_statement_fragmentContext* originirParser::Qif_ifContext::qelse_statement_fragment() {
  return getRuleContext<originirParser::Qelse_statement_fragmentContext>(0);
}

tree::TerminalNode* originirParser::Qif_ifContext::ENDIF_KEY() {
  return getToken(originirParser::ENDIF_KEY, 0);
}

std::vector<originirParser::StatementContext *> originirParser::Qif_ifContext::statement() {
  return getRuleContexts<originirParser::StatementContext>();
}

originirParser::StatementContext* originirParser::Qif_ifContext::statement(size_t i) {
  return getRuleContext<originirParser::StatementContext>(i);
}

originirParser::Qif_ifContext::Qif_ifContext(Qif_statementContext *ctx) { copyFrom(ctx); }

void originirParser::Qif_ifContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQif_if(this);
}
void originirParser::Qif_ifContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQif_if(this);
}

antlrcpp::Any originirParser::Qif_ifContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitQif_if(this);
  else
    return visitor->visitChildren(this);
}
//----------------- Qif_ifelseContext ------------------------------------------------------------------

tree::TerminalNode* originirParser::Qif_ifelseContext::QIF_KEY() {
  return getToken(originirParser::QIF_KEY, 0);
}

originirParser::ExpressionContext* originirParser::Qif_ifelseContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Qif_ifelseContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Qif_ifelseContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

tree::TerminalNode* originirParser::Qif_ifelseContext::ENDIF_KEY() {
  return getToken(originirParser::ENDIF_KEY, 0);
}

std::vector<originirParser::StatementContext *> originirParser::Qif_ifelseContext::statement() {
  return getRuleContexts<originirParser::StatementContext>();
}

originirParser::StatementContext* originirParser::Qif_ifelseContext::statement(size_t i) {
  return getRuleContext<originirParser::StatementContext>(i);
}

originirParser::Qif_ifelseContext::Qif_ifelseContext(Qif_statementContext *ctx) { copyFrom(ctx); }

void originirParser::Qif_ifelseContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQif_ifelse(this);
}
void originirParser::Qif_ifelseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQif_ifelse(this);
}

antlrcpp::Any originirParser::Qif_ifelseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitQif_ifelse(this);
  else
    return visitor->visitChildren(this);
}
originirParser::Qif_statementContext* originirParser::qif_statement() {
  Qif_statementContext *_localctx = _tracker.createInstance<Qif_statementContext>(_ctx, getState());
  enterRule(_localctx, 82, originirParser::RuleQif_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(427);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 22, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<Qif_statementContext *>(_tracker.createInstance<originirParser::Qif_ifContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(402);
      match(originirParser::QIF_KEY);
      setState(403);
      expression();
      setState(404);
      match(originirParser::NEWLINE);
      setState(408);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << originirParser::PI)
        | (1ULL << originirParser::C_KEY)
        | (1ULL << originirParser::BARRIER_KEY)
        | (1ULL << originirParser::ECHO_GATE)
        | (1ULL << originirParser::H_GATE)
        | (1ULL << originirParser::X_GATE)
        | (1ULL << originirParser::T_GATE)
        | (1ULL << originirParser::S_GATE)
        | (1ULL << originirParser::Y_GATE)
        | (1ULL << originirParser::Z_GATE)
        | (1ULL << originirParser::X1_GATE)
        | (1ULL << originirParser::Y1_GATE)
        | (1ULL << originirParser::Z1_GATE)
        | (1ULL << originirParser::I_GATE)
        | (1ULL << originirParser::U2_GATE)
        | (1ULL << originirParser::RPHI_GATE)
        | (1ULL << originirParser::U3_GATE)
        | (1ULL << originirParser::U4_GATE)
        | (1ULL << originirParser::RX_GATE)
        | (1ULL << originirParser::RY_GATE)
        | (1ULL << originirParser::RZ_GATE)
        | (1ULL << originirParser::U1_GATE)
        | (1ULL << originirParser::CNOT_GATE)
        | (1ULL << originirParser::CZ_GATE)
        | (1ULL << originirParser::CU_GATE)
        | (1ULL << originirParser::ISWAP_GATE)
        | (1ULL << originirParser::SQISWAP_GATE)
        | (1ULL << originirParser::SWAPZ1_GATE)
        | (1ULL << originirParser::ISWAPTHETA_GATE)
        | (1ULL << originirParser::CR_GATE)
        | (1ULL << originirParser::TOFFOLI_GATE)
        | (1ULL << originirParser::DAGGER_KEY)
        | (1ULL << originirParser::CONTROL_KEY)
        | (1ULL << originirParser::QIF_KEY)
        | (1ULL << originirParser::QWHILE_KEY)
        | (1ULL << originirParser::MEASURE_KEY)
        | (1ULL << originirParser::RESET_KEY)
        | (1ULL << originirParser::NOT)
        | (1ULL << originirParser::PLUS)
        | (1ULL << originirParser::MINUS)
        | (1ULL << originirParser::LPAREN))) != 0) || _la == originirParser::Integer_Literal

      || _la == originirParser::Double_Literal) {
        setState(405);
        statement();
        setState(410);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(411);
      qelse_statement_fragment();
      setState(412);
      match(originirParser::ENDIF_KEY);
      setState(413);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      _localctx = dynamic_cast<Qif_statementContext *>(_tracker.createInstance<originirParser::Qif_ifelseContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(415);
      match(originirParser::QIF_KEY);
      setState(416);
      expression();
      setState(417);
      match(originirParser::NEWLINE);
      setState(421);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << originirParser::PI)
        | (1ULL << originirParser::C_KEY)
        | (1ULL << originirParser::BARRIER_KEY)
        | (1ULL << originirParser::ECHO_GATE)
        | (1ULL << originirParser::H_GATE)
        | (1ULL << originirParser::X_GATE)
        | (1ULL << originirParser::T_GATE)
        | (1ULL << originirParser::S_GATE)
        | (1ULL << originirParser::Y_GATE)
        | (1ULL << originirParser::Z_GATE)
        | (1ULL << originirParser::X1_GATE)
        | (1ULL << originirParser::Y1_GATE)
        | (1ULL << originirParser::Z1_GATE)
        | (1ULL << originirParser::I_GATE)
        | (1ULL << originirParser::U2_GATE)
        | (1ULL << originirParser::RPHI_GATE)
        | (1ULL << originirParser::U3_GATE)
        | (1ULL << originirParser::U4_GATE)
        | (1ULL << originirParser::RX_GATE)
        | (1ULL << originirParser::RY_GATE)
        | (1ULL << originirParser::RZ_GATE)
        | (1ULL << originirParser::U1_GATE)
        | (1ULL << originirParser::CNOT_GATE)
        | (1ULL << originirParser::CZ_GATE)
        | (1ULL << originirParser::CU_GATE)
        | (1ULL << originirParser::ISWAP_GATE)
        | (1ULL << originirParser::SQISWAP_GATE)
        | (1ULL << originirParser::SWAPZ1_GATE)
        | (1ULL << originirParser::ISWAPTHETA_GATE)
        | (1ULL << originirParser::CR_GATE)
        | (1ULL << originirParser::TOFFOLI_GATE)
        | (1ULL << originirParser::DAGGER_KEY)
        | (1ULL << originirParser::CONTROL_KEY)
        | (1ULL << originirParser::QIF_KEY)
        | (1ULL << originirParser::QWHILE_KEY)
        | (1ULL << originirParser::MEASURE_KEY)
        | (1ULL << originirParser::RESET_KEY)
        | (1ULL << originirParser::NOT)
        | (1ULL << originirParser::PLUS)
        | (1ULL << originirParser::MINUS)
        | (1ULL << originirParser::LPAREN))) != 0) || _la == originirParser::Integer_Literal

      || _la == originirParser::Double_Literal) {
        setState(418);
        statement();
        setState(423);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(424);
      match(originirParser::ENDIF_KEY);
      setState(425);
      match(originirParser::NEWLINE);
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Qwhile_statementContext ------------------------------------------------------------------

originirParser::Qwhile_statementContext::Qwhile_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Qwhile_statementContext::QWHILE_KEY() {
  return getToken(originirParser::QWHILE_KEY, 0);
}

originirParser::ExpressionContext* originirParser::Qwhile_statementContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Qwhile_statementContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Qwhile_statementContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

tree::TerminalNode* originirParser::Qwhile_statementContext::ENDQWHILE_KEY() {
  return getToken(originirParser::ENDQWHILE_KEY, 0);
}

std::vector<originirParser::StatementContext *> originirParser::Qwhile_statementContext::statement() {
  return getRuleContexts<originirParser::StatementContext>();
}

originirParser::StatementContext* originirParser::Qwhile_statementContext::statement(size_t i) {
  return getRuleContext<originirParser::StatementContext>(i);
}


size_t originirParser::Qwhile_statementContext::getRuleIndex() const {
  return originirParser::RuleQwhile_statement;
}

void originirParser::Qwhile_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQwhile_statement(this);
}

void originirParser::Qwhile_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQwhile_statement(this);
}


antlrcpp::Any originirParser::Qwhile_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitQwhile_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Qwhile_statementContext* originirParser::qwhile_statement() {
  Qwhile_statementContext *_localctx = _tracker.createInstance<Qwhile_statementContext>(_ctx, getState());
  enterRule(_localctx, 84, originirParser::RuleQwhile_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(429);
    match(originirParser::QWHILE_KEY);
    setState(430);
    expression();
    setState(431);
    match(originirParser::NEWLINE);
    setState(435);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::ECHO_GATE)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
      | (1ULL << originirParser::I_GATE)
      | (1ULL << originirParser::U2_GATE)
      | (1ULL << originirParser::RPHI_GATE)
      | (1ULL << originirParser::U3_GATE)
      | (1ULL << originirParser::U4_GATE)
      | (1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN))) != 0) || _la == originirParser::Integer_Literal

    || _la == originirParser::Double_Literal) {
      setState(432);
      statement();
      setState(437);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(438);
    match(originirParser::ENDQWHILE_KEY);
    setState(439);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Measure_statementContext ------------------------------------------------------------------

originirParser::Measure_statementContext::Measure_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Measure_statementContext::MEASURE_KEY() {
  return getToken(originirParser::MEASURE_KEY, 0);
}

originirParser::Q_KEY_declarationContext* originirParser::Measure_statementContext::q_KEY_declaration() {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(0);
}

tree::TerminalNode* originirParser::Measure_statementContext::COMMA() {
  return getToken(originirParser::COMMA, 0);
}

originirParser::C_KEY_declarationContext* originirParser::Measure_statementContext::c_KEY_declaration() {
  return getRuleContext<originirParser::C_KEY_declarationContext>(0);
}


size_t originirParser::Measure_statementContext::getRuleIndex() const {
  return originirParser::RuleMeasure_statement;
}

void originirParser::Measure_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMeasure_statement(this);
}

void originirParser::Measure_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMeasure_statement(this);
}


antlrcpp::Any originirParser::Measure_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitMeasure_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Measure_statementContext* originirParser::measure_statement() {
  Measure_statementContext *_localctx = _tracker.createInstance<Measure_statementContext>(_ctx, getState());
  enterRule(_localctx, 86, originirParser::RuleMeasure_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(441);
    match(originirParser::MEASURE_KEY);
    setState(442);
    q_KEY_declaration();
    setState(443);
    match(originirParser::COMMA);
    setState(444);
    c_KEY_declaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Reset_statementContext ------------------------------------------------------------------

originirParser::Reset_statementContext::Reset_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Reset_statementContext::RESET_KEY() {
  return getToken(originirParser::RESET_KEY, 0);
}

originirParser::Q_KEY_declarationContext* originirParser::Reset_statementContext::q_KEY_declaration() {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(0);
}


size_t originirParser::Reset_statementContext::getRuleIndex() const {
  return originirParser::RuleReset_statement;
}

void originirParser::Reset_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReset_statement(this);
}

void originirParser::Reset_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReset_statement(this);
}


antlrcpp::Any originirParser::Reset_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitReset_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Reset_statementContext* originirParser::reset_statement() {
  Reset_statementContext *_localctx = _tracker.createInstance<Reset_statementContext>(_ctx, getState());
  enterRule(_localctx, 88, originirParser::RuleReset_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(446);
    match(originirParser::RESET_KEY);
    setState(447);
    q_KEY_declaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Barrier_statementContext ------------------------------------------------------------------

originirParser::Barrier_statementContext::Barrier_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Barrier_statementContext::BARRIER_KEY() {
  return getToken(originirParser::BARRIER_KEY, 0);
}

originirParser::Controlbit_listContext* originirParser::Barrier_statementContext::controlbit_list() {
  return getRuleContext<originirParser::Controlbit_listContext>(0);
}


size_t originirParser::Barrier_statementContext::getRuleIndex() const {
  return originirParser::RuleBarrier_statement;
}

void originirParser::Barrier_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBarrier_statement(this);
}

void originirParser::Barrier_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBarrier_statement(this);
}


antlrcpp::Any originirParser::Barrier_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitBarrier_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Barrier_statementContext* originirParser::barrier_statement() {
  Barrier_statementContext *_localctx = _tracker.createInstance<Barrier_statementContext>(_ctx, getState());
  enterRule(_localctx, 90, originirParser::RuleBarrier_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(449);
    match(originirParser::BARRIER_KEY);
    setState(450);
    controlbit_list();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Expression_statementContext ------------------------------------------------------------------

originirParser::Expression_statementContext::Expression_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::ExpressionContext* originirParser::Expression_statementContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}


size_t originirParser::Expression_statementContext::getRuleIndex() const {
  return originirParser::RuleExpression_statement;
}

void originirParser::Expression_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression_statement(this);
}

void originirParser::Expression_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression_statement(this);
}


antlrcpp::Any originirParser::Expression_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitExpression_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Expression_statementContext* originirParser::expression_statement() {
  Expression_statementContext *_localctx = _tracker.createInstance<Expression_statementContext>(_ctx, getState());
  enterRule(_localctx, 92, originirParser::RuleExpression_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(452);
    expression();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ConstantContext ------------------------------------------------------------------

originirParser::ConstantContext::ConstantContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::ConstantContext::Integer_Literal() {
  return getToken(originirParser::Integer_Literal, 0);
}

tree::TerminalNode* originirParser::ConstantContext::Double_Literal() {
  return getToken(originirParser::Double_Literal, 0);
}

tree::TerminalNode* originirParser::ConstantContext::PI() {
  return getToken(originirParser::PI, 0);
}


size_t originirParser::ConstantContext::getRuleIndex() const {
  return originirParser::RuleConstant;
}

void originirParser::ConstantContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstant(this);
}

void originirParser::ConstantContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstant(this);
}


antlrcpp::Any originirParser::ConstantContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirVisitor*>(visitor))
    return parserVisitor->visitConstant(this);
  else
    return visitor->visitChildren(this);
}

originirParser::ConstantContext* originirParser::constant() {
  ConstantContext *_localctx = _tracker.createInstance<ConstantContext>(_ctx, getState());
  enterRule(_localctx, 94, originirParser::RuleConstant);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(454);
    _la = _input->LA(1);
    if (!(_la == originirParser::PI || _la == originirParser::Integer_Literal

    || _la == originirParser::Double_Literal)) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool originirParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 28: return multiplicative_expressionSempred(dynamic_cast<Multiplicative_expressionContext *>(context), predicateIndex);
    case 29: return addtive_expressionSempred(dynamic_cast<Addtive_expressionContext *>(context), predicateIndex);
    case 30: return relational_expressionSempred(dynamic_cast<Relational_expressionContext *>(context), predicateIndex);
    case 31: return equality_expressionSempred(dynamic_cast<Equality_expressionContext *>(context), predicateIndex);
    case 32: return logical_and_expressionSempred(dynamic_cast<Logical_and_expressionContext *>(context), predicateIndex);
    case 33: return logical_or_expressionSempred(dynamic_cast<Logical_or_expressionContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool originirParser::multiplicative_expressionSempred(Multiplicative_expressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 2);
    case 1: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool originirParser::addtive_expressionSempred(Addtive_expressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 2: return precpred(_ctx, 2);
    case 3: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool originirParser::relational_expressionSempred(Relational_expressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 4: return precpred(_ctx, 4);
    case 5: return precpred(_ctx, 3);
    case 6: return precpred(_ctx, 2);
    case 7: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool originirParser::equality_expressionSempred(Equality_expressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 8: return precpred(_ctx, 2);
    case 9: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool originirParser::logical_and_expressionSempred(Logical_and_expressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 10: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool originirParser::logical_or_expressionSempred(Logical_or_expressionContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 11: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> originirParser::_decisionToDFA;
atn::PredictionContextCache originirParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN originirParser::_atn;
std::vector<uint16_t> originirParser::_serializedATN;

std::vector<std::string> originirParser::_ruleNames = {
  "translationunit", "declaration", "qinit_declaration", "cinit_declaration", 
  "quantum_gate_declaration", "index", "c_KEY_declaration", "q_KEY_declaration", 
  "single_gate_without_parameter_declaration", "single_gate_with_one_parameter_declaration", 
  "single_gate_with_two_parameter_declaration", "single_gate_with_three_parameter_declaration", 
  "single_gate_with_four_parameter_declaration", "double_gate_without_parameter_declaration", 
  "double_gate_with_one_parameter_declaration", "double_gate_with_four_parameter_declaration", 
  "triple_gate_without_parameter_declaration", "single_gate_without_parameter_type", 
  "single_gate_with_one_parameter_type", "single_gate_with_two_parameter_type", 
  "single_gate_with_three_parameter_type", "single_gate_with_four_parameter_type", 
  "double_gate_without_parameter_type", "double_gate_with_one_parameter_type", 
  "double_gate_with_four_parameter_type", "triple_gate_without_parameter_type", 
  "primary_expression", "unary_expression", "multiplicative_expression", 
  "addtive_expression", "relational_expression", "equality_expression", 
  "logical_and_expression", "logical_or_expression", "assignment_expression", 
  "expression", "controlbit_list", "statement", "dagger_statement", "control_statement", 
  "qelse_statement_fragment", "qif_statement", "qwhile_statement", "measure_statement", 
  "reset_statement", "barrier_statement", "expression_statement", "constant"
};

std::vector<std::string> originirParser::_literalNames = {
  "", "'PI'", "'QINIT'", "'CREG'", "'q'", "'c'", "'BARRIER'", "'ECHO'", 
  "'H'", "'X'", "'NOT'", "'T'", "'S'", "'Y'", "'Z'", "'X1'", "'Y1'", "'Z1'", 
  "'I'", "'U2'", "'RPhi'", "'U3'", "'U4'", "'RX'", "'RY'", "'RZ'", "'U1'", 
  "'CNOT'", "'CZ'", "'CU'", "'ISWAP'", "'SQISWAP'", "'SWAP'", "'ISWAPTHETA'", 
  "'CR'", "'TOFFOLI'", "'DAGGER'", "'ENDDAGGER'", "'CONTROL'", "'ENDCONTROL'", 
  "'QIF'", "'ELSE'", "'ENDQIF'", "'QWHILE'", "'ENDQWHILE'", "'MEASURE'", 
  "'RESET'", "'='", "'>'", "'<'", "'!'", "'=='", "'<='", "'>='", "'!='", 
  "'&&'", "'||'", "'+'", "'-'", "'*'", "'/'", "','", "'('", "')'", "'['", 
  "']'"
};

std::vector<std::string> originirParser::_symbolicNames = {
  "", "PI", "QINIT_KEY", "CREG_KEY", "Q_KEY", "C_KEY", "BARRIER_KEY", "ECHO_GATE", 
  "H_GATE", "X_GATE", "NOT_GATE", "T_GATE", "S_GATE", "Y_GATE", "Z_GATE", 
  "X1_GATE", "Y1_GATE", "Z1_GATE", "I_GATE", "U2_GATE", "RPHI_GATE", "U3_GATE", 
  "U4_GATE", "RX_GATE", "RY_GATE", "RZ_GATE", "U1_GATE", "CNOT_GATE", "CZ_GATE", 
  "CU_GATE", "ISWAP_GATE", "SQISWAP_GATE", "SWAPZ1_GATE", "ISWAPTHETA_GATE", 
  "CR_GATE", "TOFFOLI_GATE", "DAGGER_KEY", "ENDDAGGER_KEY", "CONTROL_KEY", 
  "ENDCONTROL_KEY", "QIF_KEY", "ELSE_KEY", "ENDIF_KEY", "QWHILE_KEY", "ENDQWHILE_KEY", 
  "MEASURE_KEY", "RESET_KEY", "ASSIGN", "GT", "LT", "NOT", "EQ", "LEQ", 
  "GEQ", "NE", "AND", "OR", "PLUS", "MINUS", "MUL", "DIV", "COMMA", "LPAREN", 
  "RPAREN", "LBRACK", "RBRACK", "NEWLINE", "Identifier", "Integer_Literal", 
  "Double_Literal", "Digit_Sequence", "REALEXP", "WhiteSpace", "SingleLineComment"
};

dfa::Vocabulary originirParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> originirParser::_tokenNames;

originirParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x4b, 0x1cb, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x4, 0x1b, 0x9, 0x1b, 
    0x4, 0x1c, 0x9, 0x1c, 0x4, 0x1d, 0x9, 0x1d, 0x4, 0x1e, 0x9, 0x1e, 0x4, 
    0x1f, 0x9, 0x1f, 0x4, 0x20, 0x9, 0x20, 0x4, 0x21, 0x9, 0x21, 0x4, 0x22, 
    0x9, 0x22, 0x4, 0x23, 0x9, 0x23, 0x4, 0x24, 0x9, 0x24, 0x4, 0x25, 0x9, 
    0x25, 0x4, 0x26, 0x9, 0x26, 0x4, 0x27, 0x9, 0x27, 0x4, 0x28, 0x9, 0x28, 
    0x4, 0x29, 0x9, 0x29, 0x4, 0x2a, 0x9, 0x2a, 0x4, 0x2b, 0x9, 0x2b, 0x4, 
    0x2c, 0x9, 0x2c, 0x4, 0x2d, 0x9, 0x2d, 0x4, 0x2e, 0x9, 0x2e, 0x4, 0x2f, 
    0x9, 0x2f, 0x4, 0x30, 0x9, 0x30, 0x4, 0x31, 0x9, 0x31, 0x3, 0x2, 0x3, 
    0x2, 0x7, 0x2, 0x65, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x68, 0xb, 0x2, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x7e, 
    0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 
    0x3, 0xc, 0x3, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 
    0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x15, 
    0x3, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 
    0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 
    0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x5, 
    0x1c, 0xf1, 0xa, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 
    0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0xfa, 0xa, 0x1d, 0x3, 0x1e, 0x3, 
    0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 
    0x3, 0x1e, 0x7, 0x1e, 0x105, 0xa, 0x1e, 0xc, 0x1e, 0xe, 0x1e, 0x108, 
    0xb, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 
    0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x7, 0x1f, 0x113, 0xa, 0x1f, 
    0xc, 0x1f, 0xe, 0x1f, 0x116, 0xb, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
    0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 
    0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x7, 0x20, 
    0x127, 0xa, 0x20, 0xc, 0x20, 0xe, 0x20, 0x12a, 0xb, 0x20, 0x3, 0x21, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 
    0x21, 0x3, 0x21, 0x7, 0x21, 0x135, 0xa, 0x21, 0xc, 0x21, 0xe, 0x21, 
    0x138, 0xb, 0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x7, 0x22, 0x140, 0xa, 0x22, 0xc, 0x22, 0xe, 0x22, 0x143, 
    0xb, 0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 
    0x23, 0x7, 0x23, 0x14b, 0xa, 0x23, 0xc, 0x23, 0xe, 0x23, 0x14e, 0xb, 
    0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x5, 0x24, 
    0x155, 0xa, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 
    0x7, 0x26, 0x15c, 0xa, 0x26, 0xc, 0x26, 0xe, 0x26, 0x15f, 0xb, 0x26, 
    0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 
    0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 
    0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x5, 
    0x27, 0x174, 0xa, 0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x7, 0x28, 
    0x179, 0xa, 0x28, 0xc, 0x28, 0xe, 0x28, 0x17c, 0xb, 0x28, 0x3, 0x28, 
    0x3, 0x28, 0x3, 0x28, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x7, 
    0x29, 0x185, 0xa, 0x29, 0xc, 0x29, 0xe, 0x29, 0x188, 0xb, 0x29, 0x3, 
    0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x7, 0x2a, 
    0x190, 0xa, 0x2a, 0xc, 0x2a, 0xe, 0x2a, 0x193, 0xb, 0x2a, 0x3, 0x2b, 
    0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x7, 0x2b, 0x199, 0xa, 0x2b, 0xc, 0x2b, 
    0xe, 0x2b, 0x19c, 0xb, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 
    0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x7, 0x2b, 0x1a6, 0xa, 0x2b, 
    0xc, 0x2b, 0xe, 0x2b, 0x1a9, 0xb, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 
    0x5, 0x2b, 0x1ae, 0xa, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 
    0x7, 0x2c, 0x1b4, 0xa, 0x2c, 0xc, 0x2c, 0xe, 0x2c, 0x1b7, 0xb, 0x2c, 
    0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 
    0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 
    0x3, 0x2f, 0x3, 0x30, 0x3, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x2, 
    0x8, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x32, 0x2, 0x4, 0x6, 0x8, 0xa, 
    0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 
    0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 
    0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 
    0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x2, 0x8, 0x4, 0x2, 0x9, 0xb, 
    0xd, 0x14, 0x3, 0x2, 0x19, 0x1c, 0x3, 0x2, 0x15, 0x16, 0x4, 0x2, 0x1d, 
    0x1e, 0x20, 0x22, 0x3, 0x2, 0x23, 0x24, 0x4, 0x2, 0x3, 0x3, 0x46, 0x47, 
    0x2, 0x1c5, 0x2, 0x62, 0x3, 0x2, 0x2, 0x2, 0x4, 0x69, 0x3, 0x2, 0x2, 
    0x2, 0x6, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x8, 0x70, 0x3, 0x2, 0x2, 0x2, 0xa, 
    0x7d, 0x3, 0x2, 0x2, 0x2, 0xc, 0x7f, 0x3, 0x2, 0x2, 0x2, 0xe, 0x83, 
    0x3, 0x2, 0x2, 0x2, 0x10, 0x86, 0x3, 0x2, 0x2, 0x2, 0x12, 0x89, 0x3, 
    0x2, 0x2, 0x2, 0x14, 0x8c, 0x3, 0x2, 0x2, 0x2, 0x16, 0x93, 0x3, 0x2, 
    0x2, 0x2, 0x18, 0x9c, 0x3, 0x2, 0x2, 0x2, 0x1a, 0xa7, 0x3, 0x2, 0x2, 
    0x2, 0x1c, 0xb4, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xb9, 0x3, 0x2, 0x2, 0x2, 
    0x20, 0xc2, 0x3, 0x2, 0x2, 0x2, 0x22, 0xd1, 0x3, 0x2, 0x2, 0x2, 0x24, 
    0xd8, 0x3, 0x2, 0x2, 0x2, 0x26, 0xda, 0x3, 0x2, 0x2, 0x2, 0x28, 0xdc, 
    0x3, 0x2, 0x2, 0x2, 0x2a, 0xde, 0x3, 0x2, 0x2, 0x2, 0x2c, 0xe0, 0x3, 
    0x2, 0x2, 0x2, 0x2e, 0xe2, 0x3, 0x2, 0x2, 0x2, 0x30, 0xe4, 0x3, 0x2, 
    0x2, 0x2, 0x32, 0xe6, 0x3, 0x2, 0x2, 0x2, 0x34, 0xe8, 0x3, 0x2, 0x2, 
    0x2, 0x36, 0xf0, 0x3, 0x2, 0x2, 0x2, 0x38, 0xf9, 0x3, 0x2, 0x2, 0x2, 
    0x3a, 0xfb, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x109, 0x3, 0x2, 0x2, 0x2, 0x3e, 
    0x117, 0x3, 0x2, 0x2, 0x2, 0x40, 0x12b, 0x3, 0x2, 0x2, 0x2, 0x42, 0x139, 
    0x3, 0x2, 0x2, 0x2, 0x44, 0x144, 0x3, 0x2, 0x2, 0x2, 0x46, 0x154, 0x3, 
    0x2, 0x2, 0x2, 0x48, 0x156, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x158, 0x3, 0x2, 
    0x2, 0x2, 0x4c, 0x173, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x175, 0x3, 0x2, 0x2, 
    0x2, 0x50, 0x180, 0x3, 0x2, 0x2, 0x2, 0x52, 0x18c, 0x3, 0x2, 0x2, 0x2, 
    0x54, 0x1ad, 0x3, 0x2, 0x2, 0x2, 0x56, 0x1af, 0x3, 0x2, 0x2, 0x2, 0x58, 
    0x1bb, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x1c0, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x1c3, 
    0x3, 0x2, 0x2, 0x2, 0x5e, 0x1c6, 0x3, 0x2, 0x2, 0x2, 0x60, 0x1c8, 0x3, 
    0x2, 0x2, 0x2, 0x62, 0x66, 0x5, 0x4, 0x3, 0x2, 0x63, 0x65, 0x5, 0x4c, 
    0x27, 0x2, 0x64, 0x63, 0x3, 0x2, 0x2, 0x2, 0x65, 0x68, 0x3, 0x2, 0x2, 
    0x2, 0x66, 0x64, 0x3, 0x2, 0x2, 0x2, 0x66, 0x67, 0x3, 0x2, 0x2, 0x2, 
    0x67, 0x3, 0x3, 0x2, 0x2, 0x2, 0x68, 0x66, 0x3, 0x2, 0x2, 0x2, 0x69, 
    0x6a, 0x5, 0x6, 0x4, 0x2, 0x6a, 0x6b, 0x5, 0x8, 0x5, 0x2, 0x6b, 0x5, 
    0x3, 0x2, 0x2, 0x2, 0x6c, 0x6d, 0x7, 0x4, 0x2, 0x2, 0x6d, 0x6e, 0x7, 
    0x46, 0x2, 0x2, 0x6e, 0x6f, 0x7, 0x44, 0x2, 0x2, 0x6f, 0x7, 0x3, 0x2, 
    0x2, 0x2, 0x70, 0x71, 0x7, 0x5, 0x2, 0x2, 0x71, 0x72, 0x7, 0x46, 0x2, 
    0x2, 0x72, 0x73, 0x7, 0x44, 0x2, 0x2, 0x73, 0x9, 0x3, 0x2, 0x2, 0x2, 
    0x74, 0x7e, 0x5, 0x12, 0xa, 0x2, 0x75, 0x7e, 0x5, 0x14, 0xb, 0x2, 0x76, 
    0x7e, 0x5, 0x16, 0xc, 0x2, 0x77, 0x7e, 0x5, 0x18, 0xd, 0x2, 0x78, 0x7e, 
    0x5, 0x1a, 0xe, 0x2, 0x79, 0x7e, 0x5, 0x1c, 0xf, 0x2, 0x7a, 0x7e, 0x5, 
    0x1e, 0x10, 0x2, 0x7b, 0x7e, 0x5, 0x20, 0x11, 0x2, 0x7c, 0x7e, 0x5, 
    0x22, 0x12, 0x2, 0x7d, 0x74, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x75, 0x3, 0x2, 
    0x2, 0x2, 0x7d, 0x76, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x77, 0x3, 0x2, 0x2, 
    0x2, 0x7d, 0x78, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x79, 0x3, 0x2, 0x2, 0x2, 
    0x7d, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x7b, 0x3, 0x2, 0x2, 0x2, 0x7d, 
    0x7c, 0x3, 0x2, 0x2, 0x2, 0x7e, 0xb, 0x3, 0x2, 0x2, 0x2, 0x7f, 0x80, 
    0x7, 0x42, 0x2, 0x2, 0x80, 0x81, 0x5, 0x48, 0x25, 0x2, 0x81, 0x82, 0x7, 
    0x43, 0x2, 0x2, 0x82, 0xd, 0x3, 0x2, 0x2, 0x2, 0x83, 0x84, 0x7, 0x7, 
    0x2, 0x2, 0x84, 0x85, 0x5, 0xc, 0x7, 0x2, 0x85, 0xf, 0x3, 0x2, 0x2, 
    0x2, 0x86, 0x87, 0x7, 0x6, 0x2, 0x2, 0x87, 0x88, 0x5, 0xc, 0x7, 0x2, 
    0x88, 0x11, 0x3, 0x2, 0x2, 0x2, 0x89, 0x8a, 0x5, 0x24, 0x13, 0x2, 0x8a, 
    0x8b, 0x5, 0x10, 0x9, 0x2, 0x8b, 0x13, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x8d, 
    0x5, 0x26, 0x14, 0x2, 0x8d, 0x8e, 0x5, 0x10, 0x9, 0x2, 0x8e, 0x8f, 0x7, 
    0x3f, 0x2, 0x2, 0x8f, 0x90, 0x7, 0x40, 0x2, 0x2, 0x90, 0x91, 0x5, 0x48, 
    0x25, 0x2, 0x91, 0x92, 0x7, 0x41, 0x2, 0x2, 0x92, 0x15, 0x3, 0x2, 0x2, 
    0x2, 0x93, 0x94, 0x5, 0x28, 0x15, 0x2, 0x94, 0x95, 0x5, 0x10, 0x9, 0x2, 
    0x95, 0x96, 0x7, 0x3f, 0x2, 0x2, 0x96, 0x97, 0x7, 0x40, 0x2, 0x2, 0x97, 
    0x98, 0x5, 0x48, 0x25, 0x2, 0x98, 0x99, 0x7, 0x3f, 0x2, 0x2, 0x99, 0x9a, 
    0x5, 0x48, 0x25, 0x2, 0x9a, 0x9b, 0x7, 0x41, 0x2, 0x2, 0x9b, 0x17, 0x3, 
    0x2, 0x2, 0x2, 0x9c, 0x9d, 0x5, 0x2a, 0x16, 0x2, 0x9d, 0x9e, 0x5, 0x10, 
    0x9, 0x2, 0x9e, 0x9f, 0x7, 0x3f, 0x2, 0x2, 0x9f, 0xa0, 0x7, 0x40, 0x2, 
    0x2, 0xa0, 0xa1, 0x5, 0x48, 0x25, 0x2, 0xa1, 0xa2, 0x7, 0x3f, 0x2, 0x2, 
    0xa2, 0xa3, 0x5, 0x48, 0x25, 0x2, 0xa3, 0xa4, 0x7, 0x3f, 0x2, 0x2, 0xa4, 
    0xa5, 0x5, 0x48, 0x25, 0x2, 0xa5, 0xa6, 0x7, 0x41, 0x2, 0x2, 0xa6, 0x19, 
    0x3, 0x2, 0x2, 0x2, 0xa7, 0xa8, 0x5, 0x2c, 0x17, 0x2, 0xa8, 0xa9, 0x5, 
    0x10, 0x9, 0x2, 0xa9, 0xaa, 0x7, 0x3f, 0x2, 0x2, 0xaa, 0xab, 0x7, 0x40, 
    0x2, 0x2, 0xab, 0xac, 0x5, 0x48, 0x25, 0x2, 0xac, 0xad, 0x7, 0x3f, 0x2, 
    0x2, 0xad, 0xae, 0x5, 0x48, 0x25, 0x2, 0xae, 0xaf, 0x7, 0x3f, 0x2, 0x2, 
    0xaf, 0xb0, 0x5, 0x48, 0x25, 0x2, 0xb0, 0xb1, 0x7, 0x3f, 0x2, 0x2, 0xb1, 
    0xb2, 0x5, 0x48, 0x25, 0x2, 0xb2, 0xb3, 0x7, 0x41, 0x2, 0x2, 0xb3, 0x1b, 
    0x3, 0x2, 0x2, 0x2, 0xb4, 0xb5, 0x5, 0x2e, 0x18, 0x2, 0xb5, 0xb6, 0x5, 
    0x10, 0x9, 0x2, 0xb6, 0xb7, 0x7, 0x3f, 0x2, 0x2, 0xb7, 0xb8, 0x5, 0x10, 
    0x9, 0x2, 0xb8, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xb9, 0xba, 0x5, 0x30, 0x19, 
    0x2, 0xba, 0xbb, 0x5, 0x10, 0x9, 0x2, 0xbb, 0xbc, 0x7, 0x3f, 0x2, 0x2, 
    0xbc, 0xbd, 0x5, 0x10, 0x9, 0x2, 0xbd, 0xbe, 0x7, 0x3f, 0x2, 0x2, 0xbe, 
    0xbf, 0x7, 0x40, 0x2, 0x2, 0xbf, 0xc0, 0x5, 0x48, 0x25, 0x2, 0xc0, 0xc1, 
    0x7, 0x41, 0x2, 0x2, 0xc1, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xc2, 0xc3, 0x5, 
    0x32, 0x1a, 0x2, 0xc3, 0xc4, 0x5, 0x10, 0x9, 0x2, 0xc4, 0xc5, 0x7, 0x3f, 
    0x2, 0x2, 0xc5, 0xc6, 0x5, 0x10, 0x9, 0x2, 0xc6, 0xc7, 0x7, 0x3f, 0x2, 
    0x2, 0xc7, 0xc8, 0x7, 0x40, 0x2, 0x2, 0xc8, 0xc9, 0x5, 0x48, 0x25, 0x2, 
    0xc9, 0xca, 0x7, 0x3f, 0x2, 0x2, 0xca, 0xcb, 0x5, 0x48, 0x25, 0x2, 0xcb, 
    0xcc, 0x7, 0x3f, 0x2, 0x2, 0xcc, 0xcd, 0x5, 0x48, 0x25, 0x2, 0xcd, 0xce, 
    0x7, 0x3f, 0x2, 0x2, 0xce, 0xcf, 0x5, 0x48, 0x25, 0x2, 0xcf, 0xd0, 0x7, 
    0x41, 0x2, 0x2, 0xd0, 0x21, 0x3, 0x2, 0x2, 0x2, 0xd1, 0xd2, 0x5, 0x34, 
    0x1b, 0x2, 0xd2, 0xd3, 0x5, 0x10, 0x9, 0x2, 0xd3, 0xd4, 0x7, 0x3f, 0x2, 
    0x2, 0xd4, 0xd5, 0x5, 0x10, 0x9, 0x2, 0xd5, 0xd6, 0x7, 0x3f, 0x2, 0x2, 
    0xd6, 0xd7, 0x5, 0x10, 0x9, 0x2, 0xd7, 0x23, 0x3, 0x2, 0x2, 0x2, 0xd8, 
    0xd9, 0x9, 0x2, 0x2, 0x2, 0xd9, 0x25, 0x3, 0x2, 0x2, 0x2, 0xda, 0xdb, 
    0x9, 0x3, 0x2, 0x2, 0xdb, 0x27, 0x3, 0x2, 0x2, 0x2, 0xdc, 0xdd, 0x9, 
    0x4, 0x2, 0x2, 0xdd, 0x29, 0x3, 0x2, 0x2, 0x2, 0xde, 0xdf, 0x7, 0x17, 
    0x2, 0x2, 0xdf, 0x2b, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xe1, 0x7, 0x18, 0x2, 
    0x2, 0xe1, 0x2d, 0x3, 0x2, 0x2, 0x2, 0xe2, 0xe3, 0x9, 0x5, 0x2, 0x2, 
    0xe3, 0x2f, 0x3, 0x2, 0x2, 0x2, 0xe4, 0xe5, 0x9, 0x6, 0x2, 0x2, 0xe5, 
    0x31, 0x3, 0x2, 0x2, 0x2, 0xe6, 0xe7, 0x7, 0x1f, 0x2, 0x2, 0xe7, 0x33, 
    0x3, 0x2, 0x2, 0x2, 0xe8, 0xe9, 0x7, 0x25, 0x2, 0x2, 0xe9, 0x35, 0x3, 
    0x2, 0x2, 0x2, 0xea, 0xf1, 0x5, 0xe, 0x8, 0x2, 0xeb, 0xf1, 0x5, 0x60, 
    0x31, 0x2, 0xec, 0xed, 0x7, 0x40, 0x2, 0x2, 0xed, 0xee, 0x5, 0x48, 0x25, 
    0x2, 0xee, 0xef, 0x7, 0x40, 0x2, 0x2, 0xef, 0xf1, 0x3, 0x2, 0x2, 0x2, 
    0xf0, 0xea, 0x3, 0x2, 0x2, 0x2, 0xf0, 0xeb, 0x3, 0x2, 0x2, 0x2, 0xf0, 
    0xec, 0x3, 0x2, 0x2, 0x2, 0xf1, 0x37, 0x3, 0x2, 0x2, 0x2, 0xf2, 0xfa, 
    0x5, 0x36, 0x1c, 0x2, 0xf3, 0xf4, 0x7, 0x3b, 0x2, 0x2, 0xf4, 0xfa, 0x5, 
    0x36, 0x1c, 0x2, 0xf5, 0xf6, 0x7, 0x3c, 0x2, 0x2, 0xf6, 0xfa, 0x5, 0x36, 
    0x1c, 0x2, 0xf7, 0xf8, 0x7, 0x34, 0x2, 0x2, 0xf8, 0xfa, 0x5, 0x36, 0x1c, 
    0x2, 0xf9, 0xf2, 0x3, 0x2, 0x2, 0x2, 0xf9, 0xf3, 0x3, 0x2, 0x2, 0x2, 
    0xf9, 0xf5, 0x3, 0x2, 0x2, 0x2, 0xf9, 0xf7, 0x3, 0x2, 0x2, 0x2, 0xfa, 
    0x39, 0x3, 0x2, 0x2, 0x2, 0xfb, 0xfc, 0x8, 0x1e, 0x1, 0x2, 0xfc, 0xfd, 
    0x5, 0x38, 0x1d, 0x2, 0xfd, 0x106, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xff, 0xc, 
    0x4, 0x2, 0x2, 0xff, 0x100, 0x7, 0x3d, 0x2, 0x2, 0x100, 0x105, 0x5, 
    0x38, 0x1d, 0x2, 0x101, 0x102, 0xc, 0x3, 0x2, 0x2, 0x102, 0x103, 0x7, 
    0x3e, 0x2, 0x2, 0x103, 0x105, 0x5, 0x38, 0x1d, 0x2, 0x104, 0xfe, 0x3, 
    0x2, 0x2, 0x2, 0x104, 0x101, 0x3, 0x2, 0x2, 0x2, 0x105, 0x108, 0x3, 
    0x2, 0x2, 0x2, 0x106, 0x104, 0x3, 0x2, 0x2, 0x2, 0x106, 0x107, 0x3, 
    0x2, 0x2, 0x2, 0x107, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x108, 0x106, 0x3, 0x2, 
    0x2, 0x2, 0x109, 0x10a, 0x8, 0x1f, 0x1, 0x2, 0x10a, 0x10b, 0x5, 0x3a, 
    0x1e, 0x2, 0x10b, 0x114, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10d, 0xc, 0x4, 
    0x2, 0x2, 0x10d, 0x10e, 0x7, 0x3b, 0x2, 0x2, 0x10e, 0x113, 0x5, 0x3a, 
    0x1e, 0x2, 0x10f, 0x110, 0xc, 0x3, 0x2, 0x2, 0x110, 0x111, 0x7, 0x3c, 
    0x2, 0x2, 0x111, 0x113, 0x5, 0x3a, 0x1e, 0x2, 0x112, 0x10c, 0x3, 0x2, 
    0x2, 0x2, 0x112, 0x10f, 0x3, 0x2, 0x2, 0x2, 0x113, 0x116, 0x3, 0x2, 
    0x2, 0x2, 0x114, 0x112, 0x3, 0x2, 0x2, 0x2, 0x114, 0x115, 0x3, 0x2, 
    0x2, 0x2, 0x115, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x116, 0x114, 0x3, 0x2, 0x2, 
    0x2, 0x117, 0x118, 0x8, 0x20, 0x1, 0x2, 0x118, 0x119, 0x5, 0x3c, 0x1f, 
    0x2, 0x119, 0x128, 0x3, 0x2, 0x2, 0x2, 0x11a, 0x11b, 0xc, 0x6, 0x2, 
    0x2, 0x11b, 0x11c, 0x7, 0x33, 0x2, 0x2, 0x11c, 0x127, 0x5, 0x3c, 0x1f, 
    0x2, 0x11d, 0x11e, 0xc, 0x5, 0x2, 0x2, 0x11e, 0x11f, 0x7, 0x32, 0x2, 
    0x2, 0x11f, 0x127, 0x5, 0x3c, 0x1f, 0x2, 0x120, 0x121, 0xc, 0x4, 0x2, 
    0x2, 0x121, 0x122, 0x7, 0x36, 0x2, 0x2, 0x122, 0x127, 0x5, 0x3c, 0x1f, 
    0x2, 0x123, 0x124, 0xc, 0x3, 0x2, 0x2, 0x124, 0x125, 0x7, 0x37, 0x2, 
    0x2, 0x125, 0x127, 0x5, 0x3c, 0x1f, 0x2, 0x126, 0x11a, 0x3, 0x2, 0x2, 
    0x2, 0x126, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x126, 0x120, 0x3, 0x2, 0x2, 
    0x2, 0x126, 0x123, 0x3, 0x2, 0x2, 0x2, 0x127, 0x12a, 0x3, 0x2, 0x2, 
    0x2, 0x128, 0x126, 0x3, 0x2, 0x2, 0x2, 0x128, 0x129, 0x3, 0x2, 0x2, 
    0x2, 0x129, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x12a, 0x128, 0x3, 0x2, 0x2, 0x2, 
    0x12b, 0x12c, 0x8, 0x21, 0x1, 0x2, 0x12c, 0x12d, 0x5, 0x3e, 0x20, 0x2, 
    0x12d, 0x136, 0x3, 0x2, 0x2, 0x2, 0x12e, 0x12f, 0xc, 0x4, 0x2, 0x2, 
    0x12f, 0x130, 0x7, 0x35, 0x2, 0x2, 0x130, 0x135, 0x5, 0x3e, 0x20, 0x2, 
    0x131, 0x132, 0xc, 0x3, 0x2, 0x2, 0x132, 0x133, 0x7, 0x38, 0x2, 0x2, 
    0x133, 0x135, 0x5, 0x3e, 0x20, 0x2, 0x134, 0x12e, 0x3, 0x2, 0x2, 0x2, 
    0x134, 0x131, 0x3, 0x2, 0x2, 0x2, 0x135, 0x138, 0x3, 0x2, 0x2, 0x2, 
    0x136, 0x134, 0x3, 0x2, 0x2, 0x2, 0x136, 0x137, 0x3, 0x2, 0x2, 0x2, 
    0x137, 0x41, 0x3, 0x2, 0x2, 0x2, 0x138, 0x136, 0x3, 0x2, 0x2, 0x2, 0x139, 
    0x13a, 0x8, 0x22, 0x1, 0x2, 0x13a, 0x13b, 0x5, 0x40, 0x21, 0x2, 0x13b, 
    0x141, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x13d, 0xc, 0x3, 0x2, 0x2, 0x13d, 
    0x13e, 0x7, 0x39, 0x2, 0x2, 0x13e, 0x140, 0x5, 0x40, 0x21, 0x2, 0x13f, 
    0x13c, 0x3, 0x2, 0x2, 0x2, 0x140, 0x143, 0x3, 0x2, 0x2, 0x2, 0x141, 
    0x13f, 0x3, 0x2, 0x2, 0x2, 0x141, 0x142, 0x3, 0x2, 0x2, 0x2, 0x142, 
    0x43, 0x3, 0x2, 0x2, 0x2, 0x143, 0x141, 0x3, 0x2, 0x2, 0x2, 0x144, 0x145, 
    0x8, 0x23, 0x1, 0x2, 0x145, 0x146, 0x5, 0x42, 0x22, 0x2, 0x146, 0x14c, 
    0x3, 0x2, 0x2, 0x2, 0x147, 0x148, 0xc, 0x3, 0x2, 0x2, 0x148, 0x149, 
    0x7, 0x3a, 0x2, 0x2, 0x149, 0x14b, 0x5, 0x42, 0x22, 0x2, 0x14a, 0x147, 
    0x3, 0x2, 0x2, 0x2, 0x14b, 0x14e, 0x3, 0x2, 0x2, 0x2, 0x14c, 0x14a, 
    0x3, 0x2, 0x2, 0x2, 0x14c, 0x14d, 0x3, 0x2, 0x2, 0x2, 0x14d, 0x45, 0x3, 
    0x2, 0x2, 0x2, 0x14e, 0x14c, 0x3, 0x2, 0x2, 0x2, 0x14f, 0x155, 0x5, 
    0x44, 0x23, 0x2, 0x150, 0x151, 0x5, 0xe, 0x8, 0x2, 0x151, 0x152, 0x7, 
    0x31, 0x2, 0x2, 0x152, 0x153, 0x5, 0x44, 0x23, 0x2, 0x153, 0x155, 0x3, 
    0x2, 0x2, 0x2, 0x154, 0x14f, 0x3, 0x2, 0x2, 0x2, 0x154, 0x150, 0x3, 
    0x2, 0x2, 0x2, 0x155, 0x47, 0x3, 0x2, 0x2, 0x2, 0x156, 0x157, 0x5, 0x46, 
    0x24, 0x2, 0x157, 0x49, 0x3, 0x2, 0x2, 0x2, 0x158, 0x15d, 0x5, 0x10, 
    0x9, 0x2, 0x159, 0x15a, 0x7, 0x3f, 0x2, 0x2, 0x15a, 0x15c, 0x5, 0x10, 
    0x9, 0x2, 0x15b, 0x159, 0x3, 0x2, 0x2, 0x2, 0x15c, 0x15f, 0x3, 0x2, 
    0x2, 0x2, 0x15d, 0x15b, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15e, 0x3, 0x2, 
    0x2, 0x2, 0x15e, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x15f, 0x15d, 0x3, 0x2, 0x2, 
    0x2, 0x160, 0x161, 0x5, 0xa, 0x6, 0x2, 0x161, 0x162, 0x7, 0x44, 0x2, 
    0x2, 0x162, 0x174, 0x3, 0x2, 0x2, 0x2, 0x163, 0x174, 0x5, 0x50, 0x29, 
    0x2, 0x164, 0x174, 0x5, 0x54, 0x2b, 0x2, 0x165, 0x174, 0x5, 0x56, 0x2c, 
    0x2, 0x166, 0x174, 0x5, 0x4e, 0x28, 0x2, 0x167, 0x168, 0x5, 0x58, 0x2d, 
    0x2, 0x168, 0x169, 0x7, 0x44, 0x2, 0x2, 0x169, 0x174, 0x3, 0x2, 0x2, 
    0x2, 0x16a, 0x16b, 0x5, 0x5a, 0x2e, 0x2, 0x16b, 0x16c, 0x7, 0x44, 0x2, 
    0x2, 0x16c, 0x174, 0x3, 0x2, 0x2, 0x2, 0x16d, 0x16e, 0x5, 0x5e, 0x30, 
    0x2, 0x16e, 0x16f, 0x7, 0x44, 0x2, 0x2, 0x16f, 0x174, 0x3, 0x2, 0x2, 
    0x2, 0x170, 0x171, 0x5, 0x5c, 0x2f, 0x2, 0x171, 0x172, 0x7, 0x44, 0x2, 
    0x2, 0x172, 0x174, 0x3, 0x2, 0x2, 0x2, 0x173, 0x160, 0x3, 0x2, 0x2, 
    0x2, 0x173, 0x163, 0x3, 0x2, 0x2, 0x2, 0x173, 0x164, 0x3, 0x2, 0x2, 
    0x2, 0x173, 0x165, 0x3, 0x2, 0x2, 0x2, 0x173, 0x166, 0x3, 0x2, 0x2, 
    0x2, 0x173, 0x167, 0x3, 0x2, 0x2, 0x2, 0x173, 0x16a, 0x3, 0x2, 0x2, 
    0x2, 0x173, 0x16d, 0x3, 0x2, 0x2, 0x2, 0x173, 0x170, 0x3, 0x2, 0x2, 
    0x2, 0x174, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x175, 0x176, 0x7, 0x26, 0x2, 
    0x2, 0x176, 0x17a, 0x7, 0x44, 0x2, 0x2, 0x177, 0x179, 0x5, 0x4c, 0x27, 
    0x2, 0x178, 0x177, 0x3, 0x2, 0x2, 0x2, 0x179, 0x17c, 0x3, 0x2, 0x2, 
    0x2, 0x17a, 0x178, 0x3, 0x2, 0x2, 0x2, 0x17a, 0x17b, 0x3, 0x2, 0x2, 
    0x2, 0x17b, 0x17d, 0x3, 0x2, 0x2, 0x2, 0x17c, 0x17a, 0x3, 0x2, 0x2, 
    0x2, 0x17d, 0x17e, 0x7, 0x27, 0x2, 0x2, 0x17e, 0x17f, 0x7, 0x44, 0x2, 
    0x2, 0x17f, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x180, 0x181, 0x7, 0x28, 0x2, 
    0x2, 0x181, 0x182, 0x5, 0x4a, 0x26, 0x2, 0x182, 0x186, 0x7, 0x44, 0x2, 
    0x2, 0x183, 0x185, 0x5, 0x4c, 0x27, 0x2, 0x184, 0x183, 0x3, 0x2, 0x2, 
    0x2, 0x185, 0x188, 0x3, 0x2, 0x2, 0x2, 0x186, 0x184, 0x3, 0x2, 0x2, 
    0x2, 0x186, 0x187, 0x3, 0x2, 0x2, 0x2, 0x187, 0x189, 0x3, 0x2, 0x2, 
    0x2, 0x188, 0x186, 0x3, 0x2, 0x2, 0x2, 0x189, 0x18a, 0x7, 0x29, 0x2, 
    0x2, 0x18a, 0x18b, 0x7, 0x44, 0x2, 0x2, 0x18b, 0x51, 0x3, 0x2, 0x2, 
    0x2, 0x18c, 0x18d, 0x7, 0x2b, 0x2, 0x2, 0x18d, 0x191, 0x7, 0x44, 0x2, 
    0x2, 0x18e, 0x190, 0x5, 0x4c, 0x27, 0x2, 0x18f, 0x18e, 0x3, 0x2, 0x2, 
    0x2, 0x190, 0x193, 0x3, 0x2, 0x2, 0x2, 0x191, 0x18f, 0x3, 0x2, 0x2, 
    0x2, 0x191, 0x192, 0x3, 0x2, 0x2, 0x2, 0x192, 0x53, 0x3, 0x2, 0x2, 0x2, 
    0x193, 0x191, 0x3, 0x2, 0x2, 0x2, 0x194, 0x195, 0x7, 0x2a, 0x2, 0x2, 
    0x195, 0x196, 0x5, 0x48, 0x25, 0x2, 0x196, 0x19a, 0x7, 0x44, 0x2, 0x2, 
    0x197, 0x199, 0x5, 0x4c, 0x27, 0x2, 0x198, 0x197, 0x3, 0x2, 0x2, 0x2, 
    0x199, 0x19c, 0x3, 0x2, 0x2, 0x2, 0x19a, 0x198, 0x3, 0x2, 0x2, 0x2, 
    0x19a, 0x19b, 0x3, 0x2, 0x2, 0x2, 0x19b, 0x19d, 0x3, 0x2, 0x2, 0x2, 
    0x19c, 0x19a, 0x3, 0x2, 0x2, 0x2, 0x19d, 0x19e, 0x5, 0x52, 0x2a, 0x2, 
    0x19e, 0x19f, 0x7, 0x2c, 0x2, 0x2, 0x19f, 0x1a0, 0x7, 0x44, 0x2, 0x2, 
    0x1a0, 0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1a1, 0x1a2, 0x7, 0x2a, 0x2, 0x2, 
    0x1a2, 0x1a3, 0x5, 0x48, 0x25, 0x2, 0x1a3, 0x1a7, 0x7, 0x44, 0x2, 0x2, 
    0x1a4, 0x1a6, 0x5, 0x4c, 0x27, 0x2, 0x1a5, 0x1a4, 0x3, 0x2, 0x2, 0x2, 
    0x1a6, 0x1a9, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x1a5, 0x3, 0x2, 0x2, 0x2, 
    0x1a7, 0x1a8, 0x3, 0x2, 0x2, 0x2, 0x1a8, 0x1aa, 0x3, 0x2, 0x2, 0x2, 
    0x1a9, 0x1a7, 0x3, 0x2, 0x2, 0x2, 0x1aa, 0x1ab, 0x7, 0x2c, 0x2, 0x2, 
    0x1ab, 0x1ac, 0x7, 0x44, 0x2, 0x2, 0x1ac, 0x1ae, 0x3, 0x2, 0x2, 0x2, 
    0x1ad, 0x194, 0x3, 0x2, 0x2, 0x2, 0x1ad, 0x1a1, 0x3, 0x2, 0x2, 0x2, 
    0x1ae, 0x55, 0x3, 0x2, 0x2, 0x2, 0x1af, 0x1b0, 0x7, 0x2d, 0x2, 0x2, 
    0x1b0, 0x1b1, 0x5, 0x48, 0x25, 0x2, 0x1b1, 0x1b5, 0x7, 0x44, 0x2, 0x2, 
    0x1b2, 0x1b4, 0x5, 0x4c, 0x27, 0x2, 0x1b3, 0x1b2, 0x3, 0x2, 0x2, 0x2, 
    0x1b4, 0x1b7, 0x3, 0x2, 0x2, 0x2, 0x1b5, 0x1b3, 0x3, 0x2, 0x2, 0x2, 
    0x1b5, 0x1b6, 0x3, 0x2, 0x2, 0x2, 0x1b6, 0x1b8, 0x3, 0x2, 0x2, 0x2, 
    0x1b7, 0x1b5, 0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1b9, 0x7, 0x2e, 0x2, 0x2, 
    0x1b9, 0x1ba, 0x7, 0x44, 0x2, 0x2, 0x1ba, 0x57, 0x3, 0x2, 0x2, 0x2, 
    0x1bb, 0x1bc, 0x7, 0x2f, 0x2, 0x2, 0x1bc, 0x1bd, 0x5, 0x10, 0x9, 0x2, 
    0x1bd, 0x1be, 0x7, 0x3f, 0x2, 0x2, 0x1be, 0x1bf, 0x5, 0xe, 0x8, 0x2, 
    0x1bf, 0x59, 0x3, 0x2, 0x2, 0x2, 0x1c0, 0x1c1, 0x7, 0x30, 0x2, 0x2, 
    0x1c1, 0x1c2, 0x5, 0x10, 0x9, 0x2, 0x1c2, 0x5b, 0x3, 0x2, 0x2, 0x2, 
    0x1c3, 0x1c4, 0x7, 0x8, 0x2, 0x2, 0x1c4, 0x1c5, 0x5, 0x4a, 0x26, 0x2, 
    0x1c5, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x1c7, 0x5, 0x48, 0x25, 0x2, 
    0x1c7, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x1c8, 0x1c9, 0x9, 0x7, 0x2, 0x2, 0x1c9, 
    0x61, 0x3, 0x2, 0x2, 0x2, 0x1a, 0x66, 0x7d, 0xf0, 0xf9, 0x104, 0x106, 
    0x112, 0x114, 0x126, 0x128, 0x134, 0x136, 0x141, 0x14c, 0x154, 0x15d, 
    0x173, 0x17a, 0x186, 0x191, 0x19a, 0x1a7, 0x1ad, 0x1b5, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

originirParser::Initializer originirParser::_init;
