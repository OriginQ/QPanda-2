
// Generated from originir.g4 by ANTLR 4.8
#pragma once



#include "Core/Utilities/Compiler/OriginIRCompiler/originirParserListener.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirParserVisitor.h"

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

std::vector<tree::TerminalNode *> originirParser::TranslationunitContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::TranslationunitContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

std::vector<originirParser::Gate_func_statementContext *> originirParser::TranslationunitContext::gate_func_statement() {
  return getRuleContexts<originirParser::Gate_func_statementContext>();
}

originirParser::Gate_func_statementContext* originirParser::TranslationunitContext::gate_func_statement(size_t i) {
  return getRuleContext<originirParser::Gate_func_statementContext>(i);
}

std::vector<originirParser::DeclarationContext *> originirParser::TranslationunitContext::declaration() {
  return getRuleContexts<originirParser::DeclarationContext>();
}

originirParser::DeclarationContext* originirParser::TranslationunitContext::declaration(size_t i) {
  return getRuleContext<originirParser::DeclarationContext>(i);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTranslationunit(this);
}

void originirParser::TranslationunitContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTranslationunit(this);
}


antlrcpp::Any originirParser::TranslationunitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(121);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == originirParser::NEWLINE) {
      setState(118);
      match(originirParser::NEWLINE);
      setState(123);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(127);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(124);
        gate_func_statement(); 
      }
      setState(129);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    }
    setState(133);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == originirParser::QINIT_KEY) {
      setState(130);
      declaration();
      setState(135);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(137); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(136);
      statement();
      setState(139); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::QGATE_KEY)
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
      | (1ULL << originirParser::P_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (originirParser::PLUS - 64))
      | (1ULL << (originirParser::MINUS - 64))
      | (1ULL << (originirParser::LPAREN - 64))
      | (1ULL << (originirParser::Identifier - 64))
      | (1ULL << (originirParser::Integer_Literal - 64))
      | (1ULL << (originirParser::Double_Literal - 64)))) != 0));
   
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclaration(this);
}

void originirParser::DeclarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclaration(this);
}


antlrcpp::Any originirParser::DeclarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(141);
    qinit_declaration();
    setState(142);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQinit_declaration(this);
}

void originirParser::Qinit_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQinit_declaration(this);
}


antlrcpp::Any originirParser::Qinit_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(144);
    match(originirParser::QINIT_KEY);
    setState(145);
    match(originirParser::Integer_Literal);
    setState(146);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCinit_declaration(this);
}

void originirParser::Cinit_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCinit_declaration(this);
}


antlrcpp::Any originirParser::Cinit_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(148);
    match(originirParser::CREG_KEY);
    setState(149);
    match(originirParser::Integer_Literal);
    setState(150);
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

originirParser::Define_gate_declarationContext* originirParser::Quantum_gate_declarationContext::define_gate_declaration() {
  return getRuleContext<originirParser::Define_gate_declarationContext>(0);
}


size_t originirParser::Quantum_gate_declarationContext::getRuleIndex() const {
  return originirParser::RuleQuantum_gate_declaration;
}

void originirParser::Quantum_gate_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQuantum_gate_declaration(this);
}

void originirParser::Quantum_gate_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQuantum_gate_declaration(this);
}


antlrcpp::Any originirParser::Quantum_gate_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(162);
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
        setState(152);
        single_gate_without_parameter_declaration();
        break;
      }

      case originirParser::RX_GATE:
      case originirParser::RY_GATE:
      case originirParser::RZ_GATE:
      case originirParser::U1_GATE:
      case originirParser::P_GATE: {
        enterOuterAlt(_localctx, 2);
        setState(153);
        single_gate_with_one_parameter_declaration();
        break;
      }

      case originirParser::U2_GATE:
      case originirParser::RPHI_GATE: {
        enterOuterAlt(_localctx, 3);
        setState(154);
        single_gate_with_two_parameter_declaration();
        break;
      }

      case originirParser::U3_GATE: {
        enterOuterAlt(_localctx, 4);
        setState(155);
        single_gate_with_three_parameter_declaration();
        break;
      }

      case originirParser::U4_GATE: {
        enterOuterAlt(_localctx, 5);
        setState(156);
        single_gate_with_four_parameter_declaration();
        break;
      }

      case originirParser::CNOT_GATE:
      case originirParser::CZ_GATE:
      case originirParser::ISWAP_GATE:
      case originirParser::SQISWAP_GATE:
      case originirParser::SWAPZ1_GATE: {
        enterOuterAlt(_localctx, 6);
        setState(157);
        double_gate_without_parameter_declaration();
        break;
      }

      case originirParser::ISWAPTHETA_GATE:
      case originirParser::CR_GATE:
      case originirParser::RXX_GATE:
      case originirParser::RYY_GATE:
      case originirParser::RZZ_GATE:
      case originirParser::RZX_GATE: {
        enterOuterAlt(_localctx, 7);
        setState(158);
        double_gate_with_one_parameter_declaration();
        break;
      }

      case originirParser::CU_GATE: {
        enterOuterAlt(_localctx, 8);
        setState(159);
        double_gate_with_four_parameter_declaration();
        break;
      }

      case originirParser::TOFFOLI_GATE: {
        enterOuterAlt(_localctx, 9);
        setState(160);
        triple_gate_without_parameter_declaration();
        break;
      }

      case originirParser::Identifier: {
        enterOuterAlt(_localctx, 10);
        setState(161);
        define_gate_declaration();
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIndex(this);
}

void originirParser::IndexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIndex(this);
}


antlrcpp::Any originirParser::IndexContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(164);
    match(originirParser::LBRACK);
    setState(165);
    expression();
    setState(166);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterC_KEY_declaration(this);
}

void originirParser::C_KEY_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitC_KEY_declaration(this);
}


antlrcpp::Any originirParser::C_KEY_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(168);
    match(originirParser::C_KEY);
    setState(169);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQ_KEY_declaration(this);
}

void originirParser::Q_KEY_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQ_KEY_declaration(this);
}


antlrcpp::Any originirParser::Q_KEY_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(171);
    match(originirParser::Q_KEY);
    setState(172);
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

tree::TerminalNode* originirParser::Single_gate_without_parameter_declarationContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}


size_t originirParser::Single_gate_without_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_without_parameter_declaration;
}

void originirParser::Single_gate_without_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_without_parameter_declaration(this);
}

void originirParser::Single_gate_without_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_without_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_without_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(180);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(174);
      single_gate_without_parameter_type();
      setState(175);
      q_KEY_declaration();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(177);
      single_gate_without_parameter_type();
      setState(178);
      match(originirParser::Q_KEY);
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

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_declarationContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}


size_t originirParser::Single_gate_with_one_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_one_parameter_declaration;
}

void originirParser::Single_gate_with_one_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_one_parameter_declaration(this);
}

void originirParser::Single_gate_with_one_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_one_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_one_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(196);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(182);
      single_gate_with_one_parameter_type();
      setState(183);
      q_KEY_declaration();
      setState(184);
      match(originirParser::COMMA);
      setState(185);
      match(originirParser::LPAREN);
      setState(186);
      expression();
      setState(187);
      match(originirParser::RPAREN);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(189);
      single_gate_with_one_parameter_type();
      setState(190);
      match(originirParser::Q_KEY);
      setState(191);
      match(originirParser::COMMA);
      setState(192);
      match(originirParser::LPAREN);
      setState(193);
      expression();
      setState(194);
      match(originirParser::RPAREN);
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

tree::TerminalNode* originirParser::Single_gate_with_two_parameter_declarationContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}


size_t originirParser::Single_gate_with_two_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_two_parameter_declaration;
}

void originirParser::Single_gate_with_two_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_two_parameter_declaration(this);
}

void originirParser::Single_gate_with_two_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_two_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_two_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(216);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(198);
      single_gate_with_two_parameter_type();
      setState(199);
      q_KEY_declaration();
      setState(200);
      match(originirParser::COMMA);
      setState(201);
      match(originirParser::LPAREN);
      setState(202);
      expression();
      setState(203);
      match(originirParser::COMMA);
      setState(204);
      expression();
      setState(205);
      match(originirParser::RPAREN);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(207);
      single_gate_with_two_parameter_type();
      setState(208);
      match(originirParser::Q_KEY);
      setState(209);
      match(originirParser::COMMA);
      setState(210);
      match(originirParser::LPAREN);
      setState(211);
      expression();
      setState(212);
      match(originirParser::COMMA);
      setState(213);
      expression();
      setState(214);
      match(originirParser::RPAREN);
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

tree::TerminalNode* originirParser::Single_gate_with_three_parameter_declarationContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}


size_t originirParser::Single_gate_with_three_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_three_parameter_declaration;
}

void originirParser::Single_gate_with_three_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_three_parameter_declaration(this);
}

void originirParser::Single_gate_with_three_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_three_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_three_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(240);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(218);
      single_gate_with_three_parameter_type();
      setState(219);
      q_KEY_declaration();
      setState(220);
      match(originirParser::COMMA);
      setState(221);
      match(originirParser::LPAREN);
      setState(222);
      expression();
      setState(223);
      match(originirParser::COMMA);
      setState(224);
      expression();
      setState(225);
      match(originirParser::COMMA);
      setState(226);
      expression();
      setState(227);
      match(originirParser::RPAREN);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(229);
      single_gate_with_three_parameter_type();
      setState(230);
      match(originirParser::Q_KEY);
      setState(231);
      match(originirParser::COMMA);
      setState(232);
      match(originirParser::LPAREN);
      setState(233);
      expression();
      setState(234);
      match(originirParser::COMMA);
      setState(235);
      expression();
      setState(236);
      match(originirParser::COMMA);
      setState(237);
      expression();
      setState(238);
      match(originirParser::RPAREN);
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

tree::TerminalNode* originirParser::Single_gate_with_four_parameter_declarationContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}


size_t originirParser::Single_gate_with_four_parameter_declarationContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_four_parameter_declaration;
}

void originirParser::Single_gate_with_four_parameter_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_four_parameter_declaration(this);
}

void originirParser::Single_gate_with_four_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_four_parameter_declaration(this);
}


antlrcpp::Any originirParser::Single_gate_with_four_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(268);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(242);
      single_gate_with_four_parameter_type();
      setState(243);
      q_KEY_declaration();
      setState(244);
      match(originirParser::COMMA);
      setState(245);
      match(originirParser::LPAREN);
      setState(246);
      expression();
      setState(247);
      match(originirParser::COMMA);
      setState(248);
      expression();
      setState(249);
      match(originirParser::COMMA);
      setState(250);
      expression();
      setState(251);
      match(originirParser::COMMA);
      setState(252);
      expression();
      setState(253);
      match(originirParser::RPAREN);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(255);
      single_gate_with_four_parameter_type();
      setState(256);
      match(originirParser::Q_KEY);
      setState(257);
      match(originirParser::COMMA);
      setState(258);
      match(originirParser::LPAREN);
      setState(259);
      expression();
      setState(260);
      match(originirParser::COMMA);
      setState(261);
      expression();
      setState(262);
      match(originirParser::COMMA);
      setState(263);
      expression();
      setState(264);
      match(originirParser::COMMA);
      setState(265);
      expression();
      setState(266);
      match(originirParser::RPAREN);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_without_parameter_declaration(this);
}

void originirParser::Double_gate_without_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_without_parameter_declaration(this);
}


antlrcpp::Any originirParser::Double_gate_without_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(270);
    double_gate_without_parameter_type();
    setState(271);
    q_KEY_declaration();
    setState(272);
    match(originirParser::COMMA);
    setState(273);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_one_parameter_declaration(this);
}

void originirParser::Double_gate_with_one_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_one_parameter_declaration(this);
}


antlrcpp::Any originirParser::Double_gate_with_one_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(275);
    double_gate_with_one_parameter_type();
    setState(276);
    q_KEY_declaration();
    setState(277);
    match(originirParser::COMMA);
    setState(278);
    q_KEY_declaration();
    setState(279);
    match(originirParser::COMMA);
    setState(280);
    match(originirParser::LPAREN);
    setState(281);
    expression();
    setState(282);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_four_parameter_declaration(this);
}

void originirParser::Double_gate_with_four_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_four_parameter_declaration(this);
}


antlrcpp::Any originirParser::Double_gate_with_four_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(284);
    double_gate_with_four_parameter_type();
    setState(285);
    q_KEY_declaration();
    setState(286);
    match(originirParser::COMMA);
    setState(287);
    q_KEY_declaration();
    setState(288);
    match(originirParser::COMMA);
    setState(289);
    match(originirParser::LPAREN);
    setState(290);
    expression();
    setState(291);
    match(originirParser::COMMA);
    setState(292);
    expression();
    setState(293);
    match(originirParser::COMMA);
    setState(294);
    expression();
    setState(295);
    match(originirParser::COMMA);
    setState(296);
    expression();
    setState(297);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTriple_gate_without_parameter_declaration(this);
}

void originirParser::Triple_gate_without_parameter_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTriple_gate_without_parameter_declaration(this);
}


antlrcpp::Any originirParser::Triple_gate_without_parameter_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
    setState(299);
    triple_gate_without_parameter_type();
    setState(300);
    q_KEY_declaration();
    setState(301);
    match(originirParser::COMMA);
    setState(302);
    q_KEY_declaration();
    setState(303);
    match(originirParser::COMMA);
    setState(304);
    q_KEY_declaration();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Define_gate_declarationContext ------------------------------------------------------------------

originirParser::Define_gate_declarationContext::Define_gate_declarationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::IdContext* originirParser::Define_gate_declarationContext::id() {
  return getRuleContext<originirParser::IdContext>(0);
}

std::vector<originirParser::Q_KEY_declarationContext *> originirParser::Define_gate_declarationContext::q_KEY_declaration() {
  return getRuleContexts<originirParser::Q_KEY_declarationContext>();
}

originirParser::Q_KEY_declarationContext* originirParser::Define_gate_declarationContext::q_KEY_declaration(size_t i) {
  return getRuleContext<originirParser::Q_KEY_declarationContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::Define_gate_declarationContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Define_gate_declarationContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}

tree::TerminalNode* originirParser::Define_gate_declarationContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

std::vector<originirParser::ExpressionContext *> originirParser::Define_gate_declarationContext::expression() {
  return getRuleContexts<originirParser::ExpressionContext>();
}

originirParser::ExpressionContext* originirParser::Define_gate_declarationContext::expression(size_t i) {
  return getRuleContext<originirParser::ExpressionContext>(i);
}

tree::TerminalNode* originirParser::Define_gate_declarationContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Define_gate_declarationContext::getRuleIndex() const {
  return originirParser::RuleDefine_gate_declaration;
}

void originirParser::Define_gate_declarationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDefine_gate_declaration(this);
}

void originirParser::Define_gate_declarationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDefine_gate_declaration(this);
}


antlrcpp::Any originirParser::Define_gate_declarationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDefine_gate_declaration(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Define_gate_declarationContext* originirParser::define_gate_declaration() {
  Define_gate_declarationContext *_localctx = _tracker.createInstance<Define_gate_declarationContext>(_ctx, getState());
  enterRule(_localctx, 34, originirParser::RuleDefine_gate_declaration);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    setState(336);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(306);
      id();
      setState(307);
      q_KEY_declaration();
      setState(312);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == originirParser::COMMA) {
        setState(308);
        match(originirParser::COMMA);
        setState(309);
        q_KEY_declaration();
        setState(314);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(315);
      id();
      setState(316);
      q_KEY_declaration();
      setState(321);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
      while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
        if (alt == 1) {
          setState(317);
          match(originirParser::COMMA);
          setState(318);
          q_KEY_declaration(); 
        }
        setState(323);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
      }
      setState(324);
      match(originirParser::COMMA);
      setState(325);
      match(originirParser::LPAREN);
      setState(326);
      expression();
      setState(331);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == originirParser::COMMA) {
        setState(327);
        match(originirParser::COMMA);
        setState(328);
        expression();
        setState(333);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(334);
      match(originirParser::RPAREN);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_without_parameter_type(this);
}

void originirParser::Single_gate_without_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_without_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_without_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_without_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_without_parameter_typeContext* originirParser::single_gate_without_parameter_type() {
  Single_gate_without_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_without_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 36, originirParser::RuleSingle_gate_without_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(338);
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

tree::TerminalNode* originirParser::Single_gate_with_one_parameter_typeContext::P_GATE() {
  return getToken(originirParser::P_GATE, 0);
}


size_t originirParser::Single_gate_with_one_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleSingle_gate_with_one_parameter_type;
}

void originirParser::Single_gate_with_one_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_one_parameter_type(this);
}

void originirParser::Single_gate_with_one_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_one_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_one_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_one_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_one_parameter_typeContext* originirParser::single_gate_with_one_parameter_type() {
  Single_gate_with_one_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_one_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 38, originirParser::RuleSingle_gate_with_one_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(340);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::P_GATE))) != 0))) {
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_two_parameter_type(this);
}

void originirParser::Single_gate_with_two_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_two_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_two_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_two_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_two_parameter_typeContext* originirParser::single_gate_with_two_parameter_type() {
  Single_gate_with_two_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_two_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 40, originirParser::RuleSingle_gate_with_two_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(342);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_three_parameter_type(this);
}

void originirParser::Single_gate_with_three_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_three_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_three_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_three_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_three_parameter_typeContext* originirParser::single_gate_with_three_parameter_type() {
  Single_gate_with_three_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_three_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 42, originirParser::RuleSingle_gate_with_three_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(344);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSingle_gate_with_four_parameter_type(this);
}

void originirParser::Single_gate_with_four_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSingle_gate_with_four_parameter_type(this);
}


antlrcpp::Any originirParser::Single_gate_with_four_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitSingle_gate_with_four_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Single_gate_with_four_parameter_typeContext* originirParser::single_gate_with_four_parameter_type() {
  Single_gate_with_four_parameter_typeContext *_localctx = _tracker.createInstance<Single_gate_with_four_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 44, originirParser::RuleSingle_gate_with_four_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(346);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_without_parameter_type(this);
}

void originirParser::Double_gate_without_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_without_parameter_type(this);
}


antlrcpp::Any originirParser::Double_gate_without_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_without_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_without_parameter_typeContext* originirParser::double_gate_without_parameter_type() {
  Double_gate_without_parameter_typeContext *_localctx = _tracker.createInstance<Double_gate_without_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 46, originirParser::RuleDouble_gate_without_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(348);
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

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_typeContext::RXX_GATE() {
  return getToken(originirParser::RXX_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_typeContext::RYY_GATE() {
  return getToken(originirParser::RYY_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_typeContext::RZZ_GATE() {
  return getToken(originirParser::RZZ_GATE, 0);
}

tree::TerminalNode* originirParser::Double_gate_with_one_parameter_typeContext::RZX_GATE() {
  return getToken(originirParser::RZX_GATE, 0);
}


size_t originirParser::Double_gate_with_one_parameter_typeContext::getRuleIndex() const {
  return originirParser::RuleDouble_gate_with_one_parameter_type;
}

void originirParser::Double_gate_with_one_parameter_typeContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_one_parameter_type(this);
}

void originirParser::Double_gate_with_one_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_one_parameter_type(this);
}


antlrcpp::Any originirParser::Double_gate_with_one_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_with_one_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_with_one_parameter_typeContext* originirParser::double_gate_with_one_parameter_type() {
  Double_gate_with_one_parameter_typeContext *_localctx = _tracker.createInstance<Double_gate_with_one_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 48, originirParser::RuleDouble_gate_with_one_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(350);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE))) != 0))) {
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDouble_gate_with_four_parameter_type(this);
}

void originirParser::Double_gate_with_four_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDouble_gate_with_four_parameter_type(this);
}


antlrcpp::Any originirParser::Double_gate_with_four_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDouble_gate_with_four_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Double_gate_with_four_parameter_typeContext* originirParser::double_gate_with_four_parameter_type() {
  Double_gate_with_four_parameter_typeContext *_localctx = _tracker.createInstance<Double_gate_with_four_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 50, originirParser::RuleDouble_gate_with_four_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(352);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterTriple_gate_without_parameter_type(this);
}

void originirParser::Triple_gate_without_parameter_typeContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitTriple_gate_without_parameter_type(this);
}


antlrcpp::Any originirParser::Triple_gate_without_parameter_typeContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitTriple_gate_without_parameter_type(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Triple_gate_without_parameter_typeContext* originirParser::triple_gate_without_parameter_type() {
  Triple_gate_without_parameter_typeContext *_localctx = _tracker.createInstance<Triple_gate_without_parameter_typeContext>(_ctx, getState());
  enterRule(_localctx, 52, originirParser::RuleTriple_gate_without_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(354);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPri_ckey(this);
}
void originirParser::Pri_ckeyContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPri_ckey(this);
}

antlrcpp::Any originirParser::Pri_ckeyContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPri_cst(this);
}
void originirParser::Pri_cstContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPri_cst(this);
}

antlrcpp::Any originirParser::Pri_cstContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterPri_expr(this);
}
void originirParser::Pri_exprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitPri_expr(this);
}

antlrcpp::Any originirParser::Pri_exprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitPri_expr(this);
  else
    return visitor->visitChildren(this);
}
originirParser::Primary_expressionContext* originirParser::primary_expression() {
  Primary_expressionContext *_localctx = _tracker.createInstance<Primary_expressionContext>(_ctx, getState());
  enterRule(_localctx, 54, originirParser::RulePrimary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(362);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::C_KEY: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_ckeyContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(356);
        c_KEY_declaration();
        break;
      }

      case originirParser::PI:
      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_cstContext>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(357);
        constant();
        break;
      }

      case originirParser::LPAREN: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_exprContext>(_localctx));
        enterOuterAlt(_localctx, 3);
        setState(358);
        match(originirParser::LPAREN);
        setState(359);
        expression();
        setState(360);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterConstant(this);
}

void originirParser::ConstantContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitConstant(this);
}


antlrcpp::Any originirParser::ConstantContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitConstant(this);
  else
    return visitor->visitChildren(this);
}

originirParser::ConstantContext* originirParser::constant() {
  ConstantContext *_localctx = _tracker.createInstance<ConstantContext>(_ctx, getState());
  enterRule(_localctx, 56, originirParser::RuleConstant);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(364);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUnary_expression(this);
}

void originirParser::Unary_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUnary_expression(this);
}


antlrcpp::Any originirParser::Unary_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitUnary_expression(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Unary_expressionContext* originirParser::unary_expression() {
  Unary_expressionContext *_localctx = _tracker.createInstance<Unary_expressionContext>(_ctx, getState());
  enterRule(_localctx, 56, originirParser::RuleUnary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(371);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::PI:
      case originirParser::C_KEY:
      case originirParser::LPAREN:
      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        enterOuterAlt(_localctx, 1);
        setState(364);
        primary_expression();
        break;
      }

      case originirParser::PLUS: {
        enterOuterAlt(_localctx, 2);
        setState(365);
        match(originirParser::PLUS);
        setState(366);
        primary_expression();
        break;
      }

      case originirParser::MINUS: {
        enterOuterAlt(_localctx, 3);
        setState(367);
        match(originirParser::MINUS);
        setState(368);
        primary_expression();
        break;
      }

      case originirParser::NOT: {
        enterOuterAlt(_localctx, 4);
        setState(369);
        match(originirParser::NOT);
        setState(370);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMultiplicative_expression(this);
}

void originirParser::Multiplicative_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMultiplicative_expression(this);
}


antlrcpp::Any originirParser::Multiplicative_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  size_t startState = 58;
  enterRecursionRule(_localctx, 58, originirParser::RuleMultiplicative_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(374);
    unary_expression();
    _ctx->stop = _input->LT(-1);
    setState(384);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(382);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Multiplicative_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression);
          setState(376);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(377);
          match(originirParser::MUL);
          setState(378);
          unary_expression();
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Multiplicative_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression);
          setState(379);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(380);
          match(originirParser::DIV);
          setState(381);
          unary_expression();
          break;
        }

        } 
      }
      setState(386);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAddtive_expression(this);
}

void originirParser::Addtive_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAddtive_expression(this);
}


antlrcpp::Any originirParser::Addtive_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  size_t startState = 60;
  enterRecursionRule(_localctx, 60, originirParser::RuleAddtive_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(388);
    multiplicative_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(398);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(396);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Addtive_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleAddtive_expression);
          setState(390);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(391);
          match(originirParser::PLUS);
          setState(392);
          multiplicative_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Addtive_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleAddtive_expression);
          setState(393);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(394);
          match(originirParser::MINUS);
          setState(395);
          multiplicative_expression(0);
          break;
        }

        } 
      }
      setState(400);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterRelational_expression(this);
}

void originirParser::Relational_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitRelational_expression(this);
}


antlrcpp::Any originirParser::Relational_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  size_t startState = 62;
  enterRecursionRule(_localctx, 62, originirParser::RuleRelational_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(402);
    addtive_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(418);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(416);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(404);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(405);
          match(originirParser::LT);
          setState(406);
          addtive_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(407);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(408);
          match(originirParser::GT);
          setState(409);
          addtive_expression(0);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(410);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(411);
          match(originirParser::LEQ);
          setState(412);
          addtive_expression(0);
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(413);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(414);
          match(originirParser::GEQ);
          setState(415);
          addtive_expression(0);
          break;
        }

        } 
      }
      setState(420);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 21, _ctx);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterEquality_expression(this);
}

void originirParser::Equality_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitEquality_expression(this);
}


antlrcpp::Any originirParser::Equality_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  size_t startState = 64;
  enterRecursionRule(_localctx, 64, originirParser::RuleEquality_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(422);
    relational_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(432);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(430);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 22, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Equality_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleEquality_expression);
          setState(424);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(425);
          match(originirParser::EQ);
          setState(426);
          relational_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Equality_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleEquality_expression);
          setState(427);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(428);
          match(originirParser::NE);
          setState(429);
          relational_expression(0);
          break;
        }

        } 
      }
      setState(434);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 23, _ctx);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogical_and_expression(this);
}

void originirParser::Logical_and_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogical_and_expression(this);
}


antlrcpp::Any originirParser::Logical_and_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  size_t startState = 66;
  enterRecursionRule(_localctx, 66, originirParser::RuleLogical_and_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(436);
    equality_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(443);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<Logical_and_expressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLogical_and_expression);
        setState(438);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(439);
        match(originirParser::AND);
        setState(440);
        equality_expression(0); 
      }
      setState(445);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 24, _ctx);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLogical_or_expression(this);
}

void originirParser::Logical_or_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLogical_or_expression(this);
}


antlrcpp::Any originirParser::Logical_or_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  size_t startState = 68;
  enterRecursionRule(_localctx, 68, originirParser::RuleLogical_or_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(447);
    logical_and_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(454);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<Logical_or_expressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLogical_or_expression);
        setState(449);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(450);
        match(originirParser::OR);
        setState(451);
        logical_and_expression(0); 
      }
      setState(456);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 25, _ctx);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAssignment_expression(this);
}

void originirParser::Assignment_expressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAssignment_expression(this);
}


antlrcpp::Any originirParser::Assignment_expressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitAssignment_expression(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Assignment_expressionContext* originirParser::assignment_expression() {
  Assignment_expressionContext *_localctx = _tracker.createInstance<Assignment_expressionContext>(_ctx, getState());
  enterRule(_localctx, 70, originirParser::RuleAssignment_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(462);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 26, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(457);
      logical_or_expression(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(458);
      c_KEY_declaration();
      setState(459);
      match(originirParser::ASSIGN);
      setState(460);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression(this);
}

void originirParser::ExpressionContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression(this);
}


antlrcpp::Any originirParser::ExpressionContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitExpression(this);
  else
    return visitor->visitChildren(this);
}

originirParser::ExpressionContext* originirParser::expression() {
  ExpressionContext *_localctx = _tracker.createInstance<ExpressionContext>(_ctx, getState());
  enterRule(_localctx, 72, originirParser::RuleExpression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(464);
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

std::vector<tree::TerminalNode *> originirParser::Controlbit_listContext::Identifier() {
  return getTokens(originirParser::Identifier);
}

tree::TerminalNode* originirParser::Controlbit_listContext::Identifier(size_t i) {
  return getToken(originirParser::Identifier, i);
}


size_t originirParser::Controlbit_listContext::getRuleIndex() const {
  return originirParser::RuleControlbit_list;
}

void originirParser::Controlbit_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterControlbit_list(this);
}

void originirParser::Controlbit_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitControlbit_list(this);
}


antlrcpp::Any originirParser::Controlbit_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitControlbit_list(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Controlbit_listContext* originirParser::controlbit_list() {
  Controlbit_listContext *_localctx = _tracker.createInstance<Controlbit_listContext>(_ctx, getState());
  enterRule(_localctx, 74, originirParser::RuleControlbit_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(482);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::Q_KEY: {
        enterOuterAlt(_localctx, 1);
        setState(466);
        q_KEY_declaration();
        setState(471);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == originirParser::COMMA) {
          setState(467);
          match(originirParser::COMMA);
          setState(468);
          q_KEY_declaration();
          setState(473);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        break;
      }

      case originirParser::Identifier: {
        enterOuterAlt(_localctx, 2);
        setState(474);
        match(originirParser::Identifier);
        setState(479);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == originirParser::COMMA) {
          setState(475);
          match(originirParser::COMMA);
          setState(476);
          match(originirParser::Identifier);
          setState(481);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
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

originirParser::Gate_func_statementContext* originirParser::StatementContext::gate_func_statement() {
  return getRuleContext<originirParser::Gate_func_statementContext>(0);
}


size_t originirParser::StatementContext::getRuleIndex() const {
  return originirParser::RuleStatement;
}

void originirParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void originirParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}


antlrcpp::Any originirParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::StatementContext* originirParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 76, originirParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(496);
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
      case originirParser::P_GATE:
      case originirParser::CNOT_GATE:
      case originirParser::CZ_GATE:
      case originirParser::CU_GATE:
      case originirParser::ISWAP_GATE:
      case originirParser::SQISWAP_GATE:
      case originirParser::SWAPZ1_GATE:
      case originirParser::ISWAPTHETA_GATE:
      case originirParser::CR_GATE:
      case originirParser::RXX_GATE:
      case originirParser::RYY_GATE:
      case originirParser::RZZ_GATE:
      case originirParser::RZX_GATE:
      case originirParser::TOFFOLI_GATE:
      case originirParser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(484);
        quantum_gate_declaration();
        setState(485);
        match(originirParser::NEWLINE);
        break;
      }

      case originirParser::CONTROL_KEY: {
        enterOuterAlt(_localctx, 2);
        setState(487);
        control_statement();
        break;
      }

      case originirParser::QIF_KEY: {
        enterOuterAlt(_localctx, 3);
        setState(488);
        qif_statement();
        break;
      }

      case originirParser::QWHILE_KEY: {
        enterOuterAlt(_localctx, 4);
        setState(489);
        qwhile_statement();
        break;
      }

      case originirParser::DAGGER_KEY: {
        enterOuterAlt(_localctx, 5);
        setState(490);
        dagger_statement();
        break;
      }

      case originirParser::MEASURE_KEY: {
        enterOuterAlt(_localctx, 6);
        setState(491);
        measure_statement();
        break;
      }

      case originirParser::RESET_KEY: {
        enterOuterAlt(_localctx, 7);
        setState(492);
        reset_statement();
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
        setState(493);
        expression_statement();
        break;
      }

      case originirParser::BARRIER_KEY: {
        enterOuterAlt(_localctx, 9);
        setState(494);
        barrier_statement();
        break;
      }

      case originirParser::QGATE_KEY: {
        enterOuterAlt(_localctx, 10);
        setState(495);
        gate_func_statement();
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDagger_statement(this);
}

void originirParser::Dagger_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDagger_statement(this);
}


antlrcpp::Any originirParser::Dagger_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDagger_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Dagger_statementContext* originirParser::dagger_statement() {
  Dagger_statementContext *_localctx = _tracker.createInstance<Dagger_statementContext>(_ctx, getState());
  enterRule(_localctx, 78, originirParser::RuleDagger_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(498);
    match(originirParser::DAGGER_KEY);
    setState(499);
    match(originirParser::NEWLINE);
    setState(503);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::QGATE_KEY)
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
      | (1ULL << originirParser::P_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (originirParser::PLUS - 64))
      | (1ULL << (originirParser::MINUS - 64))
      | (1ULL << (originirParser::LPAREN - 64))
      | (1ULL << (originirParser::Identifier - 64))
      | (1ULL << (originirParser::Integer_Literal - 64))
      | (1ULL << (originirParser::Double_Literal - 64)))) != 0)) {
      setState(500);
      statement();
      setState(505);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(506);
    match(originirParser::ENDDAGGER_KEY);
    setState(507);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterControl_statement(this);
}

void originirParser::Control_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitControl_statement(this);
}


antlrcpp::Any originirParser::Control_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitControl_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Control_statementContext* originirParser::control_statement() {
  Control_statementContext *_localctx = _tracker.createInstance<Control_statementContext>(_ctx, getState());
  enterRule(_localctx, 80, originirParser::RuleControl_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(509);
    match(originirParser::CONTROL_KEY);
    setState(510);
    controlbit_list();
    setState(511);
    match(originirParser::NEWLINE);
    setState(515);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::QGATE_KEY)
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
      | (1ULL << originirParser::P_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (originirParser::PLUS - 64))
      | (1ULL << (originirParser::MINUS - 64))
      | (1ULL << (originirParser::LPAREN - 64))
      | (1ULL << (originirParser::Identifier - 64))
      | (1ULL << (originirParser::Integer_Literal - 64))
      | (1ULL << (originirParser::Double_Literal - 64)))) != 0)) {
      setState(512);
      statement();
      setState(517);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(518);
    match(originirParser::ENDCONTROL_KEY);
    setState(519);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQelse_statement_fragment(this);
}

void originirParser::Qelse_statement_fragmentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQelse_statement_fragment(this);
}


antlrcpp::Any originirParser::Qelse_statement_fragmentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitQelse_statement_fragment(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Qelse_statement_fragmentContext* originirParser::qelse_statement_fragment() {
  Qelse_statement_fragmentContext *_localctx = _tracker.createInstance<Qelse_statement_fragmentContext>(_ctx, getState());
  enterRule(_localctx, 82, originirParser::RuleQelse_statement_fragment);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(521);
    match(originirParser::ELSE_KEY);
    setState(522);
    match(originirParser::NEWLINE);
    setState(526);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::QGATE_KEY)
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
      | (1ULL << originirParser::P_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (originirParser::PLUS - 64))
      | (1ULL << (originirParser::MINUS - 64))
      | (1ULL << (originirParser::LPAREN - 64))
      | (1ULL << (originirParser::Identifier - 64))
      | (1ULL << (originirParser::Integer_Literal - 64))
      | (1ULL << (originirParser::Double_Literal - 64)))) != 0)) {
      setState(523);
      statement();
      setState(528);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQif_if(this);
}
void originirParser::Qif_ifContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQif_if(this);
}

antlrcpp::Any originirParser::Qif_ifContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQif_ifelse(this);
}
void originirParser::Qif_ifelseContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQif_ifelse(this);
}

antlrcpp::Any originirParser::Qif_ifelseContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitQif_ifelse(this);
  else
    return visitor->visitChildren(this);
}
originirParser::Qif_statementContext* originirParser::qif_statement() {
  Qif_statementContext *_localctx = _tracker.createInstance<Qif_statementContext>(_ctx, getState());
  enterRule(_localctx, 84, originirParser::RuleQif_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(554);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 36, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<Qif_statementContext *>(_tracker.createInstance<originirParser::Qif_ifContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(529);
      match(originirParser::QIF_KEY);
      setState(530);
      expression();
      setState(531);
      match(originirParser::NEWLINE);
      setState(535);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << originirParser::PI)
        | (1ULL << originirParser::C_KEY)
        | (1ULL << originirParser::BARRIER_KEY)
        | (1ULL << originirParser::QGATE_KEY)
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
        | (1ULL << originirParser::P_GATE)
        | (1ULL << originirParser::CNOT_GATE)
        | (1ULL << originirParser::CZ_GATE)
        | (1ULL << originirParser::CU_GATE)
        | (1ULL << originirParser::ISWAP_GATE)
        | (1ULL << originirParser::SQISWAP_GATE)
        | (1ULL << originirParser::SWAPZ1_GATE)
        | (1ULL << originirParser::ISWAPTHETA_GATE)
        | (1ULL << originirParser::CR_GATE)
        | (1ULL << originirParser::RXX_GATE)
        | (1ULL << originirParser::RYY_GATE)
        | (1ULL << originirParser::RZZ_GATE)
        | (1ULL << originirParser::RZX_GATE)
        | (1ULL << originirParser::TOFFOLI_GATE)
        | (1ULL << originirParser::DAGGER_KEY)
        | (1ULL << originirParser::CONTROL_KEY)
        | (1ULL << originirParser::QIF_KEY)
        | (1ULL << originirParser::QWHILE_KEY)
        | (1ULL << originirParser::MEASURE_KEY)
        | (1ULL << originirParser::RESET_KEY)
        | (1ULL << originirParser::NOT))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 64)) & ((1ULL << (originirParser::PLUS - 64))
        | (1ULL << (originirParser::MINUS - 64))
        | (1ULL << (originirParser::LPAREN - 64))
        | (1ULL << (originirParser::Identifier - 64))
        | (1ULL << (originirParser::Integer_Literal - 64))
        | (1ULL << (originirParser::Double_Literal - 64)))) != 0)) {
        setState(532);
        statement();
        setState(537);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(538);
      qelse_statement_fragment();
      setState(539);
      match(originirParser::ENDIF_KEY);
      setState(540);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      _localctx = dynamic_cast<Qif_statementContext *>(_tracker.createInstance<originirParser::Qif_ifelseContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(542);
      match(originirParser::QIF_KEY);
      setState(543);
      expression();
      setState(544);
      match(originirParser::NEWLINE);
      setState(548);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << originirParser::PI)
        | (1ULL << originirParser::C_KEY)
        | (1ULL << originirParser::BARRIER_KEY)
        | (1ULL << originirParser::QGATE_KEY)
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
        | (1ULL << originirParser::P_GATE)
        | (1ULL << originirParser::CNOT_GATE)
        | (1ULL << originirParser::CZ_GATE)
        | (1ULL << originirParser::CU_GATE)
        | (1ULL << originirParser::ISWAP_GATE)
        | (1ULL << originirParser::SQISWAP_GATE)
        | (1ULL << originirParser::SWAPZ1_GATE)
        | (1ULL << originirParser::ISWAPTHETA_GATE)
        | (1ULL << originirParser::CR_GATE)
        | (1ULL << originirParser::RXX_GATE)
        | (1ULL << originirParser::RYY_GATE)
        | (1ULL << originirParser::RZZ_GATE)
        | (1ULL << originirParser::RZX_GATE)
        | (1ULL << originirParser::TOFFOLI_GATE)
        | (1ULL << originirParser::DAGGER_KEY)
        | (1ULL << originirParser::CONTROL_KEY)
        | (1ULL << originirParser::QIF_KEY)
        | (1ULL << originirParser::QWHILE_KEY)
        | (1ULL << originirParser::MEASURE_KEY)
        | (1ULL << originirParser::RESET_KEY)
        | (1ULL << originirParser::NOT))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
        ((1ULL << (_la - 64)) & ((1ULL << (originirParser::PLUS - 64))
        | (1ULL << (originirParser::MINUS - 64))
        | (1ULL << (originirParser::LPAREN - 64))
        | (1ULL << (originirParser::Identifier - 64))
        | (1ULL << (originirParser::Integer_Literal - 64))
        | (1ULL << (originirParser::Double_Literal - 64)))) != 0)) {
        setState(545);
        statement();
        setState(550);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(551);
      match(originirParser::ENDIF_KEY);
      setState(552);
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
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQwhile_statement(this);
}

void originirParser::Qwhile_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQwhile_statement(this);
}


antlrcpp::Any originirParser::Qwhile_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitQwhile_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Qwhile_statementContext* originirParser::qwhile_statement() {
  Qwhile_statementContext *_localctx = _tracker.createInstance<Qwhile_statementContext>(_ctx, getState());
  enterRule(_localctx, 86, originirParser::RuleQwhile_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(556);
    match(originirParser::QWHILE_KEY);
    setState(557);
    expression();
    setState(558);
    match(originirParser::NEWLINE);
    setState(562);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::PI)
      | (1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::BARRIER_KEY)
      | (1ULL << originirParser::QGATE_KEY)
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
      | (1ULL << originirParser::P_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY)
      | (1ULL << originirParser::QIF_KEY)
      | (1ULL << originirParser::QWHILE_KEY)
      | (1ULL << originirParser::MEASURE_KEY)
      | (1ULL << originirParser::RESET_KEY)
      | (1ULL << originirParser::NOT))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
      ((1ULL << (_la - 64)) & ((1ULL << (originirParser::PLUS - 64))
      | (1ULL << (originirParser::MINUS - 64))
      | (1ULL << (originirParser::LPAREN - 64))
      | (1ULL << (originirParser::Identifier - 64))
      | (1ULL << (originirParser::Integer_Literal - 64))
      | (1ULL << (originirParser::Double_Literal - 64)))) != 0)) {
      setState(559);
      statement();
      setState(564);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(565);
    match(originirParser::ENDQWHILE_KEY);
    setState(566);
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

tree::TerminalNode* originirParser::Measure_statementContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}

tree::TerminalNode* originirParser::Measure_statementContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}

tree::TerminalNode* originirParser::Measure_statementContext::C_KEY() {
  return getToken(originirParser::C_KEY, 0);
}


size_t originirParser::Measure_statementContext::getRuleIndex() const {
  return originirParser::RuleMeasure_statement;
}

void originirParser::Measure_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMeasure_statement(this);
}

void originirParser::Measure_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMeasure_statement(this);
}


antlrcpp::Any originirParser::Measure_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitMeasure_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Measure_statementContext* originirParser::measure_statement() {
  Measure_statementContext *_localctx = _tracker.createInstance<Measure_statementContext>(_ctx, getState());
  enterRule(_localctx, 88, originirParser::RuleMeasure_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(579);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 38, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(568);
      match(originirParser::MEASURE_KEY);
      setState(569);
      q_KEY_declaration();
      setState(570);
      match(originirParser::COMMA);
      setState(571);
      c_KEY_declaration();
      setState(572);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(574);
      match(originirParser::MEASURE_KEY);
      setState(575);
      match(originirParser::Q_KEY);
      setState(576);
      match(originirParser::COMMA);
      setState(577);
      match(originirParser::C_KEY);
      setState(578);
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

tree::TerminalNode* originirParser::Reset_statementContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}

tree::TerminalNode* originirParser::Reset_statementContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}


size_t originirParser::Reset_statementContext::getRuleIndex() const {
  return originirParser::RuleReset_statement;
}

void originirParser::Reset_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReset_statement(this);
}

void originirParser::Reset_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReset_statement(this);
}


antlrcpp::Any originirParser::Reset_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitReset_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Reset_statementContext* originirParser::reset_statement() {
  Reset_statementContext *_localctx = _tracker.createInstance<Reset_statementContext>(_ctx, getState());
  enterRule(_localctx, 90, originirParser::RuleReset_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(588);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 39, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(581);
      match(originirParser::RESET_KEY);
      setState(582);
      q_KEY_declaration();
      setState(583);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(585);
      match(originirParser::RESET_KEY);
      setState(586);
      match(originirParser::Q_KEY);
      setState(587);
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

tree::TerminalNode* originirParser::Barrier_statementContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}

tree::TerminalNode* originirParser::Barrier_statementContext::Q_KEY() {
  return getToken(originirParser::Q_KEY, 0);
}


size_t originirParser::Barrier_statementContext::getRuleIndex() const {
  return originirParser::RuleBarrier_statement;
}

void originirParser::Barrier_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBarrier_statement(this);
}

void originirParser::Barrier_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBarrier_statement(this);
}


antlrcpp::Any originirParser::Barrier_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitBarrier_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Barrier_statementContext* originirParser::barrier_statement() {
  Barrier_statementContext *_localctx = _tracker.createInstance<Barrier_statementContext>(_ctx, getState());
  enterRule(_localctx, 92, originirParser::RuleBarrier_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(597);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 40, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(590);
      match(originirParser::BARRIER_KEY);
      setState(591);
      controlbit_list();
      setState(592);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(594);
      match(originirParser::BARRIER_KEY);
      setState(595);
      match(originirParser::Q_KEY);
      setState(596);
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

//----------------- Expression_statementContext ------------------------------------------------------------------

originirParser::Expression_statementContext::Expression_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::ExpressionContext* originirParser::Expression_statementContext::expression() {
  return getRuleContext<originirParser::ExpressionContext>(0);
}

tree::TerminalNode* originirParser::Expression_statementContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}


size_t originirParser::Expression_statementContext::getRuleIndex() const {
  return originirParser::RuleExpression_statement;
}

void originirParser::Expression_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpression_statement(this);
}

void originirParser::Expression_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpression_statement(this);
}


antlrcpp::Any originirParser::Expression_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitExpression_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Expression_statementContext* originirParser::expression_statement() {
  Expression_statementContext *_localctx = _tracker.createInstance<Expression_statementContext>(_ctx, getState());
  enterRule(_localctx, 94, originirParser::RuleExpression_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(599);
    expression();
    setState(600);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Define_gate_statementContext ------------------------------------------------------------------

originirParser::Define_gate_statementContext::Define_gate_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Gate_nameContext* originirParser::Define_gate_statementContext::gate_name() {
  return getRuleContext<originirParser::Gate_nameContext>(0);
}

originirParser::Id_listContext* originirParser::Define_gate_statementContext::id_list() {
  return getRuleContext<originirParser::Id_listContext>(0);
}

tree::TerminalNode* originirParser::Define_gate_statementContext::NEWLINE() {
  return getToken(originirParser::NEWLINE, 0);
}

tree::TerminalNode* originirParser::Define_gate_statementContext::COMMA() {
  return getToken(originirParser::COMMA, 0);
}

tree::TerminalNode* originirParser::Define_gate_statementContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

originirParser::ExplistContext* originirParser::Define_gate_statementContext::explist() {
  return getRuleContext<originirParser::ExplistContext>(0);
}

tree::TerminalNode* originirParser::Define_gate_statementContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Define_gate_statementContext::getRuleIndex() const {
  return originirParser::RuleDefine_gate_statement;
}

void originirParser::Define_gate_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDefine_gate_statement(this);
}

void originirParser::Define_gate_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDefine_gate_statement(this);
}


antlrcpp::Any originirParser::Define_gate_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDefine_gate_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Define_gate_statementContext* originirParser::define_gate_statement() {
  Define_gate_statementContext *_localctx = _tracker.createInstance<Define_gate_statementContext>(_ctx, getState());
  enterRule(_localctx, 96, originirParser::RuleDefine_gate_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(614);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 41, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(602);
      gate_name();
      setState(603);
      id_list();
      setState(604);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(606);
      gate_name();
      setState(607);
      id_list();
      setState(608);
      match(originirParser::COMMA);
      setState(609);
      match(originirParser::LPAREN);
      setState(610);
      explist();
      setState(611);
      match(originirParser::RPAREN);
      setState(612);
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

//----------------- Define_dagger_statementContext ------------------------------------------------------------------

originirParser::Define_dagger_statementContext::Define_dagger_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Define_dagger_statementContext::DAGGER_KEY() {
  return getToken(originirParser::DAGGER_KEY, 0);
}

std::vector<tree::TerminalNode *> originirParser::Define_dagger_statementContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Define_dagger_statementContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

tree::TerminalNode* originirParser::Define_dagger_statementContext::ENDDAGGER_KEY() {
  return getToken(originirParser::ENDDAGGER_KEY, 0);
}

std::vector<originirParser::User_defined_gateContext *> originirParser::Define_dagger_statementContext::user_defined_gate() {
  return getRuleContexts<originirParser::User_defined_gateContext>();
}

originirParser::User_defined_gateContext* originirParser::Define_dagger_statementContext::user_defined_gate(size_t i) {
  return getRuleContext<originirParser::User_defined_gateContext>(i);
}


size_t originirParser::Define_dagger_statementContext::getRuleIndex() const {
  return originirParser::RuleDefine_dagger_statement;
}

void originirParser::Define_dagger_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDefine_dagger_statement(this);
}

void originirParser::Define_dagger_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDefine_dagger_statement(this);
}


antlrcpp::Any originirParser::Define_dagger_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDefine_dagger_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Define_dagger_statementContext* originirParser::define_dagger_statement() {
  Define_dagger_statementContext *_localctx = _tracker.createInstance<Define_dagger_statementContext>(_ctx, getState());
  enterRule(_localctx, 98, originirParser::RuleDefine_dagger_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(616);
    match(originirParser::DAGGER_KEY);
    setState(617);
    match(originirParser::NEWLINE);
    setState(619); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(618);
      user_defined_gate();
      setState(621); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
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
      | (1ULL << originirParser::I_GATE)
      | (1ULL << originirParser::U2_GATE)
      | (1ULL << originirParser::RPHI_GATE)
      | (1ULL << originirParser::U3_GATE)
      | (1ULL << originirParser::U4_GATE)
      | (1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::P_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY))) != 0) || _la == originirParser::Identifier);
    setState(623);
    match(originirParser::ENDDAGGER_KEY);
    setState(624);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Define_control_statementContext ------------------------------------------------------------------

originirParser::Define_control_statementContext::Define_control_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Define_control_statementContext::CONTROL_KEY() {
  return getToken(originirParser::CONTROL_KEY, 0);
}

originirParser::Controlbit_listContext* originirParser::Define_control_statementContext::controlbit_list() {
  return getRuleContext<originirParser::Controlbit_listContext>(0);
}

std::vector<tree::TerminalNode *> originirParser::Define_control_statementContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Define_control_statementContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

tree::TerminalNode* originirParser::Define_control_statementContext::ENDCONTROL_KEY() {
  return getToken(originirParser::ENDCONTROL_KEY, 0);
}

std::vector<originirParser::User_defined_gateContext *> originirParser::Define_control_statementContext::user_defined_gate() {
  return getRuleContexts<originirParser::User_defined_gateContext>();
}

originirParser::User_defined_gateContext* originirParser::Define_control_statementContext::user_defined_gate(size_t i) {
  return getRuleContext<originirParser::User_defined_gateContext>(i);
}


size_t originirParser::Define_control_statementContext::getRuleIndex() const {
  return originirParser::RuleDefine_control_statement;
}

void originirParser::Define_control_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDefine_control_statement(this);
}

void originirParser::Define_control_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDefine_control_statement(this);
}


antlrcpp::Any originirParser::Define_control_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitDefine_control_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Define_control_statementContext* originirParser::define_control_statement() {
  Define_control_statementContext *_localctx = _tracker.createInstance<Define_control_statementContext>(_ctx, getState());
  enterRule(_localctx, 100, originirParser::RuleDefine_control_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(626);
    match(originirParser::CONTROL_KEY);
    setState(627);
    controlbit_list();
    setState(628);
    match(originirParser::NEWLINE);
    setState(630); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(629);
      user_defined_gate();
      setState(632); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
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
      | (1ULL << originirParser::I_GATE)
      | (1ULL << originirParser::U2_GATE)
      | (1ULL << originirParser::RPHI_GATE)
      | (1ULL << originirParser::U3_GATE)
      | (1ULL << originirParser::U4_GATE)
      | (1ULL << originirParser::RX_GATE)
      | (1ULL << originirParser::RY_GATE)
      | (1ULL << originirParser::RZ_GATE)
      | (1ULL << originirParser::U1_GATE)
      | (1ULL << originirParser::P_GATE)
      | (1ULL << originirParser::CNOT_GATE)
      | (1ULL << originirParser::CZ_GATE)
      | (1ULL << originirParser::CU_GATE)
      | (1ULL << originirParser::ISWAP_GATE)
      | (1ULL << originirParser::SQISWAP_GATE)
      | (1ULL << originirParser::SWAPZ1_GATE)
      | (1ULL << originirParser::ISWAPTHETA_GATE)
      | (1ULL << originirParser::CR_GATE)
      | (1ULL << originirParser::RXX_GATE)
      | (1ULL << originirParser::RYY_GATE)
      | (1ULL << originirParser::RZZ_GATE)
      | (1ULL << originirParser::RZX_GATE)
      | (1ULL << originirParser::TOFFOLI_GATE)
      | (1ULL << originirParser::DAGGER_KEY)
      | (1ULL << originirParser::CONTROL_KEY))) != 0) || _la == originirParser::Identifier);
    setState(634);
    match(originirParser::ENDCONTROL_KEY);
    setState(635);
    match(originirParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- User_defined_gateContext ------------------------------------------------------------------

originirParser::User_defined_gateContext::User_defined_gateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Define_gate_statementContext* originirParser::User_defined_gateContext::define_gate_statement() {
  return getRuleContext<originirParser::Define_gate_statementContext>(0);
}

originirParser::Define_dagger_statementContext* originirParser::User_defined_gateContext::define_dagger_statement() {
  return getRuleContext<originirParser::Define_dagger_statementContext>(0);
}

originirParser::Define_control_statementContext* originirParser::User_defined_gateContext::define_control_statement() {
  return getRuleContext<originirParser::Define_control_statementContext>(0);
}


size_t originirParser::User_defined_gateContext::getRuleIndex() const {
  return originirParser::RuleUser_defined_gate;
}

void originirParser::User_defined_gateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUser_defined_gate(this);
}

void originirParser::User_defined_gateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUser_defined_gate(this);
}


antlrcpp::Any originirParser::User_defined_gateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitUser_defined_gate(this);
  else
    return visitor->visitChildren(this);
}

originirParser::User_defined_gateContext* originirParser::user_defined_gate() {
  User_defined_gateContext *_localctx = _tracker.createInstance<User_defined_gateContext>(_ctx, getState());
  enterRule(_localctx, 102, originirParser::RuleUser_defined_gate);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(640);
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
      case originirParser::P_GATE:
      case originirParser::CNOT_GATE:
      case originirParser::CZ_GATE:
      case originirParser::CU_GATE:
      case originirParser::ISWAP_GATE:
      case originirParser::SQISWAP_GATE:
      case originirParser::SWAPZ1_GATE:
      case originirParser::ISWAPTHETA_GATE:
      case originirParser::CR_GATE:
      case originirParser::RXX_GATE:
      case originirParser::RYY_GATE:
      case originirParser::RZZ_GATE:
      case originirParser::RZX_GATE:
      case originirParser::TOFFOLI_GATE:
      case originirParser::Identifier: {
        enterOuterAlt(_localctx, 1);
        setState(637);
        define_gate_statement();
        break;
      }

      case originirParser::DAGGER_KEY: {
        enterOuterAlt(_localctx, 2);
        setState(638);
        define_dagger_statement();
        break;
      }

      case originirParser::CONTROL_KEY: {
        enterOuterAlt(_localctx, 3);
        setState(639);
        define_control_statement();
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

//----------------- ExplistContext ------------------------------------------------------------------

originirParser::ExplistContext::ExplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<originirParser::ExpContext *> originirParser::ExplistContext::exp() {
  return getRuleContexts<originirParser::ExpContext>();
}

originirParser::ExpContext* originirParser::ExplistContext::exp(size_t i) {
  return getRuleContext<originirParser::ExpContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::ExplistContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::ExplistContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}


size_t originirParser::ExplistContext::getRuleIndex() const {
  return originirParser::RuleExplist;
}

void originirParser::ExplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExplist(this);
}

void originirParser::ExplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExplist(this);
}


antlrcpp::Any originirParser::ExplistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitExplist(this);
  else
    return visitor->visitChildren(this);
}

originirParser::ExplistContext* originirParser::explist() {
  ExplistContext *_localctx = _tracker.createInstance<ExplistContext>(_ctx, getState());
  enterRule(_localctx, 104, originirParser::RuleExplist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(642);
    exp(0);
    setState(647);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == originirParser::COMMA) {
      setState(643);
      match(originirParser::COMMA);
      setState(644);
      exp(0);
      setState(649);
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

//----------------- ExpContext ------------------------------------------------------------------

originirParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::IdContext* originirParser::ExpContext::id() {
  return getRuleContext<originirParser::IdContext>(0);
}

tree::TerminalNode* originirParser::ExpContext::Integer_Literal() {
  return getToken(originirParser::Integer_Literal, 0);
}

tree::TerminalNode* originirParser::ExpContext::Double_Literal() {
  return getToken(originirParser::Double_Literal, 0);
}

tree::TerminalNode* originirParser::ExpContext::PI() {
  return getToken(originirParser::PI, 0);
}

tree::TerminalNode* originirParser::ExpContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

std::vector<originirParser::ExpContext *> originirParser::ExpContext::exp() {
  return getRuleContexts<originirParser::ExpContext>();
}

originirParser::ExpContext* originirParser::ExpContext::exp(size_t i) {
  return getRuleContext<originirParser::ExpContext>(i);
}

tree::TerminalNode* originirParser::ExpContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}

tree::TerminalNode* originirParser::ExpContext::MINUS() {
  return getToken(originirParser::MINUS, 0);
}

tree::TerminalNode* originirParser::ExpContext::MUL() {
  return getToken(originirParser::MUL, 0);
}

tree::TerminalNode* originirParser::ExpContext::DIV() {
  return getToken(originirParser::DIV, 0);
}

tree::TerminalNode* originirParser::ExpContext::PLUS() {
  return getToken(originirParser::PLUS, 0);
}


size_t originirParser::ExpContext::getRuleIndex() const {
  return originirParser::RuleExp;
}

void originirParser::ExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExp(this);
}

void originirParser::ExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExp(this);
}


antlrcpp::Any originirParser::ExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitExp(this);
  else
    return visitor->visitChildren(this);
}


originirParser::ExpContext* originirParser::exp() {
   return exp(0);
}

originirParser::ExpContext* originirParser::exp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  originirParser::ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, parentState);
  originirParser::ExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 106;
  enterRecursionRule(_localctx, 106, originirParser::RuleExp, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(661);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::Identifier: {
        setState(651);
        id();
        break;
      }

      case originirParser::Integer_Literal: {
        setState(652);
        match(originirParser::Integer_Literal);
        break;
      }

      case originirParser::Double_Literal: {
        setState(653);
        match(originirParser::Double_Literal);
        break;
      }

      case originirParser::PI: {
        setState(654);
        match(originirParser::PI);
        break;
      }

      case originirParser::LPAREN: {
        setState(655);
        match(originirParser::LPAREN);
        setState(656);
        exp(0);
        setState(657);
        match(originirParser::RPAREN);
        break;
      }

      case originirParser::MINUS: {
        setState(659);
        match(originirParser::MINUS);
        setState(660);
        exp(5);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(677);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 48, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(675);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 47, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(663);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(664);
          match(originirParser::MUL);
          setState(665);
          exp(5);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(666);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(667);
          match(originirParser::DIV);
          setState(668);
          exp(4);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(669);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(670);
          match(originirParser::PLUS);
          setState(671);
          exp(3);
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(672);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(673);
          match(originirParser::MINUS);
          setState(674);
          exp(2);
          break;
        }

        } 
      }
      setState(679);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 48, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- Gate_func_statementContext ------------------------------------------------------------------

originirParser::Gate_func_statementContext::Gate_func_statementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::Gate_func_statementContext::QGATE_KEY() {
  return getToken(originirParser::QGATE_KEY, 0);
}

originirParser::IdContext* originirParser::Gate_func_statementContext::id() {
  return getRuleContext<originirParser::IdContext>(0);
}

std::vector<originirParser::Id_listContext *> originirParser::Gate_func_statementContext::id_list() {
  return getRuleContexts<originirParser::Id_listContext>();
}

originirParser::Id_listContext* originirParser::Gate_func_statementContext::id_list(size_t i) {
  return getRuleContext<originirParser::Id_listContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::Gate_func_statementContext::NEWLINE() {
  return getTokens(originirParser::NEWLINE);
}

tree::TerminalNode* originirParser::Gate_func_statementContext::NEWLINE(size_t i) {
  return getToken(originirParser::NEWLINE, i);
}

tree::TerminalNode* originirParser::Gate_func_statementContext::ENDQGATE_KEY() {
  return getToken(originirParser::ENDQGATE_KEY, 0);
}

std::vector<originirParser::User_defined_gateContext *> originirParser::Gate_func_statementContext::user_defined_gate() {
  return getRuleContexts<originirParser::User_defined_gateContext>();
}

originirParser::User_defined_gateContext* originirParser::Gate_func_statementContext::user_defined_gate(size_t i) {
  return getRuleContext<originirParser::User_defined_gateContext>(i);
}

tree::TerminalNode* originirParser::Gate_func_statementContext::COMMA() {
  return getToken(originirParser::COMMA, 0);
}

tree::TerminalNode* originirParser::Gate_func_statementContext::LPAREN() {
  return getToken(originirParser::LPAREN, 0);
}

tree::TerminalNode* originirParser::Gate_func_statementContext::RPAREN() {
  return getToken(originirParser::RPAREN, 0);
}


size_t originirParser::Gate_func_statementContext::getRuleIndex() const {
  return originirParser::RuleGate_func_statement;
}

void originirParser::Gate_func_statementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGate_func_statement(this);
}

void originirParser::Gate_func_statementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGate_func_statement(this);
}


antlrcpp::Any originirParser::Gate_func_statementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitGate_func_statement(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Gate_func_statementContext* originirParser::gate_func_statement() {
  Gate_func_statementContext *_localctx = _tracker.createInstance<Gate_func_statementContext>(_ctx, getState());
  enterRule(_localctx, 108, originirParser::RuleGate_func_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(710);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 51, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(680);
      match(originirParser::QGATE_KEY);
      setState(681);
      id();
      setState(682);
      id_list();
      setState(683);
      match(originirParser::NEWLINE);
      setState(687);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
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
        | (1ULL << originirParser::I_GATE)
        | (1ULL << originirParser::U2_GATE)
        | (1ULL << originirParser::RPHI_GATE)
        | (1ULL << originirParser::U3_GATE)
        | (1ULL << originirParser::U4_GATE)
        | (1ULL << originirParser::RX_GATE)
        | (1ULL << originirParser::RY_GATE)
        | (1ULL << originirParser::RZ_GATE)
        | (1ULL << originirParser::U1_GATE)
        | (1ULL << originirParser::P_GATE)
        | (1ULL << originirParser::CNOT_GATE)
        | (1ULL << originirParser::CZ_GATE)
        | (1ULL << originirParser::CU_GATE)
        | (1ULL << originirParser::ISWAP_GATE)
        | (1ULL << originirParser::SQISWAP_GATE)
        | (1ULL << originirParser::SWAPZ1_GATE)
        | (1ULL << originirParser::ISWAPTHETA_GATE)
        | (1ULL << originirParser::CR_GATE)
        | (1ULL << originirParser::RXX_GATE)
        | (1ULL << originirParser::RYY_GATE)
        | (1ULL << originirParser::RZZ_GATE)
        | (1ULL << originirParser::RZX_GATE)
        | (1ULL << originirParser::TOFFOLI_GATE)
        | (1ULL << originirParser::DAGGER_KEY)
        | (1ULL << originirParser::CONTROL_KEY))) != 0) || _la == originirParser::Identifier) {
        setState(684);
        user_defined_gate();
        setState(689);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(690);
      match(originirParser::ENDQGATE_KEY);
      setState(691);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(693);
      match(originirParser::QGATE_KEY);
      setState(694);
      id();
      setState(695);
      id_list();
      setState(696);
      match(originirParser::COMMA);
      setState(697);
      match(originirParser::LPAREN);
      setState(698);
      id_list();
      setState(699);
      match(originirParser::RPAREN);
      setState(700);
      match(originirParser::NEWLINE);
      setState(704);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
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
        | (1ULL << originirParser::I_GATE)
        | (1ULL << originirParser::U2_GATE)
        | (1ULL << originirParser::RPHI_GATE)
        | (1ULL << originirParser::U3_GATE)
        | (1ULL << originirParser::U4_GATE)
        | (1ULL << originirParser::RX_GATE)
        | (1ULL << originirParser::RY_GATE)
        | (1ULL << originirParser::RZ_GATE)
        | (1ULL << originirParser::U1_GATE)
        | (1ULL << originirParser::P_GATE)
        | (1ULL << originirParser::CNOT_GATE)
        | (1ULL << originirParser::CZ_GATE)
        | (1ULL << originirParser::CU_GATE)
        | (1ULL << originirParser::ISWAP_GATE)
        | (1ULL << originirParser::SQISWAP_GATE)
        | (1ULL << originirParser::SWAPZ1_GATE)
        | (1ULL << originirParser::ISWAPTHETA_GATE)
        | (1ULL << originirParser::CR_GATE)
        | (1ULL << originirParser::RXX_GATE)
        | (1ULL << originirParser::RYY_GATE)
        | (1ULL << originirParser::RZZ_GATE)
        | (1ULL << originirParser::RZX_GATE)
        | (1ULL << originirParser::TOFFOLI_GATE)
        | (1ULL << originirParser::DAGGER_KEY)
        | (1ULL << originirParser::CONTROL_KEY))) != 0) || _la == originirParser::Identifier) {
        setState(701);
        user_defined_gate();
        setState(706);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(707);
      match(originirParser::ENDQGATE_KEY);
      setState(708);
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

//----------------- IdContext ------------------------------------------------------------------

originirParser::IdContext::IdContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* originirParser::IdContext::Identifier() {
  return getToken(originirParser::Identifier, 0);
}


size_t originirParser::IdContext::getRuleIndex() const {
  return originirParser::RuleId;
}

void originirParser::IdContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterId(this);
}

void originirParser::IdContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitId(this);
}


antlrcpp::Any originirParser::IdContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitId(this);
  else
    return visitor->visitChildren(this);
}

originirParser::IdContext* originirParser::id() {
  IdContext *_localctx = _tracker.createInstance<IdContext>(_ctx, getState());
  enterRule(_localctx, 110, originirParser::RuleId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(712);
    match(originirParser::Identifier);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Id_listContext ------------------------------------------------------------------

originirParser::Id_listContext::Id_listContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<originirParser::IdContext *> originirParser::Id_listContext::id() {
  return getRuleContexts<originirParser::IdContext>();
}

originirParser::IdContext* originirParser::Id_listContext::id(size_t i) {
  return getRuleContext<originirParser::IdContext>(i);
}

std::vector<tree::TerminalNode *> originirParser::Id_listContext::COMMA() {
  return getTokens(originirParser::COMMA);
}

tree::TerminalNode* originirParser::Id_listContext::COMMA(size_t i) {
  return getToken(originirParser::COMMA, i);
}


size_t originirParser::Id_listContext::getRuleIndex() const {
  return originirParser::RuleId_list;
}

void originirParser::Id_listContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterId_list(this);
}

void originirParser::Id_listContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitId_list(this);
}


antlrcpp::Any originirParser::Id_listContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitId_list(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Id_listContext* originirParser::id_list() {
  Id_listContext *_localctx = _tracker.createInstance<Id_listContext>(_ctx, getState());
  enterRule(_localctx, 112, originirParser::RuleId_list);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(714);
    id();
    setState(719);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        setState(715);
        match(originirParser::COMMA);
        setState(716);
        id(); 
      }
      setState(721);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 52, _ctx);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Gate_nameContext ------------------------------------------------------------------

originirParser::Gate_nameContext::Gate_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

originirParser::Single_gate_without_parameter_typeContext* originirParser::Gate_nameContext::single_gate_without_parameter_type() {
  return getRuleContext<originirParser::Single_gate_without_parameter_typeContext>(0);
}

originirParser::Single_gate_with_one_parameter_typeContext* originirParser::Gate_nameContext::single_gate_with_one_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_one_parameter_typeContext>(0);
}

originirParser::Single_gate_with_two_parameter_typeContext* originirParser::Gate_nameContext::single_gate_with_two_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_two_parameter_typeContext>(0);
}

originirParser::Single_gate_with_three_parameter_typeContext* originirParser::Gate_nameContext::single_gate_with_three_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_three_parameter_typeContext>(0);
}

originirParser::Single_gate_with_four_parameter_typeContext* originirParser::Gate_nameContext::single_gate_with_four_parameter_type() {
  return getRuleContext<originirParser::Single_gate_with_four_parameter_typeContext>(0);
}

originirParser::Double_gate_without_parameter_typeContext* originirParser::Gate_nameContext::double_gate_without_parameter_type() {
  return getRuleContext<originirParser::Double_gate_without_parameter_typeContext>(0);
}

originirParser::Double_gate_with_one_parameter_typeContext* originirParser::Gate_nameContext::double_gate_with_one_parameter_type() {
  return getRuleContext<originirParser::Double_gate_with_one_parameter_typeContext>(0);
}

originirParser::Double_gate_with_four_parameter_typeContext* originirParser::Gate_nameContext::double_gate_with_four_parameter_type() {
  return getRuleContext<originirParser::Double_gate_with_four_parameter_typeContext>(0);
}

originirParser::Triple_gate_without_parameter_typeContext* originirParser::Gate_nameContext::triple_gate_without_parameter_type() {
  return getRuleContext<originirParser::Triple_gate_without_parameter_typeContext>(0);
}

originirParser::IdContext* originirParser::Gate_nameContext::id() {
  return getRuleContext<originirParser::IdContext>(0);
}


size_t originirParser::Gate_nameContext::getRuleIndex() const {
  return originirParser::RuleGate_name;
}

void originirParser::Gate_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGate_name(this);
}

void originirParser::Gate_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<originirParserListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGate_name(this);
}


antlrcpp::Any originirParser::Gate_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<originirParserVisitor*>(visitor))
    return parserVisitor->visitGate_name(this);
  else
    return visitor->visitChildren(this);
}

originirParser::Gate_nameContext* originirParser::gate_name() {
  Gate_nameContext *_localctx = _tracker.createInstance<Gate_nameContext>(_ctx, getState());
  enterRule(_localctx, 114, originirParser::RuleGate_name);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(732);
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
        setState(722);
        single_gate_without_parameter_type();
        break;
      }

      case originirParser::RX_GATE:
      case originirParser::RY_GATE:
      case originirParser::RZ_GATE:
      case originirParser::U1_GATE:
      case originirParser::P_GATE: {
        enterOuterAlt(_localctx, 2);
        setState(723);
        single_gate_with_one_parameter_type();
        break;
      }

      case originirParser::U2_GATE:
      case originirParser::RPHI_GATE: {
        enterOuterAlt(_localctx, 3);
        setState(724);
        single_gate_with_two_parameter_type();
        break;
      }

      case originirParser::U3_GATE: {
        enterOuterAlt(_localctx, 4);
        setState(725);
        single_gate_with_three_parameter_type();
        break;
      }

      case originirParser::U4_GATE: {
        enterOuterAlt(_localctx, 5);
        setState(726);
        single_gate_with_four_parameter_type();
        break;
      }

      case originirParser::CNOT_GATE:
      case originirParser::CZ_GATE:
      case originirParser::ISWAP_GATE:
      case originirParser::SQISWAP_GATE:
      case originirParser::SWAPZ1_GATE: {
        enterOuterAlt(_localctx, 6);
        setState(727);
        double_gate_without_parameter_type();
        break;
      }

      case originirParser::ISWAPTHETA_GATE:
      case originirParser::CR_GATE:
      case originirParser::RXX_GATE:
      case originirParser::RYY_GATE:
      case originirParser::RZZ_GATE:
      case originirParser::RZX_GATE: {
        enterOuterAlt(_localctx, 7);
        setState(728);
        double_gate_with_one_parameter_type();
        break;
      }

      case originirParser::CU_GATE: {
        enterOuterAlt(_localctx, 8);
        setState(729);
        double_gate_with_four_parameter_type();
        break;
      }

      case originirParser::TOFFOLI_GATE: {
        enterOuterAlt(_localctx, 9);
        setState(730);
        triple_gate_without_parameter_type();
        break;
      }

      case originirParser::Identifier: {
        enterOuterAlt(_localctx, 10);
        setState(731);
        id();
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

//----------------- ConstantContext ------------------------------------------------------------------

bool originirParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 29: return multiplicative_expressionSempred(dynamic_cast<Multiplicative_expressionContext *>(context), predicateIndex);
    case 30: return addtive_expressionSempred(dynamic_cast<Addtive_expressionContext *>(context), predicateIndex);
    case 31: return relational_expressionSempred(dynamic_cast<Relational_expressionContext *>(context), predicateIndex);
    case 32: return equality_expressionSempred(dynamic_cast<Equality_expressionContext *>(context), predicateIndex);
    case 33: return logical_and_expressionSempred(dynamic_cast<Logical_and_expressionContext *>(context), predicateIndex);
    case 34: return logical_or_expressionSempred(dynamic_cast<Logical_or_expressionContext *>(context), predicateIndex);
    case 53: return expSempred(dynamic_cast<ExpContext *>(context), predicateIndex);

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

bool originirParser::expSempred(ExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 12: return precpred(_ctx, 4);
    case 13: return precpred(_ctx, 3);
    case 14: return precpred(_ctx, 2);
    case 15: return precpred(_ctx, 1);

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
  "triple_gate_without_parameter_declaration", "define_gate_declaration", 
  "single_gate_without_parameter_type", "single_gate_with_one_parameter_type", 
  "single_gate_with_two_parameter_type", "single_gate_with_three_parameter_type", 
  "single_gate_with_four_parameter_type", "double_gate_without_parameter_type", 
  "double_gate_with_one_parameter_type", "double_gate_with_four_parameter_type", 
  "triple_gate_without_parameter_type", "primary_expression", "unary_expression", 
  "multiplicative_expression", "addtive_expression", "relational_expression", 
  "equality_expression", "logical_and_expression", "logical_or_expression", 
  "assignment_expression", "expression", "controlbit_list", "statement", 
  "dagger_statement", "control_statement", "qelse_statement_fragment", "qif_statement", 
  "qwhile_statement", "measure_statement", "reset_statement", "barrier_statement", 
  "expression_statement", "define_gate_statement", "define_dagger_statement", 
  "define_control_statement", "user_defined_gate", "explist", "exp", "gate_func_statement", 
  "id", "id_list", "gate_name", "constant"
};

std::vector<std::string> originirParser::_literalNames = {
  "", "'PI'", "'QINIT'", "'CREG'", "'q'", "'c'", "'BARRIER'", "'QGATE'", 
  "'ENDQGATE'", "'ECHO'", "'H'", "'X'", "'NOT'", "'T'", "'S'", "'Y'", "'Z'", 
  "'X1'", "'Y1'", "'Z1'", "'I'", "'U2'", "'RPhi'", "'U3'", "'U4'", "'RX'", 
  "'RY'", "'RZ'", "'U1'", "'P'", "'CNOT'", "'CZ'", "'CU'", "'ISWAP'", "'SQISWAP'", 
  "'SWAP'", "'ISWAPTHETA'", "'CR'", "'RXX'", "'RYY'", "'RZZ'", "'RZX'", 
  "'TOFFOLI'", "'DAGGER'", "'ENDDAGGER'", "'CONTROL'", "'ENDCONTROL'", "'QIF'", 
  "'ELSE'", "'ENDQIF'", "'QWHILE'", "'ENDQWHILE'", "'MEASURE'", "'RESET'", 
  "'='", "'>'", "'<'", "'!'", "'=='", "'<='", "'>='", "'!='", "'&&'", "'||'", 
  "'+'", "'-'", "'*'", "'/'", "','", "'('", "')'", "'['", "']'"
};

std::vector<std::string> originirParser::_symbolicNames = {
  "", "PI", "QINIT_KEY", "CREG_KEY", "Q_KEY", "C_KEY", "BARRIER_KEY", "QGATE_KEY", 
  "ENDQGATE_KEY", "ECHO_GATE", "H_GATE", "X_GATE", "NOT_GATE", "T_GATE", 
  "S_GATE", "Y_GATE", "Z_GATE", "X1_GATE", "Y1_GATE", "Z1_GATE", "I_GATE", 
  "U2_GATE", "RPHI_GATE", "U3_GATE", "U4_GATE", "RX_GATE", "RY_GATE", "RZ_GATE", 
  "U1_GATE", "P_GATE", "CNOT_GATE", "CZ_GATE", "CU_GATE", "ISWAP_GATE", 
  "SQISWAP_GATE", "SWAPZ1_GATE", "ISWAPTHETA_GATE", "CR_GATE", "RXX_GATE", 
  "RYY_GATE", "RZZ_GATE", "RZX_GATE", "TOFFOLI_GATE", "DAGGER_KEY", "ENDDAGGER_KEY", 
  "CONTROL_KEY", "ENDCONTROL_KEY", "QIF_KEY", "ELSE_KEY", "ENDIF_KEY", "QWHILE_KEY", 
  "ENDQWHILE_KEY", "MEASURE_KEY", "RESET_KEY", "ASSIGN", "GT", "LT", "NOT", 
  "EQ", "LEQ", "GEQ", "NE", "AND", "OR", "PLUS", "MINUS", "MUL", "DIV", 
  "COMMA", "LPAREN", "RPAREN", "LBRACK", "RBRACK", "NEWLINE", "Identifier", 
  "Integer_Literal", "Double_Literal", "Digit_Sequence", "REALEXP", "WhiteSpace", 
  "SingleLineComment"
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
    0x3, 0x52, 0x2e3, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x9, 0x2f, 0x4, 0x30, 0x9, 0x30, 0x4, 0x31, 0x9, 0x31, 0x4, 0x32, 0x9, 
    0x32, 0x4, 0x33, 0x9, 0x33, 0x4, 0x34, 0x9, 0x34, 0x4, 0x35, 0x9, 0x35, 
    0x4, 0x36, 0x9, 0x36, 0x4, 0x37, 0x9, 0x37, 0x4, 0x38, 0x9, 0x38, 0x4, 
    0x39, 0x9, 0x39, 0x4, 0x3a, 0x9, 0x3a, 0x4, 0x3b, 0x9, 0x3b, 0x4, 0x3c, 
    0x9, 0x3c, 0x3, 0x2, 0x7, 0x2, 0x7a, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x7d, 
    0xb, 0x2, 0x3, 0x2, 0x7, 0x2, 0x80, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x83, 
    0xb, 0x2, 0x3, 0x2, 0x7, 0x2, 0x86, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x89, 
    0xb, 0x2, 0x3, 0x2, 0x6, 0x2, 0x8c, 0xa, 0x2, 0xd, 0x2, 0xe, 0x2, 0x8d, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 
    0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
    0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 
    0x5, 0x6, 0xa5, 0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0xb7, 0xa, 0xa, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x5, 0xb, 0xc7, 0xa, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 
    0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x5, 
    0xc, 0xdb, 0xa, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0xf3, 0xa, 0xd, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 
    0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x5, 0xe, 0x10f, 0xa, 0xe, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x7, 
    0x13, 0x139, 0xa, 0x13, 0xc, 0x13, 0xe, 0x13, 0x13c, 0xb, 0x13, 0x3, 
    0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x7, 0x13, 0x142, 0xa, 0x13, 
    0xc, 0x13, 0xe, 0x13, 0x145, 0xb, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x13, 0x7, 0x13, 0x14c, 0xa, 0x13, 0xc, 0x13, 0xe, 0x13, 
    0x14f, 0xb, 0x13, 0x3, 0x13, 0x3, 0x13, 0x5, 0x13, 0x153, 0xa, 0x13, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 
    0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x1a, 
    0x3, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1d, 0x3, 
    0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x5, 0x1d, 0x16d, 
    0xa, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 
    0x1e, 0x3, 0x1e, 0x5, 0x1e, 0x176, 0xa, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 
    0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 
    0x1f, 0x7, 0x1f, 0x181, 0xa, 0x1f, 0xc, 0x1f, 0xe, 0x1f, 0x184, 0xb, 
    0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
    0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x7, 0x20, 0x18f, 0xa, 0x20, 0xc, 0x20, 
    0xe, 0x20, 0x192, 0xb, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 
    0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 
    0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x7, 0x21, 0x1a3, 
    0xa, 0x21, 0xc, 0x21, 0xe, 0x21, 0x1a6, 0xb, 0x21, 0x3, 0x22, 0x3, 0x22, 
    0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x3, 
    0x22, 0x7, 0x22, 0x1b1, 0xa, 0x22, 0xc, 0x22, 0xe, 0x22, 0x1b4, 0xb, 
    0x22, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x7, 0x23, 0x1bc, 0xa, 0x23, 0xc, 0x23, 0xe, 0x23, 0x1bf, 0xb, 0x23, 
    0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x7, 
    0x24, 0x1c7, 0xa, 0x24, 0xc, 0x24, 0xe, 0x24, 0x1ca, 0xb, 0x24, 0x3, 
    0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x5, 0x25, 0x1d1, 
    0xa, 0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 
    0x27, 0x1d8, 0xa, 0x27, 0xc, 0x27, 0xe, 0x27, 0x1db, 0xb, 0x27, 0x3, 
    0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 0x1e0, 0xa, 0x27, 0xc, 0x27, 
    0xe, 0x27, 0x1e3, 0xb, 0x27, 0x5, 0x27, 0x1e5, 0xa, 0x27, 0x3, 0x28, 
    0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 
    0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 0x1f3, 
    0xa, 0x28, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x7, 0x29, 0x1f8, 0xa, 0x29, 
    0xc, 0x29, 0xe, 0x29, 0x1fb, 0xb, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 
    0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x7, 0x2a, 0x204, 0xa, 0x2a, 
    0xc, 0x2a, 0xe, 0x2a, 0x207, 0xb, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 
    0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x7, 0x2b, 0x20f, 0xa, 0x2b, 0xc, 0x2b, 
    0xe, 0x2b, 0x212, 0xb, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 
    0x7, 0x2c, 0x218, 0xa, 0x2c, 0xc, 0x2c, 0xe, 0x2c, 0x21b, 0xb, 0x2c, 
    0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 
    0x2c, 0x3, 0x2c, 0x7, 0x2c, 0x225, 0xa, 0x2c, 0xc, 0x2c, 0xe, 0x2c, 
    0x228, 0xb, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x5, 0x2c, 0x22d, 
    0xa, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x7, 0x2d, 0x233, 
    0xa, 0x2d, 0xc, 0x2d, 0xe, 0x2d, 0x236, 0xb, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 
    0x3, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 
    0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x3, 0x2e, 0x5, 0x2e, 
    0x246, 0xa, 0x2e, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 
    0x3, 0x2f, 0x3, 0x2f, 0x5, 0x2f, 0x24f, 0xa, 0x2f, 0x3, 0x30, 0x3, 0x30, 
    0x3, 0x30, 0x3, 0x30, 0x3, 0x30, 0x3, 0x30, 0x3, 0x30, 0x5, 0x30, 0x258, 
    0xa, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x3, 0x32, 0x3, 0x32, 0x3, 
    0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 
    0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x5, 0x32, 0x269, 0xa, 0x32, 0x3, 0x33, 
    0x3, 0x33, 0x3, 0x33, 0x6, 0x33, 0x26e, 0xa, 0x33, 0xd, 0x33, 0xe, 0x33, 
    0x26f, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 
    0x3, 0x34, 0x6, 0x34, 0x279, 0xa, 0x34, 0xd, 0x34, 0xe, 0x34, 0x27a, 
    0x3, 0x34, 0x3, 0x34, 0x3, 0x34, 0x3, 0x35, 0x3, 0x35, 0x3, 0x35, 0x5, 
    0x35, 0x283, 0xa, 0x35, 0x3, 0x36, 0x3, 0x36, 0x3, 0x36, 0x7, 0x36, 
    0x288, 0xa, 0x36, 0xc, 0x36, 0xe, 0x36, 0x28b, 0xb, 0x36, 0x3, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x5, 0x37, 0x298, 0xa, 0x37, 
    0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 
    0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x3, 0x37, 0x7, 0x37, 
    0x2a6, 0xa, 0x37, 0xc, 0x37, 0xe, 0x37, 0x2a9, 0xb, 0x37, 0x3, 0x38, 
    0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x7, 0x38, 0x2b0, 0xa, 0x38, 
    0xc, 0x38, 0xe, 0x38, 0x2b3, 0xb, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 
    0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 
    0x38, 0x3, 0x38, 0x3, 0x38, 0x7, 0x38, 0x2c1, 0xa, 0x38, 0xc, 0x38, 
    0xe, 0x38, 0x2c4, 0xb, 0x38, 0x3, 0x38, 0x3, 0x38, 0x3, 0x38, 0x5, 0x38, 
    0x2c9, 0xa, 0x38, 0x3, 0x39, 0x3, 0x39, 0x3, 0x3a, 0x3, 0x3a, 0x3, 0x3a, 
    0x7, 0x3a, 0x2d0, 0xa, 0x3a, 0xc, 0x3a, 0xe, 0x3a, 0x2d3, 0xb, 0x3a, 
    0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 
    0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x3, 0x3b, 0x5, 0x3b, 0x2df, 0xa, 0x3b, 
    0x3, 0x3c, 0x3, 0x3c, 0x3, 0x3c, 0x2, 0x9, 0x3c, 0x3e, 0x40, 0x42, 0x44, 
    0x46, 0x6c, 0x3d, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 
    0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 
    0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 
    0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 
    0x5e, 0x60, 0x62, 0x64, 0x66, 0x68, 0x6a, 0x6c, 0x6e, 0x70, 0x72, 0x74, 
    0x76, 0x2, 0x8, 0x4, 0x2, 0xb, 0xd, 0xf, 0x16, 0x3, 0x2, 0x1b, 0x1f, 
    0x3, 0x2, 0x17, 0x18, 0x4, 0x2, 0x20, 0x21, 0x23, 0x25, 0x3, 0x2, 0x26, 
    0x2b, 0x4, 0x2, 0x3, 0x3, 0x4d, 0x4e, 0x2, 0x301, 0x2, 0x7b, 0x3, 0x2, 
    0x2, 0x2, 0x4, 0x8f, 0x3, 0x2, 0x2, 0x2, 0x6, 0x92, 0x3, 0x2, 0x2, 0x2, 
    0x8, 0x96, 0x3, 0x2, 0x2, 0x2, 0xa, 0xa4, 0x3, 0x2, 0x2, 0x2, 0xc, 0xa6, 
    0x3, 0x2, 0x2, 0x2, 0xe, 0xaa, 0x3, 0x2, 0x2, 0x2, 0x10, 0xad, 0x3, 
    0x2, 0x2, 0x2, 0x12, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x14, 0xc6, 0x3, 0x2, 
    0x2, 0x2, 0x16, 0xda, 0x3, 0x2, 0x2, 0x2, 0x18, 0xf2, 0x3, 0x2, 0x2, 
    0x2, 0x1a, 0x10e, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x110, 0x3, 0x2, 0x2, 0x2, 
    0x1e, 0x115, 0x3, 0x2, 0x2, 0x2, 0x20, 0x11e, 0x3, 0x2, 0x2, 0x2, 0x22, 
    0x12d, 0x3, 0x2, 0x2, 0x2, 0x24, 0x152, 0x3, 0x2, 0x2, 0x2, 0x26, 0x154, 
    0x3, 0x2, 0x2, 0x2, 0x28, 0x156, 0x3, 0x2, 0x2, 0x2, 0x2a, 0x158, 0x3, 
    0x2, 0x2, 0x2, 0x2c, 0x15a, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x15c, 0x3, 0x2, 
    0x2, 0x2, 0x30, 0x15e, 0x3, 0x2, 0x2, 0x2, 0x32, 0x160, 0x3, 0x2, 0x2, 
    0x2, 0x34, 0x162, 0x3, 0x2, 0x2, 0x2, 0x36, 0x164, 0x3, 0x2, 0x2, 0x2, 
    0x38, 0x16c, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x175, 0x3, 0x2, 0x2, 0x2, 0x3c, 
    0x177, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x185, 0x3, 0x2, 0x2, 0x2, 0x40, 0x193, 
    0x3, 0x2, 0x2, 0x2, 0x42, 0x1a7, 0x3, 0x2, 0x2, 0x2, 0x44, 0x1b5, 0x3, 
    0x2, 0x2, 0x2, 0x46, 0x1c0, 0x3, 0x2, 0x2, 0x2, 0x48, 0x1d0, 0x3, 0x2, 
    0x2, 0x2, 0x4a, 0x1d2, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x1e4, 0x3, 0x2, 0x2, 
    0x2, 0x4e, 0x1f2, 0x3, 0x2, 0x2, 0x2, 0x50, 0x1f4, 0x3, 0x2, 0x2, 0x2, 
    0x52, 0x1ff, 0x3, 0x2, 0x2, 0x2, 0x54, 0x20b, 0x3, 0x2, 0x2, 0x2, 0x56, 
    0x22c, 0x3, 0x2, 0x2, 0x2, 0x58, 0x22e, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x245, 
    0x3, 0x2, 0x2, 0x2, 0x5c, 0x24e, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x257, 0x3, 
    0x2, 0x2, 0x2, 0x60, 0x259, 0x3, 0x2, 0x2, 0x2, 0x62, 0x268, 0x3, 0x2, 
    0x2, 0x2, 0x64, 0x26a, 0x3, 0x2, 0x2, 0x2, 0x66, 0x274, 0x3, 0x2, 0x2, 
    0x2, 0x68, 0x282, 0x3, 0x2, 0x2, 0x2, 0x6a, 0x284, 0x3, 0x2, 0x2, 0x2, 
    0x6c, 0x297, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x2c8, 0x3, 0x2, 0x2, 0x2, 0x70, 
    0x2ca, 0x3, 0x2, 0x2, 0x2, 0x72, 0x2cc, 0x3, 0x2, 0x2, 0x2, 0x74, 0x2de, 
    0x3, 0x2, 0x2, 0x2, 0x76, 0x2e0, 0x3, 0x2, 0x2, 0x2, 0x78, 0x7a, 0x7, 
    0x4b, 0x2, 0x2, 0x79, 0x78, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x7d, 0x3, 0x2, 
    0x2, 0x2, 0x7b, 0x79, 0x3, 0x2, 0x2, 0x2, 0x7b, 0x7c, 0x3, 0x2, 0x2, 
    0x2, 0x7c, 0x81, 0x3, 0x2, 0x2, 0x2, 0x7d, 0x7b, 0x3, 0x2, 0x2, 0x2, 
    0x7e, 0x80, 0x5, 0x6e, 0x38, 0x2, 0x7f, 0x7e, 0x3, 0x2, 0x2, 0x2, 0x80, 
    0x83, 0x3, 0x2, 0x2, 0x2, 0x81, 0x7f, 0x3, 0x2, 0x2, 0x2, 0x81, 0x82, 
    0x3, 0x2, 0x2, 0x2, 0x82, 0x87, 0x3, 0x2, 0x2, 0x2, 0x83, 0x81, 0x3, 
    0x2, 0x2, 0x2, 0x84, 0x86, 0x5, 0x4, 0x3, 0x2, 0x85, 0x84, 0x3, 0x2, 
    0x2, 0x2, 0x86, 0x89, 0x3, 0x2, 0x2, 0x2, 0x87, 0x85, 0x3, 0x2, 0x2, 
    0x2, 0x87, 0x88, 0x3, 0x2, 0x2, 0x2, 0x88, 0x8b, 0x3, 0x2, 0x2, 0x2, 
    0x89, 0x87, 0x3, 0x2, 0x2, 0x2, 0x8a, 0x8c, 0x5, 0x4e, 0x28, 0x2, 0x8b, 
    0x8a, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x8b, 
    0x3, 0x2, 0x2, 0x2, 0x8d, 0x8e, 0x3, 0x2, 0x2, 0x2, 0x8e, 0x3, 0x3, 
    0x2, 0x2, 0x2, 0x8f, 0x90, 0x5, 0x6, 0x4, 0x2, 0x90, 0x91, 0x5, 0x8, 
    0x5, 0x2, 0x91, 0x5, 0x3, 0x2, 0x2, 0x2, 0x92, 0x93, 0x7, 0x4, 0x2, 
    0x2, 0x93, 0x94, 0x7, 0x4d, 0x2, 0x2, 0x94, 0x95, 0x7, 0x4b, 0x2, 0x2, 
    0x95, 0x7, 0x3, 0x2, 0x2, 0x2, 0x96, 0x97, 0x7, 0x5, 0x2, 0x2, 0x97, 
    0x98, 0x7, 0x4d, 0x2, 0x2, 0x98, 0x99, 0x7, 0x4b, 0x2, 0x2, 0x99, 0x9, 
    0x3, 0x2, 0x2, 0x2, 0x9a, 0xa5, 0x5, 0x12, 0xa, 0x2, 0x9b, 0xa5, 0x5, 
    0x14, 0xb, 0x2, 0x9c, 0xa5, 0x5, 0x16, 0xc, 0x2, 0x9d, 0xa5, 0x5, 0x18, 
    0xd, 0x2, 0x9e, 0xa5, 0x5, 0x1a, 0xe, 0x2, 0x9f, 0xa5, 0x5, 0x1c, 0xf, 
    0x2, 0xa0, 0xa5, 0x5, 0x1e, 0x10, 0x2, 0xa1, 0xa5, 0x5, 0x20, 0x11, 
    0x2, 0xa2, 0xa5, 0x5, 0x22, 0x12, 0x2, 0xa3, 0xa5, 0x5, 0x24, 0x13, 
    0x2, 0xa4, 0x9a, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x9b, 0x3, 0x2, 0x2, 0x2, 
    0xa4, 0x9c, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x9d, 0x3, 0x2, 0x2, 0x2, 0xa4, 
    0x9e, 0x3, 0x2, 0x2, 0x2, 0xa4, 0x9f, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa0, 
    0x3, 0x2, 0x2, 0x2, 0xa4, 0xa1, 0x3, 0x2, 0x2, 0x2, 0xa4, 0xa2, 0x3, 
    0x2, 0x2, 0x2, 0xa4, 0xa3, 0x3, 0x2, 0x2, 0x2, 0xa5, 0xb, 0x3, 0x2, 
    0x2, 0x2, 0xa6, 0xa7, 0x7, 0x49, 0x2, 0x2, 0xa7, 0xa8, 0x5, 0x4a, 0x26, 
    0x2, 0xa8, 0xa9, 0x7, 0x4a, 0x2, 0x2, 0xa9, 0xd, 0x3, 0x2, 0x2, 0x2, 
    0xaa, 0xab, 0x7, 0x7, 0x2, 0x2, 0xab, 0xac, 0x5, 0xc, 0x7, 0x2, 0xac, 
    0xf, 0x3, 0x2, 0x2, 0x2, 0xad, 0xae, 0x7, 0x6, 0x2, 0x2, 0xae, 0xaf, 
    0x5, 0xc, 0x7, 0x2, 0xaf, 0x11, 0x3, 0x2, 0x2, 0x2, 0xb0, 0xb1, 0x5, 
    0x26, 0x14, 0x2, 0xb1, 0xb2, 0x5, 0x10, 0x9, 0x2, 0xb2, 0xb7, 0x3, 0x2, 
    0x2, 0x2, 0xb3, 0xb4, 0x5, 0x26, 0x14, 0x2, 0xb4, 0xb5, 0x7, 0x6, 0x2, 
    0x2, 0xb5, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xb6, 0xb0, 0x3, 0x2, 0x2, 0x2, 
    0xb6, 0xb3, 0x3, 0x2, 0x2, 0x2, 0xb7, 0x13, 0x3, 0x2, 0x2, 0x2, 0xb8, 
    0xb9, 0x5, 0x28, 0x15, 0x2, 0xb9, 0xba, 0x5, 0x10, 0x9, 0x2, 0xba, 0xbb, 
    0x7, 0x46, 0x2, 0x2, 0xbb, 0xbc, 0x7, 0x47, 0x2, 0x2, 0xbc, 0xbd, 0x5, 
    0x4a, 0x26, 0x2, 0xbd, 0xbe, 0x7, 0x48, 0x2, 0x2, 0xbe, 0xc7, 0x3, 0x2, 
    0x2, 0x2, 0xbf, 0xc0, 0x5, 0x28, 0x15, 0x2, 0xc0, 0xc1, 0x7, 0x6, 0x2, 
    0x2, 0xc1, 0xc2, 0x7, 0x46, 0x2, 0x2, 0xc2, 0xc3, 0x7, 0x47, 0x2, 0x2, 
    0xc3, 0xc4, 0x5, 0x4a, 0x26, 0x2, 0xc4, 0xc5, 0x7, 0x48, 0x2, 0x2, 0xc5, 
    0xc7, 0x3, 0x2, 0x2, 0x2, 0xc6, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xc6, 0xbf, 
    0x3, 0x2, 0x2, 0x2, 0xc7, 0x15, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xc9, 0x5, 
    0x2a, 0x16, 0x2, 0xc9, 0xca, 0x5, 0x10, 0x9, 0x2, 0xca, 0xcb, 0x7, 0x46, 
    0x2, 0x2, 0xcb, 0xcc, 0x7, 0x47, 0x2, 0x2, 0xcc, 0xcd, 0x5, 0x4a, 0x26, 
    0x2, 0xcd, 0xce, 0x7, 0x46, 0x2, 0x2, 0xce, 0xcf, 0x5, 0x4a, 0x26, 0x2, 
    0xcf, 0xd0, 0x7, 0x48, 0x2, 0x2, 0xd0, 0xdb, 0x3, 0x2, 0x2, 0x2, 0xd1, 
    0xd2, 0x5, 0x2a, 0x16, 0x2, 0xd2, 0xd3, 0x7, 0x6, 0x2, 0x2, 0xd3, 0xd4, 
    0x7, 0x46, 0x2, 0x2, 0xd4, 0xd5, 0x7, 0x47, 0x2, 0x2, 0xd5, 0xd6, 0x5, 
    0x4a, 0x26, 0x2, 0xd6, 0xd7, 0x7, 0x46, 0x2, 0x2, 0xd7, 0xd8, 0x5, 0x4a, 
    0x26, 0x2, 0xd8, 0xd9, 0x7, 0x48, 0x2, 0x2, 0xd9, 0xdb, 0x3, 0x2, 0x2, 
    0x2, 0xda, 0xc8, 0x3, 0x2, 0x2, 0x2, 0xda, 0xd1, 0x3, 0x2, 0x2, 0x2, 
    0xdb, 0x17, 0x3, 0x2, 0x2, 0x2, 0xdc, 0xdd, 0x5, 0x2c, 0x17, 0x2, 0xdd, 
    0xde, 0x5, 0x10, 0x9, 0x2, 0xde, 0xdf, 0x7, 0x46, 0x2, 0x2, 0xdf, 0xe0, 
    0x7, 0x47, 0x2, 0x2, 0xe0, 0xe1, 0x5, 0x4a, 0x26, 0x2, 0xe1, 0xe2, 0x7, 
    0x46, 0x2, 0x2, 0xe2, 0xe3, 0x5, 0x4a, 0x26, 0x2, 0xe3, 0xe4, 0x7, 0x46, 
    0x2, 0x2, 0xe4, 0xe5, 0x5, 0x4a, 0x26, 0x2, 0xe5, 0xe6, 0x7, 0x48, 0x2, 
    0x2, 0xe6, 0xf3, 0x3, 0x2, 0x2, 0x2, 0xe7, 0xe8, 0x5, 0x2c, 0x17, 0x2, 
    0xe8, 0xe9, 0x7, 0x6, 0x2, 0x2, 0xe9, 0xea, 0x7, 0x46, 0x2, 0x2, 0xea, 
    0xeb, 0x7, 0x47, 0x2, 0x2, 0xeb, 0xec, 0x5, 0x4a, 0x26, 0x2, 0xec, 0xed, 
    0x7, 0x46, 0x2, 0x2, 0xed, 0xee, 0x5, 0x4a, 0x26, 0x2, 0xee, 0xef, 0x7, 
    0x46, 0x2, 0x2, 0xef, 0xf0, 0x5, 0x4a, 0x26, 0x2, 0xf0, 0xf1, 0x7, 0x48, 
    0x2, 0x2, 0xf1, 0xf3, 0x3, 0x2, 0x2, 0x2, 0xf2, 0xdc, 0x3, 0x2, 0x2, 
    0x2, 0xf2, 0xe7, 0x3, 0x2, 0x2, 0x2, 0xf3, 0x19, 0x3, 0x2, 0x2, 0x2, 
    0xf4, 0xf5, 0x5, 0x2e, 0x18, 0x2, 0xf5, 0xf6, 0x5, 0x10, 0x9, 0x2, 0xf6, 
    0xf7, 0x7, 0x46, 0x2, 0x2, 0xf7, 0xf8, 0x7, 0x47, 0x2, 0x2, 0xf8, 0xf9, 
    0x5, 0x4a, 0x26, 0x2, 0xf9, 0xfa, 0x7, 0x46, 0x2, 0x2, 0xfa, 0xfb, 0x5, 
    0x4a, 0x26, 0x2, 0xfb, 0xfc, 0x7, 0x46, 0x2, 0x2, 0xfc, 0xfd, 0x5, 0x4a, 
    0x26, 0x2, 0xfd, 0xfe, 0x7, 0x46, 0x2, 0x2, 0xfe, 0xff, 0x5, 0x4a, 0x26, 
    0x2, 0xff, 0x100, 0x7, 0x48, 0x2, 0x2, 0x100, 0x10f, 0x3, 0x2, 0x2, 
    0x2, 0x101, 0x102, 0x5, 0x2e, 0x18, 0x2, 0x102, 0x103, 0x7, 0x6, 0x2, 
    0x2, 0x103, 0x104, 0x7, 0x46, 0x2, 0x2, 0x104, 0x105, 0x7, 0x47, 0x2, 
    0x2, 0x105, 0x106, 0x5, 0x4a, 0x26, 0x2, 0x106, 0x107, 0x7, 0x46, 0x2, 
    0x2, 0x107, 0x108, 0x5, 0x4a, 0x26, 0x2, 0x108, 0x109, 0x7, 0x46, 0x2, 
    0x2, 0x109, 0x10a, 0x5, 0x4a, 0x26, 0x2, 0x10a, 0x10b, 0x7, 0x46, 0x2, 
    0x2, 0x10b, 0x10c, 0x5, 0x4a, 0x26, 0x2, 0x10c, 0x10d, 0x7, 0x48, 0x2, 
    0x2, 0x10d, 0x10f, 0x3, 0x2, 0x2, 0x2, 0x10e, 0xf4, 0x3, 0x2, 0x2, 0x2, 
    0x10e, 0x101, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x1b, 0x3, 0x2, 0x2, 0x2, 0x110, 
    0x111, 0x5, 0x30, 0x19, 0x2, 0x111, 0x112, 0x5, 0x10, 0x9, 0x2, 0x112, 
    0x113, 0x7, 0x46, 0x2, 0x2, 0x113, 0x114, 0x5, 0x10, 0x9, 0x2, 0x114, 
    0x1d, 0x3, 0x2, 0x2, 0x2, 0x115, 0x116, 0x5, 0x32, 0x1a, 0x2, 0x116, 
    0x117, 0x5, 0x10, 0x9, 0x2, 0x117, 0x118, 0x7, 0x46, 0x2, 0x2, 0x118, 
    0x119, 0x5, 0x10, 0x9, 0x2, 0x119, 0x11a, 0x7, 0x46, 0x2, 0x2, 0x11a, 
    0x11b, 0x7, 0x47, 0x2, 0x2, 0x11b, 0x11c, 0x5, 0x4a, 0x26, 0x2, 0x11c, 
    0x11d, 0x7, 0x48, 0x2, 0x2, 0x11d, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x11e, 
    0x11f, 0x5, 0x34, 0x1b, 0x2, 0x11f, 0x120, 0x5, 0x10, 0x9, 0x2, 0x120, 
    0x121, 0x7, 0x46, 0x2, 0x2, 0x121, 0x122, 0x5, 0x10, 0x9, 0x2, 0x122, 
    0x123, 0x7, 0x46, 0x2, 0x2, 0x123, 0x124, 0x7, 0x47, 0x2, 0x2, 0x124, 
    0x125, 0x5, 0x4a, 0x26, 0x2, 0x125, 0x126, 0x7, 0x46, 0x2, 0x2, 0x126, 
    0x127, 0x5, 0x4a, 0x26, 0x2, 0x127, 0x128, 0x7, 0x46, 0x2, 0x2, 0x128, 
    0x129, 0x5, 0x4a, 0x26, 0x2, 0x129, 0x12a, 0x7, 0x46, 0x2, 0x2, 0x12a, 
    0x12b, 0x5, 0x4a, 0x26, 0x2, 0x12b, 0x12c, 0x7, 0x48, 0x2, 0x2, 0x12c, 
    0x21, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x12e, 0x5, 0x36, 0x1c, 0x2, 0x12e, 
    0x12f, 0x5, 0x10, 0x9, 0x2, 0x12f, 0x130, 0x7, 0x46, 0x2, 0x2, 0x130, 
    0x131, 0x5, 0x10, 0x9, 0x2, 0x131, 0x132, 0x7, 0x46, 0x2, 0x2, 0x132, 
    0x133, 0x5, 0x10, 0x9, 0x2, 0x133, 0x23, 0x3, 0x2, 0x2, 0x2, 0x134, 
    0x135, 0x5, 0x70, 0x39, 0x2, 0x135, 0x13a, 0x5, 0x10, 0x9, 0x2, 0x136, 
    0x137, 0x7, 0x46, 0x2, 0x2, 0x137, 0x139, 0x5, 0x10, 0x9, 0x2, 0x138, 
    0x136, 0x3, 0x2, 0x2, 0x2, 0x139, 0x13c, 0x3, 0x2, 0x2, 0x2, 0x13a, 
    0x138, 0x3, 0x2, 0x2, 0x2, 0x13a, 0x13b, 0x3, 0x2, 0x2, 0x2, 0x13b, 
    0x153, 0x3, 0x2, 0x2, 0x2, 0x13c, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x13d, 
    0x13e, 0x5, 0x70, 0x39, 0x2, 0x13e, 0x143, 0x5, 0x10, 0x9, 0x2, 0x13f, 
    0x140, 0x7, 0x46, 0x2, 0x2, 0x140, 0x142, 0x5, 0x10, 0x9, 0x2, 0x141, 
    0x13f, 0x3, 0x2, 0x2, 0x2, 0x142, 0x145, 0x3, 0x2, 0x2, 0x2, 0x143, 
    0x141, 0x3, 0x2, 0x2, 0x2, 0x143, 0x144, 0x3, 0x2, 0x2, 0x2, 0x144, 
    0x146, 0x3, 0x2, 0x2, 0x2, 0x145, 0x143, 0x3, 0x2, 0x2, 0x2, 0x146, 
    0x147, 0x7, 0x46, 0x2, 0x2, 0x147, 0x148, 0x7, 0x47, 0x2, 0x2, 0x148, 
    0x14d, 0x5, 0x4a, 0x26, 0x2, 0x149, 0x14a, 0x7, 0x46, 0x2, 0x2, 0x14a, 
    0x14c, 0x5, 0x4a, 0x26, 0x2, 0x14b, 0x149, 0x3, 0x2, 0x2, 0x2, 0x14c, 
    0x14f, 0x3, 0x2, 0x2, 0x2, 0x14d, 0x14b, 0x3, 0x2, 0x2, 0x2, 0x14d, 
    0x14e, 0x3, 0x2, 0x2, 0x2, 0x14e, 0x150, 0x3, 0x2, 0x2, 0x2, 0x14f, 
    0x14d, 0x3, 0x2, 0x2, 0x2, 0x150, 0x151, 0x7, 0x48, 0x2, 0x2, 0x151, 
    0x153, 0x3, 0x2, 0x2, 0x2, 0x152, 0x134, 0x3, 0x2, 0x2, 0x2, 0x152, 
    0x13d, 0x3, 0x2, 0x2, 0x2, 0x153, 0x25, 0x3, 0x2, 0x2, 0x2, 0x154, 0x155, 
    0x9, 0x2, 0x2, 0x2, 0x155, 0x27, 0x3, 0x2, 0x2, 0x2, 0x156, 0x157, 0x9, 
    0x3, 0x2, 0x2, 0x157, 0x29, 0x3, 0x2, 0x2, 0x2, 0x158, 0x159, 0x9, 0x4, 
    0x2, 0x2, 0x159, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x15a, 0x15b, 0x7, 0x19, 
    0x2, 0x2, 0x15b, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x15c, 0x15d, 0x7, 0x1a, 
    0x2, 0x2, 0x15d, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x15e, 0x15f, 0x9, 0x5, 0x2, 
    0x2, 0x15f, 0x31, 0x3, 0x2, 0x2, 0x2, 0x160, 0x161, 0x9, 0x6, 0x2, 0x2, 
    0x161, 0x33, 0x3, 0x2, 0x2, 0x2, 0x162, 0x163, 0x7, 0x22, 0x2, 0x2, 
    0x163, 0x35, 0x3, 0x2, 0x2, 0x2, 0x164, 0x165, 0x7, 0x2c, 0x2, 0x2, 
    0x165, 0x37, 0x3, 0x2, 0x2, 0x2, 0x166, 0x16d, 0x5, 0xe, 0x8, 0x2, 0x167, 
    0x16d, 0x5, 0x76, 0x3c, 0x2, 0x168, 0x169, 0x7, 0x47, 0x2, 0x2, 0x169, 
    0x16a, 0x5, 0x4a, 0x26, 0x2, 0x16a, 0x16b, 0x7, 0x47, 0x2, 0x2, 0x16b, 
    0x16d, 0x3, 0x2, 0x2, 0x2, 0x16c, 0x166, 0x3, 0x2, 0x2, 0x2, 0x16c, 
    0x167, 0x3, 0x2, 0x2, 0x2, 0x16c, 0x168, 0x3, 0x2, 0x2, 0x2, 0x16d, 
    0x39, 0x3, 0x2, 0x2, 0x2, 0x16e, 0x176, 0x5, 0x38, 0x1d, 0x2, 0x16f, 
    0x170, 0x7, 0x42, 0x2, 0x2, 0x170, 0x176, 0x5, 0x38, 0x1d, 0x2, 0x171, 
    0x172, 0x7, 0x43, 0x2, 0x2, 0x172, 0x176, 0x5, 0x38, 0x1d, 0x2, 0x173, 
    0x174, 0x7, 0x3b, 0x2, 0x2, 0x174, 0x176, 0x5, 0x38, 0x1d, 0x2, 0x175, 
    0x16e, 0x3, 0x2, 0x2, 0x2, 0x175, 0x16f, 0x3, 0x2, 0x2, 0x2, 0x175, 
    0x171, 0x3, 0x2, 0x2, 0x2, 0x175, 0x173, 0x3, 0x2, 0x2, 0x2, 0x176, 
    0x3b, 0x3, 0x2, 0x2, 0x2, 0x177, 0x178, 0x8, 0x1f, 0x1, 0x2, 0x178, 
    0x179, 0x5, 0x3a, 0x1e, 0x2, 0x179, 0x182, 0x3, 0x2, 0x2, 0x2, 0x17a, 
    0x17b, 0xc, 0x4, 0x2, 0x2, 0x17b, 0x17c, 0x7, 0x44, 0x2, 0x2, 0x17c, 
    0x181, 0x5, 0x3a, 0x1e, 0x2, 0x17d, 0x17e, 0xc, 0x3, 0x2, 0x2, 0x17e, 
    0x17f, 0x7, 0x45, 0x2, 0x2, 0x17f, 0x181, 0x5, 0x3a, 0x1e, 0x2, 0x180, 
    0x17a, 0x3, 0x2, 0x2, 0x2, 0x180, 0x17d, 0x3, 0x2, 0x2, 0x2, 0x181, 
    0x184, 0x3, 0x2, 0x2, 0x2, 0x182, 0x180, 0x3, 0x2, 0x2, 0x2, 0x182, 
    0x183, 0x3, 0x2, 0x2, 0x2, 0x183, 0x3d, 0x3, 0x2, 0x2, 0x2, 0x184, 0x182, 
    0x3, 0x2, 0x2, 0x2, 0x185, 0x186, 0x8, 0x20, 0x1, 0x2, 0x186, 0x187, 
    0x5, 0x3c, 0x1f, 0x2, 0x187, 0x190, 0x3, 0x2, 0x2, 0x2, 0x188, 0x189, 
    0xc, 0x4, 0x2, 0x2, 0x189, 0x18a, 0x7, 0x42, 0x2, 0x2, 0x18a, 0x18f, 
    0x5, 0x3c, 0x1f, 0x2, 0x18b, 0x18c, 0xc, 0x3, 0x2, 0x2, 0x18c, 0x18d, 
    0x7, 0x43, 0x2, 0x2, 0x18d, 0x18f, 0x5, 0x3c, 0x1f, 0x2, 0x18e, 0x188, 
    0x3, 0x2, 0x2, 0x2, 0x18e, 0x18b, 0x3, 0x2, 0x2, 0x2, 0x18f, 0x192, 
    0x3, 0x2, 0x2, 0x2, 0x190, 0x18e, 0x3, 0x2, 0x2, 0x2, 0x190, 0x191, 
    0x3, 0x2, 0x2, 0x2, 0x191, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x192, 0x190, 0x3, 
    0x2, 0x2, 0x2, 0x193, 0x194, 0x8, 0x21, 0x1, 0x2, 0x194, 0x195, 0x5, 
    0x3e, 0x20, 0x2, 0x195, 0x1a4, 0x3, 0x2, 0x2, 0x2, 0x196, 0x197, 0xc, 
    0x6, 0x2, 0x2, 0x197, 0x198, 0x7, 0x3a, 0x2, 0x2, 0x198, 0x1a3, 0x5, 
    0x3e, 0x20, 0x2, 0x199, 0x19a, 0xc, 0x5, 0x2, 0x2, 0x19a, 0x19b, 0x7, 
    0x39, 0x2, 0x2, 0x19b, 0x1a3, 0x5, 0x3e, 0x20, 0x2, 0x19c, 0x19d, 0xc, 
    0x4, 0x2, 0x2, 0x19d, 0x19e, 0x7, 0x3d, 0x2, 0x2, 0x19e, 0x1a3, 0x5, 
    0x3e, 0x20, 0x2, 0x19f, 0x1a0, 0xc, 0x3, 0x2, 0x2, 0x1a0, 0x1a1, 0x7, 
    0x3e, 0x2, 0x2, 0x1a1, 0x1a3, 0x5, 0x3e, 0x20, 0x2, 0x1a2, 0x196, 0x3, 
    0x2, 0x2, 0x2, 0x1a2, 0x199, 0x3, 0x2, 0x2, 0x2, 0x1a2, 0x19c, 0x3, 
    0x2, 0x2, 0x2, 0x1a2, 0x19f, 0x3, 0x2, 0x2, 0x2, 0x1a3, 0x1a6, 0x3, 
    0x2, 0x2, 0x2, 0x1a4, 0x1a2, 0x3, 0x2, 0x2, 0x2, 0x1a4, 0x1a5, 0x3, 
    0x2, 0x2, 0x2, 0x1a5, 0x41, 0x3, 0x2, 0x2, 0x2, 0x1a6, 0x1a4, 0x3, 0x2, 
    0x2, 0x2, 0x1a7, 0x1a8, 0x8, 0x22, 0x1, 0x2, 0x1a8, 0x1a9, 0x5, 0x40, 
    0x21, 0x2, 0x1a9, 0x1b2, 0x3, 0x2, 0x2, 0x2, 0x1aa, 0x1ab, 0xc, 0x4, 
    0x2, 0x2, 0x1ab, 0x1ac, 0x7, 0x3c, 0x2, 0x2, 0x1ac, 0x1b1, 0x5, 0x40, 
    0x21, 0x2, 0x1ad, 0x1ae, 0xc, 0x3, 0x2, 0x2, 0x1ae, 0x1af, 0x7, 0x3f, 
    0x2, 0x2, 0x1af, 0x1b1, 0x5, 0x40, 0x21, 0x2, 0x1b0, 0x1aa, 0x3, 0x2, 
    0x2, 0x2, 0x1b0, 0x1ad, 0x3, 0x2, 0x2, 0x2, 0x1b1, 0x1b4, 0x3, 0x2, 
    0x2, 0x2, 0x1b2, 0x1b0, 0x3, 0x2, 0x2, 0x2, 0x1b2, 0x1b3, 0x3, 0x2, 
    0x2, 0x2, 0x1b3, 0x43, 0x3, 0x2, 0x2, 0x2, 0x1b4, 0x1b2, 0x3, 0x2, 0x2, 
    0x2, 0x1b5, 0x1b6, 0x8, 0x23, 0x1, 0x2, 0x1b6, 0x1b7, 0x5, 0x42, 0x22, 
    0x2, 0x1b7, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x1b8, 0x1b9, 0xc, 0x3, 0x2, 
    0x2, 0x1b9, 0x1ba, 0x7, 0x40, 0x2, 0x2, 0x1ba, 0x1bc, 0x5, 0x42, 0x22, 
    0x2, 0x1bb, 0x1b8, 0x3, 0x2, 0x2, 0x2, 0x1bc, 0x1bf, 0x3, 0x2, 0x2, 
    0x2, 0x1bd, 0x1bb, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1be, 0x3, 0x2, 0x2, 
    0x2, 0x1be, 0x45, 0x3, 0x2, 0x2, 0x2, 0x1bf, 0x1bd, 0x3, 0x2, 0x2, 0x2, 
    0x1c0, 0x1c1, 0x8, 0x24, 0x1, 0x2, 0x1c1, 0x1c2, 0x5, 0x44, 0x23, 0x2, 
    0x1c2, 0x1c8, 0x3, 0x2, 0x2, 0x2, 0x1c3, 0x1c4, 0xc, 0x3, 0x2, 0x2, 
    0x1c4, 0x1c5, 0x7, 0x41, 0x2, 0x2, 0x1c5, 0x1c7, 0x5, 0x44, 0x23, 0x2, 
    0x1c6, 0x1c3, 0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1ca, 0x3, 0x2, 0x2, 0x2, 
    0x1c8, 0x1c6, 0x3, 0x2, 0x2, 0x2, 0x1c8, 0x1c9, 0x3, 0x2, 0x2, 0x2, 
    0x1c9, 0x47, 0x3, 0x2, 0x2, 0x2, 0x1ca, 0x1c8, 0x3, 0x2, 0x2, 0x2, 0x1cb, 
    0x1d1, 0x5, 0x46, 0x24, 0x2, 0x1cc, 0x1cd, 0x5, 0xe, 0x8, 0x2, 0x1cd, 
    0x1ce, 0x7, 0x38, 0x2, 0x2, 0x1ce, 0x1cf, 0x5, 0x46, 0x24, 0x2, 0x1cf, 
    0x1d1, 0x3, 0x2, 0x2, 0x2, 0x1d0, 0x1cb, 0x3, 0x2, 0x2, 0x2, 0x1d0, 
    0x1cc, 0x3, 0x2, 0x2, 0x2, 0x1d1, 0x49, 0x3, 0x2, 0x2, 0x2, 0x1d2, 0x1d3, 
    0x5, 0x48, 0x25, 0x2, 0x1d3, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x1d4, 0x1d9, 
    0x5, 0x10, 0x9, 0x2, 0x1d5, 0x1d6, 0x7, 0x46, 0x2, 0x2, 0x1d6, 0x1d8, 
    0x5, 0x10, 0x9, 0x2, 0x1d7, 0x1d5, 0x3, 0x2, 0x2, 0x2, 0x1d8, 0x1db, 
    0x3, 0x2, 0x2, 0x2, 0x1d9, 0x1d7, 0x3, 0x2, 0x2, 0x2, 0x1d9, 0x1da, 
    0x3, 0x2, 0x2, 0x2, 0x1da, 0x1e5, 0x3, 0x2, 0x2, 0x2, 0x1db, 0x1d9, 
    0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1e1, 0x7, 0x4c, 0x2, 0x2, 0x1dd, 0x1de, 
    0x7, 0x46, 0x2, 0x2, 0x1de, 0x1e0, 0x7, 0x4c, 0x2, 0x2, 0x1df, 0x1dd, 
    0x3, 0x2, 0x2, 0x2, 0x1e0, 0x1e3, 0x3, 0x2, 0x2, 0x2, 0x1e1, 0x1df, 
    0x3, 0x2, 0x2, 0x2, 0x1e1, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x1e2, 0x1e5, 
    0x3, 0x2, 0x2, 0x2, 0x1e3, 0x1e1, 0x3, 0x2, 0x2, 0x2, 0x1e4, 0x1d4, 
    0x3, 0x2, 0x2, 0x2, 0x1e4, 0x1dc, 0x3, 0x2, 0x2, 0x2, 0x1e5, 0x4d, 0x3, 
    0x2, 0x2, 0x2, 0x1e6, 0x1e7, 0x5, 0xa, 0x6, 0x2, 0x1e7, 0x1e8, 0x7, 
    0x4b, 0x2, 0x2, 0x1e8, 0x1f3, 0x3, 0x2, 0x2, 0x2, 0x1e9, 0x1f3, 0x5, 
    0x52, 0x2a, 0x2, 0x1ea, 0x1f3, 0x5, 0x56, 0x2c, 0x2, 0x1eb, 0x1f3, 0x5, 
    0x58, 0x2d, 0x2, 0x1ec, 0x1f3, 0x5, 0x50, 0x29, 0x2, 0x1ed, 0x1f3, 0x5, 
    0x5a, 0x2e, 0x2, 0x1ee, 0x1f3, 0x5, 0x5c, 0x2f, 0x2, 0x1ef, 0x1f3, 0x5, 
    0x60, 0x31, 0x2, 0x1f0, 0x1f3, 0x5, 0x5e, 0x30, 0x2, 0x1f1, 0x1f3, 0x5, 
    0x6e, 0x38, 0x2, 0x1f2, 0x1e6, 0x3, 0x2, 0x2, 0x2, 0x1f2, 0x1e9, 0x3, 
    0x2, 0x2, 0x2, 0x1f2, 0x1ea, 0x3, 0x2, 0x2, 0x2, 0x1f2, 0x1eb, 0x3, 
    0x2, 0x2, 0x2, 0x1f2, 0x1ec, 0x3, 0x2, 0x2, 0x2, 0x1f2, 0x1ed, 0x3, 
    0x2, 0x2, 0x2, 0x1f2, 0x1ee, 0x3, 0x2, 0x2, 0x2, 0x1f2, 0x1ef, 0x3, 
    0x2, 0x2, 0x2, 0x1f2, 0x1f0, 0x3, 0x2, 0x2, 0x2, 0x1f2, 0x1f1, 0x3, 
    0x2, 0x2, 0x2, 0x1f3, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x1f4, 0x1f5, 0x7, 0x2d, 
    0x2, 0x2, 0x1f5, 0x1f9, 0x7, 0x4b, 0x2, 0x2, 0x1f6, 0x1f8, 0x5, 0x4e, 
    0x28, 0x2, 0x1f7, 0x1f6, 0x3, 0x2, 0x2, 0x2, 0x1f8, 0x1fb, 0x3, 0x2, 
    0x2, 0x2, 0x1f9, 0x1f7, 0x3, 0x2, 0x2, 0x2, 0x1f9, 0x1fa, 0x3, 0x2, 
    0x2, 0x2, 0x1fa, 0x1fc, 0x3, 0x2, 0x2, 0x2, 0x1fb, 0x1f9, 0x3, 0x2, 
    0x2, 0x2, 0x1fc, 0x1fd, 0x7, 0x2e, 0x2, 0x2, 0x1fd, 0x1fe, 0x7, 0x4b, 
    0x2, 0x2, 0x1fe, 0x51, 0x3, 0x2, 0x2, 0x2, 0x1ff, 0x200, 0x7, 0x2f, 
    0x2, 0x2, 0x200, 0x201, 0x5, 0x4c, 0x27, 0x2, 0x201, 0x205, 0x7, 0x4b, 
    0x2, 0x2, 0x202, 0x204, 0x5, 0x4e, 0x28, 0x2, 0x203, 0x202, 0x3, 0x2, 
    0x2, 0x2, 0x204, 0x207, 0x3, 0x2, 0x2, 0x2, 0x205, 0x203, 0x3, 0x2, 
    0x2, 0x2, 0x205, 0x206, 0x3, 0x2, 0x2, 0x2, 0x206, 0x208, 0x3, 0x2, 
    0x2, 0x2, 0x207, 0x205, 0x3, 0x2, 0x2, 0x2, 0x208, 0x209, 0x7, 0x30, 
    0x2, 0x2, 0x209, 0x20a, 0x7, 0x4b, 0x2, 0x2, 0x20a, 0x53, 0x3, 0x2, 
    0x2, 0x2, 0x20b, 0x20c, 0x7, 0x32, 0x2, 0x2, 0x20c, 0x210, 0x7, 0x4b, 
    0x2, 0x2, 0x20d, 0x20f, 0x5, 0x4e, 0x28, 0x2, 0x20e, 0x20d, 0x3, 0x2, 
    0x2, 0x2, 0x20f, 0x212, 0x3, 0x2, 0x2, 0x2, 0x210, 0x20e, 0x3, 0x2, 
    0x2, 0x2, 0x210, 0x211, 0x3, 0x2, 0x2, 0x2, 0x211, 0x55, 0x3, 0x2, 0x2, 
    0x2, 0x212, 0x210, 0x3, 0x2, 0x2, 0x2, 0x213, 0x214, 0x7, 0x31, 0x2, 
    0x2, 0x214, 0x215, 0x5, 0x4a, 0x26, 0x2, 0x215, 0x219, 0x7, 0x4b, 0x2, 
    0x2, 0x216, 0x218, 0x5, 0x4e, 0x28, 0x2, 0x217, 0x216, 0x3, 0x2, 0x2, 
    0x2, 0x218, 0x21b, 0x3, 0x2, 0x2, 0x2, 0x219, 0x217, 0x3, 0x2, 0x2, 
    0x2, 0x219, 0x21a, 0x3, 0x2, 0x2, 0x2, 0x21a, 0x21c, 0x3, 0x2, 0x2, 
    0x2, 0x21b, 0x219, 0x3, 0x2, 0x2, 0x2, 0x21c, 0x21d, 0x5, 0x54, 0x2b, 
    0x2, 0x21d, 0x21e, 0x7, 0x33, 0x2, 0x2, 0x21e, 0x21f, 0x7, 0x4b, 0x2, 
    0x2, 0x21f, 0x22d, 0x3, 0x2, 0x2, 0x2, 0x220, 0x221, 0x7, 0x31, 0x2, 
    0x2, 0x221, 0x222, 0x5, 0x4a, 0x26, 0x2, 0x222, 0x226, 0x7, 0x4b, 0x2, 
    0x2, 0x223, 0x225, 0x5, 0x4e, 0x28, 0x2, 0x224, 0x223, 0x3, 0x2, 0x2, 
    0x2, 0x225, 0x228, 0x3, 0x2, 0x2, 0x2, 0x226, 0x224, 0x3, 0x2, 0x2, 
    0x2, 0x226, 0x227, 0x3, 0x2, 0x2, 0x2, 0x227, 0x229, 0x3, 0x2, 0x2, 
    0x2, 0x228, 0x226, 0x3, 0x2, 0x2, 0x2, 0x229, 0x22a, 0x7, 0x33, 0x2, 
    0x2, 0x22a, 0x22b, 0x7, 0x4b, 0x2, 0x2, 0x22b, 0x22d, 0x3, 0x2, 0x2, 
    0x2, 0x22c, 0x213, 0x3, 0x2, 0x2, 0x2, 0x22c, 0x220, 0x3, 0x2, 0x2, 
    0x2, 0x22d, 0x57, 0x3, 0x2, 0x2, 0x2, 0x22e, 0x22f, 0x7, 0x34, 0x2, 
    0x2, 0x22f, 0x230, 0x5, 0x4a, 0x26, 0x2, 0x230, 0x234, 0x7, 0x4b, 0x2, 
    0x2, 0x231, 0x233, 0x5, 0x4e, 0x28, 0x2, 0x232, 0x231, 0x3, 0x2, 0x2, 
    0x2, 0x233, 0x236, 0x3, 0x2, 0x2, 0x2, 0x234, 0x232, 0x3, 0x2, 0x2, 
    0x2, 0x234, 0x235, 0x3, 0x2, 0x2, 0x2, 0x235, 0x237, 0x3, 0x2, 0x2, 
    0x2, 0x236, 0x234, 0x3, 0x2, 0x2, 0x2, 0x237, 0x238, 0x7, 0x35, 0x2, 
    0x2, 0x238, 0x239, 0x7, 0x4b, 0x2, 0x2, 0x239, 0x59, 0x3, 0x2, 0x2, 
    0x2, 0x23a, 0x23b, 0x7, 0x36, 0x2, 0x2, 0x23b, 0x23c, 0x5, 0x10, 0x9, 
    0x2, 0x23c, 0x23d, 0x7, 0x46, 0x2, 0x2, 0x23d, 0x23e, 0x5, 0xe, 0x8, 
    0x2, 0x23e, 0x23f, 0x7, 0x4b, 0x2, 0x2, 0x23f, 0x246, 0x3, 0x2, 0x2, 
    0x2, 0x240, 0x241, 0x7, 0x36, 0x2, 0x2, 0x241, 0x242, 0x7, 0x6, 0x2, 
    0x2, 0x242, 0x243, 0x7, 0x46, 0x2, 0x2, 0x243, 0x244, 0x7, 0x7, 0x2, 
    0x2, 0x244, 0x246, 0x7, 0x4b, 0x2, 0x2, 0x245, 0x23a, 0x3, 0x2, 0x2, 
    0x2, 0x245, 0x240, 0x3, 0x2, 0x2, 0x2, 0x246, 0x5b, 0x3, 0x2, 0x2, 0x2, 
    0x247, 0x248, 0x7, 0x37, 0x2, 0x2, 0x248, 0x249, 0x5, 0x10, 0x9, 0x2, 
    0x249, 0x24a, 0x7, 0x4b, 0x2, 0x2, 0x24a, 0x24f, 0x3, 0x2, 0x2, 0x2, 
    0x24b, 0x24c, 0x7, 0x37, 0x2, 0x2, 0x24c, 0x24d, 0x7, 0x6, 0x2, 0x2, 
    0x24d, 0x24f, 0x7, 0x4b, 0x2, 0x2, 0x24e, 0x247, 0x3, 0x2, 0x2, 0x2, 
    0x24e, 0x24b, 0x3, 0x2, 0x2, 0x2, 0x24f, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x250, 
    0x251, 0x7, 0x8, 0x2, 0x2, 0x251, 0x252, 0x5, 0x4c, 0x27, 0x2, 0x252, 
    0x253, 0x7, 0x4b, 0x2, 0x2, 0x253, 0x258, 0x3, 0x2, 0x2, 0x2, 0x254, 
    0x255, 0x7, 0x8, 0x2, 0x2, 0x255, 0x256, 0x7, 0x6, 0x2, 0x2, 0x256, 
    0x258, 0x7, 0x4b, 0x2, 0x2, 0x257, 0x250, 0x3, 0x2, 0x2, 0x2, 0x257, 
    0x254, 0x3, 0x2, 0x2, 0x2, 0x258, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x259, 0x25a, 
    0x5, 0x4a, 0x26, 0x2, 0x25a, 0x25b, 0x7, 0x4b, 0x2, 0x2, 0x25b, 0x61, 
    0x3, 0x2, 0x2, 0x2, 0x25c, 0x25d, 0x5, 0x74, 0x3b, 0x2, 0x25d, 0x25e, 
    0x5, 0x72, 0x3a, 0x2, 0x25e, 0x25f, 0x7, 0x4b, 0x2, 0x2, 0x25f, 0x269, 
    0x3, 0x2, 0x2, 0x2, 0x260, 0x261, 0x5, 0x74, 0x3b, 0x2, 0x261, 0x262, 
    0x5, 0x72, 0x3a, 0x2, 0x262, 0x263, 0x7, 0x46, 0x2, 0x2, 0x263, 0x264, 
    0x7, 0x47, 0x2, 0x2, 0x264, 0x265, 0x5, 0x6a, 0x36, 0x2, 0x265, 0x266, 
    0x7, 0x48, 0x2, 0x2, 0x266, 0x267, 0x7, 0x4b, 0x2, 0x2, 0x267, 0x269, 
    0x3, 0x2, 0x2, 0x2, 0x268, 0x25c, 0x3, 0x2, 0x2, 0x2, 0x268, 0x260, 
    0x3, 0x2, 0x2, 0x2, 0x269, 0x63, 0x3, 0x2, 0x2, 0x2, 0x26a, 0x26b, 0x7, 
    0x2d, 0x2, 0x2, 0x26b, 0x26d, 0x7, 0x4b, 0x2, 0x2, 0x26c, 0x26e, 0x5, 
    0x68, 0x35, 0x2, 0x26d, 0x26c, 0x3, 0x2, 0x2, 0x2, 0x26e, 0x26f, 0x3, 
    0x2, 0x2, 0x2, 0x26f, 0x26d, 0x3, 0x2, 0x2, 0x2, 0x26f, 0x270, 0x3, 
    0x2, 0x2, 0x2, 0x270, 0x271, 0x3, 0x2, 0x2, 0x2, 0x271, 0x272, 0x7, 
    0x2e, 0x2, 0x2, 0x272, 0x273, 0x7, 0x4b, 0x2, 0x2, 0x273, 0x65, 0x3, 
    0x2, 0x2, 0x2, 0x274, 0x275, 0x7, 0x2f, 0x2, 0x2, 0x275, 0x276, 0x5, 
    0x4c, 0x27, 0x2, 0x276, 0x278, 0x7, 0x4b, 0x2, 0x2, 0x277, 0x279, 0x5, 
    0x68, 0x35, 0x2, 0x278, 0x277, 0x3, 0x2, 0x2, 0x2, 0x279, 0x27a, 0x3, 
    0x2, 0x2, 0x2, 0x27a, 0x278, 0x3, 0x2, 0x2, 0x2, 0x27a, 0x27b, 0x3, 
    0x2, 0x2, 0x2, 0x27b, 0x27c, 0x3, 0x2, 0x2, 0x2, 0x27c, 0x27d, 0x7, 
    0x30, 0x2, 0x2, 0x27d, 0x27e, 0x7, 0x4b, 0x2, 0x2, 0x27e, 0x67, 0x3, 
    0x2, 0x2, 0x2, 0x27f, 0x283, 0x5, 0x62, 0x32, 0x2, 0x280, 0x283, 0x5, 
    0x64, 0x33, 0x2, 0x281, 0x283, 0x5, 0x66, 0x34, 0x2, 0x282, 0x27f, 0x3, 
    0x2, 0x2, 0x2, 0x282, 0x280, 0x3, 0x2, 0x2, 0x2, 0x282, 0x281, 0x3, 
    0x2, 0x2, 0x2, 0x283, 0x69, 0x3, 0x2, 0x2, 0x2, 0x284, 0x289, 0x5, 0x6c, 
    0x37, 0x2, 0x285, 0x286, 0x7, 0x46, 0x2, 0x2, 0x286, 0x288, 0x5, 0x6c, 
    0x37, 0x2, 0x287, 0x285, 0x3, 0x2, 0x2, 0x2, 0x288, 0x28b, 0x3, 0x2, 
    0x2, 0x2, 0x289, 0x287, 0x3, 0x2, 0x2, 0x2, 0x289, 0x28a, 0x3, 0x2, 
    0x2, 0x2, 0x28a, 0x6b, 0x3, 0x2, 0x2, 0x2, 0x28b, 0x289, 0x3, 0x2, 0x2, 
    0x2, 0x28c, 0x28d, 0x8, 0x37, 0x1, 0x2, 0x28d, 0x298, 0x5, 0x70, 0x39, 
    0x2, 0x28e, 0x298, 0x7, 0x4d, 0x2, 0x2, 0x28f, 0x298, 0x7, 0x4e, 0x2, 
    0x2, 0x290, 0x298, 0x7, 0x3, 0x2, 0x2, 0x291, 0x292, 0x7, 0x47, 0x2, 
    0x2, 0x292, 0x293, 0x5, 0x6c, 0x37, 0x2, 0x293, 0x294, 0x7, 0x48, 0x2, 
    0x2, 0x294, 0x298, 0x3, 0x2, 0x2, 0x2, 0x295, 0x296, 0x7, 0x43, 0x2, 
    0x2, 0x296, 0x298, 0x5, 0x6c, 0x37, 0x7, 0x297, 0x28c, 0x3, 0x2, 0x2, 
    0x2, 0x297, 0x28e, 0x3, 0x2, 0x2, 0x2, 0x297, 0x28f, 0x3, 0x2, 0x2, 
    0x2, 0x297, 0x290, 0x3, 0x2, 0x2, 0x2, 0x297, 0x291, 0x3, 0x2, 0x2, 
    0x2, 0x297, 0x295, 0x3, 0x2, 0x2, 0x2, 0x298, 0x2a7, 0x3, 0x2, 0x2, 
    0x2, 0x299, 0x29a, 0xc, 0x6, 0x2, 0x2, 0x29a, 0x29b, 0x7, 0x44, 0x2, 
    0x2, 0x29b, 0x2a6, 0x5, 0x6c, 0x37, 0x7, 0x29c, 0x29d, 0xc, 0x5, 0x2, 
    0x2, 0x29d, 0x29e, 0x7, 0x45, 0x2, 0x2, 0x29e, 0x2a6, 0x5, 0x6c, 0x37, 
    0x6, 0x29f, 0x2a0, 0xc, 0x4, 0x2, 0x2, 0x2a0, 0x2a1, 0x7, 0x42, 0x2, 
    0x2, 0x2a1, 0x2a6, 0x5, 0x6c, 0x37, 0x5, 0x2a2, 0x2a3, 0xc, 0x3, 0x2, 
    0x2, 0x2a3, 0x2a4, 0x7, 0x43, 0x2, 0x2, 0x2a4, 0x2a6, 0x5, 0x6c, 0x37, 
    0x4, 0x2a5, 0x299, 0x3, 0x2, 0x2, 0x2, 0x2a5, 0x29c, 0x3, 0x2, 0x2, 
    0x2, 0x2a5, 0x29f, 0x3, 0x2, 0x2, 0x2, 0x2a5, 0x2a2, 0x3, 0x2, 0x2, 
    0x2, 0x2a6, 0x2a9, 0x3, 0x2, 0x2, 0x2, 0x2a7, 0x2a5, 0x3, 0x2, 0x2, 
    0x2, 0x2a7, 0x2a8, 0x3, 0x2, 0x2, 0x2, 0x2a8, 0x6d, 0x3, 0x2, 0x2, 0x2, 
    0x2a9, 0x2a7, 0x3, 0x2, 0x2, 0x2, 0x2aa, 0x2ab, 0x7, 0x9, 0x2, 0x2, 
    0x2ab, 0x2ac, 0x5, 0x70, 0x39, 0x2, 0x2ac, 0x2ad, 0x5, 0x72, 0x3a, 0x2, 
    0x2ad, 0x2b1, 0x7, 0x4b, 0x2, 0x2, 0x2ae, 0x2b0, 0x5, 0x68, 0x35, 0x2, 
    0x2af, 0x2ae, 0x3, 0x2, 0x2, 0x2, 0x2b0, 0x2b3, 0x3, 0x2, 0x2, 0x2, 
    0x2b1, 0x2af, 0x3, 0x2, 0x2, 0x2, 0x2b1, 0x2b2, 0x3, 0x2, 0x2, 0x2, 
    0x2b2, 0x2b4, 0x3, 0x2, 0x2, 0x2, 0x2b3, 0x2b1, 0x3, 0x2, 0x2, 0x2, 
    0x2b4, 0x2b5, 0x7, 0xa, 0x2, 0x2, 0x2b5, 0x2b6, 0x7, 0x4b, 0x2, 0x2, 
    0x2b6, 0x2c9, 0x3, 0x2, 0x2, 0x2, 0x2b7, 0x2b8, 0x7, 0x9, 0x2, 0x2, 
    0x2b8, 0x2b9, 0x5, 0x70, 0x39, 0x2, 0x2b9, 0x2ba, 0x5, 0x72, 0x3a, 0x2, 
    0x2ba, 0x2bb, 0x7, 0x46, 0x2, 0x2, 0x2bb, 0x2bc, 0x7, 0x47, 0x2, 0x2, 
    0x2bc, 0x2bd, 0x5, 0x72, 0x3a, 0x2, 0x2bd, 0x2be, 0x7, 0x48, 0x2, 0x2, 
    0x2be, 0x2c2, 0x7, 0x4b, 0x2, 0x2, 0x2bf, 0x2c1, 0x5, 0x68, 0x35, 0x2, 
    0x2c0, 0x2bf, 0x3, 0x2, 0x2, 0x2, 0x2c1, 0x2c4, 0x3, 0x2, 0x2, 0x2, 
    0x2c2, 0x2c0, 0x3, 0x2, 0x2, 0x2, 0x2c2, 0x2c3, 0x3, 0x2, 0x2, 0x2, 
    0x2c3, 0x2c5, 0x3, 0x2, 0x2, 0x2, 0x2c4, 0x2c2, 0x3, 0x2, 0x2, 0x2, 
    0x2c5, 0x2c6, 0x7, 0xa, 0x2, 0x2, 0x2c6, 0x2c7, 0x7, 0x4b, 0x2, 0x2, 
    0x2c7, 0x2c9, 0x3, 0x2, 0x2, 0x2, 0x2c8, 0x2aa, 0x3, 0x2, 0x2, 0x2, 
    0x2c8, 0x2b7, 0x3, 0x2, 0x2, 0x2, 0x2c9, 0x6f, 0x3, 0x2, 0x2, 0x2, 0x2ca, 
    0x2cb, 0x7, 0x4c, 0x2, 0x2, 0x2cb, 0x71, 0x3, 0x2, 0x2, 0x2, 0x2cc, 
    0x2d1, 0x5, 0x70, 0x39, 0x2, 0x2cd, 0x2ce, 0x7, 0x46, 0x2, 0x2, 0x2ce, 
    0x2d0, 0x5, 0x70, 0x39, 0x2, 0x2cf, 0x2cd, 0x3, 0x2, 0x2, 0x2, 0x2d0, 
    0x2d3, 0x3, 0x2, 0x2, 0x2, 0x2d1, 0x2cf, 0x3, 0x2, 0x2, 0x2, 0x2d1, 
    0x2d2, 0x3, 0x2, 0x2, 0x2, 0x2d2, 0x73, 0x3, 0x2, 0x2, 0x2, 0x2d3, 0x2d1, 
    0x3, 0x2, 0x2, 0x2, 0x2d4, 0x2df, 0x5, 0x26, 0x14, 0x2, 0x2d5, 0x2df, 
    0x5, 0x28, 0x15, 0x2, 0x2d6, 0x2df, 0x5, 0x2a, 0x16, 0x2, 0x2d7, 0x2df, 
    0x5, 0x2c, 0x17, 0x2, 0x2d8, 0x2df, 0x5, 0x2e, 0x18, 0x2, 0x2d9, 0x2df, 
    0x5, 0x30, 0x19, 0x2, 0x2da, 0x2df, 0x5, 0x32, 0x1a, 0x2, 0x2db, 0x2df, 
    0x5, 0x34, 0x1b, 0x2, 0x2dc, 0x2df, 0x5, 0x36, 0x1c, 0x2, 0x2dd, 0x2df, 
    0x5, 0x70, 0x39, 0x2, 0x2de, 0x2d4, 0x3, 0x2, 0x2, 0x2, 0x2de, 0x2d5, 
    0x3, 0x2, 0x2, 0x2, 0x2de, 0x2d6, 0x3, 0x2, 0x2, 0x2, 0x2de, 0x2d7, 
    0x3, 0x2, 0x2, 0x2, 0x2de, 0x2d8, 0x3, 0x2, 0x2, 0x2, 0x2de, 0x2d9, 
    0x3, 0x2, 0x2, 0x2, 0x2de, 0x2da, 0x3, 0x2, 0x2, 0x2, 0x2de, 0x2db, 
    0x3, 0x2, 0x2, 0x2, 0x2de, 0x2dc, 0x3, 0x2, 0x2, 0x2, 0x2de, 0x2dd, 
    0x3, 0x2, 0x2, 0x2, 0x2df, 0x75, 0x3, 0x2, 0x2, 0x2, 0x2e0, 0x2e1, 0x9, 
    0x7, 0x2, 0x2, 0x2e1, 0x77, 0x3, 0x2, 0x2, 0x2, 0x38, 0x7b, 0x81, 0x87, 
    0x8d, 0xa4, 0xb6, 0xc6, 0xda, 0xf2, 0x10e, 0x13a, 0x143, 0x14d, 0x152, 
    0x16c, 0x175, 0x180, 0x182, 0x18e, 0x190, 0x1a2, 0x1a4, 0x1b0, 0x1b2, 
    0x1bd, 0x1c8, 0x1d0, 0x1d9, 0x1e1, 0x1e4, 0x1f2, 0x1f9, 0x205, 0x210, 
    0x219, 0x226, 0x22c, 0x234, 0x245, 0x24e, 0x257, 0x268, 0x26f, 0x27a, 
    0x282, 0x289, 0x297, 0x2a5, 0x2a7, 0x2b1, 0x2c2, 0x2c8, 0x2d1, 0x2de, 
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
