
// Generated from .\originir.g4 by ANTLR 4.7.2


#include "originirListener.h"
#include "originirVisitor.h"

#include "originirParser.h"


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
    setState(84);
    declaration();
    setState(88);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
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
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN)
      | (1ULL << originirParser::Integer_Literal)
      | (1ULL << originirParser::Double_Literal))) != 0)) {
      setState(85);
      statement();
      setState(90);
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
    setState(91);
    qinit_declaration();
    setState(92);
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
    setState(94);
    match(originirParser::QINIT_KEY);
    setState(95);
    match(originirParser::Integer_Literal);
    setState(96);
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
    setState(98);
    match(originirParser::CREG_KEY);
    setState(99);
    match(originirParser::Integer_Literal);
    setState(100);
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
    setState(109);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::H_GATE:
      case originirParser::X_GATE:
      case originirParser::T_GATE:
      case originirParser::S_GATE:
      case originirParser::Y_GATE:
      case originirParser::Z_GATE:
      case originirParser::X1_GATE:
      case originirParser::Y1_GATE:
      case originirParser::Z1_GATE: {
        enterOuterAlt(_localctx, 1);
        setState(102);
        single_gate_without_parameter_declaration();
        break;
      }

      case originirParser::RX_GATE:
      case originirParser::RY_GATE:
      case originirParser::RZ_GATE:
      case originirParser::U1_GATE: {
        enterOuterAlt(_localctx, 2);
        setState(103);
        single_gate_with_one_parameter_declaration();
        break;
      }

      case originirParser::U4_GATE: {
        enterOuterAlt(_localctx, 3);
        setState(104);
        single_gate_with_four_parameter_declaration();
        break;
      }

      case originirParser::CNOT_GATE:
      case originirParser::CZ_GATE:
      case originirParser::ISWAP_GATE:
      case originirParser::SQISWAP_GATE:
      case originirParser::SWAPZ1_GATE: {
        enterOuterAlt(_localctx, 4);
        setState(105);
        double_gate_without_parameter_declaration();
        break;
      }

      case originirParser::ISWAPTHETA_GATE:
      case originirParser::CR_GATE: {
        enterOuterAlt(_localctx, 5);
        setState(106);
        double_gate_with_one_parameter_declaration();
        break;
      }

      case originirParser::CU_GATE: {
        enterOuterAlt(_localctx, 6);
        setState(107);
        double_gate_with_four_parameter_declaration();
        break;
      }

      case originirParser::TOFFOLI_GATE: {
        enterOuterAlt(_localctx, 7);
        setState(108);
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
    setState(111);
    match(originirParser::LBRACK);
    setState(112);
    expression();
    setState(113);
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
    setState(115);
    match(originirParser::C_KEY);
    setState(116);
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
    setState(118);
    match(originirParser::Q_KEY);
    setState(119);
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
    setState(121);
    single_gate_without_parameter_type();
    setState(122);
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
    setState(124);
    single_gate_with_one_parameter_type();
    setState(125);
    q_KEY_declaration();
    setState(126);
    match(originirParser::COMMA);
    setState(127);
    match(originirParser::LPAREN);
    setState(128);
    expression();
    setState(129);
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
  enterRule(_localctx, 20, originirParser::RuleSingle_gate_with_four_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(131);
    single_gate_with_four_parameter_type();
    setState(132);
    q_KEY_declaration();
    setState(133);
    match(originirParser::COMMA);
    setState(134);
    match(originirParser::LPAREN);
    setState(135);
    expression();
    setState(136);
    match(originirParser::COMMA);
    setState(137);
    expression();
    setState(138);
    match(originirParser::COMMA);
    setState(139);
    expression();
    setState(140);
    match(originirParser::COMMA);
    setState(141);
    expression();
    setState(142);
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
  enterRule(_localctx, 22, originirParser::RuleDouble_gate_without_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(144);
    double_gate_without_parameter_type();
    setState(145);
    q_KEY_declaration();
    setState(146);
    match(originirParser::COMMA);
    setState(147);
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
  enterRule(_localctx, 24, originirParser::RuleDouble_gate_with_one_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(149);
    double_gate_with_one_parameter_type();
    setState(150);
    q_KEY_declaration();
    setState(151);
    match(originirParser::COMMA);
    setState(152);
    q_KEY_declaration();
    setState(153);
    match(originirParser::COMMA);
    setState(154);
    match(originirParser::LPAREN);
    setState(155);
    expression();
    setState(156);
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
  enterRule(_localctx, 26, originirParser::RuleDouble_gate_with_four_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(158);
    double_gate_with_four_parameter_type();
    setState(159);
    q_KEY_declaration();
    setState(160);
    match(originirParser::COMMA);
    setState(161);
    q_KEY_declaration();
    setState(162);
    match(originirParser::COMMA);
    setState(163);
    match(originirParser::LPAREN);
    setState(164);
    expression();
    setState(165);
    match(originirParser::COMMA);
    setState(166);
    expression();
    setState(167);
    match(originirParser::COMMA);
    setState(168);
    expression();
    setState(169);
    match(originirParser::COMMA);
    setState(170);
    expression();
    setState(171);
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
  enterRule(_localctx, 28, originirParser::RuleTriple_gate_without_parameter_declaration);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(173);
    triple_gate_without_parameter_type();
    setState(174);
    q_KEY_declaration();
    setState(175);
    match(originirParser::COMMA);
    setState(176);
    q_KEY_declaration();
    setState(177);
    match(originirParser::COMMA);
    setState(178);
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
  enterRule(_localctx, 30, originirParser::RuleSingle_gate_without_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(180);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE))) != 0))) {
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
  enterRule(_localctx, 32, originirParser::RuleSingle_gate_with_one_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(182);
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
  enterRule(_localctx, 34, originirParser::RuleSingle_gate_with_four_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(184);
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
  enterRule(_localctx, 36, originirParser::RuleDouble_gate_without_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(186);
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
  enterRule(_localctx, 38, originirParser::RuleDouble_gate_with_one_parameter_type);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(188);
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
  enterRule(_localctx, 40, originirParser::RuleDouble_gate_with_four_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(190);
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
  enterRule(_localctx, 42, originirParser::RuleTriple_gate_without_parameter_type);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(192);
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
  enterRule(_localctx, 44, originirParser::RulePrimary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(200);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::C_KEY: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_ckeyContext>(_localctx));
        enterOuterAlt(_localctx, 1);
        setState(194);
        c_KEY_declaration();
        break;
      }

      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_cstContext>(_localctx));
        enterOuterAlt(_localctx, 2);
        setState(195);
        constant();
        break;
      }

      case originirParser::LPAREN: {
        _localctx = dynamic_cast<Primary_expressionContext *>(_tracker.createInstance<originirParser::Pri_exprContext>(_localctx));
        enterOuterAlt(_localctx, 3);
        setState(196);
        match(originirParser::LPAREN);
        setState(197);
        expression();
        setState(198);
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
  enterRule(_localctx, 46, originirParser::RuleUnary_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(209);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::C_KEY:
      case originirParser::LPAREN:
      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        enterOuterAlt(_localctx, 1);
        setState(202);
        primary_expression();
        break;
      }

      case originirParser::PLUS: {
        enterOuterAlt(_localctx, 2);
        setState(203);
        match(originirParser::PLUS);
        setState(204);
        primary_expression();
        break;
      }

      case originirParser::MINUS: {
        enterOuterAlt(_localctx, 3);
        setState(205);
        match(originirParser::MINUS);
        setState(206);
        primary_expression();
        break;
      }

      case originirParser::NOT: {
        enterOuterAlt(_localctx, 4);
        setState(207);
        match(originirParser::NOT);
        setState(208);
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
  size_t startState = 48;
  enterRecursionRule(_localctx, 48, originirParser::RuleMultiplicative_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(212);
    unary_expression();
    _ctx->stop = _input->LT(-1);
    setState(222);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(220);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Multiplicative_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression);
          setState(214);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(215);
          match(originirParser::MUL);
          setState(216);
          unary_expression();
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Multiplicative_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression);
          setState(217);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(218);
          match(originirParser::DIV);
          setState(219);
          unary_expression();
          break;
        }

        } 
      }
      setState(224);
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
  size_t startState = 50;
  enterRecursionRule(_localctx, 50, originirParser::RuleAddtive_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(226);
    multiplicative_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(236);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(234);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Addtive_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleAddtive_expression);
          setState(228);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(229);
          match(originirParser::PLUS);
          setState(230);
          multiplicative_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Addtive_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleAddtive_expression);
          setState(231);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(232);
          match(originirParser::MINUS);
          setState(233);
          multiplicative_expression(0);
          break;
        }

        } 
      }
      setState(238);
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
  size_t startState = 52;
  enterRecursionRule(_localctx, 52, originirParser::RuleRelational_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(240);
    addtive_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(256);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(254);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(242);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(243);
          match(originirParser::LT);
          setState(244);
          addtive_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(245);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(246);
          match(originirParser::GT);
          setState(247);
          addtive_expression(0);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(248);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(249);
          match(originirParser::LEQ);
          setState(250);
          addtive_expression(0);
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<Relational_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleRelational_expression);
          setState(251);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(252);
          match(originirParser::GEQ);
          setState(253);
          addtive_expression(0);
          break;
        }

        } 
      }
      setState(258);
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
  size_t startState = 54;
  enterRecursionRule(_localctx, 54, originirParser::RuleEquality_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(260);
    relational_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(270);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(268);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<Equality_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleEquality_expression);
          setState(262);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(263);
          match(originirParser::EQ);
          setState(264);
          relational_expression(0);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<Equality_expressionContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleEquality_expression);
          setState(265);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(266);
          match(originirParser::NE);
          setState(267);
          relational_expression(0);
          break;
        }

        } 
      }
      setState(272);
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
  size_t startState = 56;
  enterRecursionRule(_localctx, 56, originirParser::RuleLogical_and_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(274);
    equality_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(281);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<Logical_and_expressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLogical_and_expression);
        setState(276);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(277);
        match(originirParser::AND);
        setState(278);
        equality_expression(0); 
      }
      setState(283);
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
  size_t startState = 58;
  enterRecursionRule(_localctx, 58, originirParser::RuleLogical_or_expression, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(285);
    logical_and_expression(0);
    _ctx->stop = _input->LT(-1);
    setState(292);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<Logical_or_expressionContext>(parentContext, parentState);
        pushNewRecursionContext(_localctx, startState, RuleLogical_or_expression);
        setState(287);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(288);
        match(originirParser::OR);
        setState(289);
        logical_and_expression(0); 
      }
      setState(294);
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
  enterRule(_localctx, 60, originirParser::RuleAssignment_expression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(300);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(295);
      logical_or_expression(0);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(296);
      c_KEY_declaration();
      setState(297);
      match(originirParser::ASSIGN);
      setState(298);
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
  enterRule(_localctx, 62, originirParser::RuleExpression);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(302);
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
  enterRule(_localctx, 64, originirParser::RuleControlbit_list);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(304);
    q_KEY_declaration();
    setState(309);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == originirParser::COMMA) {
      setState(305);
      match(originirParser::COMMA);
      setState(306);
      q_KEY_declaration();
      setState(311);
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

originirParser::Expression_statementContext* originirParser::StatementContext::expression_statement() {
  return getRuleContext<originirParser::Expression_statementContext>(0);
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
  enterRule(_localctx, 66, originirParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(325);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case originirParser::H_GATE:
      case originirParser::X_GATE:
      case originirParser::T_GATE:
      case originirParser::S_GATE:
      case originirParser::Y_GATE:
      case originirParser::Z_GATE:
      case originirParser::X1_GATE:
      case originirParser::Y1_GATE:
      case originirParser::Z1_GATE:
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
        setState(312);
        quantum_gate_declaration();
        setState(313);
        match(originirParser::NEWLINE);
        break;
      }

      case originirParser::CONTROL_KEY: {
        enterOuterAlt(_localctx, 2);
        setState(315);
        control_statement();
        break;
      }

      case originirParser::QIF_KEY: {
        enterOuterAlt(_localctx, 3);
        setState(316);
        qif_statement();
        break;
      }

      case originirParser::QWHILE_KEY: {
        enterOuterAlt(_localctx, 4);
        setState(317);
        qwhile_statement();
        break;
      }

      case originirParser::DAGGER_KEY: {
        enterOuterAlt(_localctx, 5);
        setState(318);
        dagger_statement();
        break;
      }

      case originirParser::MEASURE_KEY: {
        enterOuterAlt(_localctx, 6);
        setState(319);
        measure_statement();
        setState(320);
        match(originirParser::NEWLINE);
        break;
      }

      case originirParser::C_KEY:
      case originirParser::NOT:
      case originirParser::PLUS:
      case originirParser::MINUS:
      case originirParser::LPAREN:
      case originirParser::Integer_Literal:
      case originirParser::Double_Literal: {
        enterOuterAlt(_localctx, 7);
        setState(322);
        expression_statement();
        setState(323);
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
  enterRule(_localctx, 68, originirParser::RuleDagger_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(327);
    match(originirParser::DAGGER_KEY);
    setState(328);
    match(originirParser::NEWLINE);
    setState(332);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
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
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN)
      | (1ULL << originirParser::Integer_Literal)
      | (1ULL << originirParser::Double_Literal))) != 0)) {
      setState(329);
      statement();
      setState(334);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(335);
    match(originirParser::ENDDAGGER_KEY);
    setState(336);
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
  enterRule(_localctx, 70, originirParser::RuleControl_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(338);
    match(originirParser::CONTROL_KEY);
    setState(339);
    controlbit_list();
    setState(340);
    match(originirParser::NEWLINE);
    setState(344);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
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
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN)
      | (1ULL << originirParser::Integer_Literal)
      | (1ULL << originirParser::Double_Literal))) != 0)) {
      setState(341);
      statement();
      setState(346);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(347);
    match(originirParser::ENDCONTROL_KEY);
    setState(348);
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
  enterRule(_localctx, 72, originirParser::RuleQelse_statement_fragment);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(350);
    match(originirParser::ELSE_KEY);
    setState(351);
    match(originirParser::NEWLINE);
    setState(355);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
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
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN)
      | (1ULL << originirParser::Integer_Literal)
      | (1ULL << originirParser::Double_Literal))) != 0)) {
      setState(352);
      statement();
      setState(357);
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
  enterRule(_localctx, 74, originirParser::RuleQif_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(383);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 22, _ctx)) {
    case 1: {
      _localctx = dynamic_cast<Qif_statementContext *>(_tracker.createInstance<originirParser::Qif_ifContext>(_localctx));
      enterOuterAlt(_localctx, 1);
      setState(358);
      match(originirParser::QIF_KEY);
      setState(359);
      expression();
      setState(360);
      match(originirParser::NEWLINE);
      setState(364);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << originirParser::C_KEY)
        | (1ULL << originirParser::H_GATE)
        | (1ULL << originirParser::X_GATE)
        | (1ULL << originirParser::T_GATE)
        | (1ULL << originirParser::S_GATE)
        | (1ULL << originirParser::Y_GATE)
        | (1ULL << originirParser::Z_GATE)
        | (1ULL << originirParser::X1_GATE)
        | (1ULL << originirParser::Y1_GATE)
        | (1ULL << originirParser::Z1_GATE)
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
        | (1ULL << originirParser::NOT)
        | (1ULL << originirParser::PLUS)
        | (1ULL << originirParser::MINUS)
        | (1ULL << originirParser::LPAREN)
        | (1ULL << originirParser::Integer_Literal)
        | (1ULL << originirParser::Double_Literal))) != 0)) {
        setState(361);
        statement();
        setState(366);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(367);
      qelse_statement_fragment();
      setState(368);
      match(originirParser::ENDIF_KEY);
      setState(369);
      match(originirParser::NEWLINE);
      break;
    }

    case 2: {
      _localctx = dynamic_cast<Qif_statementContext *>(_tracker.createInstance<originirParser::Qif_ifelseContext>(_localctx));
      enterOuterAlt(_localctx, 2);
      setState(371);
      match(originirParser::QIF_KEY);
      setState(372);
      expression();
      setState(373);
      match(originirParser::NEWLINE);
      setState(377);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << originirParser::C_KEY)
        | (1ULL << originirParser::H_GATE)
        | (1ULL << originirParser::X_GATE)
        | (1ULL << originirParser::T_GATE)
        | (1ULL << originirParser::S_GATE)
        | (1ULL << originirParser::Y_GATE)
        | (1ULL << originirParser::Z_GATE)
        | (1ULL << originirParser::X1_GATE)
        | (1ULL << originirParser::Y1_GATE)
        | (1ULL << originirParser::Z1_GATE)
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
        | (1ULL << originirParser::NOT)
        | (1ULL << originirParser::PLUS)
        | (1ULL << originirParser::MINUS)
        | (1ULL << originirParser::LPAREN)
        | (1ULL << originirParser::Integer_Literal)
        | (1ULL << originirParser::Double_Literal))) != 0)) {
        setState(374);
        statement();
        setState(379);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      setState(380);
      match(originirParser::ENDIF_KEY);
      setState(381);
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
  enterRule(_localctx, 76, originirParser::RuleQwhile_statement);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(385);
    match(originirParser::QWHILE_KEY);
    setState(386);
    expression();
    setState(387);
    match(originirParser::NEWLINE);
    setState(391);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << originirParser::C_KEY)
      | (1ULL << originirParser::H_GATE)
      | (1ULL << originirParser::X_GATE)
      | (1ULL << originirParser::T_GATE)
      | (1ULL << originirParser::S_GATE)
      | (1ULL << originirParser::Y_GATE)
      | (1ULL << originirParser::Z_GATE)
      | (1ULL << originirParser::X1_GATE)
      | (1ULL << originirParser::Y1_GATE)
      | (1ULL << originirParser::Z1_GATE)
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
      | (1ULL << originirParser::NOT)
      | (1ULL << originirParser::PLUS)
      | (1ULL << originirParser::MINUS)
      | (1ULL << originirParser::LPAREN)
      | (1ULL << originirParser::Integer_Literal)
      | (1ULL << originirParser::Double_Literal))) != 0)) {
      setState(388);
      statement();
      setState(393);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(394);
    match(originirParser::ENDQWHILE_KEY);
    setState(395);
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
  enterRule(_localctx, 78, originirParser::RuleMeasure_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(397);
    match(originirParser::MEASURE_KEY);
    setState(398);
    q_KEY_declaration();
    setState(399);
    match(originirParser::COMMA);
    setState(400);
    c_KEY_declaration();
   
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
  enterRule(_localctx, 80, originirParser::RuleExpression_statement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(402);
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
  enterRule(_localctx, 82, originirParser::RuleConstant);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(404);
    _la = _input->LA(1);
    if (!(_la == originirParser::Integer_Literal

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
    case 24: return multiplicative_expressionSempred(dynamic_cast<Multiplicative_expressionContext *>(context), predicateIndex);
    case 25: return addtive_expressionSempred(dynamic_cast<Addtive_expressionContext *>(context), predicateIndex);
    case 26: return relational_expressionSempred(dynamic_cast<Relational_expressionContext *>(context), predicateIndex);
    case 27: return equality_expressionSempred(dynamic_cast<Equality_expressionContext *>(context), predicateIndex);
    case 28: return logical_and_expressionSempred(dynamic_cast<Logical_and_expressionContext *>(context), predicateIndex);
    case 29: return logical_or_expressionSempred(dynamic_cast<Logical_or_expressionContext *>(context), predicateIndex);

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
  "single_gate_with_four_parameter_declaration", "double_gate_without_parameter_declaration", 
  "double_gate_with_one_parameter_declaration", "double_gate_with_four_parameter_declaration", 
  "triple_gate_without_parameter_declaration", "single_gate_without_parameter_type", 
  "single_gate_with_one_parameter_type", "single_gate_with_four_parameter_type", 
  "double_gate_without_parameter_type", "double_gate_with_one_parameter_type", 
  "double_gate_with_four_parameter_type", "triple_gate_without_parameter_type", 
  "primary_expression", "unary_expression", "multiplicative_expression", 
  "addtive_expression", "relational_expression", "equality_expression", 
  "logical_and_expression", "logical_or_expression", "assignment_expression", 
  "expression", "controlbit_list", "statement", "dagger_statement", "control_statement", 
  "qelse_statement_fragment", "qif_statement", "qwhile_statement", "measure_statement", 
  "expression_statement", "constant"
};

std::vector<std::string> originirParser::_literalNames = {
  "", "'Pi'", "'QINIT'", "'CREG'", "'q'", "'c'", "'H'", "'X'", "'NOT'", 
  "'T'", "'S'", "'Y'", "'Z'", "'X1'", "'Y1'", "'Z1'", "'U4'", "'RX'", "'RY'", 
  "'RZ'", "'U1'", "'CNOT'", "'CZ'", "'CU'", "'ISWAP'", "'SQISWAP'", "'SWAP'", 
  "'ISWAPTHETA'", "'CR'", "'TOFFOLI'", "'DAGGER'", "'ENDDAGGER'", "'CONTROL'", 
  "'ENDCONTROL'", "'QIF'", "'ELSE'", "'ENDQIF'", "'QWHILE'", "'ENDQWHILE'", 
  "'MEASURE'", "'='", "'>'", "'<'", "'!'", "'=='", "'<='", "'>='", "'!='", 
  "'&&'", "'||'", "'+'", "'-'", "'*'", "'/'", "','", "'('", "')'", "'['", 
  "']'"
};

std::vector<std::string> originirParser::_symbolicNames = {
  "", "PI", "QINIT_KEY", "CREG_KEY", "Q_KEY", "C_KEY", "H_GATE", "X_GATE", 
  "NOT_GATE", "T_GATE", "S_GATE", "Y_GATE", "Z_GATE", "X1_GATE", "Y1_GATE", 
  "Z1_GATE", "U4_GATE", "RX_GATE", "RY_GATE", "RZ_GATE", "U1_GATE", "CNOT_GATE", 
  "CZ_GATE", "CU_GATE", "ISWAP_GATE", "SQISWAP_GATE", "SWAPZ1_GATE", "ISWAPTHETA_GATE", 
  "CR_GATE", "TOFFOLI_GATE", "DAGGER_KEY", "ENDDAGGER_KEY", "CONTROL_KEY", 
  "ENDCONTROL_KEY", "QIF_KEY", "ELSE_KEY", "ENDIF_KEY", "QWHILE_KEY", "ENDQWHILE_KEY", 
  "MEASURE_KEY", "ASSIGN", "GT", "LT", "NOT", "EQ", "LEQ", "GEQ", "NE", 
  "AND", "OR", "PLUS", "MINUS", "MUL", "DIV", "COMMA", "LPAREN", "RPAREN", 
  "LBRACK", "RBRACK", "NEWLINE", "Identifier", "Integer_Literal", "Double_Literal", 
  "Digit_Sequence", "WhiteSpace", "SingleLineComment"
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
    0x3, 0x43, 0x199, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
    0x4, 0x29, 0x9, 0x29, 0x4, 0x2a, 0x9, 0x2a, 0x4, 0x2b, 0x9, 0x2b, 0x3, 
    0x2, 0x3, 0x2, 0x7, 0x2, 0x59, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x5c, 0xb, 
    0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x70, 0xa, 0x6, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 
    0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 
    0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x11, 0x3, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 
    0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 0x3, 0x18, 
    0x5, 0x18, 0xcb, 0xa, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 
    0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x5, 0x19, 0xd4, 0xa, 0x19, 0x3, 0x1a, 
    0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 
    0x1a, 0x3, 0x1a, 0x7, 0x1a, 0xdf, 0xa, 0x1a, 0xc, 0x1a, 0xe, 0x1a, 0xe2, 
    0xb, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 
    0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x7, 0x1b, 0xed, 0xa, 0x1b, 0xc, 
    0x1b, 0xe, 0x1b, 0xf0, 0xb, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 
    0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
    0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x7, 0x1c, 0x101, 
    0xa, 0x1c, 0xc, 0x1c, 0xe, 0x1c, 0x104, 0xb, 0x1c, 0x3, 0x1d, 0x3, 0x1d, 
    0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 
    0x1d, 0x7, 0x1d, 0x10f, 0xa, 0x1d, 0xc, 0x1d, 0xe, 0x1d, 0x112, 0xb, 
    0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 
    0x7, 0x1e, 0x11a, 0xa, 0x1e, 0xc, 0x1e, 0xe, 0x1e, 0x11d, 0xb, 0x1e, 
    0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x7, 
    0x1f, 0x125, 0xa, 0x1f, 0xc, 0x1f, 0xe, 0x1f, 0x128, 0xb, 0x1f, 0x3, 
    0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x5, 0x20, 0x12f, 
    0xa, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 0x22, 0x7, 
    0x22, 0x136, 0xa, 0x22, 0xc, 0x22, 0xe, 0x22, 0x139, 0xb, 0x22, 0x3, 
    0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 
    0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x5, 
    0x23, 0x148, 0xa, 0x23, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x7, 0x24, 
    0x14d, 0xa, 0x24, 0xc, 0x24, 0xe, 0x24, 0x150, 0xb, 0x24, 0x3, 0x24, 
    0x3, 0x24, 0x3, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x7, 
    0x25, 0x159, 0xa, 0x25, 0xc, 0x25, 0xe, 0x25, 0x15c, 0xb, 0x25, 0x3, 
    0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x7, 0x26, 
    0x164, 0xa, 0x26, 0xc, 0x26, 0xe, 0x26, 0x167, 0xb, 0x26, 0x3, 0x27, 
    0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 0x16d, 0xa, 0x27, 0xc, 0x27, 
    0xe, 0x27, 0x170, 0xb, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 
    0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 0x17a, 0xa, 0x27, 
    0xc, 0x27, 0xe, 0x27, 0x17d, 0xb, 0x27, 0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 
    0x5, 0x27, 0x182, 0xa, 0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 
    0x7, 0x28, 0x188, 0xa, 0x28, 0xc, 0x28, 0xe, 0x28, 0x18b, 0xb, 0x28, 
    0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 
    0x29, 0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 
    0x2, 0x8, 0x32, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x2c, 0x2, 0x4, 0x6, 0x8, 
    0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 
    0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 0x38, 
    0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 0x50, 
    0x52, 0x54, 0x2, 0x7, 0x4, 0x2, 0x8, 0x9, 0xb, 0x11, 0x3, 0x2, 0x13, 
    0x16, 0x4, 0x2, 0x17, 0x18, 0x1a, 0x1c, 0x3, 0x2, 0x1d, 0x1e, 0x3, 0x2, 
    0x3f, 0x40, 0x2, 0x195, 0x2, 0x56, 0x3, 0x2, 0x2, 0x2, 0x4, 0x5d, 0x3, 
    0x2, 0x2, 0x2, 0x6, 0x60, 0x3, 0x2, 0x2, 0x2, 0x8, 0x64, 0x3, 0x2, 0x2, 
    0x2, 0xa, 0x6f, 0x3, 0x2, 0x2, 0x2, 0xc, 0x71, 0x3, 0x2, 0x2, 0x2, 0xe, 
    0x75, 0x3, 0x2, 0x2, 0x2, 0x10, 0x78, 0x3, 0x2, 0x2, 0x2, 0x12, 0x7b, 
    0x3, 0x2, 0x2, 0x2, 0x14, 0x7e, 0x3, 0x2, 0x2, 0x2, 0x16, 0x85, 0x3, 
    0x2, 0x2, 0x2, 0x18, 0x92, 0x3, 0x2, 0x2, 0x2, 0x1a, 0x97, 0x3, 0x2, 
    0x2, 0x2, 0x1c, 0xa0, 0x3, 0x2, 0x2, 0x2, 0x1e, 0xaf, 0x3, 0x2, 0x2, 
    0x2, 0x20, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x22, 0xb8, 0x3, 0x2, 0x2, 0x2, 
    0x24, 0xba, 0x3, 0x2, 0x2, 0x2, 0x26, 0xbc, 0x3, 0x2, 0x2, 0x2, 0x28, 
    0xbe, 0x3, 0x2, 0x2, 0x2, 0x2a, 0xc0, 0x3, 0x2, 0x2, 0x2, 0x2c, 0xc2, 
    0x3, 0x2, 0x2, 0x2, 0x2e, 0xca, 0x3, 0x2, 0x2, 0x2, 0x30, 0xd3, 0x3, 
    0x2, 0x2, 0x2, 0x32, 0xd5, 0x3, 0x2, 0x2, 0x2, 0x34, 0xe3, 0x3, 0x2, 
    0x2, 0x2, 0x36, 0xf1, 0x3, 0x2, 0x2, 0x2, 0x38, 0x105, 0x3, 0x2, 0x2, 
    0x2, 0x3a, 0x113, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x11e, 0x3, 0x2, 0x2, 0x2, 
    0x3e, 0x12e, 0x3, 0x2, 0x2, 0x2, 0x40, 0x130, 0x3, 0x2, 0x2, 0x2, 0x42, 
    0x132, 0x3, 0x2, 0x2, 0x2, 0x44, 0x147, 0x3, 0x2, 0x2, 0x2, 0x46, 0x149, 
    0x3, 0x2, 0x2, 0x2, 0x48, 0x154, 0x3, 0x2, 0x2, 0x2, 0x4a, 0x160, 0x3, 
    0x2, 0x2, 0x2, 0x4c, 0x181, 0x3, 0x2, 0x2, 0x2, 0x4e, 0x183, 0x3, 0x2, 
    0x2, 0x2, 0x50, 0x18f, 0x3, 0x2, 0x2, 0x2, 0x52, 0x194, 0x3, 0x2, 0x2, 
    0x2, 0x54, 0x196, 0x3, 0x2, 0x2, 0x2, 0x56, 0x5a, 0x5, 0x4, 0x3, 0x2, 
    0x57, 0x59, 0x5, 0x44, 0x23, 0x2, 0x58, 0x57, 0x3, 0x2, 0x2, 0x2, 0x59, 
    0x5c, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x58, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x5b, 
    0x3, 0x2, 0x2, 0x2, 0x5b, 0x3, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x5a, 0x3, 
    0x2, 0x2, 0x2, 0x5d, 0x5e, 0x5, 0x6, 0x4, 0x2, 0x5e, 0x5f, 0x5, 0x8, 
    0x5, 0x2, 0x5f, 0x5, 0x3, 0x2, 0x2, 0x2, 0x60, 0x61, 0x7, 0x4, 0x2, 
    0x2, 0x61, 0x62, 0x7, 0x3f, 0x2, 0x2, 0x62, 0x63, 0x7, 0x3d, 0x2, 0x2, 
    0x63, 0x7, 0x3, 0x2, 0x2, 0x2, 0x64, 0x65, 0x7, 0x5, 0x2, 0x2, 0x65, 
    0x66, 0x7, 0x3f, 0x2, 0x2, 0x66, 0x67, 0x7, 0x3d, 0x2, 0x2, 0x67, 0x9, 
    0x3, 0x2, 0x2, 0x2, 0x68, 0x70, 0x5, 0x12, 0xa, 0x2, 0x69, 0x70, 0x5, 
    0x14, 0xb, 0x2, 0x6a, 0x70, 0x5, 0x16, 0xc, 0x2, 0x6b, 0x70, 0x5, 0x18, 
    0xd, 0x2, 0x6c, 0x70, 0x5, 0x1a, 0xe, 0x2, 0x6d, 0x70, 0x5, 0x1c, 0xf, 
    0x2, 0x6e, 0x70, 0x5, 0x1e, 0x10, 0x2, 0x6f, 0x68, 0x3, 0x2, 0x2, 0x2, 
    0x6f, 0x69, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x6a, 0x3, 0x2, 0x2, 0x2, 0x6f, 
    0x6b, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x6d, 
    0x3, 0x2, 0x2, 0x2, 0x6f, 0x6e, 0x3, 0x2, 0x2, 0x2, 0x70, 0xb, 0x3, 
    0x2, 0x2, 0x2, 0x71, 0x72, 0x7, 0x3b, 0x2, 0x2, 0x72, 0x73, 0x5, 0x40, 
    0x21, 0x2, 0x73, 0x74, 0x7, 0x3c, 0x2, 0x2, 0x74, 0xd, 0x3, 0x2, 0x2, 
    0x2, 0x75, 0x76, 0x7, 0x7, 0x2, 0x2, 0x76, 0x77, 0x5, 0xc, 0x7, 0x2, 
    0x77, 0xf, 0x3, 0x2, 0x2, 0x2, 0x78, 0x79, 0x7, 0x6, 0x2, 0x2, 0x79, 
    0x7a, 0x5, 0xc, 0x7, 0x2, 0x7a, 0x11, 0x3, 0x2, 0x2, 0x2, 0x7b, 0x7c, 
    0x5, 0x20, 0x11, 0x2, 0x7c, 0x7d, 0x5, 0x10, 0x9, 0x2, 0x7d, 0x13, 0x3, 
    0x2, 0x2, 0x2, 0x7e, 0x7f, 0x5, 0x22, 0x12, 0x2, 0x7f, 0x80, 0x5, 0x10, 
    0x9, 0x2, 0x80, 0x81, 0x7, 0x38, 0x2, 0x2, 0x81, 0x82, 0x7, 0x39, 0x2, 
    0x2, 0x82, 0x83, 0x5, 0x40, 0x21, 0x2, 0x83, 0x84, 0x7, 0x3a, 0x2, 0x2, 
    0x84, 0x15, 0x3, 0x2, 0x2, 0x2, 0x85, 0x86, 0x5, 0x24, 0x13, 0x2, 0x86, 
    0x87, 0x5, 0x10, 0x9, 0x2, 0x87, 0x88, 0x7, 0x38, 0x2, 0x2, 0x88, 0x89, 
    0x7, 0x39, 0x2, 0x2, 0x89, 0x8a, 0x5, 0x40, 0x21, 0x2, 0x8a, 0x8b, 0x7, 
    0x38, 0x2, 0x2, 0x8b, 0x8c, 0x5, 0x40, 0x21, 0x2, 0x8c, 0x8d, 0x7, 0x38, 
    0x2, 0x2, 0x8d, 0x8e, 0x5, 0x40, 0x21, 0x2, 0x8e, 0x8f, 0x7, 0x38, 0x2, 
    0x2, 0x8f, 0x90, 0x5, 0x40, 0x21, 0x2, 0x90, 0x91, 0x7, 0x3a, 0x2, 0x2, 
    0x91, 0x17, 0x3, 0x2, 0x2, 0x2, 0x92, 0x93, 0x5, 0x26, 0x14, 0x2, 0x93, 
    0x94, 0x5, 0x10, 0x9, 0x2, 0x94, 0x95, 0x7, 0x38, 0x2, 0x2, 0x95, 0x96, 
    0x5, 0x10, 0x9, 0x2, 0x96, 0x19, 0x3, 0x2, 0x2, 0x2, 0x97, 0x98, 0x5, 
    0x28, 0x15, 0x2, 0x98, 0x99, 0x5, 0x10, 0x9, 0x2, 0x99, 0x9a, 0x7, 0x38, 
    0x2, 0x2, 0x9a, 0x9b, 0x5, 0x10, 0x9, 0x2, 0x9b, 0x9c, 0x7, 0x38, 0x2, 
    0x2, 0x9c, 0x9d, 0x7, 0x39, 0x2, 0x2, 0x9d, 0x9e, 0x5, 0x40, 0x21, 0x2, 
    0x9e, 0x9f, 0x7, 0x3a, 0x2, 0x2, 0x9f, 0x1b, 0x3, 0x2, 0x2, 0x2, 0xa0, 
    0xa1, 0x5, 0x2a, 0x16, 0x2, 0xa1, 0xa2, 0x5, 0x10, 0x9, 0x2, 0xa2, 0xa3, 
    0x7, 0x38, 0x2, 0x2, 0xa3, 0xa4, 0x5, 0x10, 0x9, 0x2, 0xa4, 0xa5, 0x7, 
    0x38, 0x2, 0x2, 0xa5, 0xa6, 0x7, 0x39, 0x2, 0x2, 0xa6, 0xa7, 0x5, 0x40, 
    0x21, 0x2, 0xa7, 0xa8, 0x7, 0x38, 0x2, 0x2, 0xa8, 0xa9, 0x5, 0x40, 0x21, 
    0x2, 0xa9, 0xaa, 0x7, 0x38, 0x2, 0x2, 0xaa, 0xab, 0x5, 0x40, 0x21, 0x2, 
    0xab, 0xac, 0x7, 0x38, 0x2, 0x2, 0xac, 0xad, 0x5, 0x40, 0x21, 0x2, 0xad, 
    0xae, 0x7, 0x3a, 0x2, 0x2, 0xae, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xaf, 0xb0, 
    0x5, 0x2c, 0x17, 0x2, 0xb0, 0xb1, 0x5, 0x10, 0x9, 0x2, 0xb1, 0xb2, 0x7, 
    0x38, 0x2, 0x2, 0xb2, 0xb3, 0x5, 0x10, 0x9, 0x2, 0xb3, 0xb4, 0x7, 0x38, 
    0x2, 0x2, 0xb4, 0xb5, 0x5, 0x10, 0x9, 0x2, 0xb5, 0x1f, 0x3, 0x2, 0x2, 
    0x2, 0xb6, 0xb7, 0x9, 0x2, 0x2, 0x2, 0xb7, 0x21, 0x3, 0x2, 0x2, 0x2, 
    0xb8, 0xb9, 0x9, 0x3, 0x2, 0x2, 0xb9, 0x23, 0x3, 0x2, 0x2, 0x2, 0xba, 
    0xbb, 0x7, 0x12, 0x2, 0x2, 0xbb, 0x25, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xbd, 
    0x9, 0x4, 0x2, 0x2, 0xbd, 0x27, 0x3, 0x2, 0x2, 0x2, 0xbe, 0xbf, 0x9, 
    0x5, 0x2, 0x2, 0xbf, 0x29, 0x3, 0x2, 0x2, 0x2, 0xc0, 0xc1, 0x7, 0x19, 
    0x2, 0x2, 0xc1, 0x2b, 0x3, 0x2, 0x2, 0x2, 0xc2, 0xc3, 0x7, 0x1f, 0x2, 
    0x2, 0xc3, 0x2d, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xcb, 0x5, 0xe, 0x8, 0x2, 
    0xc5, 0xcb, 0x5, 0x54, 0x2b, 0x2, 0xc6, 0xc7, 0x7, 0x39, 0x2, 0x2, 0xc7, 
    0xc8, 0x5, 0x40, 0x21, 0x2, 0xc8, 0xc9, 0x7, 0x39, 0x2, 0x2, 0xc9, 0xcb, 
    0x3, 0x2, 0x2, 0x2, 0xca, 0xc4, 0x3, 0x2, 0x2, 0x2, 0xca, 0xc5, 0x3, 
    0x2, 0x2, 0x2, 0xca, 0xc6, 0x3, 0x2, 0x2, 0x2, 0xcb, 0x2f, 0x3, 0x2, 
    0x2, 0x2, 0xcc, 0xd4, 0x5, 0x2e, 0x18, 0x2, 0xcd, 0xce, 0x7, 0x34, 0x2, 
    0x2, 0xce, 0xd4, 0x5, 0x2e, 0x18, 0x2, 0xcf, 0xd0, 0x7, 0x35, 0x2, 0x2, 
    0xd0, 0xd4, 0x5, 0x2e, 0x18, 0x2, 0xd1, 0xd2, 0x7, 0x2d, 0x2, 0x2, 0xd2, 
    0xd4, 0x5, 0x2e, 0x18, 0x2, 0xd3, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xd3, 0xcd, 
    0x3, 0x2, 0x2, 0x2, 0xd3, 0xcf, 0x3, 0x2, 0x2, 0x2, 0xd3, 0xd1, 0x3, 
    0x2, 0x2, 0x2, 0xd4, 0x31, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd6, 0x8, 0x1a, 
    0x1, 0x2, 0xd6, 0xd7, 0x5, 0x30, 0x19, 0x2, 0xd7, 0xe0, 0x3, 0x2, 0x2, 
    0x2, 0xd8, 0xd9, 0xc, 0x4, 0x2, 0x2, 0xd9, 0xda, 0x7, 0x36, 0x2, 0x2, 
    0xda, 0xdf, 0x5, 0x30, 0x19, 0x2, 0xdb, 0xdc, 0xc, 0x3, 0x2, 0x2, 0xdc, 
    0xdd, 0x7, 0x37, 0x2, 0x2, 0xdd, 0xdf, 0x5, 0x30, 0x19, 0x2, 0xde, 0xd8, 
    0x3, 0x2, 0x2, 0x2, 0xde, 0xdb, 0x3, 0x2, 0x2, 0x2, 0xdf, 0xe2, 0x3, 
    0x2, 0x2, 0x2, 0xe0, 0xde, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xe1, 0x3, 0x2, 
    0x2, 0x2, 0xe1, 0x33, 0x3, 0x2, 0x2, 0x2, 0xe2, 0xe0, 0x3, 0x2, 0x2, 
    0x2, 0xe3, 0xe4, 0x8, 0x1b, 0x1, 0x2, 0xe4, 0xe5, 0x5, 0x32, 0x1a, 0x2, 
    0xe5, 0xee, 0x3, 0x2, 0x2, 0x2, 0xe6, 0xe7, 0xc, 0x4, 0x2, 0x2, 0xe7, 
    0xe8, 0x7, 0x34, 0x2, 0x2, 0xe8, 0xed, 0x5, 0x32, 0x1a, 0x2, 0xe9, 0xea, 
    0xc, 0x3, 0x2, 0x2, 0xea, 0xeb, 0x7, 0x35, 0x2, 0x2, 0xeb, 0xed, 0x5, 
    0x32, 0x1a, 0x2, 0xec, 0xe6, 0x3, 0x2, 0x2, 0x2, 0xec, 0xe9, 0x3, 0x2, 
    0x2, 0x2, 0xed, 0xf0, 0x3, 0x2, 0x2, 0x2, 0xee, 0xec, 0x3, 0x2, 0x2, 
    0x2, 0xee, 0xef, 0x3, 0x2, 0x2, 0x2, 0xef, 0x35, 0x3, 0x2, 0x2, 0x2, 
    0xf0, 0xee, 0x3, 0x2, 0x2, 0x2, 0xf1, 0xf2, 0x8, 0x1c, 0x1, 0x2, 0xf2, 
    0xf3, 0x5, 0x34, 0x1b, 0x2, 0xf3, 0x102, 0x3, 0x2, 0x2, 0x2, 0xf4, 0xf5, 
    0xc, 0x6, 0x2, 0x2, 0xf5, 0xf6, 0x7, 0x2c, 0x2, 0x2, 0xf6, 0x101, 0x5, 
    0x34, 0x1b, 0x2, 0xf7, 0xf8, 0xc, 0x5, 0x2, 0x2, 0xf8, 0xf9, 0x7, 0x2b, 
    0x2, 0x2, 0xf9, 0x101, 0x5, 0x34, 0x1b, 0x2, 0xfa, 0xfb, 0xc, 0x4, 0x2, 
    0x2, 0xfb, 0xfc, 0x7, 0x2f, 0x2, 0x2, 0xfc, 0x101, 0x5, 0x34, 0x1b, 
    0x2, 0xfd, 0xfe, 0xc, 0x3, 0x2, 0x2, 0xfe, 0xff, 0x7, 0x30, 0x2, 0x2, 
    0xff, 0x101, 0x5, 0x34, 0x1b, 0x2, 0x100, 0xf4, 0x3, 0x2, 0x2, 0x2, 
    0x100, 0xf7, 0x3, 0x2, 0x2, 0x2, 0x100, 0xfa, 0x3, 0x2, 0x2, 0x2, 0x100, 
    0xfd, 0x3, 0x2, 0x2, 0x2, 0x101, 0x104, 0x3, 0x2, 0x2, 0x2, 0x102, 0x100, 
    0x3, 0x2, 0x2, 0x2, 0x102, 0x103, 0x3, 0x2, 0x2, 0x2, 0x103, 0x37, 0x3, 
    0x2, 0x2, 0x2, 0x104, 0x102, 0x3, 0x2, 0x2, 0x2, 0x105, 0x106, 0x8, 
    0x1d, 0x1, 0x2, 0x106, 0x107, 0x5, 0x36, 0x1c, 0x2, 0x107, 0x110, 0x3, 
    0x2, 0x2, 0x2, 0x108, 0x109, 0xc, 0x4, 0x2, 0x2, 0x109, 0x10a, 0x7, 
    0x2e, 0x2, 0x2, 0x10a, 0x10f, 0x5, 0x36, 0x1c, 0x2, 0x10b, 0x10c, 0xc, 
    0x3, 0x2, 0x2, 0x10c, 0x10d, 0x7, 0x31, 0x2, 0x2, 0x10d, 0x10f, 0x5, 
    0x36, 0x1c, 0x2, 0x10e, 0x108, 0x3, 0x2, 0x2, 0x2, 0x10e, 0x10b, 0x3, 
    0x2, 0x2, 0x2, 0x10f, 0x112, 0x3, 0x2, 0x2, 0x2, 0x110, 0x10e, 0x3, 
    0x2, 0x2, 0x2, 0x110, 0x111, 0x3, 0x2, 0x2, 0x2, 0x111, 0x39, 0x3, 0x2, 
    0x2, 0x2, 0x112, 0x110, 0x3, 0x2, 0x2, 0x2, 0x113, 0x114, 0x8, 0x1e, 
    0x1, 0x2, 0x114, 0x115, 0x5, 0x38, 0x1d, 0x2, 0x115, 0x11b, 0x3, 0x2, 
    0x2, 0x2, 0x116, 0x117, 0xc, 0x3, 0x2, 0x2, 0x117, 0x118, 0x7, 0x32, 
    0x2, 0x2, 0x118, 0x11a, 0x5, 0x38, 0x1d, 0x2, 0x119, 0x116, 0x3, 0x2, 
    0x2, 0x2, 0x11a, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x11b, 0x119, 0x3, 0x2, 
    0x2, 0x2, 0x11b, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x11c, 0x3b, 0x3, 0x2, 0x2, 
    0x2, 0x11d, 0x11b, 0x3, 0x2, 0x2, 0x2, 0x11e, 0x11f, 0x8, 0x1f, 0x1, 
    0x2, 0x11f, 0x120, 0x5, 0x3a, 0x1e, 0x2, 0x120, 0x126, 0x3, 0x2, 0x2, 
    0x2, 0x121, 0x122, 0xc, 0x3, 0x2, 0x2, 0x122, 0x123, 0x7, 0x33, 0x2, 
    0x2, 0x123, 0x125, 0x5, 0x3a, 0x1e, 0x2, 0x124, 0x121, 0x3, 0x2, 0x2, 
    0x2, 0x125, 0x128, 0x3, 0x2, 0x2, 0x2, 0x126, 0x124, 0x3, 0x2, 0x2, 
    0x2, 0x126, 0x127, 0x3, 0x2, 0x2, 0x2, 0x127, 0x3d, 0x3, 0x2, 0x2, 0x2, 
    0x128, 0x126, 0x3, 0x2, 0x2, 0x2, 0x129, 0x12f, 0x5, 0x3c, 0x1f, 0x2, 
    0x12a, 0x12b, 0x5, 0xe, 0x8, 0x2, 0x12b, 0x12c, 0x7, 0x2a, 0x2, 0x2, 
    0x12c, 0x12d, 0x5, 0x3c, 0x1f, 0x2, 0x12d, 0x12f, 0x3, 0x2, 0x2, 0x2, 
    0x12e, 0x129, 0x3, 0x2, 0x2, 0x2, 0x12e, 0x12a, 0x3, 0x2, 0x2, 0x2, 
    0x12f, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x130, 0x131, 0x5, 0x3e, 0x20, 0x2, 
    0x131, 0x41, 0x3, 0x2, 0x2, 0x2, 0x132, 0x137, 0x5, 0x10, 0x9, 0x2, 
    0x133, 0x134, 0x7, 0x38, 0x2, 0x2, 0x134, 0x136, 0x5, 0x10, 0x9, 0x2, 
    0x135, 0x133, 0x3, 0x2, 0x2, 0x2, 0x136, 0x139, 0x3, 0x2, 0x2, 0x2, 
    0x137, 0x135, 0x3, 0x2, 0x2, 0x2, 0x137, 0x138, 0x3, 0x2, 0x2, 0x2, 
    0x138, 0x43, 0x3, 0x2, 0x2, 0x2, 0x139, 0x137, 0x3, 0x2, 0x2, 0x2, 0x13a, 
    0x13b, 0x5, 0xa, 0x6, 0x2, 0x13b, 0x13c, 0x7, 0x3d, 0x2, 0x2, 0x13c, 
    0x148, 0x3, 0x2, 0x2, 0x2, 0x13d, 0x148, 0x5, 0x48, 0x25, 0x2, 0x13e, 
    0x148, 0x5, 0x4c, 0x27, 0x2, 0x13f, 0x148, 0x5, 0x4e, 0x28, 0x2, 0x140, 
    0x148, 0x5, 0x46, 0x24, 0x2, 0x141, 0x142, 0x5, 0x50, 0x29, 0x2, 0x142, 
    0x143, 0x7, 0x3d, 0x2, 0x2, 0x143, 0x148, 0x3, 0x2, 0x2, 0x2, 0x144, 
    0x145, 0x5, 0x52, 0x2a, 0x2, 0x145, 0x146, 0x7, 0x3d, 0x2, 0x2, 0x146, 
    0x148, 0x3, 0x2, 0x2, 0x2, 0x147, 0x13a, 0x3, 0x2, 0x2, 0x2, 0x147, 
    0x13d, 0x3, 0x2, 0x2, 0x2, 0x147, 0x13e, 0x3, 0x2, 0x2, 0x2, 0x147, 
    0x13f, 0x3, 0x2, 0x2, 0x2, 0x147, 0x140, 0x3, 0x2, 0x2, 0x2, 0x147, 
    0x141, 0x3, 0x2, 0x2, 0x2, 0x147, 0x144, 0x3, 0x2, 0x2, 0x2, 0x148, 
    0x45, 0x3, 0x2, 0x2, 0x2, 0x149, 0x14a, 0x7, 0x20, 0x2, 0x2, 0x14a, 
    0x14e, 0x7, 0x3d, 0x2, 0x2, 0x14b, 0x14d, 0x5, 0x44, 0x23, 0x2, 0x14c, 
    0x14b, 0x3, 0x2, 0x2, 0x2, 0x14d, 0x150, 0x3, 0x2, 0x2, 0x2, 0x14e, 
    0x14c, 0x3, 0x2, 0x2, 0x2, 0x14e, 0x14f, 0x3, 0x2, 0x2, 0x2, 0x14f, 
    0x151, 0x3, 0x2, 0x2, 0x2, 0x150, 0x14e, 0x3, 0x2, 0x2, 0x2, 0x151, 
    0x152, 0x7, 0x21, 0x2, 0x2, 0x152, 0x153, 0x7, 0x3d, 0x2, 0x2, 0x153, 
    0x47, 0x3, 0x2, 0x2, 0x2, 0x154, 0x155, 0x7, 0x22, 0x2, 0x2, 0x155, 
    0x156, 0x5, 0x42, 0x22, 0x2, 0x156, 0x15a, 0x7, 0x3d, 0x2, 0x2, 0x157, 
    0x159, 0x5, 0x44, 0x23, 0x2, 0x158, 0x157, 0x3, 0x2, 0x2, 0x2, 0x159, 
    0x15c, 0x3, 0x2, 0x2, 0x2, 0x15a, 0x158, 0x3, 0x2, 0x2, 0x2, 0x15a, 
    0x15b, 0x3, 0x2, 0x2, 0x2, 0x15b, 0x15d, 0x3, 0x2, 0x2, 0x2, 0x15c, 
    0x15a, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15e, 0x7, 0x23, 0x2, 0x2, 0x15e, 
    0x15f, 0x7, 0x3d, 0x2, 0x2, 0x15f, 0x49, 0x3, 0x2, 0x2, 0x2, 0x160, 
    0x161, 0x7, 0x25, 0x2, 0x2, 0x161, 0x165, 0x7, 0x3d, 0x2, 0x2, 0x162, 
    0x164, 0x5, 0x44, 0x23, 0x2, 0x163, 0x162, 0x3, 0x2, 0x2, 0x2, 0x164, 
    0x167, 0x3, 0x2, 0x2, 0x2, 0x165, 0x163, 0x3, 0x2, 0x2, 0x2, 0x165, 
    0x166, 0x3, 0x2, 0x2, 0x2, 0x166, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x167, 0x165, 
    0x3, 0x2, 0x2, 0x2, 0x168, 0x169, 0x7, 0x24, 0x2, 0x2, 0x169, 0x16a, 
    0x5, 0x40, 0x21, 0x2, 0x16a, 0x16e, 0x7, 0x3d, 0x2, 0x2, 0x16b, 0x16d, 
    0x5, 0x44, 0x23, 0x2, 0x16c, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x16d, 0x170, 
    0x3, 0x2, 0x2, 0x2, 0x16e, 0x16c, 0x3, 0x2, 0x2, 0x2, 0x16e, 0x16f, 
    0x3, 0x2, 0x2, 0x2, 0x16f, 0x171, 0x3, 0x2, 0x2, 0x2, 0x170, 0x16e, 
    0x3, 0x2, 0x2, 0x2, 0x171, 0x172, 0x5, 0x4a, 0x26, 0x2, 0x172, 0x173, 
    0x7, 0x26, 0x2, 0x2, 0x173, 0x174, 0x7, 0x3d, 0x2, 0x2, 0x174, 0x182, 
    0x3, 0x2, 0x2, 0x2, 0x175, 0x176, 0x7, 0x24, 0x2, 0x2, 0x176, 0x177, 
    0x5, 0x40, 0x21, 0x2, 0x177, 0x17b, 0x7, 0x3d, 0x2, 0x2, 0x178, 0x17a, 
    0x5, 0x44, 0x23, 0x2, 0x179, 0x178, 0x3, 0x2, 0x2, 0x2, 0x17a, 0x17d, 
    0x3, 0x2, 0x2, 0x2, 0x17b, 0x179, 0x3, 0x2, 0x2, 0x2, 0x17b, 0x17c, 
    0x3, 0x2, 0x2, 0x2, 0x17c, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x17d, 0x17b, 
    0x3, 0x2, 0x2, 0x2, 0x17e, 0x17f, 0x7, 0x26, 0x2, 0x2, 0x17f, 0x180, 
    0x7, 0x3d, 0x2, 0x2, 0x180, 0x182, 0x3, 0x2, 0x2, 0x2, 0x181, 0x168, 
    0x3, 0x2, 0x2, 0x2, 0x181, 0x175, 0x3, 0x2, 0x2, 0x2, 0x182, 0x4d, 0x3, 
    0x2, 0x2, 0x2, 0x183, 0x184, 0x7, 0x27, 0x2, 0x2, 0x184, 0x185, 0x5, 
    0x40, 0x21, 0x2, 0x185, 0x189, 0x7, 0x3d, 0x2, 0x2, 0x186, 0x188, 0x5, 
    0x44, 0x23, 0x2, 0x187, 0x186, 0x3, 0x2, 0x2, 0x2, 0x188, 0x18b, 0x3, 
    0x2, 0x2, 0x2, 0x189, 0x187, 0x3, 0x2, 0x2, 0x2, 0x189, 0x18a, 0x3, 
    0x2, 0x2, 0x2, 0x18a, 0x18c, 0x3, 0x2, 0x2, 0x2, 0x18b, 0x189, 0x3, 
    0x2, 0x2, 0x2, 0x18c, 0x18d, 0x7, 0x28, 0x2, 0x2, 0x18d, 0x18e, 0x7, 
    0x3d, 0x2, 0x2, 0x18e, 0x4f, 0x3, 0x2, 0x2, 0x2, 0x18f, 0x190, 0x7, 
    0x29, 0x2, 0x2, 0x190, 0x191, 0x5, 0x10, 0x9, 0x2, 0x191, 0x192, 0x7, 
    0x38, 0x2, 0x2, 0x192, 0x193, 0x5, 0xe, 0x8, 0x2, 0x193, 0x51, 0x3, 
    0x2, 0x2, 0x2, 0x194, 0x195, 0x5, 0x40, 0x21, 0x2, 0x195, 0x53, 0x3, 
    0x2, 0x2, 0x2, 0x196, 0x197, 0x9, 0x6, 0x2, 0x2, 0x197, 0x55, 0x3, 0x2, 
    0x2, 0x2, 0x1a, 0x5a, 0x6f, 0xca, 0xd3, 0xde, 0xe0, 0xec, 0xee, 0x100, 
    0x102, 0x10e, 0x110, 0x11b, 0x126, 0x12e, 0x137, 0x147, 0x14e, 0x15a, 
    0x165, 0x16e, 0x17b, 0x181, 0x189, 
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
