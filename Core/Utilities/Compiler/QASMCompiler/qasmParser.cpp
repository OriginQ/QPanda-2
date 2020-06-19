
// Generated from .\qasm.g4 by ANTLR 4.7.2


#include "Core/Utilities/Compiler/QASMCompiler/qasmListener.h"
#include "Core/Utilities/Compiler/QASMCompiler/qasmVisitor.h"

#include "Core/Utilities/Compiler/QASMCompiler/qasmParser.h"


using namespace antlrcpp;
using namespace antlr4;

qasmParser::qasmParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

qasmParser::~qasmParser() {
  delete _interpreter;
}

std::string qasmParser::getGrammarFileName() const {
  return "qasm.g4";
}

const std::vector<std::string>& qasmParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& qasmParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- MainprogramContext ------------------------------------------------------------------

qasmParser::MainprogramContext::MainprogramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasmParser::Head_declContext* qasmParser::MainprogramContext::head_decl() {
  return getRuleContext<qasmParser::Head_declContext>(0);
}

std::vector<qasmParser::StatementContext *> qasmParser::MainprogramContext::statement() {
  return getRuleContexts<qasmParser::StatementContext>();
}

qasmParser::StatementContext* qasmParser::MainprogramContext::statement(size_t i) {
  return getRuleContext<qasmParser::StatementContext>(i);
}


size_t qasmParser::MainprogramContext::getRuleIndex() const {
  return qasmParser::RuleMainprogram;
}

void qasmParser::MainprogramContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMainprogram(this);
}

void qasmParser::MainprogramContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMainprogram(this);
}


antlrcpp::Any qasmParser::MainprogramContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitMainprogram(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::MainprogramContext* qasmParser::mainprogram() {
  MainprogramContext *_localctx = _tracker.createInstance<MainprogramContext>(_ctx, getState());
  enterRule(_localctx, 0, qasmParser::RuleMainprogram);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(50);
    head_decl();
    setState(54);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << qasmParser::OPAQUE_KEY)
      | (1ULL << qasmParser::QREG_KEY)
      | (1ULL << qasmParser::CREG_KEY)
      | (1ULL << qasmParser::BARRIER_KEY)
      | (1ULL << qasmParser::IF_KEY)
      | (1ULL << qasmParser::MEASURE_KEY)
      | (1ULL << qasmParser::RESET_KEY)
      | (1ULL << qasmParser::GATE_KEY)
      | (1ULL << qasmParser::U_GATE_KEY)
      | (1ULL << qasmParser::CX_GATE_KEY)
      | (1ULL << qasmParser::IDENTIFIER))) != 0)) {
      setState(51);
      statement();
      setState(56);
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

//----------------- Head_declContext ------------------------------------------------------------------

qasmParser::Head_declContext::Head_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasmParser::Version_declContext* qasmParser::Head_declContext::version_decl() {
  return getRuleContext<qasmParser::Version_declContext>(0);
}

qasmParser::Include_declContext* qasmParser::Head_declContext::include_decl() {
  return getRuleContext<qasmParser::Include_declContext>(0);
}


size_t qasmParser::Head_declContext::getRuleIndex() const {
  return qasmParser::RuleHead_decl;
}

void qasmParser::Head_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterHead_decl(this);
}

void qasmParser::Head_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitHead_decl(this);
}


antlrcpp::Any qasmParser::Head_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitHead_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Head_declContext* qasmParser::head_decl() {
  Head_declContext *_localctx = _tracker.createInstance<Head_declContext>(_ctx, getState());
  enterRule(_localctx, 2, qasmParser::RuleHead_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(61);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(57);
      version_decl();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(58);
      version_decl();
      setState(59);
      include_decl();
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

//----------------- Version_declContext ------------------------------------------------------------------

qasmParser::Version_declContext::Version_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::Version_declContext::OPENQASM_KEY() {
  return getToken(qasmParser::OPENQASM_KEY, 0);
}

qasmParser::DecimalContext* qasmParser::Version_declContext::decimal() {
  return getRuleContext<qasmParser::DecimalContext>(0);
}

tree::TerminalNode* qasmParser::Version_declContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}


size_t qasmParser::Version_declContext::getRuleIndex() const {
  return qasmParser::RuleVersion_decl;
}

void qasmParser::Version_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVersion_decl(this);
}

void qasmParser::Version_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVersion_decl(this);
}


antlrcpp::Any qasmParser::Version_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitVersion_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Version_declContext* qasmParser::version_decl() {
  Version_declContext *_localctx = _tracker.createInstance<Version_declContext>(_ctx, getState());
  enterRule(_localctx, 4, qasmParser::RuleVersion_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(63);
    match(qasmParser::OPENQASM_KEY);
    setState(64);
    decimal();
    setState(65);
    match(qasmParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Include_declContext ------------------------------------------------------------------

qasmParser::Include_declContext::Include_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::Include_declContext::INCLUDE_KEY() {
  return getToken(qasmParser::INCLUDE_KEY, 0);
}

std::vector<tree::TerminalNode *> qasmParser::Include_declContext::DQM() {
  return getTokens(qasmParser::DQM);
}

tree::TerminalNode* qasmParser::Include_declContext::DQM(size_t i) {
  return getToken(qasmParser::DQM, i);
}

qasmParser::FilenameContext* qasmParser::Include_declContext::filename() {
  return getRuleContext<qasmParser::FilenameContext>(0);
}

tree::TerminalNode* qasmParser::Include_declContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}


size_t qasmParser::Include_declContext::getRuleIndex() const {
  return qasmParser::RuleInclude_decl;
}

void qasmParser::Include_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInclude_decl(this);
}

void qasmParser::Include_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInclude_decl(this);
}


antlrcpp::Any qasmParser::Include_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitInclude_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Include_declContext* qasmParser::include_decl() {
  Include_declContext *_localctx = _tracker.createInstance<Include_declContext>(_ctx, getState());
  enterRule(_localctx, 6, qasmParser::RuleInclude_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(67);
    match(qasmParser::INCLUDE_KEY);
    setState(68);
    match(qasmParser::DQM);
    setState(69);
    filename();
    setState(70);
    match(qasmParser::DQM);
    setState(71);
    match(qasmParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StatementContext ------------------------------------------------------------------

qasmParser::StatementContext::StatementContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasmParser::Reg_declContext* qasmParser::StatementContext::reg_decl() {
  return getRuleContext<qasmParser::Reg_declContext>(0);
}

qasmParser::Gate_declContext* qasmParser::StatementContext::gate_decl() {
  return getRuleContext<qasmParser::Gate_declContext>(0);
}

qasmParser::Opaque_declContext* qasmParser::StatementContext::opaque_decl() {
  return getRuleContext<qasmParser::Opaque_declContext>(0);
}

qasmParser::If_declContext* qasmParser::StatementContext::if_decl() {
  return getRuleContext<qasmParser::If_declContext>(0);
}

qasmParser::Barrier_declContext* qasmParser::StatementContext::barrier_decl() {
  return getRuleContext<qasmParser::Barrier_declContext>(0);
}

qasmParser::QopContext* qasmParser::StatementContext::qop() {
  return getRuleContext<qasmParser::QopContext>(0);
}


size_t qasmParser::StatementContext::getRuleIndex() const {
  return qasmParser::RuleStatement;
}

void qasmParser::StatementContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterStatement(this);
}

void qasmParser::StatementContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitStatement(this);
}


antlrcpp::Any qasmParser::StatementContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitStatement(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::StatementContext* qasmParser::statement() {
  StatementContext *_localctx = _tracker.createInstance<StatementContext>(_ctx, getState());
  enterRule(_localctx, 8, qasmParser::RuleStatement);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(79);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasmParser::QREG_KEY:
      case qasmParser::CREG_KEY: {
        enterOuterAlt(_localctx, 1);
        setState(73);
        reg_decl();
        break;
      }

      case qasmParser::GATE_KEY: {
        enterOuterAlt(_localctx, 2);
        setState(74);
        gate_decl();
        break;
      }

      case qasmParser::OPAQUE_KEY: {
        enterOuterAlt(_localctx, 3);
        setState(75);
        opaque_decl();
        break;
      }

      case qasmParser::IF_KEY: {
        enterOuterAlt(_localctx, 4);
        setState(76);
        if_decl();
        break;
      }

      case qasmParser::BARRIER_KEY: {
        enterOuterAlt(_localctx, 5);
        setState(77);
        barrier_decl();
        break;
      }

      case qasmParser::MEASURE_KEY:
      case qasmParser::RESET_KEY:
      case qasmParser::U_GATE_KEY:
      case qasmParser::CX_GATE_KEY:
      case qasmParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 6);
        setState(78);
        qop();
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

//----------------- Reg_declContext ------------------------------------------------------------------

qasmParser::Reg_declContext::Reg_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::Reg_declContext::QREG_KEY() {
  return getToken(qasmParser::QREG_KEY, 0);
}

qasmParser::IdContext* qasmParser::Reg_declContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

tree::TerminalNode* qasmParser::Reg_declContext::LBRACKET() {
  return getToken(qasmParser::LBRACKET, 0);
}

qasmParser::IntegerContext* qasmParser::Reg_declContext::integer() {
  return getRuleContext<qasmParser::IntegerContext>(0);
}

tree::TerminalNode* qasmParser::Reg_declContext::RBRACKET() {
  return getToken(qasmParser::RBRACKET, 0);
}

tree::TerminalNode* qasmParser::Reg_declContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}

tree::TerminalNode* qasmParser::Reg_declContext::CREG_KEY() {
  return getToken(qasmParser::CREG_KEY, 0);
}


size_t qasmParser::Reg_declContext::getRuleIndex() const {
  return qasmParser::RuleReg_decl;
}

void qasmParser::Reg_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReg_decl(this);
}

void qasmParser::Reg_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReg_decl(this);
}


antlrcpp::Any qasmParser::Reg_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitReg_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Reg_declContext* qasmParser::reg_decl() {
  Reg_declContext *_localctx = _tracker.createInstance<Reg_declContext>(_ctx, getState());
  enterRule(_localctx, 10, qasmParser::RuleReg_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(95);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasmParser::QREG_KEY: {
        enterOuterAlt(_localctx, 1);
        setState(81);
        match(qasmParser::QREG_KEY);
        setState(82);
        id();
        setState(83);
        match(qasmParser::LBRACKET);
        setState(84);
        integer();
        setState(85);
        match(qasmParser::RBRACKET);
        setState(86);
        match(qasmParser::SEMI);
        break;
      }

      case qasmParser::CREG_KEY: {
        enterOuterAlt(_localctx, 2);
        setState(88);
        match(qasmParser::CREG_KEY);
        setState(89);
        id();
        setState(90);
        match(qasmParser::LBRACKET);
        setState(91);
        integer();
        setState(92);
        match(qasmParser::RBRACKET);
        setState(93);
        match(qasmParser::SEMI);
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

//----------------- Opaque_declContext ------------------------------------------------------------------

qasmParser::Opaque_declContext::Opaque_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::Opaque_declContext::OPAQUE_KEY() {
  return getToken(qasmParser::OPAQUE_KEY, 0);
}

qasmParser::IdContext* qasmParser::Opaque_declContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

std::vector<qasmParser::IdlistContext *> qasmParser::Opaque_declContext::idlist() {
  return getRuleContexts<qasmParser::IdlistContext>();
}

qasmParser::IdlistContext* qasmParser::Opaque_declContext::idlist(size_t i) {
  return getRuleContext<qasmParser::IdlistContext>(i);
}

tree::TerminalNode* qasmParser::Opaque_declContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}

tree::TerminalNode* qasmParser::Opaque_declContext::LPAREN() {
  return getToken(qasmParser::LPAREN, 0);
}

tree::TerminalNode* qasmParser::Opaque_declContext::RPAREN() {
  return getToken(qasmParser::RPAREN, 0);
}


size_t qasmParser::Opaque_declContext::getRuleIndex() const {
  return qasmParser::RuleOpaque_decl;
}

void qasmParser::Opaque_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOpaque_decl(this);
}

void qasmParser::Opaque_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOpaque_decl(this);
}


antlrcpp::Any qasmParser::Opaque_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitOpaque_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Opaque_declContext* qasmParser::opaque_decl() {
  Opaque_declContext *_localctx = _tracker.createInstance<Opaque_declContext>(_ctx, getState());
  enterRule(_localctx, 12, qasmParser::RuleOpaque_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(117);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(97);
      match(qasmParser::OPAQUE_KEY);
      setState(98);
      id();
      setState(99);
      idlist();
      setState(100);
      match(qasmParser::SEMI);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(102);
      match(qasmParser::OPAQUE_KEY);
      setState(103);
      id();
      setState(104);
      match(qasmParser::LPAREN);
      setState(105);
      match(qasmParser::RPAREN);
      setState(106);
      idlist();
      setState(107);
      match(qasmParser::SEMI);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(109);
      match(qasmParser::OPAQUE_KEY);
      setState(110);
      id();
      setState(111);
      match(qasmParser::LPAREN);
      setState(112);
      idlist();
      setState(113);
      match(qasmParser::RPAREN);
      setState(114);
      idlist();
      setState(115);
      match(qasmParser::SEMI);
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

//----------------- If_declContext ------------------------------------------------------------------

qasmParser::If_declContext::If_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::If_declContext::IF_KEY() {
  return getToken(qasmParser::IF_KEY, 0);
}

tree::TerminalNode* qasmParser::If_declContext::LPAREN() {
  return getToken(qasmParser::LPAREN, 0);
}

qasmParser::IdContext* qasmParser::If_declContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

tree::TerminalNode* qasmParser::If_declContext::EQ() {
  return getToken(qasmParser::EQ, 0);
}

qasmParser::IntegerContext* qasmParser::If_declContext::integer() {
  return getRuleContext<qasmParser::IntegerContext>(0);
}

tree::TerminalNode* qasmParser::If_declContext::RPAREN() {
  return getToken(qasmParser::RPAREN, 0);
}

qasmParser::QopContext* qasmParser::If_declContext::qop() {
  return getRuleContext<qasmParser::QopContext>(0);
}


size_t qasmParser::If_declContext::getRuleIndex() const {
  return qasmParser::RuleIf_decl;
}

void qasmParser::If_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIf_decl(this);
}

void qasmParser::If_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIf_decl(this);
}


antlrcpp::Any qasmParser::If_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitIf_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::If_declContext* qasmParser::if_decl() {
  If_declContext *_localctx = _tracker.createInstance<If_declContext>(_ctx, getState());
  enterRule(_localctx, 14, qasmParser::RuleIf_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(119);
    match(qasmParser::IF_KEY);
    setState(120);
    match(qasmParser::LPAREN);
    setState(121);
    id();
    setState(122);
    match(qasmParser::EQ);
    setState(123);
    integer();
    setState(124);
    match(qasmParser::RPAREN);
    setState(125);
    qop();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Barrier_declContext ------------------------------------------------------------------

qasmParser::Barrier_declContext::Barrier_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::Barrier_declContext::BARRIER_KEY() {
  return getToken(qasmParser::BARRIER_KEY, 0);
}

qasmParser::AnylistContext* qasmParser::Barrier_declContext::anylist() {
  return getRuleContext<qasmParser::AnylistContext>(0);
}

tree::TerminalNode* qasmParser::Barrier_declContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}


size_t qasmParser::Barrier_declContext::getRuleIndex() const {
  return qasmParser::RuleBarrier_decl;
}

void qasmParser::Barrier_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBarrier_decl(this);
}

void qasmParser::Barrier_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBarrier_decl(this);
}


antlrcpp::Any qasmParser::Barrier_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitBarrier_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Barrier_declContext* qasmParser::barrier_decl() {
  Barrier_declContext *_localctx = _tracker.createInstance<Barrier_declContext>(_ctx, getState());
  enterRule(_localctx, 16, qasmParser::RuleBarrier_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(127);
    match(qasmParser::BARRIER_KEY);
    setState(128);
    anylist();
    setState(129);
    match(qasmParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Gate_declContext ------------------------------------------------------------------

qasmParser::Gate_declContext::Gate_declContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::Gate_declContext::GATE_KEY() {
  return getToken(qasmParser::GATE_KEY, 0);
}

qasmParser::IdContext* qasmParser::Gate_declContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

std::vector<qasmParser::IdlistContext *> qasmParser::Gate_declContext::idlist() {
  return getRuleContexts<qasmParser::IdlistContext>();
}

qasmParser::IdlistContext* qasmParser::Gate_declContext::idlist(size_t i) {
  return getRuleContext<qasmParser::IdlistContext>(i);
}

tree::TerminalNode* qasmParser::Gate_declContext::LBRACE() {
  return getToken(qasmParser::LBRACE, 0);
}

qasmParser::GoplistContext* qasmParser::Gate_declContext::goplist() {
  return getRuleContext<qasmParser::GoplistContext>(0);
}

tree::TerminalNode* qasmParser::Gate_declContext::RBRACE() {
  return getToken(qasmParser::RBRACE, 0);
}

tree::TerminalNode* qasmParser::Gate_declContext::LPAREN() {
  return getToken(qasmParser::LPAREN, 0);
}

tree::TerminalNode* qasmParser::Gate_declContext::RPAREN() {
  return getToken(qasmParser::RPAREN, 0);
}


size_t qasmParser::Gate_declContext::getRuleIndex() const {
  return qasmParser::RuleGate_decl;
}

void qasmParser::Gate_declContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGate_decl(this);
}

void qasmParser::Gate_declContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGate_decl(this);
}


antlrcpp::Any qasmParser::Gate_declContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitGate_decl(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Gate_declContext* qasmParser::gate_decl() {
  Gate_declContext *_localctx = _tracker.createInstance<Gate_declContext>(_ctx, getState());
  enterRule(_localctx, 18, qasmParser::RuleGate_decl);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(180);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(131);
      match(qasmParser::GATE_KEY);
      setState(132);
      id();
      setState(133);
      idlist();
      setState(134);
      match(qasmParser::LBRACE);
      setState(135);
      goplist();
      setState(136);
      match(qasmParser::RBRACE);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(138);
      match(qasmParser::GATE_KEY);
      setState(139);
      id();
      setState(140);
      match(qasmParser::LPAREN);
      setState(141);
      match(qasmParser::RPAREN);
      setState(142);
      idlist();
      setState(143);
      match(qasmParser::LBRACE);
      setState(144);
      goplist();
      setState(145);
      match(qasmParser::RBRACE);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(147);
      match(qasmParser::GATE_KEY);
      setState(148);
      id();
      setState(149);
      match(qasmParser::LPAREN);
      setState(150);
      idlist();
      setState(151);
      match(qasmParser::RPAREN);
      setState(152);
      idlist();
      setState(153);
      match(qasmParser::LBRACE);
      setState(154);
      goplist();
      setState(155);
      match(qasmParser::RBRACE);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(157);
      match(qasmParser::GATE_KEY);
      setState(158);
      id();
      setState(159);
      idlist();
      setState(160);
      match(qasmParser::LBRACE);
      setState(161);
      match(qasmParser::RBRACE);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(163);
      match(qasmParser::GATE_KEY);
      setState(164);
      id();
      setState(165);
      match(qasmParser::LPAREN);
      setState(166);
      match(qasmParser::RPAREN);
      setState(167);
      idlist();
      setState(168);
      match(qasmParser::LBRACE);
      setState(169);
      match(qasmParser::RBRACE);
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(171);
      match(qasmParser::GATE_KEY);
      setState(172);
      id();
      setState(173);
      match(qasmParser::LPAREN);
      setState(174);
      idlist();
      setState(175);
      match(qasmParser::RPAREN);
      setState(176);
      idlist();
      setState(177);
      match(qasmParser::LBRACE);
      setState(178);
      match(qasmParser::RBRACE);
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

//----------------- GoplistContext ------------------------------------------------------------------

qasmParser::GoplistContext::GoplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasmParser::UopContext *> qasmParser::GoplistContext::uop() {
  return getRuleContexts<qasmParser::UopContext>();
}

qasmParser::UopContext* qasmParser::GoplistContext::uop(size_t i) {
  return getRuleContext<qasmParser::UopContext>(i);
}

std::vector<qasmParser::BopContext *> qasmParser::GoplistContext::bop() {
  return getRuleContexts<qasmParser::BopContext>();
}

qasmParser::BopContext* qasmParser::GoplistContext::bop(size_t i) {
  return getRuleContext<qasmParser::BopContext>(i);
}


size_t qasmParser::GoplistContext::getRuleIndex() const {
  return qasmParser::RuleGoplist;
}

void qasmParser::GoplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGoplist(this);
}

void qasmParser::GoplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGoplist(this);
}


antlrcpp::Any qasmParser::GoplistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitGoplist(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::GoplistContext* qasmParser::goplist() {
  GoplistContext *_localctx = _tracker.createInstance<GoplistContext>(_ctx, getState());
  enterRule(_localctx, 20, qasmParser::RuleGoplist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(194);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(185);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while ((((_la & ~ 0x3fULL) == 0) &&
        ((1ULL << _la) & ((1ULL << qasmParser::U_GATE_KEY)
        | (1ULL << qasmParser::CX_GATE_KEY)
        | (1ULL << qasmParser::IDENTIFIER))) != 0)) {
        setState(182);
        uop();
        setState(187);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(191);
      _errHandler->sync(this);
      _la = _input->LA(1);
      while (_la == qasmParser::BARRIER_KEY) {
        setState(188);
        bop();
        setState(193);
        _errHandler->sync(this);
        _la = _input->LA(1);
      }
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

//----------------- BopContext ------------------------------------------------------------------

qasmParser::BopContext::BopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::BopContext::BARRIER_KEY() {
  return getToken(qasmParser::BARRIER_KEY, 0);
}

qasmParser::IdlistContext* qasmParser::BopContext::idlist() {
  return getRuleContext<qasmParser::IdlistContext>(0);
}

tree::TerminalNode* qasmParser::BopContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}


size_t qasmParser::BopContext::getRuleIndex() const {
  return qasmParser::RuleBop;
}

void qasmParser::BopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBop(this);
}

void qasmParser::BopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBop(this);
}


antlrcpp::Any qasmParser::BopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitBop(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::BopContext* qasmParser::bop() {
  BopContext *_localctx = _tracker.createInstance<BopContext>(_ctx, getState());
  enterRule(_localctx, 22, qasmParser::RuleBop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(196);
    match(qasmParser::BARRIER_KEY);
    setState(197);
    idlist();
    setState(198);
    match(qasmParser::SEMI);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QopContext ------------------------------------------------------------------

qasmParser::QopContext::QopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasmParser::UopContext* qasmParser::QopContext::uop() {
  return getRuleContext<qasmParser::UopContext>(0);
}

tree::TerminalNode* qasmParser::QopContext::MEASURE_KEY() {
  return getToken(qasmParser::MEASURE_KEY, 0);
}

std::vector<qasmParser::ArgumentContext *> qasmParser::QopContext::argument() {
  return getRuleContexts<qasmParser::ArgumentContext>();
}

qasmParser::ArgumentContext* qasmParser::QopContext::argument(size_t i) {
  return getRuleContext<qasmParser::ArgumentContext>(i);
}

tree::TerminalNode* qasmParser::QopContext::ARROW() {
  return getToken(qasmParser::ARROW, 0);
}

tree::TerminalNode* qasmParser::QopContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}

tree::TerminalNode* qasmParser::QopContext::RESET_KEY() {
  return getToken(qasmParser::RESET_KEY, 0);
}


size_t qasmParser::QopContext::getRuleIndex() const {
  return qasmParser::RuleQop;
}

void qasmParser::QopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQop(this);
}

void qasmParser::QopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQop(this);
}


antlrcpp::Any qasmParser::QopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitQop(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::QopContext* qasmParser::qop() {
  QopContext *_localctx = _tracker.createInstance<QopContext>(_ctx, getState());
  enterRule(_localctx, 24, qasmParser::RuleQop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(211);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasmParser::U_GATE_KEY:
      case qasmParser::CX_GATE_KEY:
      case qasmParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 1);
        setState(200);
        uop();
        break;
      }

      case qasmParser::MEASURE_KEY: {
        enterOuterAlt(_localctx, 2);
        setState(201);
        match(qasmParser::MEASURE_KEY);
        setState(202);
        argument();
        setState(203);
        match(qasmParser::ARROW);
        setState(204);
        argument();
        setState(205);
        match(qasmParser::SEMI);
        break;
      }

      case qasmParser::RESET_KEY: {
        enterOuterAlt(_localctx, 3);
        setState(207);
        match(qasmParser::RESET_KEY);
        setState(208);
        argument();
        setState(209);
        match(qasmParser::SEMI);
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

//----------------- UopContext ------------------------------------------------------------------

qasmParser::UopContext::UopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::UopContext::U_GATE_KEY() {
  return getToken(qasmParser::U_GATE_KEY, 0);
}

tree::TerminalNode* qasmParser::UopContext::LPAREN() {
  return getToken(qasmParser::LPAREN, 0);
}

qasmParser::ExplistContext* qasmParser::UopContext::explist() {
  return getRuleContext<qasmParser::ExplistContext>(0);
}

tree::TerminalNode* qasmParser::UopContext::RPAREN() {
  return getToken(qasmParser::RPAREN, 0);
}

std::vector<qasmParser::ArgumentContext *> qasmParser::UopContext::argument() {
  return getRuleContexts<qasmParser::ArgumentContext>();
}

qasmParser::ArgumentContext* qasmParser::UopContext::argument(size_t i) {
  return getRuleContext<qasmParser::ArgumentContext>(i);
}

tree::TerminalNode* qasmParser::UopContext::SEMI() {
  return getToken(qasmParser::SEMI, 0);
}

tree::TerminalNode* qasmParser::UopContext::CX_GATE_KEY() {
  return getToken(qasmParser::CX_GATE_KEY, 0);
}

tree::TerminalNode* qasmParser::UopContext::COMMA() {
  return getToken(qasmParser::COMMA, 0);
}

qasmParser::IdContext* qasmParser::UopContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

qasmParser::AnylistContext* qasmParser::UopContext::anylist() {
  return getRuleContext<qasmParser::AnylistContext>(0);
}


size_t qasmParser::UopContext::getRuleIndex() const {
  return qasmParser::RuleUop;
}

void qasmParser::UopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterUop(this);
}

void qasmParser::UopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitUop(this);
}


antlrcpp::Any qasmParser::UopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitUop(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::UopContext* qasmParser::uop() {
  UopContext *_localctx = _tracker.createInstance<UopContext>(_ctx, getState());
  enterRule(_localctx, 26, qasmParser::RuleUop);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(243);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(213);
      match(qasmParser::U_GATE_KEY);
      setState(214);
      match(qasmParser::LPAREN);
      setState(215);
      explist();
      setState(216);
      match(qasmParser::RPAREN);
      setState(217);
      argument();
      setState(218);
      match(qasmParser::SEMI);
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(220);
      match(qasmParser::CX_GATE_KEY);
      setState(221);
      argument();
      setState(222);
      match(qasmParser::COMMA);
      setState(223);
      argument();
      setState(224);
      match(qasmParser::SEMI);
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(226);
      id();
      setState(227);
      anylist();
      setState(228);
      match(qasmParser::SEMI);
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(230);
      id();
      setState(231);
      match(qasmParser::LPAREN);
      setState(232);
      match(qasmParser::RPAREN);
      setState(233);
      anylist();
      setState(234);
      match(qasmParser::SEMI);
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(236);
      id();
      setState(237);
      match(qasmParser::LPAREN);
      setState(238);
      explist();
      setState(239);
      match(qasmParser::RPAREN);
      setState(240);
      anylist();
      setState(241);
      match(qasmParser::SEMI);
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

//----------------- AnylistContext ------------------------------------------------------------------

qasmParser::AnylistContext::AnylistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasmParser::Id_indexContext *> qasmParser::AnylistContext::id_index() {
  return getRuleContexts<qasmParser::Id_indexContext>();
}

qasmParser::Id_indexContext* qasmParser::AnylistContext::id_index(size_t i) {
  return getRuleContext<qasmParser::Id_indexContext>(i);
}

std::vector<qasmParser::IdContext *> qasmParser::AnylistContext::id() {
  return getRuleContexts<qasmParser::IdContext>();
}

qasmParser::IdContext* qasmParser::AnylistContext::id(size_t i) {
  return getRuleContext<qasmParser::IdContext>(i);
}

std::vector<tree::TerminalNode *> qasmParser::AnylistContext::COMMA() {
  return getTokens(qasmParser::COMMA);
}

tree::TerminalNode* qasmParser::AnylistContext::COMMA(size_t i) {
  return getToken(qasmParser::COMMA, i);
}


size_t qasmParser::AnylistContext::getRuleIndex() const {
  return qasmParser::RuleAnylist;
}

void qasmParser::AnylistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterAnylist(this);
}

void qasmParser::AnylistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitAnylist(this);
}


antlrcpp::Any qasmParser::AnylistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitAnylist(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::AnylistContext* qasmParser::anylist() {
  AnylistContext *_localctx = _tracker.createInstance<AnylistContext>(_ctx, getState());
  enterRule(_localctx, 28, qasmParser::RuleAnylist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(260);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasmParser::SEMI: {
        enterOuterAlt(_localctx, 1);

        break;
      }

      case qasmParser::IDENTIFIER: {
        enterOuterAlt(_localctx, 2);
        setState(248);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
        case 1: {
          setState(246);
          id_index();
          break;
        }

        case 2: {
          setState(247);
          id();
          break;
        }

        }
        setState(257);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == qasmParser::COMMA) {
          setState(250);
          match(qasmParser::COMMA);
          setState(253);
          _errHandler->sync(this);
          switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
          case 1: {
            setState(251);
            id_index();
            break;
          }

          case 2: {
            setState(252);
            id();
            break;
          }

          }
          setState(259);
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

//----------------- IdlistContext ------------------------------------------------------------------

qasmParser::IdlistContext::IdlistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasmParser::IdContext *> qasmParser::IdlistContext::id() {
  return getRuleContexts<qasmParser::IdContext>();
}

qasmParser::IdContext* qasmParser::IdlistContext::id(size_t i) {
  return getRuleContext<qasmParser::IdContext>(i);
}

std::vector<tree::TerminalNode *> qasmParser::IdlistContext::COMMA() {
  return getTokens(qasmParser::COMMA);
}

tree::TerminalNode* qasmParser::IdlistContext::COMMA(size_t i) {
  return getToken(qasmParser::COMMA, i);
}


size_t qasmParser::IdlistContext::getRuleIndex() const {
  return qasmParser::RuleIdlist;
}

void qasmParser::IdlistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdlist(this);
}

void qasmParser::IdlistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdlist(this);
}


antlrcpp::Any qasmParser::IdlistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitIdlist(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::IdlistContext* qasmParser::idlist() {
  IdlistContext *_localctx = _tracker.createInstance<IdlistContext>(_ctx, getState());
  enterRule(_localctx, 30, qasmParser::RuleIdlist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(262);
    id();
    setState(267);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == qasmParser::COMMA) {
      setState(263);
      match(qasmParser::COMMA);
      setState(264);
      id();
      setState(269);
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

//----------------- Id_indexContext ------------------------------------------------------------------

qasmParser::Id_indexContext::Id_indexContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasmParser::IdContext* qasmParser::Id_indexContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

tree::TerminalNode* qasmParser::Id_indexContext::LBRACKET() {
  return getToken(qasmParser::LBRACKET, 0);
}

qasmParser::IntegerContext* qasmParser::Id_indexContext::integer() {
  return getRuleContext<qasmParser::IntegerContext>(0);
}

tree::TerminalNode* qasmParser::Id_indexContext::RBRACKET() {
  return getToken(qasmParser::RBRACKET, 0);
}


size_t qasmParser::Id_indexContext::getRuleIndex() const {
  return qasmParser::RuleId_index;
}

void qasmParser::Id_indexContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterId_index(this);
}

void qasmParser::Id_indexContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitId_index(this);
}


antlrcpp::Any qasmParser::Id_indexContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitId_index(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::Id_indexContext* qasmParser::id_index() {
  Id_indexContext *_localctx = _tracker.createInstance<Id_indexContext>(_ctx, getState());
  enterRule(_localctx, 32, qasmParser::RuleId_index);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(270);
    id();
    setState(271);
    match(qasmParser::LBRACKET);
    setState(272);
    integer();
    setState(273);
    match(qasmParser::RBRACKET);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArgumentContext ------------------------------------------------------------------

qasmParser::ArgumentContext::ArgumentContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasmParser::IdContext* qasmParser::ArgumentContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

tree::TerminalNode* qasmParser::ArgumentContext::LBRACKET() {
  return getToken(qasmParser::LBRACKET, 0);
}

qasmParser::IntegerContext* qasmParser::ArgumentContext::integer() {
  return getRuleContext<qasmParser::IntegerContext>(0);
}

tree::TerminalNode* qasmParser::ArgumentContext::RBRACKET() {
  return getToken(qasmParser::RBRACKET, 0);
}


size_t qasmParser::ArgumentContext::getRuleIndex() const {
  return qasmParser::RuleArgument;
}

void qasmParser::ArgumentContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArgument(this);
}

void qasmParser::ArgumentContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArgument(this);
}


antlrcpp::Any qasmParser::ArgumentContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitArgument(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::ArgumentContext* qasmParser::argument() {
  ArgumentContext *_localctx = _tracker.createInstance<ArgumentContext>(_ctx, getState());
  enterRule(_localctx, 34, qasmParser::RuleArgument);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(281);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(275);
      id();
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(276);
      id();
      setState(277);
      match(qasmParser::LBRACKET);
      setState(278);
      integer();
      setState(279);
      match(qasmParser::RBRACKET);
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

//----------------- ExplistContext ------------------------------------------------------------------

qasmParser::ExplistContext::ExplistContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<qasmParser::ExpContext *> qasmParser::ExplistContext::exp() {
  return getRuleContexts<qasmParser::ExpContext>();
}

qasmParser::ExpContext* qasmParser::ExplistContext::exp(size_t i) {
  return getRuleContext<qasmParser::ExpContext>(i);
}

std::vector<tree::TerminalNode *> qasmParser::ExplistContext::COMMA() {
  return getTokens(qasmParser::COMMA);
}

tree::TerminalNode* qasmParser::ExplistContext::COMMA(size_t i) {
  return getToken(qasmParser::COMMA, i);
}


size_t qasmParser::ExplistContext::getRuleIndex() const {
  return qasmParser::RuleExplist;
}

void qasmParser::ExplistContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExplist(this);
}

void qasmParser::ExplistContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExplist(this);
}


antlrcpp::Any qasmParser::ExplistContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitExplist(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::ExplistContext* qasmParser::explist() {
  ExplistContext *_localctx = _tracker.createInstance<ExplistContext>(_ctx, getState());
  enterRule(_localctx, 36, qasmParser::RuleExplist);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(283);
    exp(0);
    setState(288);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while (_la == qasmParser::COMMA) {
      setState(284);
      match(qasmParser::COMMA);
      setState(285);
      exp(0);
      setState(290);
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

qasmParser::ExpContext::ExpContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

qasmParser::IdContext* qasmParser::ExpContext::id() {
  return getRuleContext<qasmParser::IdContext>(0);
}

qasmParser::RealContext* qasmParser::ExpContext::real() {
  return getRuleContext<qasmParser::RealContext>(0);
}

qasmParser::DecimalContext* qasmParser::ExpContext::decimal() {
  return getRuleContext<qasmParser::DecimalContext>(0);
}

qasmParser::IntegerContext* qasmParser::ExpContext::integer() {
  return getRuleContext<qasmParser::IntegerContext>(0);
}

tree::TerminalNode* qasmParser::ExpContext::PI_KEY() {
  return getToken(qasmParser::PI_KEY, 0);
}

tree::TerminalNode* qasmParser::ExpContext::LPAREN() {
  return getToken(qasmParser::LPAREN, 0);
}

std::vector<qasmParser::ExpContext *> qasmParser::ExpContext::exp() {
  return getRuleContexts<qasmParser::ExpContext>();
}

qasmParser::ExpContext* qasmParser::ExpContext::exp(size_t i) {
  return getRuleContext<qasmParser::ExpContext>(i);
}

tree::TerminalNode* qasmParser::ExpContext::RPAREN() {
  return getToken(qasmParser::RPAREN, 0);
}

tree::TerminalNode* qasmParser::ExpContext::MINUS() {
  return getToken(qasmParser::MINUS, 0);
}

tree::TerminalNode* qasmParser::ExpContext::MUL() {
  return getToken(qasmParser::MUL, 0);
}

tree::TerminalNode* qasmParser::ExpContext::DIV() {
  return getToken(qasmParser::DIV, 0);
}

tree::TerminalNode* qasmParser::ExpContext::PLUS() {
  return getToken(qasmParser::PLUS, 0);
}


size_t qasmParser::ExpContext::getRuleIndex() const {
  return qasmParser::RuleExp;
}

void qasmParser::ExpContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExp(this);
}

void qasmParser::ExpContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExp(this);
}


antlrcpp::Any qasmParser::ExpContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitExp(this);
  else
    return visitor->visitChildren(this);
}


qasmParser::ExpContext* qasmParser::exp() {
   return exp(0);
}

qasmParser::ExpContext* qasmParser::exp(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  qasmParser::ExpContext *_localctx = _tracker.createInstance<ExpContext>(_ctx, parentState);
  qasmParser::ExpContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 38;
  enterRecursionRule(_localctx, 38, qasmParser::RuleExp, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(303);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case qasmParser::IDENTIFIER: {
        setState(292);
        id();
        break;
      }

      case qasmParser::REALEXP: {
        setState(293);
        real();
        break;
      }

      case qasmParser::DECIMAL: {
        setState(294);
        decimal();
        break;
      }

      case qasmParser::INTEGER: {
        setState(295);
        integer();
        break;
      }

      case qasmParser::PI_KEY: {
        setState(296);
        match(qasmParser::PI_KEY);
        break;
      }

      case qasmParser::LPAREN: {
        setState(297);
        match(qasmParser::LPAREN);
        setState(298);
        exp(0);
        setState(299);
        match(qasmParser::RPAREN);
        break;
      }

      case qasmParser::MINUS: {
        setState(301);
        match(qasmParser::MINUS);
        setState(302);
        exp(5);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(319);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(317);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(305);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(306);
          match(qasmParser::MUL);
          setState(307);
          exp(5);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(308);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(309);
          match(qasmParser::DIV);
          setState(310);
          exp(4);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(311);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(312);
          match(qasmParser::PLUS);
          setState(313);
          exp(3);
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<ExpContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExp);
          setState(314);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(315);
          match(qasmParser::MINUS);
          setState(316);
          exp(2);
          break;
        }

        } 
      }
      setState(321);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx);
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

qasmParser::IdContext::IdContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::IdContext::IDENTIFIER() {
  return getToken(qasmParser::IDENTIFIER, 0);
}


size_t qasmParser::IdContext::getRuleIndex() const {
  return qasmParser::RuleId;
}

void qasmParser::IdContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterId(this);
}

void qasmParser::IdContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitId(this);
}


antlrcpp::Any qasmParser::IdContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitId(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::IdContext* qasmParser::id() {
  IdContext *_localctx = _tracker.createInstance<IdContext>(_ctx, getState());
  enterRule(_localctx, 40, qasmParser::RuleId);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(322);
    match(qasmParser::IDENTIFIER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- RealContext ------------------------------------------------------------------

qasmParser::RealContext::RealContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::RealContext::REALEXP() {
  return getToken(qasmParser::REALEXP, 0);
}


size_t qasmParser::RealContext::getRuleIndex() const {
  return qasmParser::RuleReal;
}

void qasmParser::RealContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterReal(this);
}

void qasmParser::RealContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitReal(this);
}


antlrcpp::Any qasmParser::RealContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitReal(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::RealContext* qasmParser::real() {
  RealContext *_localctx = _tracker.createInstance<RealContext>(_ctx, getState());
  enterRule(_localctx, 42, qasmParser::RuleReal);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(324);
    match(qasmParser::REALEXP);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IntegerContext ------------------------------------------------------------------

qasmParser::IntegerContext::IntegerContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::IntegerContext::INTEGER() {
  return getToken(qasmParser::INTEGER, 0);
}


size_t qasmParser::IntegerContext::getRuleIndex() const {
  return qasmParser::RuleInteger;
}

void qasmParser::IntegerContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterInteger(this);
}

void qasmParser::IntegerContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitInteger(this);
}


antlrcpp::Any qasmParser::IntegerContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitInteger(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::IntegerContext* qasmParser::integer() {
  IntegerContext *_localctx = _tracker.createInstance<IntegerContext>(_ctx, getState());
  enterRule(_localctx, 44, qasmParser::RuleInteger);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(326);
    match(qasmParser::INTEGER);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- DecimalContext ------------------------------------------------------------------

qasmParser::DecimalContext::DecimalContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::DecimalContext::DECIMAL() {
  return getToken(qasmParser::DECIMAL, 0);
}


size_t qasmParser::DecimalContext::getRuleIndex() const {
  return qasmParser::RuleDecimal;
}

void qasmParser::DecimalContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDecimal(this);
}

void qasmParser::DecimalContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDecimal(this);
}


antlrcpp::Any qasmParser::DecimalContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitDecimal(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::DecimalContext* qasmParser::decimal() {
  DecimalContext *_localctx = _tracker.createInstance<DecimalContext>(_ctx, getState());
  enterRule(_localctx, 46, qasmParser::RuleDecimal);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(328);
    match(qasmParser::DECIMAL);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FilenameContext ------------------------------------------------------------------

qasmParser::FilenameContext::FilenameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* qasmParser::FilenameContext::FILENAME() {
  return getToken(qasmParser::FILENAME, 0);
}


size_t qasmParser::FilenameContext::getRuleIndex() const {
  return qasmParser::RuleFilename;
}

void qasmParser::FilenameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterFilename(this);
}

void qasmParser::FilenameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<qasmListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitFilename(this);
}


antlrcpp::Any qasmParser::FilenameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<qasmVisitor*>(visitor))
    return parserVisitor->visitFilename(this);
  else
    return visitor->visitChildren(this);
}

qasmParser::FilenameContext* qasmParser::filename() {
  FilenameContext *_localctx = _tracker.createInstance<FilenameContext>(_ctx, getState());
  enterRule(_localctx, 48, qasmParser::RuleFilename);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(330);
    match(qasmParser::FILENAME);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool qasmParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 19: return expSempred(dynamic_cast<ExpContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool qasmParser::expSempred(ExpContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 4);
    case 1: return precpred(_ctx, 3);
    case 2: return precpred(_ctx, 2);
    case 3: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> qasmParser::_decisionToDFA;
atn::PredictionContextCache qasmParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN qasmParser::_atn;
std::vector<uint16_t> qasmParser::_serializedATN;

std::vector<std::string> qasmParser::_ruleNames = {
  "mainprogram", "head_decl", "version_decl", "include_decl", "statement", 
  "reg_decl", "opaque_decl", "if_decl", "barrier_decl", "gate_decl", "goplist", 
  "bop", "qop", "uop", "anylist", "idlist", "id_index", "argument", "explist", 
  "exp", "id", "real", "integer", "decimal", "filename"
};

std::vector<std::string> qasmParser::_literalNames = {
  "", "'OPENQASM'", "'include'", "'opaque'", "'qreg'", "'creg'", "'barrier'", 
  "'if'", "'measure'", "'reset'", "'gate'", "'pi'", "'U'", "'CX'", "'->'", 
  "'=='", "'+'", "'-'", "'*'", "'/'", "','", "';'", "'('", "')'", "'['", 
  "']'", "'{'", "'}'", "'\"'"
};

std::vector<std::string> qasmParser::_symbolicNames = {
  "", "OPENQASM_KEY", "INCLUDE_KEY", "OPAQUE_KEY", "QREG_KEY", "CREG_KEY", 
  "BARRIER_KEY", "IF_KEY", "MEASURE_KEY", "RESET_KEY", "GATE_KEY", "PI_KEY", 
  "U_GATE_KEY", "CX_GATE_KEY", "ARROW", "EQ", "PLUS", "MINUS", "MUL", "DIV", 
  "COMMA", "SEMI", "LPAREN", "RPAREN", "LBRACKET", "RBRACKET", "LBRACE", 
  "RBRACE", "DQM", "IDENTIFIER", "INTEGER", "DECIMAL", "FILENAME", "REALEXP", 
  "NL", "WS", "LC"
};

dfa::Vocabulary qasmParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> qasmParser::_tokenNames;

qasmParser::Initializer::Initializer() {
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
    0x3, 0x26, 0x14f, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 
    0x12, 0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 
    0x9, 0x15, 0x4, 0x16, 0x9, 0x16, 0x4, 0x17, 0x9, 0x17, 0x4, 0x18, 0x9, 
    0x18, 0x4, 0x19, 0x9, 0x19, 0x4, 0x1a, 0x9, 0x1a, 0x3, 0x2, 0x3, 0x2, 
    0x7, 0x2, 0x37, 0xa, 0x2, 0xc, 0x2, 0xe, 0x2, 0x3a, 0xb, 0x2, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x40, 0xa, 0x3, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x5, 0x6, 0x52, 0xa, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x5, 0x7, 0x62, 0xa, 0x7, 0x3, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x78, 0xa, 0x8, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x5, 0xb, 0xb7, 0xa, 
    0xb, 0x3, 0xc, 0x7, 0xc, 0xba, 0xa, 0xc, 0xc, 0xc, 0xe, 0xc, 0xbd, 0xb, 
    0xc, 0x3, 0xc, 0x7, 0xc, 0xc0, 0xa, 0xc, 0xc, 0xc, 0xe, 0xc, 0xc3, 0xb, 
    0xc, 0x5, 0xc, 0xc5, 0xa, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x5, 0xe, 0xd6, 0xa, 0xe, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x5, 0xf, 0xf6, 0xa, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x5, 0x10, 0xfb, 0xa, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x5, 
    0x10, 0x100, 0xa, 0x10, 0x7, 0x10, 0x102, 0xa, 0x10, 0xc, 0x10, 0xe, 
    0x10, 0x105, 0xb, 0x10, 0x5, 0x10, 0x107, 0xa, 0x10, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x11, 0x7, 0x11, 0x10c, 0xa, 0x11, 0xc, 0x11, 0xe, 0x11, 
    0x10f, 0xb, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 0x3, 0x12, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x5, 
    0x13, 0x11c, 0xa, 0x13, 0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x7, 0x14, 
    0x121, 0xa, 0x14, 0xc, 0x14, 0xe, 0x14, 0x124, 0xb, 0x14, 0x3, 0x15, 
    0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x5, 0x15, 0x132, 
    0xa, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 
    0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 0x3, 0x15, 
    0x7, 0x15, 0x140, 0xa, 0x15, 0xc, 0x15, 0xe, 0x15, 0x143, 0xb, 0x15, 
    0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 0x3, 0x18, 0x3, 0x18, 0x3, 
    0x19, 0x3, 0x19, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x2, 0x3, 0x28, 0x1b, 
    0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 
    0x1c, 0x1e, 0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 
    0x2, 0x2, 0x2, 0x15e, 0x2, 0x34, 0x3, 0x2, 0x2, 0x2, 0x4, 0x3f, 0x3, 
    0x2, 0x2, 0x2, 0x6, 0x41, 0x3, 0x2, 0x2, 0x2, 0x8, 0x45, 0x3, 0x2, 0x2, 
    0x2, 0xa, 0x51, 0x3, 0x2, 0x2, 0x2, 0xc, 0x61, 0x3, 0x2, 0x2, 0x2, 0xe, 
    0x77, 0x3, 0x2, 0x2, 0x2, 0x10, 0x79, 0x3, 0x2, 0x2, 0x2, 0x12, 0x81, 
    0x3, 0x2, 0x2, 0x2, 0x14, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x16, 0xc4, 0x3, 
    0x2, 0x2, 0x2, 0x18, 0xc6, 0x3, 0x2, 0x2, 0x2, 0x1a, 0xd5, 0x3, 0x2, 
    0x2, 0x2, 0x1c, 0xf5, 0x3, 0x2, 0x2, 0x2, 0x1e, 0x106, 0x3, 0x2, 0x2, 
    0x2, 0x20, 0x108, 0x3, 0x2, 0x2, 0x2, 0x22, 0x110, 0x3, 0x2, 0x2, 0x2, 
    0x24, 0x11b, 0x3, 0x2, 0x2, 0x2, 0x26, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x28, 
    0x131, 0x3, 0x2, 0x2, 0x2, 0x2a, 0x144, 0x3, 0x2, 0x2, 0x2, 0x2c, 0x146, 
    0x3, 0x2, 0x2, 0x2, 0x2e, 0x148, 0x3, 0x2, 0x2, 0x2, 0x30, 0x14a, 0x3, 
    0x2, 0x2, 0x2, 0x32, 0x14c, 0x3, 0x2, 0x2, 0x2, 0x34, 0x38, 0x5, 0x4, 
    0x3, 0x2, 0x35, 0x37, 0x5, 0xa, 0x6, 0x2, 0x36, 0x35, 0x3, 0x2, 0x2, 
    0x2, 0x37, 0x3a, 0x3, 0x2, 0x2, 0x2, 0x38, 0x36, 0x3, 0x2, 0x2, 0x2, 
    0x38, 0x39, 0x3, 0x2, 0x2, 0x2, 0x39, 0x3, 0x3, 0x2, 0x2, 0x2, 0x3a, 
    0x38, 0x3, 0x2, 0x2, 0x2, 0x3b, 0x40, 0x5, 0x6, 0x4, 0x2, 0x3c, 0x3d, 
    0x5, 0x6, 0x4, 0x2, 0x3d, 0x3e, 0x5, 0x8, 0x5, 0x2, 0x3e, 0x40, 0x3, 
    0x2, 0x2, 0x2, 0x3f, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x3f, 0x3c, 0x3, 0x2, 
    0x2, 0x2, 0x40, 0x5, 0x3, 0x2, 0x2, 0x2, 0x41, 0x42, 0x7, 0x3, 0x2, 
    0x2, 0x42, 0x43, 0x5, 0x30, 0x19, 0x2, 0x43, 0x44, 0x7, 0x17, 0x2, 0x2, 
    0x44, 0x7, 0x3, 0x2, 0x2, 0x2, 0x45, 0x46, 0x7, 0x4, 0x2, 0x2, 0x46, 
    0x47, 0x7, 0x1e, 0x2, 0x2, 0x47, 0x48, 0x5, 0x32, 0x1a, 0x2, 0x48, 0x49, 
    0x7, 0x1e, 0x2, 0x2, 0x49, 0x4a, 0x7, 0x17, 0x2, 0x2, 0x4a, 0x9, 0x3, 
    0x2, 0x2, 0x2, 0x4b, 0x52, 0x5, 0xc, 0x7, 0x2, 0x4c, 0x52, 0x5, 0x14, 
    0xb, 0x2, 0x4d, 0x52, 0x5, 0xe, 0x8, 0x2, 0x4e, 0x52, 0x5, 0x10, 0x9, 
    0x2, 0x4f, 0x52, 0x5, 0x12, 0xa, 0x2, 0x50, 0x52, 0x5, 0x1a, 0xe, 0x2, 
    0x51, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x51, 0x4c, 0x3, 0x2, 0x2, 0x2, 0x51, 
    0x4d, 0x3, 0x2, 0x2, 0x2, 0x51, 0x4e, 0x3, 0x2, 0x2, 0x2, 0x51, 0x4f, 
    0x3, 0x2, 0x2, 0x2, 0x51, 0x50, 0x3, 0x2, 0x2, 0x2, 0x52, 0xb, 0x3, 
    0x2, 0x2, 0x2, 0x53, 0x54, 0x7, 0x6, 0x2, 0x2, 0x54, 0x55, 0x5, 0x2a, 
    0x16, 0x2, 0x55, 0x56, 0x7, 0x1a, 0x2, 0x2, 0x56, 0x57, 0x5, 0x2e, 0x18, 
    0x2, 0x57, 0x58, 0x7, 0x1b, 0x2, 0x2, 0x58, 0x59, 0x7, 0x17, 0x2, 0x2, 
    0x59, 0x62, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x5b, 0x7, 0x7, 0x2, 0x2, 0x5b, 
    0x5c, 0x5, 0x2a, 0x16, 0x2, 0x5c, 0x5d, 0x7, 0x1a, 0x2, 0x2, 0x5d, 0x5e, 
    0x5, 0x2e, 0x18, 0x2, 0x5e, 0x5f, 0x7, 0x1b, 0x2, 0x2, 0x5f, 0x60, 0x7, 
    0x17, 0x2, 0x2, 0x60, 0x62, 0x3, 0x2, 0x2, 0x2, 0x61, 0x53, 0x3, 0x2, 
    0x2, 0x2, 0x61, 0x5a, 0x3, 0x2, 0x2, 0x2, 0x62, 0xd, 0x3, 0x2, 0x2, 
    0x2, 0x63, 0x64, 0x7, 0x5, 0x2, 0x2, 0x64, 0x65, 0x5, 0x2a, 0x16, 0x2, 
    0x65, 0x66, 0x5, 0x20, 0x11, 0x2, 0x66, 0x67, 0x7, 0x17, 0x2, 0x2, 0x67, 
    0x78, 0x3, 0x2, 0x2, 0x2, 0x68, 0x69, 0x7, 0x5, 0x2, 0x2, 0x69, 0x6a, 
    0x5, 0x2a, 0x16, 0x2, 0x6a, 0x6b, 0x7, 0x18, 0x2, 0x2, 0x6b, 0x6c, 0x7, 
    0x19, 0x2, 0x2, 0x6c, 0x6d, 0x5, 0x20, 0x11, 0x2, 0x6d, 0x6e, 0x7, 0x17, 
    0x2, 0x2, 0x6e, 0x78, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x70, 0x7, 0x5, 0x2, 
    0x2, 0x70, 0x71, 0x5, 0x2a, 0x16, 0x2, 0x71, 0x72, 0x7, 0x18, 0x2, 0x2, 
    0x72, 0x73, 0x5, 0x20, 0x11, 0x2, 0x73, 0x74, 0x7, 0x19, 0x2, 0x2, 0x74, 
    0x75, 0x5, 0x20, 0x11, 0x2, 0x75, 0x76, 0x7, 0x17, 0x2, 0x2, 0x76, 0x78, 
    0x3, 0x2, 0x2, 0x2, 0x77, 0x63, 0x3, 0x2, 0x2, 0x2, 0x77, 0x68, 0x3, 
    0x2, 0x2, 0x2, 0x77, 0x6f, 0x3, 0x2, 0x2, 0x2, 0x78, 0xf, 0x3, 0x2, 
    0x2, 0x2, 0x79, 0x7a, 0x7, 0x9, 0x2, 0x2, 0x7a, 0x7b, 0x7, 0x18, 0x2, 
    0x2, 0x7b, 0x7c, 0x5, 0x2a, 0x16, 0x2, 0x7c, 0x7d, 0x7, 0x11, 0x2, 0x2, 
    0x7d, 0x7e, 0x5, 0x2e, 0x18, 0x2, 0x7e, 0x7f, 0x7, 0x19, 0x2, 0x2, 0x7f, 
    0x80, 0x5, 0x1a, 0xe, 0x2, 0x80, 0x11, 0x3, 0x2, 0x2, 0x2, 0x81, 0x82, 
    0x7, 0x8, 0x2, 0x2, 0x82, 0x83, 0x5, 0x1e, 0x10, 0x2, 0x83, 0x84, 0x7, 
    0x17, 0x2, 0x2, 0x84, 0x13, 0x3, 0x2, 0x2, 0x2, 0x85, 0x86, 0x7, 0xc, 
    0x2, 0x2, 0x86, 0x87, 0x5, 0x2a, 0x16, 0x2, 0x87, 0x88, 0x5, 0x20, 0x11, 
    0x2, 0x88, 0x89, 0x7, 0x1c, 0x2, 0x2, 0x89, 0x8a, 0x5, 0x16, 0xc, 0x2, 
    0x8a, 0x8b, 0x7, 0x1d, 0x2, 0x2, 0x8b, 0xb7, 0x3, 0x2, 0x2, 0x2, 0x8c, 
    0x8d, 0x7, 0xc, 0x2, 0x2, 0x8d, 0x8e, 0x5, 0x2a, 0x16, 0x2, 0x8e, 0x8f, 
    0x7, 0x18, 0x2, 0x2, 0x8f, 0x90, 0x7, 0x19, 0x2, 0x2, 0x90, 0x91, 0x5, 
    0x20, 0x11, 0x2, 0x91, 0x92, 0x7, 0x1c, 0x2, 0x2, 0x92, 0x93, 0x5, 0x16, 
    0xc, 0x2, 0x93, 0x94, 0x7, 0x1d, 0x2, 0x2, 0x94, 0xb7, 0x3, 0x2, 0x2, 
    0x2, 0x95, 0x96, 0x7, 0xc, 0x2, 0x2, 0x96, 0x97, 0x5, 0x2a, 0x16, 0x2, 
    0x97, 0x98, 0x7, 0x18, 0x2, 0x2, 0x98, 0x99, 0x5, 0x20, 0x11, 0x2, 0x99, 
    0x9a, 0x7, 0x19, 0x2, 0x2, 0x9a, 0x9b, 0x5, 0x20, 0x11, 0x2, 0x9b, 0x9c, 
    0x7, 0x1c, 0x2, 0x2, 0x9c, 0x9d, 0x5, 0x16, 0xc, 0x2, 0x9d, 0x9e, 0x7, 
    0x1d, 0x2, 0x2, 0x9e, 0xb7, 0x3, 0x2, 0x2, 0x2, 0x9f, 0xa0, 0x7, 0xc, 
    0x2, 0x2, 0xa0, 0xa1, 0x5, 0x2a, 0x16, 0x2, 0xa1, 0xa2, 0x5, 0x20, 0x11, 
    0x2, 0xa2, 0xa3, 0x7, 0x1c, 0x2, 0x2, 0xa3, 0xa4, 0x7, 0x1d, 0x2, 0x2, 
    0xa4, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xa5, 0xa6, 0x7, 0xc, 0x2, 0x2, 0xa6, 
    0xa7, 0x5, 0x2a, 0x16, 0x2, 0xa7, 0xa8, 0x7, 0x18, 0x2, 0x2, 0xa8, 0xa9, 
    0x7, 0x19, 0x2, 0x2, 0xa9, 0xaa, 0x5, 0x20, 0x11, 0x2, 0xaa, 0xab, 0x7, 
    0x1c, 0x2, 0x2, 0xab, 0xac, 0x7, 0x1d, 0x2, 0x2, 0xac, 0xb7, 0x3, 0x2, 
    0x2, 0x2, 0xad, 0xae, 0x7, 0xc, 0x2, 0x2, 0xae, 0xaf, 0x5, 0x2a, 0x16, 
    0x2, 0xaf, 0xb0, 0x7, 0x18, 0x2, 0x2, 0xb0, 0xb1, 0x5, 0x20, 0x11, 0x2, 
    0xb1, 0xb2, 0x7, 0x19, 0x2, 0x2, 0xb2, 0xb3, 0x5, 0x20, 0x11, 0x2, 0xb3, 
    0xb4, 0x7, 0x1c, 0x2, 0x2, 0xb4, 0xb5, 0x7, 0x1d, 0x2, 0x2, 0xb5, 0xb7, 
    0x3, 0x2, 0x2, 0x2, 0xb6, 0x85, 0x3, 0x2, 0x2, 0x2, 0xb6, 0x8c, 0x3, 
    0x2, 0x2, 0x2, 0xb6, 0x95, 0x3, 0x2, 0x2, 0x2, 0xb6, 0x9f, 0x3, 0x2, 
    0x2, 0x2, 0xb6, 0xa5, 0x3, 0x2, 0x2, 0x2, 0xb6, 0xad, 0x3, 0x2, 0x2, 
    0x2, 0xb7, 0x15, 0x3, 0x2, 0x2, 0x2, 0xb8, 0xba, 0x5, 0x1c, 0xf, 0x2, 
    0xb9, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xba, 0xbd, 0x3, 0x2, 0x2, 0x2, 0xbb, 
    0xb9, 0x3, 0x2, 0x2, 0x2, 0xbb, 0xbc, 0x3, 0x2, 0x2, 0x2, 0xbc, 0xc5, 
    0x3, 0x2, 0x2, 0x2, 0xbd, 0xbb, 0x3, 0x2, 0x2, 0x2, 0xbe, 0xc0, 0x5, 
    0x18, 0xd, 0x2, 0xbf, 0xbe, 0x3, 0x2, 0x2, 0x2, 0xc0, 0xc3, 0x3, 0x2, 
    0x2, 0x2, 0xc1, 0xbf, 0x3, 0x2, 0x2, 0x2, 0xc1, 0xc2, 0x3, 0x2, 0x2, 
    0x2, 0xc2, 0xc5, 0x3, 0x2, 0x2, 0x2, 0xc3, 0xc1, 0x3, 0x2, 0x2, 0x2, 
    0xc4, 0xbb, 0x3, 0x2, 0x2, 0x2, 0xc4, 0xc1, 0x3, 0x2, 0x2, 0x2, 0xc5, 
    0x17, 0x3, 0x2, 0x2, 0x2, 0xc6, 0xc7, 0x7, 0x8, 0x2, 0x2, 0xc7, 0xc8, 
    0x5, 0x20, 0x11, 0x2, 0xc8, 0xc9, 0x7, 0x17, 0x2, 0x2, 0xc9, 0x19, 0x3, 
    0x2, 0x2, 0x2, 0xca, 0xd6, 0x5, 0x1c, 0xf, 0x2, 0xcb, 0xcc, 0x7, 0xa, 
    0x2, 0x2, 0xcc, 0xcd, 0x5, 0x24, 0x13, 0x2, 0xcd, 0xce, 0x7, 0x10, 0x2, 
    0x2, 0xce, 0xcf, 0x5, 0x24, 0x13, 0x2, 0xcf, 0xd0, 0x7, 0x17, 0x2, 0x2, 
    0xd0, 0xd6, 0x3, 0x2, 0x2, 0x2, 0xd1, 0xd2, 0x7, 0xb, 0x2, 0x2, 0xd2, 
    0xd3, 0x5, 0x24, 0x13, 0x2, 0xd3, 0xd4, 0x7, 0x17, 0x2, 0x2, 0xd4, 0xd6, 
    0x3, 0x2, 0x2, 0x2, 0xd5, 0xca, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xcb, 0x3, 
    0x2, 0x2, 0x2, 0xd5, 0xd1, 0x3, 0x2, 0x2, 0x2, 0xd6, 0x1b, 0x3, 0x2, 
    0x2, 0x2, 0xd7, 0xd8, 0x7, 0xe, 0x2, 0x2, 0xd8, 0xd9, 0x7, 0x18, 0x2, 
    0x2, 0xd9, 0xda, 0x5, 0x26, 0x14, 0x2, 0xda, 0xdb, 0x7, 0x19, 0x2, 0x2, 
    0xdb, 0xdc, 0x5, 0x24, 0x13, 0x2, 0xdc, 0xdd, 0x7, 0x17, 0x2, 0x2, 0xdd, 
    0xf6, 0x3, 0x2, 0x2, 0x2, 0xde, 0xdf, 0x7, 0xf, 0x2, 0x2, 0xdf, 0xe0, 
    0x5, 0x24, 0x13, 0x2, 0xe0, 0xe1, 0x7, 0x16, 0x2, 0x2, 0xe1, 0xe2, 0x5, 
    0x24, 0x13, 0x2, 0xe2, 0xe3, 0x7, 0x17, 0x2, 0x2, 0xe3, 0xf6, 0x3, 0x2, 
    0x2, 0x2, 0xe4, 0xe5, 0x5, 0x2a, 0x16, 0x2, 0xe5, 0xe6, 0x5, 0x1e, 0x10, 
    0x2, 0xe6, 0xe7, 0x7, 0x17, 0x2, 0x2, 0xe7, 0xf6, 0x3, 0x2, 0x2, 0x2, 
    0xe8, 0xe9, 0x5, 0x2a, 0x16, 0x2, 0xe9, 0xea, 0x7, 0x18, 0x2, 0x2, 0xea, 
    0xeb, 0x7, 0x19, 0x2, 0x2, 0xeb, 0xec, 0x5, 0x1e, 0x10, 0x2, 0xec, 0xed, 
    0x7, 0x17, 0x2, 0x2, 0xed, 0xf6, 0x3, 0x2, 0x2, 0x2, 0xee, 0xef, 0x5, 
    0x2a, 0x16, 0x2, 0xef, 0xf0, 0x7, 0x18, 0x2, 0x2, 0xf0, 0xf1, 0x5, 0x26, 
    0x14, 0x2, 0xf1, 0xf2, 0x7, 0x19, 0x2, 0x2, 0xf2, 0xf3, 0x5, 0x1e, 0x10, 
    0x2, 0xf3, 0xf4, 0x7, 0x17, 0x2, 0x2, 0xf4, 0xf6, 0x3, 0x2, 0x2, 0x2, 
    0xf5, 0xd7, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xde, 0x3, 0x2, 0x2, 0x2, 0xf5, 
    0xe4, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xe8, 0x3, 0x2, 0x2, 0x2, 0xf5, 0xee, 
    0x3, 0x2, 0x2, 0x2, 0xf6, 0x1d, 0x3, 0x2, 0x2, 0x2, 0xf7, 0x107, 0x3, 
    0x2, 0x2, 0x2, 0xf8, 0xfb, 0x5, 0x22, 0x12, 0x2, 0xf9, 0xfb, 0x5, 0x2a, 
    0x16, 0x2, 0xfa, 0xf8, 0x3, 0x2, 0x2, 0x2, 0xfa, 0xf9, 0x3, 0x2, 0x2, 
    0x2, 0xfb, 0x103, 0x3, 0x2, 0x2, 0x2, 0xfc, 0xff, 0x7, 0x16, 0x2, 0x2, 
    0xfd, 0x100, 0x5, 0x22, 0x12, 0x2, 0xfe, 0x100, 0x5, 0x2a, 0x16, 0x2, 
    0xff, 0xfd, 0x3, 0x2, 0x2, 0x2, 0xff, 0xfe, 0x3, 0x2, 0x2, 0x2, 0x100, 
    0x102, 0x3, 0x2, 0x2, 0x2, 0x101, 0xfc, 0x3, 0x2, 0x2, 0x2, 0x102, 0x105, 
    0x3, 0x2, 0x2, 0x2, 0x103, 0x101, 0x3, 0x2, 0x2, 0x2, 0x103, 0x104, 
    0x3, 0x2, 0x2, 0x2, 0x104, 0x107, 0x3, 0x2, 0x2, 0x2, 0x105, 0x103, 
    0x3, 0x2, 0x2, 0x2, 0x106, 0xf7, 0x3, 0x2, 0x2, 0x2, 0x106, 0xfa, 0x3, 
    0x2, 0x2, 0x2, 0x107, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x108, 0x10d, 0x5, 0x2a, 
    0x16, 0x2, 0x109, 0x10a, 0x7, 0x16, 0x2, 0x2, 0x10a, 0x10c, 0x5, 0x2a, 
    0x16, 0x2, 0x10b, 0x109, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10f, 0x3, 0x2, 
    0x2, 0x2, 0x10d, 0x10b, 0x3, 0x2, 0x2, 0x2, 0x10d, 0x10e, 0x3, 0x2, 
    0x2, 0x2, 0x10e, 0x21, 0x3, 0x2, 0x2, 0x2, 0x10f, 0x10d, 0x3, 0x2, 0x2, 
    0x2, 0x110, 0x111, 0x5, 0x2a, 0x16, 0x2, 0x111, 0x112, 0x7, 0x1a, 0x2, 
    0x2, 0x112, 0x113, 0x5, 0x2e, 0x18, 0x2, 0x113, 0x114, 0x7, 0x1b, 0x2, 
    0x2, 0x114, 0x23, 0x3, 0x2, 0x2, 0x2, 0x115, 0x11c, 0x5, 0x2a, 0x16, 
    0x2, 0x116, 0x117, 0x5, 0x2a, 0x16, 0x2, 0x117, 0x118, 0x7, 0x1a, 0x2, 
    0x2, 0x118, 0x119, 0x5, 0x2e, 0x18, 0x2, 0x119, 0x11a, 0x7, 0x1b, 0x2, 
    0x2, 0x11a, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x11b, 0x115, 0x3, 0x2, 0x2, 
    0x2, 0x11b, 0x116, 0x3, 0x2, 0x2, 0x2, 0x11c, 0x25, 0x3, 0x2, 0x2, 0x2, 
    0x11d, 0x122, 0x5, 0x28, 0x15, 0x2, 0x11e, 0x11f, 0x7, 0x16, 0x2, 0x2, 
    0x11f, 0x121, 0x5, 0x28, 0x15, 0x2, 0x120, 0x11e, 0x3, 0x2, 0x2, 0x2, 
    0x121, 0x124, 0x3, 0x2, 0x2, 0x2, 0x122, 0x120, 0x3, 0x2, 0x2, 0x2, 
    0x122, 0x123, 0x3, 0x2, 0x2, 0x2, 0x123, 0x27, 0x3, 0x2, 0x2, 0x2, 0x124, 
    0x122, 0x3, 0x2, 0x2, 0x2, 0x125, 0x126, 0x8, 0x15, 0x1, 0x2, 0x126, 
    0x132, 0x5, 0x2a, 0x16, 0x2, 0x127, 0x132, 0x5, 0x2c, 0x17, 0x2, 0x128, 
    0x132, 0x5, 0x30, 0x19, 0x2, 0x129, 0x132, 0x5, 0x2e, 0x18, 0x2, 0x12a, 
    0x132, 0x7, 0xd, 0x2, 0x2, 0x12b, 0x12c, 0x7, 0x18, 0x2, 0x2, 0x12c, 
    0x12d, 0x5, 0x28, 0x15, 0x2, 0x12d, 0x12e, 0x7, 0x19, 0x2, 0x2, 0x12e, 
    0x132, 0x3, 0x2, 0x2, 0x2, 0x12f, 0x130, 0x7, 0x13, 0x2, 0x2, 0x130, 
    0x132, 0x5, 0x28, 0x15, 0x7, 0x131, 0x125, 0x3, 0x2, 0x2, 0x2, 0x131, 
    0x127, 0x3, 0x2, 0x2, 0x2, 0x131, 0x128, 0x3, 0x2, 0x2, 0x2, 0x131, 
    0x129, 0x3, 0x2, 0x2, 0x2, 0x131, 0x12a, 0x3, 0x2, 0x2, 0x2, 0x131, 
    0x12b, 0x3, 0x2, 0x2, 0x2, 0x131, 0x12f, 0x3, 0x2, 0x2, 0x2, 0x132, 
    0x141, 0x3, 0x2, 0x2, 0x2, 0x133, 0x134, 0xc, 0x6, 0x2, 0x2, 0x134, 
    0x135, 0x7, 0x14, 0x2, 0x2, 0x135, 0x140, 0x5, 0x28, 0x15, 0x7, 0x136, 
    0x137, 0xc, 0x5, 0x2, 0x2, 0x137, 0x138, 0x7, 0x15, 0x2, 0x2, 0x138, 
    0x140, 0x5, 0x28, 0x15, 0x6, 0x139, 0x13a, 0xc, 0x4, 0x2, 0x2, 0x13a, 
    0x13b, 0x7, 0x12, 0x2, 0x2, 0x13b, 0x140, 0x5, 0x28, 0x15, 0x5, 0x13c, 
    0x13d, 0xc, 0x3, 0x2, 0x2, 0x13d, 0x13e, 0x7, 0x13, 0x2, 0x2, 0x13e, 
    0x140, 0x5, 0x28, 0x15, 0x4, 0x13f, 0x133, 0x3, 0x2, 0x2, 0x2, 0x13f, 
    0x136, 0x3, 0x2, 0x2, 0x2, 0x13f, 0x139, 0x3, 0x2, 0x2, 0x2, 0x13f, 
    0x13c, 0x3, 0x2, 0x2, 0x2, 0x140, 0x143, 0x3, 0x2, 0x2, 0x2, 0x141, 
    0x13f, 0x3, 0x2, 0x2, 0x2, 0x141, 0x142, 0x3, 0x2, 0x2, 0x2, 0x142, 
    0x29, 0x3, 0x2, 0x2, 0x2, 0x143, 0x141, 0x3, 0x2, 0x2, 0x2, 0x144, 0x145, 
    0x7, 0x1f, 0x2, 0x2, 0x145, 0x2b, 0x3, 0x2, 0x2, 0x2, 0x146, 0x147, 
    0x7, 0x23, 0x2, 0x2, 0x147, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x148, 0x149, 
    0x7, 0x20, 0x2, 0x2, 0x149, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x14a, 0x14b, 
    0x7, 0x21, 0x2, 0x2, 0x14b, 0x31, 0x3, 0x2, 0x2, 0x2, 0x14c, 0x14d, 
    0x7, 0x22, 0x2, 0x2, 0x14d, 0x33, 0x3, 0x2, 0x2, 0x2, 0x17, 0x38, 0x3f, 
    0x51, 0x61, 0x77, 0xb6, 0xbb, 0xc1, 0xc4, 0xd5, 0xf5, 0xfa, 0xff, 0x103, 
    0x106, 0x10d, 0x11b, 0x122, 0x131, 0x13f, 0x141, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

qasmParser::Initializer qasmParser::_init;
