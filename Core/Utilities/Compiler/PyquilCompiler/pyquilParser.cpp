
// Generated from .\pyquil.g4 by ANTLR 4.8


#include "Core/Utilities/Compiler/PyquilCompiler/pyquilListener.h"
#include "Core/Utilities/Compiler/PyquilCompiler/pyquilVisitor.h"

#include "Core/Utilities/Compiler/PyquilCompiler/pyquilParser.h"


using namespace antlrcpp;
using namespace antlr4;

pyquilParser::pyquilParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

pyquilParser::~pyquilParser() {
  delete _interpreter;
}

std::string pyquilParser::getGrammarFileName() const {
  return "pyquil.g4";
}

const std::vector<std::string>& pyquilParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& pyquilParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- ProgContext ------------------------------------------------------------------

pyquilParser::ProgContext::ProgContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyquilParser::DeclareContext *> pyquilParser::ProgContext::declare() {
  return getRuleContexts<pyquilParser::DeclareContext>();
}

pyquilParser::DeclareContext* pyquilParser::ProgContext::declare(size_t i) {
  return getRuleContext<pyquilParser::DeclareContext>(i);
}

std::vector<pyquilParser::Code_blockContext *> pyquilParser::ProgContext::code_block() {
  return getRuleContexts<pyquilParser::Code_blockContext>();
}

pyquilParser::Code_blockContext* pyquilParser::ProgContext::code_block(size_t i) {
  return getRuleContext<pyquilParser::Code_blockContext>(i);
}


size_t pyquilParser::ProgContext::getRuleIndex() const {
  return pyquilParser::RuleProg;
}

void pyquilParser::ProgContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterProg(this);
}

void pyquilParser::ProgContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitProg(this);
}


antlrcpp::Any pyquilParser::ProgContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitProg(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::ProgContext* pyquilParser::prog() {
  ProgContext *_localctx = _tracker.createInstance<ProgContext>(_ctx, getState());
  enterRule(_localctx, 0, pyquilParser::RuleProg);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(43); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(42);
      declare();
      setState(45); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while (_la == pyquilParser::T__0);
    setState(48); 
    _errHandler->sync(this);
    _la = _input->LA(1);
    do {
      setState(47);
      code_block();
      setState(50); 
      _errHandler->sync(this);
      _la = _input->LA(1);
    } while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyquilParser::T__1)
      | (1ULL << pyquilParser::T__2)
      | (1ULL << pyquilParser::T__3)
      | (1ULL << pyquilParser::LABEL_LINE)
      | (1ULL << pyquilParser::MODIFY_OP)
      | (1ULL << pyquilParser::GATE1Q)
      | (1ULL << pyquilParser::GATE2Q)
      | (1ULL << pyquilParser::GATE3Q)
      | (1ULL << pyquilParser::GATE1Q1P)
      | (1ULL << pyquilParser::GATE2Q1P))) != 0));
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Code_blockContext ------------------------------------------------------------------

pyquilParser::Code_blockContext::Code_blockContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyquilParser::LoopContext* pyquilParser::Code_blockContext::loop() {
  return getRuleContext<pyquilParser::LoopContext>(0);
}

std::vector<pyquilParser::OperationContext *> pyquilParser::Code_blockContext::operation() {
  return getRuleContexts<pyquilParser::OperationContext>();
}

pyquilParser::OperationContext* pyquilParser::Code_blockContext::operation(size_t i) {
  return getRuleContext<pyquilParser::OperationContext>(i);
}


size_t pyquilParser::Code_blockContext::getRuleIndex() const {
  return pyquilParser::RuleCode_block;
}

void pyquilParser::Code_blockContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterCode_block(this);
}

void pyquilParser::Code_blockContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitCode_block(this);
}


antlrcpp::Any pyquilParser::Code_blockContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitCode_block(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Code_blockContext* pyquilParser::code_block() {
  Code_blockContext *_localctx = _tracker.createInstance<Code_blockContext>(_ctx, getState());
  enterRule(_localctx, 2, pyquilParser::RuleCode_block);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    setState(58);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyquilParser::LABEL_LINE: {
        enterOuterAlt(_localctx, 1);
        setState(52);
        loop();
        break;
      }

      case pyquilParser::T__1:
      case pyquilParser::T__2:
      case pyquilParser::T__3:
      case pyquilParser::MODIFY_OP:
      case pyquilParser::GATE1Q:
      case pyquilParser::GATE2Q:
      case pyquilParser::GATE3Q:
      case pyquilParser::GATE1Q1P:
      case pyquilParser::GATE2Q1P: {
        enterOuterAlt(_localctx, 2);
        setState(54); 
        _errHandler->sync(this);
        alt = 1;
        do {
          switch (alt) {
            case 1: {
                  setState(53);
                  operation();
                  break;
                }

          default:
            throw NoViableAltException(this);
          }
          setState(56); 
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx);
        } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
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

//----------------- LoopContext ------------------------------------------------------------------

pyquilParser::LoopContext::LoopContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyquilParser::Loop_startContext* pyquilParser::LoopContext::loop_start() {
  return getRuleContext<pyquilParser::Loop_startContext>(0);
}

pyquilParser::Loop_if_continueContext* pyquilParser::LoopContext::loop_if_continue() {
  return getRuleContext<pyquilParser::Loop_if_continueContext>(0);
}

tree::TerminalNode* pyquilParser::LoopContext::JUMP_LINE() {
  return getToken(pyquilParser::JUMP_LINE, 0);
}

pyquilParser::Loop_endContext* pyquilParser::LoopContext::loop_end() {
  return getRuleContext<pyquilParser::Loop_endContext>(0);
}

std::vector<pyquilParser::OperationContext *> pyquilParser::LoopContext::operation() {
  return getRuleContexts<pyquilParser::OperationContext>();
}

pyquilParser::OperationContext* pyquilParser::LoopContext::operation(size_t i) {
  return getRuleContext<pyquilParser::OperationContext>(i);
}


size_t pyquilParser::LoopContext::getRuleIndex() const {
  return pyquilParser::RuleLoop;
}

void pyquilParser::LoopContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoop(this);
}

void pyquilParser::LoopContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoop(this);
}


antlrcpp::Any pyquilParser::LoopContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitLoop(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::LoopContext* pyquilParser::loop() {
  LoopContext *_localctx = _tracker.createInstance<LoopContext>(_ctx, getState());
  enterRule(_localctx, 4, pyquilParser::RuleLoop);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(60);
    loop_start();
    setState(64);
    _errHandler->sync(this);
    _la = _input->LA(1);
    while ((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyquilParser::T__1)
      | (1ULL << pyquilParser::T__2)
      | (1ULL << pyquilParser::T__3)
      | (1ULL << pyquilParser::MODIFY_OP)
      | (1ULL << pyquilParser::GATE1Q)
      | (1ULL << pyquilParser::GATE2Q)
      | (1ULL << pyquilParser::GATE3Q)
      | (1ULL << pyquilParser::GATE1Q1P)
      | (1ULL << pyquilParser::GATE2Q1P))) != 0)) {
      setState(61);
      operation();
      setState(66);
      _errHandler->sync(this);
      _la = _input->LA(1);
    }
    setState(67);
    loop_if_continue();
    setState(68);
    match(pyquilParser::JUMP_LINE);
    setState(69);
    loop_end();
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Loop_startContext ------------------------------------------------------------------

pyquilParser::Loop_startContext::Loop_startContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::Loop_startContext::LABEL_LINE() {
  return getToken(pyquilParser::LABEL_LINE, 0);
}


size_t pyquilParser::Loop_startContext::getRuleIndex() const {
  return pyquilParser::RuleLoop_start;
}

void pyquilParser::Loop_startContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoop_start(this);
}

void pyquilParser::Loop_startContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoop_start(this);
}


antlrcpp::Any pyquilParser::Loop_startContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitLoop_start(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Loop_startContext* pyquilParser::loop_start() {
  Loop_startContext *_localctx = _tracker.createInstance<Loop_startContext>(_ctx, getState());
  enterRule(_localctx, 6, pyquilParser::RuleLoop_start);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(71);
    match(pyquilParser::LABEL_LINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Loop_endContext ------------------------------------------------------------------

pyquilParser::Loop_endContext::Loop_endContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::Loop_endContext::LABEL_LINE() {
  return getToken(pyquilParser::LABEL_LINE, 0);
}


size_t pyquilParser::Loop_endContext::getRuleIndex() const {
  return pyquilParser::RuleLoop_end;
}

void pyquilParser::Loop_endContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoop_end(this);
}

void pyquilParser::Loop_endContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoop_end(this);
}


antlrcpp::Any pyquilParser::Loop_endContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitLoop_end(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Loop_endContext* pyquilParser::loop_end() {
  Loop_endContext *_localctx = _tracker.createInstance<Loop_endContext>(_ctx, getState());
  enterRule(_localctx, 8, pyquilParser::RuleLoop_end);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(73);
    match(pyquilParser::LABEL_LINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Loop_if_continueContext ------------------------------------------------------------------

pyquilParser::Loop_if_continueContext::Loop_if_continueContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::Loop_if_continueContext::JUMP_UNLESS() {
  return getToken(pyquilParser::JUMP_UNLESS, 0);
}

std::vector<tree::TerminalNode *> pyquilParser::Loop_if_continueContext::SPACES() {
  return getTokens(pyquilParser::SPACES);
}

tree::TerminalNode* pyquilParser::Loop_if_continueContext::SPACES(size_t i) {
  return getToken(pyquilParser::SPACES, i);
}

tree::TerminalNode* pyquilParser::Loop_if_continueContext::LABELNAME() {
  return getToken(pyquilParser::LABELNAME, 0);
}

pyquilParser::Bool_valContext* pyquilParser::Loop_if_continueContext::bool_val() {
  return getRuleContext<pyquilParser::Bool_valContext>(0);
}

tree::TerminalNode* pyquilParser::Loop_if_continueContext::NEWLINE() {
  return getToken(pyquilParser::NEWLINE, 0);
}


size_t pyquilParser::Loop_if_continueContext::getRuleIndex() const {
  return pyquilParser::RuleLoop_if_continue;
}

void pyquilParser::Loop_if_continueContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterLoop_if_continue(this);
}

void pyquilParser::Loop_if_continueContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitLoop_if_continue(this);
}


antlrcpp::Any pyquilParser::Loop_if_continueContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitLoop_if_continue(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Loop_if_continueContext* pyquilParser::loop_if_continue() {
  Loop_if_continueContext *_localctx = _tracker.createInstance<Loop_if_continueContext>(_ctx, getState());
  enterRule(_localctx, 10, pyquilParser::RuleLoop_if_continue);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(75);
    match(pyquilParser::JUMP_UNLESS);
    setState(76);
    match(pyquilParser::SPACES);
    setState(77);
    match(pyquilParser::LABELNAME);
    setState(78);
    match(pyquilParser::SPACES);
    setState(79);
    bool_val();
    setState(80);
    match(pyquilParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- OperationContext ------------------------------------------------------------------

pyquilParser::OperationContext::OperationContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyquilParser::MeasureContext* pyquilParser::OperationContext::measure() {
  return getRuleContext<pyquilParser::MeasureContext>(0);
}

pyquilParser::GateContext* pyquilParser::OperationContext::gate() {
  return getRuleContext<pyquilParser::GateContext>(0);
}

pyquilParser::MoveContext* pyquilParser::OperationContext::move() {
  return getRuleContext<pyquilParser::MoveContext>(0);
}

pyquilParser::SubContext* pyquilParser::OperationContext::sub() {
  return getRuleContext<pyquilParser::SubContext>(0);
}


size_t pyquilParser::OperationContext::getRuleIndex() const {
  return pyquilParser::RuleOperation;
}

void pyquilParser::OperationContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterOperation(this);
}

void pyquilParser::OperationContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitOperation(this);
}


antlrcpp::Any pyquilParser::OperationContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitOperation(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::OperationContext* pyquilParser::operation() {
  OperationContext *_localctx = _tracker.createInstance<OperationContext>(_ctx, getState());
  enterRule(_localctx, 12, pyquilParser::RuleOperation);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(86);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyquilParser::T__1: {
        enterOuterAlt(_localctx, 1);
        setState(82);
        measure();
        break;
      }

      case pyquilParser::MODIFY_OP:
      case pyquilParser::GATE1Q:
      case pyquilParser::GATE2Q:
      case pyquilParser::GATE3Q:
      case pyquilParser::GATE1Q1P:
      case pyquilParser::GATE2Q1P: {
        enterOuterAlt(_localctx, 2);
        setState(83);
        gate();
        break;
      }

      case pyquilParser::T__2: {
        enterOuterAlt(_localctx, 3);
        setState(84);
        move();
        break;
      }

      case pyquilParser::T__3: {
        enterOuterAlt(_localctx, 4);
        setState(85);
        sub();
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

//----------------- DeclareContext ------------------------------------------------------------------

pyquilParser::DeclareContext::DeclareContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> pyquilParser::DeclareContext::SPACES() {
  return getTokens(pyquilParser::SPACES);
}

tree::TerminalNode* pyquilParser::DeclareContext::SPACES(size_t i) {
  return getToken(pyquilParser::SPACES, i);
}

pyquilParser::Var_nameContext* pyquilParser::DeclareContext::var_name() {
  return getRuleContext<pyquilParser::Var_nameContext>(0);
}

pyquilParser::Var_memContext* pyquilParser::DeclareContext::var_mem() {
  return getRuleContext<pyquilParser::Var_memContext>(0);
}

tree::TerminalNode* pyquilParser::DeclareContext::NEWLINE() {
  return getToken(pyquilParser::NEWLINE, 0);
}


size_t pyquilParser::DeclareContext::getRuleIndex() const {
  return pyquilParser::RuleDeclare;
}

void pyquilParser::DeclareContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterDeclare(this);
}

void pyquilParser::DeclareContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitDeclare(this);
}


antlrcpp::Any pyquilParser::DeclareContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitDeclare(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::DeclareContext* pyquilParser::declare() {
  DeclareContext *_localctx = _tracker.createInstance<DeclareContext>(_ctx, getState());
  enterRule(_localctx, 14, pyquilParser::RuleDeclare);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(88);
    match(pyquilParser::T__0);
    setState(89);
    match(pyquilParser::SPACES);
    setState(90);
    var_name();
    setState(91);
    match(pyquilParser::SPACES);
    setState(92);
    var_mem();
    setState(93);
    match(pyquilParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MeasureContext ------------------------------------------------------------------

pyquilParser::MeasureContext::MeasureContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> pyquilParser::MeasureContext::SPACES() {
  return getTokens(pyquilParser::SPACES);
}

tree::TerminalNode* pyquilParser::MeasureContext::SPACES(size_t i) {
  return getToken(pyquilParser::SPACES, i);
}

pyquilParser::QbitContext* pyquilParser::MeasureContext::qbit() {
  return getRuleContext<pyquilParser::QbitContext>(0);
}

pyquilParser::Array_itemContext* pyquilParser::MeasureContext::array_item() {
  return getRuleContext<pyquilParser::Array_itemContext>(0);
}

tree::TerminalNode* pyquilParser::MeasureContext::NEWLINE() {
  return getToken(pyquilParser::NEWLINE, 0);
}


size_t pyquilParser::MeasureContext::getRuleIndex() const {
  return pyquilParser::RuleMeasure;
}

void pyquilParser::MeasureContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMeasure(this);
}

void pyquilParser::MeasureContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMeasure(this);
}


antlrcpp::Any pyquilParser::MeasureContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitMeasure(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::MeasureContext* pyquilParser::measure() {
  MeasureContext *_localctx = _tracker.createInstance<MeasureContext>(_ctx, getState());
  enterRule(_localctx, 16, pyquilParser::RuleMeasure);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(95);
    match(pyquilParser::T__1);
    setState(96);
    match(pyquilParser::SPACES);
    setState(97);
    qbit();
    setState(98);
    match(pyquilParser::SPACES);
    setState(99);
    array_item();
    setState(100);
    match(pyquilParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- MoveContext ------------------------------------------------------------------

pyquilParser::MoveContext::MoveContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> pyquilParser::MoveContext::SPACES() {
  return getTokens(pyquilParser::SPACES);
}

tree::TerminalNode* pyquilParser::MoveContext::SPACES(size_t i) {
  return getToken(pyquilParser::SPACES, i);
}

pyquilParser::Array_itemContext* pyquilParser::MoveContext::array_item() {
  return getRuleContext<pyquilParser::Array_itemContext>(0);
}

pyquilParser::ExprContext* pyquilParser::MoveContext::expr() {
  return getRuleContext<pyquilParser::ExprContext>(0);
}

tree::TerminalNode* pyquilParser::MoveContext::NEWLINE() {
  return getToken(pyquilParser::NEWLINE, 0);
}


size_t pyquilParser::MoveContext::getRuleIndex() const {
  return pyquilParser::RuleMove;
}

void pyquilParser::MoveContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterMove(this);
}

void pyquilParser::MoveContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitMove(this);
}


antlrcpp::Any pyquilParser::MoveContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitMove(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::MoveContext* pyquilParser::move() {
  MoveContext *_localctx = _tracker.createInstance<MoveContext>(_ctx, getState());
  enterRule(_localctx, 18, pyquilParser::RuleMove);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(102);
    match(pyquilParser::T__2);
    setState(103);
    match(pyquilParser::SPACES);
    setState(104);
    array_item();
    setState(105);
    match(pyquilParser::SPACES);
    setState(106);
    expr(0);
    setState(107);
    match(pyquilParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- SubContext ------------------------------------------------------------------

pyquilParser::SubContext::SubContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<tree::TerminalNode *> pyquilParser::SubContext::SPACES() {
  return getTokens(pyquilParser::SPACES);
}

tree::TerminalNode* pyquilParser::SubContext::SPACES(size_t i) {
  return getToken(pyquilParser::SPACES, i);
}

pyquilParser::Array_itemContext* pyquilParser::SubContext::array_item() {
  return getRuleContext<pyquilParser::Array_itemContext>(0);
}

pyquilParser::ExprContext* pyquilParser::SubContext::expr() {
  return getRuleContext<pyquilParser::ExprContext>(0);
}

tree::TerminalNode* pyquilParser::SubContext::NEWLINE() {
  return getToken(pyquilParser::NEWLINE, 0);
}


size_t pyquilParser::SubContext::getRuleIndex() const {
  return pyquilParser::RuleSub;
}

void pyquilParser::SubContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterSub(this);
}

void pyquilParser::SubContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitSub(this);
}


antlrcpp::Any pyquilParser::SubContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitSub(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::SubContext* pyquilParser::sub() {
  SubContext *_localctx = _tracker.createInstance<SubContext>(_ctx, getState());
  enterRule(_localctx, 20, pyquilParser::RuleSub);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(109);
    match(pyquilParser::T__3);
    setState(110);
    match(pyquilParser::SPACES);
    setState(111);
    array_item();
    setState(112);
    match(pyquilParser::SPACES);
    setState(113);
    expr(0);
    setState(114);
    match(pyquilParser::NEWLINE);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Var_nameContext ------------------------------------------------------------------

pyquilParser::Var_nameContext::Var_nameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::Var_nameContext::STRING() {
  return getToken(pyquilParser::STRING, 0);
}


size_t pyquilParser::Var_nameContext::getRuleIndex() const {
  return pyquilParser::RuleVar_name;
}

void pyquilParser::Var_nameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVar_name(this);
}

void pyquilParser::Var_nameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVar_name(this);
}


antlrcpp::Any pyquilParser::Var_nameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitVar_name(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Var_nameContext* pyquilParser::var_name() {
  Var_nameContext *_localctx = _tracker.createInstance<Var_nameContext>(_ctx, getState());
  enterRule(_localctx, 22, pyquilParser::RuleVar_name);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(116);
    match(pyquilParser::STRING);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- Var_memContext ------------------------------------------------------------------

pyquilParser::Var_memContext::Var_memContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyquilParser::IdxContext* pyquilParser::Var_memContext::idx() {
  return getRuleContext<pyquilParser::IdxContext>(0);
}


size_t pyquilParser::Var_memContext::getRuleIndex() const {
  return pyquilParser::RuleVar_mem;
}

void pyquilParser::Var_memContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterVar_mem(this);
}

void pyquilParser::Var_memContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitVar_mem(this);
}


antlrcpp::Any pyquilParser::Var_memContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitVar_mem(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Var_memContext* pyquilParser::var_mem() {
  Var_memContext *_localctx = _tracker.createInstance<Var_memContext>(_ctx, getState());
  enterRule(_localctx, 24, pyquilParser::RuleVar_mem);
  size_t _la = 0;

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(118);
    _la = _input->LA(1);
    if (!((((_la & ~ 0x3fULL) == 0) &&
      ((1ULL << _la) & ((1ULL << pyquilParser::T__4)
      | (1ULL << pyquilParser::T__5)
      | (1ULL << pyquilParser::T__6)
      | (1ULL << pyquilParser::T__7))) != 0))) {
    _errHandler->recoverInline(this);
    }
    else {
      _errHandler->reportMatch(this);
      consume();
    }
    setState(119);
    match(pyquilParser::T__8);
    setState(120);
    idx();
    setState(121);
    match(pyquilParser::T__9);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- QbitContext ------------------------------------------------------------------

pyquilParser::QbitContext::QbitContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::QbitContext::INT() {
  return getToken(pyquilParser::INT, 0);
}


size_t pyquilParser::QbitContext::getRuleIndex() const {
  return pyquilParser::RuleQbit;
}

void pyquilParser::QbitContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterQbit(this);
}

void pyquilParser::QbitContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitQbit(this);
}


antlrcpp::Any pyquilParser::QbitContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitQbit(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::QbitContext* pyquilParser::qbit() {
  QbitContext *_localctx = _tracker.createInstance<QbitContext>(_ctx, getState());
  enterRule(_localctx, 26, pyquilParser::RuleQbit);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(123);
    match(pyquilParser::INT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GateContext ------------------------------------------------------------------

pyquilParser::GateContext::GateContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyquilParser::GateContext* pyquilParser::GateContext::gate() {
  return getRuleContext<pyquilParser::GateContext>(0);
}

std::vector<tree::TerminalNode *> pyquilParser::GateContext::MODIFY_OP() {
  return getTokens(pyquilParser::MODIFY_OP);
}

tree::TerminalNode* pyquilParser::GateContext::MODIFY_OP(size_t i) {
  return getToken(pyquilParser::MODIFY_OP, i);
}

tree::TerminalNode* pyquilParser::GateContext::GATE1Q() {
  return getToken(pyquilParser::GATE1Q, 0);
}

std::vector<tree::TerminalNode *> pyquilParser::GateContext::SPACES() {
  return getTokens(pyquilParser::SPACES);
}

tree::TerminalNode* pyquilParser::GateContext::SPACES(size_t i) {
  return getToken(pyquilParser::SPACES, i);
}

std::vector<pyquilParser::QbitContext *> pyquilParser::GateContext::qbit() {
  return getRuleContexts<pyquilParser::QbitContext>();
}

pyquilParser::QbitContext* pyquilParser::GateContext::qbit(size_t i) {
  return getRuleContext<pyquilParser::QbitContext>(i);
}

tree::TerminalNode* pyquilParser::GateContext::NEWLINE() {
  return getToken(pyquilParser::NEWLINE, 0);
}

tree::TerminalNode* pyquilParser::GateContext::GATE2Q() {
  return getToken(pyquilParser::GATE2Q, 0);
}

tree::TerminalNode* pyquilParser::GateContext::GATE3Q() {
  return getToken(pyquilParser::GATE3Q, 0);
}

tree::TerminalNode* pyquilParser::GateContext::GATE1Q1P() {
  return getToken(pyquilParser::GATE1Q1P, 0);
}

pyquilParser::ParamContext* pyquilParser::GateContext::param() {
  return getRuleContext<pyquilParser::ParamContext>(0);
}

tree::TerminalNode* pyquilParser::GateContext::GATE2Q1P() {
  return getToken(pyquilParser::GATE2Q1P, 0);
}


size_t pyquilParser::GateContext::getRuleIndex() const {
  return pyquilParser::RuleGate;
}

void pyquilParser::GateContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterGate(this);
}

void pyquilParser::GateContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitGate(this);
}


antlrcpp::Any pyquilParser::GateContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitGate(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::GateContext* pyquilParser::gate() {
  GateContext *_localctx = _tracker.createInstance<GateContext>(_ctx, getState());
  enterRule(_localctx, 28, pyquilParser::RuleGate);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    size_t alt;
    setState(170);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyquilParser::MODIFY_OP: {
        enterOuterAlt(_localctx, 1);
        setState(126); 
        _errHandler->sync(this);
        alt = 1;
        do {
          switch (alt) {
            case 1: {
                  setState(125);
                  match(pyquilParser::MODIFY_OP);
                  break;
                }

          default:
            throw NoViableAltException(this);
          }
          setState(128); 
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx);
        } while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER);
        setState(130);
        gate();
        break;
      }

      case pyquilParser::GATE1Q: {
        enterOuterAlt(_localctx, 2);
        setState(131);
        match(pyquilParser::GATE1Q);
        setState(132);
        match(pyquilParser::SPACES);
        setState(133);
        qbit();
        setState(134);
        match(pyquilParser::NEWLINE);
        break;
      }

      case pyquilParser::GATE2Q: {
        enterOuterAlt(_localctx, 3);
        setState(136);
        match(pyquilParser::GATE2Q);
        setState(137);
        match(pyquilParser::SPACES);
        setState(138);
        qbit();
        setState(139);
        match(pyquilParser::SPACES);
        setState(140);
        qbit();
        setState(141);
        match(pyquilParser::NEWLINE);
        break;
      }

      case pyquilParser::GATE3Q: {
        enterOuterAlt(_localctx, 4);
        setState(143);
        match(pyquilParser::GATE3Q);
        setState(144);
        match(pyquilParser::SPACES);
        setState(145);
        qbit();
        setState(146);
        match(pyquilParser::SPACES);
        setState(147);
        qbit();
        setState(148);
        match(pyquilParser::SPACES);
        setState(149);
        qbit();
        setState(150);
        match(pyquilParser::NEWLINE);
        break;
      }

      case pyquilParser::GATE1Q1P: {
        enterOuterAlt(_localctx, 5);
        setState(152);
        match(pyquilParser::GATE1Q1P);
        setState(153);
        match(pyquilParser::T__10);
        setState(154);
        param();
        setState(155);
        match(pyquilParser::T__11);
        setState(156);
        match(pyquilParser::SPACES);
        setState(157);
        qbit();
        setState(158);
        match(pyquilParser::NEWLINE);
        break;
      }

      case pyquilParser::GATE2Q1P: {
        enterOuterAlt(_localctx, 6);
        setState(160);
        match(pyquilParser::GATE2Q1P);
        setState(161);
        match(pyquilParser::T__10);
        setState(162);
        param();
        setState(163);
        match(pyquilParser::T__11);
        setState(164);
        match(pyquilParser::SPACES);
        setState(165);
        qbit();
        setState(166);
        match(pyquilParser::SPACES);
        setState(167);
        qbit();
        setState(168);
        match(pyquilParser::NEWLINE);
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

//----------------- Bool_valContext ------------------------------------------------------------------

pyquilParser::Bool_valContext::Bool_valContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::Bool_valContext::INT() {
  return getToken(pyquilParser::INT, 0);
}

pyquilParser::Array_itemContext* pyquilParser::Bool_valContext::array_item() {
  return getRuleContext<pyquilParser::Array_itemContext>(0);
}


size_t pyquilParser::Bool_valContext::getRuleIndex() const {
  return pyquilParser::RuleBool_val;
}

void pyquilParser::Bool_valContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterBool_val(this);
}

void pyquilParser::Bool_valContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitBool_val(this);
}


antlrcpp::Any pyquilParser::Bool_valContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitBool_val(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Bool_valContext* pyquilParser::bool_val() {
  Bool_valContext *_localctx = _tracker.createInstance<Bool_valContext>(_ctx, getState());
  enterRule(_localctx, 30, pyquilParser::RuleBool_val);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(174);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyquilParser::INT: {
        enterOuterAlt(_localctx, 1);
        setState(172);
        match(pyquilParser::INT);
        break;
      }

      case pyquilParser::STRING: {
        enterOuterAlt(_localctx, 2);
        setState(173);
        array_item();
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

//----------------- ParamContext ------------------------------------------------------------------

pyquilParser::ParamContext::ParamContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyquilParser::ExprContext* pyquilParser::ParamContext::expr() {
  return getRuleContext<pyquilParser::ExprContext>(0);
}


size_t pyquilParser::ParamContext::getRuleIndex() const {
  return pyquilParser::RuleParam;
}

void pyquilParser::ParamContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterParam(this);
}

void pyquilParser::ParamContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitParam(this);
}


antlrcpp::Any pyquilParser::ParamContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitParam(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::ParamContext* pyquilParser::param() {
  ParamContext *_localctx = _tracker.createInstance<ParamContext>(_ctx, getState());
  enterRule(_localctx, 32, pyquilParser::RuleParam);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(176);
    expr(0);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprContext ------------------------------------------------------------------

pyquilParser::ExprContext::ExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

std::vector<pyquilParser::ExprContext *> pyquilParser::ExprContext::expr() {
  return getRuleContexts<pyquilParser::ExprContext>();
}

pyquilParser::ExprContext* pyquilParser::ExprContext::expr(size_t i) {
  return getRuleContext<pyquilParser::ExprContext>(i);
}

tree::TerminalNode* pyquilParser::ExprContext::INT() {
  return getToken(pyquilParser::INT, 0);
}

tree::TerminalNode* pyquilParser::ExprContext::FLOAT() {
  return getToken(pyquilParser::FLOAT, 0);
}

pyquilParser::Array_itemContext* pyquilParser::ExprContext::array_item() {
  return getRuleContext<pyquilParser::Array_itemContext>(0);
}


size_t pyquilParser::ExprContext::getRuleIndex() const {
  return pyquilParser::RuleExpr;
}

void pyquilParser::ExprContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterExpr(this);
}

void pyquilParser::ExprContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitExpr(this);
}


antlrcpp::Any pyquilParser::ExprContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitExpr(this);
  else
    return visitor->visitChildren(this);
}


pyquilParser::ExprContext* pyquilParser::expr() {
   return expr(0);
}

pyquilParser::ExprContext* pyquilParser::expr(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  pyquilParser::ExprContext *_localctx = _tracker.createInstance<ExprContext>(_ctx, parentState);
  pyquilParser::ExprContext *previousContext = _localctx;
  (void)previousContext; // Silence compiler, in case the context is not used by generated code.
  size_t startState = 34;
  enterRecursionRule(_localctx, 34, pyquilParser::RuleExpr, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(188);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case pyquilParser::T__12: {
        setState(179);
        match(pyquilParser::T__12);
        setState(180);
        expr(8);
        break;
      }

      case pyquilParser::INT: {
        setState(181);
        match(pyquilParser::INT);
        break;
      }

      case pyquilParser::FLOAT: {
        setState(182);
        match(pyquilParser::FLOAT);
        break;
      }

      case pyquilParser::STRING: {
        setState(183);
        array_item();
        break;
      }

      case pyquilParser::T__10: {
        setState(184);
        match(pyquilParser::T__10);
        setState(185);
        expr(0);
        setState(186);
        match(pyquilParser::T__11);
        break;
      }

    default:
      throw NoViableAltException(this);
    }
    _ctx->stop = _input->LT(-1);
    setState(201);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(199);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(190);

          if (!(precpred(_ctx, 7))) throw FailedPredicateException(this, "precpred(_ctx, 7)");
          setState(191);
          _la = _input->LA(1);
          if (!(_la == pyquilParser::T__13

          || _la == pyquilParser::T__14)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(192);
          expr(8);
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(193);

          if (!(precpred(_ctx, 6))) throw FailedPredicateException(this, "precpred(_ctx, 6)");
          setState(194);
          _la = _input->LA(1);
          if (!(_la == pyquilParser::T__12

          || _la == pyquilParser::T__15)) {
          _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(195);
          expr(7);
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(196);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(197);
          match(pyquilParser::T__16);
          setState(198);
          expr(2);
          break;
        }

        } 
      }
      setState(203);
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

//----------------- Array_itemContext ------------------------------------------------------------------

pyquilParser::Array_itemContext::Array_itemContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

pyquilParser::ArraynameContext* pyquilParser::Array_itemContext::arrayname() {
  return getRuleContext<pyquilParser::ArraynameContext>(0);
}

pyquilParser::IdxContext* pyquilParser::Array_itemContext::idx() {
  return getRuleContext<pyquilParser::IdxContext>(0);
}


size_t pyquilParser::Array_itemContext::getRuleIndex() const {
  return pyquilParser::RuleArray_item;
}

void pyquilParser::Array_itemContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArray_item(this);
}

void pyquilParser::Array_itemContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArray_item(this);
}


antlrcpp::Any pyquilParser::Array_itemContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitArray_item(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::Array_itemContext* pyquilParser::array_item() {
  Array_itemContext *_localctx = _tracker.createInstance<Array_itemContext>(_ctx, getState());
  enterRule(_localctx, 36, pyquilParser::RuleArray_item);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(204);
    arrayname();
    setState(205);
    match(pyquilParser::T__8);
    setState(206);
    idx();
    setState(207);
    match(pyquilParser::T__9);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArraynameContext ------------------------------------------------------------------

pyquilParser::ArraynameContext::ArraynameContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::ArraynameContext::STRING() {
  return getToken(pyquilParser::STRING, 0);
}


size_t pyquilParser::ArraynameContext::getRuleIndex() const {
  return pyquilParser::RuleArrayname;
}

void pyquilParser::ArraynameContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterArrayname(this);
}

void pyquilParser::ArraynameContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitArrayname(this);
}


antlrcpp::Any pyquilParser::ArraynameContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitArrayname(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::ArraynameContext* pyquilParser::arrayname() {
  ArraynameContext *_localctx = _tracker.createInstance<ArraynameContext>(_ctx, getState());
  enterRule(_localctx, 38, pyquilParser::RuleArrayname);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(209);
    match(pyquilParser::STRING);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- IdxContext ------------------------------------------------------------------

pyquilParser::IdxContext::IdxContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* pyquilParser::IdxContext::INT() {
  return getToken(pyquilParser::INT, 0);
}


size_t pyquilParser::IdxContext::getRuleIndex() const {
  return pyquilParser::RuleIdx;
}

void pyquilParser::IdxContext::enterRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->enterIdx(this);
}

void pyquilParser::IdxContext::exitRule(tree::ParseTreeListener *listener) {
  auto parserListener = dynamic_cast<pyquilListener *>(listener);
  if (parserListener != nullptr)
    parserListener->exitIdx(this);
}


antlrcpp::Any pyquilParser::IdxContext::accept(tree::ParseTreeVisitor *visitor) {
  if (auto parserVisitor = dynamic_cast<pyquilVisitor*>(visitor))
    return parserVisitor->visitIdx(this);
  else
    return visitor->visitChildren(this);
}

pyquilParser::IdxContext* pyquilParser::idx() {
  IdxContext *_localctx = _tracker.createInstance<IdxContext>(_ctx, getState());
  enterRule(_localctx, 40, pyquilParser::RuleIdx);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(211);
    match(pyquilParser::INT);
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

bool pyquilParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 17: return exprSempred(dynamic_cast<ExprContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool pyquilParser::exprSempred(ExprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 7);
    case 1: return precpred(_ctx, 6);
    case 2: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> pyquilParser::_decisionToDFA;
atn::PredictionContextCache pyquilParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN pyquilParser::_atn;
std::vector<uint16_t> pyquilParser::_serializedATN;

std::vector<std::string> pyquilParser::_ruleNames = {
  "prog", "code_block", "loop", "loop_start", "loop_end", "loop_if_continue", 
  "operation", "declare", "measure", "move", "sub", "var_name", "var_mem", 
  "qbit", "gate", "bool_val", "param", "expr", "array_item", "arrayname", 
  "idx"
};

std::vector<std::string> pyquilParser::_literalNames = {
  "", "'DECLARE'", "'MEASURE'", "'MOVE'", "'SUB'", "'BIT'", "'REAL'", "'OCTET'", 
  "'INTEGER'", "'['", "']'", "'('", "')'", "'-'", "'*'", "'/'", "'+'", "'^'", 
  "", "", "", "", "", "'CSWAP'", "'PHASE'", "", "", "", "", "", "", "", 
  "", "'JUMP-UNLESS'"
};

std::vector<std::string> pyquilParser::_symbolicNames = {
  "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 
  "LABEL_LINE", "JUMP_LINE", "MODIFY_OP", "GATE1Q", "GATE2Q", "GATE3Q", 
  "GATE1Q1P", "GATE2Q1P", "LABELNAME", "STRING", "FLOAT", "INT", "SPACES", 
  "NEWLINE", "WS", "JUMP_UNLESS"
};

dfa::Vocabulary pyquilParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> pyquilParser::_tokenNames;

pyquilParser::Initializer::Initializer() {
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
    0x3, 0x23, 0xd8, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 0x9, 
    0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 0x4, 
    0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 0x9, 
    0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 0x4, 
    0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x4, 0x12, 
    0x9, 0x12, 0x4, 0x13, 0x9, 0x13, 0x4, 0x14, 0x9, 0x14, 0x4, 0x15, 0x9, 
    0x15, 0x4, 0x16, 0x9, 0x16, 0x3, 0x2, 0x6, 0x2, 0x2e, 0xa, 0x2, 0xd, 
    0x2, 0xe, 0x2, 0x2f, 0x3, 0x2, 0x6, 0x2, 0x33, 0xa, 0x2, 0xd, 0x2, 0xe, 
    0x2, 0x34, 0x3, 0x3, 0x3, 0x3, 0x6, 0x3, 0x39, 0xa, 0x3, 0xd, 0x3, 0xe, 
    0x3, 0x3a, 0x5, 0x3, 0x3d, 0xa, 0x3, 0x3, 0x4, 0x3, 0x4, 0x7, 0x4, 0x41, 
    0xa, 0x4, 0xc, 0x4, 0xe, 0x4, 0x44, 0xb, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 
    0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x8, 0x3, 
    0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x59, 0xa, 0x8, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xb, 0x3, 0xb, 
    0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xc, 0x3, 0xc, 
    0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 
    0x3, 0x10, 0x6, 0x10, 0x81, 0xa, 0x10, 0xd, 0x10, 0xe, 0x10, 0x82, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 
    0x10, 0x5, 0x10, 0xad, 0xa, 0x10, 0x3, 0x11, 0x3, 0x11, 0x5, 0x11, 0xb1, 
    0xa, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 
    0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
    0x5, 0x13, 0xbf, 0xa, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 
    0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x3, 0x13, 0x7, 0x13, 0xca, 
    0xa, 0x13, 0xc, 0x13, 0xe, 0x13, 0xcd, 0xb, 0x13, 0x3, 0x14, 0x3, 0x14, 
    0x3, 0x14, 0x3, 0x14, 0x3, 0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x16, 0x3, 
    0x16, 0x3, 0x16, 0x2, 0x3, 0x24, 0x17, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 
    0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x22, 0x24, 
    0x26, 0x28, 0x2a, 0x2, 0x5, 0x3, 0x2, 0x7, 0xa, 0x3, 0x2, 0x10, 0x11, 
    0x4, 0x2, 0xf, 0xf, 0x12, 0x12, 0x2, 0xd8, 0x2, 0x2d, 0x3, 0x2, 0x2, 
    0x2, 0x4, 0x3c, 0x3, 0x2, 0x2, 0x2, 0x6, 0x3e, 0x3, 0x2, 0x2, 0x2, 0x8, 
    0x49, 0x3, 0x2, 0x2, 0x2, 0xa, 0x4b, 0x3, 0x2, 0x2, 0x2, 0xc, 0x4d, 
    0x3, 0x2, 0x2, 0x2, 0xe, 0x58, 0x3, 0x2, 0x2, 0x2, 0x10, 0x5a, 0x3, 
    0x2, 0x2, 0x2, 0x12, 0x61, 0x3, 0x2, 0x2, 0x2, 0x14, 0x68, 0x3, 0x2, 
    0x2, 0x2, 0x16, 0x6f, 0x3, 0x2, 0x2, 0x2, 0x18, 0x76, 0x3, 0x2, 0x2, 
    0x2, 0x1a, 0x78, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x7d, 0x3, 0x2, 0x2, 0x2, 
    0x1e, 0xac, 0x3, 0x2, 0x2, 0x2, 0x20, 0xb0, 0x3, 0x2, 0x2, 0x2, 0x22, 
    0xb2, 0x3, 0x2, 0x2, 0x2, 0x24, 0xbe, 0x3, 0x2, 0x2, 0x2, 0x26, 0xce, 
    0x3, 0x2, 0x2, 0x2, 0x28, 0xd3, 0x3, 0x2, 0x2, 0x2, 0x2a, 0xd5, 0x3, 
    0x2, 0x2, 0x2, 0x2c, 0x2e, 0x5, 0x10, 0x9, 0x2, 0x2d, 0x2c, 0x3, 0x2, 
    0x2, 0x2, 0x2e, 0x2f, 0x3, 0x2, 0x2, 0x2, 0x2f, 0x2d, 0x3, 0x2, 0x2, 
    0x2, 0x2f, 0x30, 0x3, 0x2, 0x2, 0x2, 0x30, 0x32, 0x3, 0x2, 0x2, 0x2, 
    0x31, 0x33, 0x5, 0x4, 0x3, 0x2, 0x32, 0x31, 0x3, 0x2, 0x2, 0x2, 0x33, 
    0x34, 0x3, 0x2, 0x2, 0x2, 0x34, 0x32, 0x3, 0x2, 0x2, 0x2, 0x34, 0x35, 
    0x3, 0x2, 0x2, 0x2, 0x35, 0x3, 0x3, 0x2, 0x2, 0x2, 0x36, 0x3d, 0x5, 
    0x6, 0x4, 0x2, 0x37, 0x39, 0x5, 0xe, 0x8, 0x2, 0x38, 0x37, 0x3, 0x2, 
    0x2, 0x2, 0x39, 0x3a, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x38, 0x3, 0x2, 0x2, 
    0x2, 0x3a, 0x3b, 0x3, 0x2, 0x2, 0x2, 0x3b, 0x3d, 0x3, 0x2, 0x2, 0x2, 
    0x3c, 0x36, 0x3, 0x2, 0x2, 0x2, 0x3c, 0x38, 0x3, 0x2, 0x2, 0x2, 0x3d, 
    0x5, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x42, 0x5, 0x8, 0x5, 0x2, 0x3f, 0x41, 
    0x5, 0xe, 0x8, 0x2, 0x40, 0x3f, 0x3, 0x2, 0x2, 0x2, 0x41, 0x44, 0x3, 
    0x2, 0x2, 0x2, 0x42, 0x40, 0x3, 0x2, 0x2, 0x2, 0x42, 0x43, 0x3, 0x2, 
    0x2, 0x2, 0x43, 0x45, 0x3, 0x2, 0x2, 0x2, 0x44, 0x42, 0x3, 0x2, 0x2, 
    0x2, 0x45, 0x46, 0x5, 0xc, 0x7, 0x2, 0x46, 0x47, 0x7, 0x15, 0x2, 0x2, 
    0x47, 0x48, 0x5, 0xa, 0x6, 0x2, 0x48, 0x7, 0x3, 0x2, 0x2, 0x2, 0x49, 
    0x4a, 0x7, 0x14, 0x2, 0x2, 0x4a, 0x9, 0x3, 0x2, 0x2, 0x2, 0x4b, 0x4c, 
    0x7, 0x14, 0x2, 0x2, 0x4c, 0xb, 0x3, 0x2, 0x2, 0x2, 0x4d, 0x4e, 0x7, 
    0x23, 0x2, 0x2, 0x4e, 0x4f, 0x7, 0x20, 0x2, 0x2, 0x4f, 0x50, 0x7, 0x1c, 
    0x2, 0x2, 0x50, 0x51, 0x7, 0x20, 0x2, 0x2, 0x51, 0x52, 0x5, 0x20, 0x11, 
    0x2, 0x52, 0x53, 0x7, 0x21, 0x2, 0x2, 0x53, 0xd, 0x3, 0x2, 0x2, 0x2, 
    0x54, 0x59, 0x5, 0x12, 0xa, 0x2, 0x55, 0x59, 0x5, 0x1e, 0x10, 0x2, 0x56, 
    0x59, 0x5, 0x14, 0xb, 0x2, 0x57, 0x59, 0x5, 0x16, 0xc, 0x2, 0x58, 0x54, 
    0x3, 0x2, 0x2, 0x2, 0x58, 0x55, 0x3, 0x2, 0x2, 0x2, 0x58, 0x56, 0x3, 
    0x2, 0x2, 0x2, 0x58, 0x57, 0x3, 0x2, 0x2, 0x2, 0x59, 0xf, 0x3, 0x2, 
    0x2, 0x2, 0x5a, 0x5b, 0x7, 0x3, 0x2, 0x2, 0x5b, 0x5c, 0x7, 0x20, 0x2, 
    0x2, 0x5c, 0x5d, 0x5, 0x18, 0xd, 0x2, 0x5d, 0x5e, 0x7, 0x20, 0x2, 0x2, 
    0x5e, 0x5f, 0x5, 0x1a, 0xe, 0x2, 0x5f, 0x60, 0x7, 0x21, 0x2, 0x2, 0x60, 
    0x11, 0x3, 0x2, 0x2, 0x2, 0x61, 0x62, 0x7, 0x4, 0x2, 0x2, 0x62, 0x63, 
    0x7, 0x20, 0x2, 0x2, 0x63, 0x64, 0x5, 0x1c, 0xf, 0x2, 0x64, 0x65, 0x7, 
    0x20, 0x2, 0x2, 0x65, 0x66, 0x5, 0x26, 0x14, 0x2, 0x66, 0x67, 0x7, 0x21, 
    0x2, 0x2, 0x67, 0x13, 0x3, 0x2, 0x2, 0x2, 0x68, 0x69, 0x7, 0x5, 0x2, 
    0x2, 0x69, 0x6a, 0x7, 0x20, 0x2, 0x2, 0x6a, 0x6b, 0x5, 0x26, 0x14, 0x2, 
    0x6b, 0x6c, 0x7, 0x20, 0x2, 0x2, 0x6c, 0x6d, 0x5, 0x24, 0x13, 0x2, 0x6d, 
    0x6e, 0x7, 0x21, 0x2, 0x2, 0x6e, 0x15, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x70, 
    0x7, 0x6, 0x2, 0x2, 0x70, 0x71, 0x7, 0x20, 0x2, 0x2, 0x71, 0x72, 0x5, 
    0x26, 0x14, 0x2, 0x72, 0x73, 0x7, 0x20, 0x2, 0x2, 0x73, 0x74, 0x5, 0x24, 
    0x13, 0x2, 0x74, 0x75, 0x7, 0x21, 0x2, 0x2, 0x75, 0x17, 0x3, 0x2, 0x2, 
    0x2, 0x76, 0x77, 0x7, 0x1d, 0x2, 0x2, 0x77, 0x19, 0x3, 0x2, 0x2, 0x2, 
    0x78, 0x79, 0x9, 0x2, 0x2, 0x2, 0x79, 0x7a, 0x7, 0xb, 0x2, 0x2, 0x7a, 
    0x7b, 0x5, 0x2a, 0x16, 0x2, 0x7b, 0x7c, 0x7, 0xc, 0x2, 0x2, 0x7c, 0x1b, 
    0x3, 0x2, 0x2, 0x2, 0x7d, 0x7e, 0x7, 0x1f, 0x2, 0x2, 0x7e, 0x1d, 0x3, 
    0x2, 0x2, 0x2, 0x7f, 0x81, 0x7, 0x16, 0x2, 0x2, 0x80, 0x7f, 0x3, 0x2, 
    0x2, 0x2, 0x81, 0x82, 0x3, 0x2, 0x2, 0x2, 0x82, 0x80, 0x3, 0x2, 0x2, 
    0x2, 0x82, 0x83, 0x3, 0x2, 0x2, 0x2, 0x83, 0x84, 0x3, 0x2, 0x2, 0x2, 
    0x84, 0xad, 0x5, 0x1e, 0x10, 0x2, 0x85, 0x86, 0x7, 0x17, 0x2, 0x2, 0x86, 
    0x87, 0x7, 0x20, 0x2, 0x2, 0x87, 0x88, 0x5, 0x1c, 0xf, 0x2, 0x88, 0x89, 
    0x7, 0x21, 0x2, 0x2, 0x89, 0xad, 0x3, 0x2, 0x2, 0x2, 0x8a, 0x8b, 0x7, 
    0x18, 0x2, 0x2, 0x8b, 0x8c, 0x7, 0x20, 0x2, 0x2, 0x8c, 0x8d, 0x5, 0x1c, 
    0xf, 0x2, 0x8d, 0x8e, 0x7, 0x20, 0x2, 0x2, 0x8e, 0x8f, 0x5, 0x1c, 0xf, 
    0x2, 0x8f, 0x90, 0x7, 0x21, 0x2, 0x2, 0x90, 0xad, 0x3, 0x2, 0x2, 0x2, 
    0x91, 0x92, 0x7, 0x19, 0x2, 0x2, 0x92, 0x93, 0x7, 0x20, 0x2, 0x2, 0x93, 
    0x94, 0x5, 0x1c, 0xf, 0x2, 0x94, 0x95, 0x7, 0x20, 0x2, 0x2, 0x95, 0x96, 
    0x5, 0x1c, 0xf, 0x2, 0x96, 0x97, 0x7, 0x20, 0x2, 0x2, 0x97, 0x98, 0x5, 
    0x1c, 0xf, 0x2, 0x98, 0x99, 0x7, 0x21, 0x2, 0x2, 0x99, 0xad, 0x3, 0x2, 
    0x2, 0x2, 0x9a, 0x9b, 0x7, 0x1a, 0x2, 0x2, 0x9b, 0x9c, 0x7, 0xd, 0x2, 
    0x2, 0x9c, 0x9d, 0x5, 0x22, 0x12, 0x2, 0x9d, 0x9e, 0x7, 0xe, 0x2, 0x2, 
    0x9e, 0x9f, 0x7, 0x20, 0x2, 0x2, 0x9f, 0xa0, 0x5, 0x1c, 0xf, 0x2, 0xa0, 
    0xa1, 0x7, 0x21, 0x2, 0x2, 0xa1, 0xad, 0x3, 0x2, 0x2, 0x2, 0xa2, 0xa3, 
    0x7, 0x1b, 0x2, 0x2, 0xa3, 0xa4, 0x7, 0xd, 0x2, 0x2, 0xa4, 0xa5, 0x5, 
    0x22, 0x12, 0x2, 0xa5, 0xa6, 0x7, 0xe, 0x2, 0x2, 0xa6, 0xa7, 0x7, 0x20, 
    0x2, 0x2, 0xa7, 0xa8, 0x5, 0x1c, 0xf, 0x2, 0xa8, 0xa9, 0x7, 0x20, 0x2, 
    0x2, 0xa9, 0xaa, 0x5, 0x1c, 0xf, 0x2, 0xaa, 0xab, 0x7, 0x21, 0x2, 0x2, 
    0xab, 0xad, 0x3, 0x2, 0x2, 0x2, 0xac, 0x80, 0x3, 0x2, 0x2, 0x2, 0xac, 
    0x85, 0x3, 0x2, 0x2, 0x2, 0xac, 0x8a, 0x3, 0x2, 0x2, 0x2, 0xac, 0x91, 
    0x3, 0x2, 0x2, 0x2, 0xac, 0x9a, 0x3, 0x2, 0x2, 0x2, 0xac, 0xa2, 0x3, 
    0x2, 0x2, 0x2, 0xad, 0x1f, 0x3, 0x2, 0x2, 0x2, 0xae, 0xb1, 0x7, 0x1f, 
    0x2, 0x2, 0xaf, 0xb1, 0x5, 0x26, 0x14, 0x2, 0xb0, 0xae, 0x3, 0x2, 0x2, 
    0x2, 0xb0, 0xaf, 0x3, 0x2, 0x2, 0x2, 0xb1, 0x21, 0x3, 0x2, 0x2, 0x2, 
    0xb2, 0xb3, 0x5, 0x24, 0x13, 0x2, 0xb3, 0x23, 0x3, 0x2, 0x2, 0x2, 0xb4, 
    0xb5, 0x8, 0x13, 0x1, 0x2, 0xb5, 0xb6, 0x7, 0xf, 0x2, 0x2, 0xb6, 0xbf, 
    0x5, 0x24, 0x13, 0xa, 0xb7, 0xbf, 0x7, 0x1f, 0x2, 0x2, 0xb8, 0xbf, 0x7, 
    0x1e, 0x2, 0x2, 0xb9, 0xbf, 0x5, 0x26, 0x14, 0x2, 0xba, 0xbb, 0x7, 0xd, 
    0x2, 0x2, 0xbb, 0xbc, 0x5, 0x24, 0x13, 0x2, 0xbc, 0xbd, 0x7, 0xe, 0x2, 
    0x2, 0xbd, 0xbf, 0x3, 0x2, 0x2, 0x2, 0xbe, 0xb4, 0x3, 0x2, 0x2, 0x2, 
    0xbe, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xbe, 0xb8, 0x3, 0x2, 0x2, 0x2, 0xbe, 
    0xb9, 0x3, 0x2, 0x2, 0x2, 0xbe, 0xba, 0x3, 0x2, 0x2, 0x2, 0xbf, 0xcb, 
    0x3, 0x2, 0x2, 0x2, 0xc0, 0xc1, 0xc, 0x9, 0x2, 0x2, 0xc1, 0xc2, 0x9, 
    0x3, 0x2, 0x2, 0xc2, 0xca, 0x5, 0x24, 0x13, 0xa, 0xc3, 0xc4, 0xc, 0x8, 
    0x2, 0x2, 0xc4, 0xc5, 0x9, 0x4, 0x2, 0x2, 0xc5, 0xca, 0x5, 0x24, 0x13, 
    0x9, 0xc6, 0xc7, 0xc, 0x3, 0x2, 0x2, 0xc7, 0xc8, 0x7, 0x13, 0x2, 0x2, 
    0xc8, 0xca, 0x5, 0x24, 0x13, 0x4, 0xc9, 0xc0, 0x3, 0x2, 0x2, 0x2, 0xc9, 
    0xc3, 0x3, 0x2, 0x2, 0x2, 0xc9, 0xc6, 0x3, 0x2, 0x2, 0x2, 0xca, 0xcd, 
    0x3, 0x2, 0x2, 0x2, 0xcb, 0xc9, 0x3, 0x2, 0x2, 0x2, 0xcb, 0xcc, 0x3, 
    0x2, 0x2, 0x2, 0xcc, 0x25, 0x3, 0x2, 0x2, 0x2, 0xcd, 0xcb, 0x3, 0x2, 
    0x2, 0x2, 0xce, 0xcf, 0x5, 0x28, 0x15, 0x2, 0xcf, 0xd0, 0x7, 0xb, 0x2, 
    0x2, 0xd0, 0xd1, 0x5, 0x2a, 0x16, 0x2, 0xd1, 0xd2, 0x7, 0xc, 0x2, 0x2, 
    0xd2, 0x27, 0x3, 0x2, 0x2, 0x2, 0xd3, 0xd4, 0x7, 0x1d, 0x2, 0x2, 0xd4, 
    0x29, 0x3, 0x2, 0x2, 0x2, 0xd5, 0xd6, 0x7, 0x1f, 0x2, 0x2, 0xd6, 0x2b, 
    0x3, 0x2, 0x2, 0x2, 0xe, 0x2f, 0x34, 0x3a, 0x3c, 0x42, 0x58, 0x82, 0xac, 
    0xb0, 0xbe, 0xc9, 0xcb, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

pyquilParser::Initializer pyquilParser::_init;
