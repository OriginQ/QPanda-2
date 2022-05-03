
// Generated from .\originir.g4 by ANTLR 4.9.2
#pragma once

#include "Core/Utilities/Compiler/OriginIRCompiler/statementListener.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/statementVisitor.h"

#include "Core/Utilities/Compiler/OriginIRCompiler/statementParser.h"

namespace statement {

    using namespace antlrcpp;
    using namespace antlr4;

    statementParser::statementParser(TokenStream *input) : Parser(input) {
      _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
    }

    statementParser::~statementParser() {
      delete _interpreter;
    }

    std::string statementParser::getGrammarFileName() const {
      return "statement.g4";
    }

    const std::vector<std::string>& statementParser::getRuleNames() const {
      return _ruleNames;
    }

    dfa::Vocabulary& statementParser::getVocabulary() const {
      return _vocabulary;
    }


    //----------------- Translationunit_sContext ------------------------------------------------------------------

    statementParser::Translationunit_sContext::Translationunit_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    std::vector<tree::TerminalNode *> statementParser::Translationunit_sContext::NEWLINE() {
      return getTokens(statementParser::NEWLINE);
    }

    tree::TerminalNode* statementParser::Translationunit_sContext::NEWLINE(size_t i) {
      return getToken(statementParser::NEWLINE, i);
    }

    std::vector<statementParser::Statement_sContext *> statementParser::Translationunit_sContext::statement_s() {
      return getRuleContexts<statementParser::Statement_sContext>();
    }

    statementParser::Statement_sContext* statementParser::Translationunit_sContext::statement_s(size_t i) {
      return getRuleContext<statementParser::Statement_sContext>(i);
    }


    size_t statementParser::Translationunit_sContext::getRuleIndex() const {
      return statementParser::RuleTranslationunit_s;
    }

    void statementParser::Translationunit_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterTranslationunit_s(this);
    }

    void statementParser::Translationunit_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitTranslationunit_s(this);
    }


    antlrcpp::Any statementParser::Translationunit_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitTranslationunit_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Translationunit_sContext* statementParser::translationunit_s() {
      Translationunit_sContext *_localctx = _tracker.createInstance<Translationunit_sContext>(_ctx, getState());
      enterRule(_localctx, 0, statementParser::RuleTranslationunit_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(103);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == statementParser::NEWLINE) {
          setState(100);
          match(statementParser::NEWLINE);
          setState(105);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(107); 
        _errHandler->sync(this);
        _la = _input->LA(1);
        do {
          setState(106);
          statement_s();
          setState(109); 
          _errHandler->sync(this);
          _la = _input->LA(1);
        } while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::PI)
          | (1ULL << statementParser::C_KEY)
          | (1ULL << statementParser::BARRIER_KEY)
          | (1ULL << statementParser::ECHO_GATE)
          | (1ULL << statementParser::H_GATE)
          | (1ULL << statementParser::X_GATE)
          | (1ULL << statementParser::T_GATE)
          | (1ULL << statementParser::S_GATE)
          | (1ULL << statementParser::Y_GATE)
          | (1ULL << statementParser::Z_GATE)
          | (1ULL << statementParser::X1_GATE)
          | (1ULL << statementParser::Y1_GATE)
          | (1ULL << statementParser::Z1_GATE)
          | (1ULL << statementParser::I_GATE)
          | (1ULL << statementParser::U2_GATE)
          | (1ULL << statementParser::RPHI_GATE)
          | (1ULL << statementParser::U3_GATE)
          | (1ULL << statementParser::U4_GATE)
          | (1ULL << statementParser::RX_GATE)
          | (1ULL << statementParser::RY_GATE)
          | (1ULL << statementParser::RZ_GATE)
          | (1ULL << statementParser::U1_GATE)
          | (1ULL << statementParser::CNOT_GATE)
          | (1ULL << statementParser::CZ_GATE)
          | (1ULL << statementParser::CU_GATE)
          | (1ULL << statementParser::ISWAP_GATE)
          | (1ULL << statementParser::SQISWAP_GATE)
          | (1ULL << statementParser::SWAPZ1_GATE)
          | (1ULL << statementParser::ISWAPTHETA_GATE)
          | (1ULL << statementParser::CR_GATE)
          | (1ULL << statementParser::TOFFOLI_GATE)
          | (1ULL << statementParser::DAGGER_KEY)
          | (1ULL << statementParser::CONTROL_KEY)
          | (1ULL << statementParser::QIF_KEY)
          | (1ULL << statementParser::QWHILE_KEY)
          | (1ULL << statementParser::MEASURE_KEY)
          | (1ULL << statementParser::RESET_KEY)
          | (1ULL << statementParser::NOT)
          | (1ULL << statementParser::PLUS)
          | (1ULL << statementParser::MINUS))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 64)) & ((1ULL << (statementParser::LPAREN - 64))
          | (1ULL << (statementParser::Integer_Literal_s - 64))
          | (1ULL << (statementParser::Double_Literal_s - 64)))) != 0));
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Quantum_gate_declaration_sContext ------------------------------------------------------------------

    statementParser::Quantum_gate_declaration_sContext::Quantum_gate_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Single_gate_without_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::single_gate_without_parameter_declaration_s() {
      return getRuleContext<statementParser::Single_gate_without_parameter_declaration_sContext>(0);
    }

    statementParser::Single_gate_with_one_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::single_gate_with_one_parameter_declaration_s() {
      return getRuleContext<statementParser::Single_gate_with_one_parameter_declaration_sContext>(0);
    }

    statementParser::Single_gate_with_two_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::single_gate_with_two_parameter_declaration_s() {
      return getRuleContext<statementParser::Single_gate_with_two_parameter_declaration_sContext>(0);
    }

    statementParser::Single_gate_with_three_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::single_gate_with_three_parameter_declaration_s() {
      return getRuleContext<statementParser::Single_gate_with_three_parameter_declaration_sContext>(0);
    }

    statementParser::Single_gate_with_four_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::single_gate_with_four_parameter_declaration_s() {
      return getRuleContext<statementParser::Single_gate_with_four_parameter_declaration_sContext>(0);
    }

    statementParser::Double_gate_without_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::double_gate_without_parameter_declaration_s() {
      return getRuleContext<statementParser::Double_gate_without_parameter_declaration_sContext>(0);
    }

    statementParser::Double_gate_with_one_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::double_gate_with_one_parameter_declaration_s() {
      return getRuleContext<statementParser::Double_gate_with_one_parameter_declaration_sContext>(0);
    }

    statementParser::Double_gate_with_four_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::double_gate_with_four_parameter_declaration_s() {
      return getRuleContext<statementParser::Double_gate_with_four_parameter_declaration_sContext>(0);
    }

    statementParser::Triple_gate_without_parameter_declaration_sContext* statementParser::Quantum_gate_declaration_sContext::triple_gate_without_parameter_declaration_s() {
      return getRuleContext<statementParser::Triple_gate_without_parameter_declaration_sContext>(0);
    }


    size_t statementParser::Quantum_gate_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleQuantum_gate_declaration_s;
    }

    void statementParser::Quantum_gate_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterQuantum_gate_declaration_s(this);
    }

    void statementParser::Quantum_gate_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitQuantum_gate_declaration_s(this);
    }


    antlrcpp::Any statementParser::Quantum_gate_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitQuantum_gate_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Quantum_gate_declaration_sContext* statementParser::quantum_gate_declaration_s() {
      Quantum_gate_declaration_sContext *_localctx = _tracker.createInstance<Quantum_gate_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 2, statementParser::RuleQuantum_gate_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(120);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case statementParser::ECHO_GATE:
          case statementParser::H_GATE:
          case statementParser::X_GATE:
          case statementParser::T_GATE:
          case statementParser::S_GATE:
          case statementParser::Y_GATE:
          case statementParser::Z_GATE:
          case statementParser::X1_GATE:
          case statementParser::Y1_GATE:
          case statementParser::Z1_GATE:
          case statementParser::I_GATE: {
            enterOuterAlt(_localctx, 1);
            setState(111);
            single_gate_without_parameter_declaration_s();
            break;
          }

          case statementParser::RX_GATE:
          case statementParser::RY_GATE:
          case statementParser::RZ_GATE:
          case statementParser::U1_GATE: {
            enterOuterAlt(_localctx, 2);
            setState(112);
            single_gate_with_one_parameter_declaration_s();
            break;
          }

          case statementParser::U2_GATE:
          case statementParser::RPHI_GATE: {
            enterOuterAlt(_localctx, 3);
            setState(113);
            single_gate_with_two_parameter_declaration_s();
            break;
          }

          case statementParser::U3_GATE: {
            enterOuterAlt(_localctx, 4);
            setState(114);
            single_gate_with_three_parameter_declaration_s();
            break;
          }

          case statementParser::U4_GATE: {
            enterOuterAlt(_localctx, 5);
            setState(115);
            single_gate_with_four_parameter_declaration_s();
            break;
          }

          case statementParser::CNOT_GATE:
          case statementParser::CZ_GATE:
          case statementParser::ISWAP_GATE:
          case statementParser::SQISWAP_GATE:
          case statementParser::SWAPZ1_GATE: {
            enterOuterAlt(_localctx, 6);
            setState(116);
            double_gate_without_parameter_declaration_s();
            break;
          }

          case statementParser::ISWAPTHETA_GATE:
          case statementParser::CR_GATE: {
            enterOuterAlt(_localctx, 7);
            setState(117);
            double_gate_with_one_parameter_declaration_s();
            break;
          }

          case statementParser::CU_GATE: {
            enterOuterAlt(_localctx, 8);
            setState(118);
            double_gate_with_four_parameter_declaration_s();
            break;
          }

          case statementParser::TOFFOLI_GATE: {
            enterOuterAlt(_localctx, 9);
            setState(119);
            triple_gate_without_parameter_declaration_s();
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

    //----------------- Index_sContext ------------------------------------------------------------------

    statementParser::Index_sContext::Index_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Index_sContext::LBRACK() {
      return getToken(statementParser::LBRACK, 0);
    }

    statementParser::Expression_sContext* statementParser::Index_sContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Index_sContext::RBRACK() {
      return getToken(statementParser::RBRACK, 0);
    }


    size_t statementParser::Index_sContext::getRuleIndex() const {
      return statementParser::RuleIndex_s;
    }

    void statementParser::Index_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterIndex_s(this);
    }

    void statementParser::Index_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitIndex_s(this);
    }


    antlrcpp::Any statementParser::Index_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitIndex_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Index_sContext* statementParser::index_s() {
      Index_sContext *_localctx = _tracker.createInstance<Index_sContext>(_ctx, getState());
      enterRule(_localctx, 4, statementParser::RuleIndex_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(122);
        match(statementParser::LBRACK);
        setState(123);
        expression_s();
        setState(124);
        match(statementParser::RBRACK);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- C_KEY_declaration_sContext ------------------------------------------------------------------

    statementParser::C_KEY_declaration_sContext::C_KEY_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::C_KEY_declaration_sContext::C_KEY() {
      return getToken(statementParser::C_KEY, 0);
    }

    statementParser::Index_sContext* statementParser::C_KEY_declaration_sContext::index_s() {
      return getRuleContext<statementParser::Index_sContext>(0);
    }


    size_t statementParser::C_KEY_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleC_KEY_declaration_s;
    }

    void statementParser::C_KEY_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterC_KEY_declaration_s(this);
    }

    void statementParser::C_KEY_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitC_KEY_declaration_s(this);
    }


    antlrcpp::Any statementParser::C_KEY_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitC_KEY_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::C_KEY_declaration_sContext* statementParser::c_KEY_declaration_s() {
      C_KEY_declaration_sContext *_localctx = _tracker.createInstance<C_KEY_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 6, statementParser::RuleC_KEY_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(126);
        match(statementParser::C_KEY);
        setState(127);
        index_s();
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Q_KEY_declaration_sContext ------------------------------------------------------------------

    statementParser::Q_KEY_declaration_sContext::Q_KEY_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Q_KEY_declaration_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }

    statementParser::Index_sContext* statementParser::Q_KEY_declaration_sContext::index_s() {
      return getRuleContext<statementParser::Index_sContext>(0);
    }


    size_t statementParser::Q_KEY_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleQ_KEY_declaration_s;
    }

    void statementParser::Q_KEY_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterQ_KEY_declaration_s(this);
    }

    void statementParser::Q_KEY_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitQ_KEY_declaration_s(this);
    }


    antlrcpp::Any statementParser::Q_KEY_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitQ_KEY_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::q_KEY_declaration_s() {
      Q_KEY_declaration_sContext *_localctx = _tracker.createInstance<Q_KEY_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 8, statementParser::RuleQ_KEY_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(129);
        match(statementParser::Q_KEY);
        setState(130);
        index_s();
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Single_gate_without_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Single_gate_without_parameter_declaration_sContext::Single_gate_without_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Single_gate_without_parameter_type_sContext* statementParser::Single_gate_without_parameter_declaration_sContext::single_gate_without_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_without_parameter_type_sContext>(0);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Single_gate_without_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_declaration_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }


    size_t statementParser::Single_gate_without_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_without_parameter_declaration_s;
    }

    void statementParser::Single_gate_without_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_without_parameter_declaration_s(this);
    }

    void statementParser::Single_gate_without_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_without_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_without_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_without_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_without_parameter_declaration_sContext* statementParser::single_gate_without_parameter_declaration_s() {
      Single_gate_without_parameter_declaration_sContext *_localctx = _tracker.createInstance<Single_gate_without_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 10, statementParser::RuleSingle_gate_without_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(138);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 3, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(132);
          single_gate_without_parameter_type_s();
          setState(133);
          q_KEY_declaration_s();
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(135);
          single_gate_without_parameter_type_s();
          setState(136);
          match(statementParser::Q_KEY);
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

    //----------------- Single_gate_with_one_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_one_parameter_declaration_sContext::Single_gate_with_one_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Single_gate_with_one_parameter_type_sContext* statementParser::Single_gate_with_one_parameter_declaration_sContext::single_gate_with_one_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_one_parameter_type_sContext>(0);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Single_gate_with_one_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_declaration_sContext::COMMA() {
      return getToken(statementParser::COMMA, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_declaration_sContext::LPAREN() {
      return getToken(statementParser::LPAREN, 0);
    }

    statementParser::Expression_sContext* statementParser::Single_gate_with_one_parameter_declaration_sContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_declaration_sContext::RPAREN() {
      return getToken(statementParser::RPAREN, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_declaration_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }


    size_t statementParser::Single_gate_with_one_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_one_parameter_declaration_s;
    }

    void statementParser::Single_gate_with_one_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_one_parameter_declaration_s(this);
    }

    void statementParser::Single_gate_with_one_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_one_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_one_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_one_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_one_parameter_declaration_sContext* statementParser::single_gate_with_one_parameter_declaration_s() {
      Single_gate_with_one_parameter_declaration_sContext *_localctx = _tracker.createInstance<Single_gate_with_one_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 12, statementParser::RuleSingle_gate_with_one_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(154);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(140);
          single_gate_with_one_parameter_type_s();
          setState(141);
          q_KEY_declaration_s();
          setState(142);
          match(statementParser::COMMA);
          setState(143);
          match(statementParser::LPAREN);
          setState(144);
          expression_s();
          setState(145);
          match(statementParser::RPAREN);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(147);
          single_gate_with_one_parameter_type_s();
          setState(148);
          match(statementParser::Q_KEY);
          setState(149);
          match(statementParser::COMMA);
          setState(150);
          match(statementParser::LPAREN);
          setState(151);
          expression_s();
          setState(152);
          match(statementParser::RPAREN);
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

    //----------------- Single_gate_with_two_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_two_parameter_declaration_sContext::Single_gate_with_two_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Single_gate_with_two_parameter_type_sContext* statementParser::Single_gate_with_two_parameter_declaration_sContext::single_gate_with_two_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_two_parameter_type_sContext>(0);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Single_gate_with_two_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(0);
    }

    std::vector<tree::TerminalNode *> statementParser::Single_gate_with_two_parameter_declaration_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Single_gate_with_two_parameter_declaration_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }

    tree::TerminalNode* statementParser::Single_gate_with_two_parameter_declaration_sContext::LPAREN() {
      return getToken(statementParser::LPAREN, 0);
    }

    std::vector<statementParser::Expression_sContext *> statementParser::Single_gate_with_two_parameter_declaration_sContext::expression_s() {
      return getRuleContexts<statementParser::Expression_sContext>();
    }

    statementParser::Expression_sContext* statementParser::Single_gate_with_two_parameter_declaration_sContext::expression_s(size_t i) {
      return getRuleContext<statementParser::Expression_sContext>(i);
    }

    tree::TerminalNode* statementParser::Single_gate_with_two_parameter_declaration_sContext::RPAREN() {
      return getToken(statementParser::RPAREN, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_two_parameter_declaration_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }


    size_t statementParser::Single_gate_with_two_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_two_parameter_declaration_s;
    }

    void statementParser::Single_gate_with_two_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_two_parameter_declaration_s(this);
    }

    void statementParser::Single_gate_with_two_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_two_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_two_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_two_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_two_parameter_declaration_sContext* statementParser::single_gate_with_two_parameter_declaration_s() {
      Single_gate_with_two_parameter_declaration_sContext *_localctx = _tracker.createInstance<Single_gate_with_two_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 14, statementParser::RuleSingle_gate_with_two_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(174);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(156);
          single_gate_with_two_parameter_type_s();
          setState(157);
          q_KEY_declaration_s();
          setState(158);
          match(statementParser::COMMA);
          setState(159);
          match(statementParser::LPAREN);
          setState(160);
          expression_s();
          setState(161);
          match(statementParser::COMMA);
          setState(162);
          expression_s();
          setState(163);
          match(statementParser::RPAREN);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(165);
          single_gate_with_two_parameter_type_s();
          setState(166);
          match(statementParser::Q_KEY);
          setState(167);
          match(statementParser::COMMA);
          setState(168);
          match(statementParser::LPAREN);
          setState(169);
          expression_s();
          setState(170);
          match(statementParser::COMMA);
          setState(171);
          expression_s();
          setState(172);
          match(statementParser::RPAREN);
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

    //----------------- Single_gate_with_three_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_three_parameter_declaration_sContext::Single_gate_with_three_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Single_gate_with_three_parameter_type_sContext* statementParser::Single_gate_with_three_parameter_declaration_sContext::single_gate_with_three_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_three_parameter_type_sContext>(0);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Single_gate_with_three_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(0);
    }

    std::vector<tree::TerminalNode *> statementParser::Single_gate_with_three_parameter_declaration_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Single_gate_with_three_parameter_declaration_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }

    tree::TerminalNode* statementParser::Single_gate_with_three_parameter_declaration_sContext::LPAREN() {
      return getToken(statementParser::LPAREN, 0);
    }

    std::vector<statementParser::Expression_sContext *> statementParser::Single_gate_with_three_parameter_declaration_sContext::expression_s() {
      return getRuleContexts<statementParser::Expression_sContext>();
    }

    statementParser::Expression_sContext* statementParser::Single_gate_with_three_parameter_declaration_sContext::expression_s(size_t i) {
      return getRuleContext<statementParser::Expression_sContext>(i);
    }

    tree::TerminalNode* statementParser::Single_gate_with_three_parameter_declaration_sContext::RPAREN() {
      return getToken(statementParser::RPAREN, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_three_parameter_declaration_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }


    size_t statementParser::Single_gate_with_three_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_three_parameter_declaration_s;
    }

    void statementParser::Single_gate_with_three_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_three_parameter_declaration_s(this);
    }

    void statementParser::Single_gate_with_three_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_three_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_three_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_three_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_three_parameter_declaration_sContext* statementParser::single_gate_with_three_parameter_declaration_s() {
      Single_gate_with_three_parameter_declaration_sContext *_localctx = _tracker.createInstance<Single_gate_with_three_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 16, statementParser::RuleSingle_gate_with_three_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(198);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(176);
          single_gate_with_three_parameter_type_s();
          setState(177);
          q_KEY_declaration_s();
          setState(178);
          match(statementParser::COMMA);
          setState(179);
          match(statementParser::LPAREN);
          setState(180);
          expression_s();
          setState(181);
          match(statementParser::COMMA);
          setState(182);
          expression_s();
          setState(183);
          match(statementParser::COMMA);
          setState(184);
          expression_s();
          setState(185);
          match(statementParser::RPAREN);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(187);
          single_gate_with_three_parameter_type_s();
          setState(188);
          match(statementParser::Q_KEY);
          setState(189);
          match(statementParser::COMMA);
          setState(190);
          match(statementParser::LPAREN);
          setState(191);
          expression_s();
          setState(192);
          match(statementParser::COMMA);
          setState(193);
          expression_s();
          setState(194);
          match(statementParser::COMMA);
          setState(195);
          expression_s();
          setState(196);
          match(statementParser::RPAREN);
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

    //----------------- Single_gate_with_four_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_four_parameter_declaration_sContext::Single_gate_with_four_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Single_gate_with_four_parameter_type_sContext* statementParser::Single_gate_with_four_parameter_declaration_sContext::single_gate_with_four_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_four_parameter_type_sContext>(0);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Single_gate_with_four_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(0);
    }

    std::vector<tree::TerminalNode *> statementParser::Single_gate_with_four_parameter_declaration_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Single_gate_with_four_parameter_declaration_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }

    tree::TerminalNode* statementParser::Single_gate_with_four_parameter_declaration_sContext::LPAREN() {
      return getToken(statementParser::LPAREN, 0);
    }

    std::vector<statementParser::Expression_sContext *> statementParser::Single_gate_with_four_parameter_declaration_sContext::expression_s() {
      return getRuleContexts<statementParser::Expression_sContext>();
    }

    statementParser::Expression_sContext* statementParser::Single_gate_with_four_parameter_declaration_sContext::expression_s(size_t i) {
      return getRuleContext<statementParser::Expression_sContext>(i);
    }

    tree::TerminalNode* statementParser::Single_gate_with_four_parameter_declaration_sContext::RPAREN() {
      return getToken(statementParser::RPAREN, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_four_parameter_declaration_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }


    size_t statementParser::Single_gate_with_four_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_four_parameter_declaration_s;
    }

    void statementParser::Single_gate_with_four_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_four_parameter_declaration_s(this);
    }

    void statementParser::Single_gate_with_four_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_four_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_four_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_four_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_four_parameter_declaration_sContext* statementParser::single_gate_with_four_parameter_declaration_s() {
      Single_gate_with_four_parameter_declaration_sContext *_localctx = _tracker.createInstance<Single_gate_with_four_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 18, statementParser::RuleSingle_gate_with_four_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(226);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(200);
          single_gate_with_four_parameter_type_s();
          setState(201);
          q_KEY_declaration_s();
          setState(202);
          match(statementParser::COMMA);
          setState(203);
          match(statementParser::LPAREN);
          setState(204);
          expression_s();
          setState(205);
          match(statementParser::COMMA);
          setState(206);
          expression_s();
          setState(207);
          match(statementParser::COMMA);
          setState(208);
          expression_s();
          setState(209);
          match(statementParser::COMMA);
          setState(210);
          expression_s();
          setState(211);
          match(statementParser::RPAREN);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(213);
          single_gate_with_four_parameter_type_s();
          setState(214);
          match(statementParser::Q_KEY);
          setState(215);
          match(statementParser::COMMA);
          setState(216);
          match(statementParser::LPAREN);
          setState(217);
          expression_s();
          setState(218);
          match(statementParser::COMMA);
          setState(219);
          expression_s();
          setState(220);
          match(statementParser::COMMA);
          setState(221);
          expression_s();
          setState(222);
          match(statementParser::COMMA);
          setState(223);
          expression_s();
          setState(224);
          match(statementParser::RPAREN);
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

    //----------------- Double_gate_without_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Double_gate_without_parameter_declaration_sContext::Double_gate_without_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Double_gate_without_parameter_type_sContext* statementParser::Double_gate_without_parameter_declaration_sContext::double_gate_without_parameter_type_s() {
      return getRuleContext<statementParser::Double_gate_without_parameter_type_sContext>(0);
    }

    std::vector<statementParser::Q_KEY_declaration_sContext *> statementParser::Double_gate_without_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContexts<statementParser::Q_KEY_declaration_sContext>();
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Double_gate_without_parameter_declaration_sContext::q_KEY_declaration_s(size_t i) {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(i);
    }

    tree::TerminalNode* statementParser::Double_gate_without_parameter_declaration_sContext::COMMA() {
      return getToken(statementParser::COMMA, 0);
    }


    size_t statementParser::Double_gate_without_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleDouble_gate_without_parameter_declaration_s;
    }

    void statementParser::Double_gate_without_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterDouble_gate_without_parameter_declaration_s(this);
    }

    void statementParser::Double_gate_without_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitDouble_gate_without_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Double_gate_without_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitDouble_gate_without_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Double_gate_without_parameter_declaration_sContext* statementParser::double_gate_without_parameter_declaration_s() {
      Double_gate_without_parameter_declaration_sContext *_localctx = _tracker.createInstance<Double_gate_without_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 20, statementParser::RuleDouble_gate_without_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(228);
        double_gate_without_parameter_type_s();
        setState(229);
        q_KEY_declaration_s();
        setState(230);
        match(statementParser::COMMA);
        setState(231);
        q_KEY_declaration_s();
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Double_gate_with_one_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Double_gate_with_one_parameter_declaration_sContext::Double_gate_with_one_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Double_gate_with_one_parameter_type_sContext* statementParser::Double_gate_with_one_parameter_declaration_sContext::double_gate_with_one_parameter_type_s() {
      return getRuleContext<statementParser::Double_gate_with_one_parameter_type_sContext>(0);
    }

    std::vector<statementParser::Q_KEY_declaration_sContext *> statementParser::Double_gate_with_one_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContexts<statementParser::Q_KEY_declaration_sContext>();
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Double_gate_with_one_parameter_declaration_sContext::q_KEY_declaration_s(size_t i) {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(i);
    }

    std::vector<tree::TerminalNode *> statementParser::Double_gate_with_one_parameter_declaration_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Double_gate_with_one_parameter_declaration_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }

    tree::TerminalNode* statementParser::Double_gate_with_one_parameter_declaration_sContext::LPAREN() {
      return getToken(statementParser::LPAREN, 0);
    }

    statementParser::Expression_sContext* statementParser::Double_gate_with_one_parameter_declaration_sContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Double_gate_with_one_parameter_declaration_sContext::RPAREN() {
      return getToken(statementParser::RPAREN, 0);
    }


    size_t statementParser::Double_gate_with_one_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleDouble_gate_with_one_parameter_declaration_s;
    }

    void statementParser::Double_gate_with_one_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterDouble_gate_with_one_parameter_declaration_s(this);
    }

    void statementParser::Double_gate_with_one_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitDouble_gate_with_one_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Double_gate_with_one_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitDouble_gate_with_one_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Double_gate_with_one_parameter_declaration_sContext* statementParser::double_gate_with_one_parameter_declaration_s() {
      Double_gate_with_one_parameter_declaration_sContext *_localctx = _tracker.createInstance<Double_gate_with_one_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 22, statementParser::RuleDouble_gate_with_one_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(233);
        double_gate_with_one_parameter_type_s();
        setState(234);
        q_KEY_declaration_s();
        setState(235);
        match(statementParser::COMMA);
        setState(236);
        q_KEY_declaration_s();
        setState(237);
        match(statementParser::COMMA);
        setState(238);
        match(statementParser::LPAREN);
        setState(239);
        expression_s();
        setState(240);
        match(statementParser::RPAREN);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Double_gate_with_four_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Double_gate_with_four_parameter_declaration_sContext::Double_gate_with_four_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Double_gate_with_four_parameter_type_sContext* statementParser::Double_gate_with_four_parameter_declaration_sContext::double_gate_with_four_parameter_type_s() {
      return getRuleContext<statementParser::Double_gate_with_four_parameter_type_sContext>(0);
    }

    std::vector<statementParser::Q_KEY_declaration_sContext *> statementParser::Double_gate_with_four_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContexts<statementParser::Q_KEY_declaration_sContext>();
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Double_gate_with_four_parameter_declaration_sContext::q_KEY_declaration_s(size_t i) {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(i);
    }

    std::vector<tree::TerminalNode *> statementParser::Double_gate_with_four_parameter_declaration_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Double_gate_with_four_parameter_declaration_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }

    tree::TerminalNode* statementParser::Double_gate_with_four_parameter_declaration_sContext::LPAREN() {
      return getToken(statementParser::LPAREN, 0);
    }

    std::vector<statementParser::Expression_sContext *> statementParser::Double_gate_with_four_parameter_declaration_sContext::expression_s() {
      return getRuleContexts<statementParser::Expression_sContext>();
    }

    statementParser::Expression_sContext* statementParser::Double_gate_with_four_parameter_declaration_sContext::expression_s(size_t i) {
      return getRuleContext<statementParser::Expression_sContext>(i);
    }

    tree::TerminalNode* statementParser::Double_gate_with_four_parameter_declaration_sContext::RPAREN() {
      return getToken(statementParser::RPAREN, 0);
    }


    size_t statementParser::Double_gate_with_four_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleDouble_gate_with_four_parameter_declaration_s;
    }

    void statementParser::Double_gate_with_four_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterDouble_gate_with_four_parameter_declaration_s(this);
    }

    void statementParser::Double_gate_with_four_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitDouble_gate_with_four_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Double_gate_with_four_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitDouble_gate_with_four_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Double_gate_with_four_parameter_declaration_sContext* statementParser::double_gate_with_four_parameter_declaration_s() {
      Double_gate_with_four_parameter_declaration_sContext *_localctx = _tracker.createInstance<Double_gate_with_four_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 24, statementParser::RuleDouble_gate_with_four_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(242);
        double_gate_with_four_parameter_type_s();
        setState(243);
        q_KEY_declaration_s();
        setState(244);
        match(statementParser::COMMA);
        setState(245);
        q_KEY_declaration_s();
        setState(246);
        match(statementParser::COMMA);
        setState(247);
        match(statementParser::LPAREN);
        setState(248);
        expression_s();
        setState(249);
        match(statementParser::COMMA);
        setState(250);
        expression_s();
        setState(251);
        match(statementParser::COMMA);
        setState(252);
        expression_s();
        setState(253);
        match(statementParser::COMMA);
        setState(254);
        expression_s();
        setState(255);
        match(statementParser::RPAREN);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Triple_gate_without_parameter_declaration_sContext ------------------------------------------------------------------

    statementParser::Triple_gate_without_parameter_declaration_sContext::Triple_gate_without_parameter_declaration_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Triple_gate_without_parameter_type_sContext* statementParser::Triple_gate_without_parameter_declaration_sContext::triple_gate_without_parameter_type_s() {
      return getRuleContext<statementParser::Triple_gate_without_parameter_type_sContext>(0);
    }

    std::vector<statementParser::Q_KEY_declaration_sContext *> statementParser::Triple_gate_without_parameter_declaration_sContext::q_KEY_declaration_s() {
      return getRuleContexts<statementParser::Q_KEY_declaration_sContext>();
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Triple_gate_without_parameter_declaration_sContext::q_KEY_declaration_s(size_t i) {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(i);
    }

    std::vector<tree::TerminalNode *> statementParser::Triple_gate_without_parameter_declaration_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Triple_gate_without_parameter_declaration_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }


    size_t statementParser::Triple_gate_without_parameter_declaration_sContext::getRuleIndex() const {
      return statementParser::RuleTriple_gate_without_parameter_declaration_s;
    }

    void statementParser::Triple_gate_without_parameter_declaration_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterTriple_gate_without_parameter_declaration_s(this);
    }

    void statementParser::Triple_gate_without_parameter_declaration_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitTriple_gate_without_parameter_declaration_s(this);
    }


    antlrcpp::Any statementParser::Triple_gate_without_parameter_declaration_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitTriple_gate_without_parameter_declaration_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Triple_gate_without_parameter_declaration_sContext* statementParser::triple_gate_without_parameter_declaration_s() {
      Triple_gate_without_parameter_declaration_sContext *_localctx = _tracker.createInstance<Triple_gate_without_parameter_declaration_sContext>(_ctx, getState());
      enterRule(_localctx, 26, statementParser::RuleTriple_gate_without_parameter_declaration_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(257);
        triple_gate_without_parameter_type_s();
        setState(258);
        q_KEY_declaration_s();
        setState(259);
        match(statementParser::COMMA);
        setState(260);
        q_KEY_declaration_s();
        setState(261);
        match(statementParser::COMMA);
        setState(262);
        q_KEY_declaration_s();
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Single_gate_without_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Single_gate_without_parameter_type_sContext::Single_gate_without_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::H_GATE() {
      return getToken(statementParser::H_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::T_GATE() {
      return getToken(statementParser::T_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::S_GATE() {
      return getToken(statementParser::S_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::X_GATE() {
      return getToken(statementParser::X_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::Y_GATE() {
      return getToken(statementParser::Y_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::Z_GATE() {
      return getToken(statementParser::Z_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::X1_GATE() {
      return getToken(statementParser::X1_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::Y1_GATE() {
      return getToken(statementParser::Y1_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::Z1_GATE() {
      return getToken(statementParser::Z1_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::I_GATE() {
      return getToken(statementParser::I_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_without_parameter_type_sContext::ECHO_GATE() {
      return getToken(statementParser::ECHO_GATE, 0);
    }


    size_t statementParser::Single_gate_without_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_without_parameter_type_s;
    }

    void statementParser::Single_gate_without_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_without_parameter_type_s(this);
    }

    void statementParser::Single_gate_without_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_without_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_without_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_without_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_without_parameter_type_sContext* statementParser::single_gate_without_parameter_type_s() {
      Single_gate_without_parameter_type_sContext *_localctx = _tracker.createInstance<Single_gate_without_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 28, statementParser::RuleSingle_gate_without_parameter_type_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(264);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::ECHO_GATE)
          | (1ULL << statementParser::H_GATE)
          | (1ULL << statementParser::X_GATE)
          | (1ULL << statementParser::T_GATE)
          | (1ULL << statementParser::S_GATE)
          | (1ULL << statementParser::Y_GATE)
          | (1ULL << statementParser::Z_GATE)
          | (1ULL << statementParser::X1_GATE)
          | (1ULL << statementParser::Y1_GATE)
          | (1ULL << statementParser::Z1_GATE)
          | (1ULL << statementParser::I_GATE))) != 0))) {
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

    //----------------- Single_gate_with_one_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_one_parameter_type_sContext::Single_gate_with_one_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_type_sContext::RX_GATE() {
      return getToken(statementParser::RX_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_type_sContext::RY_GATE() {
      return getToken(statementParser::RY_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_type_sContext::RZ_GATE() {
      return getToken(statementParser::RZ_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_one_parameter_type_sContext::U1_GATE() {
      return getToken(statementParser::U1_GATE, 0);
    }


    size_t statementParser::Single_gate_with_one_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_one_parameter_type_s;
    }

    void statementParser::Single_gate_with_one_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_one_parameter_type_s(this);
    }

    void statementParser::Single_gate_with_one_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_one_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_one_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_one_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_one_parameter_type_sContext* statementParser::single_gate_with_one_parameter_type_s() {
      Single_gate_with_one_parameter_type_sContext *_localctx = _tracker.createInstance<Single_gate_with_one_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 30, statementParser::RuleSingle_gate_with_one_parameter_type_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(266);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::RX_GATE)
          | (1ULL << statementParser::RY_GATE)
          | (1ULL << statementParser::RZ_GATE)
          | (1ULL << statementParser::U1_GATE))) != 0))) {
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

    //----------------- Single_gate_with_two_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_two_parameter_type_sContext::Single_gate_with_two_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Single_gate_with_two_parameter_type_sContext::U2_GATE() {
      return getToken(statementParser::U2_GATE, 0);
    }

    tree::TerminalNode* statementParser::Single_gate_with_two_parameter_type_sContext::RPHI_GATE() {
      return getToken(statementParser::RPHI_GATE, 0);
    }


    size_t statementParser::Single_gate_with_two_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_two_parameter_type_s;
    }

    void statementParser::Single_gate_with_two_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_two_parameter_type_s(this);
    }

    void statementParser::Single_gate_with_two_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_two_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_two_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_two_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_two_parameter_type_sContext* statementParser::single_gate_with_two_parameter_type_s() {
      Single_gate_with_two_parameter_type_sContext *_localctx = _tracker.createInstance<Single_gate_with_two_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 32, statementParser::RuleSingle_gate_with_two_parameter_type_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(268);
        _la = _input->LA(1);
        if (!(_la == statementParser::U2_GATE

        || _la == statementParser::RPHI_GATE)) {
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

    //----------------- Single_gate_with_three_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_three_parameter_type_sContext::Single_gate_with_three_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Single_gate_with_three_parameter_type_sContext::U3_GATE() {
      return getToken(statementParser::U3_GATE, 0);
    }


    size_t statementParser::Single_gate_with_three_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_three_parameter_type_s;
    }

    void statementParser::Single_gate_with_three_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_three_parameter_type_s(this);
    }

    void statementParser::Single_gate_with_three_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_three_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_three_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_three_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_three_parameter_type_sContext* statementParser::single_gate_with_three_parameter_type_s() {
      Single_gate_with_three_parameter_type_sContext *_localctx = _tracker.createInstance<Single_gate_with_three_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 34, statementParser::RuleSingle_gate_with_three_parameter_type_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(270);
        match(statementParser::U3_GATE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Single_gate_with_four_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Single_gate_with_four_parameter_type_sContext::Single_gate_with_four_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Single_gate_with_four_parameter_type_sContext::U4_GATE() {
      return getToken(statementParser::U4_GATE, 0);
    }


    size_t statementParser::Single_gate_with_four_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleSingle_gate_with_four_parameter_type_s;
    }

    void statementParser::Single_gate_with_four_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterSingle_gate_with_four_parameter_type_s(this);
    }

    void statementParser::Single_gate_with_four_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitSingle_gate_with_four_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Single_gate_with_four_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitSingle_gate_with_four_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Single_gate_with_four_parameter_type_sContext* statementParser::single_gate_with_four_parameter_type_s() {
      Single_gate_with_four_parameter_type_sContext *_localctx = _tracker.createInstance<Single_gate_with_four_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 36, statementParser::RuleSingle_gate_with_four_parameter_type_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(272);
        match(statementParser::U4_GATE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Double_gate_without_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Double_gate_without_parameter_type_sContext::Double_gate_without_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Double_gate_without_parameter_type_sContext::CNOT_GATE() {
      return getToken(statementParser::CNOT_GATE, 0);
    }

    tree::TerminalNode* statementParser::Double_gate_without_parameter_type_sContext::CZ_GATE() {
      return getToken(statementParser::CZ_GATE, 0);
    }

    tree::TerminalNode* statementParser::Double_gate_without_parameter_type_sContext::ISWAP_GATE() {
      return getToken(statementParser::ISWAP_GATE, 0);
    }

    tree::TerminalNode* statementParser::Double_gate_without_parameter_type_sContext::SQISWAP_GATE() {
      return getToken(statementParser::SQISWAP_GATE, 0);
    }

    tree::TerminalNode* statementParser::Double_gate_without_parameter_type_sContext::SWAPZ1_GATE() {
      return getToken(statementParser::SWAPZ1_GATE, 0);
    }


    size_t statementParser::Double_gate_without_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleDouble_gate_without_parameter_type_s;
    }

    void statementParser::Double_gate_without_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterDouble_gate_without_parameter_type_s(this);
    }

    void statementParser::Double_gate_without_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitDouble_gate_without_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Double_gate_without_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitDouble_gate_without_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Double_gate_without_parameter_type_sContext* statementParser::double_gate_without_parameter_type_s() {
      Double_gate_without_parameter_type_sContext *_localctx = _tracker.createInstance<Double_gate_without_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 38, statementParser::RuleDouble_gate_without_parameter_type_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(274);
        _la = _input->LA(1);
        if (!((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::CNOT_GATE)
          | (1ULL << statementParser::CZ_GATE)
          | (1ULL << statementParser::ISWAP_GATE)
          | (1ULL << statementParser::SQISWAP_GATE)
          | (1ULL << statementParser::SWAPZ1_GATE))) != 0))) {
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

    //----------------- Double_gate_with_one_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Double_gate_with_one_parameter_type_sContext::Double_gate_with_one_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Double_gate_with_one_parameter_type_sContext::ISWAPTHETA_GATE() {
      return getToken(statementParser::ISWAPTHETA_GATE, 0);
    }

    tree::TerminalNode* statementParser::Double_gate_with_one_parameter_type_sContext::CR_GATE() {
      return getToken(statementParser::CR_GATE, 0);
    }


    size_t statementParser::Double_gate_with_one_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleDouble_gate_with_one_parameter_type_s;
    }

    void statementParser::Double_gate_with_one_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterDouble_gate_with_one_parameter_type_s(this);
    }

    void statementParser::Double_gate_with_one_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitDouble_gate_with_one_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Double_gate_with_one_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitDouble_gate_with_one_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Double_gate_with_one_parameter_type_sContext* statementParser::double_gate_with_one_parameter_type_s() {
      Double_gate_with_one_parameter_type_sContext *_localctx = _tracker.createInstance<Double_gate_with_one_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 40, statementParser::RuleDouble_gate_with_one_parameter_type_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(276);
        _la = _input->LA(1);
        if (!(_la == statementParser::ISWAPTHETA_GATE

        || _la == statementParser::CR_GATE)) {
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

    //----------------- Double_gate_with_four_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Double_gate_with_four_parameter_type_sContext::Double_gate_with_four_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Double_gate_with_four_parameter_type_sContext::CU_GATE() {
      return getToken(statementParser::CU_GATE, 0);
    }


    size_t statementParser::Double_gate_with_four_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleDouble_gate_with_four_parameter_type_s;
    }

    void statementParser::Double_gate_with_four_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterDouble_gate_with_four_parameter_type_s(this);
    }

    void statementParser::Double_gate_with_four_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitDouble_gate_with_four_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Double_gate_with_four_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitDouble_gate_with_four_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Double_gate_with_four_parameter_type_sContext* statementParser::double_gate_with_four_parameter_type_s() {
      Double_gate_with_four_parameter_type_sContext *_localctx = _tracker.createInstance<Double_gate_with_four_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 42, statementParser::RuleDouble_gate_with_four_parameter_type_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(278);
        match(statementParser::CU_GATE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Triple_gate_without_parameter_type_sContext ------------------------------------------------------------------

    statementParser::Triple_gate_without_parameter_type_sContext::Triple_gate_without_parameter_type_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Triple_gate_without_parameter_type_sContext::TOFFOLI_GATE() {
      return getToken(statementParser::TOFFOLI_GATE, 0);
    }


    size_t statementParser::Triple_gate_without_parameter_type_sContext::getRuleIndex() const {
      return statementParser::RuleTriple_gate_without_parameter_type_s;
    }

    void statementParser::Triple_gate_without_parameter_type_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterTriple_gate_without_parameter_type_s(this);
    }

    void statementParser::Triple_gate_without_parameter_type_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitTriple_gate_without_parameter_type_s(this);
    }


    antlrcpp::Any statementParser::Triple_gate_without_parameter_type_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitTriple_gate_without_parameter_type_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Triple_gate_without_parameter_type_sContext* statementParser::triple_gate_without_parameter_type_s() {
      Triple_gate_without_parameter_type_sContext *_localctx = _tracker.createInstance<Triple_gate_without_parameter_type_sContext>(_ctx, getState());
      enterRule(_localctx, 44, statementParser::RuleTriple_gate_without_parameter_type_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(280);
        match(statementParser::TOFFOLI_GATE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Primary_expression_sContext ------------------------------------------------------------------

    statementParser::Primary_expression_sContext::Primary_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }


    size_t statementParser::Primary_expression_sContext::getRuleIndex() const {
      return statementParser::RulePrimary_expression_s;
    }

    void statementParser::Primary_expression_sContext::copyFrom(Primary_expression_sContext *ctx) {
      ParserRuleContext::copyFrom(ctx);
    }

    //----------------- Pri_ckeyContext ------------------------------------------------------------------

    statementParser::C_KEY_declaration_sContext* statementParser::Pri_ckeyContext::c_KEY_declaration_s() {
      return getRuleContext<statementParser::C_KEY_declaration_sContext>(0);
    }

    statementParser::Pri_ckeyContext::Pri_ckeyContext(Primary_expression_sContext *ctx) { copyFrom(ctx); }

    void statementParser::Pri_ckeyContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterPri_ckey(this);
    }
    void statementParser::Pri_ckeyContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitPri_ckey(this);
    }

    antlrcpp::Any statementParser::Pri_ckeyContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitPri_ckey(this);
      else
        return visitor->visitChildren(this);
    }
    //----------------- Pri_cstContext ------------------------------------------------------------------

    statementParser::Constant_sContext* statementParser::Pri_cstContext::constant_s() {
      return getRuleContext<statementParser::Constant_sContext>(0);
    }

    statementParser::Pri_cstContext::Pri_cstContext(Primary_expression_sContext *ctx) { copyFrom(ctx); }

    void statementParser::Pri_cstContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterPri_cst(this);
    }
    void statementParser::Pri_cstContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitPri_cst(this);
    }

    antlrcpp::Any statementParser::Pri_cstContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitPri_cst(this);
      else
        return visitor->visitChildren(this);
    }
    //----------------- Pri_exprContext ------------------------------------------------------------------

    std::vector<tree::TerminalNode *> statementParser::Pri_exprContext::LPAREN() {
      return getTokens(statementParser::LPAREN);
    }

    tree::TerminalNode* statementParser::Pri_exprContext::LPAREN(size_t i) {
      return getToken(statementParser::LPAREN, i);
    }

    statementParser::Expression_sContext* statementParser::Pri_exprContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    statementParser::Pri_exprContext::Pri_exprContext(Primary_expression_sContext *ctx) { copyFrom(ctx); }

    void statementParser::Pri_exprContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterPri_expr(this);
    }
    void statementParser::Pri_exprContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitPri_expr(this);
    }

    antlrcpp::Any statementParser::Pri_exprContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitPri_expr(this);
      else
        return visitor->visitChildren(this);
    }
    statementParser::Primary_expression_sContext* statementParser::primary_expression_s() {
      Primary_expression_sContext *_localctx = _tracker.createInstance<Primary_expression_sContext>(_ctx, getState());
      enterRule(_localctx, 46, statementParser::RulePrimary_expression_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(288);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case statementParser::C_KEY: {
            _localctx = dynamic_cast<Primary_expression_sContext *>(_tracker.createInstance<statementParser::Pri_ckeyContext>(_localctx));
            enterOuterAlt(_localctx, 1);
            setState(282);
            c_KEY_declaration_s();
            break;
          }

          case statementParser::PI:
          case statementParser::Integer_Literal_s:
          case statementParser::Double_Literal_s: {
            _localctx = dynamic_cast<Primary_expression_sContext *>(_tracker.createInstance<statementParser::Pri_cstContext>(_localctx));
            enterOuterAlt(_localctx, 2);
            setState(283);
            constant_s();
            break;
          }

          case statementParser::LPAREN: {
            _localctx = dynamic_cast<Primary_expression_sContext *>(_tracker.createInstance<statementParser::Pri_exprContext>(_localctx));
            enterOuterAlt(_localctx, 3);
            setState(284);
            match(statementParser::LPAREN);
            setState(285);
            expression_s();
            setState(286);
            match(statementParser::LPAREN);
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

    //----------------- Unary_expression_sContext ------------------------------------------------------------------

    statementParser::Unary_expression_sContext::Unary_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Primary_expression_sContext* statementParser::Unary_expression_sContext::primary_expression_s() {
      return getRuleContext<statementParser::Primary_expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Unary_expression_sContext::PLUS() {
      return getToken(statementParser::PLUS, 0);
    }

    tree::TerminalNode* statementParser::Unary_expression_sContext::MINUS() {
      return getToken(statementParser::MINUS, 0);
    }

    tree::TerminalNode* statementParser::Unary_expression_sContext::NOT() {
      return getToken(statementParser::NOT, 0);
    }


    size_t statementParser::Unary_expression_sContext::getRuleIndex() const {
      return statementParser::RuleUnary_expression_s;
    }

    void statementParser::Unary_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterUnary_expression_s(this);
    }

    void statementParser::Unary_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitUnary_expression_s(this);
    }


    antlrcpp::Any statementParser::Unary_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitUnary_expression_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Unary_expression_sContext* statementParser::unary_expression_s() {
      Unary_expression_sContext *_localctx = _tracker.createInstance<Unary_expression_sContext>(_ctx, getState());
      enterRule(_localctx, 48, statementParser::RuleUnary_expression_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(297);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case statementParser::PI:
          case statementParser::C_KEY:
          case statementParser::LPAREN:
          case statementParser::Integer_Literal_s:
          case statementParser::Double_Literal_s: {
            enterOuterAlt(_localctx, 1);
            setState(290);
            primary_expression_s();
            break;
          }

          case statementParser::PLUS: {
            enterOuterAlt(_localctx, 2);
            setState(291);
            match(statementParser::PLUS);
            setState(292);
            primary_expression_s();
            break;
          }

          case statementParser::MINUS: {
            enterOuterAlt(_localctx, 3);
            setState(293);
            match(statementParser::MINUS);
            setState(294);
            primary_expression_s();
            break;
          }

          case statementParser::NOT: {
            enterOuterAlt(_localctx, 4);
            setState(295);
            match(statementParser::NOT);
            setState(296);
            primary_expression_s();
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

    //----------------- Multiplicative_expression_sContext ------------------------------------------------------------------

    statementParser::Multiplicative_expression_sContext::Multiplicative_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Unary_expression_sContext* statementParser::Multiplicative_expression_sContext::unary_expression_s() {
      return getRuleContext<statementParser::Unary_expression_sContext>(0);
    }

    statementParser::Multiplicative_expression_sContext* statementParser::Multiplicative_expression_sContext::multiplicative_expression_s() {
      return getRuleContext<statementParser::Multiplicative_expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Multiplicative_expression_sContext::MUL() {
      return getToken(statementParser::MUL, 0);
    }

    tree::TerminalNode* statementParser::Multiplicative_expression_sContext::DIV() {
      return getToken(statementParser::DIV, 0);
    }


    size_t statementParser::Multiplicative_expression_sContext::getRuleIndex() const {
      return statementParser::RuleMultiplicative_expression_s;
    }

    void statementParser::Multiplicative_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterMultiplicative_expression_s(this);
    }

    void statementParser::Multiplicative_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitMultiplicative_expression_s(this);
    }


    antlrcpp::Any statementParser::Multiplicative_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitMultiplicative_expression_s(this);
      else
        return visitor->visitChildren(this);
    }


    statementParser::Multiplicative_expression_sContext* statementParser::multiplicative_expression_s() {
       return multiplicative_expression_s(0);
    }

    statementParser::Multiplicative_expression_sContext* statementParser::multiplicative_expression_s(int precedence) {
      ParserRuleContext *parentContext = _ctx;
      size_t parentState = getState();
      statementParser::Multiplicative_expression_sContext *_localctx = _tracker.createInstance<Multiplicative_expression_sContext>(_ctx, parentState);
      statementParser::Multiplicative_expression_sContext *previousContext = _localctx;
      (void)previousContext; // Silence compiler, in case the context is not used by generated code.
      size_t startState = 50;
      enterRecursionRule(_localctx, 50, statementParser::RuleMultiplicative_expression_s, precedence);

    

      auto onExit = finally([=] {
        unrollRecursionContexts(parentContext);
      });
      try {
        size_t alt;
        enterOuterAlt(_localctx, 1);
        setState(300);
        unary_expression_s();
        _ctx->stop = _input->LT(-1);
        setState(310);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            if (!_parseListeners.empty())
              triggerExitRuleEvent();
            previousContext = _localctx;
            setState(308);
            _errHandler->sync(this);
            switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 10, _ctx)) {
            case 1: {
              _localctx = _tracker.createInstance<Multiplicative_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression_s);
              setState(302);

              if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
              setState(303);
              match(statementParser::MUL);
              setState(304);
              unary_expression_s();
              break;
            }

            case 2: {
              _localctx = _tracker.createInstance<Multiplicative_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleMultiplicative_expression_s);
              setState(305);

              if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
              setState(306);
              match(statementParser::DIV);
              setState(307);
              unary_expression_s();
              break;
            }

            } 
          }
          setState(312);
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

    //----------------- Addtive_expression_sContext ------------------------------------------------------------------

    statementParser::Addtive_expression_sContext::Addtive_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Multiplicative_expression_sContext* statementParser::Addtive_expression_sContext::multiplicative_expression_s() {
      return getRuleContext<statementParser::Multiplicative_expression_sContext>(0);
    }

    statementParser::Addtive_expression_sContext* statementParser::Addtive_expression_sContext::addtive_expression_s() {
      return getRuleContext<statementParser::Addtive_expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Addtive_expression_sContext::PLUS() {
      return getToken(statementParser::PLUS, 0);
    }

    tree::TerminalNode* statementParser::Addtive_expression_sContext::MINUS() {
      return getToken(statementParser::MINUS, 0);
    }


    size_t statementParser::Addtive_expression_sContext::getRuleIndex() const {
      return statementParser::RuleAddtive_expression_s;
    }

    void statementParser::Addtive_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterAddtive_expression_s(this);
    }

    void statementParser::Addtive_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitAddtive_expression_s(this);
    }


    antlrcpp::Any statementParser::Addtive_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitAddtive_expression_s(this);
      else
        return visitor->visitChildren(this);
    }


    statementParser::Addtive_expression_sContext* statementParser::addtive_expression_s() {
       return addtive_expression_s(0);
    }

    statementParser::Addtive_expression_sContext* statementParser::addtive_expression_s(int precedence) {
      ParserRuleContext *parentContext = _ctx;
      size_t parentState = getState();
      statementParser::Addtive_expression_sContext *_localctx = _tracker.createInstance<Addtive_expression_sContext>(_ctx, parentState);
      statementParser::Addtive_expression_sContext *previousContext = _localctx;
      (void)previousContext; // Silence compiler, in case the context is not used by generated code.
      size_t startState = 52;
      enterRecursionRule(_localctx, 52, statementParser::RuleAddtive_expression_s, precedence);

    

      auto onExit = finally([=] {
        unrollRecursionContexts(parentContext);
      });
      try {
        size_t alt;
        enterOuterAlt(_localctx, 1);
        setState(314);
        multiplicative_expression_s(0);
        _ctx->stop = _input->LT(-1);
        setState(324);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            if (!_parseListeners.empty())
              triggerExitRuleEvent();
            previousContext = _localctx;
            setState(322);
            _errHandler->sync(this);
            switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx)) {
            case 1: {
              _localctx = _tracker.createInstance<Addtive_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleAddtive_expression_s);
              setState(316);

              if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
              setState(317);
              match(statementParser::PLUS);
              setState(318);
              multiplicative_expression_s(0);
              break;
            }

            case 2: {
              _localctx = _tracker.createInstance<Addtive_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleAddtive_expression_s);
              setState(319);

              if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
              setState(320);
              match(statementParser::MINUS);
              setState(321);
              multiplicative_expression_s(0);
              break;
            }

            } 
          }
          setState(326);
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

    //----------------- Relational_expression_sContext ------------------------------------------------------------------

    statementParser::Relational_expression_sContext::Relational_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Addtive_expression_sContext* statementParser::Relational_expression_sContext::addtive_expression_s() {
      return getRuleContext<statementParser::Addtive_expression_sContext>(0);
    }

    statementParser::Relational_expression_sContext* statementParser::Relational_expression_sContext::relational_expression_s() {
      return getRuleContext<statementParser::Relational_expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Relational_expression_sContext::LT() {
      return getToken(statementParser::LT, 0);
    }

    tree::TerminalNode* statementParser::Relational_expression_sContext::GT() {
      return getToken(statementParser::GT, 0);
    }

    tree::TerminalNode* statementParser::Relational_expression_sContext::LEQ() {
      return getToken(statementParser::LEQ, 0);
    }

    tree::TerminalNode* statementParser::Relational_expression_sContext::GEQ() {
      return getToken(statementParser::GEQ, 0);
    }


    size_t statementParser::Relational_expression_sContext::getRuleIndex() const {
      return statementParser::RuleRelational_expression_s;
    }

    void statementParser::Relational_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterRelational_expression_s(this);
    }

    void statementParser::Relational_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitRelational_expression_s(this);
    }


    antlrcpp::Any statementParser::Relational_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitRelational_expression_s(this);
      else
        return visitor->visitChildren(this);
    }


    statementParser::Relational_expression_sContext* statementParser::relational_expression_s() {
       return relational_expression_s(0);
    }

    statementParser::Relational_expression_sContext* statementParser::relational_expression_s(int precedence) {
      ParserRuleContext *parentContext = _ctx;
      size_t parentState = getState();
      statementParser::Relational_expression_sContext *_localctx = _tracker.createInstance<Relational_expression_sContext>(_ctx, parentState);
      statementParser::Relational_expression_sContext *previousContext = _localctx;
      (void)previousContext; // Silence compiler, in case the context is not used by generated code.
      size_t startState = 54;
      enterRecursionRule(_localctx, 54, statementParser::RuleRelational_expression_s, precedence);

    

      auto onExit = finally([=] {
        unrollRecursionContexts(parentContext);
      });
      try {
        size_t alt;
        enterOuterAlt(_localctx, 1);
        setState(328);
        addtive_expression_s(0);
        _ctx->stop = _input->LT(-1);
        setState(344);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            if (!_parseListeners.empty())
              triggerExitRuleEvent();
            previousContext = _localctx;
            setState(342);
            _errHandler->sync(this);
            switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx)) {
            case 1: {
              _localctx = _tracker.createInstance<Relational_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleRelational_expression_s);
              setState(330);

              if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
              setState(331);
              match(statementParser::LT);
              setState(332);
              addtive_expression_s(0);
              break;
            }

            case 2: {
              _localctx = _tracker.createInstance<Relational_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleRelational_expression_s);
              setState(333);

              if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
              setState(334);
              match(statementParser::GT);
              setState(335);
              addtive_expression_s(0);
              break;
            }

            case 3: {
              _localctx = _tracker.createInstance<Relational_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleRelational_expression_s);
              setState(336);

              if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
              setState(337);
              match(statementParser::LEQ);
              setState(338);
              addtive_expression_s(0);
              break;
            }

            case 4: {
              _localctx = _tracker.createInstance<Relational_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleRelational_expression_s);
              setState(339);

              if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
              setState(340);
              match(statementParser::GEQ);
              setState(341);
              addtive_expression_s(0);
              break;
            }

            } 
          }
          setState(346);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx);
        }
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }
      return _localctx;
    }

    //----------------- Equality_expression_sContext ------------------------------------------------------------------

    statementParser::Equality_expression_sContext::Equality_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Relational_expression_sContext* statementParser::Equality_expression_sContext::relational_expression_s() {
      return getRuleContext<statementParser::Relational_expression_sContext>(0);
    }

    statementParser::Equality_expression_sContext* statementParser::Equality_expression_sContext::equality_expression_s() {
      return getRuleContext<statementParser::Equality_expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Equality_expression_sContext::EQ() {
      return getToken(statementParser::EQ, 0);
    }

    tree::TerminalNode* statementParser::Equality_expression_sContext::NE() {
      return getToken(statementParser::NE, 0);
    }


    size_t statementParser::Equality_expression_sContext::getRuleIndex() const {
      return statementParser::RuleEquality_expression_s;
    }

    void statementParser::Equality_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterEquality_expression_s(this);
    }

    void statementParser::Equality_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitEquality_expression_s(this);
    }


    antlrcpp::Any statementParser::Equality_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitEquality_expression_s(this);
      else
        return visitor->visitChildren(this);
    }


    statementParser::Equality_expression_sContext* statementParser::equality_expression_s() {
       return equality_expression_s(0);
    }

    statementParser::Equality_expression_sContext* statementParser::equality_expression_s(int precedence) {
      ParserRuleContext *parentContext = _ctx;
      size_t parentState = getState();
      statementParser::Equality_expression_sContext *_localctx = _tracker.createInstance<Equality_expression_sContext>(_ctx, parentState);
      statementParser::Equality_expression_sContext *previousContext = _localctx;
      (void)previousContext; // Silence compiler, in case the context is not used by generated code.
      size_t startState = 56;
      enterRecursionRule(_localctx, 56, statementParser::RuleEquality_expression_s, precedence);

    

      auto onExit = finally([=] {
        unrollRecursionContexts(parentContext);
      });
      try {
        size_t alt;
        enterOuterAlt(_localctx, 1);
        setState(348);
        relational_expression_s(0);
        _ctx->stop = _input->LT(-1);
        setState(358);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 17, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            if (!_parseListeners.empty())
              triggerExitRuleEvent();
            previousContext = _localctx;
            setState(356);
            _errHandler->sync(this);
            switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx)) {
            case 1: {
              _localctx = _tracker.createInstance<Equality_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleEquality_expression_s);
              setState(350);

              if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
              setState(351);
              match(statementParser::EQ);
              setState(352);
              relational_expression_s(0);
              break;
            }

            case 2: {
              _localctx = _tracker.createInstance<Equality_expression_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleEquality_expression_s);
              setState(353);

              if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
              setState(354);
              match(statementParser::NE);
              setState(355);
              relational_expression_s(0);
              break;
            }

            } 
          }
          setState(360);
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

    //----------------- Logical_and_expression_sContext ------------------------------------------------------------------

    statementParser::Logical_and_expression_sContext::Logical_and_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Equality_expression_sContext* statementParser::Logical_and_expression_sContext::equality_expression_s() {
      return getRuleContext<statementParser::Equality_expression_sContext>(0);
    }

    statementParser::Logical_and_expression_sContext* statementParser::Logical_and_expression_sContext::logical_and_expression_s() {
      return getRuleContext<statementParser::Logical_and_expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Logical_and_expression_sContext::AND() {
      return getToken(statementParser::AND, 0);
    }


    size_t statementParser::Logical_and_expression_sContext::getRuleIndex() const {
      return statementParser::RuleLogical_and_expression_s;
    }

    void statementParser::Logical_and_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterLogical_and_expression_s(this);
    }

    void statementParser::Logical_and_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitLogical_and_expression_s(this);
    }


    antlrcpp::Any statementParser::Logical_and_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitLogical_and_expression_s(this);
      else
        return visitor->visitChildren(this);
    }


    statementParser::Logical_and_expression_sContext* statementParser::logical_and_expression_s() {
       return logical_and_expression_s(0);
    }

    statementParser::Logical_and_expression_sContext* statementParser::logical_and_expression_s(int precedence) {
      ParserRuleContext *parentContext = _ctx;
      size_t parentState = getState();
      statementParser::Logical_and_expression_sContext *_localctx = _tracker.createInstance<Logical_and_expression_sContext>(_ctx, parentState);
      statementParser::Logical_and_expression_sContext *previousContext = _localctx;
      (void)previousContext; // Silence compiler, in case the context is not used by generated code.
      size_t startState = 58;
      enterRecursionRule(_localctx, 58, statementParser::RuleLogical_and_expression_s, precedence);

    

      auto onExit = finally([=] {
        unrollRecursionContexts(parentContext);
      });
      try {
        size_t alt;
        enterOuterAlt(_localctx, 1);
        setState(362);
        equality_expression_s(0);
        _ctx->stop = _input->LT(-1);
        setState(369);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            if (!_parseListeners.empty())
              triggerExitRuleEvent();
            previousContext = _localctx;
            _localctx = _tracker.createInstance<Logical_and_expression_sContext>(parentContext, parentState);
            pushNewRecursionContext(_localctx, startState, RuleLogical_and_expression_s);
            setState(364);

            if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
            setState(365);
            match(statementParser::AND);
            setState(366);
            equality_expression_s(0); 
          }
          setState(371);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 18, _ctx);
        }
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }
      return _localctx;
    }

    //----------------- Logical_or_expression_sContext ------------------------------------------------------------------

    statementParser::Logical_or_expression_sContext::Logical_or_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Logical_and_expression_sContext* statementParser::Logical_or_expression_sContext::logical_and_expression_s() {
      return getRuleContext<statementParser::Logical_and_expression_sContext>(0);
    }

    statementParser::Logical_or_expression_sContext* statementParser::Logical_or_expression_sContext::logical_or_expression_s() {
      return getRuleContext<statementParser::Logical_or_expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Logical_or_expression_sContext::OR() {
      return getToken(statementParser::OR, 0);
    }


    size_t statementParser::Logical_or_expression_sContext::getRuleIndex() const {
      return statementParser::RuleLogical_or_expression_s;
    }

    void statementParser::Logical_or_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterLogical_or_expression_s(this);
    }

    void statementParser::Logical_or_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitLogical_or_expression_s(this);
    }


    antlrcpp::Any statementParser::Logical_or_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitLogical_or_expression_s(this);
      else
        return visitor->visitChildren(this);
    }


    statementParser::Logical_or_expression_sContext* statementParser::logical_or_expression_s() {
       return logical_or_expression_s(0);
    }

    statementParser::Logical_or_expression_sContext* statementParser::logical_or_expression_s(int precedence) {
      ParserRuleContext *parentContext = _ctx;
      size_t parentState = getState();
      statementParser::Logical_or_expression_sContext *_localctx = _tracker.createInstance<Logical_or_expression_sContext>(_ctx, parentState);
      statementParser::Logical_or_expression_sContext *previousContext = _localctx;
      (void)previousContext; // Silence compiler, in case the context is not used by generated code.
      size_t startState = 60;
      enterRecursionRule(_localctx, 60, statementParser::RuleLogical_or_expression_s, precedence);

    

      auto onExit = finally([=] {
        unrollRecursionContexts(parentContext);
      });
      try {
        size_t alt;
        enterOuterAlt(_localctx, 1);
        setState(373);
        logical_and_expression_s(0);
        _ctx->stop = _input->LT(-1);
        setState(380);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 19, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            if (!_parseListeners.empty())
              triggerExitRuleEvent();
            previousContext = _localctx;
            _localctx = _tracker.createInstance<Logical_or_expression_sContext>(parentContext, parentState);
            pushNewRecursionContext(_localctx, startState, RuleLogical_or_expression_s);
            setState(375);

            if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
            setState(376);
            match(statementParser::OR);
            setState(377);
            logical_and_expression_s(0); 
          }
          setState(382);
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

    //----------------- Assignment_expression_sContext ------------------------------------------------------------------

    statementParser::Assignment_expression_sContext::Assignment_expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Logical_or_expression_sContext* statementParser::Assignment_expression_sContext::logical_or_expression_s() {
      return getRuleContext<statementParser::Logical_or_expression_sContext>(0);
    }

    statementParser::C_KEY_declaration_sContext* statementParser::Assignment_expression_sContext::c_KEY_declaration_s() {
      return getRuleContext<statementParser::C_KEY_declaration_sContext>(0);
    }

    tree::TerminalNode* statementParser::Assignment_expression_sContext::ASSIGN() {
      return getToken(statementParser::ASSIGN, 0);
    }


    size_t statementParser::Assignment_expression_sContext::getRuleIndex() const {
      return statementParser::RuleAssignment_expression_s;
    }

    void statementParser::Assignment_expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterAssignment_expression_s(this);
    }

    void statementParser::Assignment_expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitAssignment_expression_s(this);
    }


    antlrcpp::Any statementParser::Assignment_expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitAssignment_expression_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Assignment_expression_sContext* statementParser::assignment_expression_s() {
      Assignment_expression_sContext *_localctx = _tracker.createInstance<Assignment_expression_sContext>(_ctx, getState());
      enterRule(_localctx, 62, statementParser::RuleAssignment_expression_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(388);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 20, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(383);
          logical_or_expression_s(0);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(384);
          c_KEY_declaration_s();
          setState(385);
          match(statementParser::ASSIGN);
          setState(386);
          logical_or_expression_s(0);
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

    //----------------- Expression_sContext ------------------------------------------------------------------

    statementParser::Expression_sContext::Expression_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Assignment_expression_sContext* statementParser::Expression_sContext::assignment_expression_s() {
      return getRuleContext<statementParser::Assignment_expression_sContext>(0);
    }


    size_t statementParser::Expression_sContext::getRuleIndex() const {
      return statementParser::RuleExpression_s;
    }

    void statementParser::Expression_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterExpression_s(this);
    }

    void statementParser::Expression_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitExpression_s(this);
    }


    antlrcpp::Any statementParser::Expression_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitExpression_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Expression_sContext* statementParser::expression_s() {
      Expression_sContext *_localctx = _tracker.createInstance<Expression_sContext>(_ctx, getState());
      enterRule(_localctx, 64, statementParser::RuleExpression_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(390);
        assignment_expression_s();
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Controlbit_list_sContext ------------------------------------------------------------------

    statementParser::Controlbit_list_sContext::Controlbit_list_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    std::vector<statementParser::Q_KEY_declaration_sContext *> statementParser::Controlbit_list_sContext::q_KEY_declaration_s() {
      return getRuleContexts<statementParser::Q_KEY_declaration_sContext>();
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Controlbit_list_sContext::q_KEY_declaration_s(size_t i) {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(i);
    }

    std::vector<tree::TerminalNode *> statementParser::Controlbit_list_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Controlbit_list_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }

    std::vector<tree::TerminalNode *> statementParser::Controlbit_list_sContext::Identifier_s() {
      return getTokens(statementParser::Identifier_s);
    }

    tree::TerminalNode* statementParser::Controlbit_list_sContext::Identifier_s(size_t i) {
      return getToken(statementParser::Identifier_s, i);
    }


    size_t statementParser::Controlbit_list_sContext::getRuleIndex() const {
      return statementParser::RuleControlbit_list_s;
    }

    void statementParser::Controlbit_list_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterControlbit_list_s(this);
    }

    void statementParser::Controlbit_list_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitControlbit_list_s(this);
    }


    antlrcpp::Any statementParser::Controlbit_list_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitControlbit_list_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Controlbit_list_sContext* statementParser::controlbit_list_s() {
      Controlbit_list_sContext *_localctx = _tracker.createInstance<Controlbit_list_sContext>(_ctx, getState());
      enterRule(_localctx, 66, statementParser::RuleControlbit_list_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(408);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case statementParser::Q_KEY: {
            enterOuterAlt(_localctx, 1);
            setState(392);
            q_KEY_declaration_s();
            setState(397);
            _errHandler->sync(this);
            _la = _input->LA(1);
            while (_la == statementParser::COMMA) {
              setState(393);
              match(statementParser::COMMA);
              setState(394);
              q_KEY_declaration_s();
              setState(399);
              _errHandler->sync(this);
              _la = _input->LA(1);
            }
            break;
          }

          case statementParser::Identifier_s: {
            enterOuterAlt(_localctx, 2);
            setState(400);
            match(statementParser::Identifier_s);
            setState(405);
            _errHandler->sync(this);
            _la = _input->LA(1);
            while (_la == statementParser::COMMA) {
              setState(401);
              match(statementParser::COMMA);
              setState(402);
              match(statementParser::Identifier_s);
              setState(407);
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

    //----------------- Statement_sContext ------------------------------------------------------------------

    statementParser::Statement_sContext::Statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Quantum_gate_declaration_sContext* statementParser::Statement_sContext::quantum_gate_declaration_s() {
      return getRuleContext<statementParser::Quantum_gate_declaration_sContext>(0);
    }

    tree::TerminalNode* statementParser::Statement_sContext::NEWLINE() {
      return getToken(statementParser::NEWLINE, 0);
    }

    statementParser::Control_statement_sContext* statementParser::Statement_sContext::control_statement_s() {
      return getRuleContext<statementParser::Control_statement_sContext>(0);
    }

    statementParser::Qif_statement_sContext* statementParser::Statement_sContext::qif_statement_s() {
      return getRuleContext<statementParser::Qif_statement_sContext>(0);
    }

    statementParser::Qwhile_statement_sContext* statementParser::Statement_sContext::qwhile_statement_s() {
      return getRuleContext<statementParser::Qwhile_statement_sContext>(0);
    }

    statementParser::Dagger_statement_sContext* statementParser::Statement_sContext::dagger_statement_s() {
      return getRuleContext<statementParser::Dagger_statement_sContext>(0);
    }

    statementParser::Measure_statement_sContext* statementParser::Statement_sContext::measure_statement_s() {
      return getRuleContext<statementParser::Measure_statement_sContext>(0);
    }

    statementParser::Reset_statement_sContext* statementParser::Statement_sContext::reset_statement_s() {
      return getRuleContext<statementParser::Reset_statement_sContext>(0);
    }

    statementParser::Expression_statement_sContext* statementParser::Statement_sContext::expression_statement_s() {
      return getRuleContext<statementParser::Expression_statement_sContext>(0);
    }

    statementParser::Barrier_statement_sContext* statementParser::Statement_sContext::barrier_statement_s() {
      return getRuleContext<statementParser::Barrier_statement_sContext>(0);
    }


    size_t statementParser::Statement_sContext::getRuleIndex() const {
      return statementParser::RuleStatement_s;
    }

    void statementParser::Statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterStatement_s(this);
    }

    void statementParser::Statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitStatement_s(this);
    }


    antlrcpp::Any statementParser::Statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitStatement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Statement_sContext* statementParser::statement_s() {
      Statement_sContext *_localctx = _tracker.createInstance<Statement_sContext>(_ctx, getState());
      enterRule(_localctx, 68, statementParser::RuleStatement_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(421);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case statementParser::ECHO_GATE:
          case statementParser::H_GATE:
          case statementParser::X_GATE:
          case statementParser::T_GATE:
          case statementParser::S_GATE:
          case statementParser::Y_GATE:
          case statementParser::Z_GATE:
          case statementParser::X1_GATE:
          case statementParser::Y1_GATE:
          case statementParser::Z1_GATE:
          case statementParser::I_GATE:
          case statementParser::U2_GATE:
          case statementParser::RPHI_GATE:
          case statementParser::U3_GATE:
          case statementParser::U4_GATE:
          case statementParser::RX_GATE:
          case statementParser::RY_GATE:
          case statementParser::RZ_GATE:
          case statementParser::U1_GATE:
          case statementParser::CNOT_GATE:
          case statementParser::CZ_GATE:
          case statementParser::CU_GATE:
          case statementParser::ISWAP_GATE:
          case statementParser::SQISWAP_GATE:
          case statementParser::SWAPZ1_GATE:
          case statementParser::ISWAPTHETA_GATE:
          case statementParser::CR_GATE:
          case statementParser::TOFFOLI_GATE: {
            enterOuterAlt(_localctx, 1);
            setState(410);
            quantum_gate_declaration_s();
            setState(411);
            match(statementParser::NEWLINE);
            break;
          }

          case statementParser::CONTROL_KEY: {
            enterOuterAlt(_localctx, 2);
            setState(413);
            control_statement_s();
            break;
          }

          case statementParser::QIF_KEY: {
            enterOuterAlt(_localctx, 3);
            setState(414);
            qif_statement_s();
            break;
          }

          case statementParser::QWHILE_KEY: {
            enterOuterAlt(_localctx, 4);
            setState(415);
            qwhile_statement_s();
            break;
          }

          case statementParser::DAGGER_KEY: {
            enterOuterAlt(_localctx, 5);
            setState(416);
            dagger_statement_s();
            break;
          }

          case statementParser::MEASURE_KEY: {
            enterOuterAlt(_localctx, 6);
            setState(417);
            measure_statement_s();
            break;
          }

          case statementParser::RESET_KEY: {
            enterOuterAlt(_localctx, 7);
            setState(418);
            reset_statement_s();
            break;
          }

          case statementParser::PI:
          case statementParser::C_KEY:
          case statementParser::NOT:
          case statementParser::PLUS:
          case statementParser::MINUS:
          case statementParser::LPAREN:
          case statementParser::Integer_Literal_s:
          case statementParser::Double_Literal_s: {
            enterOuterAlt(_localctx, 8);
            setState(419);
            expression_statement_s();
            break;
          }

          case statementParser::BARRIER_KEY: {
            enterOuterAlt(_localctx, 9);
            setState(420);
            barrier_statement_s();
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

    //----------------- Dagger_statement_sContext ------------------------------------------------------------------

    statementParser::Dagger_statement_sContext::Dagger_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Dagger_statement_sContext::DAGGER_KEY() {
      return getToken(statementParser::DAGGER_KEY, 0);
    }

    std::vector<tree::TerminalNode *> statementParser::Dagger_statement_sContext::NEWLINE() {
      return getTokens(statementParser::NEWLINE);
    }

    tree::TerminalNode* statementParser::Dagger_statement_sContext::NEWLINE(size_t i) {
      return getToken(statementParser::NEWLINE, i);
    }

    tree::TerminalNode* statementParser::Dagger_statement_sContext::ENDDAGGER_KEY() {
      return getToken(statementParser::ENDDAGGER_KEY, 0);
    }

    std::vector<statementParser::Statement_sContext *> statementParser::Dagger_statement_sContext::statement_s() {
      return getRuleContexts<statementParser::Statement_sContext>();
    }

    statementParser::Statement_sContext* statementParser::Dagger_statement_sContext::statement_s(size_t i) {
      return getRuleContext<statementParser::Statement_sContext>(i);
    }


    size_t statementParser::Dagger_statement_sContext::getRuleIndex() const {
      return statementParser::RuleDagger_statement_s;
    }

    void statementParser::Dagger_statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterDagger_statement_s(this);
    }

    void statementParser::Dagger_statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitDagger_statement_s(this);
    }


    antlrcpp::Any statementParser::Dagger_statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitDagger_statement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Dagger_statement_sContext* statementParser::dagger_statement_s() {
      Dagger_statement_sContext *_localctx = _tracker.createInstance<Dagger_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 70, statementParser::RuleDagger_statement_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(423);
        match(statementParser::DAGGER_KEY);
        setState(424);
        match(statementParser::NEWLINE);
        setState(428);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::PI)
          | (1ULL << statementParser::C_KEY)
          | (1ULL << statementParser::BARRIER_KEY)
          | (1ULL << statementParser::ECHO_GATE)
          | (1ULL << statementParser::H_GATE)
          | (1ULL << statementParser::X_GATE)
          | (1ULL << statementParser::T_GATE)
          | (1ULL << statementParser::S_GATE)
          | (1ULL << statementParser::Y_GATE)
          | (1ULL << statementParser::Z_GATE)
          | (1ULL << statementParser::X1_GATE)
          | (1ULL << statementParser::Y1_GATE)
          | (1ULL << statementParser::Z1_GATE)
          | (1ULL << statementParser::I_GATE)
          | (1ULL << statementParser::U2_GATE)
          | (1ULL << statementParser::RPHI_GATE)
          | (1ULL << statementParser::U3_GATE)
          | (1ULL << statementParser::U4_GATE)
          | (1ULL << statementParser::RX_GATE)
          | (1ULL << statementParser::RY_GATE)
          | (1ULL << statementParser::RZ_GATE)
          | (1ULL << statementParser::U1_GATE)
          | (1ULL << statementParser::CNOT_GATE)
          | (1ULL << statementParser::CZ_GATE)
          | (1ULL << statementParser::CU_GATE)
          | (1ULL << statementParser::ISWAP_GATE)
          | (1ULL << statementParser::SQISWAP_GATE)
          | (1ULL << statementParser::SWAPZ1_GATE)
          | (1ULL << statementParser::ISWAPTHETA_GATE)
          | (1ULL << statementParser::CR_GATE)
          | (1ULL << statementParser::TOFFOLI_GATE)
          | (1ULL << statementParser::DAGGER_KEY)
          | (1ULL << statementParser::CONTROL_KEY)
          | (1ULL << statementParser::QIF_KEY)
          | (1ULL << statementParser::QWHILE_KEY)
          | (1ULL << statementParser::MEASURE_KEY)
          | (1ULL << statementParser::RESET_KEY)
          | (1ULL << statementParser::NOT)
          | (1ULL << statementParser::PLUS)
          | (1ULL << statementParser::MINUS))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 64)) & ((1ULL << (statementParser::LPAREN - 64))
          | (1ULL << (statementParser::Integer_Literal_s - 64))
          | (1ULL << (statementParser::Double_Literal_s - 64)))) != 0)) {
          setState(425);
          statement_s();
          setState(430);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(431);
        match(statementParser::ENDDAGGER_KEY);
        setState(432);
        match(statementParser::NEWLINE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Control_statement_sContext ------------------------------------------------------------------

    statementParser::Control_statement_sContext::Control_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Control_statement_sContext::CONTROL_KEY() {
      return getToken(statementParser::CONTROL_KEY, 0);
    }

    statementParser::Controlbit_list_sContext* statementParser::Control_statement_sContext::controlbit_list_s() {
      return getRuleContext<statementParser::Controlbit_list_sContext>(0);
    }

    std::vector<tree::TerminalNode *> statementParser::Control_statement_sContext::NEWLINE() {
      return getTokens(statementParser::NEWLINE);
    }

    tree::TerminalNode* statementParser::Control_statement_sContext::NEWLINE(size_t i) {
      return getToken(statementParser::NEWLINE, i);
    }

    tree::TerminalNode* statementParser::Control_statement_sContext::ENDCONTROL_KEY() {
      return getToken(statementParser::ENDCONTROL_KEY, 0);
    }

    std::vector<statementParser::Statement_sContext *> statementParser::Control_statement_sContext::statement_s() {
      return getRuleContexts<statementParser::Statement_sContext>();
    }

    statementParser::Statement_sContext* statementParser::Control_statement_sContext::statement_s(size_t i) {
      return getRuleContext<statementParser::Statement_sContext>(i);
    }


    size_t statementParser::Control_statement_sContext::getRuleIndex() const {
      return statementParser::RuleControl_statement_s;
    }

    void statementParser::Control_statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterControl_statement_s(this);
    }

    void statementParser::Control_statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitControl_statement_s(this);
    }


    antlrcpp::Any statementParser::Control_statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitControl_statement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Control_statement_sContext* statementParser::control_statement_s() {
      Control_statement_sContext *_localctx = _tracker.createInstance<Control_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 72, statementParser::RuleControl_statement_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(434);
        match(statementParser::CONTROL_KEY);
        setState(435);
        controlbit_list_s();
        setState(436);
        match(statementParser::NEWLINE);
        setState(440);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::PI)
          | (1ULL << statementParser::C_KEY)
          | (1ULL << statementParser::BARRIER_KEY)
          | (1ULL << statementParser::ECHO_GATE)
          | (1ULL << statementParser::H_GATE)
          | (1ULL << statementParser::X_GATE)
          | (1ULL << statementParser::T_GATE)
          | (1ULL << statementParser::S_GATE)
          | (1ULL << statementParser::Y_GATE)
          | (1ULL << statementParser::Z_GATE)
          | (1ULL << statementParser::X1_GATE)
          | (1ULL << statementParser::Y1_GATE)
          | (1ULL << statementParser::Z1_GATE)
          | (1ULL << statementParser::I_GATE)
          | (1ULL << statementParser::U2_GATE)
          | (1ULL << statementParser::RPHI_GATE)
          | (1ULL << statementParser::U3_GATE)
          | (1ULL << statementParser::U4_GATE)
          | (1ULL << statementParser::RX_GATE)
          | (1ULL << statementParser::RY_GATE)
          | (1ULL << statementParser::RZ_GATE)
          | (1ULL << statementParser::U1_GATE)
          | (1ULL << statementParser::CNOT_GATE)
          | (1ULL << statementParser::CZ_GATE)
          | (1ULL << statementParser::CU_GATE)
          | (1ULL << statementParser::ISWAP_GATE)
          | (1ULL << statementParser::SQISWAP_GATE)
          | (1ULL << statementParser::SWAPZ1_GATE)
          | (1ULL << statementParser::ISWAPTHETA_GATE)
          | (1ULL << statementParser::CR_GATE)
          | (1ULL << statementParser::TOFFOLI_GATE)
          | (1ULL << statementParser::DAGGER_KEY)
          | (1ULL << statementParser::CONTROL_KEY)
          | (1ULL << statementParser::QIF_KEY)
          | (1ULL << statementParser::QWHILE_KEY)
          | (1ULL << statementParser::MEASURE_KEY)
          | (1ULL << statementParser::RESET_KEY)
          | (1ULL << statementParser::NOT)
          | (1ULL << statementParser::PLUS)
          | (1ULL << statementParser::MINUS))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 64)) & ((1ULL << (statementParser::LPAREN - 64))
          | (1ULL << (statementParser::Integer_Literal_s - 64))
          | (1ULL << (statementParser::Double_Literal_s - 64)))) != 0)) {
          setState(437);
          statement_s();
          setState(442);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(443);
        match(statementParser::ENDCONTROL_KEY);
        setState(444);
        match(statementParser::NEWLINE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Qelse_statement_fragment_sContext ------------------------------------------------------------------

    statementParser::Qelse_statement_fragment_sContext::Qelse_statement_fragment_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Qelse_statement_fragment_sContext::ELSE_KEY() {
      return getToken(statementParser::ELSE_KEY, 0);
    }

    tree::TerminalNode* statementParser::Qelse_statement_fragment_sContext::NEWLINE() {
      return getToken(statementParser::NEWLINE, 0);
    }

    std::vector<statementParser::Statement_sContext *> statementParser::Qelse_statement_fragment_sContext::statement_s() {
      return getRuleContexts<statementParser::Statement_sContext>();
    }

    statementParser::Statement_sContext* statementParser::Qelse_statement_fragment_sContext::statement_s(size_t i) {
      return getRuleContext<statementParser::Statement_sContext>(i);
    }


    size_t statementParser::Qelse_statement_fragment_sContext::getRuleIndex() const {
      return statementParser::RuleQelse_statement_fragment_s;
    }

    void statementParser::Qelse_statement_fragment_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterQelse_statement_fragment_s(this);
    }

    void statementParser::Qelse_statement_fragment_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitQelse_statement_fragment_s(this);
    }


    antlrcpp::Any statementParser::Qelse_statement_fragment_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitQelse_statement_fragment_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Qelse_statement_fragment_sContext* statementParser::qelse_statement_fragment_s() {
      Qelse_statement_fragment_sContext *_localctx = _tracker.createInstance<Qelse_statement_fragment_sContext>(_ctx, getState());
      enterRule(_localctx, 74, statementParser::RuleQelse_statement_fragment_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(446);
        match(statementParser::ELSE_KEY);
        setState(447);
        match(statementParser::NEWLINE);
        setState(451);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::PI)
          | (1ULL << statementParser::C_KEY)
          | (1ULL << statementParser::BARRIER_KEY)
          | (1ULL << statementParser::ECHO_GATE)
          | (1ULL << statementParser::H_GATE)
          | (1ULL << statementParser::X_GATE)
          | (1ULL << statementParser::T_GATE)
          | (1ULL << statementParser::S_GATE)
          | (1ULL << statementParser::Y_GATE)
          | (1ULL << statementParser::Z_GATE)
          | (1ULL << statementParser::X1_GATE)
          | (1ULL << statementParser::Y1_GATE)
          | (1ULL << statementParser::Z1_GATE)
          | (1ULL << statementParser::I_GATE)
          | (1ULL << statementParser::U2_GATE)
          | (1ULL << statementParser::RPHI_GATE)
          | (1ULL << statementParser::U3_GATE)
          | (1ULL << statementParser::U4_GATE)
          | (1ULL << statementParser::RX_GATE)
          | (1ULL << statementParser::RY_GATE)
          | (1ULL << statementParser::RZ_GATE)
          | (1ULL << statementParser::U1_GATE)
          | (1ULL << statementParser::CNOT_GATE)
          | (1ULL << statementParser::CZ_GATE)
          | (1ULL << statementParser::CU_GATE)
          | (1ULL << statementParser::ISWAP_GATE)
          | (1ULL << statementParser::SQISWAP_GATE)
          | (1ULL << statementParser::SWAPZ1_GATE)
          | (1ULL << statementParser::ISWAPTHETA_GATE)
          | (1ULL << statementParser::CR_GATE)
          | (1ULL << statementParser::TOFFOLI_GATE)
          | (1ULL << statementParser::DAGGER_KEY)
          | (1ULL << statementParser::CONTROL_KEY)
          | (1ULL << statementParser::QIF_KEY)
          | (1ULL << statementParser::QWHILE_KEY)
          | (1ULL << statementParser::MEASURE_KEY)
          | (1ULL << statementParser::RESET_KEY)
          | (1ULL << statementParser::NOT)
          | (1ULL << statementParser::PLUS)
          | (1ULL << statementParser::MINUS))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 64)) & ((1ULL << (statementParser::LPAREN - 64))
          | (1ULL << (statementParser::Integer_Literal_s - 64))
          | (1ULL << (statementParser::Double_Literal_s - 64)))) != 0)) {
          setState(448);
          statement_s();
          setState(453);
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

    //----------------- Qif_statement_sContext ------------------------------------------------------------------

    statementParser::Qif_statement_sContext::Qif_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }


    size_t statementParser::Qif_statement_sContext::getRuleIndex() const {
      return statementParser::RuleQif_statement_s;
    }

    void statementParser::Qif_statement_sContext::copyFrom(Qif_statement_sContext *ctx) {
      ParserRuleContext::copyFrom(ctx);
    }

    //----------------- Qif_ifContext ------------------------------------------------------------------

    tree::TerminalNode* statementParser::Qif_ifContext::QIF_KEY() {
      return getToken(statementParser::QIF_KEY, 0);
    }

    statementParser::Expression_sContext* statementParser::Qif_ifContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    std::vector<tree::TerminalNode *> statementParser::Qif_ifContext::NEWLINE() {
      return getTokens(statementParser::NEWLINE);
    }

    tree::TerminalNode* statementParser::Qif_ifContext::NEWLINE(size_t i) {
      return getToken(statementParser::NEWLINE, i);
    }

    statementParser::Qelse_statement_fragment_sContext* statementParser::Qif_ifContext::qelse_statement_fragment_s() {
      return getRuleContext<statementParser::Qelse_statement_fragment_sContext>(0);
    }

    tree::TerminalNode* statementParser::Qif_ifContext::ENDIF_KEY() {
      return getToken(statementParser::ENDIF_KEY, 0);
    }

    std::vector<statementParser::Statement_sContext *> statementParser::Qif_ifContext::statement_s() {
      return getRuleContexts<statementParser::Statement_sContext>();
    }

    statementParser::Statement_sContext* statementParser::Qif_ifContext::statement_s(size_t i) {
      return getRuleContext<statementParser::Statement_sContext>(i);
    }

    statementParser::Qif_ifContext::Qif_ifContext(Qif_statement_sContext *ctx) { copyFrom(ctx); }

    void statementParser::Qif_ifContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterQif_if(this);
    }
    void statementParser::Qif_ifContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitQif_if(this);
    }

    antlrcpp::Any statementParser::Qif_ifContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitQif_if(this);
      else
        return visitor->visitChildren(this);
    }
    //----------------- Qif_ifelseContext ------------------------------------------------------------------

    tree::TerminalNode* statementParser::Qif_ifelseContext::QIF_KEY() {
      return getToken(statementParser::QIF_KEY, 0);
    }

    statementParser::Expression_sContext* statementParser::Qif_ifelseContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    std::vector<tree::TerminalNode *> statementParser::Qif_ifelseContext::NEWLINE() {
      return getTokens(statementParser::NEWLINE);
    }

    tree::TerminalNode* statementParser::Qif_ifelseContext::NEWLINE(size_t i) {
      return getToken(statementParser::NEWLINE, i);
    }

    tree::TerminalNode* statementParser::Qif_ifelseContext::ENDIF_KEY() {
      return getToken(statementParser::ENDIF_KEY, 0);
    }

    std::vector<statementParser::Statement_sContext *> statementParser::Qif_ifelseContext::statement_s() {
      return getRuleContexts<statementParser::Statement_sContext>();
    }

    statementParser::Statement_sContext* statementParser::Qif_ifelseContext::statement_s(size_t i) {
      return getRuleContext<statementParser::Statement_sContext>(i);
    }

    statementParser::Qif_ifelseContext::Qif_ifelseContext(Qif_statement_sContext *ctx) { copyFrom(ctx); }

    void statementParser::Qif_ifelseContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterQif_ifelse(this);
    }
    void statementParser::Qif_ifelseContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitQif_ifelse(this);
    }

    antlrcpp::Any statementParser::Qif_ifelseContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitQif_ifelse(this);
      else
        return visitor->visitChildren(this);
    }
    statementParser::Qif_statement_sContext* statementParser::qif_statement_s() {
      Qif_statement_sContext *_localctx = _tracker.createInstance<Qif_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 76, statementParser::RuleQif_statement_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(479);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 30, _ctx)) {
        case 1: {
          _localctx = dynamic_cast<Qif_statement_sContext *>(_tracker.createInstance<statementParser::Qif_ifContext>(_localctx));
          enterOuterAlt(_localctx, 1);
          setState(454);
          match(statementParser::QIF_KEY);
          setState(455);
          expression_s();
          setState(456);
          match(statementParser::NEWLINE);
          setState(460);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while ((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & ((1ULL << statementParser::PI)
            | (1ULL << statementParser::C_KEY)
            | (1ULL << statementParser::BARRIER_KEY)
            | (1ULL << statementParser::ECHO_GATE)
            | (1ULL << statementParser::H_GATE)
            | (1ULL << statementParser::X_GATE)
            | (1ULL << statementParser::T_GATE)
            | (1ULL << statementParser::S_GATE)
            | (1ULL << statementParser::Y_GATE)
            | (1ULL << statementParser::Z_GATE)
            | (1ULL << statementParser::X1_GATE)
            | (1ULL << statementParser::Y1_GATE)
            | (1ULL << statementParser::Z1_GATE)
            | (1ULL << statementParser::I_GATE)
            | (1ULL << statementParser::U2_GATE)
            | (1ULL << statementParser::RPHI_GATE)
            | (1ULL << statementParser::U3_GATE)
            | (1ULL << statementParser::U4_GATE)
            | (1ULL << statementParser::RX_GATE)
            | (1ULL << statementParser::RY_GATE)
            | (1ULL << statementParser::RZ_GATE)
            | (1ULL << statementParser::U1_GATE)
            | (1ULL << statementParser::CNOT_GATE)
            | (1ULL << statementParser::CZ_GATE)
            | (1ULL << statementParser::CU_GATE)
            | (1ULL << statementParser::ISWAP_GATE)
            | (1ULL << statementParser::SQISWAP_GATE)
            | (1ULL << statementParser::SWAPZ1_GATE)
            | (1ULL << statementParser::ISWAPTHETA_GATE)
            | (1ULL << statementParser::CR_GATE)
            | (1ULL << statementParser::TOFFOLI_GATE)
            | (1ULL << statementParser::DAGGER_KEY)
            | (1ULL << statementParser::CONTROL_KEY)
            | (1ULL << statementParser::QIF_KEY)
            | (1ULL << statementParser::QWHILE_KEY)
            | (1ULL << statementParser::MEASURE_KEY)
            | (1ULL << statementParser::RESET_KEY)
            | (1ULL << statementParser::NOT)
            | (1ULL << statementParser::PLUS)
            | (1ULL << statementParser::MINUS))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
            ((1ULL << (_la - 64)) & ((1ULL << (statementParser::LPAREN - 64))
            | (1ULL << (statementParser::Integer_Literal_s - 64))
            | (1ULL << (statementParser::Double_Literal_s - 64)))) != 0)) {
            setState(457);
            statement_s();
            setState(462);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
          setState(463);
          qelse_statement_fragment_s();
          setState(464);
          match(statementParser::ENDIF_KEY);
          setState(465);
          match(statementParser::NEWLINE);
          break;
        }

        case 2: {
          _localctx = dynamic_cast<Qif_statement_sContext *>(_tracker.createInstance<statementParser::Qif_ifelseContext>(_localctx));
          enterOuterAlt(_localctx, 2);
          setState(467);
          match(statementParser::QIF_KEY);
          setState(468);
          expression_s();
          setState(469);
          match(statementParser::NEWLINE);
          setState(473);
          _errHandler->sync(this);
          _la = _input->LA(1);
          while ((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & ((1ULL << statementParser::PI)
            | (1ULL << statementParser::C_KEY)
            | (1ULL << statementParser::BARRIER_KEY)
            | (1ULL << statementParser::ECHO_GATE)
            | (1ULL << statementParser::H_GATE)
            | (1ULL << statementParser::X_GATE)
            | (1ULL << statementParser::T_GATE)
            | (1ULL << statementParser::S_GATE)
            | (1ULL << statementParser::Y_GATE)
            | (1ULL << statementParser::Z_GATE)
            | (1ULL << statementParser::X1_GATE)
            | (1ULL << statementParser::Y1_GATE)
            | (1ULL << statementParser::Z1_GATE)
            | (1ULL << statementParser::I_GATE)
            | (1ULL << statementParser::U2_GATE)
            | (1ULL << statementParser::RPHI_GATE)
            | (1ULL << statementParser::U3_GATE)
            | (1ULL << statementParser::U4_GATE)
            | (1ULL << statementParser::RX_GATE)
            | (1ULL << statementParser::RY_GATE)
            | (1ULL << statementParser::RZ_GATE)
            | (1ULL << statementParser::U1_GATE)
            | (1ULL << statementParser::CNOT_GATE)
            | (1ULL << statementParser::CZ_GATE)
            | (1ULL << statementParser::CU_GATE)
            | (1ULL << statementParser::ISWAP_GATE)
            | (1ULL << statementParser::SQISWAP_GATE)
            | (1ULL << statementParser::SWAPZ1_GATE)
            | (1ULL << statementParser::ISWAPTHETA_GATE)
            | (1ULL << statementParser::CR_GATE)
            | (1ULL << statementParser::TOFFOLI_GATE)
            | (1ULL << statementParser::DAGGER_KEY)
            | (1ULL << statementParser::CONTROL_KEY)
            | (1ULL << statementParser::QIF_KEY)
            | (1ULL << statementParser::QWHILE_KEY)
            | (1ULL << statementParser::MEASURE_KEY)
            | (1ULL << statementParser::RESET_KEY)
            | (1ULL << statementParser::NOT)
            | (1ULL << statementParser::PLUS)
            | (1ULL << statementParser::MINUS))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
            ((1ULL << (_la - 64)) & ((1ULL << (statementParser::LPAREN - 64))
            | (1ULL << (statementParser::Integer_Literal_s - 64))
            | (1ULL << (statementParser::Double_Literal_s - 64)))) != 0)) {
            setState(470);
            statement_s();
            setState(475);
            _errHandler->sync(this);
            _la = _input->LA(1);
          }
          setState(476);
          match(statementParser::ENDIF_KEY);
          setState(477);
          match(statementParser::NEWLINE);
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

    //----------------- Qwhile_statement_sContext ------------------------------------------------------------------

    statementParser::Qwhile_statement_sContext::Qwhile_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Qwhile_statement_sContext::QWHILE_KEY() {
      return getToken(statementParser::QWHILE_KEY, 0);
    }

    statementParser::Expression_sContext* statementParser::Qwhile_statement_sContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    std::vector<tree::TerminalNode *> statementParser::Qwhile_statement_sContext::NEWLINE() {
      return getTokens(statementParser::NEWLINE);
    }

    tree::TerminalNode* statementParser::Qwhile_statement_sContext::NEWLINE(size_t i) {
      return getToken(statementParser::NEWLINE, i);
    }

    tree::TerminalNode* statementParser::Qwhile_statement_sContext::ENDQWHILE_KEY() {
      return getToken(statementParser::ENDQWHILE_KEY, 0);
    }

    std::vector<statementParser::Statement_sContext *> statementParser::Qwhile_statement_sContext::statement_s() {
      return getRuleContexts<statementParser::Statement_sContext>();
    }

    statementParser::Statement_sContext* statementParser::Qwhile_statement_sContext::statement_s(size_t i) {
      return getRuleContext<statementParser::Statement_sContext>(i);
    }


    size_t statementParser::Qwhile_statement_sContext::getRuleIndex() const {
      return statementParser::RuleQwhile_statement_s;
    }

    void statementParser::Qwhile_statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterQwhile_statement_s(this);
    }

    void statementParser::Qwhile_statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitQwhile_statement_s(this);
    }


    antlrcpp::Any statementParser::Qwhile_statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitQwhile_statement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Qwhile_statement_sContext* statementParser::qwhile_statement_s() {
      Qwhile_statement_sContext *_localctx = _tracker.createInstance<Qwhile_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 78, statementParser::RuleQwhile_statement_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(481);
        match(statementParser::QWHILE_KEY);
        setState(482);
        expression_s();
        setState(483);
        match(statementParser::NEWLINE);
        setState(487);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while ((((_la & ~ 0x3fULL) == 0) &&
          ((1ULL << _la) & ((1ULL << statementParser::PI)
          | (1ULL << statementParser::C_KEY)
          | (1ULL << statementParser::BARRIER_KEY)
          | (1ULL << statementParser::ECHO_GATE)
          | (1ULL << statementParser::H_GATE)
          | (1ULL << statementParser::X_GATE)
          | (1ULL << statementParser::T_GATE)
          | (1ULL << statementParser::S_GATE)
          | (1ULL << statementParser::Y_GATE)
          | (1ULL << statementParser::Z_GATE)
          | (1ULL << statementParser::X1_GATE)
          | (1ULL << statementParser::Y1_GATE)
          | (1ULL << statementParser::Z1_GATE)
          | (1ULL << statementParser::I_GATE)
          | (1ULL << statementParser::U2_GATE)
          | (1ULL << statementParser::RPHI_GATE)
          | (1ULL << statementParser::U3_GATE)
          | (1ULL << statementParser::U4_GATE)
          | (1ULL << statementParser::RX_GATE)
          | (1ULL << statementParser::RY_GATE)
          | (1ULL << statementParser::RZ_GATE)
          | (1ULL << statementParser::U1_GATE)
          | (1ULL << statementParser::CNOT_GATE)
          | (1ULL << statementParser::CZ_GATE)
          | (1ULL << statementParser::CU_GATE)
          | (1ULL << statementParser::ISWAP_GATE)
          | (1ULL << statementParser::SQISWAP_GATE)
          | (1ULL << statementParser::SWAPZ1_GATE)
          | (1ULL << statementParser::ISWAPTHETA_GATE)
          | (1ULL << statementParser::CR_GATE)
          | (1ULL << statementParser::TOFFOLI_GATE)
          | (1ULL << statementParser::DAGGER_KEY)
          | (1ULL << statementParser::CONTROL_KEY)
          | (1ULL << statementParser::QIF_KEY)
          | (1ULL << statementParser::QWHILE_KEY)
          | (1ULL << statementParser::MEASURE_KEY)
          | (1ULL << statementParser::RESET_KEY)
          | (1ULL << statementParser::NOT)
          | (1ULL << statementParser::PLUS)
          | (1ULL << statementParser::MINUS))) != 0) || ((((_la - 64) & ~ 0x3fULL) == 0) &&
          ((1ULL << (_la - 64)) & ((1ULL << (statementParser::LPAREN - 64))
          | (1ULL << (statementParser::Integer_Literal_s - 64))
          | (1ULL << (statementParser::Double_Literal_s - 64)))) != 0)) {
          setState(484);
          statement_s();
          setState(489);
          _errHandler->sync(this);
          _la = _input->LA(1);
        }
        setState(490);
        match(statementParser::ENDQWHILE_KEY);
        setState(491);
        match(statementParser::NEWLINE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Measure_statement_sContext ------------------------------------------------------------------

    statementParser::Measure_statement_sContext::Measure_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Measure_statement_sContext::MEASURE_KEY() {
      return getToken(statementParser::MEASURE_KEY, 0);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Measure_statement_sContext::q_KEY_declaration_s() {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(0);
    }

    tree::TerminalNode* statementParser::Measure_statement_sContext::COMMA() {
      return getToken(statementParser::COMMA, 0);
    }

    statementParser::C_KEY_declaration_sContext* statementParser::Measure_statement_sContext::c_KEY_declaration_s() {
      return getRuleContext<statementParser::C_KEY_declaration_sContext>(0);
    }

    tree::TerminalNode* statementParser::Measure_statement_sContext::NEWLINE() {
      return getToken(statementParser::NEWLINE, 0);
    }

    tree::TerminalNode* statementParser::Measure_statement_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }

    tree::TerminalNode* statementParser::Measure_statement_sContext::C_KEY() {
      return getToken(statementParser::C_KEY, 0);
    }


    size_t statementParser::Measure_statement_sContext::getRuleIndex() const {
      return statementParser::RuleMeasure_statement_s;
    }

    void statementParser::Measure_statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterMeasure_statement_s(this);
    }

    void statementParser::Measure_statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitMeasure_statement_s(this);
    }


    antlrcpp::Any statementParser::Measure_statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitMeasure_statement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Measure_statement_sContext* statementParser::measure_statement_s() {
      Measure_statement_sContext *_localctx = _tracker.createInstance<Measure_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 80, statementParser::RuleMeasure_statement_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(504);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 32, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(493);
          match(statementParser::MEASURE_KEY);
          setState(494);
          q_KEY_declaration_s();
          setState(495);
          match(statementParser::COMMA);
          setState(496);
          c_KEY_declaration_s();
          setState(497);
          match(statementParser::NEWLINE);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(499);
          match(statementParser::MEASURE_KEY);
          setState(500);
          match(statementParser::Q_KEY);
          setState(501);
          match(statementParser::COMMA);
          setState(502);
          match(statementParser::C_KEY);
          setState(503);
          match(statementParser::NEWLINE);
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

    //----------------- Reset_statement_sContext ------------------------------------------------------------------

    statementParser::Reset_statement_sContext::Reset_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Reset_statement_sContext::RESET_KEY() {
      return getToken(statementParser::RESET_KEY, 0);
    }

    statementParser::Q_KEY_declaration_sContext* statementParser::Reset_statement_sContext::q_KEY_declaration_s() {
      return getRuleContext<statementParser::Q_KEY_declaration_sContext>(0);
    }

    tree::TerminalNode* statementParser::Reset_statement_sContext::NEWLINE() {
      return getToken(statementParser::NEWLINE, 0);
    }

    tree::TerminalNode* statementParser::Reset_statement_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }


    size_t statementParser::Reset_statement_sContext::getRuleIndex() const {
      return statementParser::RuleReset_statement_s;
    }

    void statementParser::Reset_statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterReset_statement_s(this);
    }

    void statementParser::Reset_statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitReset_statement_s(this);
    }


    antlrcpp::Any statementParser::Reset_statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitReset_statement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Reset_statement_sContext* statementParser::reset_statement_s() {
      Reset_statement_sContext *_localctx = _tracker.createInstance<Reset_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 82, statementParser::RuleReset_statement_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(513);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 33, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(506);
          match(statementParser::RESET_KEY);
          setState(507);
          q_KEY_declaration_s();
          setState(508);
          match(statementParser::NEWLINE);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(510);
          match(statementParser::RESET_KEY);
          setState(511);
          match(statementParser::Q_KEY);
          setState(512);
          match(statementParser::NEWLINE);
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

    //----------------- Barrier_statement_sContext ------------------------------------------------------------------

    statementParser::Barrier_statement_sContext::Barrier_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Barrier_statement_sContext::BARRIER_KEY() {
      return getToken(statementParser::BARRIER_KEY, 0);
    }

    statementParser::Controlbit_list_sContext* statementParser::Barrier_statement_sContext::controlbit_list_s() {
      return getRuleContext<statementParser::Controlbit_list_sContext>(0);
    }

    tree::TerminalNode* statementParser::Barrier_statement_sContext::NEWLINE() {
      return getToken(statementParser::NEWLINE, 0);
    }

    tree::TerminalNode* statementParser::Barrier_statement_sContext::Q_KEY() {
      return getToken(statementParser::Q_KEY, 0);
    }


    size_t statementParser::Barrier_statement_sContext::getRuleIndex() const {
      return statementParser::RuleBarrier_statement_s;
    }

    void statementParser::Barrier_statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterBarrier_statement_s(this);
    }

    void statementParser::Barrier_statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitBarrier_statement_s(this);
    }


    antlrcpp::Any statementParser::Barrier_statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitBarrier_statement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Barrier_statement_sContext* statementParser::barrier_statement_s() {
      Barrier_statement_sContext *_localctx = _tracker.createInstance<Barrier_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 84, statementParser::RuleBarrier_statement_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(522);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 34, _ctx)) {
        case 1: {
          enterOuterAlt(_localctx, 1);
          setState(515);
          match(statementParser::BARRIER_KEY);
          setState(516);
          controlbit_list_s();
          setState(517);
          match(statementParser::NEWLINE);
          break;
        }

        case 2: {
          enterOuterAlt(_localctx, 2);
          setState(519);
          match(statementParser::BARRIER_KEY);
          setState(520);
          match(statementParser::Q_KEY);
          setState(521);
          match(statementParser::NEWLINE);
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

    //----------------- Expression_statement_sContext ------------------------------------------------------------------

    statementParser::Expression_statement_sContext::Expression_statement_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Expression_sContext* statementParser::Expression_statement_sContext::expression_s() {
      return getRuleContext<statementParser::Expression_sContext>(0);
    }

    tree::TerminalNode* statementParser::Expression_statement_sContext::NEWLINE() {
      return getToken(statementParser::NEWLINE, 0);
    }


    size_t statementParser::Expression_statement_sContext::getRuleIndex() const {
      return statementParser::RuleExpression_statement_s;
    }

    void statementParser::Expression_statement_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterExpression_statement_s(this);
    }

    void statementParser::Expression_statement_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitExpression_statement_s(this);
    }


    antlrcpp::Any statementParser::Expression_statement_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitExpression_statement_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Expression_statement_sContext* statementParser::expression_statement_s() {
      Expression_statement_sContext *_localctx = _tracker.createInstance<Expression_statement_sContext>(_ctx, getState());
      enterRule(_localctx, 86, statementParser::RuleExpression_statement_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(524);
        expression_s();
        setState(525);
        match(statementParser::NEWLINE);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Explist_sContext ------------------------------------------------------------------

    statementParser::Explist_sContext::Explist_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    std::vector<statementParser::Exp_sContext *> statementParser::Explist_sContext::exp_s() {
      return getRuleContexts<statementParser::Exp_sContext>();
    }

    statementParser::Exp_sContext* statementParser::Explist_sContext::exp_s(size_t i) {
      return getRuleContext<statementParser::Exp_sContext>(i);
    }

    std::vector<tree::TerminalNode *> statementParser::Explist_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Explist_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }


    size_t statementParser::Explist_sContext::getRuleIndex() const {
      return statementParser::RuleExplist_s;
    }

    void statementParser::Explist_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterExplist_s(this);
    }

    void statementParser::Explist_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitExplist_s(this);
    }


    antlrcpp::Any statementParser::Explist_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitExplist_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Explist_sContext* statementParser::explist_s() {
      Explist_sContext *_localctx = _tracker.createInstance<Explist_sContext>(_ctx, getState());
      enterRule(_localctx, 88, statementParser::RuleExplist_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(527);
        exp_s(0);
        setState(532);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == statementParser::COMMA) {
          setState(528);
          match(statementParser::COMMA);
          setState(529);
          exp_s(0);
          setState(534);
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

    //----------------- Exp_sContext ------------------------------------------------------------------

    statementParser::Exp_sContext::Exp_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Id_sContext* statementParser::Exp_sContext::id_s() {
      return getRuleContext<statementParser::Id_sContext>(0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::Integer_Literal_s() {
      return getToken(statementParser::Integer_Literal_s, 0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::Double_Literal_s() {
      return getToken(statementParser::Double_Literal_s, 0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::PI() {
      return getToken(statementParser::PI, 0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::LPAREN() {
      return getToken(statementParser::LPAREN, 0);
    }

    std::vector<statementParser::Exp_sContext *> statementParser::Exp_sContext::exp_s() {
      return getRuleContexts<statementParser::Exp_sContext>();
    }

    statementParser::Exp_sContext* statementParser::Exp_sContext::exp_s(size_t i) {
      return getRuleContext<statementParser::Exp_sContext>(i);
    }

    tree::TerminalNode* statementParser::Exp_sContext::RPAREN() {
      return getToken(statementParser::RPAREN, 0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::MINUS() {
      return getToken(statementParser::MINUS, 0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::MUL() {
      return getToken(statementParser::MUL, 0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::DIV() {
      return getToken(statementParser::DIV, 0);
    }

    tree::TerminalNode* statementParser::Exp_sContext::PLUS() {
      return getToken(statementParser::PLUS, 0);
    }


    size_t statementParser::Exp_sContext::getRuleIndex() const {
      return statementParser::RuleExp_s;
    }

    void statementParser::Exp_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterExp_s(this);
    }

    void statementParser::Exp_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitExp_s(this);
    }


    antlrcpp::Any statementParser::Exp_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitExp_s(this);
      else
        return visitor->visitChildren(this);
    }


    statementParser::Exp_sContext* statementParser::exp_s() {
       return exp_s(0);
    }

    statementParser::Exp_sContext* statementParser::exp_s(int precedence) {
      ParserRuleContext *parentContext = _ctx;
      size_t parentState = getState();
      statementParser::Exp_sContext *_localctx = _tracker.createInstance<Exp_sContext>(_ctx, parentState);
      statementParser::Exp_sContext *previousContext = _localctx;
      (void)previousContext; // Silence compiler, in case the context is not used by generated code.
      size_t startState = 90;
      enterRecursionRule(_localctx, 90, statementParser::RuleExp_s, precedence);

    

      auto onExit = finally([=] {
        unrollRecursionContexts(parentContext);
      });
      try {
        size_t alt;
        enterOuterAlt(_localctx, 1);
        setState(546);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case statementParser::Identifier_s: {
            setState(536);
            id_s();
            break;
          }

          case statementParser::Integer_Literal_s: {
            setState(537);
            match(statementParser::Integer_Literal_s);
            break;
          }

          case statementParser::Double_Literal_s: {
            setState(538);
            match(statementParser::Double_Literal_s);
            break;
          }

          case statementParser::PI: {
            setState(539);
            match(statementParser::PI);
            break;
          }

          case statementParser::LPAREN: {
            setState(540);
            match(statementParser::LPAREN);
            setState(541);
            exp_s(0);
            setState(542);
            match(statementParser::RPAREN);
            break;
          }

          case statementParser::MINUS: {
            setState(544);
            match(statementParser::MINUS);
            setState(545);
            exp_s(5);
            break;
          }

        default:
          throw NoViableAltException(this);
        }
        _ctx->stop = _input->LT(-1);
        setState(562);
        _errHandler->sync(this);
        alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 38, _ctx);
        while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
          if (alt == 1) {
            if (!_parseListeners.empty())
              triggerExitRuleEvent();
            previousContext = _localctx;
            setState(560);
            _errHandler->sync(this);
            switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 37, _ctx)) {
            case 1: {
              _localctx = _tracker.createInstance<Exp_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleExp_s);
              setState(548);

              if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
              setState(549);
              match(statementParser::MUL);
              setState(550);
              exp_s(5);
              break;
            }

            case 2: {
              _localctx = _tracker.createInstance<Exp_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleExp_s);
              setState(551);

              if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
              setState(552);
              match(statementParser::DIV);
              setState(553);
              exp_s(4);
              break;
            }

            case 3: {
              _localctx = _tracker.createInstance<Exp_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleExp_s);
              setState(554);

              if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
              setState(555);
              match(statementParser::PLUS);
              setState(556);
              exp_s(3);
              break;
            }

            case 4: {
              _localctx = _tracker.createInstance<Exp_sContext>(parentContext, parentState);
              pushNewRecursionContext(_localctx, startState, RuleExp_s);
              setState(557);

              if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
              setState(558);
              match(statementParser::MINUS);
              setState(559);
              exp_s(2);
              break;
            }

            } 
          }
          setState(564);
          _errHandler->sync(this);
          alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 38, _ctx);
        }
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }
      return _localctx;
    }

    //----------------- Id_sContext ------------------------------------------------------------------

    statementParser::Id_sContext::Id_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Id_sContext::Identifier_s() {
      return getToken(statementParser::Identifier_s, 0);
    }


    size_t statementParser::Id_sContext::getRuleIndex() const {
      return statementParser::RuleId_s;
    }

    void statementParser::Id_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterId_s(this);
    }

    void statementParser::Id_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitId_s(this);
    }


    antlrcpp::Any statementParser::Id_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitId_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Id_sContext* statementParser::id_s() {
      Id_sContext *_localctx = _tracker.createInstance<Id_sContext>(_ctx, getState());
      enterRule(_localctx, 92, statementParser::RuleId_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(565);
        match(statementParser::Identifier_s);
   
      }
      catch (RecognitionException &e) {
        _errHandler->reportError(this, e);
        _localctx->exception = std::current_exception();
        _errHandler->recover(this, _localctx->exception);
      }

      return _localctx;
    }

    //----------------- Id_list_sContext ------------------------------------------------------------------

    statementParser::Id_list_sContext::Id_list_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    std::vector<statementParser::Id_sContext *> statementParser::Id_list_sContext::id_s() {
      return getRuleContexts<statementParser::Id_sContext>();
    }

    statementParser::Id_sContext* statementParser::Id_list_sContext::id_s(size_t i) {
      return getRuleContext<statementParser::Id_sContext>(i);
    }

    std::vector<tree::TerminalNode *> statementParser::Id_list_sContext::COMMA() {
      return getTokens(statementParser::COMMA);
    }

    tree::TerminalNode* statementParser::Id_list_sContext::COMMA(size_t i) {
      return getToken(statementParser::COMMA, i);
    }


    size_t statementParser::Id_list_sContext::getRuleIndex() const {
      return statementParser::RuleId_list_s;
    }

    void statementParser::Id_list_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterId_list_s(this);
    }

    void statementParser::Id_list_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitId_list_s(this);
    }


    antlrcpp::Any statementParser::Id_list_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitId_list_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Id_list_sContext* statementParser::id_list_s() {
      Id_list_sContext *_localctx = _tracker.createInstance<Id_list_sContext>(_ctx, getState());
      enterRule(_localctx, 94, statementParser::RuleId_list_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(567);
        id_s();
        setState(572);
        _errHandler->sync(this);
        _la = _input->LA(1);
        while (_la == statementParser::COMMA) {
          setState(568);
          match(statementParser::COMMA);
          setState(569);
          id_s();
          setState(574);
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

    //----------------- Gate_name_sContext ------------------------------------------------------------------

    statementParser::Gate_name_sContext::Gate_name_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    statementParser::Single_gate_without_parameter_type_sContext* statementParser::Gate_name_sContext::single_gate_without_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_without_parameter_type_sContext>(0);
    }

    statementParser::Single_gate_with_one_parameter_type_sContext* statementParser::Gate_name_sContext::single_gate_with_one_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_one_parameter_type_sContext>(0);
    }

    statementParser::Single_gate_with_two_parameter_type_sContext* statementParser::Gate_name_sContext::single_gate_with_two_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_two_parameter_type_sContext>(0);
    }

    statementParser::Single_gate_with_three_parameter_type_sContext* statementParser::Gate_name_sContext::single_gate_with_three_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_three_parameter_type_sContext>(0);
    }

    statementParser::Single_gate_with_four_parameter_type_sContext* statementParser::Gate_name_sContext::single_gate_with_four_parameter_type_s() {
      return getRuleContext<statementParser::Single_gate_with_four_parameter_type_sContext>(0);
    }

    statementParser::Double_gate_without_parameter_type_sContext* statementParser::Gate_name_sContext::double_gate_without_parameter_type_s() {
      return getRuleContext<statementParser::Double_gate_without_parameter_type_sContext>(0);
    }

    statementParser::Double_gate_with_one_parameter_type_sContext* statementParser::Gate_name_sContext::double_gate_with_one_parameter_type_s() {
      return getRuleContext<statementParser::Double_gate_with_one_parameter_type_sContext>(0);
    }

    statementParser::Double_gate_with_four_parameter_type_sContext* statementParser::Gate_name_sContext::double_gate_with_four_parameter_type_s() {
      return getRuleContext<statementParser::Double_gate_with_four_parameter_type_sContext>(0);
    }

    statementParser::Triple_gate_without_parameter_type_sContext* statementParser::Gate_name_sContext::triple_gate_without_parameter_type_s() {
      return getRuleContext<statementParser::Triple_gate_without_parameter_type_sContext>(0);
    }


    size_t statementParser::Gate_name_sContext::getRuleIndex() const {
      return statementParser::RuleGate_name_s;
    }

    void statementParser::Gate_name_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterGate_name_s(this);
    }

    void statementParser::Gate_name_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitGate_name_s(this);
    }


    antlrcpp::Any statementParser::Gate_name_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitGate_name_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Gate_name_sContext* statementParser::gate_name_s() {
      Gate_name_sContext *_localctx = _tracker.createInstance<Gate_name_sContext>(_ctx, getState());
      enterRule(_localctx, 96, statementParser::RuleGate_name_s);

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        setState(584);
        _errHandler->sync(this);
        switch (_input->LA(1)) {
          case statementParser::ECHO_GATE:
          case statementParser::H_GATE:
          case statementParser::X_GATE:
          case statementParser::T_GATE:
          case statementParser::S_GATE:
          case statementParser::Y_GATE:
          case statementParser::Z_GATE:
          case statementParser::X1_GATE:
          case statementParser::Y1_GATE:
          case statementParser::Z1_GATE:
          case statementParser::I_GATE: {
            enterOuterAlt(_localctx, 1);
            setState(575);
            single_gate_without_parameter_type_s();
            break;
          }

          case statementParser::RX_GATE:
          case statementParser::RY_GATE:
          case statementParser::RZ_GATE:
          case statementParser::U1_GATE: {
            enterOuterAlt(_localctx, 2);
            setState(576);
            single_gate_with_one_parameter_type_s();
            break;
          }

          case statementParser::U2_GATE:
          case statementParser::RPHI_GATE: {
            enterOuterAlt(_localctx, 3);
            setState(577);
            single_gate_with_two_parameter_type_s();
            break;
          }

          case statementParser::U3_GATE: {
            enterOuterAlt(_localctx, 4);
            setState(578);
            single_gate_with_three_parameter_type_s();
            break;
          }

          case statementParser::U4_GATE: {
            enterOuterAlt(_localctx, 5);
            setState(579);
            single_gate_with_four_parameter_type_s();
            break;
          }

          case statementParser::CNOT_GATE:
          case statementParser::CZ_GATE:
          case statementParser::ISWAP_GATE:
          case statementParser::SQISWAP_GATE:
          case statementParser::SWAPZ1_GATE: {
            enterOuterAlt(_localctx, 6);
            setState(580);
            double_gate_without_parameter_type_s();
            break;
          }

          case statementParser::ISWAPTHETA_GATE:
          case statementParser::CR_GATE: {
            enterOuterAlt(_localctx, 7);
            setState(581);
            double_gate_with_one_parameter_type_s();
            break;
          }

          case statementParser::CU_GATE: {
            enterOuterAlt(_localctx, 8);
            setState(582);
            double_gate_with_four_parameter_type_s();
            break;
          }

          case statementParser::TOFFOLI_GATE: {
            enterOuterAlt(_localctx, 9);
            setState(583);
            triple_gate_without_parameter_type_s();
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

    //----------------- Constant_sContext ------------------------------------------------------------------

    statementParser::Constant_sContext::Constant_sContext(ParserRuleContext *parent, size_t invokingState)
      : ParserRuleContext(parent, invokingState) {
    }

    tree::TerminalNode* statementParser::Constant_sContext::Integer_Literal_s() {
      return getToken(statementParser::Integer_Literal_s, 0);
    }

    tree::TerminalNode* statementParser::Constant_sContext::Double_Literal_s() {
      return getToken(statementParser::Double_Literal_s, 0);
    }

    tree::TerminalNode* statementParser::Constant_sContext::PI() {
      return getToken(statementParser::PI, 0);
    }


    size_t statementParser::Constant_sContext::getRuleIndex() const {
      return statementParser::RuleConstant_s;
    }

    void statementParser::Constant_sContext::enterRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->enterConstant_s(this);
    }

    void statementParser::Constant_sContext::exitRule(tree::ParseTreeListener *listener) {
      auto parserListener = dynamic_cast<statementListener *>(listener);
      if (parserListener != nullptr)
        parserListener->exitConstant_s(this);
    }


    antlrcpp::Any statementParser::Constant_sContext::accept(tree::ParseTreeVisitor *visitor) {
      if (auto parserVisitor = dynamic_cast<statementVisitor*>(visitor))
        return parserVisitor->visitConstant_s(this);
      else
        return visitor->visitChildren(this);
    }

    statementParser::Constant_sContext* statementParser::constant_s() {
      Constant_sContext *_localctx = _tracker.createInstance<Constant_sContext>(_ctx, getState());
      enterRule(_localctx, 98, statementParser::RuleConstant_s);
      size_t _la = 0;

      auto onExit = finally([=] {
        exitRule();
      });
      try {
        enterOuterAlt(_localctx, 1);
        setState(586);
        _la = _input->LA(1);
        if (!(_la == statementParser::PI || _la == statementParser::Integer_Literal_s

        || _la == statementParser::Double_Literal_s)) {
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

    bool statementParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
      switch (ruleIndex) {
        case 25: return multiplicative_expression_sSempred(dynamic_cast<Multiplicative_expression_sContext *>(context), predicateIndex);
        case 26: return addtive_expression_sSempred(dynamic_cast<Addtive_expression_sContext *>(context), predicateIndex);
        case 27: return relational_expression_sSempred(dynamic_cast<Relational_expression_sContext *>(context), predicateIndex);
        case 28: return equality_expression_sSempred(dynamic_cast<Equality_expression_sContext *>(context), predicateIndex);
        case 29: return logical_and_expression_sSempred(dynamic_cast<Logical_and_expression_sContext *>(context), predicateIndex);
        case 30: return logical_or_expression_sSempred(dynamic_cast<Logical_or_expression_sContext *>(context), predicateIndex);
        case 45: return exp_sSempred(dynamic_cast<Exp_sContext *>(context), predicateIndex);

      default:
        break;
      }
      return true;
    }

    bool statementParser::multiplicative_expression_sSempred(Multiplicative_expression_sContext *_localctx, size_t predicateIndex) {
      switch (predicateIndex) {
        case 0: return precpred(_ctx, 2);
        case 1: return precpred(_ctx, 1);

      default:
        break;
      }
      return true;
    }

    bool statementParser::addtive_expression_sSempred(Addtive_expression_sContext *_localctx, size_t predicateIndex) {
      switch (predicateIndex) {
        case 2: return precpred(_ctx, 2);
        case 3: return precpred(_ctx, 1);

      default:
        break;
      }
      return true;
    }

    bool statementParser::relational_expression_sSempred(Relational_expression_sContext *_localctx, size_t predicateIndex) {
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

    bool statementParser::equality_expression_sSempred(Equality_expression_sContext *_localctx, size_t predicateIndex) {
      switch (predicateIndex) {
        case 8: return precpred(_ctx, 2);
        case 9: return precpred(_ctx, 1);

      default:
        break;
      }
      return true;
    }

    bool statementParser::logical_and_expression_sSempred(Logical_and_expression_sContext *_localctx, size_t predicateIndex) {
      switch (predicateIndex) {
        case 10: return precpred(_ctx, 1);

      default:
        break;
      }
      return true;
    }

    bool statementParser::logical_or_expression_sSempred(Logical_or_expression_sContext *_localctx, size_t predicateIndex) {
      switch (predicateIndex) {
        case 11: return precpred(_ctx, 1);

      default:
        break;
      }
      return true;
    }

    bool statementParser::exp_sSempred(Exp_sContext *_localctx, size_t predicateIndex) {
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
    std::vector<dfa::DFA> statementParser::_decisionToDFA;
    atn::PredictionContextCache statementParser::_sharedContextCache;

    // We own the ATN which in turn owns the ATN states.
    atn::ATN statementParser::_atn;
    std::vector<uint16_t> statementParser::_serializedATN;

    std::vector<std::string> statementParser::_ruleNames = {
      "translationunit_s", "quantum_gate_declaration_s", "index_s", "c_KEY_declaration_s", 
      "q_KEY_declaration_s", "single_gate_without_parameter_declaration_s", 
      "single_gate_with_one_parameter_declaration_s", "single_gate_with_two_parameter_declaration_s", 
      "single_gate_with_three_parameter_declaration_s", "single_gate_with_four_parameter_declaration_s", 
      "double_gate_without_parameter_declaration_s", "double_gate_with_one_parameter_declaration_s", 
      "double_gate_with_four_parameter_declaration_s", "triple_gate_without_parameter_declaration_s", 
      "single_gate_without_parameter_type_s", "single_gate_with_one_parameter_type_s", 
      "single_gate_with_two_parameter_type_s", "single_gate_with_three_parameter_type_s", 
      "single_gate_with_four_parameter_type_s", "double_gate_without_parameter_type_s", 
      "double_gate_with_one_parameter_type_s", "double_gate_with_four_parameter_type_s", 
      "triple_gate_without_parameter_type_s", "primary_expression_s", "unary_expression_s", 
      "multiplicative_expression_s", "addtive_expression_s", "relational_expression_s", 
      "equality_expression_s", "logical_and_expression_s", "logical_or_expression_s", 
      "assignment_expression_s", "expression_s", "controlbit_list_s", "statement_s", 
      "dagger_statement_s", "control_statement_s", "qelse_statement_fragment_s", 
      "qif_statement_s", "qwhile_statement_s", "measure_statement_s", "reset_statement_s", 
      "barrier_statement_s", "expression_statement_s", "explist_s", "exp_s", 
      "id_s", "id_list_s", "gate_name_s", "constant_s"
    };

    std::vector<std::string> statementParser::_literalNames = {
      "", "'PI'", "'QINIT'", "'CREG'", "'q'", "'c'", "'BARRIER'", "'QGATE'", 
      "'ENDQGATE'", "'ECHO'", "'H'", "'X'", "'NOT'", "'T'", "'S'", "'Y'", "'Z'", 
      "'X1'", "'Y1'", "'Z1'", "'I'", "'U2'", "'RPhi'", "'U3'", "'U4'", "'RX'", 
      "'RY'", "'RZ'", "'U1'", "'CNOT'", "'CZ'", "'CU'", "'ISWAP'", "'SQISWAP'", 
      "'SWAP'", "'ISWAPTHETA'", "'CR'", "'TOFFOLI'", "'DAGGER'", "'ENDDAGGER'", 
      "'CONTROL'", "'ENDCONTROL'", "'QIF'", "'ELSE'", "'ENDQIF'", "'QWHILE'", 
      "'ENDQWHILE'", "'MEASURE'", "'RESET'", "'='", "'>'", "'<'", "'!'", "'=='", 
      "'<='", "'>='", "'!='", "'&&'", "'||'", "'+'", "'-'", "'*'", "'/'", "','", 
      "'('", "')'", "'['", "']'"
    };

    std::vector<std::string> statementParser::_symbolicNames = {
      "", "PI", "QINIT_KEY", "CREG_KEY", "Q_KEY", "C_KEY", "BARRIER_KEY", "QGATE_KEY", 
      "ENDQGATE_KEY", "ECHO_GATE", "H_GATE", "X_GATE", "NOT_GATE", "T_GATE", 
      "S_GATE", "Y_GATE", "Z_GATE", "X1_GATE", "Y1_GATE", "Z1_GATE", "I_GATE", 
      "U2_GATE", "RPHI_GATE", "U3_GATE", "U4_GATE", "RX_GATE", "RY_GATE", "RZ_GATE", 
      "U1_GATE", "CNOT_GATE", "CZ_GATE", "CU_GATE", "ISWAP_GATE", "SQISWAP_GATE", 
      "SWAPZ1_GATE", "ISWAPTHETA_GATE", "CR_GATE", "TOFFOLI_GATE", "DAGGER_KEY", 
      "ENDDAGGER_KEY", "CONTROL_KEY", "ENDCONTROL_KEY", "QIF_KEY", "ELSE_KEY", 
      "ENDIF_KEY", "QWHILE_KEY", "ENDQWHILE_KEY", "MEASURE_KEY", "RESET_KEY", 
      "ASSIGN", "GT", "LT", "NOT", "EQ", "LEQ", "GEQ", "NE", "AND", "OR", "PLUS", 
      "MINUS", "MUL", "DIV", "COMMA", "LPAREN", "RPAREN", "LBRACK", "RBRACK", 
      "NEWLINE", "Identifier_s", "Integer_Literal_s", "Double_Literal_s", "Digit_Sequence_s", 
      "REALEXP_s", "WhiteSpace_s", "SingleLineComment_s"
    };

    dfa::Vocabulary statementParser::_vocabulary(_literalNames, _symbolicNames);

    std::vector<std::string> statementParser::_tokenNames;

    statementParser::Initializer::Initializer() {
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
        0x3, 0x4d, 0x24f, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
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
        0x32, 0x4, 0x33, 0x9, 0x33, 0x3, 0x2, 0x7, 0x2, 0x68, 0xa, 0x2, 0xc, 
        0x2, 0xe, 0x2, 0x6b, 0xb, 0x2, 0x3, 0x2, 0x6, 0x2, 0x6e, 0xa, 0x2, 0xd, 
        0x2, 0xe, 0x2, 0x6f, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
        0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x5, 0x3, 0x7b, 0xa, 0x3, 0x3, 
        0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
        0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 
        0x7, 0x3, 0x7, 0x5, 0x7, 0x8d, 0xa, 0x7, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
        0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
        0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0x9d, 0xa, 0x8, 0x3, 
        0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
        0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
        0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x5, 0x9, 0xb1, 0xa, 0x9, 0x3, 0xa, 
        0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
        0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
        0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 
        0x5, 0xa, 0xc9, 0xa, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 
        0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 
        0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 
        0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 
        0xb, 0x5, 0xb, 0xe5, 0xa, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 
        0x3, 0xc, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 
        0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
        0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
        0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 
        0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 
        0x3, 0x11, 0x3, 0x12, 0x3, 0x12, 0x3, 0x13, 0x3, 0x13, 0x3, 0x14, 0x3, 
        0x14, 0x3, 0x15, 0x3, 0x15, 0x3, 0x16, 0x3, 0x16, 0x3, 0x17, 0x3, 0x17, 
        0x3, 0x18, 0x3, 0x18, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 0x19, 0x3, 
        0x19, 0x3, 0x19, 0x5, 0x19, 0x123, 0xa, 0x19, 0x3, 0x1a, 0x3, 0x1a, 
        0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x3, 0x1a, 0x5, 0x1a, 0x12c, 
        0xa, 0x1a, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 
        0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x3, 0x1b, 0x7, 0x1b, 0x137, 0xa, 0x1b, 
        0xc, 0x1b, 0xe, 0x1b, 0x13a, 0xb, 0x1b, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 
        0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x3, 0x1c, 0x7, 
        0x1c, 0x145, 0xa, 0x1c, 0xc, 0x1c, 0xe, 0x1c, 0x148, 0xb, 0x1c, 0x3, 
        0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 
        0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 0x1d, 0x3, 
        0x1d, 0x3, 0x1d, 0x7, 0x1d, 0x159, 0xa, 0x1d, 0xc, 0x1d, 0xe, 0x1d, 
        0x15c, 0xb, 0x1d, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 
        0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x3, 0x1e, 0x7, 0x1e, 0x167, 0xa, 0x1e, 
        0xc, 0x1e, 0xe, 0x1e, 0x16a, 0xb, 0x1e, 0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 
        0x3, 0x1f, 0x3, 0x1f, 0x3, 0x1f, 0x7, 0x1f, 0x172, 0xa, 0x1f, 0xc, 0x1f, 
        0xe, 0x1f, 0x175, 0xb, 0x1f, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 0x3, 0x20, 
        0x3, 0x20, 0x3, 0x20, 0x7, 0x20, 0x17d, 0xa, 0x20, 0xc, 0x20, 0xe, 0x20, 
        0x180, 0xb, 0x20, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 0x3, 0x21, 
        0x5, 0x21, 0x187, 0xa, 0x21, 0x3, 0x22, 0x3, 0x22, 0x3, 0x23, 0x3, 0x23, 
        0x3, 0x23, 0x7, 0x23, 0x18e, 0xa, 0x23, 0xc, 0x23, 0xe, 0x23, 0x191, 
        0xb, 0x23, 0x3, 0x23, 0x3, 0x23, 0x3, 0x23, 0x7, 0x23, 0x196, 0xa, 0x23, 
        0xc, 0x23, 0xe, 0x23, 0x199, 0xb, 0x23, 0x5, 0x23, 0x19b, 0xa, 0x23, 
        0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 
        0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x3, 0x24, 0x5, 0x24, 0x1a8, 
        0xa, 0x24, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 0x7, 0x25, 0x1ad, 0xa, 0x25, 
        0xc, 0x25, 0xe, 0x25, 0x1b0, 0xb, 0x25, 0x3, 0x25, 0x3, 0x25, 0x3, 0x25, 
        0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 0x7, 0x26, 0x1b9, 0xa, 0x26, 
        0xc, 0x26, 0xe, 0x26, 0x1bc, 0xb, 0x26, 0x3, 0x26, 0x3, 0x26, 0x3, 0x26, 
        0x3, 0x27, 0x3, 0x27, 0x3, 0x27, 0x7, 0x27, 0x1c4, 0xa, 0x27, 0xc, 0x27, 
        0xe, 0x27, 0x1c7, 0xb, 0x27, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 
        0x7, 0x28, 0x1cd, 0xa, 0x28, 0xc, 0x28, 0xe, 0x28, 0x1d0, 0xb, 0x28, 
        0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 
        0x28, 0x3, 0x28, 0x7, 0x28, 0x1da, 0xa, 0x28, 0xc, 0x28, 0xe, 0x28, 
        0x1dd, 0xb, 0x28, 0x3, 0x28, 0x3, 0x28, 0x3, 0x28, 0x5, 0x28, 0x1e2, 
        0xa, 0x28, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x3, 0x29, 0x7, 0x29, 0x1e8, 
        0xa, 0x29, 0xc, 0x29, 0xe, 0x29, 0x1eb, 0xb, 0x29, 0x3, 0x29, 0x3, 0x29, 
        0x3, 0x29, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 
        0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x3, 0x2a, 0x5, 0x2a, 
        0x1fb, 0xa, 0x2a, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 0x3, 0x2b, 
        0x3, 0x2b, 0x3, 0x2b, 0x5, 0x2b, 0x204, 0xa, 0x2b, 0x3, 0x2c, 0x3, 0x2c, 
        0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x3, 0x2c, 0x5, 0x2c, 0x20d, 
        0xa, 0x2c, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2d, 0x3, 0x2e, 0x3, 0x2e, 0x3, 
        0x2e, 0x7, 0x2e, 0x215, 0xa, 0x2e, 0xc, 0x2e, 0xe, 0x2e, 0x218, 0xb, 
        0x2e, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 
        0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x5, 0x2f, 0x225, 
        0xa, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 
        0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 0x3, 0x2f, 
        0x7, 0x2f, 0x233, 0xa, 0x2f, 0xc, 0x2f, 0xe, 0x2f, 0x236, 0xb, 0x2f, 
        0x3, 0x30, 0x3, 0x30, 0x3, 0x31, 0x3, 0x31, 0x3, 0x31, 0x7, 0x31, 0x23d, 
        0xa, 0x31, 0xc, 0x31, 0xe, 0x31, 0x240, 0xb, 0x31, 0x3, 0x32, 0x3, 0x32, 
        0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 0x32, 0x3, 
        0x32, 0x5, 0x32, 0x24b, 0xa, 0x32, 0x3, 0x33, 0x3, 0x33, 0x3, 0x33, 
        0x2, 0x9, 0x34, 0x36, 0x38, 0x3a, 0x3c, 0x3e, 0x5c, 0x34, 0x2, 0x4, 
        0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 
        0x20, 0x22, 0x24, 0x26, 0x28, 0x2a, 0x2c, 0x2e, 0x30, 0x32, 0x34, 0x36, 
        0x38, 0x3a, 0x3c, 0x3e, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a, 0x4c, 0x4e, 
        0x50, 0x52, 0x54, 0x56, 0x58, 0x5a, 0x5c, 0x5e, 0x60, 0x62, 0x64, 0x2, 
        0x8, 0x4, 0x2, 0xb, 0xd, 0xf, 0x16, 0x3, 0x2, 0x1b, 0x1e, 0x3, 0x2, 
        0x17, 0x18, 0x4, 0x2, 0x1f, 0x20, 0x22, 0x24, 0x3, 0x2, 0x25, 0x26, 
        0x4, 0x2, 0x3, 0x3, 0x48, 0x49, 0x2, 0x265, 0x2, 0x69, 0x3, 0x2, 0x2, 
        0x2, 0x4, 0x7a, 0x3, 0x2, 0x2, 0x2, 0x6, 0x7c, 0x3, 0x2, 0x2, 0x2, 0x8, 
        0x80, 0x3, 0x2, 0x2, 0x2, 0xa, 0x83, 0x3, 0x2, 0x2, 0x2, 0xc, 0x8c, 
        0x3, 0x2, 0x2, 0x2, 0xe, 0x9c, 0x3, 0x2, 0x2, 0x2, 0x10, 0xb0, 0x3, 
        0x2, 0x2, 0x2, 0x12, 0xc8, 0x3, 0x2, 0x2, 0x2, 0x14, 0xe4, 0x3, 0x2, 
        0x2, 0x2, 0x16, 0xe6, 0x3, 0x2, 0x2, 0x2, 0x18, 0xeb, 0x3, 0x2, 0x2, 
        0x2, 0x1a, 0xf4, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x103, 0x3, 0x2, 0x2, 0x2, 
        0x1e, 0x10a, 0x3, 0x2, 0x2, 0x2, 0x20, 0x10c, 0x3, 0x2, 0x2, 0x2, 0x22, 
        0x10e, 0x3, 0x2, 0x2, 0x2, 0x24, 0x110, 0x3, 0x2, 0x2, 0x2, 0x26, 0x112, 
        0x3, 0x2, 0x2, 0x2, 0x28, 0x114, 0x3, 0x2, 0x2, 0x2, 0x2a, 0x116, 0x3, 
        0x2, 0x2, 0x2, 0x2c, 0x118, 0x3, 0x2, 0x2, 0x2, 0x2e, 0x11a, 0x3, 0x2, 
        0x2, 0x2, 0x30, 0x122, 0x3, 0x2, 0x2, 0x2, 0x32, 0x12b, 0x3, 0x2, 0x2, 
        0x2, 0x34, 0x12d, 0x3, 0x2, 0x2, 0x2, 0x36, 0x13b, 0x3, 0x2, 0x2, 0x2, 
        0x38, 0x149, 0x3, 0x2, 0x2, 0x2, 0x3a, 0x15d, 0x3, 0x2, 0x2, 0x2, 0x3c, 
        0x16b, 0x3, 0x2, 0x2, 0x2, 0x3e, 0x176, 0x3, 0x2, 0x2, 0x2, 0x40, 0x186, 
        0x3, 0x2, 0x2, 0x2, 0x42, 0x188, 0x3, 0x2, 0x2, 0x2, 0x44, 0x19a, 0x3, 
        0x2, 0x2, 0x2, 0x46, 0x1a7, 0x3, 0x2, 0x2, 0x2, 0x48, 0x1a9, 0x3, 0x2, 
        0x2, 0x2, 0x4a, 0x1b4, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x1c0, 0x3, 0x2, 0x2, 
        0x2, 0x4e, 0x1e1, 0x3, 0x2, 0x2, 0x2, 0x50, 0x1e3, 0x3, 0x2, 0x2, 0x2, 
        0x52, 0x1fa, 0x3, 0x2, 0x2, 0x2, 0x54, 0x203, 0x3, 0x2, 0x2, 0x2, 0x56, 
        0x20c, 0x3, 0x2, 0x2, 0x2, 0x58, 0x20e, 0x3, 0x2, 0x2, 0x2, 0x5a, 0x211, 
        0x3, 0x2, 0x2, 0x2, 0x5c, 0x224, 0x3, 0x2, 0x2, 0x2, 0x5e, 0x237, 0x3, 
        0x2, 0x2, 0x2, 0x60, 0x239, 0x3, 0x2, 0x2, 0x2, 0x62, 0x24a, 0x3, 0x2, 
        0x2, 0x2, 0x64, 0x24c, 0x3, 0x2, 0x2, 0x2, 0x66, 0x68, 0x7, 0x46, 0x2, 
        0x2, 0x67, 0x66, 0x3, 0x2, 0x2, 0x2, 0x68, 0x6b, 0x3, 0x2, 0x2, 0x2, 
        0x69, 0x67, 0x3, 0x2, 0x2, 0x2, 0x69, 0x6a, 0x3, 0x2, 0x2, 0x2, 0x6a, 
        0x6d, 0x3, 0x2, 0x2, 0x2, 0x6b, 0x69, 0x3, 0x2, 0x2, 0x2, 0x6c, 0x6e, 
        0x5, 0x46, 0x24, 0x2, 0x6d, 0x6c, 0x3, 0x2, 0x2, 0x2, 0x6e, 0x6f, 0x3, 
        0x2, 0x2, 0x2, 0x6f, 0x6d, 0x3, 0x2, 0x2, 0x2, 0x6f, 0x70, 0x3, 0x2, 
        0x2, 0x2, 0x70, 0x3, 0x3, 0x2, 0x2, 0x2, 0x71, 0x7b, 0x5, 0xc, 0x7, 
        0x2, 0x72, 0x7b, 0x5, 0xe, 0x8, 0x2, 0x73, 0x7b, 0x5, 0x10, 0x9, 0x2, 
        0x74, 0x7b, 0x5, 0x12, 0xa, 0x2, 0x75, 0x7b, 0x5, 0x14, 0xb, 0x2, 0x76, 
        0x7b, 0x5, 0x16, 0xc, 0x2, 0x77, 0x7b, 0x5, 0x18, 0xd, 0x2, 0x78, 0x7b, 
        0x5, 0x1a, 0xe, 0x2, 0x79, 0x7b, 0x5, 0x1c, 0xf, 0x2, 0x7a, 0x71, 0x3, 
        0x2, 0x2, 0x2, 0x7a, 0x72, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x73, 0x3, 0x2, 
        0x2, 0x2, 0x7a, 0x74, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x75, 0x3, 0x2, 0x2, 
        0x2, 0x7a, 0x76, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x77, 0x3, 0x2, 0x2, 0x2, 
        0x7a, 0x78, 0x3, 0x2, 0x2, 0x2, 0x7a, 0x79, 0x3, 0x2, 0x2, 0x2, 0x7b, 
        0x5, 0x3, 0x2, 0x2, 0x2, 0x7c, 0x7d, 0x7, 0x44, 0x2, 0x2, 0x7d, 0x7e, 
        0x5, 0x42, 0x22, 0x2, 0x7e, 0x7f, 0x7, 0x45, 0x2, 0x2, 0x7f, 0x7, 0x3, 
        0x2, 0x2, 0x2, 0x80, 0x81, 0x7, 0x7, 0x2, 0x2, 0x81, 0x82, 0x5, 0x6, 
        0x4, 0x2, 0x82, 0x9, 0x3, 0x2, 0x2, 0x2, 0x83, 0x84, 0x7, 0x6, 0x2, 
        0x2, 0x84, 0x85, 0x5, 0x6, 0x4, 0x2, 0x85, 0xb, 0x3, 0x2, 0x2, 0x2, 
        0x86, 0x87, 0x5, 0x1e, 0x10, 0x2, 0x87, 0x88, 0x5, 0xa, 0x6, 0x2, 0x88, 
        0x8d, 0x3, 0x2, 0x2, 0x2, 0x89, 0x8a, 0x5, 0x1e, 0x10, 0x2, 0x8a, 0x8b, 
        0x7, 0x6, 0x2, 0x2, 0x8b, 0x8d, 0x3, 0x2, 0x2, 0x2, 0x8c, 0x86, 0x3, 
        0x2, 0x2, 0x2, 0x8c, 0x89, 0x3, 0x2, 0x2, 0x2, 0x8d, 0xd, 0x3, 0x2, 
        0x2, 0x2, 0x8e, 0x8f, 0x5, 0x20, 0x11, 0x2, 0x8f, 0x90, 0x5, 0xa, 0x6, 
        0x2, 0x90, 0x91, 0x7, 0x41, 0x2, 0x2, 0x91, 0x92, 0x7, 0x42, 0x2, 0x2, 
        0x92, 0x93, 0x5, 0x42, 0x22, 0x2, 0x93, 0x94, 0x7, 0x43, 0x2, 0x2, 0x94, 
        0x9d, 0x3, 0x2, 0x2, 0x2, 0x95, 0x96, 0x5, 0x20, 0x11, 0x2, 0x96, 0x97, 
        0x7, 0x6, 0x2, 0x2, 0x97, 0x98, 0x7, 0x41, 0x2, 0x2, 0x98, 0x99, 0x7, 
        0x42, 0x2, 0x2, 0x99, 0x9a, 0x5, 0x42, 0x22, 0x2, 0x9a, 0x9b, 0x7, 0x43, 
        0x2, 0x2, 0x9b, 0x9d, 0x3, 0x2, 0x2, 0x2, 0x9c, 0x8e, 0x3, 0x2, 0x2, 
        0x2, 0x9c, 0x95, 0x3, 0x2, 0x2, 0x2, 0x9d, 0xf, 0x3, 0x2, 0x2, 0x2, 
        0x9e, 0x9f, 0x5, 0x22, 0x12, 0x2, 0x9f, 0xa0, 0x5, 0xa, 0x6, 0x2, 0xa0, 
        0xa1, 0x7, 0x41, 0x2, 0x2, 0xa1, 0xa2, 0x7, 0x42, 0x2, 0x2, 0xa2, 0xa3, 
        0x5, 0x42, 0x22, 0x2, 0xa3, 0xa4, 0x7, 0x41, 0x2, 0x2, 0xa4, 0xa5, 0x5, 
        0x42, 0x22, 0x2, 0xa5, 0xa6, 0x7, 0x43, 0x2, 0x2, 0xa6, 0xb1, 0x3, 0x2, 
        0x2, 0x2, 0xa7, 0xa8, 0x5, 0x22, 0x12, 0x2, 0xa8, 0xa9, 0x7, 0x6, 0x2, 
        0x2, 0xa9, 0xaa, 0x7, 0x41, 0x2, 0x2, 0xaa, 0xab, 0x7, 0x42, 0x2, 0x2, 
        0xab, 0xac, 0x5, 0x42, 0x22, 0x2, 0xac, 0xad, 0x7, 0x41, 0x2, 0x2, 0xad, 
        0xae, 0x5, 0x42, 0x22, 0x2, 0xae, 0xaf, 0x7, 0x43, 0x2, 0x2, 0xaf, 0xb1, 
        0x3, 0x2, 0x2, 0x2, 0xb0, 0x9e, 0x3, 0x2, 0x2, 0x2, 0xb0, 0xa7, 0x3, 
        0x2, 0x2, 0x2, 0xb1, 0x11, 0x3, 0x2, 0x2, 0x2, 0xb2, 0xb3, 0x5, 0x24, 
        0x13, 0x2, 0xb3, 0xb4, 0x5, 0xa, 0x6, 0x2, 0xb4, 0xb5, 0x7, 0x41, 0x2, 
        0x2, 0xb5, 0xb6, 0x7, 0x42, 0x2, 0x2, 0xb6, 0xb7, 0x5, 0x42, 0x22, 0x2, 
        0xb7, 0xb8, 0x7, 0x41, 0x2, 0x2, 0xb8, 0xb9, 0x5, 0x42, 0x22, 0x2, 0xb9, 
        0xba, 0x7, 0x41, 0x2, 0x2, 0xba, 0xbb, 0x5, 0x42, 0x22, 0x2, 0xbb, 0xbc, 
        0x7, 0x43, 0x2, 0x2, 0xbc, 0xc9, 0x3, 0x2, 0x2, 0x2, 0xbd, 0xbe, 0x5, 
        0x24, 0x13, 0x2, 0xbe, 0xbf, 0x7, 0x6, 0x2, 0x2, 0xbf, 0xc0, 0x7, 0x41, 
        0x2, 0x2, 0xc0, 0xc1, 0x7, 0x42, 0x2, 0x2, 0xc1, 0xc2, 0x5, 0x42, 0x22, 
        0x2, 0xc2, 0xc3, 0x7, 0x41, 0x2, 0x2, 0xc3, 0xc4, 0x5, 0x42, 0x22, 0x2, 
        0xc4, 0xc5, 0x7, 0x41, 0x2, 0x2, 0xc5, 0xc6, 0x5, 0x42, 0x22, 0x2, 0xc6, 
        0xc7, 0x7, 0x43, 0x2, 0x2, 0xc7, 0xc9, 0x3, 0x2, 0x2, 0x2, 0xc8, 0xb2, 
        0x3, 0x2, 0x2, 0x2, 0xc8, 0xbd, 0x3, 0x2, 0x2, 0x2, 0xc9, 0x13, 0x3, 
        0x2, 0x2, 0x2, 0xca, 0xcb, 0x5, 0x26, 0x14, 0x2, 0xcb, 0xcc, 0x5, 0xa, 
        0x6, 0x2, 0xcc, 0xcd, 0x7, 0x41, 0x2, 0x2, 0xcd, 0xce, 0x7, 0x42, 0x2, 
        0x2, 0xce, 0xcf, 0x5, 0x42, 0x22, 0x2, 0xcf, 0xd0, 0x7, 0x41, 0x2, 0x2, 
        0xd0, 0xd1, 0x5, 0x42, 0x22, 0x2, 0xd1, 0xd2, 0x7, 0x41, 0x2, 0x2, 0xd2, 
        0xd3, 0x5, 0x42, 0x22, 0x2, 0xd3, 0xd4, 0x7, 0x41, 0x2, 0x2, 0xd4, 0xd5, 
        0x5, 0x42, 0x22, 0x2, 0xd5, 0xd6, 0x7, 0x43, 0x2, 0x2, 0xd6, 0xe5, 0x3, 
        0x2, 0x2, 0x2, 0xd7, 0xd8, 0x5, 0x26, 0x14, 0x2, 0xd8, 0xd9, 0x7, 0x6, 
        0x2, 0x2, 0xd9, 0xda, 0x7, 0x41, 0x2, 0x2, 0xda, 0xdb, 0x7, 0x42, 0x2, 
        0x2, 0xdb, 0xdc, 0x5, 0x42, 0x22, 0x2, 0xdc, 0xdd, 0x7, 0x41, 0x2, 0x2, 
        0xdd, 0xde, 0x5, 0x42, 0x22, 0x2, 0xde, 0xdf, 0x7, 0x41, 0x2, 0x2, 0xdf, 
        0xe0, 0x5, 0x42, 0x22, 0x2, 0xe0, 0xe1, 0x7, 0x41, 0x2, 0x2, 0xe1, 0xe2, 
        0x5, 0x42, 0x22, 0x2, 0xe2, 0xe3, 0x7, 0x43, 0x2, 0x2, 0xe3, 0xe5, 0x3, 
        0x2, 0x2, 0x2, 0xe4, 0xca, 0x3, 0x2, 0x2, 0x2, 0xe4, 0xd7, 0x3, 0x2, 
        0x2, 0x2, 0xe5, 0x15, 0x3, 0x2, 0x2, 0x2, 0xe6, 0xe7, 0x5, 0x28, 0x15, 
        0x2, 0xe7, 0xe8, 0x5, 0xa, 0x6, 0x2, 0xe8, 0xe9, 0x7, 0x41, 0x2, 0x2, 
        0xe9, 0xea, 0x5, 0xa, 0x6, 0x2, 0xea, 0x17, 0x3, 0x2, 0x2, 0x2, 0xeb, 
        0xec, 0x5, 0x2a, 0x16, 0x2, 0xec, 0xed, 0x5, 0xa, 0x6, 0x2, 0xed, 0xee, 
        0x7, 0x41, 0x2, 0x2, 0xee, 0xef, 0x5, 0xa, 0x6, 0x2, 0xef, 0xf0, 0x7, 
        0x41, 0x2, 0x2, 0xf0, 0xf1, 0x7, 0x42, 0x2, 0x2, 0xf1, 0xf2, 0x5, 0x42, 
        0x22, 0x2, 0xf2, 0xf3, 0x7, 0x43, 0x2, 0x2, 0xf3, 0x19, 0x3, 0x2, 0x2, 
        0x2, 0xf4, 0xf5, 0x5, 0x2c, 0x17, 0x2, 0xf5, 0xf6, 0x5, 0xa, 0x6, 0x2, 
        0xf6, 0xf7, 0x7, 0x41, 0x2, 0x2, 0xf7, 0xf8, 0x5, 0xa, 0x6, 0x2, 0xf8, 
        0xf9, 0x7, 0x41, 0x2, 0x2, 0xf9, 0xfa, 0x7, 0x42, 0x2, 0x2, 0xfa, 0xfb, 
        0x5, 0x42, 0x22, 0x2, 0xfb, 0xfc, 0x7, 0x41, 0x2, 0x2, 0xfc, 0xfd, 0x5, 
        0x42, 0x22, 0x2, 0xfd, 0xfe, 0x7, 0x41, 0x2, 0x2, 0xfe, 0xff, 0x5, 0x42, 
        0x22, 0x2, 0xff, 0x100, 0x7, 0x41, 0x2, 0x2, 0x100, 0x101, 0x5, 0x42, 
        0x22, 0x2, 0x101, 0x102, 0x7, 0x43, 0x2, 0x2, 0x102, 0x1b, 0x3, 0x2, 
        0x2, 0x2, 0x103, 0x104, 0x5, 0x2e, 0x18, 0x2, 0x104, 0x105, 0x5, 0xa, 
        0x6, 0x2, 0x105, 0x106, 0x7, 0x41, 0x2, 0x2, 0x106, 0x107, 0x5, 0xa, 
        0x6, 0x2, 0x107, 0x108, 0x7, 0x41, 0x2, 0x2, 0x108, 0x109, 0x5, 0xa, 
        0x6, 0x2, 0x109, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x10a, 0x10b, 0x9, 0x2, 0x2, 
        0x2, 0x10b, 0x1f, 0x3, 0x2, 0x2, 0x2, 0x10c, 0x10d, 0x9, 0x3, 0x2, 0x2, 
        0x10d, 0x21, 0x3, 0x2, 0x2, 0x2, 0x10e, 0x10f, 0x9, 0x4, 0x2, 0x2, 0x10f, 
        0x23, 0x3, 0x2, 0x2, 0x2, 0x110, 0x111, 0x7, 0x19, 0x2, 0x2, 0x111, 
        0x25, 0x3, 0x2, 0x2, 0x2, 0x112, 0x113, 0x7, 0x1a, 0x2, 0x2, 0x113, 
        0x27, 0x3, 0x2, 0x2, 0x2, 0x114, 0x115, 0x9, 0x5, 0x2, 0x2, 0x115, 0x29, 
        0x3, 0x2, 0x2, 0x2, 0x116, 0x117, 0x9, 0x6, 0x2, 0x2, 0x117, 0x2b, 0x3, 
        0x2, 0x2, 0x2, 0x118, 0x119, 0x7, 0x21, 0x2, 0x2, 0x119, 0x2d, 0x3, 
        0x2, 0x2, 0x2, 0x11a, 0x11b, 0x7, 0x27, 0x2, 0x2, 0x11b, 0x2f, 0x3, 
        0x2, 0x2, 0x2, 0x11c, 0x123, 0x5, 0x8, 0x5, 0x2, 0x11d, 0x123, 0x5, 
        0x64, 0x33, 0x2, 0x11e, 0x11f, 0x7, 0x42, 0x2, 0x2, 0x11f, 0x120, 0x5, 
        0x42, 0x22, 0x2, 0x120, 0x121, 0x7, 0x42, 0x2, 0x2, 0x121, 0x123, 0x3, 
        0x2, 0x2, 0x2, 0x122, 0x11c, 0x3, 0x2, 0x2, 0x2, 0x122, 0x11d, 0x3, 
        0x2, 0x2, 0x2, 0x122, 0x11e, 0x3, 0x2, 0x2, 0x2, 0x123, 0x31, 0x3, 0x2, 
        0x2, 0x2, 0x124, 0x12c, 0x5, 0x30, 0x19, 0x2, 0x125, 0x126, 0x7, 0x3d, 
        0x2, 0x2, 0x126, 0x12c, 0x5, 0x30, 0x19, 0x2, 0x127, 0x128, 0x7, 0x3e, 
        0x2, 0x2, 0x128, 0x12c, 0x5, 0x30, 0x19, 0x2, 0x129, 0x12a, 0x7, 0x36, 
        0x2, 0x2, 0x12a, 0x12c, 0x5, 0x30, 0x19, 0x2, 0x12b, 0x124, 0x3, 0x2, 
        0x2, 0x2, 0x12b, 0x125, 0x3, 0x2, 0x2, 0x2, 0x12b, 0x127, 0x3, 0x2, 
        0x2, 0x2, 0x12b, 0x129, 0x3, 0x2, 0x2, 0x2, 0x12c, 0x33, 0x3, 0x2, 0x2, 
        0x2, 0x12d, 0x12e, 0x8, 0x1b, 0x1, 0x2, 0x12e, 0x12f, 0x5, 0x32, 0x1a, 
        0x2, 0x12f, 0x138, 0x3, 0x2, 0x2, 0x2, 0x130, 0x131, 0xc, 0x4, 0x2, 
        0x2, 0x131, 0x132, 0x7, 0x3f, 0x2, 0x2, 0x132, 0x137, 0x5, 0x32, 0x1a, 
        0x2, 0x133, 0x134, 0xc, 0x3, 0x2, 0x2, 0x134, 0x135, 0x7, 0x40, 0x2, 
        0x2, 0x135, 0x137, 0x5, 0x32, 0x1a, 0x2, 0x136, 0x130, 0x3, 0x2, 0x2, 
        0x2, 0x136, 0x133, 0x3, 0x2, 0x2, 0x2, 0x137, 0x13a, 0x3, 0x2, 0x2, 
        0x2, 0x138, 0x136, 0x3, 0x2, 0x2, 0x2, 0x138, 0x139, 0x3, 0x2, 0x2, 
        0x2, 0x139, 0x35, 0x3, 0x2, 0x2, 0x2, 0x13a, 0x138, 0x3, 0x2, 0x2, 0x2, 
        0x13b, 0x13c, 0x8, 0x1c, 0x1, 0x2, 0x13c, 0x13d, 0x5, 0x34, 0x1b, 0x2, 
        0x13d, 0x146, 0x3, 0x2, 0x2, 0x2, 0x13e, 0x13f, 0xc, 0x4, 0x2, 0x2, 
        0x13f, 0x140, 0x7, 0x3d, 0x2, 0x2, 0x140, 0x145, 0x5, 0x34, 0x1b, 0x2, 
        0x141, 0x142, 0xc, 0x3, 0x2, 0x2, 0x142, 0x143, 0x7, 0x3e, 0x2, 0x2, 
        0x143, 0x145, 0x5, 0x34, 0x1b, 0x2, 0x144, 0x13e, 0x3, 0x2, 0x2, 0x2, 
        0x144, 0x141, 0x3, 0x2, 0x2, 0x2, 0x145, 0x148, 0x3, 0x2, 0x2, 0x2, 
        0x146, 0x144, 0x3, 0x2, 0x2, 0x2, 0x146, 0x147, 0x3, 0x2, 0x2, 0x2, 
        0x147, 0x37, 0x3, 0x2, 0x2, 0x2, 0x148, 0x146, 0x3, 0x2, 0x2, 0x2, 0x149, 
        0x14a, 0x8, 0x1d, 0x1, 0x2, 0x14a, 0x14b, 0x5, 0x36, 0x1c, 0x2, 0x14b, 
        0x15a, 0x3, 0x2, 0x2, 0x2, 0x14c, 0x14d, 0xc, 0x6, 0x2, 0x2, 0x14d, 
        0x14e, 0x7, 0x35, 0x2, 0x2, 0x14e, 0x159, 0x5, 0x36, 0x1c, 0x2, 0x14f, 
        0x150, 0xc, 0x5, 0x2, 0x2, 0x150, 0x151, 0x7, 0x34, 0x2, 0x2, 0x151, 
        0x159, 0x5, 0x36, 0x1c, 0x2, 0x152, 0x153, 0xc, 0x4, 0x2, 0x2, 0x153, 
        0x154, 0x7, 0x38, 0x2, 0x2, 0x154, 0x159, 0x5, 0x36, 0x1c, 0x2, 0x155, 
        0x156, 0xc, 0x3, 0x2, 0x2, 0x156, 0x157, 0x7, 0x39, 0x2, 0x2, 0x157, 
        0x159, 0x5, 0x36, 0x1c, 0x2, 0x158, 0x14c, 0x3, 0x2, 0x2, 0x2, 0x158, 
        0x14f, 0x3, 0x2, 0x2, 0x2, 0x158, 0x152, 0x3, 0x2, 0x2, 0x2, 0x158, 
        0x155, 0x3, 0x2, 0x2, 0x2, 0x159, 0x15c, 0x3, 0x2, 0x2, 0x2, 0x15a, 
        0x158, 0x3, 0x2, 0x2, 0x2, 0x15a, 0x15b, 0x3, 0x2, 0x2, 0x2, 0x15b, 
        0x39, 0x3, 0x2, 0x2, 0x2, 0x15c, 0x15a, 0x3, 0x2, 0x2, 0x2, 0x15d, 0x15e, 
        0x8, 0x1e, 0x1, 0x2, 0x15e, 0x15f, 0x5, 0x38, 0x1d, 0x2, 0x15f, 0x168, 
        0x3, 0x2, 0x2, 0x2, 0x160, 0x161, 0xc, 0x4, 0x2, 0x2, 0x161, 0x162, 
        0x7, 0x37, 0x2, 0x2, 0x162, 0x167, 0x5, 0x38, 0x1d, 0x2, 0x163, 0x164, 
        0xc, 0x3, 0x2, 0x2, 0x164, 0x165, 0x7, 0x3a, 0x2, 0x2, 0x165, 0x167, 
        0x5, 0x38, 0x1d, 0x2, 0x166, 0x160, 0x3, 0x2, 0x2, 0x2, 0x166, 0x163, 
        0x3, 0x2, 0x2, 0x2, 0x167, 0x16a, 0x3, 0x2, 0x2, 0x2, 0x168, 0x166, 
        0x3, 0x2, 0x2, 0x2, 0x168, 0x169, 0x3, 0x2, 0x2, 0x2, 0x169, 0x3b, 0x3, 
        0x2, 0x2, 0x2, 0x16a, 0x168, 0x3, 0x2, 0x2, 0x2, 0x16b, 0x16c, 0x8, 
        0x1f, 0x1, 0x2, 0x16c, 0x16d, 0x5, 0x3a, 0x1e, 0x2, 0x16d, 0x173, 0x3, 
        0x2, 0x2, 0x2, 0x16e, 0x16f, 0xc, 0x3, 0x2, 0x2, 0x16f, 0x170, 0x7, 
        0x3b, 0x2, 0x2, 0x170, 0x172, 0x5, 0x3a, 0x1e, 0x2, 0x171, 0x16e, 0x3, 
        0x2, 0x2, 0x2, 0x172, 0x175, 0x3, 0x2, 0x2, 0x2, 0x173, 0x171, 0x3, 
        0x2, 0x2, 0x2, 0x173, 0x174, 0x3, 0x2, 0x2, 0x2, 0x174, 0x3d, 0x3, 0x2, 
        0x2, 0x2, 0x175, 0x173, 0x3, 0x2, 0x2, 0x2, 0x176, 0x177, 0x8, 0x20, 
        0x1, 0x2, 0x177, 0x178, 0x5, 0x3c, 0x1f, 0x2, 0x178, 0x17e, 0x3, 0x2, 
        0x2, 0x2, 0x179, 0x17a, 0xc, 0x3, 0x2, 0x2, 0x17a, 0x17b, 0x7, 0x3c, 
        0x2, 0x2, 0x17b, 0x17d, 0x5, 0x3c, 0x1f, 0x2, 0x17c, 0x179, 0x3, 0x2, 
        0x2, 0x2, 0x17d, 0x180, 0x3, 0x2, 0x2, 0x2, 0x17e, 0x17c, 0x3, 0x2, 
        0x2, 0x2, 0x17e, 0x17f, 0x3, 0x2, 0x2, 0x2, 0x17f, 0x3f, 0x3, 0x2, 0x2, 
        0x2, 0x180, 0x17e, 0x3, 0x2, 0x2, 0x2, 0x181, 0x187, 0x5, 0x3e, 0x20, 
        0x2, 0x182, 0x183, 0x5, 0x8, 0x5, 0x2, 0x183, 0x184, 0x7, 0x33, 0x2, 
        0x2, 0x184, 0x185, 0x5, 0x3e, 0x20, 0x2, 0x185, 0x187, 0x3, 0x2, 0x2, 
        0x2, 0x186, 0x181, 0x3, 0x2, 0x2, 0x2, 0x186, 0x182, 0x3, 0x2, 0x2, 
        0x2, 0x187, 0x41, 0x3, 0x2, 0x2, 0x2, 0x188, 0x189, 0x5, 0x40, 0x21, 
        0x2, 0x189, 0x43, 0x3, 0x2, 0x2, 0x2, 0x18a, 0x18f, 0x5, 0xa, 0x6, 0x2, 
        0x18b, 0x18c, 0x7, 0x41, 0x2, 0x2, 0x18c, 0x18e, 0x5, 0xa, 0x6, 0x2, 
        0x18d, 0x18b, 0x3, 0x2, 0x2, 0x2, 0x18e, 0x191, 0x3, 0x2, 0x2, 0x2, 
        0x18f, 0x18d, 0x3, 0x2, 0x2, 0x2, 0x18f, 0x190, 0x3, 0x2, 0x2, 0x2, 
        0x190, 0x19b, 0x3, 0x2, 0x2, 0x2, 0x191, 0x18f, 0x3, 0x2, 0x2, 0x2, 
        0x192, 0x197, 0x7, 0x47, 0x2, 0x2, 0x193, 0x194, 0x7, 0x41, 0x2, 0x2, 
        0x194, 0x196, 0x7, 0x47, 0x2, 0x2, 0x195, 0x193, 0x3, 0x2, 0x2, 0x2, 
        0x196, 0x199, 0x3, 0x2, 0x2, 0x2, 0x197, 0x195, 0x3, 0x2, 0x2, 0x2, 
        0x197, 0x198, 0x3, 0x2, 0x2, 0x2, 0x198, 0x19b, 0x3, 0x2, 0x2, 0x2, 
        0x199, 0x197, 0x3, 0x2, 0x2, 0x2, 0x19a, 0x18a, 0x3, 0x2, 0x2, 0x2, 
        0x19a, 0x192, 0x3, 0x2, 0x2, 0x2, 0x19b, 0x45, 0x3, 0x2, 0x2, 0x2, 0x19c, 
        0x19d, 0x5, 0x4, 0x3, 0x2, 0x19d, 0x19e, 0x7, 0x46, 0x2, 0x2, 0x19e, 
        0x1a8, 0x3, 0x2, 0x2, 0x2, 0x19f, 0x1a8, 0x5, 0x4a, 0x26, 0x2, 0x1a0, 
        0x1a8, 0x5, 0x4e, 0x28, 0x2, 0x1a1, 0x1a8, 0x5, 0x50, 0x29, 0x2, 0x1a2, 
        0x1a8, 0x5, 0x48, 0x25, 0x2, 0x1a3, 0x1a8, 0x5, 0x52, 0x2a, 0x2, 0x1a4, 
        0x1a8, 0x5, 0x54, 0x2b, 0x2, 0x1a5, 0x1a8, 0x5, 0x58, 0x2d, 0x2, 0x1a6, 
        0x1a8, 0x5, 0x56, 0x2c, 0x2, 0x1a7, 0x19c, 0x3, 0x2, 0x2, 0x2, 0x1a7, 
        0x19f, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x1a0, 0x3, 0x2, 0x2, 0x2, 0x1a7, 
        0x1a1, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x1a2, 0x3, 0x2, 0x2, 0x2, 0x1a7, 
        0x1a3, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x1a4, 0x3, 0x2, 0x2, 0x2, 0x1a7, 
        0x1a5, 0x3, 0x2, 0x2, 0x2, 0x1a7, 0x1a6, 0x3, 0x2, 0x2, 0x2, 0x1a8, 
        0x47, 0x3, 0x2, 0x2, 0x2, 0x1a9, 0x1aa, 0x7, 0x28, 0x2, 0x2, 0x1aa, 
        0x1ae, 0x7, 0x46, 0x2, 0x2, 0x1ab, 0x1ad, 0x5, 0x46, 0x24, 0x2, 0x1ac, 
        0x1ab, 0x3, 0x2, 0x2, 0x2, 0x1ad, 0x1b0, 0x3, 0x2, 0x2, 0x2, 0x1ae, 
        0x1ac, 0x3, 0x2, 0x2, 0x2, 0x1ae, 0x1af, 0x3, 0x2, 0x2, 0x2, 0x1af, 
        0x1b1, 0x3, 0x2, 0x2, 0x2, 0x1b0, 0x1ae, 0x3, 0x2, 0x2, 0x2, 0x1b1, 
        0x1b2, 0x7, 0x29, 0x2, 0x2, 0x1b2, 0x1b3, 0x7, 0x46, 0x2, 0x2, 0x1b3, 
        0x49, 0x3, 0x2, 0x2, 0x2, 0x1b4, 0x1b5, 0x7, 0x2a, 0x2, 0x2, 0x1b5, 
        0x1b6, 0x5, 0x44, 0x23, 0x2, 0x1b6, 0x1ba, 0x7, 0x46, 0x2, 0x2, 0x1b7, 
        0x1b9, 0x5, 0x46, 0x24, 0x2, 0x1b8, 0x1b7, 0x3, 0x2, 0x2, 0x2, 0x1b9, 
        0x1bc, 0x3, 0x2, 0x2, 0x2, 0x1ba, 0x1b8, 0x3, 0x2, 0x2, 0x2, 0x1ba, 
        0x1bb, 0x3, 0x2, 0x2, 0x2, 0x1bb, 0x1bd, 0x3, 0x2, 0x2, 0x2, 0x1bc, 
        0x1ba, 0x3, 0x2, 0x2, 0x2, 0x1bd, 0x1be, 0x7, 0x2b, 0x2, 0x2, 0x1be, 
        0x1bf, 0x7, 0x46, 0x2, 0x2, 0x1bf, 0x4b, 0x3, 0x2, 0x2, 0x2, 0x1c0, 
        0x1c1, 0x7, 0x2d, 0x2, 0x2, 0x1c1, 0x1c5, 0x7, 0x46, 0x2, 0x2, 0x1c2, 
        0x1c4, 0x5, 0x46, 0x24, 0x2, 0x1c3, 0x1c2, 0x3, 0x2, 0x2, 0x2, 0x1c4, 
        0x1c7, 0x3, 0x2, 0x2, 0x2, 0x1c5, 0x1c3, 0x3, 0x2, 0x2, 0x2, 0x1c5, 
        0x1c6, 0x3, 0x2, 0x2, 0x2, 0x1c6, 0x4d, 0x3, 0x2, 0x2, 0x2, 0x1c7, 0x1c5, 
        0x3, 0x2, 0x2, 0x2, 0x1c8, 0x1c9, 0x7, 0x2c, 0x2, 0x2, 0x1c9, 0x1ca, 
        0x5, 0x42, 0x22, 0x2, 0x1ca, 0x1ce, 0x7, 0x46, 0x2, 0x2, 0x1cb, 0x1cd, 
        0x5, 0x46, 0x24, 0x2, 0x1cc, 0x1cb, 0x3, 0x2, 0x2, 0x2, 0x1cd, 0x1d0, 
        0x3, 0x2, 0x2, 0x2, 0x1ce, 0x1cc, 0x3, 0x2, 0x2, 0x2, 0x1ce, 0x1cf, 
        0x3, 0x2, 0x2, 0x2, 0x1cf, 0x1d1, 0x3, 0x2, 0x2, 0x2, 0x1d0, 0x1ce, 
        0x3, 0x2, 0x2, 0x2, 0x1d1, 0x1d2, 0x5, 0x4c, 0x27, 0x2, 0x1d2, 0x1d3, 
        0x7, 0x2e, 0x2, 0x2, 0x1d3, 0x1d4, 0x7, 0x46, 0x2, 0x2, 0x1d4, 0x1e2, 
        0x3, 0x2, 0x2, 0x2, 0x1d5, 0x1d6, 0x7, 0x2c, 0x2, 0x2, 0x1d6, 0x1d7, 
        0x5, 0x42, 0x22, 0x2, 0x1d7, 0x1db, 0x7, 0x46, 0x2, 0x2, 0x1d8, 0x1da, 
        0x5, 0x46, 0x24, 0x2, 0x1d9, 0x1d8, 0x3, 0x2, 0x2, 0x2, 0x1da, 0x1dd, 
        0x3, 0x2, 0x2, 0x2, 0x1db, 0x1d9, 0x3, 0x2, 0x2, 0x2, 0x1db, 0x1dc, 
        0x3, 0x2, 0x2, 0x2, 0x1dc, 0x1de, 0x3, 0x2, 0x2, 0x2, 0x1dd, 0x1db, 
        0x3, 0x2, 0x2, 0x2, 0x1de, 0x1df, 0x7, 0x2e, 0x2, 0x2, 0x1df, 0x1e0, 
        0x7, 0x46, 0x2, 0x2, 0x1e0, 0x1e2, 0x3, 0x2, 0x2, 0x2, 0x1e1, 0x1c8, 
        0x3, 0x2, 0x2, 0x2, 0x1e1, 0x1d5, 0x3, 0x2, 0x2, 0x2, 0x1e2, 0x4f, 0x3, 
        0x2, 0x2, 0x2, 0x1e3, 0x1e4, 0x7, 0x2f, 0x2, 0x2, 0x1e4, 0x1e5, 0x5, 
        0x42, 0x22, 0x2, 0x1e5, 0x1e9, 0x7, 0x46, 0x2, 0x2, 0x1e6, 0x1e8, 0x5, 
        0x46, 0x24, 0x2, 0x1e7, 0x1e6, 0x3, 0x2, 0x2, 0x2, 0x1e8, 0x1eb, 0x3, 
        0x2, 0x2, 0x2, 0x1e9, 0x1e7, 0x3, 0x2, 0x2, 0x2, 0x1e9, 0x1ea, 0x3, 
        0x2, 0x2, 0x2, 0x1ea, 0x1ec, 0x3, 0x2, 0x2, 0x2, 0x1eb, 0x1e9, 0x3, 
        0x2, 0x2, 0x2, 0x1ec, 0x1ed, 0x7, 0x30, 0x2, 0x2, 0x1ed, 0x1ee, 0x7, 
        0x46, 0x2, 0x2, 0x1ee, 0x51, 0x3, 0x2, 0x2, 0x2, 0x1ef, 0x1f0, 0x7, 
        0x31, 0x2, 0x2, 0x1f0, 0x1f1, 0x5, 0xa, 0x6, 0x2, 0x1f1, 0x1f2, 0x7, 
        0x41, 0x2, 0x2, 0x1f2, 0x1f3, 0x5, 0x8, 0x5, 0x2, 0x1f3, 0x1f4, 0x7, 
        0x46, 0x2, 0x2, 0x1f4, 0x1fb, 0x3, 0x2, 0x2, 0x2, 0x1f5, 0x1f6, 0x7, 
        0x31, 0x2, 0x2, 0x1f6, 0x1f7, 0x7, 0x6, 0x2, 0x2, 0x1f7, 0x1f8, 0x7, 
        0x41, 0x2, 0x2, 0x1f8, 0x1f9, 0x7, 0x7, 0x2, 0x2, 0x1f9, 0x1fb, 0x7, 
        0x46, 0x2, 0x2, 0x1fa, 0x1ef, 0x3, 0x2, 0x2, 0x2, 0x1fa, 0x1f5, 0x3, 
        0x2, 0x2, 0x2, 0x1fb, 0x53, 0x3, 0x2, 0x2, 0x2, 0x1fc, 0x1fd, 0x7, 0x32, 
        0x2, 0x2, 0x1fd, 0x1fe, 0x5, 0xa, 0x6, 0x2, 0x1fe, 0x1ff, 0x7, 0x46, 
        0x2, 0x2, 0x1ff, 0x204, 0x3, 0x2, 0x2, 0x2, 0x200, 0x201, 0x7, 0x32, 
        0x2, 0x2, 0x201, 0x202, 0x7, 0x6, 0x2, 0x2, 0x202, 0x204, 0x7, 0x46, 
        0x2, 0x2, 0x203, 0x1fc, 0x3, 0x2, 0x2, 0x2, 0x203, 0x200, 0x3, 0x2, 
        0x2, 0x2, 0x204, 0x55, 0x3, 0x2, 0x2, 0x2, 0x205, 0x206, 0x7, 0x8, 0x2, 
        0x2, 0x206, 0x207, 0x5, 0x44, 0x23, 0x2, 0x207, 0x208, 0x7, 0x46, 0x2, 
        0x2, 0x208, 0x20d, 0x3, 0x2, 0x2, 0x2, 0x209, 0x20a, 0x7, 0x8, 0x2, 
        0x2, 0x20a, 0x20b, 0x7, 0x6, 0x2, 0x2, 0x20b, 0x20d, 0x7, 0x46, 0x2, 
        0x2, 0x20c, 0x205, 0x3, 0x2, 0x2, 0x2, 0x20c, 0x209, 0x3, 0x2, 0x2, 
        0x2, 0x20d, 0x57, 0x3, 0x2, 0x2, 0x2, 0x20e, 0x20f, 0x5, 0x42, 0x22, 
        0x2, 0x20f, 0x210, 0x7, 0x46, 0x2, 0x2, 0x210, 0x59, 0x3, 0x2, 0x2, 
        0x2, 0x211, 0x216, 0x5, 0x5c, 0x2f, 0x2, 0x212, 0x213, 0x7, 0x41, 0x2, 
        0x2, 0x213, 0x215, 0x5, 0x5c, 0x2f, 0x2, 0x214, 0x212, 0x3, 0x2, 0x2, 
        0x2, 0x215, 0x218, 0x3, 0x2, 0x2, 0x2, 0x216, 0x214, 0x3, 0x2, 0x2, 
        0x2, 0x216, 0x217, 0x3, 0x2, 0x2, 0x2, 0x217, 0x5b, 0x3, 0x2, 0x2, 0x2, 
        0x218, 0x216, 0x3, 0x2, 0x2, 0x2, 0x219, 0x21a, 0x8, 0x2f, 0x1, 0x2, 
        0x21a, 0x225, 0x5, 0x5e, 0x30, 0x2, 0x21b, 0x225, 0x7, 0x48, 0x2, 0x2, 
        0x21c, 0x225, 0x7, 0x49, 0x2, 0x2, 0x21d, 0x225, 0x7, 0x3, 0x2, 0x2, 
        0x21e, 0x21f, 0x7, 0x42, 0x2, 0x2, 0x21f, 0x220, 0x5, 0x5c, 0x2f, 0x2, 
        0x220, 0x221, 0x7, 0x43, 0x2, 0x2, 0x221, 0x225, 0x3, 0x2, 0x2, 0x2, 
        0x222, 0x223, 0x7, 0x3e, 0x2, 0x2, 0x223, 0x225, 0x5, 0x5c, 0x2f, 0x7, 
        0x224, 0x219, 0x3, 0x2, 0x2, 0x2, 0x224, 0x21b, 0x3, 0x2, 0x2, 0x2, 
        0x224, 0x21c, 0x3, 0x2, 0x2, 0x2, 0x224, 0x21d, 0x3, 0x2, 0x2, 0x2, 
        0x224, 0x21e, 0x3, 0x2, 0x2, 0x2, 0x224, 0x222, 0x3, 0x2, 0x2, 0x2, 
        0x225, 0x234, 0x3, 0x2, 0x2, 0x2, 0x226, 0x227, 0xc, 0x6, 0x2, 0x2, 
        0x227, 0x228, 0x7, 0x3f, 0x2, 0x2, 0x228, 0x233, 0x5, 0x5c, 0x2f, 0x7, 
        0x229, 0x22a, 0xc, 0x5, 0x2, 0x2, 0x22a, 0x22b, 0x7, 0x40, 0x2, 0x2, 
        0x22b, 0x233, 0x5, 0x5c, 0x2f, 0x6, 0x22c, 0x22d, 0xc, 0x4, 0x2, 0x2, 
        0x22d, 0x22e, 0x7, 0x3d, 0x2, 0x2, 0x22e, 0x233, 0x5, 0x5c, 0x2f, 0x5, 
        0x22f, 0x230, 0xc, 0x3, 0x2, 0x2, 0x230, 0x231, 0x7, 0x3e, 0x2, 0x2, 
        0x231, 0x233, 0x5, 0x5c, 0x2f, 0x4, 0x232, 0x226, 0x3, 0x2, 0x2, 0x2, 
        0x232, 0x229, 0x3, 0x2, 0x2, 0x2, 0x232, 0x22c, 0x3, 0x2, 0x2, 0x2, 
        0x232, 0x22f, 0x3, 0x2, 0x2, 0x2, 0x233, 0x236, 0x3, 0x2, 0x2, 0x2, 
        0x234, 0x232, 0x3, 0x2, 0x2, 0x2, 0x234, 0x235, 0x3, 0x2, 0x2, 0x2, 
        0x235, 0x5d, 0x3, 0x2, 0x2, 0x2, 0x236, 0x234, 0x3, 0x2, 0x2, 0x2, 0x237, 
        0x238, 0x7, 0x47, 0x2, 0x2, 0x238, 0x5f, 0x3, 0x2, 0x2, 0x2, 0x239, 
        0x23e, 0x5, 0x5e, 0x30, 0x2, 0x23a, 0x23b, 0x7, 0x41, 0x2, 0x2, 0x23b, 
        0x23d, 0x5, 0x5e, 0x30, 0x2, 0x23c, 0x23a, 0x3, 0x2, 0x2, 0x2, 0x23d, 
        0x240, 0x3, 0x2, 0x2, 0x2, 0x23e, 0x23c, 0x3, 0x2, 0x2, 0x2, 0x23e, 
        0x23f, 0x3, 0x2, 0x2, 0x2, 0x23f, 0x61, 0x3, 0x2, 0x2, 0x2, 0x240, 0x23e, 
        0x3, 0x2, 0x2, 0x2, 0x241, 0x24b, 0x5, 0x1e, 0x10, 0x2, 0x242, 0x24b, 
        0x5, 0x20, 0x11, 0x2, 0x243, 0x24b, 0x5, 0x22, 0x12, 0x2, 0x244, 0x24b, 
        0x5, 0x24, 0x13, 0x2, 0x245, 0x24b, 0x5, 0x26, 0x14, 0x2, 0x246, 0x24b, 
        0x5, 0x28, 0x15, 0x2, 0x247, 0x24b, 0x5, 0x2a, 0x16, 0x2, 0x248, 0x24b, 
        0x5, 0x2c, 0x17, 0x2, 0x249, 0x24b, 0x5, 0x2e, 0x18, 0x2, 0x24a, 0x241, 
        0x3, 0x2, 0x2, 0x2, 0x24a, 0x242, 0x3, 0x2, 0x2, 0x2, 0x24a, 0x243, 
        0x3, 0x2, 0x2, 0x2, 0x24a, 0x244, 0x3, 0x2, 0x2, 0x2, 0x24a, 0x245, 
        0x3, 0x2, 0x2, 0x2, 0x24a, 0x246, 0x3, 0x2, 0x2, 0x2, 0x24a, 0x247, 
        0x3, 0x2, 0x2, 0x2, 0x24a, 0x248, 0x3, 0x2, 0x2, 0x2, 0x24a, 0x249, 
        0x3, 0x2, 0x2, 0x2, 0x24b, 0x63, 0x3, 0x2, 0x2, 0x2, 0x24c, 0x24d, 0x9, 
        0x7, 0x2, 0x2, 0x24d, 0x65, 0x3, 0x2, 0x2, 0x2, 0x2b, 0x69, 0x6f, 0x7a, 
        0x8c, 0x9c, 0xb0, 0xc8, 0xe4, 0x122, 0x12b, 0x136, 0x138, 0x144, 0x146, 
        0x158, 0x15a, 0x166, 0x168, 0x173, 0x17e, 0x186, 0x18f, 0x197, 0x19a, 
        0x1a7, 0x1ae, 0x1ba, 0x1c5, 0x1ce, 0x1db, 0x1e1, 0x1e9, 0x1fa, 0x203, 
        0x20c, 0x216, 0x224, 0x232, 0x234, 0x23e, 0x24a, 
      };

      atn::ATNDeserializer deserializer;
      _atn = deserializer.deserialize(_serializedATN);

      size_t count = _atn.getNumberOfDecisions();
      _decisionToDFA.reserve(count);
      for (size_t i = 0; i < count; i++) { 
        _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
      }
    }

    statementParser::Initializer statementParser::_init;
}
