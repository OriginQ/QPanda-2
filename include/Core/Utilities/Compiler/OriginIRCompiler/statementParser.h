
// Generated from .\statement.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"

namespace statement {


class  statementParser : public antlr4::Parser {
public:
  enum {
    PI = 1, QINIT_KEY = 2, CREG_KEY = 3, Q_KEY = 4, C_KEY = 5, BARRIER_KEY = 6, 
    QGATE_KEY = 7, ENDQGATE_KEY = 8, ECHO_GATE = 9, H_GATE = 10, X_GATE = 11, 
    NOT_GATE = 12, T_GATE = 13, S_GATE = 14, Y_GATE = 15, Z_GATE = 16, X1_GATE = 17, 
    Y1_GATE = 18, Z1_GATE = 19, I_GATE = 20, U2_GATE = 21, RPHI_GATE = 22, 
    U3_GATE = 23, U4_GATE = 24, RX_GATE = 25, RY_GATE = 26, RZ_GATE = 27, 
    U1_GATE = 28, CNOT_GATE = 29, CZ_GATE = 30, CU_GATE = 31, ISWAP_GATE = 32, 
    SQISWAP_GATE = 33, SWAPZ1_GATE = 34, ISWAPTHETA_GATE = 35, CR_GATE = 36, 
    TOFFOLI_GATE = 37, DAGGER_KEY = 38, ENDDAGGER_KEY = 39, CONTROL_KEY = 40, 
    ENDCONTROL_KEY = 41, QIF_KEY = 42, ELSE_KEY = 43, ENDIF_KEY = 44, QWHILE_KEY = 45, 
    ENDQWHILE_KEY = 46, MEASURE_KEY = 47, RESET_KEY = 48, ASSIGN = 49, GT = 50, 
    LT = 51, NOT = 52, EQ = 53, LEQ = 54, GEQ = 55, NE = 56, AND = 57, OR = 58, 
    PLUS = 59, MINUS = 60, MUL = 61, DIV = 62, COMMA = 63, LPAREN = 64, 
    RPAREN = 65, LBRACK = 66, RBRACK = 67, NEWLINE = 68, Identifier_s = 69, 
    Integer_Literal_s = 70, Double_Literal_s = 71, Digit_Sequence_s = 72, 
    REALEXP_s = 73, WhiteSpace_s = 74, SingleLineComment_s = 75
  };

  enum {
    RuleTranslationunit_s = 0, RuleQuantum_gate_declaration_s = 1, RuleIndex_s = 2, 
    RuleC_KEY_declaration_s = 3, RuleQ_KEY_declaration_s = 4, RuleSingle_gate_without_parameter_declaration_s = 5, 
    RuleSingle_gate_with_one_parameter_declaration_s = 6, RuleSingle_gate_with_two_parameter_declaration_s = 7, 
    RuleSingle_gate_with_three_parameter_declaration_s = 8, RuleSingle_gate_with_four_parameter_declaration_s = 9, 
    RuleDouble_gate_without_parameter_declaration_s = 10, RuleDouble_gate_with_one_parameter_declaration_s = 11, 
    RuleDouble_gate_with_four_parameter_declaration_s = 12, RuleTriple_gate_without_parameter_declaration_s = 13, 
    RuleSingle_gate_without_parameter_type_s = 14, RuleSingle_gate_with_one_parameter_type_s = 15, 
    RuleSingle_gate_with_two_parameter_type_s = 16, RuleSingle_gate_with_three_parameter_type_s = 17, 
    RuleSingle_gate_with_four_parameter_type_s = 18, RuleDouble_gate_without_parameter_type_s = 19, 
    RuleDouble_gate_with_one_parameter_type_s = 20, RuleDouble_gate_with_four_parameter_type_s = 21, 
    RuleTriple_gate_without_parameter_type_s = 22, RulePrimary_expression_s = 23, 
    RuleUnary_expression_s = 24, RuleMultiplicative_expression_s = 25, RuleAddtive_expression_s = 26, 
    RuleRelational_expression_s = 27, RuleEquality_expression_s = 28, RuleLogical_and_expression_s = 29, 
    RuleLogical_or_expression_s = 30, RuleAssignment_expression_s = 31, 
    RuleExpression_s = 32, RuleControlbit_list_s = 33, RuleStatement_s = 34, 
    RuleDagger_statement_s = 35, RuleControl_statement_s = 36, RuleQelse_statement_fragment_s = 37, 
    RuleQif_statement_s = 38, RuleQwhile_statement_s = 39, RuleMeasure_statement_s = 40, 
    RuleReset_statement_s = 41, RuleBarrier_statement_s = 42, RuleExpression_statement_s = 43, 
    RuleExplist_s = 44, RuleExp_s = 45, RuleId_s = 46, RuleId_list_s = 47, 
    RuleGate_name_s = 48, RuleConstant_s = 49
  };

  statementParser(antlr4::TokenStream *input);
  ~statementParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class Translationunit_sContext;
  class Quantum_gate_declaration_sContext;
  class Index_sContext;
  class C_KEY_declaration_sContext;
  class Q_KEY_declaration_sContext;
  class Single_gate_without_parameter_declaration_sContext;
  class Single_gate_with_one_parameter_declaration_sContext;
  class Single_gate_with_two_parameter_declaration_sContext;
  class Single_gate_with_three_parameter_declaration_sContext;
  class Single_gate_with_four_parameter_declaration_sContext;
  class Double_gate_without_parameter_declaration_sContext;
  class Double_gate_with_one_parameter_declaration_sContext;
  class Double_gate_with_four_parameter_declaration_sContext;
  class Triple_gate_without_parameter_declaration_sContext;
  class Single_gate_without_parameter_type_sContext;
  class Single_gate_with_one_parameter_type_sContext;
  class Single_gate_with_two_parameter_type_sContext;
  class Single_gate_with_three_parameter_type_sContext;
  class Single_gate_with_four_parameter_type_sContext;
  class Double_gate_without_parameter_type_sContext;
  class Double_gate_with_one_parameter_type_sContext;
  class Double_gate_with_four_parameter_type_sContext;
  class Triple_gate_without_parameter_type_sContext;
  class Primary_expression_sContext;
  class Unary_expression_sContext;
  class Multiplicative_expression_sContext;
  class Addtive_expression_sContext;
  class Relational_expression_sContext;
  class Equality_expression_sContext;
  class Logical_and_expression_sContext;
  class Logical_or_expression_sContext;
  class Assignment_expression_sContext;
  class Expression_sContext;
  class Controlbit_list_sContext;
  class Statement_sContext;
  class Dagger_statement_sContext;
  class Control_statement_sContext;
  class Qelse_statement_fragment_sContext;
  class Qif_statement_sContext;
  class Qwhile_statement_sContext;
  class Measure_statement_sContext;
  class Reset_statement_sContext;
  class Barrier_statement_sContext;
  class Expression_statement_sContext;
  class Explist_sContext;
  class Exp_sContext;
  class Id_sContext;
  class Id_list_sContext;
  class Gate_name_sContext;
  class Constant_sContext; 

  class  Translationunit_sContext : public antlr4::ParserRuleContext {
  public:
    Translationunit_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    std::vector<Statement_sContext *> statement_s();
    Statement_sContext* statement_s(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Translationunit_sContext* translationunit_s();

  class  Quantum_gate_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Quantum_gate_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_without_parameter_declaration_sContext *single_gate_without_parameter_declaration_s();
    Single_gate_with_one_parameter_declaration_sContext *single_gate_with_one_parameter_declaration_s();
    Single_gate_with_two_parameter_declaration_sContext *single_gate_with_two_parameter_declaration_s();
    Single_gate_with_three_parameter_declaration_sContext *single_gate_with_three_parameter_declaration_s();
    Single_gate_with_four_parameter_declaration_sContext *single_gate_with_four_parameter_declaration_s();
    Double_gate_without_parameter_declaration_sContext *double_gate_without_parameter_declaration_s();
    Double_gate_with_one_parameter_declaration_sContext *double_gate_with_one_parameter_declaration_s();
    Double_gate_with_four_parameter_declaration_sContext *double_gate_with_four_parameter_declaration_s();
    Triple_gate_without_parameter_declaration_sContext *triple_gate_without_parameter_declaration_s();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Quantum_gate_declaration_sContext* quantum_gate_declaration_s();

  class  Index_sContext : public antlr4::ParserRuleContext {
  public:
    Index_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACK();
    Expression_sContext *expression_s();
    antlr4::tree::TerminalNode *RBRACK();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Index_sContext* index_s();

  class  C_KEY_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    C_KEY_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *C_KEY();
    Index_sContext *index_s();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  C_KEY_declaration_sContext* c_KEY_declaration_s();

  class  Q_KEY_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Q_KEY_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Q_KEY();
    Index_sContext *index_s();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Q_KEY_declaration_sContext* q_KEY_declaration_s();

  class  Single_gate_without_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_without_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_without_parameter_type_sContext *single_gate_without_parameter_type_s();
    Q_KEY_declaration_sContext *q_KEY_declaration_s();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_without_parameter_declaration_sContext* single_gate_without_parameter_declaration_s();

  class  Single_gate_with_one_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_one_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_one_parameter_type_sContext *single_gate_with_one_parameter_type_s();
    Q_KEY_declaration_sContext *q_KEY_declaration_s();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *LPAREN();
    Expression_sContext *expression_s();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_one_parameter_declaration_sContext* single_gate_with_one_parameter_declaration_s();

  class  Single_gate_with_two_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_two_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_two_parameter_type_sContext *single_gate_with_two_parameter_type_s();
    Q_KEY_declaration_sContext *q_KEY_declaration_s();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<Expression_sContext *> expression_s();
    Expression_sContext* expression_s(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_two_parameter_declaration_sContext* single_gate_with_two_parameter_declaration_s();

  class  Single_gate_with_three_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_three_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_three_parameter_type_sContext *single_gate_with_three_parameter_type_s();
    Q_KEY_declaration_sContext *q_KEY_declaration_s();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<Expression_sContext *> expression_s();
    Expression_sContext* expression_s(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_three_parameter_declaration_sContext* single_gate_with_three_parameter_declaration_s();

  class  Single_gate_with_four_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_four_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_four_parameter_type_sContext *single_gate_with_four_parameter_type_s();
    Q_KEY_declaration_sContext *q_KEY_declaration_s();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<Expression_sContext *> expression_s();
    Expression_sContext* expression_s(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_four_parameter_declaration_sContext* single_gate_with_four_parameter_declaration_s();

  class  Double_gate_without_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_without_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Double_gate_without_parameter_type_sContext *double_gate_without_parameter_type_s();
    std::vector<Q_KEY_declaration_sContext *> q_KEY_declaration_s();
    Q_KEY_declaration_sContext* q_KEY_declaration_s(size_t i);
    antlr4::tree::TerminalNode *COMMA();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_without_parameter_declaration_sContext* double_gate_without_parameter_declaration_s();

  class  Double_gate_with_one_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_one_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Double_gate_with_one_parameter_type_sContext *double_gate_with_one_parameter_type_s();
    std::vector<Q_KEY_declaration_sContext *> q_KEY_declaration_s();
    Q_KEY_declaration_sContext* q_KEY_declaration_s(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    Expression_sContext *expression_s();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_one_parameter_declaration_sContext* double_gate_with_one_parameter_declaration_s();

  class  Double_gate_with_four_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_four_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Double_gate_with_four_parameter_type_sContext *double_gate_with_four_parameter_type_s();
    std::vector<Q_KEY_declaration_sContext *> q_KEY_declaration_s();
    Q_KEY_declaration_sContext* q_KEY_declaration_s(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<Expression_sContext *> expression_s();
    Expression_sContext* expression_s(size_t i);
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_four_parameter_declaration_sContext* double_gate_with_four_parameter_declaration_s();

  class  Triple_gate_without_parameter_declaration_sContext : public antlr4::ParserRuleContext {
  public:
    Triple_gate_without_parameter_declaration_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Triple_gate_without_parameter_type_sContext *triple_gate_without_parameter_type_s();
    std::vector<Q_KEY_declaration_sContext *> q_KEY_declaration_s();
    Q_KEY_declaration_sContext* q_KEY_declaration_s(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Triple_gate_without_parameter_declaration_sContext* triple_gate_without_parameter_declaration_s();

  class  Single_gate_without_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_without_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *H_GATE();
    antlr4::tree::TerminalNode *T_GATE();
    antlr4::tree::TerminalNode *S_GATE();
    antlr4::tree::TerminalNode *X_GATE();
    antlr4::tree::TerminalNode *Y_GATE();
    antlr4::tree::TerminalNode *Z_GATE();
    antlr4::tree::TerminalNode *X1_GATE();
    antlr4::tree::TerminalNode *Y1_GATE();
    antlr4::tree::TerminalNode *Z1_GATE();
    antlr4::tree::TerminalNode *I_GATE();
    antlr4::tree::TerminalNode *ECHO_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_without_parameter_type_sContext* single_gate_without_parameter_type_s();

  class  Single_gate_with_one_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_one_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RX_GATE();
    antlr4::tree::TerminalNode *RY_GATE();
    antlr4::tree::TerminalNode *RZ_GATE();
    antlr4::tree::TerminalNode *U1_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_one_parameter_type_sContext* single_gate_with_one_parameter_type_s();

  class  Single_gate_with_two_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_two_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *U2_GATE();
    antlr4::tree::TerminalNode *RPHI_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_two_parameter_type_sContext* single_gate_with_two_parameter_type_s();

  class  Single_gate_with_three_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_three_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *U3_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_three_parameter_type_sContext* single_gate_with_three_parameter_type_s();

  class  Single_gate_with_four_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_four_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *U4_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_four_parameter_type_sContext* single_gate_with_four_parameter_type_s();

  class  Double_gate_without_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_without_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CNOT_GATE();
    antlr4::tree::TerminalNode *CZ_GATE();
    antlr4::tree::TerminalNode *ISWAP_GATE();
    antlr4::tree::TerminalNode *SQISWAP_GATE();
    antlr4::tree::TerminalNode *SWAPZ1_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_without_parameter_type_sContext* double_gate_without_parameter_type_s();

  class  Double_gate_with_one_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_one_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ISWAPTHETA_GATE();
    antlr4::tree::TerminalNode *CR_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_one_parameter_type_sContext* double_gate_with_one_parameter_type_s();

  class  Double_gate_with_four_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_four_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CU_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_four_parameter_type_sContext* double_gate_with_four_parameter_type_s();

  class  Triple_gate_without_parameter_type_sContext : public antlr4::ParserRuleContext {
  public:
    Triple_gate_without_parameter_type_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *TOFFOLI_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Triple_gate_without_parameter_type_sContext* triple_gate_without_parameter_type_s();

  class  Primary_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Primary_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Primary_expression_sContext() = default;
    void copyFrom(Primary_expression_sContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Pri_ckeyContext : public Primary_expression_sContext {
  public:
    Pri_ckeyContext(Primary_expression_sContext *ctx);

    C_KEY_declaration_sContext *c_KEY_declaration_s();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Pri_cstContext : public Primary_expression_sContext {
  public:
    Pri_cstContext(Primary_expression_sContext *ctx);

    Constant_sContext *constant_s();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Pri_exprContext : public Primary_expression_sContext {
  public:
    Pri_exprContext(Primary_expression_sContext *ctx);

    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    Expression_sContext *expression_s();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Primary_expression_sContext* primary_expression_s();

  class  Unary_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Unary_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Primary_expression_sContext *primary_expression_s();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *NOT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Unary_expression_sContext* unary_expression_s();

  class  Multiplicative_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Multiplicative_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Unary_expression_sContext *unary_expression_s();
    Multiplicative_expression_sContext *multiplicative_expression_s();
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Multiplicative_expression_sContext* multiplicative_expression_s();
  Multiplicative_expression_sContext* multiplicative_expression_s(int precedence);
  class  Addtive_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Addtive_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Multiplicative_expression_sContext *multiplicative_expression_s();
    Addtive_expression_sContext *addtive_expression_s();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Addtive_expression_sContext* addtive_expression_s();
  Addtive_expression_sContext* addtive_expression_s(int precedence);
  class  Relational_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Relational_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Addtive_expression_sContext *addtive_expression_s();
    Relational_expression_sContext *relational_expression_s();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *LEQ();
    antlr4::tree::TerminalNode *GEQ();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Relational_expression_sContext* relational_expression_s();
  Relational_expression_sContext* relational_expression_s(int precedence);
  class  Equality_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Equality_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Relational_expression_sContext *relational_expression_s();
    Equality_expression_sContext *equality_expression_s();
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *NE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Equality_expression_sContext* equality_expression_s();
  Equality_expression_sContext* equality_expression_s(int precedence);
  class  Logical_and_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Logical_and_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Equality_expression_sContext *equality_expression_s();
    Logical_and_expression_sContext *logical_and_expression_s();
    antlr4::tree::TerminalNode *AND();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Logical_and_expression_sContext* logical_and_expression_s();
  Logical_and_expression_sContext* logical_and_expression_s(int precedence);
  class  Logical_or_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Logical_or_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Logical_and_expression_sContext *logical_and_expression_s();
    Logical_or_expression_sContext *logical_or_expression_s();
    antlr4::tree::TerminalNode *OR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Logical_or_expression_sContext* logical_or_expression_s();
  Logical_or_expression_sContext* logical_or_expression_s(int precedence);
  class  Assignment_expression_sContext : public antlr4::ParserRuleContext {
  public:
    Assignment_expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Logical_or_expression_sContext *logical_or_expression_s();
    C_KEY_declaration_sContext *c_KEY_declaration_s();
    antlr4::tree::TerminalNode *ASSIGN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Assignment_expression_sContext* assignment_expression_s();

  class  Expression_sContext : public antlr4::ParserRuleContext {
  public:
    Expression_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Assignment_expression_sContext *assignment_expression_s();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Expression_sContext* expression_s();

  class  Controlbit_list_sContext : public antlr4::ParserRuleContext {
  public:
    Controlbit_list_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Q_KEY_declaration_sContext *> q_KEY_declaration_s();
    Q_KEY_declaration_sContext* q_KEY_declaration_s(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    std::vector<antlr4::tree::TerminalNode *> Identifier_s();
    antlr4::tree::TerminalNode* Identifier_s(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Controlbit_list_sContext* controlbit_list_s();

  class  Statement_sContext : public antlr4::ParserRuleContext {
  public:
    Statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Quantum_gate_declaration_sContext *quantum_gate_declaration_s();
    antlr4::tree::TerminalNode *NEWLINE();
    Control_statement_sContext *control_statement_s();
    Qif_statement_sContext *qif_statement_s();
    Qwhile_statement_sContext *qwhile_statement_s();
    Dagger_statement_sContext *dagger_statement_s();
    Measure_statement_sContext *measure_statement_s();
    Reset_statement_sContext *reset_statement_s();
    Expression_statement_sContext *expression_statement_s();
    Barrier_statement_sContext *barrier_statement_s();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Statement_sContext* statement_s();

  class  Dagger_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Dagger_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DAGGER_KEY();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDDAGGER_KEY();
    std::vector<Statement_sContext *> statement_s();
    Statement_sContext* statement_s(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Dagger_statement_sContext* dagger_statement_s();

  class  Control_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Control_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CONTROL_KEY();
    Controlbit_list_sContext *controlbit_list_s();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDCONTROL_KEY();
    std::vector<Statement_sContext *> statement_s();
    Statement_sContext* statement_s(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Control_statement_sContext* control_statement_s();

  class  Qelse_statement_fragment_sContext : public antlr4::ParserRuleContext {
  public:
    Qelse_statement_fragment_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ELSE_KEY();
    antlr4::tree::TerminalNode *NEWLINE();
    std::vector<Statement_sContext *> statement_s();
    Statement_sContext* statement_s(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Qelse_statement_fragment_sContext* qelse_statement_fragment_s();

  class  Qif_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Qif_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Qif_statement_sContext() = default;
    void copyFrom(Qif_statement_sContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Qif_ifContext : public Qif_statement_sContext {
  public:
    Qif_ifContext(Qif_statement_sContext *ctx);

    antlr4::tree::TerminalNode *QIF_KEY();
    Expression_sContext *expression_s();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    Qelse_statement_fragment_sContext *qelse_statement_fragment_s();
    antlr4::tree::TerminalNode *ENDIF_KEY();
    std::vector<Statement_sContext *> statement_s();
    Statement_sContext* statement_s(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Qif_ifelseContext : public Qif_statement_sContext {
  public:
    Qif_ifelseContext(Qif_statement_sContext *ctx);

    antlr4::tree::TerminalNode *QIF_KEY();
    Expression_sContext *expression_s();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDIF_KEY();
    std::vector<Statement_sContext *> statement_s();
    Statement_sContext* statement_s(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Qif_statement_sContext* qif_statement_s();

  class  Qwhile_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Qwhile_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *QWHILE_KEY();
    Expression_sContext *expression_s();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDQWHILE_KEY();
    std::vector<Statement_sContext *> statement_s();
    Statement_sContext* statement_s(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Qwhile_statement_sContext* qwhile_statement_s();

  class  Measure_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Measure_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MEASURE_KEY();
    Q_KEY_declaration_sContext *q_KEY_declaration_s();
    antlr4::tree::TerminalNode *COMMA();
    C_KEY_declaration_sContext *c_KEY_declaration_s();
    antlr4::tree::TerminalNode *NEWLINE();
    antlr4::tree::TerminalNode *Q_KEY();
    antlr4::tree::TerminalNode *C_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Measure_statement_sContext* measure_statement_s();

  class  Reset_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Reset_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RESET_KEY();
    Q_KEY_declaration_sContext *q_KEY_declaration_s();
    antlr4::tree::TerminalNode *NEWLINE();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Reset_statement_sContext* reset_statement_s();

  class  Barrier_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Barrier_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BARRIER_KEY();
    Controlbit_list_sContext *controlbit_list_s();
    antlr4::tree::TerminalNode *NEWLINE();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Barrier_statement_sContext* barrier_statement_s();

  class  Expression_statement_sContext : public antlr4::ParserRuleContext {
  public:
    Expression_statement_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Expression_sContext *expression_s();
    antlr4::tree::TerminalNode *NEWLINE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Expression_statement_sContext* expression_statement_s();

  class  Explist_sContext : public antlr4::ParserRuleContext {
  public:
    Explist_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Exp_sContext *> exp_s();
    Exp_sContext* exp_s(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Explist_sContext* explist_s();

  class  Exp_sContext : public antlr4::ParserRuleContext {
  public:
    Exp_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Id_sContext *id_s();
    antlr4::tree::TerminalNode *Integer_Literal_s();
    antlr4::tree::TerminalNode *Double_Literal_s();
    antlr4::tree::TerminalNode *PI();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<Exp_sContext *> exp_s();
    Exp_sContext* exp_s(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();
    antlr4::tree::TerminalNode *PLUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Exp_sContext* exp_s();
  Exp_sContext* exp_s(int precedence);
  class  Id_sContext : public antlr4::ParserRuleContext {
  public:
    Id_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier_s();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Id_sContext* id_s();

  class  Id_list_sContext : public antlr4::ParserRuleContext {
  public:
    Id_list_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Id_sContext *> id_s();
    Id_sContext* id_s(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Id_list_sContext* id_list_s();

  class  Gate_name_sContext : public antlr4::ParserRuleContext {
  public:
    Gate_name_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_without_parameter_type_sContext *single_gate_without_parameter_type_s();
    Single_gate_with_one_parameter_type_sContext *single_gate_with_one_parameter_type_s();
    Single_gate_with_two_parameter_type_sContext *single_gate_with_two_parameter_type_s();
    Single_gate_with_three_parameter_type_sContext *single_gate_with_three_parameter_type_s();
    Single_gate_with_four_parameter_type_sContext *single_gate_with_four_parameter_type_s();
    Double_gate_without_parameter_type_sContext *double_gate_without_parameter_type_s();
    Double_gate_with_one_parameter_type_sContext *double_gate_with_one_parameter_type_s();
    Double_gate_with_four_parameter_type_sContext *double_gate_with_four_parameter_type_s();
    Triple_gate_without_parameter_type_sContext *triple_gate_without_parameter_type_s();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Gate_name_sContext* gate_name_s();

  class  Constant_sContext : public antlr4::ParserRuleContext {
  public:
    Constant_sContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Integer_Literal_s();
    antlr4::tree::TerminalNode *Double_Literal_s();
    antlr4::tree::TerminalNode *PI();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Constant_sContext* constant_s();


  virtual bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;
  bool multiplicative_expression_sSempred(Multiplicative_expression_sContext *_localctx, size_t predicateIndex);
  bool addtive_expression_sSempred(Addtive_expression_sContext *_localctx, size_t predicateIndex);
  bool relational_expression_sSempred(Relational_expression_sContext *_localctx, size_t predicateIndex);
  bool equality_expression_sSempred(Equality_expression_sContext *_localctx, size_t predicateIndex);
  bool logical_and_expression_sSempred(Logical_and_expression_sContext *_localctx, size_t predicateIndex);
  bool logical_or_expression_sSempred(Logical_or_expression_sContext *_localctx, size_t predicateIndex);
  bool exp_sSempred(Exp_sContext *_localctx, size_t predicateIndex);

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

}


