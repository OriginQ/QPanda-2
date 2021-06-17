
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  originirParser : public antlr4::Parser {
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
    RPAREN = 65, LBRACK = 66, RBRACK = 67, NEWLINE = 68, Identifier = 69, 
    Integer_Literal = 70, Double_Literal = 71, Digit_Sequence = 72, REALEXP = 73, 
    WhiteSpace = 74, SingleLineComment = 75
  };

  enum {
    RuleTranslationunit = 0, RuleDeclaration = 1, RuleQinit_declaration = 2, 
    RuleCinit_declaration = 3, RuleQuantum_gate_declaration = 4, RuleIndex = 5, 
    RuleC_KEY_declaration = 6, RuleQ_KEY_declaration = 7, RuleSingle_gate_without_parameter_declaration = 8, 
    RuleSingle_gate_with_one_parameter_declaration = 9, RuleSingle_gate_with_two_parameter_declaration = 10, 
    RuleSingle_gate_with_three_parameter_declaration = 11, RuleSingle_gate_with_four_parameter_declaration = 12, 
    RuleDouble_gate_without_parameter_declaration = 13, RuleDouble_gate_with_one_parameter_declaration = 14, 
    RuleDouble_gate_with_four_parameter_declaration = 15, RuleTriple_gate_without_parameter_declaration = 16, 
    RuleDefine_gate_declaration = 17, RuleSingle_gate_without_parameter_type = 18, 
    RuleSingle_gate_with_one_parameter_type = 19, RuleSingle_gate_with_two_parameter_type = 20, 
    RuleSingle_gate_with_three_parameter_type = 21, RuleSingle_gate_with_four_parameter_type = 22, 
    RuleDouble_gate_without_parameter_type = 23, RuleDouble_gate_with_one_parameter_type = 24, 
    RuleDouble_gate_with_four_parameter_type = 25, RuleTriple_gate_without_parameter_type = 26, 
    RulePrimary_expression = 27, RuleUnary_expression = 28, RuleMultiplicative_expression = 29, 
    RuleAddtive_expression = 30, RuleRelational_expression = 31, RuleEquality_expression = 32, 
    RuleLogical_and_expression = 33, RuleLogical_or_expression = 34, RuleAssignment_expression = 35, 
    RuleExpression = 36, RuleControlbit_list = 37, RuleStatement = 38, RuleDagger_statement = 39, 
    RuleControl_statement = 40, RuleQelse_statement_fragment = 41, RuleQif_statement = 42, 
    RuleQwhile_statement = 43, RuleMeasure_statement = 44, RuleReset_statement = 45, 
    RuleBarrier_statement = 46, RuleExpression_statement = 47, RuleDefine_gate_statement = 48, 
    RuleExplist = 49, RuleExp = 50, RuleGate_func_statement = 51, RuleId = 52, 
    RuleId_list = 53, RuleGate_name = 54, RuleConstant = 55
  };

  originirParser(antlr4::TokenStream *input);
  ~originirParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class TranslationunitContext;
  class DeclarationContext;
  class Qinit_declarationContext;
  class Cinit_declarationContext;
  class Quantum_gate_declarationContext;
  class IndexContext;
  class C_KEY_declarationContext;
  class Q_KEY_declarationContext;
  class Single_gate_without_parameter_declarationContext;
  class Single_gate_with_one_parameter_declarationContext;
  class Single_gate_with_two_parameter_declarationContext;
  class Single_gate_with_three_parameter_declarationContext;
  class Single_gate_with_four_parameter_declarationContext;
  class Double_gate_without_parameter_declarationContext;
  class Double_gate_with_one_parameter_declarationContext;
  class Double_gate_with_four_parameter_declarationContext;
  class Triple_gate_without_parameter_declarationContext;
  class Define_gate_declarationContext;
  class Single_gate_without_parameter_typeContext;
  class Single_gate_with_one_parameter_typeContext;
  class Single_gate_with_two_parameter_typeContext;
  class Single_gate_with_three_parameter_typeContext;
  class Single_gate_with_four_parameter_typeContext;
  class Double_gate_without_parameter_typeContext;
  class Double_gate_with_one_parameter_typeContext;
  class Double_gate_with_four_parameter_typeContext;
  class Triple_gate_without_parameter_typeContext;
  class Primary_expressionContext;
  class Unary_expressionContext;
  class Multiplicative_expressionContext;
  class Addtive_expressionContext;
  class Relational_expressionContext;
  class Equality_expressionContext;
  class Logical_and_expressionContext;
  class Logical_or_expressionContext;
  class Assignment_expressionContext;
  class ExpressionContext;
  class Controlbit_listContext;
  class StatementContext;
  class Dagger_statementContext;
  class Control_statementContext;
  class Qelse_statement_fragmentContext;
  class Qif_statementContext;
  class Qwhile_statementContext;
  class Measure_statementContext;
  class Reset_statementContext;
  class Barrier_statementContext;
  class Expression_statementContext;
  class Define_gate_statementContext;
  class ExplistContext;
  class ExpContext;
  class Gate_func_statementContext;
  class IdContext;
  class Id_listContext;
  class Gate_nameContext;
  class ConstantContext; 

  class  TranslationunitContext : public antlr4::ParserRuleContext {
  public:
    TranslationunitContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    DeclarationContext *declaration();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  TranslationunitContext* translationunit();

  class  DeclarationContext : public antlr4::ParserRuleContext {
  public:
    DeclarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Qinit_declarationContext *qinit_declaration();
    Cinit_declarationContext *cinit_declaration();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  DeclarationContext* declaration();

  class  Qinit_declarationContext : public antlr4::ParserRuleContext {
  public:
    Qinit_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *QINIT_KEY();
    antlr4::tree::TerminalNode *Integer_Literal();
    antlr4::tree::TerminalNode *NEWLINE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Qinit_declarationContext* qinit_declaration();

  class  Cinit_declarationContext : public antlr4::ParserRuleContext {
  public:
    Cinit_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CREG_KEY();
    antlr4::tree::TerminalNode *Integer_Literal();
    antlr4::tree::TerminalNode *NEWLINE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Cinit_declarationContext* cinit_declaration();

  class  Quantum_gate_declarationContext : public antlr4::ParserRuleContext {
  public:
    Quantum_gate_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_without_parameter_declarationContext *single_gate_without_parameter_declaration();
    Single_gate_with_one_parameter_declarationContext *single_gate_with_one_parameter_declaration();
    Single_gate_with_two_parameter_declarationContext *single_gate_with_two_parameter_declaration();
    Single_gate_with_three_parameter_declarationContext *single_gate_with_three_parameter_declaration();
    Single_gate_with_four_parameter_declarationContext *single_gate_with_four_parameter_declaration();
    Double_gate_without_parameter_declarationContext *double_gate_without_parameter_declaration();
    Double_gate_with_one_parameter_declarationContext *double_gate_with_one_parameter_declaration();
    Double_gate_with_four_parameter_declarationContext *double_gate_with_four_parameter_declaration();
    Triple_gate_without_parameter_declarationContext *triple_gate_without_parameter_declaration();
    Define_gate_declarationContext *define_gate_declaration();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Quantum_gate_declarationContext* quantum_gate_declaration();

  class  IndexContext : public antlr4::ParserRuleContext {
  public:
    IndexContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *LBRACK();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RBRACK();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IndexContext* index();

  class  C_KEY_declarationContext : public antlr4::ParserRuleContext {
  public:
    C_KEY_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *C_KEY();
    IndexContext *index();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  C_KEY_declarationContext* c_KEY_declaration();

  class  Q_KEY_declarationContext : public antlr4::ParserRuleContext {
  public:
    Q_KEY_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Q_KEY();
    IndexContext *index();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Q_KEY_declarationContext* q_KEY_declaration();

  class  Single_gate_without_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_without_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_without_parameter_typeContext *single_gate_without_parameter_type();
    Q_KEY_declarationContext *q_KEY_declaration();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_without_parameter_declarationContext* single_gate_without_parameter_declaration();

  class  Single_gate_with_one_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_one_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_one_parameter_typeContext *single_gate_with_one_parameter_type();
    Q_KEY_declarationContext *q_KEY_declaration();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *LPAREN();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_one_parameter_declarationContext* single_gate_with_one_parameter_declaration();

  class  Single_gate_with_two_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_two_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_two_parameter_typeContext *single_gate_with_two_parameter_type();
    Q_KEY_declarationContext *q_KEY_declaration();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_two_parameter_declarationContext* single_gate_with_two_parameter_declaration();

  class  Single_gate_with_three_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_three_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_three_parameter_typeContext *single_gate_with_three_parameter_type();
    Q_KEY_declarationContext *q_KEY_declaration();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_three_parameter_declarationContext* single_gate_with_three_parameter_declaration();

  class  Single_gate_with_four_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_four_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_with_four_parameter_typeContext *single_gate_with_four_parameter_type();
    Q_KEY_declarationContext *q_KEY_declaration();
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_four_parameter_declarationContext* single_gate_with_four_parameter_declaration();

  class  Double_gate_without_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_without_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Double_gate_without_parameter_typeContext *double_gate_without_parameter_type();
    std::vector<Q_KEY_declarationContext *> q_KEY_declaration();
    Q_KEY_declarationContext* q_KEY_declaration(size_t i);
    antlr4::tree::TerminalNode *COMMA();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_without_parameter_declarationContext* double_gate_without_parameter_declaration();

  class  Double_gate_with_one_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_one_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Double_gate_with_one_parameter_typeContext *double_gate_with_one_parameter_type();
    std::vector<Q_KEY_declarationContext *> q_KEY_declaration();
    Q_KEY_declarationContext* q_KEY_declaration(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_one_parameter_declarationContext* double_gate_with_one_parameter_declaration();

  class  Double_gate_with_four_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_four_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Double_gate_with_four_parameter_typeContext *double_gate_with_four_parameter_type();
    std::vector<Q_KEY_declarationContext *> q_KEY_declaration();
    Q_KEY_declarationContext* q_KEY_declaration(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_four_parameter_declarationContext* double_gate_with_four_parameter_declaration();

  class  Triple_gate_without_parameter_declarationContext : public antlr4::ParserRuleContext {
  public:
    Triple_gate_without_parameter_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Triple_gate_without_parameter_typeContext *triple_gate_without_parameter_type();
    std::vector<Q_KEY_declarationContext *> q_KEY_declaration();
    Q_KEY_declarationContext* q_KEY_declaration(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Triple_gate_without_parameter_declarationContext* triple_gate_without_parameter_declaration();

  class  Define_gate_declarationContext : public antlr4::ParserRuleContext {
  public:
    Define_gate_declarationContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();
    std::vector<Q_KEY_declarationContext *> q_KEY_declaration();
    Q_KEY_declarationContext* q_KEY_declaration(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExpressionContext *> expression();
    ExpressionContext* expression(size_t i);
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Define_gate_declarationContext* define_gate_declaration();

  class  Single_gate_without_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_without_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
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

  Single_gate_without_parameter_typeContext* single_gate_without_parameter_type();

  class  Single_gate_with_one_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_one_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RX_GATE();
    antlr4::tree::TerminalNode *RY_GATE();
    antlr4::tree::TerminalNode *RZ_GATE();
    antlr4::tree::TerminalNode *U1_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_one_parameter_typeContext* single_gate_with_one_parameter_type();

  class  Single_gate_with_two_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_two_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *U2_GATE();
    antlr4::tree::TerminalNode *RPHI_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_two_parameter_typeContext* single_gate_with_two_parameter_type();

  class  Single_gate_with_three_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_three_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *U3_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_three_parameter_typeContext* single_gate_with_three_parameter_type();

  class  Single_gate_with_four_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Single_gate_with_four_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *U4_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Single_gate_with_four_parameter_typeContext* single_gate_with_four_parameter_type();

  class  Double_gate_without_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_without_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
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

  Double_gate_without_parameter_typeContext* double_gate_without_parameter_type();

  class  Double_gate_with_one_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_one_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ISWAPTHETA_GATE();
    antlr4::tree::TerminalNode *CR_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_one_parameter_typeContext* double_gate_with_one_parameter_type();

  class  Double_gate_with_four_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Double_gate_with_four_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CU_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Double_gate_with_four_parameter_typeContext* double_gate_with_four_parameter_type();

  class  Triple_gate_without_parameter_typeContext : public antlr4::ParserRuleContext {
  public:
    Triple_gate_without_parameter_typeContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *TOFFOLI_GATE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Triple_gate_without_parameter_typeContext* triple_gate_without_parameter_type();

  class  Primary_expressionContext : public antlr4::ParserRuleContext {
  public:
    Primary_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Primary_expressionContext() = default;
    void copyFrom(Primary_expressionContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Pri_ckeyContext : public Primary_expressionContext {
  public:
    Pri_ckeyContext(Primary_expressionContext *ctx);

    C_KEY_declarationContext *c_KEY_declaration();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Pri_cstContext : public Primary_expressionContext {
  public:
    Pri_cstContext(Primary_expressionContext *ctx);

    ConstantContext *constant();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Pri_exprContext : public Primary_expressionContext {
  public:
    Pri_exprContext(Primary_expressionContext *ctx);

    std::vector<antlr4::tree::TerminalNode *> LPAREN();
    antlr4::tree::TerminalNode* LPAREN(size_t i);
    ExpressionContext *expression();
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Primary_expressionContext* primary_expression();

  class  Unary_expressionContext : public antlr4::ParserRuleContext {
  public:
    Unary_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Primary_expressionContext *primary_expression();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *NOT();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Unary_expressionContext* unary_expression();

  class  Multiplicative_expressionContext : public antlr4::ParserRuleContext {
  public:
    Multiplicative_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Unary_expressionContext *unary_expression();
    Multiplicative_expressionContext *multiplicative_expression();
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Multiplicative_expressionContext* multiplicative_expression();
  Multiplicative_expressionContext* multiplicative_expression(int precedence);
  class  Addtive_expressionContext : public antlr4::ParserRuleContext {
  public:
    Addtive_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Multiplicative_expressionContext *multiplicative_expression();
    Addtive_expressionContext *addtive_expression();
    antlr4::tree::TerminalNode *PLUS();
    antlr4::tree::TerminalNode *MINUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Addtive_expressionContext* addtive_expression();
  Addtive_expressionContext* addtive_expression(int precedence);
  class  Relational_expressionContext : public antlr4::ParserRuleContext {
  public:
    Relational_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Addtive_expressionContext *addtive_expression();
    Relational_expressionContext *relational_expression();
    antlr4::tree::TerminalNode *LT();
    antlr4::tree::TerminalNode *GT();
    antlr4::tree::TerminalNode *LEQ();
    antlr4::tree::TerminalNode *GEQ();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Relational_expressionContext* relational_expression();
  Relational_expressionContext* relational_expression(int precedence);
  class  Equality_expressionContext : public antlr4::ParserRuleContext {
  public:
    Equality_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Relational_expressionContext *relational_expression();
    Equality_expressionContext *equality_expression();
    antlr4::tree::TerminalNode *EQ();
    antlr4::tree::TerminalNode *NE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Equality_expressionContext* equality_expression();
  Equality_expressionContext* equality_expression(int precedence);
  class  Logical_and_expressionContext : public antlr4::ParserRuleContext {
  public:
    Logical_and_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Equality_expressionContext *equality_expression();
    Logical_and_expressionContext *logical_and_expression();
    antlr4::tree::TerminalNode *AND();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Logical_and_expressionContext* logical_and_expression();
  Logical_and_expressionContext* logical_and_expression(int precedence);
  class  Logical_or_expressionContext : public antlr4::ParserRuleContext {
  public:
    Logical_or_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Logical_and_expressionContext *logical_and_expression();
    Logical_or_expressionContext *logical_or_expression();
    antlr4::tree::TerminalNode *OR();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Logical_or_expressionContext* logical_or_expression();
  Logical_or_expressionContext* logical_or_expression(int precedence);
  class  Assignment_expressionContext : public antlr4::ParserRuleContext {
  public:
    Assignment_expressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Logical_or_expressionContext *logical_or_expression();
    C_KEY_declarationContext *c_KEY_declaration();
    antlr4::tree::TerminalNode *ASSIGN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Assignment_expressionContext* assignment_expression();

  class  ExpressionContext : public antlr4::ParserRuleContext {
  public:
    ExpressionContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Assignment_expressionContext *assignment_expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpressionContext* expression();

  class  Controlbit_listContext : public antlr4::ParserRuleContext {
  public:
    Controlbit_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<Q_KEY_declarationContext *> q_KEY_declaration();
    Q_KEY_declarationContext* q_KEY_declaration(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Controlbit_listContext* controlbit_list();

  class  StatementContext : public antlr4::ParserRuleContext {
  public:
    StatementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Quantum_gate_declarationContext *quantum_gate_declaration();
    antlr4::tree::TerminalNode *NEWLINE();
    Control_statementContext *control_statement();
    Qif_statementContext *qif_statement();
    Qwhile_statementContext *qwhile_statement();
    Dagger_statementContext *dagger_statement();
    Measure_statementContext *measure_statement();
    Reset_statementContext *reset_statement();
    Expression_statementContext *expression_statement();
    Barrier_statementContext *barrier_statement();
    Gate_func_statementContext *gate_func_statement();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  StatementContext* statement();

  class  Dagger_statementContext : public antlr4::ParserRuleContext {
  public:
    Dagger_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *DAGGER_KEY();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDDAGGER_KEY();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Dagger_statementContext* dagger_statement();

  class  Control_statementContext : public antlr4::ParserRuleContext {
  public:
    Control_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *CONTROL_KEY();
    Controlbit_listContext *controlbit_list();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDCONTROL_KEY();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Control_statementContext* control_statement();

  class  Qelse_statement_fragmentContext : public antlr4::ParserRuleContext {
  public:
    Qelse_statement_fragmentContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *ELSE_KEY();
    antlr4::tree::TerminalNode *NEWLINE();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Qelse_statement_fragmentContext* qelse_statement_fragment();

  class  Qif_statementContext : public antlr4::ParserRuleContext {
  public:
    Qif_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
   
    Qif_statementContext() = default;
    void copyFrom(Qif_statementContext *context);
    using antlr4::ParserRuleContext::copyFrom;

    virtual size_t getRuleIndex() const override;

   
  };

  class  Qif_ifContext : public Qif_statementContext {
  public:
    Qif_ifContext(Qif_statementContext *ctx);

    antlr4::tree::TerminalNode *QIF_KEY();
    ExpressionContext *expression();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    Qelse_statement_fragmentContext *qelse_statement_fragment();
    antlr4::tree::TerminalNode *ENDIF_KEY();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  class  Qif_ifelseContext : public Qif_statementContext {
  public:
    Qif_ifelseContext(Qif_statementContext *ctx);

    antlr4::tree::TerminalNode *QIF_KEY();
    ExpressionContext *expression();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDIF_KEY();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);
    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
  };

  Qif_statementContext* qif_statement();

  class  Qwhile_statementContext : public antlr4::ParserRuleContext {
  public:
    Qwhile_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *QWHILE_KEY();
    ExpressionContext *expression();
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDQWHILE_KEY();
    std::vector<StatementContext *> statement();
    StatementContext* statement(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Qwhile_statementContext* qwhile_statement();

  class  Measure_statementContext : public antlr4::ParserRuleContext {
  public:
    Measure_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *MEASURE_KEY();
    Q_KEY_declarationContext *q_KEY_declaration();
    antlr4::tree::TerminalNode *COMMA();
    C_KEY_declarationContext *c_KEY_declaration();
    antlr4::tree::TerminalNode *NEWLINE();
    antlr4::tree::TerminalNode *Q_KEY();
    antlr4::tree::TerminalNode *C_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Measure_statementContext* measure_statement();

  class  Reset_statementContext : public antlr4::ParserRuleContext {
  public:
    Reset_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *RESET_KEY();
    Q_KEY_declarationContext *q_KEY_declaration();
    antlr4::tree::TerminalNode *NEWLINE();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Reset_statementContext* reset_statement();

  class  Barrier_statementContext : public antlr4::ParserRuleContext {
  public:
    Barrier_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *BARRIER_KEY();
    Controlbit_listContext *controlbit_list();
    antlr4::tree::TerminalNode *NEWLINE();
    antlr4::tree::TerminalNode *Q_KEY();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Barrier_statementContext* barrier_statement();

  class  Expression_statementContext : public antlr4::ParserRuleContext {
  public:
    Expression_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();
    antlr4::tree::TerminalNode *NEWLINE();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Expression_statementContext* expression_statement();

  class  Define_gate_statementContext : public antlr4::ParserRuleContext {
  public:
    Define_gate_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Gate_nameContext *gate_name();
    Id_listContext *id_list();
    antlr4::tree::TerminalNode *NEWLINE();
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *LPAREN();
    ExplistContext *explist();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Define_gate_statementContext* define_gate_statement();

  class  ExplistContext : public antlr4::ParserRuleContext {
  public:
    ExplistContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<ExpContext *> exp();
    ExpContext* exp(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExplistContext* explist();

  class  ExpContext : public antlr4::ParserRuleContext {
  public:
    ExpContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    IdContext *id();
    antlr4::tree::TerminalNode *Integer_Literal();
    antlr4::tree::TerminalNode *Double_Literal();
    antlr4::tree::TerminalNode *PI();
    antlr4::tree::TerminalNode *LPAREN();
    std::vector<ExpContext *> exp();
    ExpContext* exp(size_t i);
    antlr4::tree::TerminalNode *RPAREN();
    antlr4::tree::TerminalNode *MINUS();
    antlr4::tree::TerminalNode *MUL();
    antlr4::tree::TerminalNode *DIV();
    antlr4::tree::TerminalNode *PLUS();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ExpContext* exp();
  ExpContext* exp(int precedence);
  class  Gate_func_statementContext : public antlr4::ParserRuleContext {
  public:
    Gate_func_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *QGATE_KEY();
    IdContext *id();
    std::vector<Id_listContext *> id_list();
    Id_listContext* id_list(size_t i);
    std::vector<antlr4::tree::TerminalNode *> NEWLINE();
    antlr4::tree::TerminalNode* NEWLINE(size_t i);
    antlr4::tree::TerminalNode *ENDQGATE_KEY();
    std::vector<Define_gate_statementContext *> define_gate_statement();
    Define_gate_statementContext* define_gate_statement(size_t i);
    antlr4::tree::TerminalNode *COMMA();
    antlr4::tree::TerminalNode *LPAREN();
    antlr4::tree::TerminalNode *RPAREN();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Gate_func_statementContext* gate_func_statement();

  class  IdContext : public antlr4::ParserRuleContext {
  public:
    IdContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  IdContext* id();

  class  Id_listContext : public antlr4::ParserRuleContext {
  public:
    Id_listContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    std::vector<IdContext *> id();
    IdContext* id(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Id_listContext* id_list();

  class  Gate_nameContext : public antlr4::ParserRuleContext {
  public:
    Gate_nameContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    Single_gate_without_parameter_typeContext *single_gate_without_parameter_type();
    Single_gate_with_one_parameter_typeContext *single_gate_with_one_parameter_type();
    Single_gate_with_two_parameter_typeContext *single_gate_with_two_parameter_type();
    Single_gate_with_three_parameter_typeContext *single_gate_with_three_parameter_type();
    Single_gate_with_four_parameter_typeContext *single_gate_with_four_parameter_type();
    Double_gate_without_parameter_typeContext *double_gate_without_parameter_type();
    Double_gate_with_one_parameter_typeContext *double_gate_with_one_parameter_type();
    Double_gate_with_four_parameter_typeContext *double_gate_with_four_parameter_type();
    Triple_gate_without_parameter_typeContext *triple_gate_without_parameter_type();
    IdContext *id();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Gate_nameContext* gate_name();

  class  ConstantContext : public antlr4::ParserRuleContext {
  public:
    ConstantContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Integer_Literal();
    antlr4::tree::TerminalNode *Double_Literal();
    antlr4::tree::TerminalNode *PI();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  ConstantContext* constant();


  virtual bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;
  bool multiplicative_expressionSempred(Multiplicative_expressionContext *_localctx, size_t predicateIndex);
  bool addtive_expressionSempred(Addtive_expressionContext *_localctx, size_t predicateIndex);
  bool relational_expressionSempred(Relational_expressionContext *_localctx, size_t predicateIndex);
  bool equality_expressionSempred(Equality_expressionContext *_localctx, size_t predicateIndex);
  bool logical_and_expressionSempred(Logical_and_expressionContext *_localctx, size_t predicateIndex);
  bool logical_or_expressionSempred(Logical_or_expressionContext *_localctx, size_t predicateIndex);
  bool expSempred(ExpContext *_localctx, size_t predicateIndex);

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

