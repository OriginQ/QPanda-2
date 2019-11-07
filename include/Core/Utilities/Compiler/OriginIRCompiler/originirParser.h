
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"




class  originirParser : public antlr4::Parser {
public:
  enum {
    PI = 1, QINIT_KEY = 2, CREG_KEY = 3, Q_KEY = 4, C_KEY = 5, H_GATE = 6, 
    X_GATE = 7, NOT_GATE = 8, T_GATE = 9, S_GATE = 10, Y_GATE = 11, Z_GATE = 12, 
    X1_GATE = 13, Y1_GATE = 14, Z1_GATE = 15, I_GATE = 16, U2_GATE = 17, 
    U3_GATE = 18, U4_GATE = 19, RX_GATE = 20, RY_GATE = 21, RZ_GATE = 22, 
    U1_GATE = 23, CNOT_GATE = 24, CZ_GATE = 25, CU_GATE = 26, ISWAP_GATE = 27, 
    SQISWAP_GATE = 28, SWAPZ1_GATE = 29, ISWAPTHETA_GATE = 30, CR_GATE = 31, 
    TOFFOLI_GATE = 32, DAGGER_KEY = 33, ENDDAGGER_KEY = 34, CONTROL_KEY = 35, 
    ENDCONTROL_KEY = 36, QIF_KEY = 37, ELSE_KEY = 38, ENDIF_KEY = 39, QWHILE_KEY = 40, 
    ENDQWHILE_KEY = 41, MEASURE_KEY = 42, PMEASURE_KEY = 43, ASSIGN = 44, 
    GT = 45, LT = 46, NOT = 47, EQ = 48, LEQ = 49, GEQ = 50, NE = 51, AND = 52, 
    OR = 53, PLUS = 54, MINUS = 55, MUL = 56, DIV = 57, COMMA = 58, LPAREN = 59, 
    RPAREN = 60, LBRACK = 61, RBRACK = 62, NEWLINE = 63, Identifier = 64, 
    Integer_Literal = 65, Double_Literal = 66, Digit_Sequence = 67, WhiteSpace = 68, 
    SingleLineComment = 69
  };

  enum {
    RuleTranslationunit = 0, RuleDeclaration = 1, RuleQinit_declaration = 2, 
    RuleCinit_declaration = 3, RuleQuantum_gate_declaration = 4, RuleIndex = 5, 
    RuleC_KEY_declaration = 6, RuleQ_KEY_declaration = 7, RuleSingle_gate_without_parameter_declaration = 8, 
    RuleSingle_gate_with_one_parameter_declaration = 9, RuleSingle_gate_with_two_parameter_declaration = 10, 
    RuleSingle_gate_with_three_parameter_declaration = 11, RuleSingle_gate_with_four_parameter_declaration = 12, 
    RuleDouble_gate_without_parameter_declaration = 13, RuleDouble_gate_with_one_parameter_declaration = 14, 
    RuleDouble_gate_with_four_parameter_declaration = 15, RuleTriple_gate_without_parameter_declaration = 16, 
    RuleSingle_gate_without_parameter_type = 17, RuleSingle_gate_with_one_parameter_type = 18, 
    RuleSingle_gate_with_two_parameter_type = 19, RuleSingle_gate_with_three_parameter_type = 20, 
    RuleSingle_gate_with_four_parameter_type = 21, RuleDouble_gate_without_parameter_type = 22, 
    RuleDouble_gate_with_one_parameter_type = 23, RuleDouble_gate_with_four_parameter_type = 24, 
    RuleTriple_gate_without_parameter_type = 25, RulePrimary_expression = 26, 
    RuleUnary_expression = 27, RuleMultiplicative_expression = 28, RuleAddtive_expression = 29, 
    RuleRelational_expression = 30, RuleEquality_expression = 31, RuleLogical_and_expression = 32, 
    RuleLogical_or_expression = 33, RuleAssignment_expression = 34, RuleExpression = 35, 
    RuleControlbit_list = 36, RuleStatement = 37, RuleDagger_statement = 38, 
    RuleControl_statement = 39, RuleQelse_statement_fragment = 40, RuleQif_statement = 41, 
    RuleQwhile_statement = 42, RuleMeasure_statement = 43, RulePmeasure_statement = 44, 
    RuleExpression_statement = 45, RuleConstant = 46
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
  class Pmeasure_statementContext;
  class Expression_statementContext;
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
    Pmeasure_statementContext *pmeasure_statement();
    Expression_statementContext *expression_statement();

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

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Measure_statementContext* measure_statement();

  class  Pmeasure_statementContext : public antlr4::ParserRuleContext {
  public:
    Pmeasure_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *PMEASURE_KEY();
    std::vector<Q_KEY_declarationContext *> q_KEY_declaration();
    Q_KEY_declarationContext* q_KEY_declaration(size_t i);
    std::vector<antlr4::tree::TerminalNode *> COMMA();
    antlr4::tree::TerminalNode* COMMA(size_t i);

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Pmeasure_statementContext* pmeasure_statement();

  class  Expression_statementContext : public antlr4::ParserRuleContext {
  public:
    Expression_statementContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExpressionContext *expression();

    virtual void enterRule(antlr4::tree::ParseTreeListener *listener) override;
    virtual void exitRule(antlr4::tree::ParseTreeListener *listener) override;

    virtual antlrcpp::Any accept(antlr4::tree::ParseTreeVisitor *visitor) override;
   
  };

  Expression_statementContext* expression_statement();

  class  ConstantContext : public antlr4::ParserRuleContext {
  public:
    ConstantContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Integer_Literal();
    antlr4::tree::TerminalNode *Double_Literal();

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

