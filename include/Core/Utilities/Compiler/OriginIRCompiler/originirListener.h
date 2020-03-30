
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "originirParser.h"


/**
 * This interface defines an abstract listener for a parse tree produced by originirParser.
 */
class  originirListener : public antlr4::tree::ParseTreeListener {
public:

  virtual void enterTranslationunit(originirParser::TranslationunitContext *ctx) = 0;
  virtual void exitTranslationunit(originirParser::TranslationunitContext *ctx) = 0;

  virtual void enterDeclaration(originirParser::DeclarationContext *ctx) = 0;
  virtual void exitDeclaration(originirParser::DeclarationContext *ctx) = 0;

  virtual void enterQinit_declaration(originirParser::Qinit_declarationContext *ctx) = 0;
  virtual void exitQinit_declaration(originirParser::Qinit_declarationContext *ctx) = 0;

  virtual void enterCinit_declaration(originirParser::Cinit_declarationContext *ctx) = 0;
  virtual void exitCinit_declaration(originirParser::Cinit_declarationContext *ctx) = 0;

  virtual void enterQuantum_gate_declaration(originirParser::Quantum_gate_declarationContext *ctx) = 0;
  virtual void exitQuantum_gate_declaration(originirParser::Quantum_gate_declarationContext *ctx) = 0;

  virtual void enterIndex(originirParser::IndexContext *ctx) = 0;
  virtual void exitIndex(originirParser::IndexContext *ctx) = 0;

  virtual void enterC_KEY_declaration(originirParser::C_KEY_declarationContext *ctx) = 0;
  virtual void exitC_KEY_declaration(originirParser::C_KEY_declarationContext *ctx) = 0;

  virtual void enterQ_KEY_declaration(originirParser::Q_KEY_declarationContext *ctx) = 0;
  virtual void exitQ_KEY_declaration(originirParser::Q_KEY_declarationContext *ctx) = 0;

  virtual void enterSingle_gate_without_parameter_declaration(originirParser::Single_gate_without_parameter_declarationContext *ctx) = 0;
  virtual void exitSingle_gate_without_parameter_declaration(originirParser::Single_gate_without_parameter_declarationContext *ctx) = 0;

  virtual void enterSingle_gate_with_one_parameter_declaration(originirParser::Single_gate_with_one_parameter_declarationContext *ctx) = 0;
  virtual void exitSingle_gate_with_one_parameter_declaration(originirParser::Single_gate_with_one_parameter_declarationContext *ctx) = 0;

  virtual void enterSingle_gate_with_two_parameter_declaration(originirParser::Single_gate_with_two_parameter_declarationContext *ctx) = 0;
  virtual void exitSingle_gate_with_two_parameter_declaration(originirParser::Single_gate_with_two_parameter_declarationContext *ctx) = 0;

  virtual void enterSingle_gate_with_three_parameter_declaration(originirParser::Single_gate_with_three_parameter_declarationContext *ctx) = 0;
  virtual void exitSingle_gate_with_three_parameter_declaration(originirParser::Single_gate_with_three_parameter_declarationContext *ctx) = 0;

  virtual void enterSingle_gate_with_four_parameter_declaration(originirParser::Single_gate_with_four_parameter_declarationContext *ctx) = 0;
  virtual void exitSingle_gate_with_four_parameter_declaration(originirParser::Single_gate_with_four_parameter_declarationContext *ctx) = 0;

  virtual void enterDouble_gate_without_parameter_declaration(originirParser::Double_gate_without_parameter_declarationContext *ctx) = 0;
  virtual void exitDouble_gate_without_parameter_declaration(originirParser::Double_gate_without_parameter_declarationContext *ctx) = 0;

  virtual void enterDouble_gate_with_one_parameter_declaration(originirParser::Double_gate_with_one_parameter_declarationContext *ctx) = 0;
  virtual void exitDouble_gate_with_one_parameter_declaration(originirParser::Double_gate_with_one_parameter_declarationContext *ctx) = 0;

  virtual void enterDouble_gate_with_four_parameter_declaration(originirParser::Double_gate_with_four_parameter_declarationContext *ctx) = 0;
  virtual void exitDouble_gate_with_four_parameter_declaration(originirParser::Double_gate_with_four_parameter_declarationContext *ctx) = 0;

  virtual void enterTriple_gate_without_parameter_declaration(originirParser::Triple_gate_without_parameter_declarationContext *ctx) = 0;
  virtual void exitTriple_gate_without_parameter_declaration(originirParser::Triple_gate_without_parameter_declarationContext *ctx) = 0;

  virtual void enterSingle_gate_without_parameter_type(originirParser::Single_gate_without_parameter_typeContext *ctx) = 0;
  virtual void exitSingle_gate_without_parameter_type(originirParser::Single_gate_without_parameter_typeContext *ctx) = 0;

  virtual void enterSingle_gate_with_one_parameter_type(originirParser::Single_gate_with_one_parameter_typeContext *ctx) = 0;
  virtual void exitSingle_gate_with_one_parameter_type(originirParser::Single_gate_with_one_parameter_typeContext *ctx) = 0;

  virtual void enterSingle_gate_with_two_parameter_type(originirParser::Single_gate_with_two_parameter_typeContext *ctx) = 0;
  virtual void exitSingle_gate_with_two_parameter_type(originirParser::Single_gate_with_two_parameter_typeContext *ctx) = 0;

  virtual void enterSingle_gate_with_three_parameter_type(originirParser::Single_gate_with_three_parameter_typeContext *ctx) = 0;
  virtual void exitSingle_gate_with_three_parameter_type(originirParser::Single_gate_with_three_parameter_typeContext *ctx) = 0;

  virtual void enterSingle_gate_with_four_parameter_type(originirParser::Single_gate_with_four_parameter_typeContext *ctx) = 0;
  virtual void exitSingle_gate_with_four_parameter_type(originirParser::Single_gate_with_four_parameter_typeContext *ctx) = 0;

  virtual void enterDouble_gate_without_parameter_type(originirParser::Double_gate_without_parameter_typeContext *ctx) = 0;
  virtual void exitDouble_gate_without_parameter_type(originirParser::Double_gate_without_parameter_typeContext *ctx) = 0;

  virtual void enterDouble_gate_with_one_parameter_type(originirParser::Double_gate_with_one_parameter_typeContext *ctx) = 0;
  virtual void exitDouble_gate_with_one_parameter_type(originirParser::Double_gate_with_one_parameter_typeContext *ctx) = 0;

  virtual void enterDouble_gate_with_four_parameter_type(originirParser::Double_gate_with_four_parameter_typeContext *ctx) = 0;
  virtual void exitDouble_gate_with_four_parameter_type(originirParser::Double_gate_with_four_parameter_typeContext *ctx) = 0;

  virtual void enterTriple_gate_without_parameter_type(originirParser::Triple_gate_without_parameter_typeContext *ctx) = 0;
  virtual void exitTriple_gate_without_parameter_type(originirParser::Triple_gate_without_parameter_typeContext *ctx) = 0;

  virtual void enterPri_ckey(originirParser::Pri_ckeyContext *ctx) = 0;
  virtual void exitPri_ckey(originirParser::Pri_ckeyContext *ctx) = 0;

  virtual void enterPri_cst(originirParser::Pri_cstContext *ctx) = 0;
  virtual void exitPri_cst(originirParser::Pri_cstContext *ctx) = 0;

  virtual void enterPri_expr(originirParser::Pri_exprContext *ctx) = 0;
  virtual void exitPri_expr(originirParser::Pri_exprContext *ctx) = 0;

  virtual void enterUnary_expression(originirParser::Unary_expressionContext *ctx) = 0;
  virtual void exitUnary_expression(originirParser::Unary_expressionContext *ctx) = 0;

  virtual void enterMultiplicative_expression(originirParser::Multiplicative_expressionContext *ctx) = 0;
  virtual void exitMultiplicative_expression(originirParser::Multiplicative_expressionContext *ctx) = 0;

  virtual void enterAddtive_expression(originirParser::Addtive_expressionContext *ctx) = 0;
  virtual void exitAddtive_expression(originirParser::Addtive_expressionContext *ctx) = 0;

  virtual void enterRelational_expression(originirParser::Relational_expressionContext *ctx) = 0;
  virtual void exitRelational_expression(originirParser::Relational_expressionContext *ctx) = 0;

  virtual void enterEquality_expression(originirParser::Equality_expressionContext *ctx) = 0;
  virtual void exitEquality_expression(originirParser::Equality_expressionContext *ctx) = 0;

  virtual void enterLogical_and_expression(originirParser::Logical_and_expressionContext *ctx) = 0;
  virtual void exitLogical_and_expression(originirParser::Logical_and_expressionContext *ctx) = 0;

  virtual void enterLogical_or_expression(originirParser::Logical_or_expressionContext *ctx) = 0;
  virtual void exitLogical_or_expression(originirParser::Logical_or_expressionContext *ctx) = 0;

  virtual void enterAssignment_expression(originirParser::Assignment_expressionContext *ctx) = 0;
  virtual void exitAssignment_expression(originirParser::Assignment_expressionContext *ctx) = 0;

  virtual void enterExpression(originirParser::ExpressionContext *ctx) = 0;
  virtual void exitExpression(originirParser::ExpressionContext *ctx) = 0;

  virtual void enterControlbit_list(originirParser::Controlbit_listContext *ctx) = 0;
  virtual void exitControlbit_list(originirParser::Controlbit_listContext *ctx) = 0;

  virtual void enterStatement(originirParser::StatementContext *ctx) = 0;
  virtual void exitStatement(originirParser::StatementContext *ctx) = 0;

  virtual void enterDagger_statement(originirParser::Dagger_statementContext *ctx) = 0;
  virtual void exitDagger_statement(originirParser::Dagger_statementContext *ctx) = 0;

  virtual void enterControl_statement(originirParser::Control_statementContext *ctx) = 0;
  virtual void exitControl_statement(originirParser::Control_statementContext *ctx) = 0;

  virtual void enterQelse_statement_fragment(originirParser::Qelse_statement_fragmentContext *ctx) = 0;
  virtual void exitQelse_statement_fragment(originirParser::Qelse_statement_fragmentContext *ctx) = 0;

  virtual void enterQif_if(originirParser::Qif_ifContext *ctx) = 0;
  virtual void exitQif_if(originirParser::Qif_ifContext *ctx) = 0;

  virtual void enterQif_ifelse(originirParser::Qif_ifelseContext *ctx) = 0;
  virtual void exitQif_ifelse(originirParser::Qif_ifelseContext *ctx) = 0;

  virtual void enterQwhile_statement(originirParser::Qwhile_statementContext *ctx) = 0;
  virtual void exitQwhile_statement(originirParser::Qwhile_statementContext *ctx) = 0;

  virtual void enterMeasure_statement(originirParser::Measure_statementContext *ctx) = 0;
  virtual void exitMeasure_statement(originirParser::Measure_statementContext *ctx) = 0;

  virtual void enterReset_statement(originirParser::Reset_statementContext *ctx) = 0;
  virtual void exitReset_statement(originirParser::Reset_statementContext *ctx) = 0;

  virtual void enterExpression_statement(originirParser::Expression_statementContext *ctx) = 0;
  virtual void exitExpression_statement(originirParser::Expression_statementContext *ctx) = 0;

  virtual void enterConstant(originirParser::ConstantContext *ctx) = 0;
  virtual void exitConstant(originirParser::ConstantContext *ctx) = 0;


};

