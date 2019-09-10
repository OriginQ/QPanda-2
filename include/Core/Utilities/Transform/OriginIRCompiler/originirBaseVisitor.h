
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "originirVisitor.h"


/**
 * This class provides an empty implementation of originirVisitor, which can be
 * extended to create a visitor which only needs to handle a subset of the available methods.
 */
class  originirBaseVisitor : public originirVisitor {
public:

  virtual antlrcpp::Any visitTranslationunit(originirParser::TranslationunitContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDeclaration(originirParser::DeclarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQinit_declaration(originirParser::Qinit_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitCinit_declaration(originirParser::Cinit_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQuantum_gate_declaration(originirParser::Quantum_gate_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitIndex(originirParser::IndexContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitC_KEY_declaration(originirParser::C_KEY_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQ_KEY_declaration(originirParser::Q_KEY_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingle_gate_without_parameter_declaration(originirParser::Single_gate_without_parameter_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingle_gate_with_one_parameter_declaration(originirParser::Single_gate_with_one_parameter_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingle_gate_with_four_parameter_declaration(originirParser::Single_gate_with_four_parameter_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDouble_gate_without_parameter_declaration(originirParser::Double_gate_without_parameter_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDouble_gate_with_one_parameter_declaration(originirParser::Double_gate_with_one_parameter_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDouble_gate_with_four_parameter_declaration(originirParser::Double_gate_with_four_parameter_declarationContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingle_gate_without_parameter_type(originirParser::Single_gate_without_parameter_typeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingle_gate_with_one_parameter_type(originirParser::Single_gate_with_one_parameter_typeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitSingle_gate_with_four_parameter_type(originirParser::Single_gate_with_four_parameter_typeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDouble_gate_without_parameter_type(originirParser::Double_gate_without_parameter_typeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDouble_gate_with_one_parameter_type(originirParser::Double_gate_with_one_parameter_typeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDouble_gate_with_four_parameter_type(originirParser::Double_gate_with_four_parameter_typeContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPri_ckey(originirParser::Pri_ckeyContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPri_cst(originirParser::Pri_cstContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitPri_expr(originirParser::Pri_exprContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitUnary_expression(originirParser::Unary_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMultiplicative_expression(originirParser::Multiplicative_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAddtive_expression(originirParser::Addtive_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitRelational_expression(originirParser::Relational_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitEquality_expression(originirParser::Equality_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLogical_and_expression(originirParser::Logical_and_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitLogical_or_expression(originirParser::Logical_or_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitAssignment_expression(originirParser::Assignment_expressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression(originirParser::ExpressionContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitControlbit_list(originirParser::Controlbit_listContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitStatement(originirParser::StatementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitDagger_statement(originirParser::Dagger_statementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitControl_statement(originirParser::Control_statementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQelse_statement_fragment(originirParser::Qelse_statement_fragmentContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQif_if(originirParser::Qif_ifContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQif_ifelse(originirParser::Qif_ifelseContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitQwhile_statement(originirParser::Qwhile_statementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitMeasure_statement(originirParser::Measure_statementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitExpression_statement(originirParser::Expression_statementContext *ctx) override {
    return visitChildren(ctx);
  }

  virtual antlrcpp::Any visitConstant(originirParser::ConstantContext *ctx) override {
    return visitChildren(ctx);
  }


};

