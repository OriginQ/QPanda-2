
// Generated from .\originir.g4 by ANTLR 4.7.2

#pragma once


#include "antlr4-runtime.h"
#include "originirParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by originirParser.
 */
class  originirVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by originirParser.
   */
    virtual antlrcpp::Any visitTranslationunit(originirParser::TranslationunitContext *context) = 0;

    virtual antlrcpp::Any visitDeclaration(originirParser::DeclarationContext *context) = 0;

    virtual antlrcpp::Any visitQinit_declaration(originirParser::Qinit_declarationContext *context) = 0;

    virtual antlrcpp::Any visitCinit_declaration(originirParser::Cinit_declarationContext *context) = 0;

    virtual antlrcpp::Any visitQuantum_gate_declaration(originirParser::Quantum_gate_declarationContext *context) = 0;

    virtual antlrcpp::Any visitIndex(originirParser::IndexContext *context) = 0;

    virtual antlrcpp::Any visitC_KEY_declaration(originirParser::C_KEY_declarationContext *context) = 0;

    virtual antlrcpp::Any visitQ_KEY_declaration(originirParser::Q_KEY_declarationContext *context) = 0;

    virtual antlrcpp::Any visitSingle_gate_without_parameter_declaration(originirParser::Single_gate_without_parameter_declarationContext *context) = 0;

    virtual antlrcpp::Any visitSingle_gate_with_one_parameter_declaration(originirParser::Single_gate_with_one_parameter_declarationContext *context) = 0;

    virtual antlrcpp::Any visitSingle_gate_with_four_parameter_declaration(originirParser::Single_gate_with_four_parameter_declarationContext *context) = 0;

    virtual antlrcpp::Any visitDouble_gate_without_parameter_declaration(originirParser::Double_gate_without_parameter_declarationContext *context) = 0;

    virtual antlrcpp::Any visitDouble_gate_with_one_parameter_declaration(originirParser::Double_gate_with_one_parameter_declarationContext *context) = 0;

    virtual antlrcpp::Any visitDouble_gate_with_four_parameter_declaration(originirParser::Double_gate_with_four_parameter_declarationContext *context) = 0;

    virtual antlrcpp::Any visitSingle_gate_without_parameter_type(originirParser::Single_gate_without_parameter_typeContext *context) = 0;

    virtual antlrcpp::Any visitSingle_gate_with_one_parameter_type(originirParser::Single_gate_with_one_parameter_typeContext *context) = 0;

    virtual antlrcpp::Any visitSingle_gate_with_four_parameter_type(originirParser::Single_gate_with_four_parameter_typeContext *context) = 0;

    virtual antlrcpp::Any visitDouble_gate_without_parameter_type(originirParser::Double_gate_without_parameter_typeContext *context) = 0;

    virtual antlrcpp::Any visitDouble_gate_with_one_parameter_type(originirParser::Double_gate_with_one_parameter_typeContext *context) = 0;

    virtual antlrcpp::Any visitDouble_gate_with_four_parameter_type(originirParser::Double_gate_with_four_parameter_typeContext *context) = 0;

    virtual antlrcpp::Any visitPri_ckey(originirParser::Pri_ckeyContext *context) = 0;

    virtual antlrcpp::Any visitPri_cst(originirParser::Pri_cstContext *context) = 0;

    virtual antlrcpp::Any visitPri_expr(originirParser::Pri_exprContext *context) = 0;

    virtual antlrcpp::Any visitUnary_expression(originirParser::Unary_expressionContext *context) = 0;

    virtual antlrcpp::Any visitMultiplicative_expression(originirParser::Multiplicative_expressionContext *context) = 0;

    virtual antlrcpp::Any visitAddtive_expression(originirParser::Addtive_expressionContext *context) = 0;

    virtual antlrcpp::Any visitRelational_expression(originirParser::Relational_expressionContext *context) = 0;

    virtual antlrcpp::Any visitEquality_expression(originirParser::Equality_expressionContext *context) = 0;

    virtual antlrcpp::Any visitLogical_and_expression(originirParser::Logical_and_expressionContext *context) = 0;

    virtual antlrcpp::Any visitLogical_or_expression(originirParser::Logical_or_expressionContext *context) = 0;

    virtual antlrcpp::Any visitAssignment_expression(originirParser::Assignment_expressionContext *context) = 0;

    virtual antlrcpp::Any visitExpression(originirParser::ExpressionContext *context) = 0;

    virtual antlrcpp::Any visitControlbit_list(originirParser::Controlbit_listContext *context) = 0;

    virtual antlrcpp::Any visitStatement(originirParser::StatementContext *context) = 0;

    virtual antlrcpp::Any visitDagger_statement(originirParser::Dagger_statementContext *context) = 0;

    virtual antlrcpp::Any visitControl_statement(originirParser::Control_statementContext *context) = 0;

    virtual antlrcpp::Any visitQelse_statement_fragment(originirParser::Qelse_statement_fragmentContext *context) = 0;

    virtual antlrcpp::Any visitQif_if(originirParser::Qif_ifContext *context) = 0;

    virtual antlrcpp::Any visitQif_ifelse(originirParser::Qif_ifelseContext *context) = 0;

    virtual antlrcpp::Any visitQwhile_statement(originirParser::Qwhile_statementContext *context) = 0;

    virtual antlrcpp::Any visitMeasure_statement(originirParser::Measure_statementContext *context) = 0;

    virtual antlrcpp::Any visitExpression_statement(originirParser::Expression_statementContext *context) = 0;

    virtual antlrcpp::Any visitConstant(originirParser::ConstantContext *context) = 0;


};

