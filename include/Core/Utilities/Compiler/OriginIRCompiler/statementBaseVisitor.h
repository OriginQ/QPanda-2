
// Generated from .\statement.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "statementVisitor.h"


namespace statement {
    /**
     * This class provides an empty implementation of statementVisitor, which can be
     * extended to create a visitor which only needs to handle a subset of the available methods.
     */
    class  statementBaseVisitor : public statementVisitor {
    public:

        virtual antlrcpp::Any visitTranslationunit_s(statementParser::Translationunit_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitQuantum_gate_declaration_s(statementParser::Quantum_gate_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitIndex_s(statementParser::Index_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitC_KEY_declaration_s(statementParser::C_KEY_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitQ_KEY_declaration_s(statementParser::Q_KEY_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_without_parameter_declaration_s(statementParser::Single_gate_without_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_one_parameter_declaration_s(statementParser::Single_gate_with_one_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_two_parameter_declaration_s(statementParser::Single_gate_with_two_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_three_parameter_declaration_s(statementParser::Single_gate_with_three_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_four_parameter_declaration_s(statementParser::Single_gate_with_four_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitDouble_gate_without_parameter_declaration_s(statementParser::Double_gate_without_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitDouble_gate_with_one_parameter_declaration_s(statementParser::Double_gate_with_one_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitDouble_gate_with_four_parameter_declaration_s(statementParser::Double_gate_with_four_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitTriple_gate_without_parameter_declaration_s(statementParser::Triple_gate_without_parameter_declaration_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_without_parameter_type_s(statementParser::Single_gate_without_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_one_parameter_type_s(statementParser::Single_gate_with_one_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_two_parameter_type_s(statementParser::Single_gate_with_two_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_three_parameter_type_s(statementParser::Single_gate_with_three_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitSingle_gate_with_four_parameter_type_s(statementParser::Single_gate_with_four_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitDouble_gate_without_parameter_type_s(statementParser::Double_gate_without_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitDouble_gate_with_one_parameter_type_s(statementParser::Double_gate_with_one_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitDouble_gate_with_four_parameter_type_s(statementParser::Double_gate_with_four_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitTriple_gate_without_parameter_type_s(statementParser::Triple_gate_without_parameter_type_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitPri_ckey(statementParser::Pri_ckeyContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitPri_cst(statementParser::Pri_cstContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitPri_expr(statementParser::Pri_exprContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitUnary_expression_s(statementParser::Unary_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitMultiplicative_expression_s(statementParser::Multiplicative_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitAddtive_expression_s(statementParser::Addtive_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitRelational_expression_s(statementParser::Relational_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitEquality_expression_s(statementParser::Equality_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitLogical_and_expression_s(statementParser::Logical_and_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitLogical_or_expression_s(statementParser::Logical_or_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitAssignment_expression_s(statementParser::Assignment_expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitExpression_s(statementParser::Expression_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitControlbit_list_s(statementParser::Controlbit_list_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitStatement_s(statementParser::Statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitDagger_statement_s(statementParser::Dagger_statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitControl_statement_s(statementParser::Control_statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitQelse_statement_fragment_s(statementParser::Qelse_statement_fragment_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitQif_if(statementParser::Qif_ifContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitQif_ifelse(statementParser::Qif_ifelseContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitQwhile_statement_s(statementParser::Qwhile_statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitMeasure_statement_s(statementParser::Measure_statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitReset_statement_s(statementParser::Reset_statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitBarrier_statement_s(statementParser::Barrier_statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitExpression_statement_s(statementParser::Expression_statement_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitExplist_s(statementParser::Explist_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitExp_s(statementParser::Exp_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitId_s(statementParser::Id_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitId_list_s(statementParser::Id_list_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitGate_name_s(statementParser::Gate_name_sContext* ctx) override {
            return visitChildren(ctx);
        }

        virtual antlrcpp::Any visitConstant_s(statementParser::Constant_sContext* ctx) override {
            return visitChildren(ctx);
        }


    };
}
