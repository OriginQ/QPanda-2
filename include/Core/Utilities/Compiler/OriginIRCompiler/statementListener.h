
// Generated from .\statement.g4 by ANTLR 4.8

#pragma once


#include "antlr4-runtime.h"
#include "statementParser.h"

namespace statement {
	/**
	 * This interface defines an abstract listener for a parse tree produced by statementParser.
	 */
	class  statementListener : public antlr4::tree::ParseTreeListener {
	public:

		virtual void enterTranslationunit_s(statementParser::Translationunit_sContext* ctx) = 0;
		virtual void exitTranslationunit_s(statementParser::Translationunit_sContext* ctx) = 0;

		virtual void enterQuantum_gate_declaration_s(statementParser::Quantum_gate_declaration_sContext* ctx) = 0;
		virtual void exitQuantum_gate_declaration_s(statementParser::Quantum_gate_declaration_sContext* ctx) = 0;

		virtual void enterIndex_s(statementParser::Index_sContext* ctx) = 0;
		virtual void exitIndex_s(statementParser::Index_sContext* ctx) = 0;

		virtual void enterC_KEY_declaration_s(statementParser::C_KEY_declaration_sContext* ctx) = 0;
		virtual void exitC_KEY_declaration_s(statementParser::C_KEY_declaration_sContext* ctx) = 0;

		virtual void enterQ_KEY_declaration_s(statementParser::Q_KEY_declaration_sContext* ctx) = 0;
		virtual void exitQ_KEY_declaration_s(statementParser::Q_KEY_declaration_sContext* ctx) = 0;

		virtual void enterSingle_gate_without_parameter_declaration_s(statementParser::Single_gate_without_parameter_declaration_sContext* ctx) = 0;
		virtual void exitSingle_gate_without_parameter_declaration_s(statementParser::Single_gate_without_parameter_declaration_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_one_parameter_declaration_s(statementParser::Single_gate_with_one_parameter_declaration_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_one_parameter_declaration_s(statementParser::Single_gate_with_one_parameter_declaration_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_two_parameter_declaration_s(statementParser::Single_gate_with_two_parameter_declaration_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_two_parameter_declaration_s(statementParser::Single_gate_with_two_parameter_declaration_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_three_parameter_declaration_s(statementParser::Single_gate_with_three_parameter_declaration_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_three_parameter_declaration_s(statementParser::Single_gate_with_three_parameter_declaration_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_four_parameter_declaration_s(statementParser::Single_gate_with_four_parameter_declaration_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_four_parameter_declaration_s(statementParser::Single_gate_with_four_parameter_declaration_sContext* ctx) = 0;

		virtual void enterDouble_gate_without_parameter_declaration_s(statementParser::Double_gate_without_parameter_declaration_sContext* ctx) = 0;
		virtual void exitDouble_gate_without_parameter_declaration_s(statementParser::Double_gate_without_parameter_declaration_sContext* ctx) = 0;

		virtual void enterDouble_gate_with_one_parameter_declaration_s(statementParser::Double_gate_with_one_parameter_declaration_sContext* ctx) = 0;
		virtual void exitDouble_gate_with_one_parameter_declaration_s(statementParser::Double_gate_with_one_parameter_declaration_sContext* ctx) = 0;

		virtual void enterDouble_gate_with_four_parameter_declaration_s(statementParser::Double_gate_with_four_parameter_declaration_sContext* ctx) = 0;
		virtual void exitDouble_gate_with_four_parameter_declaration_s(statementParser::Double_gate_with_four_parameter_declaration_sContext* ctx) = 0;

		virtual void enterTriple_gate_without_parameter_declaration_s(statementParser::Triple_gate_without_parameter_declaration_sContext* ctx) = 0;
		virtual void exitTriple_gate_without_parameter_declaration_s(statementParser::Triple_gate_without_parameter_declaration_sContext* ctx) = 0;

		virtual void enterSingle_gate_without_parameter_type_s(statementParser::Single_gate_without_parameter_type_sContext* ctx) = 0;
		virtual void exitSingle_gate_without_parameter_type_s(statementParser::Single_gate_without_parameter_type_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_one_parameter_type_s(statementParser::Single_gate_with_one_parameter_type_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_one_parameter_type_s(statementParser::Single_gate_with_one_parameter_type_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_two_parameter_type_s(statementParser::Single_gate_with_two_parameter_type_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_two_parameter_type_s(statementParser::Single_gate_with_two_parameter_type_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_three_parameter_type_s(statementParser::Single_gate_with_three_parameter_type_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_three_parameter_type_s(statementParser::Single_gate_with_three_parameter_type_sContext* ctx) = 0;

		virtual void enterSingle_gate_with_four_parameter_type_s(statementParser::Single_gate_with_four_parameter_type_sContext* ctx) = 0;
		virtual void exitSingle_gate_with_four_parameter_type_s(statementParser::Single_gate_with_four_parameter_type_sContext* ctx) = 0;

		virtual void enterDouble_gate_without_parameter_type_s(statementParser::Double_gate_without_parameter_type_sContext* ctx) = 0;
		virtual void exitDouble_gate_without_parameter_type_s(statementParser::Double_gate_without_parameter_type_sContext* ctx) = 0;

		virtual void enterDouble_gate_with_one_parameter_type_s(statementParser::Double_gate_with_one_parameter_type_sContext* ctx) = 0;
		virtual void exitDouble_gate_with_one_parameter_type_s(statementParser::Double_gate_with_one_parameter_type_sContext* ctx) = 0;

		virtual void enterDouble_gate_with_four_parameter_type_s(statementParser::Double_gate_with_four_parameter_type_sContext* ctx) = 0;
		virtual void exitDouble_gate_with_four_parameter_type_s(statementParser::Double_gate_with_four_parameter_type_sContext* ctx) = 0;

		virtual void enterTriple_gate_without_parameter_type_s(statementParser::Triple_gate_without_parameter_type_sContext* ctx) = 0;
		virtual void exitTriple_gate_without_parameter_type_s(statementParser::Triple_gate_without_parameter_type_sContext* ctx) = 0;

		virtual void enterPri_ckey(statementParser::Pri_ckeyContext* ctx) = 0;
		virtual void exitPri_ckey(statementParser::Pri_ckeyContext* ctx) = 0;

		virtual void enterPri_cst(statementParser::Pri_cstContext* ctx) = 0;
		virtual void exitPri_cst(statementParser::Pri_cstContext* ctx) = 0;

		virtual void enterPri_expr(statementParser::Pri_exprContext* ctx) = 0;
		virtual void exitPri_expr(statementParser::Pri_exprContext* ctx) = 0;

		virtual void enterUnary_expression_s(statementParser::Unary_expression_sContext* ctx) = 0;
		virtual void exitUnary_expression_s(statementParser::Unary_expression_sContext* ctx) = 0;

		virtual void enterMultiplicative_expression_s(statementParser::Multiplicative_expression_sContext* ctx) = 0;
		virtual void exitMultiplicative_expression_s(statementParser::Multiplicative_expression_sContext* ctx) = 0;

		virtual void enterAddtive_expression_s(statementParser::Addtive_expression_sContext* ctx) = 0;
		virtual void exitAddtive_expression_s(statementParser::Addtive_expression_sContext* ctx) = 0;

		virtual void enterRelational_expression_s(statementParser::Relational_expression_sContext* ctx) = 0;
		virtual void exitRelational_expression_s(statementParser::Relational_expression_sContext* ctx) = 0;

		virtual void enterEquality_expression_s(statementParser::Equality_expression_sContext* ctx) = 0;
		virtual void exitEquality_expression_s(statementParser::Equality_expression_sContext* ctx) = 0;

		virtual void enterLogical_and_expression_s(statementParser::Logical_and_expression_sContext* ctx) = 0;
		virtual void exitLogical_and_expression_s(statementParser::Logical_and_expression_sContext* ctx) = 0;

		virtual void enterLogical_or_expression_s(statementParser::Logical_or_expression_sContext* ctx) = 0;
		virtual void exitLogical_or_expression_s(statementParser::Logical_or_expression_sContext* ctx) = 0;

		virtual void enterAssignment_expression_s(statementParser::Assignment_expression_sContext* ctx) = 0;
		virtual void exitAssignment_expression_s(statementParser::Assignment_expression_sContext* ctx) = 0;

		virtual void enterExpression_s(statementParser::Expression_sContext* ctx) = 0;
		virtual void exitExpression_s(statementParser::Expression_sContext* ctx) = 0;

		virtual void enterControlbit_list_s(statementParser::Controlbit_list_sContext* ctx) = 0;
		virtual void exitControlbit_list_s(statementParser::Controlbit_list_sContext* ctx) = 0;

		virtual void enterStatement_s(statementParser::Statement_sContext* ctx) = 0;
		virtual void exitStatement_s(statementParser::Statement_sContext* ctx) = 0;

		virtual void enterDagger_statement_s(statementParser::Dagger_statement_sContext* ctx) = 0;
		virtual void exitDagger_statement_s(statementParser::Dagger_statement_sContext* ctx) = 0;

		virtual void enterControl_statement_s(statementParser::Control_statement_sContext* ctx) = 0;
		virtual void exitControl_statement_s(statementParser::Control_statement_sContext* ctx) = 0;

		virtual void enterQelse_statement_fragment_s(statementParser::Qelse_statement_fragment_sContext* ctx) = 0;
		virtual void exitQelse_statement_fragment_s(statementParser::Qelse_statement_fragment_sContext* ctx) = 0;

		virtual void enterQif_if(statementParser::Qif_ifContext* ctx) = 0;
		virtual void exitQif_if(statementParser::Qif_ifContext* ctx) = 0;

		virtual void enterQif_ifelse(statementParser::Qif_ifelseContext* ctx) = 0;
		virtual void exitQif_ifelse(statementParser::Qif_ifelseContext* ctx) = 0;

		virtual void enterQwhile_statement_s(statementParser::Qwhile_statement_sContext* ctx) = 0;
		virtual void exitQwhile_statement_s(statementParser::Qwhile_statement_sContext* ctx) = 0;

		virtual void enterMeasure_statement_s(statementParser::Measure_statement_sContext* ctx) = 0;
		virtual void exitMeasure_statement_s(statementParser::Measure_statement_sContext* ctx) = 0;

		virtual void enterReset_statement_s(statementParser::Reset_statement_sContext* ctx) = 0;
		virtual void exitReset_statement_s(statementParser::Reset_statement_sContext* ctx) = 0;

		virtual void enterBarrier_statement_s(statementParser::Barrier_statement_sContext* ctx) = 0;
		virtual void exitBarrier_statement_s(statementParser::Barrier_statement_sContext* ctx) = 0;

		virtual void enterExpression_statement_s(statementParser::Expression_statement_sContext* ctx) = 0;
		virtual void exitExpression_statement_s(statementParser::Expression_statement_sContext* ctx) = 0;

		virtual void enterExplist_s(statementParser::Explist_sContext* ctx) = 0;
		virtual void exitExplist_s(statementParser::Explist_sContext* ctx) = 0;

		virtual void enterExp_s(statementParser::Exp_sContext* ctx) = 0;
		virtual void exitExp_s(statementParser::Exp_sContext* ctx) = 0;

		virtual void enterId_s(statementParser::Id_sContext* ctx) = 0;
		virtual void exitId_s(statementParser::Id_sContext* ctx) = 0;

		virtual void enterId_list_s(statementParser::Id_list_sContext* ctx) = 0;
		virtual void exitId_list_s(statementParser::Id_list_sContext* ctx) = 0;

		virtual void enterGate_name_s(statementParser::Gate_name_sContext* ctx) = 0;
		virtual void exitGate_name_s(statementParser::Gate_name_sContext* ctx) = 0;

		virtual void enterConstant_s(statementParser::Constant_sContext* ctx) = 0;
		virtual void exitConstant_s(statementParser::Constant_sContext* ctx) = 0;


	};

}