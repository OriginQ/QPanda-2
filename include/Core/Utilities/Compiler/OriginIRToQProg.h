#ifndef  _ORIGINIR_TO_QPROG_H
#define  _ORIGINIR_TO_QPROG_H
#include "ThirdParty/antlr4/runtime/src/antlr4-runtime.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirBaseVisitor.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirLexer.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirParser.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirVisitor.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"

QPANDA_BEGIN


/**
* @brief  Convert OriginIR  To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR file path
* @param[in]  QuantumMachine*	quantum machine pointer
* @param[out]  QVec	qubit  pointer vector
* @param[out]  std::vector<ClassicalCondition>	classical register  vector
* @return     QProg    quantum program
*/
QProg convert_originir_to_qprog(std::string file_path, QuantumMachine *qm, QVec &qv, std::vector<ClassicalCondition> &cv);

/**
* @brief  Convert OriginIR  To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR file path
* @param[in]  QuantumMachine*	quantum machine pointer
* @return     QProg    quantum program
*/
QProg convert_originir_to_qprog(std::string file_path, QuantumMachine *qm);


/**
* @brief  Convert OriginIR String To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR String
* @param[in]  QuantumMachine*	quantum machine pointer
* @param[out]  QVec	qubit  pointer
* @param[out]  std::vector<ClassicalCondition>	classical register  vector
* @return     QProg    quantum program
*/
QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine *qm, QVec &qv, std::vector<ClassicalCondition> &cv);

/**
* @brief  Convert OriginIR String To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR String
* @param[in]  QuantumMachine*	quantum machine pointer
* @return     QProg    quantum program
*/
QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine *qm);


/**
* @brief  OriginIR Transform To  Quantum Program 
* @ingroup Utilities
* @param[in]  std::string		OriginIR file path
* @param[in]  QuantumMachine*	quantum machine pointer
* @param[out]  QVec	qubit  pointer
* @param[out]  std::vector<ClassicalCondition>	classical register  vector
* @return     QProg    quantum program
*/
QProg transformOriginIRToQProg(std::string filePath, QuantumMachine* qm, QVec &qv, std::vector<ClassicalCondition> &cv);


/**
* @brief  Quantum Program Builder
* @ingroup Utilities
*/
class QProgBuilder {
	QuantumMachine* m_machine;
	std::unordered_map<size_t, QProg> m_progid_set;
	size_t qid = 0;
	std::unordered_map<size_t, ClassicalCondition> m_exprid_set;
	size_t cid = 0;

	QVec &qs;
	std::vector<ClassicalCondition> &ccs;

public:
	QProgBuilder(QuantumMachine *qm, QVec &qv, std::vector<ClassicalCondition> &cv);
	QProg get_qprog();
	
	enum class GateType {
		H, T, S, X, Y, Z, X1, Y1, Z1, I,
		RX, RY, RZ, U1,
		U2,
		U3,
		U4,
		CNOT, CZ, ISWAP, SQISWAP, SWAP,

		ISWAPTHETA, CR,

		CU,

		TOFFOLI
	};

#define MACRO_GET_GATETYPE(name) if (gatename==#name){return GateType::name;}
	static GateType get_gatetype(std::string gatename) {
		MACRO_GET_GATETYPE(H);
		MACRO_GET_GATETYPE(T);
		MACRO_GET_GATETYPE(S);
		MACRO_GET_GATETYPE(X);
		MACRO_GET_GATETYPE(Y);
		MACRO_GET_GATETYPE(Z);
		MACRO_GET_GATETYPE(X1);
		MACRO_GET_GATETYPE(Y1);
		MACRO_GET_GATETYPE(Z1);
		MACRO_GET_GATETYPE(I);
		MACRO_GET_GATETYPE(RX);
		MACRO_GET_GATETYPE(RY);
		MACRO_GET_GATETYPE(RZ);
		MACRO_GET_GATETYPE(U1);
		MACRO_GET_GATETYPE(U2);
		MACRO_GET_GATETYPE(U3);
		MACRO_GET_GATETYPE(U4);
		MACRO_GET_GATETYPE(CNOT);
		MACRO_GET_GATETYPE(CZ);
		MACRO_GET_GATETYPE(ISWAP);
		MACRO_GET_GATETYPE(SQISWAP);
		MACRO_GET_GATETYPE(SWAP);
		MACRO_GET_GATETYPE(ISWAPTHETA);
		MACRO_GET_GATETYPE(CR);
		MACRO_GET_GATETYPE(CU);
		MACRO_GET_GATETYPE(TOFFOLI);

	}
	// add up a level on the prog stack.
	
	void alloc_qubit(int num);
	void alloc_cbit(int num);

	size_t add_prog();

	void insert_subprog(size_t progid_dst, size_t progid_src);
	size_t add_qgate(GateType type, std::vector<int> index, std::vector<double> parameters);
	size_t add_qgate_cc(GateType type, std::vector<size_t> exprid, std::vector<int> index, std::vector<double> parameters);
	size_t add_measure_literal(size_t qidx, size_t cidx);
	size_t add_measure_cc(size_t exprid, size_t cidx);
	size_t add_reset_literal(size_t qidx);
	size_t add_reset_cc(size_t exprid);
	size_t add_expr_stat(size_t exprid);

	size_t make_qif(size_t exprid, size_t progid);
	size_t make_qifelse(size_t exprid, size_t progid_true, size_t progid_false);
	size_t make_qwhile(size_t exprid, size_t progid);

	void delete_prog(size_t progid);

	// return the expression id
	size_t cc_init_id(size_t cidx);
	size_t cc_init_literal(double value);

	// binary
	size_t cc_op_cc(size_t exprid1, size_t exprid2, int op_type);
	size_t cc_op_literal(size_t exprid1, double literal2, int op_type);
	size_t literal_op_cc(double literal1, size_t exprid2, int op_type);

	// unary
	size_t op_cc(size_t exprid, int op_type);

	// dagger the prog
	void make_dagger(size_t progid);
	size_t make_dagger_new(size_t progid);

	// make the prog controlled
	void make_control(size_t progid, std::vector<int> idx);
	size_t make_control_new(size_t progid, std::vector<int> idx);

	// make the stack top controlled by ccidx
	// like CONTROL q[c[1]], q[c[2]], q[3]
	// expridx contains the idx of c[1] and c[2]
	// idx = {-1, -1, 3}, -1 represents it a cc, 3 represents it
	// const idx
	void make_control_cc(size_t progid, std::vector<size_t> expridx, std::vector<int> idx);
	size_t make_control_cc_new(size_t progid, std::vector<size_t> expridx, std::vector<int> idx);
};

/**
* @brief OriginIR  Visitor
* @ingroup Utilities
*/
class OriginIRVisitor : public originirBaseVisitor {

	QProgBuilder builder;
	int qinit = -1;
	int cinit = -1;

	enum class Error {
		None,
		BadArgument,
	};
	Error errcode = Error::None;

	struct ExprContext {
		// true if the expr is a constant
		bool isConstant;

		// isConstant = true, then value is valid.
		double value;

		// isConstant = false, then ccid is valid.
		size_t ccid;
	};

	struct GateContext {
		QProgBuilder::GateType gatetype;
	};

public:
	OriginIRVisitor(QuantumMachine* qm, QVec &qv, std::vector<ClassicalCondition> &cv)
		:builder(qm, qv, cv) { }

	enum OpType {
		UnaryPlus,
		UnaryMinus,
		UnaryNot,

		Plus,
		Minus,
		Mul,
		Div,

		LT,
		GT,
		LEQ,
		GEQ,

		EQ,
		NE,

		AND,
		OR,
		ASSIGN,
	};

	QProg get_qprog(size_t progid) {
		return builder.get_qprog();
	}

	antlrcpp::Any visitTranslationunit(originirParser::TranslationunitContext *ctx) {
		visit(ctx->children[0]);
		builder.alloc_qubit(qinit);
		builder.alloc_cbit(cinit);
		auto fullprog = builder.add_prog();
		for (int i = 1; i < ctx->children.size(); ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(fullprog, prog);
		}
		return fullprog;
	}

	antlrcpp::Any visitQinit_declaration(originirParser::Qinit_declarationContext *ctx) {
		std::string s = ctx->children[1]->getText();
		int num = atoi(s.c_str());
		qinit = num;
		return 0;
	}

	antlrcpp::Any visitCinit_declaration(originirParser::Cinit_declarationContext *ctx) {
		std::string s = ctx->children[1]->getText();
		int num = atoi(s.c_str());
		cinit = num;
		return 0;
	}

	antlrcpp::Any visitIndex(originirParser::IndexContext *ctx) {
		return visit(ctx->children[1]);
	}

	antlrcpp::Any visitC_KEY_declaration(originirParser::C_KEY_declarationContext *ctx) {
		ExprContext retcontext= visit(ctx->children[1]);
		retcontext.ccid = builder.cc_init_id((size_t)retcontext.value);
		retcontext.isConstant = false;
		return retcontext;
	}

	antlrcpp::Any visitSingle_gate_without_parameter_declaration(
		originirParser::Single_gate_without_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context = visit(ctx->children[1]);
		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, {});
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, {-1}, {});
		}
	}

	antlrcpp::Any visitSingle_gate_with_one_parameter_declaration(
		originirParser::Single_gate_with_one_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context = visit(ctx->children[1]);
		ExprContext angle = visit(ctx->children[4]);

		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, { angle.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, {-1}, { angle.value });
		}
	}

	antlrcpp::Any visitSingle_gate_with_two_parameter_declaration(
		originirParser::Single_gate_with_two_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context = visit(ctx->children[1]);
		ExprContext angle1 = visit(ctx->children[4]);
		ExprContext angle2 = visit(ctx->children[6]);

		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, { angle1.value, angle2.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, { -1 }, { angle1.value, angle2.value });
		}
	}

	antlrcpp::Any visitSingle_gate_with_three_parameter_declaration(
		originirParser::Single_gate_with_three_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context = visit(ctx->children[1]);
		ExprContext angle1 = visit(ctx->children[4]);
		ExprContext angle2 = visit(ctx->children[6]);
		ExprContext angle3 = visit(ctx->children[8]);

		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, { angle1.value, angle2.value, angle3.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, { -1 }, { angle1.value, angle2.value, angle3.value });
		}
	}

	antlrcpp::Any visitSingle_gate_with_four_parameter_declaration(
		originirParser::Single_gate_with_four_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context = visit(ctx->children[1]);
		ExprContext angle1 = visit(ctx->children[4]);
		ExprContext angle2 = visit(ctx->children[6]);
		ExprContext angle3 = visit(ctx->children[8]);
		ExprContext angle4 = visit(ctx->children[10]);

		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, 
				{ angle1.value, angle2.value, angle3.value, angle4.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, {-1}, 
				{ angle1.value, angle2.value, angle3.value, angle4.value });
		}
	}

	antlrcpp::Any visitDouble_gate_without_parameter_declaration(
		originirParser::Double_gate_without_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);

		// insert gate
		if (context1.isConstant&&context2.isConstant) {
			return builder.add_qgate(gatetype, { (int)context1.value, (int)context2.value }, {});
		}
		else if (context1.isConstant) {
			return builder.add_qgate_cc(gatetype, { context2.ccid }, { (int)context1.value, -1 }, { });
		}
		else if (context2.isConstant) {
			return builder.add_qgate_cc(gatetype, { context1.ccid }, { -1, (int)context2.value }, { });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context1.ccid, context2.ccid }, { -1, -1 }, { });
		}
	}

	antlrcpp::Any visitDouble_gate_with_one_parameter_declaration(
		originirParser::Double_gate_with_one_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);
		ExprContext angle = visit(ctx->children[6]);

		if (context1.isConstant&&context2.isConstant) {
			return builder.add_qgate(gatetype, { (int)context1.value, (int)context2.value }, { angle.value });
		}
		else if (context1.isConstant) {
			return builder.add_qgate_cc(gatetype, { context2.ccid }, { (int)context1.value, -1 }, { angle.value });
		}
		else if (context2.isConstant) {
			return builder.add_qgate_cc(gatetype, { context1.ccid }, { -1, (int)context2.value }, { angle.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context1.ccid, context2.ccid }, { -1, -1 }, { angle.value });
		}
	}

	antlrcpp::Any visitDouble_gate_with_four_parameter_declaration(
		originirParser::Double_gate_with_four_parameter_declarationContext *ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);
		ExprContext angle1 = visit(ctx->children[6]);
		ExprContext angle2 = visit(ctx->children[8]);
		ExprContext angle3 = visit(ctx->children[10]);
		ExprContext angle4 = visit(ctx->children[12]);

		if (context1.isConstant&&context2.isConstant) {
			return builder.add_qgate(gatetype, { (int)context1.value, (int)context2.value }, 
				{ angle1.value, angle2.value,angle3.value,angle4.value });
		}
		else if (context1.isConstant) {
			return builder.add_qgate_cc(gatetype, { context2.ccid }, { (int)context1.value, -1 }, 
				{ angle1.value, angle2.value,angle3.value,angle4.value });
		}
		else if (context2.isConstant) {
			return builder.add_qgate_cc(gatetype, { context1.ccid }, { -1, (int)context2.value }, 
				{ angle1.value, angle2.value,angle3.value,angle4.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context1.ccid, context2.ccid }, { -1, -1 }, 
				{ angle1.value, angle2.value,angle3.value,angle4.value });
		}
	}
	antlrcpp::Any visitTriple_gate_without_parameter_declaration(
		originirParser::Triple_gate_without_parameter_declarationContext *ctx){
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);
		ExprContext context3 = visit(ctx->children[5]);

		// insert gate
		if (context1.isConstant && context2.isConstant && context3.isConstant){
			return builder.add_qgate(gatetype, { (int)context1.value, (int)context2.value, (int)context3.value }, {});
		}
		else if (context1.isConstant&&context2.isConstant){
			return builder.add_qgate_cc(gatetype, { context3.ccid } ,{ (int)context1.value, (int)context2.value, -1}, {});
		}
		else if (context1.isConstant&&context3.isConstant){
			return builder.add_qgate_cc(gatetype, { context2.ccid }, { (int)context1.value, -1, (int)context3.value }, {});
		}
		else if (context2.isConstant&&context3.isConstant){
			return builder.add_qgate_cc(gatetype, { context1.ccid }, { -1, (int)context2.value, (int)context3.value }, {});
		}
		else if (context1.isConstant) {
			return builder.add_qgate_cc(gatetype, { context2.ccid, context3.ccid }, { (int)context1.value, -1, -1 }, { });
		}
		else if (context2.isConstant) {
			return builder.add_qgate_cc(gatetype, { context1.ccid, context3.ccid }, { -1, (int)context2.value, -1 }, { });
		}
		else if (context3.isConstant) {
			return builder.add_qgate_cc(gatetype, { context1.ccid, context2.ccid }, { -1, (int)context3.value, -1 }, { });
		}
		else{
			return builder.add_qgate_cc(gatetype, { context1.ccid, context2.ccid, context3.ccid }, { -1, -1, -1 }, { });
		}
	}

	antlrcpp::Any visitSingle_gate_without_parameter_type(
		originirParser::Single_gate_without_parameter_typeContext *ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_one_parameter_type(
		originirParser::Single_gate_with_one_parameter_typeContext *ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_two_parameter_type(
		originirParser::Single_gate_with_two_parameter_typeContext *ctx)  {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_three_parameter_type(
		originirParser::Single_gate_with_three_parameter_typeContext *ctx) override {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_four_parameter_type(
		originirParser::Single_gate_with_four_parameter_typeContext *ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitDouble_gate_without_parameter_type(
		originirParser::Double_gate_without_parameter_typeContext *ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitDouble_gate_with_one_parameter_type(
		originirParser::Double_gate_with_one_parameter_typeContext *ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitDouble_gate_with_four_parameter_type(
		originirParser::Double_gate_with_four_parameter_typeContext *ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitTriple_gate_without_parameter_type(
		originirParser::Triple_gate_without_parameter_typeContext *ctx)
	{
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitPri_ckey(originirParser::Pri_ckeyContext *ctx) {
		ExprContext retcontext = visit(ctx->children[0]);
		return retcontext;
	}

	antlrcpp::Any visitPri_cst(originirParser::Pri_cstContext *ctx) {
		ExprContext context;
		context.isConstant = true;
		context.value = atof(ctx->children[0]->getText().c_str());
		return context;
	}

	antlrcpp::Any visitPri_expr(originirParser::Pri_exprContext *ctx) {
		// expect an object of ExprContext type.
		return visit(ctx->children[0]);
	}

	antlrcpp::Any visitUnary_expression(
		originirParser::Unary_expressionContext *ctx) {
		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[0]->getText();
			ExprContext context = visit(ctx->children[1]);
			ExprContext retcontext;
			if (context.isConstant) {
				retcontext.isConstant = true;
				retcontext.value = context.value;
				if (s == "-")
					retcontext.value = -retcontext.value;
				else if (s == "!")
					retcontext.value = !retcontext.value;

			}
			else {
				retcontext.isConstant = false;
				if (s == "+") {
					retcontext.ccid = builder.op_cc(context.ccid, UnaryPlus);
				}
				else if (s == "-") {
					retcontext.ccid = builder.op_cc(context.ccid, UnaryMinus);
				}
				else if (s == "!") {
					retcontext.ccid = builder.op_cc(context.ccid, UnaryNot);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitMultiplicative_expression(
		originirParser::Multiplicative_expressionContext *ctx) {
		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[1]->getText();
			ExprContext context1 = visit(ctx->children[0]);
			ExprContext context2 = visit(ctx->children[2]);
			ExprContext retcontext;
			if (context1.isConstant && context2.isConstant) {
				retcontext.isConstant = true;
				if (s == "*")
					retcontext.value = context1.value * context2.value;
				else if (s == "/")
					retcontext.value = context1.value / context2.value;
			}
			else if (context1.isConstant) {
				retcontext.isConstant = false;
				if (s == "*") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, Mul);
				}
				else if (s == "/") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, Div);
				}
			}
			else if (context2.isConstant) {
				retcontext.isConstant = false;
				if (s == "*") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, Mul);
				}
				else if (s == "/") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, Div);
				}
			}
			else {
				retcontext.isConstant = false;
				if (s == "*") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, Mul);
				}
				else if (s == "/") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, Div);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitAddtive_expression(
		originirParser::Addtive_expressionContext *ctx) {

		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[1]->getText();
			ExprContext context1 = visit(ctx->children[0]);
			ExprContext context2 = visit(ctx->children[2]);
			ExprContext retcontext;
			if (context1.isConstant && context2.isConstant) {
				retcontext.isConstant = true;
				if (s == "+")
					retcontext.value = context1.value + context2.value;
				else if (s == "-")
					retcontext.value = context1.value - context2.value;
			}
			else if (context1.isConstant) {
				retcontext.isConstant = false;
				if (s == "+") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, Plus);
				}
				else if (s == "-") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, Minus);
				}
			}
			else if (context2.isConstant) {
				retcontext.isConstant = false;
				if (s == "+") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, Plus);
				}
				else if (s == "-") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, Minus);
				}
			}
			else {
				retcontext.isConstant = false;
				if (s == "+") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, Plus);
				}
				else if (s == "-") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, Minus);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitRelational_expression(
		originirParser::Relational_expressionContext *ctx) {
		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[1]->getText();
			ExprContext context1 = visit(ctx->children[0]);
			ExprContext context2 = visit(ctx->children[2]);
			ExprContext retcontext;
			if (context1.isConstant && context2.isConstant) {
				retcontext.isConstant = true;
				if (s == "<")
					retcontext.value = context1.value < context2.value;
				else if (s == ">")
					retcontext.value = context1.value > context2.value;
				else if (s == "<=")
					retcontext.value = context1.value <= context2.value;
				else if (s == ">=")
					retcontext.value = context1.value >= context2.value;
			}
			else if (context1.isConstant) {
				retcontext.isConstant = false;
				if (s == "<") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, LT);
				}
				else if (s == ">") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, GT);
				}
				else if (s == "<=") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, LEQ);
				}
				else if (s == ">=") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, GEQ);
				}
			}
			else if (context2.isConstant) {
				retcontext.isConstant = false;
				if (s == "<") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, LT);
				}
				else if (s == ">") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, GT);
				}
				else if (s == "<=") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, LEQ);
				}
				else if (s == ">=") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, GEQ);
				}
			}
			else {
				retcontext.isConstant = false;
				if (s == "<") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, LT);
				}
				else if (s == ">") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, GT);
				}
				else if (s == "<=") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, LEQ);
				}
				else if (s == ">=") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, GEQ);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitEquality_expression(
		originirParser::Equality_expressionContext *ctx) {

		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[1]->getText();
			ExprContext context1 = visit(ctx->children[0]);
			ExprContext context2 = visit(ctx->children[2]);
			ExprContext retcontext;
			if (context1.isConstant && context2.isConstant) {
				retcontext.isConstant = true;
				if (s == "==")
					retcontext.value = context1.value == context2.value;
				else if (s == "!=")
					retcontext.value = context1.value != context2.value;
			}
			else if (context1.isConstant) {
				retcontext.isConstant = false;
				if (s == "==") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, EQ);
				}
				else if (s == "!=") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, NE);
				}
			}
			else if (context2.isConstant) {
				retcontext.isConstant = false;
				if (s == "==") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, EQ);
				}
				else if (s == "!=") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, NE);
				}
			}
			else {
				retcontext.isConstant = false;
				if (s == "==") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, EQ);
				}
				else if (s == "!=") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, NE);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitLogical_and_expression(
		originirParser::Logical_and_expressionContext *ctx) {
		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[1]->getText();
			ExprContext context1 = visit(ctx->children[0]);
			ExprContext context2 = visit(ctx->children[2]);
			ExprContext retcontext;
			if (context1.isConstant && context2.isConstant) {
				retcontext.isConstant = true;
				if (s == "&&")
					retcontext.value = context1.value == context2.value;
			}
			else if (context1.isConstant) {
				retcontext.isConstant = false;
				if (s == "&&") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, AND);
				}
			}
			else if (context2.isConstant) {
				retcontext.isConstant = false;
				if (s == "&&") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, AND);
				}
			}
			else {
				retcontext.isConstant = false;
				if (s == "&&") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, AND);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitLogical_or_expression(
		originirParser::Logical_or_expressionContext *ctx) {
		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[1]->getText();
			ExprContext context1 = visit(ctx->children[0]);
			ExprContext context2 = visit(ctx->children[2]);
			ExprContext retcontext;
			if (context1.isConstant && context2.isConstant) {
				retcontext.isConstant = true;
				if (s == "||")
					retcontext.value = context1.value == context2.value;
			}
			else if (context1.isConstant) {
				retcontext.isConstant = false;
				if (s == "||") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, OR);
				}
			}
			else if (context2.isConstant) {
				retcontext.isConstant = false;
				if (s == "||") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, OR);
				}
			}
			else {
				retcontext.isConstant = false;
				if (s == "||") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, OR);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitAssignment_expression(
		originirParser::Assignment_expressionContext *ctx) {
		if (ctx->children.size() == 1) {
			return visit(ctx->children[0]);
		}
		else {
			std::string s = ctx->children[1]->getText();
			ExprContext context1 = visit(ctx->children[0]);
			ExprContext context2 = visit(ctx->children[2]);
			ExprContext retcontext;
			if (context1.isConstant && context2.isConstant) {
				retcontext.isConstant = true;
				if (s == "=")
					retcontext.value = context1.value == context2.value;
			}
			else if (context1.isConstant) {
				retcontext.isConstant = false;
				if (s == "=") {
					retcontext.ccid = builder.literal_op_cc(context1.value, context2.ccid, ASSIGN);
				}
			}
			else if (context2.isConstant) {
				retcontext.isConstant = false;
				if (s == "=") {
					retcontext.ccid = builder.cc_op_literal(context1.ccid, context2.value, ASSIGN);
				}
			}
			else {
				retcontext.isConstant = false;
				if (s == "=") {
					retcontext.ccid = builder.cc_op_cc(context1.ccid, context2.ccid, ASSIGN);
				}
			}
			return retcontext;
		}
	}

	antlrcpp::Any visitControlbit_list(originirParser::Controlbit_listContext *ctx) {
		int n_children = ctx->children.size();
		std::vector<ExprContext> qkeys;
		for (int i = 0; i < n_children; i += 2) {
			qkeys.push_back(visit(ctx->children[i]));
		}
		return qkeys;
	}

	antlrcpp::Any visitStatement(originirParser::StatementContext *ctx) {
		return visit(ctx->children[0]);
	}

	antlrcpp::Any visitDagger_statement(originirParser::Dagger_statementContext *ctx) {
		size_t progid = builder.add_prog();

		for (int i = 2; i < ctx->children.size() - 2; ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}

		builder.make_dagger(progid);
		return progid;
	}

	antlrcpp::Any visitControl_statement(originirParser::Control_statementContext *ctx) {
		size_t progid = builder.add_prog();
		std::vector<ExprContext> qkeys = visit(ctx->children[1]);

		for (int i = 3; i < ctx->children.size() - 2; ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}
		bool isAllConstant = true;
		std::vector<int> indices;
		std::vector<size_t> exprindices;
		for (auto qkey : qkeys) {
			isAllConstant &= qkey.isConstant;
			if (qkey.isConstant)
				indices.push_back((int)qkey.value);
			else {
				indices.push_back(-1);
				exprindices.push_back(qkey.ccid);
			}
		}
		if (isAllConstant)
			builder.make_control(progid, indices);
		else
			builder.make_control_cc(progid, exprindices, indices);

		return progid;
	}

	antlrcpp::Any visitQelse_statement_fragment(
		originirParser::Qelse_statement_fragmentContext *ctx) {
		size_t progid = builder.add_prog();
		for (int i = 2; i < ctx->children.size(); ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}
		return progid;
	}

	antlrcpp::Any visitQif_if(originirParser::Qif_ifContext *ctx) {
		ExprContext context = visit(ctx->children[1]);
		size_t ccid = context.ccid;
		if (context.isConstant) {
			ccid = builder.cc_init_literal(context.value);
		}
		size_t progid = builder.add_prog();
		for (int i = 3; i < ctx->children.size() - 3; ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}
		size_t falseprogid = visit(ctx->children[ctx->children.size() - 3]);
		return builder.make_qifelse(ccid, progid, falseprogid);
	}

	antlrcpp::Any visitQif_ifelse(originirParser::Qif_ifelseContext *ctx) {
		ExprContext context = visit(ctx->children[1]);
		size_t ccid = context.ccid;
		if (context.isConstant) {
			ccid = builder.cc_init_literal(context.value);
		}
		size_t progid = builder.add_prog();
		for (int i = 3; i < ctx->children.size() - 2; ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}
		return builder.make_qif(ccid, progid);
	}

	antlrcpp::Any visitQwhile_statement(
		originirParser::Qwhile_statementContext *ctx) {
		ExprContext context = visit(ctx->children[1]);
		size_t ccid = context.ccid;
		if (context.isConstant) {
			ccid = builder.cc_init_literal(context.value);
		}
		size_t progid = builder.add_prog();
		for (int i = 3; i < ctx->children.size() - 2; ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}
		return builder.make_qwhile(ccid, progid);
	}

	antlrcpp::Any visitMeasure_statement(
		originirParser::Measure_statementContext *ctx) {
		ExprContext qcontext = visit(ctx->children[1]);
		ExprContext ccontext = visit(ctx->children[3]);
		if (qcontext.isConstant)
			return builder.add_measure_literal(qcontext.value, ccontext.value);
		else
			return builder.add_measure_cc(qcontext.ccid, ccontext.value);
	}

	antlrcpp::Any visitReset_statement(
		originirParser::Reset_statementContext *ctx) override {
		ExprContext qcontext = visit(ctx->children[1]);
		if (qcontext.isConstant)
			return builder.add_reset_literal(qcontext.value);
		else
			return builder.add_reset_cc(qcontext.ccid);
	}

	antlrcpp::Any visitExpression_statement(
		originirParser::Expression_statementContext *ctx) {
		ExprContext retcontext = visit(ctx->children[0]);
		return builder.add_expr_stat(retcontext.ccid);
	}

};

QPANDA_END
#endif