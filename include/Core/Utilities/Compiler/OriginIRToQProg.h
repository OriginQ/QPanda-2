#ifndef  _ORIGINIR_TO_QPROG_H
#define  _ORIGINIR_TO_QPROG_H
#include "ThirdParty/antlr4/runtime/src/antlr4-runtime.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirBaseVisitor.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirLexer.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirParser.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirVisitor.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/originirListener.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/statementBaseVisitor.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/statementLexer.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/statementParser.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/statementVisitor.h"
#include "Core/Utilities/Compiler/OriginIRCompiler/statementListener.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/Utilities/Tools/Traversal.h"
#include "Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/ClassicalProgram.h"
#include "Core/QuantumCircuit/QGlobalVariable.h"
#include "Core/Utilities/Compiler/QASMToQProg.h"
#include "include/Core/Utilities/Tools/QString.h"
#include <Core/Utilities/Compiler/OriginIRCompiler/originirParser.h>
#include <Core/Utilities/Compiler/OriginIRCompiler/originirParser.h>
#include <include/Core/Utilities/Tools/Utils.h>
#include "Core/Utilities/QProgTransform/QProgToQCircuit.h"
#include "Core/Utilities/Tools/QPandaException.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <regex>


QPANDA_BEGIN

class StatementErrorListener : public antlr4::BaseErrorListener {
public:
	void syntaxError(antlr4::Recognizer* recognizer, antlr4::Token* offendingSymbol, size_t line,
		size_t charPositionInLine, const std::string& msg,
		std::exception_ptr e) override {
		std::ostringstream output;
		output << "Invalid UserDefinedGate source ---> parameter error";
		output << ":" << charPositionInLine << " " << msg;
		QCERR_AND_THROW(run_fail, output.str());
	}
};

std::vector<std::string> all_supported_gate =
{
	"H","T","S","X","Y","Z","X1","Y1","Z1","I","ECHO",
	"RX","RY","RZ","U1",
	"U2","RPhi",
	"U3",
	"U4",
	"CNOT","CZ","ISWAP","SQISWAP","SWAP",
	"CR","ISWAPTHETA",
	"CU",
	"TOFFOLI"
};

std::regex id("[a-zA-Z_][a-zA-Z0-9_]*");

/**
* @brief  repalce A with X in define_gate_declaration statement
* @ingroup Utilities
* @param[in]  std::string& src_str  source string
* @param[in]  std::string strA:  string to be replaced
* @param[in]  std::string strX:  string to replace
* @return   std::string: the replaced string
*/
std::string replaceAWithX(std::string& src_str, std::string strA, std::string strX)
{
	if (src_str[src_str.size() - 1] != '\n')
	{
		src_str += "\n";
	}
	std::string aim_result;
	std::string::size_type pos;
	std::vector<std::string> result = split(src_str, " ");
	std::string remaining = accumulate(result.begin() + 1, result.end(), aim_result);
	int left_bracket = remaining.find_last_of('(');
	std::vector<std::string> aim_context_info;
	std::string before_bracket_context = remaining.substr(0, left_bracket - 1);
	std::string target_bracket_info;
	if (left_bracket != -1)
	{
		std::string bracket_context = remaining.substr(left_bracket, remaining.size() - left_bracket);
		std::string bracket_context_info = bracket_context.substr(1, bracket_context.size() - 3);
		std::vector<std::string> bracket_context_info_vector = split(bracket_context_info, ",");
		aim_context_info.push_back("(");
		for (int i = 0; i < bracket_context_info_vector.size(); i++)
		{
			std::string tmp = bracket_context_info_vector[i];
			size_t size = tmp.size();
			if (tmp[size - 1] == '\n')
			{
				int first_line_feed = tmp.find_first_of('\n');
				tmp = tmp.substr(0, first_line_feed);
			}
			if (tmp == strA)
			{
				aim_context_info.push_back(strX);
			}
			else
			{
				aim_context_info.push_back(tmp);
			}
		}
		aim_context_info.push_back(")");
		target_bracket_info.assign("," + aim_context_info[0]);
		for (int i = 1; i < aim_context_info.size() - 1; i++)
		{
			if (i != aim_context_info.size() - 2)
				target_bracket_info += (aim_context_info[i] + ",");
			else
				target_bracket_info += (aim_context_info[i] + ")");
		}
	}
	else
	{
		target_bracket_info += " ";
	}
	std::vector<std::string> remaining_str = split(before_bracket_context, ",");
	int flag;
	for (int i = 0; i < remaining_str.size(); i++)
	{
		flag = 0;
		std::string tmp = remaining_str[i];
		size_t size = tmp.size();
		if (tmp[size - 1] == '\n')
		{
			int first_line_feed = tmp.find_first_of('\n');
			tmp = tmp.substr(0, first_line_feed);
		}
		if (tmp[0] == '(' && tmp[size - 1] == ')')
		{
			flag = 1;
			tmp = tmp.substr(1, size - 2);
		}
		if (tmp == strA)
		{
			if (flag == 1)
				remaining_str[i].assign("(" + strX);
			else if (flag == 0)
				remaining_str[i].assign(strX);
		}
	}
	aim_result.erase();
	aim_result.assign(result[0] + " ");
	for (int i = 0; i < remaining_str.size(); i++)
	{
		if (i != remaining_str.size() - 1)
			aim_result += (remaining_str[i] + ",");
		else
			aim_result += (remaining_str[i]);
	}
	//target_bracket_info += ")";
	aim_result += target_bracket_info;
	return aim_result;
}


bool is_element_in_vector(std::vector<std::string> v, std::string element) 
{
	std::vector<std::string>::iterator it;
	it = find(v.begin(), v.end(), element);
	if (it != v.end()) 
	{
		return true;
	}
	else 
	{
		return false;
	}
}


//template<class src_type>
//std::string type2str(src_type src) {
//	std::strstream  ss;
//	ss << src;
//	string ret;
//	ss >> ret;
//	return ret;
//}

/**
* @brief  Convert OriginIR  To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR file path
* @param[in]  QuantumMachine*	quantum machine pointer
* @param[out]  QVec	qubit  pointer vector
* @param[out]  std::vector<ClassicalCondition>	classical register  vector
* @return     QProg    quantum program
*/
QProg convert_originir_to_qprog(std::string file_path, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);

/**
* @brief  Convert OriginIR  To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR file path
* @param[in]  QuantumMachine*	quantum machine pointer
* @return     QProg    quantum program
*/
QProg convert_originir_to_qprog(std::string file_path, QuantumMachine* qm);


/**
* @brief  Convert OriginIR String To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR String
* @param[in]  QuantumMachine*	quantum machine pointer
* @param[out]  QVec	qubit  pointer
* @param[out]  std::vector<ClassicalCondition>	classical register  vector
* @return     QProg    quantum program
*/
QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);

/**
* @brief  Convert OriginIR String To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR String
* @param[in]  QuantumMachine*	quantum machine pointer
* @return     QProg    quantum program
*/
QProg convert_originir_string_to_qprog(std::string str_originir, QuantumMachine* qm);


/**
* @brief  OriginIR Transform To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		OriginIR file path
* @param[in]  QuantumMachine*	quantum machine pointer
* @param[out]  QVec	qubit  pointer
* @param[out]  std::vector<ClassicalCondition>	classical register  vector
* @return     QProg    quantum program
*/
QProg transformOriginIRToQProg(std::string filePath, QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);



/**
* @brief  define QGate function info
* @ingroup Utilities
*/
struct CallGateInfo
{
	std::string gate_name;
	std::vector<std::string> qubits;
	std::vector<std::shared_ptr<Exp>> angles;
};

struct FunctionDetailedInfo
{
	std::string function_name;
	std::vector<std::string> par_info;
	std::vector<std::string> par_info_angle;
	std::vector<std::pair<std::string, std::vector<std::string>>> sub_function_info;
	std::vector<std::pair<std::string, std::vector<std::string>>> sub_function_info_angle;
};


struct UserDefineGateInfo {
	std::vector<std::string> gate_bodys;
	std::vector<std::string> tobe_replaced_par;
};


std::vector<FunctionDetailedInfo> functionSymtab;

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

	QVec& qs;
	std::vector<ClassicalCondition>& ccs;

public:
	QProgBuilder(QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv);
	QProg get_qprog();
	size_t get_qubits_size() { return qs.size(); }
	size_t get_cbits_size() { return ccs.size(); }

	enum class GateType {
		H, T, S, X, Y, Z, X1, Y1, Z1, I, ECHO,
		RX, RY, RZ, U1,
		U2, RPhi,
		U3,
		U4,
		CNOT, CZ, ISWAP, SQISWAP, SWAP,

		ISWAPTHETA, CR,

		CU,
		DAGGER, CONTROL,

		TOFFOLI,
		DEFINE_QAGE,
	};

#define MACRO_GET_GATETYPE(name) if (gatename==#name){return GateType::name;}
	static GateType get_gatetype(std::string gatename) {
		MACRO_GET_GATETYPE(H);
		MACRO_GET_GATETYPE(ECHO);
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
		MACRO_GET_GATETYPE(RPhi);
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
		return GateType::DEFINE_QAGE;
	}

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
	size_t add_barrier_literal(size_t exprid, QVec qv);
	size_t add_barrier_cc(size_t exprid, QVec qv);
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


	QVec make_qvec(std::vector<size_t> expridx, std::vector<int> idx);

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

	std::map<std::string, UserDefineGateInfo> UserDefinedGateInf;

public:
	OriginIRVisitor(QuantumMachine* qm, QVec& qv, std::vector<ClassicalCondition>& cv)
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

	antlrcpp::Any visitTranslationunit(originirParser::TranslationunitContext* ctx) {
		auto fullprog = builder.add_prog();
		for (int i = 0; i < ctx->children.size(); ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(fullprog, prog);
		}
		return fullprog;
	}
	antlrcpp::Any visitDeclaration(originirParser::DeclarationContext* ctx) {

		visit(ctx->qinit_declaration());
		visit(ctx->cinit_declaration());

		builder.alloc_qubit(qinit);
		builder.alloc_cbit(cinit);
		return builder.add_prog();
	}

	antlrcpp::Any visitQinit_declaration(originirParser::Qinit_declarationContext* ctx) {
		std::string s = ctx->children[1]->getText();
		int num = atoi(s.c_str());
		qinit = num;
		return 0;
	}

	antlrcpp::Any visitCinit_declaration(originirParser::Cinit_declarationContext* ctx) {
		std::string s = ctx->children[1]->getText();
		int num = atoi(s.c_str());
		cinit = num;
		return 0;
	}

	antlrcpp::Any visitIndex(originirParser::IndexContext* ctx) {
		return visit(ctx->children[1]);
	}

	antlrcpp::Any visitC_KEY_declaration(originirParser::C_KEY_declarationContext* ctx) {
		ExprContext retcontext = visit(ctx->children[1]);
		retcontext.ccid = builder.cc_init_id((size_t)retcontext.value);
		retcontext.isConstant = false;
		return retcontext;
	}

	antlrcpp::Any visitSingle_gate_without_parameter_declaration(
		originirParser::Single_gate_without_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		if (ctx->Q_KEY()) {
			size_t qb_size = builder.get_qubits_size();
			size_t prog_id = builder.add_prog();

			for (int i = 0; i < qb_size; i++) {
				size_t sub_id = builder.add_qgate(gatetype, { i }, {});
				builder.insert_subprog(prog_id, sub_id);
			}
			return prog_id;
		}

		ExprContext context = visit(ctx->children[1]);
		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, {});
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, { -1 }, {});
		}
	}

	antlrcpp::Any visitSingle_gate_with_one_parameter_declaration(
		originirParser::Single_gate_with_one_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext angle = visit(ctx->children[4]);
		if (ctx->Q_KEY()) {
			size_t qb_size = builder.get_qubits_size();
			size_t prog_id = builder.add_prog();

			for (int i = 0; i < qb_size; i++) {
				size_t sub_id = builder.add_qgate(gatetype, { i }, { angle.value });
				builder.insert_subprog(prog_id, sub_id);
			}
			return prog_id;
		}

		ExprContext context = visit(ctx->children[1]);
		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, { angle.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, { -1 }, { angle.value });
		}
	}

	antlrcpp::Any visitSingle_gate_with_two_parameter_declaration(
		originirParser::Single_gate_with_two_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext angle1 = visit(ctx->children[4]);
		ExprContext angle2 = visit(ctx->children[6]);
		if (ctx->Q_KEY()) {
			size_t qb_size = builder.get_qubits_size();
			size_t prog_id = builder.add_prog();

			for (int i = 0; i < qb_size; i++) {
				size_t sub_id = builder.add_qgate(gatetype, { i }, { angle1.value, angle2.value });
				builder.insert_subprog(prog_id, sub_id);
			}
			return prog_id;
		}

		ExprContext context = visit(ctx->children[1]);
		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, { angle1.value, angle2.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, { -1 }, { angle1.value, angle2.value });
		}
	}

	antlrcpp::Any visitSingle_gate_with_three_parameter_declaration(
		originirParser::Single_gate_with_three_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext angle1 = visit(ctx->children[4]);
		ExprContext angle2 = visit(ctx->children[6]);
		ExprContext angle3 = visit(ctx->children[8]);
		if (ctx->Q_KEY()) {
			size_t qb_size = builder.get_qubits_size();
			size_t prog_id = builder.add_prog();

			for (int i = 0; i < qb_size; i++) {
				size_t sub_id = builder.add_qgate(gatetype, { i }, { angle1.value, angle2.value, angle3.value });
				builder.insert_subprog(prog_id, sub_id);
			}
			return prog_id;
		}

		ExprContext context = visit(ctx->children[1]);
		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value }, { angle1.value, angle2.value, angle3.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, { -1 }, { angle1.value, angle2.value, angle3.value });
		}
	}

	antlrcpp::Any visitSingle_gate_with_four_parameter_declaration(
		originirParser::Single_gate_with_four_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext angle1 = visit(ctx->children[4]);
		ExprContext angle2 = visit(ctx->children[6]);
		ExprContext angle3 = visit(ctx->children[8]);
		ExprContext angle4 = visit(ctx->children[10]);

		if (ctx->Q_KEY()) {
			size_t qb_size = builder.get_qubits_size();
			size_t prog_id = builder.add_prog();

			for (int i = 0; i < qb_size; i++) {
				size_t sub_id = builder.add_qgate(gatetype, { i }, { angle1.value, angle2.value, angle3.value, angle4.value });
				builder.insert_subprog(prog_id, sub_id);
			}
			return prog_id;
		}

		ExprContext context = visit(ctx->children[1]);
		if (context.isConstant) {
			return builder.add_qgate(gatetype, { (int)context.value },
				{ angle1.value, angle2.value, angle3.value, angle4.value });
		}
		else {
			return builder.add_qgate_cc(gatetype, { context.ccid }, { -1 },
				{ angle1.value, angle2.value, angle3.value, angle4.value });
		}
	}

	antlrcpp::Any visitDouble_gate_without_parameter_declaration(
		originirParser::Double_gate_without_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);

		// insert gate
		if (context1.isConstant && context2.isConstant) {
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
		originirParser::Double_gate_with_one_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);
		ExprContext angle = visit(ctx->children[6]);

		if (context1.isConstant && context2.isConstant) {
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
		originirParser::Double_gate_with_four_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);
		ExprContext angle1 = visit(ctx->children[6]);
		ExprContext angle2 = visit(ctx->children[8]);
		ExprContext angle3 = visit(ctx->children[10]);
		ExprContext angle4 = visit(ctx->children[12]);

		if (context1.isConstant && context2.isConstant) {
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
		originirParser::Triple_gate_without_parameter_declarationContext* ctx) {
		QProgBuilder::GateType gatetype = visit(ctx->children[0]);
		ExprContext context1 = visit(ctx->children[1]);
		ExprContext context2 = visit(ctx->children[3]);
		ExprContext context3 = visit(ctx->children[5]);

		// insert gate
		if (context1.isConstant && context2.isConstant && context3.isConstant) {
			return builder.add_qgate(gatetype, { (int)context1.value, (int)context2.value, (int)context3.value }, {});
		}
		else if (context1.isConstant && context2.isConstant) {
			return builder.add_qgate_cc(gatetype, { context3.ccid }, { (int)context1.value, (int)context2.value, -1 }, {});
		}
		else if (context1.isConstant && context3.isConstant) {
			return builder.add_qgate_cc(gatetype, { context2.ccid }, { (int)context1.value, -1, (int)context3.value }, {});
		}
		else if (context2.isConstant && context3.isConstant) {
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
		else {
			return builder.add_qgate_cc(gatetype, { context1.ccid, context2.ccid, context3.ccid }, { -1, -1, -1 }, { });
		}
	}

	antlrcpp::Any visitSingle_gate_without_parameter_type(
		originirParser::Single_gate_without_parameter_typeContext* ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_one_parameter_type(
		originirParser::Single_gate_with_one_parameter_typeContext* ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_two_parameter_type(
		originirParser::Single_gate_with_two_parameter_typeContext* ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_three_parameter_type(
		originirParser::Single_gate_with_three_parameter_typeContext* ctx) override {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitSingle_gate_with_four_parameter_type(
		originirParser::Single_gate_with_four_parameter_typeContext* ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitDouble_gate_without_parameter_type(
		originirParser::Double_gate_without_parameter_typeContext* ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitDouble_gate_with_one_parameter_type(
		originirParser::Double_gate_with_one_parameter_typeContext* ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitDouble_gate_with_four_parameter_type(
		originirParser::Double_gate_with_four_parameter_typeContext* ctx) {
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitTriple_gate_without_parameter_type(
		originirParser::Triple_gate_without_parameter_typeContext* ctx)
	{
		std::string gatename = ctx->children[0]->getText();
		return QProgBuilder::get_gatetype(gatename);
	}

	antlrcpp::Any visitPri_ckey(originirParser::Pri_ckeyContext* ctx) {
		ExprContext retcontext = visit(ctx->children[0]);
		return retcontext;
	}

	antlrcpp::Any visitPri_cst(originirParser::Pri_cstContext* ctx) {
		ExprContext context;
		context.isConstant = true;
		if (ctx->children[0]->getText() == "PI")
			context.value = PI;
		else
			context.value = atof(ctx->children[0]->getText().c_str());
		return context;
	}

	antlrcpp::Any visitPri_expr(originirParser::Pri_exprContext* ctx) {
		// expect an object of ExprContext type.
		return visit(ctx->children[0]);
	}

	antlrcpp::Any visitUnary_expression(
		originirParser::Unary_expressionContext* ctx) {
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
		originirParser::Multiplicative_expressionContext* ctx) {
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
		originirParser::Addtive_expressionContext* ctx) {

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
		originirParser::Relational_expressionContext* ctx) {
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
		originirParser::Equality_expressionContext* ctx) {

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
		originirParser::Logical_and_expressionContext* ctx) {
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
		originirParser::Logical_or_expressionContext* ctx) {
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
		originirParser::Assignment_expressionContext* ctx) {
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

	antlrcpp::Any visitControlbit_list(originirParser::Controlbit_listContext* ctx) {
		int n_children = ctx->children.size();
		std::vector<ExprContext> qkeys;
		for (int i = 0; i < n_children; i += 2) {
			qkeys.push_back(visit(ctx->children[i]));
		}
		return qkeys;
	}

	antlrcpp::Any visitStatement(originirParser::StatementContext* ctx) {
		return visit(ctx->children[0]);
	}


	antlrcpp::Any visitUser_defined_gate(originirParser::User_defined_gateContext* ctx) {
		auto startToken = ctx->start;
		auto stopToken = ctx->stop;
		auto interval = antlr4::misc::Interval(startToken->getStartIndex(), stopToken->getStopIndex());
		auto text = startToken->getInputStream()->getText(interval);
		std::string actual_text = text;
		return actual_text;
	}

	virtual antlrcpp::Any visitDefine_dagger_statement(originirParser::Define_dagger_statementContext* ctx) {
		size_t progid = builder.add_prog();

		for (int i = 2; i < ctx->children.size() - 2; ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}

		builder.make_dagger(progid);
		return progid;
	}

	antlrcpp::Any visitDefine_control_statement(originirParser::Define_control_statementContext* ctx) {
		size_t progid = builder.add_prog();
		std::vector<ExprContext> qkeys = visit(ctx->children[1]);

		for (int i = 3; i < ctx->children.size() - 2; ++i) {
			size_t id = visit(ctx->children[i]);
			builder.insert_subprog(progid, id);
			builder.delete_prog(id);
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

	antlrcpp::Any visitDagger_statement(originirParser::Dagger_statementContext* ctx) {
		size_t progid = builder.add_prog();

		for (int i = 2; i < ctx->children.size() - 2; ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}

		builder.make_dagger(progid);
		return progid;
	}

	antlrcpp::Any visitControl_statement(originirParser::Control_statementContext* ctx) {
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
		originirParser::Qelse_statement_fragmentContext* ctx) {
		size_t progid = builder.add_prog();
		for (int i = 2; i < ctx->children.size(); ++i) {
			size_t prog = visit(ctx->children[i]);
			builder.insert_subprog(progid, prog);
			builder.delete_prog(prog);
		}
		return progid;
	}

	antlrcpp::Any visitQif_if(originirParser::Qif_ifContext* ctx) {
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

	antlrcpp::Any visitQif_ifelse(originirParser::Qif_ifelseContext* ctx) {
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
		originirParser::Qwhile_statementContext* ctx) {
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
		originirParser::Measure_statementContext* ctx) {
		if (ctx->C_KEY() && ctx->Q_KEY())
		{
			size_t qb_size = builder.get_qubits_size();
			size_t cb_size = builder.get_cbits_size();

			size_t prog_id = builder.add_prog();
			if (qb_size != cb_size)
				QCERR_AND_THROW(run_fail, " qubit/cbit size  error!");

			for (int i = 0; i < qb_size; i++) {
				size_t sub_id = builder.add_measure_literal(i, i);
				builder.insert_subprog(prog_id, sub_id);
			}
			return prog_id;
		}

		ExprContext qcontext = visit(ctx->children[1]);
		ExprContext ccontext = visit(ctx->children[3]);
		if (qcontext.isConstant)
			return builder.add_measure_literal(qcontext.value, ccontext.value);
		else
			return builder.add_measure_cc(qcontext.ccid, ccontext.value);
	}

	antlrcpp::Any visitReset_statement(
		originirParser::Reset_statementContext* ctx) override {
		if (ctx->Q_KEY()) {
			size_t qb_size = builder.get_qubits_size();
			size_t prog_id = builder.add_prog();

			for (int i = 0; i < qb_size; i++) {
				size_t sub_id = builder.add_reset_literal(i);
				builder.insert_subprog(prog_id, sub_id);
			}
			return prog_id;
		}
		ExprContext qcontext = visit(ctx->children[1]);
		if (qcontext.isConstant)
			return builder.add_reset_literal(qcontext.value);
		else
			return builder.add_reset_cc(qcontext.ccid);
	}

	antlrcpp::Any visitBarrier_statement(originirParser::Barrier_statementContext* ctx) override
	{
		if (ctx->Q_KEY()) {
			size_t qb_size = builder.get_qubits_size();
			QVec qv;
			std::vector<int> indices;
			std::vector<size_t> exprindices;
			for (int i = 1; i < qb_size; i++) {
				indices.push_back(i);
			}
			qv = builder.make_qvec(exprindices, indices);
			return builder.add_barrier_literal(0, qv);
		}

		std::vector<ExprContext> all_qkeys = visit(ctx->children[1]);

		ExprContext qcontext = all_qkeys[0];
		QVec q;

		std::vector<ExprContext> qkeys = all_qkeys;
		qkeys.erase(qkeys.begin());
		if (!qkeys.empty())
		{
			std::vector<int> indices;
			std::vector<size_t> exprindices;
			for (auto qkey : qkeys) {
				if (qkey.isConstant)
					indices.push_back((int)qkey.value);
				else {
					indices.push_back(-1);
					exprindices.push_back(qkey.ccid);
				}
			}
			q = builder.make_qvec(exprindices, indices);
		}

		if (qcontext.isConstant)
			return builder.add_barrier_literal(qcontext.value, q);
		else
			return builder.add_barrier_cc(qcontext.ccid, q);
	}


	antlrcpp::Any visitExpression_statement(
		originirParser::Expression_statementContext* ctx) {
		ExprContext retcontext = visit(ctx->children[0]);
		return builder.add_expr_stat(retcontext.ccid);
	}

	//antlrcpp::Any visitGate_func_statement(originirParser::Gate_func_statementContext* ctx)
	//{
	//	FunctionDetailedInfo inf_;
	//	inf_.function_name = ctx->id()->getText();
	//	inf_.par_num = ctx->id_list().size() / 2;
	//	/*if (ctx->user_defined_gate()
	//	{

	//	}*/
	//	for (int i = 0; i < ctx->user_defined_gate().size(); i++)
	//	{
	//		//std::unordered_map<std::string,int> sub_info=
	//	}
	//	


	//}


	antlrcpp::Any visitDefine_gate_statement(originirParser::Define_gate_statementContext* ctx) {

		CallGateInfo call_gate;

		call_gate.gate_name = ctx->gate_name()->getText();
		std::vector<std::string> qubits = visit(ctx->id_list());
		//inf_.par_num = qubits.size();

		call_gate.qubits = qubits;

		if (ctx->explist())
		{
			std::vector<std::shared_ptr<Exp>> angles = visit(ctx->explist());
			call_gate.angles = angles;
		}

		return call_gate;
	}

	antlrcpp::Any visitExplist(originirParser::ExplistContext* ctx) {
		std::vector<std::shared_ptr<Exp>>angel_params_vec;
		for (auto exp_ctx : ctx->exp())
		{
			std::shared_ptr<Exp> exp_ptr = visit(exp_ctx);
			angel_params_vec.push_back(exp_ptr);
		}
		return angel_params_vec;
	}

	antlrcpp::Any visitExp(originirParser::ExpContext* ctx) {
		std::shared_ptr<Exp> exp_ptr;
		int children_size = ctx->children.size();
		if (1 == children_size)
		{
			if (ctx->id())
			{
				std::string id = ctx->id()->getText();
				exp_ptr = Exp(id).clone();
			}
			else if (ctx->PI())
			{
				exp_ptr = Exp(double(PI)).clone();
			}
			else if (ctx->Integer_Literal())
			{
				std::string str_temp = ctx->Integer_Literal()->getText();
				double val = atoi(str_temp.c_str());
				exp_ptr = Exp(val).clone();
			}
			else if (ctx->Double_Literal())
			{
				std::string str_temp = ctx->Double_Literal()->getText();
				double val = atof(str_temp.c_str());
				exp_ptr = Exp(val).clone();
			}
			else
			{
				QCERR_AND_THROW(run_fail, "error!");
			}
		}
		else if (2 == children_size)  // -exp
		{
			std::shared_ptr<Exp> left_exp_ptr = Exp(double(0)).clone();
			std::string op_type = ctx->children[0]->getText();
			std::shared_ptr<Exp> right_exp_ptr = visit(ctx->children[1]);

			exp_ptr = Exp(left_exp_ptr, right_exp_ptr, op_type).clone();
		}
		else if (3 == children_size)  // exp + - * / exp   ¡¢   (  exp ) 
		{
			if (ctx->LPAREN() && ctx->RPAREN())
			{
				return visit(ctx->children[1]);
			}
			std::shared_ptr<Exp> left_exp_ptr = visit(ctx->children[0]);
			std::string op_type = ctx->children[1]->getText(); //    + - * /
			std::shared_ptr<Exp> right_exp_ptr = visit(ctx->children[2]);

			exp_ptr = Exp(left_exp_ptr, right_exp_ptr, op_type).clone();
		}
		else
		{
			QCERR_AND_THROW(run_fail, "error!");
		}

		return exp_ptr;
	}


	antlrcpp::Any visitGate_func_statement(originirParser::Gate_func_statementContext* ctx) {

		UserDefineGateInfo defined_gate;
		std::string userDefinedGateName = ctx->id()->getText();
		std::vector<std::string> tobe_replaced_par = visit(ctx->id_list(0));

		FunctionDetailedInfo present_info;
		present_info.function_name = userDefinedGateName;
		present_info.par_info = tobe_replaced_par;

		if (ctx->id_list(1))
		{
			std::vector<std::string> tmp= visit(ctx->id_list(1));
			present_info.par_info_angle = tmp;
			for (int i = 0; i < tmp.size(); i++)
			{
				tobe_replaced_par.push_back(tmp[i]);
			}
		}

		std::vector<std::string> str_body_vect;
		for (int i = 0; i < ctx->user_defined_gate().size(); i++)
		{
			std::vector<std::string>::iterator it;
			it = find(all_supported_gate.begin(), all_supported_gate.end(), ctx->user_defined_gate(i)->define_gate_statement()->gate_name()->getText());
			if (it == all_supported_gate.end()) //user_defined
			{
				std::string sub_fun_name = ctx->user_defined_gate(i)->define_gate_statement()->gate_name()->getText();
				std::vector<std::string> sub_par_info = visit(ctx->user_defined_gate(i)->define_gate_statement()->id_list());
				//std::vector<std::shared_ptr<Exp>> angles = visit(ctx->user_defined_gate(i)->define_gate_statement()->explist()->exp());
				present_info.sub_function_info.push_back(make_pair(sub_fun_name, sub_par_info));
				if (ctx->user_defined_gate(i)->define_gate_statement()->explist())
				{
					std::vector<std::string> sub_angles_info;
					std::string tmp = ctx->user_defined_gate(i)->define_gate_statement()->explist()->getText();
					sub_angles_info = split(tmp, ",");
					//sub_angles_info.push_back(ctx->user_defined_gate(i)->define_gate_statement()->explist()->getText());
					present_info.sub_function_info_angle.push_back(make_pair(sub_fun_name, sub_angles_info));
				}
			}
			str_body_vect.push_back(visit(ctx->user_defined_gate(i)));
		}
		defined_gate.gate_bodys = str_body_vect;
		defined_gate.tobe_replaced_par = tobe_replaced_par;
		UserDefinedGateInf.insert(make_pair(userDefinedGateName, defined_gate));
		functionSymtab.push_back(present_info);
		return builder.add_prog();
	}

	antlrcpp::Any visitDefine_gate_declaration(originirParser::Define_gate_declarationContext* ctx) {

		auto prog_id = builder.add_prog();

		antlr4::Token* tk = ctx->getStart();
		size_t line = tk->getLine();
		auto func_name = ctx->id()->getText();
		std::string error_inf = "line" + std::to_string(line) + ":" + "UserDefinedGate " + func_name + " undefined error!";

		if (UserDefinedGateInf.find(func_name) == UserDefinedGateInf.end())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, error_inf);
		}
		
		for (int i = 0; i < functionSymtab.size(); i++)
		{
			if (functionSymtab[i].function_name == func_name)
			{
				std::vector<std::string> standard_par_info = functionSymtab[i].par_info;
				std::vector<std::pair<std::string, std::vector<std::string>>> sub_par_info(functionSymtab[i].sub_function_info);
				for (int k = 0; k < sub_par_info.size(); k++)
				{
					if (sub_par_info[k].second.size() > standard_par_info.size())
					{
						QCERR_AND_THROW_ERRSTR(run_fail, "the number of parameters of the invoked logical gate should be smaller than the number of parameters of the main logical gate !!");
					}
					for (int s = 0; s < sub_par_info[k].second.size(); s++)
					{
						if (!is_element_in_vector(standard_par_info, sub_par_info[k].second.at(s)))
						{
							QCERR_AND_THROW_ERRSTR(run_fail, "the number of parameters of the invoked logical gate should from the parameters of the main logical gate !!");
						}
					}
				}
				if (functionSymtab[i].par_info_angle.size() != 0)
				{
					std::vector<std::string>  standard_par_angle_info = functionSymtab[i].par_info_angle;
					std::vector<std::pair<std::string, std::vector<std::string>>> sub_par_angle_info(functionSymtab[i].sub_function_info_angle);
					for (int s = 0; s < sub_par_angle_info.size(); s++)
					{
						for (int t = 0; t < sub_par_angle_info[s].second.size(); t++)
						{
							if (!is_element_in_vector(standard_par_angle_info, sub_par_angle_info[s].second.at(t))
								&& std::regex_match(sub_par_angle_info[s].second.at(t), id))
							{
								QCERR_AND_THROW_ERRSTR(run_fail, "Wrong Angle parameter occurs !!");
							}
						}
					}
				}
			}
		}

		auto information = UserDefinedGateInf[func_name];
		std::vector<std::string> gate_bodys = information.gate_bodys;
		std::vector<std::string> tobe_replaced_par = information.tobe_replaced_par;

		std::map < std::string, std::string> formal2actual_map;
		int idx = 0;
		for (int i = 0; i < ctx->q_KEY_declaration().size(); i++)
		{
			std::string str_shi_id = ctx->q_KEY_declaration(i)->getText();
			formal2actual_map[tobe_replaced_par[idx]] = str_shi_id;
			idx++;
		}

		for (int i = 0; i < ctx->expression().size(); i++)
		{
			std::string str_shi_id = ctx->expression(i)->getText();
			formal2actual_map[tobe_replaced_par[idx]] = str_shi_id;
			idx++;
		}

		for (auto& str : gate_bodys)
		{
			std::vector<QString> detail_str = QString(str).splitByStr(" ");
			if (detail_str.front().data() == func_name)
			{
				QCERR_AND_THROW_ERRSTR(run_fail, " user defined gate cannot call itself!");
			}

			for (auto& it : formal2actual_map)
			{
				str = replaceAWithX(str, it.first, it.second);
			}

			std::vector<std::string>::iterator it;
			it = find(all_supported_gate.begin(), all_supported_gate.end(), detail_str[0]);
			if (it != all_supported_gate.end())
			{
				str += "\r\n";
				antlr4::ANTLRInputStream input_statement(str);
				statement::statementLexer lexer_statement(&input_statement);
				antlr4::CommonTokenStream tokens_statement(&lexer_statement);
				statement::statementParser parser_statement(&tokens_statement);
				parser_statement.removeErrorListeners();
				StatementErrorListener _e;
				parser_statement.addErrorListener(&_e);
				antlr4::tree::ParseTree* tree_ = parser_statement.translationunit_s();
			}

			antlr4::ANTLRInputStream input(str);
			originirLexer lexer(&input);
			antlr4::CommonTokenStream tokens(&lexer);
			originirParser parser(&tokens);
			parser.removeErrorListeners();
			antlr4::tree::ParseTree* tree = parser.statement();
			size_t _tmp_id = visit(tree);
			builder.insert_subprog(prog_id, _tmp_id);
		}

		return prog_id;
	}

	antlrcpp::Any visitId_list(originirParser::Id_listContext* ctx) {
		std::vector<std::string> id_list;
		for (int i = 0; i < ctx->id().size(); i++)
		{
			id_list.push_back(ctx->id(i)->getText());
		}
		return id_list;
	}

};

QPANDA_END
#endif