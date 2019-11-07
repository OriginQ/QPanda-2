/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
/*! \file QASMToQPorg.h */

#ifndef  _QASMTOQPORG_H
#define  _QASMTOQPORG_H
#include "ThirdParty/antlr4/runtime/src/antlr4-runtime.h"
#include "Core/Utilities/Compiler/QASMCompiler/qasmLexer.h"
#include "Core/Utilities/Compiler/QASMCompiler/qasmParser.h"
#include "Core/Utilities/Compiler/QASMCompiler/qasmVisitor.h"
#include "Core/Utilities/Compiler/QASMCompiler/qasmBaseVisitor.h"

#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/ControlFlow.h"
#include "Core/QuantumCircuit/QuantumMeasure.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"

QPANDA_BEGIN
/**
* @namespace QPanda
*/


static std::map<std::string, std::function<double(double , double)>> _binary_operation =
{
	{"+", [](double  lval,double rval) {return lval + rval; }},
	{"-", [](double lval,double rval) {return lval - rval; } },
	{"*", [](double lval,double rval) {return lval * rval; } },
	{"/", [](double lval,double  rval) {return lval / rval; } },
};

/**
* @class  Exp
* @brief  Saves the expression containing the variable	
*/
class Exp
{
public:
	struct Content
	{
		std::string var_name;
		std::string op_specifier;
		double const_value;
	};

	enum ContentType
	{
		VAR_NAME = 0,
		OP_EXPR,
		CONST_VAL,
	};

	Exp(std::string name)
	{
		m_content.var_name = name;
		m_content_type = VAR_NAME;
	}

	Exp(Exp *left_expr, Exp *right_expr, std::string op)
	{
		m_left_exp = left_expr;
		m_right_exp = right_expr;
		m_content.op_specifier = op;
		m_content_type = OP_EXPR;
	}

	Exp(double val)
	{
		m_content.const_value = val;
		m_content_type = CONST_VAL;
	}

	~Exp()
	{
		if (m_content_type == OP_EXPR)
		{
			if (m_right_exp != nullptr)
				delete m_right_exp;

			if (m_left_exp != nullptr)
				delete m_left_exp;
		}
	}

	void set_formal_actual_var_map(std::map <std::string, double> name_val_map)
	{
		m_formal_actual_var_map = name_val_map;
		if (m_content_type == OP_EXPR)
		{
			m_left_exp->set_formal_actual_var_map(name_val_map);
			m_right_exp->set_formal_actual_var_map(name_val_map);
		}
	}

	/**
	* @brief   evaluation
	* @return  double   operation rusult
	*/
	double eval()
	{
		if (m_content_type == VAR_NAME)
		{
			std::string var_name = m_content.var_name;
			auto iter_actual = m_formal_actual_var_map.find(var_name);
			if (iter_actual == m_formal_actual_var_map.end())
			{
				QCERR("get actual val error!");
				throw std::runtime_error("get actual val error!");
			}
			return iter_actual->second;
		}
		else if (m_content_type == OP_EXPR)
		{
			double left_val = m_left_exp->eval();
			double right_val = m_right_exp->eval();

			auto iter_func = _binary_operation.find(m_content.op_specifier);
			if (iter_func == _binary_operation.end())
			{
				QCERR("get binary operation  function error!");
				throw std::runtime_error("get binary operation  function error!");
			}
			return iter_func->second(left_val, right_val);
		}
		else if (m_content_type == CONST_VAL)
		{
			return m_content.const_value;
		}
		else
		{
			QCERR("content typer error!");
			throw std::invalid_argument("content typer error!");
		}
	}

private:
	Exp *m_left_exp;
	Exp *m_right_exp;
	int m_content_type;
	Content m_content;
	std::map<std::string, double> m_formal_actual_var_map;
};

struct RegParamInfo
{
	std::string reg_name;
	int reg_index;
};

struct GateOperationInfo
{
	std::string op_id;
	std::vector<RegParamInfo> regs_vec;
	std::vector<Exp *> angles_vec;
};

struct GataFuncInfo
{
	std::string func_name;
	std::vector<std::string> angle_names_vec;
	std::vector<std::string> reg_names_vec;
	std::vector<GateOperationInfo> ops_vec;
};

enum  QASMGateType
{
	ID_GATE=0,
	X_GATE,
	Y_GATE,
	Z_GATE,
	H_GATE,
	S_GATE,
	SDG_GATE,
	T_GATE,
	TDG_GATE,
	RX_GATE,
	RY_GATE,
	RZ_GATE,
	CX_GATE,
	CZ_GATE,
	CY_GATE,
	CH_GATE,
	U3_GATE,
	U2_GATE,
	U1_GATE,
	CCX_GATE,
	CRZ_GATE,
	CU1_GATE,
	CU3_GATE,
	U_BASE_GATE,
	CX_BASE_GATE
};

/**
* @class  QASMToQProg
* @ingroup Utilities
* @brief QASM instruction sets  convert  to quantum program
*/
class QASMToQProg : public qasmBaseVisitor
{
public:
	QASMToQProg(QuantumMachine* qvm);

	~QASMToQProg();

	/**
	 * @brief  They are abstract visitors for a parse tree produced by qasmParser.
    */
	antlrcpp::Any visitMainprogram(qasmParser::MainprogramContext *ctx);
	antlrcpp::Any visitHead_decl(qasmParser::Head_declContext *ctx);
	antlrcpp::Any visitVersion_decl(qasmParser::Version_declContext *ctx);
	antlrcpp::Any visitInclude_decl(qasmParser::Include_declContext *ctx);
	antlrcpp::Any visitStatement(qasmParser::StatementContext *ctx);
	antlrcpp::Any visitReg_decl(qasmParser::Reg_declContext *ctx);
	antlrcpp::Any visitOpaque_decl(qasmParser::Opaque_declContext *ctx);
	antlrcpp::Any visitIf_decl(qasmParser::If_declContext *ctx);
	antlrcpp::Any visitBarrier_decl(qasmParser::Barrier_declContext *ctx);
	antlrcpp::Any visitGate_decl(qasmParser::Gate_declContext *ctx);
	antlrcpp::Any visitGoplist(qasmParser::GoplistContext *ctx);
	antlrcpp::Any visitBop(qasmParser::BopContext *ctx);
	antlrcpp::Any visitQop(qasmParser::QopContext *ctx);
	antlrcpp::Any visitUop(qasmParser::UopContext *ctx);
	antlrcpp::Any visitAnylist(qasmParser::AnylistContext *ctx);
	antlrcpp::Any visitIdlist(qasmParser::IdlistContext *ctx);
	antlrcpp::Any visitArgument(qasmParser::ArgumentContext *ctx);
	antlrcpp::Any visitId_index(qasmParser::Id_indexContext *ctx);
	antlrcpp::Any visitExplist(qasmParser::ExplistContext *ctx);
	antlrcpp::Any visitExp(qasmParser::ExpContext *ctx);
	antlrcpp::Any visitId(qasmParser::IdContext *ctx);
	antlrcpp::Any visitInteger(qasmParser::IntegerContext *ctx);
	antlrcpp::Any visitDecimal(qasmParser::DecimalContext *ctx);
	antlrcpp::Any visitFilename(qasmParser::FilenameContext *ctx);

	QVec find_qvec_map_value(std::string str_key);
	std::vector<ClassicalCondition> find_cvec_map_value(std::string str_key);
	void execute_gate_function(GateOperationInfo op_info, QProg  &prog);

	void build_zero_param_single_gate(QASMGateType type, bool is_dagger, GateOperationInfo op_info, QProg  &prog);
	void build_one_param_single_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog);
	void build_two_param_single_gate_func(QASMGateType type, GateOperationInfo op_info, QProg  &prog);
	void build_three_param_single_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog);
	void build_zero_param_double_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog);
	void build_zero_param_triple_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog);

	void build_zero_param_double_circuit(QASMGateType type, GateOperationInfo op_info, QProg  &prog);
	void build_one_param_double_circuit(QASMGateType type, GateOperationInfo op_info, QProg  &prog);
	void build_three_param_double_circuit(QASMGateType type, GateOperationInfo op_info, QProg  &prog);

	void build_qprog(GateOperationInfo op_info, QProg  &prog);
	QProg get_qprog();

private:
	QuantumMachine * m_qvm;
	bool m_support_qelib1;
	QProg m_build_qprog;
	std::map<std::string, QVec> m_alloc_qvec_map;
	std::map<std::string, std::vector<ClassicalCondition> > m_alloc_cvec_map;
	std::map<std::string, QASMGateType> m_qasm_gate_type;
	std::map<std::string, GataFuncInfo> m_gate_func_map;
	std::vector<Exp *> m_need_delete_exp;

	std::map<int, std::function<QGate(Qubit *)> > m_zero_param_single_gate_func;
	std::map<int, std::function<QGate(Qubit *, double)> > m_one_param_single_gate_func;
	std::map<int, std::function<QGate(Qubit *, double, double)> > m_two_param_single_gate_func;
	std::map<int, std::function<QGate(Qubit *, double, double, double)> > m_three_param_single_gate_func;
	std::map<int, std::function<QGate(Qubit *, Qubit*)> > m_zero_param_double_gate_func;
	std::map<int, std::function<QGate(Qubit *, Qubit*, Qubit*)> > m_zero_param_triple_gate_func;

	std::map<int, std::function<QCircuit(Qubit *, Qubit*)> > m_zero_param_double_circuit_func;
	std::map<int, std::function<QCircuit(Qubit *, Qubit*, double)> > m_one_param_double_circuit_func;
	std::map<int, std::function<QCircuit(Qubit *, Qubit*, double, double, double)> > m_three_param_double_circuit_func;
};



/**
* @brief  QASM Transform To  Quantum Program
* @ingroup Utilities
* @param[in]  std::string		QASM file path
* @param[in]  QuantumMachine*	quantum machine pointer
* @return     QProg    quantum program
*/
QProg convert_qasm_to_qprog(std::string file_path, QuantumMachine* qvm);

QPANDA_END
#endif //!_QASMTOQPORG_H