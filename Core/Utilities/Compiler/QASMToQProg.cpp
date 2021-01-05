#include "Core/Utilities/Compiler/QASMToQProg.h"
#include "Core/Utilities/Tools/Utils.h"
#include <queue>
using namespace std;
USING_QPANDA


static QCircuit _qasm_cy(Qubit *ctrl, Qubit *target)
{
	QCircuit cir;
	cir << S(target).dagger()
		<< CNOT(ctrl, target)
		<< S(target);
	return cir;
}

static QCircuit _qasm_ch(Qubit *ctrl, Qubit *target)
{
	QCircuit cir;
	cir << H(target)
		<< S(target).dagger()
		<< CNOT(ctrl, target)
		<< H(target)
		<< T(target)
		<< CNOT(ctrl, target)
		<< T(target)
		<< H(target)
		<< S(target)
		<< X(target)
		<< S(ctrl);
	return cir;
}

static QCircuit _qasm_crz(Qubit *ctrl, Qubit *target, double lambda)
{
	QCircuit cir;
	cir << U1(target, lambda / 2)
		<< CNOT(ctrl, target)
		<< U1(target, -lambda / 2)
		<< CNOT(ctrl, target);
	return cir;
}

static QCircuit _qasm_cu1(Qubit *ctrl, Qubit *target, double lambda)
{
	QCircuit cir;
	cir << U1(ctrl, lambda / 2)
		<< CNOT(ctrl, target)
		<< U1(target, -lambda / 2)
		<< CNOT(ctrl, target)
		<< U1(target, lambda / 2);
	return cir;
}

static QCircuit _qasm_cu3(Qubit *ctrl, Qubit *target, double theta, double phi, double lambda)
{
	QCircuit cir;
	cir << U1(target, (lambda - phi) / 2)
		<< CNOT(ctrl, target)
		<< U3(target, theta, phi, lambda)
		<< CNOT(ctrl, target)
		<< U3(target, theta / 2, phi, 0);
	return cir;
}

QASMToQProg::QASMToQProg(QuantumMachine* qvm,  QVec &qv, std::vector<ClassicalCondition> &cv)
	:m_qvm(qvm), m_qvec(qv), m_cvec(cv)
{
	m_qasm_gate_type.insert(make_pair("u3", QASMGateType::U3_GATE));
	m_qasm_gate_type.insert(make_pair("u2", QASMGateType::U2_GATE));
	m_qasm_gate_type.insert(make_pair("u1", QASMGateType::U1_GATE));
	m_qasm_gate_type.insert(make_pair("id", QASMGateType::ID_GATE));
	m_qasm_gate_type.insert(make_pair("x", QASMGateType::X_GATE));
	m_qasm_gate_type.insert(make_pair("y", QASMGateType::Y_GATE));
	m_qasm_gate_type.insert(make_pair("z", QASMGateType::Z_GATE));
	m_qasm_gate_type.insert(make_pair("h", QASMGateType::H_GATE));
	m_qasm_gate_type.insert(make_pair("s", QASMGateType::S_GATE));
	m_qasm_gate_type.insert(make_pair("sdg", QASMGateType::SDG_GATE));
	m_qasm_gate_type.insert(make_pair("t", QASMGateType::T_GATE));
	m_qasm_gate_type.insert(make_pair("tdg", QASMGateType::TDG_GATE));
	m_qasm_gate_type.insert(make_pair("cx", QASMGateType::CX_GATE));
	m_qasm_gate_type.insert(make_pair("rx", QASMGateType::RX_GATE));
	m_qasm_gate_type.insert(make_pair("ry", QASMGateType::RY_GATE));
	m_qasm_gate_type.insert(make_pair("rz", QASMGateType::RZ_GATE));
	m_qasm_gate_type.insert(make_pair("cz", QASMGateType::CZ_GATE));
	m_qasm_gate_type.insert(make_pair("cy", QASMGateType::CY_GATE));
	m_qasm_gate_type.insert(make_pair("ch", QASMGateType::CH_GATE));
	m_qasm_gate_type.insert(make_pair("ccx", QASMGateType::CCX_GATE));
	m_qasm_gate_type.insert(make_pair("crz", QASMGateType::CRZ_GATE));
	m_qasm_gate_type.insert(make_pair("cu1", QASMGateType::CU1_GATE));
	m_qasm_gate_type.insert(make_pair("cu3", QASMGateType::CU3_GATE));
	m_qasm_gate_type.insert(make_pair("U", QASMGateType::U_BASE_GATE));
	m_qasm_gate_type.insert(make_pair("CX", QASMGateType::CX_BASE_GATE));


	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::ID_GATE, I));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::X_GATE, X));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::Y_GATE, Y));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::Z_GATE, Z));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::H_GATE, H));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::S_GATE, S));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::SDG_GATE, S));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::T_GATE, T));
	m_zero_param_single_gate_func.insert(make_pair(QASMGateType::TDG_GATE, T));

	m_one_param_single_gate_func.insert(make_pair(QASMGateType::U1_GATE, U1));
	m_one_param_single_gate_func.insert(make_pair(QASMGateType::RX_GATE, RX));
	m_one_param_single_gate_func.insert(make_pair(QASMGateType::RY_GATE, RY));
	m_one_param_single_gate_func.insert(make_pair(QASMGateType::RZ_GATE, U1));

	m_two_param_single_gate_func.insert(make_pair(QASMGateType::U2_GATE, U2));

	m_three_param_single_gate_func.insert(make_pair(QASMGateType::U_BASE_GATE, U3));
	m_three_param_single_gate_func.insert(make_pair(QASMGateType::U3_GATE, U3));

	m_zero_param_double_gate_func.insert(make_pair(QASMGateType::CX_BASE_GATE, CNOT));
	m_zero_param_double_gate_func.insert(make_pair(QASMGateType::CX_GATE, CNOT));
	m_zero_param_double_gate_func.insert(make_pair(QASMGateType::CZ_GATE, CZ));

	m_zero_param_triple_gate_func.insert(make_pair(QASMGateType::CCX_GATE, Toffoli));

	m_zero_param_double_circuit_func.insert(make_pair(QASMGateType::CY_GATE, _qasm_cy));
	m_zero_param_double_circuit_func.insert(make_pair(QASMGateType::CH_GATE, _qasm_ch));
	m_one_param_double_circuit_func.insert(make_pair(QASMGateType::CRZ_GATE, _qasm_crz));
	m_one_param_double_circuit_func.insert(make_pair(QASMGateType::CU1_GATE, _qasm_cu1));
	m_three_param_double_circuit_func.insert(make_pair(QASMGateType::CU3_GATE, _qasm_cu3));
}

QASMToQProg::~QASMToQProg()
{
}

antlrcpp::Any QASMToQProg::visitMainprogram(qasmParser::MainprogramContext *ctx)
{
	return visitChildren(ctx);
}

antlrcpp::Any QASMToQProg::visitHead_decl(qasmParser::Head_declContext *ctx)
{
	if (!ctx->version_decl())
	{
		QCERR("without QASM version info!!");
		throw runtime_error("without QASM version info!!");
	}
	double ver = visit(ctx->version_decl());
	if (ver > 2.0 || ver < 2.0)
	{
		QCERR("QASM version error!!");
		throw runtime_error("QASM version error!!");
	}

	m_support_qelib1 = false;
	if (ctx->include_decl())
	{
		string str_inc = visit(ctx->include_decl());
		if (str_inc == "qelib1.inc")
		{
			m_support_qelib1 = true;
		}
	}
	return  0;
}

antlrcpp::Any QASMToQProg::visitVersion_decl(qasmParser::Version_declContext *ctx)
{
	return visit(ctx->decimal());
}

antlrcpp::Any QASMToQProg::visitInclude_decl(qasmParser::Include_declContext *ctx)
{
	return visit(ctx->filename());
}

antlrcpp::Any QASMToQProg::visitStatement(qasmParser::StatementContext *ctx)
{
	return visitChildren(ctx);
}

antlrcpp::Any QASMToQProg::visitReg_decl(qasmParser::Reg_declContext *ctx)
{
	string reg_id = visit(ctx->id());
	int reg_size = visit(ctx->integer());
	if (ctx->CREG_KEY())
	{
		m_cvec = m_qvm->allocateCBits(reg_size);
		m_alloc_cvec_map.insert(make_pair(reg_id, m_cvec));
	}
	else if (ctx->QREG_KEY())
	{
		m_qvec = m_qvm->allocateQubits(reg_size);
		m_alloc_qvec_map.insert(make_pair(reg_id, m_qvec));
	}
	else
	{
		QCERR("reg type error!");
		throw runtime_error("reg type error!");
	}
	return 0;
}

antlrcpp::Any QASMToQProg::visitOpaque_decl(qasmParser::Opaque_declContext *ctx)
{
	//do nothing
	return 0;
}

antlrcpp::Any QASMToQProg::visitIf_decl(qasmParser::If_declContext *ctx)
{
	QProg prog;
	int qop_childen_size = ctx->qop()->children.size();
	if (qop_childen_size == 5) //measure
	{
		RegParamInfo q_reg_info = visit(ctx->qop()->argument(0));
		RegParamInfo c_reg_info = visit(ctx->qop()->argument(1));
		QVec qv = find_qvec_map_value(q_reg_info.reg_name);
		auto cv = find_cvec_map_value(c_reg_info.reg_name);
		if (q_reg_info.reg_index == -1
			&& c_reg_info.reg_index == -1
			&& qv.size() == cv.size())  // measure q->c
		{
			for (int i = 0; i < qv.size(); i++)
			{
				prog << Measure(qv[i], cv[i]);
			}
		}
		else if (-1 != q_reg_info.reg_index
			&& -1 != c_reg_info.reg_index)  // measure q[1]->c[1]
		{
			prog << Measure(qv[c_reg_info.reg_index], cv[c_reg_info.reg_index]);
		}
		else
		{
			QCERR("measure error!");
			throw runtime_error("measure error!");
		}
	}
	else if (qop_childen_size == 3) //reset
	{
		RegParamInfo q_reg_info = visit(ctx->qop()->argument(0));
		QVec qv = find_qvec_map_value(q_reg_info.reg_name);
		if (-1 == q_reg_info.reg_index)   // reset q
		{
			for (int i = 0; i < qv.size(); i++)
			{
				m_build_qprog << Reset(qv[i]);
			}
		}
		else  // reset q[1]
		{
			m_build_qprog << Reset(qv[q_reg_info.reg_index]);
		}
	}
	else  //uop
	{
		GateOperationInfo op_info;
		op_info = visit(ctx->qop()->uop());
		build_qprog(op_info, prog);
	}

	int rvalue = visit(ctx->integer());
	string c_vec_name = visit(ctx->id());
	std::vector<ClassicalCondition> c_vec = find_cvec_map_value(c_vec_name);

	auto qif = createIfProg((c_vec[0] == rvalue), prog);
	m_build_qprog << qif;
	return 0;
}

antlrcpp::Any QASMToQProg::visitBarrier_decl(qasmParser::Barrier_declContext *ctx)
{
	//do nothing
	return 0;
}

antlrcpp::Any QASMToQProg::visitGate_decl(qasmParser::Gate_declContext *ctx)
{
	//gate func
	GataFuncInfo func_info;
	int children_size = ctx->children.size();
	if (6 == children_size || 8 == children_size)  // gate id idlist { goplist }      // gate id ( ) idlist { goplist }  
	{
		string gate_func_name = visit(ctx->id());
		std::vector<std::string> reg_params_vec = visit(ctx->idlist(0));
		std::vector<GateOperationInfo> ops_vec = visit(ctx->goplist());

		func_info.func_name = gate_func_name;
		func_info.ops_vec = ops_vec;
		func_info.reg_names_vec = reg_params_vec;
		m_gate_func_map.insert(make_pair(gate_func_name, func_info));

	}
	else if (9 == children_size)  // gate id ( idlist ) idlist { goplist }
	{
		string gate_func_name = visit(ctx->id());
		std::vector<std::string> angle_params_vec = visit(ctx->idlist(0));
		std::vector<std::string> reg_params_vec = visit(ctx->idlist(1));
		std::vector<GateOperationInfo> ops_vec = visit(ctx->goplist());

		func_info.func_name = gate_func_name;
		func_info.ops_vec = ops_vec;
		func_info.angle_names_vec = angle_params_vec;
		func_info.reg_names_vec = reg_params_vec;
		m_gate_func_map.insert(make_pair(gate_func_name, func_info));
	}
	else  // without goplist
	{
		string gate_func_name = visit(ctx->id());
		m_gate_func_map.insert(make_pair(gate_func_name, func_info));
	}

	return 0;
}

antlrcpp::Any QASMToQProg::visitGoplist(qasmParser::GoplistContext *ctx)
{
	std::vector<GateOperationInfo> ops_vec;
	GateOperationInfo op_info;
	for (int i = 0; i < ctx->children.size(); i++)
	{
		op_info = visit(ctx->children[i]);
		ops_vec.push_back(op_info);
	}
	return ops_vec;
}

antlrcpp::Any QASMToQProg::visitBop(qasmParser::BopContext *ctx)
{
	//do nothing
	return 0;
}

antlrcpp::Any QASMToQProg::visitQop(qasmParser::QopContext *ctx)
{
	int childen_size = ctx->children.size();
	if (5 == childen_size)
	{
		RegParamInfo q_reg_info = visit(ctx->argument(0));
		RegParamInfo c_reg_info = visit(ctx->argument(1));

		QVec qv = find_qvec_map_value(q_reg_info.reg_name);
		auto cv = find_cvec_map_value(c_reg_info.reg_name);

		if (-1 == q_reg_info.reg_index
			&& -1 == c_reg_info.reg_index
			&& qv.size() == cv.size())  // measure q->c
		{
			for (int i = 0; i < qv.size(); i++)
			{
				m_build_qprog << Measure(qv[i], cv[i]);
			}
		}
		else if (-1 != q_reg_info.reg_index
			&& -1 != c_reg_info.reg_index)  // measure q[1]->c[1]
		{
			m_build_qprog << Measure(qv[q_reg_info.reg_index], cv[c_reg_info.reg_index]);
		}
		else
		{
			QCERR("measure error!");
			throw runtime_error("measure error!");
		}
	}
	else if (3 == childen_size) //reset
	{
		RegParamInfo q_reg_info = visit(ctx->argument(0));
		QVec qv = find_qvec_map_value(q_reg_info.reg_name);
		if (-1 == q_reg_info.reg_index)   // reset q
		{
			for (int i = 0; i < qv.size(); i++)
			{
				m_build_qprog << Reset(qv[i]);
			}
		}
		else  // reset q[1]
		{
			m_build_qprog << Reset(qv[q_reg_info.reg_index]);
		}
	}
	else  //uop
	{
		GateOperationInfo op_info;
		op_info = visit(ctx->uop());
		build_qprog(op_info, m_build_qprog);
	}

	return 0;
}

antlrcpp::Any QASMToQProg::visitUop(qasmParser::UopContext *ctx)
{
	GateOperationInfo op_info;
	for (auto argument_ctx : ctx->argument())
	{
		RegParamInfo reg_info;
		reg_info = visit(argument_ctx);
		op_info.regs_vec.push_back(reg_info);
	}
	if (ctx->anylist())
	{
		std::vector<RegParamInfo> reg_formal_params_vec = visit(ctx->anylist());
		op_info.regs_vec = reg_formal_params_vec;
	}

	if (ctx->explist())
	{
		std::vector<std::shared_ptr<Exp>> angle_vec = visit(ctx->explist());
		op_info.angles_vec = angle_vec;
	}

	if (ctx->id())
	{
		string func_name = visit(ctx->id());
		op_info.op_id = func_name;
	}
	else
	{
		op_info.op_id = ctx->children[0]->getText();
	}
	return op_info;
}

antlrcpp::Any QASMToQProg::visitAnylist(qasmParser::AnylistContext *ctx)
{
	std::vector<RegParamInfo> reg_formal_params_vec;

	for (int i = 0; i < ctx->children.size(); i++)
	{
		for (auto id_ctx : ctx->id())
		{
			if (ctx->children[i] == id_ctx)
			{
				RegParamInfo param;
				string id = visit(id_ctx);
				param.reg_name = id;
				param.reg_index = -1;
				reg_formal_params_vec.push_back(param);
			}
		}

		for (auto id_index_ctx : ctx->id_index())
		{
			if (ctx->children[i] == id_index_ctx)
			{
				RegParamInfo param = visit(id_index_ctx);
				reg_formal_params_vec.push_back(param);
			}
		}
	}

	return reg_formal_params_vec;
}

antlrcpp::Any QASMToQProg::visitIdlist(qasmParser::IdlistContext *ctx)
{
	std::vector<std::string> ids_vec;
	for (auto id_ctx : ctx->id())
	{
		string str_id = visit(id_ctx);
		ids_vec.push_back(str_id);
	}
	return ids_vec;
}

antlrcpp::Any QASMToQProg::visitArgument(qasmParser::ArgumentContext *ctx)
{
	RegParamInfo info;
	string id = visit(ctx->id());
	info.reg_name = id;
	info.reg_index = -1;
	if (ctx->integer())
	{
		info.reg_index = visit(ctx->integer());
	}
	return info;
}

antlrcpp::Any QASMToQProg::visitId_index(qasmParser::Id_indexContext *ctx)
{
	RegParamInfo info;
	string str_id = visit(ctx->id());
	int index = visit(ctx->integer());
	info.reg_name = str_id;
	info.reg_index = index;
	return info;
}

antlrcpp::Any QASMToQProg::visitExplist(qasmParser::ExplistContext *ctx)
{
	std::vector<std::shared_ptr<Exp>>angel_params_vec;
	for (auto exp_ctx : ctx->exp())
	{
		std::shared_ptr<Exp> exp_ptr = visit(exp_ctx);
		angel_params_vec.push_back(exp_ptr);
	}

	return angel_params_vec;
}


antlrcpp::Any QASMToQProg::visitExp(qasmParser::ExpContext *ctx)
{
	std::shared_ptr<Exp> exp_ptr;
	int children_size = ctx->children.size();
	if (1 == children_size)
	{
		if (ctx->id())
		{
			string id = ctx->id()->getText();
			exp_ptr = Exp(id).clone();
		}
		else if (ctx->PI_KEY())
		{
			exp_ptr =  Exp(double(PI)).clone();
		}
		else if (ctx->decimal())
		{
			double val = visit(ctx->decimal());
			exp_ptr =  Exp(val).clone();
		}
		else if (ctx->integer())
		{
			int val = visit(ctx->integer());
			exp_ptr =  Exp((double)val).clone();
		}
		else if (ctx->real())
		{
			double val = visit(ctx->real());
			exp_ptr = Exp(val).clone();
		}
		else
		{
			QCERR("error!");
			throw runtime_error("error!");
		}
	}
	else if (2 == children_size)  // -exp
	{
		std::shared_ptr<Exp> left_exp_ptr  = Exp(double(0)).clone();
		string op_type = ctx->children[0]->getText();
		std::shared_ptr<Exp> right_exp_ptr = visit(ctx->children[1]);

		exp_ptr =  Exp(left_exp_ptr, right_exp_ptr, op_type).clone();
	}
	else if (3 == children_size)  // exp + - * / exp   ¡¢   (  exp ) 
	{
		if (ctx->LPAREN() && ctx->RPAREN())
		{
			return visit(ctx->children[1]);
		}
		std::shared_ptr<Exp> left_exp_ptr = visit(ctx->children[0]);
		string op_type = ctx->children[1]->getText(); //    + - * /
		std::shared_ptr<Exp> right_exp_ptr = visit(ctx->children[2]);

		exp_ptr = Exp(left_exp_ptr, right_exp_ptr, op_type).clone();
	}
	else
	{
		QCERR("error!");
		throw runtime_error("error!");
	}

	return exp_ptr;
}

antlrcpp::Any QASMToQProg::visitId(qasmParser::IdContext *ctx)
{
	return  ctx->children[0]->getText();
}

antlrcpp::Any QASMToQProg::visitInteger(qasmParser::IntegerContext *ctx)
{
	string str_integer = ctx->children[0]->getText();
	return atoi(str_integer.c_str());
}

antlrcpp::Any QASMToQProg::visitReal(qasmParser::RealContext *ctx)
{
	string str_real = ctx->children[0]->getText();
	return atof(str_real.c_str());
}

antlrcpp::Any QASMToQProg::visitDecimal(qasmParser::DecimalContext *ctx)
{
	string str_dec = ctx->children[0]->getText();
	return atof(str_dec.c_str());
}

antlrcpp::Any QASMToQProg::visitFilename(qasmParser::FilenameContext *ctx)
{
	return ctx->children[0]->getText();
}

QVec QASMToQProg::find_qvec_map_value(std::string str_key)
{
	auto iter = m_alloc_qvec_map.find(str_key);
	if (iter == m_alloc_qvec_map.end())
	{
		QCERR("qvec map not find " + str_key);
		throw runtime_error("qvec map find error");
	}
	return iter->second;
}

std::vector<ClassicalCondition> QASMToQProg::find_cvec_map_value(std::string str_key)
{
	auto iter = m_alloc_cvec_map.find(str_key);
	if (iter == m_alloc_cvec_map.end())
	{
		QCERR("cvec map not find " + str_key);
		throw runtime_error("cvec map find error");
	}
	return iter->second;
}

void QASMToQProg::build_zero_param_single_gate(QASMGateType type, bool is_dagger, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_zero_param_single_gate_func.find(type);
	if (iter_func == m_zero_param_single_gate_func.end())
	{
		QCERR("gate type is not supported!");
		throw runtime_error("gate type is not supported!");
	}
	if (1 != op_info.regs_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}
	auto reg = op_info.regs_vec[0];
	QVec qv = find_qvec_map_value(reg.reg_name);
	if (-1 == reg.reg_index)  // h q
	{
		for (auto q : qv)
		{
			QGate gate = iter_func->second(q);
			gate.setDagger(is_dagger);
			prog << gate;
		}
	}
	else   // h q[1]
	{
		QGate gate = iter_func->second(qv[reg.reg_index]);
		gate.setDagger(is_dagger);
		prog << gate;
	}
}

void QASMToQProg::build_three_param_single_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_three_param_single_gate_func.find(type);
	if (iter_func == m_three_param_single_gate_func.end())
	{
		QCERR(type + " gate type is not supported!");
		throw runtime_error(type + " gate type is not supported!");
	}
	if (1 != op_info.regs_vec.size() || 3 != op_info.angles_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}
	auto reg = op_info.regs_vec[0];
	double angle_1 = op_info.angles_vec[0]->eval();
	double angle_2 = op_info.angles_vec[1]->eval();
	double angle_3 = op_info.angles_vec[2]->eval();
	QVec qv = find_qvec_map_value(reg.reg_name);
	if (-1 == reg.reg_index)  //  u3(theta,phi,lambda) q
	{
		for (auto q : qv)
		{
			prog << iter_func->second(q, angle_1, angle_2, angle_3);
		}
	}
	else   //  u3(theta,phi,lambda) q[1]
	{
		prog << iter_func->second(qv[reg.reg_index], angle_1, angle_2, angle_3);
	}
}

void QASMToQProg::build_two_param_single_gate_func(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_two_param_single_gate_func.find(type);
	if (iter_func == m_two_param_single_gate_func.end())
	{
		QCERR(type + " gate type is not supported!");
		throw runtime_error(type + " gate type is not supported!");
	}
	if (1 != op_info.regs_vec.size() || 2 != op_info.angles_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}
	auto reg = op_info.regs_vec[0];
	double angle_1 = op_info.angles_vec[0]->eval();
	double angle_2 = op_info.angles_vec[1]->eval();
	QVec qv = find_qvec_map_value(reg.reg_name);
	if (-1 == reg.reg_index)  //  u2(phi,lambda) q
	{
		for (auto q : qv)
		{
			prog << iter_func->second(q, angle_1, angle_2);
		}
	}
	else   // u2(phi,lambda) q[1]
	{
		prog << iter_func->second(qv[reg.reg_index], angle_1, angle_2);
	}
}

void QASMToQProg::build_one_param_single_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_one_param_single_gate_func.find(type);
	if (iter_func == m_one_param_single_gate_func.end())
	{
		QCERR(type + " gate type is not supported!");
		throw runtime_error(type + " gate type is not supported!");
	}
	if (1 != op_info.regs_vec.size() || 1 != op_info.angles_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}
	auto reg = op_info.regs_vec[0];
	double angle = op_info.angles_vec[0]->eval();
	QVec qv = find_qvec_map_value(reg.reg_name);
	if (-1 == reg.reg_index)  // rx(pi) q
	{
		for (auto q : qv)
		{
			prog << iter_func->second(q, angle);
		}
	}
	else   // rx(pi) q[1]
	{
		prog << iter_func->second(qv[reg.reg_index], angle);
	}
}

void QASMToQProg::build_zero_param_double_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_zero_param_double_gate_func.find(type);
	if (iter_func == m_zero_param_double_gate_func.end())
	{
		QCERR(type + " gate type is not supported!");
		throw runtime_error(type + " gate type is not supported!");
	}
	if (2 != op_info.regs_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}

	auto reg_1 = op_info.regs_vec[0];
	auto reg_2 = op_info.regs_vec[1];

	QVec qv_1 = find_qvec_map_value(reg_1.reg_name);
	QVec qv_2 = find_qvec_map_value(reg_2.reg_name);

	if (-1 == reg_1.reg_index  && -1 == reg_2.reg_index)  //  cx q, t
	{
		if (qv_1.size() != qv_2.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[i]);
		}
	}
	else  if (-1 == reg_1.reg_index)  //cx q,t[1]
	{
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[reg_2.reg_index]);
		}
	}
	else if (-1 == reg_2.reg_index)  // cx q[1],t
	{
		for (int i = 0; i < qv_2.size(); i++)
		{
			prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[i]);
		}
	}
	else  //cx q[1],t[1]
	{
		prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[reg_2.reg_index]);
	}
}

void QASMToQProg::build_zero_param_double_circuit(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_zero_param_double_circuit_func.find(type);
	if (iter_func == m_zero_param_double_circuit_func.end())
	{
		QCERR(type + " gate type is not supported!");
		throw runtime_error(type + " gate type is not supported!");
	}
	if (2 != op_info.regs_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}

	auto reg_1 = op_info.regs_vec[0];
	auto reg_2 = op_info.regs_vec[1];

	QVec qv_1 = find_qvec_map_value(reg_1.reg_name);
	QVec qv_2 = find_qvec_map_value(reg_2.reg_name);

	if (-1 == reg_1.reg_index && -1 == reg_2.reg_index)  //  cy q, t
	{
		if (qv_1.size() != qv_2.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[i]);
		}
	}
	else  if (-1 == reg_1.reg_index)  //cy q,t[1]
	{
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[reg_2.reg_index]);
		}
	}
	else if (-1 == reg_2.reg_index)  // cy q[1],t
	{
		for (int i = 0; i < qv_2.size(); i++)
		{
			prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[i]);
		}
	}
	else  //cy q[1],t[1]
	{
		prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[reg_2.reg_index]);
	}
}

void QASMToQProg::build_one_param_double_circuit(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_one_param_double_circuit_func.find(type);
	if (iter_func == m_one_param_double_circuit_func.end())
	{
		QCERR(type + " gate type is not supported!");
		throw runtime_error(type + " gate type is not supported!");
	}
	if (2 != op_info.regs_vec.size() || 1 != op_info.angles_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}

	auto reg_1 = op_info.regs_vec[0];
	auto reg_2 = op_info.regs_vec[1];

	QVec qv_1 = find_qvec_map_value(reg_1.reg_name);
	QVec qv_2 = find_qvec_map_value(reg_2.reg_name);
	double angle = op_info.angles_vec[0]->eval();
	if (-1 == reg_1.reg_index  && -1 == reg_2.reg_index)  //  cu1(lambda) a,b
	{
		if (qv_1.size() != qv_2.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[i], angle);
		}
	}
	else  if (-1 == reg_1.reg_index)  //cu1(lambda) a,b[1]
	{
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[reg_2.reg_index], angle);
		}
	}
	else if (-1 == reg_2.reg_index)  // cu1(lambda) a[1],b
	{
		for (int i = 0; i < qv_2.size(); i++)
		{
			prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[i], angle);
		}
	}
	else  //cu1(lambda) a[1],b[2]
	{
		prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[reg_2.reg_index], angle);
	}
}

void QASMToQProg::build_three_param_double_circuit(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_three_param_double_circuit_func.find(type);
	if (iter_func == m_three_param_double_circuit_func.end())
	{
		QCERR(type + " gate type is not supported!");
		throw runtime_error(type + " gate type is not supported!");
	}
	if (2 != op_info.regs_vec.size() || 3 != op_info.angles_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}

	auto reg_1 = op_info.regs_vec[0];
	auto reg_2 = op_info.regs_vec[1];

	QVec qv_1 = find_qvec_map_value(reg_1.reg_name);
	QVec qv_2 = find_qvec_map_value(reg_2.reg_name);
	double angle_1 = op_info.angles_vec[0]->eval();
	double angle_2 = op_info.angles_vec[1]->eval();
	double angle_3 = op_info.angles_vec[2]->eval();

	if (-1 == reg_1.reg_index && -1 == reg_2.reg_index)  //  cu3(theta,phi,lambda) c, t
	{
		if (qv_1.size() != qv_2.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[i], angle_1, angle_2, angle_3);
		}
	}
	else  if (-1 == reg_1.reg_index)  //cu3(theta,phi,lambda) c, t[1]
	{
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[reg_2.reg_index], angle_1, angle_2, angle_3);
		}
	}
	else if (-1 == reg_2.reg_index)  // cu3(theta,phi,lambda) c[1], t
	{
		for (int i = 0; i < qv_2.size(); i++)
		{
			prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[i], angle_1, angle_2, angle_3);
		}
	}
	else  //cu3(theta,phi,lambda) c[1], t[2]
	{
		prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[reg_2.reg_index], angle_1, angle_2, angle_3);
	}
}

void QASMToQProg::build_zero_param_triple_gate(QASMGateType type, GateOperationInfo op_info, QProg  &prog)
{
	auto iter_func = m_zero_param_triple_gate_func.find(type);
	if (iter_func == m_zero_param_triple_gate_func.end())
	{
		QCERR("gate type is not supported!");
		throw runtime_error("gate type is not supported!");
	}
	if (3 != op_info.regs_vec.size())
	{
		QCERR("parameter number error !");
		throw runtime_error("parameter number error!");
	}

	auto reg_1 = op_info.regs_vec[0];
	auto reg_2 = op_info.regs_vec[1];
	auto reg_3 = op_info.regs_vec[2];

	QVec qv_1 = find_qvec_map_value(reg_1.reg_name);
	QVec qv_2 = find_qvec_map_value(reg_2.reg_name);
	QVec qv_3 = find_qvec_map_value(reg_3.reg_name);

	if (-1 == reg_1.reg_index
		&&  -1 == reg_2.reg_index
		&& -1 == reg_3.reg_index)  //ccx a, b, c
	{
		if (qv_1.size() != qv_2.size()
			&& qv_1.size() != qv_3.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[i], qv_3[i]);
		}
	}
	else if (-1 == reg_1.reg_index
		&& -1 == reg_2.reg_index)  // ccx a, b, c[1]
	{
		if (qv_1.size() != qv_2.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[i], qv_3[reg_3.reg_index]);
		}
	}
	else if (-1 == reg_1.reg_index
		&& -1 == reg_3.reg_index)  //ccx a, b[1], c
	{
		if (qv_1.size() != qv_3.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}

		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[reg_2.reg_index], qv_3[i]);
		}
	}
	else if (-1 == reg_2.reg_index
		&& -1 == reg_3.reg_index)  // //ccx a[1], b, c
	{
		if (qv_2.size() != qv_3.size())
		{
			QCERR(" not supported!");
			throw runtime_error(" not supported!");
		}
		for (int i = 0; i < qv_2.size(); i++)
		{
			prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[i], qv_3[i]);
		}
	}
	else if (-1 == reg_1.reg_index)
	{
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[i], qv_2[reg_2.reg_index], qv_3[reg_3.reg_index]);
		}
	}
	else if (-1 == reg_2.reg_index)
	{
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[i], qv_3[reg_3.reg_index]);
		}
	}
	else if (-1 == reg_3.reg_index)
	{
		for (int i = 0; i < qv_1.size(); i++)
		{
			prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[reg_2.reg_index], qv_3[i]);
		}
	}
	else
	{
		prog << iter_func->second(qv_1[reg_1.reg_index], qv_2[reg_2.reg_index], qv_3[reg_3.reg_index]);
	}

}

void QASMToQProg::execute_gate_function(GateOperationInfo op_info, QProg  &prog)
{
	// gate func , formal -> actual parameter 
	auto iter_func_info = m_gate_func_map.find(op_info.op_id);
	if (iter_func_info == m_gate_func_map.end())
	{
		QCERR("operation id error, not find the function!");
		throw runtime_error("operation id error, not find the function!");
	}
	GataFuncInfo func_info = iter_func_info->second;
	if (func_info.ops_vec.size() == 0)
	{
		return;
	}

	if (op_info.regs_vec.size() != func_info.reg_names_vec.size())
	{
		QCERR("function reg params error!");
		throw runtime_error("function reg params error!");
	}
	std::map<string, RegParamInfo> reg_formal_actual_param_map;
	for (int i = 0; i < op_info.regs_vec.size(); i++)
	{
		reg_formal_actual_param_map.insert(make_pair(func_info.reg_names_vec[i], op_info.regs_vec[i]));
	}

	if (op_info.angles_vec.size() != func_info.angle_names_vec.size())
	{
		QCERR("function angle params error!");
		throw runtime_error("function angle params error!");
	}
	std::map<string, double> angle_formal_actual_param_map;
	for (int i = 0; i < op_info.angles_vec.size(); i++)
	{
		angle_formal_actual_param_map.insert(make_pair(func_info.angle_names_vec[i], op_info.angles_vec[i]->eval()));
	}

	for (auto op : func_info.ops_vec)
	{
		GateOperationInfo cur_gate_op;
		cur_gate_op.op_id = op.op_id;
		for (auto reg_formal : op.regs_vec)
		{
			RegParamInfo reg_actual_param;
			auto iter = reg_formal_actual_param_map.find(reg_formal.reg_name);
			if (iter == reg_formal_actual_param_map.end())
			{
				QCERR("error!");
				throw runtime_error("error!");
			}
			reg_actual_param = iter->second;
			cur_gate_op.regs_vec.push_back(reg_actual_param);
		}

		for (auto angle_param : op.angles_vec)
		{
			if (!angle_formal_actual_param_map.empty())
				angle_param->set_formal_actual_var_map(angle_formal_actual_param_map);

			cur_gate_op.angles_vec.push_back(angle_param);
		}

		build_qprog(cur_gate_op, prog);
	}
}

void QASMToQProg::build_qprog(GateOperationInfo op_info, QProg  &prog)
{
	auto iter_type = m_qasm_gate_type.find(op_info.op_id);
	if (iter_type == m_qasm_gate_type.end()) //gate func
	{
		execute_gate_function(op_info, prog);
	}
	else  // QASM basal gate
	{
		switch (iter_type->second)
		{
		case QASMGateType::SDG_GATE:
		case QASMGateType::TDG_GATE:
		{
			build_zero_param_single_gate(iter_type->second, true, op_info, prog);
		}
		break;
		case QASMGateType::X_GATE:
		case QASMGateType::Y_GATE:
		case QASMGateType::Z_GATE:
		case QASMGateType::H_GATE:
		case QASMGateType::S_GATE:
		case QASMGateType::T_GATE:
		case QASMGateType::ID_GATE:
		{
			build_zero_param_single_gate(iter_type->second, false, op_info, prog);
		}
		break;

		case QASMGateType::U1_GATE:
		case QASMGateType::RX_GATE:
		case QASMGateType::RY_GATE:
		case QASMGateType::RZ_GATE:
		{
			build_one_param_single_gate(iter_type->second, op_info, prog);
		}
		break;

		case QASMGateType::U2_GATE:
		{
			build_two_param_single_gate_func(iter_type->second, op_info, prog);
		}
		break;
		case QASMGateType::U_BASE_GATE:
		case QASMGateType::U3_GATE:
		{
			build_three_param_single_gate(iter_type->second, op_info, prog);
		}
		break;

		case QASMGateType::CX_BASE_GATE:
		case QASMGateType::CX_GATE:
		case QASMGateType::CZ_GATE:
		{
			build_zero_param_double_gate(iter_type->second, op_info, prog);
		}
		break;

		case QASMGateType::CY_GATE:
		case QASMGateType::CH_GATE:
		{
			build_zero_param_double_circuit(iter_type->second, op_info, prog);
		}
		break;

		case QASMGateType::CCX_GATE:
		{
			build_zero_param_triple_gate(iter_type->second, op_info, prog);
		}
		break;

		case QASMGateType::CRZ_GATE:
		case QASMGateType::CU1_GATE:
		{
			build_one_param_double_circuit(iter_type->second, op_info, prog);
		}
		break;

		case QASMGateType::CU3_GATE:
		{
			build_three_param_double_circuit(iter_type->second, op_info, prog);
		}
		break;
		default:
		{
			QCERR("qasm gate type error!");
			throw runtime_error("qasm gate type error!");
		}
		break;
		}
	}
}

QProg QASMToQProg::get_qprog()
{
	return m_build_qprog;
}

QProg QPanda::convert_qasm_to_qprog(std::string file_path, QuantumMachine* qvm)
{
	QVec qv;
	std::vector<ClassicalCondition> cv;
	return convert_qasm_to_qprog(file_path, qvm, qv, cv);
}

QProg QPanda::convert_qasm_to_qprog(std::string file_path, QuantumMachine* qvm, QVec &qv, std::vector<ClassicalCondition> &cv)
{
	std::ifstream stream;
	stream.open(file_path);
	if (!stream)
	{
		QCERR("File opening fail");
		throw invalid_argument("File opening fail");
	}
	antlr4::ANTLRInputStream input(stream);
	stream.close();
	qasmLexer lexer(&input);
	antlr4::CommonTokenStream tokens(&lexer);
	qasmParser parser(&tokens);

	antlr4::tree::ParseTree *tree = parser.mainprogram();
	QASMToQProg visitor(qvm , qv, cv);
	try
	{
		visitor.visit(tree);
	}
	catch (const std::exception&e)
	{
		QCERR(e.what());
		throw e;
	}

	return visitor.get_qprog();
}

QProg QPanda::convert_qasm_string_to_qprog(std::string qasm_str, QuantumMachine* qvm)
{
	QVec qv;
	std::vector<ClassicalCondition> cv;
	return convert_qasm_string_to_qprog(qasm_str, qvm, qv, cv);
}

QProg QPanda::convert_qasm_string_to_qprog(std::string qasm_str, QuantumMachine* qvm, QVec &qv, std::vector<ClassicalCondition> &cv)
{
	antlr4::ANTLRInputStream input(qasm_str);
	qasmLexer lexer(&input);
	antlr4::CommonTokenStream tokens(&lexer);
	qasmParser parser(&tokens);

	antlr4::tree::ParseTree *tree = parser.mainprogram();
	QASMToQProg visitor(qvm, qv, cv);
	try
	{
		visitor.visit(tree);
	}
	catch (const std::exception&e)
	{
		QCERR(e.what());
		throw e;
	}

	return visitor.get_qprog();
}



