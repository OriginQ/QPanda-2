#include "Core/Utilities/Compiler/OriginIRToQProg.h"
#include "Core/Utilities/QProgTransform/QProgToQCircuit.h"
#include "Core/Utilities/Tools/QPandaException.h"

using namespace std;
USING_QPANDA

QProg QPanda::convert_originir_to_qprog(std::string file_path, QuantumMachine *qm)
{
	QVec qv; 
	std::vector<ClassicalCondition> cv;
	return convert_originir_to_qprog(file_path, qm, qv, cv);
}

QProg QPanda::convert_originir_to_qprog(std::string file_path, QuantumMachine *qm, QVec &qv,  std::vector<ClassicalCondition> &cv)
{
	std::ifstream stream;
	stream.open(file_path);
	if (!stream)
	{
		QCERR_AND_THROW(run_fail, "Error: Filed to open originir file.");
	}
	try
	{
		antlr4::ANTLRInputStream input(stream);
		stream.close();
		originirLexer lexer(&input);
		antlr4::CommonTokenStream tokens(&lexer);
		originirParser parser(&tokens);

		antlr4::tree::ParseTree *tree = parser.translationunit();
		OriginIRVisitor visitor(qm, qv, cv);
		size_t fullprog = visitor.visit(tree);
		return visitor.get_qprog(fullprog);
	}
	catch (const std::exception&e)
	{
		QCERR_AND_THROW(run_fail, "Error: catch a exception: " << e.what());
	}
}

QProg QPanda::convert_originir_string_to_qprog(std::string str_originir, QuantumMachine *qm)
{
	QVec qv;
	std::vector<ClassicalCondition> cv;
	return convert_originir_string_to_qprog(str_originir, qm, qv, cv);
}

QProg QPanda::convert_originir_string_to_qprog(std::string str_originir, QuantumMachine *qm, QVec &qv, std::vector<ClassicalCondition> &cv)
{
	try
	{
		antlr4::ANTLRInputStream input(str_originir);
		originirLexer lexer(&input);
		antlr4::CommonTokenStream tokens(&lexer);
		originirParser parser(&tokens);

		antlr4::tree::ParseTree *tree = parser.translationunit();
		OriginIRVisitor visitor(qm, qv, cv);

		size_t fullprog = visitor.visit(tree);
		return visitor.get_qprog(fullprog);
	}
	catch (const std::exception&e)
	{
		QCERR(e.what());
		throw e;
	}
}


QProg QPanda::transformOriginIRToQProg(std::string filePath, QuantumMachine * qm, QVec &qv, std::vector<ClassicalCondition> &cv)
{
	std::ifstream stream;
	stream.open(filePath);
    if (!stream)
    {
        QCERR("File opening fail");
        throw invalid_argument("File opening fail");
    }

    try
    {
        antlr4::ANTLRInputStream input(stream);
		stream.close();
        originirLexer lexer(&input);
        antlr4::CommonTokenStream tokens(&lexer);
        originirParser parser(&tokens);

        antlr4::tree::ParseTree *tree = parser.translationunit();
        OriginIRVisitor visitor(qm, qv, cv);

        size_t fullprog = visitor.visit(tree);
        return visitor.get_qprog(fullprog);
    }
    catch (const std::exception&e)
    {
        QCERR(e.what());
        throw e;
    }
}

QProgBuilder::QProgBuilder(QuantumMachine * qm, QVec &qv, std::vector<ClassicalCondition> &cv)
	:m_machine(qm), qs(qv), ccs(cv)
{ }

QProg QProgBuilder::get_qprog()
{
	return m_progid_set[0];
}

void QProgBuilder::alloc_qubit(int num)
{
	qs = m_machine->allocateQubits(num);
}

void QProgBuilder::alloc_cbit(int num)
{
	ccs = m_machine->allocateCBits(num);
}

size_t QProgBuilder::add_prog()
{
	m_progid_set.insert({ qid, QProg() });
	return qid++;	
}

void QProgBuilder::insert_subprog(size_t progid_dst, size_t progid_src)
{
	m_progid_set[progid_dst] << m_progid_set[progid_src];
}

size_t QProgBuilder::add_qgate(GateType type, std::vector<int> index, std::vector<double> parameters)
{
	return add_qgate_cc(type, {}, index, parameters);
}

size_t QProgBuilder::add_qgate_cc(
	GateType type, 
	std::vector<size_t> exprid, 
	std::vector<int> index, 
	std::vector<double> parameters) {
	
	size_t progid = add_prog();
	vector<Qubit*> qubits;
	int counter = 0;
	for (int i = 0; i < index.size(); ++i) {
		if (index[i] == -1) {
			qubits.push_back(qs[m_exprid_set.at(exprid[counter])]);
			counter++;
		}
		else {
			if (index[i]+1 > qs.size())
				throw runtime_error("too little qubits is allocated");

			qubits.push_back(qs[index[i]]);
		}
	}
	switch (type) {
	case GateType::H:
		m_progid_set[progid] << H(qubits[0]);
		break;
	case GateType::ECHO:
		m_progid_set[progid] << ECHO(qubits[0]);
        break;
	case GateType::T:
		m_progid_set[progid] << T(qubits[0]);
		break;
	case GateType::S:
		m_progid_set[progid] << S(qubits[0]);
		break;
	case GateType::X:
		m_progid_set[progid] << X(qubits[0]);
		break;
	case GateType::Y:
		m_progid_set[progid] << Y(qubits[0]);
		break;
	case GateType::Z:
		m_progid_set[progid] << Z(qubits[0]);
		break;
	case GateType::X1:
		m_progid_set[progid] << X1(qubits[0]);
		break;
	case GateType::Y1:
		m_progid_set[progid] << Y1(qubits[0]);
		break;
	case GateType::Z1:
		m_progid_set[progid] << Z1(qubits[0]);
		break;
	case GateType::I:
		m_progid_set[progid] << I(qubits[0]);
		break;
	case GateType::RX:
		m_progid_set[progid] << RX(qubits[0], parameters[0]);
		break;
	case GateType::RY:
		m_progid_set[progid] << RY(qubits[0], parameters[0]);
		break;
	case GateType::RZ:
		m_progid_set[progid] << RZ(qubits[0], parameters[0]);
		break;
	case GateType::U1:
		m_progid_set[progid] << U1(qubits[0], parameters[0]);
		break;
	case GateType::U2:
		m_progid_set[progid] << U2(qubits[0], parameters[0], parameters[1]);
		break;
	case GateType::RPhi:
		m_progid_set[progid] << RPhi(qubits[0], parameters[0], parameters[1]);
		break;

	case GateType::U3:
		m_progid_set[progid] << U3(qubits[0], parameters[0], parameters[1], parameters[2]);
		break;
	case GateType::U4:
		m_progid_set[progid] << U4(parameters[0], parameters[1], 
			parameters[2], parameters[3], qubits[0]);
		break;

	case GateType::CNOT:
		m_progid_set[progid] << CNOT(qubits[0], qubits[1]);
		break;
	case GateType::CZ:
		m_progid_set[progid] << CZ(qubits[0], qubits[1]);
		break;
	case GateType::ISWAP:
		m_progid_set[progid] << iSWAP(qubits[0], qubits[1]);
		break;
	case GateType::SQISWAP:
		m_progid_set[progid] << SqiSWAP(qubits[0], qubits[1]);
		break;
	case GateType::SWAP:
        m_progid_set[progid] << SWAP(qubits[0], qubits[1]);
		break;

	case GateType::ISWAPTHETA:
		m_progid_set[progid] << iSWAP(qubits[0], qubits[1], parameters[0]);
		break;
	case GateType::CR:
		m_progid_set[progid] << CR(qubits[0], qubits[1], parameters[0]);
		break;
	case GateType::CU:
		m_progid_set[progid] << CU(parameters[0], parameters[1],
			parameters[2], parameters[3], qubits[0], qubits[1]);
		break;
	case GateType::TOFFOLI:
	{
		auto toffoli_gate = X(qubits[2]);
		toffoli_gate.setControl({ qubits[0], qubits[1] });
		m_progid_set[progid] << toffoli_gate;
	}
		break;
	default:
		throw runtime_error("Bad Argument.");
	}
	return progid;
}

size_t QProgBuilder::add_measure_literal(size_t qidx, size_t cidx)
{
	size_t progid = add_prog();
	if (ccs.size()  <  cidx+1)
		throw runtime_error("add_measure_literal too little cbits is allocated");

	m_progid_set[progid] << Measure(qs[qidx], ccs[cidx]);
	return progid;
}

size_t QProgBuilder::add_measure_cc(size_t exprid, size_t cidx)
{
	size_t progid = add_prog();

	if (ccs.size() < cidx + 1)
		throw runtime_error("add_measure_cc too little cbits is allocated");

	m_progid_set[progid] << Measure(qs[m_exprid_set.at(exprid)], ccs[cidx]);
	return progid;
}

size_t QProgBuilder::add_reset_literal(size_t qidx)
{
	size_t progid = add_prog();
	m_progid_set[progid] << Reset(qs[qidx]);
	return progid;
}

size_t QProgBuilder::add_reset_cc(size_t exprid)
{
	size_t progid = add_prog();
	m_progid_set[progid] << Reset(qs[m_exprid_set.at(exprid)]);
	return progid;
}

size_t QProgBuilder::add_barrier_literal(size_t qidx, QVec qv)
{
	size_t progid = add_prog();
	m_progid_set[progid] << BARRIER(qs[qidx]).control(qv);
	return progid;
}

size_t QProgBuilder::add_barrier_cc(size_t exprid, QVec qv)
{
	size_t progid = add_prog();
	m_progid_set[progid] << BARRIER(qs[m_exprid_set.at(exprid)]).control(qv);
	return progid;
}

size_t QProgBuilder::add_expr_stat(size_t exprid)
{
	size_t progid = add_prog();
	m_progid_set[progid] << m_exprid_set.at(exprid);
	return progid;
}

size_t QProgBuilder::make_qif(size_t exprid, size_t progid)
{
	size_t prog = add_prog();
	m_progid_set[prog] << CreateIfProg(m_exprid_set.at(exprid), m_progid_set[progid]);
	return prog;
}

size_t QProgBuilder::make_qifelse(size_t exprid, size_t progid_true, size_t progid_false)
{
	size_t progid = add_prog();
	m_progid_set[progid] << CreateIfProg(
		m_exprid_set.at(exprid), m_progid_set[progid_true], m_progid_set[progid_false]);
	return progid;
}

size_t QProgBuilder::make_qwhile(size_t exprid, size_t progid)
{
	size_t prog = add_prog();
	m_progid_set[prog] << CreateWhileProg(m_exprid_set.at(exprid), m_progid_set[progid]);
	return prog;
}

void QProgBuilder::delete_prog(size_t progid)
{
	m_progid_set.erase(progid);
}

size_t QProgBuilder::cc_init_id(size_t cidx)
{
	if (ccs.size() < cidx + 1)
		throw runtime_error("cc_init_id too little cbits is allocated");

	m_exprid_set.insert({ cid, ccs[cidx] });
	return cid++;
}

size_t QProgBuilder::cc_init_literal(double value)
{
	m_exprid_set.insert({ cid, ClassicalCondition((cbit_size_t)value) });
    return cid++;
}

size_t QProgBuilder::cc_op_cc(size_t exprid1, size_t exprid2, int op_type)
{
	switch (op_type) {
	case OriginIRVisitor::Plus:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) + m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::Minus:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) - m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::Mul:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) * m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::Div:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) / m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::LT:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) < m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::GT:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) > m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::LEQ:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) <= m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::GEQ:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) >= m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::EQ:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) == m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::NE:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) != m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::AND:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) && m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::OR:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) || m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::ASSIGN:
	{
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) = m_exprid_set.at(exprid2) });
		m_exprid_set.at(cid).get_val();
	}
		break;
	default:
		throw runtime_error("Bad Argument.");
	}
	return cid++;
}

size_t QProgBuilder::cc_op_literal(size_t exprid1, double literal2, int op_type)
{
	switch (op_type) {
	case OriginIRVisitor::Plus:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) + literal2 });
		break;
	case OriginIRVisitor::Minus:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) - literal2 });
		break;
	case OriginIRVisitor::Mul:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) * literal2 });
		break;
	case OriginIRVisitor::Div:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) / literal2 });
		break;
	case OriginIRVisitor::LT:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) < literal2 });
		break;
	case OriginIRVisitor::GT:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) > literal2 });
		break;
	case OriginIRVisitor::LEQ:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) <= literal2 });
		break;
	case OriginIRVisitor::GEQ:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) >= literal2 });
		break;
	case OriginIRVisitor::EQ:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) == literal2 });
		break;
	case OriginIRVisitor::NE:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) != literal2 });
		break;
	case OriginIRVisitor::AND:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) && literal2 });
		break;
	case OriginIRVisitor::OR:
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) || literal2 });
		break;
	case OriginIRVisitor::ASSIGN:
	{
		m_exprid_set.insert({ cid, m_exprid_set.at(exprid1) = literal2 });
		m_exprid_set.at(cid).get_val();
	}
		break;
	default:
		throw runtime_error("Bad Argument.");
	}
	return cid++;
}

size_t QProgBuilder::literal_op_cc(double literal1, size_t exprid2, int op_type)
{
	switch (op_type) {
	case OriginIRVisitor::Plus:
		m_exprid_set.insert({ cid, literal1 + m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::Minus:
		m_exprid_set.insert({ cid, literal1 - m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::Mul:
		m_exprid_set.insert({ cid, literal1 * m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::Div:
		m_exprid_set.insert({ cid, literal1 / m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::LT:
		m_exprid_set.insert({ cid, literal1 < m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::GT:
		m_exprid_set.insert({ cid, literal1 > m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::LEQ:
		m_exprid_set.insert({ cid, literal1 <= m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::GEQ:
		m_exprid_set.insert({ cid, literal1 >= m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::EQ:
		m_exprid_set.insert({ cid, literal1 == m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::NE:
		m_exprid_set.insert({ cid, literal1 != m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::AND:
		m_exprid_set.insert({ cid, literal1 && m_exprid_set.at(exprid2) });
		break;
	case OriginIRVisitor::OR:
		m_exprid_set.insert({ cid, literal1 || m_exprid_set.at(exprid2) });
		break;
	default:
		throw runtime_error("Bad Argument.");
	}
	return cid++;
}

size_t QProgBuilder::op_cc(size_t exprid, int op_type)
{
	switch (op_type) {
	case OriginIRVisitor::UnaryPlus:
		return literal_op_cc(0, exprid, OriginIRVisitor::Plus);
		break;
	case OriginIRVisitor::UnaryMinus:
		return literal_op_cc(0, exprid, OriginIRVisitor::Minus);
		break;
	case OriginIRVisitor::UnaryNot:
		m_exprid_set.insert({ cid, !m_exprid_set.at(exprid) });
		return cid++;
		break;
	default:
		throw runtime_error("Bad Argument.");
	}
}

void QProgBuilder::make_dagger(size_t progid)
{
	QCircuit c;
	if (cast_qprog_qcircuit(m_progid_set[progid], c)) {
		c.setDagger(true);
		m_progid_set[progid] = c;
	}
	else {
		throw runtime_error("Non-Circuit Components when daggering.");
	}
}

size_t QProgBuilder::make_dagger_new(size_t progid)
{
	QCircuit c;
	if (cast_qprog_qcircuit(m_progid_set[progid], c)) {
		c.setDagger(true);
		m_progid_set.insert({ qid, c });
		return qid++;
	}
	else {
		throw runtime_error("Non-Circuit Components when daggering.");
	}
}

QVec QProgBuilder::make_qvec( std::vector<size_t> expridx, std::vector<int> idx)
{
	QVec q;
	int counter = 0;
	for (int i = 0; i < idx.size(); ++i) 
	{
		if (idx[i] != -1)
			q.push_back(qs[idx[i]]);
		else {
			q.push_back(qs[m_exprid_set.at(expridx[counter++])]);
		}
	}
	return q;
}


void QProgBuilder::make_control(size_t progid, std::vector<int> idx)
{
	make_control_cc(progid, {}, idx);
}

size_t QProgBuilder::make_control_new(size_t progid, std::vector<int> idx)
{
	return make_control_cc_new(progid, {}, idx);
}

void QProgBuilder::make_control_cc(size_t progid, std::vector<size_t> expridx, std::vector<int> idx)
{
	QCircuit c;
	QVec q;
	int counter = 0;
	for (int i = 0; i < idx.size(); ++i) {
		if (idx[i] != -1)
			q.push_back(qs[idx[i]]);
		else {
			q.push_back(qs[m_exprid_set.at(expridx[counter++])]);
		}
	}
	if (cast_qprog_qcircuit(m_progid_set[progid], c)) {
		c.setControl(q);
		m_progid_set[progid] = c;
	}
	else {
		throw runtime_error("Non-Circuit Components when controlling.");
	}
}

size_t QProgBuilder::make_control_cc_new(size_t progid, std::vector<size_t> expridx, std::vector<int> idx)
{
	QCircuit c;
	QVec q;
	int counter = 0;
	for (int i = 0; i < idx.size(); ++i) {
		if (idx[i]!=-1)
			q.push_back(qs[i]);
		else {
			q.push_back(qs[m_exprid_set.at(expridx[counter++])]);
		}
	}
	if (cast_qprog_qcircuit(m_progid_set[progid], c)) {
		c.setControl(q);
		m_progid_set.insert({ qid, c });
		return qid++;
	}
	else {
		throw runtime_error("Non-Circuit Components when controlling.");
	}
}
