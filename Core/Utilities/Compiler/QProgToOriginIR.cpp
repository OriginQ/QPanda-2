#include "Core/Utilities/Compiler/QProgToOriginIR.h"

using namespace std;
using namespace QGATE_SPACE;
USING_QPANDA

static void traversalInOrderPCtr(const CExpr* pCtrFlow, string &ctr_statement)
{
	if (nullptr != pCtrFlow)
	{
		traversalInOrderPCtr(pCtrFlow->getLeftExpr(), ctr_statement);
		string str_expr = pCtrFlow->getName();
		if (str_expr.at(0) == 'c')  //c0 --> c[0]
		{
			str_expr = "c[" + str_expr.substr(1) + "]";
		}
		ctr_statement = ctr_statement + str_expr;
		traversalInOrderPCtr(pCtrFlow->getRightExpr(), ctr_statement);
	}
}

static string transformQubitFormat(Qubit *qubit)
{
	string str_qubit;
	string str_expr;

	auto qubit_ref = dynamic_cast<QubitReferenceInterface *>(qubit);
	if (qubit_ref)  //q[c[1]] 
	{
		qubit->getPhysicalQubitPtr()->getQubitAddr();
		traversalInOrderPCtr(qubit_ref->getExprPtr().get(), str_expr);
		str_qubit = "q[" + str_expr + "]";
	}
	else   //q[1]
	{
		str_qubit = "q[" + to_string(qubit->getPhysicalQubitPtr()->getQubitAddr()) + "]";
	}

	return str_qubit;
}

QProgToOriginIR::QProgToOriginIR(QuantumMachine * quantum_machine)
{
	m_gatetype.insert(pair<int, string>(PAULI_X_GATE, "X"));
	m_gatetype.insert(pair<int, string>(PAULI_Y_GATE, "Y"));
	m_gatetype.insert(pair<int, string>(PAULI_Z_GATE, "Z"));

	m_gatetype.insert(pair<int, string>(X_HALF_PI, "X1"));
	m_gatetype.insert(pair<int, string>(Y_HALF_PI, "Y1"));
	m_gatetype.insert(pair<int, string>(Z_HALF_PI, "Z1"));
	m_gatetype.insert(pair<int, string>(I_GATE, "I"));
    m_gatetype.insert(pair<int, string>(HADAMARD_GATE, "H"));
	m_gatetype.insert(pair<int, string>(T_GATE, "T"));
	m_gatetype.insert(pair<int, string>(S_GATE, "S"));

	m_gatetype.insert(pair<int, string>(RX_GATE, "RX"));
	m_gatetype.insert(pair<int, string>(RY_GATE, "RY"));
	m_gatetype.insert(pair<int, string>(RZ_GATE, "RZ"));

	m_gatetype.insert(pair<int, string>(U1_GATE, "U1"));
	m_gatetype.insert(pair<int, string>(U2_GATE, "U2"));
	m_gatetype.insert(pair<int, string>(U3_GATE, "U3"));
	m_gatetype.insert(pair<int, string>(U4_GATE, "U4"));

	m_gatetype.insert(pair<int, string>(CU_GATE, "CU"));
	m_gatetype.insert(pair<int, string>(CNOT_GATE, "CNOT"));
	m_gatetype.insert(pair<int, string>(CZ_GATE, "CZ"));
	m_gatetype.insert(pair<int, string>(CPHASE_GATE, "CR"));
	m_gatetype.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
	m_gatetype.insert(pair<int, string>(SWAP_GATE, "SWAP"));
	m_gatetype.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));
	m_gatetype.insert(pair<int, string>(TWO_QUBIT_GATE, "QDoubleGate"));

	m_gatetype.insert(pair<int, string>(TOFFOLI_GATE, "TOFFOLI"));
	m_gatetype.insert(pair<int, string>(ORACLE_GATE, "ORACLE_GATE"));

	m_OriginIR.clear();

	m_quantum_machine = quantum_machine;
}

void QProgToOriginIR::transformQGate(AbstractQGateNode * pQGate, bool is_dagger)
{
	if (nullptr == pQGate || nullptr == pQGate->getQGate())
	{
		QCERR("pQGate is null");
		throw invalid_argument("pQGate is null");
	}

	QVec qubits_vector;
	QVec ctr_qubits_vector;
	bool is_toffoli = false;
	string all_ctr_qubits;
	pQGate->getQuBitVector(qubits_vector);
	pQGate->getControlVector(ctr_qubits_vector);

	if (PAULI_X_GATE == pQGate->getQGate()->getGateType() 
		&& 2 == ctr_qubits_vector.size())
	{
		is_toffoli = true;
	}

	if (pQGate->isDagger())
	{
		m_OriginIR.emplace_back("DAGGER");
	}
	if (!ctr_qubits_vector.empty())
	{
		for (auto val : ctr_qubits_vector)
		{
			all_ctr_qubits = all_ctr_qubits + transformQubitFormat(val) + ",";
		}
		if(!is_toffoli)
		{ 
			m_OriginIR.emplace_back("CONTROL " + all_ctr_qubits.substr(0, all_ctr_qubits.length() - 1));
		}
	}
	auto iter = m_gatetype.find(pQGate->getQGate()->getGateType());
	if (iter == m_gatetype.end())
	{
		QCERR("unknown error");
		throw runtime_error("unknown error");
	}

	string first_qubit = transformQubitFormat(qubits_vector.front());
	string all_qubits;
	for (auto _val : qubits_vector)
	{
		all_qubits = all_qubits + transformQubitFormat(_val) + ",";
	}
	all_qubits = all_qubits.substr(0, all_qubits.length() - 1);

	string item = iter->second;
	switch (iter->first)
	{		
	case ORACLE_GATE: 
	{
		QGATE_SPACE::OracularGate *oracle_gate = dynamic_cast<QGATE_SPACE::OracularGate*>(pQGate->getQGate());
		m_OriginIR.emplace_back(oracle_gate->get_name() + " " + all_qubits);
	}
		break;
	case PAULI_X_GATE:
	{	
		if (is_toffoli)
		{
			string str_toffoli = m_gatetype.find(TOFFOLI_GATE)->second;
			m_OriginIR.emplace_back(str_toffoli + " " + all_ctr_qubits + first_qubit);
		}
		else
		{
			m_OriginIR.emplace_back(item + " " + first_qubit);
		}
	}
	break;
	case PAULI_Y_GATE:
	case PAULI_Z_GATE:
	case X_HALF_PI:
	case Y_HALF_PI:
	case Z_HALF_PI:
	case HADAMARD_GATE:
	case T_GATE:
	case S_GATE:
	case I_GATE:
		m_OriginIR.emplace_back(item + " " + first_qubit);
		break;
	case U4_GATE:
	{
		auto u4gate = dynamic_cast<AbstractAngleParameter *>(pQGate->getQGate());
		m_OriginIR.emplace_back(item
			+ " " + first_qubit + ","
			+ "(" + to_string(u4gate->getAlpha())
			+ "," + to_string(u4gate->getBeta())
			+ "," + to_string(u4gate->getGamma())
			+ "," + to_string(u4gate->getDelta())
			+ ") ");
	}
	break;

	case U1_GATE:
	case RX_GATE:
	case RY_GATE:
	case RZ_GATE:
	{
		auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter*>(pQGate->getQGate());
		string  gate_angle = to_string(gate_parameter->getParameter());
		m_OriginIR.emplace_back(item + " " + first_qubit + "," + "(" + gate_angle + ")");
	}
	break;

	case U2_GATE:
	{
		QGATE_SPACE::U2 *u2_gate = dynamic_cast<QGATE_SPACE::U2*>(pQGate->getQGate());
		string gate_two_angle = to_string(u2_gate->get_phi()) + ',' + to_string(u2_gate->get_lambda());
		m_OriginIR.emplace_back(item + " " + all_qubits + "," + "(" + gate_two_angle + ")");
	}
	break;
	case U3_GATE:
	{
		QGATE_SPACE::U3 *u3_gate = dynamic_cast<QGATE_SPACE::U3*>(pQGate->getQGate());
		string gate_three_angle = to_string(u3_gate->get_theta()) + ',' +to_string(u3_gate->get_phi()) + ',' + to_string(u3_gate->get_lambda());
		m_OriginIR.emplace_back(item + " " + all_qubits + "," + "(" + gate_three_angle + ")");
	}
	break;

	case CNOT_GATE:
	case CZ_GATE:
	case ISWAP_GATE:
	case SQISWAP_GATE:
	case SWAP_GATE:
	case TWO_QUBIT_GATE:
	{
		m_OriginIR.emplace_back(item + " " + all_qubits);
	}
	break;

	case CPHASE_GATE:
	{
		auto gate_parameter = dynamic_cast<AbstractSingleAngleParameter *>(pQGate->getQGate());
		string  gate_theta = to_string(gate_parameter->getParameter());
		m_OriginIR.emplace_back(item + " " + all_qubits + "," + "(" + gate_theta + ")");
	}
	break;

	case CU_GATE:
	{
		auto gate_parameter = dynamic_cast<AbstractAngleParameter *>(pQGate->getQGate());
		string gate_four_theta = to_string(gate_parameter->getAlpha()) + ',' +
			to_string(gate_parameter->getBeta()) + ',' +
			to_string(gate_parameter->getGamma()) + ',' +
			to_string(gate_parameter->getDelta());
		m_OriginIR.emplace_back(item + " " + all_qubits + "," + "(" + gate_four_theta + ")");
	}
	break;

	default:m_OriginIR.emplace_back("Unsupported GateNode");
	}

	if (!ctr_qubits_vector.empty() && !is_toffoli)
	{
		m_OriginIR.emplace_back("ENDCONTROL");
	}
	if (pQGate->isDagger())
	{
		m_OriginIR.emplace_back("ENDDAGGER");
	}
}

void QProgToOriginIR::transformQMeasure(AbstractQuantumMeasure *pMeasure)
{
	if (nullptr == pMeasure || nullptr == pMeasure->getQuBit()->getPhysicalQubitPtr())
	{
		QCERR("pMeasure is null");
		throw invalid_argument("pMeasure is null");
	}

	string tar_qubit = transformQubitFormat(pMeasure->getQuBit());

	string creg_name = pMeasure->getCBit()->getName();

	creg_name = "c[" + creg_name.substr(1) + "]";  //c0 --->  c[0]

	m_OriginIR.emplace_back("MEASURE " + tar_qubit + "," + creg_name);
}

void QProgToOriginIR::transformClassicalProg(AbstractClassicalProg *pClassicalProg)
{
	if (nullptr == pClassicalProg)
	{
		QCERR("pClassicalProg is null");
		throw invalid_argument("pClassicalProg is null");
		return;
	}
	string exper;
	auto expr = dynamic_cast<OriginClassicalProg *>(pClassicalProg)->getExpr().get();
	traversalInOrderPCtr(expr, exper);
	m_OriginIR.emplace_back(exper);
}

void QProgToOriginIR::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node)
{
	transformQGate(cur_node.get());
}

void QProgToOriginIR::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node)
{
	transformQMeasure(cur_node.get());
}

void QProgToOriginIR::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node)
{
	auto pNode = dynamic_pointer_cast<QNode>(cur_node);

	switch (pNode->getNodeType())
	{
	case NodeType::WHILE_START_NODE:
	{
		string exper;
		auto expr = cur_node->getCExpr().getExprPtr().get();

		traversalInOrderPCtr(expr, exper);
		if (exper.empty())
		{
			QCERR("expression is null!");
			throw invalid_argument("expression is null!");
		}

		m_OriginIR.emplace_back("QWHILE " + exper);
		auto while_branch_node = cur_node->getTrueBranch();
		if (nullptr != while_branch_node)
		{
			Traversal::traversalByType(while_branch_node, nullptr, *this);
		}
		m_OriginIR.emplace_back("ENDQWHILE");
	}
	break;

	case NodeType::QIF_START_NODE:
	{
		string exper;
		auto expr = cur_node->getCExpr().getExprPtr().get();

		traversalInOrderPCtr(expr, exper);
		if (exper.empty())
		{
			QCERR("expression is null!");
			throw invalid_argument("expression is null!");
		}
		m_OriginIR.emplace_back("QIF " + exper);
		auto truth_branch_node = cur_node->getTrueBranch();
		if (nullptr != truth_branch_node)
		{
			Traversal::traversalByType(truth_branch_node, nullptr, *this);
		}

		auto false_branch_node = cur_node->getFalseBranch();
		if (nullptr != false_branch_node)
		{
			m_OriginIR.emplace_back("ELSE");
			Traversal::traversalByType(false_branch_node, nullptr, *this);
		}
		m_OriginIR.emplace_back("ENDQIF");
	}
	break;
	}
}

void QProgToOriginIR::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node)
{
	Traversal::traversal(cur_node, *this);
}

void QProgToOriginIR::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node)
{
	if (cur_node->isDagger())
	{
		m_OriginIR.emplace_back("DAGGER");
	}
	QVec circuit_ctr_qubits;
	string all_ctr_qubits;
	cur_node->getControlVector(circuit_ctr_qubits);
	if (!circuit_ctr_qubits.empty())
	{
		for (auto val : circuit_ctr_qubits)
		{
			all_ctr_qubits = all_ctr_qubits + transformQubitFormat(val) + ",";
		}
		all_ctr_qubits = all_ctr_qubits.substr(0, all_ctr_qubits.length() - 1);
		m_OriginIR.emplace_back("CONTROL " + all_ctr_qubits);
	}

	Traversal::traversal(cur_node, false, *this);

	if (!circuit_ctr_qubits.empty())
	{
		m_OriginIR.emplace_back("ENDCONTROL");
	}
	if (cur_node->isDagger())
	{
		m_OriginIR.emplace_back("ENDDAGGER");
	}
}

void QProgToOriginIR::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node)
{
	transformClassicalProg(cur_node.get());
}

string QProgToOriginIR::getInsturctions()
{
	string instructions;

	for (auto &instruct_out : m_OriginIR)
	{
		instructions.append(instruct_out).append("\n");
	}
	instructions.erase(instructions.size() - 1);

	return instructions;
}




