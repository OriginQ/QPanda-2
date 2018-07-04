#include "QGateCompare.h"

map<int, string>  _G_gateTypeInt_map_gateTypeStr = {
	{ PAULI_X_GATE,		"X" },
	{ PAULI_Y_GATE,		"Y" },
	{ PAULI_Z_GATE,		"Z" },
	{ HADAMARD_GATE,	"H" },
	{ T_GATE,			"T" },
	{ S_GATE,			"S" },
	{ RX_GATE,			"RX" },
	{ RY_GATE,			"RY" },
	{ RZ_GATE,			"RZ" },
	{ U1_GATE,			"U1" },
	{ U2_GATE,			"U2" },
	{ U3_GATE,			"U3" },
	{ U4_GATE,			"U4" },
	{ CU_GATE,			"CU" },
	{ CNOT_GATE,		"CNOT" },
	{ CZ_GATE,			"CZ" },
	{ CPHASE_GATE,		"CPHASE" },
	{ ISWAP_THETA_GATE,	"ISWAP_THETA" },
	{ ISWAP_GATE,		"ISWAP" },
	{ SQISWAP_GATE,		"SQISWAP" },
	{ TWO_QUBIT_GATE,	"TWO_QUBIT" }
};

QGateCompare::QGateCompare()
{
	
}

QGateCompare::~QGateCompare()
{
}

size_t QGateCompare::countQGateNotSupport(AbstractQGateNode * PQGata, const vector<vector<string>>& instructionSet)
{
	if (nullptr == PQGata)
	{
		throw exception();
	}

	size_t iCount = 0;
	int qopNum = PQGata->getQGate()->getOpNum();
	int gateType = PQGata->getQGate()->getGateType();

	auto iter = _G_gateTypeInt_map_gateTypeStr.find(gateType);
	if (iter == _G_gateTypeInt_map_gateTypeStr.end())
	{
		throw exception();
	}

	string item = iter->second;
	if (!isItemExist(item, instructionSet))
	{
		iCount++;
	}

	return iCount;
}

size_t QGateCompare::countQGateNotSupport(AbstractQuantumProgram *pQPro, const vector<vector<string>>& instructionSet)
{
	if (nullptr == pQPro)
	{
		throw param_error_exception("pQPro is null", false);
	}

	size_t iCount = 0;
	for (auto aiter = pQPro->getFirstNodeIter(); aiter != pQPro->getEndNodeIter(); aiter++)
	{
		QNode * pNode = *aiter;
		iCount += countQGateNotSupport(pNode, instructionSet);
	}

	return iCount;
}


size_t QGateCompare::countQGateNotSupport(AbstractControlFlowNode * pCtr, const vector<vector<string>>& instructionSet)
{
	if (nullptr == pCtr)
	{
		throw param_error_exception("pCtr is null", false);
	}

	QNode *pNode = dynamic_cast<QNode *>(pCtr);
	if (nullptr == pNode)
	{
		throw param_error_exception("pNode is null", false);
	}

	size_t iCount = 0;
	QNode *pTrueBranchNode = pCtr->getTrueBranch();

	if (nullptr != pTrueBranchNode)
	{
		iCount += countQGateNotSupport(pTrueBranchNode, instructionSet);
	}

	if (NodeType::QIF_START_NODE == pNode->getNodeType())
	{
		QNode *pFalseBranchNode = pCtr->getFalseBranch();
		if (nullptr != pFalseBranchNode)
		{
			iCount += countQGateNotSupport(pFalseBranchNode, instructionSet);
		}
	}

	return iCount;
}


size_t QGateCompare::countQGateNotSupport(AbstractQuantumCircuit * pCircuit, const vector<vector<string>>& instructionSet)
{
	if (nullptr == pCircuit)
	{
		throw param_error_exception("pCircuit is null", false);
	}

	size_t iCount = 0;
	for (auto aiter = pCircuit->getFirstNodeIter(); aiter != pCircuit->getEndNodeIter(); aiter++)
	{
		QNode * pNode = *aiter;
		iCount += countQGateNotSupport(pNode, instructionSet);
	}

	return iCount;
}

size_t QGateCompare::countQGateNotSupport(QNode * pNode, const vector<vector<string>>& instructionSet)
{
	if (nullptr == pNode)
	{
		throw param_error_exception("pNode is null", false);
	}

	size_t iCount = 0;
	int type = pNode->getNodeType();
	switch (type)
	{
	case NodeType::GATE_NODE :
		iCount += countQGateNotSupport(dynamic_cast<AbstractQGateNode *>(pNode), instructionSet);
		break;
	case NodeType::CIRCUIT_NODE:
		iCount += countQGateNotSupport(dynamic_cast<AbstractQuantumCircuit *>(pNode), instructionSet);
		break;
	case NodeType::PROG_NODE:
		iCount += countQGateNotSupport(dynamic_cast<AbstractQuantumProgram *>(pNode), instructionSet);
		break;
	case NodeType::QIF_START_NODE:
	case NodeType::WHILE_START_NODE:
		iCount += countQGateNotSupport(dynamic_cast<AbstractControlFlowNode *>(pNode), instructionSet);
		break;
	case NodeType::MEASURE_GATE:
		break;
	case NodeType::NODE_UNDEFINED:
		break;
	default:
		throw exception();
		break;
	}

	return iCount;
}


bool QGateCompare::isItemExist(const string & item, const vector<vector<string>>& instructionSet)
{
	for (auto &vec : instructionSet)
	{
		for (auto val : vec)
		{
			if (item == val)
			{
				return true;
			}
		}
	}

	return false;
}

