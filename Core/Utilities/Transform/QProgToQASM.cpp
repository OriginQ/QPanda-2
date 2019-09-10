#include "Core/Utilities/MetadataValidity.h"
#include "Core/Utilities/Transform/QProgToQASM.h"
#include "Core/Utilities/Transform/TransformDecomposition.h"
#include "Core/Utilities/Transform/backends/IBMQ.h"
#include "QPanda.h"

using namespace std;
USING_QPANDA

#define ENUM_TO_STR(x) #x
#ifndef MAX_PATH
#define MAX_PATH 260
#endif

#define QASM_HEAD "OPENQASM 2.0;"
#define QASM_UNSUPPORT_EXCEPTIONAL(gate, dg) {\
    std::string excepStr;\
    if (dg){\
        excepStr = string("Error: Qasm unsport: ") + gate + "dg";\
    }else{excepStr = string("Error: Qasm unsport: ") + gate;}\
	QCERR(excepStr.c_str());\
	throw std::invalid_argument(excepStr.c_str());\
}

QProgToQASM::QProgToQASM(QuantumMachine * quantum_machine, IBMQBackends ibmBackend/* = IBMQ_QASM_SIMULATOR*/)
{
	_ibmBackend = ibmBackend;
    m_gatetype.insert(pair<int, string>(PAULI_X_GATE, "X"));
    m_gatetype.insert(pair<int, string>(PAULI_Y_GATE, "Y"));
    m_gatetype.insert(pair<int, string>(PAULI_Z_GATE, "Z"));

    m_gatetype.insert(pair<int, string>(X_HALF_PI, "X1"));
    m_gatetype.insert(pair<int, string>(Y_HALF_PI, "Y1"));
    m_gatetype.insert(pair<int, string>(Z_HALF_PI, "Z1"));

    m_gatetype.insert(pair<int, string>(HADAMARD_GATE, "H"));
    m_gatetype.insert(pair<int, string>(T_GATE, "T"));
    m_gatetype.insert(pair<int, string>(S_GATE, "S"));

    m_gatetype.insert(pair<int, string>(RX_GATE, "RX"));
    m_gatetype.insert(pair<int, string>(RY_GATE, "RY"));
    m_gatetype.insert(pair<int, string>(RZ_GATE, "RZ"));
    m_gatetype.insert(pair<int, string>(U1_GATE, "U1"));

    m_gatetype.insert(pair<int, string>(CU_GATE, "CU"));
    m_gatetype.insert(pair<int, string>(CNOT_GATE, "CNOT"));
    m_gatetype.insert(pair<int, string>(CZ_GATE, "CZ"));
    m_gatetype.insert(pair<int, string>(CPHASE_GATE, "CR"));
	m_gatetype.insert(pair<int, string>(SWAP_GATE, "SWAP"));
    m_gatetype.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_gatetype.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));

    m_qasm.clear();
    m_quantum_machine = quantum_machine;
}

void QProgToQASM::transform(QProg &prog)
{
    m_qasm.emplace_back(QASM_HEAD);
	m_qasm.emplace_back("include \"qelib1.inc\";");
    m_qasm.emplace_back("qreg q[" + to_string(m_quantum_machine->getAllocateQubit()) + "];");
    m_qasm.emplace_back("creg c[" + to_string(m_quantum_machine->getAllocateCMem()) + "];");
    if (nullptr == m_quantum_machine)
    {
        QCERR("Quantum machine is nullptr");
        throw std::invalid_argument("Quantum machine is nullptr");
    }

    const int KMETADATA_GATE_TYPE_COUNT = 2;
    vector<vector<string>> ValidQGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
    vector<vector<string>> QGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));

	//根据后端类型加载拓扑结构
	vector<vector<int>> vAdjacentMatrix;
	int qubitsNum = 0;
	string dataElementStr = getIBMQBackendName(_ibmBackend);
	loadIBMQuantumTopology(string(IBMQ_BACKENDS_CONFIG), dataElementStr, qubitsNum, vAdjacentMatrix);

	//判断类型是否合法，Qubit数目是否匹配
	if (m_quantum_machine->getAllocateQubit() > qubitsNum)
	{
		QCERR("Quantum machine has too many qubits");
		throw std::invalid_argument("Quantum machine has too many qubits");
		return;
	}

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[PAULI_X_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[PAULI_Y_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[PAULI_Z_GATE]);

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[HADAMARD_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[T_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[S_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[RX_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[RY_GATE]);

    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[RZ_GATE]);
    QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[U1_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CU_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CNOT_GATE]);

    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CZ_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CPHASE_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[ISWAP_GATE]);

    SingleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
    DoubleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */
    TransformDecomposition traversal_vector(ValidQGateMatrix, QGateMatrix, vAdjacentMatrix, m_quantum_machine);

    
    traversal_vector.TraversalOptimizationMerge(prog);

	transformQProgByTraversalAlg(&prog);
}

std::string QProgToQASM::getIBMQBackendName(IBMQBackends typeNum)
{
	std::string backendName;
	switch (typeNum)
	{
	case IBMQ_QASM_SIMULATOR:
		backendName = ENUM_TO_STR(IBMQ_QASM_SIMULATOR);
		break;
		
	case IBMQ_16_MELBOURNE:
		backendName = ENUM_TO_STR(IBMQ_16_MELBOURNE);
		break;

	case IBMQX2:
		backendName = ENUM_TO_STR(IBMQX2);
		break;

	case IBMQX4:
		backendName = ENUM_TO_STR(IBMQX4);
		break;

	default:
		//error
		break;
	}

	std::transform(backendName.begin(), backendName.end(), backendName.begin(), ::tolower);
	return backendName;
}

void QProgToQASM::transformQGate(AbstractQGateNode * pQGate,bool is_dagger)
{
    if (nullptr == pQGate || nullptr == pQGate->getQGate())
    {
        QCERR("pQGate is null");
        throw invalid_argument("pQGate is null");
    }

    QVec qubits_vector;
    pQGate->getQuBitVector(qubits_vector);
    auto iter = m_gatetype.find(pQGate->getQGate()->getGateType());
	if (iter == m_gatetype.end())
	{
		return;
	}
    string tarQubit = to_string(qubits_vector.front()->getPhysicalQubitPtr()->getQubitAddr());
    string all_qubits;
    for (auto _val : qubits_vector)
    {
        all_qubits = all_qubits + "q[" + to_string(_val->getPhysicalQubitPtr()->getQubitAddr()) + "]" + ",";
    }
    all_qubits = all_qubits.substr(0, all_qubits.length() - 1);
	auto dagger = pQGate->isDagger() ^ is_dagger;
    string sTemp = iter->second;
	int iLabel = dagger ? -1 : 1;   /* iLabel is 1 or -1 */
	char tmpStr[MAX_PATH] = "";
    switch (iter->first)
    {
        case PAULI_X_GATE:
        case PAULI_Y_GATE:
        case PAULI_Z_GATE:
        case HADAMARD_GATE:
		{
			//ignore dagger，because dagger is equal to itself
			sTemp.append(" q[" + tarQubit + "];");
		}
		break;

		case X_HALF_PI:
		{
			string  gate_angle = to_string((dynamic_cast<angleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
			sTemp = ("rx(" + gate_angle + ")");
			sTemp.append(" q[" + tarQubit + "];");
		}
			break;

		case Y_HALF_PI:
		{
			string  gate_angle = to_string((dynamic_cast<angleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
			sTemp = ("ry(" + gate_angle + ")");
			sTemp.append(" q[" + tarQubit + "];");
		}
			break;

		case Z_HALF_PI:
		{
			string  gate_angle = to_string((dynamic_cast<angleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
			sTemp = ("rz(" + gate_angle + ")");
			sTemp.append(" q[" + tarQubit + "];");
		}
			break;

		case T_GATE:
		case S_GATE:
		{
			sTemp.append(dagger ? "dg q[" + tarQubit + "];" : " q[" + tarQubit + "];");
		}
			break;

        case U1_GATE:
		case RX_GATE:
		case RY_GATE:
		case RZ_GATE:
		{
			string  gate_angle = to_string((dynamic_cast<angleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
			sTemp.append("(" + gate_angle + ")");
			sTemp.append(" q[" + tarQubit + "];");
		}
			break;

		case CNOT_GATE:
		{
			sTemp = "cx";
			//dagger is equal to itself
			sTemp.append(" " + all_qubits + ";");
		}
			break;

        case CZ_GATE:
		{
			//dagger is equal to itself
			sTemp.append(" " + all_qubits + ";");
		}
			break;

        case ISWAP_GATE:
		case SQISWAP_GATE:
            {
			    QASM_UNSUPPORT_EXCEPTIONAL(sTemp.c_str(), dagger);
                sTemp.append(dagger ? "dg " + all_qubits + ";" : " " + all_qubits + ";");
            }
            break;

        case CPHASE_GATE: 
            {
			    //U1(θ)等价于PHASE(θ)
				sTemp = "u1";
				string  gate_angle = to_string((dynamic_cast<angleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
				sTemp.append("(" + gate_angle + ")");
				sTemp.append(" q[" + tarQubit + "];");

            }
            break;

        case CU_GATE: 
            {
			    QuantumGate * gate_parameter = dynamic_cast<QuantumGate *>(pQGate->getQGate());
				if (dagger)
				{
					/*qCircuit << U1(target_qubits[0], -alpha) << RZ(target_qubits[0], -beta)
						<< RY(target_qubits[0], -gamma / 2) << CNOT(target_qubits[0], target_qubits[1])
						<< RY(target_qubits[0], gamma / 2) << RZ(target_qubits[0], (delta + beta) / 2)
						<< CNOT(target_qubits[0], target_qubits[1]) << RZ(target_qubits[0], -(delta - beta) / 2);*/

					snprintf(tmpStr, MAX_PATH, "u1(%f) q[%d];\n"
						"rz(%f) q[%d];\n"
						"ry(%f) q[%d];\n"
						"cx q[%d], q[%d];\n"
						"ry(%f) q[%d];\n"
						"rz(%f) q[%d];\n"
						"cx q[%d], q[%d];\n"
						"rz(%f) q[%d];\n", 
						(-1)*(gate_parameter->getAlpha()), qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						(-1)*(gate_parameter->getBeta()), qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						(-1)*(gate_parameter->getGamma())/2.0, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(), qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr(),
						(gate_parameter->getGamma()) / 2.0, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						(gate_parameter->getDelta() + gate_parameter->getBeta())/2.0, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(), qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr(),
						(-1)*(gate_parameter->getDelta() - gate_parameter->getBeta())/2, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr());
				}
				else
				{
					//QASM_UNSUPPORT_EXCEPTIONAL(sTemp.c_str(), dagger);
					/*qCircuit << RZ(target_qubits[0], (delta - beta) / 2) << CNOT(target_qubits[0], target_qubits[1])
						<< RZ(target_qubits[0], -(delta + beta) / 2) << RY(target_qubits[0], -gamma / 2)
						<< CNOT(target_qubits[0], target_qubits[1]) << RY(target_qubits[0], gamma / 2)
						<< RZ(target_qubits[0], beta) << U1(target_qubits[0], alpha);*/

					snprintf(tmpStr, MAX_PATH, "rz(%f) [%d];\n"
						"cx q[%d], q[%d];\n"
						"rz(%f) q[%d];\n"
						"ry(%f) q[%d];\n"
						"cx q[%d], q[%d];\n"
						"ry(%f) q[%d];\n"
						"rz(%f) q[%d];\n"
						"u1(%f) q[%d];\n",
						(gate_parameter->getDelta() - gate_parameter->getBeta())/2, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(), qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr(),
						(-1)*(gate_parameter->getDelta() + gate_parameter->getBeta()) / 2.0, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						(-1)*(gate_parameter->getGamma()) / 2.0, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(), qubits_vector[1]->getPhysicalQubitPtr()->getQubitAddr(),
						(gate_parameter->getGamma()) / 2.0, qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						(gate_parameter->getBeta()), qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr(),
						(gate_parameter->getAlpha()), qubits_vector[0]->getPhysicalQubitPtr()->getQubitAddr());
				}

				sTemp = tmpStr;
            }
            break;

		case SWAP_GATE:
			//dagger is equal to itself
			sTemp.append(" " + all_qubits + ";");
			break;

        default:sTemp = "UnSupportedQuantumGate;";
            break;
    }
    m_qasm.emplace_back(sTemp);
}

void QProgToQASM::transformQMeasure(AbstractQuantumMeasure *pMeasure)
{
    if (nullptr == pMeasure)
    {
        QCERR("pMeasure is null");
        throw invalid_argument("pMeasure is null");
    }
    if (nullptr == pMeasure->getQuBit()->getPhysicalQubitPtr())
    {
        QCERR("PhysicalQubitPtr is null");
        throw invalid_argument("PhysicalQubitPtr is null");
    }

    std::string tar_qubit = to_string(pMeasure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
    std::string creg_name = pMeasure->getCBit()->getName().substr(1);
    m_qasm.emplace_back("measure q[" + tar_qubit + "]" +" -> "+ "c[" + creg_name + "];");
}

void QProgToQASM::transformQProgByTraversalAlg(QProg *prog)
{
	if (nullptr == prog)
	{
		QCERR("p_prog is null");
		throw runtime_error("p_prog is null");
		return;
	}

	bool isDagger = false;
	Traversal::traversalByType(prog->getImplementationPtr(),nullptr, *this, isDagger);
}

void QProgToQASM::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	transformQGate(cur_node.get(),is_dagger);
}

void QProgToQASM::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	transformQMeasure(cur_node.get());
}

void QProgToQASM::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, bool &)
{}

void QProgToQASM::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	Traversal::traversal(cur_node, *this, is_dagger);
}

void QProgToQASM::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, bool &is_dagger)
{
	bool bDagger = cur_node->isDagger() ^ is_dagger;
	Traversal::traversal(cur_node, true, *this, bDagger);
}

void QProgToQASM::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, bool&)
{
	//抛出异常
	QCERR("transform error, there shouldn't be classicalProg here.");
	throw invalid_argument("transform error, there shouldn't be classicalProg here.");
}

static void traversalInOrderPCtr(const CExpr* pCtrFlow, string &ctr_statement)
{
    if (nullptr != pCtrFlow)
    {
        traversalInOrderPCtr(pCtrFlow->getLeftExpr(), ctr_statement);
        ctr_statement = ctr_statement + pCtrFlow->getName();
        traversalInOrderPCtr(pCtrFlow->getRightExpr(), ctr_statement);
    }
}

string QProgToQASM::getInsturctions()
{
    string instructions;

    for (auto &val : m_qasm)
    {
		//首行OPENQASM必须大写
		if (0 != val.compare(QASM_HEAD))
		{
			std::transform(val.begin(), val.end(), val.begin(), ::tolower);
		}
        instructions.append(val).append("\n");
    }
    instructions.erase(instructions.size() - 1);
    return instructions;
}

string QPanda::transformQProgToQASM(QProg &prog, QuantumMachine* quantum_machine, IBMQBackends ibmBackend /*= IBMQ_QASM_SIMULATOR*/)
{
    if (nullptr == quantum_machine)
    {
        QCERR("Quantum machine is nullptr");
        throw std::invalid_argument("Quantum machine is nullptr");
    }
    QProgToQASM pQASMTraverse(quantum_machine, ibmBackend);
    pQASMTraverse.transform(prog);
    return pQASMTraverse.getInsturctions();
}
