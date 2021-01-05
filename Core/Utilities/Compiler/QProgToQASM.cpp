#include "Core/Utilities/QProgInfo/MetadataValidity.h"
#include "Core/Utilities/Compiler/QProgToQASM.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Core.h"

using namespace std;
using namespace QGATE_SPACE;

USING_QPANDA

#ifndef MAX_PATH
#define MAX_PATH 260
#endif

#define QASM_HEAD "OPENQASM 2.0;"
#define QASM_UNSUPPORT_EXCEPTIONAL(gate, dg) {\
    std::string excepStr;\
    if (dg){\
        excepStr = string("Error: Qasm unsupport: ") + gate + "dg";\
    }else{excepStr = string("Error: Qasm unsupport: ") + gate;}\
	QCERR(excepStr.c_str());\
	throw std::invalid_argument(excepStr.c_str());\
}

QProgToQASM::QProgToQASM(QProg src_prog, QuantumMachine * quantum_machine)
	:m_src_prog(src_prog)
{
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
	m_gatetype.insert(pair<int, string>(U3_GATE, "U3"));

    m_gatetype.insert(pair<int, string>(CU_GATE, "CU"));
    m_gatetype.insert(pair<int, string>(CNOT_GATE, "CNOT"));
    m_gatetype.insert(pair<int, string>(CZ_GATE, "CZ"));
    m_gatetype.insert(pair<int, string>(CPHASE_GATE, "CPHASE"));
	m_gatetype.insert(pair<int, string>(SWAP_GATE, "SWAP"));
    m_gatetype.insert(pair<int, string>(ISWAP_GATE, "ISWAP"));
    m_gatetype.insert(pair<int, string>(SQISWAP_GATE, "SQISWAP"));

    m_qasm.clear();
    m_quantum_machine = quantum_machine;
}

void QProgToQASM::transform()
{
	if (nullptr == m_quantum_machine)
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: Quantum machine is nullptr.");
	}

    m_qasm.emplace_back(QASM_HEAD);
	m_qasm.emplace_back("include \"qelib1.inc\";");
    m_qasm.emplace_back("qreg q[" + to_string(m_quantum_machine->getAllocateQubit()) + "];");
    m_qasm.emplace_back("creg c[" + to_string(m_quantum_machine->getAllocateCMem()) + "];");

    vector<vector<string>> ValidQGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));
    vector<vector<string>> QGateMatrix(KMETADATA_GATE_TYPE_COUNT, vector<string>(0));

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
	QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE].emplace_back(m_gatetype[U3_GATE]);

    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CNOT_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CZ_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[CPHASE_GATE]);
    QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE].emplace_back(m_gatetype[SWAP_GATE]);

    SingleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_SINGLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_SINGLE_GATE]);  /* single gate data MetadataValidity */
    DoubleGateTypeValidator::GateType(QGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE],
        ValidQGateMatrix[MetadataGateType::METADATA_DOUBLE_GATE]);  /* double gate data MetadataValidity */
    TransformDecomposition traversal_vector(ValidQGateMatrix, QGateMatrix, m_quantum_machine);

    traversal_vector.TraversalOptimizationMerge(m_src_prog);

	traverse_qprog(m_src_prog);
}

void QProgToQASM::transformQGate(AbstractQGateNode * pQGate,bool is_dagger)
{
    if (nullptr == pQGate || nullptr == pQGate->getQGate())
    {
		QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: QGate is null.");
    }

    QVec qubits_vector;
    pQGate->getQuBitVector(qubits_vector);
	if (qubits_vector.size() == 2)
	{
		if (qubits_vector.front()->get_phy_addr() == qubits_vector.back()->get_phy_addr())
		{
			QCERR_AND_THROW_ERRSTR(run_fail, "Error: the control qubit and the target qubit are the same qubit.");
		}
	}
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
			//ignore dagger, because dagger is equal to itself
			sTemp.append(" q[" + tarQubit + "];");
		}
		break;

		case X_HALF_PI:
		{
			string  gate_angle = double_to_string((dynamic_cast<AbstractSingleAngleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
			sTemp = ("rx(" + gate_angle + ")");
			sTemp.append(" q[" + tarQubit + "];");
		}
			break;

		case Y_HALF_PI:
		{
			string  gate_angle = double_to_string((dynamic_cast<AbstractSingleAngleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
			sTemp = ("ry(" + gate_angle + ")");
			sTemp.append(" q[" + tarQubit + "];");
		}
			break;

		case Z_HALF_PI:
		{
			string  gate_angle = double_to_string((dynamic_cast<AbstractSingleAngleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
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
			string  gate_angle = double_to_string((dynamic_cast<AbstractSingleAngleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
			sTemp.append("(" + gate_angle + ")");
			sTemp.append(" q[" + tarQubit + "];");
		}
			break;

		case U3_GATE:
		{
			auto u3_gate = dynamic_cast<QGATE_SPACE::U3*>(pQGate->getQGate());
			string theta = double_to_string(u3_gate->get_theta() * iLabel);
			string lambda;
			string phi;
			if (dagger)
			{
				lambda = double_to_string(u3_gate->get_phi() * iLabel);
				phi = double_to_string(u3_gate->get_lambda() * iLabel);
			}
			else
			{
				phi = double_to_string(u3_gate->get_phi());
				lambda = double_to_string(u3_gate->get_lambda());
			}

			sTemp.append("(" + theta + "," + phi + ","+ lambda + ")");
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
				sTemp = "u1";
				string  gate_angle = double_to_string((dynamic_cast<AbstractSingleAngleParameter *>(pQGate->getQGate()))->getParameter() * iLabel);
				sTemp.append("(" + gate_angle + ")");
				sTemp.append(" q[" + tarQubit + "];");

            }
            break;

        case CU_GATE: 
            {
			    auto gate_parameter = dynamic_cast<AbstractAngleParameter *>(pQGate->getQGate());
				if (dagger)
				{
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

        default:
			QASM_UNSUPPORT_EXCEPTIONAL(iter->second, dagger);
            break;
    }
    m_qasm.emplace_back(sTemp);
}

void QProgToQASM::transformQMeasure(AbstractQuantumMeasure *pMeasure)
{
    if (nullptr == pMeasure->getQuBit()->getPhysicalQubitPtr())
    {
		QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: measure node is null.");
    }

    std::string tar_qubit = to_string(pMeasure->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
    std::string creg_name = pMeasure->getCBit()->getName().substr(1);
    m_qasm.emplace_back("measure q[" + tar_qubit + "]" +" -> "+ "c[" + creg_name + "];");
}

void QProgToQASM::transformQReset(AbstractQuantumReset* pReset)
{
	if (nullptr == pReset)
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: reset node is null.");
	}
	if (nullptr == pReset->getQuBit()->getPhysicalQubitPtr())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: PhysicalQubitPtr is null.");
	}

	std::string tar_qubit = to_string(pReset->getQuBit()->getPhysicalQubitPtr()->getQubitAddr());
	m_qasm.emplace_back("reset q[" + tar_qubit + "];");
}

std::string QProgToQASM::double_to_string(const double d, const int precision /*= 17*/)
{
	std::ostringstream stream;
	stream.precision(precision);
	stream << d;
	return stream.str();
}

void QProgToQASM::execute(std::shared_ptr<AbstractQGateNode>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	transformQGate(cur_node.get(), cir_param.m_is_dagger);
}

void QProgToQASM::execute(std::shared_ptr<AbstractQuantumMeasure> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	transformQMeasure(cur_node.get());
}

void QProgToQASM::execute(std::shared_ptr<AbstractQuantumReset> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	transformQReset(cur_node.get());
}

void QProgToQASM::execute(std::shared_ptr<AbstractControlFlowNode> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: unsupport control-flow-node here.");
}

void QProgToQASM::execute(std::shared_ptr<AbstractQuantumProgram>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
}

void QProgToQASM::execute(std::shared_ptr<AbstractQuantumCircuit> cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	TraverseByNodeIter::execute(cur_node, parent_node, cir_param, cur_node_iter);
}

void QProgToQASM::execute(std::shared_ptr<AbstractClassicalProg>  cur_node, std::shared_ptr<QNode> parent_node, QCircuitParam &cir_param, NodeIter& cur_node_iter)
{
	QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: unsupport classicalProg here.");
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
		if (0 != val.compare(QASM_HEAD))
		{
			std::transform(val.begin(), val.end(), val.begin(), ::tolower);
		}
        instructions.append(val).append("\n");
    }
    instructions.erase(instructions.size() - 1);
    return instructions;
}

string QPanda::convert_qprog_to_qasm(QProg &prog, QuantumMachine* qm)
{
	if (nullptr == qm)
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error on transformQProgToQASM: Quantum machine is nullptr.");
	}
	QProgToQASM pQASMTraverse(prog, qm);
	pQASMTraverse.transform();
	return pQASMTraverse.getInsturctions();
}

void QPanda::write_to_qasm_file(QProg prog, QuantumMachine * qvm, const string file_name)
{
	std::ofstream out_file;
	auto qasm_str = convert_qprog_to_qasm(prog, qvm);
	out_file.open(file_name, ios::out);
	if (!out_file.is_open())
	{
		QCERR_AND_THROW_ERRSTR(run_fail, "Error: failed to open file.");
	}

	out_file << qasm_str;
	out_file.close();
}