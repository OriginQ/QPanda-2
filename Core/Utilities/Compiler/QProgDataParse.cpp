#include "Core/Utilities/Compiler/QProgDataParse.h"
#include "Core/QuantumCircuit/ClassicalConditionInterface.h"
#include "Core/Utilities/QProgTransform/QProgToQCircuit.h"
#include "Core/Utilities/Compiler/QProgStored.h"

using namespace std;
USING_QPANDA

const uint32_t kBinaryOperatorValue =
(1u << PLUS) | (1u << MINUS) | (1u << MUL) | (1u << DIV) |
(1u << GT) | (1u << EGT) | (1u << LT) | (1u << ELT) |
(1u << EQUAL) | (1u << NE) | (1u << AND) | (1u << OR) |
(1u << ASSIGN);
const uint32_t kUnaryOperatorValue = 1u << NOT;

const std::map<int, function<ClassicalCondition(ClassicalCondition &, ClassicalCondition &)>> kBinaryOperationFun =
{
    {PLUS,[](ClassicalCondition & a,ClassicalCondition &b) {return a + b; }},
    {MINUS,[](ClassicalCondition& a,ClassicalCondition & b) {return a - b; } },
    {MUL,[](ClassicalCondition& a,ClassicalCondition & b) {return a * b; } },
    {DIV,[](ClassicalCondition& a,ClassicalCondition & b) {return a / b; } },
    {EQUAL,[](ClassicalCondition& a,ClassicalCondition & b) {return a == b; }},
    {NE,[](ClassicalCondition& a,ClassicalCondition & b) {return a != b; } },
    {GT,[](ClassicalCondition& a,ClassicalCondition & b) {return a > b; } },
    {EGT,[](ClassicalCondition& a,ClassicalCondition & b) {return a >= b; } },
    {LT,[](ClassicalCondition& a,ClassicalCondition & b) {return a < b; } },
    {ELT,[](ClassicalCondition& a,ClassicalCondition & b) {return a <= b; } },
    {AND,[](ClassicalCondition &a,ClassicalCondition &b) {return a && b; } },
    {OR,[](ClassicalCondition &a,ClassicalCondition &b) {return a || b; } },
	{ASSIGN,[](ClassicalCondition &a,ClassicalCondition &b) { return a = b; } }
};

const std::map<int, function<ClassicalCondition(ClassicalCondition)>> kUnaryOperationFun =
{
    {1u << NOT,[](ClassicalCondition a) {return !a; }}
};

const uint32_t kSingleGateValue =
(1u << QPROG_PAULI_X_GATE) | (1u << QPROG_PAULI_Y_GATE) | (1u << QPROG_PAULI_Z_GATE) |
(1u << QPROG_X_HALF_PI) | (1u << QPROG_Y_HALF_PI) | (1u << QPROG_Z_HALF_PI) |
(1u << QPROG_HADAMARD_GATE) | (1u << QPROG_T_GATE) | (1u << QPROG_S_GATE);
const std::map<int, function<QGate(Qubit *)>> kSingleGateFun =
{
    {QPROG_PAULI_X_GATE, X},
    {QPROG_PAULI_Y_GATE, Y},
    {QPROG_PAULI_Z_GATE, Z},
    {QPROG_X_HALF_PI,    X1},
    {QPROG_Y_HALF_PI,    Y1},
    {QPROG_Z_HALF_PI,    Z1},
    {QPROG_HADAMARD_GATE,H},
    {QPROG_T_GATE,       T},
    {QPROG_S_GATE,       S}
};

const uint32_t kSingleGateAngleValue =
(1u << QPROG_RX_GATE) | (1u << QPROG_RY_GATE) | (1u << QPROG_RZ_GATE) |
(1u << QPROG_U1_GATE);
const std::map<int, function<QGate(Qubit *, double)>> kSingleGateAngelFun =
{
    {QPROG_RX_GATE, RX},
    {QPROG_RY_GATE, RY},
    {QPROG_RZ_GATE, RZ},
    {QPROG_U1_GATE, U1}
};
const uint32_t kCUValue = 1u << QPROG_CU_GATE;
const std::map<int, function<QGate(Qubit*, Qubit*, double, double, double, double)>> kCUFun =
{
	{QPROG_CU_GATE, [](Qubit *q1, Qubit *q2, double alpha, double beta, double gamma, double delta)
					 {return CU(alpha, beta, gamma, delta, q1, q2); }}
};

const uint32_t kU4Value = 1u << QPROG_U4_GATE;
const std::map<int, function<QGate(Qubit*, double, double, double, double)>> kU4Fun =
{
    { QPROG_U4_GATE, [](Qubit *q, double alpha, double beta, double gamma, double delta)
					 {return U4(alpha, beta, gamma, delta, q); } }
};

const uint32_t kDoubleGateValue =
(1u << QPROG_CNOT_GATE) | (1u << QPROG_CZ_GATE) |
(1u << QPROG_ISWAP_GATE) | (1u << QPROG_SQISWAP_GATE)| 
(1u << QPROG_SWAP_GATE);
const std::map<int, function<QGate(Qubit *, Qubit *)>> kDoubleGateFun =
{
    {QPROG_CNOT_GATE, CNOT},
    {QPROG_CZ_GATE, CZ},
    {QPROG_ISWAP_GATE, [](Qubit *q1, Qubit *q2) {return iSWAP(q1, q2); }},
	{QPROG_SWAP_GATE, [](Qubit *q1, Qubit *q2) {return SWAP(q1, q2); }},
    {QPROG_SQISWAP_GATE, SqiSWAP}
};

const uint32_t kDoubleGateAngleValue =
(1u << QPROG_CPHASE_GATE) | (1u << QPROG_ISWAP_THETA_GATE);
const std::map<int, function<QGate(Qubit *, Qubit *, double)>> kDoubleGateAngleFun =
{
    {QPROG_ISWAP_THETA_GATE, [](Qubit *q1, Qubit *q2, double a) {return iSWAP(q1, q2, a); }},
    {QPROG_CPHASE_GATE, CR}
};


QProgDataParse::QProgDataParse(QuantumMachine* qm)
{
    if (nullptr == qm)
    {
        throw std::invalid_argument("QuantumMachine is nullptr");
    }
    m_quantum_machine = qm;
}

QProgDataParse::~QProgDataParse()
{
}

bool QProgDataParse::load(const std::string &filename)
{
    ifstream in;
    in.open(filename, std::ios::in | std::ios::binary);
    if (!in)
    {
        QCERR("open file error");
        return false;
    }

    uint32_t file_length = 0;
    std::streampos pos = in.tellg();
    in.seekg(0, std::ios::end);
    file_length = in.tellg();

    in.seekg(pos);
    pair<uint32_t, DataNode> one_data_node;
    in.read((char *)&one_data_node, sizeof(one_data_node));

    if (file_length != one_data_node.first)
    {
        return false;
    }
    m_node_counter = one_data_node.second.qubit_data;

    in.read((char *)&one_data_node, sizeof(one_data_node));
    uint32_t qubit_number = one_data_node.first;
    uint32_t cbit_number = one_data_node.second.qubit_data;
    m_qubits_count = qubit_number;
    m_cbits_count = cbit_number;

    m_data_vector.resize(m_node_counter);
    in.read((char *)m_data_vector.data(), m_node_counter * sizeof(one_data_node));
    in.close();
    m_qubits_addr.clear();
    m_cbits_addr.clear();

    return true;
}


bool QProgDataParse::load(const std::vector<uint8_t>& data)
{
    pair<uint32_t, DataNode> one_data_node;
    memcpy(&one_data_node, data.data(), sizeof(one_data_node));
    m_node_counter = one_data_node.second.qubit_data;
    size_t size = (m_node_counter + 2) * sizeof(one_data_node);

    if (size != data.size())
    {
        QCERR("QProg data is invalid");
        return false;
    }

    memcpy(&one_data_node, data.data() + sizeof(one_data_node),
        sizeof(one_data_node));
    uint32_t qubit_number = one_data_node.first;
    uint32_t cbit_number = one_data_node.second.qubit_data;
    m_qubits_count = qubit_number;
    m_cbits_count = cbit_number;

    m_data_vector.resize(m_node_counter);
    memcpy(m_data_vector.data(), data.data() + 2 * sizeof(one_data_node),
        m_node_counter * sizeof(one_data_node));

    m_qubits_addr.clear();
    m_cbits_addr.clear();
    return true;
}

bool QProgDataParse::parse(QProg &prog)
{
    for (uint32_t addr = 0; addr < m_qubits_count; addr++)
    {
        m_qubits_addr.push_back(addr);
    }

    for (uint32_t addr = 0; addr < m_cbits_count; addr++)
    {
        m_cbits_addr.push_back(addr);
    }

    if (prog.getFirstNodeIter() != prog.getEndNodeIter())
    {
        QCERR("QProg is not empty");
        throw invalid_argument("QProg is not empty");
    }

    if (m_node_counter != m_data_vector.size())
    {
        return false;
    }

    m_iter = m_data_vector.begin();
    parseDataNode(prog, m_node_counter);

    return true;
}

QVec QProgDataParse::getQubits()
{
    std::stable_sort(m_qubits_addr.begin(), m_qubits_addr.end(),
                     [](size_t a, size_t b){return a < b ;});
    QVec qubits;
    for (auto qubit_addr : m_qubits_addr)
    {
        auto qubit = m_quantum_machine->allocateQubitThroughVirAddress(qubit_addr);
        qubits.push_back(qubit);
    }

    return qubits;
}

std::vector<ClassicalCondition> QProgDataParse::getCbits()
{
    std::stable_sort(m_cbits_addr.begin(), m_cbits_addr.end(),
                     [](size_t a, size_t b){return a < b ;});
    std::vector<ClassicalCondition> cbits;
    for (auto &cbit_addr : m_cbits_addr)
    {
        auto cbit = m_quantum_machine->allocateCBit(cbit_addr);
        cbits.push_back(cbit);
    }

    return cbits;
}


void QProgDataParse::parseQGateDataNode(QProg &prog, const uint32_t &type_and_number, const uint32_t &qubits_data)
{
	QVec ctrl_qubits_vector;
	if (!m_control_qubits_addr.empty())
	{
		for (auto qubit_addr : m_control_qubits_addr)
		{
			auto qubit = m_quantum_machine->allocateQubitThroughVirAddress(qubit_addr);
			ctrl_qubits_vector.push_back(qubit);
		}
		m_control_qubits_addr.clear();
	}

    uint16_t type = (type_and_number & 0xffff) >> 1;
    bool is_dagger = (type_and_number & 0x01) ? true : false;

    const int kQubitMax = 2;
    uint16_t qubit_array[kQubitMax] = { 0 };
    qubit_array[0] = qubits_data & 0xffff;
    qubit_array[1] = qubits_data >> (kCountMoveBit);

    auto qubit0 = m_quantum_machine->allocateQubitThroughVirAddress(qubit_array[0]);
    auto iter = std::find(m_qubits_addr.begin(), m_qubits_addr.end(), qubit_array[0]);
    if (m_qubits_addr.end() == iter)
    {
        m_qubits_addr.push_back(qubit_array[0]);
    }

    uint32_t tmp = 1u << type;
    if (kSingleGateValue & tmp)
    {
        auto iter = kSingleGateFun.find(type);
        if (iter == kSingleGateFun.end())
        {
            QCERR("parse gate type error!");
            throw runtime_error("parse gate type error!");
        }

        auto gate = iter->second(qubit0);
        gate.setDagger(is_dagger);
		gate.setControl(ctrl_qubits_vector);
        prog << gate;
    }
    else if (kSingleGateAngleValue & tmp)
    {
        auto iter = kSingleGateAngelFun.find(type);
        if (iter == kSingleGateAngelFun.end())
        {
            QCERR("parse gate type error!");
            throw runtime_error("parse gate type error!");
        }

        m_iter++;
        float angle = getAngle(*m_iter);
        auto gate = iter->second(qubit0, angle);
        gate.setDagger(is_dagger);
		gate.setControl(ctrl_qubits_vector);
        prog << gate;
    }
    else if (kDoubleGateValue & tmp)
    {
        auto iter = kDoubleGateFun.find(type);
        if (iter == kDoubleGateFun.end())
        {
            QCERR("parse gate type error!");
            throw runtime_error("parse gate type error!");
        }

        auto qubit1 = m_quantum_machine->allocateQubitThroughVirAddress(qubit_array[1]);
        auto qubit1_iter = std::find(m_qubits_addr.begin(), m_qubits_addr.end(), qubit_array[1]);
        if (m_qubits_addr.end() == qubit1_iter)
        {
            m_qubits_addr.push_back(qubit_array[1]);
        }

        auto gate = iter->second(qubit0, qubit1);
        gate.setDagger(is_dagger);
		gate.setControl(ctrl_qubits_vector);
        prog << gate;
    }
    else if (kDoubleGateAngleValue & tmp)
    {
        auto iter = kDoubleGateAngleFun.find(type);
        if (iter == kDoubleGateAngleFun.end())
        {
            QCERR("parse gate type error!");
            throw runtime_error("parse gate type error!");
        }

        m_iter++;
        float angle = getAngle(*m_iter);

        auto qubit1 = m_quantum_machine->allocateQubitThroughVirAddress(qubit_array[1]);
        auto iter1 = std::find(m_qubits_addr.begin(), m_qubits_addr.end(), qubit_array[1]);
        if (m_qubits_addr.end() == iter1)
        {
            m_qubits_addr.push_back(qubit_array[1]);
        }

        auto gate = iter->second(qubit0, qubit1, angle);
        gate.setDagger(is_dagger);
		gate.setControl(ctrl_qubits_vector);
        prog << gate;
    }
    else if (kU4Value & tmp)
    {
        auto iter = kU4Fun.find(type);
        if (iter == kU4Fun.end())
        {
            QCERR("parse gate type error!");
            throw runtime_error("parse gate type error!");
        }
        m_iter++;
        float alpha = getAngle(*m_iter);
        m_iter++;
        float beta = getAngle(*m_iter);
        m_iter++;
        float gamma = getAngle(*m_iter);
        m_iter++;
        float delta = getAngle(*m_iter);

        auto gate = iter->second(qubit0, alpha, beta, gamma, delta);
        gate.setDagger(is_dagger);
		gate.setControl(ctrl_qubits_vector);
        prog << gate;
    }
	else if (kCUValue & tmp)
	{
		auto iter = kCUFun.find(type);
		if (iter == kCUFun.end())
		{
			QCERR("parse gate type error!");
			throw runtime_error("parse gate type error!");
		}
		auto qubit1 = m_quantum_machine->allocateQubitThroughVirAddress(qubit_array[1]);
		auto qubit1_iter = std::find(m_qubits_addr.begin(), m_qubits_addr.end(), qubit_array[1]);
		if (m_qubits_addr.end() == qubit1_iter)
		{
			m_qubits_addr.push_back(qubit_array[1]);
		}
		m_iter++;
		float alpha = getAngle(*m_iter);
		m_iter++;
		float beta = getAngle(*m_iter);
		m_iter++;
		float gamma = getAngle(*m_iter);
		m_iter++;
		float delta = getAngle(*m_iter);

		auto gate = iter->second(qubit0, qubit1, alpha, beta, gamma, delta);
		gate.setDagger(is_dagger);
		gate.setControl(ctrl_qubits_vector);
		prog << gate;
	}
    else
    {
        QCERR("Invaild QGate Type");
        throw runtime_error("Invaild QGate Type");
    }
    return;
}

float QProgDataParse::getAngle(const pair<uint32_t, DataNode> &data_node)
{
    uint16_t type_iter = (data_node.first & 0xffff) >> 1;
    if (QPROG_GATE_ANGLE != type_iter)
    {
        QCERR("parsing QPROG_GATE_ANGLE failure");
        throw runtime_error("parsing QPROG_GATE_ANGLE failure");
    }

    return data_node.second.angle_data;
}

int QProgDataParse::getCBitValue(const std::pair<uint32_t, DataNode> &data_node)
{
    uint16_t type_iter = (data_node.first & 0xffff) >> 1;
    if (QPROG_CEXPR_EVAL != type_iter)
    {
        QCERR("parsing QPROG_CEXPR_EVAL failure");
        throw runtime_error("parsing QPROG_CEXPR_EVAL failure");
    }

    return (int)data_node.second.qubit_data;
}


void QProgDataParse::parseQMeasureDataNode(QProg &prog, uint32_t qubits_data)
{
    const int kQubitMax = 2;
    uint16_t qubit_array[kQubitMax] = { 0 };
    qubit_array[0] = qubits_data & 0xffff;
    qubit_array[1] = (qubits_data >> (kCountMoveBit));

    auto qubit = m_quantum_machine->allocateQubitThroughVirAddress(qubit_array[0]);
    auto qubit_iter = std::find(m_qubits_addr.begin(), m_qubits_addr.end(), qubit_array[0]);
    if (m_qubits_addr.end() == qubit_iter)
    {
        m_qubits_addr.push_back(qubit_array[0]);
    }

    auto cbit = m_quantum_machine->allocateCBit(qubit_array[1]);
    auto cbit_iter = std::find(m_cbits_addr.begin(), m_cbits_addr.end(), qubit_array[1]);
    if (m_cbits_addr.end() == cbit_iter)
    {
        m_cbits_addr.push_back(qubit_array[1]);
    }

    auto measure = Measure(qubit, cbit);
    prog << measure;
    return;
}

void QProgDataParse::parseCExprCBitDataNode(const uint32_t &data)
{
    m_iter++;
    auto value = getCBitValue(*m_iter);

    auto cbit = m_quantum_machine->allocateCBit(data);
    auto cbit_iter = std::find(m_cbits_addr.begin(), m_cbits_addr.end(), data);
    if (m_cbits_addr.end() == cbit_iter)
    {
        m_cbits_addr.push_back(data);
    }

    cbit.setValue(value);
    m_stack_cc.push(cbit);
    return;
}

void QProgDataParse::parseCExprOperateDataNode(const uint32_t &data)
{
    uint32_t tmp = 1u << data;
    if (kBinaryOperatorValue & tmp)
    {
        ClassicalCondition cc_right(m_stack_cc.top());
        m_stack_cc.pop();
        ClassicalCondition cc_left(m_stack_cc.top());
        m_stack_cc.pop();

        auto iter = kBinaryOperationFun.find(data);
        if (iter == kBinaryOperationFun.end())
        {
            QCERR("parse ClassicalCondition Operator error");
            throw runtime_error("parse ClassicalCondition Operator error");
        }

        ClassicalCondition cc_root = iter->second(cc_left, cc_right);
        m_stack_cc.push(cc_root);
    }
    else if (kUnaryOperatorValue & tmp)
    {
        ClassicalCondition cc_right(m_stack_cc.top());
        m_stack_cc.pop();

        auto iter = kUnaryOperationFun.find(data);
        if (iter == kUnaryOperationFun.end())
        {
            QCERR("parse ClassicalCondition Operator error");
            throw runtime_error("parse ClassicalCondition Operator error");
        }

        ClassicalCondition cc_root = iter->second(cc_right);
        m_stack_cc.push(cc_root);
    }
    else
    {
        QCERR("parse ClassicalCondition Operator error");
        throw runtime_error("parse ClassicalCondition Operator error");
    }

    return;
}

void QProgDataParse::parseCExprConstValueDataNode(const int & data)
{
    auto &fac = CExprFactory::GetFactoryInstance();
    auto expr = fac.GetCExprByValue(data);
    if (expr == nullptr)
    {
        QCERR("CExpr factory fails");
        throw std::runtime_error("CExpr factory fails");
    }

    ClassicalCondition cc(expr);
    m_stack_cc.push(cc);
    return;
}

void QProgDataParse::parseCExprEvalDataNode(const int &data)
{
    m_stack_cc.top().setValue(data);
    return;
}


void QProgDataParse::parseQIfDataNode(QProg &prog, const uint32_t &data)
{
    ClassicalCondition cc(m_stack_cc.top());
    m_stack_cc.pop();
    uint32_t tail_number_true_branch = data >> kCountMoveBit;
    uint32_t tail_number_false_branch = data & 0xffff;

    QProg prog_true_branch = CreateEmptyQProg();
    m_iter++;
    parseDataNode(prog_true_branch, tail_number_true_branch);

    if (tail_number_false_branch)
    {
        QProg prog_false_brach = CreateEmptyQProg();
        m_iter++;
        parseDataNode(prog_false_brach, tail_number_false_branch);
        QIfProg prog_If = CreateIfProg(cc, prog_true_branch, prog_false_brach);
        prog << prog_If;
    }
    else
    {
        QIfProg prog_If = CreateIfProg(cc, prog_true_branch);
        prog << prog_If;
    }

    return;
}


void QProgDataParse::parseQWhileDataNode(QProg &prog, uint32_t data)
{
    ClassicalCondition cc(m_stack_cc.top());
    m_stack_cc.pop();
    uint32_t tail_number_true_branch = data >> kCountMoveBit;
    QProg prog_true_branch = CreateEmptyQProg();

    m_iter++;
    parseDataNode(prog_true_branch, tail_number_true_branch);
    QWhileProg prog_while = CreateWhileProg(cc, prog_true_branch);
    prog << prog_while;

    return;
}
void QProgDataParse::parseCircuitDataNode(QProg &prog, const uint32_t &type_and_number, const uint32_t &data)
{
	QVec ctrl_qubits_vector;
	if (!m_control_qubits_addr.empty())
	{
		for (auto qubit_addr : m_control_qubits_addr)
		{
			auto qubit = m_quantum_machine->allocateQubitThroughVirAddress(qubit_addr);
			ctrl_qubits_vector.push_back(qubit);
		}
		m_control_qubits_addr.clear();
	}

	bool is_dagger = (type_and_number & 0x01) ? true : false;
	uint32_t tail_number_circuit = data & 0xffff/*>> kCountMoveBit*/;

	QCircuit cir = CreateEmptyCircuit();
	QProg porg_tmp = CreateEmptyQProg();
	m_iter++;
	parseDataNode(porg_tmp, tail_number_circuit);

	cast_qprog_qcircuit(porg_tmp, cir);

	cir.setDagger(is_dagger);
	cir.setControl(ctrl_qubits_vector);
	prog << cir;

	return;
}

void QProgDataParse::parseClassicalExprDataNode(QProg &prog, uint32_t data)
{
	ClassicalCondition cc(m_stack_cc.top());
	m_stack_cc.pop();
	prog << cc;
	return;
}

void QProgDataParse::parseControlNodeData(const uint32_t &data)
{
	size_t qubit_addr_1 = data & 0xffff;
	m_control_qubits_addr.push_back(qubit_addr_1);

	size_t qubit_addr_2 = (data >> (kCountMoveBit)); 
	if (0 != qubit_addr_2 )
	{
		m_control_qubits_addr.push_back(qubit_addr_2);
	}

	return;
}

void QProgDataParse::parseDataNode(QProg &prog, const uint32_t &tail_number)
{
    if (0 == tail_number)
    {
        return;
    }

    uint16_t type = (m_iter->first & 0xffff) >> 1;
    uint32_t data = m_iter->second.qubit_data;
    switch (type)
    {
    case QPROG_PAULI_X_GATE:
    case QPROG_PAULI_Y_GATE:
    case QPROG_PAULI_Z_GATE:
    case QPROG_X_HALF_PI:
    case QPROG_Y_HALF_PI:
    case QPROG_Z_HALF_PI:
    case QPROG_HADAMARD_GATE:
    case QPROG_T_GATE:
    case QPROG_S_GATE:
    case QPROG_U4_GATE:
    case QPROG_CU_GATE:
    case QPROG_CNOT_GATE:
    case QPROG_CZ_GATE:
    case QPROG_ISWAP_GATE:
    case QPROG_ISWAP_THETA_GATE:
    case QPROG_SQISWAP_GATE:
	case	QPROG_SWAP_GATE:
    case QPROG_RX_GATE:
    case QPROG_RY_GATE:
    case QPROG_RZ_GATE:
    case QPROG_U1_GATE:
    case QPROG_CPHASE_GATE:
        parseQGateDataNode(prog, m_iter->first, data);
        break;
    case QPROG_MEASURE_GATE:
        parseQMeasureDataNode(prog, data);
        break;
    case QPROG_QIF_NODE:
        parseQIfDataNode(prog, data);
        break;
    case QPROG_QWHILE_NODE:
        parseQWhileDataNode(prog, data);
        break;
    case QPROG_CEXPR_CBIT:
        parseCExprCBitDataNode(data);
        break;
    case QPROG_CEXPR_OPERATOR:
        parseCExprOperateDataNode(data);
        break;
    case QPROG_CEXPR_CONSTVALUE:
        parseCExprConstValueDataNode(data);
        break;
	case QPROG_CEXPR_NODE:
		parseClassicalExprDataNode(prog, data);
		break;
	case QPROG_CONTROL:
		parseControlNodeData(data);
		break;
	case QPROG_CIRCUIT_NODE:
		parseCircuitDataNode(prog, m_iter->first, data);
		break;
	default:
        QCERR("invalid QProg node type");
        throw runtime_error("invalid QProg node type");
        break;
    }

    uint32_t node_count = m_iter->first >> kCountMoveBit;
    if (tail_number == node_count)
    {
        return;
    }

    m_iter++;
    parseDataNode(prog, tail_number);
}


bool QPanda::binaryQProgFileParse(QuantumMachine *qm, const std::string &filename, QVec &qubits, 
                                  std::vector<ClassicalCondition> &cbits, QProg &prog)
{
    QProgDataParse dataParse(qm);
    if (!dataParse.load(filename))
    {
        std::cout << "load file error" << std::endl;
        throw runtime_error("Parse file error");
    }

    if (!dataParse.parse(prog))
    {
        throw runtime_error("Parse file error");
    }

    qubits = dataParse.getQubits();
    cbits = dataParse.getCbits();

    return true;
}

bool QPanda::binaryQProgDataParse(QuantumMachine *qm, const std::vector<uint8_t>& data, QVec & qubits,
                                  std::vector<ClassicalCondition>& cbits, QProg & prog)
{
    QProgDataParse dataParse(qm);
    if (!dataParse.load(data))
    {
        std::cout << "load binary data error" << std::endl;
        throw runtime_error("load binary data error");
    }

    if (!dataParse.parse(prog))
    {
        throw runtime_error("Parse binary data error");
    }

    qubits = dataParse.getQubits();
    cbits = dataParse.getCbits();

    return true;
}

bool QPanda::transformBinaryDataToQProg(QuantumMachine *qm, const std::string &filename, QVec &qubits,
	std::vector<ClassicalCondition> &cbits, QProg &prog)
{
	QProgDataParse dataParse(qm);
	if (!dataParse.load(filename))
	{
		std::cout << "load file error" << std::endl;
		throw runtime_error("Parse file error");
	}

	if (!dataParse.parse(prog))
	{
		throw runtime_error("Parse file error");
	}

	qubits = dataParse.getQubits();
	cbits = dataParse.getCbits();

	return true;
}

bool QPanda::transformBinaryDataToQProg(QuantumMachine *qm, const std::vector<uint8_t>& data, QVec & qubits,
	std::vector<ClassicalCondition>& cbits, QProg & prog)
{
	QProgDataParse dataParse(qm);
	if (!dataParse.load(data))
	{
		std::cout << "load binary data error" << std::endl;
		throw runtime_error("load binary data error");
	}

	if (!dataParse.parse(prog))
	{
		throw runtime_error("Parse binary data error");
	}

	qubits = dataParse.getQubits();
	cbits = dataParse.getCbits();

	return true;
}

bool QPanda::convert_binary_data_to_qprog(QuantumMachine *qm, const std::string &filename, QVec &qubits,
	std::vector<ClassicalCondition> &cbits, QProg &prog)
{
	return transformBinaryDataToQProg(qm, filename, qubits, cbits, prog);
}

bool QPanda::convert_binary_data_to_qprog(QuantumMachine *qm, const std::vector<uint8_t>& data, QVec & qubits,
	std::vector<ClassicalCondition>& cbits, QProg & prog)
{
	return transformBinaryDataToQProg(qm, data, qubits, cbits, prog);
}












