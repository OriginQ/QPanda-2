#include "QProgDataParse.h"
#include "../QPanda.h"
using namespace std;
USING_QPANDA
QProgDataParse::QProgDataParse(const string &filename) :
    m_file_length(0), m_node_counter(0), m_filename(filename)
{ }


QProgDataParse::~QProgDataParse()
{ }

bool QProgDataParse::loadFile()
{
    FILE *fp = nullptr;

#if defined(_MSC_VER) && (_MSC_VER >= 1400 )
    errno_t errno_number = fopen_s(&fp, m_filename.c_str(), "rb");
    if (errno_number || !fp)
    {
        return false;
    }
#else
    fp = fopen(m_filename.c_str(), "rb");
    if (!fp)
    {
        return false;
    }
#endif

    fseek(fp, 0l, SEEK_END);
    m_file_length = ftell(fp);
    fseek(fp, 0l, SEEK_SET);

    const int kMemNumber = 1;
    pair<uint_t, DataNode> one_data_node;

    if (kMemNumber != fread(&one_data_node, sizeof(uint_t) + sizeof(DataNode), kMemNumber, fp))
    {
        return false;
    }

    if (m_file_length != one_data_node.first)
    {
        return false;
    }
    m_node_counter = one_data_node.second.qubit_data;

    while (kMemNumber == fread(&one_data_node, sizeof(uint_t) + sizeof(DataNode), kMemNumber, fp))
    {
        m_data_list.push_back(one_data_node);
    }

    fclose(fp);
    return true;
}


bool QProgDataParse::parse(QProg &prog)
{
    if (prog.getFirstNodeIter() != prog.getEndNodeIter())
    {
        QCERR("qProg is not empty");
        throw invalid_argument("qProg is not empty");
    }

    if (m_node_counter != m_data_list.size())
    {
        return false;
    }

    m_iter = m_data_list.begin();
    parseDataNode(prog, m_node_counter);

    return true;
}


void QProgDataParse::parseQGateDataNode(QProg &prog, const uint_t type_and_number, const uint_t qubits_data)
{
    ushort_t type = (type_and_number & 0xffff) >> 1;
    bool is_dagger = (type_and_number & 0x01) ? true : false;

    const int kQubitMax = 2;
    ushort_t qubit_array[kQubitMax] = {0};
    qubit_array[0] = qubits_data & 0xffff;
    qubit_array[1] = (qubits_data >> (kCountMoveBit));

    auto qubit0 = qAlloc(qubit_array[0]);
    if (!qubit0)
    {
        QCERR("qAlloc fail");
        throw runtime_error("qAlloc fail");
    }

    switch (type)
    {
    case QPROG_NODE_TYPE_PAULI_X_GATE:
        {
            auto gate = X(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_PAULI_Y_GATE:
        {
            auto gate = Y(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_PAULI_Z_GATE:
        {
            auto gate = Z(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_X_HALF_PI:
        {
            auto gate = X1(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_Y_HALF_PI:
        {
            auto gate = Y1(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_Z_HALF_PI:
        {
            auto gate = Z1(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_HADAMARD_GATE:
        {
            auto gate = H(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_T_GATE:
        {
            auto gate = T(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_S_GATE:
        {
            auto gate = S(qubit0);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_CNOT_GATE:
        {
            auto qubit1 = qAlloc(qubit_array[1]);
            if (!qubit1)
            {
                QCERR("qAlloc fail");
                throw runtime_error("qAlloc fail");
            }

            auto gate = CNOT(qubit0, qubit1);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_CZ_GATE:
        {
            auto qubit1 = qAlloc(qubit_array[1]);
            if (!qubit1)
            {
                QCERR("qAlloc fail");
                throw runtime_error("qAlloc fail");
            }

            auto gate = CZ(qubit0, qubit1);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_ISWAP_GATE:
        {
            auto qubit1 = qAlloc(qubit_array[1]);
            if (!qubit1)
            {
                QCERR("qAlloc fail");
                throw runtime_error("qAlloc fail");
            }

            auto gate = iSWAP(qubit0, qubit1);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_SQISWAP_GATE:
        {
            auto qubit1 = qAlloc(qubit_array[1]);
            if (!qubit1)
            {
                throw runtime_error("qAlloc fail");
            }

            auto gate = SqiSWAP(qubit0, qubit1);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_RX_GATE:
        {
            m_iter++;
            float angle = getAngle(*m_iter);
            auto gate = RX(qubit0, angle);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_RY_GATE:
        {
            m_iter++;
            float angle = getAngle(*m_iter);
            auto gate = RY(qubit0, angle);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_RZ_GATE:
        {
            m_iter++;
            float angle = getAngle(*m_iter);
            auto gate = RZ(qubit0, angle);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_U1_GATE:
        {
            m_iter++;
            float angle = getAngle(*m_iter);
            auto gate = U1(qubit0, angle);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    case QPROG_NODE_TYPE_CPHASE_GATE:
        {
            m_iter++;
            float angle = getAngle(*m_iter);
            auto qubit1 = qAlloc(qubit_array[1]);
            if (!qubit1)
            {
                QCERR("qAlloc fail");
                throw runtime_error("qAlloc fail");
            }

            auto gate = CR(qubit0,  qubit1, angle);
            gate.setDagger(is_dagger);
            prog << gate;
        }
        break;
    default:
        QCERR("Bad QGate Type");
        throw runtime_error("Bad QGate Type");
        break;
    }
    return;
}

float QProgDataParse::getAngle(const pair<uint_t, DataNode> &data_node)
{
    ushort_t type_iter = (data_node.first & 0xffff) >> 1;
    if (QPROG_NODE_TYPE_GATE_ANGLE != type_iter)
    {
        QCERR("parsing DAT fails");
        throw runtime_error("parsing DAT fails");
    }

    return data_node.second.angle_data;
}

void QProgDataParse::parseQMeasureDataNode(QProg &prog, uint_t qubits_data)
{
    const int kQubitMax = 2;
    ushort_t qubit_array[kQubitMax] = {0};
    qubit_array[0] = qubits_data & 0xffff;
    qubit_array[1] = (qubits_data >> (kCountMoveBit));

    auto qubit = qAlloc(qubit_array[0]);
    if (!qubit)
    {
        QCERR("qAlloc fail");
        throw runtime_error("qAlloc fail");
    }

    auto cbit = cAlloc(qubit_array[1]);

    auto measure = Measure(qubit, cbit);
    prog << measure;
    return ;
}

void QProgDataParse::parseCExprCBitDataNode(const uint_t data)
{
    auto cbit = cAlloc(data);
    m_stack_cc.push(cbit);
    return;
}

void QProgDataParse::parseCExprOperateDataNode(const uint_t data)
{    
    switch (data)
    {
    case PLUS:
        {
            ClassicalCondition cc_right(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_left(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_root = cc_left + cc_right;
            m_stack_cc.push(cc_root);
        }
        break;
    case MINUS:
        {
            ClassicalCondition cc_right(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_left(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_root = cc_left - cc_right;
            m_stack_cc.push(cc_root);
        }
        break;
    case AND:
        {
            ClassicalCondition cc_right(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_left(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_root = cc_left && cc_right;
            m_stack_cc.push(cc_root);
        }
        break;
    case OR:
        {
            ClassicalCondition cc_right(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_left(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_root = cc_left || cc_right;
            m_stack_cc.push(cc_root);
        }
        break;
    case NOT:
        {
            ClassicalCondition cc_right(m_stack_cc.top());
            m_stack_cc.pop();
            ClassicalCondition cc_root = !cc_right;
            m_stack_cc.push(cc_root);
        }
        break;
    default:
        QCERR("Unknown internal error");
        throw runtime_error("Unknown internal error");
    }

    return ;
}


void QProgDataParse::parseQIfDataNode(QProg &prog, const uint_t data)
{
    ClassicalCondition cc(m_stack_cc.top());
    m_stack_cc.pop();

    uint_t tail_number_true_branch = data >> kCountMoveBit;
    uint_t tail_number_false_branch = data & 0xffff;
    QProg prog_true_branch = CreateEmptyQProg();
    m_iter++;
    parseDataNode(prog_true_branch, tail_number_true_branch);

    if (tail_number_false_branch)
    {
        QProg prog_false_brach = CreateEmptyQProg();
        m_iter++;
        parseDataNode(prog_false_brach, tail_number_false_branch);
        QIfProg prog_If = CreateIfProg(cc, &prog_true_branch, &prog_false_brach);
        prog << prog_If;
    }
    else
    {
        QIfProg prog_If = CreateIfProg(cc, &prog_true_branch);
        prog << prog_If;
    }

    return ;
}


void QProgDataParse::parseQWhileDataNode(QProg &prog, QProgDataParse::uint_t data)
{
    ClassicalCondition cc(m_stack_cc.top());
    m_stack_cc.pop();

    uint_t tail_number_true_branch = data >> kCountMoveBit;
    QProg prog_true_branch = CreateEmptyQProg();
    m_iter++;
    parseDataNode(prog_true_branch, tail_number_true_branch);

    QWhileProg prog_while = CreateWhileProg(cc, &prog_true_branch);
    prog << prog_while;

    return ;
}


void QProgDataParse::parseDataNode(QProg &prog, const uint_t tail_number)
{
    if(0 == tail_number)
    {
        return ;
    }

    ushort_t type = (m_iter->first & 0xffff) >> 1;
    uint_t data = m_iter->second.qubit_data;
    switch (type)
    {
    case QPROG_NODE_TYPE_PAULI_X_GATE:
    case QPROG_NODE_TYPE_PAULI_Y_GATE:
    case QPROG_NODE_TYPE_PAULI_Z_GATE:
    case QPROG_NODE_TYPE_X_HALF_PI:
    case QPROG_NODE_TYPE_Y_HALF_PI:
    case QPROG_NODE_TYPE_Z_HALF_PI:
    case QPROG_NODE_TYPE_HADAMARD_GATE:
    case QPROG_NODE_TYPE_T_GATE:
    case QPROG_NODE_TYPE_S_GATE:
    case QPROG_NODE_TYPE_CU_GATE:
    case QPROG_NODE_TYPE_CNOT_GATE:
    case QPROG_NODE_TYPE_CZ_GATE:
    case QPROG_NODE_TYPE_ISWAP_GATE:
    case QPROG_NODE_TYPE_SQISWAP_GATE:
    case QPROG_NODE_TYPE_RX_GATE:
    case QPROG_NODE_TYPE_RY_GATE:
    case QPROG_NODE_TYPE_RZ_GATE:
    case QPROG_NODE_TYPE_U1_GATE:
    case QPROG_NODE_TYPE_CPHASE_GATE:
        parseQGateDataNode(prog, m_iter->first, data);
        break;
    case QPROG_NODE_TYPE_MEASURE_GATE:
        parseQMeasureDataNode(prog, data);
        break;
    case QPROG_NODE_TYPE_QIF_NODE:
        parseQIfDataNode(prog, data);
        break;
    case QPROG_NODE_TYPE_QWHILE_NODE:
        parseQWhileDataNode(prog, data);
        break;
    case QPROG_NODE_TYPE_CEXPR_CBIT:
        parseCExprCBitDataNode(data);
        break;
    case QPROG_NODE_TYPE_CEXPR_OPERATOR:
        parseCExprOperateDataNode(data);
        break;
    default:
        QCERR("Bad type");
        throw runtime_error("Bad type");
        break;
    }

    uint_t node_count = m_iter->first >> kCountMoveBit;
    if (tail_number == node_count)
    {
        return ;
    }

    m_iter++;
    parseDataNode(prog, tail_number);
}


bool QPanda::binaryQProgFileParse(QProg &prog, const string &filename)
{
    QProgDataParse datParse(filename);
    if (!datParse.loadFile())
    {
        std::cout << "load file error" << std::endl;
        return false;
    }

    if (!datParse.parse(prog))
    {
        std::cout << "parse file error" << std::endl;
        return false;
    }

    return true;
}
















