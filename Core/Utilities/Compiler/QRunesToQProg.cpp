#include <regex>
#include <fstream>
#include <algorithm>
#include "Core/Utilities/Compiler/QRunesToQProg.h"
USING_QPANDA
using namespace std;

static vector<string> extract_value(string sQRunes)
{
    if ('%' == sQRunes[0])
    {
        vector<string> val_vec;
        return  val_vec;
    }

    smatch result;
    vector<string> val_vec;
    regex key("^X1|^Y1|^Z1|^[A-Z]+|[0-9]+(?!\\.)|-?[0-9]+\\.[0-9]+");

    string::const_iterator start_iter = sQRunes.begin();
    string::const_iterator end_iter = sQRunes.end();
    while (regex_search(start_iter, end_iter, result, key))
    {
        val_vec.emplace_back(result[0]);
        start_iter = result[0].second;
    }

    return val_vec;
}

QRunesToQProg::QRunesToQProg()
{
	typedef QGate(*gate_f1)(Qubit*);
    m_singleGateFunc.insert(make_pair("H", (gate_f1)H));
    m_singleGateFunc.insert(make_pair("T", (gate_f1)T));
    m_singleGateFunc.insert(make_pair("S", (gate_f1)S));

    m_singleGateFunc.insert(make_pair("X", (gate_f1)X));
    m_singleGateFunc.insert(make_pair("Y", (gate_f1)Y));
    m_singleGateFunc.insert(make_pair("Z", (gate_f1)Z));

    m_singleGateFunc.insert(make_pair("X1", (gate_f1)X1));
    m_singleGateFunc.insert(make_pair("Y1", (gate_f1)Y1));
    m_singleGateFunc.insert(make_pair("Z1", (gate_f1)Z1));

	typedef QGate(*gate_f2)(Qubit*, Qubit*);
    m_doubleGateFunc.insert(make_pair("CNOT", (gate_f2)CNOT));
    m_doubleGateFunc.insert(make_pair("CZ", (gate_f2)CZ));
    m_doubleGateFunc.insert(make_pair("ISWAP", (gate_f2)iSWAP));
    m_doubleGateFunc.insert(make_pair("SQISWAP", (gate_f2)SqiSWAP));

	typedef QGate(*gate_f3)(Qubit*, double);
    m_angleGateFunc.insert(make_pair("RX", (gate_f3)RX));
    m_angleGateFunc.insert(make_pair("RY", (gate_f3)RY));
    m_angleGateFunc.insert(make_pair("RZ", (gate_f3)RZ));
    m_angleGateFunc.insert(make_pair("U1", (gate_f3)U1));

	typedef QGate(*gate_f4)(Qubit*, Qubit*, double);
    m_doubleAngleGateFunc.insert(make_pair("CR", (gate_f4)CR));
}

size_t  QRunesToQProg::handleDaggerCircuit(std::shared_ptr<QNode> qNode, size_t pos)
{
    size_t cir_size{ 0 }, increment{ 0 };
    auto qCircuit = CreateEmptyCircuit();

    auto m_QRunes_value = extract_value(m_QRunes[pos]);
    string end_sign = m_QRunes_value.empty() ? "" : m_QRunes_value[0];

    for (; end_sign != "ENDDAGGER" && pos < m_QRunes.size();)
    {
        increment = traversalQRunes(pos, dynamic_pointer_cast<QNode>(qCircuit.getImplementationPtr()));
        pos += increment;
        cir_size += increment;

        m_QRunes_value = extract_value(m_QRunes[pos]);
        end_sign = m_QRunes_value.empty() ? "" : m_QRunes_value[0];
    }

    if (PROG_NODE == qNode->getNodeType())
    {
        auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
        if (nullptr == abstract_prog)
        {
            QCERR(" Error");
            throw invalid_argument("error");
        }

        qCircuit.setDagger(true);
        QProg prog = QProg(abstract_prog);
        prog << qCircuit;
    }
    else if (CIRCUIT_NODE == qNode->getNodeType())
    {
        auto  abstract_cir = dynamic_pointer_cast<AbstractQuantumCircuit>(qNode);
        if (nullptr == abstract_cir)
        {
            QCERR("Error");
            throw invalid_argument("error");
        }

        qCircuit.setDagger(true);
        QCircuit cir = QCircuit(abstract_cir);
        cir << qCircuit;
    }
    else
    {
        QCERR("Error");
        throw invalid_argument("Error");
    }

    return cir_size + 2;
}

size_t  QRunesToQProg::handleControlCircuit(std::shared_ptr<QNode> qNode, size_t pos)
{
    QVec ctr_qbits;
    QCircuit qCircuit = QCircuit();
    string end_sign = "ENDCONTROL ";
    for_each(m_QRunes_value.begin() + 1, m_QRunes_value.end(), [&](string value)
    {
        ctr_qbits.emplace_back(qvm->allocateQubitThroughPhyAddress(stoi(value)));
        end_sign.append(value).append(",");
    });
    end_sign.pop_back();

    size_t cir_size{ 0 }, increment{ 0 };
    QPANDA_ASSERT(pos >= m_QRunes.size(), "pos limits error");

    for (; m_QRunes[pos] != end_sign && pos < m_QRunes.size();)
    {
        increment = traversalQRunes(pos, dynamic_pointer_cast<QNode>(qCircuit.getImplementationPtr()));
        pos += increment;
        cir_size += increment;
    }

    if (PROG_NODE == qNode->getNodeType())
    {
        auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
        if (nullptr == abstract_prog)
        {
            QCERR("Error");
            throw invalid_argument("error");
        }
        QProg prog = QProg(abstract_prog);
        prog << qCircuit.control(ctr_qbits);
    }
    else if (CIRCUIT_NODE == qNode->getNodeType())
    {
        auto  abstract_cir = dynamic_pointer_cast<AbstractQuantumCircuit>(qNode);
        if (nullptr == abstract_cir)
        {
            QCERR("Error");
            throw invalid_argument("error");
        }
        QCircuit cir = QCircuit(abstract_cir);
        cir << qCircuit.control(ctr_qbits);
    }
    else
    {
        QCERR("Error");
        throw invalid_argument("Error");
    }

    return cir_size + 2;
}

size_t  QRunesToQProg::handleSingleGate(std::shared_ptr<QNode> qNode)
{
    auto iter = m_singleGateFunc.find(m_QRunes_value[0]);
    if (m_singleGateFunc.end() == iter)
    {
        QCERR("undefined Gate");
        throw invalid_argument("undefined Gate");
    }
    else
    {
        if (CIRCUIT_NODE == qNode->getNodeType())
        {
            auto  abstract_cir = dynamic_pointer_cast<AbstractQuantumCircuit>(qNode);
            if (nullptr == abstract_cir)
            {
                QCERR("CircuitError");
                throw invalid_argument("CircuitError");
            }
            QCircuit cir = QCircuit(abstract_cir);
            cir << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
            if (nullptr == abstract_prog)
            {
                QCERR("QProgError");
                throw invalid_argument("QProgError");
            }
            QProg prog = QProg(abstract_prog);
            prog << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])));
        }
        else
        {
            QCERR("NodeTypeError");
            throw invalid_argument("NodeTypeError");
        }

        return 1;
    }
}

size_t  QRunesToQProg::handleDoubleGate(std::shared_ptr<QNode> qNode)
{
    auto iter = m_doubleGateFunc.find(m_QRunes_value[0]);
    if (m_doubleGateFunc.end() == iter)
    {
        QCERR("undefined Gate");
        throw invalid_argument("undefined Gate");
    }
    else
    {
        if (CIRCUIT_NODE == qNode->getNodeType())
        {
            auto  abstract_cir = dynamic_pointer_cast<AbstractQuantumCircuit>(qNode);
            if (nullptr == abstract_cir)
            {
                QCERR("error");
                throw invalid_argument("error");
            }
            QCircuit cir = QCircuit(abstract_cir);
            cir << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
            if (nullptr == abstract_prog)
            {
                QCERR("error");
                throw invalid_argument("error");
            }
            QProg prog = QProg(abstract_prog);
            prog << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])));
        }

        return 1;
    }
}

size_t  QRunesToQProg::handleDoubleAngleGate(std::shared_ptr<QNode> qNode)
{
    auto iter = m_doubleAngleGateFunc.find(m_QRunes_value[0]);
    if (m_doubleAngleGateFunc.end() == iter)
    {
        QCERR("undefined Gate");
        throw invalid_argument("undefined Gate");
    }
    else
    {
        if (CIRCUIT_NODE == qNode->getNodeType())
        {
            auto  abstract_cir = dynamic_pointer_cast<AbstractQuantumCircuit>(qNode);
            if (nullptr == abstract_cir)
            {
                QCERR("error");
                throw invalid_argument("error");
            }
            QCircuit cir = QCircuit(abstract_cir);
            cir << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])),
                stod(m_QRunes_value[3]));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
            if (nullptr == abstract_prog)
            {
                QCERR("error");
                throw invalid_argument("error");
            }
            QProg prog = QProg(abstract_prog);
            prog << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])),
                stod(m_QRunes_value[3]));
        }
        return 1;
    }
}

size_t  QRunesToQProg::handleAngleGate(std::shared_ptr<QNode> qNode)
{
    auto iter = m_angleGateFunc.find(m_QRunes_value[0]);
    if (m_angleGateFunc.end() == iter)
    {
        QCERR("undefined Gate");
        throw invalid_argument("undefined Gate");
    }
    else
    {
        if (CIRCUIT_NODE == qNode->getNodeType())
        {
            auto  abstract_cir = dynamic_pointer_cast<AbstractQuantumCircuit>(qNode);
            if (nullptr == abstract_cir)
            {
                QCERR("Error");
                throw invalid_argument("error");
            }
            QCircuit cir = QCircuit(abstract_cir);
            cir << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                stod(m_QRunes_value[2]));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
            if (nullptr == abstract_prog)
            {
                QCERR("Formal Error");
                throw invalid_argument("error");
            }
            QProg prog = QProg(abstract_prog);
            prog << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                stod(m_QRunes_value[2]));
        }

        return 1;
    }
}

size_t  QRunesToQProg::handleMeasureGate(std::shared_ptr<QNode> qNode)
{
    if (nullptr == qNode || PROG_NODE != qNode->getNodeType())
    {
        QCERR("NodeError");
        throw invalid_argument("NodeError");
    }
    else
    {
        auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
        if (nullptr == abstract_prog)
        {
            QCERR("Formal Error");
            throw invalid_argument("error");
        }
        m_cbit_vec.emplace_back(qvm->allocateCBit(stoi(m_QRunes_value[2])));
        QProg prog = QProg(abstract_prog);
        prog << Measure(
            qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
            m_cbit_vec.back());

        return 1;
    }
}

size_t  QRunesToQProg::traversalQRunes(size_t pos, std::shared_ptr<QNode> qNode)
{
    if (nullptr == qNode)
    {
        QCERR("NodeError");
        throw invalid_argument("NodeError");
    }

    m_QRunes_value = extract_value(m_QRunes[pos]);
    if (m_QRunes_value.empty())
    {
        return 1;
    }
    string keyWord = m_QRunes_value[0];
    if (keyWord == "H" || keyWord == "T" || keyWord == "S" ||
        keyWord == "X" || keyWord == "Y" || keyWord == "Z" ||
        keyWord == "X1" || keyWord == "Y1" || keyWord == "Z1")
    {
        return handleSingleGate(qNode);
    }
    else if (keyWord == "RX" || keyWord == "RY" || keyWord == "RZ")
    {
        return handleAngleGate(qNode);
    }
    else if (keyWord == "CNOT" || keyWord == "CZ" || keyWord == "SQISWAP")
    {
        return handleDoubleGate(qNode);
    }
    else if (keyWord == "CR")
    {
        return handleDoubleAngleGate(qNode);
    }
    else if (keyWord == "MEASURE")
    {
        return handleMeasureGate(qNode);
    }
    else if (keyWord == "DAGGER")
    {
        return handleDaggerCircuit(qNode, ++pos);
    }
    else if (keyWord == "CONTROL")
    {
        return handleControlCircuit(qNode, ++pos);
    }
    else if (keyWord == "TOFFOLI")
    {
        return handleToffoliGate(qNode);
    }
    else
    {
        if (m_QRunes[pos].find('%'))
        {
            return 1;
        }
        else
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
    }
}

size_t QRunesToQProg::handleToffoliGate(std::shared_ptr<QNode> qNode)
{
    auto Toffoli = X(qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[3])));
    Toffoli.setControl({ { qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])) ,
        qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])) } });

    if (CIRCUIT_NODE == qNode->getNodeType())
    {
        auto  abstract_cir = dynamic_pointer_cast<AbstractQuantumCircuit>(qNode);
        if (nullptr == abstract_cir)
        {
            QCERR("CircuitError");
            throw invalid_argument("CircuitError");
        }
        QCircuit cir = QCircuit(abstract_cir);
        cir << Toffoli;
    }
    else if (PROG_NODE == qNode->getNodeType())
    {
        auto abstract_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qNode);
        if (nullptr == abstract_prog)
        {
            QCERR("QProgError");
            throw invalid_argument("QProgError");
        }
        QProg prog = QProg(abstract_prog);
        prog << Toffoli;
    }
    else
    {
        QCERR("NodeTypeError");
        throw invalid_argument("NodeTypeError");
    }

    return 1;
}


void QRunesToQProg::qRunesParser(std::string sFilePath, QProg& prog, QuantumMachine*qvm)
{
    ifstream fin(sFilePath);
    if (!fin)
    {
        QCERR("FileOpenError");
        throw init_fail("Open File Failed");
    }
    else
    {
        smatch result;
        regex dec("[0-9]+");
        regex val("PI/[0-9]+|PI", regex::icase);

        string sQRunes;
        while (!fin.eof())
        {
            getline(fin, sQRunes);
            if (regex_search(sQRunes, result, val))
            {
                string temp = result[0];
                if (regex_search(temp, result, dec))
                {
                    string theta = to_string(PI / stod(result[0]));
                    sQRunes = regex_replace(sQRunes, val, theta);
                }
                else
                {
                    sQRunes = regex_replace(sQRunes, val, to_string(PI));
                }
            }
            m_QRunes.emplace_back(sQRunes);
        }
        fin.close();

        if (regex_search(m_QRunes[0], result, dec))
        {
            int qbit_num = stoi(result[0]);
            for (int i = 0; i < qbit_num; ++i)
            {
                qvm->allocateQubitThroughPhyAddress(i);
            }
        }

        if (regex_search(m_QRunes[1], result, dec))
        {
            int cbit_num = stoi(result[0]);
            for (int i = 0; i < cbit_num; ++i)
            {
                qvm->allocateCBit(i);
            }
        }

        for (size_t pos = 2; pos < m_QRunes.size();)
        {
            //auto qnode = dynamic_pointer_cast<QNode>(prog.getImplementationPtr());

            //auto abs_prog = dynamic_pointer_cast<AbstractQuantumProgram>(qnode);
            //QProg prog(abs_prog);
            pos += traversalQRunes(pos, dynamic_pointer_cast<QNode>(prog.getImplementationPtr()));
        }
    }
}

std::vector<ClassicalCondition> QPanda::transformQRunesToQProg(std::string sFilePath, QProg& prog, QuantumMachine* qvm)
{
    QRunesToQProg qRunesTraverse;
    qRunesTraverse.qvm = qvm;
    qRunesTraverse.qRunesParser(sFilePath, prog, qvm);

    return qRunesTraverse.m_cbit_vec;
}
