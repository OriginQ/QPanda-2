#include <regex>
#include <fstream>
#include <algorithm>
#include "Core/Utilities/Transform/QRunesToQProg.h"
USING_QPANDA
using namespace std;

QGate _iSWAP(Qubit * targitBit_fisrt, Qubit * targitBit_second)
{
    return iSWAP(targitBit_fisrt, targitBit_second);
}

static vector<string> extract_value(string sQRunes)
{
    if ('%' == sQRunes[0])
    {
        vector<string> val_vec;
        return  val_vec;
    }

    smatch result;
    vector<string> val_vec;
    //regex key("[A-Z]+|[0-9]+(?!\\.)|-?[0-9]+\\.[0-9]+");
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
    m_singleGateFunc.insert(make_pair("H", H));
    m_singleGateFunc.insert(make_pair("T", T));
    m_singleGateFunc.insert(make_pair("S", S));

    m_singleGateFunc.insert(make_pair("X", X));
    m_singleGateFunc.insert(make_pair("Y", Y));
    m_singleGateFunc.insert(make_pair("Z", Z));

    m_singleGateFunc.insert(make_pair("X1", X1));
    m_singleGateFunc.insert(make_pair("Y1", Y1));
    m_singleGateFunc.insert(make_pair("Z1", Z1));

    m_doubleGateFunc.insert(make_pair("CNOT", CNOT));
    m_doubleGateFunc.insert(make_pair("CZ", CZ));
    m_doubleGateFunc.insert(make_pair("ISWAP", _iSWAP));
    m_doubleGateFunc.insert(make_pair("SQISWAP", SqiSWAP));

    m_angleGateFunc.insert(make_pair("RX", RX));
    m_angleGateFunc.insert(make_pair("RY", RY));
    m_angleGateFunc.insert(make_pair("RZ", RZ));
    m_angleGateFunc.insert(make_pair("U1", U1));

    m_doubleAngleGateFunc.insert(make_pair("CR", CR));
}

size_t  QRunesToQProg::handleDaggerCircuit(QNode *qNode, size_t pos)
{

    cout << qNode << endl;
    size_t cir_size{ 0 }, increment{ 0 };
    auto qCircuit = CreateEmptyCircuit();
    cout << qCircuit.getImplementationPtr().get() << endl;

    for (; m_QRunes[pos] != "ENDDAGGER" && pos < m_QRunes.size();)
    {
        increment = traversalQRunes(pos, &qCircuit);
        pos += increment;
        cir_size += increment;
    }

    if (PROG_NODE == qNode->getNodeType())
    {
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR(" Error");
            throw invalid_argument("error");
        }
        (*qProg) << qCircuit.dagger();
    }
    else if (CIRCUIT_NODE == qNode->getNodeType())
    {
        QCircuit *qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            QCERR("Error");
            throw invalid_argument(" error");
        }
        (*qCir) << qCircuit.dagger();
    }
    else
    {
        QCERR(" Error");
        throw invalid_argument("Error");
    }

    return cir_size + 2;
}

size_t  QRunesToQProg::handleControlCircuit(QNode *qNode, size_t pos)
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
    for (; m_QRunes[pos] != end_sign && pos < m_QRunes.size();)
    {
        increment = traversalQRunes(pos, &qCircuit);
        pos += increment;
        cir_size += increment;
    }

    if (PROG_NODE == qNode->getNodeType())
    {
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR("Error");
            throw invalid_argument("error");
        }
        (*qProg) << qCircuit.control(ctr_qbits);
    }
    else if (CIRCUIT_NODE == qNode->getNodeType())
    {
        QCircuit *qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            QCERR("Error");
            throw invalid_argument("error");
        }
        (*qCir) << qCircuit.control(ctr_qbits);
    }
    else
    {
        QCERR(" Error");
        throw invalid_argument("Error");
    }

    return cir_size + 2;
}

size_t  QRunesToQProg::handleSingleGate(QNode *qNode)
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
            QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
            if (nullptr == qCir)
            {
                QCERR("CircuitError");
                throw invalid_argument("CircuitError");
            }
            (*qCir) << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            QProg * qProg = dynamic_cast<QProg*>(qNode);
            if (nullptr == qProg)
            {
                QCERR("QProgError");
                throw invalid_argument("QProgError");
            }

            (*qProg) << iter->second(
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

size_t  QRunesToQProg::handleDoubleGate(QNode *qNode)
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
            QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
            if (nullptr == qCir)
            {
                QCERR("error");
                throw invalid_argument(" error");
            }

            (*qCir) << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            QProg * qProg = dynamic_cast<QProg*>(qNode);
            if (nullptr == qProg)
            {
                QCERR("error");
                throw invalid_argument(" error");
            }

            (*qProg) << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])));
        }

        return 1;
    }
}

size_t  QRunesToQProg::handleDoubleAngleGate(QNode *qNode)
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
            QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
            if (nullptr == qCir)
            {
                QCERR("error");
                throw invalid_argument(" error");
            }
            (*qCir) << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])),
                stod(m_QRunes_value[3]));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            QProg * qProg = dynamic_cast<QProg*>(qNode);
            if (nullptr == qProg)
            {
                QCERR("error");
                throw invalid_argument("error");
            }
            (*qProg) << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])),
                stod(m_QRunes_value[3]));
        }
        return 1;
    }
}

size_t  QRunesToQProg::handleAngleGate(QNode *qNode)
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
            QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
            if (nullptr == qCir)
            {
                QCERR("Error");
                throw invalid_argument(" error");
            }

            (*qCir) << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                stod(m_QRunes_value[2]));
        }
        else if (PROG_NODE == qNode->getNodeType())
        {
            QProg * qProg = dynamic_cast<QProg*>(qNode);
            if (nullptr == qProg)
            {
                QCERR("Formal Error");
                throw invalid_argument(" error");
            }
            (*qProg) << iter->second(
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
                stod(m_QRunes_value[2]));
        }

        return 1;
    }
}

size_t  QRunesToQProg::handleMeasureGate(QNode *qNode)
{
    cout << qNode << endl;

    if (nullptr == qNode || PROG_NODE != qNode->getNodeType())
    {
        cout << qNode->getNodeType() << endl;
        QCERR("NodeError");
        throw invalid_argument("NodeError");
    }
    else
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        m_cbit_vec.emplace_back(qvm->allocateCBit(stoi(m_QRunes_value[2])));
        *qProg << Measure(
            qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])),
            m_cbit_vec.back());

        return 1;
    }
}

size_t  QRunesToQProg::traversalQRunes(size_t pos, QNode *qNode)
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
    else if (keyWord == "CNOT" || keyWord == "CZ" || 
             keyWord == "SQISWAP" || keyWord == "ISWAP")
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

size_t QRunesToQProg::handleToffoliGate(QNode* qNode)
{
    if (CIRCUIT_NODE == qNode->getNodeType())
    {
        QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            QCERR("CircuitError");
            throw invalid_argument("CircuitError");
        }

        (*qCir) << X(qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[3])))
            .control({ qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])) ,
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])) });
    }
    else if (PROG_NODE == qNode->getNodeType())
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR("QProgError");
            throw invalid_argument("QProgError");
        }

        (*qProg) << X(qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[3])))
            .control({ qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[1])) ,
                qvm->allocateQubitThroughPhyAddress(stoi(m_QRunes_value[2])) });
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
            pos += traversalQRunes(pos, &prog);
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
