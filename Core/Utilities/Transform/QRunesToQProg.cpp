#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <string.h>
#include "QRunesToQProg.h"
#include "QPanda.h"
USING_QPANDA
using namespace std;

#define EXPR                int notSize{ 0 };\
                            for (; cExpr.substr(pos + notSize + 1, 1) == "!"; notSize++);\
                            if (cExpr.substr(pos + notSize + 1, 1) != "c")\
                            {\
                                QCERR("Formal Error");\
                                throw invalid_argument("Formal Error");\
                            }\
                            for (; (pos + notSize + NumSize + 2) < cExpr.length() &&\
                                    cExpr.substr(pos + notSize + NumSize + 2, 1) != "|" &&\
                                    cExpr.substr(pos + notSize + NumSize + 2, 1) != "+" &&\
                                    cExpr.substr(pos + notSize + NumSize + 2, 1) != "&" &&\
                                    cExpr.substr(pos + notSize + NumSize + 2, 1) != "-";\
                                    NumSize++);\
                            if (NumSize == 0)\
                            {\
                                QCERR("Formal Error");\
                                throw invalid_argument("Formal Error");\
                            }\
                            int creg_addr = isIntNum(cExpr.substr(pos + notSize + 2, NumSize));\
                            checkNumberLegal(creg_addr, m_all_cregs.size());\
                            ClassicalCondition condition_exper = m_all_cregs[creg_addr];\
                            for (int i = 0; i < notSize; i++)\
                            {\
                                condition_exper = !condition_exper;\
                            }\
                            pos = pos + notSize + NumSize + 2;\


int operationType(char c) 
{
    if (c == '+' || c == '-') 
    {
        return 1;
    }
    else if (c == '*' || c == '/') 
    {
        return 2;
    }
    else 
    {
        return 0;
    }

}

vector<string> operationStack(string exper) 
{
    vector<string> operation_vector;

    stack<char> operation;
    for (int i = 0; i < exper.length(); i++) 
    {
        string tmp = "";
        switch (exper[i]) 
        {
        case '+':
        case '-':
        case '*':
        case '/':
            if (operation.empty() || operation.top() == '(') 
            {
                operation.push(exper[i]);
            }
            else {
                while (!operation.empty() && operationType(operation.top()) >= operationType(exper[i])) 
                {
                    tmp += operation.top();
                    operation_vector.push_back(tmp);
                    operation.pop();
                    tmp = "";
                }
                operation.push(exper[i]);
            }
            break;

        case '(':
            operation.push(exper[i]);
            break;
        case ')':
            while (operation.top() != '(') 
            {
                tmp += operation.top();
                operation_vector.push_back(tmp);
                operation.pop();
                tmp = "";
            }
            operation.pop();
            break;

        default:
            if ((exper[i] >= '0' && exper[i] <= '9')) 
            {
                tmp += exper[i];
                while (i + 1 < exper.size() && exper[i + 1] >= '0' && exper[i + 1] <= '9' || exper[i + 1] == '.')
                {
                    tmp += exper[i + 1];
                    ++i;
                }
                operation_vector.push_back(tmp);
            }
        }
    }
    while (!operation.empty()) 
    {
        string tmp = "";
        tmp += operation.top();
        operation_vector.push_back(tmp);
        operation.pop();
    }
    return operation_vector;
}

double calculationResult(vector<string> calculationsolve) 
{
    stack<double> operation;

    double num, op1, op2;
    for (int i = 0; i < calculationsolve.size(); i++) 
    {
        string tmp = calculationsolve[i];
        if (tmp[0] >= '0'&&tmp[0] <= '9') 
        {
            num = atof(tmp.c_str());
            operation.push(num);
        }
        else if (calculationsolve[i] == "+")
        {
            op2 = operation.top();
            operation.pop();
            op1 = operation.top();
            operation.pop();
            operation.push(op1 + op2);
        }
        else if (calculationsolve[i] == "-")
        {
            op2 = operation.top();
            operation.pop();
            op1 = operation.top();
            operation.pop();
            operation.push(op1 - op2);
        }
        else if (calculationsolve[i] == "*")
        {
            op2 = operation.top();
            operation.pop();
            op1 = operation.top();
            operation.pop();
            operation.push(op1*op2);
        }
        else if (calculationsolve[i] == "/")
        {
            op2 = operation.top();
            operation.pop();
            op1 = operation.top();
            operation.pop();
            operation.push(op1 / op2);
        }
        else
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
    }
    return operation.top();
}

double calculationSolve(string exper) 
{
    for (int i = 0; i < exper.length(); i++)
    {
        if (exper[i] == '-')
        {
            if (i == 0)
            {
                exper.insert(0, 1, '0');
            }
            else if (exper[i - 1] == '(')
            {
                exper.insert(i, 1, '0');
            }
        }
    }

    vector<string> calculationsolve = operationStack(exper);
    double result = calculationResult(calculationsolve);
    return result;
}

void checkAngleExper(string str)
{
    stack<char> s;

    for (int i = 0; i < str.length(); i++)
    {
        switch (str[i])
        {
        case '(':
            switch (str[i - 1])
            {
            case '(':
            case '+':
            case '-':
            case '*':
            case '/':
            case NULL:break;
            default:
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
                break;
            }
            switch (str[i + 1])
            {
            case ')':
            case '+':
            case '*':
            case '/':
            case '.':
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
                     break;

            default: 
                break;
            }
            s.push(str[i]);
            break;

        case ')':
        {
            if (s.empty() || s.top() != '(' || str[i - 1] == '(')
            {
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
            }
            switch (str[i - 1])
            {
            case '(':
            case '+':
            case '-':
            case '*':
            case '/':
            case '.':
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
                break;
            default:
                break;

            }
            switch (str[i + 1])
            {
            case ')':
            case '+':
            case '-':
            case '*':
            case '/':
            case NULL:break;
            default:
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
                break;

            }
            s.pop();
            break;

        }
        default:
            break;

        }
    }
    if (!s.empty())
    {
        QCERR("Formal Error");
        throw invalid_argument("Formal Error");
    }

    for (auto iter=str.begin();iter!=str.end();iter++)
    {
        if ('+'!= *iter && '-' != *iter && '*' != *iter && '/' != *iter && 
            '(' != *iter && ')' != *iter && '.' != *iter && '0' != *iter &&
            '1' != *iter && '2' != *iter && '3' != *iter && '4' != *iter &&
            '5' != *iter && '6' != *iter && '7' != *iter&&'8' != *iter && '9' != *iter)
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
        if ((*iter == '+') || (*iter == '-') || (*iter == '*')
            || (*iter == '/'))
        {
            if ((iter + 1) == str.end())
            {
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
            }
            else if ((*(iter + 1) == '+') || (*(iter + 1) == '-') || (*(iter + 1) == '*')
                || (*(iter + 1) == '/'))
            {
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
            }
            else if ((*iter == '/') && (*(iter + 1) == '0'))
            {
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
            }
            else
            {
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
            }
        }
    }
}

QRunesToQprog::QRunesToQprog(string sFilePath):
m_sFilePath(sFilePath)
{
    m_keyWords.emplace_back("QINIT");
    m_keyWords.emplace_back("CREG");
    m_keyWords.emplace_back("X");
    m_keyWords.emplace_back("Y");
    m_keyWords.emplace_back("Z");

    m_keyWords.emplace_back("H");
    m_keyWords.emplace_back("T");
    m_keyWords.emplace_back("S");

    m_keyWords.emplace_back("X1");
    m_keyWords.emplace_back("Y1");
    m_keyWords.emplace_back("Z1");

    m_keyWords.emplace_back("RX");
    m_keyWords.emplace_back("RY");
    m_keyWords.emplace_back("RZ");
    m_keyWords.emplace_back("U1");

    m_keyWords.emplace_back("CNOT");
    m_keyWords.emplace_back("CZ");

    m_keyWords.emplace_back("ISWAP");
    m_keyWords.emplace_back("SQISWAP");

    m_keyWords.emplace_back("DAGGER");
    m_keyWords.emplace_back("ENDAGGER");
    m_keyWords.emplace_back("CONTROL");
    m_keyWords.emplace_back("ENDCONTROL");

    m_keyWords.emplace_back("QIF");
    m_keyWords.emplace_back("ELSE");
    m_keyWords.emplace_back("ENDQIF");
    m_keyWords.emplace_back("QWHILE");
    m_keyWords.emplace_back("ENDQWHILE");

    m_keyWords.emplace_back("MEASURE");

    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("H",  H));
    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("T",  T));
    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("S",  S));

    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("X",  X));
    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("Y",  Y));
    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("Z",  Z));

    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("X1", X1));
    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("Y1", Y1));
    m_singleGateFunc.insert(pair<string, std::function<QGate(Qubit *)> >("Z1", Z1));

    m_doubleGateFunc.insert(pair<string, std::function<QGate(Qubit *, Qubit *)> >("CNOT", CNOT));
    m_doubleGateFunc.insert(pair<string, std::function<QGate(Qubit *, Qubit *)> >("CZ", CZ));
    //m_doubleGateFunc.insert(pair<string, std::function<QGate(Qubit *, Qubit *)> >("ISWAP", iSWAP);
    m_doubleGateFunc.insert(pair<string, std::function<QGate(Qubit *, Qubit *)> >("SQISWAP", SqiSWAP));

    m_angleGateFunc.insert(pair<string, std::function<QGate(Qubit *, double)> >("RX", RX));
    m_angleGateFunc.insert(pair<string, std::function<QGate(Qubit *, double)> >("RY", RY));
    m_angleGateFunc.insert(pair<string, std::function<QGate(Qubit *, double)> >("RZ", RZ));
    m_angleGateFunc.insert(pair<string, std::function<QGate(Qubit *, double)> >("U1", U1));

}

QRunesToQprog::~QRunesToQprog()
{}

int isIntNum(string str)
{
    stringstream sin(str);
    int int_number;

    char char_number;

    if (!(sin >> int_number))
    {
        QCERR("isIntError");
        throw invalid_argument("Formal Error");
    }
    if (sin >> char_number)
    {
        QCERR("Formal Error");
        throw invalid_argument("Formal Error");
    }

    return stoi(str);
}

void checkNumberLegal(int number,size_t max_number)
{
    if ((number < 0 ) || (number>=max_number))
    {
        QCERR("Formal Error");
        throw invalid_argument("Illegal Number");
    }
}

void countKeywords(vector<string> &keyVec)
{
    auto countFunc = [=](string keyWords) 
    { return (int)count(keyVec.begin(), keyVec.end(), keyWords); };

    if (countFunc("DAGGER")  != countFunc("ENDDAGGER")  ||
        countFunc("CONTROL") != countFunc("ENDCONTROL") ||
        countFunc("QIF")     != countFunc("ENDQIF")     ||
        countFunc("QWHILE")  != countFunc("ENDQWHILE")  )
        {
            QCERR("MatchingError");
            throw invalid_argument("Illegal KeyWords");
        }
}

ClassicalCondition QRunesToQprog::checkIfWhileLegal(string cExpr)
{
    int notSize{ 0 }, firstNumSize{ 0 }, NumSize{ 0 };
    if (cExpr.substr(0, 1) == "!")
    {
        for (; cExpr.substr(notSize, 1) == "!"; notSize++);
        if (   cExpr.substr(notSize, 1) != "c")
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
    }
    else if (cExpr.substr(0, 1) == "c")
    {}
    else
    {
        QCERR("Formal Error");
        throw invalid_argument("Formal Error");
    }
    for (; cExpr.substr(notSize + firstNumSize + 1, 1) != "|" && 
           cExpr.substr(notSize + firstNumSize + 1, 1) != "+" && 
           cExpr.substr(notSize + firstNumSize + 1, 1) != "&" &&
           cExpr.substr(notSize + firstNumSize + 1, 1) != "-" &&
           cExpr.length() > notSize + firstNumSize + 1;
           firstNumSize++);
    if (firstNumSize == 0)
    {
        QCERR("Formal Error");
        throw invalid_argument("Formal Error");
    }

    isIntNum(cExpr.substr(notSize + 1, firstNumSize ));
    int creg_addr = stoi(cExpr.substr(notSize + 1, firstNumSize + 1));
    checkNumberLegal(creg_addr, m_all_cregs.size());
    ClassicalCondition condition_exper = m_all_cregs[creg_addr];

    for (int i = 0; i < notSize; i++)
    {
        condition_exper = !condition_exper;
    }

    for (int pos = firstNumSize + 1 + notSize; pos != cExpr.length();)
    {
        NumSize = 0;
        if (cExpr.substr(pos, 1) == "+")
        {
            EXPR

            condition_exper = condition_exper + condition_exper;
        }

        else if (cExpr.substr(pos, 1) == "-")
        {
            EXPR

            condition_exper = condition_exper - condition_exper;
        }

        else if (cExpr.substr(pos, 1) == "|" && cExpr.substr(++pos, 1) != "|")
        {
            EXPR

            condition_exper = condition_exper || condition_exper;
        }

        else if (cExpr.substr(pos, 1) == "&" && cExpr.substr(++pos, 1) != "&")
        {
            EXPR

            condition_exper = condition_exper && condition_exper;
        }

        else
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
    }

    return condition_exper;

}

void QRunesToQprog::qRunesAllocation(vector<string> &m_QRunes, QProg& newQProg)
{
    string qubits_number = m_QRunes[0].substr(m_QRunes[0].find(" ") + 1, m_QRunes[0].length());
    string cregs_number  = m_QRunes[1].substr(m_QRunes[1].find(" ") + 1, m_QRunes[1].length());

    for (int i = 0; i < isIntNum(qubits_number); i++)
    {
        m_all_qubits.emplace_back(qAlloc());
    }

    for (int i = 0; i < isIntNum(cregs_number); i++)
    {
        m_all_cregs.emplace_back(cAlloc());
    }

    for (auto iter = m_QRunes.begin() + 2; iter != m_QRunes.end();)
    {
        iter += QRunesToQprog::traversalQRunes(iter, &newQProg);
    }

}

int  QRunesToQprog::handleDaggerCircuit(vector<string>::iterator iter, QNode *qNode)
{
    int node_type = qNode->getNodeType();
    int cirSize{ 0 }, increment{ 0 };

    auto qCircuit = CreateEmptyCircuit();

    for (; (*iter != "ENDAGGER");)
    {
        increment = traversalQRunes(iter, &qCircuit);
        iter += increment;
        cirSize += increment;
    }
    qCircuit.setDagger(true);
    if (PROG_NODE == node_type)
    {
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR(" Error");
            throw invalid_argument(" error");
        }
        *qProg << qCircuit;
    }
    else if (CIRCUIT_NODE == node_type)
    {
        QCircuit *qCirCuit = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCirCuit)
        {
            QCERR(" Error");
            throw invalid_argument(" error");
        }
        *qCirCuit << qCircuit;
    }
    else
    {
        QCERR(" Error");
        throw invalid_argument("Error");
    }

    return cirSize;

}

int  QRunesToQprog::handleControlCircuit(vector<string>::iterator iter, QNode *qNode,
                                         vector<Qubit*> &all_ctr_qubits, 
                                         string &cExpr)
{
    auto qCircuit = CreateEmptyCircuit();
    int cirSize{ 0 }, increment{ 0 };

    for (; (*iter).substr(0, (*iter).find(" ")) != "ENDCONTROL";)
    {
        increment = traversalQRunes(iter, &qCircuit);
        iter += increment;
        cirSize += increment;
    }

    if ((*iter).substr((*iter).find(" ") + 1) != cExpr)
    {
        QCERR(" Error");
        throw invalid_argument("Error");
    }
    qCircuit.setControl(all_ctr_qubits);

    if (PROG_NODE == qNode->getNodeType())
    {
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR(" Error");
            throw invalid_argument(" error");
        }
        *qProg << qCircuit;
    }
    else if (CIRCUIT_NODE == qNode->getNodeType())
    {
        QCircuit *qCirCuit = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCirCuit)
        {
            QCERR(" Error");
            throw invalid_argument(" error");
        }
        *qCirCuit << qCircuit;
    }
    else
    {
        QCERR(" Error");
        throw invalid_argument("Error");
    }

    return cirSize;

}

int  QRunesToQprog::handleSingleGate (vector<string>::iterator iter, QNode *qNode,
                                     const string &gateName,int qubit_addr)
{
    if ( CIRCUIT_NODE == qNode->getNodeType())
    {
        QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            QCERR("CircuitError");
            throw invalid_argument("CircuitError");
        }

        auto iter = m_singleGateFunc.find(gateName);
        if (iter != m_singleGateFunc.end())
        {
            *qCir << iter->second(m_all_qubits[qubit_addr]);
        } 
        else
        {
            QCERR("undefined Gate");
            throw invalid_argument("undefined Gate");
        }
    }
    else if ( PROG_NODE == qNode->getNodeType())
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR("QProgError");
            throw invalid_argument("QProgError");
        }

        auto iter = m_singleGateFunc.find(gateName);
        if (iter != m_singleGateFunc.end())
        {
            *qProg << iter->second(m_all_qubits[qubit_addr]);
        }
        else
        {
            QCERR("undefined Gate");
            throw invalid_argument("undefined Gate");
        }

    }
    else
    {
        QCERR("error");
        throw invalid_argument(" error");
    }
    return 1;
}

int  QRunesToQprog::handleDoubleGate (vector<string>::iterator iter, QNode *qNode, 
                                     const string &gateName, int ctr_qubit_addr,int tar_qubit_addr)
{
    if (CIRCUIT_NODE == qNode->getNodeType())
    {
        QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            QCERR("error");
            throw invalid_argument(" error");
        }

        auto iter = m_doubleGateFunc.find(gateName);
        if (iter != m_doubleGateFunc.end())
        {
            *qCir << iter->second(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else
        {
            QCERR("undefined Gate");
            throw invalid_argument("undefined Gate");
        }

    }
    else if (PROG_NODE == qNode->getNodeType())
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR("error");
            throw invalid_argument(" error");
        }
        auto iter = m_doubleGateFunc.find(gateName);
        if (iter != m_doubleGateFunc.end())
        {
            *qProg << iter->second(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else
        {
            QCERR("undefined Gate");
            throw invalid_argument("undefined Gate");
        }
    }
    return 1;
}

int  QRunesToQprog::handleAngleGate  (vector<string>::iterator iter, QNode *qNode, 
                                     const string &gateName, int qubit_addr, double gate_angle)
{
    if (CIRCUIT_NODE == qNode->getNodeType())
    {
        QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            QCERR("Error");
            throw invalid_argument(" error");
        }

        auto iter = m_angleGateFunc.find(gateName);
        if (iter != m_angleGateFunc.end())
        {
            *qCir << iter->second(m_all_qubits[qubit_addr], gate_angle);
        }
        else
        {
            QCERR("undefined Gate");
            throw invalid_argument("undefined Gate");
        }

    }
    else if (PROG_NODE == qNode->getNodeType())
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR("Formal Error");
            throw invalid_argument(" error");
        }
        auto iter = m_angleGateFunc.find(gateName);
        if (iter != m_angleGateFunc.end())
        {
            *qProg << iter->second(m_all_qubits[qubit_addr], gate_angle);
        }
        else
        {
            QCERR("undefined Gate");
            throw invalid_argument("undefined Gate");
        }
    }
    return 1;
}

int  QRunesToQprog::handleMeasureGate(vector<string>::iterator iter, QNode *qNode, 
                                     const string &keyWord, int qubit_addr, int creg_addr)
{
    if (nullptr == qNode || PROG_NODE != qNode->getNodeType())
    {
        QCERR("NodeError");
        throw invalid_argument("NodeError");
    }
    else 
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR("error");
            throw invalid_argument(" error");
        }
        *qProg << Measure(m_all_qubits[qubit_addr], m_all_cregs[creg_addr]);
    }
    return 1;
}

int  QRunesToQprog::handleQIfProg(vector<string>::iterator iter, QNode *qNode,
                                     ClassicalCondition &condition_exper)
{
    if (nullptr == qNode || PROG_NODE != qNode->getNodeType())
    {
        QCERR("NodeError");
        throw invalid_argument("NodeError");
    }

    auto trueProg = CreateEmptyQProg();
    auto falseProg= CreateEmptyQProg();

    int cirSize{ 0 }, increment{ 0 }, elseincrement{ 0 };
    for (; (*iter != "ENDQIF") && (*iter != "ELSE");)
    {
        increment = traversalQRunes(iter, &trueProg);
        iter += increment;
        cirSize += increment;
    }

    if (*iter == "ELSE")
    {
        ++iter;
        ++cirSize;

        for (; *iter != "ENDQIF" ;)
        {
            increment = traversalQRunes(iter, &falseProg);
            iter += increment;
            cirSize += increment;
        }
        auto ifProg = CreateIfProg(condition_exper, &trueProg, &falseProg);
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR(" error");
            throw invalid_argument(" error");
        }
        *qProg << ifProg;
    }
    else
    {
        auto ifProg = CreateIfProg(condition_exper, &trueProg);
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            QCERR(" error");
            throw invalid_argument(" error");
        }
        *qProg << ifProg;
    }

    return cirSize;
}

int  QRunesToQprog::handleQWhileProg(vector<string>::iterator iter, QNode *qNode, 
                                     ClassicalCondition &condition_exper)
{
    if (nullptr == qNode || PROG_NODE != qNode->getNodeType())
    {
        QCERR("NodeError");
        throw invalid_argument("NodeError");
    }

    auto trueProg=CreateEmptyQProg();
    int cirSize{ 0 }, increment{ 0 };

    for (; *iter != "ENDQWHILE";)
    {
        increment = traversalQRunes(iter, &trueProg);
        iter += increment;
        cirSize += increment;
    }
    auto ifProg = CreateWhileProg(condition_exper, &trueProg);
    QProg *qProg = dynamic_cast<QProg*>(qNode);
    if (nullptr == qProg)
    {
        QCERR("error");
        throw invalid_argument(" error");
    }
    *qProg << ifProg;

    return cirSize;
}

int  QRunesToQprog::traversalQRunes(vector<string>::iterator iter, QNode *qNode)
{
    if (nullptr == qNode)
    {
        QCERR("NodeError");
        throw invalid_argument("NodeError");
    }

    string keyWord = (*iter).substr(0, (*iter).find(" "));
    if (keyWord == "H"  || keyWord == "T"  || keyWord == "S"||
        keyWord == "X"  || keyWord == "Y"  || keyWord == "Z"||
        keyWord == "X1" || keyWord == "Y1" || keyWord == "Z1")
    {
        int qubit_addr = isIntNum((*iter).substr((*iter).find(" ") + 1));
        checkNumberLegal(qubit_addr, m_all_qubits.size());
        return handleSingleGate(iter, qNode, keyWord, qubit_addr);
    }

    else if (keyWord == "RX" || keyWord == "RY" || keyWord == "RZ" )
    {
        if ((-1) == (*iter).substr((*iter).find(" ") + 1).find(","))
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
        if ((*iter).substr((*iter).find(",") + 1,1)!="\"" && 
            (*iter).substr((*iter).length()-1,1) != "\"")
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }

        int NumSize = (int)((*iter).find(",")-(*iter).find(" ")-1);
        
        int qubit_addr = isIntNum((*iter).substr(3, NumSize));
        checkNumberLegal(qubit_addr, m_all_qubits.size());

        checkAngleExper((*iter).substr((*iter).find(",") + 2, (*iter).length() - (*iter).find(",") - 3));
        double gate_angle = calculationSolve((*iter).substr((*iter).find(",") + 2, (*iter).length() - (*iter).find(",") - 3));
        return handleAngleGate(++iter, qNode, keyWord, qubit_addr,gate_angle);
    }

    else if (keyWord == "CNOT" || keyWord == "CZ" || keyWord == "ISWAP" || keyWord == "SQISWAP")
    {
        if ((-1) == (*iter).substr((*iter).find(" ") + 1).find(","))
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }

        int NumSize = (int)((*iter).find(",") - (*iter).find(" ") - 1);
        int ctr_qubit_addr = isIntNum((*iter).substr((*iter).find(" ") + 1, NumSize));
        int tar_qubit_addr = isIntNum((*iter).substr((*iter).find(",") + 1));

        checkNumberLegal(ctr_qubit_addr, m_all_qubits.size());
        checkNumberLegal(tar_qubit_addr, m_all_qubits.size());

        if ((ctr_qubit_addr == tar_qubit_addr))
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
        return handleDoubleGate(++iter, qNode, keyWord, ctr_qubit_addr, tar_qubit_addr);
    }

    else if (keyWord == "MEASURE")
    {
        if ((-1) == (*iter).substr((*iter).find(" ") + 1).find(",$"))
        {
            QCERR("Formal Error");
            throw invalid_argument("Formal Error");
        }
        int NumSize = (int)((*iter).find(",") - (*iter).find(" ") - 1);

        int qubit_addr = isIntNum((*iter).substr((*iter).find(" ") + 1, NumSize));
        checkNumberLegal(qubit_addr, m_all_qubits.size());
        
        int creg_addr = isIntNum((*iter).substr((*iter).find(",") + 2));
        checkNumberLegal(creg_addr, m_all_cregs.size());

        return handleMeasureGate(++iter, qNode, keyWord, qubit_addr, creg_addr);
    }

    else if (keyWord == "DAGGER")
    {
        return handleDaggerCircuit(++iter, qNode) + 2;
    }
    else if (keyWord == "CONTROL")
    {
        string cExpr=(*iter).substr((*iter).find(" ") + 1);
        vector<Qubit* > all_ctr_qubits;

        for (auto i = 0; i <= cExpr.length(); i += 2)
        {
            checkNumberLegal(isIntNum(cExpr.substr(i, 1)), m_all_qubits.size());
            all_ctr_qubits.emplace_back(m_all_qubits[stoi(cExpr.substr(i, 1))]);
        }

        for (auto i = 1; i <= cExpr.length()-1; i += 2)
        {
            if ("," != cExpr.substr(i, 1))
            {
                QCERR("Formal Error");
                throw invalid_argument("Formal Error");
            }
        }
        return handleControlCircuit(++iter, qNode, all_ctr_qubits, cExpr) + 2;
    }

    else if (keyWord == "QIF")
    {
        string cExpr = (*iter).substr((*iter).find(" ") + 1);
        ClassicalCondition condition_exper = checkIfWhileLegal(cExpr);
        return handleQIfProg(++iter, qNode, condition_exper)+2;
    }

    else if (keyWord == "QWHILE")
    {
        string cExpr = (*iter).substr((*iter).find(" ") + 1);
        ClassicalCondition condition_exper = checkIfWhileLegal(cExpr);
        return handleQWhileProg(++iter, qNode, condition_exper) + 2;
    }

    else
    {
        QCERR("Formal Error");
        throw invalid_argument("Formal Error");
    }
}

void QRunesToQprog::qRunesParser(QProg& newQProg)
{
    ifstream fin(m_sFilePath);
    std::vector<std::string> firstCheck;

    if (!fin)
    {
        QCERR("FileOpenError");
        throw invalid_argument("Open File Failed");
    }
    else
    {
        string sQRunes;
        while (!fin.eof())
        {
            getline(fin, sQRunes);
            string instruction = sQRunes.substr(0, sQRunes.find(" "));

            if (find(m_keyWords.begin(), m_keyWords.end(), instruction) == m_keyWords.end())
            {
                QCERR("KeyWordsError");
                throw invalid_argument("UnSupported KeyWords");
            }
            else
            {
                firstCheck.emplace_back(instruction);
                m_QRunes.emplace_back(sQRunes);
            }
        }
        fin.close();
    }

    countKeywords(firstCheck);
    QRunesToQprog::qRunesAllocation(m_QRunes,newQProg);
}
