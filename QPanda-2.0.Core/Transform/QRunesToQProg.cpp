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

string QRunesToQprog::m_file_path ;


string calculationFormat(string exper) 
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
    return exper;
}

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
            throw param_error_exception("Formal Error", false);
        }
    }
    return operation.top();
}

double calculationSolve(string exper) 
{
    exper = calculationFormat(exper);
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
                throw param_error_exception("Formal Error", false);
                break;
            }
            switch (str[i + 1])
            {
            case ')':
            case '+':
            case '*':
            case '/':
            case '.':
                throw param_error_exception("Formal Error", false);
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
                throw param_error_exception("Formal Error", false);
            }
            switch (str[i - 1])
            {
            case '(':
            case '+':
            case '-':
            case '*':
            case '/':
            case '.':
                throw param_error_exception("Formal Error", false);
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
                throw param_error_exception("Formal Error", false);
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
        throw param_error_exception("Formal Error", false);
    }

    for (auto iter=str.begin();iter!=str.end();iter++)
    {
        if ('+'!= *iter && '-' != *iter && '*' != *iter && '/' != *iter && 
            '(' != *iter && ')' != *iter && '.' != *iter && '0' != *iter &&
            '1' != *iter && '2' != *iter && '3' != *iter && '4' != *iter &&
            '5' != *iter && '6' != *iter && '7' != *iter&&'8' != *iter && '9' != *iter)
        {
            throw param_error_exception("Formal Error", false);
        }
        if ((*iter == '+') || (*iter == '-') || (*iter == '*')
            || (*iter == '/'))
        {
            if ((iter + 1) == str.end())
            {
                throw param_error_exception("Formal Error", false);
            }
            else if ((*(iter + 1) == '+') || (*(iter + 1) == '-') || (*(iter + 1) == '*')
                || (*(iter + 1) == '/'))
            {
                throw param_error_exception("Formal Error", false);
            }
            else if ((*iter == '/') && (*(iter + 1) == '0'))
            {
                throw param_error_exception("Formal Error", false);
            }
            else
            {
                throw param_error_exception("Formal Error", false);
            }
        }
    }
}

QRunesToQprog::QRunesToQprog()
{
    m_check_legal.emplace_back("#QUBITS_NUM");
    m_check_legal.emplace_back("#CREGS_NUM");
    m_check_legal.emplace_back("X");
    m_check_legal.emplace_back("Y");
    m_check_legal.emplace_back("Z");

    m_check_legal.emplace_back("H");
    m_check_legal.emplace_back("T");
    m_check_legal.emplace_back("S");

    m_check_legal.emplace_back("RX");
    m_check_legal.emplace_back("RY");
    m_check_legal.emplace_back("RZ");

    m_check_legal.emplace_back("X1");
    m_check_legal.emplace_back("Y1");
    m_check_legal.emplace_back("Z1");

    m_check_legal.emplace_back("U1");
    m_check_legal.emplace_back("U2");
    m_check_legal.emplace_back("U3");
    m_check_legal.emplace_back("U4");

    m_check_legal.emplace_back("CU");
    m_check_legal.emplace_back("CNOT");
    m_check_legal.emplace_back("CZ");

    m_check_legal.emplace_back("CPHASE");
    m_check_legal.emplace_back("ISWAP");
    m_check_legal.emplace_back("SQISWAP");

    m_check_legal.emplace_back("DAGGER");
    m_check_legal.emplace_back("ENDAGGER");
    m_check_legal.emplace_back("CONTROL");
    m_check_legal.emplace_back("ENDCONTROL");

    m_check_legal.emplace_back("QIF");
    m_check_legal.emplace_back("ELSE");
    m_check_legal.emplace_back("ENDQIF");
    m_check_legal.emplace_back("QWHILE");
    m_check_legal.emplace_back("ENDQWHILE");

    m_check_legal.emplace_back("MEASURE");

}

QRunesToQprog::~QRunesToQprog()
{
}

void isIntNum(string str)
{
    stringstream sin(str);
    int int_number;

    char char_number;

    if (!(sin >> int_number))
        {
          throw param_error_exception("Formal Error", false);
        }
    if (sin >> char_number)
        {
          throw param_error_exception("Formal Error", false);
        }
}

void isDoubleNum(string str)
{
    stringstream sin(str);

    double double_number;
    char char_number;
    if (!(sin >> double_number))
    {
        throw param_error_exception("Formal Error", false);
    }
    if (sin >> char_number)
    {
        throw param_error_exception("Formal Error", false);
    }
}

void checkNumberLegal(int number,int max_number)
{
    if ((number < 0 ) || (number>=max_number))
    {
        throw param_error_exception("Illegal Number", false);
    }
}

void countKeywords(vector<string> &keywords_vector)
{
    int dagger_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "DAGGER");
    int endagger_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "ENDAGGER");
    int control_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "CONTROL");
    int endcontrol_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "ENDCONTROL");
    int qif_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "QIF");
    int endqif_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "ENDQIF");
    int qwhile_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "QWHILE");
    int endqwhile_number = (int)count(keywords_vector.begin(), keywords_vector.end(), "ENDQWHILE");

    if (dagger_number != endagger_number || control_number != endcontrol_number || 
             qif_number != endqif_number || qwhile_number  != endqwhile_number )
    {
        throw param_error_exception("Illegal KeyWords", false);
    }
}

ClassicalCondition QRunesToQprog::checkIfWhileLegal(string ctr_info)
{
    int count_not_number = 0;
    int first_number_length = 0;
    int number_length = 0;
    if (ctr_info.substr(0, 1) == "!")
    {
        for (; ctr_info.substr(count_not_number, 1) == "!"; count_not_number++);
        if (   ctr_info.substr(count_not_number, 1) != "c")
        {
            throw param_error_exception("Formal Error", false);
        }
    }
    else if (ctr_info.substr(0, 1) == "c")
    {}
    else
    {
        throw param_error_exception("Formal Error", false);
    }
    for (; ctr_info.substr(count_not_number + first_number_length + 1, 1) != "|" && 
           ctr_info.substr(count_not_number + first_number_length + 1, 1) != "+" && 
           ctr_info.substr(count_not_number + first_number_length + 1, 1) != "&" && 
           ctr_info.substr(count_not_number + first_number_length + 1, 1) != "-"; 
           first_number_length++);
    if (first_number_length == 0)
    {
        throw param_error_exception("Formal Error", false);
    }

    isIntNum(ctr_info.substr(count_not_number + 1, first_number_length ));
    int creg_addr = stoi(ctr_info.substr(count_not_number + 1, first_number_length + 1));
    checkNumberLegal(creg_addr, m_max_cregs);
    ClassicalCondition condition_exper = bind_a_cbit(m_all_cregs[creg_addr]);

    for (int i = 0;i < count_not_number;i++)
    {
        condition_exper = !condition_exper;
    }

    for (int position = first_number_length + 1 + count_not_number ; 
                                       position !=ctr_info.length();  )
    {
        number_length = 0;
        if (ctr_info.substr(position, 1) == "+")
        {
            int count_not_number = 0;
            for (; ctr_info.substr(position + count_not_number + 1, 1) == "!";
                count_not_number++);
            if (ctr_info.substr(position + count_not_number + 1, 1) != "c")
            {
                throw param_error_exception("Formal Error", false);
            }

            for (; (position + count_not_number + number_length + 2) < ctr_info.length() &&
                ctr_info.substr(position + count_not_number + number_length + 2, 1) != "|" &&
                ctr_info.substr(position + count_not_number + number_length + 2, 1) != "+" &&
                ctr_info.substr(position + count_not_number + number_length + 2, 1) != "&" &&
                ctr_info.substr(position + count_not_number + number_length + 2, 1) != "-";
                number_length++);
            if (number_length == 0)
            {
                throw param_error_exception("Formal Error", false);
            }

            isIntNum(ctr_info.substr(position + count_not_number + 2, number_length));
            int creg_addr = stoi(ctr_info.substr(position + count_not_number + 2, number_length));
            checkNumberLegal(creg_addr, m_max_cregs);
            ClassicalCondition condition_exper = bind_a_cbit(m_all_cregs[creg_addr]);

            for (int i = 0; i < count_not_number; i++)
            {
                condition_exper = !condition_exper;
            }
            condition_exper = condition_exper + condition_exper;
            position = position + count_not_number + number_length + 2;
        }

        else if (ctr_info.substr(position, 1) == "-")
        {
            int count_not_number = 0;
            for (; ctr_info.substr(position + count_not_number + 1, 1) == "!"; count_not_number++);
            if (ctr_info.substr(position + count_not_number + 1, 1) != "c")
            {
                throw param_error_exception("Formal Error", false);
            }
            for (; (position + count_not_number + number_length + 2) < ctr_info.length() &&
                   ctr_info.substr(position + count_not_number + number_length + 2, 1) != "|" &&
                   ctr_info.substr(position + count_not_number + number_length + 2, 1) != "+" &&
                   ctr_info.substr(position + count_not_number + number_length + 2, 1) != "&" &&
                   ctr_info.substr(position + count_not_number + number_length + 2, 1) != "-"; number_length++);

            if (number_length == 0)
            {
                throw param_error_exception("Formal Error", false);
            }

            isIntNum(ctr_info.substr(position + count_not_number + 2, number_length));
            int creg_addr = stoi(ctr_info.substr(position + count_not_number + 2, number_length));
            checkNumberLegal(creg_addr, m_max_cregs);
            ClassicalCondition condition_exper = bind_a_cbit(m_all_cregs[creg_addr]);

            for (int i = 0; i < count_not_number; i++)
            {
                condition_exper = !condition_exper;
            }

            condition_exper = condition_exper - condition_exper;
            position = position + count_not_number + number_length + 2;
        }

        else if (ctr_info.substr(  position, 1) == "|")
        {
             if (ctr_info.substr(++position, 1) != "|")
            {
                throw param_error_exception("Formal Error", false);
            }

            int count_not_number = 0;
            for (; ctr_info.substr(position + count_not_number + 1, 1) == "!"; count_not_number++);

            if (ctr_info.substr(position + count_not_number + 1, 1) != "c")
            {
                throw param_error_exception("Formal Error", false);
            }

            for (; (position + count_not_number + number_length + 2) < ctr_info.length() &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "|" &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "+" &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "&" &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "-"; number_length++);

            if (number_length == 0)
            {
                throw param_error_exception("Formal Error", false);
            }

            isIntNum(ctr_info.substr(position + count_not_number + 2, number_length));
            int creg_addr = stoi(ctr_info.substr(position + count_not_number + 2, number_length));
            checkNumberLegal(creg_addr, m_max_cregs);
            ClassicalCondition condition_exper = bind_a_cbit(m_all_cregs[creg_addr]);

            for (int i = 0; i < count_not_number; i++)
            {
                condition_exper = !condition_exper;
            }

            condition_exper = condition_exper || condition_exper;
            position = position + count_not_number + number_length + 2;
        }

        else if (ctr_info.substr(  position, 1) == "&")
        {
            if ( ctr_info.substr(++position, 1) != "&")
            {
                throw param_error_exception("Formal Error", false);
            }

            int count_not_number = 0;
            for (; ctr_info.substr(position + count_not_number + 1,1) == "!"; count_not_number++);
            if (ctr_info.substr(position + count_not_number + 1, 1) != "c")
            {
                throw param_error_exception("Formal Error", false);
            }

            for (; (position + count_not_number + number_length + 2) < ctr_info.length() &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "|" &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "+" &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "&" &&
                    ctr_info.substr(position + count_not_number + number_length + 2, 1) != "-"; number_length++);

            if (number_length == 0)
            {
                throw param_error_exception("Formal Error", false);
            }

            isIntNum(ctr_info.substr(position + count_not_number + 2, number_length));
            int creg_addr = stoi(ctr_info.substr(position + count_not_number + 2, number_length));
            checkNumberLegal(creg_addr, m_max_cregs);
            ClassicalCondition condition_exper = bind_a_cbit(m_all_cregs[creg_addr]);

            for (int i = 0; i < count_not_number; i++)
            {
                condition_exper = !condition_exper;
            }

            condition_exper = condition_exper && condition_exper;
            position = position + count_not_number + number_length + 2;
        }

        else
        {
            throw param_error_exception("Formal Error", false);
        }
    }

    return condition_exper;

}

void QRunesToQprog::qRunesAllocation(vector<string> &m_qrunes)
{
    string first_line_keyword  = m_qrunes[FIRST_LINE ].substr(0, m_qrunes[FIRST_LINE ].find(" "));
    string second_line_keyword = m_qrunes[SECOND_LINE].substr(0, m_qrunes[SECOND_LINE].find(" "));
    string qubits_number = m_qrunes[FIRST_LINE ].substr(m_qrunes[FIRST_LINE].find(" ") + 1, 
                                                          m_qrunes[FIRST_LINE].length());
    string cregs_number  = m_qrunes[SECOND_LINE].substr(m_qrunes[SECOND_LINE].find(" ") + 1, 
                                                          m_qrunes[SECOND_LINE].length());


    isIntNum(qubits_number);
    isIntNum(cregs_number);

    m_max_qubits = stoi(qubits_number);
    m_max_cregs = stoi(cregs_number);

    for ( int i = 0; i<m_max_qubits; i++ )
    {
        m_all_qubits.emplace_back(qAlloc());
    }
    for ( int i = 0; i < m_max_cregs; i++)
    {
        m_all_cregs.emplace_back(cAlloc());
    }


    for (auto iter = m_qrunes.begin()+2 ; iter!=m_qrunes .end() ;)
    {
        iter += QRunesToQprog::traversalQRunes(iter,&m_new_qprog);
    }

}

int  QRunesToQprog::handleDaggerCircuit(vector<string>::iterator iter, QNode *qNode)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();
    int circuit_length = 0;
    int increment = 0;

    auto quantum_circuit = CreateEmptyCircuit();

    for (; (*iter != "ENDAGGER");)
    {
        increment = traversalQRunes(iter, &quantum_circuit);
        iter += increment;
        circuit_length += increment;
    }
    quantum_circuit.setDagger(true);
    if (PROG_NODE == node_type)
    {
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        *qProg << quantum_circuit;
    }
    else if (CIRCUIT_NODE == node_type)
    {
        QCircuit *qCirCuit = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCirCuit)
        {
            throw param_error_exception(" error", false);
        }
        *qCirCuit << quantum_circuit;
    }
    else
    {
        throw param_error_exception("Error", false);
    }

    return circuit_length;

}

int  QRunesToQprog::handleControlCircuit(vector<string>::iterator iter, QNode *qNode,
                                         vector<Qubit*> &all_ctr_qubits, string &ctr_info)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();

    auto quantum_circuit = CreateEmptyCircuit();

    int circuit_length = 0;
    int increment = 0;

    for (; (*iter).substr(0, (*iter).find(" ")) != "ENDCONTROL";)
    {
        increment = traversalQRunes(iter, &quantum_circuit);
        iter += increment;
        circuit_length += increment;
    }
    string sEnCtrStr = (*iter).substr((*iter).find(" ") + 1);
    if (sEnCtrStr != ctr_info)
    {
        throw param_error_exception("Error", false);
    }
    quantum_circuit.setControl(all_ctr_qubits);

    if (PROG_NODE == node_type)
    {
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        *qProg << quantum_circuit;
    }
    else if (CIRCUIT_NODE == node_type)
    {
        QCircuit *qCirCuit = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCirCuit)
        {
            throw param_error_exception(" error", false);
        }
        *qCirCuit << quantum_circuit;
    }
    else
    {
        throw param_error_exception("Error", false);
    }

    return circuit_length;

}

int  QRunesToQprog::handleSingleGate (vector<string>::iterator iter, QNode *qNode,
                                     const string &gate_name,int qubit_addr)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();
    if ( NODE_UNDEFINED == node_type)
    {
        throw param_error_exception("param error", false);
    }
    else if ( CIRCUIT_NODE == node_type )
    {
        QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            throw param_error_exception(" error", false);
        }
        if (gate_name == "H")
        {
            *qCir << H(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "T")
        {
            *qCir << T(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "S")
        {
            *qCir << S(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "X")
        {
            *qCir << X(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Y")
        {
            *qCir << Y(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Z")
        {
            *qCir << Z(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "X1")
        {
            *qCir << X1(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Y1")
        {
            *qCir << Y1(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Z1")
        {
            *qCir << Z1(m_all_qubits[qubit_addr]);
        }
    }
    else if ( PROG_NODE == node_type)
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        if (gate_name == "H")
        {
            *qProg << H(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "T")
        {
            cout << "T " << qubit_addr << endl;
            *qProg << T(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "S")
        {
            *qProg << S(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "X")
        {
            *qProg << X(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Y")
        {
            *qProg << Y(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Z")
        {
            *qProg << Z(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "X1")
        {
            *qProg << X1(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Y1")
        {
            *qProg << Y1(m_all_qubits[qubit_addr]);
        }
        else if (gate_name == "Z1")
        {
            *qProg << Z1(m_all_qubits[qubit_addr]);
        }
    }
    return 1;
}

int  QRunesToQprog::handleDoubleGate (vector<string>::iterator iter, QNode *qNode, 
                                     const string &gate_name, int ctr_qubit_addr,int tar_qubit_addr)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();
    if (NODE_UNDEFINED == node_type)
    {
        throw param_error_exception("param error", false);
    }
    else if (CIRCUIT_NODE == node_type)
    {
        QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            throw param_error_exception(" error", false);
        }
        if (gate_name == "CNOT")
        {
            *qCir << CNOT(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else if (gate_name == "CZ")
        {
            *qCir << CZ(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else if (gate_name == "ISWAP")
        {
            *qCir << iSWAP(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else if (gate_name == "SQISWAP")
        {
            *qCir << SqiSWAP(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
    }
    else if (PROG_NODE == node_type)
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        if (gate_name == "CNOT")
        {
            *qProg << CNOT(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else if (gate_name == "CZ")
        {
            *qProg << CZ(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else if (gate_name == "ISWAP")
        {
            *qProg << iSWAP(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
        else if (gate_name == "SQISWAP")
        {
            *qProg << SqiSWAP(m_all_qubits[ctr_qubit_addr], m_all_qubits[tar_qubit_addr]);
        }
    }
    return 1;
}

int  QRunesToQprog::handleAngleGate  (vector<string>::iterator iter, QNode *qNode, 
                                     const string &gate_name, int qubit_addr, double gate_angle)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();
    if (NODE_UNDEFINED == node_type)
    {
        throw param_error_exception("param error", false);
    }
    else if (CIRCUIT_NODE == node_type)
    {
        QCircuit * qCir = dynamic_cast<QCircuit*>(qNode);
        if (nullptr == qCir)
        {
            throw param_error_exception(" error", false);
        }
        if (gate_name == "RX")
        {
            *qCir << RX(m_all_qubits[qubit_addr], gate_angle);
        }
        else if (gate_name == "RY")
        {
            *qCir << RY(m_all_qubits[qubit_addr], gate_angle);
        }
        else if (gate_name == "RZ")
        {
            *qCir << RZ(m_all_qubits[qubit_addr], gate_angle);
        }
    }
    else if (PROG_NODE == node_type)
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        if (gate_name == "RX")
        {
            *qProg << RX(m_all_qubits[qubit_addr], gate_angle);
        }
        else if (gate_name == "RY")
        {
            *qProg << RY(m_all_qubits[qubit_addr], gate_angle);
        }
        else if (gate_name == "RZ")
        {
            *qProg << RZ(m_all_qubits[qubit_addr], gate_angle);
        }
    }
    return 1;
}

int  QRunesToQprog::handleMeasureGate(vector<string>::iterator iter, QNode *qNode, 
                                     const string &key_word, int qubit_addr, int creg_addr)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();
    if (PROG_NODE != node_type)
    {
        throw param_error_exception("param error", false);
    }
    else 
    {
        QProg * qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        *qProg << Measure(m_all_qubits[qubit_addr], m_all_cregs[creg_addr]);
    }
    return 1;
}

int  QRunesToQprog::handleQifProg(vector<string>::iterator iter, QNode *qNode,
                                     ClassicalCondition &condition_exper)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();
    if (PROG_NODE != node_type)
    {
        throw param_error_exception("param error", false);
    }

    auto true_prog = CreateEmptyQProg();
    auto pFalseProg= CreateEmptyQProg();

    int circuit_length = 0;
    int increment = 0;
    int elseincrement = 0;
    for (; (*iter != "ENDQIF") && (*iter != "ELSE");)
    {
        increment = traversalQRunes(iter, &true_prog);
        iter += increment;
        circuit_length += increment;
    }

    if (*iter == "ELSE")
    {
        ++iter;
        ++circuit_length;

        for (; *iter != "ENDQIF" ;)
        {
            increment = traversalQRunes(iter, &pFalseProg);
            iter += increment;
            circuit_length += increment;
        }
        auto ifProg = CreateIfProg(condition_exper, &true_prog, &pFalseProg);
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        *qProg << ifProg;
    }
    else
    {
        auto ifProg = CreateIfProg(condition_exper, &true_prog);
        QProg *qProg = dynamic_cast<QProg*>(qNode);
        if (nullptr == qProg)
        {
            throw param_error_exception(" error", false);
        }
        *qProg << ifProg;
    }

    return circuit_length;
}

int  QRunesToQprog::handleQWhileProg(vector<string>::iterator iter, QNode *qNode, 
                                     ClassicalCondition &condition_exper)
{
    if (nullptr == qNode)
    {
        throw param_error_exception("qNode is Null", false);
    }
    int node_type = qNode->getNodeType();
    if (PROG_NODE != node_type)
    {
        throw param_error_exception("param error", false);
    }

    auto true_prog=CreateEmptyQProg();
    int circuit_length = 0;
    int increment = 0;

    for (; *iter != "ENDQWHILE";)
    {
        increment = traversalQRunes(iter, &true_prog);
        iter += increment;
        circuit_length += increment;
    }
    auto ifProg = CreateWhileProg(condition_exper, &true_prog);
    QProg *qProg = dynamic_cast<QProg*>(qNode);
    if (nullptr == qProg)
    {
        throw param_error_exception(" error", false);
    }
    *qProg << ifProg;

    return circuit_length;
}

int  QRunesToQprog::traversalQRunes(vector<string>::iterator iter, QNode *qNode)
{
    string key_word = (*iter).substr(0, (*iter).find(" "));
    if (key_word == "H"  || key_word == "T"  || key_word == "S"||
        key_word == "X"  || key_word == "Y"  || key_word == "Z"||
        key_word == "X1" || key_word == "Y1" || key_word == "Z1")
    {
        isIntNum((*iter).substr((*iter).find(" ") + 1));
        int qubit_addr = stoi((*iter).substr((*iter).find(" ") + 1));
        checkNumberLegal(qubit_addr, m_max_qubits);
        return handleSingleGate(iter, qNode, key_word, qubit_addr);
    }

    else if (key_word == "RX" || key_word == "RY" || key_word == "RZ" )
    {
        if ( (-1) ==(*iter).substr((*iter).find(" ")+1).find(","))
        {
            throw param_error_exception("Formal Error", false);
        }
        if ((*iter).substr((*iter).find(",") + 1,1)!="\"" && 
            (*iter).substr((*iter).length()-1,1) != "\"")
        {
            throw param_error_exception("Formal Error", false);
        }

        int number_length = (int)((*iter).find(",")-(*iter).find(" ")-1);
        isIntNum((*iter).substr( 3 , number_length));
        int qubit_addr = stoi((*iter).substr(3, number_length));
        checkNumberLegal(qubit_addr, m_max_qubits);

        checkAngleExper((*iter).substr((*iter).find(",") + 2, (*iter).length()- (*iter).find(",")-3));
        double gate_angle = calculationSolve((*iter).substr((*iter).find(",") + 2, (*iter).length() - (*iter).find(",") - 3));
        return handleAngleGate(++iter, qNode, key_word, qubit_addr,gate_angle);
    }

    else if (key_word == "CNOT" || key_word == "CZ" || key_word == "ISWAP" || key_word == "SQISWAP")
    {
        if ((-1) == (*iter).substr((*iter).find(" ") + 1).find(","))
        {
            throw param_error_exception("Formal Error", false);
        }
        int number_length = (int)((*iter).find(",") - (*iter).find(" ") - 1);
        isIntNum((*iter).substr((*iter).find(" ") + 1, number_length));

        int ctr_qubit_addr = stoi((*iter).substr((*iter).find(" ") + 1, number_length));
        isIntNum((*iter).substr((*iter).find(",") + 1));

        int tar_qubit_addr = stoi((*iter).substr((*iter).find(",") + 1));
        checkNumberLegal(ctr_qubit_addr, m_max_qubits);
        checkNumberLegal(tar_qubit_addr, m_max_qubits);

        if ((ctr_qubit_addr == tar_qubit_addr))
        {
            throw param_error_exception("Formal Error", false);
        }
        return handleDoubleGate(++iter, qNode, key_word, ctr_qubit_addr, tar_qubit_addr);
    }

    else if (key_word == "MEASURE")
    {
        if ((-1) == (*iter).substr((*iter).find(" ") + 1).find(",$"))
        {
            throw param_error_exception("Formal Error", false);
        }
        int number_length = (int)((*iter).find(",") - (*iter).find(" ") - 1);
        isIntNum((*iter).substr((*iter).find(" ") + 1, number_length));

        int qubit_addr = stoi((*iter).substr((*iter).find(" ") + 1, number_length));
        checkNumberLegal(qubit_addr, m_max_qubits);

        isIntNum((*iter).substr((*iter).find(",") + 2));
        int creg_addr = stoi((*iter).substr((*iter).find(",") + 2));

        checkNumberLegal(creg_addr, m_max_cregs);
        return handleMeasureGate(++iter, qNode, key_word, qubit_addr, creg_addr);
    }

    else if (key_word == "DAGGER")
    {
        return handleDaggerCircuit(++iter, qNode)+2;
    }
    else if (key_word == "CONTROL")
    {
        string ctr_info=(*iter).substr((*iter).find(" ") + 1);
        vector<Qubit* > all_ctr_qubits;

        for (auto i = 0; i <= ctr_info.length(); i += 2)
        {
            isIntNum(ctr_info.substr(i,1));
            checkNumberLegal(stoi(ctr_info.substr(i, 1)), m_max_qubits);
            all_ctr_qubits.emplace_back(m_all_qubits[stoi(ctr_info.substr(i, 1))]);
        }

        for (auto i = 1; i <= ctr_info.length()-1; i += 2)
        {
            if ("," != ctr_info.substr(i, 1))
            {
                throw param_error_exception("Formal Error", false);
            }
        }
        return handleControlCircuit(++iter, qNode, all_ctr_qubits, ctr_info) + 2;
    }

    else if (key_word == "QIF")
    {
        string ctr_info = (*iter).substr((*iter).find(" ") + 1);
        ClassicalCondition condition_exper = checkIfWhileLegal(ctr_info);
        return handleQifProg(++iter, qNode, condition_exper)+2;
    }

    else if (key_word == "QWHILE")
    {
        string ctr_info = (*iter).substr((*iter).find(" ") + 1);
        ClassicalCondition condition_exper = checkIfWhileLegal(ctr_info);
        return handleQWhileProg(++iter, qNode, condition_exper)+2;
    }

    else
    {
        throw param_error_exception("Formal Error", false);
    }


}

QProg QRunesToQprog::qRunesParser()
{
    ifstream fin(m_file_path);
    if (!fin)
    {
        throw param_error_exception("Open File Failed", false);
    }
    else
    {
        string qrunes_line;
        while (!fin.eof())
        {
            getline(fin, qrunes_line);
            string instructions_keyword = qrunes_line.substr(0, qrunes_line.find(" "));

            auto iter = find(m_check_legal.begin(), m_check_legal.end(), instructions_keyword);

            if (iter == m_check_legal.end())
            {
                throw param_error_exception("UnSupported KeyWords", false);
            }
            else
            {
                m_first_check.emplace_back(instructions_keyword);
                m_qrunes.emplace_back(qrunes_line);
            }
        }
        fin.close();
    }
    countKeywords(m_first_check);

    QRunesToQprog::qRunesAllocation(m_qrunes);

    return m_new_qprog;

}

void QRunesToQprog::setFilePath(string s_FilePath   )
{
    m_file_path = s_FilePath;
 }
