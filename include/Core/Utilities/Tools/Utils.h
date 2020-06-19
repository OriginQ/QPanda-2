/*
Copyright (c) 2017-2019 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

Traversal.h
Author: doumenghan
Created in 2019-4-16

Classes for get the shortes path of graph

*/
#ifndef _UTILITIES_H
#define _UTILITIES_H

#include"Core/Utilities/QPandaNamespace.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include <iostream>
#include <map>
#include <stddef.h>

#pragma warning( disable : 4996)

QPANDA_BEGIN

#ifndef MAX_PRECISION
#define MAX_PRECISION 0.000001
#endif // !MAX_PRECISION

std::string dec2bin(unsigned n, size_t size);

double RandomNumberGenerator();

void add_up_a_map(std::map<std::string, size_t> &meas_result, std::string key);

void replace_qcircuit(AbstractQGateNode * pGateNode,
    QCircuit & qCircuit,
    QNode * pParentNode);

QProg Reset_Qubit_Circuit(Qubit *q, ClassicalCondition& cbit, bool setVal);

QProg Reset_Qubit(Qubit* q, bool setVal, QuantumMachine * qvm);

QProg Reset_All(std::vector<Qubit*> qubit_vector, bool setVal,QuantumMachine * qvm);


/**
* @brief  CNOT all qubits (except last) with the last qubit
* @ingroup Utilities
* @param[in]  std::vector<Qubit*>  qubit vector
* @return     QCircuit
*/
QCircuit parityCheckCircuit(std::vector<Qubit*> qubit_vec);

/**
* @brief  Apply Quantum Gate on a series of Qubit
* @ingroup Utilities
* @param[in]  QVec  qubit vector
* @param[in]  std::function<QGate(Qubit*)>  QGate function
* @return     QCircuit
*/
inline QCircuit apply_QGate(QVec qubits, std::function<QGate(Qubit*)> gate) {
	QCircuit c;
	for (auto qubit : qubits) {
		c << gate(qubit);
	}
	return c;
}

inline QCircuit applyQGate(QVec qubits, std::function<QGate(Qubit*)> gate) {
	QCircuit c;
	for (auto qubit : qubits) {
		c << gate(qubit);
	}
	return c;
}

template<typename InputType, typename OutputType>
using Oracle = std::function<QCircuit(InputType, OutputType)>;

/**
* @brief  Toffoli Quantum Gate 
* @ingroup Utilities
* @param[in]  Qubit*  first control qubit 
* @param[in]  Qubit*  second control qubit 
* @param[in]  Qubit*  target qubit
* @return     QGate
*/
inline QGate Toffoli(Qubit* qc1, Qubit* qc2, Qubit* target) {
	auto gate = X(target);
	gate.setControl({ qc1,qc2 });
	return gate;
}

/**
* @brief  Splits the string by symbol
* @ingroup Utilities
* @param[in]  std::string&  string
* @param[in]  std::string&  delim 
* @return     std::vector<std::string>
*/
inline std::vector<std::string> split(const std::string& str, const std::string& delim) {
	std::vector<std::string> res;
	if ("" == str) return res;
	char * strs = new char[str.length() + 1];  
	strcpy(strs, str.c_str());

	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());

	char *p = strtok(strs, d);
	while (p) {
		std::string s = p;
		res.push_back(s);
		p = strtok(NULL, d);
	}

	return res;
}

/**
* @brief  cast string to int
* @ingroup Utilities
* @param[in]  std::string&  string
* @param[out]   long long&  number
* @return     bool
*/
inline bool str2int(std::string s, long long& num) {
	try {
		num = atoll(s.c_str());
		return true;
	}
	catch (std::exception& e) {
		return false;
	}
}

/**
* @brief  cast string to double
* @ingroup Utilities
* @param[in]  std::string&  string
* @param[out]  double&  number
* @return     bool
*/
inline bool str2double(std::string s, double& num) {
	try {
		num = atof(s.c_str());
		return true;
	}
	catch (std::exception& e) {
		return false;
	}
}


/**
* @brief  cast long long to string
* @ingroup Utilities
* @param[in]  long long  number
* @return     std::string
*/
inline std::string ll2str(long long num) {
	char p[100];
    snprintf(p, sizeof(p), "%lld", num);
	return std::string(p);
}

/**
* @brief  cast int to string
* @ingroup Utilities
* @param[in]  int  number
* @return     std::string
*/
inline std::string int2str(int num) {
	char p[100];
    snprintf(p, sizeof(p), "%d", num);
	return std::string(p);
}

/**
* @brief  cast double to string
* @ingroup Utilities
* @param[in]  double  number
* @return     std::string
*/
inline std::string double2str(double num) {
	char p[100];
    snprintf(p, sizeof(p), "%lf", num);
	return std::string(p);
}

inline bool parse_oracle_name(std::string rawname,
	std::string &oraclename, std::vector<size_t> &qubits) {

	auto strs = split(rawname, "_");
	oraclename = strs[0];
	for (size_t i = 1; i < strs.size(); ++i) {
		ptrdiff_t s = 0;
		if ((str2int(strs[i], (long long&)s)) && (s >= 0)) {
			qubits.push_back((size_t)s);
		}
		else return false;
	}
}

inline std::string generate_oracle_name(std::string oraclename,
	std::vector<size_t> qubits) {
	for (auto qubit : qubits) {
		oraclename += ll2str(qubit);
	}
	return oraclename;
}

inline double argc(qcomplex_t num)
{
	if ((abs(num.imag()) < MAX_PRECISION) && (abs(num.real()) < MAX_PRECISION))
	{
		return acos(1);
	}
    else if (num.imag() >0)
    {
        return acos((qstate_type)(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag())));
    }
    else if(num.imag() <0)
    {
        return -acos((qstate_type)(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag())));
    }
    else
    {
        return acos((qstate_type)(num.real() / sqrt(num.real()*num.real() + num.imag()*num.imag())));
    }
}

QPANDA_END

#endif // !1

