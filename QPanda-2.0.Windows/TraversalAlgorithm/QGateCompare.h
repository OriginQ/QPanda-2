#ifndef  QGATE_COMPARE_H_
#define  QGATE_COMPARE_H_

#pragma once

#include "QuantumCircuit/QProgram.h"
#include <map>
#include "QuantumCircuit/QGlobalVariable.h"


using std::map;
using std::vector;


class QGateCompare {
public:
	QGateCompare();
	virtual ~QGateCompare();
	
	static size_t countQGateNotSupport(AbstractQuantumProgram *pQpro, const vector<vector<string>> &instructionSet);
	static size_t countQGateNotSupport(AbstractQGateNode *PQGata, const vector<vector<string>> &instructionSet);
	static size_t countQGateNotSupport(AbstractControlFlowNode *pCtr, const vector<vector<string>> &instructionSet);
	static size_t countQGateNotSupport(AbstractQuantumCircuit *pCircuit, const vector<vector<string>> &instructionSet);

protected:
	static size_t countQGateNotSupport(QNode *pNode, const vector<vector<string>> &instructionSet);
	static bool isItemExist(const string &item, const vector<vector<string>> &instructionSet);
private:
};






#endif