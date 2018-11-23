/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QuantumMetadata.h
Author: Wangjing
Created in 2018-8-31

Classes for get the shortes path of graph

*/

#ifndef QUBITCONFIG_H
#define QUBITCONFIG_H

#include "TinyXML/tinyxml.h"
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include "QuantumCircuit/QGlobalVariable.h"
#include "TranformQGateTypeStringAndEnum.h"


using std::string;
using std::vector;
using std::map;
using std::pair;


/*
Parse xml config and get metadata
*/
class QuantumMetadata
{   
public:
    QuantumMetadata() ;
    QuantumMetadata & operator =(const QuantumMetadata &) = delete;
    QuantumMetadata(const string & filename);

    /*
    Parse xml config file and get qubit count
    param:
        None
    return:
        qubit count

    Note:
    */
    size_t getQubitCount();

    /*
    Parse xml config file and get qubit matrix
    param:
        qubit_matrix: output qubit matrix
    return:
        sucess or not

    Note:
    */
    bool getQubiteMatrix(vector<vector<int> > &qubit_matrix);

    /*
    Parse xml config file and get single gate
    param:
        single_gate: output single gate
    return:
        sucess or not

    Note:
    */
    bool getSingleGate(vector<string> &single_gate);

    /*
    Parse xml config file and get double gate
    param:
        double_gate: output double gate
    return:
        sucess or not

    Note:
    */
    bool getDoubleGate(vector<string> &double_gate);

    /*
    Parse xml config file and get gate time
    param:
        gate_time_map: gate type map gate time
    return:
        sucess or not

    Note:
    */
    bool getGateTime(map<int, size_t> &gate_time_map);

    ~QuantumMetadata();

protected:
    /*
    insert time to gate_time_map
    param:
        gate: gate string type
        time: gate clock cycle
        gate_time_map: gate type map gate time
    return:
        None

    Note:
    */
    void insertGateTimeMap(const pair<string, size_t> &gate_time, map<int, size_t> &gate_time_map);

private:
    TiXmlDocument m_doc;
    TiXmlElement *m_root_element;
};


#endif // QubitConfig_H
