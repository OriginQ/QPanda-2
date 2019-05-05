#ifndef TENSOR_ENGINE_H
#define TENSOR_ENGINE_H
#include "Core/VirtualQuantumProcessor/SingleAmplitude/TensorNode.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <thread>
#include <iostream>

using namespace std;

typedef struct QubitVertice
{
    qsize_t m_qubit_id;
    qsize_t m_num;
    qsize_t m_max;
    int   m_count;
    QubitVertice(const QubitVertice & old)
    {
        m_num = old.m_num;
        m_qubit_id = old.m_qubit_id;   
        m_max = old.m_max;
        m_count = old.m_count;
    }

    QubitVertice operator = (const QubitVertice & old)
    {
        m_num = old.m_num;
        m_qubit_id = old.m_qubit_id;
        m_max = old.m_max;
        m_count = old.m_count;
        return *this;
    }
    QubitVertice() :m_num(0), m_qubit_id(0), m_max(0), m_count(0)
    {}
}
qubit_vertice_t;

void split(QuantumProgMap* prog_map,
	qubit_vertice_t* qubit_vertice,
	qcomplex_data_t* result);

class TensorEngine
{
public:
    static qubit_vertice_t getNoValueVertice(QuantumProgMap& prog_map,size_t contect_edge);
    static qubit_vertice_t getNoValueAndContectEdgeMaxVertice(QuantumProgMap& prog_map);
    static void Merge(QuantumProgMap& prog_map);
    static qcomplex_data_t Merge(QuantumProgMap& prog_map, qubit_vertice_t*);
    static qcomplex_data_t computing(QuantumProgMap & prog_map);
    static map<qsize_t, Vertice>::iterator 
		MergeQuantumProgMap(QuantumProgMap&, qubit_vertice_t&);
    
    static void dimDecrementbyValue(QuantumProgMap&, qubit_vertice_t &,int value);
};

#endif // !TENSOR_ENGINE_H


