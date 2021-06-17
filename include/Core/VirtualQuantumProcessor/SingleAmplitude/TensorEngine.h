#ifndef TENSOR_ENGINE_H
#define TENSOR_ENGINE_H
#include "Core/VirtualQuantumProcessor/SingleAmplitude/TensorNode.h"

using qprog_sequence_t = std::vector<std::pair<size_t, bool>>;

class TensorEngine
{
public:
    static void split(QProgMap &prog_map, qubit_vertice_t &qubit_vertice);
    static qubit_vertice_t getNoValueVertice(QProgMap& prog_map,size_t contect_edge);
    static qubit_vertice_t getNoValueAndContectEdgeMaxVertice(QProgMap& prog_map);

    static qcomplex_data_t Merge(QProgMap &prog_map,
                                 const qprog_sequence_t &sequence);

    static qcomplex_data_t computing(QProgMap & prog_map);
    static std::map<qsize_t, Vertice>::iterator
           MergeQuantumProgMap(QProgMap&, qubit_vertice_t&, bool &is_success);
    static void MergeByVerticeVector(QProgMap & , qprog_sequence_t &sequence);
    
    static void dimDecrementbyValue(QProgMap&, qubit_vertice_t &,int value);
    static void dimDecrementbyValueAndNum(QProgMap&, qubit_vertice_t &,int value);
    static void getVerticeMap(QProgMap &, std::vector<std::pair<size_t, size_t>> &);
    static size_t getMaxRank(QProgMap &);
    static qubit_vertice_t getMaxQubitVertice(QProgMap &prog_map);

	static void seq_merge_by_vertices(QProgMap& prog_map, std::vector<size_t> vertice_vector, qprog_sequence_t& sequence);
	static void seq_merge(QProgMap& prog_map, qprog_sequence_t& vertice_vector);

};


#endif // !TENSOR_ENGINE_H


