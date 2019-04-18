/*
* Copyright (c) 2019 Origin Quantum Computing. All Right Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
*/
/*! \file PartialAmplitudeQVM.h */
#ifndef  _PARTIALAMPLITUDE_H_
#define  _PARTIALAMPLITUDE_H_
#include "include/Core/VirtualQuantumProcessor/CPUImplQPU.h"
#include "include/Core/VirtualQuantumProcessor/PartialAmplitude/MergeMap.h"
QPANDA_BEGIN
/**
* @namespace QPanda
*/


/**
* @class PartialAmplitudeQVM
* @brief Quantum machine for partial amplitude simulation
* @ingroup QuantumMachine
*/
class PartialAmplitudeQVM : public QVM,public TraversalQProg
{
public:
    /**
    * @brief  Init  the quantum  machine environment
    * @return     void
    * @note   use  this at the begin
    */
    void init();
    PartialAmplitudeQVM();
    ~PartialAmplitudeQVM();

    /**
    * @brief  load the quanyum program
    * @param[in]  QProg&  the reference to a quantum program
    * @return     void
    */
    void run(QProg&);

    /**
    * @brief  Get the quantum state of QProg
    * @return     QStat   quantum state
    * @exception   run_fail   pQProg is null
    */
    QStat getQStat();

    std::vector<double> PMeasure(QVec, int);
    std::vector<std::pair<size_t, double> > PMeasure(int);

    std::vector<double> getProbList(QVec, int);
    std::vector<double> probRunList(QProg &, QVec, int);

    std::map<std::string, double> getProbDict(QVec, int);
    std::map<std::string, double> probRunDict(QProg &, QVec, int);

    std::vector<std::pair<size_t, double>> getProbTupleList(QVec, int);
    std::vector<std::pair<size_t, double>> probRunTupleList(QProg &, QVec, int);

private:
    MergeMap *m_prog_map;
    long long low_pos, high_pos;

    void getAvgBinary(long long, size_t);
    void traversal(AbstractQGateNode *);
    void traversalAll(AbstractQuantumProgram *);
};

QPANDA_END
#endif
