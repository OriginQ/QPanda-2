/*
Copyright (c) 2017-2023 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QST_simulation.h
Author: cby_Albert
Created in 2022-2-16

funtions for the simulation tomography

*/
#ifndef _QST_SIMULATION_H_
#define _QST_SIMULATION_H_

#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Core/QuantumMachine/QuantumMachineInterface.h"
#include "Core/QuantumCircuit/QNodeDeepCopy.h"
#include "QAlg/Base_QCircuit/AmplitudeEncode.h"

QPANDA_BEGIN

const double threshold = 1e-2;
const double sampling_const = 36;

class QSTSIM : public QuantumStateTomography
{
public:
    QSTSIM();
    QSTSIM(QCircuit& cir);
    virtual ~QSTSIM();

    QCircuit source_circuit_operation()
    {
        QProg prog = QProg(m_source_cir);
        init();
        auto qvec = qAllocMany(m_source_cir.get_qgate_num());
        m_qlist = qvec;
        auto cbits = cAllocMany(m_qlist.size());
        prog << MeasureAll(m_qlist, cbits);
        auto results = runWithConfiguration(prog, cbits, 1024);
        for (auto &ele : results)
        {
            m_source_prob.push_back(sqrt(double(ele.second) / double(1024)));
        }
        return m_source_cir;
    }

    QCircuit encoded_circuit(const QVec& qubits)
    {
        QCircuit cir = amplitude_encode(qubits, m_source_prob);
        return cir;
    }



    QCircuit get_source_circuit()
    {
        return m_source_cir;
    }

private:
    QVec m_qlist;
    QCircuit m_source_cir;
    QCircuit m_cir;
    std::vector<double> m_source_prob;
    std::vector<double> m_prob;
    std::vector<double> m_s;
};


/**
* @brief  QST_simulation
* @ingroup Utilities
* @param[in]  QCircuit& cir: source circuit
* @param[in]  QVec& qvec: the qvector
* @param[in]  shot: the measure number
* @return  std::vector<double> : the estimated vector after the tomography
*/
std::vector<double> QST_simulation(QCircuit& cir,QVec& qvec,const size_t shot);


QPANDA_END
#endif //_QST_SIMULATION_H_
