/*
Copyright (c) 2017-2018 Origin Quantum Computing. All Right Reserved.
Licensed under the Apache License 2.0

QAOA.h

Author: LiYe
Created in 2018-09-06


*/

#ifndef QAOA_H
#define QAOA_H

#include <map>
#include <memory>
#include "QPanda.h"
#include "Utilities/QAlgDataStruct.h"

namespace QPanda
{
    /*
    Quantum Approximate Optimization Algorithm
    */
    class AbstractQOptimizer;
    class QAOA
    {
    public:
        QAOA(Optimizer optimizer = NELDER_MEAD);
        QAOA(QAOA &) = delete;
        QAOA operator =(QAOA &) = delete;

        void setHamiltonian(const QPauliMap &pauli_map)
        {
            m_pauli_map = pauli_map;
        }

        void setDeltaT(double delta_t)
        {
            m_delta_t = delta_t;
        }

        void setOptimizedPara(const vector_d &optimized_para)
        {
            m_optimized_para = optimized_para;
        }

        void setShots(size_t shots)
        {
            m_shots = shots;
        }

        void regiestUserDefinedFunc(const QUserDefinedFunc &func)
        {
            m_user_defined_func = func;
        }

        AbstractQOptimizer* getOptimizer()
        {
            return m_optimizer.get();
        }

        OptimizationResult exec();
    private:
        bool initQAOA(OptimizationResult &result);
        bool checkPara(std::string &err_msg);

        QResultPair callQAOA(const vector_d &para);
        void insertSampleCircuit(
            QProg &prog,
            const std::vector<Qubit*> &vec,
            const vector_d &para);
        void insertComplexCircuit(
            QProg &prog,
            const std::vector<Qubit*> &vec,
            const vector_d &para);

        QResultPair getResult(const std::map<string, size_t> &result);
    private:
        QPauliMap m_pauli_map;
        double m_delta_t;

        QHamiltonian m_hamiltonia;

        vector_d m_optimized_para;

        size_t m_shots;
        size_t m_qn;

        bool m_simple_construct;

        std::vector<Qubit*> m_qubit_vec;
        std::vector<CBit*> m_cbit_vec;
        QProg m_prog;
        QCircuit m_circuit;

        QUserDefinedFunc m_user_defined_func;
        std::unique_ptr<AbstractQOptimizer> m_optimizer;
    };

}

#endif // QAOA_H
