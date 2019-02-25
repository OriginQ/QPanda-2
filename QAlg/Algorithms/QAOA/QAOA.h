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
#include <fstream>
#include "Core/QPanda.h"
#include "QAlg/DataStruct.h"

namespace QPanda
{
    /*
    Quantum Approximate Optimization Algorithm
    */
    class AbstractOptimizer;
    class QAOA
    {
    public:
        QAOA(OptimizerType optimizer = OptimizerType::NELDER_MEAD);
        QAOA(const std::string &optimizer);
        QAOA(QAOA &) = delete;
        QAOA& operator =(QAOA &) = delete;
        ~QAOA();

        void setHamiltonian(const QPauliMap &pauli_map)
        {
            m_pauli_map = pauli_map;
        }

        void setDeltaT(double delta_t)
        {
            m_delta_t = delta_t;
        }

        void setStep(size_t step)
        {
            m_step = step;
        }

        size_t step()
        {
            return m_step;
        }

        void setShots(size_t shots)
        {
            m_shots = shots;
        }

        void regiestUserDefinedFunc(const QUserDefinedFunc &func)
        {
            m_user_defined_func = func;
        }

        void setDefaultOptimizePara(const vector_d &para)
        {
            m_default_optimizer_para = para;
        }

        void enableLog(bool enabled, std::string filename = "")
        {
            m_enable_log = enabled;
            m_log_filename = filename;
        }

        AbstractOptimizer* getOptimizer()
        {
            return m_optimizer.get();
        }

        bool exec();

        auto getOptimizerResult()
        {
            return m_optimization_result;
        }

        bool scan2Para(const QScanPara &data);
    private:
        bool initQAOA(QOptimizationResult &result);
        bool checkPara(std::string &err_msg);
        bool initLog(std::string &err_msg);
        QResultPair callQAOA(const vector_d &para);
        void insertSampleCircuit(
            QProg &prog,
            const std::vector<Qubit*> &vec,
            const double &gamma);
        void insertComplexCircuit(
            QProg &prog,
            const std::vector<Qubit*> &vec,
            const double &gamma);
        void insertPaulixModel(QProg &prog,
            const std::vector<Qubit*> &vec,
            const double &beta);

        QResultPair getResult(const std::map<std::string, size_t> &result);
        void saveParaToLogFile(const vector_d &para);

        void QInit();
        void QFinalize();
        vector_d getDefaultPara();
        double getKeyProbability(
            const vector_d &para, 
            const vector_i &key_vec);
    private:
        QPauliMap m_pauli_map;
        double m_delta_t{0.01};

        QHamiltonian m_hamiltonia;

        size_t m_step{1};
        size_t m_shots{100};
        size_t m_qn{0};

        bool m_simple_construct{false};

        QVec m_qubit_vec;
        std::vector<ClassicalCondition> m_cbit_vec;
        QProg m_prog;
        QCircuit m_circuit;

        QUserDefinedFunc m_user_defined_func;
        vector_d m_default_optimizer_para;
        std::unique_ptr<AbstractOptimizer> m_optimizer;

        QOptimizationResult m_optimization_result;

        bool m_enable_log{false};
        std::string m_log_filename;
        std::fstream m_log_stream;
    };

}

#endif // QAOA_H
