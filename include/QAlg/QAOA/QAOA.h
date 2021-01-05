/*
Copyright (c) 2017-2020 Origin Quantum Computing. All Right Reserved.
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
#include "Core/QuantumCircuit/QProgram.h"
#include "Core/QuantumCircuit/QCircuit.h"
#include "Core/QuantumCircuit/QGate.h"
#include "Core/QuantumMachine/OriginQuantumMachine.h"
#include "Components/Operator/PauliOperator.h"
#include "Components/DataStruct.h"

namespace QPanda
{
    class AbstractOptimizer;
	
	/**
	* @brief Quantum Approximate Optimization Algorithm
    * @ingroup QAOA
    */
    class QAOA
    {
    public:
		/**
	    * @brief Constructor of ChemiQ
	    */
        QAOA(OptimizerType optimizer = OptimizerType::NELDER_MEAD);
        QAOA(const std::string &optimizer);
        QAOA(QAOA &) = delete;
        QAOA& operator =(QAOA &) = delete;
        ~QAOA();

		/**
	    * @brief set Hamiltonian
	    * @param[in] QPauliMap& pauli map
	    */
        void setHamiltonian(const PauliOperator &pauli)
        {
            m_pauli = pauli;
        }

		/**
		* @brief set val of Delta T
		* @param[in] double the val of Delta T
		*/
        void setDeltaT(double delta_t)
        {
            m_delta_t = delta_t;
        }

		/**
		* @brief set step
		* @param[in] size_t the val of step
		*/
        void setStep(size_t step)
        {
            m_step = step;
        }

		/**
		* @brief get step
		* @return return the val of step
		*/
        size_t step()
        {
            return m_step;
        }

		/**
		* @brief set Shots val
		* @param[in] size_t the val of Shots
		*/
        void setShots(size_t shots)
        {
            m_shots = shots;
        }

		/**
		* @brief regiest user defined functional
		* @param[in] QUserDefinedFunc& the user defined functional
		*/
        void regiestUserDefinedFunc(const QUserDefinedFunc &func)
        {
            m_user_defined_func = func;
        }

		/**
		* @brief set default optimize parameter
		* @param[in] vector_d& the default optimize parameters
		*/
        void setDefaultOptimizePara(const vector_d &para)
        {
            m_default_optimizer_para = para;
        }

		/**
		* @brief whether or not enable the log file
		* @param[in] bool whether or not
		* @param[in] string filename log file name
		*/
        void enableLog(bool enabled, std::string filename = "")
        {
            m_enable_log = enabled;
            m_log_filename = filename;
        }

		/**
		* @brief get optimizer object
		* @return AbstractOptimizer* the optimizer object ptr
		*/
        AbstractOptimizer* getOptimizer()
        {
            return m_optimizer.get();
        }

		/**
		* @brief execute optimizer
		* @return return true on success, or else return false
		*/
        bool exec();

		/**
		* @brief get optimizer result
		* @return return QOptimizationResult
		*/
        QOptimizationResult getOptimizerResult()
        {
            return m_optimization_result;
        }

		/**
		* @brief scan Para to file
		* @return return true on success, or else return false
		*/
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
        PauliOperator m_pauli;
		QuantumMachine * m_qvm;
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
