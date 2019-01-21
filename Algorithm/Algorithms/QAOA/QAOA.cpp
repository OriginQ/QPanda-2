#include <algorithm>
#include <numeric>
#include "QAOA.h"
#include "Utilities/QOptimizer/QOptimizerFactor.h"
#include "Utilities/QOptimizer/AbstractQOptimizer.h"
#include "Utilities/QPauliOperator/QPauliOperator.h"
#include "Utilities/QHamiltonian/QHamiltonian.h"

namespace QPanda
{

    QAOA::QAOA(Optimizer optimizer) :
        m_delta_t(0.01),
        m_optimized_para(2, 0),
        m_shots(100),
        m_qn(0),
        m_simple_construct(false)
    {
        m_optimizer = QOptimizerFactor::makeQOptimizer(optimizer);
        if (nullptr == m_optimizer.get())
        {
            throw std::string("Bad alloc.");
        }
    }

    OptimizationResult QAOA::exec()
    {
        OptimizationResult result;
        do
        {
            if (!initQAOA(result))
            {
                break;
            }

            m_optimizer->registerFunc(std::bind(&QAOA::callQAOA,
                this,
                std::placeholders::_1),
                m_optimized_para);

            init();

            for (size_t i = 0; i < m_qn; i++)
            {
                m_qubit_vec.emplace_back(qAlloc());
                m_cbit_vec.emplace_back(cAlloc());
            }

            applySingleGateToAll("H", m_qubit_vec, m_circuit);

            m_optimizer->exec();

            std::for_each(m_qubit_vec.begin(), m_qubit_vec.end(),
                [](Qubit* qbit) { qFree(qbit); });
            std::for_each(m_cbit_vec.begin(), m_cbit_vec.end(),
                [](CBit* cbit) { cFree(cbit); });

            finalize();

            return m_optimizer->getResult();
        } while (0);

        return result;
    }

    bool QAOA::initQAOA(OptimizationResult &result)
    {
        std::string err_msg;

        do {
            if (!checkPara(err_msg))
            {
                break;
            }

            QPauliOperator op(m_pauli_map);

            bool ok = false;
            m_hamiltonia = op.toHamiltonian(&ok);
            if (!ok)
            {
                err_msg = "It is not a hamiltonian.";
                break;
            }

            m_qn = op.getMaxIndex();
            m_simple_construct = op.isAllPauliZorI();

            return true;
        } while (0);

        result[DEF_MESSAGE] = err_msg;
        std::cout << DEF_WARING + result[DEF_MESSAGE]
            << std::endl;

        return false;
    }

    bool QAOA::checkPara(std::string &err_msg)
    {
        do
        {
            if (m_pauli_map.empty())
            {
                err_msg = "Pauli map is empty.";
                break;
            }

            if ((m_delta_t < 0) ||
                (m_delta_t < 1e-6))
            {
                err_msg = "Delta T is closed to Zero.";
                break;
            }

            if (m_optimized_para.empty() ||
                (0 != (m_optimized_para.size() % 2)))
            {
                err_msg = "Optimizer parameter setting error.";
                break;
            }

            if (0 == m_shots)
            {
                err_msg = "Shots num is zero.";
                break;
            }

            return true;
        } while (0);

        return false;
    }

    QResultPair QAOA::callQAOA(const vector_d &para)
    {
        m_prog.clear();

        m_prog << m_circuit;

        if (m_simple_construct)
        {
            insertSampleCircuit(m_prog, m_qubit_vec, para);
        }
        else
        {
            insertComplexCircuit(m_prog, m_qubit_vec, para);
        }

        load(m_prog);
        run();

        vector_d prob_vec = PMeasure_no_index(m_qubit_vec);
        prob_vec = accumulateProbability(prob_vec);

        map<string, size_t> result = quick_measure(
                                         m_qubit_vec, 
                                         m_shots, 
                                         prob_vec);

        return getResult(result);
    }

    void QAOA::insertSampleCircuit(
        QProg &prog,
        const std::vector<Qubit*> &vec,
        const vector_d &para)
    {
        for (auto i = 0; i < para.size(); i++)
        {
            prog << simulatePauliZHamiltonian(vec, m_hamiltonia, para[i]);
        }
    }

    void QAOA::insertComplexCircuit(
        QProg &prog, 
        const std::vector<Qubit*> &vec, 
        const vector_d &para)
    {
        for (auto i = 0; i < para.size(); i++)
        {
            size_t slices = std::ceil(para[i] / m_delta_t);
            prog << simulateHamiltonian(
                        vec, 
                        m_hamiltonia, 
                        para[i],
                        slices);
        }
    }

    QResultPair QAOA::getResult(const std::map<string, size_t> &result)
    {
        double target = std::numeric_limits<double>::min();
        std::string result_key;

        for_each(result.begin(),
            result.end(),
            [&, this](const std::pair<string, size_t> item)
        {
            const std::string &key = item.first;
            double ret_value = m_user_defined_func(key);
            if (ret_value > target)
            {
                target = ret_value;
                result_key = key;
            }
        });

        return QResultPair(result_key, -target);
    }

}
