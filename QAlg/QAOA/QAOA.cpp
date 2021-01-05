#include <algorithm>
#include <numeric>
#include <ctime>
#include "QAlg/QAOA/QAOA.h"
#include "Components/Optimizer/OptimizerFactory.h"
#include "Components/Optimizer/AbstractOptimizer.h"
#include "Components/Operator/PauliOperator.h"
#include "Components/HamiltonianSimulation/HamiltonianSimulation.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"
using namespace std;
namespace QPanda
{
    QAOA::QAOA(OptimizerType optimizer)
    {
        m_optimizer = OptimizerFactory::makeOptimizer(optimizer);
        if (nullptr == m_optimizer.get())
        {
            QCERR("No optimizer.");
            throw runtime_error("No optimizer.");
        }
    }

    QAOA::QAOA(const string &optimizer)
    {
        m_optimizer = OptimizerFactory::makeOptimizer(optimizer);
        if (nullptr == m_optimizer.get())
        {
            QCERR("No optimizer.");
            throw runtime_error("No optimizer.");
        }
    }

    QAOA::~QAOA()
    {  }

    bool QAOA::exec()
    {
        do
        {
            if (!initQAOA(m_optimization_result))
            {
                return false;
            }

            vector_d optimized_para;
            if (m_default_optimizer_para.size() == m_step * 2)
            {
                optimized_para = std::move(m_default_optimizer_para);
            }
            else
            {
                optimized_para.resize(m_step * 2, 0.5);
            }

            m_optimizer->registerFunc(std::bind(&QAOA::callQAOA,
                this,
                std::placeholders::_1),
                optimized_para);

            QInit();
            m_optimizer->exec();
            QFinalize();

            m_optimization_result = m_optimizer->getResult();
        } while (0);

        if (m_log_stream.is_open())
        {
            m_log_stream.close();
        }

        return true;
    }

    bool QAOA::scan2Para(const QScanPara &data)
    {
        if ((data.pos1 >= m_step*2)
                || (data.pos2 >= m_step*2))
        {
            std::cout << "Bad para position in scan2Para." << std::endl;
            return false;
        }

        if (!initQAOA(m_optimization_result))
        {
            return false;
        }

        std::fstream f(data.filename, std::ios::app);
        if (f.fail())
        {
            std::cout << "Open file failed! " << data.filename
                << std::endl;
            return false;
        }

        QInit();

        vector_d para = getDefaultPara();
        const QTwoPara &p = data.two_para;
        for (auto i = p.x_min; i < p.x_max + p.x_step*0.9; i += p.x_step)
        {
            for (auto j = p.y_min; j < p.y_max + p.y_step*0.9; j += p.y_step)
            {
                para[data.pos1] = i;
                para[data.pos2] = j;

                if (data.keys.empty())
                {
                    auto ret = callQAOA(para);
                    std::cout << ret.first << "\t" << ret.second
                              << "\t" << i << "\t" <<j << std::endl;
                    f << i << "\t" << j << "\t" << ret.second << std::endl;
                }
                else
                {
                    auto ret = getKeyProbability(para, data.keys);
                    std::cout << i << "\t" << j << "\t" << ret << std::endl;
                    f << i << "\t" << j << "\t" << ret << std::endl;
                }
            }
        }

        f.close();

        QFinalize();

        return true;
    }

    bool QAOA::initQAOA(QOptimizationResult &result)
    {
        std::string err_msg;

        do {
            if (!checkPara(err_msg))
            {
                break;
            }

            if (!initLog(err_msg))
            {
                break;
            }

            bool ok = false;
            m_hamiltonia = m_pauli.toHamiltonian(&ok);
            if (!ok)
            {
                err_msg = "It is not a hamiltonian.";
                break;
            }

            m_qn = m_pauli.getMaxIndex();
            m_simple_construct = m_pauli.isAllPauliZorI();

            return true;
        } while (0);

        result.message = err_msg;
        std::cout << DEF_WARING + result.message << std::endl;

        return false;
    }

    bool QAOA::checkPara(std::string &err_msg)
    {
        do
        {
            if (m_pauli.isEmpty())
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

            if (0 == m_step)
            {
                err_msg = "Step cannot set to 0.";
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

    bool QAOA::initLog(std::string & err_msg)
    {
        if (!m_enable_log)
        {
            return true;
        }

        if (m_log_filename.empty())
        {
            time_t now = time(nullptr);
            tm *ltm = localtime(&now);
            auto year = 1900 + ltm->tm_year;
            auto month = 1 + ltm->tm_mon;
            auto day = ltm->tm_mday;
            auto hour = ltm->tm_hour;
            auto min = ltm->tm_min;
            auto sec = ltm->tm_sec;

            char tmp_str[20];
            sprintf(tmp_str, "%04d%02d%02d_%02d%02d%02d", year, month, day,
                hour, min, sec);
            m_log_filename = std::string("QAOA_")
                    + std::string(tmp_str) + ".log";
        }

        m_log_stream = std::fstream(m_log_filename, std::ios::out);
        if (m_log_stream.fail())
        {
            err_msg = std::string("Open QAOA log file failed. filename :")
                + m_log_filename;

            return false;
        }

        return true;
    }

    QResultPair QAOA::callQAOA(const vector_d &para)
    {
        saveParaToLogFile(para);

        m_prog.clear();

        m_prog << m_circuit;

        vector_d beta_vec;
        vector_d gamma_vec;
        for (size_t i = 0; i < para.size()/2; i++)
        {
            beta_vec.push_back(para[i]);
            gamma_vec.push_back(para[i + para.size() / 2]);
        }

        for (size_t i = 0; i < m_step; i++)
        {
            if (m_simple_construct)
            {
                insertSampleCircuit(m_prog, m_qubit_vec, gamma_vec[i]);
            }
            else
            {
                insertComplexCircuit(m_prog, m_qubit_vec, gamma_vec[i]);
            }
            insertPaulixModel(m_prog, m_qubit_vec, beta_vec[i]);
        }

        //std::cout << convert_qprog_to_originir(m_prog, m_qvm) << std::endl;
        m_qvm->directlyRun(m_prog);

		auto ideal_qvm = dynamic_cast<IdealQVM *>(m_qvm);
		if (nullptr == ideal_qvm)
		{
			QCERR("qvm is null");
			throw runtime_error("qvm is null");
		}

        vector_d prob_vec = ideal_qvm->PMeasure_no_index(m_qubit_vec);

        map<string, size_t> result = ideal_qvm->quickMeasure(
            m_qubit_vec,
            static_cast<int>(m_shots));

        return getResult(result);
    }

    void QAOA::insertSampleCircuit(
        QProg &prog,
        const std::vector<Qubit*> &vec,
        const double &gamma)
    {
        prog << simulatePauliZHamiltonian(vec, m_hamiltonia, gamma);
    }

    void QAOA::insertComplexCircuit(
        QProg &prog, 
        const std::vector<Qubit*> &vec, 
        const double &gamma)
    {
        size_t slices = static_cast<size_t>(std::ceil(gamma / m_delta_t));
        prog << simulateHamiltonian(
            vec,
            m_hamiltonia,
            gamma,
            slices);
    }

    void QAOA::insertPaulixModel(QProg &prog,
        const std::vector<Qubit*> &vec,
        const double &beta)
    {
        for (size_t i = 0; i < vec.size(); i++)
        {
            prog << RX(vec[i], 2 * beta);
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

    void QAOA::saveParaToLogFile(const vector_d & para)
    {
        if (m_log_stream.is_open())
        {
            for (size_t i = 0; i < para.size(); i++)
            {
                if (i != 0)
                {
                    m_log_stream << "\t";
                }

                m_log_stream << para[i];
            }

            if (para.size() > 0)
            {
                m_log_stream << std::endl;
            }
        }
    }

    void QAOA::QInit()
    {
		m_qvm = new CPUQVM();
		m_qvm->init();

        for (size_t i = 0; i < m_qn; i++)
        {
            m_qubit_vec.emplace_back(m_qvm->allocateQubit());
            m_cbit_vec.emplace_back(m_qvm->allocateCBit());
        }

        applySingleGateToAll("H", m_qubit_vec, m_circuit);
    }

    void QAOA::QFinalize()
    {
		m_qvm->cFreeAll(m_cbit_vec);
		m_qvm->finalize();
		delete m_qvm;
		m_qvm = nullptr;
    }

    vector_d QAOA::getDefaultPara()
    {
        vector_d para;
        if (m_default_optimizer_para.size() == m_step*2)
        {
            para = m_default_optimizer_para;
        }
        else
        {
            std::cout << "00000000000000000000000" << std::endl;
            para.resize(m_step*2, 0.5);
        }

        return para;
    }

    double QAOA::getKeyProbability(
        const vector_d & para, 
        const vector_i & key_vec)
    {
        m_prog.clear();

        m_prog << m_circuit;

        vector_d beta_vec;
        vector_d gamma_vec;
        for (size_t i = 0; i < para.size() / 2; i++)
        {
            beta_vec.push_back(para[i]);
            gamma_vec.push_back(para[i + para.size() / 2]);
        }

        for (size_t i = 0; i < m_step; i++)
        {
            if (m_simple_construct)
            {
                insertSampleCircuit(m_prog, m_qubit_vec, gamma_vec[i]);
            }
            else
            {
                insertComplexCircuit(m_prog, m_qubit_vec, gamma_vec[i]);
            }
            insertPaulixModel(m_prog, m_qubit_vec, beta_vec[i]);
        }

        m_qvm->directlyRun(m_prog);
		auto ideal_qvm = dynamic_cast<IdealQVM *>(m_qvm);
		if (nullptr == ideal_qvm)
		{
			QCERR("qvm is null");
			throw runtime_error("qvm is null");
		}
        vector_d prob_vec = ideal_qvm->pMeasureNoIndex(m_qubit_vec);
        double key_probability = 0.0;
        int prob_vec_size = static_cast<int>(prob_vec.size());
        for (size_t i = 0; i < key_vec.size(); i++)
        {
            if (key_vec[i] < prob_vec_size)
            {
                size_t index = static_cast<size_t>(key_vec[i]);
                key_probability += prob_vec[index];
            }

            std::cout << std::endl;
        }

        return key_probability;
    }

}
